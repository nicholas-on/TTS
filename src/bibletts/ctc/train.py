"""Training one round of the recogniser, with checkpoints that survive
Colab disconnects: rerun the same call and it continues where it stopped."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..evaluate import cer
from ..text import CTCVocab
from .data import ClipDataset, collate, length_batches, spec_augment
from .model import CTCRecogniser, greedy_decode


def build_model(vocab: CTCVocab, model_cfg: dict, stride: int, n_mels: int = 80) -> CTCRecogniser:
    return CTCRecogniser(len(vocab), n_mels=n_mels, stride=stride, **model_cfg)


def load_model(path: str | Path, device: str = "cpu") -> tuple[CTCRecogniser, dict]:
    """Load a trained model from a checkpoint file. Returns (model, checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    vocab = CTCVocab(ckpt["alphabet"])
    model = build_model(vocab, ckpt["model_cfg"], ckpt["stride"], ckpt["n_mels"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def _lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warm-up, then cosine decay to 1% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model: CTCRecogniser, dataset: ClipDataset, device, max_clips: int = 300,
             batch_frames: int = 40000) -> dict:
    """Greedy-decode up to max_clips dev clips; return CER and a few examples."""
    model.eval()
    idx = list(range(min(len(dataset), max_clips)))
    refs, hyps = [], []
    for batch in length_batches([dataset.frames(i) for i in idx], batch_frames, shuffle=False):
        mel, lengths, _, _ = collate([dataset[idx[i]] for i in batch])
        lp = model(mel.to(device), lengths.to(device))
        out_lengths = model.output_lengths(lengths)
        for k, i in enumerate(batch):
            refs.append(dataset.vocab.decode(dataset[idx[i]][1].tolist()))
            hyps.append(dataset.vocab.decode(greedy_decode(lp[k], int(out_lengths[k]))))
    model.train()
    return {"cer": cer(refs, hyps), "examples": list(zip(refs[:3], hyps[:3]))}


def train_round(train_segments: list[dict], dev_segments: list[dict], mels: dict,
                vocab: CTCVocab, cfg: dict, stride: int, epochs: int, out_dir: str | Path,
                mel_mean: torch.Tensor, mel_std: torch.Tensor, hop_s: float,
                device: str | None = None, log_every: int = 50) -> CTCRecogniser:
    """Train a fresh recogniser on train_segments. Checkpoints go to out_dir.

    If out_dir already holds checkpoints from an interrupted run, training
    resumes from the latest one. If it holds a finished model (final.pt),
    that model is loaded and returned without training.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if (out_dir / "final.pt").exists():
        print(f"{out_dir / 'final.pt'} exists; loading it instead of training.")
        return load_model(out_dir / "final.pt", device)[0]

    tc = cfg["train"]
    random.seed(tc["seed"]); np.random.seed(tc["seed"]); torch.manual_seed(tc["seed"])

    # CTC needs at least one output frame per character (plus one between
    # repeated characters); drop clips that cannot satisfy that.
    def fits(s):
        n_out = math.ceil((s["mel_end"] - s["mel_start"]) / stride)
        return n_out > 1.2 * len(vocab.encode(s["text"])) and len(s["text"]) > 0
    train_segments = [s for s in train_segments if fits(s)]
    dev_segments = [s for s in dev_segments if fits(s)]
    train_ds = ClipDataset(train_segments, mels, vocab)
    dev_ds = ClipDataset(dev_segments, mels, vocab)
    frames = [train_ds.frames(i) for i in range(len(train_ds))]
    max_frames = int(tc["batch_seconds"] / hop_s)
    steps_per_epoch = len(length_batches(frames, max_frames, seed=0))
    total_steps = steps_per_epoch * epochs
    print(f"{len(train_ds)} train clips ({sum(frames) * hop_s / 3600:.2f} h), "
          f"{len(dev_ds)} dev clips; {steps_per_epoch} steps/epoch, {total_steps} steps total; "
          f"device {device}")

    model = build_model(vocab, cfg["model"], stride).to(device)
    model.set_normalisation(mel_mean, mel_std)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")
    optimiser = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    use_amp = tc["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ctc_loss = nn.CTCLoss(blank=vocab.blank, zero_infinity=True)

    step, start_epoch, start_batch = 0, 0, 0
    checkpoints = sorted(out_dir.glob("step_*.pt"))
    if checkpoints:
        ckpt = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scaler.load_state_dict(ckpt["scaler"])
        step, start_epoch, start_batch = ckpt["step"], ckpt["epoch"], ckpt["batch"]
        print(f"Resumed from {checkpoints[-1].name} (epoch {start_epoch}, step {step})")

    def save(path: Path, epoch: int, batch: int) -> None:
        torch.save({
            "model": model.state_dict(), "optimiser": optimiser.state_dict(),
            "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "batch": batch,
            "alphabet": vocab.alphabet, "model_cfg": cfg["model"], "stride": stride,
            "n_mels": int(mel_mean.numel()), "hop_s": hop_s,
        }, path)

    log_path = out_dir / "log.jsonl"
    model.train()
    t0 = time.time()
    running = []
    for epoch in range(start_epoch, epochs):
        batches = length_batches(frames, max_frames, seed=tc["seed"] + epoch)
        for b, batch in enumerate(batches):
            if epoch == start_epoch and b < start_batch:
                continue
            mel, lengths, targets, target_lengths = collate([train_ds[i] for i in batch])
            mel = spec_augment(mel.to(device), lengths, tc["specaugment"])
            for g in optimiser.param_groups:
                g["lr"] = _lr_at(step, total_steps, tc["lr"], tc["warmup_steps"])
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                log_probs = model(mel, lengths.to(device))
            loss = ctc_loss(log_probs.transpose(0, 1), targets.to(device),
                            model.output_lengths(lengths).to(device), target_lengths.to(device))
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimiser)
            scaler.update()
            step += 1
            running.append(loss.item())

            record = None
            if step % log_every == 0:
                record = {"step": step, "epoch": epoch, "loss": float(np.mean(running)),
                          "lr": optimiser.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60}
                running = []
            if step % tc["eval_every"] == 0 and len(dev_ds):
                result = evaluate(model, dev_ds, device)
                record = (record or {"step": step, "epoch": epoch}) | {"dev_cer": result["cer"]}
                print(f"step {step}: dev CER {result['cer']:.3f}")
                for ref, hyp in result["examples"][:2]:
                    print(f"   ref: {ref[:90]}\n   hyp: {hyp[:90]}")
            if record:
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
                if "loss" in record:
                    print(f"step {step} epoch {epoch} loss {record['loss']:.3f} "
                          f"({record['minutes']:.1f} min)")
            if step % tc["checkpoint_every"] == 0:
                save(out_dir / f"step_{step:07d}.pt", epoch, b + 1)
                for old in sorted(out_dir.glob("step_*.pt"))[: -tc["keep_checkpoints"]]:
                    old.unlink()

    if len(dev_ds):
        result = evaluate(model, dev_ds, device, max_clips=10**9)
        print(f"Final dev CER {result['cer']:.3f}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "final_dev_cer": result["cer"]}) + "\n")
    save(out_dir / "final.pt", epochs, 0)
    return model.eval()


def read_log(out_dir: str | Path) -> list[dict]:
    path = Path(out_dir) / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]
