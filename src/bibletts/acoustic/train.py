"""Training the acoustic model, with checkpoints that survive Colab disconnects."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..audio import MelSpectrogram, save_wav
from ..ctc.data import length_batches
from ..ctc.model import CTCRecogniser, greedy_decode
from ..ctc.train import _lr_at
from ..evaluate import cer
from ..text import CTCVocab, TTSVocab, for_recogniser
from .data import TTSDataset, collate
from .model import AcousticModel, acoustic_loss
from .synthesis import synthesise_mel


def build_model(vocab: TTSVocab, model_cfg: dict, n_mels: int = 80) -> AcousticModel:
    return AcousticModel(len(vocab), n_mels=n_mels, **model_cfg)


def load_model(path: str | Path, device: str = "cpu") -> tuple[AcousticModel, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(TTSVocab(ckpt["alphabet"]), ckpt["model_cfg"], ckpt["n_mels"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def recogniser_cer(model: AcousticModel, recogniser: CTCRecogniser, clips: list[dict],
                   tts_vocab: TTSVocab, ctc_vocab: CTCVocab, device, pace: float = 1.0) -> dict:
    """Synthesise each clip's text from scratch (predicted durations), run the
    recogniser straight on the predicted mel-spectrogram, and measure CER.
    No vocoder involved, so this accuracy check works from the first hour."""
    model.eval()
    recogniser.eval()
    refs, hyps = [], []
    for c in clips:
        mel = synthesise_mel(model, tts_vocab, c["text"], pace, device)
        lp = recogniser(torch.from_numpy(mel)[None].to(device))[0]
        refs.append(for_recogniser(c["text"]))
        hyps.append(ctc_vocab.decode(greedy_decode(lp)))
    model.train()
    return {"cer": cer(refs, hyps), "examples": list(zip(refs[:3], hyps[:3]))}


@torch.no_grad()
def dev_loss(model: AcousticModel, ds: TTSDataset, device, max_frames: int) -> tuple[float, float]:
    model.eval()
    mel_losses, dur_losses = [], []
    for batch in length_batches([ds.frames(i) for i in range(len(ds))], max_frames, shuffle=False):
        tok, dur, mel = (x.to(device) for x in collate([ds[i] for i in batch]))
        pred, log_dur, lengths = model(tok, dur)
        _, ml, dl = acoustic_loss(pred.float(), mel, lengths, log_dur.float(), dur, tok)
        mel_losses.append(ml.item()); dur_losses.append(dl.item())
    model.train()
    return float(np.mean(mel_losses)), float(np.mean(dur_losses))


def train_acoustic(train_clips: list[dict], dev_clips: list[dict], mels: dict, vocab: TTSVocab,
                   cfg: dict, out_dir: str | Path, hop_s: float, mel: MelSpectrogram,
                   recogniser: CTCRecogniser | None = None, ctc_vocab: CTCVocab | None = None,
                   steps: int | None = None, device: str | None = None,
                   log_every: int = 100) -> AcousticModel:
    """Train on train_clips. Resumes from the latest checkpoint in out_dir; if
    out_dir holds final.pt, loads it instead. `steps` overrides cfg epochs
    (used for the quick overfitting check)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if (out_dir / "final.pt").exists():
        print(f"{out_dir / 'final.pt'} exists; loading it instead of training.")
        return load_model(out_dir / "final.pt", device)[0]

    tc = cfg["train"]
    random.seed(tc["seed"]); np.random.seed(tc["seed"]); torch.manual_seed(tc["seed"])
    train_ds, dev_ds = TTSDataset(train_clips, mels), TTSDataset(dev_clips, mels)
    frames = [train_ds.frames(i) for i in range(len(train_ds))]
    max_frames = int(tc["batch_seconds"] / hop_s)
    steps_per_epoch = len(length_batches(frames, max_frames, seed=0))
    total_steps = steps or steps_per_epoch * tc["epochs"]
    epochs = -(-total_steps // steps_per_epoch)
    print(f"{len(train_ds)} train clips ({sum(frames) * hop_s / 3600:.2f} h), {len(dev_ds)} dev "
          f"clips; {steps_per_epoch} steps/epoch, {total_steps} steps; device {device}")

    model = build_model(vocab, cfg["model"]).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")
    optimiser = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"],
                                  betas=(0.9, 0.98))
    use_amp = tc["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "batch": batch,
                    "alphabet": vocab.alphabet,
                    "model_cfg": cfg["model"], "n_mels": mel.cfg.n_mels, "hop_s": hop_s}, path)

    eval_clips = dev_clips[: tc["eval_clips"]]
    sample_clips = dev_clips[:2]
    log_path = out_dir / "log.jsonl"

    def evaluate_now() -> dict:
        record = {}
        if len(dev_ds):
            record["dev_mel_loss"], record["dev_dur_loss"] = dev_loss(model, dev_ds, device, max_frames)
        if recogniser is not None and eval_clips:
            r = recogniser_cer(model, recogniser.to(device), eval_clips, vocab, ctc_vocab, device)
            record["dev_synth_cer"] = r["cer"]
            for ref, hyp in r["examples"][:2]:
                print(f"   ref: {ref[:90]}\n   hyp: {hyp[:90]}")
        for k, c in enumerate(sample_clips):
            m = synthesise_mel(model, vocab, c["text"], 1.0, device)
            audio = mel.griffin_lim(torch.from_numpy(m), n_iter=32)
            save_wav(out_dir / "samples" / f"step_{step:07d}_{k}.wav", audio, mel.cfg.sample_rate)
        model.train()
        return record

    model.train()
    t0 = time.time()
    running = []
    for epoch in range(start_epoch, epochs):
        for b, batch in enumerate(length_batches(frames, max_frames, seed=tc["seed"] + epoch)):
            if epoch == start_epoch and b < start_batch:
                continue
            if step >= total_steps:
                break
            tok, dur, target = (x.to(device) for x in collate([train_ds[i] for i in batch]))
            for g in optimiser.param_groups:
                g["lr"] = _lr_at(step, total_steps, tc["lr"], min(tc["warmup_steps"], total_steps // 5))
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred, log_dur, lengths = model(tok, dur)
            loss, ml, dl = acoustic_loss(pred.float(), target, lengths, log_dur.float(), dur, tok)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimiser)
            scaler.update()
            step += 1
            running.append((ml.item(), dl.item()))

            record = None
            if step % log_every == 0:
                m, d = np.mean(running, axis=0)
                record = {"step": step, "epoch": epoch, "mel_loss": float(m), "dur_loss": float(d),
                          "lr": optimiser.param_groups[0]["lr"], "minutes": (time.time() - t0) / 60}
                running = []
                print(f"step {step} epoch {epoch}: mel {m:.3f} dur {d:.3f} "
                      f"({record['minutes']:.1f} min)")
            if step % tc["eval_every"] == 0:
                ev = evaluate_now()
                print(f"step {step}: " + ", ".join(f"{k} {v:.3f}" for k, v in ev.items()))
                record = (record or {"step": step, "epoch": epoch}) | ev
            if record:
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            if step % tc["checkpoint_every"] == 0:
                save(out_dir / f"step_{step:07d}.pt", epoch, b + 1)
                for old in sorted(out_dir.glob("step_*.pt"))[: -tc["keep_checkpoints"]]:
                    old.unlink()

    ev = evaluate_now()
    print("Final: " + ", ".join(f"{k} {v:.3f}" for k, v in ev.items()))
    with open(log_path, "a") as f:
        f.write(json.dumps({"step": step, "final": True} | ev) + "\n")
    save(out_dir / "final.pt", epochs, 0)
    return model.eval()
