"""Mel-spectrogram caching, clip datasets and length-based batching."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..audio import MelSpectrogram, load_audio
from ..text import CTCVocab

SILENCE = math.log(1e-5)   # log-mel value of digital silence, used for padding


def compute_chapter_mels(data_root: str | Path, chapter_ids: list[str], mel: MelSpectrogram,
                         device: str = "cpu", overwrite: bool = False) -> None:
    """Save each chapter's full log mel-spectrogram as chapters/<id>/mel.npy (float16)."""
    for cid in tqdm(chapter_ids, desc="mel-spectrograms"):
        out = Path(data_root) / "chapters" / cid / "mel.npy"
        if out.exists() and not overwrite:
            continue
        audio = load_audio(out.parent / "audio.wav", mel.cfg.sample_rate)
        x = torch.from_numpy(audio).to(device)
        mel.filterbank = mel.filterbank.to(device)
        m = mel(x).cpu().numpy().astype(np.float16)
        np.save(out, m)
    mel.filterbank = mel.filterbank.cpu()


def load_mels(data_root: str | Path, chapter_ids: list[str]) -> dict[str, np.ndarray]:
    return {cid: np.load(Path(data_root) / "chapters" / cid / "mel.npy") for cid in chapter_ids}


def mel_stats(mels: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard deviation of each mel band over all frames."""
    total = sum(m.shape[1] for m in mels.values())
    s = sum(m.astype(np.float64).sum(axis=1) for m in mels.values())
    mean = s / total
    sq = sum(((m.astype(np.float64) - mean[:, None]) ** 2).sum(axis=1) for m in mels.values())
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(np.sqrt(sq / total), dtype=torch.float32)


class ClipDataset(Dataset):
    """(mel [n_mels, T], target ids) for each segment, sliced from chapter mels."""

    def __init__(self, segments: list[dict], mels: dict[str, np.ndarray], vocab: CTCVocab):
        self.segments = segments
        self.mels = mels
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.segments)

    def frames(self, i: int) -> int:
        s = self.segments[i]
        return s["mel_end"] - s["mel_start"]

    def __getitem__(self, i: int):
        s = self.segments[i]
        mel = self.mels[s["chapter"]][:, s["mel_start"]: s["mel_end"]]
        return torch.from_numpy(mel.astype(np.float32)), torch.tensor(self.vocab.encode(s["text"]))


def collate(batch):
    mels, targets = zip(*batch)
    lengths = torch.tensor([m.shape[1] for m in mels])
    padded = torch.full((len(mels), mels[0].shape[0], int(lengths.max())), SILENCE)
    for i, m in enumerate(mels):
        padded[i, :, : m.shape[1]] = m
    target_lengths = torch.tensor([len(t) for t in targets])
    return padded, lengths, torch.cat(targets), target_lengths


def length_batches(frames: list[int], max_frames: int, shuffle: bool = True,
                   seed: int = 0) -> list[list[int]]:
    """Group clips of similar length so little of each batch is padding.

    Clips are sorted by length (with a little noise, so batches differ between
    epochs), then packed until the padded batch would exceed max_frames.
    """
    rng = random.Random(seed)
    order = sorted(range(len(frames)),
                   key=lambda i: frames[i] * (1 + (rng.uniform(-0.1, 0.1) if shuffle else 0)))
    batches, current, longest = [], [], 0
    for i in order:
        new_longest = max(longest, frames[i])
        if current and new_longest * (len(current) + 1) > max_frames:
            batches.append(current)
            current, new_longest = [], frames[i]
        current.append(i)
        longest = new_longest
    if current:
        batches.append(current)
    if shuffle:
        rng.shuffle(batches)
    return batches


def spec_augment(mel: torch.Tensor, lengths: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Mask random bands of frequencies and stretches of time (Park et al. 2019).

    Forces the model to use context rather than rely on any one band or
    moment, which matters with only ~10 hours of data.
    """
    mel = mel.clone()
    B, F, _ = mel.shape
    fill = mel.mean()
    for b in range(B):
        for _ in range(cfg["freq_masks"]):
            w = random.randint(0, cfg["freq_width"])
            f = random.randint(0, F - w)
            mel[b, f: f + w, :] = fill
        T = int(lengths[b])
        for _ in range(cfg["time_masks"]):
            w = random.randint(0, int(T * cfg["time_width"]))
            t = random.randint(0, max(0, T - w))
            mel[b, :, t: t + w] = fill
    return mel
