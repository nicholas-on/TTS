"""Dataset of (tokens, durations, mel) clips from milestone 2's clips_final.jsonl."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..ctc.data import SILENCE


class TTSDataset(Dataset):
    def __init__(self, clips: list[dict], mels: dict[str, np.ndarray]):
        self.clips = clips
        self.mels = mels

    def __len__(self) -> int:
        return len(self.clips)

    def frames(self, i: int) -> int:
        c = self.clips[i]
        return c["mel_end"] - c["mel_start"]

    def __getitem__(self, i: int):
        c = self.clips[i]
        mel = self.mels[c["chapter"]][:, c["mel_start"]: c["mel_end"]].astype(np.float32)
        return (torch.tensor(c["tokens"]), torch.tensor(c["durations"]), torch.from_numpy(mel))


def collate(batch):
    tokens, durations, mels = zip(*batch)
    B = len(batch)
    n_tok = max(len(t) for t in tokens)
    n_frames = max(m.shape[1] for m in mels)
    tok = torch.zeros(B, n_tok, dtype=torch.long)
    dur = torch.zeros(B, n_tok, dtype=torch.long)
    mel = torch.full((B, mels[0].shape[0], n_frames), SILENCE)
    for i, (t, d, m) in enumerate(batch):
        tok[i, : len(t)] = t
        dur[i, : len(d)] = d
        mel[i, :, : m.shape[1]] = m
    return tok, dur, mel
