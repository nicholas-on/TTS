"""Turning text into speech with a trained acoustic model.

Long text is split at sentence and verse ends and each piece is synthesised
separately, then joined with pauses sized by the punctuation, which is how
the model was trained (clips of 2-12 seconds).
"""

from __future__ import annotations

import re

import numpy as np
import torch

from ..audio import MelSpectrogram
from ..ctc.data import SILENCE
from ..text import TTSVocab, clean_text
from .model import AcousticModel


def split_sentences(cleaned: str, max_chars: int = 180) -> list[str]:
    """Split cleaned text after . ? ! ; and, if a piece is still long, after commas."""
    pieces = [p.strip() for p in re.split(r"(?<=[.?!;])\s+", cleaned) if p.strip()]
    out = []
    for p in pieces:
        while len(p) > max_chars:
            cut = p.rfind(", ", 0, max_chars)
            if cut <= 0:
                cut = p.rfind(" ", 0, max_chars)
            if cut <= 0:
                break
            out.append(p[: cut + 1].strip())
            p = p[cut + 1:].strip()
        out.append(p)
    return out


@torch.no_grad()
def synthesise_mel(model: AcousticModel, vocab: TTSVocab, cleaned: str, pace: float = 1.0,
                   device: str = "cpu") -> np.ndarray:
    """One piece of cleaned text -> log mel-spectrogram [n_mels, T]."""
    model.eval()
    tokens = torch.tensor([vocab.encode(cleaned)], device=device)
    mel, _ = model.synthesise(tokens, pace)
    return mel[0].float().cpu().numpy()


def synthesise_text(model: AcousticModel, vocab: TTSVocab, raw_text: str, mel: MelSpectrogram,
                    text_cfg: dict, synth_cfg: dict, device: str = "cpu",
                    return_mel: bool = False):
    """Raw text (any length) -> waveform, using Griffin-Lim for now.

    Returns (audio float32, sample rate), plus the joined mel if return_mel.
    """
    cleaned = clean_text(raw_text, text_cfg["alphabet"], text_cfg["strip_accents"])
    hop_s = mel.cfg.hop_length / mel.cfg.sample_rate
    parts = []
    for piece in split_sentences(cleaned):
        parts.append(synthesise_mel(model, vocab, piece, synth_cfg["pace"], device))
        pause = synth_cfg["pause_s"].get(piece[-1] if piece[-1] in ",;.?!" else " ", 0.1)
        parts.append(np.full((parts[-1].shape[0], int(pause / hop_s)), SILENCE, np.float32))
    joined = np.concatenate(parts, axis=1) if parts else np.zeros((mel.cfg.n_mels, 1), np.float32)
    audio = mel.griffin_lim(torch.from_numpy(joined), n_iter=synth_cfg["griffin_lim_iters"])
    return (audio, mel.cfg.sample_rate, joined) if return_mel else (audio, mel.cfg.sample_rate)
