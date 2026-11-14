"""Plots used by the notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_mel(log_mel, sample_rate: int, hop_length: int, title: str = "", ax=None):
    log_mel = np.asarray(log_mel)
    ax = ax or plt.subplots(figsize=(12, 3))[1]
    seconds = log_mel.shape[1] * hop_length / sample_rate
    im = ax.imshow(log_mel, origin="lower", aspect="auto", extent=[0, seconds, 0, log_mel.shape[0]])
    ax.set(xlabel="seconds", ylabel="mel band", title=title)
    plt.colorbar(im, ax=ax, label="log energy")
    return ax


def plot_spectrum(freqs, power_db, bandwidth: float | None = None, title: str = "", ax=None):
    ax = ax or plt.subplots(figsize=(10, 3))[1]
    ax.plot(freqs / 1000, power_db)
    if bandwidth:
        ax.axvline(bandwidth / 1000, color="r", ls="--", label=f"content ends ≈ {bandwidth/1000:.1f} kHz")
        ax.legend()
    ax.set(xlabel="kHz", ylabel="average dB", title=title)
    return ax


def plot_pauses(audio, sample_rate: int, pauses, start_s: float = 0.0, end_s: float = 60.0,
                title: str = "", ax=None):
    """Waveform with detected pauses shaded."""
    ax = ax or plt.subplots(figsize=(14, 2.5))[1]
    a, b = int(start_s * sample_rate), int(min(end_s * sample_rate, len(audio)))
    t = np.arange(a, b) / sample_rate
    step = max(1, (b - a) // 20000)  # thin out for speed
    ax.plot(t[::step], audio[a:b:step], lw=0.5)
    for s, e in pauses:
        if e >= start_s and s <= end_s:
            ax.axvspan(max(s, start_s), min(e, end_s), color="orange", alpha=0.35)
    ax.set(xlim=(start_s, b / sample_rate), xlabel="seconds", title=title)
    return ax


def plot_durations(log_mel, token_labels: list[str], durations: list[int], sample_rate: int,
                   hop_length: int, title: str = "", ax=None):
    """Mel-spectrogram with each token's span marked: the TTS model's training target.

    Each label sits at the start of its token's frames; dashed lines mark the
    boundaries. Character boundaries should fall roughly where the sound changes.
    """
    log_mel = np.asarray(log_mel)
    ax = ax or plt.subplots(figsize=(max(12, len(token_labels) * 0.18), 3.5))[1]
    ax.imshow(log_mel, origin="lower", aspect="auto")
    edges = np.concatenate([[0], np.cumsum(durations)])
    for label, a, b in zip(token_labels, edges[:-1], edges[1:]):
        if b > a:
            ax.axvline(a, color="w", lw=0.5, ls="--")
            ax.text((a + b) / 2, log_mel.shape[0] + 1, label, ha="center", va="bottom", fontsize=7)
    ax.set(xlabel=f"mel frames ({1000 * hop_length / sample_rate:.1f} ms each)", ylabel="mel band")
    ax.set_title(title, pad=14)
    return ax
