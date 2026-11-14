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
