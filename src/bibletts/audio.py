"""Audio I/O, mel-spectrograms, Griffin-Lim, loudness and pause detection.

Nothing here has learned weights. The mel filterbank comes from librosa, but it
is only a fixed formula: triangular filters spaced evenly on the mel scale.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------


def load_audio(path: str | Path, sample_rate: int, start_s: float = 0.0,
               duration_s: float | None = None) -> np.ndarray:
    """Decode any format ffmpeg reads (mp3, ogg, wav...) to mono float32.

    Resampling and down-mixing to mono are done by ffmpeg.
    """
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start_s), "-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", str(duration_s)]
    cmd += ["-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if not raw:
        raise ValueError(f"No audio decoded from {path} at {start_s} s (past the end?)")
    return np.frombuffer(raw, dtype=np.float32).copy()


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def probe(path: str | Path) -> dict:
    """Format details from ffprobe: codec, sample rate, channels, bitrate, duration."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
         "-show_streams", str(path)],
        capture_output=True, check=True, text=True).stdout
    info = json.loads(out)
    stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    fmt = info["format"]
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": stream.get("channels"),
        "bit_rate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000),
        "duration_s": float(fmt.get("duration", 0.0)),
    }


# ---------------------------------------------------------------------------
# Loudness
# ---------------------------------------------------------------------------


def frame_rms_db(audio: np.ndarray, sample_rate: int, win_s: float = 0.025,
                 hop_s: float = 0.010) -> np.ndarray:
    """Loudness of short overlapping frames, in dB relative to full scale."""
    win = int(win_s * sample_rate)
    hop = int(hop_s * sample_rate)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, win)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-8))


def speech_level_db(audio: np.ndarray, sample_rate: int) -> float:
    """Typical loudness of the speech: the 95th percentile frame level.

    A percentile ignores silence and is not thrown off by one loud click, the
    way the peak would be.
    """
    return float(np.percentile(frame_rms_db(audio, sample_rate), 95))


def normalise_loudness(audio: np.ndarray, sample_rate: int,
                       target_dbfs: float = -20.0) -> np.ndarray:
    """Scale so speech sits at target_dbfs, then make sure nothing clips."""
    gain_db = target_dbfs - speech_level_db(audio, sample_rate)
    out = audio * (10 ** (gain_db / 20))
    peak = np.max(np.abs(out)) if len(out) else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Pauses
# ---------------------------------------------------------------------------


def detect_pauses(audio: np.ndarray, sample_rate: int, threshold_db: float = -40.0,
                  min_duration_s: float = 0.3, hop_s: float = 0.010) -> list[tuple[float, float]]:
    """Find silences: stretches quieter than threshold_db relative to the speech
    level, lasting at least min_duration_s. Returns (start, end) in seconds."""
    levels = frame_rms_db(audio, sample_rate, hop_s=hop_s)
    silent = levels < speech_level_db(audio, sample_rate) + threshold_db
    pauses = []
    start = None
    for i, s in enumerate(np.append(silent, False)):  # False closes a final run
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) * hop_s >= min_duration_s:
                pauses.append((round(start * hop_s, 3), round(i * hop_s, 3)))
            start = None
    return pauses


# ---------------------------------------------------------------------------
# Mel-spectrograms and Griffin-Lim
# ---------------------------------------------------------------------------


@dataclass
class MelConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0

    @classmethod
    def from_dict(cls, d: dict) -> "MelConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class MelSpectrogram:
    """Waveform -> log mel-spectrogram, and back again with Griffin-Lim.

    Steps of the forward direction:
    1. Cut the waveform into overlapping windows of win_length samples, one
       every hop_length samples (about 11.6 ms at 22,050 Hz).
    2. Fourier transform each window: how much of each frequency it contains.
    3. Keep the magnitude, drop the phase. This is the spectrogram.
    4. Pool the n_fft / 2 + 1 frequency bins into n_mels bands that are narrow
       at low frequencies and wide at high ones, the way hearing works.
    5. Take the log, because loudness is perceived logarithmically.

    Padding matches the HiFi-GAN convention, so a waveform of L samples gives
    about L / hop_length frames. The vocoder in milestone 4 relies on that.
    """

    def __init__(self, cfg: MelConfig):
        self.cfg = cfg
        fb = librosa.filters.mel(sr=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels,
                                 fmin=cfg.fmin, fmax=cfg.fmax)
        self.filterbank = torch.from_numpy(fb).float()          # [n_mels, n_fft/2+1]
        self.inverse_filterbank = torch.linalg.pinv(self.filterbank)
        self.window = torch.hann_window(cfg.win_length)

    def magnitude(self, audio: torch.Tensor) -> torch.Tensor:
        """[T_samples] or [B, T_samples] -> magnitude spectrogram [.., freq, frames]."""
        pad = (self.cfg.n_fft - self.cfg.hop_length) // 2
        x = audio.unsqueeze(0) if audio.dim() == 1 else audio
        x = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        spec = torch.stft(x, self.cfg.n_fft, self.cfg.hop_length, self.cfg.win_length,
                          window=self.window.to(x.device), center=False, return_complex=True)
        mag = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)
        return mag.squeeze(0) if audio.dim() == 1 else mag

    def __call__(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Waveform -> log mel-spectrogram [n_mels, frames] (or batched)."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        mel = self.filterbank.to(audio.device) @ self.magnitude(audio.float())
        return torch.log(torch.clamp(mel, min=1e-5))

    def griffin_lim(self, log_mel: torch.Tensor, n_iter: int = 60,
                    momentum: float = 0.99) -> np.ndarray:
        """Log mel-spectrogram -> waveform, with no learned model.

        1. Undo the log, then undo the mel pooling approximately (pseudo-inverse
           of the filterbank). Fine frequency detail lost in pooling stays lost.
        2. The phase was thrown away, so guess it: start random, then repeat
           "make a waveform from magnitude + phase, re-analyse it, keep the new
           phase, put the known magnitude back". Each round makes the phase
           more self-consistent. Momentum speeds this up (Perraudin et al. 2013).

        The result sounds buzzy and metallic. That is expected: the point is
        that the words are intelligible, which proves the mel-spectrogram
        holds what the TTS model needs.
        """
        c = self.cfg
        mel = torch.exp(log_mel.float())
        mag = torch.clamp(self.inverse_filterbank @ mel, min=0.0)   # [freq, frames]
        n_samples = mag.shape[-1] * c.hop_length
        window = self.window

        def stft(x):
            return torch.stft(x, c.n_fft, c.hop_length, c.win_length, window=window,
                              center=True, return_complex=True)[..., : mag.shape[-1]]

        def istft(s):
            return torch.istft(s, c.n_fft, c.hop_length, c.win_length, window=window,
                               center=True, length=n_samples)

        gen = torch.Generator().manual_seed(0)
        angles = torch.exp(2j * torch.pi * torch.rand(mag.shape, generator=gen))
        previous = torch.zeros_like(angles)
        for _ in range(n_iter):
            rebuilt = stft(istft(mag * angles))
            accelerated = rebuilt - (momentum / (1 + momentum)) * previous
            angles = accelerated / (accelerated.abs() + 1e-8)
            previous = rebuilt
        return istft(mag * angles).numpy().astype(np.float32)


def average_spectrum_db(audio: np.ndarray, sample_rate: int,
                        n_fft: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Mean energy per frequency over the whole clip, for spotting where the
    recording's content stops (MP3 encoders often cut everything above 11-16 kHz)."""
    spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=n_fft // 4)) ** 2
    power = 10 * np.log10(np.mean(spec, axis=1) + 1e-12)
    freqs = np.linspace(0, sample_rate / 2, len(power))
    return freqs, power


def bandwidth_hz(audio: np.ndarray, sample_rate: int, drop_db: float = 50.0) -> float:
    """Highest frequency whose average energy is within drop_db of the loudest band."""
    freqs, power = average_spectrum_db(audio, sample_rate)
    above = np.nonzero(power > power.max() - drop_db)[0]
    return float(freqs[above[-1]]) if len(above) else 0.0
