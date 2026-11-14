import numpy as np
import torch

from bibletts.audio import (MelConfig, MelSpectrogram, bandwidth_hz, detect_pauses,
                            load_audio, normalise_loudness, probe, save_wav, speech_level_db)

SR = 22050


def tone(seconds, freq=440.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_mel_shape_matches_hop():
    mel = MelSpectrogram(MelConfig())
    x = tone(1.0)
    out = mel(x)
    assert out.shape == (80, len(x) // 256)


def test_griffin_lim_recovers_a_tone():
    """Round trip a pure tone: the output should still peak at the same pitch."""
    mel = MelSpectrogram(MelConfig())
    x = tone(1.0, freq=1000.0)
    y = mel.griffin_lim(mel(x), n_iter=30)
    assert abs(len(y) - len(x)) < 256
    spectrum = np.abs(np.fft.rfft(y))
    peak_hz = np.argmax(spectrum) * SR / len(y)
    assert abs(peak_hz - 1000) < 60  # mel pooling blurs frequency a little


def test_pauses_found_where_silence_is():
    x = np.concatenate([tone(1.0), np.zeros(int(0.5 * SR), np.float32), tone(1.0),
                        np.zeros(int(0.1 * SR), np.float32), tone(1.0)])
    pauses = detect_pauses(x, SR, threshold_db=-40, min_duration_s=0.3)
    assert len(pauses) == 1  # the 0.1 s gap is too short
    start, end = pauses[0]
    assert abs(start - 1.0) < 0.05 and abs(end - 1.5) < 0.05


def test_loudness_normalisation():
    x = normalise_loudness(tone(2.0, amp=0.01), SR, target_dbfs=-20)
    assert abs(speech_level_db(x, SR) - (-20)) < 0.5
    assert np.max(np.abs(x)) <= 0.99


def test_bandwidth_detects_lowpass():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(SR * 2).astype(np.float32)
    # Crude low-pass at about 5 kHz by zeroing FFT bins.
    spec = np.fft.rfft(noise)
    spec[int(5000 / (SR / 2) * len(spec)):] = 0
    lowpassed = np.fft.irfft(spec, n=len(noise)).astype(np.float32)
    assert 4500 < bandwidth_hz(lowpassed, SR) < 5500


def test_ffmpeg_round_trip(tmp_path):
    path = tmp_path / "x.wav"
    save_wav(path, np.repeat(tone(1.0), 2), 44100)  # 1 s at 44.1 kHz
    info = probe(path)
    assert info["sample_rate"] == 44100 and abs(info["duration_s"] - 1.0) < 0.01
    y = load_audio(path, SR)  # resampled by ffmpeg
    assert abs(len(y) - SR) < 10
    assert torch.is_tensor(MelSpectrogram(MelConfig())(y))
