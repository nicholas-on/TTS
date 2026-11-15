from pathlib import Path

import numpy as np
import torch

from bibletts.acoustic.data import TTSDataset, collate
from bibletts.acoustic.model import AcousticModel, acoustic_loss, length_regulate
from bibletts.acoustic.synthesis import split_sentences, synthesise_text
from bibletts.audio import MelConfig, MelSpectrogram
from bibletts.config import load_config
from bibletts.text import TTSVocab

REPO = Path(__file__).resolve().parents[1]
ALPHABET = "abcdefghijklmnopqrstuvwxyz'"
TINY = dict(dim=32, heads=2, ffn_dim=64, kernel=3, encoder_layers=1, decoder_layers=1,
            duration_hidden=32, dropout=0.0)


def test_length_regulate():
    x = torch.arange(6.0).reshape(1, 3, 2)
    out, lengths = length_regulate(x, torch.tensor([[2, 0, 3]]))
    assert lengths.tolist() == [5]
    assert out[0, :, 0].tolist() == [0, 0, 4, 4, 4]


def test_forward_shapes_and_masking():
    vocab = TTSVocab(ALPHABET)
    m = AcousticModel(len(vocab), **TINY).eval()
    tok = torch.tensor([vocab.encode("in the beginning."), vocab.encode("god.") + [0] * 13])
    dur = torch.randint(1, 4, tok.shape) * (tok != 0)
    mel, log_dur, lengths = m(tok, dur)
    assert mel.shape == (2, 80, int(dur.sum(1).max()))
    assert lengths.tolist() == dur.sum(1).tolist()
    assert torch.all(log_dur[1, 6:] == 0)  # padding predicts nothing
    syn, d = m.synthesise(tok)
    assert syn.shape[0] == 2 and torch.all(d[1, 6:] == 0)


def test_tiny_model_overfits_two_clips():
    """Loss on two fixed clips must fall a lot: the training wiring works."""
    torch.manual_seed(0)
    vocab = TTSVocab(ALPHABET)
    texts = ["let there be light.", "and there was light."]
    clips, mels = [], {}
    for i, t in enumerate(texts):
        tokens = vocab.encode(t)
        durs = [3] * len(tokens)
        n = sum(durs)
        mels[f"c{i}"] = np.random.default_rng(i).normal(-5, 1, (80, n)).astype(np.float16)
        clips.append({"chapter": f"c{i}", "mel_start": 0, "mel_end": n, "tokens": tokens,
                      "durations": durs, "text": t})
    tok, dur, mel = collate([TTSDataset(clips, mels)[i] for i in range(2)])
    m = AcousticModel(len(vocab), **TINY)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    losses = []
    for _ in range(300):
        pred, log_dur, lengths = m(tok, dur)
        loss, _, _ = acoustic_loss(pred, mel, lengths, log_dur, dur, tok)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.3 * losses[0]
    # Learned durations should come back at synthesis time.
    _, d = m.eval().synthesise(tok[:1])
    assert abs(int(d.sum()) - int(dur[0].sum())) <= 4


def test_split_sentences():
    text = "in the beginning, god created. the earth was empty; and dark? yes!"
    assert split_sentences(text) == ["in the beginning, god created.", "the earth was empty;",
                                     "and dark?", "yes!"]
    long = ", ".join(["word word word"] * 30) + "."
    parts = split_sentences(long, max_chars=60)
    assert all(len(p) <= 60 for p in parts) and " ".join(parts) == long


def test_synthesise_text_runs_untrained():
    cfg = load_config(REPO / "configs" / "acoustic.yaml")
    data_cfg = load_config(REPO / "configs" / "data.yaml")
    vocab = TTSVocab(ALPHABET)
    m = AcousticModel(len(vocab), **TINY)
    mel = MelSpectrogram(MelConfig())
    synth = cfg["synthesis"] | {"griffin_lim_iters": 2}
    audio, sr, joined = synthesise_text(m, vocab, "In the beginning, God created. Amen!", mel,
                                        data_cfg["text"], synth, return_mel=True)
    assert sr == 22050 and audio.dtype == np.float32
    assert len(audio) == joined.shape[1] * 256
