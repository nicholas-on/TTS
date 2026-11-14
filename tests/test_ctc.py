import itertools
import math

import numpy as np
import pytest
import torch

from bibletts.config import load_config
from bibletts.ctc.align import forced_align
from bibletts.ctc.data import collate, length_batches, spec_augment
from bibletts.ctc.model import CTCRecogniser, chapter_log_probs, greedy_decode
from bibletts.ctc.segment import (aligned_segments, chapter_words, clip_text, cut_shifts,
                                  drop_worst, recogniser_text, rough_segments, tts_durations)
from bibletts.evaluate import cer, edit_distance, wer
from bibletts.text import CTCVocab, TTSVocab, for_recogniser

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = load_config(REPO / "configs" / "ctc.yaml")
ALPHABET = "abcdefghijklmnopqrstuvwxyz'"
HOP = 256 / 22050


# ---------------------------------------------------------------- alignment


def brute_force_best_path(log_probs, targets, blank=0):
    """Best CTC path by trying every frame labelling (tiny inputs only)."""
    T, V = log_probs.shape
    best, best_score = None, -math.inf
    for labels in itertools.product(range(V), repeat=T):
        collapsed = [k for k, _ in itertools.groupby(labels) if k != blank]
        if collapsed != list(targets):
            continue
        score = sum(log_probs[t, labels[t]] for t in range(T))
        if score > best_score:
            best, best_score = labels, score
    return best, best_score


@pytest.mark.parametrize("targets", [[1, 2], [1, 1], [2, 1, 2]])
def test_forced_align_matches_brute_force(targets):
    rng = np.random.default_rng(len(targets))
    T, V = 6, 3
    lp = np.log(rng.dirichlet(np.ones(V), size=T)).astype(np.float32)
    al = forced_align(lp, targets)
    _, best_score = brute_force_best_path(lp, targets)
    assert al.frame_scores.sum() == pytest.approx(best_score, abs=1e-4)
    # The path, collapsed, spells the targets.
    tok = al.token_of_frame
    labels = [targets[t] if t >= 0 else 0 for t in tok]
    collapsed = [k for k, _ in itertools.groupby(zip(labels, al.states)) if k[0] != 0]
    assert [k[0] for k in collapsed] == targets


def test_forced_align_finds_planted_timing():
    """Characters planted at known frames are recovered there."""
    T, V = 40, 5
    lp = np.full((T, V), math.log(0.01), dtype=np.float32)
    lp[:, 0] = math.log(0.96)
    planted = {5: 1, 12: 2, 13: 2, 25: 3, 33: 4}   # frame -> token; token 2 spans two frames
    targets = [1, 2, 3, 4]
    for f, k in planted.items():
        lp[f, :] = math.log(0.01)
        lp[f, k] = math.log(0.96)
    spans = forced_align(lp, targets).spans()
    assert spans[:, 0].tolist() == [5, 12, 25, 33]
    assert spans[1].tolist() == [12, 13]


def test_forced_align_rejects_impossible():
    with pytest.raises(ValueError):
        forced_align(np.zeros((2, 3), np.float32), [1, 2, 1])


# ---------------------------------------------------------------- model


def test_model_shapes_and_windows():
    torch.manual_seed(0)
    for stride in (1, 2):
        m = CTCRecogniser(29, stride=stride, conv_channels=32, gru_hidden=16, gru_layers=2).eval()
        mel = torch.randn(1, 80, 301)
        full = m(mel)[0]
        assert full.shape == (math.ceil(301 / stride), 29)
        windowed = chapter_log_probs(m, mel[0], window_frames=100, context_frames=20)
        assert windowed.shape == full.shape


def test_greedy_decode_merges_and_drops_blanks():
    lp = torch.log(torch.eye(4)[[0, 1, 1, 0, 1, 2, 2, 3]] + 1e-6)
    assert greedy_decode(lp) == [1, 1, 2, 3]


def test_tiny_model_learns_to_overfit():
    """CTC loss on two fixed examples must fall: the training wiring works."""
    torch.manual_seed(0)
    vocab = CTCVocab(ALPHABET)
    m = CTCRecogniser(len(vocab), stride=2, conv_channels=32, gru_hidden=32, gru_layers=1, dropout=0)
    mels = [torch.randn(80, 120), torch.randn(80, 90)]
    texts = ["in the beginning", "god said"]
    batch = collate([(x, torch.tensor(vocab.encode(t))) for x, t in zip(mels, texts)])
    mel, lengths, targets, target_lengths = batch
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    loss_fn = torch.nn.CTCLoss(zero_infinity=True)
    losses = []
    for _ in range(150):
        lp = m(mel, lengths)
        loss = loss_fn(lp.transpose(0, 1), targets, m.output_lengths(lengths), target_lengths)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.3 * losses[0]


def test_length_batches_and_specaugment():
    frames = [100, 900, 120, 880, 500, 510]
    batches = length_batches(frames, max_frames=1800, shuffle=False)
    assert sorted(i for b in batches for i in b) == list(range(6))
    for b in batches:
        assert max(frames[i] for i in b) * len(b) <= 1800 or len(b) == 1
    mel = torch.zeros(2, 80, 100)
    out = spec_augment(mel + torch.arange(80.0)[:, None], torch.tensor([100, 60]),
                       CFG["train"]["specaugment"])
    assert out.shape == mel.shape


# ---------------------------------------------------------------- segmentation


def fake_chapter():
    """Six verses; audio with a 0.5 s pause after each verse."""
    verses = ["in the beginning, god created the heavens.", "the earth was empty;",
              "god said, let there be light.", "and there was light.",
              "god saw the light, that it was good.", "he divided the light from the darkness."]
    words_per_s = 2.5
    t, pauses = 0.5, [(0.0, 0.5)]
    for v in verses:
        t += len(v.split()) / words_per_s
        pauses.append((round(t, 3), round(t + 0.5, 3)))
        t += 0.5
    return {"id": "GEN_001", "split": "train", "duration_s": round(t, 3),
            "announcement": "genesis, chapter one.",
            "verses": [{"verse": i + 1, "raw": v, "clean": v} for i, v in enumerate(verses)],
            "pauses": pauses}


def test_chapter_words_match_recogniser_text():
    meta = fake_chapter()
    words = chapter_words(meta)
    full = " ".join([meta["announcement"]] + [v["clean"] for v in meta["verses"]])
    assert recogniser_text(words) == for_recogniser(full)
    assert clip_text(words) == full
    for w in words:
        assert recogniser_text(words)[w.rec_start: w.rec_end] == w.text
    assert sum(w.piece_end for w in words) == 7


def test_rough_segments_cover_chapter_in_order():
    meta = fake_chapter()
    seg_cfg = CFG["segment"] | {"rough_min_s": 3, "rough_target_s": 5, "rough_max_s": 8}
    segs = rough_segments(meta, seg_cfg, HOP)
    assert len(segs) >= 2
    assert segs[0]["word_start"] == 0 and segs[-1]["word_end"] == len(chapter_words(meta))
    for a, b in zip(segs, segs[1:]):
        assert a["word_end"] == b["word_start"] and a["end_s"] <= b["start_s"]
    assert segs[0]["text"].startswith("genesis, chapter one.")


def synthetic_alignment(meta, words, frame_s):
    """An alignment placing each word evenly within its verse's speech time."""
    from bibletts.ctc.align import Alignment
    T = int(meta["duration_s"] / frame_s)
    rec = recogniser_text(words)
    states = np.full(T, -2, dtype=np.int64)   # -2 = not yet assigned
    # verse k spans pause[k].end .. pause[k+1].start; announcement shares verse 1's slot
    piece_ranges, current = [], []
    for i, w in enumerate(words):
        current.append(i)
        if w.piece_end:
            piece_ranges.append(current); current = []
    piece_ranges[0:2] = [piece_ranges[0] + piece_ranges[1]]
    for k, idx in enumerate(piece_ranges):
        start, end = meta["pauses"][k][1], meta["pauses"][k + 1][0]
        c0, c1 = words[idx[0]].rec_start, words[idx[-1]].rec_end
        if k + 1 < len(piece_ranges):
            c1 += 1   # the space before the next verse
        n = c1 - c0
        for j, c in enumerate(range(c0, c1)):
            f = int((start + (end - start) * j / n) / frame_s)
            states[f] = 2 * c + 1
    # fill the frames between spikes with the preceding blank state
    last = -1
    for t in range(T):
        if states[t] >= 0:
            last = states[t]
        else:
            states[t] = last + 1
    return Alignment(states=states, frame_scores=np.full(T, -0.1, np.float32), n_tokens=len(rec))


def test_aligned_segments_and_durations():
    meta = fake_chapter()
    words = chapter_words(meta)
    al = synthetic_alignment(meta, words, HOP)
    seg_cfg = CFG["segment"] | {"min_s": 1.0, "target_s": 4.0, "max_s": 7.0}
    segs = aligned_segments(meta, words, al, HOP, seg_cfg, HOP)
    assert segs, "no clips"
    for s in segs:
        assert seg_cfg["min_s"] <= s["end_s"] - s["start_s"] <= seg_cfg["max_s"]
        last_word = words[s["word_end"] - 1]
        assert last_word.punct in ".;?!" or last_word.piece_end
    vocab = TTSVocab(ALPHABET)
    for s in segs:
        tokens, durs = tts_durations(s, words, al, vocab, HOP, HOP)
        assert tokens == vocab.encode(s["text"])
        assert len(durs) == len(tokens)
        assert sum(durs) == s["mel_end"] - s["mel_start"]
        assert min(durs) >= 0 and all(d >= 1 for d in durs[1:-2])


def test_drop_worst_and_cut_shifts():
    segs = [{"id": str(i), "split": "train", "score": -i, "chapter": "c", "speech_end_s": i}
            for i in range(10)]
    kept, dropped = drop_worst(segs, 0.2)
    assert [s["id"] for s in dropped] == ["9", "8"] and len(kept) == 8
    moved = [s | {"speech_end_s": s["speech_end_s"] + 0.01} for s in segs]
    assert np.allclose(cut_shifts(segs, moved), 0.01)


# ---------------------------------------------------------------- evaluation


def test_error_rates():
    assert edit_distance("kitten", "sitting") == 3
    assert cer(["abc"], ["abc"]) == 0
    assert cer(["abcd"], ["abxd"]) == 0.25
    assert wer(["the light was good"], ["the night was good"]) == 0.25
