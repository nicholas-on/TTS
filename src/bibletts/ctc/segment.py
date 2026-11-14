"""Cutting chapters into clips, before and after alignment, and turning an
alignment into per-token durations for the TTS model.

A "segment" is a dict describing one clip:
    id, chapter, split           e.g. "MAT_005_007", "MAT_005", "train"
    start_s, end_s               where the clip sits in the chapter audio
    mel_start, mel_end           the same, in mel frames
    word_start, word_end         which words of the chapter text it covers
    text                         cleaned text of those words
    score                        (aligned clips only) mean log-probability
    tokens, durations            (final round only) TTS tokens and frames each
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..text import TTSVocab
from .align import Alignment


@dataclass
class Word:
    text: str          # letters only
    punct: str         # punctuation after it ("" , ; . ? !)
    rec_start: int     # position of its first character in the recogniser text
    rec_end: int       # one past its last character
    piece_end: bool    # last word of a verse (or of the announcement)


@dataclass
class Cut:
    """A place a clip may start or end: a silence from start to end (seconds)."""
    start: float
    end: float


def chapter_words(meta: dict) -> list[Word]:
    """The chapter's words, in order, with positions in the recogniser text.

    The recogniser text is the words joined by single spaces, which is exactly
    text.for_recogniser(meta["text"]).
    """
    pieces = ([meta["announcement"]] if meta.get("announcement") else []) + \
             [v["clean"] for v in meta["verses"]]
    words: list[Word] = []
    pos = 0
    for piece in pieces:
        found = re.findall(r"([^ ,;.?!]+)([,;.?!]?)", piece)
        for i, (w, p) in enumerate(found):
            words.append(Word(w, p, pos, pos + len(w), i == len(found) - 1))
            pos += len(w) + 1
    return words


def recogniser_text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


def clip_text(words: list[Word]) -> str:
    return "".join(w.text + w.punct + " " for w in words).rstrip(" ")


def _speech_bounds(meta: dict, lead_tolerance_s: float = 0.05) -> tuple[Cut, Cut]:
    """Pseudo-cuts for the start and end of the chapter, skipping silence."""
    duration = meta["duration_s"]
    pauses = meta["pauses"]
    first = Cut(0.0, 0.0)
    last = Cut(duration, duration)
    if pauses and pauses[0][0] <= lead_tolerance_s:
        first = Cut(0.0, pauses[0][1])
    if pauses and pauses[-1][1] >= duration - lead_tolerance_s:
        last = Cut(pauses[-1][0], duration)
    return first, last


def _make_segment(meta: dict, words: list[Word], w0: int, w1: int, a: Cut, b: Cut,
                  edge_s: float, hop_s: float) -> dict:
    start = max(a.start, a.end - edge_s)
    end = min(b.end, b.start + edge_s)
    return {
        "id": f"{meta['id']}_{w0:04d}",
        "chapter": meta["id"],
        "split": meta["split"],
        "start_s": round(start, 3),
        "end_s": round(end, 3),
        "mel_start": int(round(start / hop_s)),
        "mel_end": int(round(end / hop_s)),
        "word_start": w0,
        "word_end": w1,
        "text": clip_text(words[w0:w1]),
        "speech_end_s": round(b.start, 3),
    }


# ---------------------------------------------------------------------------
# Round 0: no model yet
# ---------------------------------------------------------------------------


def rough_segments(meta: dict, seg_cfg: dict, hop_s: float) -> list[dict]:
    """Cut a chapter into ~25 s clips without any model.

    Assumes the reader keeps a constant pace, so the time a verse ends is
    proportional to the characters before it. Cuts are placed at the detected
    pause closest to a predicted verse end. Clip edges may be off by a few
    words; that is acceptable for training a first recogniser.
    """
    words = chapter_words(meta)
    if not words:
        return []
    first, last = _speech_bounds(meta)
    n_chars = words[-1].rec_end
    rate = n_chars / max(last.start - first.end, 1e-3)          # characters per second
    predicted = {i: first.end + w.rec_end / rate for i, w in enumerate(words) if w.piece_end}
    pauses = [Cut(s, e) for s, e in meta["pauses"]]
    mids = np.array([(p.start + p.end) / 2 for p in pauses]) if pauses else np.array([])

    lo, target, hi = seg_cfg["rough_min_s"], seg_cfg["rough_target_s"], seg_cfg["rough_max_s"]
    tol = seg_cfg["rough_pause_tolerance_s"]
    edge = seg_cfg["edge_silence_s"]

    segments = []
    cur_word, cur_cut = 0, first
    while cur_word < len(words):
        remaining = last.start - cur_cut.end
        if remaining <= hi:
            segments.append(_make_segment(meta, words, cur_word, len(words), cur_cut, last, edge, hop_s))
            break
        best = None
        for widen in (0, 10, 20):
            for i, t in predicted.items():
                if i < cur_word or not (lo <= t - cur_cut.end <= hi + widen) or not len(mids):
                    continue
                j = int(np.argmin(np.abs(mids - t)))
                miss = abs(mids[j] - t)
                if miss > tol or mids[j] <= cur_cut.end:
                    continue
                key = (miss, abs(t - cur_cut.end - target))
                if best is None or key < best[0]:
                    best = (key, i, pauses[j])
            if best:
                break
        if best is None:  # no usable pause: keep the rest as one clip
            segments.append(_make_segment(meta, words, cur_word, len(words), cur_cut, last, edge, hop_s))
            break
        _, i, cut = best
        segments.append(_make_segment(meta, words, cur_word, i + 1, cur_cut, cut, edge, hop_s))
        cur_word, cur_cut = i + 1, cut
    return segments


# ---------------------------------------------------------------------------
# Rounds 1 and 2: cut at aligned sentence or verse ends
# ---------------------------------------------------------------------------


def aligned_segments(meta: dict, words: list[Word], alignment: Alignment, frame_s: float,
                     seg_cfg: dict, hop_s: float) -> list[dict]:
    """Cut a chapter at boundaries where the alignment puts a detected pause.

    A boundary qualifies if it is in seg_cfg["cut_at"] (punctuation marks,
    or "verse" for any verse end) and a detected pause overlaps the gap
    between the last character of one word and the first of the next.
    Clips are chosen greedily, each as close to target_s as possible within
    [min_s, max_s]. Stretches with no qualifying boundary within max_s are
    dropped.
    """
    spans = alignment.spans() * frame_s     # seconds, [n_chars, 2]
    pauses = [Cut(s, e) for s, e in meta["pauses"]]
    allowed = set(seg_cfg["cut_at"])
    first, last = _speech_bounds(meta)

    candidates: list[tuple[int, Cut]] = []   # (index of first word after the cut, cut)
    for i, w in enumerate(words[:-1]):
        if not (w.punct in allowed or (w.piece_end and "verse" in allowed)):
            continue
        gap_start = spans[w.rec_end - 1, 1] + frame_s
        gap_end = spans[words[i + 1].rec_start, 0]
        overlaps = [(min(p.end, gap_end) - max(p.start, gap_start), p) for p in pauses
                    if p.start < gap_end and p.end > gap_start]
        if overlaps:
            candidates.append((i + 1, max(overlaps, key=lambda o: o[0])[1]))
    candidates.append((len(words), last))

    lo, target, hi = seg_cfg["min_s"], seg_cfg["target_s"], seg_cfg["max_s"]
    edge = seg_cfg["edge_silence_s"]

    def length(a: Cut, b: Cut) -> float:
        return min(b.end, b.start + edge) - max(a.start, a.end - edge)

    def make(w0: int, w1: int, a: Cut, b: Cut) -> dict:
        # Edges come from the pauses, but must also contain every character
        # of the clip: a CTC spike can sit slightly inside a detected pause.
        first_spike = spans[words[w0].rec_start, 0]
        last_spike_end = spans[words[w1 - 1].rec_end - 1, 1] + frame_s
        seg = _make_segment(meta, words, w0, w1, a, b, edge, hop_s)
        seg["start_s"] = round(min(seg["start_s"], first_spike), 3)
        seg["end_s"] = round(max(seg["end_s"], last_spike_end + frame_s), 3)
        seg["speech_end_s"] = round(max(seg["speech_end_s"], last_spike_end), 3)
        seg["mel_start"] = int(np.floor(seg["start_s"] / hop_s))
        seg["mel_end"] = int(np.ceil(seg["end_s"] / hop_s))
        return seg

    segments = []
    cur_word, cur_cut = 0, first
    k = 0
    while k < len(candidates):
        options = [(abs(length(cur_cut, c) - target), j) for j, (_, c) in enumerate(candidates)
                   if j >= k and lo <= length(cur_cut, c) <= hi]
        if options:
            _, j = min(options)
            w_end, cut = candidates[j]
            seg = make(cur_word, w_end, cur_cut, cut)
            a, b = int(seg["start_s"] / frame_s), int(np.ceil(seg["end_s"] / frame_s))
            seg["score"] = round(float(alignment.frame_scores[a:b].mean()), 4)
            segments.append(seg)
            cur_word, cur_cut, k = w_end, cut, j + 1
        else:
            # Nothing fits: skip to the first candidate that is at least min_s on.
            later = [j for j in range(k, len(candidates)) if length(cur_cut, candidates[j][1]) > hi]
            if not later:
                break
            cur_word, cur_cut = candidates[later[0]][0], candidates[later[0]][1]
            k = later[0] + 1
    return segments


def drop_worst(segments: list[dict], fraction: float) -> tuple[list[dict], list[dict]]:
    """Remove the lowest-scoring fraction of clips in each split."""
    kept, dropped = [], []
    for split in sorted({s["split"] for s in segments}):
        group = sorted((s for s in segments if s["split"] == split), key=lambda s: s["score"])
        n = int(round(len(group) * fraction))
        dropped += group[:n]
        kept += group[n:]
    order = {s["id"]: i for i, s in enumerate(segments)}
    return sorted(kept, key=lambda s: order[s["id"]]), dropped


def cut_shifts(old: list[dict], new: list[dict]) -> np.ndarray:
    """For every clip end in `new`, distance (s) to the nearest clip end in `old`
    in the same chapter. Small values mean the alignment has settled."""
    by_chapter: dict[str, list[float]] = {}
    for s in old:
        by_chapter.setdefault(s["chapter"], []).append(s["speech_end_s"])
    shifts = []
    for s in new:
        ends = by_chapter.get(s["chapter"])
        if ends:
            shifts.append(min(abs(e - s["speech_end_s"]) for e in ends))
    return np.array(shifts)


# ---------------------------------------------------------------------------
# Durations for the TTS model
# ---------------------------------------------------------------------------


def tts_durations(seg: dict, words: list[Word], alignment: Alignment, vocab: TTSVocab,
                  hop_s: float, frame_s: float) -> tuple[list[int], list[int]] | None:
    """Tokens and mel frames per token for one clip, or None if it cannot be done.

    The alignment must be at the mel frame rate (a stride-1 model). Each
    character owns the frames from its own spike to the next character's
    spike. <bos> owns the leading silence; the last character runs until
    speech ends; <eos> owns the trailing silence; final punctuation gets 0.
    Durations always sum to the clip's number of mel frames.
    """
    if abs(frame_s - hop_s) > 1e-9:
        raise ValueError("Durations need a full-rate alignment (stride 1)")
    w0, w1 = seg["word_start"], seg["word_end"]
    c0, c1 = words[w0].rec_start, words[w1 - 1].rec_end
    f0, n = seg["mel_start"], seg["mel_end"] - seg["mel_start"]

    spikes = alignment.spans()[c0:c1, 0] - f0
    if np.any(spikes < 0) or np.any(spikes >= n):
        return None
    # Each spike must be at least one frame after the previous one.
    for i in range(1, len(spikes)):
        spikes[i] = max(spikes[i], spikes[i - 1] + 1)
    speech_end = int(round(seg["speech_end_s"] / hop_s)) - f0
    last_dur = max(1, speech_end - int(spikes[-1]))
    char_durs = list(np.diff(spikes)) + [last_dur]

    tokens = vocab.encode(seg["text"])
    has_final_punct = seg["text"][-1:] in ",;.?!"
    expected = 1 + (c1 - c0) + int(has_final_punct) + 1
    if len(tokens) != expected:
        raise AssertionError(f"token mismatch in {seg['id']}: {len(tokens)} vs {expected}")

    durs = [int(spikes[0])] + [int(d) for d in char_durs] + ([0] if has_final_punct else [])
    eos = n - sum(durs)
    if eos < 0:
        return None
    return tokens, durs + [eos]
