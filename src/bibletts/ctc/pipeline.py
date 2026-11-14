"""The steps of milestone 2, as functions the notebook calls in order."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from ..text import CTCVocab, TTSVocab
from .align import Alignment, forced_align
from .model import CTCRecogniser, chapter_log_probs
from .segment import (Word, aligned_segments, chapter_words, drop_worst, recogniser_text,
                      rough_segments, tts_durations)


def save_segments(segments: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in segments:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def load_segments(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def make_rough_segments(metas: list[dict], seg_cfg: dict, hop_s: float) -> list[dict]:
    segments = []
    for m in metas:
        segments += rough_segments(m, seg_cfg, hop_s)
    return segments


def align_chapters(model: CTCRecogniser, metas: list[dict], mels: dict, vocab: CTCVocab,
                   align_cfg: dict, hop_s: float, device: str | None = None
                   ) -> tuple[dict[str, tuple[list[Word], Alignment]], dict[str, str]]:
    """Force-align every chapter's full text to its full audio.

    Returns ({chapter id: (words, alignment)}, {chapter id: error message}).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    window = int(align_cfg["window_s"] / hop_s)
    context = int(align_cfg["context_s"] / hop_s)
    results, failures = {}, {}
    for m in tqdm(metas, desc="aligning"):
        words = chapter_words(m)
        targets = vocab.encode(recogniser_text(words))
        mel = torch.from_numpy(mels[m["id"]].astype("float32"))
        lp = chapter_log_probs(model, mel, window, context, device).numpy()
        try:
            results[m["id"]] = (words, forced_align(lp, targets, vocab.blank))
        except ValueError as e:
            failures[m["id"]] = str(e)
    return results, failures


def make_aligned_segments(metas: list[dict], alignments: dict, seg_cfg: dict, hop_s: float,
                          stride: int) -> tuple[list[dict], list[dict]]:
    """Clips cut at aligned boundaries, minus the worst-scoring few percent.
    Returns (kept, dropped)."""
    frame_s = hop_s * stride
    segments = []
    for m in metas:
        if m["id"] in alignments:
            words, alignment = alignments[m["id"]]
            segments += aligned_segments(m, words, alignment, frame_s, seg_cfg, hop_s)
    return drop_worst(segments, seg_cfg["drop_fraction"])


def add_durations(segments: list[dict], alignments: dict, tts_vocab: TTSVocab, hop_s: float,
                  stride: int) -> tuple[list[dict], list[str]]:
    """Attach TTS tokens and per-token durations to each clip.
    Returns (clips with durations, ids of clips where it failed)."""
    out, failed = [], []
    for s in segments:
        words, alignment = alignments[s["chapter"]]
        result = tts_durations(s, words, alignment, tts_vocab, hop_s, hop_s * stride)
        if result is None:
            failed.append(s["id"])
            continue
        tokens, durations = result
        out.append(s | {"tokens": tokens, "durations": durations})
    return out, failed
