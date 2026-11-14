"""Error rates. Character error rate (CER) is the main accuracy number."""

from __future__ import annotations


def edit_distance(ref, hyp) -> int:
    """Minimum insertions, deletions and substitutions turning hyp into ref."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_rate(refs: list, hyps: list) -> float:
    """Total edits divided by total reference length, over many pairs."""
    edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total = sum(len(r) for r in refs)
    return edits / max(total, 1)


def cer(refs: list[str], hyps: list[str]) -> float:
    return error_rate(refs, hyps)


def wer(refs: list[str], hyps: list[str]) -> float:
    return error_rate([r.split() for r in refs], [h.split() for h in hyps])
