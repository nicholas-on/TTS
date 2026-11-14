"""CTC forced alignment: given the audio's per-frame character probabilities
and the known text, find the single most likely timing of every character.

This is the Viterbi algorithm on the CTC graph. The text "ab" becomes the
state sequence  blank a blank b blank. Each frame occupies one state, and
moving from frame to frame you may:
    - stay in the same state,
    - move to the next state,
    - skip a blank, if the characters either side of it differ ("aa" needs
      the blank in between, or it would read as one "a").
The path must start in the first blank or first character and end in the last
character or last blank. Dynamic programming finds the best path in
O(frames x states) time, keeping one back-pointer per cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEG_INF = -1e30


@dataclass
class Alignment:
    states: np.ndarray        # [T] state index per frame (even = blank, odd = character)
    frame_scores: np.ndarray  # [T] log-probability of the path at each frame
    n_tokens: int             # number of target characters

    @property
    def token_of_frame(self) -> np.ndarray:
        """[T] index of the target character each frame is in, or -1 for blank."""
        return np.where(self.states % 2 == 1, (self.states - 1) // 2, -1)

    def spans(self) -> np.ndarray:
        """[n_tokens, 2] first and last frame (inclusive) of each character's
        non-blank frames, its "spike"."""
        tok = self.token_of_frame
        frames = np.nonzero(tok >= 0)[0]
        ids = tok[frames]
        first = np.full(self.n_tokens, np.iinfo(np.int64).max)
        last = np.full(self.n_tokens, -1)
        np.minimum.at(first, ids, frames)
        np.maximum.at(last, ids, frames)
        return np.stack([first, last], axis=1)

    @property
    def score(self) -> float:
        """Mean log-probability per frame along the path."""
        return float(self.frame_scores.mean())


def forced_align(log_probs: np.ndarray, targets: list[int] | np.ndarray,
                 blank: int = 0) -> Alignment:
    """Best CTC path. log_probs [T, V] (log-softmax output), targets [L] (no blanks)."""
    log_probs = np.asarray(log_probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64)
    T, L = log_probs.shape[0], len(targets)
    S = 2 * L + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = targets
    if T < L + int(np.sum(targets[1:] == targets[:-1])):
        raise ValueError(f"{T} frames cannot hold {L} characters")

    # A skip from s-2 to s is allowed for characters that differ from the
    # character two states back.
    can_skip = np.zeros(S, dtype=bool)
    can_skip[3::2] = targets[1:] != targets[:-1]

    alpha = np.full(S, NEG_INF, dtype=np.float32)
    alpha[0] = log_probs[0, blank]
    if L:
        alpha[1] = log_probs[0, ext[1]]
    back = np.zeros((T, S), dtype=np.uint8)   # 0 = stay, 1 = from s-1, 2 = from s-2

    for t in range(1, T):
        stay = alpha
        step = np.concatenate(([NEG_INF], alpha[:-1]))
        skip = np.concatenate(([NEG_INF, NEG_INF], alpha[:-2]))
        skip = np.where(can_skip, skip, NEG_INF)
        best = np.maximum(stay, np.maximum(step, skip))
        choice = np.where(best == stay, 0, np.where(best == step, 1, 2)).astype(np.uint8)
        back[t] = choice
        alpha = best + log_probs[t, ext]

    end = S - 1 if L == 0 or alpha[S - 1] >= alpha[S - 2] else S - 2
    if alpha[end] <= NEG_INF / 2:
        raise ValueError("No valid alignment")

    states = np.empty(T, dtype=np.int64)
    s = end
    for t in range(T - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])
    frame_scores = log_probs[np.arange(T), ext[states]]
    return Alignment(states=states, frame_scores=frame_scores, n_tokens=L)
