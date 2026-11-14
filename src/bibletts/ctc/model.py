"""The recogniser: log mel-spectrogram in, per-frame character probabilities out.

    mel [B, 80, T]
      -> normalise each mel band with statistics from the training set
      -> 2 x Conv1d (local patterns: formants, transitions); the first may
         halve the frame rate (stride 2)
      -> 4 x bidirectional GRU (context in both directions)
      -> Linear -> log-probabilities over [blank, space, letters...]

Trained with CTC loss, which sums over every way of spreading the target
characters across the frames, so no frame-level labels are needed.
"""

from __future__ import annotations

import torch
from torch import nn


class CTCRecogniser(nn.Module):
    def __init__(self, n_tokens: int, n_mels: int = 80, stride: int = 1,
                 conv_channels: int = 256, conv_kernel: int = 5, gru_hidden: int = 320,
                 gru_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.stride = stride
        pad = conv_kernel // 2
        self.register_buffer("mel_mean", torch.zeros(n_mels, 1))
        self.register_buffer("mel_std", torch.ones(n_mels, 1))
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, conv_channels, conv_kernel, stride=stride, padding=pad),
            nn.BatchNorm1d(conv_channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, conv_kernel, padding=pad),
            nn.BatchNorm1d(conv_channels), nn.GELU(), nn.Dropout(dropout),
        )
        self.gru = nn.GRU(conv_channels, gru_hidden, num_layers=gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.out = nn.Linear(2 * gru_hidden, n_tokens)

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mel_mean.copy_(mean.reshape(-1, 1))
        self.mel_std.copy_(std.reshape(-1, 1).clamp(min=1e-3))

    def output_lengths(self, mel_lengths: torch.Tensor) -> torch.Tensor:
        # Conv1d with kernel k, padding k//2 and stride s gives ceil(T / s) frames.
        return (mel_lengths + self.stride - 1) // self.stride

    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """mel [B, n_mels, T] -> log-probabilities [B, T_out, n_tokens] (float32)."""
        x = (mel - self.mel_mean) / self.mel_std
        x = self.conv(x).transpose(1, 2)                       # [B, T_out, C]
        if mel_lengths is not None:
            lengths = self.output_lengths(mel_lengths).cpu()
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True,
                                                  enforce_sorted=False)
            x, _ = self.gru(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.gru(x)
        return self.out(x).float().log_softmax(dim=-1)


def greedy_decode(log_probs: torch.Tensor, length: int | None = None, blank: int = 0) -> list[int]:
    """Best token per frame, then merge repeats and remove blanks."""
    best = log_probs[:length].argmax(dim=-1).tolist()
    out, prev = [], None
    for t in best:
        if t != prev and t != blank:
            out.append(t)
        prev = t
    return out


@torch.no_grad()
def chapter_log_probs(model: CTCRecogniser, mel: torch.Tensor, window_frames: int,
                      context_frames: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Run a whole chapter through the model in overlapping windows.

    The model is trained on clips of seconds, not minutes, so a chapter is
    processed as windows of window_frames with context_frames of extra audio
    either side. Only each window's middle part is kept. Returns [T_out, V].
    """
    model.eval()
    s = model.stride
    window_frames = max(s, window_frames // s * s)        # keep windows on the output grid
    context_frames = context_frames // s * s
    total = mel.shape[-1]
    pieces = []
    for start in range(0, total, window_frames):
        a = max(0, start - context_frames)
        b = min(total, start + window_frames + context_frames)
        lp = model(mel[None, :, a:b].to(device).float())[0]
        keep_from = (start - a) // s
        keep_to = keep_from + model.output_lengths(torch.tensor(min(window_frames, total - start))).item()
        pieces.append(lp[keep_from:keep_to].cpu())
    return torch.cat(pieces)
