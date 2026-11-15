"""The acoustic model: TTS tokens in, mel-spectrogram out (FastSpeech-style).

    tokens [B, N]
      -> embedding + position encoding
      -> encoder: transformer blocks, each token sees the whole sentence
      -> duration predictor: how many mel frames each token lasts
      -> length regulator: repeat each token's vector for its frames
      -> decoder: transformer blocks over frames
      -> linear -> mel [B, n_mels, T]

During training the durations come from the alignment (milestone 2), so the
model never has to learn alignment itself. At synthesis time the duration
predictor supplies them. Because every token gets its own block of frames, the
model cannot skip or repeat words.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoid_positions(length: int, dim: int, device=None) -> torch.Tensor:
    """Fixed position encoding (Vaswani et al. 2017): [length, dim]."""
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    angle = pos / torch.pow(10000.0, i / dim)
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


class FFTBlock(nn.Module):
    """Self-attention, then a convolutional feed-forward layer. Each with a
    residual connection and layer norm. Convolutions instead of the usual
    linear feed-forward help with neighbouring sounds (FastSpeech)."""

    def __init__(self, dim: int, heads: int, ffn_dim: int, kernel: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv1d(dim, ffn_dim, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(ffn_dim, dim, 1)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x [B, L, D]; pad_mask [B, L] True where padding."""
        a, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = self.norm1(x + self.dropout(a))
        f = self.conv2(F.relu(self.conv1(x.transpose(1, 2)))).transpose(1, 2)
        x = self.norm2(x + self.dropout(f))
        return x.masked_fill(pad_mask[..., None], 0.0)


class DurationPredictor(nn.Module):
    """Predicts log(1 + frames) for each token from the encoder output."""

    def __init__(self, dim: int, hidden: int, kernel: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, hidden, kernel, padding=kernel // 2)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.norm2 = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.norm1(F.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)))
        h = self.dropout(self.norm2(F.relu(self.conv2(h.transpose(1, 2))).transpose(1, 2)))
        return self.out(h).squeeze(-1).masked_fill(pad_mask, 0.0)


def length_regulate(x: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat each token vector by its duration.

    x [B, N, D], durations [B, N] (integers) -> frames [B, T, D], lengths [B].
    """
    lengths = durations.sum(dim=1)
    T = max(int(lengths.max()), 1)
    out = x.new_zeros(x.shape[0], T, x.shape[2])
    for b in range(x.shape[0]):
        expanded = torch.repeat_interleave(x[b], durations[b], dim=0)
        out[b, : expanded.shape[0]] = expanded
    return out, lengths


class AcousticModel(nn.Module):
    def __init__(self, n_tokens: int, n_mels: int = 80, dim: int = 256, heads: int = 2,
                 ffn_dim: int = 1024, kernel: int = 9, encoder_layers: int = 4,
                 decoder_layers: int = 4, duration_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(n_tokens, dim, padding_idx=0)
        self.encoder = nn.ModuleList(FFTBlock(dim, heads, ffn_dim, kernel, dropout)
                                     for _ in range(encoder_layers))
        self.duration = DurationPredictor(dim, duration_hidden, 3, dropout)
        self.decoder = nn.ModuleList(FFTBlock(dim, heads, ffn_dim, kernel, dropout)
                                     for _ in range(decoder_layers))
        self.to_mel = nn.Linear(dim, n_mels)

    def encode(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = tokens == 0
        x = self.embed(tokens) * math.sqrt(self.dim)
        x = x + sinusoid_positions(tokens.shape[1], self.dim, tokens.device)
        for block in self.encoder:
            x = block(x, pad)
        return x, pad

    def decode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        pad = torch.arange(x.shape[1], device=x.device)[None] >= lengths[:, None]
        x = x + sinusoid_positions(x.shape[1], self.dim, x.device)
        for block in self.decoder:
            x = block(x, pad)
        return self.to_mel(x).transpose(1, 2)            # [B, n_mels, T]

    def forward(self, tokens: torch.Tensor, durations: torch.Tensor):
        """Training: decode with the given (aligned) durations.
        Returns (mel [B, n_mels, T], predicted log durations [B, N], mel lengths [B])."""
        x, pad = self.encode(tokens)
        log_dur = self.duration(x, pad)
        frames, lengths = length_regulate(x, durations)
        return self.decode(frames, lengths), log_dur, lengths

    @torch.no_grad()
    def synthesise(self, tokens: torch.Tensor, pace: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Synthesis: durations from the predictor. pace > 1 reads slower.
        Returns (mel [B, n_mels, T], durations [B, N])."""
        x, pad = self.encode(tokens)
        log_dur = self.duration(x, pad)
        durations = torch.clamp(torch.round((torch.exp(log_dur) - 1) * pace), min=0).long()
        durations = durations.masked_fill(pad, 0)
        frames, lengths = length_regulate(x, durations)
        return self.decode(frames, lengths), durations


def acoustic_loss(mel_pred: torch.Tensor, mel_target: torch.Tensor, mel_lengths: torch.Tensor,
                  log_dur_pred: torch.Tensor, durations: torch.Tensor, tokens: torch.Tensor):
    """Mean absolute error on real mel frames, plus squared error on log durations.
    Returns (total, mel loss, duration loss)."""
    T = mel_target.shape[-1]
    frame_mask = (torch.arange(T, device=mel_target.device)[None] < mel_lengths[:, None])[:, None]
    mel_loss = (torch.abs(mel_pred[..., :T] - mel_target) * frame_mask).sum() / (
        frame_mask.sum() * mel_target.shape[1])
    token_mask = tokens != 0
    dur_target = torch.log1p(durations.float())
    dur_loss = ((log_dur_pred - dur_target) ** 2 * token_mask).sum() / token_mask.sum()
    return mel_loss + dur_loss, mel_loss, dur_loss
