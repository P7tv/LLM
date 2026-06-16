import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    รองรับ Dynamic Extension — ถ้า seq_len > cache จะ rebuild อัตโนมัติ
    """
    def __init__(
        self,
        dim:         int,
        max_seq_len: int   = 131_072,
        theta:       float = 1_000_000.0,
    ):
        super().__init__()
        self.dim         = dim
        self.theta       = theta
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)       # [T, dim/2]
        emb   = torch.cat([freqs, freqs], dim=-1)   # [T, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len * 2)   # extend เผื่อไว้
        if position_ids is None:
            return (
                self.cos_cached[:seq_len],
                self.sin_cached[:seq_len],
            )
        else:
            # position_ids shape: [B, T]
            cos = self.cos_cached[position_ids]  # [B, T, dim]
            sin = self.sin_cached[position_ids]  # [B, T, dim]
            return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q:   torch.Tensor,
    k:   torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to Q and K
    q, k   : [B, heads, T, head_dim]
    cos/sin: [T, head_dim] or [B, T, head_dim]
    """
    cos = cos.to(dtype=q.dtype, device=q.device)
    sin = sin.to(dtype=q.dtype, device=q.device)
    
    if cos.ndim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, T, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.ndim == 3:
        cos = cos.unsqueeze(1)                # [B, 1, T, head_dim]
        sin = sin.unsqueeze(1)
        
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
