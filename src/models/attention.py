import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rmsnorm import RMSNorm
from .rope    import RotaryEmbedding, apply_rotary_emb


# ── FlashAttention-3 Import & Version Check ───────────────────────────────────
_FA3_AVAILABLE  = False
_flash_attn_fn  = None
_FA3_MIN        = (3, 0, 0)


def _parse_version(v: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in v.split(".")[:3])


_FA3_INITIALIZED = False

def _init_flash() -> None:
    global _FA3_AVAILABLE, _flash_attn_fn, _FA3_INITIALIZED
    if _FA3_INITIALIZED:
        return
    _FA3_INITIALIZED = True
    try:
        import flash_attn
        from flash_attn import flash_attn_func

        fa_ver = _parse_version(flash_attn.__version__)
        if fa_ver < _FA3_MIN:
            raise ImportError(
                f"FlashAttention >= 3.0.0 required for B200 SWA kernel, "
                f"got {flash_attn.__version__}. "
                f"Install: pip install flash-attn --no-build-isolation"
            )
        _flash_attn_fn  = flash_attn_func
        _FA3_AVAILABLE  = True
        print(f"[FlashAttention] v{flash_attn.__version__} ✅")
    except ImportError as e:
        print(f"[FlashAttention] ⚠️  {e}")
        print("[FlashAttention] Fallback to SDPA — max seq_len=8K for local layers")
        _FA3_AVAILABLE = False


# ── Sliding Window Mask (SDPA only — short seq) ──────────────────────────────
def _create_sliding_window_mask(
    seq_len:     int,
    window_size: int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Causal Sliding Window Mask
    ⚠️  ใช้ได้เฉพาะ seq_len < 8K เท่านั้น (memory = seq²)
    """
    row = torch.arange(seq_len, device=device).unsqueeze(1)
    col = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = col <= row
    window = (row - col) < window_size
    return causal & window                          # [T, T] bool


# ── Main Attention Module ─────────────────────────────────────────────────────
class HybridGQAAttention(nn.Module):
    """
    Grouped Query Attention with:
      ✅ QK-Norm (Qwen3)    — norm ก่อน RoPE เสมอ
      ✅ Hybrid Attention   — Local SWA / Global per layer pattern
      ✅ FA3 SWA kernel     — ไม่สร้าง explicit mask ที่ 128K
      ✅ SDPA fallback      — raise error ถ้า local + T > 8K

    Bug fixes vs v1:
      ❌ เดิม: q_norm(q_proj(x))              → norm บน hidden_size ทั้งก้อน
      ✅ ใหม่: reshape ก่อน → norm บน head_dim ทีละหัว

      ❌ เดิม: RoPE → QK-Norm
      ✅ ใหม่: QK-Norm → RoPE  (Qwen3 paper order)

      ❌ เดิม: explicit (T×T) mask → OOM ที่ 128K
      ✅ ใหม่: FA3 window_size kernel / raise error fallback
    """

    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        _init_flash()

        self.hidden_size    = config["hidden_size"]
        self.num_q_heads    = config["num_attention_heads"]
        self.num_kv_heads   = config["num_key_value_heads"]
        self.head_dim       = config["head_dim"]
        self.layer_idx      = layer_idx
        self.sliding_window = config.get("sliding_window", 4096)

        assert self.num_q_heads % self.num_kv_heads == 0, (
            f"Q heads ({self.num_q_heads}) must be divisible "
            f"by KV heads ({self.num_kv_heads})"
        )
        assert self.head_dim == self.hidden_size // self.num_q_heads, (
            f"head_dim mismatch: config says {self.head_dim}, "
            f"but hidden/Q_heads = {self.hidden_size // self.num_q_heads}"
        )

        self.gqa_groups = self.num_q_heads // self.num_kv_heads

        # ── Local vs Global ────────────────────────────────────────────
        pattern      = config.get(
            "attn_pattern",
            ["local", "local", "local", "global"],
        )
        self.is_local = pattern[layer_idx % len(pattern)] == "local"

        # ── Projections ────────────────────────────────────────────────
        self.q_proj = nn.Linear(self.hidden_size, self.num_q_heads  * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, self.hidden_size,  bias=False)

        # ── QK-Norm ────────────────────────────────────────────────────
        # ✅ dim = head_dim (ไม่ใช่ hidden_size)
        self.use_qk_norm = config.get("qk_norm", True)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # ── RoPE ───────────────────────────────────────────────────────
        self.rotary_emb = RotaryEmbedding(
            dim         = self.head_dim,
            max_seq_len = config.get("max_position_embeddings", 131_072),
            theta       = config.get("rope_theta", 1_000_000.0),
        )

    # ─────────────────────────────────────────────────────────────────
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match Q heads (GQA)"""
        # x: [B, kv_heads, T, head_dim]
        if self.gqa_groups == 1:
            return x
        B, kv, T, hd = x.shape
        x = x.unsqueeze(2).expand(B, kv, self.gqa_groups, T, hd)
        return x.reshape(B, kv * self.gqa_groups, T, hd)

    # ─────────────────────────────────────────────────────────────────
    def _flash_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        FlashAttention-3 path
        Input  : [B, heads, T, head_dim]
        Output : [B, heads, T, head_dim]
        FA3 expects [B, T, heads, head_dim]
        """
        q = q.transpose(1, 2)   # [B, T, heads, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ✅ FA3 SWA kernel — ไม่มีการสร้าง (T×T) matrix เลย
        window = (self.sliding_window, 0) if self.is_local else (-1, -1)

        out = _flash_attn_fn(
            q, k, v,
            dropout_p   = 0.0,
            causal      = True,
            window_size = window,
        )
        return out.transpose(1, 2)   # [B, heads, T, hd]

    # ─────────────────────────────────────────────────────────────────
    def _sdpa_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """
        PyTorch SDPA fallback
        ✅ ป้องกัน OOM: raise error ถ้า local + T > 8192
        """
        if self.is_local:
            if T > 8_192:
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"[LOCAL Attention] OOM Prevention — seq_len={T:,} > 8,192\n"
                    f"Explicit mask size = {T}² × 2B = {T*T*2/1e9:.1f} GB\n"
                    f"Solution: ติดตั้ง FlashAttention-3\n"
                    f"  pip install flash-attn --no-build-isolation\n"
                    f"{'='*60}"
                )
            # Safe for T <= 8K
            mask = _create_sliding_window_mask(T, self.sliding_window, q.device)
            bias = torch.zeros(T, T, device=q.device, dtype=q.dtype)
            bias = bias.masked_fill_(~mask, float("-inf"))
            bias = bias.unsqueeze(0).unsqueeze(0)   # [1, 1, T, T]
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = bias,
                dropout_p = 0.0,
                is_causal = False,
            )
        else:
            # Global Full Attention
            return F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = 0.0,
                is_causal = True,
            )

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        x:            torch.Tensor,                    # [B, T, hidden]
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, T, _ = x.shape

        # 1. Project Q K V
        q = self.q_proj(x)   # [B, T, q_heads * head_dim]
        k = self.k_proj(x)   # [B, T, kv_heads * head_dim]
        v = self.v_proj(x)   # [B, T, kv_heads * head_dim]

        # 2. Reshape → [B, heads, T, head_dim]
        q = q.view(B, T, self.num_q_heads,  self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ✅ 3. QK-Norm BEFORE RoPE (Qwen3 paper order)
        #    ❌ เดิม: RoPE → norm  (norm ล้าง positional info)
        #    ✅ ใหม่: norm → RoPE  (magnitude stable ก่อนหมุน)
        if self.use_qk_norm:
            q = self.q_norm(q)   # norm ทีละ head [B, heads, T, head_dim]
            k = self.k_norm(k)

        # 4. Apply RoPE AFTER QK-Norm
        cos, sin = self.rotary_emb(T, position_ids)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # ✅ 6. Hybrid Attention — FA3 kernel หรือ SDPA fallback
        if _FA3_AVAILABLE:
            # FlashAttention-3 handles GQA natively without repeating KV heads
            out = self._flash_forward(q, k, v)
        else:
            # SDPA fallback requires repeating KV heads
            k = self._repeat_kv(k)   # [B, q_heads, T, head_dim]
            v = self._repeat_kv(v)
            out = self._sdpa_forward(q, k, v, T)

        # 7. Reshape + Output projection
        out = out.transpose(1, 2).contiguous().view(
            B, T, self.num_q_heads * self.head_dim
        )
        return self.o_proj(out)

    def extra_repr(self) -> str:
        return (
            f"layer={self.layer_idx}, "
            f"type={'LOCAL' if self.is_local else 'GLOBAL'}, "
            f"Q={self.num_q_heads}, KV={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"qk_norm={self.use_qk_norm}, "
            f"window={self.sliding_window if self.is_local else 'full'}"
        )
