import torch
import torch.nn as nn
from typing import Optional, Tuple

from .rmsnorm  import RMSNorm
from .attention import HybridGQAAttention
from .ffn       import SwiGLUFFN


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer Block
    Structure:
      x → RMSNorm → Attention → + (residual) → RMSNorm → FFN → + (residual)
    """

    def __init__(self, config: dict, layer_idx: int):
        super().__init__()
        eps = config.get("rms_norm_eps", 1e-5)

        self.attn_norm = RMSNorm(config["hidden_size"], eps)
        self.attn      = HybridGQAAttention(config, layer_idx)
        self.ffn_norm  = RMSNorm(config["hidden_size"], eps)
        self.ffn       = SwiGLUFFN(config)
        self.layer_idx = layer_idx

    def forward(
        self,
        x:              torch.Tensor,                    # [B, T, hidden]
        position_ids:   Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache:      bool = False,
    ):
        # Attention sub-block
        attn_out, present = self.attn(
            self.attn_norm(x), position_ids, past_key_value, use_cache
        )
        x = x + attn_out

        # FFN sub-block
        x = x + self.ffn(self.ffn_norm(x))

        return x, present
