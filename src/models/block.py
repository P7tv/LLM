import torch
import torch.nn as nn
from typing import Optional

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
        x:            torch.Tensor,                    # [B, T, hidden]
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Attention sub-block
        x = x + self.attn(self.attn_norm(x), position_ids)

        # FFN sub-block
        x = x + self.ffn(self.ffn_norm(x))

        return x
