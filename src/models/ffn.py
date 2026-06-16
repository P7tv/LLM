import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network
    FFN(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

    ✅ Bug fixes vs v1:
       ❌ เดิม: down_proj shape = (hidden → intermediate)  ← ผิด
       ✅ ใหม่: down_proj shape = (intermediate → hidden)  ← ถูก

       ❌ เดิม: SiLU ใส่ที่ up_proj
       ✅ ใหม่: SiLU ใส่ที่ gate_proj  (SwiGLU definition จริง)
    """

    def __init__(self, config: dict):
        super().__init__()
        hidden = config["hidden_size"]
        inter  = config["intermediate_size"]

        # ✅ shape ถูกต้อง
        self.gate_proj = nn.Linear(hidden, inter,  bias=False)  # hidden → inter
        self.up_proj   = nn.Linear(hidden, inter,  bias=False)  # hidden → inter
        self.down_proj = nn.Linear(inter,  hidden, bias=False)  # inter  → hidden ✅

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ SiLU อยู่ที่ gate เท่านั้น
        gate = self.gate_proj(x)          # [B, T, inter]
        up   = self.up_proj(x)            # [B, T, inter]
        return self.down_proj(F.silu(gate) * up)   # [B, T, hidden]

    def extra_repr(self) -> str:
        return (
            f"hidden={self.gate_proj.in_features}, "
            f"intermediate={self.gate_proj.out_features}"
        )
