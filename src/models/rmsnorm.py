import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Pre-LN variant — วางก่อน Attention และ FFN ทุกครั้ง
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * norm.to(input_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"
