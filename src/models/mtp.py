import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .rmsnorm import RMSNorm


class MTPHead(nn.Module):
    """
    Multi-Token Prediction Head (DeepSeek V3 style)

    ทำนาย token อนาคตที่ตำแหน่ง t + depth โดย:
      1. Concat hidden_state(t) + embed(token_(t + depth - 1))
      2. Project → hidden_size
      3. Norm → LM Head (shared weight กับ main lm_head)

    depth=1 → ทำนาย next+1  (เพิ่มเติมจาก main head)
    depth=2 → ทำนาย next+2
    """

    def __init__(self, config: dict, depth: int):
        super().__init__()
        self.depth  = depth
        hidden      = config["hidden_size"]
        vocab       = config["vocab_size"]

        self.norm    = RMSNorm(hidden, config.get("rms_norm_eps", 1e-5))
        self.proj    = nn.Linear(hidden * 2, hidden, bias=False)
        # lm_head weight จะถูก tie กับ main model ใน CustomLLM
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, T, hidden]
        future_embeds: torch.Tensor,   # [B, T, hidden] — embed(token_{t+depth-1})
    ) -> torch.Tensor:
        """Returns logits [B, T, vocab]"""
        normed   = self.norm(hidden_states)                        # [B, T, hidden]
        combined = torch.cat([normed, future_embeds], dim=-1)      # [B, T, hidden*2]
        x        = self.proj(combined)                             # [B, T, hidden]
        return self.lm_head(x)                                     # [B, T, vocab]


def compute_mtp_loss(
    mtp_logits_list: List[torch.Tensor],   # List of [B, T', vocab]
    input_ids:       torch.Tensor,         # [B, T]
    num_mtp_heads:   int,
    vocab_size:      int,
    lam:             float = 0.3,          # λ weight (DeepSeek V3 default)
    labels:          Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Compute weighted MTP loss รวมทุก depth

    Labels สำหรับ depth d:
      main head → labels[t]   = input_ids[t+1]
      MTP  d=1  → labels[t]   = input_ids[t+2]
      MTP  d=2  → labels[t]   = input_ids[t+3]

    Returns:
      total_mtp_loss  — weighted sum (scalar)
      per_head_losses — list of individual losses (for logging)
    """
    T                = input_ids.shape[1]
    per_head_losses  = []
    total_mtp_loss   = torch.tensor(0.0, device=input_ids.device)

    for depth, logits in enumerate(mtp_logits_list, start=1):
        # logits: [B, T - depth, vocab]
        # labels: input_ids shifted by (depth + 1)
        label_start = depth + 1
        label_end   = T

        if label_start >= label_end:
            # seq_len สั้นเกินไปสำหรับ depth นี้
            per_head_losses.append(torch.tensor(0.0, device=input_ids.device))
            continue

        target_source = labels if labels is not None else input_ids
        mtp_labels = target_source[:, label_start:label_end].contiguous()   # [B, T-depth-1]
        logits_trimmed = logits[:, :mtp_labels.shape[1], :].contiguous()

        loss = F.cross_entropy(
            logits_trimmed.reshape(-1, vocab_size),
            mtp_labels.reshape(-1),
            ignore_index=-100,
        )
        per_head_losses.append(loss)
        total_mtp_loss = total_mtp_loss + loss

    # weighted average
    if len(per_head_losses) > 0:
        total_mtp_loss = lam * (total_mtp_loss / num_mtp_heads)

    return total_mtp_loss, per_head_losses
