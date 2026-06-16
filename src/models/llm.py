import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

from .rmsnorm import RMSNorm
from .block   import TransformerBlock
from .mtp     import MTPHead, compute_mtp_loss


class CustomLLM(nn.Module):
    """
    Custom Thai LLM — Decoder-Only Frontier Architecture

    Features:
      ✅ Qwen3-style QK-Norm     (norm ก่อน RoPE)
      ✅ Gemma4-style Hybrid Attn (Local SWA / Global interleaved)
      ✅ GQA                      (Grouped Query Attention)
      ✅ SwiGLU FFN
      ✅ RMSNorm Pre-LN
      ✅ RoPE (theta configurable)
      ✅ MTP Heads (Multi-Token Prediction, optional)
      ✅ FlashAttention-3 (auto-detect + version check)
      ✅ tie_word_embeddings=False (LM Head แยกชิ้น)

    Bug fixes vs v1:
      ✅ QK-Norm order (norm → RoPE ไม่ใช่ RoPE → norm)
      ✅ down_proj direction (inter → hidden)
      ✅ SWA ไม่สร้าง explicit T×T mask ที่ 128K
      ✅ MTP weight tying กับ main lm_head
      ✅ Parameter count รวม LM Head แยกชิ้น
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config      = config
        self.vocab_size  = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers  = config["num_hidden_layers"]
        self.num_mtp     = config.get("num_mtp_heads", 0)
        self.mtp_lambda  = config.get("mtp_lambda", 0.3)

        # ── Token Embedding ───────────────────────────────────────────
        self.embed_tokens = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
        )

        # ── Transformer Blocks ────────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(self.num_layers)
        ])

        # ── Final Norm ────────────────────────────────────────────────
        self.norm = RMSNorm(
            self.hidden_size,
            config.get("rms_norm_eps", 1e-5),
        )

        # ── LM Head (แยกชิ้น — tie_word_embeddings=False) ────────────
        self.lm_head = nn.Linear(
            self.hidden_size,
            self.vocab_size,
            bias=False,
        )

        # ── MTP Heads ─────────────────────────────────────────────────
        if self.num_mtp > 0:
            self.mtp_heads = nn.ModuleList([
                MTPHead(config, depth=d + 1)
                for d in range(self.num_mtp)
            ])
            # ✅ Weight tying: MTP lm_head share weights กับ main lm_head
            for head in self.mtp_heads:
                head.lm_head.weight = self.lm_head.weight
        else:
            self.mtp_heads = None

        # ── Init Weights ──────────────────────────────────────────────
        self.apply(self._init_weights)
        # Scaled init สำหรับ residual projections (GPT-2 style)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(
                    p,
                    mean=0.0,
                    std=0.02 / (2 * self.num_layers) ** 0.5,
                )

        # ── Startup Report ────────────────────────────────────────────
        self._print_model_info()

    # ─────────────────────────────────────────────────────────────────
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ─────────────────────────────────────────────────────────────────
    def _print_model_info(self) -> None:
        total  = sum(p.numel() for p in self.parameters())
        unique = sum(
            p.numel() for p in {id(p): p for p in self.parameters()}.values()
        )   # หัก weight tying
        print(f"\n{'='*55}")
        print(f"  CustomLLM — Thai LLM v2.0")
        print(f"{'='*55}")
        print(f"  Total params  : {total  / 1e9:.3f}B")
        print(f"  Unique params : {unique / 1e9:.3f}B  (after weight tying)")
        print(f"  Vocab size    : {self.vocab_size:,}")
        print(f"  Hidden size   : {self.hidden_size}")
        print(f"  Num layers    : {self.num_layers}")
        print(f"  MTP heads     : {self.num_mtp}")

        # แสดง pattern ของทุก layer
        local_idx  = []
        global_idx = []
        for i, layer in enumerate(self.layers):
            (local_idx if layer.attn.is_local else global_idx).append(i)
        print(f"  LOCAL  layers : {local_idx}")
        print(f"  GLOBAL layers : {global_idx}")
        print(f"{'='*55}\n")

    # ─────────────────────────────────────────────────────────────────
    def _get_mtp_inputs(
        self,
        input_ids:    torch.Tensor,    # [B, T]
        hidden_states: torch.Tensor,   # [B, T, hidden]
        depth:        int,
    ):
        """
        เตรียม hidden + future_embeds สำหรับ MTP head ที่ depth d
        hidden  : [B, T-depth, hidden]  — trim หาง
        embeds  : [B, T-depth, hidden]  — embed ของ token_{t+depth-1}
        """
        T = input_ids.shape[1]

        # trim hidden states (ตัด token สุดท้าย depth ตัว)
        h_trimmed = hidden_states[:, :T - depth, :]       # [B, T-depth, hidden]

        # future token ids: token ที่ตำแหน่ง t+depth-1
        future_ids    = input_ids[:, depth:T]             # [B, T-depth]
        future_embeds = self.embed_tokens(future_ids)     # [B, T-depth, hidden]

        return h_trimmed, future_embeds

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:    torch.Tensor,                        # [B, T]
        labels:       Optional[torch.Tensor]  = None,      # [B, T]
        position_ids: Optional[torch.Tensor]  = None,
        return_dict:  bool                    = True,
    ) -> Dict[str, Any]:
        """
        Returns dict:
          loss          — total loss (main + MTP) ถ้ามี labels
          logits        — main logits [B, T, vocab]
          mtp_losses    — list of per-head MTP loss (for logging)
          hidden_states — final hidden states [B, T, hidden]
        """
        B, T = input_ids.shape

        # ── 1. Embedding ──────────────────────────────────────────────
        x = self.embed_tokens(input_ids)       # [B, T, hidden]

        # ── 2. Transformer Blocks ─────────────────────────────────────
        for layer in self.layers:
            x = layer(x, position_ids)         # [B, T, hidden]

        # ── 3. Final Norm ─────────────────────────────────────────────
        hidden = self.norm(x)                  # [B, T, hidden]

        # ── 4. Main LM Head → Logits ──────────────────────────────────
        logits = self.lm_head(hidden)          # [B, T, vocab]

        # ── 5. Loss Computation ───────────────────────────────────────
        main_loss  = None
        mtp_loss   = None
        mtp_losses = []

        if labels is not None:
            # Main causal LM loss — shift right
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            main_loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

            # MTP loss
            if self.mtp_heads is not None:
                mtp_logits_list = []
                for d, mtp_head in enumerate(self.mtp_heads, start=1):
                    h_trim, f_emb = self._get_mtp_inputs(input_ids, hidden, d)
                    mtp_logit     = mtp_head(h_trim, f_emb)   # [B, T-d, vocab]
                    mtp_logits_list.append(mtp_logit)

                mtp_loss, mtp_losses = compute_mtp_loss(
                    mtp_logits_list = mtp_logits_list,
                    input_ids       = input_ids,
                    num_mtp_heads   = self.num_mtp,
                    vocab_size      = self.vocab_size,
                    lam             = self.mtp_lambda,
                    labels          = labels,
                )

            total_loss = main_loss + (mtp_loss if mtp_loss is not None else 0.0)
        else:
            total_loss = None

        if return_dict:
            return {
                "loss":          total_loss,
                "main_loss":     main_loss,
                "mtp_loss":      mtp_loss,
                "mtp_losses":    mtp_losses,   # per-head สำหรับ logging
                "logits":        logits,
                "hidden_states": hidden,
            }
        return total_loss, logits

    # ─────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids:   torch.Tensor,         # [B, T]
        max_new_tokens: int   = 256,
        temperature:    float = 1.0,
        top_p:          float = 0.9,
        eos_token_id:   int   = 2,
    ) -> torch.Tensor:
        """
        Greedy + Top-p sampling (inference only)
        ไม่ใช้ KV-cache ยังในเวอร์ชันนี้ — Phase 2 เพิ่ม
        """
        was_training = self.training
        self.eval()
        try:
            generated = input_ids.clone()

            for _ in range(max_new_tokens):
                out    = self.forward(generated, return_dict=True)
                logits = out["logits"][:, -1, :]   # [B, vocab] — last token only

                # Temperature
                logits = logits / max(temperature, 1e-6)

                # Top-p (nucleus) sampling
                probs       = F.softmax(logits, dim=-1)
                sorted_p, sorted_idx = torch.sort(probs, descending=True)
                cumsum_p    = torch.cumsum(sorted_p, dim=-1)
                remove_mask = cumsum_p - sorted_p > top_p
                sorted_p[remove_mask] = 0.0
                sorted_p    = sorted_p / sorted_p.sum(dim=-1, keepdim=True)

                next_token  = torch.multinomial(sorted_p, num_samples=1)
                next_token  = sorted_idx.gather(-1, next_token)   # map back

                generated = torch.cat([generated, next_token], dim=1)

                # Stop ถ้าทุก sequence เจอ EOS
                if (next_token == eos_token_id).all():
                    break
        finally:
            self.train(was_training)

        return generated
