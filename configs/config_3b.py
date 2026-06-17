config_3b = {
    # ── Tokenizer ──────────────────────────────────────────────────
    "vocab_size":                151_669,   # Typhoon 2.5

    # ── Dimensions ─────────────────────────────────────────────────
    "hidden_size":               3072,
    "num_hidden_layers":         28,

    # ── Attention ──────────────────────────────────────────────────
    "num_attention_heads":       24,        # Q heads
    "num_key_value_heads":       8,         # KV heads (GQA 3:1)
    "head_dim":                  128,       # 3072/24=128 ✅
    "qk_norm":                   True,      # ✅ Qwen3 style
    "attn_pattern":              ["local", "local", "local", "global"] * 7,
    "sliding_window":            4096,

    # ── FFN ────────────────────────────────────────────────────────
    "intermediate_size":         8192,      # 2.67x hidden
    "hidden_act":                "silu",

    # ── RoPE ───────────────────────────────────────────────────────
    "rope_theta":                500_000,
    "max_position_embeddings":   32_768,    # 32K
    "max_seq_len_sft":           4096,      # SFT/DPO seq cap

    # ── Norm ───────────────────────────────────────────────────────
    "rms_norm_eps":              1e-5,

    # ── Embedding ──────────────────────────────────────────────────
    "tie_word_embeddings":       False,     # LM Head แยกชิ้น

    # ── MTP ────────────────────────────────────────────────────────
    "num_mtp_heads":             1,         # predict t+1
    "mtp_lambda":                0.3,       # DeepSeek V3 default (tune later)
}
