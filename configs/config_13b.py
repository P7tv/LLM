config_13b = {
    # ── Tokenizer ──────────────────────────────────────────────────
    "vocab_size":                151_669,

    # ── Dimensions ─────────────────────────────────────────────────
    "hidden_size":               5120,
    "num_hidden_layers":         48,

    # ── Attention ──────────────────────────────────────────────────
    "num_attention_heads":       40,
    "num_key_value_heads":       8,         # GQA 5:1
    "head_dim":                  128,       # 5120/40=128 ✅
    "qk_norm":                   True,
    "attn_pattern":              ["local", "local", "local", "global"] * 12,
    "sliding_window":            4096,

    # ── FFN ────────────────────────────────────────────────────────
    "intermediate_size":         13824,     # 2.70x — Qwen2.5-14B proven
    "hidden_act":                "silu",

    # ── RoPE ───────────────────────────────────────────────────────
    "rope_theta":                1_000_000,
    "max_position_embeddings":   131_072,

    # ── Norm ───────────────────────────────────────────────────────
    "rms_norm_eps":              1e-5,

    # ── Embedding ──────────────────────────────────────────────────
    "tie_word_embeddings":       False,

    # ── MTP ────────────────────────────────────────────────────────
    "num_mtp_heads":             1,
    "mtp_lambda":                0.3,
}
