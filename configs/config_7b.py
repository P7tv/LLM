config_7b = {
    # ── Tokenizer ──────────────────────────────────────────────────
    "vocab_size":                151_669,

    # ── Dimensions ─────────────────────────────────────────────────
    "hidden_size":               4096,
    "num_hidden_layers":         32,

    # ── Attention ──────────────────────────────────────────────────
    "num_attention_heads":       32,
    "num_key_value_heads":       8,         # GQA 4:1
    "head_dim":                  128,       # 4096/32=128 ✅
    "qk_norm":                   True,
    "attn_pattern":              ["local", "local", "local", "global"] * 8,
    "sliding_window":            4096,

    # ── FFN ────────────────────────────────────────────────────────
    "intermediate_size":         14336,     # 3.5x — Llama3 style ✅
    "hidden_act":                "silu",

    # ── RoPE ───────────────────────────────────────────────────────
    "rope_theta":                1_000_000,
    "max_position_embeddings":   131_072,   # 128K

    # ── Norm ───────────────────────────────────────────────────────
    "rms_norm_eps":              1e-5,

    # ── Embedding ──────────────────────────────────────────────────
    "tie_word_embeddings":       False,

    # ── MTP ────────────────────────────────────────────────────────
    "num_mtp_heads":             1,
    "mtp_lambda":                0.3,
}
