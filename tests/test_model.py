"""
รันก่อนเสมอ:
  python -m pytest tests/test_model.py -v
หรือ:
  python tests/test_model.py
"""

import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.models import CustomLLM
from configs.config_3b import config_3b


# ── Mini config สำหรับ CPU test ──────────────────────────────────────
config_mini = {
    "vocab_size":              1000,
    "hidden_size":             256,
    "num_hidden_layers":       4,
    "num_attention_heads":     8,
    "num_key_value_heads":     2,
    "head_dim":                32,      # 256/8=32 ✅
    "qk_norm":                 True,
    "attn_pattern":            ["local", "local", "local", "global"],
    "sliding_window":          64,
    "intermediate_size":       512,
    "hidden_act":              "silu",
    "rope_theta":              10_000,
    "max_position_embeddings": 512,
    "rms_norm_eps":            1e-5,
    "tie_word_embeddings":     False,
    "num_mtp_heads":           1,
    "mtp_lambda":              0.3,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model():
    m = CustomLLM(config_mini).to(DEVICE)
    m.eval()
    return m


# ── Test 1: Param Count ───────────────────────────────────────────────
def test_param_count(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total:,}")
    assert total > 0
    assert total < 500_000_000   # mini model < 500M


# ── Test 2: Output Shape ──────────────────────────────────────────────
def test_output_shape(model):
    B, T = 2, 128
    ids = torch.randint(0, 1000, (B, T)).to(DEVICE)
    with torch.no_grad():
        out = model(ids)
    assert out["logits"].shape == (B, T, 1000), (
        f"Expected (2, 128, 1000), got {out['logits'].shape}"
    )
    print("\n✅ Output shape correct:", out["logits"].shape)


# ── Test 3: Forward + Loss ────────────────────────────────────────────
def test_forward_with_loss(model):
    model.train()
    B, T = 2, 128
    ids    = torch.randint(0, 1000, (B, T)).to(DEVICE)
    labels = ids.clone()

    out = model(ids, labels=labels)

    assert out["loss"] is not None
    assert torch.isfinite(out["loss"]), f"Loss is not finite: {out['loss']}"
    assert out["main_loss"] is not None
    assert out["mtp_loss"]  is not None
    print(f"\n✅ Loss     : {out['loss'].item():.4f}")
    print(f"   Main loss: {out['main_loss'].item():.4f}")
    print(f"   MTP  loss: {out['mtp_loss'].item():.4f}")


# ── Test 4: Backward ──────────────────────────────────────────────────
def test_backward(model):
    model.train()
    B, T = 2, 64
    ids    = torch.randint(0, 1000, (B, T)).to(DEVICE)
    labels = ids.clone()

    out = model(ids, labels=labels)
    out["loss"].backward()

    # ตรวจว่าทุก param มี gradient
    no_grad = [
        n for n, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert len(no_grad) == 0, f"Params without grad: {no_grad}"
    print(f"\n✅ Backward OK — all {sum(1 for p in model.parameters() if p.requires_grad)} params have gradients")


# ── Test 5: No NaN/Inf ────────────────────────────────────────────────
def test_no_nan_inf(model):
    model.train()
    B, T = 2, 64
    ids    = torch.randint(0, 1000, (B, T)).to(DEVICE)
    labels = ids.clone()

    out  = model(ids, labels=labels)
    loss = out["loss"]

    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"

    out["loss"].backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
            assert not torch.isinf(p.grad).any(), f"Inf grad in {name}"

    print("\n✅ No NaN/Inf in loss or gradients")


# ── Test 6: Hybrid Attention Pattern ──────────────────────────────────
def test_hybrid_attention_pattern():
    m = CustomLLM(config_mini)
    pattern = config_mini["attn_pattern"]
    for i, layer in enumerate(m.layers):
        expected = pattern[i % len(pattern)] == "local"
        actual   = layer.attn.is_local
        assert expected == actual, (
            f"Layer {i}: expected is_local={expected}, got {actual}"
        )
    print("\n✅ Attention pattern correct:")
    for i, layer in enumerate(m.layers):
        t = "LOCAL " if layer.attn.is_local else "GLOBAL"
        print(f"   Layer {i:02d}: {t}")


# ── Test 7: Weight Tying (MTP shares LM Head) ─────────────────────────
def test_weight_tying():
    m = CustomLLM(config_mini)
    if m.mtp_heads is not None:
        main_w = m.lm_head.weight.data_ptr()
        for i, mtp in enumerate(m.mtp_heads):
            mtp_w = mtp.lm_head.weight.data_ptr()
            assert main_w == mtp_w, (
                f"MTP head {i} weight NOT tied to main lm_head!"
            )
        print(f"\n✅ Weight tying OK — {len(m.mtp_heads)} MTP heads share lm_head")


# ── Test 8: QK-Norm dimension ────────────────────────────────────────
def test_qk_norm_dim():
    m = CustomLLM(config_mini)
    for i, layer in enumerate(m.layers):
        attn = layer.attn
        if attn.use_qk_norm:
            q_norm_dim = attn.q_norm.weight.shape[0]
            k_norm_dim = attn.k_norm.weight.shape[0]
            assert q_norm_dim == config_mini["head_dim"], (
                f"Layer {i} q_norm dim={q_norm_dim} "
                f"!= head_dim={config_mini['head_dim']}"
            )
            assert k_norm_dim == config_mini["head_dim"], (
                f"Layer {i} k_norm dim={k_norm_dim} "
                f"!= head_dim={config_mini['head_dim']}"
            )
    print(f"\n✅ QK-Norm dim = head_dim = {config_mini['head_dim']} (correct)")


# ── Test 9: Generate (inference) ──────────────────────────────────────
def test_generate():
    m = CustomLLM(config_mini).to(DEVICE)
    m.eval()
    ids = torch.randint(0, 1000, (1, 10)).to(DEVICE)
    with torch.no_grad():
        out = m.generate(ids, max_new_tokens=20, eos_token_id=2)
    assert out.shape[0] == 1
    assert out.shape[1] > 10
    print(f"\n✅ Generate OK: {ids.shape} → {out.shape}")


# ── Run all ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Running CustomLLM Sanity Tests")
    print("="*55)

    m = CustomLLM(config_mini).to(DEVICE)

    test_param_count(m)
    test_output_shape(m)
    test_forward_with_loss(m)
    test_backward(m)
    test_no_nan_inf(m)
    test_hybrid_attention_pattern()
    test_weight_tying()
    test_qk_norm_dim()
    test_generate()

    print("\n" + "="*55)
    print("  ✅ All tests passed!")
    print("="*55 + "\n")
