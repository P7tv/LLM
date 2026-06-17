# Frontier Thai LLM 🇹🇭

> A decoder-only Large Language Model built **from scratch** in PyTorch, combining frontier techniques from Qwen3, Gemma, Llama 3, and DeepSeek-V3 — with a complete pipeline from data engineering → pre-training → SFT (LoRA) → DPO → Hugging Face export.
>
> โมเดลภาษาขนาดใหญ่แบบ Decoder-Only ที่เขียน**ขึ้นเองจากศูนย์**ด้วย PyTorch รวมเทคนิคระดับ frontier จาก Qwen3, Gemma, Llama 3 และ DeepSeek-V3 พร้อมไปป์ไลน์ครบตั้งแต่เตรียมข้อมูล → pre-training → SFT (LoRA) → DPO → export ขึ้น Hugging Face

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Tests" src="https://img.shields.io/badge/tests-13%20passing-brightgreen">
</p>

---

## Table of Contents

- [Why this repo](#why-this-repo)
- [Key features](#key-features)
- [Architecture](#architecture)
- [Model configurations](#model-configurations)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quickstart (5 minutes, CPU)](#quickstart-5-minutes-cpu)
- [Datasets](#datasets)
- [Full pipeline](#full-pipeline)
- [Design notes — the frontier techniques](#design-notes--the-frontier-techniques)
- [Testing](#testing)
- [Hardware notes](#hardware-notes)
- [Known limitations & roadmap](#known-limitations--roadmap)
- [License & acknowledgements](#license--acknowledgements)

---

## Why this repo

**EN** — Most "train your own LLM" tutorials stop at a toy GPT. This repository implements the architectural choices that **modern production models actually use in 2024–2025**, with correct, readable, and tested PyTorch — so you can study each component in isolation, or use it as a serious starting point for your own Thai (or multilingual) model.

**TH** — โปรเจกต์นี้ไม่ได้หยุดแค่ GPT ของเล่น แต่ implement สถาปัตยกรรมที่โมเดลระดับ production ปี 2024–2025 ใช้จริง ด้วยโค้ด PyTorch ที่ถูกต้อง อ่านง่าย และมีเทสต์ครบ — เหมาะทั้งสำหรับศึกษาทีละชิ้นส่วน และใช้เป็นจุดเริ่มต้นจริงจังในการสร้างโมเดลภาษาไทย/หลายภาษาของคุณเอง

---

## Key features

| Component | What it is | Inspired by |
|---|---|---|
| **Decoder-only Transformer** | Causal next-token prediction backbone | GPT / Llama |
| **Grouped Query Attention (GQA)** | Q heads share fewer K/V heads → smaller KV-cache | Llama 3 |
| **QK-Normalization** | RMSNorm on Q & K (per head) *before* RoPE for training stability | Qwen3 |
| **Hybrid Attention** | Interleaved Local Sliding-Window + Global layers (`local×3 → global`) | Gemma |
| **RoPE (long context)** | Rotary embeddings, `theta=1M`, dynamic cache extension up to 128K | Llama / Qwen |
| **SwiGLU FFN** | Gated feed-forward (`down(SiLU(gate)·up)`) | Llama / PaLM |
| **RMSNorm Pre-LN** | Norm before each sub-block, computed in fp32 | Llama |
| **Multi-Token Prediction (MTP)** | Auxiliary heads predicting t+2, t+3… (weight-tied to LM head) | DeepSeek-V3 |
| **KV-cache generation** | Prefill + incremental decode; local layers cap cache to the window | — |
| **FlashAttention-3** | Auto-detected; safe SDPA fallback with OOM guards | — |
| **Data engineering** | Quality filter + MinHash dedup + memory-mapped `uint32` token shards | — |
| **Training** | FSDP-ready, bf16/fp16 AMP, gradient accumulation, cosine warmup, checkpointing | — |
| **Alignment** | SFT with prompt-masking + **LoRA**, then **DPO** preference optimization | — |
| **Export & quantize** | → Hugging Face `Qwen3ForCausalLM` (safetensors), AWQ / GGUF helpers | — |

---

## Architecture

```
 Input token IDs  [B, T]
        │
        ▼
 Token Embedding                                  ┌─── Inside one TransformerBlock ───┐
        │                                         │                                   │
        ▼                                         │   x ─► RMSNorm ─► Hybrid GQA Attn  │
 ┌─────────────────┐    expands to                │            (QK-Norm → RoPE → SWA   │
 │ TransformerBlock│ ◄──────────────────────────► │             / Global) ─► (+)x      │  ← residual
 │     × N layers  │                              │   x ─► RMSNorm ─► SwiGLU FFN ─►(+)x │  ← residual
 └─────────────────┘                              └───────────────────────────────────┘
        │
        ▼
 Final RMSNorm
        │
        ▼
 LM Head  ───────────────►  Logits [B, T, vocab]  ─►  Cross-Entropy
        │
        └──(optional)──►  MTP Heads  ─►  auxiliary t+2…t+d loss (λ-weighted)
```

**EN** — Each block is **Pre-LN with residual connections**: `x = x + Attn(Norm(x))` then `x = x + FFN(Norm(x))`. Attention layers alternate between *local* (sliding-window, cheap, O(T·w)) and *global* (full, O(T²)) following the `attn_pattern`, giving long-range understanding at a fraction of the compute.

**TH** — แต่ละ block เป็นแบบ **Pre-LN + residual**: `x = x + Attn(Norm(x))` แล้วตามด้วย `x = x + FFN(Norm(x))`. ชั้น attention สลับระหว่าง *local* (sliding-window ประหยัด O(T·w)) กับ *global* (เต็ม O(T²)) ตาม `attn_pattern` — ได้ความเข้าใจระยะไกลโดยใช้ compute เพียงเศษเสี้ยว

> 📖 A component-by-component walkthrough (basic → expert) is in [Design notes](#design-notes--the-frontier-techniques).

---

## Model configurations

| Config | hidden | layers | Q / KV heads | ctx window | RoPE θ | MTP | Intended use |
|---|---|---|---|---|---|---|---|
| `tiny_llm.yaml` | 256 | 4 | 8 / 2 | 1K | 1M | 1 | CPU sanity / unit tests |
| `pilot_3b.yaml` | 3072 | 28 | 24 / 8 | 32K | 500K | 1 | Pilot runs |
| `main_7b.yaml` | 4096 | 32 | 32 / 8 | 128K | 1M | 0 | Main model |
| `full_13b.yaml` | 5120 | 48 | 40 / 8 | 128K | 1M | 1 | Full-scale |

All configs share `vocab_size = 151,669` (Typhoon 2.5 tokenizer) and `tie_word_embeddings: false`.
Python-dict equivalents live in `configs/config_{3b,7b,13b}.py`.

---

## Project structure

```text
├── configs/                  # Model + run configs (.yaml for trainer, .py dict mirrors)
│   ├── tiny_llm.yaml         #   small config for CPU testing
│   ├── pilot_3b / main_7b / full_13b
│   └── config_{3b,7b,13b}.py
├── src/
│   ├── data/
│   │   ├── tokenizer.py       # Typhoon 2.5 tokenizer wrapper
│   │   ├── dataset.py         # QualityFilter, MinHash dedup, streaming .bin dataset
│   │   └── prepare_data.py    # raw text → tokenized uint32 binary shards (CLI)
│   ├── models/
│   │   ├── rmsnorm.py         # RMSNorm (fp32 internal)
│   │   ├── rope.py            # Rotary embeddings + dynamic extension
│   │   ├── attention.py       # HybridGQAAttention: GQA + QK-Norm + SWA/Global + KV-cache
│   │   ├── ffn.py             # SwiGLU FFN
│   │   ├── block.py           # Pre-LN Transformer block
│   │   ├── mtp.py             # Multi-Token Prediction heads + loss
│   │   └── llm.py             # CustomLLM: assembles everything + generate() w/ KV-cache
│   ├── training/
│   │   ├── trainer.py         # Pre-training loop (FSDP-ready, AMP, ckpt) — CLI entry
│   │   ├── schedule.py        # Cosine LR with linear warmup
│   │   ├── checkpoint.py      # Save/load + rotation
│   │   ├── sft.py             # Supervised fine-tuning (prompt-masked) + LoRA
│   │   └── dpo.py             # Direct Preference Optimization
│   ├── evaluation/eval.py     # Perplexity + multiple-choice (MMLU-style)
│   └── export/
│       ├── export_hf.py       # → Hugging Face Qwen3ForCausalLM (safetensors)
│       └── quantize.py        # AWQ helper + GGUF (llama.cpp) instructions
├── tests/                     # pytest: shapes, loss, gradients, NaN/Inf, MTP, generate
├── requirements.txt
├── LICENSE                    # MIT
└── README.md
```

---

## Installation

Requires **Python 3.10+**.

```bash
git clone <your-repo-url>
cd LLM
pip install -r requirements.txt
```

**FlashAttention-3 (optional, GPU only).** The model auto-detects it and falls back to PyTorch SDPA if absent. For long-context *local* layers (T > 8K) on GPU it is strongly recommended:

```bash
pip install flash-attn --no-build-isolation
```

---

## Quickstart (5 minutes, CPU)

Run the test suite — it builds a tiny 4-layer model and exercises the full forward/backward/generate paths on CPU:

```bash
python -m pytest -q
# 13 passed
```

Instantiate and generate from an (untrained) model:

```python
import torch, yaml
from src.models import CustomLLM

config = yaml.safe_load(open("configs/tiny_llm.yaml"))
model = CustomLLM(config).eval()

ids = torch.randint(0, config["vocab_size"], (1, 8))
out = model.generate(ids, max_new_tokens=20, temperature=0.8, top_p=0.9)
print(out.shape)   # uses KV-cache: prefill once, then 1 token / step
```

---

## Datasets

**EN** — This repo ships the *pipeline*, not a corpus (training data is intentionally `.gitignore`d). Bring your own text, or start from public Thai/multilingual datasets. The pipeline accepts `.txt`, `.jsonl`, `.json`, and `.parquet`; for JSON/JSONL/Parquet it reads a configurable text field (`--text_key`, default `text`).

**TH** — repo นี้ให้ *ไปป์ไลน์* ไม่ได้ให้ตัว corpus (ข้อมูลเทรนถูก gitignore ไว้โดยตั้งใจ) — เอาข้อความของคุณเองมาใส่ หรือเริ่มจาก dataset ไทย/หลายภาษาแบบสาธารณะด้านล่าง

| Dataset (Hugging Face) | What it is | Notes |
|---|---|---|
| `wikimedia/wikipedia` (`20231101.th`) | Thai Wikipedia — clean, structured | Easiest start, ungated |
| `uonlp/CulturaX` (`th`) | Large filtered multilingual web text | Strong general pre-training base |
| `HuggingFaceFW/fineweb-2` (`tha_Thai`) | Modern, heavily-filtered web corpus | High quality, large |
| `oscar-corpus/OSCAR-2301` (`th`) | Web crawl | May require access agreement |
| `allenai/MADLAD-400` (`th`) | Document-level multilingual crawl | Large scale |
| `bigcode/the-stack-v2` | Source code (many languages) | Add if you want coding ability |

> Always review each dataset's license and card on Hugging Face before training.

**Example — Thai Wikipedia → JSONL the pipeline understands:**

```python
from datasets import load_dataset
import json, os

os.makedirs("raw", exist_ok=True)
ds = load_dataset("wikimedia/wikipedia", "20231101.th", split="train")
with open("raw/thwiki.jsonl", "w", encoding="utf-8") as f:
    for row in ds:                      # each line: {"text": "..."} ← prepare_data reads "text"
        f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
```

```bash
# Then tokenize + filter + dedup into binary shards (see step 1 below)
python src/data/prepare_data.py --input raw --output_dir data/shards
```

> For multi-terabyte corpora, prefer **streaming** (`load_dataset(..., streaming=True)`) and write many `.jsonl` files so the dataloader and dedup can work shard-by-shard.

---

## Full pipeline

### 1. Data engineering

Convert a raw corpus (`.txt`, `.jsonl`, `.json`, `.parquet`) into tokenized `uint32` binary shards. Applies a quality filter and MinHash near-deduplication.

```bash
python src/data/prepare_data.py \
    --input <raw_data_dir_or_file> \
    --output_dir data/shards \
    --tokenizer_id scb10x/typhoon2.5-qwen3-4b \
    --shard_size 1000000 \
    --min_len 50 \
    --max_repeat 0.3 \
    --dedup_threshold 0.85
```

> Omit `--input` to run a self-contained **mock mode** that fabricates sample files in `scratch/raw_mock_data` and verifies the pipeline end-to-end.

### 2. Pre-training

Reads a YAML config and streams the `.bin` shards from `data/shards/`.

```bash
# Single device
python src/training/trainer.py --config configs/main_7b.yaml

# Multi-GPU (FSDP via torchrun)
torchrun --nproc_per_node=8 src/training/trainer.py --config configs/main_7b.yaml
```

Features: bf16/fp16 AMP, gradient accumulation, cosine-warmup LR, TensorBoard/W&B logging, automatic checkpoint resume, tokens/sec throughput.

### 3. Supervised Fine-Tuning (SFT) + LoRA

Loss is computed **only on the response** (prompt tokens masked to `-100`); fine-tuning uses LoRA adapters.

```python
from src.training.sft import train_sft

train_sft(
    model_ckpt_path="checkpoints_7b/ckpt_step_100000.pt",
    tokenizer_path="scb10x/typhoon2.5-qwen3-4b",
    sft_data=[{"prompt": "...", "response": "..."}],
    config_path="configs/main_7b.yaml",
)
```

### 4. Direct Preference Optimization (DPO)

Aligns the SFT model on `chosen` vs `rejected` pairs against a frozen reference model. Log-probs are length-normalized; tune `beta` via `config["dpo_beta"]` (default `0.5`).

```python
from src.training.dpo import train_dpo

train_dpo(
    model_sft_path="checkpoints_7b/sft_lora/adapter_model.pt",
    tokenizer_path="scb10x/typhoon2.5-qwen3-4b",
    preference_data=[{"prompt": "...", "chosen": "...", "rejected": "..."}],
    config_path="configs/main_7b.yaml",
)
```

### 5. Evaluation

```python
from src.evaluation.eval import calculate_perplexity, evaluate_multiple_choice
# Perplexity (lower = better) on a validation shard, and MMLU-style 4-choice accuracy
```

### 6. Export to Hugging Face

Maps your custom weights onto `Qwen3ForCausalLM` and writes safetensors.

```python
from src.export.export_hf import convert_to_hf
convert_to_hf("checkpoints_7b/ckpt_step_100000.pt", "exported_model_hf", "configs/main_7b.yaml")
```

> ⚠️ **Note:** Qwen3 uses a single sliding-window setting for the whole model, so the **layer-wise hybrid pattern is not preserved** on export (the code warns about this). For faithful export, target an architecture with per-layer attention patterns (e.g. Gemma2).

### 7. Quantization (optional)

```python
from src.export.quantize import run_awq_quantization, print_gguf_instructions
run_awq_quantization("exported_model_hf", "checkpoints/awq_model")  # GPU, 4-bit AWQ
print_gguf_instructions("exported_model_hf")                        # llama.cpp / GGUF for CPU/Mac
```

---

## Design notes — the frontier techniques

A condensed tour of *why* each piece exists. (For a deep basic→expert walkthrough, this is the section to extend.)

- **GQA** — Full multi-head attention gives every query head its own K/V; that bloats the KV-cache at inference. GQA shares one K/V head across a group of query heads (here 4:1), shrinking the cache ~4× with negligible quality loss. FlashAttention-3 handles the grouping natively; the SDPA fallback expands K/V explicitly.

- **QK-Normalization (Qwen3)** — Large models suffer *attention-logit explosion* (a few dimensions blow up `QKᵀ`, destabilizing training). Applying RMSNorm to Q and K **per head** and **before** RoPE keeps logits well-scaled. Order matters: norming *after* RoPE would wash out positional information.

- **Hybrid Local/Global attention (Gemma)** — Making every layer global is O(T²) × depth — catastrophic at 128K. Here 3 of every 4 layers are local sliding-window (linear in sequence length), and every 4th is global. Crucially, local layers **never materialize a T×T mask** (which would be ~32 GB at 128K) — they use FA3's window kernel, with the SDPA fallback raising a clear error past 8K instead of OOM-ing.

- **RoPE with θ=1M** — A large base frequency rotates slowly enough that 128K positions remain distinguishable. The cache auto-extends if a longer sequence appears. Rotations are computed in fp32 then cast back, avoiding bf16 angle drift at long context.

- **SwiGLU FFN** — A gated feed-forward: `down(SiLU(gate(x)) · up(x))`. The element-wise gate consistently outperforms a plain ReLU MLP; with 3 projection matrices the intermediate size is tuned (~2.7–3.5× hidden) to keep parameter count comparable.

- **Multi-Token Prediction (DeepSeek-V3)** — Extra heads predict tokens t+2, t+3… by combining the hidden state at t with the embedding of the next token, giving denser training signal and faster learning. Heads are weight-tied to the main LM head and discarded at inference (or reused for speculative decoding).

- **KV-cache generation** — `generate()` runs a single **prefill** over the prompt (caching K/V per layer), then **decodes one token at a time** in O(1) per step instead of re-running the whole sequence. Local layers cap their cache at the sliding window so memory stays flat at long context. Decode passes the **true absolute position** into RoPE — necessary because a capped local cache no longer equals the token's position. This path is verified to match a full forward to within ~1e-6.

---

## Testing

```bash
python -m pytest -q          # full suite (13 tests, CPU)
python -m pytest tests/test_model.py -v
```

Covers: parameter counts, output shapes, forward + (main & MTP) loss, gradient flow on every parameter, NaN/Inf stability, hybrid-attention layer pattern, MTP weight-tying, QK-Norm dimensions, and KV-cached generation.

---

## Hardware notes

This is a real architecture, not a toy. The `tiny` config runs on CPU; the **3B/7B/13B configs are intended for datacenter GPUs (A100 / H200 / B200 or better)** with FSDP multi-GPU and FlashAttention-3. Do not expect the large configs to train on a laptop.

---

## Known limitations & roadmap

- [ ] **Export fidelity** — hybrid attention pattern is flattened on HF export (see note above).
- [ ] **Dynamic padding** — SFT/DPO datasets pad to a fixed `max_seq_len_sft` (default 4096); a `collate_fn` padding to the per-batch longest would be more efficient.
- [ ] **MinHash dedup** is O(n²) (linear scan of signatures); add LSH banding for million-doc corpora.
- [ ] **Speculative decoding** — reuse the trained MTP heads at inference.
- [ ] **Batched generation** stops only when *all* sequences hit EOS; per-sequence early-exit would save compute.

Contributions welcome — open an issue or PR.

---

## License & acknowledgements

Released under the **MIT License** — see [LICENSE](LICENSE).

Originally developed during the **SuperAI Engineer** program (while the author served as a Teaching Assistant).

Architecture inspired by the published work behind **Qwen3** (QK-Norm), **Gemma** (hybrid local/global attention), **Llama 3** (GQA, SwiGLU, RoPE), and **DeepSeek-V3** (Multi-Token Prediction). Tokenizer: [scb10x/typhoon2.5-qwen3-4b](https://huggingface.co/scb10x/typhoon2.5-qwen3-4b).

This is an independent, from-scratch implementation for research and educational use; it is not affiliated with or endorsed by any of the above.
