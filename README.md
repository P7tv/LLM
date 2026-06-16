# Frontier Thai LLM: Decoder-Only Custom LLM with Multi-Token Prediction (MTP)

โมเดลภาษาขนาดใหญ่ภาษาไทย (Custom LLM) ที่สร้างขึ้นบนสถาปัตยกรรม Decoder-Only สมัยใหม่ รองรับฟีเจอร์ระดับ Frontier เช่น Multi-Token Prediction (MTP), QK-Normalization, Hybrid Attention (Local Sliding Window & Global interleaved) และกระบวนการเทรนตั้งแต่ Pre-training, SFT ไปจนถึง DPO

---

## 🚀 ฟีเจอร์หลัก (Key Features)

- **Decoder-Only Frontier Architecture**:
  - **QK-Normalization (Qwen3 Style)**: การทำ Normalization ให้กับ Query (Q) และ Key (K) ก่อนคูณ RoPE เพื่อความเสถียรสูงสุดขณะฝึกสอนโมเดลขนาดใหญ่
  - **Grouped Query Attention (GQA)**: เพิ่มประสิทธิภาพในการประมวลผลและการใช้หน่วยความจำ (KV Cache)
  - **Hybrid Attention Pattern (Gemma4 Style)**: การวาง Layer ผสมระหว่าง **Local Sliding Window Attention (SWA)** และ **Global Attention** เพื่อจับความสัมพันธ์ระยะสั้นและยาวควบคู่กัน
  - **SwiGLU FFN**: Feed-Forward Network ที่ใช้แอคติเวชันฟังก์ชัน SwiGLU
  - **RoPE with Dynamic Context Extension**: ระบบกางและขยาย Cache คอสไซน์/ไซน์ของ Rotary Embedding อัตโนมัติเมื่อเจอลำดับความยาวที่เกิน
  - **Decoupled LM Head**: ยกเลิกการแชร์น้ำหนักระหว่าง Embedding และ Output Layer (`tie_word_embeddings: false`) เพื่อประสิทธิภาพสูงสุดสำหรับพจนานุกรมขนาดใหญ่
- **Multi-Token Prediction (MTP) Heads (DeepSeek-V3 Style)**:
  - โมเดลสามารถทำนาย token ถัดไปแบบคู่ขนาน $t+1, t+2, \dots, t+d$ เพื่อเพิ่มประสิทธิภาพทางด้านการเรียนรู้เชิงบริบท
  - รองรับการคูณน้ำหนักร่วมกัน (**Weight Tying**) ระหว่าง MTP heads และ Main LM Head
- **Data Engineering Pipeline**:
  - การกรองข้อมูลคุณภาพต่ำด้วย **Quality Filter** (กรองตามความยาวตัวอักษร และอัตราการซ้ำซากของบรรทัด)
  - การกรองเอกสารซ้ำซ้อนผ่าน **MinHash (LSH) Deduplicator**
  - ตัวสตรีมข้อมูลบิตไบนารี (`.bin` shards) สำหรับ Dataloader ป้อนเข้า GPU โดยตรง
- **Robust Training & Fine-Tuning**:
  - **Pre-training**: รองรับการเทรนแบบขนาน (Distributed FSDP NCCL), Mixed Precision (fp16/bf16), Gradient Accumulation และ Cosine Warmup Scheduler
  - **Supervised Fine-Tuning (SFT)**: เทรนปรับจูนเฉพาะ Response ผ่าน Masking Prompt Loss และใช้เทคนิค **LoRA (PEFT)** สำหรับ Parameter-Efficient Fine-Tuning
  - **Direct Preference Optimization (DPO)**: การจัดแนวความพึงพอใจของคำตอบ (Preference Alignment) จากคู่ข้อมูลตัวเลือก Chosen/Rejected
- **Hugging Face Exporter**:
  - แปลงน้ำหนักตัวแบบที่สร้างขึ้นเองไปเป็นฟอร์แมตมาตรฐาน Hugging Face `Qwen2ForCausalLM` (`Qwen3` target wrapper) พร้อมเปิดไฟล์ในแบบ Safetensors

---

## 📂 โครงสร้างโฟลเดอร์ (Directory Structure)

```text
├── configs/                  # ไฟล์กำหนดค่าโมเดลและขั้นตอนการรัน
│   ├── tiny_llm.yaml         # Config ขนาดเล็กสำหรับการเทรนทดสอบ
│   ├── pilot_3b.yaml         # Config สำหรับโมเดลขนาด 3B Pilot
│   ├── main_7b.yaml          # Config สำหรับโมเดลขนาด 7B Main
│   ├── full_13b.yaml         # Config สำหรับโมเดลขนาด 13B Full
│   ├── config_3b.py          # Python config สำหรับ 3B
│   ├── config_7b.py          # Python config สำหรับ 7B
│   └── config_13b.py         # Python config สำหรับ 13B
├── src/
│   ├── data/                 # จัดการข้อมูลโทเค็นและดาต้าเซ็ต
│   │   ├── dataset.py        # โครงสร้าง Streaming Dataset และ Dataloader
│   │   ├── prepare_data.py   # สคริปต์ทำความสะอาดและจัดเตรียมดาต้าเซ็ตบิต shards
│   │   └── tokenizer.py      # ตัวห่อหุ้มโมเดล Typhoon 2.5 Tokenizer
│   ├── evaluation/           # การประเมินผล
│   │   └── eval.py           # คำนวณ Perplexity และประเมินตัวเลือกคำตอบแบบหลายตัวเลือก
│   ├── export/               # ส่งออกโมเดล
│   │   └── export_hf.py      # แปลงพารามิเตอร์โมเดลเป็นฟอร์แมต Hugging Face
│   ├── models/               # สถาปัตยกรรมโครงข่ายประสาทเทียม
│   │   ├── __init__.py       # เชื่อมต่อ CustomLLM
│   │   ├── attention.py      # โมดูล GQA (Local / Global) พร้อม FlashAttention
│   │   ├── block.py          # โครงสร้าง Layer Block หลัก
│   │   ├── ffn.py            # ส่วนโครงสร้าง SwiGLU FFN
│   │   ├── llm.py            # ตัวห่อหุ้มโมเดล CustomLLM
│   │   ├── mtp.py            # ส่วนยื่นทำนายอนาคต (MTP Head และ MTP Loss)
│   │   ├── rmsnorm.py        # RMS Normalization
│   │   └── rope.py           # Rotary Position Embedding (RoPE)
│   └── training/             # โค้ดส่วนการฝึกสอนโมเดล
│       ├── checkpoint.py     # ตัวจัดการเซฟ/โหลดน้ำหนักโมเดล
│       ├── dpo.py            # โมดูล Direct Preference Optimization
│       ├── schedule.py       # คลาสช่วยคำนวณอัตราการเรียนรู้แบบ Cosine Warmup
│       ├── sft.py            # โมดูล Supervised Fine-Tuning
│       └── trainer.py        # ตัวควบคุมลูปการเทรนหลัก (Pre-training)
├── tests/                    # ชุดทดสอบความถูกต้องของซอฟต์แวร์
│   ├── test_data.py          # ทดสอบการประมวลผลข้อมูลและจำลอง token
│   └── test_model.py         # ทดสอบมิติผลลัพธ์โมเดล, loss, gradient และการจำลอง infer
├── requirements.txt          # รายการไลบรารีที่จำเป็นต้องใช้
└── README.md                 # เอกสารอธิบายโครงการ (ไฟล์นี้)
```

---

## 🛠️ การติดตั้งใช้งาน (Installation)

1. ตรวจสอบให้มั่นใจว่าระบบของคุณใช้ Python 3.10 ขึ้นไป
2. ติดตั้งไลบรารีภายนอกที่จำเป็นทั้งหมด:

```bash
pip install -r requirements.txt
```

*หมายเหตุ: หากการติดตั้ง `numpy` หรือไลบรารีอื่น ๆ มีความไม่เข้ากัน แนะนำให้ใช้งาน numpy เวอร์ชัน `1.26.0` หรือสูงกว่าในสภาพแวดล้อมที่เสถียร*

---

## 📊 1. การเตรียมข้อมูลบิตโทเค็น (Data Engineering)

ก่อนการเริ่มฝึกสอนโมเดล คุณจำเป็นต้องประมวลผลคลังข้อมูลดิบ (สามารถป้อนเข้ามาเป็น `.txt`, `.jsonl`, `.json`, หรือ `.parquet`) ให้อยู่ในฟอร์แมตไบนารีโทเค็นบิต shards

ในการรันสคริปต์เตรียมข้อมูล:

```bash
python src/data/prepare_data.py \
    --input <path_to_raw_data_dir_or_file> \
    --output_dir data/shards \
    --tokenizer_id scb10x/typhoon2.5-qwen3-4b \
    --shard_size 1000000 \
    --min_len 50 \
    --max_repeat 0.3 \
    --dedup_threshold 0.85
```

*หากไม่ระบุ `--input` สคริปต์จะสร้างข้อมูลจำลองขึ้นมาในโฟลเดอร์ `scratch/raw_mock_data` อัตโนมัติ เพื่อยืนยันว่าไปป์ไลน์การจัดการข้อมูลสามารถทำงานได้ตามปกติ*

---

## 🏋️ 2. การฝึกสอนโมเดลแบบ Pre-training

เมื่อข้อมูลโทเค็นถูกแบ่งย่อยเรียบร้อยแล้ว สามารถเริ่มฝึกสอนโมเดลผ่านสคริปต์ `trainer.py` ซึ่งจะอ่านค่า config จากไฟล์ `.yaml`

ในการเริ่มเทรน:

```bash
python src/training/trainer.py --config configs/tiny_llm.yaml
```

คุณสามารถปรับการเลือก config ไปเป็นขนาดที่ต้องการเทรน เช่น `configs/pilot_3b.yaml`, `configs/main_7b.yaml` หรือ `configs/full_13b.yaml` ตามกำลังของเครื่องเซิร์ฟเวอร์

### ฟังก์ชันสำคัญของ Trainer:
- **CLI Argument Config**: ไม่มีการฮาร์ดโค้ดพาธ config และสามารถรันสั่งเปลี่ยน config ได้อย่างอิสระ
- **Throughput Logging**: ตรวจจับความเร็วการประมวลผลโทเค็นจริงต่อวินาที (`Tokens/sec`) ได้ถูกต้องแม่นยำ (คูณเข้ากับค่าระยะ `logging_steps` เรียบร้อยแล้ว)
- **Mixed Precision**: เลือกระหว่างการใช้ `bf16` หรือ `fp16` ผ่านค่าคอนฟิกได้
- **Automatic FSDP Hook**: มีโครงสร้างห่อโมเดลตัวอย่างพร้อมทำงานสำหรับ `torchrun` เพื่อรองรับ Multi-GPU Training

---

## 🎯 3. การทำ Supervised Fine-Tuning (SFT) & DPO

### Supervised Fine-Tuning (SFT)
เทรนโมเดลผ่านพจนานุกรมคำถามและคำตอบ โดยจะบล็อกค่า loss ไม่ให้นำมาคำนวณในฝั่ง prompt (จะประเมินเฉพาะส่วนคำตอบของโมเดลเท่านั้น) และใช้วิธีแนบ LoRA Adapter เข้าไปในเลเยอร์ Projections ของโมเดล

ตัวอย่างคำสั่งเทรน SFT:
สคริปต์รองรับการใช้ฟังก์ชัน `train_sft` ผ่านการอ้างอิงและส่งค่าพารามิเตอร์:
```python
from src.training.sft import train_sft

train_sft(
    model_ckpt_path="checkpoints/ckpt_step_100.pt",
    tokenizer_path="scb10x/typhoon2.5-qwen3-4b",
    sft_data=your_sft_data_list,
    config_path="configs/tiny_llm.yaml"
)
```

### Direct Preference Optimization (DPO)
การฝึกให้โมเดลเลือกตอบคำตอบที่ดีที่สุดจากตัวเลือกสองตัว (`chosen` vs `rejected`)

ตัวอย่างการเรียกใช้:
```python
from src.training.dpo import train_dpo

train_dpo(
    model_sft_path="checkpoints/sft_lora",
    tokenizer_path="scb10x/typhoon2.5-qwen3-4b",
    preference_data=your_dpo_preference_data,
    config_path="configs/tiny_llm.yaml"
)
```

---

## 📈 4. การวัดและประเมินผลโมเดล (Evaluation)

สคริปต์ `src/evaluation/eval.py` เตรียมโมดูลเพื่อประเมินความสามารถของโมเดลไว้ 2 รูปแบบหลัก:
1. **Perplexity (PPL)**: ประเมินความราบรื่นในการเชื่อมโยงภาษาของโมเดลผ่าน Validation Set
2. **Multiple-Choice QA**: ประเมินความถูกต้องของการเลือกตอบคำถาม 4 ตัวเลือก (เช่น ชุดข้อสอบ MMLU) 

*ระบบจะประเมินโดยใส่โทเค็น `<bos>` เฉพาะในกรณีที่ตัวแปลงโทเค็นตัวนั้นรองรับ (หากไม่รองรับ/หรือค่าเป็น None ระบบจะไม่ฝืนเอาโทเค็น EOS มาใส่แทน เพื่อป้องกันพฤติกรรมข้อมูลเพี้ยน)*

---

## 📦 5. การส่งออกและนำไปใช้งานบน Hugging Face (Exporting Model)

สคริปต์ `src/export/export_hf.py` สนับสนุนการแมปชื่อเลเยอร์ทั้งหมดของตัวแบบโมเดลที่คุณเขียนขึ้นเองไปหาโครงสร้างมาตรฐานของ **Hugging Face Qwen2ForCausalLM** (เป้าหมายคือ `Qwen3` config architecture) 

ในการส่งออกและทำการบันทึกข้อมูลน้ำหนักโมเดลเป็นไฟล์ Safetensors:
```python
from src.export.export_hf import convert_to_hf

convert_to_hf(
    model_ckpt_path="checkpoints/ckpt_step_100.pt",
    hf_output_dir="exported_model_hf",
    config_path="configs/tiny_llm.yaml"
)
```

---

## 🧪 6. การทดสอบโค้ดสถาปัตยกรรม (Running Unit Tests)

เพื่อตรวจสอบความถูกต้องและเสถียรภาพของโค้ดสถาปัตยกรรม คุณสามารถรันชุดการทดสอบทั้งหมดได้ด้วย `pytest`:

```bash
python -m pytest
```

ผลลัพธ์จากการทดสอบจะประเมินส่วนประกอบต่าง ๆ ได้แก่:
- จำนวนพารามิเตอร์ ความถูกต้องของมิติการคำนวณ
- Loss ของโมเดลขณะเทรนปกติ และขณะรัน MTP
- การไหลของ Gradient ในเลเยอร์ต่างๆ
- ความเสถียร (ไม่มี NaN/Inf)
- Hybrid attention pattern
- การทำ Weight Tying ของหัว MTP
- ความถูกต้องของขนาดมิติ QK-Normalization
- ความถูกต้องของการทำ Inference (Generate)
