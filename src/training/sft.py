import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model

from src.models import CustomLLM
from src.data.tokenizer import CustomTokenizer

class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning (SFT).
    
    Constructs prompt template and masks prompt tokens so loss is computed only on responses.
    """
    def __init__(self, data_list: list[dict], tokenizer: CustomTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        
        # Special token IDs
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        
        for item in data_list:
            prompt = item["prompt"]
            response = item["response"]
            self._process_sample(prompt, response)

    def _process_sample(self, prompt: str, response: str):
        # Format using Typhoon 2.5 chat template
        prompt_str = self.tokenizer.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
        response_str = f"{response}"
        
        # Tokenize separately to calculate prefix length
        prompt_ids = self.tokenizer.encode(prompt_str)
        if self.bos_id is not None:
            prompt_ids = [self.bos_id] + prompt_ids
        response_ids = self.tokenizer.encode(response_str) + [self.eos_id]
        
        input_ids = prompt_ids + response_ids
        
        # Truncate to max sequence length if necessary
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            
        # Target labels: prompt tokens are masked to -100 (ignored in CrossEntropyLoss)
        labels = [-100] * len(prompt_ids) + response_ids
        labels = labels[:len(input_ids)]
        
        # Padding
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.token_to_id("<pad>")
            input_ids += [pad_id] * pad_len
            labels += [-100] * pad_len
            
        self.samples.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_sft(model_ckpt_path: str, tokenizer_path: str, sft_data: list[dict], config_path: str = "configs/tiny_llm.yaml"):
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = CustomTokenizer(tokenizer_path)
    
    # 2. Reconstruct Model
    model = CustomLLM(config)
    
    # Load pre-trained weights
    checkpoint = torch.load(model_ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # 3. Setup LoRA (PEFT)
    # Target linear layers in our custom transformer model
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Prepare Dataset
    max_seq_len = config.get("max_seq_len_sft", 4096)
    dataset = SFTDataset(sft_data, tokenizer, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # 5. Fine-Tuning Setup
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    # Fine-Tuning loop (For pipeline illustration - runs only if invoked explicitly)
    print("Initializing Supervised Fine-Tuning (SFT)...")
    for epoch in range(1):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if step % 5 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                
    # Save SFT LoRA checkpoint weights
    sft_ckpt_dir = os.path.join(config["checkpoint_dir"], "sft_lora")
    model.save_pretrained(sft_ckpt_dir)
    print(f"SFT completed. LoRA adapter saved to {sft_ckpt_dir}")


if __name__ == "__main__":
    # Example invocation outline
    dummy_sft_data = [
        {
            "prompt": "ช่วยแนะนำตัวหน่อยครับ", 
            "response": "สวัสดีครับ ผมคือโมเดลภาษาขนาดใหญ่ที่ถูกเทรนขึ้นมาเพื่อตอบคำถามของคุณครับ"
        },
        {
            "prompt": "Write a python function to add two numbers.",
            "response": "def add(a, b):\n    return a + b"
        }
    ]
    # In actual deployment, user runs: train_sft('checkpoints/ckpt_step_100.pt', 'configs/tokenizer.json', dummy_sft_data)
    print("SFT script initialized. Run train_sft method with appropriate checkpoints to execute.")
