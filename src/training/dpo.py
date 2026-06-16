import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.models import CustomLLM
from src.data.tokenizer import CustomTokenizer

class DPODataset(Dataset):
    """Dataset storing preference pairs for DPO training."""
    def __init__(self, data_list: list[dict], tokenizer: CustomTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")
        
        for item in data_list:
            self._process_pair(item["prompt"], item["chosen"], item["rejected"])

    def _process_pair(self, prompt: str, chosen: str, rejected: str):
        # Format using Typhoon 2.5 chat template
        prompt_str = self.tokenizer.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
        
        prompt_ids = self.tokenizer.encode(prompt_str)
        if self.bos_id is not None:
            prompt_ids = [self.bos_id] + prompt_ids
        chosen_resp_ids = self.tokenizer.encode(chosen) + [self.eos_id]
        rejected_resp_ids = self.tokenizer.encode(rejected) + [self.eos_id]
        
        # Build chosen sequence & mask prompt
        c_input_ids = prompt_ids + chosen_resp_ids
        c_labels = [-100] * len(prompt_ids) + chosen_resp_ids
        
        # Build rejected sequence & mask prompt
        r_input_ids = prompt_ids + rejected_resp_ids
        r_labels = [-100] * len(prompt_ids) + rejected_resp_ids
        
        # Crop or Pad chosen sequence
        c_input_ids, c_labels = self._pad_or_crop(c_input_ids, c_labels)
        # Crop or Pad rejected sequence
        r_input_ids, r_labels = self._pad_or_crop(r_input_ids, r_labels)
        
        self.samples.append({
            "chosen_input_ids": torch.tensor(c_input_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(c_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(r_input_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(r_labels, dtype=torch.long)
        })

    def _pad_or_crop(self, ids: list[int], labels: list[int]):
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(ids)
            pad_id = self.tokenizer.token_to_id("<pad>")
            ids += [pad_id] * pad_len
            labels += [-100] * pad_len
        return ids, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes average/sum log probabilities for response tokens in the labels."""
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    
    # Shift logits and labels to match causal prediction (predict next token)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Calculate log_softmax
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log prob of correct labels
    # Mask out -100 labels (prompt + pad tokens)
    loss_mask = (shift_labels != -100)
    
    # Replace -100 with 0 temporarily to prevent indexing issues during gather
    gather_labels = shift_labels.clone()
    gather_labels[gather_labels == -100] = 0
    
    per_token_logps = torch.gather(log_probs, dim=-1, index=gather_labels.unsqueeze(-1)).squeeze(-1)
    
    # Sum log probabilities across response sequence
    return (per_token_logps * loss_mask).sum(-1)


def compute_dpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
                     reference_chosen_logps: torch.Tensor, reference_rejected_logps: torch.Tensor,
                     beta: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO Loss = -E [ log sigmoid( beta * (log(pi_theta(yw|x) / pi_ref(yw|x)) - log(pi_theta(yl|x) / pi_ref(yl|x))) ) ]"""
    
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps
    
    logits = policy_log_ratio - reference_log_ratio
    
    loss = -F.logsigmoid(beta * logits).mean()
    
    # Diagnostic metrics (accuracies)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
    
    return loss, chosen_rewards, reward_accuracies


def train_dpo(model_sft_path: str, tokenizer_path: str, preference_data: list[dict], beta: float = 0.1, config_path: str = "configs/tiny_llm.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = CustomTokenizer(tokenizer_path)
    
    # Initialize Policy Model and Reference Model
    policy_model = CustomLLM(config)
    reference_model = CustomLLM(config)
    
    # Load SFT base weights into both models
    checkpoint = torch.load(model_sft_path, map_location="cpu")
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    reference_model.load_state_dict(checkpoint["model_state_dict"])
    
    # Freeze reference model completely
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
        
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    policy_model.to(device)
    reference_model.to(device)
    
    # Prepare DPO dataset loader
    dataset = DPODataset(preference_data, tokenizer, max_seq_len=config["max_position_embeddings"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-6)
    
    policy_model.train()
    print("Initializing Direct Preference Optimization (DPO)...")
    
    for epoch in range(1):
        for step, batch in enumerate(dataloader):
            # Move data to target device
            c_input_ids = batch["chosen_input_ids"].to(device)
            c_labels = batch["chosen_labels"].to(device)
            r_input_ids = batch["rejected_input_ids"].to(device)
            r_labels = batch["rejected_labels"].to(device)
            
            # Policy forward pass
            policy_chosen_logits = policy_model(c_input_ids)["logits"]
            policy_rejected_logits = policy_model(r_input_ids)["logits"]
            
            # Reference forward pass (without gradients)
            with torch.no_grad():
                ref_chosen_logits = reference_model(c_input_ids)["logits"]
                ref_rejected_logits = reference_model(r_input_ids)["logits"]
                
            # Log probabilities computation
            pi_c_logps = get_batch_logps(policy_chosen_logits, c_labels)
            pi_r_logps = get_batch_logps(policy_rejected_logits, r_labels)
            
            ref_c_logps = get_batch_logps(ref_chosen_logits, c_labels)
            ref_r_logps = get_batch_logps(ref_rejected_logits, r_labels)
            
            # DPO loss calculation
            loss, rewards, accuracy = compute_dpo_loss(
                pi_c_logps, pi_r_logps, ref_c_logps, ref_r_logps, beta=beta
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if step % 5 == 0:
                print(f"DPO Step {step} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.2f}")
                
    # Save DPO weights
    dpo_ckpt_path = os.path.join(config["checkpoint_dir"], "dpo_model.pt")
    torch.save({"model_state_dict": policy_model.state_dict()}, dpo_ckpt_path)
    print(f"DPO completed. Preference aligned model saved to {dpo_ckpt_path}")


if __name__ == "__main__":
    print("DPO Trainer script loaded. Use train_dpo method with preference datasets to align model.")
