import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any

from src.models import CustomLLM
from src.data.tokenizer import CustomTokenizer
from src.data.dataset import StreamingDataset

@torch.no_grad()
def calculate_perplexity(model: CustomLLM, dataloader: DataLoader, device: torch.device) -> float:
    """Calculates model perplexity (PPL) on validation tokens dataset."""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        outputs = model(x, labels=y)
        loss = outputs["loss"]
        
        total_loss += loss.item()
        total_steps += 1
        
    if total_steps == 0:
        return float('inf')
        
    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    return perplexity


@torch.no_grad()
def evaluate_multiple_choice(
    model: CustomLLM, 
    tokenizer: CustomTokenizer, 
    eval_samples: List[Dict[str, Any]], 
    device: torch.device
) -> float:
    """Evaluates the model on multiple-choice questions (e.g. MMLU task structure).
    
    Format:
      samples = [
        {
          "question": "What is the capital of Thailand?",
          "choices": ["A) Chiang Mai", "B) Bangkok", "C) Phuket", "D) Pattaya"],
          "answer": "B"
        }
      ]
    """
    model.eval()
    correct_predictions = 0
    
    bos_id = tokenizer.token_to_id("<bos>")
    
    for sample in eval_samples:
        prompt = f"<user>\n{sample['question']}\nChoices:\n"
        for choice in sample["choices"]:
            prompt += f"{choice}\n"
        prompt += "<assistant>\nThe correct choice is "
        
        prompt_ids = tokenizer.encode(prompt)
        if bos_id is not None:
            prompt_ids = [bos_id] + prompt_ids
        
        choice_logps = {}
        for option in ["A", "B", "C", "D"]:
            # Tokenize the candidate answer option
            option_id = tokenizer.encode(option)
            
            # Sequence: prompt + candidate answer option
            seq = prompt_ids + option_id
            seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            
            # Get model outputs
            logits = model(seq_tensor)["logits"]
            
            # Extract logits for the target choice token (the last position of prompt)
            # which predicts the choice
            option_logits = logits[0, -2, :]  # -2 because last input was 'option_id'
            log_probs = torch.log_softmax(option_logits, dim=-1)
            
            # Keep track of target option token log probability
            target_token_id = option_id[0]
            choice_logps[option] = log_probs[target_token_id].item()
            
        # Select key with maximum log probability
        predicted = max(choice_logps, key=choice_logps.get)
        if predicted == sample["answer"]:
            correct_predictions += 1
            
    accuracy = correct_predictions / len(eval_samples) if eval_samples else 0.0
    return accuracy


if __name__ == "__main__":
    print("Evaluation module loaded. Functions available: calculate_perplexity, evaluate_multiple_choice")
