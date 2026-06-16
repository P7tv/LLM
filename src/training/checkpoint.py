import os
import torch
from typing import Dict, Any

class CheckpointManager:
    """Manages saving and loading model checkpoints and optimizer states."""
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler_state: Dict[str, Any], 
        step: int, 
        loss: float
    ) -> str:
        """Saves a checkpoint containing model state, optimizer state, scheduler, and step."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"ckpt_step_{step}.pt")
        
        # Unwrap DDP/FSDP module if wrapped
        raw_model = model.module if hasattr(model, "module") else model
        
        state_dict = {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state,
            "step": step,
            "loss": loss
        }
        
        torch.save(state_dict, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path} (step {step})")
        
        self._rotate_checkpoints()
        return checkpoint_path

    def load_latest_checkpoint(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, Any]:
        """Loads the most recent checkpoint if available, returns state metadata."""
        checkpoints = self.get_all_checkpoints()
        if not checkpoints:
            print("No checkpoints found. Starting from scratch.")
            return {}

        latest_ckpt_path = checkpoints[-1]
        print(f"Loading checkpoint from {latest_ckpt_path}...")
        
        checkpoint = torch.load(latest_ckpt_path, map_location="cpu")
        
        # Load states
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return {
            "step": checkpoint["step"],
            "loss": checkpoint["loss"],
            "scheduler_state_dict": checkpoint["scheduler_state_dict"]
        }

    def get_all_checkpoints(self) -> list[str]:
        """Returns a list of checkpoint paths sorted by step number."""
        files = [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("ckpt_step_") and f.endswith(".pt")
        ]
        
        # Sort by step number extracted from filename
        def get_step(path):
            name = os.path.basename(path)
            return int(name.split("_")[2].split(".")[0])
            
        return sorted(files, key=get_step)

    def _rotate_checkpoints(self):
        """Deletes older checkpoints to keep only the max_to_keep most recent ones."""
        checkpoints = self.get_all_checkpoints()
        if len(checkpoints) > self.max_to_keep:
            to_delete = checkpoints[:-self.max_to_keep]
            for file_path in to_delete:
                try:
                    os.remove(file_path)
                    print(f"Deleted old checkpoint: {file_path}")
                except OSError as e:
                    print(f"Error deleting checkpoint {file_path}: {e}")
