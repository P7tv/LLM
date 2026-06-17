import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Import local modules
from src.models import CustomLLM
from src.data.dataset import get_streaming_dataloader
from src.training.schedule import CosineWarmupScheduler
from src.training.checkpoint import CheckpointManager

def setup_distributed():
    """Initializes distributed environment if multi-GPU training is detected."""
    # Detect PyTorch Distributed environment variables (set by torchrun)
    is_distributed = "WORLD_SIZE" in os.environ and "RANK" in os.environ
    if not is_distributed:
        return False, 0, 1, torch.device("cpu")
        
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Initialize NCCL process group
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    return True, rank, world_size, device


def main():
    # Load configuration
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_llm.yaml")
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Setup Distributed Mode
    is_dist, rank, world_size, device = setup_distributed()
    
    if not is_dist:
        # Fallback to local device selection
        if config["device"] == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(config["device"])
            
    if rank == 0:
        print(f"Using training device: {device}")
        print(f"Distributed mode active: {is_dist}")

    # 2. Initialize Model
    model = CustomLLM(config)
    model.to(device)

    # 3. Distributed Wrapper Hook
    if is_dist:
        # Skeleton for wrapping with FSDP (Fully Sharded Data Parallel)
        # or can be substituted with DeepSpeed wrappers.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(model)
        if rank == 0:
            print("Model wrapped with PyTorch Fully Sharded Data Parallel (FSDP)")

    # 4. Optimizer & Scheduler Setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-8
    )

    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=config["warmup_steps"],
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        min_learning_rate=config["min_learning_rate"]
    )

    # Checkpoint and Loggers (Main rank only)
    ckpt_manager = CheckpointManager(checkpoint_dir=config["checkpoint_dir"])
    tb_writer = None
    wandb_run = None
    
    if rank == 0:
        tb_writer = SummaryWriter(log_dir=config["tensorboard_dir"])
        if config["wandb_enabled"]:
            import wandb
            wandb_run = wandb.init(project=config["wandb_project"], config=config)

    # Load checkpoint if resuming
    start_step = 0
    checkpoint_meta = ckpt_manager.load_latest_checkpoint(model, optimizer)
    if checkpoint_meta:
        start_step = checkpoint_meta["step"]
        scheduler.current_step = start_step
        if rank == 0:
            print(f"Resumed training from step {start_step}")

    # 5. Data Loading Setup
    # Assumes preprocessing has converted raw text into binary shards in `data/shards/`
    shards_dir = "data/shards"
    os.makedirs(shards_dir, exist_ok=True)
    
    # In practice, if empty we skip or prompt. We check if there are binary shards
    has_shards = len(os.listdir(shards_dir)) > 0
    if not has_shards:
        if rank == 0:
            print(f"Warning: No binary data found in '{shards_dir}'. Please run data engineering scripts first.")
        return

    train_loader = get_streaming_dataloader(
        bin_dir=shards_dir,
        max_seq_len=config["max_position_embeddings"],
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    # 6. Automatic Mixed Precision Setup
    # Adapt AMP type based on configuration and hardware support
    amp_dtype = torch.bfloat16 if config["mixed_precision"] == "bf16" else torch.float16
    # MPS does not support bfloat16 AMP in some ops, fallback to float32 if on MPS
    use_amp = device.type in ["cuda", "cpu"]
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=(config["mixed_precision"] == "fp16" and use_amp))

    # 7. Pre-training Loop
    model.train()
    step = start_step
    data_iterator = iter(train_loader)
    
    accumulated_loss = 0.0
    last_logged_loss = 0.0
    t0 = time.time()

    while step < config["max_steps"]:
        optimizer.zero_grad(set_to_none=True)
        
        # Inner loop for gradient accumulation
        for micro_step in range(config["gradient_accumulation_steps"]):
            try:
                x, y = next(data_iterator)
            except StopIteration:
                # Reload iterator when data ends
                data_iterator = iter(train_loader)
                x, y = next(data_iterator)

            x, y = x.to(device), y.to(device)

            # Mixed precision forward pass
            with torch.amp.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(x, labels=y)
                loss = outputs["loss"]
                # Normalize loss to account for gradient accumulation steps
                loss = loss / config["gradient_accumulation_steps"]

            accumulated_loss += loss.item()
            
            # Backward pass
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Gradient clipping
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            
        # Unwrapped parameters for clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

        # Optimizer Step
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Update LR
        lr = scheduler.step()
        step += 1

        # Logging & Diagnostics
        if step % config["logging_steps"] == 0 and rank == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            # Compute loss and tokens processed per second
            avg_loss = accumulated_loss / config["logging_steps"]
            last_logged_loss = avg_loss
            accumulated_loss = 0.0
            tokens_per_sec = (config["logging_steps"] * config["batch_size"] * config["gradient_accumulation_steps"] * config["max_position_embeddings"]) / dt
            
            print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:.1f} | Step time: {dt*1000:.1f}ms")
            
            if tb_writer:
                tb_writer.add_scalar("train/loss", avg_loss, step)
                tb_writer.add_scalar("train/lr", lr, step)
            if wandb_run:
                wandb.log({"train/loss": avg_loss, "train/lr": lr, "train/tokens_per_sec": tokens_per_sec}, step=step)

        # Checkpointing
        if step % config["save_steps"] == 0 and rank == 0:
            ckpt_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler_state={"current_step": scheduler.current_step},
                step=step,
                loss=last_logged_loss
            )

    if rank == 0:
        print("Pre-training completed successfully!")
        if tb_writer:
            tb_writer.close()
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    main()
