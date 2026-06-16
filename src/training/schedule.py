import math

def get_cosine_lr_with_warmup(
    step: int, 
    warmup_steps: int, 
    max_steps: int, 
    learning_rate: float, 
    min_learning_rate: float
) -> float:
    """Calculates learning rate with linear warmup and cosine decay.
    
    Matches modern pre-training schedules.
    """
    # 1. Linear Warm-up Phase
    if step < warmup_steps:
        return learning_rate * (step + 1) / (warmup_steps + 1)
        
    # 2. Beyond Max Steps (constant minimum learning rate)
    if step > max_steps:
        return min_learning_rate
        
    # 3. Cosine Decay Phase
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    return min_learning_rate + coeff * (learning_rate - min_learning_rate)


class CosineWarmupScheduler:
    """Helper wrapper around get_cosine_lr_with_warmup to manage optimizer rates."""
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, 
                 learning_rate: float, min_learning_rate: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.current_step = 0

    def step(self):
        lr = get_cosine_lr_with_warmup(
            step=self.current_step,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            min_learning_rate=self.min_learning_rate
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr
