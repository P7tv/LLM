import os
import yaml
import torch
try:
    from transformers import Qwen3Config, Qwen3ForCausalLM
except ImportError:
    # Local fallback for version < 4.51.0 (dev/testing environment)
    from transformers import Qwen2Config as Qwen3Config
    from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM
from src.data.tokenizer import CustomTokenizer

def convert_to_hf(model_ckpt_path: str, hf_output_dir: str, config_path: str = "configs/tiny_llm.yaml"):
    """Converts custom model weights to standard Hugging Face Qwen2ForCausalLM format.
    
    Qwen2 architecture natively supports QK-Normalization, GQA, and sliding window attention.
    """
    import warnings
    warnings.warn(
        "Exporting to Qwen3ForCausalLM (Qwen2 architecture). "
        "Note that Qwen2 uses a single sliding window config for the entire model. "
        "The layer-wise hybrid-attention pattern (local x3 + global) of this model "
        "will NOT be preserved in the exported Hugging Face model. "
        "The model's attention behavior will differ from training. "
        "Consider mapping to an architecture that supports layer-wise patterns (e.g., Gemma2) if needed."
    )
    
    os.makedirs(hf_output_dir, exist_ok=True)
    
    # 1. Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load tokenizer to grab token IDs
    tokenizer = CustomTokenizer()

    # 2. Define Qwen3 Configuration
    qk_norm_enabled = config.get("qk_norm", False)
    
    hf_config = Qwen3Config(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        rms_norm_eps=config["rms_norm_eps"],
        max_position_embeddings=config["max_position_embeddings"],
        rope_theta=config["rope_theta"],
        bos_token_id=tokenizer.tokenizer.bos_token_id,
        eos_token_id=tokenizer.tokenizer.eos_token_id,
        pad_token_id=tokenizer.tokenizer.pad_token_id,
        tie_word_embeddings=config.get("tie_word_embeddings", False),
        use_qk_norm=qk_norm_enabled,
        sliding_window=config.get("sliding_window", 4096),
        use_sliding_window=(config.get("sliding_window") is not None)
    )
    # Ensure exported model is serialized as a Qwen3 architecture
    hf_config.model_type = "qwen3"
    
    print("Initializing HF Qwen3ForCausalLM model with target configuration...")
    hf_model = Qwen3ForCausalLM(hf_config)
    
    # 3. Load custom model state dict
    print(f"Loading custom checkpoint from {model_ckpt_path}...")
    custom_state = torch.load(model_ckpt_path, map_location="cpu")
    if "model_state_dict" in custom_state:
        custom_weights = custom_state["model_state_dict"]
    else:
        custom_weights = custom_state
        
    # Filter out MTP keys from custom state before mapping
    custom_weights = {k: v for k, v in custom_weights.items() if "mtp" not in k}
        
    new_state_dict = {}

    # 4. Map custom parameters to HF parameter naming convention
    new_state_dict["model.embed_tokens.weight"] = custom_weights["embed_tokens.weight"]
    
    for i in range(config["num_hidden_layers"]):
        # Attention Projections
        new_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = custom_weights[f"layers.{i}.attn.q_proj.weight"]
        new_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = custom_weights[f"layers.{i}.attn.k_proj.weight"]
        new_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = custom_weights[f"layers.{i}.attn.v_proj.weight"]
        new_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = custom_weights[f"layers.{i}.attn.o_proj.weight"]
        
        # QK-Normalization layers (if active)
        if qk_norm_enabled:
            new_state_dict[f"model.layers.{i}.self_attn.q_norm.weight"] = custom_weights[f"layers.{i}.attn.q_norm.weight"]
            new_state_dict[f"model.layers.{i}.self_attn.k_norm.weight"] = custom_weights[f"layers.{i}.attn.k_norm.weight"]
        
        # SwiGLU FFN
        new_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = custom_weights[f"layers.{i}.ffn.gate_proj.weight"]
        new_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = custom_weights[f"layers.{i}.ffn.up_proj.weight"]
        new_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = custom_weights[f"layers.{i}.ffn.down_proj.weight"]
        
        # Normalization Layers
        new_state_dict[f"model.layers.{i}.input_layernorm.weight"] = custom_weights[f"layers.{i}.attn_norm.weight"]
        new_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = custom_weights[f"layers.{i}.ffn_norm.weight"]
        
    # Final Normalization and Output Head
    new_state_dict["model.norm.weight"] = custom_weights["norm.weight"]
    new_state_dict["lm_head.weight"] = custom_weights["lm_head.weight"]

    # 5. Load mapped weights and save
    print("Mapping weights to Hugging Face model state dict...")
    hf_model.load_state_dict(new_state_dict, strict=False)
    
    # Save model as standard safetensors
    hf_model.save_pretrained(hf_output_dir, safe_serialization=True)
    hf_config.save_pretrained(hf_output_dir)
    # Save the tokenizer files alongside
    tokenizer.save(hf_output_dir)
    
    print(f"Hugging Face compatible Qwen3 model saved successfully at: {hf_output_dir}")


if __name__ == "__main__":
    print("HF weight converter loaded. Run convert_to_hf(checkpoint_path, output_dir) to map models.")
