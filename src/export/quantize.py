import os
import subprocess

def run_awq_quantization(hf_model_dir: str, quant_output_dir: str, w_bit: int = 4, group_size: int = 128):
    """Utility wrapper for AutoAWQ quantization.
    
    AWQ (Activation-aware Weight Quantization) is ideal for GPU-based low-latency inference.
    """
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        
        print(f"Loading HF model from {hf_model_dir} for AWQ quantization...")
        model = AutoAWQForCausalLM.from_pretrained(hf_model_dir, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, use_fast=True)
        
        # Configure AWQ options
        quant_config = {
            "zero_point": True,
            "q_group_size": group_size,
            "w_bit": w_bit,
            "version": "GEMM"
        }
        
        print(f"Quantizing model weights to {w_bit}-bit (group size {group_size})...")
        model.quantize(tokenizer, quant_config=quant_config)
        
        print(f"Saving quantized AWQ model to {quant_output_dir}...")
        model.save_quantized(quant_output_dir)
        tokenizer.save_pretrained(quant_output_dir)
        print("AWQ Quantization completed successfully!")
        
    except ImportError:
        print("\n[AWQ Notice] 'autoawq' is not installed. To run AWQ quantization, please install it using:")
        print("    pip install autoawq")
        print(f"Alternative: Convert model to GGUF format for local CPU/Mac hardware execution (see print_gguf_instructions).")


def print_gguf_instructions(hf_model_dir: str, gguf_output_name: str = "model_q4_k_m.gguf"):
    """Prints step-by-step commands to compile llama.cpp and quantize the model to GGUF format.
    
    GGUF is optimal for running on consumer hardware (especially macOS with unified memory).
    """
    print("=" * 80)
    print("📋 INSTRUCTIONS FOR GGUF CONVERSION (via llama.cpp)")
    print("=" * 80)
    print("Step 1: Clone and build llama.cpp on your machine:")
    print("    git clone https://github.com/ggerganov/llama.cpp.git")
    print("    cd llama.cpp")
    print("    make -j")
    print("\nStep 2: Install required python dependencies in the llama.cpp environment:")
    print("    pip install -r requirements.txt")
    print("\nStep 3: Convert the Hugging Face model folder into GGUF float16 format:")
    print(f"    python3 convert_hf_to_gguf.py {os.path.abspath(hf_model_dir)} --outfile model_f16.gguf")
    print("\nStep 4: Quantize the GGUF file to Q4_K_M (4-bit medium quantization, recommended balance):")
    print(f"    ./llama-quantize model_f16.gguf {gguf_output_name} Q4_K_M")
    print("\nStep 5: Run inference locally using llama.cpp CLI:")
    print(f"    ./llama-cli -m {gguf_output_name} -p \"<user>\\nสวัสดีครับแนะนำตัวหน่อย\\n<assistant>\\n\" -n 128")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantization helper for custom LLM models")
    parser.add_argument("--hf_model_dir", type=str, default="configs/hf_export", help="Path to exported Hugging Face model folder")
    parser.add_argument("--output_dir", type=str, default="checkpoints/awq_model", help="Path to save AWQ model")
    parser.add_argument("--mode", type=str, choices=["awq", "gguf", "all"], default="all", help="Quantization target mode")
    args = parser.parse_args()

    if args.mode in ["awq", "all"]:
        run_awq_quantization(args.hf_model_dir, args.output_dir)
        
    if args.mode in ["gguf", "all"]:
        print_gguf_instructions(args.hf_model_dir)
