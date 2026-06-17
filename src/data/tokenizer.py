import os
from typing import List, Union
from transformers import AutoTokenizer

class CustomTokenizer:
    """Wrapper around Hugging Face AutoTokenizer for the Typhoon 2.5 model."""
    def __init__(self, tokenizer_path_or_id: str = "scb10x/typhoon2.5-qwen3-4b"):
        # If tokenizer_path_or_id is a path that doesn't exist, we fall back to the HF model id
        if not tokenizer_path_or_id or (not os.path.exists(tokenizer_path_or_id) and "/" not in tokenizer_path_or_id):
            tokenizer_path_or_id = "scb10x/typhoon2.5-qwen3-4b"
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_id, use_fast=True)
        
        # Ensure pad_token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, files: List[str], vocab_size: int = 151669, min_frequency: int = 2) -> None:
        """Mock train method since Typhoon 2.5 tokenizer is pre-trained."""
        print("Using pre-trained Typhoon 2.5 tokenizer. Skipping custom BPE training.")

    def save(self, path: str) -> None:
        """Saves tokenizer configuration files locally."""
        self.tokenizer.save_pretrained(path)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes text into token IDs."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_special_tokens:
            bos_id = self.tokenizer.bos_token_id
            eos_id = self.tokenizer.eos_token_id
            if bos_id is not None:
                ids = [bos_id] + ids
            if eos_id is not None:
                ids = ids + [eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back to string."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def token_to_id(self, token: str) -> Union[int, None]:
        """Helper to get token ID for special identifiers or custom inputs."""
        # Translate legacy template tokens to Qwen/Typhoon equivalents
        if token == "<bos>":
            return self.tokenizer.bos_token_id
        if token == "<eos>":
            return self.tokenizer.eos_token_id
        if token == "<pad>":
            return self.tokenizer.pad_token_id
        if token == "<unk>":
            return self.tokenizer.unk_token_id or self.tokenizer.eos_token_id
            
        try:
            val = self.tokenizer.convert_tokens_to_ids(token)
            if val is not None and val != self.tokenizer.unk_token_id:
                return val
            return None
        except Exception:
            return None

    def id_to_token(self, idx: int) -> Union[str, None]:
        """Helper to translate token ID to string token."""
        try:
            return self.tokenizer.convert_ids_to_tokens(idx)
        except Exception:
            return None


def create_dummy_corpus(filepath: str):
    """Creates a small mixed corpus of Thai, English, and Python Code for tokenizer testing."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    thai_text = (
        "ภาษาไทยเป็นภาษาที่สวยงามและมีเอกลักษณ์ การเรียนรู้การเทรนโมเดลภาษาขนาดใหญ่ (LLM) "
        "ช่วยให้เราพัฒนาเทคโนโลยีที่ตอบสนองต่อภาษาไทยได้ดียิ่งขึ้น\n"
    )
    english_text = (
        "Large Language Models (LLMs) have revolutionized the field of Artificial Intelligence. "
        "Training a custom LLM from scratch allows full control over the model's vocabulary and weights.\n"
    )
    code_text = (
        "def main():\n"
        "    print('Hello World from scratch Custom LLM!')\n"
        "    x = [i for i in range(10) if i % 2 == 0]\n"
        "    return x\n"
    )
    
    with open(filepath, "w", encoding="utf-8") as f:
        for _ in range(50):
            f.write(thai_text)
            f.write(english_text)
            f.write(code_text)
            f.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load and inspect Typhoon 2.5 Tokenizer")
    parser.add_argument("--model_id", type=str, default="scb10x/typhoon2.5-qwen3-4b", help="Typhoon model repository ID")
    args = parser.parse_args()

    print(f"Loading Typhoon 2.5 Tokenizer: {args.model_id}...")
    tok = CustomTokenizer(args.model_id)
    print(f"Tokenizer loaded successfully. Vocab size: {tok.vocab_size}")
    
    # Test encoding/decoding
    test_str = "สวัสดีครับคุณไต้ฝุ่น! Let's test the Qwen/Typhoon tokenizer."
    ids = tok.encode(test_str, add_special_tokens=True)
    decoded = tok.decode(ids)
    print(f"Test string: {test_str}")
    print(f"Encoded IDs: {ids}")
    print(f"Decoded: {decoded}")
