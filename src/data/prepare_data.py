import os
import glob
import argparse
import json
import numpy as np
from typing import List, Generator

# Import local dataset and tokenizer modules
from src.data.tokenizer import CustomTokenizer
from src.data.dataset import MinHashDeduplicator, QualityFilter, tokenize_to_bin

def document_generator(input_path: str, text_key: str = "text") -> Generator[str, None, None]:
    """Scans and yields documents from files (.txt, .jsonl, .json, .parquet) in the input path."""
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        # Scan for supported file patterns recursively
        files = []
        for ext in ["*.txt", "*.jsonl", "*.json", "*.parquet"]:
            files.extend(glob.glob(os.path.join(input_path, "**", ext), recursive=True))
    else:
        raise FileNotFoundError(f"Input path {input_path} does not exist.")
        
    print(f"Found {len(files)} raw files to process.")
    
    for file_path in sorted(files):
        print(f"Reading file: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".txt":
            # Treat each paragraph or the whole file as a document
            # Here we treat the whole file as a single document
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        yield content
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                
        elif ext == ".jsonl":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        if text_key in data and isinstance(data[text_key], str):
                            yield data[text_key]
            except Exception as e:
                print(f"Error reading JSONL {file_path}: {e}")
                
        elif ext == ".json":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and text_key in item and isinstance(item[text_key], str):
                                yield item[text_key]
                            elif isinstance(item, str):
                                yield item
                    elif isinstance(data, dict) and text_key in data and isinstance(data[text_key], str):
                        yield data[text_key]
            except Exception as e:
                print(f"Error reading JSON {file_path}: {e}")
                
        elif ext == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)
                if text_key in df.columns:
                    for val in df[text_key].dropna():
                        if isinstance(val, str):
                            yield val
                else:
                    print(f"Column '{text_key}' not found in parquet file {file_path}")
            except ImportError:
                print("Pandas/Fastparquet required for Parquet files. Please install them or avoid parquet files.")
            except Exception as e:
                print(f"Error reading Parquet {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Frontier Custom Thai LLM Data Preparation Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Path to raw text directory or file. If None, runs in mock verify mode.")
    parser.add_argument("--output_dir", type=str, default="data/shards", help="Output directory for binary token shards")
    parser.add_argument("--text_key", type=str, default="text", help="Key for text/content column in JSON/Parquet files")
    parser.add_argument("--tokenizer_id", type=str, default="scb10x/typhoon2.5-qwen3-4b", help="Typhoon model tokenizer ID")
    parser.add_argument("--shard_size", type=int, default=1000000, help="Maximum tokens written per binary shard file")
    parser.add_argument("--min_len", type=int, default=50, help="Minimum character length filter")
    parser.add_argument("--max_repeat", type=float, default=0.3, help="Maximum repetitive line ratio filter")
    parser.add_argument("--dedup_threshold", type=float, default=0.85, help="MinHash similarity threshold for deduplication")
    args = parser.parse_args()

    # 1. Initialize Tokenizer and Filters
    print(f"Loading Typhoon Tokenizer: {args.tokenizer_id}")
    tokenizer = CustomTokenizer(args.tokenizer_id)
    
    q_filter = QualityFilter(min_char_len=args.min_len, max_repeat_ratio=args.max_repeat)
    deduplicator = MinHashDeduplicator(num_hashes=64, threshold=args.dedup_threshold)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 2. Check if we need to run in Mock Mode (if no input is provided)
    input_path = args.input
    if not input_path:
        input_path = "scratch/raw_mock_data"
        print(f"\n[Mock Mode] No input path specified. Generating mock raw files in '{input_path}'...")
        os.makedirs(input_path, exist_ok=True)
        
        # Write mock files
        with open(os.path.join(input_path, "sample1.txt"), "w", encoding="utf-8") as f:
            f.write("นี่คือข้อความตัวอย่างแรกสำหรับการเตรียมคลังข้อมูลก่อนป้อนเข้าเทรนโมเดลไต้ฝุ่น\n")
            f.write("This is an English paragraph to check bilingual text tokenization processing pipeline.\n")
            
        with open(os.path.join(input_path, "sample2.jsonl"), "w", encoding="utf-8") as f:
            # First item passes
            f.write(json.dumps({"text": "ข้อความแถวที่สองที่มีความยาวพอสมควรและจะนำไปเขียนเข้ารหัสเป็นโทเค็น"}) + "\n")
            # Duplicate item will be deduplicated
            f.write(json.dumps({"text": "ข้อความแถวที่สองที่มีความยาวพอสมควรและจะนำไปเขียนเข้ารหัสเป็นโทเค็น"}) + "\n")
            # Short item will be filtered out by quality filter
            f.write(json.dumps({"text": "สั้น"}) + "\n")

    # 3. Preprocess and Shard Documents
    print(f"\nStarting data preparation pipeline...")
    print(f"Input: {input_path}")
    print(f"Output Directory: {args.output_dir}")
    print("-" * 50)
    
    shard_idx = 1
    token_buffer = []
    
    total_docs = 0
    passed_docs = 0
    dup_docs = 0
    low_quality_docs = 0
    total_tokens_written = 0
    
    for doc in document_generator(input_path, text_key=args.text_key):
        total_docs += 1
        
        # Apply quality filter
        if not q_filter.is_high_quality(doc):
            low_quality_docs += 1
            continue
            
        # Apply MinHash deduplication
        if deduplicator.is_duplicate(doc):
            dup_docs += 1
            continue
            
        passed_docs += 1
        
        # Encode document text (adds BOS/EOS)
        tokens = tokenizer.encode(doc, add_special_tokens=True)
        token_buffer.extend(tokens)
        
        # Shard buffer if it exceeds shard size limit
        while len(token_buffer) >= args.shard_size:
            shard_tokens = token_buffer[:args.shard_size]
            token_buffer = token_buffer[args.shard_size:]
            
            shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.bin")
            np.array(shard_tokens, dtype=np.uint32).tofile(shard_path)
            
            total_tokens_written += len(shard_tokens)
            print(f"Saved binary shard: {shard_path} (Tokens: {len(shard_tokens)})")
            shard_idx += 1
            
    # Write remaining tokens in the buffer
    if token_buffer:
        shard_path = os.path.join(args.output_dir, f"shard_{shard_idx:05d}.bin")
        np.array(token_buffer, dtype=np.uint32).tofile(shard_path)
        total_tokens_written += len(token_buffer)
        print(f"Saved final binary shard: {shard_path} (Tokens: {len(token_buffer)})")
        
    print("-" * 50)
    print("📋 DATA PREPARATION SUMMARY")
    print("-" * 50)
    print(f"Total documents processed : {total_docs}")
    print(f"Low quality filtered out  : {low_quality_docs}")
    print(f"Near-duplicates removed   : {dup_docs}")
    print(f"Passed documents          : {passed_docs}")
    print(f"Total tokens sharded (.bin): {total_tokens_written}")
    print(f"Output directory          : {os.path.abspath(args.output_dir)}")
    print("-" * 50)


if __name__ == "__main__":
    main()
