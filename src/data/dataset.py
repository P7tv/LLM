import os
import glob
import random
import re
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Iterator, Tuple, Set

class MinHashDeduplicator:
    """Lightweight MinHash implementation for document near-deduplication."""
    def __init__(self, num_hashes: int = 64, threshold: float = 0.85):
        self.num_hashes = num_hashes
        self.threshold = threshold
        # Generate random coefficients for hash functions: h_i(x) = (a_i * x + b_i) % c
        # We use a large prime for c
        self.prime = 4294967291  # Largest prime < 2^32
        random.seed(42)
        self.a_coeffs = [random.randint(1, self.prime - 1) for _ in range(num_hashes)]
        self.b_coeffs = [random.randint(0, self.prime - 1) for _ in range(num_hashes)]
        
        # Determine LSH parameters (b bands, r rows) automatically
        best_b, best_r = num_hashes, 1
        min_error = float('inf')
        for b in range(1, num_hashes + 1):
            if num_hashes % b == 0:
                r = num_hashes // b
                t = (1.0 / b) ** (1.0 / r)
                error = abs(t - threshold)
                if error < min_error:
                    min_error = error
                    best_b, best_r = b, r
        self.b = best_b
        self.r = best_r
        
        self.buckets = [{} for _ in range(self.b)]
        self.seen_signatures: List[np.ndarray] = []

    def _get_shingles(self, text: str, k: int = 5) -> Set[str]:
        """Creates character-level k-grams from text."""
        clean_text = re.sub(r'\s+', '', text.lower())
        shingles = set()
        for i in range(len(clean_text) - k + 1):
            shingles.add(clean_text[i:i+k])
        return shingles

    def _hash_shingle(self, shingle: str) -> int:
        """Basic 32-bit FNV-1a hash of a shingle."""
        h = 2166136261
        for char in shingle.encode('utf-8', 'ignore'):
            h = h ^ char
            h = (h * 16777619) & 0xffffffff
        return h

    def compute_signature(self, text: str) -> np.ndarray:
        """Computes the MinHash signature of a document."""
        shingles = self._get_shingles(text)
        if not shingles:
            return np.full(self.num_hashes, self.prime - 1, dtype=np.uint32)
            
        shingle_hashes = [self._hash_shingle(s) for s in shingles]
        signature = np.full(self.num_hashes, self.prime, dtype=np.uint64)
        
        for val in shingle_hashes:
            for i in range(self.num_hashes):
                h_val = (self.a_coeffs[i] * val + self.b_coeffs[i]) % self.prime
                if h_val < signature[i]:
                    signature[i] = h_val
        return signature.astype(np.uint32)

    def is_duplicate(self, text: str) -> bool:
        """Checks if the document is a near-duplicate of a previously seen document."""
        sig = self.compute_signature(text)
        
        # LSH Bucket lookup
        candidates = set()
        for i in range(self.b):
            band = tuple(sig[i * self.r : (i + 1) * self.r])
            if band in self.buckets[i]:
                candidates.update(self.buckets[i][band])
                
        for cand_idx in candidates:
            seen_sig = self.seen_signatures[cand_idx]
            similarity = np.mean(seen_sig == sig)
            if similarity >= self.threshold:
                return True
                
        # Register new signature
        sig_idx = len(self.seen_signatures)
        self.seen_signatures.append(sig)
        for i in range(self.b):
            band = tuple(sig[i * self.r : (i + 1) * self.r])
            if band not in self.buckets[i]:
                self.buckets[i][band] = []
            self.buckets[i][band].append(sig_idx)
            
        return False


class QualityFilter:
    """Performs heuristic checks to filter low-quality texts."""
    def __init__(self, min_char_len: int = 50, max_char_len: int = 500000, 
                 min_thai_ratio: float = 0.0, max_repeat_ratio: float = 0.3):
        self.min_char_len = min_char_len
        self.max_char_len = max_char_len
        self.min_thai_ratio = min_thai_ratio
        self.max_repeat_ratio = max_repeat_ratio

    def is_high_quality(self, text: str) -> bool:
        # Check basic length constraints
        text_len = len(text)
        if text_len < self.min_char_len or text_len > self.max_char_len:
            return False
            
        # Optional: check Thai character ratio if specified
        if self.min_thai_ratio > 0.0:
            thai_chars = len(re.findall(r'[\u0e00-\u0e7f]', text))
            if (thai_chars / text_len) < self.min_thai_ratio:
                return False

        # Check for repetitive lines or symbols (e.g. spam)
        lines = text.split('\n')
        if len(lines) > 5:
            unique_lines = set(lines)
            repeat_ratio = 1.0 - (len(unique_lines) / len(lines))
            if repeat_ratio > self.max_repeat_ratio:
                return False

        return True


def tokenize_to_bin(texts: List[str], tokenizer, output_bin_path: str, 
                    deduplicator: MinHashDeduplicator = None, 
                    quality_filter: QualityFilter = None) -> int:
    """Preprocesses, tokenizes and saves texts as a binary uint32 token buffer."""
    os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)
    all_tokens = []
    
    for text in texts:
        # 1. Quality Filter
        if quality_filter and not quality_filter.is_high_quality(text):
            continue
            
        # 2. Near-Deduplication
        if deduplicator and deduplicator.is_duplicate(text):
            continue
            
        # 3. Tokenization (adds BOS/EOS)
        tokens = tokenizer.encode(text, add_special_tokens=True)
        all_tokens.extend(tokens)
        
    if not all_tokens:
        return 0

    # Write as a continuous uint32 array (supports vocab sizes > 65k)
    np_tokens = np.array(all_tokens, dtype=np.uint32)
    np_tokens.tofile(output_bin_path)
    return len(np_tokens)


class StreamingDataset(IterableDataset):
    """Memory-mapped streaming dataset that reads pre-tokenized binary files.
    
    Yields chunks of sequence length: max_seq_len + 1 (for input and target alignment).
    """
    def __init__(self, bin_dir: str, max_seq_len: int, shuffle: bool = True):
        super().__init__()
        self.bin_dir = bin_dir
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self.bin_files = sorted(glob.glob(os.path.join(bin_dir, "*.bin")))
        if not self.bin_files:
            raise FileNotFoundError(f"No .bin files found in directory {bin_dir}")

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        
        # Partition files across multiple DataLoader workers if applicable
        files = list(self.bin_files)
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = [f for i, f in enumerate(files) if i % num_workers == worker_id]
            
        if self.shuffle:
            random.shuffle(files)

        chunk_size = self.max_seq_len + 1
        
        for file_path in files:
            # Memory map the binary file (zero memory overhead)
            # data type is uint32 as saved in sharding phase
            token_array = np.memmap(file_path, dtype=np.uint32, mode='r')
            num_tokens = len(token_array)
            
            if num_tokens <= chunk_size:
                continue

            # Calculate total chunks possible in this file
            max_start_idx = num_tokens - chunk_size
            
            # Create indexing list
            start_indices = list(range(0, max_start_idx, self.max_seq_len))
            if self.shuffle:
                random.shuffle(start_indices)
                
            for start_idx in start_indices:
                end_idx = start_idx + chunk_size
                chunk = token_array[start_idx:end_idx].astype(np.int64) # Convert to torch-compatible int64
                
                # Split into inputs (x) and targets (y)
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


def get_streaming_dataloader(bin_dir: str, max_seq_len: int, batch_size: int, 
                             shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Returns a PyTorch DataLoader wrapping the StreamingDataset."""
    dataset = StreamingDataset(bin_dir, max_seq_len=max_seq_len, shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    # Small self-check
    from tokenizer import CustomTokenizer, create_dummy_corpus
    
    print("Self-checking data engineering pipeline...")
    # 1. Train local tokenizer on dummy data
    dummy_text = "scratch/dummy_corpus.txt"
    create_dummy_corpus(dummy_text)
    
    tok = CustomTokenizer()
    tok.train([dummy_text], vocab_size=1000)
    tok_path = "scratch/test_tokenizer.json"
    tok.save(tok_path)
    
    # 2. Preprocess, Deduplicate and write to bin
    corpus_texts = [
        "นี่คือข้อความภาษาไทยแถวแรกสำหรับการตรวจสอบความถูกต้องของท่อส่งข้อมูลการเทรน",
        "นี่คือข้อความภาษาไทยแถวแรกสำหรับการตรวจสอบความถูกต้องของท่อส่งข้อมูลการเทรน", # Exact duplicate
        "นี่คือข้อความภาษาไทยแถวแรกสำหรับการตรวจสอบความถูกต้องของท่อส่งข้อมูลการเทรน!!", # Near duplicate
        "Python testing code. def function(): return True. Short sentence.",
        "Short.", # Quality filtered out (too short)
    ]
    
    dedup = MinHashDeduplicator(num_hashes=16, threshold=0.8)
    q_filter = QualityFilter(min_char_len=20)
    
    bin_path = "scratch/shards/shard_001.bin"
    total_tokens = tokenize_to_bin(
        corpus_texts, tok, bin_path, deduplicator=dedup, quality_filter=q_filter
    )
    print(f"Tokenized text into binary file. Total tokens written: {total_tokens}")
    
    # 3. Stream from file using dataset
    dataset = StreamingDataset("scratch/shards", max_seq_len=8, shuffle=True)
    loader = DataLoader(dataset, batch_size=2)
    
    print("\nReading batches from Streaming DataLoader:")
    for i, (x, y) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  Inputs (x) shape: {x.shape} -> {x[0].tolist()}")
        print(f"  Targets (y) shape: {y.shape} -> {y[0].tolist()}")
        if i >= 1:
            break
