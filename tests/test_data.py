import os
import shutil
import numpy as np
import pytest
import torch
from src.data.tokenizer import CustomTokenizer, create_dummy_corpus
from src.data.dataset import MinHashDeduplicator, QualityFilter, tokenize_to_bin, StreamingDataset

TEMP_DIR = "scratch/test_runs_data"

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    os.makedirs(TEMP_DIR, exist_ok=True)
    yield
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

def test_tokenizer_train_save_load():
    corpus_file = os.path.join(TEMP_DIR, "corpus.txt")
    create_dummy_corpus(corpus_file)
    
    tokenizer_dir = os.path.join(TEMP_DIR, "tokenizer_dir")
    
    # Load pre-trained Typhoon 2.5 tokenizer (mock training is skipped)
    tok = CustomTokenizer()
    tok.train([corpus_file], vocab_size=151669)
    tok.save(tokenizer_dir)
    
    assert os.path.exists(tokenizer_dir)
    
    # Reload from local dir
    tok2 = CustomTokenizer(tokenizer_dir)
    assert tok2.vocab_size == tok.vocab_size
    
    text = "Hello World! สวัสดีครับ"
    ids1 = tok.encode(text, add_special_tokens=True)
    ids2 = tok2.encode(text, add_special_tokens=True)
    assert ids1 == ids2
    
    # Check BOS/EOS tokens
    # Qwen/Typhoon doesn't use a default BOS token, but has an EOS token.
    assert ids1[-1] == tok.token_to_id("<eos>")
    
    decoded = tok2.decode(ids1)
    assert "Hello World" in decoded

def test_minhash_deduplicator():
    dedup = MinHashDeduplicator(num_hashes=32, threshold=0.8)
    
    doc1 = "The quick brown fox jumps over the lazy dog to test code."
    doc2 = "The quick brown fox jumps over the lazy dog to test code." # Exact duplicate
    doc3 = "The quick brown fox jumps over the lazy dog to test code!!!" # Near duplicate
    doc4 = "This is a completely different document about deep learning neural nets." # Unique
    
    assert not dedup.is_duplicate(doc1)
    assert dedup.is_duplicate(doc2)
    assert dedup.is_duplicate(doc3)
    assert not dedup.is_duplicate(doc4)

def test_quality_filter():
    q_filter = QualityFilter(min_char_len=15, max_repeat_ratio=0.2)
    
    good_doc = "This is a valid test document that meets quality standards."
    short_doc = "Too short."
    spam_doc = "Spam line.\nSpam line.\nSpam line.\nSpam line.\nSpam line.\nSpam line."
    
    assert q_filter.is_high_quality(good_doc)
    assert not q_filter.is_high_quality(short_doc)
    assert not q_filter.is_high_quality(spam_doc)

def test_binary_sharding_and_streaming_dataset():
    corpus_file = os.path.join(TEMP_DIR, "corpus_shard.txt")
    create_dummy_corpus(corpus_file)
    
    tok = CustomTokenizer()
    
    texts = [
        "First example sentences to process.",
        "Second example sentence for pre-processing.",
        "Another unique sentence that passes filters."
    ]
    
    bin_dir = os.path.join(TEMP_DIR, "shards")
    bin_file = os.path.join(bin_dir, "shard_0.bin")
    
    tokens_written = tokenize_to_bin(texts, tok, bin_file)
    assert tokens_written > 0
    assert os.path.exists(bin_file)
    
    # Read back size and assert type
    data = np.fromfile(bin_file, dtype=np.uint32)
    assert len(data) == tokens_written
    
    # Test StreamingDataset
    max_seq_len = 16
    dataset = StreamingDataset(bin_dir, max_seq_len=max_seq_len, shuffle=False)
    
    # Test iterator
    inputs_targets = list(dataset)
    assert len(inputs_targets) > 0
    
    x, y = inputs_targets[0]
    assert x.shape == (max_seq_len,)
    assert y.shape == (max_seq_len,)
    # Target should be shifted by 1 relative to input
    assert torch.equal(x[1:], y[:-1])
