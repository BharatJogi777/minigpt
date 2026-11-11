from pathlib import Path
import torch
from minigpt.tokenizer import CharTokenizer

def load_corpus(path):
    text = Path(path).read_text(encoding="utf-8")
    if len(text) < 2: raise ValueError("Corpus must contain at least two characters")
    return text

def split_encoded_data(text, tokenizer, train_fraction=0.9):
    if not 0 < train_fraction < 1: raise ValueError("train_fraction must be between 0 and 1")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(len(tokens) * train_fraction)
    if split < 1 or len(tokens)-split < 1: raise ValueError("Corpus too small for split")
    return tokens[:split], tokens[split:]

def get_batch(data, batch_size, block_size, device):
    if batch_size < 1 or block_size < 1: raise ValueError("Batch and block sizes must be positive")
    if data.numel() <= block_size: raise ValueError("Dataset must be longer than block_size")
    starts = torch.randint(0, data.numel()-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in starts])
    y = torch.stack([data[i+1:i+block_size+1] for i in starts])
    return x.to(device), y.to(device)
