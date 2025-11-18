"""
Fast Data loading and preprocessing for GPT-X using HuggingFace BPE Tokenizer.
Handles:
1. Reading + cleaning Hindi text
2. Tokenization via Rust BPE (super fast)
3. Train/Val split
4. Batch generation
"""

import torch
import re
from config import BATCH_SIZE, BLOCK_SIZE, DEVICE


# ----------------------------------------------
#  CLEANING (same as before)
# ----------------------------------------------
def load_and_clean_text(file_path):
    print(f"Loading text from: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    print(f"Original text length: {len(text)} characters")

    # Keep only Hindi + spaces + danda
    text = re.sub(r"[^\u0900-\u097F\s।]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove short lines
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 5]
    text = " ".join(lines)

    print(f"Cleaned text length: {len(text)} characters")
    return text


# ----------------------------------------------
#  TRAIN/VAL SPLIT
# ----------------------------------------------
def create_train_val_split(data, train_ratio=0.9):
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Train tokens: {len(train_data)}")
    print(f"Val tokens: {len(val_data)}")

    return train_data, val_data


# ----------------------------------------------
#  BATCHING
# ----------------------------------------------
def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data

    # Random start positions
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    # Build batch
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    return x.to(DEVICE), y.to(DEVICE)


# ----------------------------------------------
#  DATAPROCESSOR with FAST BPE
# ----------------------------------------------
class DataProcessor:
    """
    Uses the fast HuggingFace BPE tokenizer instead of slow Python BPE.
    """

    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer  # FastBPETokenizer instance
        self.train_data = None
        self.val_data = None

    def prepare_data(self):
        """
        1. Clean text
        2. (Tokenizer is already trained or loaded in main)
        3. Encode fast with Rust BPE
        4. Create train/val tensors
        """
        print("Loading and cleaning text...")
        text = load_and_clean_text(self.file_path)

        # We DO NOT train the tokenizer here — done in main.py
        print("\nEncoding text (Fast BPE)...")
        encoded = self.tokenizer.encode(text)

        print(f"Total tokens: {len(encoded)}")

        data_tensor = torch.tensor(encoded, dtype=torch.long)

        # Train/validation split
        self.train_data, self.val_data = create_train_val_split(data_tensor)

        return self.train_data, self.val_data

    def get_batch(self, split):
        return get_batch(split, self.train_data, self.val_data)
