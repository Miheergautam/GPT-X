# data_loader.py
"""
Data loading and preprocessing utilities.
Handles reading text files, cleaning data, and creating train/validation splits.
"""

import torch
import re
from config import BATCH_SIZE, BLOCK_SIZE, DEVICE


def load_and_clean_text(file_path):
    """
    Load text file and apply cleaning for Hindi (Devanagari) text.

    Steps:
    1. Read file with UTF-8 encoding
    2. Keep only Hindi characters, spaces, and danda (।)
    3. Normalize multiple spaces
    4. Remove very short lines

    Args:
        file_path: Path to text file

    Returns:
        Cleaned text string
    """
    print(f"Loading text from: {file_path}")

    # Read the file
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    print(f"Original text length: {len(text)} characters")

    # Step 1: Keep only Hindi (Devanagari Unicode range), space, and danda
    # Devanagari Unicode range: U+0900 to U+097F
    text = re.sub(r"[^\u0900-\u097F\s।]", "", text)

    # Step 2: Normalize multiple spaces to single space
    text = re.sub(r"\s+", " ", text).strip()

    # Step 3: Remove very short lines (less than 5 characters)
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 5]
    text = " ".join(lines)

    print(f"Cleaned text length: {len(text)} characters")

    return text


def create_train_val_split(data, train_ratio=0.9):
    """
    Split data into training and validation sets.

    Args:
        data: Torch tensor of encoded tokens
        train_ratio: Fraction of data to use for training (default: 0.9)

    Returns:
        Tuple of (train_data, val_data)
    """
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Train data: {len(train_data)} tokens")
    print(f"Validation data: {len(val_data)} tokens")

    return train_data, val_data


def get_batch(split, train_data, val_data):
    """
    Generate a random batch of input-output pairs for training or validation.

    How it works:
    1. Select random starting positions in the data
    2. Extract sequences of length BLOCK_SIZE
    3. Create input (x) and target (y) where y is x shifted by 1 position

    Args:
        split: 'train' or 'val' to select dataset
        train_data: Training data tensor
        val_data: Validation data tensor

    Returns:
        Tuple of (x, y) where:
            x: Input sequences of shape (BATCH_SIZE, BLOCK_SIZE)
            y: Target sequences of shape (BATCH_SIZE, BLOCK_SIZE)
    """
    # Choose appropriate dataset
    data = train_data if split == "train" else val_data

    # Generate random starting indices for each sequence in the batch
    # Ensure we don't go past the end of data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    # Stack sequences into batch
    # x contains the input sequences
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])

    # y contains the target sequences (shifted by 1 position)
    # This means we're predicting the next token at each position
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    # Move tensors to appropriate device (CPU or GPU)
    x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y


class DataProcessor:
    """
    Main class for handling all data processing operations.
    Combines loading, cleaning, tokenization, and batching.
    """

    def __init__(self, file_path, tokenizer):
        """
        Initialize data processor.

        Args:
            file_path: Path to text data file
            tokenizer: BPE tokenizer instance
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None

    def prepare_data(self):
        """
        Complete data preparation pipeline:
        1. Load and clean text
        2. Train tokenizer
        3. Encode text to tokens
        4. Create train/val split
        """
        # Load and clean text
        print("Loading and cleaning text...")
        text = load_and_clean_text(self.file_path)

        # Train tokenizer on the text
        print("\nTraining BPE tokenizer...")
        self.tokenizer.train(text)

        # Encode the entire text
        print("\nEncoding text...")
        encoded_data = self.tokenizer.encode(text)
        data_tensor = torch.tensor(encoded_data, dtype=torch.long)

        print(f"Total tokens: {len(data_tensor)}")

        # Create train/validation split
        self.train_data, self.val_data = create_train_val_split(data_tensor)

        return self.train_data, self.val_data

    def get_batch(self, split):
        """
        Get a batch of data for training or validation.

        Args:
            split: 'train' or 'val'

        Returns:
            Tuple of (x, y) input and target tensors
        """
        return get_batch(split, self.train_data, self.val_data)
