# fast_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class FastBPETokenizer:
    def __init__(self, vocab_size=8000, min_frequency=3):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = None

    def train(self, file_path):
        """
        Train a BPE tokenizer on dataset.txt.

        Args:
            file_path: path to the raw text file
        """
        print("Training fast BPE tokenizer...")

        # BPE model (Rust backend)
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        # Whitespace pre-tokenizer (needed for Hindi words)
        tokenizer.pre_tokenizer = Whitespace()

        # Trainer settings
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["[PAD]", "[UNK]"],
        )

        # Train on file
        tokenizer.train([file_path], trainer)

        self.tokenizer = tokenizer
        print("Training complete.")
        print(f"Final vocab size: {tokenizer.get_vocab_size()}")

    def encode(self, text):
        """Fast encoding → returns list of IDs"""
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        """Fast decoding"""
        return self.tokenizer.decode(ids)

    def save(self, path):
        """Save tokenizer JSON"""
        self.tokenizer.save(path)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        """Load tokenizer JSON"""
        self.tokenizer = Tokenizer.from_file(path)
        print(f"Loaded tokenizer from {path}")

    def get_vocab_size(self):
        """Safe for Pyright — guarantees tokenizer is not None."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized.")
        return self.tokenizer.get_vocab_size()