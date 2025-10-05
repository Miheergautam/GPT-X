# config.py
"""
Configuration file for GPT model hyperparameters and training settings.
This centralizes all model and training configurations for easy modification.
"""

import torch

# Training hyperparameters
BATCH_SIZE = 32  # Number of sequences processed in parallel
BLOCK_SIZE = 128  # Maximum context length for predictions
MAX_ITERS = 4000  # Total training iterations
EVAL_INTERVAL = 500  # Evaluate model every N iterations
LEARNING_RATE = 3e-4  # Learning rate for optimizer
EVAL_ITERS = 50  # Number of iterations to average for loss estimation

# Model architecture hyperparameters
N_EMBD = 128  # Embedding dimension size
N_HEAD = 4  # Number of attention heads
N_LAYER = 2  # Number of transformer blocks
DROPOUT = 0.3  # Dropout rate for regularization

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random seed for reproducibility
SEED = 1337

# Data paths
DATA_PATH = "/Users/miheergautam/Documents/GitHub/GPT-X/dataset.txt"
OUTPUT_PATH = "generated_output.txt"

# BPE tokenizer settings
VOCAB_SIZE = 500  # Target vocabulary size for BPE (will be adjusted during training)
MIN_FREQUENCY = 2  # Minimum frequency for BPE merges
