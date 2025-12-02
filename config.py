# config.py
"""
Configuration file for GPT model hyperparameters and training settings.
This centralizes all model and training configurations for easy modification.
"""

"""
Hyperparameters for GPT-X model
"""

import torch

# # Context and batch settings -- 1st
# BLOCK_SIZE = 256    # Context length (increased from 128)
# BATCH_SIZE = 64     # Batch size (increased for better gradient estimates)

# Context and batch settings -- 2nd
BLOCK_SIZE = 512    # Context length (increased from 128)
BATCH_SIZE = 16     # Batch size (increased for better gradient estimates)

# TRAINING HYPERPARAMETERS -> 1st
# MAX_ITERS = 4000           # Training iterations (increased from 4000)
# EVAL_INTERVAL = 500         # Evaluate every 500 steps
# LEARNING_RATE = 3e-4        # Standard for Transformers
# EVAL_ITERS = 100            # Average over more batches for stable metrics

# TRAINING HYPERPARAMETERS -> 2nd
MAX_ITERS = 10000           # Training iterations (increased from 4000)
EVAL_INTERVAL = 500         # Evaluate every 500 steps
LEARNING_RATE = 3e-4        # Standard for Transformers
EVAL_ITERS = 200            # Average over more batches for stable metrics

# # Core dimensions 1st 
# N_EMBD = 384        # Embedding dimension (increased from 128)
# N_HEAD = 6          # Number of attention heads (384 / 6 = 64 per head)
# N_LAYER = 6         # Number of transformer layers (increased from 2)
# DROPOUT = 0.2       # Dropout rate (reduced from 0.3 for larger model)

# Core dimensions 1st 
N_EMBD = 384       # Embedding dimension (increased from 128)
N_HEAD = 6          # Number of attention heads (384 / 6 = 64 per head)
N_LAYER = 6         # Number of transformer layers (increased from 2)
DROPOUT = 0.1       # Dropout rate (reduced from 0.3 for larger model)

# Learning rate schedule (warmup + cosine decay)
WARMUP_ITERS = 1000         # Warmup steps
LR_DECAY_ITERS = 4000      # Total decay period
MIN_LR = 3e-5               # Minimum learning rate (10% of max)

# Gradient clipping for stability
GRAD_CLIP = 1.0             # Clip gradients to prevent exploding gradients

# TOKENIZER SETTINGS - Critical for Hindi
VOCAB_SIZE = 8000           
MIN_FREQUENCY = 3           # Minimum merge frequency (stricter filtering)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random seed for reproducibility
SEED = 1337
FINETUNE = True

# Data paths
# DATA_PATH = "/Users/miheergautam/Documents/GitHub/GPT-X/dataset.txt"
DATA_PATH = "./dataset.txt"
OUTPUT_PATH = "generated_output.txt"






# 1st
# 2nd

# 3rd -> blocksize 512, iterations = 10000, batchsize=16
# 4th model increase -> dblock = 384 heads= 6 and layers =6