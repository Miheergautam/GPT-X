# model.py
"""
GPT (Generative Pre-trained Transformer) model implementation.
Contains all the neural network components: attention, feedforward, and transformer blocks.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import N_EMBD, N_HEAD, N_LAYER, DROPOUT, BLOCK_SIZE, DEVICE


class Head(nn.Module):
    """
    Single head of self-attention mechanism.

    Self-attention allows each token to look at all previous tokens
    and decide which ones are most relevant for predicting the next token.
    """

    def __init__(self, head_size):
        """
        Initialize single attention head.

        Args:
            head_size: Dimension of query, key, and value vectors
        """
        super().__init__()

        # Three linear transformations for attention mechanism
        # Key: "What do I contain?"
        self.key = nn.Linear(N_EMBD, head_size, bias=False)

        # Query: "What am I looking for?"
        self.query = nn.Linear(N_EMBD, head_size, bias=False)

        # Value: "What do I communicate if I'm relevant?"
        self.value = nn.Linear(N_EMBD, head_size, bias=False)

        # Register lower triangular mask for causal attention
        # This ensures tokens can only attend to previous tokens, not future ones
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        # Dropout for regularization
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Forward pass of attention head.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd)

        Returns:
            Output tensor of shape (batch_size, sequence_length, head_size)
        """
        B, T, C = x.shape  # Batch size, sequence length, embedding dimension

        # Generate key, query, value vectors
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compute attention scores (affinities between tokens)
        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)

        # Apply causal mask: prevent attending to future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # Normalize attention scores to probabilities
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Apply dropout
        wei = self.dropout(wei)

        # Weighted aggregation of values
        out = wei @ v  # (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel.

    Different heads can learn to attend to different aspects of the input,
    making the model more powerful.
    """

    def __init__(self, num_heads, head_size):
        """
        Initialize multi-head attention.

        Args:
            num_heads: Number of parallel attention heads
            head_size: Dimension of each head
        """
        super().__init__()

        # Create multiple attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # Output projection to combine all heads
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)

        # Dropout for regularization
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Forward pass through all attention heads.

        Args:
            x: Input tensor of shape (B, T, N_EMBD)

        Returns:
            Output tensor of shape (B, T, N_EMBD)
        """
        # Run input through all heads and concatenate outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Project concatenated outputs
        out = self.dropout(self.proj(out))

        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Applied independently to each position after attention.
    Allows the model to process information gathered by attention.
    """

    def __init__(self, n_embd):
        """
        Initialize feed-forward network.

        Args:
            n_embd: Embedding dimension
        """
        super().__init__()

        # Two-layer MLP with expansion factor of 4
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand dimension
            nn.ReLU(),  # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Project back
            nn.Dropout(DROPOUT),  # Regularization
        )

    def forward(self, x):
        """
        Forward pass through feed-forward network.

        Args:
            x: Input tensor

        Returns:
            Output tensor (same shape as input)
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: the fundamental building block of GPT.

    Combines self-attention (for communication between tokens)
    and feed-forward network (for computation on individual tokens).
    """

    def __init__(self, n_embd, n_head):
        """
        Initialize transformer block.

        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
        """
        super().__init__()

        head_size = n_embd // n_head

        # Self-attention for token communication
        self.sa = MultiHeadAttention(n_head, head_size)

        # Feed-forward for token-wise computation
        self.ffwd = FeedForward(n_embd)

        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass with residual connections.

        Residual connections help with gradient flow during training.

        Args:
            x: Input tensor

        Returns:
            Output tensor (same shape as input)
        """
        # Attention block with residual connection
        x = x + self.sa(self.ln1(x))

        # Feed-forward block with residual connection
        x = x + self.ffwd(self.ln2(x))

        return x


class GPTLanguageModel(nn.Module):
    """
    Complete GPT language model.

    Architecture:
    1. Token + Position embeddings
    2. Stack of transformer blocks
    3. Layer normalization
    4. Linear projection to vocabulary
    """

    def __init__(self, vocab_size):
        """
        Initialize GPT model.

        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
        """
        super().__init__()

        # Token embedding: convert token IDs to dense vectors
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)

        # Position embedding: encode position information
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)]
        )

        # Final layer normalization
        self.ln_f = nn.LayerNorm(N_EMBD)

        # Language modeling head: project to vocabulary size
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize model weights for better training.

        Uses normal distribution with small standard deviation.

        Args:
            module: Neural network module to initialize
        """
        if isinstance(module, nn.Linear):
            # Initialize linear layer weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initialize embedding weights
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass through the model.

        Args:
            idx: Input token indices of shape (B, T)
            targets: Target token indices of shape (B, T) for loss calculation

        Returns:
            Tuple of (logits, loss):
                logits: Prediction scores of shape (B, T, vocab_size)
                loss: Cross-entropy loss (if targets provided, else None)
        """
        B, T = idx.shape

        # Get token embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, N_EMBD)

        # Get position embeddings
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=DEVICE)
        )  # (T, N_EMBD)

        # Combine token and position embeddings
        x = tok_emb + pos_emb  # (B, T, N_EMBD)

        # Pass through transformer blocks
        x = self.blocks(x)  # (B, T, N_EMBD)

        # Apply final layer norm
        x = self.ln_f(x)  # (B, T, N_EMBD)

        # Project to vocabulary size
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate loss if targets are provided
        if targets is None:
            loss = None
        else:
            # Reshape for cross-entropy calculation
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            # Cross-entropy loss
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens given a context.

        Autoregressive generation: predict one token at a time,
        add it to context, and repeat.

        Args:
            idx: Starting context of shape (B, T)
            max_new_tokens: Number of tokens to generate

        Returns:
            Generated sequence of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop context to last BLOCK_SIZE tokens
            idx_cond = idx[:, -BLOCK_SIZE:]

            # Get predictions
            logits, _ = self(idx_cond)

            # Focus on last time step
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx
