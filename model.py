# model_rope.py
"""
GPT-X model with RoPE (Rotary Positional Embeddings)
Architecture stays identical to your original model:
- LayerNorm
- ReLU feed-forward
- Original multi-head structure

Only positional embeddings are replaced by RoPE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_EMBD, N_HEAD, N_LAYER, DROPOUT, BLOCK_SIZE


# ============================================================
# RoPE helpers
# ============================================================

def build_rope_cache(seq_len, dim, device):
    """Precompute cos/sin rotation matrices"""
    assert dim % 2 == 0, "Head dimension must be even for RoPE."

    half_dim = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))

    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    x: (B, T, head_dim)
    cos/sin: (T, head_dim)
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    # rotation trick
    rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

    cos = cos.unsqueeze(0)  # (1, T, dim)
    sin = sin.unsqueeze(0)

    return x * cos + rot * sin


# ============================================================
# Attention Head WITH RoPE
# ============================================================

class Head(nn.Module):
    """Single self-attention head with RoPE"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

        # RoPE cache
        self.rope_cos = None
        self.rope_sin = None
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        device = x.device

        # Build rope cache lazily
        if self.rope_cos is None or self.rope_cos.device != device:
            cos, sin = build_rope_cache(BLOCK_SIZE, self.head_size, device)
            self.rope_cos = cos
            self.rope_sin = sin

        # q, k, v
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x)
        v = self.value(x)

        # Apply RoPE only to q, k
        cos = self.rope_cos[:T]  # (T, head_dim)
        sin = self.rope_sin[:T]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Self-attention
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


# ============================================================
# Multi-head, FFN, Block — unchanged
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ============================================================
# GPT model — only change: NO position embedding table
# ============================================================

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)

        # Removed: self.position_embedding_table

        self.blocks = nn.Sequential(*[
            Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)
        ])

        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        x = tok_emb                                 # RoPE added inside heads

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, ix, v)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx
