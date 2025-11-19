# model_rope.py
"""
GPT-X model with RoPE (Rotary Positional Embeddings)
Optimized to store the RoPE cache and causal mask once (shared) to
avoid per-head/per-layer duplication that causes GPU memory blowups.

Changes from original:
- Single shared RoPE cache registered as buffers on the model
- Single shared causal `tril` buffer inside MultiHeadAttention
- Heads accept `cos, sin, tril` as inputs instead of storing their own cache
- Blocks are kept as nn.ModuleList so we can forward extra args through

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_EMBD, N_HEAD, N_LAYER, DROPOUT, BLOCK_SIZE


# ============================================================
# RoPE helpers
# ============================================================

def build_rope_cache(seq_len, dim, device):
    """Precompute cos/sin rotation matrices.
    Returns tensors of shape (seq_len, dim).
    """
    assert dim % 2 == 0, "Head dimension must be even for RoPE."

    half_dim = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device).float() / half_dim))

    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (seq_len, half_dim)

    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)  # (seq_len, dim)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)  # (seq_len, dim)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    Apply RoPE to x.
    x: (B, T, head_dim)
    cos/sin: (T, head_dim)
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    # rotation trick
    rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

    # cos/sin expected to be (T, head_dim) -> make (1,T,dim)
    cos = cos.unsqueeze(0)
    sin = sin.unsqueeze(0)

    return x * cos + rot * sin


# ============================================================
# Attention Head WITH shared RoPE
# ============================================================

class Head(nn.Module):
    """Single self-attention head. RoPE & tril are passed in to avoid
    per-head buffers.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
        self.head_size = head_size

    def forward(self, x, cos, sin, tril):
        # x: (B, T, C)
        B, T, C = x.shape

        k = self.key(x)   # (B, T, head_size)
        q = self.query(x)
        v = self.value(x)

        # cos/sin expected shape: (T, head_dim)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        # Scaled dot-product attention (causal by tril)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v  # (B, T, head_size)
        return out


# ============================================================
# Multi-head, FFN, Block — optimized
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

        # single shared causal mask buffer (BLOCK_SIZE x BLOCK_SIZE)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x, cos, sin):
        # cos/sin: (BLOCK_SIZE, head_dim)
        # pass tril to every head (shared)
        head_outs = [h(x, cos, sin, self.tril) for h in self.heads]
        out = torch.cat(head_outs, dim=-1)
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

    def forward(self, x, cos, sin):
        x = x + self.sa(self.ln1(x), cos, sin)
        x = x + self.ffwd(self.ln2(x))
        return x


# ============================================================
# GPT model — now with shared RoPE cache
# ============================================================

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)

        # Blocks as ModuleList so we can pass rope args through
        self.blocks = nn.ModuleList([Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])

        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

        # Build shared RoPE cache (on cpu). When model.to(device) is called,
        # buffers move to the correct device automatically.
        head_size = N_EMBD // N_HEAD
        cos, sin = build_rope_cache(BLOCK_SIZE, head_size, device=torch.device('cpu'))
        # register as buffers so they travel with the model
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

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
        x = tok_emb

        # cos/sin are buffers -> already on model.device after .to(device)
        cos = self.rope_cos
        sin = self.rope_sin

        # pass shared RoPE cache through each block
        for b in self.blocks:
            x = b(x, cos, sin)

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
