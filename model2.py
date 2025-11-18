# model_rope_rms.py
"""
GPT-X model using:
- RoPE (Rotary Positional Embeddings)
- RMSNorm (instead of LayerNorm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_EMBD, N_HEAD, N_LAYER, DROPOUT, BLOCK_SIZE

# RMSNorm 

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, T, C)
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x * (self.weight / norm)


# ============================================================
# RoPE helpers
# ============================================================

def build_rope_cache(seq_len, dim, device):
    """
    Build cos/sin rotation cache for RoPE.
    dim must be even (each pair forms a rotation).
    """
    assert dim % 2 == 0, "Head dimension must be even for RoPE."

    half_dim = dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 1, device=device).float() / half_dim))

    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, half_dim)

    cos = freqs.cos()
    sin = freqs.sin()

    # Expand to full dim by interleaving (x0, x1) → (cos, sin) pairs
    cos = torch.repeat_interleave(cos, 2, dim=-1)  # (T, dim)
    sin = torch.repeat_interleave(sin, 2, dim=-1)  # (T, dim)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    x: (B, n_head, T, head_dim)
    cos/sin: (T, head_dim)
    """

    # Rotate half trick: (x_even, x_odd) -> (-x_odd, x_even)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,H)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return x * cos + x_rot * sin


# Multi-Head Self-Attention with RoPE

class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        # single fused linear for q/k/v
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)

        # dropout
        self.dropout = nn.Dropout(DROPOUT)

        # causal mask
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        # rope cache (built later on proper device)
        self.rope_cos = None
        self.rope_sin = None

    def _prepare_rope(self, device):
        """Build RoPE cache if not existing or wrong device."""
        if self.rope_cos is None or self.rope_cos.device != device:
            cos, sin = build_rope_cache(BLOCK_SIZE, self.head_dim, device)
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, x):
        B, T, C = x.shape
        device = x.device

        self._prepare_rope(device)

        # qkv projection
        qkv = self.qkv(x)  # (B, T, 3C)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_head, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # apply rotary embeddings
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # attention scores
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # causal mask
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        # softmax
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # attention output
        out = torch.matmul(att, v)  # (B, n_head, T, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)

        out = self.out_proj(out)
        return out


# ============================================================
# Feed-Forward Network (same as original)
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Transformer Block
# ============================================================

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = MultiHeadAttentionRoPE(n_embd, n_head)
        self.norm1 = RMSNorm(n_embd)
        self.ff = FeedForward(n_embd)
        self.norm2 = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ============================================================
# GPT Language Model
# ============================================================

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, N_EMBD)

        # NO position embeddings → RoPE replaces them
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)]
        )

        self.norm_f = RMSNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)  # (B, T, C)

        x = self.blocks(tok)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, ix = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, ix, v)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx