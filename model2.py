"""
GPT-X Model 2 (Final):
- RoPE (Rotary Position Embedding)
- RMSNorm
- Dropout (Training Stability)
- FlashAttention fallback
- Stable initialization for fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import N_EMBD, N_HEAD, N_LAYER, DROPOUT, BLOCK_SIZE


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x * (self.weight / norm)


# ============================================================
# Rotary Positional Embeddings (RoPE)
# ============================================================

def build_rope_cache(seq_len, dim, device):
    assert dim % 2 == 0
    half = dim // 2

    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)
    return cos, sin


def apply_rope(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return x * cos + x_rot * sin


# ============================================================
# Multi-Head Attention with RoPE
# ============================================================

class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(DROPOUT)

        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        # RoPE cache
        self.rope_cos = None
        self.rope_sin = None

    def _prepare_rope(self, device):
        if self.rope_cos is None or self.rope_cos.device != device:
            self.rope_cos, self.rope_sin = build_rope_cache(
                BLOCK_SIZE, self.head_dim, device
            )

    def forward(self, x):
        B, T, C = x.shape
        device = x.device

        self._prepare_rope(device)

        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ============================================================
        # FlashAttention fallback (more stability)
        # ============================================================
        try:
            from torch.nn.functional import scaled_dot_product_attention
            out = scaled_dot_product_attention(q, k, v, dropout_p=DROPOUT if self.training else 0)
        except:
            att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            out = torch.matmul(att, v)

        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)


# ============================================================
# Feed-Forward Network
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
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
        self.norm1 = RMSNorm(n_embd)
        self.attn = MultiHeadAttentionRoPE(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.ff = FeedForward(n_embd)

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
        self.drop = nn.Dropout(DROPOUT)  # ⭐ NEW: embedding dropout

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

        x = self.token_emb(idx)
        x = self.drop(x)  # ⭐ NEW: dropout

        x = self.blocks(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        return logits, loss

    # Stable text generation
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k:
                v, ix = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, ix, v)

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)

            idx = torch.cat((idx, next_tok), dim=1)

        return idx
