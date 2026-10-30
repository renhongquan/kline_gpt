import math
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import sys


# ====================== 【1】RoPE + Attention + Block + Model 定义（原样复用） ======================
class RoPE(nn.Module):
    def __init__(self, dim: int, max_kline_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_kline_len = max_kline_len
        assert dim % 2 == 0
        theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("theta", theta, persistent=False)

        pos = torch.arange(0, max_kline_len, dtype=torch.float32)
        freqs = torch.outer(pos, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        B, H, T, D = x.shape
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rot = x_complex * freqs_cis[None, None, :, :]
        x_out = torch.view_as_real(x_rot).flatten(-2)
        return x_out.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, rope: RoPE, max_seq: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.rope = rope
        mask = torch.tril(torch.ones(max_seq, max_seq, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x).split(self.n_embd, dim=2)
        q, k, v = [t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in qkv]
        q = self.rope(q)
        k = self.rope(k)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        #mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        # 不再动态生成，直接切片取 [T,T]
        mask = self.causal_mask[:T, :T]
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):


    def __init__(self, n_embd: int, n_head: int, rope: RoPE,max_seq:int,dropout_rate=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, rope,max_seq)
        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)
        self.mlp = MLP(n_embd)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class KlineGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_feat: int = 4,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 256,
        max_kline_seq: int = 512,
        use_learnable_fuse: bool = True,   # 新增开关
        dropout_rate=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_feat = n_feat
        self.n_embd = n_embd
        self.max_kline_seq = max_kline_seq
        self.use_learnable_fuse = use_learnable_fuse
        self.rope = RoPE(dim=n_embd // n_head, max_kline_len=max_kline_seq)
        self.wte = nn.Embedding(vocab_size, n_embd)
        # 只有开启才创建kline_fuse层
        if self.use_learnable_fuse:
            self.kline_fuse = nn.Sequential(
                nn.Linear(n_feat * n_embd, n_embd, bias=False),            
            )

        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, self.rope,max_kline_seq,dropout_rate) for _ in range(n_layer)
        ])
        self.ln_f = nn.RMSNorm(n_embd)

        self.heads = nn.ModuleList([
            nn.Linear(n_embd, vocab_size, bias=False) for _ in range(n_feat)
        ])
        

    def forward(self, kline_tokens: torch.Tensor, targets: torch.Tensor = None):
        B, T, C_feat = kline_tokens.shape
        emb = self.wte(kline_tokens)
        
        if self.use_learnable_fuse:
            emb_flat = emb.reshape(B, T, -1)
            x = self.kline_fuse(emb_flat)
        else:
            x = emb.mean(dim=2)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits_list = [h(x) for h in self.heads]
        loss = None
        if targets is not None:
            loss = 0.0
            for i in range(self.n_feat):
                logit = logits_list[i]
                y = targets[..., i]
                loss += F.cross_entropy(
                    logit.reshape(-1, self.vocab_size),
                    y.reshape(-1)
                )
            loss = loss / self.n_feat
        return logits_list, loss

    @torch.no_grad()
    def generate_next_kline(self, ctx_tokens: torch.Tensor, temperature=1.0):
        B, T, _ = ctx_tokens.shape
        temperature = max(temperature, 1e-6)
        ctx_cond = ctx_tokens[:, -self.max_kline_seq:, :]
        logits_list, _ = self(ctx_cond)
        last_logits = [l[:, -1, :] for l in logits_list]

        next_tokens = []
        for logit in last_logits:
            logit = logit / temperature
            prob = F.softmax(logit, dim=-1)
            tid = torch.multinomial(prob, num_samples=1)
            next_tokens.append(tid)
        next_kline = torch.cat(next_tokens, dim=-1).unsqueeze(1)
        return next_kline

