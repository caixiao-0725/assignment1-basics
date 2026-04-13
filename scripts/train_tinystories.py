from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cs336_basics.adamw import AdamW


@dataclass
class TrainConfig:
    data_path: Path
    max_bytes: int
    out_dir: Path
    steps: int
    batch_size: int
    context_length: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    lr: float
    weight_decay: float
    grad_clip: float
    log_every: int
    save_every: int
    device: str


def load_byte_tokens(path: Path, max_bytes: int) -> np.ndarray:
    raw = path.read_bytes()
    if max_bytes > 0:
        raw = raw[:max_bytes]
    if len(raw) < 2:
        raise ValueError("Data is too short; need at least 2 bytes.")
    return np.frombuffer(raw, dtype=np.uint8)


def get_batch(tokens: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= context_length:
        raise ValueError("Token length must be larger than context_length.")
    max_start = len(tokens) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    offsets = np.arange(context_length, dtype=np.int64)
    x = tokens[starts[:, None] + offsets]
    y = tokens[starts[:, None] + offsets + 1]
    x_t = torch.tensor(x, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return x_t, y_t


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq, d_model)
        return self.out(attn)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, n_heads: int, n_layers: int, d_ff: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq = x.shape
        if seq > self.context_length:
            raise ValueError(f"Input sequence length {seq} exceeds context_length {self.context_length}.")
        pos = torch.arange(seq, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.lm_head(h)


def train(cfg: TrainConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    tokens = load_byte_tokens(cfg.data_path, cfg.max_bytes)
    model = TinyGPT(
        vocab_size=256,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
    ).to(cfg.device)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    model.train()
    for step in range(1, cfg.steps + 1):
        x, y = get_batch(tokens, cfg.batch_size, cfg.context_length, cfg.device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_every == 0 or step == 1:
            ppl = math.exp(loss.item())
            print(f"step={step:5d} loss={loss.item():.4f} ppl={ppl:.2f}")

        if cfg.save_every > 0 and (step % cfg.save_every == 0 or step == cfg.steps):
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg.__dict__,
            }
            ckpt_path = cfg.out_dir / f"tinystories_step{step}.pt"
            torch.save(ckpt, ckpt_path)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a tiny byte-level GPT on TinyStories.")
    parser.add_argument("--data-path", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/tinystories"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        max_bytes=args.max_bytes,
        out_dir=args.out_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        context_length=args.context_length,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        save_every=args.save_every,
        device=args.device,
    )


if __name__ == "__main__":
    config = parse_args()
    print(f"device={config.device} data={config.data_path} steps={config.steps} max_bytes={config.max_bytes}")
    train(config)
