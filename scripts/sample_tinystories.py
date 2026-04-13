from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.train_tinystories import TinyGPT


def sample_text(
    model: TinyGPT,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> str:
    model.eval()
    prompt_bytes = prompt.encode("utf-8", errors="ignore")
    if len(prompt_bytes) == 0:
        prompt_bytes = b"Once upon a time"
    idx = torch.tensor([list(prompt_bytes)], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = idx[:, -model.context_length :]
            logits = model(x)[:, -1, :]
            logits = logits / max(temperature, 1e-6)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

    out_bytes = bytes(idx[0].tolist())
    return out_bytes.decode("utf-8", errors="ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from TinyStories checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = list(Path("checkpoints/tinystories").glob("tinystories_step*.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint found under checkpoints/tinystories.")
        ckpt_path = max(candidates, key=lambda p: int(p.stem.split("step")[-1]))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = TinyGPT(
        vocab_size=256,
        context_length=config["context_length"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])

    text = sample_text(
        model=model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    print(f"[checkpoint] {ckpt_path}")
    print(text)


if __name__ == "__main__":
    main()
