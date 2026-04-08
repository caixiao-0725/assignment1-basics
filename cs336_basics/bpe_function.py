from __future__ import annotations

import os
from collections import Counter

import regex

GPT2_PRETOKEN_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _pretokenize_to_counter(text: str) -> Counter[tuple[bytes, ...]]:
    word_freqs: Counter[tuple[bytes, ...]] = Counter()
    for pretoken in GPT2_PRETOKEN_PATTERN.findall(text):
        token_bytes = pretoken.encode("utf-8")
        symbols = tuple(bytes([b]) for b in token_bytes)
        if symbols:
            word_freqs[symbols] += 1
    return word_freqs


def _non_special_segments(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]

    escaped_tokens = [regex.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True)]
    split_pattern = regex.compile("|".join(escaped_tokens))
    return [segment for segment in split_pattern.split(text) if segment]


def _apply_merge_to_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    i = 0
    left, right = pair
    while i < len(word):
        if i + 1 < len(word) and word[i] == left and word[i + 1] == right:
            merged.append(left + right)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    word_freqs: Counter[tuple[bytes, ...]] = Counter()
    for segment in _non_special_segments(text, special_tokens):
        word_freqs.update(_pretokenize_to_counter(segment))

    vocab: dict[int, bytes] = {}
    next_id = 0
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1

    for b in range(256):
        if next_id >= vocab_size:
            break
        vocab[next_id] = bytes([b])
        next_id += 1

    merges: list[tuple[bytes, bytes]] = []
    max_merges = max(0, vocab_size - len(vocab))

    for _ in range(max_merges):
        pair_freqs: Counter[tuple[bytes, bytes]] = Counter()
        for word, count in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += count

        if not pair_freqs:
            break

        best_pair = max(pair_freqs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best_pair)

        new_word_freqs: Counter[tuple[bytes, ...]] = Counter()
        for word, count in word_freqs.items():
            new_word_freqs[_apply_merge_to_word(word, best_pair)] += count
        word_freqs = new_word_freqs

        if next_id < vocab_size:
            vocab[next_id] = best_pair[0] + best_pair[1]
            next_id += 1

    return vocab, merges


def demo_bpe_walkthrough() -> None:
    """
    小教程：用极短语料走一遍「预分词 → 按字节拆开 → 统计相邻对 → 反复合并」。
    在仓库根目录执行：uv run python -m cs336_basics.bpe_function
    """
    import tempfile
    from pathlib import Path

    # 故意重复的词，方便看到 pair 频次
    corpus = "the cat sat on the mat\nthe cat sat again\n"
    print("=== 1) 原始语料 ===")
    print(repr(corpus))

    print("\n=== 2) GPT-2 预分词（正则拆出来的「词片」）===")
    for i, pt in enumerate(GPT2_PRETOKEN_PATTERN.findall(corpus)):
        print(f"  [{i}] {pt!r}  ->  utf-8 字节: {list(pt.encode('utf-8'))}")

    print("\n=== 3) 每条预分词拆成「单字节」符号序列（即 BPE 起点）===")
    wf = _pretokenize_to_counter(corpus)
    for syms, c in sorted(wf.items(), key=lambda x: (-x[1], x[0])):
        readable = "/".join(tok.decode("latin-1", errors="replace") for tok in syms)
        print(f"  频次 {c:3d}  |  符号链 ({len(syms)} 段): {readable!r}")

    print("\n=== 4) 前 5 步合并（每步：当前最高频相邻对 -> 并成新符号）===")
    word_freqs: Counter[tuple[bytes, ...]] = Counter(wf)
    for step in range(5):
        pair_freqs: Counter[tuple[bytes, bytes]] = Counter()
        for word, count in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += count
        if not pair_freqs:
            print("  （已无相邻对，停止）")
            break
        top5 = sorted(pair_freqs.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        print(f"  步骤 {step + 1}  频次 Top5（并列时取 pair 字典序更大的那个）:")
        for (a, b), f in top5:
            print(f"    freq={f:3d}  pair=({a!r}, {b!r})  merged={a+b!r}")
        best = max(pair_freqs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        print(f"  -> 本步选中合并: {best[0]!r} + {best[1]!r} => {best[0] + best[1]!r}")
        new_wf: Counter[tuple[bytes, ...]] = Counter()
        for word, count in word_freqs.items():
            new_wf[_apply_merge_to_word(word, best)] += count
        word_freqs = new_wf

    print("\n=== 5) 调用完整 train_bpe（写临时文件，小词表）===")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        tmp.write(corpus)
        p = Path(tmp.name)
    try:
        vocab, merges = train_bpe(
            input_path=p,
            vocab_size=256 + 8,
            special_tokens=["<|endoftext|>"],
        )
        print(f"  词表大小: {len(vocab)}，合并规则条数: {len(merges)}")
        print("  前几条 merges（bytes 形式）:")
        for i, m in enumerate(merges[:6]):
            print(f"    {i}: {m[0]!r} + {m[1]!r}")
        print("  词表里几条示例 id -> bytes:")
        for tid in sorted(vocab.keys())[:6]:
            print(f"    {tid}: {vocab[tid]!r}")
    finally:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    demo_bpe_walkthrough()
