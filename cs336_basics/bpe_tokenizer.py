from __future__ import annotations

import regex
from collections.abc import Iterable

from cs336_basics.bpe_function import GPT2_PRETOKEN_PATTERN


class BPETokenizer:
    """GPT-2 风格字节 BPE：预分词 + 按 merge 优先级反复合并相邻符号。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self._vocab = dict(vocab)
        self._merge_ranks = {pair: idx for idx, pair in enumerate(merges)}
        self._bytes_to_id = {b: i for i, b in self._vocab.items()}
        self._special_tokens = list(special_tokens) if special_tokens else []

    def encode(self, text: str) -> list[int]:
        return list(self._encode_yield(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            yield from self._encode_yield(chunk)

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        parts = [self._vocab[i] for i in ids]
        # 与 tiktoken 一致：单 token 用 replace，避免 BPE 在 UTF-8 边界截断的字节单独 strict 解码失败
        if len(parts) == 1:
            return parts[0].decode("utf-8", errors="replace")
        return b"".join(parts).decode("utf-8")

    def _encode_yield(self, text: str) -> Iterable[int]:
        if not text:
            return
        yield from self._encode_with_specials(text)

    def _encode_with_specials(self, text: str) -> Iterable[int]:
        if not self._special_tokens:
            yield from self._encode_plain(text)
            return
        pattern = "|".join(
            regex.escape(t) for t in sorted(set(self._special_tokens), key=len, reverse=True)
        )
        rx = regex.compile(pattern)
        pos = 0
        for m in rx.finditer(text):
            if m.start() > pos:
                yield from self._encode_plain(text[pos : m.start()])
            st = m.group(0)
            yield self._bytes_to_id[st.encode("utf-8")]
            pos = m.end()
        if pos < len(text):
            yield from self._encode_plain(text[pos:])

    def _encode_plain(self, text: str) -> Iterable[int]:
        for pretoken in GPT2_PRETOKEN_PATTERN.findall(text):
            yield from self._bpe_encode_pretoken(pretoken)

    def _bpe_encode_pretoken(self, pretoken: str) -> Iterable[int]:
        word = [bytes([b]) for b in pretoken.encode("utf-8")]
        if not word:
            return
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: (self._merge_ranks.get(p, 10**18), p))
            if best not in self._merge_ranks:
                break
            first, second = best
            new_word: list[bytes] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        for w in word:
            yield self._bytes_to_id[w]


def build_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def demo_tokenizer_walkthrough() -> None:
    """
    小教程：展示 tokenizer 的「special token 切分 + 预分词 + BPE 编码 + decode」全过程。
    在仓库根目录执行：uv run python -m cs336_basics.bpe_tokenizer
    """
    print("=== 1) 构造一个极小 tokenizer（手工 vocab + merges）===")
    vocab = {
        0: b"<|endoftext|>",
        1: b" ",
        2: b"h",
        3: b"e",
        4: b"l",
        5: b"o",
        6: b"w",
        7: b"r",
        8: b"d",
        9: b"!",
        10: b"he",
        11: b"ll",
        12: b"o",
        13: b"wo",
        14: b"rld",
        15: b"hello",
        16: b" world",
    }
    merges = [
        (b"h", b"e"),       # he
        (b"l", b"l"),       # ll
        (b"he", b"ll"),     # hell
        (b"hell", b"o"),    # hello
        (b"w", b"o"),       # wo
        (b"r", b"l"),       # rl
        (b"rl", b"d"),      # rld
        (b" ", b"w"),       #  w
        (b" w", b"o"),      #  wo
        (b" wo", b"rld"),   #  world
    ]
    tokenizer = BPETokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
    print(f"vocab 大小: {len(vocab)}, merges 条数: {len(merges)}")

    text = "hello<|endoftext|> world!"
    print("\n=== 2) 输入文本 ===")
    print(repr(text))

    print("\n=== 3) special token 切分（_encode_with_specials 在做的事）===")
    pattern = "|".join(regex.escape(t) for t in sorted(set(tokenizer._special_tokens), key=len, reverse=True))
    rx = regex.compile(pattern)
    pos = 0
    for m in rx.finditer(text):
        if m.start() > pos:
            print(f"普通片段: {text[pos:m.start()]!r}")
        print(f"special: {m.group(0)!r}")
        pos = m.end()
    if pos < len(text):
        print(f"普通片段: {text[pos:]!r}")

    plain_text = "hello world!"
    print("\n=== 4) 对普通片段做 GPT-2 预分词（_encode_plain）===")
    pretokens = GPT2_PRETOKEN_PATTERN.findall(plain_text)
    for i, p in enumerate(pretokens):
        print(f"[{i}] {p!r} -> bytes {list(p.encode('utf-8'))}")

    print("\n=== 5) 看一个 pretoken 如何被 BPE 合并（_bpe_encode_pretoken）===")
    probe = "hello"
    word = [bytes([b]) for b in probe.encode("utf-8")]
    print("初始符号:", word)
    merge_ranks = tokenizer._merge_ranks
    step = 1
    while len(word) > 1:
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        best = min(pairs, key=lambda p: (merge_ranks.get(p, 10**18), p))
        if best not in merge_ranks:
            break
        new_word: list[bytes] = []
        i = 0
        while i < len(word):
            if i + 1 < len(word) and word[i] == best[0] and word[i + 1] == best[1]:
                new_word.append(best[0] + best[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = new_word
        print(f"step {step}: merge {best} -> {word}")
        step += 1

    print("\n=== 6) 真正 encode/decode ===")
    ids = tokenizer.encode(text)
    pieces = [tokenizer.decode([i]) for i in ids]
    print("token ids:", ids)
    print("逐 token 解码:", pieces)
    print("整体 decode:", tokenizer.decode(ids))

    print("\n=== 7) encode_iterable（流式）===")
    chunks = ["hello<|endoftext|>", " world!"]
    stream_ids = list(tokenizer.encode_iterable(chunks))
    print("输入 chunks:", chunks)
    print("stream ids :", stream_ids)
    print("一致性检查:", stream_ids == ids)


if __name__ == "__main__":
    demo_tokenizer_walkthrough()

