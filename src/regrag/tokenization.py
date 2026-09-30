"""Token counting. tiktoken when available, a deterministic whitespace fallback otherwise."""

from __future__ import annotations

from functools import lru_cache


class Tokenizer:
    def __init__(self, name: str = "cl100k_base"):
        self.name = name
        self._enc = _load(name)

    def encode(self, text: str) -> list[int] | list[str]:
        if self._enc is not None:
            return self._enc.encode(text, disallowed_special=())
        return text.split()

    def decode(self, tokens) -> str:
        if self._enc is not None:
            return self._enc.decode(tokens)
        return " ".join(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


@lru_cache(maxsize=4)
def _load(name: str):
    try:
        import tiktoken

        return tiktoken.get_encoding(name)
    except Exception:  # offline CI without the BPE file
        return None


@lru_cache(maxsize=4)
def get_tokenizer(name: str = "cl100k_base") -> Tokenizer:
    return Tokenizer(name)
