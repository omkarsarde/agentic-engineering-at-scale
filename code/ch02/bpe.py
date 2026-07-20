"""Deterministic byte-pair encoding with a byte fallback vocabulary."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class Merge:
    left: int
    right: int
    token: int


class BytePairTokenizer:
    """A minimal frozen BPE tokenizer whose base vocabulary is all 256 bytes."""

    def __init__(self, merges: list[Merge] | None = None) -> None:
        self.merges = list(merges or [])
        self.token_bytes = {index: bytes([index]) for index in range(256)}
        for merge in self.merges:
            self.token_bytes[merge.token] = self.token_bytes[merge.left] + self.token_bytes[merge.right]

    @property
    def vocab_size(self) -> int:
        """Return the fixed number of token IDs recognized by this tokenizer."""

        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BytePairTokenizer":
        """Learn deterministic adjacent-pair merges from UTF-8 text.

        Args:
            text: Training text used only to choose merge order.
            vocab_size: Desired vocabulary size, at least 256.

        Returns:
            A frozen tokenizer containing up to ``vocab_size - 256`` merges.
        """

        if vocab_size < 256:
            raise ValueError("vocab_size must retain all 256 byte tokens")
        ids = list(text.encode("utf-8"))
        tokenizer = cls()
        for token in range(256, vocab_size):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda item: (-counts[item], item))
            ids = _replace_pair(ids, pair, token)
            tokenizer.merges.append(Merge(*pair, token))
            tokenizer.token_bytes[token] = tokenizer.token_bytes[pair[0]] + tokenizer.token_bytes[pair[1]]
        return tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode any Unicode string without an unknown-token case.

        Args:
            text: Unicode text to encode as UTF-8 bytes and learned merges.

        Returns:
            Token IDs from the frozen vocabulary.
        """

        ids = list(text.encode("utf-8"))
        for merge in self.merges:
            ids = _replace_pair(ids, (merge.left, merge.right), merge.token)
        return ids

    def decode(self, ids: list[int], errors: str = "strict") -> str:
        """Decode IDs produced by this tokenizer back to the exact input text.

        Args:
            ids: Token IDs from ``encode``.
            errors: UTF-8 error policy; keep ``strict`` for round-trip checks.

        Returns:
            The reconstructed Unicode string.

        Raises:
            KeyError: If an ID is outside the frozen vocabulary.
            UnicodeDecodeError: If arbitrary IDs do not form valid UTF-8.
        """

        return b"".join(self.token_bytes[index] for index in ids).decode("utf-8", errors=errors)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable tokenizer description."""

        return {"vocab_size": self.vocab_size, "merges": [merge.__dict__ for merge in self.merges]}


def _replace_pair(ids: list[int], pair: tuple[int, int], token: int) -> list[int]:
    """Replace non-overlapping occurrences of one adjacent pair."""

    merged: list[int] = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
            merged.append(token)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged
