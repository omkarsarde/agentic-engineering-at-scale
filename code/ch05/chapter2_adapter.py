"""Import the tokenizer and decoder from Chapter 2 without duplicating them."""

import sys
from pathlib import Path


CH02 = Path(__file__).resolve().parent.parent / "ch02"
if str(CH02) not in sys.path:
    sys.path.insert(0, str(CH02))

from bpe import BytePairTokenizer  # noqa: E402
from tinygpt import GPTConfig, TinyGPT  # noqa: E402


__all__ = ["BytePairTokenizer", "GPTConfig", "TinyGPT"]
