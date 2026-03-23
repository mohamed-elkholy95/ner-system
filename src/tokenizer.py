"""Tokenization and feature extraction for NER."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from collections import Counter

from src.config import NER_TAGS, RANDOM_SEED

logger = logging.getLogger(__name__)


class NERTokenizer:
    """Simple whitespace tokenizer with vocabulary building."""

    def __init__(self, min_freq: int = 1, special_tokens: Optional[List[str]] = None) -> None:
        self.min_freq = min_freq
        self._vocab: Dict[str, int] = {}
        self._inv_vocab: Dict[int, str] = {}
        self._special = special_tokens or ["<PAD>", "<UNK>"]
        self._is_fitted = False

    def fit(self, texts: List[str]) -> "NERTokenizer":
        """Build vocabulary from texts."""
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())

        idx = 0
        for tok in self._special:
            self._vocab[tok] = idx
            idx += 1

        for word, freq in counter.most_common():
            if freq >= self.min_freq:
                self._vocab[word] = idx
                idx += 1

        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        self._is_fitted = True
        logger.info("Vocabulary: %d tokens (min_freq=%d)", len(self._vocab), self.min_freq)
        return self

    def encode(self, tokens: List[str]) -> List[int]:
        """Encode tokens to IDs."""
        if not self._is_fitted:
            raise RuntimeError("Tokenizer not fitted")
        unk_id = self._vocab.get("<UNK>", 0)
        return [self._vocab.get(t.lower(), unk_id) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        """Decode IDs to tokens."""
        return [self._inv_vocab.get(i, "<UNK>") for i in ids]

    @property
    def pad_id(self) -> int:
        return self._vocab.get("<PAD>", 0)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


class TagEncoder:
    """Encode NER tags to integers."""

    def __init__(self, tags: Optional[List[str]] = None) -> None:
        self._tags = tags or NER_TAGS
        self._tag2id = {t: i for i, t in enumerate(self._tags)}
        self._id2tag = {i: t for t, i in self._tag2id.items()}

    def encode(self, tags: List[str]) -> List[int]:
        return [self._tag2id.get(t, 0) for t in tags]

    def decode(self, ids: List[int]) -> List[str]:
        return [self._id2tag.get(i, "O") for i in ids]

    @property
    def num_tags(self) -> int:
        return len(self._tags)

    @property
    def pad_id(self) -> int:
        return self._tag2id.get("O", 0)


def word_shape(token: str) -> str:
    """Map a token to its word shape pattern.

    Word shape collapses characters into categories:
      - uppercase letter → 'X'
      - lowercase letter → 'x'
      - digit → 'd'
      - other → itself

    Consecutive identical categories are collapsed to reduce length.

    Examples:
        "John"     → "Xxxx"
        "McDonald" → "XxXxxxx"
        "2024-01"  → "dddd-dd"
        "U.S.A."   → "X.X.X."

    This is a classic NER feature — the shape of a word reveals a lot
    about its entity type without knowing the specific word.
    """
    shape_chars = []
    for ch in token:
        if ch.isupper():
            shape_chars.append("X")
        elif ch.islower():
            shape_chars.append("x")
        elif ch.isdigit():
            shape_chars.append("d")
        else:
            shape_chars.append(ch)
    return "".join(shape_chars)


def extract_features(tokens: List[str], window: int = 2) -> np.ndarray:
    """Extract word-level features for each token.

    Features per token (10 total):
      0: token length
      1: starts with uppercase (title case → likely proper noun)
      2: all uppercase (acronyms like 'NASA', 'IBM')
      3: all digits (years, IDs, quantities)
      4: suffix length (morphological signal)
      5: ends with uppercase
      6: contains hyphen (hyphenated names, compound words)
      7: contains digit (mixed alphanumeric like 'COVID-19')
      8: all alphabetic (vs punctuation/mixed)
      9: ratio of uppercase chars (higher → more likely entity)

    Args:
        tokens: Token list.
        window: Context window size (reserved for future use).

    Returns:
        Feature array of shape (n_tokens, 10).
    """
    features = []
    for i, tok in enumerate(tokens):
        if not tok:
            features.append([0.0] * 10)
            continue
        upper_ratio = sum(1 for c in tok if c.isupper()) / len(tok) if tok else 0.0
        f = [
            float(len(tok)),
            float(tok[0].isupper()),
            float(tok.isupper()),
            float(tok.isdigit()),
            float(len(tok[-3:]) if len(tok) >= 3 else len(tok)),
            float(tok[-1].isupper()),
            float("-" in tok),
            float(any(c.isdigit() for c in tok)),
            float(tok.isalpha()),
            round(upper_ratio, 4),
        ]
        features.append(f)
    if not features:
        return np.empty((0, 10))
    return np.array(features, dtype=np.float64)
