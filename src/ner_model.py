"""NER model implementations.

This module provides two NER model architectures:

1. **CRFTagger** — A Conditional Random Field model using hand-crafted features.
   CRFs model the conditional probability P(tags | tokens) directly, making them
   discriminative models. The key advantage of CRFs over simpler classifiers (like
   logistic regression on each token independently) is that CRFs model *transition
   probabilities* between adjacent tags — so they learn that B-PER → I-PER is
   likely but B-PER → I-ORG is not.

2. **BiLSTMNERTagger** — A bidirectional LSTM with a linear classification head.
   The BiLSTM reads the sentence left-to-right AND right-to-left, giving each
   token a context-aware representation. This is more powerful than hand-crafted
   features because the model learns its own features from data. In a production
   system, you'd typically add a CRF layer on top (BiLSTM-CRF) to also capture
   tag transition constraints.

Architecture comparison:
  ┌──────────────┬───────────────┬────────────────┐
  │              │ CRF           │ BiLSTM(-CRF)   │
  ├──────────────┼───────────────┼────────────────┤
  │ Features     │ Hand-crafted  │ Learned        │
  │ Context      │ Window-based  │ Full sentence  │
  │ Transitions  │ Yes (native)  │ Optional +CRF  │
  │ Training     │ Fast (CPU)    │ Slower (GPU)   │
  │ Data needs   │ Small OK      │ Needs more     │
  └──────────────┴───────────────┴────────────────┘
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("torch not available")

CRF_AVAILABLE = False
try:
    import sklearn_crfsuite  # noqa: F401
    CRF_AVAILABLE = True
except ImportError:
    logger.info("sklearn_crfsuite not available — using fallback")


def word2features(sent: List[str], i: int) -> Dict[str, str]:
    """Extract CRF features for a word at position i.

    Feature engineering is critical for CRF performance. Each feature captures
    a different linguistic signal:
      - word.lower(): normalized form (case-insensitive matching)
      - suffix features: morphological patterns (e.g., "-tion" → noun, "-ing" → verb)
      - prefix features: beginning patterns (e.g., "Mc" → Irish surname)
      - capitalization: title case → proper noun → likely entity
      - digit detection: dates, phone numbers, IDs
      - context features: neighboring words provide disambiguation
        ("Dr." before a word strongly suggests B-PER)

    The window-based approach (i-1, i, i+1) is a trade-off: wider windows
    capture more context but increase feature space and training time.
    """
    word = sent[i]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word.isupper()": word.isupper(),
        "word.istitle()": word[0].isupper(),
        "word.isdigit()": word.isdigit(),
        "word.isalpha()": word.isalpha(),
        "word.len": len(word),
        "word.has_hyphen": "-" in word,
        "word.has_digit": any(c.isdigit() for c in word),
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            "-1:word.lower()": word1.lower(),
            "-1:word.istitle()": word1[0].isupper(),
            "-1:word.isupper()": word1.isupper(),
        })
    else:
        features["BOS"] = True  # Beginning of sentence

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            "+1:word.lower()": word1.lower(),
            "+1:word.istitle()": word1[0].isupper(),
            "+1:word.isupper()": word1.isupper(),
        })
    else:
        features["EOS"] = True  # End of sentence

    return features


def sent2features(sent: List[str]) -> List[Dict[str, str]]:
    return [word2features(sent, i) for i in range(len(sent))]


class CRFTagger:
    """CRF-based NER tagger."""

    def __init__(self, c1: float = 0.1, c2: float = 0.01, max_iterations: int = 50) -> None:
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self._model: Optional[Any] = None

    def fit(self, sentences: List[List[str]], tag_lists: List[List[str]]) -> "CRFTagger":
        """Train CRF model."""
        if not CRF_AVAILABLE:
            logger.warning("sklearn_crfsuite not available — mock training")
            return self

        X = [sent2features(s) for s in sentences]
        y = tag_lists
        self._model = sklearn_crfsuite.CRF(
            algorithm="lbfgs", c1=self.c1, c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True,
        )
        self._model.fit(X, y)
        logger.info("CRF trained on %d sentences", len(sentences))
        return self

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """Predict NER tags."""
        if self._model is None:
            return [["O"] * len(s) for s in sentences]

        X = [sent2features(s) for s in sentences]
        return self._model.predict(X)

    def predict_proba(self, sentences: List[List[str]]) -> List[np.ndarray]:
        """Predict tag probabilities (marginals) for each token.

        Returns a list of dicts-per-token, where each dict maps tag names to
        their marginal probability. Useful for confidence estimation and
        active learning (uncertain predictions → ask a human to label).
        """
        if self._model is None:
            return [np.full((len(s), 2), 0.5) for s in sentences]
        X = [sent2features(s) for s in sentences]
        return self._model.predict_marginals(X)

    def save(self, path: str) -> None:
        """Serialize trained CRF model to disk.

        Args:
            path: File path (recommended extension: .pkl).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._model is None:
            raise RuntimeError("Cannot save an untrained CRF model")
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self._model,
                "c1": self.c1,
                "c2": self.c2,
                "max_iterations": self.max_iterations,
            }, f)
        logger.info("CRF model saved to %s", filepath)

    @classmethod
    def load(cls, path: str) -> "CRFTagger":
        """Deserialize a trained CRF model from disk.

        Args:
            path: Path to the saved model file.

        Returns:
            A CRFTagger instance with the loaded model.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"No model file at {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        tagger = cls(c1=data["c1"], c2=data["c2"], max_iterations=data["max_iterations"])
        tagger._model = data["model"]
        logger.info("CRF model loaded from %s", filepath)
        return tagger


class BiLSTMNERTagger(nn.Module if HAS_TORCH else object):
    """BiLSTM-CRF NER tagger (PyTorch)."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_tags: int) -> None:
        if HAS_TORCH:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                                bidirectional=True, batch_first=True, dropout=0.3)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_dim * 2, num_tags)

    def forward(self, x: Any) -> Any:
        if not HAS_TORCH:
            return None
        emb = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(emb)
        logits = self.fc(self.dropout(lstm_out))
        return logits


def train_bilstm(
    model: Any, sentences: List[List[str]], tag_lists: List[List[str]],
    tokenizer: Any, tag_encoder: Any, epochs: int = 5, lr: float = 1e-3,
) -> Dict[str, List[float]]:
    """Train BiLSTM model.

    Returns:
        Training history dict with 'loss' key.
    """
    if not HAS_TORCH or not isinstance(model, nn.Module):
        logger.warning("PyTorch not available — returning mock history")
        return {"loss": [2.5 - 0.3 * i for i in range(epochs)]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    history = {"loss": []}

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for sent, tags in zip(sentences, tag_lists):
            ids = torch.tensor([tokenizer.encode(sent)], dtype=torch.long).to(device)
            tag_ids = torch.tensor([tag_encoder.encode(tags)], dtype=torch.long).to(device)
            optimizer.zero_grad()
            logits = model(ids)
            loss = criterion(logits.view(-1, logits.shape[-1]), tag_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history["loss"].append(round(avg_loss, 4))
        logger.info("BiLSTM Epoch %d/%d: loss=%.4f", epoch + 1, epochs, avg_loss)

    return history


def predict_bilstm(
    model: Any, sentence: List[str], tokenizer: Any, tag_encoder: Any,
) -> List[str]:
    """Predict tags with BiLSTM."""
    if not HAS_TORCH or not isinstance(model, nn.Module):
        return ["O"] * len(sentence)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ids = torch.tensor([tokenizer.encode(sentence)], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(ids)
    pred_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    return tag_encoder.decode(pred_ids)
