"""Synthetic NER data generation."""
import logging
from typing import Dict, List, Tuple

import numpy as np

from src.config import RANDOM_SEED, NER_TAGS

logger = logging.getLogger(__name__)

TEMPLATES = {
    "PER": [("John Smith", "B-PER", "I-PER"), ("Sarah Johnson", "B-PER", "I-PER"),
            ("Michael Brown", "B-PER", "I-PER"), ("Emily Davis", "B-PER", "I-PER"),
            ("Dr. Robert Lee", "B-PER", "I-PER"), ("Maria Garcia", "B-PER", "I-PER")],
    "ORG": [("Google", "B-ORG"), ("Microsoft", "B-ORG"), ("Apple Inc", "B-ORG"),
            ("OpenAI", "B-ORG"), ("Stanford University", "B-ORG"),
            ("United Nations", "B-ORG"), ("Amazon Web Services", "B-ORG")],
    "LOC": [("New York", "B-LOC"), ("San Francisco", "B-LOC"), ("London", "B-LOC"),
            ("Tokyo", "B-LOC"), ("Silicon Valley", "B-LOC"),
            ("Paris", "B-LOC"), ("Berlin", "B-LOC")],
    "DATE": [("January 2024", "B-DATE"), ("Monday", "B-DATE"), ("2024-03-15", "B-DATE"),
             ("last week", "B-DATE"), ("yesterday", "B-DATE"), ("March 19th", "B-DATE")],
}

FILLER_PHRASES = [
    "visited the office of", "announced at", "traveled to", "met with",
    "gave a talk at", "moved to", "was founded in", "headquartered in",
    "joined", "left", "presented at", "graduated from", "works at",
    "is located in", "was established on", "spoke at",
]


def generate_ner_data(
    n_samples: int = 500,
    seed: int = RANDOM_SEED,
) -> List[Tuple[List[str], List[str]]]:
    """Generate synthetic NER training data.

    Args:
        n_samples: Number of samples.
        seed: Random seed.

    Returns:
        List of (tokens, tags) tuples.
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples):
        n_entities = rng.integers(1, 4)
        tokens, tags = [], []

        entity_types = list(TEMPLATES.keys())
        chosen = rng.choice(entity_types, size=n_entities, replace=False)

        for etype in chosen:
            template = rng.choice(TEMPLATES[etype])
            filler = rng.choice(FILLER_PHRASES)

            # Add filler before entity
            filler_toks = filler.split()
            tokens.extend(filler_toks)
            tags.extend(["O"] * len(filler_toks))

            # Add entity
            entity_toks = template[0].split()
            entity_tags = list(template[1:])
            # Pad or truncate tags to match tokens
            while len(entity_tags) < len(entity_toks):
                entity_tags.append("O")
            tokens.extend(entity_toks)
            tags.extend(entity_tags[:len(entity_toks)])

        # Ensure at least one entity
        if not any(t != "O" for t in tags):
            template = rng.choice(TEMPLATES["PER"])
            tokens = template[0].split()
            tags = list(template[1:])
            while len(tags) < len(tokens):
                tags.append("O")

        samples.append((tokens, tags[:len(tokens)]))

    logger.info("Generated %d NER samples", len(samples))
    return samples


def tokens_to_text(tokens: List[str]) -> str:
    """Join tokens into text."""
    return " ".join(tokens)


def bio_to_entities(tokens: List[str], tags: List[str]) -> List[Dict]:
    """Convert BIO tags to entity list.

    Args:
        tokens: Token list.
        tags: Tag list.

    Returns:
        List of entity dicts with 'text', 'type', 'start', 'end'.
    """
    entities = []
    current = None
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current:
                current["end"] = i
                entities.append(current)
            current = {"text": tok, "type": tag[2:], "start": i, "end": i + 1}
        elif tag.startswith("I-") and current and tag[2:] == current["type"]:
            current["text"] += " " + tok
            current["end"] = i + 1
        else:
            if current:
                current["end"] = i
                entities.append(current)
                current = None
    if current:
        current["end"] = len(tokens)
        entities.append(current)
    return entities


def get_data_stats(samples: List[Tuple[List[str], List[str]]]) -> Dict:
    """Compute dataset statistics."""
    all_tags = [t for _, tags in samples for t in tags]
    entity_tags = [t for t in all_tags if t != "O"]
    return {
        "n_samples": len(samples),
        "total_tokens": sum(len(toks) for toks, _ in samples),
        "total_entities": len(entity_tags),
        "entity_types": sorted(set(t.split("-")[1] for t in entity_tags if "-" in t)),
        "avg_tokens": round(np.mean([len(toks) for toks, _ in samples]), 1) if samples else 0,
    }
