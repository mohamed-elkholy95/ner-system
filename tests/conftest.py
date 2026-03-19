"""Shared fixtures."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.data_generator import generate_ner_data


@pytest.fixture
def sample_data():
    return generate_ner_data(n_samples=100, seed=42)


@pytest.fixture
def sample_texts(sample_data):
    return [" ".join(toks) for toks, _ in sample_data[:5]]


@pytest.fixture
def tokenizer(sample_texts):
    from src.tokenizer import NERTokenizer
    return NERTokenizer().fit(sample_texts)


@pytest.fixture
def tag_encoder():
    from src.tokenizer import TagEncoder
    return TagEncoder()
