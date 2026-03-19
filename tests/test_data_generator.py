"""Tests for data generator."""
import pytest
from src.data_generator import generate_ner_data, tokens_to_text, bio_to_entities, get_data_stats


class TestGenerateNerData:
    def test_returns_list(self):
        data = generate_ner_data(n_samples=50)
        assert isinstance(data, list)

    def test_correct_count(self):
        data = generate_ner_data(n_samples=75)
        assert len(data) == 75

    def test_has_entities(self):
        data = generate_ner_data(n_samples=100)
        has_entity = any(any(t != "O" for t in tags) for _, tags in data)
        assert has_entity

    def test_reproducible(self):
        d1 = generate_ner_data(n_samples=20, seed=42)
        d2 = generate_ner_data(n_samples=20, seed=42)
        assert d1 == d2


class TestTokensToText:
    def test_basic(self):
        assert tokens_to_text(["hello", "world"]) == "hello world"

    def test_empty(self):
        assert tokens_to_text([]) == ""


class TestBioToEntities:
    def test_per_entity(self):
        toks = ["John", "Smith", "works", "at", "Google"]
        tags = ["B-PER", "I-PER", "O", "O", "B-ORG"]
        ents = bio_to_entities(toks, tags)
        assert len(ents) == 2
        assert ents[0]["type"] == "PER"
        assert ents[1]["type"] == "ORG"

    def test_no_entities(self):
        ents = bio_to_entities(["hello", "world"], ["O", "O"])
        assert len(ents) == 0

    def test_single_token_entity(self):
        ents = bio_to_entities(["Google"], ["B-ORG"])
        assert ents[0]["text"] == "Google"


class TestGetStats:
    def test_stats(self):
        data = generate_ner_data(n_samples=50)
        stats = get_data_stats(data)
        assert stats["n_samples"] == 50
        assert stats["total_entities"] > 0

    def test_empty(self):
        assert get_data_stats([])["n_samples"] == 0
