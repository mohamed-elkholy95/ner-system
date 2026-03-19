"""Tests for tokenizer."""
import pytest
from src.tokenizer import NERTokenizer, TagEncoder, extract_features


class TestNERTokenizer:
    def test_fit(self, sample_texts):
        tok = NERTokenizer().fit(sample_texts)
        assert tok.vocab_size > 2  # at least special tokens

    def test_encode_decode(self, sample_texts):
        tok = NERTokenizer().fit(sample_texts)
        ids = tok.encode(["hello", "world"])
        assert len(ids) == 2

    def test_pad_id(self, sample_texts):
        tok = NERTokenizer().fit(sample_texts)
        assert tok.pad_id == 0

    def test_unfitted_raises(self):
        tok = NERTokenizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            tok.encode(["test"])


class TestTagEncoder:
    def test_encode_decode(self):
        enc = TagEncoder()
        tags = ["B-PER", "I-PER", "O", "B-ORG"]
        ids = enc.encode(tags)
        decoded = enc.decode(ids)
        assert decoded == tags

    def test_num_tags(self):
        enc = TagEncoder()
        assert enc.num_tags == 11

    def test_pad_id(self):
        enc = TagEncoder()
        assert enc.pad_id == 0  # O tag


class TestExtractFeatures:
    def test_shape(self):
        feats = extract_features(["hello", "world", "test"])
        assert feats.shape == (3, 6)

    def test_empty(self):
        feats = extract_features([])
        assert feats.shape[0] == 0
