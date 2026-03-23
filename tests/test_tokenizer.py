"""Tests for tokenizer."""
import pytest
from src.tokenizer import NERTokenizer, TagEncoder, extract_features, word_shape


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

    def test_min_freq_filters(self, sample_texts):
        """Tokens below min_freq should be excluded from vocabulary."""
        tok = NERTokenizer(min_freq=9999).fit(sample_texts)
        # Only special tokens should survive such a high threshold
        assert tok.vocab_size == 2

    def test_decode_round_trip(self, sample_texts):
        """Encode → decode should recover original tokens (lowercased)."""
        tok = NERTokenizer().fit(sample_texts)
        words = sample_texts[0].lower().split()[:3]
        ids = tok.encode(words)
        decoded = tok.decode(ids)
        assert decoded == words


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

    def test_unknown_tag_defaults_to_zero(self):
        """Tags not in the schema should map to ID 0 (the 'O' tag)."""
        enc = TagEncoder()
        ids = enc.encode(["B-UNKNOWN"])
        assert ids == [0]

    def test_custom_tags(self):
        """Encoder should work with a custom tag set."""
        enc = TagEncoder(tags=["O", "B-FOOD", "I-FOOD"])
        assert enc.num_tags == 3
        assert enc.encode(["B-FOOD", "I-FOOD"]) == [1, 2]


class TestWordShape:
    def test_title_case(self):
        assert word_shape("John") == "Xxxx"

    def test_all_caps(self):
        assert word_shape("NASA") == "XXXX"

    def test_digits(self):
        assert word_shape("2024") == "dddd"

    def test_mixed(self):
        assert word_shape("COVID-19") == "XXXXX-dd"

    def test_empty(self):
        assert word_shape("") == ""


class TestExtractFeatures:
    def test_shape(self):
        feats = extract_features(["hello", "world", "test"])
        assert feats.shape == (3, 10)

    def test_empty(self):
        feats = extract_features([])
        assert feats.shape[0] == 0

    def test_uppercase_ratio(self):
        """All-caps token should have uppercase ratio of 1.0."""
        feats = extract_features(["NASA"])
        assert feats[0, 9] == 1.0  # uppercase ratio column

    def test_hyphen_detection(self):
        """Hyphenated tokens should have the hyphen feature set."""
        feats = extract_features(["COVID-19"])
        assert feats[0, 6] == 1.0  # has_hyphen column
