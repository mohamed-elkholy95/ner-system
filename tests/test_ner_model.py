"""Tests for NER model."""
import pytest
from src.ner_model import CRFTagger, BiLSTMNERTagger, word2features, sent2features, train_bilstm, predict_bilstm


class TestWord2Features:
    def test_basic(self):
        f = word2features(["John", "works", "at", "Google"], 0)
        assert "word.lower()" in f
        assert f["word.lower()"] == "john"
        assert f["word.istitle()"] is True

    def test_context(self):
        f = word2features(["Hello", "World"], 0)
        assert "+1:word.lower()" in f

    def test_boundary(self):
        f = word2features(["Hello"], 0)
        assert "-1:word.lower()" not in f


class TestSent2Features:
    def test_length(self):
        feats = sent2features(["hello", "world"])
        assert len(feats) == 2


class TestCRFTagger:
    def test_predict_untrained(self):
        tagger = CRFTagger()
        preds = tagger.predict([["John", "Smith"]])
        assert len(preds) == 1
        assert all(t == "O" for t in preds[0])

    def test_fit_mock(self):
        tagger = CRFTagger()
        tagger.fit([["hello"]], [["O"]])
        # If sklearn_crfsuite not available, mock training
        preds = tagger.predict([["hello"]])
        assert len(preds) == 1


class TestBiLSTMNERTagger:
    def test_init(self):
        model = BiLSTMNERTagger(vocab_size=100, embedding_dim=32, hidden_dim=64, num_tags=11)
        assert model is not None

    def test_train_bilstm_mock(self):
        history = train_bilstm(None, [["hello"]], [["O"]], None, None, epochs=3)
        assert "loss" in history
        assert len(history["loss"]) == 3

    def test_predict_bilstm_mock(self):
        tags = predict_bilstm(None, ["hello", "world"], None, None)
        assert tags == ["O", "O"]
