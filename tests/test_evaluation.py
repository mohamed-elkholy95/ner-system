"""Tests for evaluation."""
import pytest
from src.evaluation import compute_ner_metrics, extract_entities_from_tags, generate_report


class TestExtractEntities:
    def test_single(self):
        tags = ["B-PER", "I-PER", "O", "B-ORG"]
        ents = extract_entities_from_tags(tags)
        assert len(ents) == 2
        assert ents[0] == ("PER", 0, 2)
        assert ents[1] == ("ORG", 3, 4)

    def test_empty(self):
        assert extract_entities_from_tags(["O", "O"]) == []

    def test_continuous(self):
        tags = ["B-PER", "I-PER", "I-PER", "O"]
        ents = extract_entities_from_tags(tags)
        assert ents[0] == ("PER", 0, 3)


class TestComputeMetrics:
    def test_perfect(self):
        y_true = [["B-PER", "O"], ["O", "B-ORG"]]
        y_pred = [["B-PER", "O"], ["O", "B-ORG"]]
        m = compute_ner_metrics(y_true, y_pred)
        assert m["precision"] == 1.0
        assert m["f1"] == 1.0

    def test_zero(self):
        m = compute_ner_metrics([["O", "O"]], [["O", "O"]])
        assert m["f1"] == 0.0  # no entities

    def test_partial(self):
        y_true = [["B-PER", "O", "B-ORG"]]
        y_pred = [["B-PER", "O", "B-ORG", "O"]]
        # Partial overlap: same entities but pred has extra token
        m = compute_ner_metrics(y_true, y_pred)
        assert m["support"] > 0

    def test_per_type(self):
        y_true = [["B-PER", "O", "B-ORG"]]
        y_pred = [["B-PER", "O", "B-ORG"]]
        m = compute_ner_metrics(y_true, y_pred)
        assert "PER" in m["per_type"]
        assert m["per_type"]["PER"]["f1"] == 1.0


class TestGenerateReport:
    def test_output(self):
        report = generate_report({"precision": 0.85, "recall": 0.80, "f1": 0.82})
        assert "# NER Evaluation Report" in report
        assert "| precision |" in report
