"""Tests for evaluation."""
import pytest
from src.evaluation import (
    compute_ner_metrics,
    compute_partial_match_metrics,
    build_tag_confusion_matrix,
    extract_entities_from_tags,
    generate_report,
)


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

    def test_adjacent_same_type(self):
        """Two adjacent B-PER tags should be separate entities."""
        tags = ["B-PER", "B-PER", "O"]
        ents = extract_entities_from_tags(tags)
        assert len(ents) == 2
        assert ents[0] == ("PER", 0, 1)
        assert ents[1] == ("PER", 1, 2)

    def test_type_mismatch_breaks_entity(self):
        """I-ORG after B-PER should close PER and start nothing (BIO violation)."""
        tags = ["B-PER", "I-ORG", "O"]
        ents = extract_entities_from_tags(tags)
        # PER ends at index 1, I-ORG is a BIO violation (no preceding B-ORG)
        assert len(ents) == 1
        assert ents[0] == ("PER", 0, 1)

    def test_entity_at_end(self):
        """Entity at the end of a sequence (no trailing O)."""
        tags = ["O", "B-LOC", "I-LOC"]
        ents = extract_entities_from_tags(tags)
        assert len(ents) == 1
        assert ents[0] == ("LOC", 1, 3)

    def test_all_entities_no_O(self):
        """Sequence with no O tags at all."""
        tags = ["B-PER", "I-PER", "B-ORG"]
        ents = extract_entities_from_tags(tags)
        assert len(ents) == 2


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
        m = compute_ner_metrics(y_true, y_pred)
        assert m["support"] > 0

    def test_per_type(self):
        y_true = [["B-PER", "O", "B-ORG"]]
        y_pred = [["B-PER", "O", "B-ORG"]]
        m = compute_ner_metrics(y_true, y_pred)
        assert "PER" in m["per_type"]
        assert m["per_type"]["PER"]["f1"] == 1.0

    def test_false_positive(self):
        """Model predicts an entity where none exists → precision drops."""
        y_true = [["O", "O", "O"]]
        y_pred = [["B-PER", "O", "O"]]
        m = compute_ner_metrics(y_true, y_pred)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0

    def test_false_negative(self):
        """Model misses a real entity → recall drops."""
        y_true = [["B-PER", "O"]]
        y_pred = [["O", "O"]]
        m = compute_ner_metrics(y_true, y_pred)
        assert m["recall"] == 0.0

    def test_boundary_error(self):
        """Predicted span partially overlaps true span → exact match fails."""
        y_true = [["B-PER", "I-PER", "O"]]
        y_pred = [["O", "B-PER", "O"]]
        m = compute_ner_metrics(y_true, y_pred)
        # (PER, 0, 2) != (PER, 1, 2) → no exact match
        assert m["f1"] == 0.0


class TestPartialMatchMetrics:
    def test_perfect(self):
        y_true = [["B-PER", "I-PER", "O"]]
        y_pred = [["B-PER", "I-PER", "O"]]
        m = compute_partial_match_metrics(y_true, y_pred)
        assert m["partial_f1"] == 1.0

    def test_overlap(self):
        """Partial overlap should give partial credit."""
        y_true = [["B-PER", "I-PER", "I-PER", "O"]]
        y_pred = [["O", "B-PER", "I-PER", "O"]]
        m = compute_partial_match_metrics(y_true, y_pred)
        assert 0.0 < m["partial_f1"] < 1.0

    def test_no_overlap(self):
        y_true = [["B-PER", "O", "O"]]
        y_pred = [["O", "O", "B-ORG"]]
        m = compute_partial_match_metrics(y_true, y_pred)
        assert m["partial_f1"] == 0.0

    def test_empty(self):
        m = compute_partial_match_metrics([["O"]], [["O"]])
        assert m["partial_f1"] == 0.0


class TestConfusionMatrix:
    def test_basic(self):
        y_true = [["B-PER", "O", "B-ORG"]]
        y_pred = [["B-PER", "O", "B-LOC"]]
        cm = build_tag_confusion_matrix(y_true, y_pred)
        assert cm["PER"]["PER"] == 1
        assert cm["ORG"]["LOC"] == 1
        assert cm["O"]["O"] == 1

    def test_all_correct(self):
        y_true = [["O", "B-PER", "O"]]
        y_pred = [["O", "B-PER", "O"]]
        cm = build_tag_confusion_matrix(y_true, y_pred)
        assert cm["O"]["O"] == 2
        assert cm["PER"]["PER"] == 1


class TestGenerateReport:
    def test_output(self):
        report = generate_report({"precision": 0.85, "recall": 0.80, "f1": 0.82})
        assert "# NER Evaluation Report" in report
        assert "| precision |" in report

    def test_per_type_section(self):
        metrics = {
            "precision": 0.9, "recall": 0.85, "f1": 0.87, "support": 100,
            "per_type": {
                "PER": {"precision": 0.95, "recall": 0.90, "f1": 0.92, "support": 40},
                "ORG": {"precision": 0.85, "recall": 0.80, "f1": 0.82, "support": 60},
            },
        }
        report = generate_report(metrics, model_name="CRF v2")
        assert "CRF v2" in report
        assert "| PER |" in report
        assert "Per-Entity-Type" in report
