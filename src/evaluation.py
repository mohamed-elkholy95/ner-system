"""NER evaluation metrics.

This module implements entity-level evaluation for NER systems. The key insight
is that **token-level accuracy is misleading** for NER: if 85% of tokens are 'O'
(outside any entity), a model that always predicts 'O' gets 85% accuracy but
finds zero entities.

Instead, we use **entity-level** metrics:
  - An entity is correct only if BOTH the type AND the exact span match.
  - Precision = correct entities / predicted entities
  - Recall = correct entities / true entities
  - F1 = harmonic mean of precision and recall

We also provide **partial match scoring** (via ``compute_partial_match_metrics``)
which gives credit for partially overlapping spans — useful during development
to see if the model is "close" even when exact boundaries are off.
"""
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_entities_from_tags(tags: List[str]) -> List[Tuple[str, int, int]]:
    """Extract entity spans from BIO tags.

    Returns:
        List of (entity_type, start_idx, end_idx).
    """
    entities = []
    current_type = None
    current_start = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if current_type is not None:
                entities.append((current_type, current_start, i))
            current_type = tag[2:]
            current_start = i
        elif tag.startswith("I-") and current_type == tag[2:]:
            pass
        else:
            if current_type is not None:
                entities.append((current_type, current_start, i))
                current_type = None

    if current_type is not None:
        entities.append((current_type, current_start, len(tags)))

    return entities


def compute_ner_metrics(
    y_true: List[List[str]], y_pred: List[List[str]],
) -> Dict[str, float]:
    """Compute entity-level precision, recall, F1.

    Args:
        y_true: True tag sequences.
        y_pred: Predicted tag sequences.

    Returns:
        Metrics dict.
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    per_type: Dict[str, Dict[str, int]] = {}

    for true_tags, pred_tags in zip(y_true, y_pred):
        true_ents = set(extract_entities_from_tags(true_tags))
        pred_ents = set(extract_entities_from_tags(pred_tags))

        tp = len(true_ents & pred_ents)
        fp = len(pred_ents - true_ents)
        fn = len(true_ents - pred_ents)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-type
        for etype in set([e[0] for e in true_ents] + [e[0] for e in pred_ents]):
            if etype not in per_type:
                per_type[etype] = {"tp": 0, "fp": 0, "fn": 0}
            true_t = set(e for e in true_ents if e[0] == etype)
            pred_t = set(e for e in pred_ents if e[0] == etype)
            per_type[etype]["tp"] += len(true_t & pred_t)
            per_type[etype]["fp"] += len(pred_t - true_t)
            per_type[etype]["fn"] += len(true_t - pred_t)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_type_metrics = {}
    for etype, counts in per_type.items():
        p = counts["tp"] / (counts["tp"] + counts["fp"]) if (counts["tp"] + counts["fp"]) > 0 else 0.0
        r = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_type_metrics[etype] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                                    "support": counts["tp"] + counts["fn"]}

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": total_tp + total_fn,
        "per_type": per_type_metrics,
    }


def compute_partial_match_metrics(
    y_true: List[List[str]], y_pred: List[List[str]],
) -> Dict[str, float]:
    """Compute partial match entity metrics using span overlap.

    Unlike exact match (where the span boundaries must match perfectly),
    partial match gives proportional credit for overlapping spans. This
    is useful for debugging: if partial F1 is high but exact F1 is low,
    the model finds the right entities but gets boundaries wrong.

    A predicted entity gets credit = |overlap| / max(|pred_span|, |true_span|)
    for the best-matching true entity of the same type.

    Args:
        y_true: True tag sequences.
        y_pred: Predicted tag sequences.

    Returns:
        Dict with partial_precision, partial_recall, partial_f1.
    """
    total_pred_score = 0.0
    total_true_score = 0.0
    total_pred_count = 0
    total_true_count = 0

    for true_tags, pred_tags in zip(y_true, y_pred):
        true_ents = extract_entities_from_tags(true_tags)
        pred_ents = extract_entities_from_tags(pred_tags)
        total_pred_count += len(pred_ents)
        total_true_count += len(true_ents)

        # For each predicted entity, find best overlap with same-type true entity
        for p_type, p_start, p_end in pred_ents:
            best_overlap = 0.0
            for t_type, t_start, t_end in true_ents:
                if p_type != t_type:
                    continue
                overlap_start = max(p_start, t_start)
                overlap_end = min(p_end, t_end)
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    span_max = max(p_end - p_start, t_end - t_start)
                    best_overlap = max(best_overlap, overlap_len / span_max)
            total_pred_score += best_overlap

        # For each true entity, find best overlap with same-type predicted entity
        for t_type, t_start, t_end in true_ents:
            best_overlap = 0.0
            for p_type, p_start, p_end in pred_ents:
                if p_type != t_type:
                    continue
                overlap_start = max(p_start, t_start)
                overlap_end = min(p_end, t_end)
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    span_max = max(p_end - p_start, t_end - t_start)
                    best_overlap = max(best_overlap, overlap_len / span_max)
            total_true_score += best_overlap

    partial_precision = total_pred_score / total_pred_count if total_pred_count > 0 else 0.0
    partial_recall = total_true_score / total_true_count if total_true_count > 0 else 0.0
    partial_f1 = (
        2 * partial_precision * partial_recall / (partial_precision + partial_recall)
        if (partial_precision + partial_recall) > 0 else 0.0
    )

    return {
        "partial_precision": round(partial_precision, 4),
        "partial_recall": round(partial_recall, 4),
        "partial_f1": round(partial_f1, 4),
    }


def build_tag_confusion_matrix(
    y_true: List[List[str]], y_pred: List[List[str]],
) -> Dict[str, Dict[str, int]]:
    """Build a token-level confusion matrix for NER tags.

    Returns a nested dict where result[true_tag][pred_tag] = count.
    Useful for identifying systematic errors like ORG↔LOC confusion
    (common because organizations are often named after locations).

    Args:
        y_true: True tag sequences.
        y_pred: Predicted tag sequences.

    Returns:
        Nested dict mapping true_tag → pred_tag → count.
    """
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for true_tags, pred_tags in zip(y_true, y_pred):
        for t_tag, p_tag in zip(true_tags, pred_tags):
            # Collapse B- and I- prefixes to just the entity type for readability
            t_key = t_tag if t_tag == "O" else t_tag.split("-", 1)[1]
            p_key = p_tag if p_tag == "O" else p_tag.split("-", 1)[1]
            matrix[t_key][p_key] += 1

    # Convert defaultdicts to regular dicts for clean serialization
    return {k: dict(v) for k, v in matrix.items()}


def generate_report(metrics: Dict[str, float], model_name: str = "NER Model") -> str:
    """Generate a markdown evaluation report.

    Args:
        metrics: Output of ``compute_ner_metrics``.
        model_name: Display name for the report header.

    Returns:
        Markdown-formatted string.
    """
    lines = [f"# NER Evaluation Report — {model_name}", ""]
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k in ["precision", "recall", "f1", "support"]:
        if k in metrics:
            lines.append(f"| {k} | {metrics[k]} |")

    if "per_type" in metrics and metrics["per_type"]:
        lines.append("\n## Per-Entity-Type Performance")
        lines.append("| Entity | Precision | Recall | F1 | Support |")
        lines.append("|--------|-----------|--------|----|---------|")
        for etype, m in sorted(metrics["per_type"].items()):
            lines.append(f"| {etype} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |")

    return "\n".join(lines)
