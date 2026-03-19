"""NER evaluation metrics."""
import logging
from typing import Dict, List, Tuple

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


def generate_report(metrics: Dict[str, float], model_name: str = "NER Model") -> str:
    """Generate markdown evaluation report."""
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
        for etype, m in metrics["per_type"].items():
            lines.append(f"| {etype} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |")

    return "\n".join(lines)
