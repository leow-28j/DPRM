"""Standalone metrics for semantic change detection.

This file deliberately has no model, dataset, or DPRM dependency. It accepts
integer class-ID arrays of identical shape and reports confusion-matrix metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _as_labels(array: np.ndarray, name: str) -> np.ndarray:
    """Return a two-dimensional integer label map or raise a clear error."""
    labels = np.asarray(array)
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    if labels.ndim != 2:
        raise ValueError(f"{name} must be a 2-D class-ID map, got {labels.shape}.")
    if not (np.issubdtype(labels.dtype, np.integer) or labels.dtype == np.bool_):
        raise ValueError(f"{name} must contain integer class IDs, got {labels.dtype}.")
    return labels


def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Build a ``target x prediction`` confusion matrix for class-ID maps."""
    if num_classes < 1:
        raise ValueError("num_classes must be positive.")
    prediction = _as_labels(prediction, "prediction")
    target = _as_labels(target, "target")
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} versus {target.shape}.")

    valid = (target >= 0) & (target < num_classes)
    if ignore_index is not None:
        valid &= target != ignore_index
    valid &= (prediction >= 0) & (prediction < num_classes)
    codes = num_classes * target[valid].astype(np.int64) + prediction[valid]
    return np.bincount(codes, minlength=num_classes**2).reshape(num_classes, num_classes)


def summarize_confusion(matrix: np.ndarray) -> dict[str, Any]:
    """Return overall accuracy, mean IoU, and per-class IoU from a matrix."""
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    total = int(matrix.sum())
    diagonal = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1) - diagonal
    iou = np.divide(
        diagonal,
        denominator,
        out=np.full(matrix.shape[0], np.nan),
        where=denominator != 0,
    )
    accuracy = float(diagonal.sum() / total) if total else float("nan")
    return {
        "overall_accuracy": accuracy,
        "mean_iou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan"),
        "per_class_iou": iou.tolist(),
        "valid_pixels": total,
    }


def binary_change_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    """Compute precision, recall, F1, IoU, and accuracy for binary masks."""
    prediction = _as_labels(prediction, "prediction") != 0
    target = _as_labels(target, "target") != 0
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} versus {target.shape}.")
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    total = tp + fp + fn + tn
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": (tp + tn) / total if total else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _kappa(matrix: np.ndarray) -> float:
    """Compute Cohen's kappa from a target-by-prediction confusion matrix."""
    total = float(matrix.sum())
    if total == 0:
        return 0.0
    observed = float(np.trace(matrix) / total)
    expected = float(matrix.sum(axis=1) @ matrix.sum(axis=0) / (total * total))
    return 0.0 if np.isclose(expected, 1.0) else (observed - expected) / (1.0 - expected)


def scd_paper_metrics(
    pred_t1: np.ndarray,
    pred_t2: np.ndarray,
    target_t1: np.ndarray,
    target_t2: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, float | int]:
    """Compute the state-based OA, Fscd, mIoU and SeK used for SCD reporting.

    A no-change pixel is state 0. A changed pixel is encoded by its ordered
    ``(T1, T2)`` class pair. The binary change map is derived by comparing the
    two semantic maps. This is a standalone implementation of the metric
    definitions, not an implementation of any DPRM component.
    """
    pred_t1, pred_t2 = _as_labels(pred_t1, "pred_t1"), _as_labels(pred_t2, "pred_t2")
    target_t1, target_t2 = _as_labels(target_t1, "target_t1"), _as_labels(target_t2, "target_t2")
    shapes = {array.shape for array in (pred_t1, pred_t2, target_t1, target_t2)}
    if len(shapes) != 1:
        raise ValueError(f"all maps must share one shape, got {sorted(shapes)}.")

    valid_target = (target_t1 >= 0) & (target_t1 < num_classes)
    valid_target &= (target_t2 >= 0) & (target_t2 < num_classes)
    if ignore_index is not None:
        valid_target &= (target_t1 != ignore_index) & (target_t2 != ignore_index)
    valid_pred = (pred_t1 >= 0) & (pred_t1 < num_classes)
    valid_pred &= (pred_t2 >= 0) & (pred_t2 < num_classes)

    state_count = num_classes * num_classes + 1
    target_state = np.full(target_t1.shape, -1, dtype=np.int64)
    pred_state = np.full(pred_t1.shape, -1, dtype=np.int64)
    target_state[valid_target] = (
        target_t1[valid_target].astype(np.int64) * num_classes
        + target_t2[valid_target].astype(np.int64)
        + 1
    )
    pred_state[valid_pred] = (
        pred_t1[valid_pred].astype(np.int64) * num_classes
        + pred_t2[valid_pred].astype(np.int64)
        + 1
    )
    target_state[valid_target & (target_t1 == target_t2)] = 0
    pred_state[valid_pred & (pred_t1 == pred_t2)] = 0
    matrix = confusion_matrix(pred_state, target_state, state_count, ignore_index=-1)

    total = int(matrix.sum())
    oa = float(np.trace(matrix) / total) if total else float("nan")
    true_unchanged = int(matrix[0, :].sum())
    predicted_unchanged = int(matrix[:, 0].sum())
    correct_unchanged = int(matrix[0, 0])
    true_changed = total - true_unchanged
    predicted_changed = total - predicted_unchanged
    correctly_classified_changed = int(np.trace(matrix[1:, 1:]))
    precision = correctly_classified_changed / predicted_changed if predicted_changed else 0.0
    recall = correctly_classified_changed / true_changed if true_changed else 0.0
    fscd = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    unchanged_union = true_unchanged + predicted_unchanged - correct_unchanged
    changed_intersection = int(matrix[1:, 1:].sum())
    changed_union = true_changed + predicted_changed - changed_intersection
    iou_unchanged = correct_unchanged / unchanged_union if unchanged_union else 0.0
    iou_changed = changed_intersection / changed_union if changed_union else 0.0
    miou = (iou_unchanged + iou_changed) / 2.0

    matrix_without_true_negatives = matrix.copy()
    matrix_without_true_negatives[0, 0] = 0
    kappa_n0 = _kappa(matrix_without_true_negatives)
    sek = kappa_n0 * float(np.exp(iou_changed)) / float(np.e)
    return {
        "OA": oa,
        "Fscd": fscd,
        "mIoU": miou,
        "SeK": sek,
        "Kappa_n0": kappa_n0,
        "binary_iou_changed": iou_changed,
        "valid_pixels": total,
    }


def semantic_change_metrics(
    pred_t1: np.ndarray,
    pred_t2: np.ndarray,
    target_t1: np.ndarray,
    target_t2: np.ndarray,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, Any]:
    """Return date-wise diagnostics plus paper-style state-based SCD metrics."""
    pred_t1, pred_t2 = _as_labels(pred_t1, "pred_t1"), _as_labels(pred_t2, "pred_t2")
    target_t1, target_t2 = _as_labels(target_t1, "target_t1"), _as_labels(target_t2, "target_t2")
    shapes = {array.shape for array in (pred_t1, pred_t2, target_t1, target_t2)}
    if len(shapes) != 1:
        raise ValueError(f"all maps must share one shape, got {sorted(shapes)}.")
    date_1 = summarize_confusion(confusion_matrix(pred_t1, target_t1, num_classes, ignore_index))
    date_2 = summarize_confusion(confusion_matrix(pred_t2, target_t2, num_classes, ignore_index))
    valid = (target_t1 >= 0) & (target_t1 < num_classes) & (target_t2 >= 0) & (target_t2 < num_classes)
    if ignore_index is not None:
        valid &= (target_t1 != ignore_index) & (target_t2 != ignore_index)
    predicted_change = (pred_t1 != pred_t2)[valid].reshape(1, -1)
    target_change = (target_t1 != target_t2)[valid].reshape(1, -1)
    return {
        "date_1": date_1,
        "date_2": date_2,
        "mean_overall_accuracy": float(np.nanmean([date_1["overall_accuracy"], date_2["overall_accuracy"]])),
        "mean_iou": float(np.nanmean([date_1["mean_iou"], date_2["mean_iou"]])),
        "binary_change": binary_change_metrics(predicted_change, target_change),
        "paper_metrics": scd_paper_metrics(
            pred_t1, pred_t2, target_t1, target_t2, num_classes, ignore_index
        ),
    }


def _read_label(path: str) -> np.ndarray:
    return np.asarray(Image.open(Path(path)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate semantic change prediction maps.")
    parser.add_argument("--pred-t1", required=True)
    parser.add_argument("--pred-t2", required=True)
    parser.add_argument("--target-t1", required=True)
    parser.add_argument("--target-t2", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--ignore-index", type=int)
    args = parser.parse_args()
    report = semantic_change_metrics(
        _read_label(args.pred_t1), _read_label(args.pred_t2),
        _read_label(args.target_t1), _read_label(args.target_t2),
        args.num_classes, args.ignore_index,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
