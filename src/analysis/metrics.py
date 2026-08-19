"""Classification metrics for CE-NBI benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0

    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def aggregate_patient_level(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: list[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Average predicted probability per patient; take max label as patient label."""
    df_true = {}
    df_prob = {}
    counts = {}
    for yt, yp, pid in zip(y_true, y_prob, patient_ids):
        pid = str(pid)
        df_true[pid] = max(df_true.get(pid, 0), int(yt))
        df_prob[pid] = df_prob.get(pid, 0.0) + float(yp)
        counts[pid] = counts.get(pid, 0) + 1
    pids = sorted(df_true.keys())
    yt = np.array([df_true[p] for p in pids], dtype=int)
    yp = np.array([df_prob[p] / counts[p] for p in pids], dtype=float)
    return yt, yp, pids
