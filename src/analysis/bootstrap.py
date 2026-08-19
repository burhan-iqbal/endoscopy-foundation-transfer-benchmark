"""Bootstrap confidence intervals for metrics."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.analysis.metrics import compute_binary_metrics


def bootstrap_metric_cis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    metric_fn: Callable | None = None,
) -> dict[str, dict[str, float]]:
    """
    Bootstrap CIs by resampling indices (images or patients, depending on input).
    Returns {metric: {point, low, high}}.
    """
    metric_fn = metric_fn or (lambda yt, yp: compute_binary_metrics(yt, yp))
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_prob)

    keys = [k for k, v in point.items() if isinstance(v, (int, float)) and k not in {"n", "threshold", "tn", "fp", "fn", "tp"}]
    samples = {k: [] for k in keys}

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m = metric_fn(y_true[idx], y_prob[idx])
        for k in keys:
            val = m.get(k, np.nan)
            samples[k].append(val)

    alpha = (1.0 - ci) / 2.0
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        out[k] = {
            "point": float(point[k]),
            "low": float(np.nanquantile(arr, alpha)),
            "high": float(np.nanquantile(arr, 1.0 - alpha)),
        }
    return out
