"""Smoke tests for metric helpers."""

from __future__ import annotations

import numpy as np

from src.analysis.bootstrap import bootstrap_metric_cis
from src.analysis.metrics import (
    aggregate_patient_level,
    compute_binary_metrics,
    expected_calibration_error,
)


def test_compute_binary_metrics_perfect() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_binary_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["auroc"] == 1.0


def test_ece_bounds() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    ece = expected_calibration_error(y_true, y_prob, n_bins=4)
    assert 0.0 <= ece <= 1.0


def test_patient_aggregation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.6, 0.8])
    pids = ["a", "a", "b", "b"]
    yt, yp, out_pids = aggregate_patient_level(y_true, y_prob, pids)
    assert out_pids == ["a", "b"]
    assert list(yt) == [0, 1]
    assert np.isclose(yp[0], 0.2)
    assert np.isclose(yp[1], 0.7)


def test_bootstrap_runs() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=40)
    y_prob = rng.random(40)
    cis = bootstrap_metric_cis(y_true, y_prob, n_resamples=50, seed=0)
    assert "auroc" in cis
    assert cis["auroc"]["low"] <= cis["auroc"]["point"] <= cis["auroc"]["high"]
