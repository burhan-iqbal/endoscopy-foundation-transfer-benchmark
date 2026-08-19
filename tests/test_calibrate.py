"""Temperature scaling must keep T > 0 and must not invert rankings."""

from __future__ import annotations

import numpy as np
import pytest

from src.training.calibrate import (
    apply_temperature,
    fit_temperature,
    is_valid_temperature,
)


def test_fit_temperature_is_strictly_positive() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=200)
    labels = (rng.random(200) < 1.0 / (1.0 + np.exp(-logits))).astype(float)
    temperature = fit_temperature(logits, labels)
    assert is_valid_temperature(temperature)


def test_apply_temperature_preserves_ranking() -> None:
    logits = np.array([-2.0, -0.5, 0.1, 1.5, 3.0])
    labels = np.array([0, 0, 1, 1, 1])
    from src.analysis.metrics import compute_binary_metrics

    before = compute_binary_metrics(labels, 1.0 / (1.0 + np.exp(-logits)))
    after = compute_binary_metrics(labels, apply_temperature(logits, 2.0))
    assert after["auroc"] == pytest.approx(before["auroc"])


def test_nonpositive_temperature_is_rejected() -> None:
    logits = np.array([-1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(logits, -0.089)
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(logits, 0.0)
    assert not is_valid_temperature(-0.089)
    assert not is_valid_temperature(0.0)
    assert is_valid_temperature(1.0)
