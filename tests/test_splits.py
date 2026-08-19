"""Tests for patient-level and low-label split integrity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
META = ROOT / "data" / "processed" / "ce_nbi_metadata.csv"


def _require_splits() -> None:
    if not (SPLITS / "patient_level_main_split.csv").exists():
        pytest.skip("Splits not generated yet; run python -m src.data.make_splits")


def test_no_patient_leakage_across_splits() -> None:
    _require_splits()
    main = pd.read_csv(SPLITS / "patient_level_main_split.csv")
    groups = {
        s: set(main.loc[main["split"] == s, "patient_id"].astype(str))
        for s in ("train", "val", "test")
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


def test_both_classes_present_in_each_split() -> None:
    _require_splits()
    main = pd.read_csv(SPLITS / "patient_level_main_split.csv")
    for split in ("train", "val", "test"):
        labels = set(main.loc[main["split"] == split, "binary_label"].unique())
        assert labels == {0, 1}, f"split={split} labels={labels}"


def test_low_label_subsets_are_training_patients_only() -> None:
    _require_splits()
    main = pd.read_csv(SPLITS / "patient_level_main_split.csv")
    train_patients = set(main.loc[main["split"] == "train", "patient_id"].astype(str))
    val_patients = set(main.loc[main["split"] == "val", "patient_id"].astype(str))
    test_patients = set(main.loc[main["split"] == "test", "patient_id"].astype(str))

    low_files = sorted(SPLITS.glob("low_label_*_seed_*.csv"))
    assert low_files, "No low-label split files found"
    for path in low_files:
        df = pd.read_csv(path)
        subset_train = set(df.loc[df["split"] == "train", "patient_id"].astype(str))
        subset_val = set(df.loc[df["split"] == "val", "patient_id"].astype(str))
        subset_test = set(df.loc[df["split"] == "test", "patient_id"].astype(str))
        assert subset_train.issubset(train_patients)
        assert subset_val == val_patients
        assert subset_test == test_patients


def test_validation_and_test_identical_across_label_regimes() -> None:
    _require_splits()
    files = sorted(SPLITS.glob("low_label_*_seed_0.csv"))
    assert files
    ref = pd.read_csv(files[0])
    ref_val = ref.loc[ref["split"] == "val", "image_id"].sort_values().tolist()
    ref_test = ref.loc[ref["split"] == "test", "image_id"].sort_values().tolist()
    for path in files[1:]:
        df = pd.read_csv(path)
        val = df.loc[df["split"] == "val", "image_id"].sort_values().tolist()
        test = df.loc[df["split"] == "test", "image_id"].sort_values().tolist()
        assert val == ref_val
        assert test == ref_test


def test_metadata_row_count_if_present() -> None:
    if not META.exists():
        pytest.skip("Metadata not built yet")
    meta = pd.read_csv(META)
    assert len(meta) > 1000
    assert meta["patient_id"].nunique() > 50
    assert set(meta["binary_label"].unique()) == {0, 1}
