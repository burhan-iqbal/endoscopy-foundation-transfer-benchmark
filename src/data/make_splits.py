"""Create patient-level and low-label CE-NBI splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logging import setup_logger
from src.utils.paths import ensure_dir, load_yaml, project_root, resolve_path
from src.utils.seed import set_seed

logger = setup_logger("make_splits")


def patient_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with majority/max binary label."""
    g = (
        meta.groupby("patient_id", as_index=False)
        .agg(
            binary_label=("binary_label", "max"),
            n_images=("image_id", "count"),
            lesion_type=("lesion_type", "first"),
        )
    )
    return g


def make_patient_level_split(
    meta: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train/val/test fractions must sum to 1")

    patients = patient_table(meta)
    y = patients["binary_label"].astype(int)

    train_patients, temp_patients = train_test_split(
        patients,
        test_size=(val_frac + test_frac),
        stratify=y,
        random_state=seed,
    )
    temp_y = temp_patients["binary_label"].astype(int)
    relative_test = test_frac / (val_frac + test_frac)
    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=relative_test,
        stratify=temp_y,
        random_state=seed,
    )

    assign = {}
    for pid in train_patients["patient_id"]:
        assign[pid] = "train"
    for pid in val_patients["patient_id"]:
        assign[pid] = "val"
    for pid in test_patients["patient_id"]:
        assign[pid] = "test"

    out = meta.copy()
    out["split"] = out["patient_id"].map(assign)
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), "patient_id"].unique()
        raise RuntimeError(f"Unassigned patients: {missing[:10]}")
    return out[["image_id", "patient_id", "binary_label", "split"]]


def make_low_label_subsets(
    main_split: pd.DataFrame,
    fractions: list[float],
    seeds: list[int],
) -> dict[str, pd.DataFrame]:
    """
    Sample a fraction of *training patients* for each seed/fraction.
    Validation and test rows are always included unchanged.
    """
    outputs: dict[str, pd.DataFrame] = {}
    train_patients = (
        main_split.loc[main_split["split"] == "train", ["patient_id", "binary_label"]]
        .drop_duplicates("patient_id")
    )
    fixed = main_split.loc[main_split["split"].isin(["val", "test"])].copy()

    for seed in seeds:
        for frac in fractions:
            set_seed(seed)
            rng = np.random.default_rng(seed)
            # Stratified patient subsample
            selected = []
            for label in sorted(train_patients["binary_label"].unique()):
                pool = train_patients.loc[
                    train_patients["binary_label"] == label, "patient_id"
                ].tolist()
                n = max(1, int(round(len(pool) * frac))) if frac < 1.0 else len(pool)
                n = min(n, len(pool))
                chosen = rng.choice(pool, size=n, replace=False).tolist()
                selected.extend(chosen)

            train_rows = main_split.loc[
                (main_split["split"] == "train")
                & (main_split["patient_id"].isin(selected))
            ].copy()
            # Mark role: train_subset vs fixed val/test
            train_rows = train_rows.copy()
            subset = pd.concat([train_rows, fixed], ignore_index=True)
            tag = f"{int(round(frac * 100)):02d}"
            key = f"low_label_{tag}_seed_{seed}"
            outputs[key] = subset
    return outputs


def make_image_level_split(
    meta: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """Cautionary random image-level split (leakage-prone)."""
    idx = np.arange(len(meta))
    y = meta["binary_label"].astype(int).to_numpy()
    train_idx, temp_idx = train_test_split(
        idx, test_size=(val_frac + test_frac), stratify=y, random_state=seed
    )
    temp_y = y[temp_idx]
    relative_test = test_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=relative_test, stratify=temp_y, random_state=seed
    )
    split = np.empty(len(meta), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    out = meta[["image_id", "patient_id", "binary_label"]].copy()
    out["split"] = split
    return out


def run(cfg_path: str = "configs/data.yaml") -> None:
    root = project_root()
    cfg = load_yaml(root / cfg_path)
    meta_path = resolve_path(cfg["paths"]["metadata_csv"], root)
    splits_dir = ensure_dir(resolve_path(cfg["paths"]["splits_dir"], root))

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}. Run: python -m src.data.build_metadata"
        )

    meta = pd.read_csv(meta_path)
    split_cfg = cfg["split"]
    seed0 = int(split_cfg["seeds"][0])

    set_seed(seed0)
    main = make_patient_level_split(
        meta,
        train_frac=float(split_cfg["train_frac"]),
        val_frac=float(split_cfg["val_frac"]),
        test_frac=float(split_cfg["test_frac"]),
        seed=seed0,
    )
    main_path = splits_dir / "patient_level_main_split.csv"
    main.to_csv(main_path, index=False)
    logger.info(
        "Wrote %s | train=%d val=%d test=%d patients",
        main_path,
        main.loc[main["split"] == "train", "patient_id"].nunique(),
        main.loc[main["split"] == "val", "patient_id"].nunique(),
        main.loc[main["split"] == "test", "patient_id"].nunique(),
    )

    low = make_low_label_subsets(
        main,
        fractions=[float(x) for x in split_cfg["label_fractions"]],
        seeds=[int(s) for s in split_cfg["seeds"]],
    )
    for name, df in low.items():
        path = splits_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        n_train_p = df.loc[df["split"] == "train", "patient_id"].nunique()
        logger.info("Wrote %s (train patients=%d)", path.name, n_train_p)

    img_split = make_image_level_split(
        meta,
        train_frac=float(split_cfg["train_frac"]),
        val_frac=float(split_cfg["val_frac"]),
        test_frac=float(split_cfg["test_frac"]),
        seed=seed0,
    )
    img_path = splits_dir / "image_level_main_split.csv"
    img_split.to_csv(img_path, index=False)
    logger.info("Wrote cautionary image-level split: %s", img_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
