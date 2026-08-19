"""Plotting helpers for label-efficiency, calibration, and confusion matrices."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cenbi")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.logging import setup_logger
from src.utils.paths import ensure_dir, load_yaml, project_root, resolve_path

logger = setup_logger("plots")


def _is_valid_temperature(temperature: object) -> bool:
    try:
        value = float(temperature)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and value > 0.0)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> None:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc, bin_conf, bin_count = [], [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (
            y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1]
        )
        if not np.any(mask):
            continue
        bin_acc.append(y_true[mask].mean())
        bin_conf.append(y_prob[mask].mean())
        bin_count.append(mask.sum())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, label="Model", color="#4C78A8")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_label_efficiency(
    metrics_table: pd.DataFrame,
    out_path: Path,
    metric: str = "image_auroc",
) -> None:
    df = metrics_table.dropna(subset=["label_frac", metric])
    if df.empty:
        logger.warning("No rows with label_frac and %s; skipping %s", metric, out_path.name)
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, g in df.groupby("model"):
        stats = (
            g.groupby("label_frac")[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("label_frac")
        )
        ax.plot(stats["label_frac"], stats["mean"], marker="o", label=model)
        band = stats[stats["count"] > 1]
        if not band.empty:
            ax.fill_between(
                band["label_frac"],
                band["mean"] - band["std"],
                band["mean"] + band["std"],
                alpha=0.2,
            )
    ax.set_xlabel("Fraction of labelled training patients")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"Label efficiency ({metric}, mean over seeds, band = +/-1 std)")
    ax.set_xticks(sorted(df["label_frac"].unique()))
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def aggregate_run_metrics(logs_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in logs_dir.rglob("metrics.json"):
        with open(metrics_path, encoding="utf-8") as f:
            payload = json.load(f)
        row = {
            "model": payload.get("model"),
            "label_frac": payload.get("label_frac"),
            "seed": payload.get("seed"),
            **payload.get("test_image", {}),
        }
        # prefix patient metrics
        for k, v in payload.get("test_patient", {}).items():
            row[f"patient_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not parse %s", path)
        return None


def _get(payload: dict | None, *keys: str):
    cur = payload
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return np.nan
        cur = cur[k]
    return cur


def _run_row(model: str, tag: str, seed: int, run_dir: Path) -> dict:
    metrics = _load_json(run_dir / "metrics.json")
    metrics_eval = _load_json(run_dir / "metrics_eval.json")
    calibration = _load_json(run_dir / "calibration.json")
    eval_src = metrics_eval if metrics_eval is not None else metrics
    m = re.fullmatch(r"label_(\d+)", tag)
    return {
        "model": model,
        "tag": tag,
        "label_frac": int(m.group(1)) / 100.0 if m else np.nan,
        "seed": seed,
        "image_auroc": _get(eval_src, "test_image", "auroc"),
        "patient_auroc": _get(eval_src, "test_patient", "auroc"),
        "image_auroc_ci_lo": _get(eval_src, "bootstrap_image", "auroc", "low"),
        "image_auroc_ci_hi": _get(eval_src, "bootstrap_image", "auroc", "high"),
        "patient_auroc_ci_lo": _get(eval_src, "bootstrap_patient", "auroc", "low"),
        "patient_auroc_ci_hi": _get(eval_src, "bootstrap_patient", "auroc", "high"),
        "ece_before": _get(calibration, "before", "ece"),
        "ece_after": _get(calibration, "after", "ece"),
        "brier_before": _get(calibration, "before", "brier"),
        "brier_after": _get(calibration, "after", "brier"),
        "temperature": _get(calibration, "temperature"),
        "valid_temperature": _is_valid_temperature(_get(calibration, "temperature")),
        "has_metrics": metrics is not None,
        "has_metrics_eval": metrics_eval is not None,
        "has_calibration": calibration is not None,
    }


def collect_run_rows(logs_dir: Path) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        for tag_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            for seed_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()):
                m = re.fullmatch(r"seed_(\d+)", seed_dir.name)
                if m is None:
                    continue
                rows.append(_run_row(model_dir.name, tag_dir.name, int(m.group(1)), seed_dir))
    return pd.DataFrame(rows)


def report_missing_runs(runs: pd.DataFrame, expected_seeds: list[int]) -> None:
    for (model, tag), g in runs.groupby(["model", "tag"]):
        if not re.fullmatch(r"label_\d+", str(tag)):
            continue
        missing = sorted(set(expected_seeds) - set(g["seed"].astype(int)))
        if missing:
            print(f"WARNING: {model}/{tag}: missing seeds {','.join(str(s) for s in missing)}")
        for _, row in g.iterrows():
            absent = [
                name
                for name, present in [
                    ("metrics.json", row["has_metrics"]),
                    ("metrics_eval.json", row["has_metrics_eval"]),
                    ("calibration.json", row["has_calibration"]),
                ]
                if not present
            ]
            if absent:
                print(f"WARNING: {model}/{tag}/seed_{int(row['seed'])}: missing {', '.join(absent)}")


def build_main_results_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, tag), g in runs.groupby(["model", "tag"]):
        rows.append(
            {
                "model": model,
                "tag": tag,
                "label_frac": g["label_frac"].iloc[0],
                "n_seeds": int(g["seed"].nunique()),
                "image_auroc_mean": g["image_auroc"].mean(),
                "image_auroc_std": g["image_auroc"].std(),
                "patient_auroc_mean": g["patient_auroc"].mean(),
                "patient_auroc_std": g["patient_auroc"].std(),
                "image_auroc_ci_lo_mean": g["image_auroc_ci_lo"].mean(),
                "image_auroc_ci_hi_mean": g["image_auroc_ci_hi"].mean(),
                "patient_auroc_ci_lo_mean": g["patient_auroc_ci_lo"].mean(),
                "patient_auroc_ci_hi_mean": g["patient_auroc_ci_hi"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "label_frac", "tag"]).reset_index(drop=True)


def build_calibration_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean calibration by model/regime.

    ECE/Brier *before* uses every seed. ECE/Brier *after* and mean T use only
    seeds with T > 0: applying a non-positive temperature inverts rankings and
    is not a valid calibration (see Methods).
    """
    rows = []
    for (model, tag), g in runs.groupby(["model", "tag"]):
        valid = g[g["valid_temperature"]] if "valid_temperature" in g.columns else g
        invalid = (
            g[~g["valid_temperature"]]
            if "valid_temperature" in g.columns
            else g.iloc[0:0]
        )
        if len(valid) < len(g):
            seeds = ", ".join(str(int(s)) for s in invalid["seed"].tolist())
            logger.warning(
                "%s/%s: excluding %d non-positive T fit(s) from ECE-after (seeds %s)",
                model,
                tag,
                len(g) - len(valid),
                seeds,
            )
        rows.append(
            {
                "model": model,
                "tag": tag,
                "label_frac": g["label_frac"].iloc[0],
                "n_seeds": int(g["seed"].nunique()),
                "n_valid_temperature": int(len(valid)),
                "ece_before_mean": g["ece_before"].mean(),
                "ece_before_std": g["ece_before"].std(),
                "ece_after_mean": valid["ece_after"].mean() if len(valid) else np.nan,
                "ece_after_std": valid["ece_after"].std() if len(valid) > 1 else np.nan,
                "brier_before_mean": g["brier_before"].mean(),
                "brier_before_std": g["brier_before"].std(),
                "brier_after_mean": valid["brier_after"].mean() if len(valid) else np.nan,
                "brier_after_std": valid["brier_after"].std() if len(valid) > 1 else np.nan,
                "temperature_mean": valid["temperature"].mean() if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "label_frac", "tag"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--config", default="configs/experiments.yaml")
    args = parser.parse_args()
    root = project_root()
    cfg = load_yaml(root / args.config)
    logs_dir = resolve_path(cfg["paths"]["logs_dir"], root)
    figures_dir = ensure_dir(resolve_path(cfg["paths"]["figures_dir"], root))
    tables_dir = ensure_dir(resolve_path(cfg["paths"]["tables_dir"], root))

    if args.aggregate:
        df = aggregate_run_metrics(logs_dir)
        if df.empty:
            logger.warning("No metrics.json files found under %s", logs_dir)
        else:
            out_csv = tables_dir / "main_metrics_image_level.csv"
            df.to_csv(out_csv, index=False)
            logger.info("Wrote %s", out_csv)

        runs = collect_run_rows(logs_dir)
        if runs.empty:
            logger.warning("No seed_N run directories found under %s", logs_dir)
            return 0
        expected_seeds = list(cfg.get("seeds", [0, 1, 2]))
        report_missing_runs(runs, expected_seeds)

        per_run_csv = tables_dir / "per_run_full.csv"
        runs.to_csv(per_run_csv, index=False)
        logger.info("Wrote %s", per_run_csv)

        main_csv = tables_dir / "main_results_by_model_fraction.csv"
        build_main_results_table(runs).to_csv(main_csv, index=False)
        logger.info("Wrote %s", main_csv)

        calib_csv = tables_dir / "calibration_by_model_fraction.csv"
        build_calibration_table(runs).to_csv(calib_csv, index=False)
        logger.info("Wrote %s", calib_csv)

        plot_label_efficiency(runs, figures_dir / "label_efficiency_auroc.png", "image_auroc")
        plot_label_efficiency(
            runs, figures_dir / "label_efficiency_patient_auroc.png", "patient_auroc"
        )
        logger.info("Wrote label-efficiency figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
