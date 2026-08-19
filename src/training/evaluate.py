"""Evaluate a saved checkpoint and optionally write bootstrap CIs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.bootstrap import bootstrap_metric_cis
from src.analysis.metrics import aggregate_patient_level, compute_binary_metrics
from src.data.dataset import CENBIDataset, collate_batch
from src.models.registry import build_classifier
from src.training.train import label_tag, predict, resolve_split_csv
from src.utils.device import get_device
from src.utils.logging import save_json, setup_logger
from src.utils.paths import load_yaml, project_root, resolve_path
from src.utils.seed import set_seed

logger = setup_logger("evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label-frac", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--split-csv", default=None, help="Override split CSV (requires --run-tag)")
    parser.add_argument("--run-tag", default=None, help="Override run dir tag (default: label_{tag})")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if metrics_eval.json exists")
    args = parser.parse_args()

    if args.split_csv and not args.run_tag:
        parser.error("--run-tag is required when --split-csv is given")

    root = project_root()
    cfg = load_yaml(root / args.config)
    set_seed(args.seed)

    run_tag = args.run_tag or f"label_{label_tag(args.label_frac)}"
    run_dir = (
        resolve_path(cfg["paths"]["logs_dir"], root)
        / args.model
        / run_tag
        / f"seed_{args.seed}"
    )
    metrics_eval_path = run_dir / "metrics_eval.json"
    if metrics_eval_path.exists() and not args.force:
        logger.info("Eval already complete (%s). Skipping (use --force to re-run).", metrics_eval_path)
        return 0

    ckpt_path = run_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")

    model_cfg = dict(cfg["models"][args.model])
    device = get_device()
    model = build_classifier(args.model, model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    if args.split_csv:
        split_csv = resolve_path(args.split_csv, root)
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
    else:
        split_csv = resolve_split_csv(
            resolve_path(cfg["paths"]["splits_dir"], root), args.label_frac, args.seed
        )
    ds = CENBIDataset(
        metadata_csv=resolve_path(cfg["paths"]["metadata_csv"], root),
        split_csv=split_csv,
        split="test",
        image_size=int(model_cfg.get("input_size", cfg["image"]["size"])),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        collate_fn=collate_batch,
    )
    pred = predict(model, loader, device)
    pred.to_csv(run_dir / "predictions.csv", index=False)

    y_true = pred["label"].to_numpy()
    y_prob = pred["prob"].to_numpy()
    image_metrics = compute_binary_metrics(y_true, y_prob)
    yt_p, yp_p, _ = aggregate_patient_level(y_true, y_prob, pred["patient_id"].tolist())
    patient_metrics = compute_binary_metrics(yt_p, yp_p)

    out = {
        "model": args.model,
        "label_frac": args.label_frac,
        "seed": args.seed,
        "test_image": image_metrics,
        "test_patient": patient_metrics,
    }
    if args.bootstrap:
        n = int(cfg["bootstrap"]["n_resamples"])
        ci = float(cfg["bootstrap"]["ci"])
        out["bootstrap_image"] = bootstrap_metric_cis(y_true, y_prob, n_resamples=n, ci=ci, seed=args.seed)
        out["bootstrap_patient"] = bootstrap_metric_cis(yt_p, yp_p, n_resamples=n, ci=ci, seed=args.seed)

    save_json(run_dir / "metrics_eval.json", out)
    # Merge into metrics.json if present
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        import json

        with open(metrics_path, encoding="utf-8") as f:
            base = json.load(f)
        base.update({"test_image": image_metrics, "test_patient": patient_metrics})
        if args.bootstrap:
            base["bootstrap_image"] = out["bootstrap_image"]
            base["bootstrap_patient"] = out["bootstrap_patient"]
        save_json(metrics_path, base)

    logger.info("Test AUROC (image)=%.4f (patient)=%.4f", image_metrics.get("auroc"), patient_metrics.get("auroc"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
