"""Build a saliency grid figure (TP/FP/TN/FN) for a trained run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cenbi")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.saliency import _overlay, grad_cam, input_gradient_saliency
from src.data.dataset import resolve_image_path
from src.data.transforms import build_transforms
from src.models.registry import build_classifier
from src.training.train import label_tag
from src.utils.device import get_device
from src.utils.logging import setup_logger
from src.utils.paths import load_yaml, project_root, resolve_path

logger = setup_logger("make_saliency")

CELL_ORDER = ("TP", "FP", "TN", "FN")
GRAD_CAM_MODELS = {"resnet50", "imagenet_resnet50"}


def pick_examples(pred: pd.DataFrame, n_per_cell: int) -> dict[str, pd.DataFrame]:
    """Highest-confidence examples per confusion cell at threshold 0.5."""
    label = pred["label"].astype(int)
    positive = pred["prob"] >= 0.5
    masks = {
        "TP": (label == 1) & positive,
        "FP": (label == 0) & positive,
        "TN": (label == 0) & ~positive,
        "FN": (label == 1) & ~positive,
    }
    picked: dict[str, pd.DataFrame] = {}
    for cell in CELL_ORDER:
        sub = pred.loc[masks[cell]]
        if sub.empty:
            logger.warning("No %s examples at threshold 0.5; skipping row", cell)
            continue
        ascending = cell in ("TN", "FN")
        picked[cell] = sub.sort_values("prob", ascending=ascending).head(n_per_cell)
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label-frac", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--run-tag", default=None, help="Override run dir tag (default: label_{tag})")
    parser.add_argument("--n-per-cell", type=int, default=3)
    parser.add_argument("--out", default=None, help="Output PNG path (default: reports/figures/saliency/)")
    args = parser.parse_args()

    root = project_root()
    cfg = load_yaml(root / args.config)

    run_tag = args.run_tag or f"label_{label_tag(args.label_frac)}"
    run_dir = (
        resolve_path(cfg["paths"]["logs_dir"], root)
        / args.model
        / run_tag
        / f"seed_{args.seed}"
    )
    ckpt_path = run_dir / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")
    pred_path = run_dir / "predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}. Evaluate first.")

    model_cfg = dict(cfg["models"][args.model])
    device = get_device()
    model = build_classifier(args.model, model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    pred = pd.read_csv(pred_path)
    picked = pick_examples(pred, args.n_per_cell)
    if not picked:
        raise RuntimeError(f"No examples found in any confusion cell from {pred_path}")

    metadata = pd.read_csv(resolve_path(cfg["paths"]["metadata_csv"], root))
    path_by_id = dict(zip(metadata["image_id"].astype(str), metadata["image_path"].astype(str)))

    input_size = int(model_cfg.get("input_size", cfg["image"]["size"]))
    transform = build_transforms(split="test", image_size=input_size)
    use_grad_cam = args.model.lower() in GRAD_CAM_MODELS

    n_rows = len(picked)
    n_cols = max(len(rows) for rows in picked.values())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.2 * n_rows), squeeze=False)

    for r, (cell, rows) in enumerate(picked.items()):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c >= len(rows):
                continue
            row = rows.iloc[c]
            image_id = str(row["image_id"])
            if image_id not in path_by_id:
                logger.warning("image_id %s not in metadata; skipping", image_id)
                continue
            img_path = resolve_image_path(path_by_id[image_id], root)
            with Image.open(img_path) as im:
                pil = im.convert("RGB")
            tensor = transform(pil).unsqueeze(0).to(device)
            if use_grad_cam:
                heat = grad_cam(model, tensor)
            else:
                heat = input_gradient_saliency(model, tensor)
            raw = np.array(pil.resize(heat.shape[::-1]))
            ax.imshow(_overlay(raw, heat))
            true_name = "malignant" if int(row["label"]) == 1 else "benign"
            ax.set_title(f"p={float(row['prob']):.2f} true={true_name}", fontsize=9)
            if c == 0:
                ax.text(
                    -0.08, 0.5, cell, transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=12, fontweight="bold",
                )

    method = "Grad-CAM" if use_grad_cam else "input-gradient"
    fig.suptitle(f"{args.model} {run_tag} seed {args.seed} ({method})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if args.out:
        out_path = resolve_path(args.out, root)
    else:
        out_path = root / "reports" / "figures" / "saliency" / f"{args.model}_{run_tag}_seed_{args.seed}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved saliency grid to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
