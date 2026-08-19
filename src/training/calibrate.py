"""Temperature scaling calibration fitted on validation logits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.metrics import compute_binary_metrics
from src.analysis.plots import plot_reliability_diagram
from src.training.train import label_tag
from src.utils.logging import save_json, setup_logger
from src.utils.paths import ensure_dir, load_yaml, project_root, resolve_path

logger = setup_logger("calibrate")


class TemperatureScaler(nn.Module):
    """Scalar temperature T = exp(log T), so T is strictly positive.

    Guo et al. (2017) require T > 0: dividing logits by T is then a monotone
    rescaling and cannot invert rankings. An earlier unconstrained Parameter
    could be saved as T <= 0 while the fit itself used clamp(min=1e-4); applying
    that raw value flipped test AUROC. Stored H100 artefacts keep those original
    fits; this class is the corrected fitter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def is_valid_temperature(temperature: float) -> bool:
    return bool(np.isfinite(temperature) and temperature > 0.0)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not is_valid_temperature(temperature):
        raise ValueError(
            f"Temperature must be finite and positive (got {temperature}). "
            "A non-positive T inverts rankings and is not a valid calibration."
        )
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-(logits / temperature)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 500) -> float:
    device = torch.device("cpu")
    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS([scaler.log_temperature], lr=0.01, max_iter=max_iter)
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = criterion(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    temperature = float(scaler.temperature.detach().cpu().item())
    if not is_valid_temperature(temperature):
        raise RuntimeError(f"Temperature fit produced non-positive T={temperature}")
    return temperature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label-frac", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--split-csv", default=None, help="Unused here; accepted for symmetry with evaluate (requires --run-tag)")
    parser.add_argument("--run-tag", default=None, help="Override run dir tag (default: label_{tag})")
    parser.add_argument("--force", action="store_true", help="Re-calibrate even if calibration.json exists")
    args = parser.parse_args()

    if args.split_csv and not args.run_tag:
        parser.error("--run-tag is required when --split-csv is given")

    root = project_root()
    cfg = load_yaml(root / args.config)
    run_tag = args.run_tag or f"label_{label_tag(args.label_frac)}"
    run_dir = (
        resolve_path(cfg["paths"]["logs_dir"], root)
        / args.model
        / run_tag
        / f"seed_{args.seed}"
    )
    calibration_path = run_dir / "calibration.json"
    if calibration_path.exists() and not args.force:
        logger.info("Calibration already complete (%s). Skipping (use --force to re-run).", calibration_path)
        return 0

    val_path = run_dir / "predictions_val.csv"
    test_path = run_dir / "predictions.csv"
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Need {val_path} and {test_path}. Run training first."
        )

    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    temperature = fit_temperature(val["logit"].to_numpy(), val["label"].to_numpy())
    logger.info("Fitted temperature=%.4f", temperature)

    def apply_temp(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["prob_calibrated"] = apply_temperature(out["logit"].to_numpy(), temperature)
        return out

    val_c = apply_temp(val)
    test_c = apply_temp(test)
    test_c.to_csv(run_dir / "predictions_calibrated.csv", index=False)

    before = compute_binary_metrics(test["label"].to_numpy(), test["prob"].to_numpy())
    after = compute_binary_metrics(test["label"].to_numpy(), test_c["prob_calibrated"].to_numpy())

    figures = ensure_dir(resolve_path(cfg["paths"]["figures_dir"], root) / "calibration")
    plot_reliability_diagram(
        test["label"].to_numpy(),
        test["prob"].to_numpy(),
        figures / f"{args.model}_{run_tag}_seed_{args.seed}_before.png",
        title=f"{args.model} before TS",
    )
    plot_reliability_diagram(
        test["label"].to_numpy(),
        test_c["prob_calibrated"].to_numpy(),
        figures / f"{args.model}_{run_tag}_seed_{args.seed}_after.png",
        title=f"{args.model} after TS",
    )

    payload = {
        "model": args.model,
        "label_frac": args.label_frac,
        "seed": args.seed,
        "temperature": temperature,
        "valid_temperature": True,
        "before": before,
        "after": after,
    }
    save_json(calibration_path, payload)
    logger.info(
        "ECE before=%.4f after=%.4f | Brier before=%.4f after=%.4f",
        before["ece"],
        after["ece"],
        before["brier"],
        after["brier"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
