"""Train a CE-NBI classifier under the shared benchmark protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.metrics import aggregate_patient_level, compute_binary_metrics
from src.data.dataset import CENBIDataset, collate_batch
from src.models.registry import build_classifier
from src.utils.device import amp_enabled, get_device, pin_memory_for
from src.utils.logging import save_json, save_yaml, setup_logger
from src.utils.paths import ensure_dir, load_yaml, project_root, resolve_path
from src.utils.seed import set_seed

logger = setup_logger("train")


def label_tag(frac: float) -> str:
    return f"{int(round(frac * 100)):02d}"


def resolve_split_csv(splits_dir: Path, label_frac: float, seed: int) -> Path:
    tag = label_tag(label_frac)
    path = splits_dir / f"low_label_{tag}_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file {path}. Run: python -m src.data.make_splits"
        )
    return path


def make_loaders(
    metadata_csv: Path,
    split_csv: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device | None = None,
) -> dict[str, DataLoader]:
    device = device or get_device()
    loaders = {}
    for split in ("train", "val", "test"):
        ds = CENBIDataset(
            metadata_csv=metadata_csv,
            split_csv=split_csv,
            split=split,
            image_size=image_size,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_batch,
            pin_memory=pin_memory_for(device),
            persistent_workers=(num_workers > 0),
        )
    return loaders


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for i, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch["label"].numpy()
        for j in range(len(probs)):
            rows.append(
                {
                    "image_id": batch["image_id"][j],
                    "patient_id": batch["patient_id"][j],
                    "label": int(labels[j]),
                    "logit": float(logits[j].detach().cpu()),
                    "prob": float(probs[j]),
                }
            )
        # Avoid MPS memory buildup on long val/test passes
        if device.type == "mps" and (i + 1) % 20 == 0:
            torch.mps.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()
    return pd.DataFrame(rows)


def evaluate_split(df: pd.DataFrame) -> dict:
    y_true = df["label"].to_numpy()
    y_prob = df["prob"].to_numpy()
    image_metrics = compute_binary_metrics(y_true, y_prob)
    yt_p, yp_p, _ = aggregate_patient_level(y_true, y_prob, df["patient_id"].tolist())
    patient_metrics = compute_binary_metrics(yt_p, yp_p)
    return {"image": image_metrics, "patient": patient_metrics}


def class_weights_from_labels(labels: np.ndarray) -> torch.Tensor:
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    # pos_weight for BCEWithLogitsLoss
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def _package_versions() -> dict[str, str]:
    from importlib import metadata

    versions = {}
    for pkg in ("torch", "torchvision", "timm", "numpy"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """Write via a temp file so a crash mid-save does not corrupt the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def build_full_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_auroc: float,
    wait: int,
    curve_rows: list[dict[str, Any]],
    model_name: str,
    label_frac: float,
    seed: int,
    finished: bool = False,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "label_frac": label_frac,
        "seed": seed,
        "epoch": epoch,
        "best_auroc": best_auroc,
        "wait": wait,
        "curve_rows": curve_rows,
        "finished": finished,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "val_auroc": best_auroc,
    }


def train_one_run(
    model_name: str,
    label_frac: float,
    seed: int,
    config_path: str = "configs/experiments.yaml",
    max_epochs_override: int | None = None,
    resume: bool = True,
    save_every_epoch: bool = False,
    split_csv_path: str | None = None,
    run_tag: str | None = None,
) -> Path:
    root = project_root()
    cfg = load_yaml(root / config_path)
    set_seed(seed)

    model_cfg = dict(cfg["models"][model_name])
    train_cfg = cfg["training"]
    paths = cfg["paths"]

    metadata_csv = resolve_path(paths["metadata_csv"], root)
    splits_dir = resolve_path(paths["splits_dir"], root)
    if split_csv_path is not None:
        if not run_tag:
            raise ValueError("--run-tag is required when --split-csv is given")
        split_csv = resolve_path(split_csv_path, root)
        if not split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
        tag = run_tag
    else:
        split_csv = resolve_split_csv(splits_dir, label_frac, seed)
        tag = f"label_{label_tag(label_frac)}"

    run_dir = ensure_dir(
        resolve_path(paths["logs_dir"], root)
        / model_name
        / tag
        / f"seed_{seed}"
    )
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and resume:
        logger.info("Run already complete (%s). Skipping train.", metrics_path)
        return run_dir

    device = get_device()
    logger.info("Device: %s | model=%s | label_frac=%s | seed=%s", device, model_name, label_frac, seed)

    if model_name == "endo_fm":
        weights = resolve_path(model_cfg.get("weights_path", paths["external_weights_dir"] + "/endo_fm.pth"), root)
        if not Path(weights).exists():
            notes = resolve_path("reports/model_access_notes.md", root)
            notes.write_text(
                "# Model access notes\n\n"
                f"Endo-FM weights missing at `{weights}`.\n\n"
                "See `scripts/download_endo_fm_weights.md`. "
                "ResNet-50 and DINOv2 can still be trained under the full protocol.\n",
                encoding="utf-8",
            )
            raise FileNotFoundError(
                f"Endo-FM weights not found at {weights}. Wrote {notes}."
            )

    model = build_classifier(model_name, model_cfg).to(device)

    run_config = {
        "model": model_name,
        "label_frac": label_frac,
        "seed": seed,
        "model_cfg": model_cfg,
        "training": train_cfg,
        "split_csv": str(split_csv),
        "run_tag": tag,
        "versions": _package_versions(),
    }
    encoder = getattr(model, "encoder", None)
    if model_name == "endo_fm" and hasattr(encoder, "_endo_fm_loaded"):
        run_config["endo_fm_load"] = {
            "loaded": int(encoder._endo_fm_loaded),
            "missing": len(encoder._endo_fm_missing),
            "unexpected": len(encoder._endo_fm_unexpected),
            "skipped_shape": len(encoder._endo_fm_skipped_shape),
        }
    save_yaml(run_dir / "config.yaml", run_config)

    image_size = int(model_cfg.get("input_size", cfg["image"]["size"]))
    # macOS MPS + forked DataLoader workers is unreliable; keep workers on CUDA/CPU only.
    num_workers = int(train_cfg.get("num_workers", 4))
    if device.type == "mps":
        num_workers = 0
    elif device.type == "cuda":
        num_workers = 8
    loaders = make_loaders(
        metadata_csv=metadata_csv,
        split_csv=split_csv,
        image_size=image_size,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=num_workers,
        device=device,
    )

    # Parameter groups: higher LR for head when frozen encoder
    head_params = list(model.head.parameters())
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    param_groups = [{"params": head_params, "lr": float(train_cfg["lr_head"])}]
    if enc_params:
        param_groups.append({"params": enc_params, "lr": float(train_cfg["lr_finetune"])})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(train_cfg["weight_decay"]))
    max_epochs = int(max_epochs_override or train_cfg["max_epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 0))
    if 0 < warmup_epochs < max_epochs:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max_epochs - warmup_epochs
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    pos_weight = None
    if train_cfg.get("use_class_weights", True):
        train_labels = loaders["train"].dataset.df["binary_label"].to_numpy()
        pos_weight = class_weights_from_labels(train_labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc = -1.0
    best_path = run_dir / "checkpoint_best.pt"
    last_path = run_dir / "checkpoint_last.pt"
    patience = int(train_cfg["early_stopping_patience"])
    wait = 0
    curve_rows: list[dict[str, Any]] = []
    start_epoch = 1

    use_amp = amp_enabled(device, bool(train_cfg.get("amp", True)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and use_amp:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not restore AMP scaler: %s", exc)
        best_auroc = float(ckpt.get("best_auroc", ckpt.get("val_auroc", -1.0)))
        wait = int(ckpt.get("wait", 0))
        curve_rows = list(ckpt.get("curve_rows", []))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info(
            "Resumed from %s at epoch %d (best_auroc=%.4f, wait=%d)",
            last_path,
            start_epoch,
            best_auroc,
            wait,
        )
        if start_epoch > max_epochs:
            logger.info("Checkpoint already past max_epochs; jumping to final eval.")
            start_epoch = max_epochs + 1
        if ckpt.get("finished", False) or wait >= patience:
            logger.info(
                "Training already finished (finished=%s, wait=%d/%d); jumping to final eval.",
                ckpt.get("finished", False),
                wait,
                patience,
            )
            start_epoch = max_epochs + 1

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_df = predict(model, loaders["val"], device)
        val_metrics = evaluate_split(val_df)
        val_auroc = val_metrics["image"].get("auroc", float("nan"))
        curve_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "val_auroc": val_auroc,
                "val_f1": val_metrics["image"].get("f1"),
            }
        )
        logger.info("Epoch %d | loss=%.4f | val_auroc=%.4f", epoch, np.mean(losses), val_auroc)

        improved = val_auroc > best_auroc if not np.isnan(val_auroc) else False
        if improved:
            best_auroc = val_auroc
            wait = 0
            _atomic_torch_save(
                {
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auroc": best_auroc,
                    "best_auroc": best_auroc,
                    "label_frac": label_frac,
                    "seed": seed,
                },
                best_path,
            )
        else:
            wait += 1

        # Always overwrite last checkpoint (full resume state)
        _atomic_torch_save(
            build_full_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_auroc=best_auroc,
                wait=wait,
                curve_rows=curve_rows,
                model_name=model_name,
                label_frac=label_frac,
                seed=seed,
                finished=False,
            ),
            last_path,
        )
        # Optional per-epoch archive (disk-heavy; off by default)
        if save_every_epoch:
            _atomic_torch_save(
                {
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_auroc": val_auroc,
                    "label_frac": label_frac,
                    "seed": seed,
                },
                run_dir / f"checkpoint_epoch_{epoch:03d}.pt",
            )

        pd.DataFrame(curve_rows).to_csv(run_dir / "training_curve.csv", index=False)

        if wait >= patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    pd.DataFrame(curve_rows).to_csv(run_dir / "training_curve.csv", index=False)

    # Load best and evaluate
    if not best_path.exists():
        raise FileNotFoundError(
            f"No best checkpoint at {best_path}. Training may have failed before first improvement."
        )
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_df = predict(model, loaders["test"], device)
    val_df = predict(model, loaders["val"], device)
    test_metrics = evaluate_split(test_df)
    val_metrics = evaluate_split(val_df)

    test_df.to_csv(run_dir / "predictions.csv", index=False)
    val_df.to_csv(run_dir / "predictions_val.csv", index=False)
    pred_dir = ensure_dir(resolve_path(paths["predictions_dir"], root))
    test_df.to_csv(
        pred_dir / f"{model_name}_{tag}_seed_{seed}.csv",
        index=False,
    )

    payload = {
        "model": model_name,
        "label_frac": label_frac,
        "seed": seed,
        "best_val_auroc": best_auroc,
        "val_image": val_metrics["image"],
        "val_patient": val_metrics["patient"],
        "test_image": test_metrics["image"],
        "test_patient": test_metrics["patient"],
    }
    save_json(run_dir / "metrics.json", payload)

    # Mark last checkpoint finished so resume knows this run completed training
    if last_path.exists():
        last_ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        last_ckpt["finished"] = True
        _atomic_torch_save(last_ckpt, last_path)

    logger.info("Finished. Best val AUROC=%.4f | test AUROC=%.4f", best_auroc, test_metrics["image"].get("auroc"))
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["resnet50", "dinov2_vits14", "endo_fm"])
    parser.add_argument("--label-frac", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override for smoke tests")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint_last.pt if present (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing checkpoints and train from scratch",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Also keep checkpoint_epoch_XXX.pt (uses more disk)",
    )
    parser.add_argument(
        "--split-csv",
        default=None,
        help="Optional split CSV (image_id,patient_id,binary_label,split) to use instead of the default low-label split",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Run directory tag under reports/logs/{model}/ (required with --split-csv)",
    )
    args = parser.parse_args()
    if args.split_csv and not args.run_tag:
        parser.error("--run-tag is required when --split-csv is given")
    train_one_run(
        model_name=args.model,
        label_frac=args.label_frac,
        seed=args.seed,
        config_path=args.config,
        max_epochs_override=args.max_epochs,
        resume=args.resume,
        save_every_epoch=args.save_every_epoch,
        split_csv_path=args.split_csv,
        run_tag=args.run_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
