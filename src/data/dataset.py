"""PyTorch Dataset for CE-NBI images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import build_transforms
from src.utils.paths import project_root


def resolve_image_path(path: str, root: Path) -> str:
    """Resolve a stored image_path against the current project root."""
    p = Path(path)
    if not p.is_absolute():
        candidate = root / p
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Image not found: {candidate} (stored path: {path})")
    if p.exists():
        return str(p)
    posix = p.as_posix()
    marker = "data/interim/"
    idx = posix.find(marker)
    if idx != -1:
        remapped = root / posix[idx:]
        if remapped.exists():
            return str(remapped)
        raise FileNotFoundError(
            f"Image not found at stored path {p} or remapped path {remapped}"
        )
    raise FileNotFoundError(
        f"Image not found at stored path {p} and no '{marker}' segment to remap onto {root}"
    )


class CENBIDataset(Dataset):
    """Reads metadata + split CSVs and returns image, label, patient_id, image_id."""

    def __init__(
        self,
        metadata_csv: str | Path,
        split_csv: str | Path,
        split: str,
        image_size: int = 224,
        transform: Any | None = None,
    ) -> None:
        self.metadata = pd.read_csv(metadata_csv)
        splits = pd.read_csv(split_csv)
        merged = self.metadata.merge(
            splits[["image_id", "split"]],
            on="image_id",
            how="inner",
            suffixes=("", "_split"),
        )
        if "split" not in merged.columns and "split_split" in merged.columns:
            merged = merged.rename(columns={"split_split": "split"})
        self.df = merged.loc[merged["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split={split!r} in {split_csv}")

        root = project_root()
        self.df["image_path"] = self.df["image_path"].map(
            lambda p: resolve_image_path(str(p), root)
        )

        self.transform = transform or build_transforms(split=split, image_size=image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as im:
            image = im.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["binary_label"])
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "patient_id": str(row["patient_id"]),
            "image_id": str(row["image_id"]),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "label": torch.stack([b["label"] for b in batch], dim=0),
        "patient_id": [b["patient_id"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }
