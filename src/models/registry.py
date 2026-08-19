"""Classifier registry combining encoder + head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.encoders import build_encoder
from src.models.heads import build_head
from src.utils.paths import project_root, resolve_path


class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return self.head(feats)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats


def build_classifier(model_name: str, model_cfg: dict[str, Any] | None = None) -> Classifier:
    model_cfg = dict(model_cfg or {})
    if "weights_path" in model_cfg:
        model_cfg["weights_path"] = str(
            resolve_path(model_cfg["weights_path"], project_root())
        )
    encoder = build_encoder(model_name, model_cfg)
    head_kind = model_cfg.get("head", "linear")
    # Fine-tune ResNet uses linear head on pooled features
    if model_cfg.get("train_mode") == "finetune" and "head" not in model_cfg:
        head_kind = "linear"
    head = build_head(head_kind, in_dim=encoder.feat_dim)
    return Classifier(encoder, head)
