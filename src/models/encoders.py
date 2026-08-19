"""Model encoders for ImageNet, DINOv2, and Endo-FM."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EncoderOutput(nn.Module):
    """Thin wrapper exposing forward -> features and optional logits path."""

    def __init__(self, backbone: nn.Module, feat_dim: int, freeze: bool = False) -> None:
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_resnet50(pretrained: bool = True, freeze: bool = False) -> EncoderOutput:
    import timm

    model = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
    feat_dim = model.num_features
    return EncoderOutput(model, feat_dim=feat_dim, freeze=freeze)


def build_dinov2_vits14(freeze: bool = True) -> EncoderOutput:
    """Load DINOv2 ViT-S/14 via torch.hub (facebookresearch/dinov2)."""
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    feat_dim = getattr(backbone, "embed_dim", 384)

    class _DinoWrap(nn.Module):
        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.m(x)

    return EncoderOutput(_DinoWrap(backbone), feat_dim=feat_dim, freeze=freeze)


def _extract_endo_fm_backbone_state(raw: Any) -> dict[str, torch.Tensor]:
    """
    Endo-FM checkpoints store DINO student/teacher TimeSformer weights.

    Prefer teacher.backbone.* (EMA) then student module.backbone.*.
    Drop temporal_* / time_embed keys for single-image ViT transfer.
    """
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected Endo-FM checkpoint type: {type(raw)}")

    candidate = None
    for key in ("teacher", "student", "model", "state_dict"):
        if key in raw and isinstance(raw[key], dict):
            candidate = raw[key]
            break
    if candidate is None:
        candidate = raw

    cleaned: dict[str, torch.Tensor] = {}
    for k, v in candidate.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        for prefix in (
            "module.backbone.",
            "backbone.",
            "module.",
            "encoder.",
        ):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
                break
        # Skip video-only temporal parameters
        if "temporal_" in nk or nk.startswith("time_embed"):
            continue
        cleaned[nk] = v
    if not cleaned:
        raise RuntimeError("No usable Endo-FM backbone tensors found in checkpoint")
    return cleaned


def build_endo_fm(
    weights_path: str | Path,
    freeze: bool = True,
    feat_dim: int = 768,
) -> EncoderOutput:
    """
    Load Endo-FM weights if present.

    Endo-FM is a video TimeSformer; for single-image CE-NBI we transfer the
    spatial ViT-B/16 trunk weights (skipping temporal attention) into a
    timm ViT-B/16 feature extractor.
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Endo-FM weights not found at {path}. "
            "See scripts/download_endo_fm_weights.md and place endo_fm.pth under "
            "models/external_weights/."
        )

    try:
        import timm

        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        # Official Endo-FM release may contain numpy scalars; trusted author checkpoint.
        raw = torch.load(path, map_location="cpu", weights_only=False)
        cleaned = _extract_endo_fm_backbone_state(raw)

        # pos_embed may include a temporal/extra token layout; keep only matching shapes
        model_sd = model.state_dict()
        filtered = {}
        skipped_shape = []
        for k, v in cleaned.items():
            if k not in model_sd:
                continue
            if model_sd[k].shape != v.shape:
                skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
            filtered[k] = v

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        model._endo_fm_missing = missing  # type: ignore[attr-defined]
        model._endo_fm_unexpected = unexpected  # type: ignore[attr-defined]
        model._endo_fm_loaded = len(filtered)  # type: ignore[attr-defined]
        model._endo_fm_skipped_shape = skipped_shape  # type: ignore[attr-defined]
        logger.info(
            "Endo-FM load: loaded=%d missing=%d unexpected=%d skipped_shape=%d",
            len(filtered),
            len(missing),
            len(unexpected),
            len(skipped_shape),
        )
        critical_skipped = [
            k
            for k, _, _ in skipped_shape
            if "pos_embed" in k or "patch_embed" in k or "cls_token" in k
        ]
        problems = []
        if len(filtered) < 140:
            problems.append(f"only {len(filtered)} keys mapped (expected >= 140)")
        if missing:
            problems.append(f"{len(missing)} missing keys, first 10: {list(missing)[:10]}")
        if critical_skipped:
            problems.append(
                f"{len(critical_skipped)} critical keys skipped on shape mismatch, "
                f"first 10: {critical_skipped[:10]}"
            )
        if problems:
            msg = (
                "Endo-FM load integrity check failed: "
                + "; ".join(problems)
                + f" (loaded={len(filtered)}, missing={len(missing)}, "
                f"unexpected={len(unexpected)}, skipped_shape={len(skipped_shape)})."
            )
            if os.environ.get("ENDO_FM_ALLOW_PARTIAL") == "1":
                logger.warning("ENDO_FM_ALLOW_PARTIAL=1 set, proceeding anyway: %s", msg)
            else:
                raise RuntimeError(
                    msg + " Set ENDO_FM_ALLOW_PARTIAL=1 to downgrade to a warning "
                    "(experimentation only)."
                )
        return EncoderOutput(model, feat_dim=model.num_features, freeze=freeze)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load Endo-FM weights from {path}: {exc}. "
            "Document this in reports/model_access_notes.md."
        ) from exc


def build_encoder(name: str, cfg: dict[str, Any] | None = None) -> EncoderOutput:
    cfg = cfg or {}
    name = name.lower()
    if name in {"resnet50", "imagenet_resnet50"}:
        freeze = cfg.get("train_mode") == "frozen_head"
        return build_resnet50(pretrained=bool(cfg.get("pretrained", True)), freeze=freeze)
    if name in {"dinov2_vits14", "dinov2"}:
        return build_dinov2_vits14(freeze=True)
    if name in {"endo_fm", "endofm"}:
        weights = cfg.get("weights_path", "models/external_weights/endo_fm.pth")
        return build_endo_fm(weights_path=weights, freeze=True)
    raise ValueError(f"Unknown encoder: {name}")
