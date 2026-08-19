"""Saliency / Grad-CAM helpers for qualitative explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    # Prefer last conv in ResNet-style encoders
    encoder = getattr(model, "encoder", model)
    backbone = getattr(encoder, "backbone", encoder)
    candidates = []
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return candidates[-1][1]


@torch.no_grad()
def _overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    heat = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colored = cm.jet(heat)[..., :3]
    base = image.astype(np.float32) / 255.0
    out = (1 - alpha) * base + alpha * colored
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def grad_cam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: torch.nn.Module | None = None,
) -> np.ndarray:
    """
    Compute Grad-CAM for a single image tensor [1,3,H,W].
    Returns heatmap [H,W] in [0,1].
    """
    model.eval()
    target_layer = target_layer or _find_target_layer(model)
    activations = {}
    gradients = {}

    def fwd_hook(_m, _i, o):
        activations["value"] = o

    def bwd_hook(_m, _gi, go):
        gradients["value"] = go[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        image_tensor = image_tensor.detach().requires_grad_(True)
        logits = model(image_tensor)
        score = logits if logits.ndim == 0 else logits.squeeze()
        model.zero_grad(set_to_none=True)
        score.backward()
        acts = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam
    finally:
        h1.remove()
        h2.remove()


def input_gradient_saliency(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    blur_sigma: float = 2.0,
) -> np.ndarray:
    """
    Input-gradient saliency for a single image tensor [1,3,H,W].

    |d logit / d input|, max over channels, light Gaussian blur, normalised.
    Architecture-agnostic; used for ViT backbones where Grad-CAM does not apply.
    Returns heatmap [H,W] in [0,1].
    """
    from torchvision.transforms import functional as TF

    model.eval()
    image_tensor = image_tensor.detach().requires_grad_(True)
    logits = model(image_tensor)
    score = logits if logits.ndim == 0 else logits.squeeze()
    model.zero_grad(set_to_none=True)
    score.backward()
    sal = image_tensor.grad.detach().abs().amax(dim=1, keepdim=True).cpu()
    if blur_sigma > 0:
        kernel = int(2 * round(3 * blur_sigma) + 1)
        sal = TF.gaussian_blur(sal, kernel_size=[kernel, kernel], sigma=blur_sigma)
    sal = sal.squeeze().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal


def save_saliency_example(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    raw_image_path: str | Path,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cam = grad_cam(model, image_tensor)
    raw = np.array(Image.open(raw_image_path).convert("RGB").resize(cam.shape[::-1]))
    overlay = _overlay(raw, cam)
    Image.fromarray(overlay).save(out_path)
    return out_path


def attention_rollout_placeholder(*_args: Any, **_kwargs: Any) -> np.ndarray:
    """
    Placeholder for ViT attention rollout.

    Full attention-map extraction depends on the specific ViT implementation.
    Use Grad-CAM on CNN baselines first; extend for DINOv2/Endo-FM as needed.
    """
    raise NotImplementedError(
        "Attention rollout for ViT backbones is not implemented yet; "
        "use Captum Integrated Gradients or extend this helper."
    )
