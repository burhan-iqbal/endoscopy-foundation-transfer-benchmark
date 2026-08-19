"""Device selection: CUDA > MPS (Apple Silicon) > CPU."""

from __future__ import annotations

import torch

from src.utils.logging import setup_logger

logger = setup_logger("device")


def get_device(prefer: str | None = None) -> torch.device:
    """
    Pick the best available accelerator.

    Order: CUDA (if available) → MPS (Apple Silicon) → CPU.
    Set prefer to 'cuda', 'mps', or 'cpu' to force a choice when available.
    """
    if prefer:
        prefer = prefer.lower()
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")
        logger.warning("Requested device %r unavailable; falling back to auto", prefer)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s", device)
    return device


def amp_enabled(device: torch.device, want_amp: bool = True) -> bool:
    """AMP GradScaler/autocast is only used on CUDA in this project."""
    return bool(want_amp) and device.type == "cuda"


def pin_memory_for(device: torch.device) -> bool:
    """pin_memory only helps CUDA host→device transfers."""
    return device.type == "cuda"
