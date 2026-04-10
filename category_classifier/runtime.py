"""Runtime and device helpers."""

from __future__ import annotations
from enum import StrEnum

import torch


class Device(StrEnum):
    """Runtime device identifier."""

    CPU = "cpu"
    MPS = "mps"
    AUTO = "auto"


def is_mps_available() -> bool:
    """Return True when Apple Metal backend is available."""
    return bool(torch.backends.mps.is_available())


def resolve_device(device: str) -> Device:
    """Resolve runtime device from cpu|mps|auto.
    
    Returns:
        Device: Resolved device enum value.
    """
    normalized = device.lower().strip()
    if normalized == "cpu":
        return Device.CPU
    if normalized == "mps":
        if not is_mps_available():
            raise ValueError("MPS requested but not available on this machine.")
        return Device.MPS
    if normalized == "auto":
        return Device.MPS if is_mps_available() else Device.CPU
    raise ValueError(f"Unsupported device '{device}'. Expected cpu|mps|auto.")
