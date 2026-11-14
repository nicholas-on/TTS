"""Helpers for running in Google Colab."""

from __future__ import annotations

import os
from pathlib import Path


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive() -> None:
    """Mount Google Drive at /content/drive (asks for permission once)."""
    if in_colab() and not Path("/content/drive/MyDrive").exists():
        from google.colab import drive
        drive.mount("/content/drive")


def gpu_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"


def disk_free_gb(path: str = "/content") -> float:
    st = os.statvfs(path if Path(path).exists() else "/")
    return st.f_bavail * st.f_frsize / 1e9
