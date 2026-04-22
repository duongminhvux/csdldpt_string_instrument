"""Shared helper functions."""

from pathlib import Path


def list_wav_files(root_dir: str):
    root = Path(root_dir)
    return sorted(str(p) for p in root.rglob("*.wav"))
