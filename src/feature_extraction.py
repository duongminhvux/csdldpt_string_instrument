"""Feature extraction utilities for WAV files."""

from pathlib import Path


def extract_features(audio_path: str) -> dict:
    """Stub for feature extraction logic."""
    path = Path(audio_path)
    return {
        "file_name": path.name,
        "file_path": str(path),
        "instrument_label": path.parent.name if path.parent.name != "query" else None,
        "duration": None,
        "sample_rate": None,
        "rms_mean": None,
        "rms_std": None,
        "zcr_mean": None,
        "zcr_std": None,
        "centroid_mean": None,
        "centroid_std": None,
        "bandwidth_mean": None,
        "bandwidth_std": None,
        "rolloff_mean": None,
        "rolloff_std": None,
        "pitch_mean": None,
        "pitch_std": None,
        "mfcc": [None] * 13,
    }
