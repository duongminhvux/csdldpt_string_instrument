import csv
import os
from typing import Dict, List, Optional


SUPPORTED_AUDIO_EXTENSIONS = {".wav"}


def is_audio_file(file_name: str) -> bool:
    ext = os.path.splitext(file_name)[1].lower()
    return ext in SUPPORTED_AUDIO_EXTENSIONS


def get_file_name(file_path: str) -> str:
    return os.path.basename(file_path)


def get_instrument_name_from_path(file_path: str, dataset_root: str = "data/dataset") -> Optional[str]:
    normalized_file_path = os.path.normpath(file_path)
    normalized_dataset_root = os.path.normpath(dataset_root)

    rel_path = os.path.relpath(normalized_file_path, normalized_dataset_root)
    parts = rel_path.split(os.sep)

    if len(parts) < 2:
        return None

    return parts[0].lower()


def list_audio_files(root_folder: str) -> List[str]:
    audio_files = []

    for root, _, files in os.walk(root_folder):
        for file_name in files:
            if is_audio_file(file_name):
                audio_files.append(os.path.join(root, file_name))

    audio_files.sort()
    return audio_files


def ensure_folder_exists(folder_path: str) -> None:
    if folder_path:
        os.makedirs(folder_path, exist_ok=True)


def save_dicts_to_csv(rows: List[Dict], output_file: str) -> None:
    if not rows:
        return

    ensure_folder_exists(os.path.dirname(output_file) if os.path.dirname(output_file) else ".")

    fieldnames = list(rows[0].keys())

    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
