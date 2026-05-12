import csv
import os
from pathlib import Path
from typing import Dict, List, Optional


SUPPORTED_AUDIO_EXTENSIONS = {".wav"}


def is_audio_file(file_name: str) -> bool:
    ext = os.path.splitext(file_name)[1].lower()
    return ext in SUPPORTED_AUDIO_EXTENSIONS


def get_file_name(file_path: str) -> str:
    return os.path.basename(file_path)


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


def find_file_in_folder(file_path: str) -> Optional[str]:
    path = Path(file_path)

    if path.exists() and path.is_file():
        return str(path)

    return None