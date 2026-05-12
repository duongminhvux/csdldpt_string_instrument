import os
import sys
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.config import DATA_ALL_ROOT, QUERY_ROOT, RESULTS_DIR, DB_CONFIG, FEATURES_CSV
from src.database_manager import DatabaseManager
from src.feature_extraction import extract_all
from src.utils import (
    get_file_name,
    list_audio_files,
    print_section,
    save_dicts_to_csv,
)


OUTPUT_CSV = str(FEATURES_CSV)


def ensure_project_folders() -> None:
    for folder in [DATA_ALL_ROOT, QUERY_ROOT, RESULTS_DIR]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def ask_clear_old_data() -> bool:
    while True:
        choice = input("Clear old dataset data first? (y/n): ").strip().lower()

        if choice in {"y", "yes"}:
            return True

        if choice in {"n", "no"}:
            return False

        print("Please enter y or n.")


def build_dataset(clear_old_data: bool = True) -> dict:
    ensure_project_folders()

    audio_files = list_audio_files(str(DATA_ALL_ROOT))

    if not audio_files:
        return {
            "success": False,
            "message": f"No .wav files found in: {DATA_ALL_ROOT}",
            "audio_count": 0,
            "frame_count": 0,
            "csv_path": str(FEATURES_CSV),
        }

    db = DatabaseManager(**DB_CONFIG)
    db.connect()

    exported_rows = []

    try:
        if clear_old_data:
            db.clear_dataset_data()
            db.reset_auto_increment_for_clean_build()

        for idx, file_path in enumerate(audio_files, start=1):
            file_name = get_file_name(file_path)

            print(f"[{idx}/{len(audio_files)}] Processing: {file_name}")

            result = extract_all(file_path)
            metadata = result["metadata"]
            frame_features = result["frame_features"]

            audio_id = db.insert_audio_file(
                file_name=file_name,
                file_path=file_path,
                dataset_type="dataset",
                duration_seconds=metadata.get("duration_seconds"),
                sample_rate=metadata.get("sample_rate"),
                bit_depth=metadata.get("bit_depth"),
                channels=metadata.get("channels"),
                file_size_bytes=metadata.get("file_size_bytes"),
                file_format=metadata.get("file_format", "wav"),
            )

            db.insert_audio_features(audio_id, frame_features)

            for frame in frame_features:
                exported_rows.append(
                    {
                        "audio_id": audio_id,
                        "file_name": file_name,
                        "file_path": file_path,
                        **metadata,
                        **frame,
                    }
                )

        save_dicts_to_csv(exported_rows, OUTPUT_CSV)

        return {
            "success": True,
            "message": "Build dataset successfully.",
            "audio_count": len(audio_files),
            "frame_count": len(exported_rows),
            "csv_path": OUTPUT_CSV,
        }

    finally:
        db.close()


def main() -> None:
    print_section("BUILD DATASET FRAME FEATURES")
    print(f"Data all root: {DATA_ALL_ROOT}")

    clear_old_data = ask_clear_old_data()

    result = build_dataset(clear_old_data=clear_old_data)

    print_section("DONE")

    if result["success"]:
        print(f"Inserted {result['audio_count']} dataset files into MySQL.")
        print(f"Inserted {result['frame_count']} frame feature rows.")
        print(f"Feature backup saved to: {result['csv_path']}")
    else:
        print(result["message"])


if __name__ == "__main__":
    main()