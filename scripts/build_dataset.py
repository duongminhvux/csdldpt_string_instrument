import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.config import DATASET_ROOT, DB_CONFIG, FEATURES_CSV
from src.database_manager import DatabaseManager
from src.feature_extraction import extract_all
from src.utils import (
    get_file_name,
    get_instrument_name_from_path,
    list_audio_files,
    print_section,
    save_dicts_to_csv,
)


OUTPUT_CSV = str(FEATURES_CSV)


def ask_clear_old_data() -> bool:
    while True:
        choice = input("Clear old dataset data first? (y/n): ").strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Please enter y or n.")


def main() -> None:
    print_section("BUILD DATASET FEATURES")
    print(f"Dataset root: {DATASET_ROOT}")

    audio_files = list_audio_files(str(DATASET_ROOT))
    if not audio_files:
        print(f"No .wav files found in: {DATASET_ROOT}")
        return

    db = DatabaseManager(**DB_CONFIG)
    db.connect()

    try:
        if ask_clear_old_data():
            print_section("CLEARING OLD DATA")
            db.clear_dataset_data()
            db.reset_auto_increment_for_clean_build()

        exported_rows = []

        for idx, file_path in enumerate(audio_files, start=1):
            file_name = get_file_name(file_path)
            instrument_name = get_instrument_name_from_path(file_path, str(DATASET_ROOT))

            print(f"[{idx}/{len(audio_files)}] Processing: {file_name} | instrument={instrument_name}")

            result = extract_all(file_path)
            metadata = result["metadata"]
            features = result["features"]

            audio_id = db.insert_audio_file(
                file_name=file_name,
                file_path=file_path,
                dataset_type="dataset",
                instrument_name=instrument_name,
                duration_seconds=metadata.get("duration_seconds"),
                sample_rate=metadata.get("sample_rate"),
                bit_depth=metadata.get("bit_depth"),
                channels=metadata.get("channels"),
                file_size_bytes=metadata.get("file_size_bytes"),
                file_format=metadata.get("file_format", "wav"),
            )

            db.insert_audio_features(audio_id, features)

            row = {
                "audio_id": audio_id,
                "file_name": file_name,
                "file_path": file_path,
                "instrument_name": instrument_name,
                **metadata,
                **features,
            }
            exported_rows.append(row)

        save_dicts_to_csv(exported_rows, OUTPUT_CSV)

        print_section("DONE")
        print(f"Inserted {len(audio_files)} dataset files into MySQL.")
        print(f"Feature backup saved to: {OUTPUT_CSV}")

    finally:
        db.close()


if __name__ == "__main__":
    main()