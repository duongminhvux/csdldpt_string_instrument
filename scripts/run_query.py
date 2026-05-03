import os
import shutil
import sys
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.config import DATA_ALL_ROOT, DB_CONFIG, QUERY_ROOT, TOP5_RESULTS_CSV, TOP5_WAV_DIR
from src.database_manager import DatabaseManager
from src.feature_extraction import extract_all
from src.retrieval import (
    print_instrument_votes,
    print_top_results,
    rank_similar_files,
    vote_instrument_by_nearest_frame,
)
from src.utils import (
    ensure_folder_exists,
    find_file_in_flat_folder,
    get_file_name,
    list_audio_files,
    print_section,
    save_dicts_to_csv,
)


TOP_K = 5


def choose_query_file() -> str:
    query_files = list_audio_files(str(QUERY_ROOT))

    if not query_files:
        raise FileNotFoundError(f"No .wav files found in: {QUERY_ROOT}")

    print_section("AVAILABLE QUERY FILES")

    for i, file_path in enumerate(query_files, start=1):
        print(f"{i}. {file_path}")

    while True:
        choice = input("\nChoose query file number: ").strip()

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        idx = int(choice)

        if 1 <= idx <= len(query_files):
            return query_files[idx - 1]

        print("Choice out of range.")


def copy_top_result_wavs(query_file_name: str, top_results: list[dict]) -> str:
    query_stem = Path(query_file_name).stem
    output_dir = TOP5_WAV_DIR / query_stem

    ensure_folder_exists(str(output_dir))

    for old_file in output_dir.glob("*.wav"):
        old_file.unlink()

    for item in top_results:
        matched_file_name = item["file_name"]
        src_path = find_file_in_flat_folder(matched_file_name, str(DATA_ALL_ROOT))

        if src_path is None:
            print(f"Warning: file not found in data_all, skip copy: {matched_file_name}")
            continue

        src = Path(src_path)
        dest_name = f"rank_{item['rank_position']}_{src.name}"
        dest = output_dir / dest_name

        shutil.copy2(src, dest)

    return str(output_dir)


def main() -> None:
    print_section("RUN AUDIO QUERY")
    print(f"Query root: {QUERY_ROOT}")
    print(f"Data all root: {DATA_ALL_ROOT}")

    query_file_path = choose_query_file()
    query_file_name = get_file_name(query_file_path)

    print(f"\nSelected query file: {query_file_name}")

    db = DatabaseManager(**DB_CONFIG)
    db.connect()

    try:
        query_result = extract_all(query_file_path)

        query_metadata = query_result["metadata"]
        query_frame_features = query_result["frame_features"]
        query_frame_vectors = query_result["feature_vectors"]

        query_audio_id = db.insert_audio_file(
            file_name=query_file_name,
            file_path=query_file_path,
            dataset_type="query",
            instrument_name=None,
            duration_seconds=query_metadata.get("duration_seconds"),
            sample_rate=query_metadata.get("sample_rate"),
            bit_depth=query_metadata.get("bit_depth"),
            channels=query_metadata.get("channels"),
            file_size_bytes=query_metadata.get("file_size_bytes"),
            file_format=query_metadata.get("file_format", "wav"),
        )

        db.insert_audio_features(query_audio_id, query_frame_features)

        dataset_rows = db.fetch_dataset_features()

        votes = vote_instrument_by_nearest_frame(
            query_frame_vectors=query_frame_vectors,
            dataset_rows=dataset_rows,
        )

        print_instrument_votes(votes)

        top_results = rank_similar_files(
            query_frame_vectors=query_frame_vectors,
            dataset_rows=dataset_rows,
            top_k=TOP_K,
            normalize=True,
        )

        print_top_results(top_results)

        copied_dir = copy_top_result_wavs(query_file_name, top_results)

        exported_rows = []

        for item in top_results:
            matched_file_name = item["file_name"]
            copied_source = find_file_in_flat_folder(matched_file_name, str(DATA_ALL_ROOT))

            exported_rows.append({
                "query_file_name": query_file_name,
                "matched_audio_id": item["matched_audio_id"],
                "matched_file_name": matched_file_name,
                "matched_file_path_in_db": item["file_path"],
                "matched_file_path_in_data_all": copied_source,
                "instrument_name": item.get("instrument_name"),
                "rank_position": item["rank_position"],
                "distance_score": item["distance_score"],
                "copied_wav_dir": copied_dir,
            })

        save_dicts_to_csv(exported_rows, str(TOP5_RESULTS_CSV))

        print_section("DONE")
        print(f"Query audio_id: {query_audio_id}")
        print(f"Query valid frames: {len(query_frame_features)}")
        print(f"CSV result saved to: {TOP5_RESULTS_CSV}")
        print(f"Top-{TOP_K} wav files copied from data_all to: {copied_dir}")

    finally:
        db.close()


if __name__ == "__main__":
    main()