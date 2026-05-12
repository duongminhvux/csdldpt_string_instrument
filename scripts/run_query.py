import os
import sys
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.config import QUERY_ROOT, DB_CONFIG, TOP5_RESULTS_CSV
from src.database_manager import DatabaseManager
from src.feature_extraction import extract_all
from src.retrieval import rank_similar_files, print_top_results
from src.utils import list_audio_files, print_section, save_dicts_to_csv


TOP_K = 5


def choose_query_file() -> str:
    query_files = list_audio_files(str(QUERY_ROOT))

    if not query_files:
        raise FileNotFoundError(f"No .wav files found in: {QUERY_ROOT}")

    print_section("QUERY FILES")

    for idx, file_path in enumerate(query_files, start=1):
        print(f"{idx}. {Path(file_path).name}")

    while True:
        choice = input("Choose query file number: ").strip()

        if choice.isdigit():
            idx = int(choice)

            if 1 <= idx <= len(query_files):
                return query_files[idx - 1]

        print("Invalid choice.")


def run_query(query_file_path: str) -> list[dict]:
    query_result = extract_all(query_file_path)
    query_frame_vectors = query_result["feature_vectors"]

    db = DatabaseManager(**DB_CONFIG)
    db.connect()

    try:
        dataset_rows = db.fetch_dataset_features()

        top_results = rank_similar_files(
            query_frame_vectors=query_frame_vectors,
            dataset_rows=dataset_rows,
            top_k=TOP_K,
            normalize=True,
        )

        exported_rows = []

        for item in top_results:
            exported_rows.append(
                {
                    "query_file_name": Path(query_file_path).name,
                    "rank": item["rank_position"],
                    "matched_audio_id": item["matched_audio_id"],
                    "file_name": item["file_name"],
                    "file_path": item["file_path"],
                    "euclidean_distance": item["distance_score"],
                }
            )

        save_dicts_to_csv(exported_rows, str(TOP5_RESULTS_CSV))

        return top_results

    finally:
        db.close()


def main() -> None:
    print_section("RUN QUERY TOP 5")

    query_file_path = choose_query_file()

    print(f"Query file: {query_file_path}")

    top_results = run_query(query_file_path)

    print_top_results(top_results)

    print(f"\nResult CSV saved to: {TOP5_RESULTS_CSV}")


if __name__ == "__main__":
    main()