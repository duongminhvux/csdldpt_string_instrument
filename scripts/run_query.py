import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database_manager import DatabaseManager
from src.feature_extraction import extract_all
from src.retrieval import print_top_results, rank_similar_files
from src.utils import get_file_name, list_audio_files, print_section, save_dicts_to_csv


QUERY_ROOT = "data/query"
OUTPUT_CSV = "results/top5_results.csv"

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "database": "string_instrument_search",
}

SIMILARITY_METHOD = "cosine"   # "cosine" hoặc "euclidean"
NORMALIZE_METHOD = "zscore"    # "zscore", "minmax", hoặc None
TOP_K = 5


def choose_query_file() -> str:
    query_files = list_audio_files(QUERY_ROOT)

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


def main() -> None:
    print_section("RUN AUDIO QUERY")

    query_file_path = choose_query_file()
    query_file_name = get_file_name(query_file_path)

    print(f"\nSelected query file: {query_file_name}")

    db = DatabaseManager(**DB_CONFIG)
    db.connect()

    try:
        query_result = extract_all(query_file_path)
        query_metadata = query_result["metadata"]
        query_features = query_result["features"]
        query_vector = query_result["feature_vector"]

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

        db.insert_audio_features(query_audio_id, query_features)

        dataset_rows = db.fetch_dataset_features()
        top_results = rank_similar_files(
            query_vector=query_vector,
            dataset_rows=dataset_rows,
            method=SIMILARITY_METHOD,
            normalize=NORMALIZE_METHOD,
            top_k=TOP_K,
        )

        print_top_results(top_results)

        query_id = db.insert_search_query(
            query_audio_id=query_audio_id,
            similarity_method=SIMILARITY_METHOD,
            top_k=TOP_K,
            notes=f"normalize={NORMALIZE_METHOD}",
        )

        db.insert_search_results(query_id, top_results)

        exported_rows = []
        for item in top_results:
            exported_rows.append({
                "query_id": query_id,
                "query_file_name": query_file_name,
                "matched_audio_id": item["matched_audio_id"],
                "matched_file_name": item["file_name"],
                "matched_file_path": item["file_path"],
                "instrument_name": item.get("instrument_name"),
                "rank_position": item["rank_position"],
                "similarity_score": item["similarity_score"],
                "distance_score": item["distance_score"],
            })

        save_dicts_to_csv(exported_rows, OUTPUT_CSV)

        print_section("DONE")
        print(f"Top-{TOP_K} results saved to database.")
        print(f"CSV result saved to: {OUTPUT_CSV}")

    finally:
        db.close()


if __name__ == "__main__":
    main()