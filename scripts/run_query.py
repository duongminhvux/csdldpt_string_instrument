"""Run top-5 search for files in data/query/."""

from pathlib import Path


def main():
    query_dir = Path("data/query")
    print(f"Query folder: {query_dir.resolve()}")
    print("Add query WAV files, then connect feature extraction and retrieval here.")


if __name__ == "__main__":
    main()
