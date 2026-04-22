"""Build feature dataset from files in data/dataset/."""

from pathlib import Path


def main():
    dataset_dir = Path("data/dataset")
    print(f"Dataset folder: {dataset_dir.resolve()}")
    print("Add WAV files, then connect feature extraction and MySQL insert here.")


if __name__ == "__main__":
    main()
