from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "data" / "dataset"
DATA_ALL_DIR = BASE_DIR / "data" / "data_all"


def copy_all_wav():
    if not DATASET_DIR.exists():
        print(f"[ERROR] Không tìm thấy folder dataset: {DATASET_DIR}")
        return

    DATA_ALL_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for wav_file in DATASET_DIR.rglob("*.wav"):
        dest_file = DATA_ALL_DIR / wav_file.name

        if dest_file.exists():
            skipped += 1
            print(f"[SKIP] Đã tồn tại, không ghi đè: {dest_file.name}")
            continue

        shutil.copy2(wav_file, dest_file)
        copied += 1
        print(f"[COPY] {wav_file.name}")

    print("\nDONE")
    print(f"Đã copy: {copied} file wav")
    print(f"Bỏ qua vì trùng tên: {skipped} file")
    print(f"Folder đích: {DATA_ALL_DIR}")


if __name__ == "__main__":
    copy_all_wav()