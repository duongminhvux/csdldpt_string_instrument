from pathlib import Path
import shutil
import re


BASE_DIR = Path(__file__).resolve().parent

DATA_ALL_DIR = BASE_DIR / "data" / "data_all"
DATASET_DIR = BASE_DIR / "data" / "dataset"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


def ensure_data_folders() -> None:
    DATA_ALL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def get_instrument_name(file_path: Path) -> str | None:
    """
    Lấy tên nhạc cụ từ tên file.

    Nhận các dạng:
        violin-001.wav      -> violin
        violin_001.wav      -> violin
        dan_tranh_01.wav    -> dan_tranh
        dan_tranh-01.wav    -> dan_tranh

    Quy tắc:
        Nếu tên file kết thúc bằng _số hoặc -số
        thì bỏ phần số cuối, phần còn lại là tên nhạc cụ.
    """
    stem = file_path.stem.strip().lower()

    parts = re.split(r"[-_]", stem)

    if len(parts) < 2:
        return None

    last_part = parts[-1]

    if not last_part.isdigit():
        return None

    instrument_name = "_".join(parts[:-1]).strip()

    if not instrument_name:
        return None

    return instrument_name


def clear_dataset_folder() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def copy_data_all_to_dataset(clear_old_dataset: bool = True) -> dict:
    ensure_data_folders()

    if clear_old_dataset:
        clear_dataset_folder()

    copied = 0
    skipped = 0
    invalid_name = 0

    audio_files = [
        item
        for item in DATA_ALL_DIR.iterdir()
        if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS
    ]

    if not audio_files:
        message = f"Không có file audio trong folder: {DATA_ALL_DIR}"
        print(f"[WARNING] {message}")
        return {
            "success": False,
            "message": message,
            "copied": copied,
            "skipped": skipped,
            "invalid_name": invalid_name,
            "data_all_dir": str(DATA_ALL_DIR),
            "dataset_dir": str(DATASET_DIR),
        }

    for audio_file in audio_files:
        instrument_name = get_instrument_name(audio_file)

        if instrument_name is None:
            invalid_name += 1
            print(f"[SKIP] Tên file không đúng dạng 'nhaccu-so' hoặc 'nhaccu_so': {audio_file.name}")
            continue

        instrument_dir = DATASET_DIR / instrument_name
        instrument_dir.mkdir(parents=True, exist_ok=True)

        dest_file = instrument_dir / audio_file.name

        if dest_file.exists():
            skipped += 1
            print(f"[SKIP] Đã tồn tại: {dest_file}")
            continue

        shutil.copy2(audio_file, dest_file)
        copied += 1
        print(f"[COPY] {audio_file.name} -> dataset/{instrument_name}/")

    message = "Copy data_all sang dataset xong."

    print("\nDONE")
    print(f"Đã copy: {copied} file")
    print(f"Bỏ qua vì đã tồn tại: {skipped} file")
    print(f"Bỏ qua vì sai tên file: {invalid_name} file")
    print(f"Folder nguồn: {DATA_ALL_DIR}")
    print(f"Folder đích: {DATASET_DIR}")

    return {
        "success": True,
        "message": message,
        "copied": copied,
        "skipped": skipped,
        "invalid_name": invalid_name,
        "data_all_dir": str(DATA_ALL_DIR),
        "dataset_dir": str(DATASET_DIR),
    }


if __name__ == "__main__":
    copy_data_all_to_dataset(clear_old_dataset=True)