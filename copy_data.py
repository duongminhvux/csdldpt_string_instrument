from pathlib import Path
import shutil
import re


BASE_DIR = Path(__file__).resolve().parent

DATA_ALL_DIR = BASE_DIR / "data" / "data_all"
DATASET_DIR = BASE_DIR / "data" / "dataset"


AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}


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

def copy_data_all_to_dataset():
    if not DATA_ALL_DIR.exists():
        print(f"[ERROR] Không tìm thấy folder data_all: {DATA_ALL_DIR}")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    invalid_name = 0

    for audio_file in DATA_ALL_DIR.iterdir():
        if not audio_file.is_file():
            continue

        if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

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

    print("\nDONE")
    print(f"Đã copy: {copied} file")
    print(f"Bỏ qua vì đã tồn tại: {skipped} file")
    print(f"Bỏ qua vì sai tên file: {invalid_name} file")
    print(f"Folder nguồn: {DATA_ALL_DIR}")
    print(f"Folder đích: {DATASET_DIR}")


if __name__ == "__main__":
    copy_data_all_to_dataset()