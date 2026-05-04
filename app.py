import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from copy_data import copy_data_all_to_dataset
from scripts.build_dataset import build_dataset
from src.config import DATA_ALL_ROOT, DATASET_ROOT, DB_CONFIG, QUERY_ROOT, RESULTS_DIR, TOP5_RESULTS_CSV, TOP5_WAV_DIR
from src.database_manager import DatabaseManager
from src.feature_extraction import extract_all
from src.retrieval import rank_similar_files, vote_instrument_by_nearest_frame
from src.utils import ensure_folder_exists, find_file_in_flat_folder, save_dicts_to_csv


TOP_K = 5

def ensure_project_folders() -> None:
    for folder in [
        DATA_ALL_ROOT,
        DATASET_ROOT,
        QUERY_ROOT,
        RESULTS_DIR,
        TOP5_WAV_DIR,
    ]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def save_uploaded_query_file(uploaded_file) -> Path:
    ensure_folder_exists(str(QUERY_ROOT))

    safe_name = uploaded_file.name.replace(" ", "_")
    query_path = QUERY_ROOT / safe_name

    with open(query_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return query_path


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
            item["copied_audio_path"] = None
            continue

        src = Path(src_path)
        dest_name = f"rank_{item['rank_position']}_{src.name}"
        dest = output_dir / dest_name

        shutil.copy2(src, dest)
        item["copied_audio_path"] = str(dest)

    return str(output_dir)


def run_query(query_file_path: Path) -> tuple[list[dict], dict, dict]:
    query_result = extract_all(str(query_file_path))

    query_frame_features = query_result["frame_features"]
    query_frame_vectors = query_result["feature_vectors"]

    db = DatabaseManager(**DB_CONFIG)
    db.connect()

    try:
        dataset_rows = db.fetch_dataset_features()

        votes = vote_instrument_by_nearest_frame(
            query_frame_vectors=query_frame_vectors,
            dataset_rows=dataset_rows,
        )

        top_results = rank_similar_files(
            query_frame_vectors=query_frame_vectors,
            dataset_rows=dataset_rows,
            top_k=TOP_K,
            normalize=True,
        )

        copied_dir = copy_top_result_wavs(query_file_path.name, top_results)

        exported_rows = []

        for item in top_results:
            matched_file_name = item["file_name"]
            copied_source = find_file_in_flat_folder(matched_file_name, str(DATA_ALL_ROOT))

            exported_rows.append({
                "query_file_name": query_file_path.name,
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

        info = {
            "query_valid_frames": len(query_frame_features),
            "copied_dir": copied_dir,
            "csv_path": str(TOP5_RESULTS_CSV),
        }

        return top_results, votes, info

    finally:
        db.close()


def show_votes(votes: dict) -> None:
    if not votes:
        st.info("Không có kết quả dự đoán nhạc cụ.")
        return

    rows = [
        {
            "Nhạc cụ": label,
            "Tỷ lệ theo frame (%)": percent,
        }
        for label, percent in votes.items()
    ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def show_top_results(top_results: list[dict]) -> None:
    if not top_results:
        st.warning("Không có kết quả top 5.")
        return

    table_rows = []

    for item in top_results:
        table_rows.append({
            "Rank": item["rank_position"],
            "File": item["file_name"],
            "Nhạc cụ": item.get("instrument_name"),
            "Distance": round(item["distance_score"], 6),
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    st.subheader("Nghe top 5 file giống nhất")

    for item in top_results:
        st.markdown(
            f"**Top {item['rank_position']}: {item['file_name']}**  \n"
            f"Nhạc cụ: `{item.get('instrument_name')}`  \n"
            f"Distance: `{item['distance_score']:.6f}`"
        )

        audio_path = item.get("copied_audio_path")

        if audio_path and Path(audio_path).exists():
            st.audio(audio_path)
        else:
            st.warning("Không tìm thấy file audio trong data_all để phát.")

        st.divider()


def main() -> None:
    st.set_page_config(
        page_title="String Instrument Search",
        page_icon="🎻",
        layout="wide",
    )
    ensure_project_folders()

    st.title("Tìm kiếm tiếng nhạc cụ bộ dây")
    st.caption("Frame 0.5s, hop 0.25s, vector 6 chiều, so sánh Euclidean.")

    tab_query, tab_build = st.tabs(["Upload file query", "Build dataset"])

    with tab_query:
        st.header("1. Upload file để tìm top 5")

        uploaded_file = st.file_uploader(
            "Chọn file âm thanh",
            type=["wav"],
        )

        if uploaded_file is not None:
            st.write(f"File đã chọn: `{uploaded_file.name}`")
            st.audio(uploaded_file)

        if st.button("Tìm kiếm top 5", type="primary", disabled=uploaded_file is None):
            try:
                with st.spinner("Đang trích đặc trưng và so sánh với dữ liệu trong MySQL..."):
                    query_path = save_uploaded_query_file(uploaded_file)
                    top_results, votes, info = run_query(query_path)

                st.success("Tìm kiếm xong.")

                col1, col2 = st.columns(2)
                col1.metric("Valid frames", info["query_valid_frames"])
                col2.metric("Top K", TOP_K)

                st.subheader("Dự đoán nhạc cụ theo frame")
                show_votes(votes)

                st.subheader("Top 5 giống nhất")
                show_top_results(top_results)

                st.info(f"CSV kết quả: {info['csv_path']}")
                st.info(f"Folder wav top 5: {info['copied_dir']}")

            except Exception as e:
                st.error("Có lỗi khi query.")
                st.exception(e)

    with tab_build:
        st.header("2. Build dataset")

        st.write("Quy trình khi bấm nút:")
        st.code(
            "data/data_all -> copy_data.py -> data/dataset/<ten_nhac_cu>/ -> extract feature -> MySQL",
            language="text",
        )

        st.warning(
            "Tên file trong data_all nên có dạng: "
            "`violin_01.wav`, `violin-01.wav`, `dan_tranh_01.wav`, `dan_tranh-01.wav`."
        )

        clear_old_data = st.checkbox("Xóa dataset cũ trong database trước khi build", value=True)

        if st.button("Copy data_all và build dataset", type="primary"):
            try:
                with st.spinner("Đang copy file từ data_all sang dataset theo tên nhạc cụ..."):
                    copy_result = copy_data_all_to_dataset(clear_old_dataset=True)

                if copy_result["success"]:
                    st.success(
                        f"Copy data_all sang dataset xong. "
                        f"Đã copy {copy_result['copied']} file, "
                        f"sai tên {copy_result['invalid_name']} file."
                    )
                else:
                    st.warning(copy_result["message"])
                    
                with st.spinner("Đang trích xuất đặc trưng và lưu vào MySQL..."):
                    result = build_dataset(clear_old_data=clear_old_data)

                if result["success"]:
                    st.success("Build dataset xong.")
                    col1, col2 = st.columns(2)
                    col1.metric("Số file audio", result["audio_count"])
                    col2.metric("Số frame feature", result["frame_count"])
                    st.info(f"CSV feature: {result['csv_path']}")
                else:
                    st.warning(result["message"])

            except Exception as e:
                st.error("Có lỗi khi build dataset.")
                st.exception(e)

        st.divider()

        st.subheader("Đường dẫn hiện tại")
        st.code(
            f"DATA_ALL_ROOT = {DATA_ALL_ROOT}\n"
            f"DATASET_ROOT  = {BASE_DIR / 'data' / 'dataset'}\n"
            f"QUERY_ROOT    = {QUERY_ROOT}\n"
            f"RESULTS_DIR   = {BASE_DIR / 'results'}",
            language="text",
        )


if __name__ == "__main__":
    main()