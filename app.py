import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from scripts.build_dataset import build_dataset
from src.config import DATA_ALL_ROOT, QUERY_ROOT, RESULTS_DIR, TOP5_RESULTS_CSV, DB_CONFIG
from src.database_manager import DatabaseManager
from src.feature_extraction import extract_all
from src.retrieval import rank_similar_files
from src.utils import ensure_folder_exists, save_dicts_to_csv


TOP_K = 5


def ensure_project_folders() -> None:
    for folder in [DATA_ALL_ROOT, QUERY_ROOT, RESULTS_DIR]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def save_uploaded_query_file(uploaded_file) -> Path:
    ensure_folder_exists(str(QUERY_ROOT))

    safe_name = uploaded_file.name.replace(" ", "_")
    query_path = QUERY_ROOT / safe_name

    with open(query_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return query_path


def run_query(query_file_path: Path) -> tuple[list[dict], dict]:
    query_result = extract_all(str(query_file_path))

    query_frame_features = query_result["frame_features"]
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
                    "query_file_name": query_file_path.name,
                    "rank": item["rank_position"],
                    "matched_audio_id": item["matched_audio_id"],
                    "file_name": item["file_name"],
                    "file_path": item["file_path"],
                    "euclidean_distance": item["distance_score"],
                }
            )

        save_dicts_to_csv(exported_rows, str(TOP5_RESULTS_CSV))

        info = {
            "query_valid_frames": len(query_frame_features),
            "csv_path": str(TOP5_RESULTS_CSV),
        }

        return top_results, info

    finally:
        db.close()


def show_top_results(top_results: list[dict]) -> None:
    if not top_results:
        st.warning("Không có kết quả top 5.")
        return

    table_rows = []

    for item in top_results:
        table_rows.append(
            {
                "Rank": item["rank_position"],
                "File name": item["file_name"],
                "File path": item["file_path"],
                "Euclidean distance": round(item["distance_score"], 6),
            }
        )

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    st.subheader("Nghe top 5 file giống nhất")

    for item in top_results:
        st.markdown(
            f"**Top {item['rank_position']}: {item['file_name']}**  \n"
            f"File path: `{item['file_path']}`  \n"
            f"Euclidean distance: `{item['distance_score']:.6f}`"
        )

        audio_path = Path(item["file_path"])

        if audio_path.exists():
            st.audio(str(audio_path))
        else:
            st.warning("Không tìm thấy file audio theo file_path trong DB.")

        st.divider()


def main() -> None:
    st.set_page_config(
        page_title="String Instrument Search",
        layout="wide",
    )

    ensure_project_folders()

    st.title("Tìm kiếm tiếng nhạc cụ bộ dây")
    st.caption("Frame 0.5s, hop 0.25s, vector 6 chiều, chuẩn hóa Z-score, so sánh Euclidean.")

    tab_query, tab_build = st.tabs(["Upload file query", "Build dataset"])

    with tab_query:
        st.header("1. Upload file để tìm top 5")

        uploaded_file = st.file_uploader(
            "Chọn file âm thanh .wav",
            type=["wav"],
        )

        if uploaded_file is not None:
            st.write(f"File đã chọn: `{uploaded_file.name}`")
            st.audio(uploaded_file)

        if st.button("Tìm kiếm top 5", type="primary", disabled=uploaded_file is None):
            try:
                with st.spinner("Đang lưu file query vào data/query và trích đặc trưng..."):
                    query_path = save_uploaded_query_file(uploaded_file)

                with st.spinner("Đang so sánh với đặc trưng dataset trong MySQL..."):
                    top_results, info = run_query(query_path)

                st.success("Tìm kiếm xong.")

                col1, col2 = st.columns(2)
                col1.metric("Valid query frames", info["query_valid_frames"])
                col2.metric("Top K", TOP_K)

                st.info(f"File query đã lưu tạm tại: `{query_path}`")
                st.info(f"CSV kết quả: `{info['csv_path']}`")

                st.subheader("Top 5 giống nhất")
                show_top_results(top_results)

            except Exception as e:
                st.error("Có lỗi khi query.")
                st.exception(e)

    with tab_build:
        st.header("2. Build dataset")

        st.write("Quy trình build dataset:")
        st.code(
            "data/data_all -> extract frame features -> audio_files + audio_features trong MySQL",
            language="text",
        )

        st.warning(
            "Hệ thống chỉ đọc file .wav trong data/data_all. "
            "Không lấy instrument_name, không đoán nhạc cụ, không copy file sang data/dataset."
        )

        clear_old_data = st.checkbox("Xóa dataset cũ trong database trước khi build", value=True)

        if st.button("Build dataset", type="primary"):
            try:
                with st.spinner("Đang trích xuất đặc trưng frame và lưu vào MySQL..."):
                    result = build_dataset(clear_old_data=clear_old_data)

                if result["success"]:
                    st.success("Build dataset xong.")

                    col1, col2 = st.columns(2)
                    col1.metric("Số file audio", result["audio_count"])
                    col2.metric("Số frame feature", result["frame_count"])

                    st.info(f"CSV feature: `{result['csv_path']}`")
                else:
                    st.warning(result["message"])

            except Exception as e:
                st.error("Có lỗi khi build dataset.")
                st.exception(e)

        st.divider()

        st.subheader("Đường dẫn hiện tại")
        st.code(
            f"DATA_ALL_ROOT = {DATA_ALL_ROOT}\n"
            f"QUERY_ROOT = {QUERY_ROOT}\n"
            f"RESULTS_DIR = {RESULTS_DIR}",
            language="text",
        )


if __name__ == "__main__":
    main()