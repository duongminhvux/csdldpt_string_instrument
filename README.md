# Hệ CSDL lưu trữ và tìm kiếm tiếng nhạc cụ bộ dây

## 1. Giới thiệu

Đây là hệ thống lưu trữ và tìm kiếm âm thanh nhạc cụ thuộc bộ dây. Hệ thống cho phép xây dựng bộ dữ liệu âm thanh, trích xuất đặc trưng theo từng frame, lưu đặc trưng vào MySQL và tìm kiếm 5 file âm thanh giống nhất với một file truy vấn đầu vào.

Hệ thống sử dụng:

- Python
- MySQL
- Streamlit
- Librosa
- NumPy
- Pandas

Phương pháp tìm kiếm chính là **Euclidean distance** trên vector đặc trưng âm thanh 6 chiều.

---

## 2. Chức năng chính

Hệ thống gồm các chức năng:

1. Tự động copy file âm thanh từ `data/data_all` sang `data/dataset/<tên_nhạc_cụ>`.
2. Trích xuất đặc trưng âm thanh theo frame.
3. Lưu metadata và đặc trưng âm thanh vào MySQL.
4. Upload file âm thanh truy vấn trên giao diện Streamlit.
5. Tìm kiếm 5 file âm thanh giống nhất với file truy vấn.
6. Hiển thị kết quả top 5 và cho phép nghe trực tiếp các file kết quả.

---

## 3. Cấu trúc thư mục

```text
csdldpt_string_instrument/
│
├── app.py
├── copy_data.py
├── requirements.txt
├── README.md
│
├── database/
│   └── schema.sql
│
├── data/
│   ├── data_all/
│   ├── dataset/
│   └── query/
│
├── results/
│   ├── features.csv
│   ├── top5_results.csv
│   └── top5_wav/
│
├── scripts/
│   ├── build_dataset.py
│   └── run_query.py
│
└── src/
    ├── config.py
    ├── database_manager.py
    ├── feature_extraction.py
    ├── retrieval.py
    └── utils.py