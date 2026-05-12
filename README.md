
# String Instrument Search

Hệ thống xây dựng cơ sở dữ liệu lưu trữ đặc trưng âm thanh và tìm kiếm **top 5 file âm thanh giống nhất** với một file truy vấn đầu vào.

Pipeline của project được xây dựng đúng theo báo cáo:

```text
data/data_all
-> đọc metadata file âm thanh
-> tiền xử lý âm thanh
-> chia frame 0.5s, hop 0.25s
-> trích xuất vector đặc trưng 6 chiều
-> lưu metadata vào audio_files
-> lưu đặc trưng theo frame vào audio_features

data/query
-> lưu tạm file query được upload
-> trích xuất vector đặc trưng 6 chiều
-> lấy đặc trưng dataset từ MySQL
-> chuẩn hóa Z-score
-> tính Euclidean distance theo frame
-> sắp xếp tăng dần distance
-> trả về top 5 file giống nhất
````

Project **không lưu nhãn nhạc cụ trong database**, **không đoán instrument_name**, **không copy file kết quả**, và **không lưu file query vào database**.

---

---

## 5. Cấu trúc thư mục

```text
csdldpt_string_instrument/
│
├── app.py
├── requirements.txt
├── README.md
│
├── database/
│   └── schema.sql
│
├── scripts/
│   ├── build_dataset.py
│   └── run_query.py
│
├── src/
│   ├── audio_utils.py
│   ├── config.py
│   ├── database_manager.py
│   ├── feature_extraction.py
│   ├── retrieval.py
│   └── utils.py
│
├── data/
│   ├── data_all/
│   │   ├── banjo_01.wav
│   │   ├── cello_01.wav
│   │   └── ...
│   │
│   └── query/
│       └── query_01.wav
│
└── results/
    ├── features.csv
    └── top5_results.csv
```

Trong đó:

| Thư mục / file             | Ý nghĩa                              |
| -------------------------- | ------------------------------------ |
| `data/data_all`            | Chứa toàn bộ file `.wav` của dataset |
| `data/query`               | Chứa file query upload tạm thời      |
| `results/features.csv`     | CSV backup đặc trưng đã trích xuất   |
| `results/top5_results.csv` | CSV kết quả top 5                    |
| `database/schema.sql`      | File tạo database và bảng            |
| `scripts/build_dataset.py` | Script build dataset                 |
| `scripts/run_query.py`     | Script query bằng terminal           |
| `app.py`                   | Giao diện Streamlit                  |

---

## 6. Cài đặt môi trường

### 6.1. Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt môi trường ảo trên Windows:

```bash
venv\Scripts\activate
```

### 6.2. Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu thiếu thư viện, có thể cài trực tiếp:

```bash
pip install librosa soundfile numpy pandas streamlit mysql-connector-python
```

---

## 7. Cấu hình MySQL

Mở file:

```text
src/config.py
```

Sửa thông tin kết nối MySQL:

```python
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "string_instrument_search",
}
```
---

## 11. Chạy giao diện Streamlit

Chạy lệnh:

```bash
streamlit run app.py
```

Giao diện có 2 tab:

```text
Upload file query
Build dataset
```


