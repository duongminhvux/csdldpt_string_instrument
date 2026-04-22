CREATE DATABASE IF NOT EXISTS string_instrument_search
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE string_instrument_search;

-- Xóa bảng theo thứ tự phụ thuộc để chạy lại schema cho tiện
DROP TABLE IF EXISTS search_results;
DROP TABLE IF EXISTS search_queries;
DROP TABLE IF EXISTS audio_features;
DROP TABLE IF EXISTS audio_files;
DROP TABLE IF EXISTS instruments;

-- 1. Danh mục nhạc cụ
CREATE TABLE instruments (
    instrument_id INT AUTO_INCREMENT PRIMARY KEY,
    instrument_name VARCHAR(100) NOT NULL UNIQUE,
    instrument_family VARCHAR(100) NOT NULL DEFAULT 'string',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Metadata file âm thanh
CREATE TABLE audio_files (
    audio_id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    instrument_id INT NULL,
    dataset_type ENUM('dataset', 'query') NOT NULL,
    duration_seconds FLOAT NULL,
    sample_rate INT NULL,
    bit_depth INT NULL,
    channels INT NULL,
    file_size_bytes BIGINT NULL,
    file_format VARCHAR(20) NOT NULL DEFAULT 'wav',
    notes TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT fk_audio_files_instrument
        FOREIGN KEY (instrument_id) REFERENCES instruments(instrument_id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
);

-- 3. Đặc trưng âm thanh
CREATE TABLE audio_features (
    feature_id INT AUTO_INCREMENT PRIMARY KEY,
    audio_id INT NOT NULL UNIQUE,

    -- Time / energy features
    rms_mean FLOAT NULL,
    rms_std FLOAT NULL,
    zcr_mean FLOAT NULL,
    zcr_std FLOAT NULL,
    attack_time FLOAT NULL,

    -- Spectral features
    spectral_centroid_mean FLOAT NULL,
    spectral_centroid_std FLOAT NULL,
    spectral_bandwidth_mean FLOAT NULL,
    spectral_bandwidth_std FLOAT NULL,
    spectral_rolloff_mean FLOAT NULL,
    spectral_rolloff_std FLOAT NULL,
    pitch_mean FLOAT NULL,
    pitch_std FLOAT NULL,

    -- MFCC mean 1..13
    mfcc_1_mean FLOAT NULL,
    mfcc_2_mean FLOAT NULL,
    mfcc_3_mean FLOAT NULL,
    mfcc_4_mean FLOAT NULL,
    mfcc_5_mean FLOAT NULL,
    mfcc_6_mean FLOAT NULL,
    mfcc_7_mean FLOAT NULL,
    mfcc_8_mean FLOAT NULL,
    mfcc_9_mean FLOAT NULL,
    mfcc_10_mean FLOAT NULL,
    mfcc_11_mean FLOAT NULL,
    mfcc_12_mean FLOAT NULL,
    mfcc_13_mean FLOAT NULL,

    -- MFCC std 1..13
    mfcc_1_std FLOAT NULL,
    mfcc_2_std FLOAT NULL,
    mfcc_3_std FLOAT NULL,
    mfcc_4_std FLOAT NULL,
    mfcc_5_std FLOAT NULL,
    mfcc_6_std FLOAT NULL,
    mfcc_7_std FLOAT NULL,
    mfcc_8_std FLOAT NULL,
    mfcc_9_std FLOAT NULL,
    mfcc_10_std FLOAT NULL,
    mfcc_11_std FLOAT NULL,
    mfcc_12_std FLOAT NULL,
    mfcc_13_std FLOAT NULL,

    -- Chroma mean 1..12
    chroma_1_mean FLOAT NULL,
    chroma_2_mean FLOAT NULL,
    chroma_3_mean FLOAT NULL,
    chroma_4_mean FLOAT NULL,
    chroma_5_mean FLOAT NULL,
    chroma_6_mean FLOAT NULL,
    chroma_7_mean FLOAT NULL,
    chroma_8_mean FLOAT NULL,
    chroma_9_mean FLOAT NULL,
    chroma_10_mean FLOAT NULL,
    chroma_11_mean FLOAT NULL,
    chroma_12_mean FLOAT NULL,

    -- Lưu vector tổng hợp nếu muốn debug / export
    feature_vector_json JSON NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT fk_audio_features_audio
        FOREIGN KEY (audio_id) REFERENCES audio_files(audio_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- 4. Lưu lịch sử các lần query
CREATE TABLE search_queries (
    query_id INT AUTO_INCREMENT PRIMARY KEY,
    query_audio_id INT NOT NULL,
    similarity_method ENUM('euclidean', 'cosine') NOT NULL DEFAULT 'cosine',
    top_k INT NOT NULL DEFAULT 5,
    queried_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT NULL,

    CONSTRAINT fk_search_queries_audio
        FOREIGN KEY (query_audio_id) REFERENCES audio_files(audio_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

-- 5. Lưu kết quả top-k của từng query
CREATE TABLE search_results (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    query_id INT NOT NULL,
    matched_audio_id INT NOT NULL,
    rank_position INT NOT NULL,
    similarity_score FLOAT NULL,
    distance_score FLOAT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_search_results_query
        FOREIGN KEY (query_id) REFERENCES search_queries(query_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,

    CONSTRAINT fk_search_results_audio
        FOREIGN KEY (matched_audio_id) REFERENCES audio_files(audio_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,

    CONSTRAINT uq_query_rank UNIQUE (query_id, rank_position)
);

-- Index để truy vấn nhanh hơn
CREATE INDEX idx_audio_files_dataset_type ON audio_files(dataset_type);
CREATE INDEX idx_audio_files_instrument_id ON audio_files(instrument_id);
CREATE INDEX idx_search_queries_audio_id ON search_queries(query_audio_id);
CREATE INDEX idx_search_results_query_id ON search_results(query_id);
CREATE INDEX idx_search_results_matched_audio_id ON search_results(matched_audio_id);

-- Dữ liệu mẫu cho nhạc cụ bộ dây
INSERT INTO instruments (instrument_name, instrument_family, description) VALUES
('violin', 'string', 'Nhac cu day am vuc cao'),
('viola', 'string', 'Nhac cu day trung'),
('cello', 'string', 'Nhac cu day am vuc tram'),
('guitar', 'string', 'Nhac cu day gam'),
('double_bass', 'string', 'Nhac cu day am vuc rat tram');