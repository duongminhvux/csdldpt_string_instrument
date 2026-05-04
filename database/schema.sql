CREATE DATABASE IF NOT EXISTS string_instrument_search
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE string_instrument_search;

DROP TABLE IF EXISTS audio_features;
DROP TABLE IF EXISTS audio_files;

CREATE TABLE audio_files (
    audio_id INT AUTO_INCREMENT PRIMARY KEY,

    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,

    -- Chỉ lưu file thuộc bộ dữ liệu gốc.
    -- File query không lưu vào CSDL, chỉ xử lý tạm trong bộ nhớ.
    dataset_type ENUM('dataset') NOT NULL DEFAULT 'dataset',

    instrument_name VARCHAR(100) NULL,

    duration_seconds FLOAT NULL,
    sample_rate INT NULL,
    bit_depth INT NULL,
    channels INT NULL,
    file_size_bytes BIGINT NULL,
    file_format VARCHAR(20) NOT NULL DEFAULT 'wav',

    notes TEXT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE audio_features (
    feature_id INT AUTO_INCREMENT PRIMARY KEY,

    audio_id INT NOT NULL,

    frame_index INT NOT NULL,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,

    rms_energy FLOAT NOT NULL DEFAULT 0,
    zero_crossing_rate FLOAT NOT NULL DEFAULT 0,
    average_frequency FLOAT NOT NULL DEFAULT 0,
    frequency_variation FLOAT NOT NULL DEFAULT 0,
    average_pitch FLOAT NOT NULL DEFAULT 0,
    pitch_variation FLOAT NOT NULL DEFAULT 0,

    feature_vector_json JSON NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT fk_audio_features_audio
        FOREIGN KEY (audio_id)
        REFERENCES audio_files(audio_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE,

    UNIQUE KEY uq_audio_frame (audio_id, frame_index)
);

CREATE INDEX idx_audio_files_dataset_type ON audio_files(dataset_type);
CREATE INDEX idx_audio_files_instrument_name ON audio_files(instrument_name);
CREATE INDEX idx_audio_features_audio_id ON audio_features(audio_id);
CREATE INDEX idx_audio_features_frame_index ON audio_features(frame_index);