-- MySQL schema for string instrument audio search

CREATE DATABASE IF NOT EXISTS string_instrument_search;
USE string_instrument_search;

CREATE TABLE IF NOT EXISTS audio_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    instrument_label VARCHAR(100),
    dataset_type VARCHAR(50) NOT NULL,
    duration FLOAT,
    sample_rate INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audio_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    audio_id INT NOT NULL,
    rms_mean FLOAT,
    rms_std FLOAT,
    zcr_mean FLOAT,
    zcr_std FLOAT,
    centroid_mean FLOAT,
    centroid_std FLOAT,
    bandwidth_mean FLOAT,
    bandwidth_std FLOAT,
    rolloff_mean FLOAT,
    rolloff_std FLOAT,
    pitch_mean FLOAT,
    pitch_std FLOAT,
    mfcc_1_mean FLOAT,
    mfcc_2_mean FLOAT,
    mfcc_3_mean FLOAT,
    mfcc_4_mean FLOAT,
    mfcc_5_mean FLOAT,
    mfcc_6_mean FLOAT,
    mfcc_7_mean FLOAT,
    mfcc_8_mean FLOAT,
    mfcc_9_mean FLOAT,
    mfcc_10_mean FLOAT,
    mfcc_11_mean FLOAT,
    mfcc_12_mean FLOAT,
    mfcc_13_mean FLOAT,
    FOREIGN KEY (audio_id) REFERENCES audio_files(id) ON DELETE CASCADE
);
