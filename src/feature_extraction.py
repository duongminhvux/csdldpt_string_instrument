import os
from typing import Dict, Tuple

import librosa
import numpy as np
import soundfile as sf


TARGET_SR = 22050
N_MFCC = 13
N_CHROMA = 12


def load_audio(file_path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return y, sr


def trim_silence(y: np.ndarray, top_db: int = 20) -> np.ndarray:
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if yt is None or len(yt) == 0:
        return y
    return yt


def normalize_audio(y: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(y))
    if max_val == 0:
        return y
    return y / max_val


def estimate_attack_time(y: np.ndarray, sr: int, hop_length: int = 512) -> float:
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) == 0:
        return 0.0

    peak = np.max(rms)
    if peak <= 0:
        return 0.0

    threshold = 0.9 * peak
    onset_frames = np.where(rms >= threshold)[0]
    if len(onset_frames) == 0:
        return 0.0

    attack_frame = int(onset_frames[0])
    attack_time = librosa.frames_to_time(attack_frame, sr=sr, hop_length=hop_length)
    return float(attack_time)


def estimate_pitch(y: np.ndarray, sr: int) -> Tuple[float, float]:
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )

    if f0 is None:
        return 0.0, 0.0

    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) == 0:
        return 0.0, 0.0

    return float(np.mean(valid_f0)), float(np.std(valid_f0))


def get_basic_metadata(file_path: str) -> Dict[str, float]:
    info = sf.info(file_path)
    return {
        "duration_seconds": float(info.duration),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "file_size_bytes": int(os.path.getsize(file_path)),
        "bit_depth": extract_bit_depth(info.subtype),
        "file_format": "wav",
    }


def extract_bit_depth(subtype: str) -> int:
    subtype = subtype.upper()

    if "PCM_16" in subtype:
        return 16
    if "PCM_24" in subtype:
        return 24
    if "PCM_32" in subtype:
        return 32
    if "PCM_U8" in subtype or "PCM_S8" in subtype:
        return 8
    if "FLOAT" in subtype:
        return 32

    return 0


def summarize_feature(values: np.ndarray) -> Tuple[float, float]:
    if values is None or len(values) == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def extract_features(file_path: str) -> Dict[str, float]:
    y, sr = load_audio(file_path)
    y = trim_silence(y)
    y = normalize_audio(y)

    features: Dict[str, float] = {}

    # Time / energy features
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    attack_time = estimate_attack_time(y, sr)

    features["rms_mean"], features["rms_std"] = summarize_feature(rms)
    features["zcr_mean"], features["zcr_std"] = summarize_feature(zcr)
    features["attack_time"] = attack_time

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    features["spectral_centroid_mean"], features["spectral_centroid_std"] = summarize_feature(spectral_centroid)
    features["spectral_bandwidth_mean"], features["spectral_bandwidth_std"] = summarize_feature(spectral_bandwidth)
    features["spectral_rolloff_mean"], features["spectral_rolloff_std"] = summarize_feature(spectral_rolloff)

    pitch_mean, pitch_std = estimate_pitch(y, sr)
    features["pitch_mean"] = pitch_mean
    features["pitch_std"] = pitch_std

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        coeff = mfcc[i]
        features[f"mfcc_{i+1}_mean"], features[f"mfcc_{i+1}_std"] = summarize_feature(coeff)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
    for i in range(N_CHROMA):
        chroma_band = chroma[i]
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma_band)) if len(chroma_band) > 0 else 0.0

    return features


def build_feature_vector(features: Dict[str, float]) -> np.ndarray:
    ordered_keys = [
        "rms_mean", "rms_std",
        "zcr_mean", "zcr_std",
        "attack_time",
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "pitch_mean", "pitch_std",
    ]

    for i in range(1, N_MFCC + 1):
        ordered_keys.append(f"mfcc_{i}_mean")
    for i in range(1, N_MFCC + 1):
        ordered_keys.append(f"mfcc_{i}_std")
    for i in range(1, N_CHROMA + 1):
        ordered_keys.append(f"chroma_{i}_mean")

    vector = np.array([features.get(k, 0.0) for k in ordered_keys], dtype=np.float32)
    return vector


def extract_all(file_path: str) -> Dict[str, Dict]:
    metadata = get_basic_metadata(file_path)
    features = extract_features(file_path)
    feature_vector = build_feature_vector(features)

    return {
        "metadata": metadata,
        "features": features,
        "feature_vector": feature_vector,
    }