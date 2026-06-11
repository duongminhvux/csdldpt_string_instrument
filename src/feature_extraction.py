import os
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf


TARGET_SR = 22050
FRAME_DURATION = 0.5
HOP_DURATION = 0.25

SILENCE_THRESHOLD = 0.01
SILENCE_RATIO = 0.8

VECTOR_KEYS = [
    "rms_energy",
    "zero_crossing_rate",
    "average_frequency",
    "frequency_variation",
    "average_pitch",
    "pitch_variation",
]


def load_audio(file_path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return y.astype(np.float32), sr


def normalize_audio(y: np.ndarray) -> np.ndarray:
    if y is None or len(y) == 0:
        return np.array([], dtype=np.float32)

    max_abs = float(np.max(np.abs(y)))
    if max_abs == 0:
        return y.astype(np.float32)

    return (y / max_abs).astype(np.float32)


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


def get_basic_metadata(file_path: str) -> Dict[str, object]:
    info = sf.info(file_path)

    return {
        "duration_seconds": float(info.duration),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "file_size_bytes": int(os.path.getsize(file_path)),
        "bit_depth": extract_bit_depth(info.subtype),
        "file_format": "wav",
    }


def is_silent_frame(frame: np.ndarray) -> bool:
    if frame is None or len(frame) == 0:
        return True

    silent_count = int(np.sum(np.abs(frame) < SILENCE_THRESHOLD))
    silent_ratio = silent_count / len(frame)

    return silent_ratio >= SILENCE_RATIO


def split_frames(y: np.ndarray, sr: int) -> List[Tuple[int, float, float, np.ndarray]]:
    frame_size = int(FRAME_DURATION * sr)
    hop_size = int(HOP_DURATION * sr)

    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("FRAME_DURATION and HOP_DURATION must be positive.")

    frames = []

    if len(y) == 0:
        padded = np.zeros(frame_size, dtype=np.float32)
        return [(0, 0.0, FRAME_DURATION, padded)]

    if len(y) < frame_size:
        padded = np.pad(y, (0, frame_size - len(y)))
        return [(0, 0.0, FRAME_DURATION, padded.astype(np.float32))]

    frame_index = 0

    for start in range(0, len(y) - frame_size + 1, hop_size):
        end = start + frame_size
        start_time = start / sr
        end_time = end / sr
        frame = y[start:end].astype(np.float32)

        frames.append((frame_index, start_time, end_time, frame))
        frame_index += 1

    return frames


def calc_rms(frame: np.ndarray) -> float:
    if len(frame) == 0:
        return 0.0

    return float(np.sqrt(np.mean(frame ** 2)))


def calc_zero_crossing_rate(frame: np.ndarray) -> float:
    if len(frame) < 2:
        return 0.0

    signs = np.sign(frame)
    signs[signs == 0] = 1

    return float(np.sum(np.abs(signs[1:] - signs[:-1])) / (2 * len(frame)))


def get_positive_spectrum(frame: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(frame) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    windowed = frame * np.hanning(len(frame))
    spectrum = np.fft.fft(windowed)
    freqs = np.fft.fftfreq(len(windowed), d=1 / sr)

    positive_mask = freqs > 0

    positive_freqs = freqs[positive_mask]
    magnitude = np.abs(spectrum[positive_mask])

    return positive_freqs.astype(np.float32), magnitude.astype(np.float32)


def calc_average_frequency(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    if len(freqs) == 0 or len(magnitude) == 0:
        return 0.0

    total_magnitude = float(np.sum(magnitude))
    if total_magnitude == 0:
        return 0.0

    return float(np.sum(freqs * magnitude) / total_magnitude)


def calc_frequency_variation(magnitude: np.ndarray) -> float:
    if len(magnitude) < 2:
        return 0.0

    return float(np.mean(np.abs(np.diff(magnitude))))


def calc_average_pitch(freqs: np.ndarray, magnitude: np.ndarray) -> float:
    if len(freqs) == 0 or len(magnitude) == 0:
        return 0.0

    max_idx = int(np.argmax(magnitude))

    return float(freqs[max_idx])


def calc_pitch_variation(frame: np.ndarray, sr: int) -> float:
    if len(frame) == 0:
        return 0.0

    sub_frame_size = int(0.01 * sr)
    sub_hop_size = max(1, sub_frame_size // 2)

    if sub_frame_size <= 0 or len(frame) < sub_frame_size:
        return 0.0

    pitches = []

    for start in range(0, len(frame) - sub_frame_size + 1, sub_hop_size):
        sub_frame = frame[start:start + sub_frame_size]
        freqs, magnitude = get_positive_spectrum(sub_frame, sr)
        pitch = calc_average_pitch(freqs, magnitude)

        if pitch > 0:
            pitches.append(pitch)

    if len(pitches) < 2:
        return 0.0

    return float(np.mean(np.abs(np.diff(pitches))))


def extract_features_from_frame(
    frame_index: int,
    start_time: float,
    end_time: float,
    frame: np.ndarray,
    sr: int,
) -> Dict[str, float]:
    freqs, magnitude = get_positive_spectrum(frame, sr)

    return {
        "frame_index": int(frame_index),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "rms_energy": calc_rms(frame),
        "zero_crossing_rate": calc_zero_crossing_rate(frame),
        "average_frequency": calc_average_frequency(freqs, magnitude),
        "frequency_variation": calc_frequency_variation(magnitude),
        "average_pitch": calc_average_pitch(freqs, magnitude),
        "pitch_variation": calc_pitch_variation(frame, sr),
    }


def extract_frame_features(file_path: str) -> List[Dict[str, float]]:
    y, sr = load_audio(file_path)
    y = normalize_audio(y)

    all_frames = split_frames(y, sr)
    frame_features = []

    for frame_index, start_time, end_time, frame in all_frames:
        if is_silent_frame(frame):
            continue

        feature = extract_features_from_frame(
            frame_index=frame_index,
            start_time=start_time,
            end_time=end_time,
            frame=frame,
            sr=sr,
        )
        frame_features.append(feature)

    if not frame_features:
        frame_index, start_time, end_time, frame = all_frames[0]
        frame_features.append(
            extract_features_from_frame(
                frame_index=frame_index,
                start_time=start_time,
                end_time=end_time,
                frame=frame,
                sr=sr,
            )
        )

    return frame_features


def build_feature_vector(frame_feature: Dict[str, float]) -> np.ndarray:
    return np.array(
        [float(frame_feature.get(key, 0.0)) for key in VECTOR_KEYS],
        dtype=np.float32,
    )


def extract_all(file_path: str) -> Dict[str, object]:
    metadata = get_basic_metadata(file_path)
    frame_features = extract_frame_features(file_path)
    feature_vectors = [build_feature_vector(item) for item in frame_features]

    return {
        "metadata": metadata,
        "frame_features": frame_features,
        "feature_vectors": feature_vectors,
    }