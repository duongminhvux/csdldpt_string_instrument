from collections import defaultdict
from typing import Any, Dict, List

import numpy as np


VECTOR_KEYS = [
    "rms_energy",
    "zero_crossing_rate",
    "average_frequency",
    "frequency_variation",
    "average_pitch",
    "pitch_variation",
]


def build_vector_from_row(row: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [float(row.get(key) or 0.0) for key in VECTOR_KEYS],
        dtype=np.float32,
    )


def zscore_normalize(
    query_vectors: List[np.ndarray],
    dataset_vectors: List[np.ndarray],
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    all_vectors = np.vstack(query_vectors + dataset_vectors)

    mean = np.mean(all_vectors, axis=0)
    std = np.std(all_vectors, axis=0)

    std[std == 0] = 1.0

    query_norm = [(v - mean) / std for v in query_vectors]
    dataset_norm = [(v - mean) / std for v in dataset_vectors]

    return query_norm, dataset_norm


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.linalg.norm(v1 - v2))


def group_dataset_frames(dataset_rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    grouped = {}

    for row in dataset_rows:
        audio_id = row["audio_id"]

        if audio_id not in grouped:
            grouped[audio_id] = {
                "audio_id": audio_id,
                "file_name": row["file_name"],
                "file_path": row["file_path"],
                "instrument_name": row.get("instrument_name"),
                "frames": [],
            }

        grouped[audio_id]["frames"].append({
            "row": row,
            "vector": build_vector_from_row(row),
        })

    return grouped


def rank_similar_files(
    query_frame_vectors: List[np.ndarray],
    dataset_rows: List[Dict[str, Any]],
    top_k: int = 5,
    normalize: bool = True,
) -> List[Dict[str, Any]]:
    if not query_frame_vectors:
        raise ValueError("Query file has no valid frame vectors.")

    if not dataset_rows:
        raise ValueError("Dataset has no frame features. Run scripts/build_dataset.py first.")

    grouped = group_dataset_frames(dataset_rows)

    dataset_all_vectors = []
    dataset_vector_refs = []

    for audio_id, item in grouped.items():
        for frame in item["frames"]:
            dataset_all_vectors.append(frame["vector"])
            dataset_vector_refs.append((audio_id, frame))

    if normalize:
        query_vectors_norm, dataset_vectors_norm = zscore_normalize(
            query_frame_vectors,
            dataset_all_vectors,
        )

        for i, (_, frame) in enumerate(dataset_vector_refs):
            frame["vector_norm"] = dataset_vectors_norm[i]
    else:
        query_vectors_norm = query_frame_vectors

        for _, frame in dataset_vector_refs:
            frame["vector_norm"] = frame["vector"]

    results = []

    for audio_id, item in grouped.items():
        dataset_vectors = [frame["vector_norm"] for frame in item["frames"]]

        min_distances = []

        for qv in query_vectors_norm:
            distances = [euclidean_distance(qv, dv) for dv in dataset_vectors]
            min_distances.append(min(distances))

        distance_score = float(np.mean(min_distances))

        results.append({
            "matched_audio_id": audio_id,
            "file_name": item["file_name"],
            "file_path": item["file_path"],
            "instrument_name": item.get("instrument_name"),
            "distance_score": distance_score,
        })

    results.sort(key=lambda x: x["distance_score"])

    top_results = results[:top_k]

    for idx, item in enumerate(top_results, start=1):
        item["rank_position"] = idx

    return top_results


def vote_instrument_by_nearest_frame(
    query_frame_vectors: List[np.ndarray],
    dataset_rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    if not query_frame_vectors or not dataset_rows:
        return {}

    dataset_vectors = [build_vector_from_row(row) for row in dataset_rows]
    query_vectors_norm, dataset_vectors_norm = zscore_normalize(
        query_frame_vectors,
        dataset_vectors,
    )

    votes = defaultdict(int)

    for qv in query_vectors_norm:
        distances = [euclidean_distance(qv, dv) for dv in dataset_vectors_norm]
        best_idx = int(np.argmin(distances))

        label = dataset_rows[best_idx].get("instrument_name") or "unknown"
        votes[label] += 1

    total = sum(votes.values())

    if total == 0:
        return {}

    return {
        label: round(count / total * 100, 2)
        for label, count in sorted(votes.items(), key=lambda x: x[1], reverse=True)
    }


def print_top_results(top_results: List[Dict[str, Any]]) -> None:
    print("\nTOP 5 FILES GIONG NHAT:")

    for item in top_results:
        print(
            f"{item['rank_position']}. "
            f"{item['file_name']} | "
            f"instrument={item.get('instrument_name')} | "
            f"distance={item['distance_score']:.6f}"
        )


def print_instrument_votes(votes: Dict[str, float]) -> None:
    print("\nTY LE DU DOAN NHAC CU THEO FRAME:")

    if not votes:
        print("No vote result.")
        return

    for label, percent in votes.items():
        print(f"{label}: {percent}%")