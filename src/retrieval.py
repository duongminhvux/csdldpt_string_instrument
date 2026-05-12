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
    if not query_vectors:
        raise ValueError("Query file has no valid frame vectors.")

    if not dataset_vectors:
        raise ValueError("Dataset has no frame features.")

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
    grouped: Dict[int, Dict[str, Any]] = {}

    for row in dataset_rows:
        audio_id = int(row["audio_id"])

        if audio_id not in grouped:
            grouped[audio_id] = {
                "audio_id": audio_id,
                "file_name": row["file_name"],
                "file_path": row["file_path"],
                "frames": [],
            }

        grouped[audio_id]["frames"].append(
            {
                "row": row,
                "vector": build_vector_from_row(row),
            }
        )

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

        results.append(
            {
                "matched_audio_id": audio_id,
                "file_name": item["file_name"],
                "file_path": item["file_path"],
                "distance_score": distance_score,
            }
        )

    results.sort(key=lambda x: x["distance_score"])

    top_results = results[:top_k]

    for idx, item in enumerate(top_results, start=1):
        item["rank_position"] = idx

    return top_results


def print_top_results(top_results: List[Dict[str, Any]]) -> None:
    print("\nTOP 5 FILES GIONG NHAT:")

    for item in top_results:
        print(
            f"{item['rank_position']}. "
            f"{item['file_name']} | "
            f"path={item['file_path']} | "
            f"distance={item['distance_score']:.6f}"
        )