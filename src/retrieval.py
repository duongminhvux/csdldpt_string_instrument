import numpy as np
from typing import Any, Dict, List, Tuple


def cosine_similarity(query_vector: np.ndarray, dataset_vector: np.ndarray) -> float:
    query_norm = np.linalg.norm(query_vector)
    dataset_norm = np.linalg.norm(dataset_vector)

    if query_norm == 0 or dataset_norm == 0:
        return 0.0

    similarity = np.dot(query_vector, dataset_vector) / (query_norm * dataset_norm)
    return float(similarity)


def euclidean_distance(query_vector: np.ndarray, dataset_vector: np.ndarray) -> float:
    return float(np.linalg.norm(query_vector - dataset_vector))


def min_max_normalize(
    query_vector: np.ndarray,
    dataset_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([query_vector, dataset_matrix])

    min_vals = np.min(combined, axis=0)
    max_vals = np.max(combined, axis=0)

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    normalized = (combined - min_vals) / ranges

    normalized_query = normalized[0]
    normalized_dataset = normalized[1:]

    return normalized_query, normalized_dataset


def z_score_normalize(
    query_vector: np.ndarray,
    dataset_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([query_vector, dataset_matrix])

    means = np.mean(combined, axis=0)
    stds = np.std(combined, axis=0)
    stds[stds == 0] = 1.0

    normalized = (combined - means) / stds

    normalized_query = normalized[0]
    normalized_dataset = normalized[1:]

    return normalized_query, normalized_dataset


def build_vector_from_db_row(row: Dict[str, Any]) -> np.ndarray:
    ordered_keys = [
        "rms_mean", "rms_std",
        "zcr_mean", "zcr_std",
        "attack_time",
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "pitch_mean", "pitch_std",
    ]

    for i in range(1, 14):
        ordered_keys.append(f"mfcc_{i}_mean")
    for i in range(1, 14):
        ordered_keys.append(f"mfcc_{i}_std")
    for i in range(1, 13):
        ordered_keys.append(f"chroma_{i}_mean")

    vector = np.array([row.get(k, 0.0) or 0.0 for k in ordered_keys], dtype=np.float32)
    return vector


def rank_similar_files(
    query_vector: np.ndarray,
    dataset_rows: List[Dict[str, Any]],
    method: str = "cosine",
    normalize: str = "zscore",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    if len(dataset_rows) == 0:
        return []

    dataset_vectors = np.array([build_vector_from_db_row(row) for row in dataset_rows], dtype=np.float32)

    if normalize == "minmax":
        norm_query, norm_dataset = min_max_normalize(query_vector, dataset_vectors)
    elif normalize == "zscore":
        norm_query, norm_dataset = z_score_normalize(query_vector, dataset_vectors)
    else:
        norm_query, norm_dataset = query_vector, dataset_vectors

    results = []

    for i, row in enumerate(dataset_rows):
        dataset_vector = norm_dataset[i]

        if method == "euclidean":
            distance = euclidean_distance(norm_query, dataset_vector)
            similarity = 1.0 / (1.0 + distance)
        else:
            similarity = cosine_similarity(norm_query, dataset_vector)
            distance = euclidean_distance(norm_query, dataset_vector)

        results.append({
            "matched_audio_id": row["audio_id"],
            "file_name": row["file_name"],
            "file_path": row["file_path"],
            "instrument_name": row.get("instrument_name"),
            "similarity_score": float(similarity),
            "distance_score": float(distance),
        })

    if method == "euclidean":
        results.sort(key=lambda x: x["distance_score"])
    else:
        results.sort(key=lambda x: x["similarity_score"], reverse=True)

    top_results = results[:top_k]

    for rank, item in enumerate(top_results, start=1):
        item["rank_position"] = rank

    return top_results


def print_top_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No matching results found.")
        return

    print("\nTop matching files:")
    print("-" * 90)
    for item in results:
        print(
            f"Rank {item['rank_position']}: "
            f"{item['file_name']} | "
            f"Instrument: {item.get('instrument_name')} | "
            f"Similarity: {item['similarity_score']:.6f} | "
            f"Distance: {item['distance_score']:.6f}"
        )
    print("-" * 90)