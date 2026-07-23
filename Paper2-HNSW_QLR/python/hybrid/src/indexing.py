from pathlib import Path

import faiss
import numpy as np


def build_hnsw_index(
    embeddings: np.ndarray,
    m: int = 32,
    ef_construction: int = 500,
    ef_search: int = 64,
    metric: str = "ip",
) -> faiss.Index:
    """
    Build a FAISS HNSW index.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape (n, d), float32
    metric : str
        "ip" for inner product, "l2" for L2 distance
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    d = embeddings.shape[1]

    if metric == "ip":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        faiss_metric = faiss.METRIC_L2
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    index = faiss.IndexHNSWFlat(d, m, faiss_metric)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    index.add(embeddings)

    return index


def set_ef_search(index: faiss.Index, ef_search: int) -> None:
    index.hnsw.efSearch = ef_search


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def save_ids(ids: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ids)


def load_ids(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)