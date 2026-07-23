from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def parquet_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.parquet"))


def load_embeddings_from_parquets(
    folder: Path,
    id_col: str = "id",
    emb_col: str = "embedding",
    max_files: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ids and embeddings from all parquet shards in a folder.

    Returns
    -------
    ids : np.ndarray of shape (n,)
        Document/query IDs as strings.
    embeddings : np.ndarray of shape (n, d)
        Embeddings as float32.
    """
    files = parquet_files(folder)
    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")

    all_ids: list[str] = []
    all_vecs: list[np.ndarray] = []

    for f in files:
        df = pd.read_parquet(f, columns=[id_col, emb_col])

        ids = df[id_col].astype(str).tolist()
        vecs = np.array(df[emb_col].tolist(), dtype=np.float32)

        all_ids.extend(ids)
        all_vecs.append(vecs)

    embeddings = np.vstack(all_vecs)
    ids = np.array(all_ids, dtype=object)

    return ids, embeddings


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def embedding_norm_stats(x: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(x, axis=1)
    return {
        "min": float(norms.min()),
        "mean": float(norms.mean()),
        "max": float(norms.max()),
        "std": float(norms.std()),
    }


def subset_arrays(
    ids: np.ndarray,
    embeddings: np.ndarray,
    n: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n is None:
        return ids, embeddings
    return ids[:n], embeddings[:n]