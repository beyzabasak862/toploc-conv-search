# ---------------------------------------------------------------------------
# toploc_paper_2 / preprocessing — path-only portability edits:
#   * Hardcoded TRAIN_QUERY_DIR / OUTPUT_DIR replaced with reads from
#     environment variables (config/paths.env): PREPROC_TRAIN_QUERY_DIR and
#     PREPROC_OUTPUT_ROOT (writes to ${PREPROC_OUTPUT_ROOT}/querylog_pca256).
#   * PROJECT_ROOT / sys.path / `from src...` import machinery is unchanged.
#
# PCA dimensions (256), svd_solver, random_state (42), normalization, metric,
# HNSW M / efConstruction / efSearch, Qmax quantile (0.75) and output schema
# are IDENTICAL to the original at msmarco_HNSW/scripts/build_query_log_pca.py
# (SHA256 pair in preprocessing/manifests/COPY_MANIFEST.tsv).
# ---------------------------------------------------------------------------
from pathlib import Path
import os
import sys
import json
import joblib
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loading import load_embeddings_from_parquets, l2_normalize
from src.indexing import build_hnsw_index, save_index, save_ids


def _env_path(var, required=True, default=None):
    v = os.environ.get(var)
    if v:
        return Path(v)
    if default is not None:
        return Path(default)
    if required:
        raise RuntimeError(
            f"[build_query_log_pca] Required environment variable {var!r} is not set. "
            f"Source toploc_paper_2/config/paths.env or export {var!r}."
        )
    return None


# -----------------
# Config (paths from config/paths.env; algorithm constants unchanged)
# -----------------
TRAIN_QUERY_DIR = _env_path("PREPROC_TRAIN_QUERY_DIR")
_OUTPUT_ROOT = _env_path("PREPROC_OUTPUT_ROOT",
                         default=str(PROJECT_ROOT / "_generated"))
OUTPUT_DIR = _OUTPUT_ROOT / "querylog_pca256"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "id"
EMB_COL = "embedding"

NORMALIZE = True
N_COMPONENTS = 256

# HNSW params for query-log index
METRIC = "ip"
M = 32
EF_CONSTRUCTION = 500
EF_SEARCH = 64

QMAX_QUANTILE = 0.75
QMAX_BATCH_SIZE = 2048

def compute_qmax_from_reduced_index(index, reduced_queries: np.ndarray, batch_size: int = 2048) -> float:
    """
    Compute Qmax as the 75th percentile of the nearest-neighbor similarity
    of each historical query to its nearest *other* historical query in the
    reduced query-log space.
    """
    nn_scores = []

    for start in range(0, len(reduced_queries), batch_size):
        end = min(start + batch_size, len(reduced_queries))
        xb = reduced_queries[start:end]

        # top-2 because top-1 is usually the query itself
        D, I = index.search(xb, 2)

        for local_i in range(end - start):
            global_i = start + local_i

            if I[local_i, 0] == global_i:
                nn_scores.append(float(D[local_i, 1]))
            else:
                nn_scores.append(float(D[local_i, 0]))

        print(f"Qmax progress: {end}/{len(reduced_queries)}")

    nn_scores = np.array(nn_scores, dtype=np.float32)
    qmax = float(np.quantile(nn_scores, QMAX_QUANTILE))
    return qmax

def main():
    print("Loading full train_query embeddings...")
    train_query_ids, train_query_embeddings = load_embeddings_from_parquets(
        TRAIN_QUERY_DIR,
        id_col=ID_COL,
        emb_col=EMB_COL,
    )
    print("train_query shape:", train_query_embeddings.shape)

    if NORMALIZE:
        print("Normalizing train_query embeddings...")
        train_query_embeddings = l2_normalize(train_query_embeddings).astype("float32")

    print(f"Fitting PCA to reduce {train_query_embeddings.shape[1]} -> {N_COMPONENTS} ...")
    pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=42)
    train_query_pca = pca.fit_transform(train_query_embeddings).astype(np.float32)

    explained = float(np.sum(pca.explained_variance_ratio_))
    print("Reduced train_query shape:", train_query_pca.shape)
    print("Explained variance ratio sum:", explained)

    print("Building HNSW query-log index on PCA-reduced vectors...")
    querylog_index = build_hnsw_index(
        train_query_pca,
        m=M,
        ef_construction=EF_CONSTRUCTION,
        ef_search=EF_SEARCH,
        metric=METRIC,
    )

    print("Computing Qmax in reduced query-log space...")
    qmax = compute_qmax_from_reduced_index(
        index=querylog_index,
        reduced_queries=train_query_pca,
        batch_size=QMAX_BATCH_SIZE,
    )
    print(f"Computed Qmax: {qmax:.6f}")

    # Save artifacts
    np.save(OUTPUT_DIR / "train_query_pca256.npy", train_query_pca)
    save_ids(train_query_ids, OUTPUT_DIR / "train_query_pca256_ids.npy")
    save_index(querylog_index, OUTPUT_DIR / "train_query_pca256_hnsw.faiss")
    joblib.dump(pca, OUTPUT_DIR / "pca_1024_to_256.joblib")
    np.save(OUTPUT_DIR / "qmax_pca256.npy", np.array(qmax, dtype=np.float32))

    meta = {
        "input_dim": int(train_query_embeddings.shape[1]),
        "output_dim": int(N_COMPONENTS),
        "num_queries": int(len(train_query_ids)),
        "normalized_input": bool(NORMALIZE),
        "explained_variance_ratio_sum": explained,
        "metric": METRIC,
        "M": int(M),
        "ef_construction": int(EF_CONSTRUCTION),
        "ef_search": int(EF_SEARCH),
        "qmax_quantile": float(QMAX_QUANTILE),
        "qmax": float(qmax),
    }
    with open(OUTPUT_DIR / "pca_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:")
    print(" ", OUTPUT_DIR / "train_query_pca256.npy")
    print(" ", OUTPUT_DIR / "train_query_pca256_ids.npy")
    print(" ", OUTPUT_DIR / "train_query_pca256_hnsw.faiss")
    print(" ", OUTPUT_DIR / "pca_1024_to_256.joblib")
    print(" ", OUTPUT_DIR / "qmax_pca256.npy")
    print(" ", OUTPUT_DIR / "pca_meta.json")

if __name__ == "__main__":
    main()
