# ---------------------------------------------------------------------------
# toploc_paper_2 / preprocessing — path-only portability edits:
#   * Hardcoded DEV_QUERY_DIR / FLAT_INDEX_PATH / OUTPUT_DIR replaced with reads
#     from environment variables (config/paths.env):
#       DEV_QUERY_DIR       (dev queries — reused from the hybrid benchmark var)
#       PREPROC_FLAT_INDEX  (exact flat index, e.g. treccast_flat.index)
#       PREPROC_OUTPUT_ROOT (writes ${PREPROC_OUTPUT_ROOT}/exact_results_full)
#   * PROJECT_ROOT / sys.path / `from src...` import machinery is unchanged.
#
# TOPK (10), BATCH_SIZE (512), normalization, metric and output schema are
# IDENTICAL to the original at msmarco_HNSW/scripts/flat_index_search_acc.py
# (SHA256 pair in preprocessing/manifests/COPY_MANIFEST.tsv).
# ---------------------------------------------------------------------------
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loading import load_embeddings_from_parquets, l2_normalize
from src.indexing import load_index


def _env_path(var, required=True, default=None):
    v = os.environ.get(var)
    if v:
        return Path(v)
    if default is not None:
        return Path(default)
    if required:
        raise RuntimeError(
            f"[flat_index_search_acc] Required environment variable {var!r} is not set. "
            f"Source toploc_paper_2/config/paths.env or export {var!r}."
        )
    return None


# -----------------
# Config (paths from config/paths.env; algorithm constants unchanged)
# -----------------
DEV_QUERY_DIR = _env_path("DEV_QUERY_DIR")
FLAT_INDEX_PATH = _env_path("PREPROC_FLAT_INDEX")
_OUTPUT_ROOT = _env_path("PREPROC_OUTPUT_ROOT",
                         default=str(PROJECT_ROOT / "_generated"))
OUTPUT_DIR = _OUTPUT_ROOT / "exact_results_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "id"
EMB_COL = "embedding"

NORMALIZE = True
TOPK = 10
BATCH_SIZE = 512

def main():
    print("Loading dev_query embeddings...")
    dev_query_ids, dev_query_embeddings = load_embeddings_from_parquets(
        DEV_QUERY_DIR,
        id_col=ID_COL,
        emb_col=EMB_COL,
    )

    print("Dev queries shape:", dev_query_embeddings.shape)

    if NORMALIZE:
        print("Normalizing dev_query embeddings...")
        dev_query_embeddings = l2_normalize(dev_query_embeddings).astype("float32")

    print("Loading flat index...")
    flat_index = load_index(FLAT_INDEX_PATH)
    print("Flat index dim:", flat_index.d)
    print("Flat index ntotal:", flat_index.ntotal)

    if flat_index.d != dev_query_embeddings.shape[1]:
        raise ValueError(
            f"Dimension mismatch: flat index dim={flat_index.d}, "
            f"dev_query dim={dev_query_embeddings.shape[1]}"
        )

    print(f"Running exact search with top-{TOPK}...")
    all_scores = []
    all_indices = []

    for start in range(0, len(dev_query_embeddings), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(dev_query_embeddings))
        D, I = flat_index.search(dev_query_embeddings[start:end], TOPK)
        all_scores.append(D)
        all_indices.append(I)
        print(f"Processed {end}/{len(dev_query_embeddings)} queries")

    exact_scores = np.vstack(all_scores).astype(np.float32)
    exact_indices = np.vstack(all_indices).astype(np.int64)

    print("Exact scores shape:", exact_scores.shape)
    print("Exact indices shape:", exact_indices.shape)

    np.save(OUTPUT_DIR / "exact_scores.npy", exact_scores)
    np.save(OUTPUT_DIR / "exact_indices.npy", exact_indices)
    np.save(OUTPUT_DIR / "dev_query_ids.npy", dev_query_ids)

    meta = pd.DataFrame([{
        "method": "exact_flat",
        "topk": TOPK,
        "num_queries": len(dev_query_ids),
        "index_path": str(FLAT_INDEX_PATH),
        "batch_size": BATCH_SIZE,
        "normalized_queries": NORMALIZE,
    }])
    meta.to_csv(OUTPUT_DIR / "exact_meta.csv", index=False)

    print("Saved:")
    print(" ", OUTPUT_DIR / "exact_scores.npy")
    print(" ", OUTPUT_DIR / "exact_indices.npy")
    print(" ", OUTPUT_DIR / "dev_query_ids.npy")
    print(" ", OUTPUT_DIR / "exact_meta.csv")

if __name__ == "__main__":
    main()
