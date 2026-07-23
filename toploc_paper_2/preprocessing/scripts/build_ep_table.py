# ---------------------------------------------------------------------------
# toploc_paper_2 / preprocessing — path-only portability edits:
#   * Hardcoded TRAIN_QUERY_DIR / DOC_INDEX_DIR / OUTPUT_DIR replaced with reads
#     from environment variables (config/paths.env):
#       PREPROC_TRAIN_QUERY_DIR  (train query embeddings)
#       HYBRID_DOC_INDEX         (the 158 GB TREC-CAsT doc index, reused)
#       PREPROC_OUTPUT_ROOT      (writes ${PREPROC_OUTPUT_ROOT}/qlr_artifacts_full)
#   * PROJECT_ROOT / sys.path / `from src...` import machinery is unchanged.
#   * The commented-out `from src.search import hnsw_search_with_latency` line
#     from the original is preserved as-is (it was already inactive).
#
# TOPK_EP (10), normalization, document-ID space and output schema are IDENTICAL
# to the original at msmarco_HNSW/scripts/build_ep_table.py (SHA256 pair in
# preprocessing/manifests/COPY_MANIFEST.tsv).
# ---------------------------------------------------------------------------
from pathlib import Path
import os
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loading import load_embeddings_from_parquets, l2_normalize
from src.indexing import load_index
#from src.search import hnsw_search_with_latency


def _env_path(var, required=True, default=None):
    v = os.environ.get(var)
    if v:
        return Path(v)
    if default is not None:
        return Path(default)
    if required:
        raise RuntimeError(
            f"[build_ep_table] Required environment variable {var!r} is not set. "
            f"Source toploc_paper_2/config/paths.env or export {var!r}."
        )
    return None


# -----------------
# Config (paths from config/paths.env; algorithm constants unchanged)
# -----------------
TRAIN_QUERY_DIR = _env_path("PREPROC_TRAIN_QUERY_DIR")
# The document HNSW index over which entry points are searched. Reuse the same
# full TREC-CAsT index the benchmarks consume (HYBRID_DOC_INDEX is the file path).
DOC_INDEX_PATH = _env_path("HYBRID_DOC_INDEX")
_OUTPUT_ROOT = _env_path("PREPROC_OUTPUT_ROOT",
                         default=str(PROJECT_ROOT / "_generated"))
OUTPUT_DIR = _OUTPUT_ROOT / "qlr_artifacts_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "id"
EMB_COL = "embedding"

NORMALIZE = True
TOPK_EP = 10

def main():
    print("Loading train_query embeddings...")
    train_query_ids, train_query_embeddings = load_embeddings_from_parquets(
        TRAIN_QUERY_DIR,
        id_col=ID_COL,
        emb_col=EMB_COL,
    )

    print("train_query embeddings shape:", train_query_embeddings.shape)

    if NORMALIZE:
        print("Normalizing train_query embeddings...")
        train_query_embeddings = l2_normalize(train_query_embeddings).astype("float32")

    print("Loading document HNSW index...")
    doc_index = load_index(DOC_INDEX_PATH)

    # print(f"Searching document index for top-{TOPK_EP} cached entry points...")
    # ep_distances, ep_indices, latency_us = hnsw_search_with_latency(
    #     doc_index,
    #     train_query_embeddings,
    #     topk=TOPK_EP,
    # )
    print(f"Searching document index for top-{TOPK_EP} cached entry points...")
    ep_distances, ep_indices = doc_index.search(train_query_embeddings, TOPK_EP)

    print("EP indices shape:", ep_indices.shape)
    print("EP distances shape:", ep_distances.shape)

    np.save(OUTPUT_DIR / "ep_indices.npy", ep_indices.astype(np.int32))
    np.save(OUTPUT_DIR / "ep_distances.npy", ep_distances.astype(np.float32))
    np.save(OUTPUT_DIR / "train_query_ids.npy", train_query_ids)

    print("Saved:")
    print(" ", OUTPUT_DIR / "ep_indices.npy")
    print(" ", OUTPUT_DIR / "ep_distances.npy")
    print(" ", OUTPUT_DIR / "train_query_ids.npy")

if __name__ == "__main__":
    main()
