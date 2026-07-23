# ---------------------------------------------------------------------------
# toploc_paper_2 / preprocessing — path-only portability edits:
#   * Hardcoded TRAIN_CONTEXT_DIR / OUTPUT_DIR replaced with reads from
#     environment variables (config/paths.env): PREPROC_DOC_EMB_DIR and
#     PREPROC_OUTPUT_ROOT (writes to ${PREPROC_OUTPUT_ROOT}/index_artifacts).
#   * PROJECT_ROOT / sys.path / `from src...` import machinery is unchanged
#     (it already derives from __file__ and resolves to preprocessing/src/).
#
# FAISS index type, HNSW M, efConstruction, efSearch, metric, normalization,
# document order and output schema are IDENTICAL to the original at
# msmarco_HNSW/scripts/build_index.py (SHA256 pair in
# preprocessing/manifests/COPY_MANIFEST.tsv).
# ---------------------------------------------------------------------------
from pathlib import Path
import os
import sys

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
            f"[build_index] Required environment variable {var!r} is not set. "
            f"Source toploc_paper_2/config/paths.env or export {var!r}."
        )
    return None


# -----------------
# Config (paths from config/paths.env; algorithm constants unchanged)
# -----------------
TRAIN_CONTEXT_DIR = _env_path("PREPROC_DOC_EMB_DIR")
_OUTPUT_ROOT = _env_path("PREPROC_OUTPUT_ROOT",
                         default=str(PROJECT_ROOT / "_generated"))
OUTPUT_DIR = _OUTPUT_ROOT / "index_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "id"
EMB_COL = "embedding"

NORMALIZE = True
METRIC = "ip"

M = 32
EF_CONSTRUCTION = 500
EF_SEARCH = 64  # default stored in index

def main():
    print("Loading document embeddings...")
    doc_ids, doc_embeddings = load_embeddings_from_parquets(
        TRAIN_CONTEXT_DIR,
        id_col=ID_COL,
        emb_col=EMB_COL,
    )

    print("Docs:", doc_embeddings.shape)

    if NORMALIZE:
        print("Normalizing document embeddings...")
        doc_embeddings = l2_normalize(doc_embeddings).astype("float32")

    print("Building HNSW index...")
    hnsw_index = build_hnsw_index(
        doc_embeddings,
        m=M,
        ef_construction=EF_CONSTRUCTION,
        ef_search=EF_SEARCH,
        metric=METRIC,
    )

    save_index(hnsw_index, OUTPUT_DIR / "train_query_full_hnsw.faiss")
    save_ids(doc_ids, OUTPUT_DIR / "train_query_full_ids.npy")

    print("Index saved.")
    print("Index path:", OUTPUT_DIR / "train_query_full_hnsw.faiss")
    print("IDs path:", OUTPUT_DIR / "train_query_full_ids.npy")

if __name__ == "__main__":
    main()
