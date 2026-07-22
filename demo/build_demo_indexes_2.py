"""
build_demo_indexes.py
=====================
Phase 2 of 2: Build Flat / IVF / HNSW FAISS indexes from the saved demo
collection produced by build_demo_collection.py.

Reads
-----
  demo_embeddings.parquet  – id | embedding
  demo_metadata.json       – provenance / stats

Writes
------
  demo_ids.npy             – integer position → original doc-ID string
  demo_flat.index          – exact inner-product index
  demo_ivf.index           – IVF approximate index
  demo_hnsw.index          – HNSW approximate index
  demo_metadata.json       – updated in-place with index paths & params
"""

import os
import sys
import json
import logging
import numpy as np
import pyarrow.parquet as pq
import faiss

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_DIR = "/home/toploc1/Datasets/toploc1/demo"

# These must match what build_demo_collection.py produced
DEMO_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "demo_embeddings.parquet")
METADATA_PATH        = os.path.join(OUTPUT_DIR, "demo_metadata.json")

# Index hyperparameters
NPROBE_DEMO       = 8    # nprobe stored in the IVF index at query time
HNSW_M            = 32   # HNSW connectivity parameter
HNSW_EF_CONSTRUCT = 200

# Column names (must match Phase 1)
COL_ID        = "id"
COL_EMBEDDING = "embedding"

# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIR, "build_demo_indexes.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

DEMO_IDS_PATH   = os.path.join(OUTPUT_DIR, "demo_ids.npy")
FLAT_INDEX_PATH = os.path.join(OUTPUT_DIR, "demo_flat.index")
IVF_INDEX_PATH  = os.path.join(OUTPUT_DIR, "demo_ivf.index")
HNSW_INDEX_PATH = os.path.join(OUTPUT_DIR, "demo_hnsw.index")


# ══════════════════════════════════════════════
# STEP 1 – load embeddings
# ══════════════════════════════════════════════
log.info("=== Step 1: Load demo embeddings ===")
if not os.path.exists(DEMO_EMBEDDINGS_PATH):
    log.error(f"Embeddings file not found: {DEMO_EMBEDDINGS_PATH}")
    log.error("Run build_demo_collection.py first.")
    sys.exit(1)

tbl      = pq.read_table(DEMO_EMBEDDINGS_PATH, columns=[COL_ID, COL_EMBEDDING])
ids_out  = tbl[COL_ID].to_pylist()
embs_raw = tbl[COL_EMBEDDING].to_pylist()

emb_matrix = np.array(embs_raw, dtype=np.float32)
faiss.normalize_L2(emb_matrix)
N, D = emb_matrix.shape
log.info(f"Loaded {N:,} embeddings, dim={D}")

# Save the ordered ID map so index positions map back to doc IDs
np.save(DEMO_IDS_PATH, np.array(ids_out))
log.info(f"ID map saved → {DEMO_IDS_PATH}")


# ══════════════════════════════════════════════
# STEP 2 – Flat (exact) index
# ══════════════════════════════════════════════
log.info("=== Step 2: Build Flat index ===")
flat_index = faiss.IndexFlatIP(D)
flat_index.add(emb_matrix)
faiss.write_index(flat_index, FLAT_INDEX_PATH)
log.info(f"Flat index saved ({flat_index.ntotal:,} vectors) → {FLAT_INDEX_PATH}")


# ══════════════════════════════════════════════
# STEP 3 – IVF index
# ══════════════════════════════════════════════
log.info("=== Step 3: Build IVF index ===")
nlist = max(4, int(np.sqrt(N)))   # rule-of-thumb: sqrt(N)
nlist = min(nlist, N // 10)       # but at most N/10 to allow training
log.info(f"  nlist={nlist}")

quantiser = faiss.IndexFlatIP(D)
ivf_index = faiss.IndexIVFFlat(quantiser, D, nlist, faiss.METRIC_INNER_PRODUCT)
ivf_index.train(emb_matrix)
ivf_index.add(emb_matrix)
ivf_index.nprobe = NPROBE_DEMO
faiss.write_index(ivf_index, IVF_INDEX_PATH)
log.info(f"IVF index saved ({ivf_index.ntotal:,} vectors) → {IVF_INDEX_PATH}")


# ══════════════════════════════════════════════
# STEP 4 – HNSW index
# ══════════════════════════════════════════════
log.info("=== Step 4: Build HNSW index ===")
hnsw_index = faiss.IndexHNSWFlat(D, HNSW_M, faiss.METRIC_INNER_PRODUCT)
hnsw_index.hnsw.efConstruction = HNSW_EF_CONSTRUCT
hnsw_index.add(emb_matrix)
faiss.write_index(hnsw_index, HNSW_INDEX_PATH)
log.info(f"HNSW index saved ({hnsw_index.ntotal:,} vectors) → {HNSW_INDEX_PATH}")


# ══════════════════════════════════════════════
# STEP 5 – update metadata with index info
# ══════════════════════════════════════════════
log.info("=== Step 5: Update metadata ===")
metadata = {}
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH) as fh:
        metadata = json.load(fh)

metadata.update({
    "ivf_nlist":            nlist,
    "ivf_nprobe_demo":      NPROBE_DEMO,
    "hnsw_M":               HNSW_M,
    "hnsw_ef_construction": HNSW_EF_CONSTRUCT,
    "demo_ids_path":        DEMO_IDS_PATH,
    "flat_index_path":      FLAT_INDEX_PATH,
    "ivf_index_path":       IVF_INDEX_PATH,
    "hnsw_index_path":      HNSW_INDEX_PATH,
})
with open(METADATA_PATH, "w") as fh:
    json.dump(metadata, fh, indent=2)
log.info(f"Metadata updated → {METADATA_PATH}")
log.info("=== Phase 2 COMPLETE ===")
