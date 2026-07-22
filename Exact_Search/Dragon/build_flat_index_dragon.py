"""
Build an exact-search (flat) FAISS index for Dragon on CAST2019.
 
IndexFlatIP over the RAW Dragon embeddings with METRIC_INNER_PRODUCT:
brute-force dot-product search, i.e. the "Exact" reference of
Muntean et al., SIGIR'25. NO normalization and NO MIPS transformation --
a flat IP index scores <q, x> directly, so the Bachrach trick used for
the HNSW index is unnecessary here (that trick exists only because HNSW
needs an L2/cosine geometry).
 
Resource note: IndexFlat stores every vector verbatim.
  38,636,446 x 768 x 4 bytes ~= 119 GB in RAM while building,
  and the same ~119 GB on disk for the saved index file.
"""
 
import faiss
import numpy as np
import pyarrow.parquet as pq
import glob
import os
import sys
import json
import logging
 
# -----------------------
# Configuration
# -----------------------
DATA_GLOB  = "/home/toploc1/Datasets/conversational/CAST2019/dragon_embeddings/**/*.parquet"
OUTPUT_DIR = "/home/toploc1/Datasets/toploc1/Exact_Search/Dragon"
 
INDEX_PATH      = os.path.join(OUTPUT_DIR, "treccast_flat_dragon.index")
IDS_PATH        = os.path.join(OUTPUT_DIR, "treccast_flat_dragon_ids.npy")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "flat_checkpoint_dragon_ip.json")
LOG_PATH        = os.path.join(OUTPUT_DIR, "indexCreationFlat_dragon.log")
 
RAW_DIM    = 768   # Dragon embedding dimension (no extra MIPS dim here)
SAVE_EVERY = 200   # each checkpoint rewrites the WHOLE index file
                   # (up to ~119 GB of I/O) -- keep this large
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
 
 
def load_embeddings(path):
    table = pq.read_table(path)
    ids = table["id"].to_pylist()
    col = table["embedding"].combine_chunks()
    try:
        flat = col.flatten().to_numpy(zero_copy_only=False)
        emb = np.ascontiguousarray(
            flat.reshape(len(col), -1).astype(np.float32, copy=False))
    except Exception:
        emb = np.array(table["embedding"].to_pylist(), dtype=np.float32)
    return ids, emb
 
 
# -----------------------
# 1. File discovery
# -----------------------
files = sorted(glob.glob(DATA_GLOB, recursive=True))
log.info(f"Total files found: {len(files)}")
if not files:
    log.error("No parquet files found. Check DATA_GLOB path.")
    sys.exit(1)
 
# -----------------------
# 2. Resume or initialize
# -----------------------
start_file = 0
all_ids = []
 
if os.path.exists(CHECKPOINT_PATH) and os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH):
    with open(CHECKPOINT_PATH) as chk:
        ckpt = json.load(chk)
    start_file = ckpt["last_file_index"] + 1
    all_ids = list(np.load(IDS_PATH, allow_pickle=True))
    index = faiss.read_index(INDEX_PATH)
    assert len(all_ids) == index.ntotal, (
        f"Checkpoint inconsistent: {len(all_ids)} ids vs {index.ntotal} vectors. "
        "Delete checkpoint files and rebuild.")
    log.info(f"Resuming from file {start_file}/{len(files)}, "
             f"index has {index.ntotal:,} vectors")
else:
    index = faiss.IndexFlatIP(RAW_DIM)   # exact dot-product search
    log.info(f"Flat IP index created: d={RAW_DIM} (raw Dragon, no transform)")
 
 
def save_checkpoint(last_file_index):
    assert len(all_ids) == index.ntotal, (
        f"ids/vectors out of sync: {len(all_ids)} vs {index.ntotal}")
    faiss.write_index(index, INDEX_PATH)
    np.save(IDS_PATH, np.array(all_ids, dtype=object))
    with open(CHECKPOINT_PATH, "w") as chk:
        json.dump({"last_file_index": last_file_index,
                   "ntotal": index.ntotal}, chk)
    log.info(f"Checkpoint saved ({index.ntotal:,} vectors).")
 
 
# -----------------------
# 3. Add loop (no training needed for a flat index)
# -----------------------
for i, f in enumerate(files[start_file:], start=start_file):
    try:
        ids, emb = load_embeddings(f)
        assert emb.shape[1] == RAW_DIM, \
            f"Expected dim {RAW_DIM}, got {emb.shape[1]} in {f}"
        index.add(emb)                     # RAW vectors: no normalize_L2
        all_ids.extend(ids)
        del emb
    except Exception as e:
        log.error(f"[{i+1}/{len(files)}] Failed on {os.path.basename(f)}: {e}")
        continue
 
    if (i + 1) % 10 == 0:
        log.info(f"[{i+1}/{len(files)}] Indexed: {index.ntotal:,} vectors")
 
    if (i + 1) % SAVE_EVERY == 0:
        log.info(f"Saving checkpoint at file {i+1}...")
        save_checkpoint(i)
 
# -----------------------
# 4. Final save
# -----------------------
log.info(f"Total vectors indexed: {index.ntotal:,}")
save_checkpoint(len(files) - 1)
os.remove(CHECKPOINT_PATH)
log.info("Pipeline complete.")
log.info(f"Index saved to:  {INDEX_PATH}")
log.info(f"IDs saved to:    {IDS_PATH}")