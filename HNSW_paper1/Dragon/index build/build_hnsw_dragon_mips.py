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
OUTPUT_DIR = "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes"

INDEX_PATH      = os.path.join(OUTPUT_DIR, "treccast_hnsw_M32_dragon_mips.index")
IDS_PATH        = os.path.join(OUTPUT_DIR, "treccast_hnsw_idsM32_dragon_mips.npy")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "hnsw_checkpoint_M32_dragon_mips.json")
MAXNORM_PATH    = os.path.join(OUTPUT_DIR, "dragon_mips_maxnorm.json")
LOG_PATH        = os.path.join(OUTPUT_DIR, "indexCreationHNSW_M32_dragon_mips.log")

M               = 32   # graph connectivity (16-64 typical)
EF_CONSTRUCTION = 500  # build quality: higher = better graph, slower build
EF_SEARCH       = 64   # query-time accuracy/speed tradeoff
SAVE_EVERY      = 50   # checkpoint every N files
RAW_DIM         = 768  # Dragon embedding dimension (index will be RAW_DIM + 1)

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

# -----------------------
# MIPS -> L2 transformation (Bachrach et al., RecSys'14; used by
# Muntean et al. SIGIR'25 for Dragon):
#   docs   : phi(x) = [x, sqrt(Mmax^2 - ||x||^2)]   (all norms == Mmax)
#   queries: q'     = [q, 0]
# Then  ||q' - phi(x)||^2 = ||q||^2 + Mmax^2 - 2<q, x>,
# so L2 ranking on the transformed vectors == inner-product ranking
# on the raw vectors. DO NOT normalize anything.
# -----------------------

# -----------------------
# 1. File discovery
# -----------------------
files = sorted(glob.glob(DATA_GLOB, recursive=True))
log.info(f"Total files found: {len(files)}")
if not files:
    log.error("No parquet files found. Check DATA_GLOB path.")
    sys.exit(1)


def load_embeddings(path):
    table = pq.read_table(path)
    ids = table["id"].to_pylist()
    emb = np.array(table["embedding"].to_pylist(), dtype=np.float32)
    return ids, emb


# -----------------------
# 2. Pass 1: global max norm over the WHOLE collection.
#    Required before any vector can be transformed. Cheap (read + norm),
#    checkpointed so it survives interruptions.
# -----------------------
if os.path.exists(MAXNORM_PATH):
    with open(MAXNORM_PATH) as f:
        mn = json.load(f)
    if mn.get("complete", False):
        MAX_NORM = float(mn["max_norm"])
        log.info(f"Pass 1 already complete: max_norm={MAX_NORM:.6f}")
    else:
        start = mn["last_file_index"] + 1
        MAX_NORM = float(mn["max_norm"])
        log.info(f"Resuming pass 1 from file {start}, running max={MAX_NORM:.6f}")
else:
    start, MAX_NORM = 0, 0.0

if not (os.path.exists(MAXNORM_PATH) and json.load(open(MAXNORM_PATH)).get("complete", False)):
    for i, f in enumerate(files[start:], start=start):
        _, emb = load_embeddings(f)
        assert emb.shape[1] == RAW_DIM, f"Expected dim {RAW_DIM}, got {emb.shape[1]}"
        MAX_NORM = max(MAX_NORM, float(np.linalg.norm(emb, axis=1).max()))
        if (i + 1) % 50 == 0:
            log.info(f"[pass1 {i+1}/{len(files)}] running max_norm={MAX_NORM:.6f}")
            with open(MAXNORM_PATH, "w") as fh:
                json.dump({"last_file_index": i, "max_norm": MAX_NORM,
                           "complete": False}, fh)
    with open(MAXNORM_PATH, "w") as fh:
        json.dump({"last_file_index": len(files) - 1, "max_norm": MAX_NORM,
                   "complete": True}, fh)
    log.info(f"Pass 1 complete: max_norm={MAX_NORM:.6f}")

MAX_NORM_SQ = MAX_NORM * MAX_NORM


def mips_transform(emb):
    """(n, 768) raw Dragon -> (n, 769) Bachrach-transformed, float32."""
    sq = np.einsum("ij,ij->i", emb, emb)
    # numerical guard: sq can exceed MAX_NORM_SQ by float error for the
    # argmax vector itself
    extra = np.sqrt(np.maximum(MAX_NORM_SQ - sq, 0.0)).astype(np.float32)
    return np.ascontiguousarray(np.hstack([emb, extra[:, None]]))


# -----------------------
# 3. Pass 2: build the 769-dim HNSW index (resume-capable)
# -----------------------
D_INDEX = RAW_DIM + 1
start_file = 0
all_ids = []

if os.path.exists(CHECKPOINT_PATH) and os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH):
    with open(CHECKPOINT_PATH) as chk:
        ckpt = json.load(chk)
    assert abs(ckpt["max_norm"] - MAX_NORM) < 1e-6, \
        "Checkpoint was built with a different max_norm - delete and rebuild."
    start_file = ckpt["last_file_index"] + 1
    all_ids = list(np.load(IDS_PATH, allow_pickle=True))
    index = faiss.read_index(INDEX_PATH)
    log.info(f"Resuming from file {start_file}/{len(files)}, "
             f"index has {index.ntotal:,} vectors")
else:
    index = faiss.IndexHNSWFlat(D_INDEX, M)   # METRIC_L2 by construction
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH
    log.info(f"HNSW index created: d={D_INDEX} (768 + 1 MIPS dim), M={M}, "
             f"efConstruction={EF_CONSTRUCTION}")

for i, f in enumerate(files[start_file:], start=start_file):
    try:
        ids, emb = load_embeddings(f)
        index.add(mips_transform(emb))        # NO normalize_L2
        all_ids.extend(ids)
    except Exception as e:
        log.error(f"[{i+1}/{len(files)}] Failed on {os.path.basename(f)}: {e}")
        continue

    if (i + 1) % 10 == 0:
        log.info(f"[{i+1}/{len(files)}] Indexed: {index.ntotal:,} vectors")

    if (i + 1) % SAVE_EVERY == 0:
        log.info(f"Saving checkpoint at file {i+1}...")
        faiss.write_index(index, INDEX_PATH)
        np.save(IDS_PATH, np.array(all_ids, dtype=object))
        with open(CHECKPOINT_PATH, "w") as chk:
            json.dump({"last_file_index": i, "ntotal": index.ntotal,
                       "max_norm": MAX_NORM}, chk)
        log.info("Checkpoint saved.")

# -----------------------
# 4. Final save
# -----------------------
log.info(f"Total vectors indexed: {index.ntotal:,}")
faiss.write_index(index, INDEX_PATH)
np.save(IDS_PATH, np.array(all_ids, dtype=object))
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
log.info("Pipeline complete.")
log.info(f"Index saved to:  {INDEX_PATH}")
log.info(f"IDs saved to:    {IDS_PATH}")
log.info(f"max_norm saved:  {MAXNORM_PATH}  <- needed at query time!")
