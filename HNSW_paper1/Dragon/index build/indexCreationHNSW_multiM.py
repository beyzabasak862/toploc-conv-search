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
OUTPUT_DIR = "/home/toploc1/Datasets/toploc1/indexes/Dragon/HNSW"
LOG_OUTPUT_PATH = "/home/toploc1/Datasets/toploc1/HNSW_paper1/Dragon/index build/logs"
# Build one HNSW index per M value. Add 64 here if you want it too.
M_LIST          = [16, 64]
EF_CONSTRUCTION = 500  # build quality: higher = better graph, slower build
EF_SEARCH       = 64   # query-time accuracy/speed tradeoff (grid overrides this later)
SAVE_EVERY      = 50   # checkpoint every N files
RAW_DIM         = 768  # Dragon embedding dimension (index will be RAW_DIM + 1)

# max_norm is a property of the COLLECTION, not of M -- computed once,
# shared by every M index.
MAXNORM_PATH = os.path.join(OUTPUT_DIR, "dragon_mips_maxnorm.json")
LOG_PATH     = os.path.join(LOG_OUTPUT_PATH, "indexCreationHNSW_dragon_multiM.log")

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


def paths_for_m(m):
    """Per-M filenames so the indexes never collide."""
    return {
        "index":      os.path.join(OUTPUT_DIR, f"treccast_hnsw_M{m}_dragon_mips.index"),
        "ids":        os.path.join(OUTPUT_DIR, f"treccast_hnsw_idsM{m}_dragon_mips.npy"),
        "checkpoint": os.path.join(OUTPUT_DIR, f"hnsw_checkpoint_M{m}_dragon_mips.json"),
    }

# -----------------------
# MIPS -> L2 transformation (Bachrach et al., RecSys'14; used by
# Muntean et al. SIGIR'25 for Dragon):
#   docs   : phi(x) = [x, sqrt(Mmax^2 - ||x||^2)]   (all norms == Mmax)
#   queries: q'     = [q, 0]
# Then  ||q' - phi(x)||^2 = ||q||^2 + Mmax^2 - 2<q, x>,
# so L2 ranking on the transformed vectors == inner-product ranking
# on the raw vectors. DO NOT normalize anything.
# The transform is IDENTICAL for every M -- only the graph parameter M differs.
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
# 2. Pass 1: global max norm over the WHOLE collection (M-independent).
#    Required before any vector can be transformed. Cheap (read + norm),
#    checkpointed so it survives interruptions. Runs ONCE for all M.
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
# 3. Pass 2: build the 769-dim HNSW index for one M value (resume-capable).
# -----------------------
D_INDEX = RAW_DIM + 1


def build_for_m(m):
    p = paths_for_m(m)
    log.info(f"================  BUILDING HNSW  M={m}  ================")

    # already finished on a previous run? (index exists, checkpoint gone)
    if os.path.exists(p["index"]) and os.path.exists(p["ids"]) \
            and not os.path.exists(p["checkpoint"]):
        idx = faiss.read_index(p["index"])
        log.info(f"[M={m}] already complete: {idx.ntotal:,} vectors -> skipping")
        return

    start_file = 0
    all_ids = []

    if os.path.exists(p["checkpoint"]) and os.path.exists(p["index"]) and os.path.exists(p["ids"]):
        with open(p["checkpoint"]) as chk:
            ckpt = json.load(chk)
        assert abs(ckpt["max_norm"] - MAX_NORM) < 1e-6, \
            f"[M={m}] checkpoint built with a different max_norm - delete and rebuild."
        start_file = ckpt["last_file_index"] + 1
        all_ids = list(np.load(p["ids"], allow_pickle=True))
        index = faiss.read_index(p["index"])
        log.info(f"[M={m}] resuming from file {start_file}/{len(files)}, "
                 f"index has {index.ntotal:,} vectors")
    else:
        index = faiss.IndexHNSWFlat(D_INDEX, m)   # METRIC_L2 by construction
        index.hnsw.efConstruction = EF_CONSTRUCTION
        index.hnsw.efSearch = EF_SEARCH
        log.info(f"[M={m}] HNSW index created: d={D_INDEX} (768 + 1 MIPS dim), "
                 f"M={m}, efConstruction={EF_CONSTRUCTION}")

    for i, f in enumerate(files[start_file:], start=start_file):
        try:
            ids, emb = load_embeddings(f)
            index.add(mips_transform(emb))        # NO normalize_L2
            all_ids.extend(ids)
        except Exception as e:
            log.error(f"[M={m}][{i+1}/{len(files)}] Failed on {os.path.basename(f)}: {e}")
            continue

        if (i + 1) % 10 == 0:
            log.info(f"[M={m}][{i+1}/{len(files)}] Indexed: {index.ntotal:,} vectors")

        if (i + 1) % SAVE_EVERY == 0:
            log.info(f"[M={m}] Saving checkpoint at file {i+1}...")
            faiss.write_index(index, p["index"])
            np.save(p["ids"], np.array(all_ids, dtype=object))
            with open(p["checkpoint"], "w") as chk:
                json.dump({"last_file_index": i, "ntotal": index.ntotal,
                           "max_norm": MAX_NORM}, chk)
            log.info(f"[M={m}] Checkpoint saved.")

    # final save for this M
    log.info(f"[M={m}] Total vectors indexed: {index.ntotal:,}")
    faiss.write_index(index, p["index"])
    np.save(p["ids"], np.array(all_ids, dtype=object))
    if os.path.exists(p["checkpoint"]):
        os.remove(p["checkpoint"])
    log.info(f"[M={m}] complete.")
    log.info(f"[M={m}] Index saved to:  {p['index']}")
    log.info(f"[M={m}] IDs saved to:    {p['ids']}")

    # free RAM before the next M (each index is ~119 GB)
    del index
    all_ids.clear()


# -----------------------
# 4. Build every M sequentially
# -----------------------
for m in M_LIST:
    build_for_m(m)

log.info("All requested M indexes complete.")
log.info(f"M values built: {M_LIST}")
log.info(f"max_norm ({MAX_NORM:.6f}) saved: {MAXNORM_PATH}  <- needed at query time!")