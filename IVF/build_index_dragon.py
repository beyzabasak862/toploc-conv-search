import faiss
import numpy as np
import pyarrow.parquet as pq
import glob
import random
import os
import sys
import json
import logging

faiss.omp_set_num_threads(24)  # 28 cores, leave 4 for OS

# -----------------------
# Configuration
# -----------------------
DATA_GLOB     = "../../conversational/CAST2019/dragon_embeddings/**/*.parquet"
OUTPUT_DIR    = "/home/toploc1/Datasets/toploc1/indexes"

INDEX_PATH      = os.path.join(OUTPUT_DIR, "treccast_dragon_ivf.index")
IDS_PATH        = os.path.join(OUTPUT_DIR, "treccast_dragon_ivf_ids.npy")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "dragon_ivf_checkpoint.json")
LOG_PATH        = os.path.join(OUTPUT_DIR, "dragon_indexCreation.log")

EXPECTED_DIM  = 768    
NLIST         = 32768
N_TRAIN_FILES = 10
SAVE_EVERY    = 20

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
log.info(f"FAISS threads: {faiss.omp_get_max_threads()}")

# -----------------------
# 1. File Discovery
# -----------------------
files = sorted(glob.glob(DATA_GLOB, recursive=True))
log.info(f"Total files found: {len(files)}")

if not files:
    log.error("No parquet files found. Check DATA_GLOB path.")
    sys.exit(1)

# -----------------------
# 2. Resume or Initialize
# -----------------------
start_file = 0
all_ids = []

if os.path.exists(CHECKPOINT_PATH) and os.path.exists(INDEX_PATH) and os.path.exists(IDS_PATH):
    with open(CHECKPOINT_PATH) as chk:
        ckpt = json.load(chk)
    start_file = ckpt["last_file_index"] + 1
    all_ids = list(np.load(IDS_PATH, allow_pickle=True))
    index = faiss.read_index(INDEX_PATH)
    log.info(f"Resuming from file {start_file}/{len(files)}, index has {index.ntotal:,} vectors")
else:
    sample_table = pq.read_table(files[0])
    sample_emb = np.array(sample_table["embedding"].to_pylist(), dtype=np.float32)
    d = sample_emb.shape[1]
    log.info(f"Vector dimension: {d}")

    if d != EXPECTED_DIM:
        log.warning(f"Expected dimension {EXPECTED_DIM}, but got {d}. Update EXPECTED_DIM if this is correct.")

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, NLIST, faiss.METRIC_INNER_PRODUCT)

    log.info(f"Selecting {N_TRAIN_FILES} random files for training...")
    train_files = random.sample(files, min(len(files), N_TRAIN_FILES))
    train_vectors = []
    for tf in train_files:
        try:
            table = pq.read_table(tf)
            emb = np.array(table["embedding"].to_pylist(), dtype=np.float32)
            faiss.normalize_L2(emb)
            train_vectors.append(emb)
        except Exception as e:
            log.warning(f"Skipping training file {tf}: {e}")

    train_vectors = np.vstack(train_vectors)
    log.info(f"Training on {train_vectors.shape[0]:,} vectors (need >= {39 * NLIST:,})...")
    if train_vectors.shape[0] < 39 * NLIST:
        log.warning(
            f"Training set may be too small for {NLIST} centroids. "
            "Consider increasing N_TRAIN_FILES."
        )
    index.train(train_vectors)
    del train_vectors
    log.info("Index training complete.")

# -----------------------
# 3. Add Loop with Checkpointing
# -----------------------
for i, f in enumerate(files[start_file:], start=start_file):
    try:
        table = pq.read_table(f)
        ids = table["id"].to_pylist()
        emb = np.array(table["embedding"].to_pylist(), dtype=np.float32)
        faiss.normalize_L2(emb)
        index.add(emb)
        all_ids.extend(ids)
        del emb, table
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
            json.dump({"last_file_index": i, "ntotal": index.ntotal}, chk)
        log.info("Checkpoint saved.")

# -----------------------
# 4. Final Save
# -----------------------
log.info("Saving final index and ID map...")
faiss.write_index(index, INDEX_PATH)
np.save(IDS_PATH, np.array(all_ids, dtype=object))
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
log.info(f"Pipeline complete. Total vectors indexed: {index.ntotal:,}")
log.info(f"Index saved to:  {INDEX_PATH}")
log.info(f"IDs saved to:    {IDS_PATH}")
