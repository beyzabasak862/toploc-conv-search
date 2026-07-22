import faiss
import numpy as np
import pyarrow.parquet as pq
import glob
import os
import sys
import json
import logging

# -----------------------
# Configurationconversational
# -----------------------
DATA_GLOB = "/home/toploc1/Datasets/conversational/CAST2019/snowflake_embeddings/**/*.parquet"
OUTPUT_DIR = "/home/toploc1/Datasets/toploc1/Exact_Search"

INDEX_PATH      = os.path.join(OUTPUT_DIR, "treccast_flat.index")
IDS_PATH        = os.path.join(OUTPUT_DIR, "treccast_flat_ids.npy")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "flat_checkpoint.json")
LOG_PATH        = os.path.join(OUTPUT_DIR, "indexCreationFlat.log")

SAVE_EVERY = 50   # checkpoint every N files

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
    assert d == 1024, f"Expected dimension 1024, got {d}"

    # Exact search: no training, no quantizer, no nlist.
    # IndexFlatIP over L2-normalized vectors == exact cosine similarity.
    index = faiss.IndexFlatIP(d)
    log.info("Initialized flat (exact) inner-product index.")

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
