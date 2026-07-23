import faiss
import numpy as np
import pyarrow.parquet as pq
import glob
import random
import os
import sys
import json
import logging
import time
 
# NOTE: no RAM limit is set. An IndexIVFFlat stores the FULL vectors in RAM:
#   38.6M vectors x 768 dims x 4 bytes ~= 119 GB for the index alone,
#   plus ~30 GB for the k-means training set, plus working buffers.
# Expect ~160 GB peak usage during the build.
 
# Training 2^18 centroids is compute-bound BLAS work: give it real threads.
N_THREADS = 64
faiss.omp_set_num_threads(N_THREADS)
 
# -----------------------
# Configuration
# -----------------------
DATA_GLOB     = "/home/toploc1/Datasets/conversational/CAST2019/dragon_embeddings/**/*.parquet"
OUTPUT_DIR    = "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes"
 
INDEX_PATH      = os.path.join(OUTPUT_DIR, "treccast_dragon_ivf_2e18.index")
IDS_PATH        = os.path.join(OUTPUT_DIR, "treccast_dragon_ivf_2e18_ids.npy")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "dragon_ivf_2e18_checkpoint.json")
LOG_PATH        = os.path.join(OUTPUT_DIR, "dragon_indexCreation_2e18.log")
 
EXPECTED_DIM  = 768        # Dragon embeddings
NLIST         = 2 ** 18    # 262,144 -- the paper's best Dragon config
                           # (2^15 = 32,768 was their best for SNOWFLAKE, not Dragon)
 
# Dragon is a DOT-PRODUCT model whose vector norms carry information.
# METRIC_INNER_PRODUCT on the RAW embeddings gives true Dragon retrieval;
# L2-normalizing first would silently turn this into cosine search and
# change the rankings (likely the source of your effectiveness gap).
NORMALIZE     = False
 
# k-means wants >= 39 points per centroid or FAISS warns and centroid
# quality degrades. 39 * 2^18 ~= 10.2M training vectors ~= 29 GB fp32.
TRAIN_TARGET  = 20 * NLIST
KMEANS_NITER  = 15         # lower than you might like, but each iteration at
                           # this scale is expensive; 15-25 is a sane range
SEED          = 42
 
# Each checkpoint rewrites the WHOLE index file (grows to ~119 GB by the
# end), which takes minutes of pure I/O. Save rarely, not every 40 files.
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
 
log.info("No RAM limit set (expect ~160 GB peak for this build)")
log.info(f"FAISS threads: {faiss.omp_get_max_threads()}")
log.info(f"NLIST = {NLIST:,} (2^18, paper's Dragon config)")
 
random.seed(SEED)
 
 
def load_embeddings(table):
    """Fast path: pull the list column out through Arrow without the
    to_pylist() detour (which materialises millions of Python floats)."""
    col = table["embedding"].combine_chunks()
    try:
        flat = col.flatten().to_numpy(zero_copy_only=False)
        emb = np.ascontiguousarray(
            flat.reshape(len(col), -1).astype(np.float32, copy=False))
    except Exception:
        emb = np.array(table["embedding"].to_pylist(), dtype=np.float32)
    return emb
 
 
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
    assert len(all_ids) == index.ntotal, (
        f"Checkpoint inconsistent: {len(all_ids)} ids vs {index.ntotal} vectors. "
        "Delete checkpoint files and rebuild.")
    log.info(f"Resuming from file {start_file}/{len(files)}, index has {index.ntotal:,} vectors")
else:
    sample_table = pq.read_table(files[0])
    d = load_embeddings(sample_table).shape[1]
    log.info(f"Vector dimension: {d}")
    if d != EXPECTED_DIM:
        log.warning(f"Expected dimension {EXPECTED_DIM}, but got {d}.")
 
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, NLIST, faiss.METRIC_INNER_PRODUCT)
    index.cp.niter = KMEANS_NITER
    index.cp.seed = SEED
    index.verbose = True
 
    # --- adaptive training sample: keep reading random files until we have
    #     enough vectors, instead of a fixed N_TRAIN_FILES guess ---
    shuffled = files[:]
    random.shuffle(shuffled)
    train_chunks, n_train = [], 0
    for tf in shuffled:
        if n_train >= TRAIN_TARGET:
            break
        try:
            emb = load_embeddings(pq.read_table(tf))
            if NORMALIZE:
                faiss.normalize_L2(emb)
            train_chunks.append(emb)
            n_train += emb.shape[0]
            log.info(f"  training sample: +{emb.shape[0]:,} from "
                     f"{os.path.basename(tf)} (total {n_train:,}/{TRAIN_TARGET:,})")
        except Exception as e:
            log.warning(f"Skipping training file {tf}: {e}")
 
    train_vectors = np.vstack(train_chunks)
    del train_chunks
    if train_vectors.shape[0] < TRAIN_TARGET:
        log.warning(f"Only {train_vectors.shape[0]:,} training vectors available "
                    f"(target {TRAIN_TARGET:,}); centroid quality may suffer.")
 
    log.info(f"Training {NLIST:,} centroids on {train_vectors.shape[0]:,} vectors, "
             f"niter={KMEANS_NITER}. THIS WILL TAKE HOURS ON CPU -- "
             "use faiss-gpu for the train step if you have a GPU.")
    t0 = time.time()
    index.train(train_vectors)
    log.info(f"Index training complete in {(time.time() - t0) / 3600:.2f} h.")
    del train_vectors
 
# -----------------------
# 3. Add Loop with Checkpointing
# -----------------------
def save_checkpoint(last_file_index):
    assert len(all_ids) == index.ntotal, (
        f"ids/vectors out of sync: {len(all_ids)} vs {index.ntotal}")
    t0 = time.time()
    faiss.write_index(index, INDEX_PATH)
    np.save(IDS_PATH, np.array(all_ids, dtype=object))
    with open(CHECKPOINT_PATH, "w") as chk:
        json.dump({"last_file_index": last_file_index, "ntotal": index.ntotal}, chk)
    log.info(f"Checkpoint saved ({index.ntotal:,} vectors, "
             f"{time.time() - t0:.0f}s write time).")
 
 
for i, f in enumerate(files[start_file:], start=start_file):
    try:
        table = pq.read_table(f)
        ids = table["id"].to_pylist()
        emb = load_embeddings(table)
        if NORMALIZE:
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
        save_checkpoint(i)
 
# -----------------------
# 4. Final Save
# -----------------------
log.info("Saving final index and ID map...")
save_checkpoint(len(files) - 1)
os.remove(CHECKPOINT_PATH)
log.info(f"Pipeline complete. Total vectors indexed: {index.ntotal:,}")
log.info(f"Index saved to:  {INDEX_PATH}")
log.info(f"IDs saved to:    {IDS_PATH}")