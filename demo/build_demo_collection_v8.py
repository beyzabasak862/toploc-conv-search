"""
build_demo_collection.py
========================
Phase 1 of 2: Build and save the demo collection (text + embeddings separately).

Saves
-----
  demo_texts.parquet       – id | text
  demo_embeddings.parquet  – id | embedding
  demo_metadata.json       – provenance / stats

Run Phase 2 (build_demo_indexes.py) afterwards to create FAISS indexes.

Data sources
------------
- Embeddings : one directory of sharded Parquet files  (id | embedding)
- Text        : a single TSV file                       (id \\t text, with header)

Pipeline
--------
1. Parse qrels  → unique query IDs
2. Load query embeddings for those IDs
3. Search the full IVF index (top-100 per query)  → relevant pool
4. Sample an equal-sized noise pool from the rest of the corpus
5. Fetch text (from TSV) + embeddings (from sharded Parquet) for demo docs
6. Save texts and embeddings as separate Parquet files
"""

import os
import sys
import json
import random
import logging
import gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import faiss
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION  – edit these paths as needed
# ─────────────────────────────────────────────
FULL_INDEX_PATH   = "/home/toploc1/Datasets/toploc1/indexes/treccast_34M_ivf.index"
FULL_IDS_PATH     = "/home/toploc1/Datasets/toploc1/indexes/treccast_34M_ivf_ids.npy"

EMBEDDINGS_DIR    = "/home/toploc1/Datasets/conversational/CAST2019/snowflake_embeddings"
COLLECTION_TSV    = "/home/toploc1/Datasets/conversational/CAST2019/CAST2019collection.tsv"
QUERY_EMB_PATH    = "/home/toploc1/Datasets/conversational/CAST2019/topics/topics_snowflake_embeddings.parquet"
QRELS_PATH        = "/home/toploc1/Datasets/conversational/CAST2019/topics/qrels.qrel"

OUTPUT_DIR        = "/home/toploc1/Datasets/toploc1/demo"

# Retrieval settings
TOP_K_BUILD       = 100
RANDOM_SEED       = 42
NPROBE_FULL       = 64

# Column names in the embedding Parquet shards (adjust if yours differ)
COL_ID        = "id"
COL_EMBEDDING = "embedding"
COL_TEXT      = "text"

# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_DIR, "build_demo_collection.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Output paths (collection only – no indexes here)
DEMO_TEXTS_PATH      = os.path.join(OUTPUT_DIR, "demo_texts.parquet")
DEMO_EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "demo_embeddings.parquet")
METADATA_PATH        = os.path.join(OUTPUT_DIR, "demo_metadata.json")


# ══════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════

def parse_qrels(path: str) -> dict:
    """Return {qid: {docid: grade}} from a comma-separated qrel file."""
    qrels = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            qid, docid, grade = parts[0], parts[2], int(parts[3])
            qrels[qid][docid] = grade
    log.info(f"qrels: {len(qrels)} queries, "
             f"{sum(len(v) for v in qrels.values())} judgments")
    return dict(qrels)


# ══════════════════════════════════════════════
# STEP 1 – unique query IDs from qrels
# ══════════════════════════════════════════════
log.info("=== Step 1: Parse qrels ===")
qrels = parse_qrels(QRELS_PATH)
unique_qids = set(qrels.keys())
log.info(f"Unique query IDs in qrels: {len(unique_qids)}")


# ══════════════════════════════════════════════
# STEP 2 – query embeddings for those IDs
# ══════════════════════════════════════════════
log.info("=== Step 2: Load query embeddings ===")
q_table   = pq.read_table(QUERY_EMB_PATH)
q_id_col  = q_table["id"].to_pylist()
q_emb_col = np.array(q_table["embedding"].to_pylist(), dtype=np.float32)

mask  = [qid in unique_qids for qid in q_id_col]
q_ids = [qid for qid, m in zip(q_id_col, mask) if m]
q_emb = q_emb_col[[i for i, m in enumerate(mask) if m]]
faiss.normalize_L2(q_emb)

log.info(f"Queries after filtering: {len(q_ids)} | dim={q_emb.shape[1]}")
DIM = q_emb.shape[1]


# ══════════════════════════════════════════════
# STEP 3 – mine relevant pool from full IVF index
# ══════════════════════════════════════════════
log.info("=== Step 3: Search full IVF index ===")
log.info(f"Loading full index from {FULL_INDEX_PATH} …")
full_index = faiss.read_index(FULL_INDEX_PATH)
log.info(f"Full index loaded: {full_index.ntotal:,} vectors")

all_ids_arr  = np.load(FULL_IDS_PATH, allow_pickle=True)
all_ids_list = all_ids_arr.tolist()
assert len(all_ids_list) == full_index.ntotal

full_index.nprobe = NPROBE_FULL
log.info(f"Searching top-{TOP_K_BUILD} per query (nprobe={NPROBE_FULL}) …")

BATCH = 50
all_D, all_I = [], []
for start in range(0, len(q_emb), BATCH):
    batch = q_emb[start:start + BATCH]
    D, I  = full_index.search(batch, TOP_K_BUILD)
    all_D.append(D)
    all_I.append(I)

indices_full = np.vstack(all_I)

relevant_ids: set = set()
for qi in range(len(q_ids)):
    for idx in indices_full[qi]:
        if idx != -1:
            relevant_ids.add(str(all_ids_list[idx]))

log.info(f"Relevant pool size: {len(relevant_ids):,} unique docs")

del full_index
gc.collect()
log.info("Full index released from memory")


# ══════════════════════════════════════════════
# STEP 4 – sample equal-sized noise pool
# ══════════════════════════════════════════════
log.info("=== Step 4: Sample noise pool ===")
all_doc_ids_str = [str(d) for d in all_ids_list]
non_relevant    = [d for d in all_doc_ids_str if d not in relevant_ids]
rng             = random.Random(RANDOM_SEED)
noise_ids       = set(rng.sample(non_relevant,
                                  min(len(relevant_ids), len(non_relevant))))
log.info(f"Noise pool size: {len(noise_ids):,} docs")

demo_doc_ids = relevant_ids | noise_ids
log.info(f"Total demo collection size: {len(demo_doc_ids):,} docs")


# ══════════════════════════════════════════════
# STEP 5a – fetch text from TSV
# ══════════════════════════════════════════════
log.info("=== Step 5a: Fetch document texts from TSV ===")
text_map: dict[str, str] = {}
with open(COLLECTION_TSV, encoding="utf-8") as fh:
    header = fh.readline()
    log.info(f"TSV header: {header.strip()!r}")
    for lineno, line in enumerate(fh, start=2):
        line = line.rstrip("\n")
        if not line:
            continue
        tab = line.index("\t")
        doc_id = line[:tab]
        if doc_id in demo_doc_ids:
            text_map[doc_id] = line[tab + 1:]
        if lineno % 5_000_000 == 0:
            log.info(f"  TSV progress: {lineno:,} lines read, "
                     f"{len(text_map):,} demo texts collected")

log.info(f"Texts collected: {len(text_map):,} (expected ≤ {len(demo_doc_ids):,})")
missing_text = demo_doc_ids - set(text_map)
if missing_text:
    log.warning(f"  {len(missing_text):,} demo doc IDs not found in TSV – skipped")


# ══════════════════════════════════════════════
# STEP 5b – fetch embeddings from sharded Parquet
# ══════════════════════════════════════════════
log.info("=== Step 5b: Fetch embeddings from sharded Parquet ===")
emb_dir     = Path(EMBEDDINGS_DIR)
shard_files = sorted(emb_dir.glob("*.parquet"))
if not shard_files:
    log.error(f"No .parquet files found in {EMBEDDINGS_DIR}")
    sys.exit(1)
log.info(f"Embedding shards: {len(shard_files)}")

# Inner join: only IDs that have both text and an embedding
target_ids = demo_doc_ids & set(text_map)

# Collected separately so we can write two clean files
ids_out:  list[str]       = []
texts_out: list[str]      = []
embs_out:  list[list]     = []
seen_emb:  set[str]       = set()

for shard in shard_files:
    if len(seen_emb) == len(target_ids):
        log.info("  All embeddings found – stopping early")
        break

    log.info(f"  Reading shard {shard.name} …")
    tbl  = pq.read_table(shard, columns=[COL_ID, COL_EMBEDDING])
    ids  = tbl[COL_ID].to_pylist()
    embs = tbl[COL_EMBEDDING].to_pylist()

    for doc_id, emb in zip(ids, embs):
        sid = str(doc_id)
        if sid in target_ids and sid not in seen_emb:
            seen_emb.add(sid)
            ids_out.append(sid)
            texts_out.append(text_map[sid])
            embs_out.append(emb)

    log.info(f"    collected so far: {len(ids_out):,} / {len(target_ids):,}")

log.info(f"Demo records collected: {len(ids_out):,}")
if len(ids_out) == 0:
    log.error(
        "No documents were fetched! Check EMBEDDINGS_DIR, COLLECTION_TSV, "
        "and COL_ID / COL_EMBEDDING column names."
    )
    sys.exit(1)


# ══════════════════════════════════════════════
# STEP 6 – save texts and embeddings separately
# ══════════════════════════════════════════════

# ── 6a  Text file ─────────────────────────────
log.info("=== Step 6a: Save demo_texts.parquet ===")
text_table = pa.table({
    COL_ID:   pa.array(ids_out,   type=pa.string()),
    COL_TEXT: pa.array(texts_out, type=pa.string()),
})
pq.write_table(text_table, DEMO_TEXTS_PATH)
log.info(f"  Saved {len(ids_out):,} rows → {DEMO_TEXTS_PATH}")

# ── 6b  Embeddings file ───────────────────────
log.info("=== Step 6b: Save demo_embeddings.parquet ===")
emb_table = pa.table({
    COL_ID:        pa.array(ids_out,  type=pa.string()),
    COL_EMBEDDING: pa.array(embs_out, type=pa.list_(pa.float32())),
})
pq.write_table(emb_table, DEMO_EMBEDDINGS_PATH)
log.info(f"  Saved {len(ids_out):,} rows → {DEMO_EMBEDDINGS_PATH}")


# ══════════════════════════════════════════════
# STEP 7 – save metadata
# ══════════════════════════════════════════════
N   = len(ids_out)
DIM = len(embs_out[0]) if embs_out else 0

metadata = {
    "num_docs":             N,
    "embedding_dim":        DIM,
    "relevant_pool_size":   len(relevant_ids),
    "noise_pool_size":      len(noise_ids),
    "top_k_build":          TOP_K_BUILD,
    "nprobe_full_search":   NPROBE_FULL,
    "random_seed":          RANDOM_SEED,
    "embeddings_dir":       EMBEDDINGS_DIR,
    "collection_tsv":       COLLECTION_TSV,
    "demo_texts_path":      DEMO_TEXTS_PATH,
    "demo_embeddings_path": DEMO_EMBEDDINGS_PATH,
}
with open(METADATA_PATH, "w") as fh:
    json.dump(metadata, fh, indent=2)
log.info(f"Metadata saved to {METADATA_PATH}")
log.info("=== Phase 1 COMPLETE – run build_demo_indexes.py next ===")
