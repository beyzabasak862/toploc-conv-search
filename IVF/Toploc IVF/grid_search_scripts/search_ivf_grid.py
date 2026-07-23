import faiss
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
import sys
import json
import time
import logging
from collections import defaultdict

# -----------------------
# Configuration
# -----------------------
INDEX_PATH     = "/home/toploc1/Datasets/toploc1/indexes/treccast_34M_ivf.index"
IDS_PATH       = "/home/toploc1/Datasets/toploc1/indexes/treccast_34M_ivf_ids.npy"
QUERY_EMB_PATH = "../conversational/CAST2019/topics/topics_snowflake_embeddings.parquet"
QRELS_PATH     = "../conversational/CAST2019/topics/qrels.qrel"
OUTPUT_DIR     = "/home/toploc1/Datasets/toploc1/indexes"
METRICS_PATH   = os.path.join(OUTPUT_DIR, "metrics_ivf_grid.json")
LOG_PATH       = os.path.join(OUTPUT_DIR, "search_ivf_grid.log")

NPROBE_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
TOP_K         = 1000
REL_THRESHOLD = 1

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
# Metrics
# -----------------------

def parse_qrels(path: str) -> dict:
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            qid, docid, grade = parts[0], parts[2], int(parts[3])
            qrels[qid][docid] = grade
    log.info(f"Loaded qrels: {len(qrels)} queries, "
             f"{sum(len(v) for v in qrels.values())} judgments")
    return dict(qrels)


def mrr_at_k(retrieved: list, relevant: set, k: int) -> float:
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list, grades: dict, k: int) -> float:
    dcg = sum(
        grades.get(doc, 0) / np.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg  = sum(g / np.log2(r + 1) for r, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(run: dict, qrels: dict) -> dict:
    per_query = {}
    totals    = defaultdict(float)
    evaluated = 0

    for qid, retrieved in run.items():
        if qid not in qrels:
            continue
        grades   = qrels[qid]
        relevant = {d for d, g in grades.items() if g >= REL_THRESHOLD}
        evaluated += 1

        q = {
            "MRR@10":  mrr_at_k(retrieved, relevant, 10),
            "NDCG@3":  ndcg_at_k(retrieved, grades, 3),
            "NDCG@10": ndcg_at_k(retrieved, grades, 10),
        }
        per_query[qid] = q
        for metric, val in q.items():
            totals[metric] += val

    aggregate = {m: v / evaluated for m, v in totals.items()}
    aggregate["num_queries"] = evaluated
    return {"aggregate": aggregate, "per_query": per_query}


# -----------------------
# 1. Load index
# -----------------------
log.info(f"Loading index from {INDEX_PATH}")
index = faiss.read_index(INDEX_PATH)
log.info(f"Index loaded: {index.ntotal:,} vectors")

# -----------------------
# 2. Load ID map
# -----------------------
all_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
assert len(all_ids) == index.ntotal, \
    f"ID map length {len(all_ids)} != index size {index.ntotal}"
log.info(f"ID map loaded: {len(all_ids):,} entries")

# -----------------------
# 3. Load query embeddings
# -----------------------
log.info(f"Loading queries from {QUERY_EMB_PATH}")
q_table = pq.read_table(QUERY_EMB_PATH)
q_ids   = q_table["id"].to_pylist()
q_emb   = np.array(q_table["embedding"].to_pylist(), dtype=np.float32)
faiss.normalize_L2(q_emb)
log.info(f"Queries: {len(q_ids)} | dim={q_emb.shape[1]}")

# -----------------------
# 4. Load qrels
# -----------------------
log.info(f"Loading qrels from {QRELS_PATH}")
qrels = parse_qrels(QRELS_PATH)

# -----------------------
# 5. Grid search over nprobe
# -----------------------
all_results = {}

for nprobe in NPROBE_VALUES:
    log.info(f"--- nprobe={nprobe} ---")
    index.nprobe = nprobe

    # Search and time it
    start_time = time.perf_counter()
    BATCH_SIZE = 50
    all_distances, all_indices = [], []
    for start in range(0, len(q_emb), BATCH_SIZE):
        batch = q_emb[start:start + BATCH_SIZE]
        D, I = index.search(batch, TOP_K)
        all_distances.append(D)
        all_indices.append(I)
    distances = np.vstack(all_distances)
    indices   = np.vstack(all_indices)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    avg_ms = elapsed_ms / len(q_ids)

    # Build run
    run = {}
    for qi, qid in enumerate(q_ids):
        retrieved = []
        for idx, score in zip(indices[qi], distances[qi]):
            if idx == -1:
                continue
            retrieved.append(str(all_ids[idx]))
        run[qid] = retrieved

    # Evaluate
    results = evaluate(run, qrels)
    agg = results["aggregate"]

    log.info(f"  avg_query_time_ms: {avg_ms:.2f}")
    log.info(f"  MRR@10:  {agg['MRR@10']:.4f}")
    log.info(f"  NDCG@3:  {agg['NDCG@3']:.4f}")
    log.info(f"  NDCG@10: {agg['NDCG@10']:.4f}")

    all_results[str(nprobe)] = {
        "nprobe": nprobe,
        "avg_query_time_ms": avg_ms,
        "MRR@10":  agg["MRR@10"],
        "NDCG@3":  agg["NDCG@3"],
        "NDCG@10": agg["NDCG@10"],
        "num_queries": agg["num_queries"],
    }

# -----------------------
# 6. Save all results
# -----------------------
with open(METRICS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
log.info(f"Grid search complete. Results saved to {METRICS_PATH}")