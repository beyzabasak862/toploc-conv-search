"""
Exact search baseline for Dragon on CAsT 2019 AND 2020, using the flat IP
index built by build_flat_dragon.py.

Brute-force dot-product retrieval = the "Exact" reference row of
Muntean et al., SIGIR'25.

CAsT 2019 and 2020 share the SAME passage collection, so the flat index and
its id map are loaded ONCE and reused; only the query embeddings and qrels
differ per dataset. One metrics json is written per dataset, each in the
{"Flat": {...}} schema the summary notebook consumes.

Note: the paper reports no query time for Exact -- it is an effectiveness
ceiling, not an efficiency baseline. Timing is logged anyway for completeness.
"""

import faiss
import numpy as np
import pyarrow.parquet as pq
import os
import sys
import json
import time
import logging
from collections import defaultdict

# -----------------------
# Configuration
# -----------------------
# Shared flat index (same collection for both datasets):
INDEX_PATH = "/home/toploc1/Datasets/toploc1/indexes/Dragon/Flat Search/treccast_flat_dragon.index"
IDS_PATH   = "/home/toploc1/Datasets/toploc1/indexes/Dragon/Flat Search/treccast_flat_dragon_ids.npy"

OUTPUT_DIR = "/home/toploc1/Datasets/toploc1/Exact_Search/Dragon"
LOG_PATH   = os.path.join(OUTPUT_DIR, "search_flat_dragon.log")

# Per-dataset inputs. Only the query embeddings + qrels change.
# EDIT the 2020 paths to match your files.
DATASETS = {
    "cast2019": {
        "query_emb": "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/topics_dragon_embeddings.parquet",
        "qrels":     "/home/toploc1/Datasets/conversational/CAST2019/topics/qrels.qrel",
    },
    "cast2020": {
        "query_emb": "/home/toploc1/Datasets/toploc1/Data Exploration/topics_dragon_embeddings_2020.parquet",
        "qrels":     "/home/toploc1/Datasets/toploc1/Data Exploration/cast2020_qrels.qrel",
    },
}

TOP_K         = 10
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
# Helpers
# -----------------------
def parse_qrels(path):
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


def mrr_at_k(retrieved, relevant, k):
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, grades, k):
    dcg = sum(
        grades.get(doc, 0) / np.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], start=1)
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg  = sum(g / np.log2(r + 1) for r, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(run, qrels):
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


def flat_search(index, all_ids, q_emb, q_ids, top_k):
    t0 = time.perf_counter()
    D, I = index.search(q_emb, top_k)
    search_ms = (time.perf_counter() - t0) * 1000

    run = {}
    for qi, qid in enumerate(q_ids):
        run[qid] = [
            str(all_ids[idx])
            for idx, score in zip(I[qi], D[qi])
            if idx != -1
        ]
    return run, search_ms


# -----------------------
# Load the flat index ONCE (shared across datasets)
# -----------------------
if not os.path.exists(INDEX_PATH):
    log.error(f"Index not found at {INDEX_PATH}")
    sys.exit(1)
if not os.path.exists(IDS_PATH):
    log.error(f"IDs not found at {IDS_PATH}")
    sys.exit(1)

log.info(f"=== Loading flat index from {INDEX_PATH} ===")
index = faiss.read_index(INDEX_PATH)
log.info(f"  ntotal={index.ntotal:,} | d={index.d} (exact inner-product search)")

all_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
assert len(all_ids) == index.ntotal, \
    f"ID map length {len(all_ids)} != index size {index.ntotal}"
log.info(f"  ID map loaded: {len(all_ids):,} entries")


# -----------------------
# Run exact search per dataset
# -----------------------
def run_dataset(name, cfg):
    log.info(f"##### DATASET: {name} #####")

    if not os.path.exists(cfg["query_emb"]):
        log.error(f"[{name}] query embeddings not found: {cfg['query_emb']} -- skipping")
        return
    if not os.path.exists(cfg["qrels"]):
        log.error(f"[{name}] qrels not found: {cfg['qrels']} -- skipping")
        return

    # queries
    log.info(f"[{name}] Loading queries from {cfg['query_emb']}")
    q_table = pq.read_table(cfg["query_emb"])
    q_ids   = q_table["id"].to_pylist()
    q_emb   = np.array(q_table["embedding"].to_pylist(), dtype=np.float32)
    faiss.normalize_L2(q_emb)                 # per-query scaling; does not change IP ranking
    assert q_emb.shape[1] == index.d, \
        f"[{name}] query dim {q_emb.shape[1]} != index dim {index.d}"
    log.info(f"[{name}] Queries: {len(q_ids)} | dim={q_emb.shape[1]}")

    # qrels
    log.info(f"[{name}] Loading qrels from {cfg['qrels']}")
    qrels = parse_qrels(cfg["qrels"])

    # exact search
    log.info(f"[{name}] --- Flat | exact search ---")
    run, search_ms = flat_search(index, all_ids, q_emb, q_ids, TOP_K)
    avg_ms = search_ms / len(q_ids)

    agg = evaluate(run, qrels)["aggregate"]
    log.info(f"[{name}]   search_ms (total):   {search_ms:.1f}")
    log.info(f"[{name}]   avg_query_time_ms:   {avg_ms:.2f}")
    log.info(f"[{name}]   MRR@10:  {agg['MRR@10']:.4f}")
    log.info(f"[{name}]   NDCG@3:  {agg['NDCG@3']:.4f}")
    log.info(f"[{name}]   NDCG@10: {agg['NDCG@10']:.4f}")

    all_results = {
        "Flat": {
            "method":             "flat",
            "dataset":            name,
            "avg_query_time_ms":  avg_ms,
            "search_ms_total":    search_ms,
            "MRR@10":             agg["MRR@10"],
            "NDCG@3":             agg["NDCG@3"],
            "NDCG@10":            agg["NDCG@10"],
            "num_queries":        agg["num_queries"],
        }
    }

    metrics_path = os.path.join(OUTPUT_DIR, f"metrics_flat_dragon_{name}.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"[{name}] Flat baseline complete. Results saved to {metrics_path}")


for name, cfg in DATASETS.items():
    run_dataset(name, cfg)

log.info("All datasets complete.")