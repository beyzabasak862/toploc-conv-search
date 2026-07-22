import faiss
import numpy as np
import pyarrow.parquet as pq
import os
import sys
import json
import time
import logging
from collections import defaultdict

# ======================================================================
# v12 — TopLoc IVF+ (centroid cache refresh), alpha grid.
#
# Method (paper, Eq. 1): for follow-up utterance q_j with current cache
# C0 (built from reference query q_ref, initially q_0):
#     I0 = top_np(q_j, C0)  INTERSECT  top_np(q_ref, C0)
# If |I0| < alpha * np, a topic shift is assumed and the cache is
# REFRESHED: C0 <- top_h(q_j, C), q_ref <- q_j. Turn j is then searched
# with the NEW cache (effectiveness-preserving interpretation).
#
# Measurement design:
#   * Processing is SEQUENTIAL per turn within each conversation --
#     a refresh at turn j changes the cache used by turn j+1, so the
#     batched structure of v11 cannot apply. Latencies here are
#     comparable across alpha values (identical loop), NOT directly
#     against v11's batched numbers.
#   * alpha = 0.0 never triggers a refresh (|I0| < 0 is impossible), so
#     it IS static TopLoc in the same code path: the built-in control.
#     The marginal effect of refreshing = (alpha>0) vs (alpha=0).
#   * REFRESH COST IS INSIDE THE FOLLOW-UP TIMER: a refresh is work done
#     while serving a follow-up query (full quantizer scan over nlist +
#     rebuild of the small centroid index), unlike the one-time turn-0
#     cache build which stays outside and is charged to first-turn
#     total, exactly as in v11.
#   * Intersection computation (two np-sized id arrays) is also timed --
#     it is TopLoc+'s per-query monitoring overhead.
#   * Refresh statistics (count, rate, per-conversation) are collected
#     in the recall pass (deterministic, identical across timed runs).
#   * parallel_mode=2 for all search_preassigned calls (per v11
#     diagnostic); refresh scan uses the flat quantizer directly.
# ======================================================================

# -----------------------
# Configuration
# -----------------------
INDEX_PATH     = os.environ.get("INDEX_PATH",
    "/home/toploc1/Datasets/toploc1/indexes/treccast_34M_ivf.index")
IDS_PATH       = os.environ.get("IDS_PATH",
    "/home/toploc1/Datasets/toploc1/indexes/treccast_34M_ivf_ids.npy")
QUERY_EMB_PATH = os.environ.get("QUERY_EMB_PATH",
    "/home/toploc1/Datasets/conversational/CAST2019/topics/topics_snowflake_embeddings.parquet")
QRELS_PATH     = os.environ.get("QRELS_PATH",
    "/home/toploc1/Datasets/conversational/CAST2019/topics/qrels.qrel")
OUTPUT_DIR     = os.environ.get("OUTPUT_DIR",
    "/home/toploc1/Datasets/toploc1/indexes")
METRICS_PATH = os.path.join(OUTPUT_DIR, "toploc_ivf_plus_grid.json")
LOG_PATH     = os.path.join(OUTPUT_DIR, "toploc_ivf_plus_grid.log")

USE_MMAP = os.environ.get("MMAP", "1") == "1"

NPROBE_VALUES    = [1, 2, 4, 8, 16, 32, 64, 128, 256]
H_VALUES         = [512, 1024, 4096, 8192]
ALPHA_VALUES     = [0.0, 0.05, 0.1, 0.2]   # 0.0 = static TopLoc control
K_LATENCY_VALUES = [10, 100, 1000]
K_METRICS        = int(os.environ.get("K_METRICS", 1000))
REL_THRESHOLD    = 1
WARMUP_RUNS      = int(os.environ.get("WARMUP_RUNS", 1))
TIMED_RUNS       = int(os.environ.get("TIMED_RUNS", 3))

PM_PREASSIGNED = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# -----------------------
# Conversation structure
# -----------------------

def get_conv_id(qid: str) -> str:
    return qid.rsplit("_", 1)[0]


def build_conv_structure(q_ids: list):
    conv_ids = [get_conv_id(str(qid)) for qid in q_ids]
    seen: dict = {}
    turn0_idx_list = []
    conv_members: list = []
    for qi, cid in enumerate(conv_ids):
        if cid not in seen:
            seen[cid] = len(turn0_idx_list)
            turn0_idx_list.append(qi)
            conv_members.append([])
        ci = seen[cid]
        if qi != turn0_idx_list[ci]:
            conv_members[ci].append(qi)
    turn0_idx      = np.array(turn0_idx_list, dtype=np.int32)
    conv_later_idx = [np.array(m, dtype=np.int32) for m in conv_members]
    return turn0_idx, conv_later_idx


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
            qrels[parts[0]][parts[2]] = int(parts[3])
    return dict(qrels)


def mrr_at_k(retrieved, relevant, k):
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, grades, k):
    dcg = sum(grades.get(doc, 0) / np.log2(rank + 1)
              for rank, doc in enumerate(retrieved[:k], start=1))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(r + 1) for r, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(run, qrels):
    totals, evaluated = defaultdict(float), 0
    for qid, retrieved in run.items():
        if qid not in qrels:
            continue
        grades = qrels[qid]
        relevant = {d for d, g in grades.items() if g >= REL_THRESHOLD}
        evaluated += 1
        totals["MRR@10"]  += mrr_at_k(retrieved, relevant, 10)
        totals["NDCG@3"]  += ndcg_at_k(retrieved, grades, 3)
        totals["NDCG@10"] += ndcg_at_k(retrieved, grades, 10)
    agg = {m: v / evaluated for m, v in totals.items()} if evaluated else {}
    agg["num_queries"] = evaluated
    return agg


def timeit(fn):
    for _ in range(WARMUP_RUNS):
        fn()
    ts = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


# -----------------------
# Initial cache build (once per h, outside all timers -- as in v11)
# -----------------------

def build_caches(index, q_emb, h, all_centroids, turn0_idx):
    quantizer = faiss.downcast_index(index.quantizer)
    d = q_emb.shape[1]
    t0 = time.perf_counter()
    turn0_vecs = q_emb[turn0_idx]
    Dq0, Iq0 = quantizer.search(turn0_vecs, h)
    cache_ids, cache_dists, cache_index = {}, {}, {}
    for c in range(len(turn0_idx)):
        cids = Iq0[c].astype("int64")
        idx = faiss.IndexFlatIP(d)
        idx.add(all_centroids[cids])
        cache_ids[c]   = cids
        cache_dists[c] = Dq0[c].astype("float32")
        cache_index[c] = idx
    build_ms = (time.perf_counter() - t0) * 1000 / max(1, len(turn0_idx))
    return cache_ids, cache_dists, cache_index, build_ms


# -----------------------
# TopLoc+ sequential follow-up pass
# -----------------------

def toploc_plus_followup(index, quantizer, q_emb, conv_later_idx,
                         master_ids, master_dists, master_index,
                         all_centroids, h, nprobe, alpha, top_k,
                         collect_results=False):
    """
    One full sequential pass over all follow-up turns of all
    conversations, applying the TopLoc+ refresh rule.

    Working caches start as (shallow) references to the master cache
    built from turn 0; refreshes REPLACE per-conversation entries with
    new objects, so the master is never mutated and every timed run
    starts from the identical initial state.

    Returns (results, n_refreshes, refreshes_per_conv, i0_fractions):
      results:      list of (query_row, I_array) if collect_results
      i0_fractions: |I0|/nprobe at each follow-up turn (pre-refresh
                    cache), for topic-shift analysis
    """
    d = q_emb.shape[1]
    threshold = alpha * nprobe
    index.parallel_mode = PM_PREASSIGNED
    index.nprobe = nprobe

    results = [] if collect_results else None
    n_refreshes = 0
    refreshes_per_conv = []
    i0_fractions = [] if collect_results else None

    for c, later in enumerate(conv_later_idx):
        if later.size == 0:
            refreshes_per_conv.append(0)
            continue

        # Working state for this conversation (references, not copies)
        w_ids   = master_ids[c]
        w_index = master_index[c]
        # Reference selection = top-nprobe of the query that built the
        # cache; cache arrays are sorted, so it's a slice.
        ref_sel = w_ids[:nprobe]
        conv_refreshes = 0

        for row in later:
            q = q_emb[row:row + 1]  # (1, d)

            # Selection from current cache
            D_local, I_local = w_index.search(q, nprobe)
            sel = w_ids[I_local[0]]

            # Monitoring: |I0| = |sel INTERSECT ref_sel| -- TopLoc+'s
            # per-query overhead, deliberately inside the timed region.
            i0 = np.intersect1d(sel, ref_sel).size
            if collect_results:
                i0_fractions.append(i0 / nprobe)

            if i0 < threshold:
                # --- REFRESH: full coarse scan for q_j, rebuild cache.
                # This cost is part of serving this follow-up query.
                Dq, Iq = quantizer.search(q, h)
                w_ids = Iq[0].astype("int64")
                new_index = faiss.IndexFlatIP(d)
                new_index.add(all_centroids[w_ids])
                w_index = new_index
                ref_sel = w_ids[:nprobe]
                # Re-select for this turn from the fresh cache: the
                # top-nprobe are the first entries of the sorted scan.
                sel = w_ids[:nprobe]
                D_sel = Dq[0][:nprobe].astype("float32")
                n_refreshes += 1
                conv_refreshes += 1
            else:
                D_sel = D_local[0].astype("float32")

            Iq_assign = sel.reshape(1, nprobe).astype(np.int64)
            Dq_assign = D_sel.reshape(1, nprobe)
            D, I = index.search_preassigned(q, top_k, Iq_assign, Dq_assign)

            if collect_results:
                results.append((int(row), I[0]))

        refreshes_per_conv.append(conv_refreshes)

    return results, n_refreshes, refreshes_per_conv, i0_fractions


# -----------------------
# First turn (identical to v11: reuse the initial cache-build scan)
# -----------------------

def toploc_firstturn(index, q_emb, turn0_idx, cache_ids, cache_dists,
                     nprobe, top_k):
    n = turn0_idx.size
    Iq = np.stack([cache_ids[c][:nprobe]   for c in range(n)]).astype(np.int64)
    Dq = np.stack([cache_dists[c][:nprobe] for c in range(n)]).astype(np.float32)
    index.parallel_mode = PM_PREASSIGNED
    index.nprobe = nprobe
    D, I = index.search_preassigned(q_emb[turn0_idx], top_k, Iq, Dq)
    return D, I


def save_results(results):
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ======================================================================
# Load everything
# ======================================================================
log.info(f"Loading index from {INDEX_PATH} (mmap={USE_MMAP})")
index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP if USE_MMAP else 0)
log.info(f"Index loaded: {index.ntotal:,} vectors, nlist={index.nlist}")

quantizer     = faiss.downcast_index(index.quantizer)
all_centroids = quantizer.reconstruct_n(0, index.nlist)
log.info(f"Centroid matrix: {all_centroids.shape}")

all_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
assert len(all_ids) == index.ntotal

q_table = pq.read_table(QUERY_EMB_PATH)
q_ids   = q_table["id"].to_pylist()
q_emb   = np.array(q_table["embedding"].to_pylist(), dtype=np.float32)
faiss.normalize_L2(q_emb)
log.info(f"Queries: {len(q_ids)} | dim={q_emb.shape[1]}")

qrels = parse_qrels(QRELS_PATH)
log.info(f"Loaded qrels: {len(qrels)} queries")

turn0_idx, conv_later_idx = build_conv_structure(q_ids)
n_first = int(turn0_idx.size)
n_fu    = int(sum(r.size for r in conv_later_idx))
avg_followups = n_fu / max(1, n_first)
log.info(f"Conversations: {n_first} | follow-up queries: {n_fu} "
         f"| avg follow-ups/conv: {avg_followups:.2f}")

results = {
    "config": {
        "nprobe_values": NPROBE_VALUES, "h_values": H_VALUES,
        "alpha_values": ALPHA_VALUES,
        "k_latency_values": K_LATENCY_VALUES, "k_metrics": K_METRICS,
        "warmup_runs": WARMUP_RUNS, "timed_runs": TIMED_RUNS,
        "n_conversations": n_first, "n_followup_queries": n_fu,
        "avg_followups_per_conv": avg_followups,
        "parallel_mode_preassigned": PM_PREASSIGNED,
        "note": ("Sequential per-turn processing; refresh cost inside "
                 "follow-up timer; alpha=0.0 is the static-TopLoc "
                 "control in the identical code path."),
    },
    "toploc_plus": {},
}

# ======================================================================
# Grid: h (outer, cache built once) x alpha x nprobe
# ======================================================================
for h in H_VALUES:
    if h > index.nlist:
        log.info(f"skip h={h}: exceeds nlist={index.nlist}")
        continue

    m_ids, m_dists, m_index, cache_build_ms = build_caches(
        index, q_emb, h, all_centroids, turn0_idx)
    log.info(f"[h={h}] initial caches built: {cache_build_ms:.2f} ms/conv "
             f"(outside timers, charged to first turn as in v11)")

    for alpha in ALPHA_VALUES:
        for nprobe in NPROBE_VALUES:
            if h <= nprobe:
                continue

            # ---- recall pass (also collects refresh stats + |I0|) ----
            res, n_ref, ref_per_conv, i0_fracs = toploc_plus_followup(
                index, quantizer, q_emb, conv_later_idx,
                m_ids, m_dists, m_index, all_centroids,
                h, nprobe, alpha, K_METRICS, collect_results=True)

            run = {}
            _, I0turn = toploc_firstturn(index, q_emb, turn0_idx,
                                         m_ids, m_dists, nprobe, K_METRICS)
            for r, row in enumerate(turn0_idx):
                run[str(q_ids[row])] = [str(all_ids[i]) for i in I0turn[r]
                                        if i != -1]
            for row, I in res:
                run[str(q_ids[row])] = [str(all_ids[i]) for i in I if i != -1]
            eval_t = evaluate(run, qrels)

            refresh_rate = n_ref / max(1, n_fu)
            mean_i0 = float(np.mean(i0_fracs)) if i0_fracs else float("nan")

            # ---- latency per k (sequential pass incl. refreshes) ----
            lat = {}
            for k in K_LATENCY_VALUES:
                fu_ms = timeit(lambda k=k: toploc_plus_followup(
                    index, quantizer, q_emb, conv_later_idx,
                    m_ids, m_dists, m_index, all_centroids,
                    h, nprobe, alpha, k,
                    collect_results=False)) / max(1, n_fu)
                lat[str(k)] = {"followup_ms_per_query": fu_ms}

            log.info(
                f"  h={h:>5} a={alpha:<4} nprobe={nprobe:>4} | "
                f"fu ms/q: k10={lat['10']['followup_ms_per_query']:6.3f} "
                f"k1000={lat['1000']['followup_ms_per_query']:6.3f} | "
                f"refreshes={n_ref:>3} ({refresh_rate*100:4.1f}% of turns) "
                f"mean|I0|/np={mean_i0:.3f} | "
                f"NDCG@10={eval_t['NDCG@10']:.4f}"
            )

            results["toploc_plus"][f"h={h}_alpha={alpha}_nprobe={nprobe}"] = {
                "h": h, "alpha": alpha, "nprobe": nprobe,
                "cache_build_ms_per_conv": cache_build_ms,
                "metrics": eval_t,
                "refresh": {
                    "total_refreshes": int(n_ref),
                    "refresh_rate": refresh_rate,
                    "refreshes_per_conv": [int(x) for x in ref_per_conv],
                    "mean_i0_fraction": mean_i0,
                },
                "latency": lat,
            }
            save_results(results)

    del m_ids, m_dists, m_index

log.info(f"Done. Results saved to {METRICS_PATH}")