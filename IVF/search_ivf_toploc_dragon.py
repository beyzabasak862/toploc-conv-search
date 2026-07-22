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
# v11 — v10 + PREASSIGNED CONTROL arm.
#
# Motivation: v10 showed large TopLoc speedups, but the diagnostic
# (diag_preassigned_overhead.py) proved that search_preassigned at
# parallel_mode=2 is faster than fused search() at parallel_mode=0 even
# when fed the FULL, unrestricted coarse assignments. So the v10 speedup
# conflates two effects:
#   (a) TopLoc's mechanism: restricting the coarse step to h cached
#       centroids (skip the 32k scan on follow-up turns);
#   (b) the preassigned code path being better-threaded than the fused
#       path on this machine.
#
# The CONTROL arm isolates them. For each (nprobe, k) it times, on the
# same follow-up rows:
#   full quantizer.search over ALL 32768 centroids  (timed)
#   + search_preassigned at parallel_mode=2          (timed)
# i.e. exactly TopLoc's execution path but WITHOUT the cache — the
# coarse step is complete, results identical to plain IVF.
#
# Decomposition reported per cell:
#   followup_speedup            = fused_baseline / toploc      (system claim)
#   followup_speedup_vs_control = control        / toploc      (pure mechanism)
#   control_vs_fused            = fused_baseline / control     (pure implementation)
# Note: system = mechanism x implementation (the two ratios multiply).
#
# Everything else is identical to v10 (parallel_mode=0 for fused
# baseline, parallel_mode=2 for all preassigned calls; recall identical
# by construction to v9/v10 so it is recomputed only for TopLoc).
# ======================================================================

# -----------------------
# Configuration
# -----------------------
INDEX_PATH     = os.environ.get("INDEX_PATH",
    "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/treccast_dragon_ivf.index")
IDS_PATH       = os.environ.get("IDS_PATH",
    "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/treccast_dragon_ivf_ids.npy")
QUERY_EMB_PATH = os.environ.get("QUERY_EMB_PATH",
    "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/topics_dragon_embeddings.parquet")
QRELS_PATH     = os.environ.get("QRELS_PATH",
    "/home/toploc1/Datasets/conversational/CAST2019/topics/qrels.qrel")
OUTPUT_DIR     = os.environ.get("OUTPUT_DIR",
    "/home/toploc1/Datasets/toploc1/IVF/dragon")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics_ivf_toploc_dragon_v11.json")
LOG_PATH     = os.path.join(OUTPUT_DIR, "search_ivf_toploc_dragon_v11.log")

USE_MMAP = os.environ.get("MMAP", "1") == "1"

NPROBE_VALUES    = [1, 2, 4, 8, 16, 32, 64, 128, 256]
H_VALUES         = [512, 1024, 4096, 8192]   # >1024 kept to show O(h) decay
K_LATENCY_VALUES = [10, 100, 1000]           # latency swept over k
K_METRICS        = int(os.environ.get("K_METRICS", 1000))
REL_THRESHOLD    = 1
BATCH_SIZE       = 50
WARMUP_RUNS      = int(os.environ.get("WARMUP_RUNS", 1))
TIMED_RUNS       = int(os.environ.get("TIMED_RUNS", 3))

PM_FUSED       = 0   # best mode for fused search() (from diagnostic)
PM_PREASSIGNED = 2   # best mode for search_preassigned (from diagnostic)

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
    """turn0_idx (n_convs,), conv_later_idx: list of arrays of turns j>0."""
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


# -----------------------
# Timing helper: warmup + median
# -----------------------

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
# Fused baseline over an arbitrary row set (parallel_mode=PM_FUSED)
# -----------------------

def baseline_search_rows(index, q_emb, rows, nprobe, top_k):
    index.nprobe = nprobe
    index.parallel_mode = PM_FUSED
    all_I = []
    for s in range(0, rows.size, BATCH_SIZE):
        _, I = index.search(q_emb[rows[s:s + BATCH_SIZE]], top_k)
        all_I.append(I)
    return np.vstack(all_I) if all_I else np.empty((0, top_k), dtype=np.int64)


# -----------------------
# PREASSIGNED CONTROL: full coarse scan + preassigned scan, no cache.
# Same execution path as TopLoc minus the caching mechanism. Results
# identical to plain IVF at the same nprobe by construction.
# -----------------------

def preassigned_control_rows(index, q_emb, rows, nprobe, top_k):
    quantizer = faiss.downcast_index(index.quantizer)
    x = q_emb[rows]
    Dq, Iq = quantizer.search(x, nprobe)      # full 32k scan — TIMED
    index.parallel_mode = PM_PREASSIGNED
    index.nprobe = nprobe
    D, I = index.search_preassigned(x, top_k, Iq.astype(np.int64),
                                    Dq.astype(np.float32))
    return D, I


# -----------------------
# TopLoc cache build (once per h, NEVER inside a latency timer)
# -----------------------

def build_caches(index, q_emb, h, all_centroids, turn0_idx):
    """
    For each conversation: full quantizer scan of the turn-0 vector,
    keep top-h centroid ids + distances, build a tiny IndexFlatIP over
    those h centroid vectors. Distances are kept so TopLoc's first turn
    can reuse this scan instead of redoing it.
    Returns (cache_ids, cache_dists, cache_index, build_ms_per_conv).
    """
    quantizer = faiss.downcast_index(index.quantizer)
    d = q_emb.shape[1]
    t0 = time.perf_counter()
    turn0_vecs = q_emb[turn0_idx]
    Dq0, Iq0 = quantizer.search(turn0_vecs, h)        # (n_convs, h), sorted
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
# TopLoc follow-up search (the part that must be fast)
# -----------------------

def toploc_followup(index, q_emb, later_by_conv, cache_ids, cache_index,
                    nprobe, top_k):
    fu_rows = np.concatenate([r for r in later_by_conv if r.size]) \
        if any(r.size for r in later_by_conv) else np.empty(0, dtype=np.int32)
    if fu_rows.size == 0:
        return None, None, fu_rows

    n_fu = fu_rows.size
    Iq = np.empty((n_fu, nprobe), dtype=np.int64)
    Dq = np.empty((n_fu, nprobe), dtype=np.float32)

    pos = 0
    for c, later in enumerate(later_by_conv):
        if later.size == 0:
            continue
        D_local, I_local = cache_index[c].search(q_emb[later], nprobe)
        sl = slice(pos, pos + later.size)
        Iq[sl] = cache_ids[c][I_local]
        Dq[sl] = D_local
        pos += later.size

    index.parallel_mode = PM_PREASSIGNED
    index.nprobe = nprobe
    D, I = index.search_preassigned(q_emb[fu_rows], top_k, Iq, Dq)
    return D, I, fu_rows


# -----------------------
# TopLoc first turn: reuse the cache-build scan (no re-scan of 32k)
# -----------------------

def toploc_firstturn(index, q_emb, turn0_idx, cache_ids, cache_dists,
                     nprobe, top_k):
    """
    Quantizer output in the cache is already sorted, so the top-nprobe
    centroids for turn 0 are just the first nprobe cache entries.
    This is the MARGINAL first-turn cost (cache assumed to exist).
    Result is identical to plain IVF at the same nprobe by construction.
    """
    n = turn0_idx.size
    Iq = np.stack([cache_ids[c][:nprobe]   for c in range(n)]).astype(np.int64)
    Dq = np.stack([cache_dists[c][:nprobe] for c in range(n)]).astype(np.float32)
    index.parallel_mode = PM_PREASSIGNED
    index.nprobe = nprobe
    D, I = index.search_preassigned(q_emb[turn0_idx], top_k, Iq, Dq)
    return D, I


# -----------------------
# Full TopLoc run dict for metrics
# -----------------------

def build_run_toploc(index, q_emb, q_ids, all_ids, turn0_idx, conv_later_idx,
                     cache_ids, cache_dists, cache_index, nprobe, top_k):
    run = {}
    _, I0 = toploc_firstturn(index, q_emb, turn0_idx, cache_ids, cache_dists,
                             nprobe, top_k)
    for r, row in enumerate(turn0_idx):
        run[str(q_ids[row])] = [str(all_ids[i]) for i in I0[r] if i != -1]
    _, Ifu, fu_rows = toploc_followup(index, q_emb, conv_later_idx,
                                      cache_ids, cache_index, nprobe, top_k)
    if fu_rows.size:
        for r, row in enumerate(fu_rows):
            run[str(q_ids[row])] = [str(all_ids[i]) for i in Ifu[r] if i != -1]
    return run


def save_results(results):
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ======================================================================
# Load everything
# ======================================================================
log.info(f"Loading index from {INDEX_PATH} (mmap={USE_MMAP})")
index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP if USE_MMAP else 0)
log.info(f"Index loaded: {index.ntotal:,} vectors, nlist={index.nlist}")

_quantizer    = faiss.downcast_index(index.quantizer)
all_centroids = _quantizer.reconstruct_n(0, index.nlist)
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
later_all = np.concatenate([r for r in conv_later_idx if r.size]) \
    if any(r.size for r in conv_later_idx) else np.empty(0, dtype=np.int32)
all_rows  = np.arange(len(q_ids), dtype=np.int32)
n_first, n_fu = int(turn0_idx.size), int(later_all.size)
avg_followups = n_fu / max(1, n_first)
log.info(f"Conversations: {n_first} | follow-up queries: {n_fu} "
         f"| avg follow-ups/conv: {avg_followups:.2f}")

results = {
    "config": {
        "nprobe_values": NPROBE_VALUES, "h_values": H_VALUES,
        "k_latency_values": K_LATENCY_VALUES, "k_metrics": K_METRICS,
        "warmup_runs": WARMUP_RUNS, "timed_runs": TIMED_RUNS,
        "n_conversations": n_first, "n_followup_queries": n_fu,
        "avg_followups_per_conv": avg_followups,
        "parallel_mode_fused": PM_FUSED,
        "parallel_mode_preassigned": PM_PREASSIGNED,
    },
    "baseline": {},
    "toploc": {},
}

# ======================================================================
# Pass 1: fused baseline + preassigned control, once per nprobe.
# ======================================================================
for nprobe in NPROBE_VALUES:
    log.info(f"=== baseline nprobe={nprobe} ===")

    # Recall over ALL queries at K_METRICS (one number set per nprobe).
    Ib = baseline_search_rows(index, q_emb, all_rows, nprobe, K_METRICS)
    run_base = {str(q_ids[r]): [str(all_ids[i]) for i in Ib[r] if i != -1]
                for r in range(len(q_ids))}
    eval_base = evaluate(run_base, qrels)

    # Latency on follow-up rows and first-turn rows, per k, for BOTH the
    # fused baseline and the preassigned control (same rows, same k).
    lat = {}
    for k in K_LATENCY_VALUES:
        fu_ms = timeit(lambda k=k: baseline_search_rows(
            index, q_emb, later_all, nprobe, k)) / max(1, n_fu)
        ft_ms = timeit(lambda k=k: baseline_search_rows(
            index, q_emb, turn0_idx, nprobe, k)) / max(1, n_first)
        ctrl_fu_ms = timeit(lambda k=k: preassigned_control_rows(
            index, q_emb, later_all, nprobe, k)) / max(1, n_fu)
        ctrl_ft_ms = timeit(lambda k=k: preassigned_control_rows(
            index, q_emb, turn0_idx, nprobe, k)) / max(1, n_first)
        lat[str(k)] = {
            "followup_ms_per_query": fu_ms,
            "firstturn_ms_per_query": ft_ms,
            "control_followup_ms_per_query": ctrl_fu_ms,
            "control_firstturn_ms_per_query": ctrl_ft_ms,
            "control_vs_fused_followup": fu_ms / ctrl_fu_ms
                if ctrl_fu_ms > 0 else float("nan"),
        }
        log.info(f"  k={k:>4}: fused fu={fu_ms:7.3f} ft={ft_ms:7.3f} | "
                 f"control fu={ctrl_fu_ms:7.3f} ft={ctrl_ft_ms:7.3f} | "
                 f"fused/control={fu_ms / ctrl_fu_ms if ctrl_fu_ms else 0:5.2f}x")

    log.info(f"  MRR@10={eval_base['MRR@10']:.4f}  "
             f"NDCG@3={eval_base['NDCG@3']:.4f}  "
             f"NDCG@10={eval_base['NDCG@10']:.4f}")

    results["baseline"][str(nprobe)] = {"metrics": eval_base, "latency": lat}
    save_results(results)

# ======================================================================
# Pass 2: TopLoc. Cache built once per h, outside all timers.
# ======================================================================
for h in H_VALUES:
    if h > index.nlist:
        log.info(f"skip h={h}: exceeds nlist={index.nlist}")
        continue

    cache_ids, cache_dists, cache_index, cache_build_ms = build_caches(
        index, q_emb, h, all_centroids, turn0_idx)
    log.info(f"[h={h}] caches built once: {cache_build_ms:.2f} ms/conv "
             f"(excluded from follow-up latency; charged to first-turn "
             f"TOTAL and break-even)")

    for nprobe in NPROBE_VALUES:
        if h <= nprobe:
            continue

        base = results["baseline"][str(nprobe)]

        # ---- recall at K_METRICS (single computation; k-invariant @10) ----
        run_t = build_run_toploc(index, q_emb, q_ids, all_ids, turn0_idx,
                                 conv_later_idx, cache_ids, cache_dists,
                                 cache_index, nprobe, K_METRICS)
        eval_t = evaluate(run_t, qrels)

        # ---- latency, swept over k ----
        lat = {}
        for k in K_LATENCY_VALUES:
            fu_t_ms = timeit(lambda k=k: toploc_followup(
                index, q_emb, conv_later_idx, cache_ids, cache_index,
                nprobe, k)) / max(1, n_fu)
            ft_marginal_ms = timeit(lambda k=k: toploc_firstturn(
                index, q_emb, turn0_idx, cache_ids, cache_dists,
                nprobe, k)) / max(1, n_first)

            base_fu = base["latency"][str(k)]["followup_ms_per_query"]
            base_ft = base["latency"][str(k)]["firstturn_ms_per_query"]
            ctrl_fu = base["latency"][str(k)]["control_followup_ms_per_query"]

            ft_total_ms = cache_build_ms + ft_marginal_ms
            ft_overhead = ft_total_ms - base_ft
            speedup_fu       = base_fu / fu_t_ms if fu_t_ms > 0 else float("nan")
            speedup_vs_ctrl  = ctrl_fu / fu_t_ms if fu_t_ms > 0 else float("nan")
            impl_vs_fused    = base_fu / ctrl_fu if ctrl_fu > 0 else float("nan")

            saving_per_fu = base_fu - fu_t_ms
            breakeven_turns = (cache_build_ms / saving_per_fu
                               if saving_per_fu > 0 else float("inf"))

            # Whole-conversation view at the dataset's average length.
            conv_base   = base_ft + avg_followups * base_fu
            conv_toploc = ft_total_ms + avg_followups * fu_t_ms
            conv_speedup = conv_base / conv_toploc if conv_toploc > 0 \
                else float("nan")

            lat[str(k)] = {
                "followup_baseline_ms": base_fu,
                "followup_control_ms": ctrl_fu,
                "followup_toploc_ms": fu_t_ms,
                "followup_speedup": speedup_fu,
                "followup_speedup_vs_control": speedup_vs_ctrl,
                "control_vs_fused": impl_vs_fused,
                "firstturn_baseline_ms": base_ft,
                "firstturn_toploc_marginal_ms": ft_marginal_ms,
                "firstturn_toploc_total_ms": ft_total_ms,
                "firstturn_overhead_ms": ft_overhead,
                "breakeven_followup_turns": breakeven_turns,
                "conversation_speedup_at_avg_len": conv_speedup,
            }
            log.info(
                f"  h={h:>5} nprobe={nprobe:>4} k={k:>4} | "
                f"fu fused={base_fu:7.3f} ctrl={ctrl_fu:7.3f} "
                f"toploc={fu_t_ms:7.3f} | "
                f"speedup total={speedup_fu:5.2f}x "
                f"mech={speedup_vs_ctrl:5.2f}x impl={impl_vs_fused:5.2f}x | "
                f"breakeven={breakeven_turns:6.1f} | "
                f"conv={conv_speedup:5.2f}x"
            )

        log.info(
            f"  h={h:>5} nprobe={nprobe:>4} recall | "
            f"NDCG@10 base={base['metrics']['NDCG@10']:.4f} "
            f"toploc={eval_t['NDCG@10']:.4f} "
            f"(delta {eval_t['NDCG@10'] - base['metrics']['NDCG@10']:+.4f})"
        )

        results["toploc"][f"h={h}_nprobe={nprobe}"] = {
            "h": h, "nprobe": nprobe,
            "cache_build_ms_per_conv": cache_build_ms,
            "metrics": eval_t,
            "metrics_delta_vs_baseline": {
                m: eval_t[m] - base["metrics"][m]
                for m in ("MRR@10", "NDCG@3", "NDCG@10")
            },
            "latency": lat,
        }
        save_results(results)

    # Free per-h caches before the next (bigger) h.
    del cache_ids, cache_dists, cache_index

log.info(f"Done. Results saved to {METRICS_PATH}")
