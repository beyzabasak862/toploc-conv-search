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
INDEX_PATH     = "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/treccast_hnsw_M32_dragon_mips.index"
IDS_PATH       = "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/treccast_hnsw_idsM32_dragon_mips.npy"
QUERY_EMB_PATH = "/home/toploc1/Datasets/toploc1/indexes/Dragon_indexes/topics_dragon_embeddings.parquet"  # <-- adjust to your Dragon query parquet
QRELS_PATH     = "/home/toploc1/Datasets/conversational/CAST2019/topics/qrels.qrel"
OUTPUT_DIR     = "/home/toploc1/Datasets/toploc1/HNSW_paper1"

SUMMARY_PATH   = os.path.join(OUTPUT_DIR, "grid_summary_level0_fu_dragon_final.parquet")
METRICS_PATH   = os.path.join(OUTPUT_DIR, "grid_metrics_level0_fu_dragon_final.json")
LOG_PATH       = os.path.join(OUTPUT_DIR, "search_grid_hnsw_level0_fu_dragon_final.log")

TOP_K         = 10
REL_THRESHOLD = 1
EF_LIST       = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
UP_LIST       = [1, 2, 4, 8, 16]
BATCH_SIZE    = 50

# --- Timing methodology -------------------------------------------------
# Each config is executed TIMING_REPS times and the reported time is the
# aggregate below.
TIMING_REPS = 5
TIMING_AGG  = "median"          # "median" or "min"



MAX_MULTI_THREADS = 32
THREAD_CONFIGS = {
    "multi":  os.cpu_count(),
    "single": 1,
}

# Also run the old serialized per-conversation TopLoc variant for comparison.
RUN_PERCONV = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

AGG_FN = {"median": np.median, "min": np.min}[TIMING_AGG]


# -----------------------
# Metrics
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
    dcg = sum(grades.get(doc, 0) / np.log2(rank + 1)
              for rank, doc in enumerate(retrieved[:k], start=1))
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


# -----------------------
# Load index / ids / queries
# -----------------------
log.info(f"Loading index from {INDEX_PATH}")
index = faiss.read_index(INDEX_PATH)
if not isinstance(index, faiss.IndexHNSW):
    index = faiss.downcast_index(index)
assert hasattr(index, "search_level_0"), \
    "This FAISS build does not expose IndexHNSW.search_level_0 - upgrade faiss (>=1.7.3)."

ORIG_ENTRY = index.hnsw.entry_point
ORIG_LEVEL = index.hnsw.max_level
IS_IP      = (index.metric_type == faiss.METRIC_INNER_PRODUCT)
log.info(f"Index: {index.ntotal:,} vectors | entry_point={ORIG_ENTRY} | "
         f"max_level={ORIG_LEVEL} | metric={'IP' if IS_IP else 'L2'}")

all_ids = np.load(IDS_PATH, allow_pickle=True).tolist()
assert len(all_ids) == index.ntotal, f"{len(all_ids)} != {index.ntotal}"

q_table = pq.read_table(QUERY_EMB_PATH)
q_ids   = q_table["id"].to_pylist()
q_emb   = np.array(q_table["embedding"].to_pylist(), dtype=np.float32)
# MIPS (Bachrach) query transform: q -> [q, 0]. DO NOT normalize -- the
# index stores phi(x) = [x, sqrt(Mmax^2 - ||x||^2)], and L2 search on
# ([q, 0], phi(x)) ranks exactly by the raw inner product <q, x>.
q_emb = np.ascontiguousarray(
    np.hstack([q_emb, np.zeros((len(q_emb), 1), dtype=np.float32)]))
assert q_emb.shape[1] == index.d, \
    f"query dim {q_emb.shape[1]} != index dim {index.d} " \
    "(is this the MIPS-transformed 769-dim index?)"
log.info(f"Queries: {len(q_ids)} | dim={q_emb.shape[1]} (768 + 1 MIPS dim)")


# -----------------------
# Group queries into conversations, ordered by turn
# -----------------------
def conv_turn(qid):
    conv, _, turn = qid.rpartition("_")
    if conv and turn.isdigit():
        return conv, int(turn)
    return qid, 0

conversations = defaultdict(list)
for i, qid in enumerate(q_ids):
    conv, turn = conv_turn(qid)
    conversations[conv].append((turn, i, qid))
for conv in conversations:
    conversations[conv].sort()
log.info(f"Conversations: {len(conversations)} | "
         f"turns/conv min={min(len(v) for v in conversations.values())} "
         f"max={max(len(v) for v in conversations.values())}")


def restore_default_entry():
    index.hnsw.entry_point = ORIG_ENTRY
    index.hnsw.max_level   = ORIG_LEVEL


# Precompute the flat followup layout once; identical for every (ef, up) config.
CONVS = list(conversations.values())
FU_ROWS, FU_CONV, FU_QIDS = [], [], []
for ci, items in enumerate(CONVS):
    for _, i, qid in items[1:]:
        FU_ROWS.append(i)
        FU_CONV.append(ci)
        FU_QIDS.append(qid)
FU_ROWS  = np.array(FU_ROWS, dtype=np.int64)
FU_CONV  = np.array(FU_CONV, dtype=np.int64)
FU_BATCH = np.ascontiguousarray(q_emb[FU_ROWS])
N_FU     = len(FU_ROWS)
log.info(f"Followup queries: {N_FU} (single batched level-0 call per config)")


def seed_distances(qs, ep_vecs):
    """Initial candidate distance in FAISS's internal convention
    (smaller = better): -<q,x> for inner product, squared L2 otherwise."""
    if IS_IP:
        return np.ascontiguousarray(
            -np.einsum("ij,ij->i", qs, ep_vecs)).astype(np.float32)
    diff = qs - ep_vecs
    return np.ascontiguousarray(
        np.einsum("ij,ij->i", diff, diff)).astype(np.float32)


# -----------------------
# Baseline HNSW: full descent, BATCH_SIZE chunks
# -----------------------
def run_baseline(ef):
    restore_default_entry()
    index.hnsw.efSearch = ef
    all_indices = []
    search_ms = 0.0
    for start in range(0, len(q_emb), BATCH_SIZE):
        batch = q_emb[start:start + BATCH_SIZE]
        t0 = time.perf_counter()
        D, I = index.search(batch, TOP_K)
        search_ms += (time.perf_counter() - t0) * 1e3
        all_indices.append(I)
    indices = np.vstack(all_indices)

    run = {}
    for qi, qid in enumerate(q_ids):
        run[qid] = [str(all_ids[idx]) for idx in indices[qi] if idx != -1]
    return run, search_ms


def run_baseline_repeated(ef, reps=TIMING_REPS):
    """Run the baseline `reps` times; results are deterministic so the run
    dict from the last repetition is kept. Times are aggregated with
    TIMING_AGG to suppress scheduler / OpenMP jitter."""
    times = []
    run = None
    for _ in range(reps):
        run, search_ms = run_baseline(ef)
        times.append(search_ms)
    times = np.asarray(times)
    return run, float(AGG_FN(times)), float(times.std())


# -----------------------
# TopLoc HNSW, batched:
#   q0 phase : one index.search() over all first turns, ef = base_ef * up
#   followups: ONE search_level_0() call over ALL followups of ALL
#              conversations, each seeded with its own privileged entry point.
# -----------------------
def run_toploc_batched(base_ef, up):
    run = {}

    # --- q0 phase ---
    restore_default_entry()
    index.hnsw.efSearch = base_ef * up
    q0_rows  = [items[0][1] for items in CONVS]
    q0_batch = np.ascontiguousarray(q_emb[q0_rows])
    t0 = time.perf_counter()
    D0, I0 = index.search(q0_batch, TOP_K)
    q0_time = time.perf_counter() - t0
    for ci, items in enumerate(CONVS):
        run[items[0][2]] = [str(all_ids[idx]) for idx in I0[ci] if idx != -1]
    privs = np.where(I0[:, 0] != -1, I0[:, 0], ORIG_ENTRY).astype(np.int64)

    # --- followup phase: single level-0 call, per-query entry points ---
    sub_time = 0.0
    if N_FU:
        index.hnsw.efSearch = base_ef
        # NOTE: search_level_0 expects storage_idx_t (= int32) entry points.
        eps = np.ascontiguousarray(privs[FU_CONV].astype(np.int32))
        D   = np.empty((N_FU, TOP_K), dtype=np.float32)
        I   = np.empty((N_FU, TOP_K), dtype=np.int64)

        # SWIG raw-pointer calls have no dtype checking beyond overload
        # matching -- fail loudly instead of producing garbage/segfaults.
        assert FU_BATCH.dtype == np.float32 and FU_BATCH.flags.c_contiguous
        assert eps.dtype == np.int32
        assert D.dtype == np.float32 and I.dtype == np.int64

        t0 = time.perf_counter()
        ep_vecs = index.reconstruct_batch(eps.astype(np.int64))   # (N_FU, d)
        d_seed  = seed_distances(FU_BATCH, ep_vecs)               # (N_FU,)
        assert d_seed.dtype == np.float32
        index.search_level_0(
            N_FU,
            faiss.swig_ptr(FU_BATCH),
            TOP_K,
            faiss.swig_ptr(eps),
            faiss.swig_ptr(d_seed),
            faiss.swig_ptr(D),
            faiss.swig_ptr(I),
            1,   # nprobe: one seed per query
            1,   # search_type 1: one search per seed
        )
        sub_time = time.perf_counter() - t0

        for qid, Irow in zip(FU_QIDS, I):
            run[qid] = [str(all_ids[idx]) for idx in Irow if idx != -1]

    return run, q0_time, sub_time, len(CONVS), N_FU


# -----------------------
# Old serialized variant (kept for diagnosis, off by default)
# -----------------------
def run_toploc_perconv(base_ef, up):
    run = {}
    restore_default_entry()
    index.hnsw.efSearch = base_ef * up
    q0_rows  = [items[0][1] for items in CONVS]
    q0_batch = np.ascontiguousarray(q_emb[q0_rows])
    t0 = time.perf_counter()
    D0, I0 = index.search(q0_batch, TOP_K)
    q0_time = time.perf_counter() - t0
    for ci, items in enumerate(CONVS):
        run[items[0][2]] = [str(all_ids[idx]) for idx in I0[ci] if idx != -1]
    privs = [int(I0[ci, 0]) if I0[ci, 0] != -1 else ORIG_ENTRY
             for ci in range(len(CONVS))]

    index.hnsw.max_level = 0
    index.hnsw.efSearch  = base_ef
    sub_time, n_sub = 0.0, 0
    for ci, items in enumerate(CONVS):
        followups = items[1:]
        if not followups:
            continue
        rows  = [i for _, i, _ in followups]
        batch = np.ascontiguousarray(q_emb[rows])
        index.hnsw.entry_point = privs[ci]
        t0 = time.perf_counter()
        D, I = index.search(batch, TOP_K)
        sub_time += time.perf_counter() - t0
        n_sub += len(followups)
        for (_, _, qid), Irow in zip(followups, I):
            run[qid] = [str(all_ids[idx]) for idx in Irow if idx != -1]
    restore_default_entry()
    return run, q0_time, sub_time, len(CONVS), n_sub


def run_toploc_repeated(fn, base_ef, up, reps=TIMING_REPS):
    """Repeat a TopLoc variant `reps` times; aggregate q0/sub times
    independently with TIMING_AGG. The run dict is deterministic."""
    q0_times, sub_times = [], []
    run, n_q0, n_sub = None, 0, 0
    for _ in range(reps):
        run, q0_time, sub_time, n_q0, n_sub = fn(base_ef, up)
        q0_times.append(q0_time)
        sub_times.append(sub_time)
    q0_times, sub_times = np.asarray(q0_times), np.asarray(sub_times)
    return (run,
            float(AGG_FN(q0_times)),  float(AGG_FN(sub_times)),
            float(q0_times.std()),    float(sub_times.std()),
            n_q0, n_sub)


# -----------------------
# Warmup: run AFTER every omp_set_num_threads() call, because changing
# the thread count reconfigures the OpenMP pool and invalidates any
# previous warmup. Warms both code paths (full search + level-0).
# -----------------------
def warmup():
    restore_default_entry()
    index.hnsw.efSearch = 64
    index.search(q_emb, TOP_K)          # full-descent path
    run_toploc_batched(64, 1)           # search_level_0 / reconstruct path
    restore_default_entry()


# -----------------------
# Grid search
# -----------------------
qrels = parse_qrels(QRELS_PATH)

all_results = {}

TOPLOC_VARIANTS = {"batched": run_toploc_batched}
if RUN_PERCONV:
    TOPLOC_VARIANTS["perconv"] = run_toploc_perconv

log.info(f"Timing: {TIMING_REPS} reps per config, reporting {TIMING_AGG}")

for thread_label, n_threads in THREAD_CONFIGS.items():
    log.info(f"##### THREAD MODE: {thread_label} (omp_set_num_threads={n_threads}) #####")
    faiss.omp_set_num_threads(n_threads)
    log.info("Warmup pass ...")
    warmup()

    baseline_times = {}

    log.info(f"=== BASELINE HNSW [{thread_label}] ===")
    for ef in EF_LIST:
        run, search_ms, search_ms_std = run_baseline_repeated(ef)
        agg = evaluate(run, qrels)["aggregate"]
        avg_ms = search_ms / len(q_ids)
        baseline_times[ef] = avg_ms
        all_results[f"baseline_ef{ef}_{thread_label}"] = {
            "thread_mode":       thread_label,
            "threads":           n_threads,
            "efSearch":          ef,
            "avg_query_time_ms": avg_ms,
            "search_ms_total":   search_ms,
            "search_ms_std":     search_ms_std,
            "reps":              TIMING_REPS,
            "MRR@10":            agg["MRR@10"],
            "NDCG@3":            agg["NDCG@3"],
            "NDCG@10":           agg["NDCG@10"],
            "num_queries":       agg["num_queries"],
        }
        log.info(f"[{thread_label}] ef={ef:<5d} avg={avg_ms:7.3f}ms "
                 f"(±{search_ms_std / len(q_ids):6.3f}) "
                 f"MRR@10={agg['MRR@10']:.4f} NDCG@3={agg['NDCG@3']:.4f} "
                 f"NDCG@10={agg['NDCG@10']:.4f}")

    for variant, fn in TOPLOC_VARIANTS.items():
        log.info(f"=== TopLoc HNSW ({variant}) [{thread_label}] ===")
        for up in UP_LIST:
            for ef in EF_LIST:
                (run, q0_time, sub_time, q0_std, sub_std,
                 n_q0, n_sub) = run_toploc_repeated(fn, ef, up)
                agg = evaluate(run, qrels)["aggregate"]
                search_ms = (q0_time + sub_time) * 1e3
                total_q   = n_q0 + n_sub
                avg_ms    = search_ms / total_q
                q0_ms     = q0_time / n_q0 * 1e3
                sub_ms    = (sub_time / n_sub * 1e3) if n_sub else float("nan")

                base_ms = baseline_times.get(ef)
                # speedup incl. q0 (amortized) vs baseline at the same ef
                speedup = (base_ms / avg_ms) if base_ms and avg_ms else float("nan")
                # steady-state speedup: followup turns only, q0 excluded.
                # sub_ms == sub_ms filters the NaN case (no followups).
                speedup_fu = (base_ms / sub_ms) \
                    if base_ms and sub_ms == sub_ms and sub_ms else float("nan")

                key = f"toploc_{variant}_ef{ef}_up{up}_{thread_label}"
                all_results[key] = {
                    "thread_mode":            thread_label,
                    "threads":                n_threads,
                    "variant":                variant,
                    "efSearch":               ef,
                    "up":                     up,
                    "avg_query_time_ms":      avg_ms,
                    "search_ms_total":        search_ms,
                    "q0_ms":                  q0_ms,
                    "sub_ms":                 sub_ms,
                    "q0_s_std":               q0_std,
                    "sub_s_std":              sub_std,
                    "reps":                   TIMING_REPS,
                    "speedup_vs_baseline":    speedup,
                    "speedup_fu_vs_baseline": speedup_fu,
                    "MRR@10":                 agg["MRR@10"],
                    "NDCG@3":                 agg["NDCG@3"],
                    "NDCG@10":                agg["NDCG@10"],
                    "num_queries":            agg["num_queries"],
                }
                log.info(f"[{thread_label}/{variant}] up={up:<2d} ef={ef:<5d} "
                         f"avg={avg_ms:7.3f}ms (q0={q0_ms:7.3f} sub/query={sub_ms:7.3f}) "
                         f"speedup={speedup:.3f}x fu_speedup={speedup_fu:.3f}x "
                         f"MRR@10={agg['MRR@10']:.4f} NDCG@3={agg['NDCG@3']:.4f} "
                         f"NDCG@10={agg['NDCG@10']:.4f}")

faiss.omp_set_num_threads(os.cpu_count())
restore_default_entry()

log.info("=== Speedup summary ===")
for thread_label in THREAD_CONFIGS:
    for variant in TOPLOC_VARIANTS:
        subset = {
            k: r for k, r in all_results.items()
            if k.startswith(f"toploc_{variant}") and k.endswith(f"_{thread_label}")
        }
        for col, label in (("speedup_vs_baseline",    "best speedup (incl. q0)"),
                           ("speedup_fu_vs_baseline", "best fu_speedup (followups only)")):
            vals = {k: r[col] for k, r in subset.items() if r.get(col) == r.get(col)}
            if vals:
                best_key = max(vals, key=vals.get)
                log.info(f"[{thread_label}/{variant}] {label}: "
                         f"{vals[best_key]:.3f}x  (config={best_key})")

# -----------------------
# Save results
# -----------------------
with open(METRICS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
log.info(f"Grid search complete. Results saved to {METRICS_PATH}")

summary = defaultdict(list)
for key, r in all_results.items():
    summary["config"].append(key)
    summary["method"].append("toploc" if key.startswith("toploc") else "baseline")
    for k in ("thread_mode", "threads", "variant", "efSearch", "up",
              "avg_query_time_ms", "search_ms_total", "search_ms_std",
              "q0_ms", "sub_ms", "q0_s_std", "sub_s_std", "reps",
              "speedup_vs_baseline", "speedup_fu_vs_baseline",
              "MRR@10", "NDCG@3", "NDCG@10", "num_queries"):
        summary[k].append(r.get(k))
pq.write_table(pa.table(dict(summary)), SUMMARY_PATH)
log.info(f"Summary table -> {SUMMARY_PATH}")
