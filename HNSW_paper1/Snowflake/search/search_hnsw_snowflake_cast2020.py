import faiss
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
import sys
import gc
import json
import time
import logging
from collections import defaultdict

# -----------------------
# Configuration
# -----------------------
INDEX_DIR      = "/home/toploc1/Datasets/toploc1/indexes/Snowflake/HNSW"
QUERY_EMB_PATH = "/home/toploc1/Datasets/toploc1/Data Exploration/cast2020_query_embeddings.parquet"
QRELS_PATH     = "/home/toploc1/Datasets/toploc1/Data Exploration/cast2020_qrels.qrel"
OUTPUT_DIR     = "/home/toploc1/Datasets/toploc1/HNSW_paper1/Snowflake/search/results/CAST2020"

# Run the whole grid once per M value. Indexes are loaded ONE AT A TIME
# (each ~150 GB) and freed before the next M loads -- never two in RAM.
M_LIST = [16, 32, 64]


def index_paths(m):
    return {
        "index": os.path.join(INDEX_DIR, f"treccast_hnsw_M{m}.index"),
        "ids":   os.path.join(INDEX_DIR, f"treccast_hnsw_idsM{m}.npy"),
    }


def output_paths(m):
    return {
        "summary": os.path.join(OUTPUT_DIR, f"snowflake_toplocHNSW_M{m}_ct2020.parquet"),
        "metrics": os.path.join(OUTPUT_DIR, f"snowflake_toplocHNSW_metrics{m}_ct2020.json"),
        "log":     os.path.join(OUTPUT_DIR, f"snowflake_toplocHNSW_M{m}_ct2020.log"),
    }


TOP_K         = 10
REL_THRESHOLD = 1
EF_LIST       = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
UP_LIST       = [2, 4, 8, 16]
BATCH_SIZE    = 50

# --- Timing methodology -------------------------------------------------
TIMING_REPS = 5
TIMING_AGG  = "median"          # "median" or "min"

MAX_MULTI_THREADS = 32
THREAD_CONFIGS = {
    "multi":  os.cpu_count(),
    "single": 1,
}

RUN_PERCONV = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base logging: stdout always; a per-M FileHandler is attached inside run_for_m.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
_LOG_FMT = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

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
# Load queries ONCE (M-independent)
# -----------------------
q_table = pq.read_table(QUERY_EMB_PATH)
q_ids   = q_table["id"].to_pylist()
q_emb   = np.array(q_table["embedding"].to_pylist(), dtype=np.float32)
faiss.normalize_L2(q_emb)
log.info(f"Queries: {len(q_ids)} | dim={q_emb.shape[1]}")


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

# Precompute the flat followup layout once; identical for every (ef, up, M).
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

# qrels loaded once
qrels = parse_qrels(QRELS_PATH)

# -----------------------
# Per-M index state (set inside run_for_m, read by the helpers below)
# -----------------------
index = None
all_ids = None
ORIG_ENTRY = None
ORIG_LEVEL = None
IS_IP = None


def restore_default_entry():
    index.hnsw.entry_point = ORIG_ENTRY
    index.hnsw.max_level   = ORIG_LEVEL


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
    times = []
    run = None
    for _ in range(reps):
        run, search_ms = run_baseline(ef)
        times.append(search_ms)
    times = np.asarray(times)
    return run, float(AGG_FN(times)), float(times.std())


# -----------------------
# TopLoc HNSW, batched
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
        eps = np.ascontiguousarray(privs[FU_CONV].astype(np.int32))
        D   = np.empty((N_FU, TOP_K), dtype=np.float32)
        I   = np.empty((N_FU, TOP_K), dtype=np.int64)

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


def warmup():
    restore_default_entry()
    index.hnsw.efSearch = 64
    index.search(q_emb, TOP_K)          # full-descent path
    run_toploc_batched(64, 1)           # search_level_0 / reconstruct path
    restore_default_entry()


# =======================================================================
# Run the entire grid for ONE M value, then free the index.
# =======================================================================
def run_for_m(m):
    global index, all_ids, ORIG_ENTRY, ORIG_LEVEL, IS_IP

    ip = index_paths(m)
    op = output_paths(m)

    # per-M log file
    m_handler = logging.FileHandler(op["log"])
    m_handler.setFormatter(_LOG_FMT)
    log.addHandler(m_handler)

    try:
        log.info(f"##################  M = {m}  ##################")
        log.info(f"Loading index from {ip['index']} (~150 GB, be patient)")
        idx = faiss.read_index(ip["index"])
        if not isinstance(idx, faiss.IndexHNSW):
            idx = faiss.downcast_index(idx)
        assert hasattr(idx, "search_level_0"), \
            "This FAISS build does not expose IndexHNSW.search_level_0 - upgrade faiss (>=1.7.3)."
        index = idx

        ORIG_ENTRY = index.hnsw.entry_point
        ORIG_LEVEL = index.hnsw.max_level
        IS_IP      = (index.metric_type == faiss.METRIC_INNER_PRODUCT)
        log.info(f"[M={m}] Index: {index.ntotal:,} vectors | entry_point={ORIG_ENTRY} | "
                 f"max_level={ORIG_LEVEL} | metric={'IP' if IS_IP else 'L2'}")

        all_ids = np.load(ip["ids"], allow_pickle=True).tolist()
        assert len(all_ids) == index.ntotal, f"{len(all_ids)} != {index.ntotal}"

        all_results = {}
        TOPLOC_VARIANTS = {"batched": run_toploc_batched}
        if RUN_PERCONV:
            TOPLOC_VARIANTS["perconv"] = run_toploc_perconv

        log.info(f"[M={m}] Timing: {TIMING_REPS} reps per config, reporting {TIMING_AGG}")

        for thread_label, n_threads in THREAD_CONFIGS.items():
            log.info(f"##### [M={m}] THREAD MODE: {thread_label} "
                     f"(omp_set_num_threads={n_threads}) #####")
            faiss.omp_set_num_threads(n_threads)
            log.info("Warmup pass ...")
            warmup()

            baseline_times = {}

            log.info(f"=== [M={m}] BASELINE HNSW [{thread_label}] ===")
            for ef in EF_LIST:
                run, search_ms, search_ms_std = run_baseline_repeated(ef)
                agg = evaluate(run, qrels)["aggregate"]
                avg_ms = search_ms / len(q_ids)
                baseline_times[ef] = avg_ms
                all_results[f"baseline_ef{ef}_{thread_label}"] = {
                    "M":                 m,
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
                log.info(f"[M={m}/{thread_label}] ef={ef:<5d} avg={avg_ms:7.3f}ms "
                         f"(±{search_ms_std / len(q_ids):6.3f}) "
                         f"MRR@10={agg['MRR@10']:.4f} NDCG@3={agg['NDCG@3']:.4f} "
                         f"NDCG@10={agg['NDCG@10']:.4f}")

            for variant, fn in TOPLOC_VARIANTS.items():
                log.info(f"=== [M={m}] TopLoc HNSW ({variant}) [{thread_label}] ===")
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
                        speedup = (base_ms / avg_ms) if base_ms and avg_ms else float("nan")
                        speedup_fu = (base_ms / sub_ms) \
                            if base_ms and sub_ms == sub_ms and sub_ms else float("nan")

                        key = f"toploc_{variant}_ef{ef}_up{up}_{thread_label}"
                        all_results[key] = {
                            "M":                      m,
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
                        log.info(f"[M={m}/{thread_label}/{variant}] up={up:<2d} ef={ef:<5d} "
                                 f"avg={avg_ms:7.3f}ms (q0={q0_ms:7.3f} sub/query={sub_ms:7.3f}) "
                                 f"speedup={speedup:.3f}x fu_speedup={speedup_fu:.3f}x "
                                 f"MRR@10={agg['MRR@10']:.4f} NDCG@3={agg['NDCG@3']:.4f} "
                                 f"NDCG@10={agg['NDCG@10']:.4f}")

        faiss.omp_set_num_threads(os.cpu_count())
        restore_default_entry()

        log.info(f"=== [M={m}] Speedup summary ===")
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
                        log.info(f"[M={m}/{thread_label}/{variant}] {label}: "
                                 f"{vals[best_key]:.3f}x  (config={best_key})")

        # -----------------------
        # Save results for this M
        # -----------------------
        with open(op["metrics"], "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"[M={m}] Grid search complete. Results saved to {op['metrics']}")

        summary = defaultdict(list)
        for key, r in all_results.items():
            summary["config"].append(key)
            summary["method"].append("toploc" if key.startswith("toploc") else "baseline")
            for k in ("M", "thread_mode", "threads", "variant", "efSearch", "up",
                      "avg_query_time_ms", "search_ms_total", "search_ms_std",
                      "q0_ms", "sub_ms", "q0_s_std", "sub_s_std", "reps",
                      "speedup_vs_baseline", "speedup_fu_vs_baseline",
                      "MRR@10", "NDCG@3", "NDCG@10", "num_queries"):
                summary[k].append(r.get(k))
        pq.write_table(pa.table(dict(summary)), op["summary"])
        log.info(f"[M={m}] Summary table -> {op['summary']}")

    finally:
        # ---- free the ~150 GB index before the next M loads ----
        index = None
        all_ids = None
        gc.collect()
        log.info(f"[M={m}] index released.")
        log.removeHandler(m_handler)
        m_handler.close()


# -----------------------
# Run every M sequentially (one index in RAM at a time)
# -----------------------
for m in M_LIST:
    run_for_m(m)

log.info(f"All M values complete: {M_LIST}")