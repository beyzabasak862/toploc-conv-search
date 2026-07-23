# ============================================================================
# Benchmark 08 producer — HYBRID FULL TREC-CAsT CACHE-WARMED B-Q_A-Q_B.
#
# SCIENTIFIC LABEL:
#   FULL TREC-CAsT DOCUMENT CORPUS /
#   FULL MS MARCO V1 DEV.SMALL QUERY WORKLOAD /
#   HYBRID FAISS QLR /
#   MULTI-CONFIG CACHE-WARMED B-Q_A-Q_B POSITION-2 OBSERVATION
#
# This is the HYBRID (Python + FAISS, TREC-CAsT) analogue of Benchmark 02.
# Benchmark 02 contributes ONLY the cache-warmed protocol and reporting style:
#   * one Python process,
#   * per-query call order [Baseline, Q_A, Q_B] (fixed query order 0..N-1),
#   * warmup + reps philosophy (default warmup=300, reps=3),
#   * per-query latency arrays + pooled statistics + BEST/BASELINE json,
#   * Q_B is the cache-warmed position-2 observation (inherits Q_A warmth).
#
# The search backend, corpus and every QLR building block come from the
# EXISTING hybrid stack:
#   * doc index          : HYBRID_DOC_INDEX (158 GB TREC-CAsT HNSW, ~38.6M docs)
#   * queries            : DEV_QUERY_DIR (ALL 6,980 MS MARCO v1 dev.small)
#   * PCA                 : PCA_MODEL joblib (same asset as Benchmarks 1/5/6)
#   * router (I_Q)        : ROUTER_INDEX (train_query_pca256_hnsw.faiss)
#   * EP table            : QLR_ARTIFACT_DIR/ep_{indices,distances}.npy
#   * exact ground truth  : EXACT_DIR/exact_indices.npy
#   * QLR algorithm       : python/faithful/faithful_qlr.py::FaithfulQLR
#
# ALGORITHM LOCK: the per-query baseline and QLR functions (`timed_baseline`,
# `timed_qlr`) and `compute_s_max` below are byte-for-byte the faithful hybrid
# implementations from python/faithful/runner.py (Benchmark 07). Only the
# surrounding execution protocol (fixed-order B-Q_A-Q_B cache-warmed loop) is
# taken from Benchmark 02. No search heuristic, threshold, adaptive policy,
# accuracy definition, normalization, fallback semantic, timing boundary or
# output field definition is changed.
#
# NEVER present the Q_B row as an isolated or paper-comparable speedup.
# ============================================================================
import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import sys
import json
import time
import argparse
import warnings
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import faiss
import joblib
from threadpoolctl import threadpool_limits


def _env_path(var, required=True, default=None):
    v = os.environ.get(var)
    if v:
        return Path(v)
    if default is not None:
        return Path(default)
    if required:
        raise RuntimeError(
            f"[cachewarmed_treccast] Required environment variable {var!r} is not set. "
            f"Source toploc_paper_2/config/paths.env or export {var!r}."
        )
    return None


# --- Package-local imports (bundled hybrid helpers + faithful algorithm) ------
_PKG_ROOT = _env_path("SUBMISSION_CODE_PKG_ROOT")
_HYBRID_LIB = str(_PKG_ROOT / "python" / "hybrid")
_FAITHFUL_LIB = str(_PKG_ROOT / "python" / "faithful")
for _p in (_HYBRID_LIB, _FAITHFUL_LIB):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.data_loading import load_embeddings_from_parquets, l2_normalize
from src.indexing import load_index
from faithful_qlr import FaithfulQLR, QLRConfig, QLRHandles

# --- External data paths from config/paths.env (all hybrid; no native vars) ---
DEV_QUERY_DIR    = _env_path("DEV_QUERY_DIR")
DOC_INDEX_PATH   = _env_path("HYBRID_DOC_INDEX")
PCA_QL_DIR       = _env_path("PCA_QL_DIR")
QUERY_INDEX_PATH = _env_path("ROUTER_INDEX", required=False) or (PCA_QL_DIR / "train_query_pca256_hnsw.faiss")
PCA_MODEL_PATH   = _env_path("PCA_MODEL",    required=False) or (PCA_QL_DIR / "pca_1024_to_256.joblib")
QLR_ARTIFACT_DIR = _env_path("QLR_ARTIFACT_DIR")
EXACT_DIR        = _env_path("EXACT_DIR")

ID_COL, EMB_COL = "id", "embedding"
NORMALIZE = True
TOPK = 10


# ============================================================================
# ==== VERBATIM from python/faithful/runner.py (algorithm lock) ==============
#   compute_s_max, timed_baseline, timed_qlr — identical semantics to
#   Benchmark 07. Reproduced here so Benchmark 08 imports no code from the
#   original repository and does not modify the Benchmark 07 producer.
# ============================================================================
def compute_s_max(ep_dist_col0: np.ndarray, quantile: float = 0.25) -> float:
    """
    Paper's s_max: 75th percentile of similarity between q_l and its top-1 doc-NN.
    Doc index metric is L2 (squared) over L2-normalized vectors:
        sim = 1 - L2^2 / 2
    So the 75th percentile of similarity == 25th percentile of squared-L2 distance.
    """
    q25_L2sq = float(np.quantile(ep_dist_col0, quantile))
    s_max = 1.0 - q25_L2sq / 2.0
    return s_max


def timed_baseline(dx, q, ef_default):
    dx.hnsw.efSearch = ef_default
    t0 = time.perf_counter_ns()
    D, I = dx.search(q, TOPK)
    t1 = time.perf_counter_ns()
    return I[0].copy(), (t1 - t0) / 1e3


def timed_qlr(qlr: FaithfulQLR, q, cfg: QLRConfig):
    """One faithful QLR query. Returns (ids, total_us, components, fallback_flag)."""
    dx = qlr.h.doc_index
    qx = qlr.h.query_index
    # PCA (timed, single-thread inside caller)
    t0 = time.perf_counter_ns()
    qp = qlr.pca_transform(q)                    # (1,256) f32
    t_pca = time.perf_counter_ns()
    # Router
    qx.hnsw.efSearch = cfg.router_ef
    Dh, Ih = qx.search(qp, cfg.kp)
    t_router = time.perf_counter_ns()
    s = float(Dh[0, 0])
    if s < cfg.th:
        # Fallback: full HNSW at ef_default
        dx.hnsw.efSearch = cfg.ef_default
        Df, If = dx.search(q, TOPK)
        t_fb = time.perf_counter_ns()
        total = (t_fb - t0) / 1e3
        comps = {
            "pca_us": (t_pca - t0) / 1e3,
            "router_us": (t_router - t_pca) / 1e3,
            "union_us": 0.0,
            "seedprep_us": 0.0,
            "beam_us": 0.0,
            "fallback_us": (t_fb - t_router) / 1e3,
            "total_us": total,
            "s": s,
            "ef_used": cfg.ef_default,
            "c_size": 0,
        }
        return If[0].copy(), total, comps, True
    # Routed: build union C
    Ih_row = Ih[0, :cfg.kp]
    ids = qlr.union_ep(Ih_row, cfg.kep)          # int32, dedup, ordered by first-occurrence
    t_union = time.perf_counter_ns()
    # Adaptive ef'
    ef_prime = qlr.adaptive_ef(s, cfg.s_max, cfg.th, cfg.ef_min, cfg.ef_default)
    # Seed dists (recompute current-query L2)
    Ic, Dc = qlr.compute_seed_dists(ids, q[0])
    t_seed = time.perf_counter_ns()
    # Seeded beam search
    ids_out, dq_out = qlr.seeded_beam(q, Ic, Dc, ef_prime, cfg.search_type)
    t_beam = time.perf_counter_ns()
    total = (t_beam - t0) / 1e3
    comps = {
        "pca_us": (t_pca - t0) / 1e3,
        "router_us": (t_router - t_pca) / 1e3,
        "union_us": (t_union - t_router) / 1e3,
        "seedprep_us": (t_seed - t_union) / 1e3,
        "beam_us": (t_beam - t_seed) / 1e3,
        "fallback_us": 0.0,
        "total_us": total,
        "s": s,
        "ef_used": ef_prime,
        "c_size": len(Ic),
    }
    return ids_out, total, comps, False
# ==== end verbatim block ====================================================


def _acc10(res_ids: np.ndarray, exact_row: np.ndarray) -> float:
    return len(set(int(x) for x in res_ids[:TOPK]) & set(int(x) for x in exact_row[:TOPK])) / TOPK


def parse_cfg(s: str, s_max: float, name: str) -> QLRConfig:
    """
    Parse "kp=20,kep=10,th=0.35,ef=112,ef_min=10,rEF=16,st=2" into a QLRConfig.

    Accepts the Benchmark-02 key names for a 1:1 mapping:
      ef   -> ef_default
      rEF  -> router_ef
      st   -> search_type (default 2 == native v2 pooled-beam equivalent)
    A legacy `backend=v2` token maps to search_type=2 (pooled beam); `backend=v1`
    maps to search_type=1 (per-seed). It is only accepted for parity with the
    Benchmark-02 config string and is otherwise ignored.
    """
    out = dict(kp=20, kep=10, th=0.30, ef_default=64, ef_min=10, router_ef=16, search_type=2)
    for kv in s.split(','):
        if not kv.strip():
            continue
        k, v = kv.split('=')
        k = k.strip(); v = v.strip()
        if k == 'ef':
            k = 'ef_default'
        elif k == 'rEF':
            k = 'router_ef'
        elif k == 'st':
            k = 'search_type'
        elif k == 'backend':
            out['search_type'] = 1 if v == 'v1' else 2
            continue
        try:
            out[k] = int(v)
        except ValueError:
            out[k] = float(v)
    return QLRConfig(kp=out['kp'], kep=out['kep'], th=out['th'],
                     ef_min=out['ef_min'], ef_default=out['ef_default'],
                     s_max=s_max, router_ef=out['router_ef'],
                     search_type=out['search_type'], name=name)


def load_all():
    print("[load] dev queries (ALL 6980)", flush=True)
    _, dev_all = load_embeddings_from_parquets(DEV_QUERY_DIR, id_col=ID_COL, emb_col=EMB_COL)
    if NORMALIZE:
        dev_all = l2_normalize(dev_all).astype(np.float32)
    dev_all = np.ascontiguousarray(dev_all)
    n_total = dev_all.shape[0]
    assert n_total == 6980, f"expected 6980 dev queries, got {n_total}"

    print("[load] PCA model (mean_, components_) from joblib", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = joblib.load(PCA_MODEL_PATH)
    pca_mean = np.ascontiguousarray(pca.mean_.astype(np.float32))
    pca_comps = np.ascontiguousarray(pca.components_.astype(np.float32))  # (256, 1024)
    assert pca_mean.shape == (1024,), pca_mean.shape
    assert pca_comps.shape == (256, 1024), pca_comps.shape
    assert not bool(getattr(pca, "whiten", False)), "hybrid PCA is expected to be non-whitening"

    print("[load] ep artifacts", flush=True)
    ep_i = np.load(QLR_ARTIFACT_DIR / "ep_indices.npy").astype(np.int32, copy=False)
    ep_d = np.load(QLR_ARTIFACT_DIR / "ep_distances.npy").astype(np.float32, copy=False)
    assert ep_i.shape[1] >= 10, f"EP width < 10: {ep_i.shape}"

    print("[load] exact GT", flush=True)
    exact_I = np.ascontiguousarray(np.load(EXACT_DIR / "exact_indices.npy"))
    assert exact_I.shape[0] == n_total, f"exact GT rows {exact_I.shape[0]} != dev {n_total}"

    print("[load] doc index (158GB TREC-CAsT, once)...", flush=True)
    t = time.time()
    doc_index = load_index(DOC_INDEX_PATH)
    print(f"[load]   ntotal={doc_index.ntotal} metric={doc_index.metric_type} in {time.time()-t:.1f}s", flush=True)

    print("[load] query index (I_Q router)...", flush=True)
    query_index = load_index(QUERY_INDEX_PATH)
    print(f"[load]   ntotal={query_index.ntotal} metric={query_index.metric_type}", flush=True)

    faiss.omp_set_num_threads(1)

    return {
        "dev_emb": dev_all,
        "pca_mean": pca_mean,
        "pca_components": pca_comps,
        "ep_indices": ep_i,
        "ep_distances": ep_d,
        "exact_I": exact_I,
        "doc_index": doc_index,
        "query_index": query_index,
        "doc_index_path": str(DOC_INDEX_PATH),
        "query_index_path": str(QUERY_INDEX_PATH),
    }


def _stats(reps):
    p = np.concatenate(reps)
    return dict(mean=float(p.mean()), median=float(np.median(p)),
                p95=float(np.quantile(p, 0.95)), p99=float(np.quantile(p, 0.99)),
                per_rep_mean=[float(r.mean()) for r in reps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6980)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baseline_ef", type=int, default=64)
    ap.add_argument("--cfg_a", type=str,
                    default="kp=20,kep=10,th=0.35,ef=112,ef_min=10,rEF=16,st=2")
    ap.add_argument("--cfg_b", type=str,
                    default="kp=20,kep=10,th=0.32,ef=112,ef_min=10,rEF=12,st=2")
    _default_out_dir = os.environ.get("OUTPUT_ROOT") or str(Path.cwd() / "outputs" / "08_cachewarmed_treccast")
    ap.add_argument("--out_dir", type=str, default=_default_out_dir)
    ap.add_argument("--core", type=int, default=21)
    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = Path(args.out_dir) / f"cachewarmed_treccast_{run_id}"
    os.makedirs(out_dir, exist_ok=False)
    print(f"[main] run_id={run_id} out_dir={out_dir}", flush=True)

    # Pin ourselves (the launcher already ran taskset -c $CORE)
    try:
        os.sched_setaffinity(0, {args.core})
    except Exception as e:
        print(f"[pin] sched_setaffinity failed: {e} (continuing; taskset should have set it)")

    data = load_all()
    dev = data["dev_emb"]
    dx = data["doc_index"]
    exact_I = data["exact_I"]
    N = min(args.n, dev.shape[0])
    print(f"[main] doc_index_path={data['doc_index_path']}", flush=True)
    print(f"[main] query_index_path={data['query_index_path']}", flush=True)

    # s_max — paper definition, hybrid L2 form (same as Benchmark 07)
    s_max = compute_s_max(data["ep_distances"][:, 0], quantile=0.25)
    print(f"[main] s_max (75th %ile top-1 doc similarity) = {s_max:.4f}", flush=True)

    cfg_a = parse_cfg(args.cfg_a, s_max, name="Q_A")
    cfg_b = parse_cfg(args.cfg_b, s_max, name="Q_B")
    cfgs = [("A", cfg_a), ("B", cfg_b)]
    print(f"[cfgs] baseline(ordinary HNSW ef={args.baseline_ef})  "
          f"Q_A={cfg_a.label()}  Q_B={cfg_b.label()}", flush=True)

    handles = QLRHandles(
        doc_index=data["doc_index"],
        query_index=data["query_index"],
        ep_indices=data["ep_indices"],
        ep_distances=data["ep_distances"],
        pca_mean=data["pca_mean"],
        pca_components=data["pca_components"],
        topk=TOPK,
    )
    qlr = FaithfulQLR(handles, max_c=250)

    # ---- warmup (Benchmark-02 philosophy: touch B + Q_A + Q_B) ----
    print(f"[warmup] {args.warmup} queries (B + Q_A + Q_B full path)", flush=True)
    with threadpool_limits(limits=1):
        for i in range(args.warmup):
            j = i % N
            q = dev[j:j+1]
            timed_baseline(dx, q, args.baseline_ef)
            for _, c in cfgs:
                timed_qlr(qlr, q, c)
    print("[warmup] done", flush=True)

    K = len(cfgs)
    B_lat = [np.zeros(N, np.float64) for _ in range(args.reps)]
    Q_lat = [[np.zeros(N, np.float64) for _ in range(args.reps)] for _ in range(K)]
    B_acc = np.zeros(N, np.float64)
    Q_acc = [np.zeros(N, np.float64) for _ in range(K)]
    Q_fb  = [np.zeros(N, bool)       for _ in range(K)]
    Q_ef  = [np.zeros(N, np.int32)   for _ in range(K)]
    Q_seed= [np.zeros(N, np.int32)   for _ in range(K)]
    Q_comp = [{k: np.zeros(N, np.float64) for k in
               ["pca_us", "router_us", "union_us", "seedprep_us", "beam_us", "fallback_us"]}
              for _ in range(K)]

    # ---- cache-warmed benchmark: fixed order 0..N-1, per query [B, Q_A, Q_B] ----
    print(f"[bench] {args.reps} reps  order: [B, Q_A, Q_B] per query (fixed 0..N-1)", flush=True)
    t0 = time.time()
    with threadpool_limits(limits=1):
        for rp in range(args.reps):
            rp_t0 = time.time()
            for i in range(N):
                q = dev[i:i+1]
                # 1. B — ordinary hybrid HNSW baseline
                b_ids, b_us = timed_baseline(dx, q, args.baseline_ef)
                B_lat[rp][i] = b_us
                if rp == 0:
                    B_acc[i] = _acc10(b_ids, exact_I[i])
                # 2/3. Q_A then Q_B — Q_B inherits Q_A's cache warmth
                for ki, (_, c) in enumerate(cfgs):
                    q_ids, q_us, comps, fb = timed_qlr(qlr, q, c)
                    Q_lat[ki][rp][i] = q_us
                    if rp == 0:
                        Q_acc[ki][i] = _acc10(q_ids, exact_I[i])
                        Q_fb[ki][i]  = fb
                        Q_ef[ki][i]  = comps["ef_used"]
                        Q_seed[ki][i]= comps["c_size"]
                        for kk in Q_comp[ki]:
                            Q_comp[ki][kk][i] = comps[kk]
            print(f"  rep {rp}: {time.time()-rp_t0:.1f}s  B={B_lat[rp].mean():.1f}us  " +
                  " ".join(f"Q{nm}={Q_lat[k][rp].mean():.1f}us" for k, (nm, _) in enumerate(cfgs)), flush=True)
    print(f"[bench] total {time.time()-t0:.1f}s", flush=True)

    Bs = _stats(B_lat)
    Qs = [_stats(Q_lat[k]) for k in range(K)]

    # Per-query arrays (Benchmark-02 transparency philosophy)
    np.save(out_dir / "PER_QUERY_BASELINE.npy", np.stack(B_lat, axis=0))
    for k, (nm, c) in enumerate(cfgs):
        np.save(out_dir / f"PER_QUERY_QLR_{nm}.npy", np.stack(Q_lat[k], axis=0))
        np.save(out_dir / f"ROUTE_MASK_{nm}.npy", ~Q_fb[k])
        np.save(out_dir / f"EF_USED_{nm}.npy", Q_ef[k])
        np.save(out_dir / f"SEED_CT_{nm}.npy", Q_seed[k])
    np.save(out_dir / "BASELINE_ACC10.npy", B_acc)

    result = {
        "benchmark_id": "08_cachewarmed_treccast",
        "scientific_label": ("FULL TREC-CAsT DOCUMENT CORPUS / FULL MS MARCO V1 DEV.SMALL "
                             "QUERY WORKLOAD / HYBRID FAISS QLR / MULTI-CONFIG CACHE-WARMED "
                             "B-Q_A-Q_B POSITION-2 OBSERVATION"),
        "backend": "hybrid_faiss_treccast",
        "document_corpus": "full TREC-CAsT (HYBRID_DOC_INDEX)",
        "query_workload": "full MS MARCO v1 dev.small (all 6,980 queries)",
        "doc_index_path": data["doc_index_path"],
        "doc_ntotal": int(dx.ntotal),
        "query_index_path": data["query_index_path"],
        "n": N, "reps": args.reps, "core": args.core, "warmup": args.warmup,
        "s_max": s_max,
        "per_query_call_order": ["baseline(ordinary HNSW, ef=%d)" % args.baseline_ef, "Q_A", "Q_B"],
        "query_order": "fixed 0..N-1",
        "one_process": True,
        "baseline": {"backend": "hybrid_hnsw", "ef": args.baseline_ef,
                     **Bs, "acc10": float(B_acc.mean())},
        "qlr": [
            {"name": nm, "config": {
                "kp": c.kp, "kep": c.kep, "th": c.th, "ef_min": c.ef_min,
                "ef_default": c.ef_default, "s_max": c.s_max, "router_ef": c.router_ef,
                "search_type": c.search_type, "label": c.label()},
             **Qs[k],
             "acc10": float(Q_acc[k].mean()),
             "fallback_rate": float(Q_fb[k].mean()),
             "ef_used_mean": float(Q_ef[k][~Q_fb[k]].mean()) if (~Q_fb[k]).any() else 0.0,
             "seed_ct_mean": float(Q_seed[k][~Q_fb[k]].mean()) if (~Q_fb[k]).any() else 0.0,
             "components_mean_us": {kk: float(vv.mean()) for kk, vv in Q_comp[k].items()},
             "speedup_mean_pooled": Bs["mean"] / Qs[k]["mean"],
             "speedup_median_pooled": Bs["median"] / Qs[k]["median"],
             "speedup_per_rep": [float(B_lat[rp].mean() / Q_lat[k][rp].mean()) for rp in range(args.reps)],
            }
            for k, (nm, c) in enumerate(cfgs)
        ],
        "caveat": ("Q_B is the position-2 cache-warmed observation: it inherits the "
                   "PCA / router / EP / doc-index cache warmth established by Q_A on the "
                   "same query. It is NOT an isolated cold-cache result and is NOT "
                   "automatically paper-comparable. Latency depends on page-cache state, "
                   "CPU load, NUMA placement, memory pressure, filesystem residency and "
                   "CPU frequency."),
    }
    with open(out_dir / "final_validation.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    with open(out_dir / "BASELINE_RESULT.json", "w") as f:
        json.dump(result["baseline"], f, indent=2, default=float)
    # Best QLR (max speedup at acc>=0.95) — reporting parity with Benchmark 02
    best = None
    for q in result["qlr"]:
        if q["acc10"] >= 0.95 and (best is None or q["speedup_mean_pooled"] > best["speedup_mean_pooled"]):
            best = q
    if best is None:
        best = max(result["qlr"], key=lambda q: q["speedup_mean_pooled"])
    with open(out_dir / "BEST_RESULT.json", "w") as f:
        json.dump(best, f, indent=2, default=float)
    with open(out_dir / "BEST_CONFIG.json", "w") as f:
        json.dump(best["config"], f, indent=2, default=float)

    with open(out_dir / "RESULT_LABEL.txt", "w") as f:
        f.write(
            "============================================================\n"
            "RESULT LABEL (mandatory when reporting Q_B):\n"
            "    FULL TREC-CAsT DOCUMENT CORPUS / FULL MS MARCO V1 DEV.SMALL\n"
            "    QUERY WORKLOAD / HYBRID FAISS QLR /\n"
            "    MULTI-CONFIG CACHE-WARMED B-Q_A-Q_B POSITION-2 OBSERVATION\n\n"
            "Never present the Q_B row as an isolated speedup or a paper-\n"
            "comparable result. Q_B (position 2) inherits Q_A's freshly warmed\n"
            "PCA weights, router index, EP rows and doc-index caches; the same-\n"
            "run baseline does not share that exact cache state.\n"
            "============================================================\n"
        )

    print("\n=== FINAL (hybrid TREC-CAsT cache-warmed) ===", flush=True)
    print(f"  Baseline (hybrid HNSW ef={args.baseline_ef}): mean={Bs['mean']:.1f}us "
          f"med={Bs['median']:.1f}us p95={Bs['p95']:.1f} acc={B_acc.mean():.4f}", flush=True)
    for k, (nm, c) in enumerate(cfgs):
        q = result["qlr"][k]
        pos = "position 2, CACHE-WARMED" if nm == "B" else "position 1"
        print(f"  Q_{nm} ({pos}) cfg={c.label()}:", flush=True)
        print(f"    mean={q['mean']:.1f}us med={q['median']:.1f}us p95={q['p95']:.1f} "
              f"acc={q['acc10']:.4f} fb={q['fallback_rate']:.4f} ef_used={q['ef_used_mean']:.1f}", flush=True)
        print(f"    speedup mean={q['speedup_mean_pooled']:.3f}x median={q['speedup_median_pooled']:.3f}x "
              f"per_rep={q['speedup_per_rep']}", flush=True)
    print(f"[DONE] wrote {out_dir}/final_validation.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
