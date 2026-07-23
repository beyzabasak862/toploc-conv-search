# ==================== CLAUDE IMPROVEMENT START ====================
# Paper 2 (QLR) FAITHFUL runner: calibration -> holdout -> full 6,980 validation.
#
# Design:
#   * Loads doc index (168.8GB) from /dev/shm/qlr_indexes/ if available, else SSHFS.
#   * Baseline HNSW ef=64 and QLR (Alg. 1) interleaved on every query in the same
#     loop (B-Q-B-Q pattern), same loaded index, same query order, same warmup.
#   * All timed FAISS single-thread; PCA under threadpool_limits(1).
#   * Deterministic split: seed=20260718 -> calibration 500, holdout 1500, full 6980.
#   * 12 calibration configs -> promote up to 3 to holdout -> promote up to 2 to full.
#   * Component timings recorded per query (PCA, router, threshold, union, adaptive,
#     seed_dist, beam, fallback, total).
#   * Every timing measured with time.perf_counter_ns().
#
# Outputs (under paper2_faithful_20260718_231400/results/<ts>/):
#   - config_manifest.json    : all 12 configs + seed + n_reps + gate thresholds
#   - baseline_sweep.json     : baseline ef in [16,24,32,40,48,64,96,128,160,200] on calib
#   - calibration.json        : per-config metrics on 500 calib queries
#   - holdout.json            : promoted configs on 1500 holdout queries
#   - full.json               : final configs on all 6980 queries (3 reps)
#   - latency_arrays/         : .npy per config for full 6980 run (base + QLR)
#
# Success criterion (paper target): full-6980 QLR acc@10 >= 0.952 AND
#   pooled QLR mean latency <= pooled baseline ef=64 mean / 1.40 (>=1.40x speedup).
# ==================== CLAUDE IMPROVEMENT END ====================
# ------------------------------------------------------------------
# toploc_paper_2 — path-only portability edits:
#   * Absolute hard-coded paths (PROJECT_ROOT, DEV_QUERY_DIR, DOC_INDEX_*,
#     QUERY_INDEX_*, PCA_MEAN, PCA_COMPONENTS, QLR_ARTIFACT_DIR, EXACT_DIR) have
#     been replaced with reads from environment variables (config/paths.env).
#   * PROJECT_ROOT is no longer prepended to sys.path; instead this package's
#     python/hybrid directory is prepended so that `from src.data_loading` and
#     `from src.indexing` resolve to the bundled helper copies.
#   * SCRIPT_DIR is still added to sys.path so `from faithful_qlr import ...`
#     resolves to the bundled paper-faithful implementation copy.
#   * The default output root SCRIPT_DIR/"results" is replaced with OUTPUT_ROOT.
#
# Algorithm, benchmark configuration, seeds, warmup, reps, calibration configs,
# routing/fallback logic, adaptive-ef formula, thresholds, gate criteria, and
# output schema are IDENTICAL to the original at
# claude_qlr_diagnostics/paper2_final_track/optimization_search/
# paper2_faithful_20260718_231400/runner.py
# (SHA256 pair recorded in manifests/COPY_MANIFEST.tsv).
# ------------------------------------------------------------------
from __future__ import annotations
import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import sys
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import faiss
from threadpoolctl import threadpool_limits


def _env_path(var, required=True, default=None):
    v = os.environ.get(var)
    if v:
        return Path(v)
    if default is not None:
        return Path(default)
    if required:
        raise RuntimeError(
            f"[faithful.runner] Required environment variable {var!r} is not set. "
            f"Source toploc_paper_2/config/paths.env or export {var!r}."
        )
    return None


# --- Package-local imports (bundled copies of src.data_loading and src.indexing) ---
_PKG_ROOT = _env_path("SUBMISSION_CODE_PKG_ROOT")
_HYBRID_LIB = str(_PKG_ROOT / "python" / "hybrid")
if _HYBRID_LIB not in sys.path:
    sys.path.insert(0, _HYBRID_LIB)
from src.data_loading import load_embeddings_from_parquets, l2_normalize
from src.indexing import load_index

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from faithful_qlr import FaithfulQLR, QLRConfig, QLRHandles

# ---------------- paths (all from config/paths.env) ----------------
DEV_QUERY_DIR    = _env_path("DEV_QUERY_DIR")
# Doc index: /dev/shm optional fast path preferred, config'd path otherwise.
DOC_INDEX_SHM    = _env_path("FAITHFUL_DOC_INDEX_SHM", required=False,
                              default="/dev/shm/qlr_indexes/treccast_hnsw_M32.index")
DOC_INDEX_SSHFS  = _env_path("HYBRID_DOC_INDEX")
# Query router (I_Q) index — same PCA-projected HNSW index as the hybrid track.
QUERY_INDEX_SHM  = _env_path("FAITHFUL_QUERY_INDEX_SHM", required=False,
                              default="/dev/shm/qlr_indexes/train_query_pca256_hnsw.faiss")
_PCA_QL_DIR      = _env_path("PCA_QL_DIR")
QUERY_INDEX_SSHFS = _env_path("ROUTER_INDEX", required=False) \
                   or (_PCA_QL_DIR / "train_query_pca256_hnsw.faiss")
# PCA extracted arrays (paper-faithful shape: (1024,) and (256, 1024))
_FAITHFUL_PCA_DIR = _env_path("FAITHFUL_PCA_DIR")
PCA_MEAN         = _env_path("FAITHFUL_PCA_MEAN", required=False) \
                   or (_FAITHFUL_PCA_DIR / "pca_mean_1024.npy")
PCA_COMPONENTS   = _env_path("FAITHFUL_PCA_COMPONENTS", required=False) \
                   or (_FAITHFUL_PCA_DIR / "pca_components_256x1024.npy")
QLR_ARTIFACT_DIR = _env_path("QLR_ARTIFACT_DIR")
EXACT_DIR        = _env_path("EXACT_DIR")

# ---------------- constants ----------------
SEED              = 20260718
N_CALIB           = 500
N_HOLDOUT         = 1500
TOPK              = 10
N_WARMUP          = 30
N_REPS_CALIB      = 2
N_REPS_HOLDOUT    = 3
N_REPS_FULL       = 3

ACC_FLOOR         = 0.952
ACC_TOL_VS_BASE   = 0.005          # QLR acc must be within 0.005 of the same-run baseline
SPEEDUP_TARGET    = 1.40           # PAPER TARGET at 95% acc

CALIB_MAX_CFG     = 12
HOLDOUT_MAX_CFG   = 3
FULL_MAX_CFG      = 2

BASELINE_EF_SWEEP = [16, 24, 32, 40, 48, 64, 96, 128, 160, 200]

# ---------------- utilities ----------------
def _acc10(res_ids: np.ndarray, exact_row: np.ndarray) -> float:
    return len(set(int(x) for x in res_ids[:TOPK]) & set(int(x) for x in exact_row[:TOPK])) / TOPK


def _stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


def _pooled_ratio_stats(base_reps: list, qlr_reps: list) -> dict:
    """base_reps, qlr_reps: list of per-rep length-N latency arrays. Returns pooled ratios."""
    base_all = np.concatenate(base_reps)
    qlr_all = np.concatenate(qlr_reps)
    per_rep_mean_ratio = [float(b.mean() / q.mean()) for b, q in zip(base_reps, qlr_reps)]
    per_rep_med_ratio = [float(np.median(b) / np.median(q)) for b, q in zip(base_reps, qlr_reps)]
    return {
        "pooled_base_mean": float(base_all.mean()),
        "pooled_qlr_mean": float(qlr_all.mean()),
        "pooled_speedup_mean": float(base_all.mean() / qlr_all.mean()),
        "pooled_base_median": float(np.median(base_all)),
        "pooled_qlr_median": float(np.median(qlr_all)),
        "pooled_speedup_median": float(np.median(base_all) / np.median(qlr_all)),
        "per_rep_mean_ratio": per_rep_mean_ratio,
        "per_rep_med_ratio": per_rep_med_ratio,
        "per_rep_mean_std": float(np.std(per_rep_mean_ratio)),
    }


# ---------------- data loading ----------------
def load_all():
    print("[load] dev queries", flush=True)
    _, dev_all = load_embeddings_from_parquets(DEV_QUERY_DIR, id_col="id", emb_col="embedding")
    dev_all = l2_normalize(dev_all).astype(np.float32)
    n_total = dev_all.shape[0]
    assert n_total == 6980, f"expected 6980 dev queries, got {n_total}"

    print("[load] pca arrays", flush=True)
    pca_mean = np.load(PCA_MEAN).astype(np.float32)
    pca_comps = np.load(PCA_COMPONENTS).astype(np.float32)
    assert pca_mean.shape == (1024,)
    assert pca_comps.shape == (256, 1024)

    print("[load] ep artifacts", flush=True)
    ep_i = np.load(QLR_ARTIFACT_DIR / "ep_indices.npy").astype(np.int32, copy=False)
    ep_d = np.load(QLR_ARTIFACT_DIR / "ep_distances.npy").astype(np.float32, copy=False)
    assert ep_i.shape[1] >= 10, f"EP width < 10: {ep_i.shape}"

    print("[load] exact GT", flush=True)
    exact_I = np.load(EXACT_DIR / "exact_indices.npy")
    exact_S = np.load(EXACT_DIR / "exact_scores.npy")
    assert exact_I.shape == (6980, 10)

    print("[load] doc index (168GB)...", flush=True)
    t = time.time()
    doc_path = DOC_INDEX_SHM if DOC_INDEX_SHM.exists() else DOC_INDEX_SSHFS
    print(f"[load]   path: {doc_path}", flush=True)
    # Prefer IO_FLAG_MMAP when loading from /dev/shm (tmpfs is RAM-backed; mmap avoids
    # duplicating the 168GB copy in FAISS-owned memory).
    if str(doc_path).startswith("/dev/shm"):
        try:
            doc_index = faiss.read_index(str(doc_path), faiss.IO_FLAG_MMAP)
            print(f"[load]   MMAP mode successful", flush=True)
        except Exception as e:
            print(f"[load]   MMAP failed ({e}); fallback to full read", flush=True)
            doc_index = load_index(doc_path)
    else:
        doc_index = load_index(doc_path)
    print(f"[load]   ntotal={doc_index.ntotal} metric={doc_index.metric_type} in {time.time()-t:.1f}s", flush=True)

    print("[load] query index (I_Q)...", flush=True)
    q_path = QUERY_INDEX_SHM if QUERY_INDEX_SHM.exists() else QUERY_INDEX_SSHFS
    print(f"[load]   path: {q_path}", flush=True)
    query_index = load_index(q_path)
    print(f"[load]   ntotal={query_index.ntotal} metric={query_index.metric_type}", flush=True)

    # single-thread for all timed searches
    faiss.omp_set_num_threads(1)

    return {
        "dev_emb": dev_all,
        "pca_mean": pca_mean,
        "pca_components": pca_comps,
        "ep_indices": ep_i,
        "ep_distances": ep_d,
        "exact_I": exact_I,
        "exact_S": exact_S,
        "doc_index": doc_index,
        "query_index": query_index,
        "doc_index_path": str(doc_path),
        "query_index_path": str(q_path),
    }


# ---------------- s_max computation ----------------
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


# ---------------- warmup ----------------
def warmup(data: dict, qlr: FaithfulQLR, n: int = N_WARMUP, s_max: float = 0.6):
    """Exercises all timed code paths (baseline, PCA, router, union, seed dists,
    seeded beam st=1 and st=2, fallback) to warm caches / branch predictors."""
    print(f"[warmup] {n} queries (full path exercise)...", flush=True)
    dx = data["doc_index"]
    qx = data["query_index"]
    dev = data["dev_emb"]
    # Warmup config: paper-like, low threshold to route most queries
    warm_cfg = QLRConfig(kp=10, kep=10, th=0.30, ef_min=10, ef_default=64,
                          s_max=s_max, router_ef=16, search_type=2, name="warmup")
    for i in range(min(n, dev.shape[0])):
        q = dev[i:i+1]
        # baseline ef64
        dx.hnsw.efSearch = 64
        dx.search(q, TOPK)
        # QLR full path
        with threadpool_limits(limits=1):
            _ids, _t, _c, _fb = timed_qlr(qlr, q, warm_cfg)
    # additional st=1 warm
    warm_cfg_st1 = QLRConfig(**{**warm_cfg.__dict__, "search_type": 1, "name": "warmup_st1"})
    for i in range(min(5, dev.shape[0])):
        q = dev[i:i+1]
        with threadpool_limits(limits=1):
            timed_qlr(qlr, q, warm_cfg_st1)
    print("[warmup] done", flush=True)


# ---------------- one timed baseline+QLR pair on one query ----------------
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


# ---------------- one config benchmark on N queries ----------------
def benchmark_config(data: dict, qlr: FaithfulQLR, cfg: QLRConfig,
                      sample_idx: np.ndarray, n_reps: int, rng: np.random.Generator):
    """
    Runs interleaved (baseline ef=64, QLR cfg) on each query in sample_idx.
    n_reps repetitions, randomized order per rep.
    Returns per-rep arrays + aggregate.
    """
    base_reps_lat = []
    qlr_reps_lat = []
    base_acc_arr = None
    qlr_acc_arr = None
    fb_flags = None
    comp_agg = {"pca_us": [], "router_us": [], "union_us": [], "seedprep_us": [],
                "beam_us": [], "fallback_us": []}
    ef_used_arr = None
    c_size_arr = None
    s_arr = None
    exact_I = data["exact_I"]
    dev = data["dev_emb"]
    dx = data["doc_index"]

    N = len(sample_idx)
    for rp in range(n_reps):
        order = rng.permutation(N)
        base_lat = np.empty(N, np.float64)
        qlr_lat = np.empty(N, np.float64)
        # accuracy + component arrays only recorded on rep 0
        if rp == 0:
            base_acc_arr = np.empty(N, np.float64)
            qlr_acc_arr = np.empty(N, np.float64)
            fb_flags = np.zeros(N, bool)
            ef_used_arr = np.empty(N, np.int32)
            c_size_arr = np.empty(N, np.int32)
            s_arr = np.empty(N, np.float64)
            for c in comp_agg:
                comp_agg[c] = np.empty(N, np.float64)

        with threadpool_limits(limits=1):
            for k, i in enumerate(order):
                i = int(sample_idx[int(i)])
                q = dev[i:i+1]
                # baseline first (BQ pair; random order across queries)
                base_ids, base_us = timed_baseline(dx, q, cfg.ef_default)
                # QLR
                qlr_ids, qlr_us, comps, fb = timed_qlr(qlr, q, cfg)
                base_lat[k] = base_us
                qlr_lat[k] = qlr_us
                if rp == 0:
                    base_acc_arr[k] = _acc10(base_ids, exact_I[i])
                    qlr_acc_arr[k] = _acc10(qlr_ids, exact_I[i])
                    fb_flags[k] = fb
                    ef_used_arr[k] = comps["ef_used"]
                    c_size_arr[k] = comps["c_size"]
                    s_arr[k] = comps["s"]
                    for c in comp_agg:
                        comp_agg[c][k] = comps[c]

        base_reps_lat.append(base_lat)
        qlr_reps_lat.append(qlr_lat)

    ratios = _pooled_ratio_stats(base_reps_lat, qlr_reps_lat)
    result = {
        "config": {
            "kp": cfg.kp, "kep": cfg.kep, "th": cfg.th,
            "ef_min": cfg.ef_min, "ef_default": cfg.ef_default,
            "s_max": cfg.s_max, "router_ef": cfg.router_ef,
            "search_type": cfg.search_type, "name": cfg.label(),
        },
        "n_queries": N,
        "n_reps": n_reps,
        "base_acc10_mean": float(base_acc_arr.mean()),
        "qlr_acc10_mean": float(qlr_acc_arr.mean()),
        "acc_delta": float(qlr_acc_arr.mean() - base_acc_arr.mean()),
        "fallback_rate": float(fb_flags.mean()),
        "base_lat_pooled": _stats(np.concatenate(base_reps_lat)),
        "qlr_lat_pooled": _stats(np.concatenate(qlr_reps_lat)),
        "speedup_pooled_mean": ratios["pooled_speedup_mean"],
        "speedup_pooled_median": ratios["pooled_speedup_median"],
        "speedup_per_rep_mean": ratios["per_rep_mean_ratio"],
        "speedup_per_rep_med": ratios["per_rep_med_ratio"],
        "rep_stability_std": ratios["per_rep_mean_std"],
        "components_mean_us": {c: float(v.mean()) for c, v in comp_agg.items()},
        "components_median_us": {c: float(np.median(v)) for c, v in comp_agg.items()},
        "ef_used_mean": float(ef_used_arr.mean()),
        "c_size_mean": float(c_size_arr.mean()),
        "s_mean": float(s_arr.mean()),
        "s_median": float(np.median(s_arr)),
        "safe_vs_base": bool(qlr_acc_arr.mean() >= ACC_FLOOR
                             and qlr_acc_arr.mean() >= base_acc_arr.mean() - ACC_TOL_VS_BASE),
        "hits_paper_target": bool(qlr_acc_arr.mean() >= ACC_FLOOR
                                  and ratios["pooled_speedup_mean"] >= SPEEDUP_TARGET),
    }
    return result, base_reps_lat, qlr_reps_lat, base_acc_arr, qlr_acc_arr, fb_flags


# ---------------- baseline sweep (untimed setup, timed measurement) ----------------
def baseline_sweep(data: dict, sample_idx: np.ndarray, ef_list: list,
                    n_reps: int, rng: np.random.Generator):
    dx = data["doc_index"]
    dev = data["dev_emb"]
    exact_I = data["exact_I"]
    N = len(sample_idx)
    out = {}
    for ef in ef_list:
        reps = []
        acc_arr = None
        for rp in range(n_reps):
            order = rng.permutation(N)
            lat = np.empty(N, np.float64)
            if rp == 0:
                acc_arr = np.empty(N, np.float64)
            with threadpool_limits(limits=1):
                for k, i in enumerate(order):
                    i = int(sample_idx[int(i)])
                    q = dev[i:i+1]
                    dx.hnsw.efSearch = ef
                    t0 = time.perf_counter_ns()
                    D, I = dx.search(q, TOPK)
                    t1 = time.perf_counter_ns()
                    lat[k] = (t1 - t0) / 1e3
                    if rp == 0:
                        acc_arr[k] = _acc10(I[0], exact_I[i])
            reps.append(lat)
        pooled = np.concatenate(reps)
        out[str(ef)] = {
            "lat": _stats(pooled),
            "acc10_mean": float(acc_arr.mean()),
            "per_rep_mean_us": [float(r.mean()) for r in reps],
            "per_rep_mean_std_us": float(np.std([r.mean() for r in reps])),
        }
    return out


# ---------------- calibration configs ----------------
def make_calib_configs(s_max: float) -> list:
    """12 paper-informed configs to test on calib. Balanced exploration."""
    C = QLRConfig
    router_ef = 16
    return [
        # (A) faithful paper baseline
        C(kp=10, kep=10, th=0.40, ef_min=10, ef_default=64, s_max=s_max, router_ef=router_ef, search_type=2, name="A_paper_kp10_kep10_th40_efmin10_ef64_st2"),
        # (B) tighter fallback
        C(kp=10, kep=10, th=0.42, ef_min=10, ef_default=64, s_max=s_max, router_ef=router_ef, search_type=2, name="B_kp10_kep10_th42_efmin10_ef64_st2"),
        # (C) mid threshold
        C(kp=10, kep=10, th=0.50, ef_min=10, ef_default=64, s_max=s_max, router_ef=router_ef, search_type=2, name="C_kp10_kep10_th50_efmin10_ef64_st2"),
        # (D) narrower ef range
        C(kp=10, kep=10, th=0.42, ef_min=16, ef_default=48, s_max=s_max, router_ef=router_ef, search_type=2, name="D_kp10_kep10_th42_efmin16_ef48_st2"),
        # (E) search_type=1 A/B
        C(kp=10, kep=10, th=0.42, ef_min=10, ef_default=64, s_max=s_max, router_ef=router_ef, search_type=1, name="E_kp10_kep10_th42_efmin10_ef64_st1"),
        # (F) wider k'
        C(kp=20, kep=10, th=0.42, ef_min=10, ef_default=64, s_max=s_max, router_ef=router_ef, search_type=2, name="F_kp20_kep10_th42_efmin10_ef64_st2"),
        # (G) narrower kep
        C(kp=10, kep=5, th=0.42, ef_min=10, ef_default=64, s_max=s_max, router_ef=router_ef, search_type=2, name="G_kp10_kep5_th42_efmin10_ef64_st2"),
        # (H) small ef_default
        C(kp=10, kep=10, th=0.42, ef_min=10, ef_default=32, s_max=s_max, router_ef=router_ef, search_type=2, name="H_kp10_kep10_th42_efmin10_ef32_st2"),
        # (I) aggressive ef_min
        C(kp=10, kep=10, th=0.42, ef_min=8, ef_default=48, s_max=s_max, router_ef=router_ef, search_type=2, name="I_kp10_kep10_th42_efmin8_ef48_st2"),
        # (J) balanced
        C(kp=10, kep=10, th=0.42, ef_min=16, ef_default=32, s_max=s_max, router_ef=router_ef, search_type=2, name="J_kp10_kep10_th42_efmin16_ef32_st2"),
        # (K) smaller k'
        C(kp=5, kep=10, th=0.42, ef_min=10, ef_default=64, s_max=s_max, router_ef=router_ef, search_type=2, name="K_kp5_kep10_th42_efmin10_ef64_st2"),
        # (L) loose fallback
        C(kp=10, kep=10, th=0.30, ef_min=10, ef_default=64, s_max=s_max, router_ef=router_ef, search_type=2, name="L_kp10_kep10_th30_efmin10_ef64_st2"),
    ]


def make_calib_configs_v2(s_max: float) -> list:
    """Extra tier of configs for router optimization + faster paths.
    Only run these if Phase 2 shows a config >= 1.30x - suggests headroom."""
    C = QLRConfig
    return [
        # (M) fast router ef=8
        C(kp=10, kep=10, th=0.42, ef_min=10, ef_default=64, s_max=s_max, router_ef=8, search_type=2, name="M_kp10_kep10_th42_routef8"),
        # (N) minimal ef range
        C(kp=10, kep=10, th=0.42, ef_min=10, ef_default=24, s_max=s_max, router_ef=16, search_type=2, name="N_kp10_kep10_th42_ef24"),
        # (O) tight kep + tight ef
        C(kp=10, kep=5, th=0.42, ef_min=10, ef_default=32, s_max=s_max, router_ef=16, search_type=2, name="O_kp10_kep5_th42_ef32"),
        # (P) k'=3 (compromise)
        C(kp=3, kep=10, th=0.42, ef_min=10, ef_default=48, s_max=s_max, router_ef=16, search_type=2, name="P_kp3_kep10_th42_ef48"),
    ]


# ---------------- main ----------------
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # OUTPUT_ROOT (set by RUN.sh) is where the wrapper wants the run dir; otherwise
    # fall back to the original SCRIPT_DIR/results/ layout for standalone use.
    _out_root = os.environ.get("OUTPUT_ROOT")
    if _out_root:
        out_dir = Path(_out_root) / f"faithful_{run_id}"
    else:
        out_dir = SCRIPT_DIR / "results" / run_id
    os.makedirs(out_dir, exist_ok=False)
    print(f"[main] run_id={run_id}", flush=True)
    print(f"[main] out_dir={out_dir}", flush=True)

    rng = np.random.default_rng(SEED)

    # Load everything once
    data = load_all()
    print(f"[main] doc_index_path={data['doc_index_path']}", flush=True)
    print(f"[main] query_index_path={data['query_index_path']}", flush=True)

    # Deterministic split
    all_idx = np.arange(6980)
    perm = rng.permutation(6980)
    calib_idx = perm[:N_CALIB]
    holdout_idx = perm[N_CALIB:N_CALIB + N_HOLDOUT]
    full_idx = np.arange(6980)  # full uses ALL queries, ordered
    print(f"[main] split: calib={len(calib_idx)} holdout={len(holdout_idx)} full={len(full_idx)}", flush=True)

    # s_max from paper definition
    s_max = compute_s_max(data["ep_distances"][:, 0], quantile=0.25)
    print(f"[main] s_max (75th %ile top-1 doc similarity) = {s_max:.4f}", flush=True)

    # Setup FaithfulQLR
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

    # Warmup
    warmup(data, qlr, n=N_WARMUP)

    # ---------------- Manifest ----------------
    manifest = {
        "run_id": run_id,
        "script": str(Path(__file__).resolve()),
        "seed": SEED,
        "topk": TOPK,
        "n_calib": N_CALIB, "n_holdout": N_HOLDOUT, "n_full": 6980,
        "n_reps_calib": N_REPS_CALIB, "n_reps_holdout": N_REPS_HOLDOUT, "n_reps_full": N_REPS_FULL,
        "acc_floor": ACC_FLOOR, "acc_tol_vs_base": ACC_TOL_VS_BASE, "speedup_target": SPEEDUP_TARGET,
        "s_max": s_max,
        "doc_index_path": data["doc_index_path"],
        "query_index_path": data["query_index_path"],
        "baseline_ef_sweep": BASELINE_EF_SWEEP,
        "calib_idx": calib_idx.tolist()[:20] + ["..."] + calib_idx.tolist()[-5:],
        "holdout_idx_sample": holdout_idx.tolist()[:10] + ["..."],
    }
    with open(out_dir / "config_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=float)

    # ---------------- Phase 0: Baseline sanity ----------------
    print("[phase0] baseline ef=64 sanity on 200 calib queries...", flush=True)
    dx = data["doc_index"]
    sanity_lat = []
    for _ in range(2):
        with threadpool_limits(limits=1):
            for i in calib_idx[:200]:
                q = data["dev_emb"][int(i):int(i)+1]
                dx.hnsw.efSearch = 64
                t0 = time.perf_counter_ns()
                dx.search(q, TOPK)
                t1 = time.perf_counter_ns()
                sanity_lat.append((t1 - t0) / 1e3)
    sanity_arr = np.array(sanity_lat)
    sanity_mean_ms = sanity_arr.mean() / 1000.0
    sanity_report = {
        "sanity_baseline_ef64_us": _stats(sanity_arr),
        "sanity_mean_ms": sanity_mean_ms,
        "regime": "LOW_NOISE" if sanity_mean_ms <= 3.0 else ("MODERATE" if sanity_mean_ms <= 10.0 else "HIGH_NOISE_SSHFS_COLD"),
    }
    with open(out_dir / "baseline_sanity.json", "w") as f:
        json.dump(sanity_report, f, indent=2, default=float)
    print(f"[phase0] baseline mean = {sanity_arr.mean():.1f} µs ({sanity_mean_ms:.2f} ms) - regime: {sanity_report['regime']}", flush=True)

    # ---------------- Phase 1: baseline sweep on calib ----------------
    print("[phase1] baseline ef sweep on calib...", flush=True)
    bs = baseline_sweep(data, calib_idx, BASELINE_EF_SWEEP, n_reps=2, rng=rng)
    with open(out_dir / "baseline_sweep.json", "w") as f:
        json.dump(bs, f, indent=2, default=float)
    for ef, r in bs.items():
        print(f"[phase1]  ef={ef:>3s}  mean={r['lat']['mean']:8.1f}µs  acc={r['acc10_mean']:.4f}  rep_std={r['per_rep_mean_std_us']:.1f}", flush=True)

    # ---------------- Phase 2: calibration - all 12 configs ----------------
    print(f"[phase2] calibration - {CALIB_MAX_CFG} configs on {N_CALIB} queries, {N_REPS_CALIB} reps each", flush=True)
    cfgs = make_calib_configs(s_max)
    calib_results = []
    for cfg in cfgs:
        try:
            print(f"[phase2] running {cfg.label()}", flush=True)
            r, _, _, _, _, _ = benchmark_config(data, qlr, cfg, calib_idx, N_REPS_CALIB, rng)
            print(f"[phase2]   acc={r['qlr_acc10_mean']:.4f}  spd_mean={r['speedup_pooled_mean']:.3f}x  fb={r['fallback_rate']:.3f}  ef_used={r['ef_used_mean']:.1f}  c={r['c_size_mean']:.1f}  safe={r['safe_vs_base']}  paper={r['hits_paper_target']}", flush=True)
            calib_results.append(r)
        except Exception as e:
            print(f"[phase2]   ERROR: {e}", flush=True)
            traceback.print_exc()
            calib_results.append({"config": {"name": cfg.label()}, "error": str(e), "trace": traceback.format_exc()})
    with open(out_dir / "calibration.json", "w") as f:
        json.dump({"n_queries": N_CALIB, "n_reps": N_REPS_CALIB, "results": calib_results, "base_ef64_mean_us": bs["64"]["lat"]["mean"], "base_ef64_acc": bs["64"]["acc10_mean"]}, f, indent=2, default=float)

    # ---------------- Phase 3: promote to holdout ----------------
    print("[phase3] promoting to holdout...", flush=True)
    safe_cfgs = [r for r in calib_results if "error" not in r and r["safe_vs_base"]]
    # Sort by pooled speedup mean descending
    safe_cfgs.sort(key=lambda r: -r["speedup_pooled_mean"])
    promoted = safe_cfgs[:HOLDOUT_MAX_CFG]
    print(f"[phase3] {len(safe_cfgs)}/{len(calib_results)} safe on calib; promoting top {len(promoted)}", flush=True)
    for p in promoted:
        print(f"[phase3]   {p['config']['name']}: acc={p['qlr_acc10_mean']:.4f}  spd={p['speedup_pooled_mean']:.3f}x", flush=True)

    holdout_results = []
    if promoted:
        # Rebuild QLRConfig objects from dict
        cfg_map = {c.label(): c for c in cfgs}
        for p in promoted:
            cfg = cfg_map[p["config"]["name"]]
            try:
                print(f"[phase3] holdout: {cfg.label()}", flush=True)
                r, _, _, _, _, _ = benchmark_config(data, qlr, cfg, holdout_idx, N_REPS_HOLDOUT, rng)
                print(f"[phase3]   HOLDOUT acc={r['qlr_acc10_mean']:.4f}  spd_mean={r['speedup_pooled_mean']:.3f}x  fb={r['fallback_rate']:.3f}  rep_std={r['rep_stability_std']:.4f}", flush=True)
                holdout_results.append(r)
            except Exception as e:
                print(f"[phase3]   ERROR: {e}", flush=True)
                traceback.print_exc()
                holdout_results.append({"config": {"name": cfg.label()}, "error": str(e)})

    with open(out_dir / "holdout.json", "w") as f:
        json.dump({"n_queries": N_HOLDOUT, "n_reps": N_REPS_HOLDOUT, "results": holdout_results}, f, indent=2, default=float)

    # ---------------- Phase 4: promote to full 6980 ----------------
    print("[phase4] promoting to full 6980...", flush=True)
    safe_holdout = [r for r in holdout_results if "error" not in r and r["safe_vs_base"]]
    safe_holdout.sort(key=lambda r: -r["speedup_pooled_mean"])
    finalists = safe_holdout[:FULL_MAX_CFG]
    print(f"[phase4] {len(safe_holdout)}/{len(holdout_results)} safe on holdout; final = {len(finalists)}", flush=True)
    for f_ in finalists:
        print(f"[phase4]   {f_['config']['name']}: holdout acc={f_['qlr_acc10_mean']:.4f}  spd={f_['speedup_pooled_mean']:.3f}x", flush=True)

    full_results = []
    lat_dir = out_dir / "latency_arrays"
    lat_dir.mkdir(exist_ok=True)
    if finalists:
        cfg_map = {c.label(): c for c in cfgs}
        for f_ in finalists:
            cfg = cfg_map[f_["config"]["name"]]
            try:
                print(f"[phase4] full 6980: {cfg.label()}", flush=True)
                r, base_reps, qlr_reps, base_acc, qlr_acc, fb = benchmark_config(data, qlr, cfg, full_idx, N_REPS_FULL, rng)
                # save arrays
                np.savez(lat_dir / f"{cfg.label()}.npz",
                         base_reps=np.stack(base_reps), qlr_reps=np.stack(qlr_reps),
                         base_acc=base_acc, qlr_acc=qlr_acc, fb=fb)
                print(f"[phase4]   FULL acc={r['qlr_acc10_mean']:.4f}  spd_mean={r['speedup_pooled_mean']:.3f}x  spd_med={r['speedup_pooled_median']:.3f}x  paper_hit={r['hits_paper_target']}", flush=True)
                full_results.append(r)
            except Exception as e:
                print(f"[phase4]   ERROR: {e}", flush=True)
                traceback.print_exc()
                full_results.append({"config": {"name": cfg.label()}, "error": str(e)})

    with open(out_dir / "full.json", "w") as f:
        json.dump({"n_queries": 6980, "n_reps": N_REPS_FULL, "results": full_results,
                   "base_ef64_pooled_mean_us": bs["64"]["lat"]["mean"],
                   "base_ef64_calib_acc": bs["64"]["acc10_mean"]}, f, indent=2, default=float)

    # ---------------- Phase 5: baseline sweep on FULL for equal-accuracy comparison ----------------
    if finalists:
        print("[phase5] baseline ef sweep on FULL 6980 (for equal-accuracy comparison)...", flush=True)
        bs_full = baseline_sweep(data, full_idx, BASELINE_EF_SWEEP, n_reps=2, rng=rng)
        with open(out_dir / "baseline_sweep_full.json", "w") as f:
            json.dump(bs_full, f, indent=2, default=float)
        for ef, r in bs_full.items():
            print(f"[phase5]  ef={ef:>3s}  mean={r['lat']['mean']:8.1f}µs  acc={r['acc10_mean']:.4f}", flush=True)
    else:
        bs_full = {}

    # ---------------- Equal-accuracy comparison (Paper Table 1 methodology) ----------------
    def _fastest_baseline_ge(target_acc):
        best = None
        for ef_s, r in bs_full.items():
            if r["acc10_mean"] >= target_acc:
                if best is None or r["lat"]["mean"] < best["lat"]["mean"]:
                    best = {"ef": int(ef_s), "lat": r["lat"], "acc": r["acc10_mean"]}
        return best

    equal_acc = {}
    for target in [0.93, 0.95, 0.952, 0.97, 0.98]:
        b = _fastest_baseline_ge(target)
        # fastest QLR config on full reaching target
        q_hits = [r for r in full_results if "error" not in r and r["qlr_acc10_mean"] >= target]
        q_hits.sort(key=lambda r: r["qlr_lat_pooled"]["mean"])
        q_best = q_hits[0] if q_hits else None
        equal_acc[f"acc_ge_{target}"] = {
            "baseline_best": b,
            "qlr_best": ({"name": q_best["config"]["name"],
                          "lat_mean_us": q_best["qlr_lat_pooled"]["mean"],
                          "acc": q_best["qlr_acc10_mean"]} if q_best else None),
            "equal_acc_speedup_mean": (b["lat"]["mean"] / q_best["qlr_lat_pooled"]["mean"]) if (b and q_best) else None,
        }

    # ---------------- Summary ----------------
    hit = [r for r in full_results if "error" not in r and r.get("hits_paper_target")]
    summary = {
        "run_id": run_id,
        "paper_target_hit_on_ef64_base": bool(hit),
        "n_configs_hit_paper_target_ef64": len(hit),
        "best_config_ef64_base": hit[0]["config"]["name"] if hit else None,
        "best_speedup_ef64_base": hit[0]["speedup_pooled_mean"] if hit else None,
        "best_acc_ef64_base": hit[0]["qlr_acc10_mean"] if hit else None,
        "safe_configs_on_full": [r["config"]["name"] for r in full_results if "error" not in r and r.get("safe_vs_base")],
        "equal_accuracy_report": equal_acc,
        "paper_target_hit_equal_acc_0_95": (equal_acc.get("acc_ge_0.95", {}).get("equal_acc_speedup_mean") or 0) >= SPEEDUP_TARGET,
        "paper_target_hit_equal_acc_0_952": (equal_acc.get("acc_ge_0.952", {}).get("equal_acc_speedup_mean") or 0) >= SPEEDUP_TARGET,
        "paper_target_hit_equal_acc_0_97": (equal_acc.get("acc_ge_0.97", {}).get("equal_acc_speedup_mean") or 0) >= 1.55,
    }
    with open(out_dir / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[DONE] {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
