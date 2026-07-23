# ==================== CLAUDE IMPROVEMENT START: Stage-2 accuracy-recovery grid (NEW FILE; scripts/son.py & Paper 1 untouched) ====================
# ------------------------------------------------------------------
# SUBMISSION_CODE_PACKAGE — path-only portability edits:
#   * Absolute hard-coded paths replaced with reads from environment variables
#     (config/paths.env: DEV_QUERY_DIR, HYBRID_DOC_INDEX, PCA_QL_DIR,
#     ROUTER_INDEX, PCA_MODEL, PCA_QMAX, QLR_ARTIFACT_DIR, EXACT_DIR).
#   * PROJECT_ROOT is no longer prepended to sys.path; instead this package's
#     python/hybrid directory is prepended so that `from src.data_loading` and
#     `from src.indexing` resolve to the bundled helper copies.
#   * OUT_DIR is chosen from OUTPUT_ROOT (if set by RUN.sh).
#
# Algorithm, benchmark configuration (N_MAIN=500, SEED=20260717, N_REPS=3,
# N_WARMUP=50, RS=0.50, EF_DEFAULT=64, ROUTER_EF=16, SEED_MODES, NPROBES,
# SEEDED_EFS, 24-config grid, mixture-mean speedup, output schema) are IDENTICAL
# to the original at claude_qlr_diagnostics/rescue_stage2_accuracy.py
# (byte-identical SHA256 recorded in manifests/COPY_MANIFEST.tsv).
# ------------------------------------------------------------------
import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import sys
import csv
import json
import time
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
            f"[rescue_stage2_accuracy] Required environment variable {var!r} is not set. "
            f"Source SUBMISSION_CODE_PACKAGE/config/paths.env or export {var!r}."
        )
    return None


# --- Package-local imports (bundled copies of src.data_loading and src.indexing) ---
_PKG_ROOT = _env_path("SUBMISSION_CODE_PKG_ROOT")
_HYBRID_LIB = str(_PKG_ROOT / "python" / "hybrid")
if _HYBRID_LIB not in sys.path:
    sys.path.insert(0, _HYBRID_LIB)
from src.data_loading import load_embeddings_from_parquets, l2_normalize
from src.indexing import load_index

# ---- paths: identical to scripts/son.py (read-only) ----
DEV_QUERY_DIR    = _env_path("DEV_QUERY_DIR")
DOC_INDEX_PATH   = _env_path("HYBRID_DOC_INDEX")
PCA_QL_DIR       = _env_path("PCA_QL_DIR")
QUERY_INDEX_PATH = _env_path("ROUTER_INDEX", required=False) or (PCA_QL_DIR / "train_query_pca256_hnsw.faiss")
PCA_MODEL_PATH   = _env_path("PCA_MODEL",    required=False) or (PCA_QL_DIR / "pca_1024_to_256.joblib")
QMAX_PATH        = _env_path("PCA_QMAX",     required=False) or (PCA_QL_DIR / "qmax_pca256.npy")
QLR_ARTIFACT_DIR = _env_path("QLR_ARTIFACT_DIR")
EXACT_DIR        = _env_path("EXACT_DIR")

# ---- config ----
ID_COL, EMB_COL = "id", "embedding"
NORMALIZE = True
TOPK = 10
RS = 0.50
EF_DEFAULT = 64          # baseline + fallback doc-search efSearch
ROUTER_EF = 16           # FIX 1 (validated Stage 1)

SEED_MODES  = ["cached", "recompute_l2"]
NPROBES     = [3, 5, 10]
SEEDED_EFS  = [16, 32, 64, 128]

N_MAIN, N_REPS, N_WARMUP = 500, 3, 50
SEED = 20260717                       # SAME sample as Stage 1 (direct comparability)
ACC_FLOOR = 0.952                     # baseline 0.962 - tol 0.01 ; hard gate for any speedup claim
BASELINE_ACC = 0.962                  # reference (re-measured live below)
ACC_TOL = 0.01
rng = np.random.default_rng(SEED)

RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
_OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT")
if _OUTPUT_ROOT:
    OUT_DIR = Path(_OUTPUT_ROOT) / f"stage2_{RUN_ID}"
else:
    OUT_DIR = Path.cwd() / "outputs" / "06_stage2_bounded_pareto" / f"stage2_{RUN_ID}"
os.makedirs(OUT_DIR, exist_ok=False)  # never overwrite

results = {"run_id": RUN_ID, "stage": 2, "seed": SEED,
           "config": {"RS": RS, "EF_DEFAULT": EF_DEFAULT, "ROUTER_EF": ROUTER_EF,
                      "SEED_MODES": SEED_MODES, "NPROBES": NPROBES, "SEEDED_EFS": SEEDED_EFS,
                      "TOPK": TOPK, "N_MAIN": N_MAIN, "N_REPS": N_REPS, "N_WARMUP": N_WARMUP,
                      "ACC_FLOOR": ACC_FLOOR, "ACC_TOL": ACC_TOL},
           "loadavg": {}}

def loadavg():
    try:
        return list(os.getloadavg())
    except Exception:
        return None
results["loadavg"]["at_start"] = loadavg()

def dump():
    with open(OUT_DIR / "stage2_accuracy.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

def acc10(res, ex):
    return len(set(int(x) for x in res[:TOPK]) & set(int(x) for x in ex[:TOPK])) / TOPK

def agg(list_of_arrays):
    allv = np.concatenate(list_of_arrays)
    rm = np.array([a.mean() for a in list_of_arrays])
    return {"mean": float(allv.mean()), "median": float(np.median(allv)),
            "p95": float(np.quantile(allv, 0.95)), "min": float(allv.min()),
            "rep_mean_mean": float(rm.mean()), "rep_mean_std": float(rm.std()),
            "n_per_rep": int(len(list_of_arrays[0]))}

# ------------------------------------------------------------------ load small artifacts (same as Stage 1)
print("[load] dev queries + PCA + router index + EP + exact GT", flush=True)
_, dev_all = load_embeddings_from_parquets(DEV_QUERY_DIR, id_col=ID_COL, emb_col=EMB_COL)
if NORMALIZE:
    dev_all = l2_normalize(dev_all).astype("float32")
N_TOTAL = dev_all.shape[0]
samp = np.sort(rng.choice(N_TOTAL, size=min(N_MAIN, N_TOTAL), replace=False))  # identical to Stage 1
np.save(OUT_DIR / "sampled_indices.npy", samp)
dev_emb = np.ascontiguousarray(dev_all[samp])
results["n_total_dev"] = int(N_TOTAL); results["n_sampled"] = int(len(samp))

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pca = joblib.load(PCA_MODEL_PATH)
qmax = float(np.load(QMAX_PATH))
query_index = load_index(QUERY_INDEX_PATH)
EP  = np.load(QLR_ARTIFACT_DIR / "ep_indices.npy").astype(np.int32, copy=False)
EP_dist = np.load(QLR_ARTIFACT_DIR / "ep_distances.npy").astype(np.float32, copy=False)
exact_I = np.ascontiguousarray(np.load(EXACT_DIR / "exact_indices.npy")[samp])
EP_COLS = int(EP.shape[1])

# FIX 2 (bare-matmul PCA), validated equivalent in Stage 1
PCA_MEAN = np.ascontiguousarray(pca.mean_.astype(np.float32))
PCA_CT   = np.ascontiguousarray(pca.components_.T.astype(np.float32))
PCA_WHITEN = bool(getattr(pca, "whiten", False))
PCA_SCALE = (np.sqrt(pca.explained_variance_).astype(np.float32) if PCA_WHITEN else None)
def bare_pca(x):
    y = (x - PCA_MEAN) @ PCA_CT
    if PCA_WHITEN:
        y = y / PCA_SCALE
    return np.ascontiguousarray(y.astype(np.float32))
results["pca_whiten"] = PCA_WHITEN
dev_pca = bare_pca(dev_emb)           # batched (untimed) for the shared router-decision pass

# ------------------------------------------------------------------ load doc index ONCE (heavy)
print("[load] doc index (158GB, once)...", flush=True); t = time.time()
doc_index = load_index(DOC_INDEX_PATH)
MT = int(doc_index.metric_type)
results["doc_metric"] = "IP" if MT == faiss.METRIC_INNER_PRODUCT else "L2"
results["doc_ntotal"] = int(doc_index.ntotal)
try:
    _ = doc_index.reconstruct(0); RECON_OK = True
except Exception as e:
    RECON_OK = False; results["reconstruct_err"] = str(e)
results["reconstruct_ok"] = RECON_OK
print(f"[load] doc done {time.time()-t:.1f}s ntotal={doc_index.ntotal} metric={results['doc_metric']} recon={RECON_OK}", flush=True)
faiss.omp_set_num_threads(1)          # ALL timed FAISS single-thread
results["loadavg"]["after_doc_load"] = loadavg()

def sl0(q, Ic, Dc, npb, Dq, Iq):
    # 8-arg call == son.py == search_type 1 (UNCHANGED)
    doc_index.search_level_0(1, faiss.swig_ptr(q), TOPK, faiss.swig_ptr(Ic),
                             faiss.swig_ptr(Dc), faiss.swig_ptr(Dq), faiss.swig_ptr(Iq), npb)

def seed_dists_l2(vecs, q0):
    # squared-L2 of current query q0 (1024,) to each reconstructed seed vec (np,1024)
    diff = vecs - q0
    return np.ascontiguousarray(np.einsum('ij,ij->i', diff, diff).astype(np.float32))

def seed_dists_ip(vecs, q0):
    return np.ascontiguousarray((vecs @ q0).astype(np.float32))

# ------------------------------------------------------------------ shared router decision @ ef16 (untimed, deterministic)
print("[route] router decisions @ ef16 (shared across all configs)", flush=True)
query_index.hnsw.efSearch = ROUTER_EF
route_row = np.zeros(N_MAIN, np.int64); route_sc = np.zeros(N_MAIN, np.float32)
for i in range(N_MAIN):
    d, ii = query_index.search(dev_pca[i:i+1], 1); route_sc[i] = d[0, 0]; route_row[i] = ii[0, 0]
route_fb = route_sc < RS
routed_idx = np.where(~route_fb)[0]
fallback_idx = np.where(route_fb)[0]
routed_frac = float(len(routed_idx) / N_MAIN)
fb_frac = float(len(fallback_idx) / N_MAIN)
results["route"] = {"fallback_rate": float(route_fb.mean()), "n_routed": int(len(routed_idx)),
                    "n_fallback": int(len(fallback_idx)), "best_score_mean": float(route_sc.mean()),
                    "best_score_median": float(np.median(route_sc))}
dump()

# ------------------------------------------------------------------ seed<->exact overlap diagnostic (untimed, seed-mode independent)
print("[overlap] EP-seed vs exact top-10 overlap per nprobe", flush=True)
overlap = {}
for npb in NPROBES:
    npb_eff = min(npb, EP_COLS)
    ov = np.zeros(len(routed_idx)); hit1 = np.zeros(len(routed_idx))
    for t_i, i in enumerate(routed_idx):
        seeds = set(int(x) for x in EP[int(route_row[i]), :npb_eff])
        truth = set(int(x) for x in exact_I[i, :TOPK])
        inter = len(seeds & truth)
        ov[t_i] = inter / npb_eff
        hit1[t_i] = 1.0 if inter >= 1 else 0.0
    overlap[str(npb)] = {"mean_frac_seeds_in_exact_top10": float(ov.mean()),
                         "mean_count_seeds_in_exact_top10": float(ov.mean() * npb_eff),
                         "frac_routed_with_ge1_seed_hit": float(hit1.mean())}
results["seed_exact_overlap"] = overlap
print("[overlap]", json.dumps(overlap), flush=True)
dump()

# ------------------------------------------------------------------ warmup (untimed)
print("[warmup]", flush=True)
query_index.hnsw.efSearch = ROUTER_EF
for i in range(min(N_WARMUP, N_MAIN)):
    q = dev_emb[i:i+1]
    doc_index.hnsw.efSearch = EF_DEFAULT; doc_index.search(q, TOPK)
    query_index.search(dev_pca[i:i+1], 1)
    r = int(route_row[i]); npb = min(3, EP_COLS)
    Ic = np.ascontiguousarray(EP[r:r+1, :npb]); Dc = np.ascontiguousarray(EP_dist[r:r+1, :npb])
    Dq = np.zeros((1, TOPK), np.float32); Iq = np.zeros((1, TOPK), np.int64)
    doc_index.hnsw.efSearch = 16; sl0(q, Ic, Dc, npb, Dq, Iq)

# ------------------------------------------------------------------ PASS 1: baseline over all 500 (full search @ ef64)
# Gives (a) same-run baseline latency+acc and (b) the fallback-subset online doc-search cost.
print("[pass1] baseline full search @ ef64 (all 500)", flush=True)
def run_baseline():
    reps = []; acc = np.zeros(N_MAIN)
    for rp in range(N_REPS):
        order = rng.permutation(N_MAIN); lat = np.empty(N_MAIN, np.float64)
        doc_index.hnsw.efSearch = EF_DEFAULT
        for k, i in enumerate(order):
            i = int(i); q = dev_emb[i:i+1]
            t0 = time.perf_counter_ns(); Db, Ib = doc_index.search(q, TOPK); t1 = time.perf_counter_ns()
            lat[k] = (t1 - t0) / 1e3
            if rp == 0: acc[i] = acc10(Ib[0], exact_I[i])
        # reorder latencies back to query index for subset masking
        inv = np.empty(N_MAIN, np.float64); inv[order] = lat
        reps.append(inv)
    base_lat = agg(reps)
    base_acc = float(acc.mean())
    # fallback-subset online cost = baseline latency on the fallback queries (same full search)
    fb_lat = agg([r[fallback_idx] for r in reps]) if len(fallback_idx) else None
    fb_acc = float(acc[fallback_idx].mean()) if len(fallback_idx) else None
    routed_base_lat = agg([r[routed_idx] for r in reps]) if len(routed_idx) else None
    return base_lat, base_acc, fb_lat, fb_acc, routed_base_lat
base_lat, base_acc, fb_lat, fb_acc, routed_base_lat = run_baseline()
results["baseline"] = {"lat_us": base_lat, "acc10": base_acc,
                       "fallback_subset_lat_us": fb_lat, "fallback_subset_acc10": fb_acc,
                       "routed_subset_baseline_lat_us": routed_base_lat}
print(f"[pass1] baseline mean={base_lat['mean']:.1f}us acc={base_acc:.4f} fb_subset_mean={(fb_lat or {}).get('mean')}", flush=True)
dump()

# ------------------------------------------------------------------ PASS 2: shared router-tax (PCA + router @ ef16) over all 500
print("[pass2] shared router tax = bare_pca + router@ef16 (all 500)", flush=True)
def run_router_tax():
    pca_reps = []; rt_reps = []
    with threadpool_limits(limits=1):
        for rp in range(N_REPS):
            order = rng.permutation(N_MAIN)
            pca_l = np.empty(N_MAIN, np.float64); rt_l = np.empty(N_MAIN, np.float64)
            query_index.hnsw.efSearch = ROUTER_EF
            for k, i in enumerate(order):
                i = int(i); q = dev_emb[i:i+1]
                a0 = time.perf_counter_ns(); qp = bare_pca(q); a1 = time.perf_counter_ns()
                query_index.search(qp, 1); a2 = time.perf_counter_ns()
                pca_l[k] = (a1 - a0) / 1e3; rt_l[k] = (a2 - a1) / 1e3
            pca_reps.append(pca_l); rt_reps.append(rt_l)
    return agg(pca_reps), agg(rt_reps)
pca_tax, router_tax = run_router_tax()
TAX_MEAN = pca_tax["mean"] + router_tax["mean"]     # per-query shared cost added to every config
results["router_tax_us"] = {"pca": pca_tax, "router": router_tax, "tax_mean_us": float(TAX_MEAN)}
print(f"[pass2] pca={pca_tax['mean']:.1f} router={router_tax['mean']:.1f} tax_mean={TAX_MEAN:.1f}", flush=True)
dump()

# ------------------------------------------------------------------ PASS 3: the 24-config accuracy grid (routed subset only)
print("[pass3] 24-config grid (mode x nprobe x seeded_ef) on routed subset", flush=True)
def run_config(mode, npb_req, ef):
    npb = min(npb_req, EP_COLS)
    seed_reps = []; doc_reps = []
    acc = np.zeros(len(routed_idx))
    with threadpool_limits(limits=1):
        for rp in range(N_REPS):
            order = rng.permutation(len(routed_idx))
            s_l = np.empty(len(routed_idx), np.float64); d_l = np.empty(len(routed_idx), np.float64)
            for kk, pos in enumerate(order):
                i = int(routed_idx[pos]); q = dev_emb[i:i+1]; r = int(route_row[i])
                if mode == "cached":
                    ts0 = time.perf_counter_ns()
                    Ic = np.ascontiguousarray(EP[r:r+1, :npb])
                    Dc = np.ascontiguousarray(EP_dist[r:r+1, :npb])
                    ts1 = time.perf_counter_ns()
                else:  # recompute_l2 : current-query distances in the doc-index native metric
                    ts0 = time.perf_counter_ns()
                    vecs = np.stack([doc_index.reconstruct(int(EP[r, j])) for j in range(npb)])
                    dd = seed_dists_ip(vecs, q[0]) if MT == faiss.METRIC_INNER_PRODUCT else seed_dists_l2(vecs, q[0])
                    Dc = np.ascontiguousarray(dd.reshape(1, npb))
                    Ic = np.ascontiguousarray(EP[r:r+1, :npb])
                    ts1 = time.perf_counter_ns()
                doc_index.hnsw.efSearch = ef
                Dq = np.zeros((1, TOPK), np.float32); Iq = np.zeros((1, TOPK), np.int64)
                td0 = time.perf_counter_ns(); sl0(q, Ic, Dc, npb, Dq, Iq); td1 = time.perf_counter_ns()
                s_l[kk] = (ts1 - ts0) / 1e3; d_l[kk] = (td1 - td0) / 1e3
                if rp == 0: acc[pos] = acc10(Iq[0], exact_I[i])
            seed_reps.append(s_l); doc_reps.append(d_l)
    seed_c = agg(seed_reps); doc_c = agg(doc_reps)
    routed_acc = float(acc.mean())
    routed_online_mean = TAX_MEAN + seed_c["mean"] + doc_c["mean"]
    fb_online_mean = TAX_MEAN + (fb_lat["mean"] if fb_lat else 0.0)
    overall_lat_mean = routed_frac * routed_online_mean + fb_frac * fb_online_mean
    overall_acc = routed_frac * routed_acc + fb_frac * (fb_acc if fb_acc is not None else 0.0)
    speedup = float(base_lat["mean"] / overall_lat_mean)
    return {"mode": mode, "nprobe": npb, "seeded_ef": ef,
            "seedcost_us": seed_c, "docsearch_us": doc_c,
            "routed_acc10": routed_acc, "overall_acc10": float(overall_acc),
            "routed_online_mean_us": float(routed_online_mean),
            "fb_online_mean_us": float(fb_online_mean),
            "overall_lat_mean_us": float(overall_lat_mean),
            "speedup_vs_baseline": speedup,
            "acc_safe": bool(overall_acc >= ACC_FLOOR)}

grid = {}
for mode in SEED_MODES:
    for npb in NPROBES:
        for ef in SEEDED_EFS:
            name = f"{mode}_np{npb}_ef{ef}"
            try:
                t = time.time(); cfg = run_config(mode, npb, ef)
                grid[name] = cfg
                print(f"[grid] {name:26s} acc={cfg['overall_acc10']:.4f} routed_acc={cfg['routed_acc10']:.4f} "
                      f"lat={cfg['overall_lat_mean_us']:.1f}us speedup={cfg['speedup_vs_baseline']:.3f} "
                      f"safe={cfg['acc_safe']} ({time.time()-t:.1f}s)", flush=True)
            except Exception as e:
                grid[name] = {"error": str(e), "trace": traceback.format_exc()}
                print(f"[grid][ERROR] {name}: {e}", flush=True)
            results["grid"] = grid
            dump()

# ------------------------------------------------------------------ selection + Pareto + honest gate
ok = [(n, c) for n, c in grid.items() if isinstance(c, dict) and "error" not in c]
safe_cfgs = [(n, c) for n, c in ok if c["acc_safe"]]
if safe_cfgs:
    best_name, best_cfg = min(safe_cfgs, key=lambda kv: kv[1]["overall_lat_mean_us"])
    best_safe = {"withheld": False, "config": best_name,
                 "overall_acc10": best_cfg["overall_acc10"], "routed_acc10": best_cfg["routed_acc10"],
                 "overall_lat_mean_us": best_cfg["overall_lat_mean_us"],
                 "speedup_vs_baseline": best_cfg["speedup_vs_baseline"],
                 "acc_floor": ACC_FLOOR, "baseline_acc10": base_acc}
else:
    best_by_acc = max(ok, key=lambda kv: kv[1]["overall_acc10"]) if ok else (None, None)
    best_safe = {"withheld": True,
                 "reason": f"no config reached acc@10 >= {ACC_FLOOR} (baseline {base_acc:.4f}); "
                           f"equal-accuracy speedup NOT claimed",
                 "best_acc_config": best_by_acc[0],
                 "best_acc_overall_acc10": (best_by_acc[1]["overall_acc10"] if best_by_acc[1] else None),
                 "best_acc_speedup": (best_by_acc[1]["speedup_vs_baseline"] if best_by_acc[1] else None),
                 "followup_lever": "search_type=2 (merged-beam) is untested and out of Stage-2 scope; "
                                   "candidate next lever if the gate must be met"}
results["best_safe_config"] = best_safe
dump()

# Pareto CSV (all configs + baseline reference)
with open(OUT_DIR / "pareto.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["config", "mode", "nprobe", "seeded_ef", "overall_acc10", "routed_acc10",
                "overall_lat_mean_us", "routed_online_mean_us", "seedcost_mean_us",
                "docsearch_mean_us", "speedup_vs_baseline", "acc_safe"])
    w.writerow(["baseline", "-", "-", EF_DEFAULT, f"{base_acc:.4f}", "-",
                f"{base_lat['mean']:.1f}", "-", "-", "-", "1.000", base_acc >= ACC_FLOOR])
    for n, c in ok:
        w.writerow([n, c["mode"], c["nprobe"], c["seeded_ef"], f"{c['overall_acc10']:.4f}",
                    f"{c['routed_acc10']:.4f}", f"{c['overall_lat_mean_us']:.1f}",
                    f"{c['routed_online_mean_us']:.1f}", f"{c['seedcost_us']['mean']:.2f}",
                    f"{c['docsearch_us']['mean']:.1f}", f"{c['speedup_vs_baseline']:.3f}", c["acc_safe"]])

results["loadavg"]["at_end"] = loadavg()
dump()
faiss.omp_set_num_threads(os.cpu_count())
print("[ALL DONE] wrote", str(OUT_DIR / "stage2_accuracy.json"), "and pareto.csv", flush=True)
print("[SUMMARY] best_safe_config =", json.dumps(results["best_safe_config"]), flush=True)
# ===================== CLAUDE IMPROVEMENT END: Stage-2 accuracy-recovery grid =====================
