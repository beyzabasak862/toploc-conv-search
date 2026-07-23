# ==================== CLAUDE IMPROVEMENT START: full 6,980-query rescue run =====================
# ------------------------------------------------------------------
# SUBMISSION_CODE_PACKAGE — path-only portability edits:
#   * Absolute hard-coded paths (PROJECT_ROOT, DEV_QUERY_DIR, DOC_INDEX_PATH,
#     PCA_QL_DIR, QUERY_INDEX_PATH, PCA_MODEL_PATH, QLR_ARTIFACT_DIR, EXACT_DIR)
#     have been replaced with reads from environment variables (config/paths.env).
#   * PROJECT_ROOT is no longer prepended to sys.path; instead this package's
#     python/hybrid directory is prepended so that `from src.data_loading` and
#     `from src.indexing` resolve to the bundled helper copies.
#   * OUT_DIR is chosen from OUTPUT_ROOT (if set by RUN.sh) instead of the
#     original claude_qlr_diagnostics/results/full_<ts>/ inside PROJECT_ROOT.
#
# Algorithm, benchmark configuration, seeds, warmup, reps, ef sweep, accuracy
# computation, routing/fallback logic, and output schema are IDENTICAL to the
# original at claude_qlr_diagnostics/rescue_full_run.py
# (byte-identical SHA256 recorded in manifests/COPY_MANIFEST.tsv).
# ------------------------------------------------------------------
import os
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import sys
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
            f"[rescue_full_run] Required environment variable {var!r} is not set. "
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

# --- External data paths from config/paths.env -------------------------------------
DEV_QUERY_DIR    = _env_path("DEV_QUERY_DIR")
DOC_INDEX_PATH   = _env_path("HYBRID_DOC_INDEX")
PCA_QL_DIR       = _env_path("PCA_QL_DIR")
QUERY_INDEX_PATH = _env_path("ROUTER_INDEX",   required=False) or (PCA_QL_DIR / "train_query_pca256_hnsw.faiss")
PCA_MODEL_PATH   = _env_path("PCA_MODEL",      required=False) or (PCA_QL_DIR / "pca_1024_to_256.joblib")
QLR_ARTIFACT_DIR = _env_path("QLR_ARTIFACT_DIR")
EXACT_DIR        = _env_path("EXACT_DIR")

ID_COL, EMB_COL = "id", "embedding"
NORMALIZE = True
TOPK = 10
RS = 0.50
EF_DEFAULT = 64          # baseline + fallback
ROUTER_EF = 16           # FIX 1
NPROBE = 3               # Stage-2 efficient-frontier point
SEEDED_EFS = [32, 16]    # 32 = acc-safe headline ; 16 = aggressive endpoint
N_REPS, N_WARMUP = 2, 50 # full 6,980 sample: 2 reps gives ample stability, kinder to the shared box
SEED = 20260717
ACC_FLOOR = 0.952
rng = np.random.default_rng(SEED)

RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
_OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT")
if _OUTPUT_ROOT:
    OUT_DIR = Path(_OUTPUT_ROOT) / f"full_{RUN_ID}"
else:
    OUT_DIR = Path.cwd() / "outputs" / "01_safe_hybrid" / f"full_{RUN_ID}"
os.makedirs(OUT_DIR, exist_ok=False)

results = {"run_id": RUN_ID, "stage": "full_6980", "seed": SEED,
           "config": {"RS": RS, "EF_DEFAULT": EF_DEFAULT, "ROUTER_EF": ROUTER_EF, "NPROBE": NPROBE,
                      "SEEDED_EFS": SEEDED_EFS, "TOPK": TOPK, "N_REPS": N_REPS, "N_WARMUP": N_WARMUP,
                      "seed_mode": "recompute_l2", "ACC_FLOOR": ACC_FLOOR},
           "loadavg": {}}

def loadavg():
    try:
        return list(os.getloadavg())
    except Exception:
        return None
results["loadavg"]["at_start"] = loadavg()

def dump():
    with open(OUT_DIR / "full_run.json", "w") as f:
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

# ---------------------------------------------------- load small artifacts (full dev set)
print("[load] dev queries (ALL 6980) + PCA + router + EP + exact GT", flush=True)
_, dev_all = load_embeddings_from_parquets(DEV_QUERY_DIR, id_col=ID_COL, emb_col=EMB_COL)
if NORMALIZE:
    dev_all = l2_normalize(dev_all).astype("float32")
dev_emb = np.ascontiguousarray(dev_all)
N = dev_emb.shape[0]
results["n_queries"] = int(N)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pca = joblib.load(PCA_MODEL_PATH)
query_index = load_index(QUERY_INDEX_PATH)
EP  = np.load(QLR_ARTIFACT_DIR / "ep_indices.npy").astype(np.int32, copy=False)
EP_dist = np.load(QLR_ARTIFACT_DIR / "ep_distances.npy").astype(np.float32, copy=False)  # loaded for parity; unused (recompute mode)
exact_I = np.ascontiguousarray(np.load(EXACT_DIR / "exact_indices.npy"))
assert exact_I.shape[0] == N, f"exact GT rows {exact_I.shape[0]} != dev {N}"
EP_COLS = int(EP.shape[1]); NPB = min(NPROBE, EP_COLS)

# FIX 2: bare-matmul PCA (validated equivalent in Stage 1)
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
dev_pca_batched = bare_pca(dev_emb)   # untimed, for the shared route-decision pass only

# ---------------------------------------------------- load doc index ONCE
print("[load] doc index (158GB, once)...", flush=True); t = time.time()
doc_index = load_index(DOC_INDEX_PATH)
MT = int(doc_index.metric_type)
results["doc_metric"] = "IP" if MT == faiss.METRIC_INNER_PRODUCT else "L2"
results["doc_ntotal"] = int(doc_index.ntotal)
print(f"[load] doc done {time.time()-t:.1f}s ntotal={doc_index.ntotal} metric={results['doc_metric']}", flush=True)
faiss.omp_set_num_threads(1)
results["loadavg"]["after_doc_load"] = loadavg()

def sl0(q, Ic, Dc, npb, Dq, Iq):
    doc_index.search_level_0(1, faiss.swig_ptr(q), TOPK, faiss.swig_ptr(Ic),
                             faiss.swig_ptr(Dc), faiss.swig_ptr(Dq), faiss.swig_ptr(Iq), npb)

def seed_dists(vecs, q0):
    if MT == faiss.METRIC_INNER_PRODUCT:
        return np.ascontiguousarray((vecs @ q0).astype(np.float32))
    diff = vecs - q0
    return np.ascontiguousarray(np.einsum('ij,ij->i', diff, diff).astype(np.float32))

# ---------------------------------------------------- shared route decision (untimed, ef16)
print("[route] router decisions @ ef16 (all 6980)", flush=True)
query_index.hnsw.efSearch = ROUTER_EF
route_row = np.zeros(N, np.int64); route_fb = np.zeros(N, bool)
for i in range(N):
    d, ii = query_index.search(dev_pca_batched[i:i+1], 1)
    route_row[i] = ii[0, 0]; route_fb[i] = d[0, 0] < RS
results["route"] = {"fallback_rate": float(route_fb.mean()), "n_routed": int((~route_fb).sum()),
                    "n_fallback": int(route_fb.sum())}
dump()

# ---------------------------------------------------- warmup
print("[warmup]", flush=True)
query_index.hnsw.efSearch = ROUTER_EF
for i in range(min(N_WARMUP, N)):
    q = dev_emb[i:i+1]
    doc_index.hnsw.efSearch = EF_DEFAULT; doc_index.search(q, TOPK)
    query_index.search(dev_pca_batched[i:i+1], 1)
    r = int(route_row[i])
    vecs = np.stack([doc_index.reconstruct(int(EP[r, j])) for j in range(NPB)])
    Ic = np.ascontiguousarray(EP[r:r+1, :NPB]); Dc = np.ascontiguousarray(seed_dists(vecs, q[0]).reshape(1, NPB))
    Dq = np.zeros((1, TOPK), np.float32); Iq = np.zeros((1, TOPK), np.int64)
    doc_index.hnsw.efSearch = 16; sl0(q, Ic, Dc, NPB, Dq, Iq)

# ---------------------------------------------------- full same-run pipeline
# Per query: baseline(ef64) + shared[bare_pca, router@ef16] + (fallback ef64 | recompute seeds -> sl0 @ each ef).
print(f"[run] full pipeline, {N_REPS} reps x {N} queries", flush=True)
base_reps = []
qlr_reps = {ef: [] for ef in SEEDED_EFS}
base_acc = np.zeros(N); acc_ef = {ef: np.zeros(N) for ef in SEEDED_EFS}; fbflag = np.zeros(N, bool)
try:
    with threadpool_limits(limits=1):
        for rp in range(N_REPS):
            order = rng.permutation(N)
            b = np.zeros(N); tot = {ef: np.zeros(N) for ef in SEEDED_EFS}
            for i in order:
                i = int(i); q = dev_emb[i:i+1]
                # --- baseline (unchanged) ---
                doc_index.hnsw.efSearch = EF_DEFAULT
                t0 = time.perf_counter_ns(); Db, Ib = doc_index.search(q, TOPK); t1 = time.perf_counter_ns()
                b[i] = (t1 - t0) / 1e3
                if rp == 0: base_acc[i] = acc10(Ib[0], exact_I[i])
                # --- shared router tax (FIX 2 + FIX 1, FIX 3 counts it) ---
                p0 = time.perf_counter_ns(); qp = bare_pca(q); p1 = time.perf_counter_ns()
                query_index.hnsw.efSearch = ROUTER_EF
                r0 = time.perf_counter_ns(); d, ii = query_index.search(qp, 1); r1 = time.perf_counter_ns()
                s = float(d[0, 0]); r = int(ii[0, 0]); f = s < RS
                tax = (p1 - p0 + r1 - r0) / 1e3
                if f:
                    doc_index.hnsw.efSearch = EF_DEFAULT
                    d0 = time.perf_counter_ns(); Dq, Iq = doc_index.search(q, TOPK); d1 = time.perf_counter_ns()
                    fb_cost = (d1 - d0) / 1e3
                    for ef in SEEDED_EFS:
                        tot[ef][i] = tax + fb_cost
                        if rp == 0: acc_ef[ef][i] = acc10(Iq[0], exact_I[i])
                    if rp == 0: fbflag[i] = True
                else:
                    # FIX 4: recompute current-query seed distances (shared across both ef points)
                    s0 = time.perf_counter_ns()
                    vecs = np.stack([doc_index.reconstruct(int(EP[r, j])) for j in range(NPB)])
                    Dc = np.ascontiguousarray(seed_dists(vecs, q[0]).reshape(1, NPB))
                    Ic = np.ascontiguousarray(EP[r:r+1, :NPB])
                    s1 = time.perf_counter_ns(); seedcost = (s1 - s0) / 1e3
                    for ef in (SEEDED_EFS if rng.integers(2) == 0 else SEEDED_EFS[::-1]):  # randomize ef order
                        doc_index.hnsw.efSearch = ef
                        Dq = np.zeros((1, TOPK), np.float32); Iq = np.zeros((1, TOPK), np.int64)
                        e0 = time.perf_counter_ns(); sl0(q, Ic, Dc, NPB, Dq, Iq); e1 = time.perf_counter_ns()
                        tot[ef][i] = tax + seedcost + (e1 - e0) / 1e3
                        if rp == 0: acc_ef[ef][i] = acc10(Iq[0], exact_I[i])
            base_reps.append(b)
            for ef in SEEDED_EFS: qlr_reps[ef].append(tot[ef])
            print(f"[run] rep {rp+1}/{N_REPS} done", flush=True)

    base_lat = agg(base_reps)
    out = {"baseline": {"lat_us": base_lat, "acc10": float(base_acc.mean())},
           "fallback_rate": float(fbflag.mean()), "variants": {}}
    for ef in SEEDED_EFS:
        qlat = agg(qlr_reps[ef])
        per_q_mean = np.mean(np.stack(qlr_reps[ef]), axis=0)   # per-query mean over reps for the routed/fb split
        rmask = ~fbflag
        out["variants"][f"recompute_l2_np{NPB}_ef{ef}"] = {
            "lat_us": qlat, "acc10": float(acc_ef[ef].mean()),
            "routed_acc10": float(acc_ef[ef][rmask].mean()) if rmask.any() else None,
            "fallback_acc10": float(acc_ef[ef][fbflag].mean()) if fbflag.any() else None,
            "routed_total_mean_us": float(per_q_mean[rmask].mean()) if rmask.any() else None,
            "fallback_total_mean_us": float(per_q_mean[fbflag].mean()) if fbflag.any() else None,
            "speedup_mean": float(base_lat["mean"] / qlat["mean"]),
            "speedup_median": float(base_lat["median"] / qlat["median"]),
            "acc_delta_vs_baseline": float(acc_ef[ef].mean() - base_acc.mean()),
            "acc_safe": bool(acc_ef[ef].mean() >= ACC_FLOOR),
        }
    results["results"] = out
    # save per-query arrays (reproducible, nothing hardcoded)
    np.save(OUT_DIR / "baseline_latency_us_rep0.npy", base_reps[0])
    np.save(OUT_DIR / "used_fallback.npy", fbflag)
    np.save(OUT_DIR / "baseline_acc10.npy", base_acc)
    for ef in SEEDED_EFS:
        np.save(OUT_DIR / f"qlr_ef{ef}_latency_us_rep0.npy", qlr_reps[ef][0])
        np.save(OUT_DIR / f"qlr_ef{ef}_acc10.npy", acc_ef[ef])
    hl = out["variants"][f"recompute_l2_np{NPB}_ef32"]
    print(f"[run] baseline mean={base_lat['mean']:.1f} median={base_lat['median']:.1f} acc={base_acc.mean():.4f}", flush=True)
    print(f"[run] ef32 mean={hl['lat_us']['mean']:.1f} median={hl['lat_us']['median']:.1f} acc={hl['acc10']:.4f} "
          f"speedup_mean={hl['speedup_mean']:.3f} speedup_median={hl['speedup_median']:.3f} safe={hl['acc_safe']}", flush=True)
except Exception as e:
    print("[run][ERROR]", e, flush=True); traceback.print_exc()
    results["results"] = {"error": str(e), "trace": traceback.format_exc()}

results["loadavg"]["at_end"] = loadavg()
dump()
faiss.omp_set_num_threads(os.cpu_count())
print("[ALL DONE] wrote", str(OUT_DIR / "full_run.json"), flush=True)
# ===================== CLAUDE IMPROVEMENT END: full 6,980-query rescue run =====================
