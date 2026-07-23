# ==================== CLAUDE IMPROVEMENT START ====================
# Correctness + benchmark driver for the native_qlr pybind11 module.
# Compares against baseline and existing Python/FAISS QLR reference.
# ==================== CLAUDE IMPROVEMENT END ====================
# ------------------------------------------------------------------
# SUBMISSION_CODE_PACKAGE — path-only portability edits:
#   * WS-relative sys.path.insert (str(WS / "build")) is replaced with the
#     NATIVE_MODULE_DIR environment variable so the .so can live in
#     native/prebuilt/ or a user-rebuilt directory.
#   * EXPORT_DIR (native_export) reads from NATIVE_EXPORT_DIR.
#   * FAITH root reads from FAITH_ROOT.
#   * --out already CLI-configurable; the caller RUN.sh supplies it.
#
# Algorithm, sweeps, warmup, reps, accuracy computation, config grid and
# output schema are IDENTICAL to the original at
# $NATIVE_WS/python/benchmark_native.py (byte-identical SHA256 in
# manifests/COPY_MANIFEST.tsv).
# ------------------------------------------------------------------
import os, sys, time, json, argparse
from pathlib import Path
import numpy as np


def _env_path(var, required=True, default=None):
    v = os.environ.get(var)
    if v:
        return Path(v)
    if default is not None:
        return Path(default)
    if required:
        raise RuntimeError(
            f"[benchmark_native] Required environment variable {var!r} is not set. "
            f"Source SUBMISSION_CODE_PACKAGE/config/paths.env or export {var!r}."
        )
    return None


NATIVE_MODULE_DIR = _env_path("NATIVE_MODULE_DIR")
if str(NATIVE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(NATIVE_MODULE_DIR))
import native_qlr as nq

EXPORT_DIR = str(_env_path("NATIVE_EXPORT_DIR"))
FAITH = _env_path("FAITH_ROOT")

TOPK = 10

def acc10(res_ids, gt_row):
    return len(set(int(x) for x in res_ids[:TOPK] if x >= 0) & set(int(x) for x in gt_row[:TOPK])) / TOPK

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6980, help="number of dev queries")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baseline_ef", type=int, nargs='+', default=[10, 20, 30, 40, 50, 64, 80, 100, 130])
    _default_out = os.environ.get("OUTPUT_ROOT")
    if _default_out:
        _default_out = str(Path(_default_out) / "native_500k.json")
    else:
        _default_out = str(Path.cwd() / "outputs" / "03_native_equal_accuracy" / "native_500k.json")
    ap.add_argument("--out", type=str, default=_default_out)
    args = ap.parse_args()

    print(f"[load] native module")
    t0 = time.time()
    ix = nq.NativeQLR(EXPORT_DIR)
    print(f"[load] doc_ntotal={ix.doc_ntotal()} iq_ntotal={ix.iq_ntotal()} dim={ix.dim()} entry_point={ix.doc_entry_point()} in {time.time()-t0:.2f}s")

    # Load dev queries + GT
    dev = np.load(FAITH / "ground_truth/dev_small_query_embs.npy").astype(np.float32)
    gt = np.load(FAITH / "ground_truth/dev_small_exact_top10_ids.npy").astype(np.int32)
    N = min(args.n, len(dev))
    dev = dev[:N]
    gt = gt[:N]

    # s_max from EP table
    ep_scores = np.load(FAITH / "ep_table/ep_scores.npy")
    s_max = float(np.quantile(ep_scores[:, 0], 0.75))
    print(f"[params] s_max = {s_max:.4f}")

    # Warmup
    print(f"[warmup] {args.warmup} queries")
    for i in range(args.warmup):
        ix.baseline(dev[i % N], 64, TOPK)
        ix.qlr(dev[i % N], 10, 10, 0.4, 64, 10, 16, s_max, TOPK)

    # ================ Native baseline sweep ================
    print("[phase] native baseline sweep")
    base_results = {}
    for ef in args.baseline_ef:
        lat_reps = []
        acc_arr = np.zeros(N)
        for rp in range(args.reps):
            lat = np.zeros(N)
            for i in range(N):
                r = ix.baseline(dev[i], ef, TOPK)
                lat[i] = r["total_us"]
                if rp == 0:
                    acc_arr[i] = acc10(r["ids"], gt[i])
            lat_reps.append(lat)
        pooled = np.concatenate(lat_reps)
        base_results[str(ef)] = {
            "mean_lat_us": float(pooled.mean()),
            "median_lat_us": float(np.median(pooled)),
            "p95_lat_us": float(np.quantile(pooled, 0.95)),
            "p99_lat_us": float(np.quantile(pooled, 0.99)),
            "acc10_mean": float(acc_arr.mean()),
            "per_rep_mean_us": [float(l.mean()) for l in lat_reps],
        }
        print(f"  ef={ef:>3}: mean={pooled.mean():7.1f}us med={np.median(pooled):7.1f}us acc={acc_arr.mean():.4f}")

    # ================ Native QLR configs ================
    print("[phase] native QLR sweep")
    qlr_cfgs = []
    for kp in [10, 20]:
        for kep in [10]:
            for th in [0.30, 0.40, 0.42, 0.50]:
                for ef_default in [64, 96, 128]:
                    qlr_cfgs.append(dict(kp=kp, kep=kep, th=th, ef_default=ef_default, ef_min=10, router_ef=16))
    # Focused ones with smaller kep
    for kep in [3, 5]:
        for th in [0.40, 0.50]:
            for ef_default in [64, 96]:
                qlr_cfgs.append(dict(kp=10, kep=kep, th=th, ef_default=ef_default, ef_min=10, router_ef=16))

    qlr_results = []
    for cfg in qlr_cfgs:
        lat_reps = []
        acc_arr = np.zeros(N)
        fb_arr = np.zeros(N, bool)
        ef_used_arr = np.zeros(N, int)
        seed_ct_arr = np.zeros(N, int)
        comp = {"pca": [], "router": [], "union": [], "beam": [], "fallback": []}
        for rp in range(args.reps):
            lat = np.zeros(N)
            for i in range(N):
                r = ix.qlr(dev[i], cfg["kp"], cfg["kep"], cfg["th"], cfg["ef_default"],
                           cfg["ef_min"], cfg["router_ef"], s_max, TOPK)
                lat[i] = r["total_us"]
                if rp == 0:
                    acc_arr[i] = acc10(r["ids"], gt[i])
                    fb_arr[i] = (r["routed"] == 0)
                    ef_used_arr[i] = r["ef_used"]
                    seed_ct_arr[i] = r["n_seeds"]
                    comp["pca"].append(r["pca_us"])
                    comp["router"].append(r["router_us"])
                    comp["union"].append(r["union_us"])
                    comp["beam"].append(r["beam_us"])
                    comp["fallback"].append(r["fallback_us"])
            lat_reps.append(lat)
        pooled = np.concatenate(lat_reps)
        r = {
            "config": cfg,
            "acc10_mean": float(acc_arr.mean()),
            "mean_lat_us": float(pooled.mean()),
            "median_lat_us": float(np.median(pooled)),
            "p95_lat_us": float(np.quantile(pooled, 0.95)),
            "p99_lat_us": float(np.quantile(pooled, 0.99)),
            "fallback_rate": float(fb_arr.mean()),
            "ef_used_mean": float(ef_used_arr[~fb_arr].mean()) if (~fb_arr).any() else 0.0,
            "seed_ct_mean": float(seed_ct_arr[~fb_arr].mean()) if (~fb_arr).any() else 0.0,
            "components_mean_us": {k: float(np.mean(v)) for k, v in comp.items()},
            "per_rep_mean_us": [float(l.mean()) for l in lat_reps],
        }
        qlr_results.append(r)
        print(f"  kp={cfg['kp']} kep={cfg['kep']:>2} th={cfg['th']:.2f} ef_d={cfg['ef_default']:>3}: "
              f"acc={r['acc10_mean']:.4f} mean={r['mean_lat_us']:7.1f}us fb={r['fallback_rate']:.3f} "
              f"ef={r['ef_used_mean']:.1f} c={r['seed_ct_mean']:.1f}")

    # ================ Equal-accuracy comparison ================
    equal = {}
    for target in [0.90, 0.93, 0.95, 0.952, 0.97, 0.98, 0.99]:
        best_b = None
        for ef, r in base_results.items():
            if r["acc10_mean"] >= target and (best_b is None or r["mean_lat_us"] < best_b["mean_lat_us"]):
                best_b = dict(ef=int(ef), **r)
        best_q = None
        for r in qlr_results:
            if r["acc10_mean"] >= target and (best_q is None or r["mean_lat_us"] < best_q["mean_lat_us"]):
                best_q = r
        equal[f"acc_ge_{target}"] = {
            "baseline": best_b,
            "qlr": best_q,
            "speedup": (best_b["mean_lat_us"] / best_q["mean_lat_us"]) if (best_b and best_q) else None,
        }

    # Print summary
    print("\n=== EQUAL-ACCURACY COMPARISON (native) ===")
    for k, v in equal.items():
        b, q, s = v["baseline"], v["qlr"], v["speedup"]
        if b and q:
            print(f"  {k}: base ef={b['ef']} {b['mean_lat_us']:.0f}us acc={b['acc10_mean']:.4f} "
                  f"| QLR {q['mean_lat_us']:.0f}us acc={q['acc10_mean']:.4f} | spd={s:.3f}x")
        elif b:
            print(f"  {k}: base ef={b['ef']} {b['mean_lat_us']:.0f}us / no QLR reaches")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n": N, "warmup": args.warmup, "reps": args.reps, "s_max": s_max,
                   "baseline": base_results, "qlr": qlr_results, "equal": equal}, f, indent=2)
    print(f"\n[DONE] wrote {args.out}")

if __name__ == "__main__":
    main()
