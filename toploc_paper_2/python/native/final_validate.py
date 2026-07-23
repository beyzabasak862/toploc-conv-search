"""
Final validation — B-Q-B-Q-B-Q on ALL 6980 dev.small queries.

Compares up to two QLR candidates (v1_winner, v2_best) against the same
native baseline at ef=50, using strict same-run interleaving.

Reports per-repetition means, pooled means, and per-query arrays; writes
overnight_final/PER_QUERY_*.npy for full transparency.
"""
# ------------------------------------------------------------------
# SUBMISSION_CODE_PACKAGE — path-only portability edits:
#   * sys.path.insert to WS/"build" replaced with NATIVE_MODULE_DIR env var.
#   * FAITH root reads from FAITH_ROOT.
#   * EXPORT_DIR reads from NATIVE_EXPORT_DIR.
#   * The --out_dir CLI default now honours OUTPUT_ROOT when the caller did
#     not pass --out_dir; the wrapper (02_cachewarmed_best/RUN.sh) still
#     passes --out_dir explicitly.
#
# Algorithm, N=6980, warmup=300, reps=3, B-Q_A-Q_B interleaving, baseline
# backend/ef, per-query arrays, pooled statistics, and output schema are
# IDENTICAL to the original at $NATIVE_WS/python/final_validate.py
# (byte-identical SHA256 in manifests/COPY_MANIFEST.tsv).
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
            f"[final_validate] Required environment variable {var!r} is not set. "
            f"Source SUBMISSION_CODE_PACKAGE/config/paths.env or export {var!r}."
        )
    return None


NATIVE_MODULE_DIR = _env_path("NATIVE_MODULE_DIR")
sys.path.insert(0, str(NATIVE_MODULE_DIR))
import native_qlr as v1
import native_qlr_v2 as v2

FAITH = _env_path("FAITH_ROOT")
EXPORT_DIR = str(_env_path("NATIVE_EXPORT_DIR"))
TOPK = 10

def acc10(ids, gt):
    return len(set(int(x) for x in ids[:TOPK] if x >= 0) & set(int(x) for x in gt[:TOPK])) / TOPK

def parse_cfg(s):
    # "kp=20,kep=10,th=0.30,ef=128,ef_min=10,rEF=16,backend=v1"
    out = dict(kp=20, kep=10, th=0.30, ef_default=128, ef_min=10, router_ef=16, backend='v2')
    for kv in s.split(','):
        k, v = kv.split('='); k = k.strip()
        if k == 'ef': k = 'ef_default'
        if k == 'rEF': k = 'router_ef'
        if k == 'backend': out[k] = v.strip(); continue
        try: out[k] = int(v)
        except ValueError: out[k] = float(v)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6980)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--baseline_ef", type=int, default=50)
    ap.add_argument("--baseline_backend", type=str, default="v2")
    ap.add_argument("--cfg_a", type=str, default="kp=20,kep=10,th=0.30,ef=128,ef_min=10,rEF=16,backend=v1")
    ap.add_argument("--cfg_b", type=str, default="")
    _default_out_dir = os.environ.get("OUTPUT_ROOT") or str(Path.cwd() / "outputs" / "02_cachewarmed_best")
    ap.add_argument("--out_dir", type=str, default=_default_out_dir)
    ap.add_argument("--core", type=int, default=21)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Try to pin ourselves (parent has already done taskset -c 21 in the launcher)
    try:
        os.sched_setaffinity(0, {args.core})
    except Exception as e:
        print(f"[pin] sched_setaffinity failed: {e} (continuing; taskset should have set it)")

    # Load both backends
    print(f"[load] v1 and v2 backends")
    ix1 = v1.NativeQLR(EXPORT_DIR)
    ix2 = v2.NativeQLR(EXPORT_DIR)

    dev = np.load(FAITH / "ground_truth/dev_small_query_embs.npy").astype(np.float32)
    gt  = np.load(FAITH / "ground_truth/dev_small_exact_top10_ids.npy").astype(np.int32)
    ep_scores = np.load(FAITH / "ep_table/ep_scores.npy")
    s_max = float(np.quantile(ep_scores[:, 0], 0.75))
    N = min(args.n, len(dev))
    dev = dev[:N]; gt = gt[:N]
    print(f"[params] N={N} s_max={s_max:.4f} baseline_ef={args.baseline_ef} baseline_backend={args.baseline_backend}")

    cfgs = [("A", parse_cfg(args.cfg_a))]
    if args.cfg_b: cfgs.append(("B", parse_cfg(args.cfg_b)))
    print(f"[cfgs] {cfgs}")

    def ix_of(backend): return ix1 if backend == "v1" else ix2
    ixB = ix_of(args.baseline_backend)

    # Warmup: touch pages for both backends and both configs
    print(f"[warmup] {args.warmup} queries")
    for i in range(args.warmup):
        j = i % N
        ixB.baseline(dev[j], args.baseline_ef, TOPK)
        for _, c in cfgs:
            ix_of(c['backend']).qlr(dev[j], c["kp"], c["kep"], c["th"], c["ef_default"],
                                     c["ef_min"], c["router_ef"], s_max, TOPK)

    K = len(cfgs)
    # Per-query arrays: baseline latency, per-config QLR latency + acc + fallback + ef_used
    B_lat = [np.zeros(N, dtype=np.float32) for _ in range(args.reps)]
    Q_lat = [[np.zeros(N, dtype=np.float32) for _ in range(args.reps)] for _ in range(K)]
    B_acc = np.zeros(N, dtype=np.float32)
    Q_acc = [np.zeros(N, dtype=np.float32) for _ in range(K)]
    Q_fb  = [np.zeros(N, dtype=bool)       for _ in range(K)]
    Q_ef  = [np.zeros(N, dtype=np.int32)   for _ in range(K)]
    Q_seed= [np.zeros(N, dtype=np.int32)   for _ in range(K)]
    Q_comp = [{k: [] for k in ["pca","router","union","beam","fallback"]} for _ in range(K)]

    print(f"[bench] {args.reps} interleaved reps  order: [B, Q_A, (Q_B)] per query")
    t0 = time.time()
    for rp in range(args.reps):
        rp_t0 = time.time()
        for i in range(N):
            rb = ixB.baseline(dev[i], args.baseline_ef, TOPK)
            B_lat[rp][i] = rb["total_us"]
            if rp == 0: B_acc[i] = acc10(rb["ids"], gt[i])
            for ki, (_, c) in enumerate(cfgs):
                rq = ix_of(c['backend']).qlr(dev[i], c["kp"], c["kep"], c["th"], c["ef_default"],
                                              c["ef_min"], c["router_ef"], s_max, TOPK)
                Q_lat[ki][rp][i] = rq["total_us"]
                if rp == 0:
                    Q_acc[ki][i] = acc10(rq["ids"], gt[i])
                    Q_fb[ki][i]  = (rq["routed"] == 0)
                    Q_ef[ki][i]  = rq["ef_used"]
                    Q_seed[ki][i]= rq["n_seeds"]
                    for k in Q_comp[ki]: Q_comp[ki][k].append(rq[k+"_us"])
        print(f"  rep {rp}: {time.time()-rp_t0:.1f}s  B={B_lat[rp].mean():.1f}us  " +
              " ".join(f"Q{i}={Q_lat[i][rp].mean():.1f}us" for i in range(K)))
    print(f"[bench] total {time.time()-t0:.1f}s")

    # Pooled stats
    def summ(reps):
        p = np.concatenate(reps)
        return dict(mean=float(p.mean()), median=float(np.median(p)),
                    p95=float(np.quantile(p,0.95)), p99=float(np.quantile(p,0.99)),
                    per_rep_mean=[float(r.mean()) for r in reps])
    Bs = summ(B_lat)
    Qs = [summ(Q_lat[k]) for k in range(K)]

    # Save arrays
    np.save(Path(args.out_dir)/"PER_QUERY_BASELINE.npy", np.stack(B_lat, axis=0))
    for k, (name, c) in enumerate(cfgs):
        np.save(Path(args.out_dir)/f"PER_QUERY_QLR_{name}.npy", np.stack(Q_lat[k], axis=0))
        np.save(Path(args.out_dir)/f"ROUTE_MASK_{name}.npy", ~Q_fb[k])
        np.save(Path(args.out_dir)/f"EF_USED_{name}.npy", Q_ef[k])
        np.save(Path(args.out_dir)/f"SEED_CT_{name}.npy", Q_seed[k])

    result = {
        "n": N, "reps": args.reps, "core": args.core, "warmup": args.warmup,
        "s_max": s_max,
        "baseline": {"backend": args.baseline_backend, "ef": args.baseline_ef,
                     **Bs, "acc10": float(B_acc.mean())},
        "qlr": [
            {"name": nm, "config": c,
             **Qs[k],
             "acc10": float(Q_acc[k].mean()),
             "fallback_rate": float(Q_fb[k].mean()),
             "ef_used_mean": float(Q_ef[k][~Q_fb[k]].mean()) if (~Q_fb[k]).any() else 0.0,
             "seed_ct_mean": float(Q_seed[k][~Q_fb[k]].mean()) if (~Q_fb[k]).any() else 0.0,
             "components_mean_us": {kk: float(np.mean(vv)) for kk,vv in Q_comp[k].items()},
             "speedup_mean_pooled": Bs["mean"] / Qs[k]["mean"],
             "speedup_median_pooled": Bs["median"] / Qs[k]["median"],
             "speedup_per_rep": [float(B_lat[rp].mean() / Q_lat[k][rp].mean()) for rp in range(args.reps)],
            }
            for k, (nm, c) in enumerate(cfgs)
        ],
    }
    with open(Path(args.out_dir)/"final_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(Path(args.out_dir)/"BASELINE_RESULT.json", "w") as f:
        json.dump(result["baseline"], f, indent=2)
    # Best QLR (max speedup at acc>=0.95)
    best = None
    for q in result["qlr"]:
        if q["acc10"] >= 0.95 and (best is None or q["speedup_mean_pooled"] > best["speedup_mean_pooled"]):
            best = q
    if best is None:
        best = max(result["qlr"], key=lambda q: q["speedup_mean_pooled"])
    with open(Path(args.out_dir)/"BEST_RESULT.json", "w") as f:
        json.dump(best, f, indent=2)
    with open(Path(args.out_dir)/"BEST_CONFIG.json", "w") as f:
        json.dump(best["config"], f, indent=2)

    print("\n=== FINAL ===")
    print(f"  Baseline ({args.baseline_backend} ef={args.baseline_ef}): mean={Bs['mean']:.1f}us med={Bs['median']:.1f}us "
          f"p95={Bs['p95']:.1f} p99={Bs['p99']:.1f} acc={B_acc.mean():.4f}")
    for k, (nm, c) in enumerate(cfgs):
        q = result["qlr"][k]
        print(f"  Q_{nm} ({c['backend']}) cfg={c}:")
        print(f"    mean={q['mean']:.1f}us med={q['median']:.1f}us p95={q['p95']:.1f} p99={q['p99']:.1f}")
        print(f"    acc={q['acc10']:.4f} fb={q['fallback_rate']:.4f} ef_used_mean={q['ef_used_mean']:.1f} seed_mean={q['seed_ct_mean']:.1f}")
        print(f"    speedup mean={q['speedup_mean_pooled']:.3f}x median={q['speedup_median_pooled']:.3f}x per_rep={q['speedup_per_rep']}")
    print(f"[DONE] wrote {args.out_dir}/final_validation.json")

if __name__ == "__main__":
    main()
