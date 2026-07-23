"""Canonical apples-to-apples: same script, same server-load moment, 6 configs.
Each measured with isolated per-backend warmup + 3x6980 reps."""
# ------------------------------------------------------------------
# SUBMISSION_CODE_PACKAGE — path-only portability edits:
#   * sys.path.insert to WS/"build" replaced with NATIVE_MODULE_DIR env var.
#   * FAITH root reads from FAITH_ROOT.
#   * EXPORT reads from NATIVE_EXPORT_DIR.
#   * Output path (was hard-coded WS/"results"/"canonical_final.json") now
#     honours OUTPUT_ROOT if set; otherwise defaults to CWD/outputs/... The
#     wrapper (04_native_canonical_v3/RUN.sh) sets OUTPUT_ROOT to a
#     timestamped sandbox.
#
# Algorithm, three-run structure, per-backend isolated warmup, 3x6980 reps,
# baseline/QLR configs (v1/v2/v3), accuracy computation, and summary schema
# are IDENTICAL to the original at $NATIVE_WS/python/canonical_final.py
# (byte-identical SHA256 in manifests/COPY_MANIFEST.tsv).
# ------------------------------------------------------------------
import os, sys, json, time
import numpy as np
from pathlib import Path


def _env_path(var, required=True, default=None):
    v = os.environ.get(var)
    if v:
        return Path(v)
    if default is not None:
        return Path(default)
    if required:
        raise RuntimeError(
            f"[canonical_final] Required environment variable {var!r} is not set. "
            f"Source SUBMISSION_CODE_PACKAGE/config/paths.env or export {var!r}."
        )
    return None


NATIVE_MODULE_DIR = _env_path("NATIVE_MODULE_DIR")
sys.path.insert(0, str(NATIVE_MODULE_DIR))
import native_qlr as v1
import native_qlr_v2 as v2
import native_qlr_v3 as v3
FAITH=_env_path("FAITH_ROOT")
EXPORT=str(_env_path("NATIVE_EXPORT_DIR"))
TOPK=10
def acc10(ids,gt): return len(set(int(x) for x in ids[:TOPK] if x>=0)&set(int(x) for x in gt[:TOPK]))/TOPK

def main():
    print("[load] v1 v2 v3 backends")
    ix1=v1.NativeQLR(EXPORT); ix2=v2.NativeQLR(EXPORT); ix3=v3.NativeQLR(EXPORT)
    dev=np.load(FAITH/"ground_truth/dev_small_query_embs.npy").astype(np.float32)
    gt =np.load(FAITH/"ground_truth/dev_small_exact_top10_ids.npy").astype(np.int32)
    ep_scores=np.load(FAITH/"ep_table/ep_scores.npy")
    s_max=float(np.quantile(ep_scores[:,0],0.75))
    N=len(dev)
    print(f"[cfg] N={N} s_max={s_max:.4f}")

    def measure_baseline(ix, ef, label):
        for i in range(200): ix.baseline(dev[i%N], ef, TOPK)
        reps=[]; acc=np.zeros(N)
        for rp in range(3):
            L=np.zeros(N)
            for i in range(N):
                r=ix.baseline(dev[i], ef, TOPK); L[i]=r["total_us"]
                if rp==0: acc[i]=acc10(r["ids"], gt[i])
            reps.append(L)
        p=np.concatenate(reps)
        print(f"  [{label}] mean={p.mean():.1f}us med={np.median(p):.1f}us p95={np.quantile(p,0.95):.1f} acc={acc.mean():.4f}  per_rep={[float(l.mean()) for l in reps]}")
        return dict(mean=float(p.mean()), median=float(np.median(p)), p95=float(np.quantile(p,0.95)),
                    p99=float(np.quantile(p,0.99)), acc10=float(acc.mean()),
                    per_rep_mean=[float(l.mean()) for l in reps])

    def measure_qlr(ix, kp, kep, th, ef_d, ef_min, rEF, label):
        for i in range(150): ix.qlr(dev[i%N], kp, kep, th, ef_d, ef_min, rEF, s_max, TOPK)
        reps=[]; acc=np.zeros(N); fb=np.zeros(N,bool); ef_used=np.zeros(N,int); seed_ct=np.zeros(N,int)
        for rp in range(3):
            L=np.zeros(N)
            for i in range(N):
                r=ix.qlr(dev[i], kp, kep, th, ef_d, ef_min, rEF, s_max, TOPK); L[i]=r["total_us"]
                if rp==0: acc[i]=acc10(r["ids"], gt[i]); fb[i]=(r["routed"]==0); ef_used[i]=r["ef_used"]; seed_ct[i]=r["n_seeds"]
            reps.append(L)
        p=np.concatenate(reps)
        print(f"  [{label}] mean={p.mean():.1f}us med={np.median(p):.1f}us p95={np.quantile(p,0.95):.1f} acc={acc.mean():.4f} fb={fb.mean():.3f}  per_rep={[float(l.mean()) for l in reps]}")
        return dict(mean=float(p.mean()), median=float(np.median(p)), p95=float(np.quantile(p,0.95)),
                    p99=float(np.quantile(p,0.99)), acc10=float(acc.mean()),
                    fallback_rate=float(fb.mean()),
                    ef_used_mean=float(ef_used[~fb].mean()) if (~fb).any() else 0.0,
                    seed_ct_mean=float(seed_ct[~fb].mean()) if (~fb).any() else 0.0,
                    per_rep_mean=[float(l.mean()) for l in reps])

    print("[measure] v1 baseline ef=50")
    v1_b = measure_baseline(ix1, 50, "V1_BASE_ef50")
    print("[measure] v1 QLR (recorded winner: kp=20 kep=10 th=0.30 ef=128 rEF=16)")
    v1_q = measure_qlr(ix1, 20, 10, 0.30, 128, 10, 16, "V1_QLR")
    print("[measure] v2 baseline ef=50")
    v2_b = measure_baseline(ix2, 50, "V2_BASE_ef50")
    print("[measure] v2 QLR (best v2: kp=20 kep=10 th=0.35 ef=112 rEF=16)")
    v2_q = measure_qlr(ix2, 20, 10, 0.35, 112, 10, 16, "V2_QLR")
    print("[measure] v3 baseline ef=50")
    v3_b = measure_baseline(ix3, 50, "V3_BASE_ef50")
    print("[measure] v3 QLR (best v3: kp=20 kep=10 th=0.32 ef=112 rEF=12)")
    v3_q = measure_qlr(ix3, 20, 10, 0.32, 112, 10, 12, "V3_QLR")

    out = {
        "v1": {"baseline_ef50": v1_b, "qlr": v1_q, "spd_mean": v1_b["mean"]/v1_q["mean"], "spd_med": v1_b["median"]/v1_q["median"], "cfg":"kp=20 kep=10 th=0.30 ef=128 rEF=16"},
        "v2": {"baseline_ef50": v2_b, "qlr": v2_q, "spd_mean": v2_b["mean"]/v2_q["mean"], "spd_med": v2_b["median"]/v2_q["median"], "cfg":"kp=20 kep=10 th=0.35 ef=112 rEF=16"},
        "v3": {"baseline_ef50": v3_b, "qlr": v3_q, "spd_mean": v3_b["mean"]/v3_q["mean"], "spd_med": v3_b["median"]/v3_q["median"], "cfg":"kp=20 kep=10 th=0.32 ef=112 rEF=12"},
        "backend_speedup_baseline": {"v2_vs_v1": v1_b["mean"]/v2_b["mean"], "v3_vs_v1": v1_b["mean"]/v3_b["mean"]},
        "compound_qlr_vs_v1_baseline": {"v2_qlr": v1_b["mean"]/v2_q["mean"], "v3_qlr": v1_b["mean"]/v3_q["mean"]},
    }
    print("\n=== CANONICAL SUMMARY ===")
    for k in ["v1","v2","v3"]:
        b=out[k]["baseline_ef50"]; q=out[k]["qlr"]
        print(f"  {k}: B={b['mean']:.1f}us(acc={b['acc10']:.4f})  Q={q['mean']:.1f}us(acc={q['acc10']:.4f})  spd={out[k]['spd_mean']:.3f}x")
    print(f"\nbackend spd (baseline): v2/v1={out['backend_speedup_baseline']['v2_vs_v1']:.3f}x  v3/v1={out['backend_speedup_baseline']['v3_vs_v1']:.3f}x")
    print(f"compound spd vs v1 baseline: v2_QLR={out['compound_qlr_vs_v1_baseline']['v2_qlr']:.3f}x  v3_QLR={out['compound_qlr_vs_v1_baseline']['v3_qlr']:.3f}x")

    _output_root = os.environ.get("OUTPUT_ROOT")
    if _output_root:
        _out_path = Path(_output_root) / "canonical_final.json"
    else:
        _out_path = Path.cwd() / "outputs" / "04_native_canonical_v3" / "canonical_final.json"
    _out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(_out_path, "w"), indent=2)
    print(f"[DONE] wrote {_out_path}")

if __name__=="__main__": main()
