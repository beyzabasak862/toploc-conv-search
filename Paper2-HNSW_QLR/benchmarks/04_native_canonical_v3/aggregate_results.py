#!/usr/bin/env python3
"""aggregate_results.py — reads the 3 canonical_final*.json files produced by
the wrapper and emits AGGREGATE.json with the SAME metrics the project's own
`BEST_RESULT.json` uses (see overnight_final/BEST_RESULT.json:primary_headline_3run_avg).

Definitions (matched to the canonical project code):
  intra_backend_v3_QLR_speedup_avg
      = mean over runs of  ( run["v3"]["baseline_ef50"]["mean"] / run["v3"]["qlr"]["mean"] )
      i.e. same as canonical_final.py line 73 (`spd_mean = v3_b["mean"]/v3_q["mean"]`),
      then averaged across the three canonical runs.
  median_speedup_avg
      = mean over runs of  ( baseline_median / qlr_median )     (per-run spd_med)
  compound_v3_QLR_vs_v1_baseline_avg
      = mean over runs of  ( run["v1"]["baseline_ef50"]["mean"] / run["v3"]["qlr"]["mean"] )
  backend_engineering_v3_vs_v1_baseline_avg
      = mean over runs of  ( run["v1"]["baseline_ef50"]["mean"] / run["v3"]["baseline_ef50"]["mean"] )

The script does NOT alter any input file; it only reads and writes AGGREGATE.json.
"""
import argparse, json, os, statistics, sys

def load_run(path):
    with open(path) as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_paths", nargs="+",
                    help="Paths to the three canonical_final*.json files (order preserved).")
    ap.add_argument("--out", required=True, help="Output AGGREGATE.json path.")
    args = ap.parse_args()

    runs = [load_run(p) for p in args.json_paths]

    def per_run(field_chain):
        return [r[field_chain[0]][field_chain[1]] for r in runs] if len(field_chain) == 2 else \
               [r[field_chain[0]][field_chain[1]][field_chain[2]] for r in runs]

    v1_base = [r["v1"]["baseline_ef50"]["mean"]   for r in runs]
    v1_qlr  = [r["v1"]["qlr"]["mean"]              for r in runs]
    v2_base = [r["v2"]["baseline_ef50"]["mean"]   for r in runs]
    v2_qlr  = [r["v2"]["qlr"]["mean"]              for r in runs]
    v3_base = [r["v3"]["baseline_ef50"]["mean"]   for r in runs]
    v3_qlr  = [r["v3"]["qlr"]["mean"]              for r in runs]

    v3_base_med = [r["v3"]["baseline_ef50"]["median"] for r in runs]
    v3_qlr_med  = [r["v3"]["qlr"]["median"]            for r in runs]

    v3_acc = [r["v3"]["qlr"]["acc10"] for r in runs]

    intra_v3_mean_per_run = [b/q for b, q in zip(v3_base, v3_qlr)]
    intra_v3_med_per_run  = [b/q for b, q in zip(v3_base_med, v3_qlr_med)]
    compound_per_run      = [b/q for b, q in zip(v1_base, v3_qlr)]
    backend_v3_over_v1_per_run = [b/v for b, v in zip(v1_base, v3_base)]

    def mean_stdev(xs):
        return {"mean": statistics.mean(xs),
                "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
                "per_run": list(xs)}

    result = {
        "n_runs": len(runs),
        "input_json_paths": [os.path.abspath(p) for p in args.json_paths],
        "config": runs[0]["v3"]["cfg"],
        "intra_backend_v3_QLR_speedup_mean":   mean_stdev(intra_v3_mean_per_run),
        "intra_backend_v3_QLR_speedup_median": mean_stdev(intra_v3_med_per_run),
        "compound_v3_QLR_vs_v1_baseline":      mean_stdev(compound_per_run),
        "backend_engineering_v3_vs_v1_baseline": mean_stdev(backend_v3_over_v1_per_run),
        "v3_qlr_acc10_per_run": v3_acc,
        "v3_qlr_acc10_mean":     statistics.mean(v3_acc),
        "v3_qlr_mean_us_per_run":  v3_qlr,
        "v3_qlr_mean_us_avg":     statistics.mean(v3_qlr),
        "v3_baseline_mean_us_per_run": v3_base,
        "v3_baseline_mean_us_avg":     statistics.mean(v3_base),
        "protocol_note": (
            "Each input JSON is one canonical run of canonical_final.py — one Python "
            "process, per-backend isolated warmup, 3 reps × 6980 queries. Aggregation "
            "here is the same as overnight_final/BEST_RESULT.json:primary_headline_3run_avg."
        ),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[aggregate_results] wrote {args.out}")
    print(f"  intra_v3_spd_mean_avg   = {result['intra_backend_v3_QLR_speedup_mean']['mean']:.4f}"
          f"  (stdev {result['intra_backend_v3_QLR_speedup_mean']['stdev']:.4f})")
    print(f"  intra_v3_spd_median_avg = {result['intra_backend_v3_QLR_speedup_median']['mean']:.4f}"
          f"  (stdev {result['intra_backend_v3_QLR_speedup_median']['stdev']:.4f})")
    print(f"  compound_v3_vs_v1_avg   = {result['compound_v3_QLR_vs_v1_baseline']['mean']:.4f}"
          f"  (stdev {result['compound_v3_QLR_vs_v1_baseline']['stdev']:.4f})")
    print(f"  v3_acc10_avg            = {result['v3_qlr_acc10_mean']:.6f}")

if __name__ == "__main__":
    main()
