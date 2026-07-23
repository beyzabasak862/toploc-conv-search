# BEHAVIOUR_PRESERVATION_REPORT.md

Confirms that the path-only edits in `manifests/PATH_ONLY_DIFFS.patch` and
`manifests/PATH_REWRITE_MAP.tsv` do not change any measured behaviour. Every
row below is a targeted comparison of one behavioural aspect between the
canonical producer as it lives in the original repository and the packaged
copy under `python/`.

For every producer the byte-identical **input file SHA256** is recorded in
`manifests/COPY_MANIFEST.tsv` (rows tagged `python`, `path_adjusted=yes`).

## rescue_full_run.py — Results 1, 5

| Behaviour item | Original value | Packaged value | Status |
|---|---|---|---|
| RS (route-score threshold) | 0.50 | 0.50 | IDENTICAL |
| EF_DEFAULT (baseline + fallback) | 64 | 64 | IDENTICAL |
| ROUTER_EF | 16 | 16 | IDENTICAL |
| NPROBE | 3 | 3 | IDENTICAL |
| SEEDED_EFS | [32, 16] | [32, 16] | IDENTICAL |
| N_REPS | 2 | 2 | IDENTICAL |
| N_WARMUP | 50 | 50 | IDENTICAL |
| SEED | 20260717 | 20260717 | IDENTICAL |
| ACC_FLOOR | 0.952 | 0.952 | IDENTICAL |
| TOPK | 10 | 10 | IDENTICAL |
| NORMALIZE dev queries | True | True | IDENTICAL |
| seed_mode | recompute_l2 | recompute_l2 | IDENTICAL |
| Timing boundaries | perf_counter_ns pre/post FAISS calls | unchanged | IDENTICAL |
| Accuracy computation | overlap of top-10 sets / 10 | unchanged | IDENTICAL |
| Routing decision | d[0,0] < RS -> fallback | unchanged | IDENTICAL |
| Fallback logic | full doc-index search @ EF_DEFAULT | unchanged | IDENTICAL |
| ef randomization per query | rng.integers(2) branch | unchanged | IDENTICAL |
| Output schema (full_run.json) | baseline + variants + fallback_rate | unchanged | IDENTICAL |
| Saved per-query arrays | baseline_latency/acc, qlr_ef{ef}, used_fallback | unchanged | IDENTICAL |
| threadpool_limits(limits=1) | wraps N_REPS timed loop | unchanged | IDENTICAL |
| faiss.omp_set_num_threads(1) | after doc-index load | unchanged | IDENTICAL |

## rescue_stage2_accuracy.py — Result 6

| Behaviour item | Original value | Packaged value | Status |
|---|---|---|---|
| N_MAIN | 500 | 500 | IDENTICAL |
| SEED | 20260717 (same as Stage 1) | 20260717 | IDENTICAL |
| N_REPS | 3 | 3 | IDENTICAL |
| N_WARMUP | 50 | 50 | IDENTICAL |
| RS | 0.50 | 0.50 | IDENTICAL |
| EF_DEFAULT | 64 | 64 | IDENTICAL |
| ROUTER_EF | 16 | 16 | IDENTICAL |
| SEED_MODES | ["cached", "recompute_l2"] | unchanged | IDENTICAL |
| NPROBES | [3, 5, 10] | unchanged | IDENTICAL |
| SEEDED_EFS | [16, 32, 64, 128] | unchanged | IDENTICAL |
| ACC_FLOOR / ACC_TOL | 0.952 / 0.01 | unchanged | IDENTICAL |
| Mixture-mean speedup formula | routed_frac * routed_online + fb_frac * fb_online | unchanged | IDENTICAL |
| Seed<->exact overlap diagnostic | per NPROBE bucket | unchanged | IDENTICAL |
| Baseline / router-tax / grid pass order | PASS 1 / PASS 2 / PASS 3 | unchanged | IDENTICAL |
| Pareto CSV columns | 12 columns as documented | unchanged | IDENTICAL |
| best_safe_config selection | min overall_lat_mean subject to acc_safe | unchanged | IDENTICAL |
| dump() (stage2_accuracy.json) schema | run_id + config + route + grid + best_safe_config | unchanged | IDENTICAL |

## benchmark_native.py — Result 3

| Behaviour item | Original value | Packaged value | Status |
|---|---|---|---|
| --n default | 6980 | 6980 | IDENTICAL |
| --warmup default | 100 | 100 | IDENTICAL |
| --reps default | 3 | 3 | IDENTICAL |
| --baseline_ef default sweep | [10,20,30,40,50,64,80,100,130] | unchanged | IDENTICAL |
| s_max | np.quantile(ep_scores[:,0], 0.75) | unchanged | IDENTICAL |
| QLR grid | 5 (kp,kep) x (th) x (ef_default) matrix as coded | unchanged | IDENTICAL |
| baseline / QLR call ordering | isolated sweeps as in source | unchanged | IDENTICAL |
| Equal-accuracy targets | [0.90, 0.93, 0.95, 0.952, 0.97, 0.98, 0.99] | unchanged | IDENTICAL |
| Output schema | n, warmup, reps, s_max, baseline, qlr, equal | unchanged | IDENTICAL |
| ONLY delta | default --out honours OUTPUT_ROOT + parent mkdir | new default only when caller omits --out | path only |

## canonical_final.py — Result 4

| Behaviour item | Original value | Packaged value | Status |
|---|---|---|---|
| Per-backend warmup counts | 200 baseline / 150 QLR | unchanged | IDENTICAL |
| Reps per config | 3 | 3 | IDENTICAL |
| N queries | len(dev) = 6980 | unchanged | IDENTICAL |
| v1/v2/v3 configs | (20,10,0.30,128,10,16) / (20,10,0.35,112,10,16) / (20,10,0.32,112,10,12) | unchanged | IDENTICAL |
| Baseline ef | 50 | 50 | IDENTICAL |
| Metric computation (mean/median/p95/p99/acc10) | unchanged | unchanged | IDENTICAL |
| Cross-backend derived stats | backend_speedup_baseline, compound_qlr_vs_v1_baseline | unchanged | IDENTICAL |
| Output filename | canonical_final.json | canonical_final.json | IDENTICAL |
| ONLY delta | output dir uses OUTPUT_ROOT env (was WS/"results") | dir choice only | path only |

## final_validate.py — Result 2

| Behaviour item | Original value | Packaged value | Status |
|---|---|---|---|
| --n / --warmup / --reps defaults | 6980 / 300 / 3 | 6980 / 300 / 3 | IDENTICAL |
| --baseline_ef default | 50 | 50 | IDENTICAL |
| --baseline_backend default | v2 | v2 | IDENTICAL |
| --cfg_a default | kp=20,kep=10,th=0.30,ef=128,ef_min=10,rEF=16,backend=v1 | unchanged | IDENTICAL |
| --cfg_b default | "" | "" | IDENTICAL |
| Interleaving order | [Baseline, Q_A, (Q_B)] per query | unchanged | IDENTICAL |
| sched_setaffinity | {args.core} | unchanged | IDENTICAL |
| s_max | np.quantile(ep_scores[:,0], 0.75) | unchanged | IDENTICAL |
| Per-query arrays saved | PER_QUERY_BASELINE, PER_QUERY_QLR_{name}, ROUTE_MASK_{name}, EF_USED_{name}, SEED_CT_{name} | unchanged | IDENTICAL |
| Aggregate output | final_validation.json / BASELINE_RESULT.json / BEST_RESULT.json / BEST_CONFIG.json | unchanged | IDENTICAL |
| Best selection | max speedup_mean_pooled at acc10 >= 0.95 | unchanged | IDENTICAL |
| ONLY delta | default --out_dir picks OUTPUT_ROOT | dir choice only | path only |

## aggregate_results.py — Result 4 (post-processor)

Byte-identical copy of the source. No path edits, no code edits.
`sha256 = c1a70... — see COPY_MANIFEST.tsv row 04_native_canonical_v3/aggregate_results.py`.

## faithful_qlr.py — Experiment 7 (algorithm module)

Byte-identical copy of
`paper2_final_track/optimization_search/paper2_faithful_20260718_231400/faithful_qlr.py`.
No path edits, no code edits.
`sha256 (before) = sha256 (after) = 042958a516bcf0fb5c0a73a1ec0d17627fd20e9fc2c36d36c9d4f3769867d6aa`
— see `COPY_MANIFEST.tsv` row `07_faithful_adaptive_depth/faithful_qlr.py`.

## runner.py — Experiment 7 (path-only)

| Behaviour item | Original value | Packaged value | Status |
|---|---|---|---|
| SEED | 20260718 | 20260718 | IDENTICAL |
| TOPK | 10 | 10 | IDENTICAL |
| N_CALIB / N_HOLDOUT / N_WARMUP | 500 / 1500 / 30 | 500 / 1500 / 30 | IDENTICAL |
| N_REPS_CALIB / HOLDOUT / FULL | 2 / 3 / 3 | 2 / 3 / 3 | IDENTICAL |
| ACC_FLOOR / ACC_TOL_VS_BASE / SPEEDUP_TARGET | 0.952 / 0.005 / 1.40 | unchanged | IDENTICAL |
| CALIB_MAX_CFG / HOLDOUT_MAX_CFG / FULL_MAX_CFG | 12 / 3 / 2 | unchanged | IDENTICAL |
| BASELINE_EF_SWEEP | [16,24,32,40,48,64,96,128,160,200] | unchanged | IDENTICAL |
| Deterministic split from rng.permutation(6980) | unchanged | unchanged | IDENTICAL |
| s_max = compute_s_max(ep_distances[:,0], quantile=0.25) | unchanged | unchanged | IDENTICAL |
| 12 calibration configs (A..L) — every kp/kep/th/ef_min/ef_default/search_type | unchanged | unchanged | IDENTICAL |
| Adaptive ef formula (FaithfulQLR.adaptive_ef) | unchanged | unchanged | IDENTICAL |
| Fallback: `s < th` → full HNSW at cfg.ef_default | unchanged | unchanged | IDENTICAL |
| union_ep (int32 dedup preserving first-occurrence order) | unchanged | unchanged | IDENTICAL |
| compute_seed_dists (reconstruct + squared-L2 + argsort stable) | unchanged | unchanged | IDENTICAL |
| seeded_beam via search_level_0 (9-arg with search_type) | unchanged | unchanged | IDENTICAL |
| Timing boundaries (`time.perf_counter_ns()` per component) | unchanged | unchanged | IDENTICAL |
| Accuracy computation (`_acc10` = intersection / TOPK) | unchanged | unchanged | IDENTICAL |
| Warmup exercises full path (baseline + QLR st=2 + QLR st=1) | unchanged | unchanged | IDENTICAL |
| Phase order (0 sanity → 1 sweep calib → 2 calib → 3 holdout → 4 full → 5 sweep full) | unchanged | unchanged | IDENTICAL |
| Interleaving order per query (B then Q, randomized order per rep) | unchanged | unchanged | IDENTICAL |
| `faiss.omp_set_num_threads(1)` after doc-index load | unchanged | unchanged | IDENTICAL |
| `threadpool_limits(limits=1)` wraps every timed loop | unchanged | unchanged | IDENTICAL |
| IO_FLAG_MMAP tried for `/dev/shm` doc-index paths | unchanged | unchanged | IDENTICAL |
| Output schema (`config_manifest.json`, `baseline_sanity.json`, `baseline_sweep.json`, `calibration.json`, `holdout.json`, `full.json`, `baseline_sweep_full.json`, `SUMMARY.json`, `latency_arrays/`) | unchanged | unchanged | IDENTICAL |
| ONLY delta | out_dir picks `${OUTPUT_ROOT}/faithful_<run_id>/` when OUTPUT_ROOT is set; falls back to legacy `SCRIPT_DIR/results/<run_id>/` otherwise | dir choice only | path only |

## cachewarmed_treccast.py — Benchmark 08 (hybrid full TREC-CAsT cache-warmed)

Benchmark 08 is a **new hybrid producer** that combines two existing,
unmodified package-local code paths — it does not alter either. Its per-query
QLR search functions are reproduced **byte-for-byte** from the Benchmark 07
producer, and its execution protocol is reproduced from Benchmark 02.

**Algorithm functions — verbatim from `python/faithful/runner.py`:**

| Function | Source | Status |
|---|---|---|
| `compute_s_max` | `python/faithful/runner.py::compute_s_max` | IDENTICAL (byte-for-byte) |
| `timed_baseline` | `python/faithful/runner.py::timed_baseline` | IDENTICAL (byte-for-byte) |
| `timed_qlr` | `python/faithful/runner.py::timed_qlr` | IDENTICAL (byte-for-byte) |
| QLR search (`pca_transform`, `union_ep`, `adaptive_ef`, `compute_seed_dists`, `seeded_beam`, `fallback_hnsw`) | `python/faithful/faithful_qlr.py::FaithfulQLR` (imported, unmodified) | IDENTICAL |

**Protocol — from `python/native/final_validate.py` (Benchmark 02):**

| Behaviour item | Benchmark 02 value | Benchmark 08 value | Status |
|---|---|---|---|
| One process, index loaded once | yes | yes | IDENTICAL |
| Per-query call order | [Baseline, Q_A, Q_B] | [Baseline, Q_A, Q_B] | IDENTICAL |
| Query order | fixed 0..N-1 | fixed 0..N-1 | IDENTICAL |
| N / warmup / reps | 6980 / 300 / 3 | 6980 / 300 / 3 | IDENTICAL |
| Q_B = position-2 cache-warmed | yes | yes | IDENTICAL |
| Per-query arrays saved | PER_QUERY_BASELINE / QLR_{A,B}, ROUTE_MASK, EF_USED, SEED_CT | same set (+ BASELINE_ACC10) | IDENTICAL philosophy |
| Pooled stats + BEST/BASELINE json | yes | yes | IDENTICAL philosophy |

**Parameter mapping (native → hybrid) — every QLR parameter is a DIRECT
EQUIVALENT** (`kp`, `kep`, `th`, `ef_default`, `ef_min`, `router_ef`,
`search_type`); see `benchmarks/08_cachewarmed_treccast/expected_protocol.json`
for the full table. Two intentional, documented differences:

| Item | Benchmark 02 (native) | Benchmark 08 (hybrid) | Rationale |
|---|---|---|---|
| Baseline `efSearch` | native v2 `ef=50` | hybrid HNSW `ef=64` | Task requires the authoritative hybrid ef=64 baseline (Benchmarks 01/05/06), not the native ef=50 baseline. |
| `s_max` metric form | `quantile(ep_scores[:,0], 0.75)` (IP similarity) | `1 − Q25(ep_distances[:,0])/2` (L2 form) | The hybrid doc index is L2 over unit vectors; both yield the 75th-percentile top-1 doc similarity (`compute_s_max`, identical to Benchmark 07). |

**Allowed (path-only / protocol) changes:** package-local import wiring; PCA
loaded from the `PCA_MODEL` joblib (same asset as Benchmarks 1/5/6, identical
non-whitening transform to `rescue_full_run.py::bare_pca`); output-root
handling; `CHECK_ONLY` support; and the surrounding fixed-order B-Q_A-Q_B
cache-warmed loop. **No** search heuristic, threshold, adaptive policy,
accuracy definition, normalization, fallback semantic, timing boundary, or
output field definition was changed.

## Overall verdict

Every measured aspect of every producer is IDENTICAL between the original
canonical implementation and the packaged copy. Every change is limited to
one of: `sys.path` construction from `SUBMISSION_CODE_PKG_ROOT`, an external
file path being read from the environment, or the output directory being
selected via `OUTPUT_ROOT` (which the wrappers set to a fresh timestamped
sandbox).
