# Benchmark 08 — Hybrid Full TREC-CAsT Cache-Warmed B → Q_A → Q_B

**Scientific label:**

> FULL TREC-CAsT DOCUMENT CORPUS /
> FULL MS MARCO V1 DEV.SMALL QUERY WORKLOAD /
> HYBRID FAISS QLR /
> MULTI-CONFIG CACHE-WARMED B-Q_A-Q_B POSITION-2 OBSERVATION

This is the **hybrid (Python + FAISS, TREC-CAsT)** analogue of Benchmark 02.
Benchmark 02 (native 500k MS MARCO) contributes the cache-warmed **protocol and
reporting style only**; the search backend, corpus, and every QLR building block
come from the existing hybrid stack used by Benchmarks 01, 05, 06, and 07.

## What this benchmark is (and is not)

* It uses the **same full TREC-CAsT document corpus** (≈38.6M documents/passages,
  `HYBRID_DOC_INDEX` = `treccast_hnsw_M32.index`) as hybrid Benchmarks 01, 05,
  06, and 07.
* It uses **all 6,980 MS MARCO v1 dev.small queries** — no sampling, no subset.
* It does **not** use TREC-CAsT conversational queries.
* It does **not** use the 500k native MS MARCO document export, `NATIVE_EXPORT_DIR`,
  the `native_qlr*` modules, native 500k document IDs, native EP tables, or
  native ground truth.
* Benchmark 02 contributes the **cache-warmed protocol and reporting style only**;
  the backend and corpus here are hybrid TREC-CAsT.
* **Q_B inherits cache warmth from Q_A** (position-2 observation).
* Q_B is **not** an isolated cold-cache result and is **not** automatically
  paper-comparable.
* Latency depends on page-cache state, CPU load, NUMA placement, memory
  pressure, filesystem residency, and CPU frequency.

It must never be conflated with:

* the native 500k Benchmark 02 (`02_cachewarmed_best`),
* the frozen fixed-`ef` Benchmark 01 (`01_safe_hybrid`),
* the fixed-`ef` Stage-2 grid Benchmark 06 (`06_stage2_bounded_pareto`).

## The evaluation

```
full MS MARCO v1 dev.small queries  (all 6,980)
                    ↓
full TREC-CAsT document corpus / index  (HYBRID_DOC_INDEX, ~38.6M docs)
                    ↓
hybrid baseline + hybrid QLR accuracy/latency evaluation
```

## Per-query protocol (from Benchmark 02)

One Python process, index loaded once, single thread, pinned to `taskset -c
$CORE`. For every query in **fixed order 0…N−1**, three methods run
back-to-back:

1. **B** — ordinary hybrid HNSW baseline (`efSearch = 64`, top-k = 10)
2. **Q_A** — first hybrid QLR configuration
3. **Q_B** — target hybrid QLR configuration (**position 2, cache-warmed**)

Q_B directly follows Q_A on the same query, so it inherits the PCA / router / EP
/ doc-index cache warmth Q_A just established. Defaults: `N=6980`,
`warmup=300`, `reps=3` (Benchmark 02's warmup/repetition philosophy).

## Baseline configuration

Ordinary hybrid HNSW baseline, `efSearch = 64`, top-k = 10, single-thread — the
authoritative hybrid baseline from Benchmarks 01/05/06 (`EF_DEFAULT=64`) and
Benchmark 07 (`timed_baseline`). This is **not** Benchmark 02's native `ef=50`
baseline; Benchmark 02 is only the cache-warmed structural template.

## Q_A / Q_B configuration and the Benchmark-02 → Benchmark-08 mapping

The hybrid `FaithfulQLR` (bundled for Benchmark 07) is the direct hybrid
equivalent of Benchmark 02's native `qlr(kp, kep, th, ef_default, ef_min,
router_ef, s_max, topk)` call — same router, EP-union, adaptive-`ef`, seeded
beam, and fallback semantics.

| Benchmark 02 native parameter | Benchmark 08 hybrid equivalent | Source code location | Verdict |
|---|---|---|---|
| `kp=20` (k′)                     | `QLRConfig.kp=20`                | `faithful_qlr.QLRConfig.kp`; `timed_qlr` `Ih[0,:kp]` | DIRECT EQUIVALENT |
| `kep=10` (entry points / query)  | `QLRConfig.kep=10`               | `faithful_qlr.FaithfulQLR.union_ep` | DIRECT EQUIVALENT |
| `th=0.35` (Q_A) / `0.32` (Q_B)   | `QLRConfig.th=0.35` / `0.32`     | `timed_qlr` (`s < th → fallback`) | DIRECT EQUIVALENT |
| `ef=112` (ef_default)            | `QLRConfig.ef_default=112`       | `FaithfulQLR.adaptive_ef` upper bound | DIRECT EQUIVALENT |
| `ef_min=10`                      | `QLRConfig.ef_min=10`            | `FaithfulQLR.adaptive_ef` lower bound | DIRECT EQUIVALENT |
| `rEF=16` (Q_A) / `12` (Q_B)      | `QLRConfig.router_ef=16` / `12`  | `timed_qlr` `qx.hnsw.efSearch = cfg.router_ef` | DIRECT EQUIVALENT |
| `backend=v2` (pooled beam)       | `search_type=2` (pooled beam)    | `FaithfulQLR.seeded_beam` `search_level_0` `search_type` | DIRECT EQUIVALENT |
| `s_max = quantile(ep_scores[:,0], 0.75)` (IP) | `compute_s_max(ep_distances[:,0], 0.25) = 1 − Q25(L2²)/2` | `runner.compute_s_max` | SEMANTIC EQUIVALENT (corpus metric differs — native IP vs hybrid L2 over unit vectors — both give the 75th-percentile top-1 doc similarity) |
| baseline `native v2 ef=50`       | baseline `hybrid HNSW ef=64`     | Benchmarks 01/05/06 `EF_DEFAULT=64`; `timed_baseline` | DELIBERATE DIFFERENCE (task requires the hybrid ef=64 baseline) |
| `N=6980, warmup=300, reps=3, fixed order, [B,Q_A,Q_B] per query` | identical | `benchmarks/02 RUN.sh` + `final_validate.py` | DIRECT EQUIVALENT |

**Incompatibilities:** none. Every Benchmark 02 QLR parameter has a direct
hybrid equivalent. The only intentional differences — the baseline `efSearch`
(hybrid 64 vs native 50, mandated by the task) and the `s_max` metric form
(L2 vs IP) — are flagged explicitly above and in `expected_protocol.json` and
`manifests/BEHAVIOUR_PRESERVATION_REPORT.md`.

Q_A and Q_B are **not optimized** — they are the direct mapping of Benchmark
02's fixed Q_A/Q_B pair. No parameter search, no ground-truth-based selection.

## Algorithm lock

`compute_s_max`, `timed_baseline`, and `timed_qlr` in
`python/hybrid/cachewarmed_treccast.py` are **byte-for-byte** the faithful
hybrid implementations from `python/faithful/runner.py` (Benchmark 07). The
QLR search itself is `FaithfulQLR` (`pca_transform`, `union_ep`, `adaptive_ef`,
`compute_seed_dists`, `seeded_beam`, `fallback_hnsw`) — unchanged. Only the
surrounding fixed-order B → Q_A → Q_B cache-warmed execution loop (from
Benchmark 02), the package-local import wiring, PCA loading from the
`PCA_MODEL` joblib, output-root handling, and `CHECK_ONLY` support are new.

## Configuration

Benchmark 08 reuses the **existing hybrid variables** — no new config variable
is introduced:

| Variable | Purpose |
|---|---|
| `HYBRID_PYTHON`     | FAISS 1.9 / Py 3.11 interpreter (also Benchmarks 01, 05, 06, 07) |
| `HYBRID_DOC_INDEX`  | 158 GB TREC-CAsT `treccast_hnsw_M32.index` |
| `DEV_QUERY_DIR`     | parquet shards of the 6,980 dev-query embeddings |
| `PCA_QL_DIR`        | dir with `pca_1024_to_256.joblib` + `train_query_pca256_hnsw.faiss` |
| `PCA_MODEL` (opt)   | override for `pca_1024_to_256.joblib` |
| `ROUTER_INDEX` (opt)| override for `train_query_pca256_hnsw.faiss` |
| `QLR_ARTIFACT_DIR`  | `ep_indices.npy` + `ep_distances.npy` (also the `s_max` source) |
| `EXACT_DIR`         | `exact_indices.npy` (acc@10 ground truth) |
| `OUTPUT_ROOT`, `CORE` | runtime output root + CPU affinity |

The PCA is loaded from the `PCA_MODEL` joblib (`pca.mean_`, `pca.components_`) —
the **same asset** as Benchmarks 01/05/06 — and fed to `FaithfulQLR`. This is
identical to the bare-matmul PCA used by `rescue_full_run.py` (both non-whitening),
so no `FAITHFUL_PCA_DIR` arrays are needed.

## Usage

```bash
cp config/paths.env.example config/paths.env
${EDITOR:-vi} config/paths.env

# Dry-run — resolves paths, prints B/Q_A/Q_B, corpus + query count,
# and the exact command; exits before loading the 158 GB index.
CHECK_ONLY=1 ./benchmarks/08_cachewarmed_treccast/RUN.sh

# Full benchmark (158 GB index, 6980 queries × 3 reps × 3 methods)
./benchmarks/08_cachewarmed_treccast/RUN.sh
```

## Output

A timestamped directory `outputs/08_cachewarmed_treccast/<UTC_STAMP>/`; the
producer writes `cachewarmed_treccast_<local_stamp>/` inside it containing:

* `final_validation.json` — corpus metadata, query count, baseline / Q_A / Q_B
  accuracy, mean/median/p95/p99 latency, Q_A & Q_B speedup vs baseline,
  routed/fallback counts, adaptive-`ef` and component-timing means, and the
  cache-warmed caveat.
* `BASELINE_RESULT.json`, `BEST_RESULT.json`, `BEST_CONFIG.json`
* `PER_QUERY_BASELINE.npy`, `PER_QUERY_QLR_A.npy`, `PER_QUERY_QLR_B.npy`
* `ROUTE_MASK_{A,B}.npy`, `EF_USED_{A,B}.npy`, `SEED_CT_{A,B}.npy`,
  `BASELINE_ACC10.npy`
* `RESULT_LABEL.txt` (mandatory cache-warmed label)

The wrapper additionally writes `command.txt`, `environment.txt`,
`pip_versions.txt`, `wrapper_metadata.json`, and the stdout/stderr/combined logs.

## Reproducibility notes

* Exact latency is environment-sensitive (page-cache warmth, `/dev/shm`
  residency, memory pressure, NUMA, CPU-thermal history). Absolute microsecond
  numbers differ across machines.
* The 158 GB TREC-CAsT index needs ~200 GB usable RAM + page-cache to run the
  full workload.
* No expected speedup is asserted here — the benchmark reports what it measures.
