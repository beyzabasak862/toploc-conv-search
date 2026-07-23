# Experiment 7 — Faithful QLR with Adaptive Search Depth

This benchmark compares **ordinary HNSW baseline** against the
**paper-faithful QLR (Algorithm 1) with query-dependent adaptive HNSW search
depth**, on the TREC-CAsT track (158 GB doc index, 6,980 dev queries).

The implementation is a byte-identical (algorithm module) / path-only-rewritten
(runner) copy of the authoritative faithful experiment at
`claude_qlr_diagnostics/paper2_final_track/optimization_search/paper2_faithful_20260718_231400/`.
It is bundled inside this package under `python/faithful/`; nothing is loaded
from the original repository.

## What is being compared

* **Baseline** — for every query, `IndexHNSWFlat.search(q, 10)` at the config's
  own `ef_default` (32, 48, or 64). One search per query, single thread. This
  is the same call FAISS end-users make with no QLR involved.
* **Adaptive QLR** — for every query, the paper's Algorithm 1: PCA-project → I_Q
  router → threshold gate → union of historical entry points → seeded HNSW beam
  search at a **query-dependent** `ef'` chosen from the router similarity.

Baseline and adaptive are **interleaved on every query** (B-Q pattern), sharing
the same loaded index, same warm-up, same query order, same single-threaded
environment (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1`), pinned to `taskset -c $CORE` (default core 21).

## Adaptive search depth — what it actually does

Adaptive search depth means the FAISS `efSearch` parameter is chosen **per
query** from the top-1 similarity `s` that the router returns for the historical
query index `I_Q`. From `python/faithful/faithful_qlr.py::FaithfulQLR.adaptive_ef`:

```
if s < th:                # low-confidence route -> full HNSW fallback
    return HnswSearch(q, ef_default)
if s > s_max:             # very-confident route -> minimum beam width
    return int(ef_min)
ef' = ef_min + (ef_default - ef_min) * (s_max - s) / (s_max - th)
ef' = clip(ef', ef_min, ef_default)
return int(round(ef'))
```

* `s_max` is computed **at run time** from
  `${QLR_ARTIFACT_DIR}/ep_distances.npy` as the 25th percentile of squared-L2
  distance — equal to the 75th percentile of similarity (paper definition:
  `runner.py::compute_s_max`).
* `th`, `ef_min`, `ef_default` are per calibration config (see the 12 rows in
  `expected_protocol.json`).

**This is different from just choosing better entry points.** The candidate
seeds (`C = union_{i=1..k'} EP(q_i)` with de-duplication) are still supplied to
the beam search, but the *depth* of the beam is what changes per-query. It is
also different from a fixed-ef sweep: no single `ef` is used across queries.

## Not to be confused with

Experiment 7 must **not** be presented as, or conflated with, any of:

* the frozen fixed-`ef` Result 1 (`01_safe_hybrid`)
* the cache-warmed native Result 2 (`02_cachewarmed_best`)
* the fixed-`ef` Stage-2 grid Result 6 (`06_stage2_bounded_pareto`)

## How the producer decides which configs advance

`runner.py` runs three promotion phases (all baked in, no CLI args):

| Phase | queries | configs | reps | promotion rule |
|-------|---------|---------|------|----------------|
| Calibration | 500  | 12 (A…L) | 2 | `qlr_acc >= 0.952 AND qlr_acc >= base_acc - 0.005` → top 3 by pooled-mean speedup advance |
| Holdout     | 1500 | ≤3       | 3 | same safe-vs-base gate → top 2 advance to full |
| Full        | 6980 | ≤2       | 3 | latency arrays written to `latency_arrays/` |

A separate baseline `ef` sweep over `[16, 24, 32, 40, 48, 64, 96, 128, 160,
200]` is measured on the full 6980 queries (2 reps) whenever a config reaches
the full phase, so the report can express equal-accuracy speedup vs the fastest
same-accuracy baseline (paper Table 1 methodology).

The exact per-config parameters are frozen in `expected_protocol.json`.

## Configuration

Environment variables read from `config/paths.env` (see
`config/paths.env.example` for the template). Experiment 7 shares every
existing external variable with the hybrid track:

| Variable | Purpose |
|---|---|
| `HYBRID_PYTHON`     | FAISS 1.9 Python interpreter (also runs Experiments 1, 5, 6) |
| `DEV_QUERY_DIR`     | parquet shards of the 6980 dev-query embeddings |
| `HYBRID_DOC_INDEX`  | 158 GB TREC-CAsT `treccast_hnsw_M32.index` |
| `PCA_QL_DIR`        | dir containing `train_query_pca256_hnsw.faiss` (I_Q router) |
| `ROUTER_INDEX`      | (optional) explicit path to the I_Q router index |
| `QLR_ARTIFACT_DIR`  | `ep_indices.npy` + `ep_distances.npy` (also used for `s_max`) |
| `EXACT_DIR`         | `exact_indices.npy` + `exact_scores.npy` (acc@10 ground truth) |

Plus these additional Experiment 7 variables:

| Variable | Purpose |
|---|---|
| `FAITHFUL_PCA_DIR`         | dir containing `pca_mean_1024.npy` (`float32 [1024]`) and `pca_components_256x1024.npy` (`float32 [256, 1024]`) |
| `FAITHFUL_PCA_MEAN`        | (optional) explicit path to `pca_mean_1024.npy` |
| `FAITHFUL_PCA_COMPONENTS`  | (optional) explicit path to `pca_components_256x1024.npy` |
| `FAITHFUL_DOC_INDEX_SHM`   | (optional) `/dev/shm/...` fast-path for the doc index (mmap-loaded) |
| `FAITHFUL_QUERY_INDEX_SHM` | (optional) `/dev/shm/...` fast-path for the router index |

The PCA files are **not bundled** with this package; they are inputs the paper
produced during router training. Approximate sizes: `pca_mean_1024.npy` ≈ 4 KB,
`pca_components_256x1024.npy` ≈ 1 MB.

## Usage

```bash
# 1. Configure once (adjust paths.env for your machine)
cp config/paths.env.example config/paths.env
${EDITOR:-vi} config/paths.env

# 2. Dry-run — resolves every path and prints the exact command, does NOT
#    load the 158 GB index or launch the benchmark
CHECK_ONLY=1 ./benchmarks/07_faithful_adaptive_depth/RUN.sh

# 3. Full benchmark (calib -> holdout -> full 6980 x 3 reps + baseline sweep on
#    full 6980 x 2 reps; wall clock ranges from several hours to a full day
#    depending on I/O regime — see the runner's baseline_sanity.json regime tag)
./benchmarks/07_faithful_adaptive_depth/RUN.sh
```

## Output

Each RUN.sh invocation creates a timestamped directory
`outputs/07_faithful_adaptive_depth/<UTC_STAMP>/`. `runner.py` writes its
own `faithful_<local_stamp>/` inside that, containing (per phase):

* `config_manifest.json`
* `baseline_sanity.json`, `baseline_sweep.json`, `baseline_sweep_full.json`
* `calibration.json`, `holdout.json`, `full.json`, `SUMMARY.json`
* `latency_arrays/<config_name>.npz` (per finalist)

The wrapper additionally writes `command.txt`, `environment.txt`,
`pip_versions.txt`, `wrapper_metadata.json`, `stdout.log`, `stderr.log`,
`combined.log`, and `canonical_output_source_path.txt`.

## Reproducibility notes

* Wall-clock latency is environment-sensitive (SSHFS-cold I/O vs `/dev/shm`
  mmap, memory pressure, CPU-thermal history). Absolute microsecond numbers on
  a different machine will differ; the equal-accuracy speedup shape should
  survive as long as the doc index fits in the OS page-cache or `/dev/shm`.
* This is the **paper-faithful hybrid corpus/setup**: 158 GB TREC-CAsT doc
  index, 6980 MS MARCO dev queries, PCA-projected query router. The
  faithful implementation's known limitations from its `CLAUDE IMPROVEMENT`
  header (single-thread FAISS, cold-cache first-touch effects) remain.
* Historical result metrics from
  `paper2_final_track/optimization_search/paper2_faithful_20260718_231400/results/`
  are **not** copied into this code-only package and are **not** submission
  evidence. If cited, they are historical reference observations from that
  artifact path only.
