# toploc_paper_2

# Benchmark Overview

This package contains eight benchmark workflows. They are not independent experiments; several evaluate the same implementation under different operating points or evaluation protocols.

The benchmarks are grouped into three categories:

- **Hybrid FAISS (Benchmarks 1, 5, 6, 8)** — evaluates the FAISS-based QLR implementation on the full TREC-CAsT document index.
- **Native HNSW (Benchmarks 2, 3, 4)** — evaluates the native C++ implementation on the faithful 500k MS MARCO-v1 export.
- **Faithful Adaptive Search (Benchmark 7)** — reproduces the adaptive search-depth strategy described in Paper 2 (Algorithm 1).

---

## Benchmark 1 — Safe Hybrid

**Purpose**

Evaluates the hybrid FAISS implementation on the full TREC-CAsT document index using all 6,980 MS MARCO dev.small queries.

This benchmark compares:

- Baseline HNSW (`ef = 64`)
- Safe QLR (`SEEDED_EF = 32`)

The selected configuration maintains nearly identical retrieval quality while reducing retrieval latency.

**Run**

```bash
./benchmarks/01_safe_hybrid/RUN.sh
```

---

## Benchmark 2 — Cache-Warmed Best (Native)

**Purpose**

Evaluates the native C++ implementation using the cache-warmed protocol

```
Baseline → Q_A → Q_B
```

where

- **Baseline** is ordinary HNSW,
- **Q_A** is the first routed query,
- **Q_B** is executed immediately afterwards using an already warmed cache.

This benchmark measures the best-case latency behaviour of the native implementation.

**Important**

The reported **Q_B latency is a cache-warmed observation** and **must not be interpreted or reported as an independent speedup.**

**Run**

```bash
./benchmarks/02_cachewarmed_best/RUN.sh
```

---

## Benchmark 3 — Native Equal Accuracy

**Purpose**

Runs parameter sweeps for both baseline HNSW and QLR and compares them at matching Accuracy@10 operating points.

Instead of comparing identical `ef` values, this benchmark asks:

> How much faster is QLR when both methods achieve approximately the same retrieval accuracy?

This benchmark produces the representative equal-accuracy comparisons (≈93%, ≈95%, ≈96%, etc.) used throughout the project.

**Run**

```bash
./benchmarks/03_native_equal_accuracy/RUN.sh
```

---

## Benchmark 4 — Native Canonical v3

**Purpose**

Produces the canonical native implementation results.

The benchmark executes the v3 backend three times and averages all runs to report

- latency,
- accuracy,
- speedup.

This benchmark is intended to reduce run-to-run variance and produce the representative native results.

**Run**

```bash
./benchmarks/04_native_canonical_v3/RUN.sh
```

---

## Benchmark 5 — Aggressive Hybrid

**Purpose**

Uses the same execution as Benchmark 1 but reports the aggressive operating point

```
SEEDED_EF = 16
```

This configuration sacrifices retrieval quality in exchange for lower latency.

Benchmark 5 is **not executed independently**—its results are extracted directly from Benchmark 1.

**Run**

```bash
./benchmarks/05_aggressive_hybrid/RUN.sh
```

---

## Benchmark 6 — Stage-2 Pareto Search

**Purpose**

Performs the bounded Stage-2 hyperparameter search.

The benchmark evaluates 24 configurations across

- cached seed strategy,
- number of cached entry-point seeds,
- seeded beam-search width,

and produces the Pareto frontier used to select the final hybrid FAISS configuration.

**Run**

```bash
./benchmarks/06_stage2_bounded_pareto/RUN.sh
```

---

## Benchmark 7 — Faithful Adaptive Depth

**Purpose**

Implements the paper-faithful Algorithm 1 from Paper 2.

Unlike the fixed-`ef` benchmarks above, this implementation dynamically adjusts the HNSW search depth (`ef`) according to the similarity between the current query and its nearest historical query.

The benchmark performs

- calibration,
- hold-out validation,
- full evaluation

using the complete TREC-CAsT document index and all 6,980 dev.small queries.

This benchmark reproduces the adaptive search algorithm itself rather than the cache-warmed evaluation protocol.

**Run**

```bash
./benchmarks/07_faithful_adaptive_depth/RUN.sh
```

---

## Benchmark 8 — Cache-Warmed TREC-CAsT

**Purpose**

Hybrid FAISS analogue of Benchmark 2.

Benchmark 8 applies

- the cache-warmed **Baseline → Q_A → Q_B** protocol from Benchmark 2

to

- the full TREC-CAsT document corpus,
- the hybrid FAISS backend,
- together with the adaptive-search implementation from Benchmark 7.

Conceptually,

```
Benchmark 8
    = Benchmark 2 evaluation protocol
    + Benchmark 7 adaptive-search implementation
    + Full TREC-CAsT corpus
```

This benchmark should not be confused with

- Benchmark 2 (native backend),
- Benchmark 7 (adaptive search evaluation),
- Benchmark 1 (safe hybrid),
- or Benchmark 6 (parameter search).

**Run**

```bash
./benchmarks/08_cachewarmed_treccast/RUN.sh
```

---

# Configuration (`config/paths.env`)

All dataset locations, generated artifacts, Python interpreters, and runtime settings are configured through `config/paths.env`.

The benchmark scripts do **not** contain hard-coded dataset paths. Instead, every `RUN.sh` script loads `config/load_config.sh`, which reads `config/paths.env` and exports the required environment variables before executing the benchmark.

The configuration file specifies:

- Python interpreters for the hybrid, native, and reporting environments.
- Locations of the document indexes.
- Query embeddings and PCA/router artifacts.
- Native benchmark export.
- Generated preprocessing artifacts.
- CPU affinity (`CORE`) used for latency measurements.
- Optional output directory overrides.

To run the package on another machine, simply copy

```
config/paths.env.example
```

to

```
config/paths.env
```

and replace every path with the corresponding local path.

---

# Generating the Required Artifacts (`preprocessing/`)

The repository contains the code required to regenerate all preprocessing artifacts used by the benchmarks. These scripts are located under

```
preprocessing/
```

and currently include

- `build_index`
- `build_query_log_pca`
- `build_ep_table`
- `flat_index_search_acc`

Their purposes are:

| Script | Purpose |
|---------|---------|
| `build_index` | Builds the FAISS HNSW document index from document embeddings. |
| `build_query_log_pca` | Builds the PCA projection model and query-log router index used by QLR. |
| `build_ep_table` | Generates the historical entry-point lookup table used during routing. |
| `flat_index_search_acc` | Computes the exact nearest-neighbour ground truth using a Flat index for accuracy evaluation. |

The preprocessing pipeline assumes that document and query embeddings are already available.

**Embedding generation is intentionally not duplicated in this repository.** The same embedding generation pipeline used for **Paper 1** should be used to produce the document and query embeddings required here.

Likewise, the **TREC-CAsT document index** used throughout the hybrid benchmarks is built using the indexing pipeline from **Paper 1**. The scripts included in `preprocessing/` operate on those generated embeddings and indexes to produce the additional artifacts required by QLR (router index, PCA model, entry-point table, and exact ground truth).

After all preprocessing artifacts have been generated, update the corresponding paths in `config/paths.env`, and the benchmark scripts can be executed without further modification.

--------------------------------------------------------------------------------

Portable code-only reproduction package for the QLR/HNSW work. Contains every
producer script, native source file, build recipe, wrapper, small helper, and
the preprocessing / artifact-generation code required to run **eight** benchmark
workflows on a fresh machine. No datasets, no indexes, no large binary blobs,
no generated evidence.

* Experiment 7 (`07_faithful_adaptive_depth`) is the paper-faithful QLR
  Algorithm 1 with query-dependent adaptive HNSW search depth.
* Benchmark 8 (`08_cachewarmed_treccast`) is the **hybrid full TREC-CAsT**
  analogue of Benchmark 02: full MS MARCO v1 dev.small queries against the full
  TREC-CAsT document corpus, using the hybrid FAISS backend under Benchmark 02's
  cache-warmed B → Q_A → Q_B protocol.
* `preprocessing/` bundles the authoritative code that generates the benchmark
  artifacts (document index, PCA/router, EP table, exact ground truth).

Everything else is inherited byte-for-byte from the validated
`SUBMISSION_CODE_PACKAGE`.

## 1. What this package implements

QLR ("Query-Log-Router") is a two-stage retrieval accelerator layered on top
of a FAISS HNSW document index. A precomputed router (an HNSW over PCA256
projections of training queries) picks a nearest historical query per
current query; the top-K historical entry points then seed a truncated HNSW
level-0 doc search, avoiding the full HNSW descent for the fraction of
queries that route well.

The six benchmark workflows measure this on two corpora:

* **Hybrid TREC-CAsT track** (Results 1, 5, 6) — 158 GB FAISS HNSW built on
  the TREC-CAsT corpus with Snowflake Arctic Embed L 1024-d embeddings.
* **Faithful MS MARCO-v1 500k track** (Results 2, 3, 4) — flat binary export
  loaded by three C++/pybind11 backends (`native_qlr`, `_v2`, `_v3`).

## 2. The eight benchmark workflows

| ID | Directory | Producer | What it measures |
|---|---|---|---|
| 1 | `benchmarks/01_safe_hybrid`               | `python/hybrid/rescue_full_run.py`       | Safe hybrid QLR full 6980-query benchmark. Baseline ef=64 vs QLR SEEDED_EF=32 with routing; reports acc@10 and pooled-mean speedup. |
| 5 | `benchmarks/05_aggressive_hybrid`         | (dedup from Result 1)                    | Aggressive endpoint SEEDED_EF=16 measured in the same rescue_full_run.py execution. Trade-off endpoint below the 0.952 safety floor. |
| 6 | `benchmarks/06_stage2_bounded_pareto`     | `python/hybrid/rescue_stage2_accuracy.py`| Bounded 500-query Stage-2 24-config Pareto grid (seed mode × nprobe × seeded ef). Target: `recompute_l2_np3_ef16`. |
| 3 | `benchmarks/03_native_equal_accuracy`     | `python/native/benchmark_native.py`      | Native v1 sweep + QLR grid on the faithful 500k track. Equal-accuracy comparison at multiple acc@10 thresholds. |
| 4 | `benchmarks/04_native_canonical_v3`       | `python/native/canonical_final.py` × 3   | 3-run average across v1/v2/v3 backends. Aggregated by `benchmarks/04_native_canonical_v3/aggregate_results.py`. |
| 2 | `benchmarks/02_cachewarmed_best`          | `python/native/final_validate.py`        | B-Q_A-Q_B interleaved 6980-query run — Q_B is a CACHE-WARMED position-2 OBSERVATION and MUST NOT be reported as an isolated speedup. |
| 7 | `benchmarks/07_faithful_adaptive_depth`   | `python/faithful/runner.py`              | Paper-faithful QLR (Algorithm 1) with **query-dependent adaptive HNSW search depth** on the TREC-CAsT track (158 GB doc index, 6980 dev queries). Baseline is ordinary HNSW at the same `ef_default`, interleaved B-Q per query. See `benchmarks/07_faithful_adaptive_depth/README.md` + `expected_protocol.json` for the exact adaptive-ef formula and per-config parameters. Not to be confused with 1/2/6. |
| 8 | `benchmarks/08_cachewarmed_treccast`      | `python/hybrid/cachewarmed_treccast.py`  | **HYBRID full TREC-CAsT** cache-warmed B → Q_A → Q_B. Full MS MARCO v1 dev.small (all 6,980 queries) against the full TREC-CAsT document corpus (`HYBRID_DOC_INDEX`) using the hybrid FAISS QLR (`FaithfulQLR`). Adopts Benchmark 02's cache-warmed protocol/reporting **only**; the backend + corpus are hybrid TREC-CAsT (never the native 500k export). Q_B is the position-2 cache-warmed observation. See `benchmarks/08_cachewarmed_treccast/README.md` + `expected_protocol.json`. Not to be confused with native 02, frozen 01, or grid 06. |

## 3. Directory structure

```
toploc_paper_2/
├── README.md                        (this file)
├── RUN_ALL.sh                       sequential dedup-aware campaign runner (7 stages)
├── VERIFY_CODE_PACKAGE.sh           full validator (paths, imports, CHECK_ONLY)
├── requirements.txt / environment.yml
├── .gitignore
├── config/
│   ├── paths.env.example            template — supervisor edits this
│   ├── load_config.sh               package-local shell config loader
│   └── README.md
├── common/                          shell helpers (source-only)
├── benchmarks/                      seven per-result RUN.sh files
├── python/
│   ├── hybrid/                      Results 1, 5, 6 + Benchmark 8 producers + bundled src/
│   ├── native/                      Results 2, 3, 4 producers
│   └── faithful/                    Experiment 7 producer + faithful_qlr.py
├── preprocessing/                   artifact-generation code (index, PCA, EP, ground truth)
├── native/
│   ├── src/                         .cpp sources for v1, v2, v3 + original build.sh copies
│   ├── prebuilt/                    optional cpython-310 x86-64 .so binaries
│   └── build/                       destination of build/BUILD_NATIVE.sh (empty by default)
├── build/                           BUILD_NATIVE.sh + README.md
├── manifests/                       provenance + integrity evidence
├── validation/                      output of VERIFY_CODE_PACKAGE.sh
├── dist/                            regenerated archives + SHA256 sums
└── outputs/                         where RUN.sh writes fresh timestamped results
```

## 4. Required OS and architecture

* **OS**: Linux, glibc ≥ 2.34 (reference: Debian 12).
* **Kernel**: any recent x86-64 kernel.
* **CPU**: x86-64 with AVX2 and FMA. AMD Zen 2 (znver2) or newer recommended
  for v2/v3 native modules; Intel/AMD without Zen 2 must rebuild with
  `NATIVE_MARCH=native ./build/BUILD_NATIVE.sh`.
* **Disk**: negligible for the package itself (~2 MB); external data can be
  up to ~165 GB (see `manifests/EXTERNAL_RESOURCES.yaml`).

## 5. Required Python versions

* **HYBRID_PYTHON**: CPython 3.11.x with FAISS 1.9 (reference: 3.11.15).
* **NATIVE_PYTHON**: CPython 3.10.x matching the shipped `.so` ABI
  (reference: 3.10.19). Rebuild if you need a different ABI.
* **REPORT_PYTHON**: any Python 3 with numpy (reference: `/usr/bin/python3`).

## 6. Required Python dependencies

See `requirements.txt` and `environment.yml`. In short:

* HYBRID env — numpy 1.26.4, faiss-cpu 1.9.0, joblib 1.5.3, threadpoolctl
  3.6.0, pandas 3.0.3, scikit-learn 1.9.0, pyarrow 24.0.0.
* NATIVE env — numpy 1.26.4. `pybind11` 3.0.4 is required only at build time
  when rebuilding the `.so` files.

## 7. Native build requirements

* g++ ≥ 11 with C++17 (reference: g++ 12.2.0).
* AVX2 + FMA CPU (v3 additionally uses F16C).
* Python 3.10 headers (`sysconfig.get_path('include')`) and pybind11 headers
  (`pybind11.get_include()`), both discovered at build time.

## 8. Building the native extensions

```bash
edit config/paths.env                           # set NATIVE_PYTHON
"${NATIVE_PYTHON}" -m pip install pybind11
./build/BUILD_NATIVE.sh                          # writes native/build/*.so
# then in paths.env change NATIVE_MODULE_DIR to point at native/build
```

Details in `build/README.md` and `manifests/NATIVE_BUILD_MANIFEST.md`.

## 9-10. External datasets / indexes / resources

Nothing in this package includes the datasets — every path is a variable
in `config/paths.env`. `manifests/EXTERNAL_RESOURCES.yaml` lists each item's
config variable, expected file format, expected dimensions, approximate
size, workflow(s) that consume it, and how the code loads it.

Notable sizes: 158 GB TREC-CAsT HNSW index, 4.2 GB native_export, ~1 GB
FAISS router index, ~207 MB EP scores.

## 11. How to edit `config/paths.env`

```bash
cp config/paths.env.example config/paths.env
${EDITOR:-vi} config/paths.env
```

Every `/path/to/…` value in the example is intentionally a placeholder that
will fail the verifier — replace with a real path on your machine. Optional
keys are commented out; uncomment only to override.

## 12. How to validate configuration

```bash
./VERIFY_CODE_PACKAGE.sh
```

Runs bash syntax checks on every shell script, byte-compiles every Python
file under both interpreters, verifies every path in `paths.env`, imports
the three native modules, and runs `CHECK_ONLY=1` for every RUN.sh. A
non-zero exit means at least one check failed — the log names the failure.

## 13. How to run each benchmark

```bash
./benchmarks/01_safe_hybrid/RUN.sh
./benchmarks/02_cachewarmed_best/RUN.sh
./benchmarks/03_native_equal_accuracy/RUN.sh
./benchmarks/04_native_canonical_v3/RUN.sh
./benchmarks/05_aggressive_hybrid/RUN.sh
./benchmarks/06_stage2_bounded_pareto/RUN.sh
./benchmarks/07_faithful_adaptive_depth/RUN.sh
./benchmarks/08_cachewarmed_treccast/RUN.sh
```

Each RUN.sh supports:

* `CHECK_ONLY=1 ./RUN.sh` — resolve dependencies, print the exact command
  that would run, exit before starting a benchmark.
* `CAMPAIGN_DIR=/path ./RUN.sh` — write output under that shared campaign
  directory instead of `outputs/<result_id>/<STAMP>/`.

## 14. How to run `RUN_ALL.sh`

```bash
CHECK_ONLY=1 ./RUN_ALL.sh   # sanity-check every wrapper without benchmarking
YES=1 ./RUN_ALL.sh          # skip the "RUN ALL SIX" confirmation gate
./RUN_ALL.sh                # standard interactive run
```

Order preserves the source campaign convention with Experiment 7 and
Benchmark 8 appended:
1) 01_safe_hybrid, 2) 05_aggressive_hybrid (dedup from #1),
3) 06_stage2_bounded_pareto, 4) 03_native_equal_accuracy,
5) 04_native_canonical_v3, 6) 02_cachewarmed_best,
7) 07_faithful_adaptive_depth, 8) 08_cachewarmed_treccast.

## 14b. Regenerating benchmark artifacts (`preprocessing/`)

The `preprocessing/` directory bundles the authoritative producers for the
document HNSW index, the PCA model + router index, the EP table, and the exact
ground truth. It is code-only — no artifact is shipped. Validate it (never
starts a job) with:

```bash
./preprocessing/VERIFY_PREPROCESSING.sh
```

See `preprocessing/README.md` for the per-producer table, the recommended
execution order, config variables (`PREPROC_*`), and the list of missing
authoritative producers (e.g. the exact flat index and the upstream embedding
pipeline).

## 15. Where outputs are written

By default `${SUBMISSION_CODE_PKG_ROOT}/outputs/<result_id>/<UTC_STAMP>/` or,
under a campaign, `${OUTPUT_ROOT}/campaign_<STAMP>/<result_id>_<STAMP>/`.
Override with `OUTPUT_ROOT` in `config/paths.env` or on the command line.
Every output directory is timestamped; nothing overwrites previous runs.

## 16. Which metrics each benchmark reports

* Result 1 — mean/median/p95 latency, per-query acc@10, and pooled-mean +
  pooled-median speedup for the safe (`recompute_l2_np3_ef32`) and
  aggressive (`_ef16`) endpoints; per-query arrays under `full_<TS>/`.
* Result 5 — same run as Result 1; only the `recompute_l2_np3_ef16` row
  extracted, labelled AGGRESSIVE ACCURACY-SPEED TRADE-OFF.
* Result 6 — 24-config grid with per-config `overall_acc10`, `routed_acc10`,
  `overall_lat_mean_us`, `speedup_vs_baseline` (mixture-mean semantics),
  plus a Pareto CSV.
* Result 3 — baseline sweep, QLR grid, and equal-accuracy comparison at
  seven acc@10 thresholds.
* Result 4 — per-backend baseline_ef50 vs QLR (mean, median, p95, p99,
  acc10), plus 3-run averages for intra-v3 spd, compound spd v3-QLR vs
  v1-baseline, and backend engineering spd (v3/v1).
* Result 2 — interleaved baseline + Q_A + (Q_B); pooled speedups per config,
  per-query numpy arrays under `PER_QUERY_*.npy`. **The Q_B row is a
  cache-warmed observation and must never be reported as isolated.**
* Result 7 — three-phase calibration→holdout→full promotion over 12
  Algorithm-1 configurations with query-dependent adaptive `ef'` (formula in
  `benchmarks/07_faithful_adaptive_depth/expected_protocol.json` and README).
  Reports pooled-mean/median speedup vs baseline HNSW at the same
  `ef_default`, plus an equal-accuracy sweep over
  `ef ∈ [16, 24, 32, 40, 48, 64, 96, 128, 160, 200]` on the full 6980
  queries. Historical faithful metrics from the source directory are **not**
  copied and are **not** submission evidence.

## 17. Environment sensitivity of latency numbers

Exact latency and speedup values are environment-sensitive: they depend on
the CPU model, thermal state, page-cache warmth, system load, and even the
specific physical core (`CORE=21` on the reference machine). Frozen
reference numbers are baselines, not targets — reproduce shape, not
digit-for-digit equality.

## 18. Datasets and indexes intentionally excluded

The 158 GB TREC-CAsT index, 4.2 GB native_export, ~1 GB router index, and
~207 MB EP scores are not bundled. `manifests/EXTERNAL_RESOURCES.yaml` fully
describes each so the supervisor can source them independently.

## 19. This package contains code only

No PDFs, presentations, benchmark results, or logs are bundled. Every file
in this package is source code, config template, build tooling, small
required helper, or provenance manifest.

## 20. Smoke tests ≠ full benchmark reproduction

`CHECK_ONLY=1` and `VERIFY_CODE_PACKAGE.sh` prove the wrappers, imports,
and paths are correct. They do NOT reproduce the speedup numbers — a full
run of any single benchmark takes 15 minutes to 4 hours.

## 21. Known architecture, compiler, and ABI limitations

* The prebuilt `.so` files match `cpython-310-x86_64-linux-gnu`; every other
  ABI must rebuild via `build/BUILD_NATIVE.sh`.
* `-march=znver2` is used for v2/v3 by default; on non-AMD Zen 2 CPUs
  `BUILD_NATIVE.sh` automatically falls back to `-march=native`.
* FAISS 1.9's `search_level_0` C++ signature is required by the hybrid
  producers; earlier FAISS versions may miss it.
* g++ 11+ with C++17 is required.

## 22. Troubleshooting

| Symptom | Fix |
|---|---|
| `config/paths.env is missing` | `cp config/paths.env.example config/paths.env` and edit. |
| `MISSING FILE: … (env DEV_QUERY_DIR)` | The named variable in `config/paths.env` is unset or wrong. |
| `ImportError: No module named native_qlr` | Wrong `NATIVE_MODULE_DIR` — must contain `native_qlr.cpython-*.so`. Rebuild with `build/BUILD_NATIVE.sh` if the ABI differs. |
| `ImportError: sklearn.decomposition._pca` at joblib.load | `HYBRID_PYTHON` is missing scikit-learn; `pip install scikit-learn==1.9.0`. |
| CHECK_ONLY passes but full run OOMs while loading index | The 158 GB TREC-CAsT index needs ≥ ~200 GB of usable RAM/page-cache. |
| Consistent low speedup | Check `CORE`, other tenant load, and CPU frequency governor. |
| `-march=znver2` compile error | `NATIVE_MARCH=native ./build/BUILD_NATIVE.sh`. |

For anything else, inspect the log file that `VERIFY_CODE_PACKAGE.sh` wrote
into `validation/`, and cross-reference the failing variable with
`manifests/EXTERNAL_RESOURCES.yaml`.

## Provenance

* `manifests/COPY_MANIFEST.tsv` — every bundled file with before/after
  SHA256, size, workflow that requires it, and whether its content was
  path-adjusted.
* `manifests/PATH_ONLY_DIFFS.patch` — full unified diff for every rewritten
  producer.
* `manifests/BEHAVIOUR_PRESERVATION_REPORT.md` — per-behaviour equivalence
  table against the original canonical implementation.
* `manifests/DEPENDENCY_GRAPH.md` — resolved import + external-path graph.
* `manifests/ORIGINAL_SUBMISSION_REPRO_SHA256SUMS.txt` — baseline SHA256 of
  the pristine `SUBMISSION_REPRO_PACKAGE` (proven unchanged by
  `manifests/ORIGINAL_INTEGRITY_REPORT.md` after this package is built).
