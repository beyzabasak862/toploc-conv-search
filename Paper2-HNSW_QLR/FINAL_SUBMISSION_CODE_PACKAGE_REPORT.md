# FINAL_SUBMISSION_CODE_PACKAGE_REPORT.md

Generated 2026-07-23 on host `pegasus`. Authoritative summary of the
`toploc_paper_2` repackage — a GitHub-ready code-only copy of the validated
`SUBMISSION_CODE_PACKAGE` with Experiment 7 (`07_faithful_adaptive_depth`)
added as a first-class benchmark.

> **Extension (2026-07-23):** Benchmark 08 (`08_cachewarmed_treccast`) and the
> `preprocessing/` directory were added on top of the Experiment-7 package. The
> Benchmark-08 + preprocessing summary is in **§EXT** immediately below;
> the original Experiment-7 report follows unchanged from §1.

---

## §EXT — Benchmark 08 + preprocessing extension

**1. Benchmark 08 path:** `benchmarks/08_cachewarmed_treccast/`
(`RUN.sh`, `README.md`, `expected_protocol.json`).

**2. Hybrid producer:** `python/hybrid/cachewarmed_treccast.py`.

**3. Backend is hybrid FAISS, not native.** The producer imports `FaithfulQLR`
(`python/faithful/faithful_qlr.py`) + `src.data_loading` + `src.indexing`, reads
`HYBRID_DOC_INDEX`, and uses `faiss` `IndexHNSWFlat.search` / `search_level_0`.
It contains **zero** `native_qlr` / `NATIVE_EXPORT_DIR` / `NATIVE_MODULE_DIR`
references (`validation/step08_bench08_backend.log`; asserted by
`verify_cachewarmed_treccast_import`).

**4. Document corpus:** full TREC-CAsT (`HYBRID_DOC_INDEX` =
`treccast_hnsw_M32.index`, ~38.6M docs) — the same doc-ID space as Benchmarks
01/05/06/07. Never the native 500k export.

**5. Query set:** full MS MARCO v1 dev.small, **all 6,980** queries, no
sampling (producer asserts `n_total == 6980`).

**6. Baseline:** ordinary hybrid HNSW, `efSearch = 64`, top-k = 10,
single-thread (authoritative hybrid baseline from 01/05/06; `timed_baseline`).
NOT the native ef=50 baseline.

**7. Q_A / Q_B (mapped 1:1 from Benchmark 02):**
* Q_A: `kp=20, kep=10, th=0.35, ef_default=112, ef_min=10, router_ef=16, search_type=2`
* Q_B: `kp=20, kep=10, th=0.32, ef_default=112, ef_min=10, router_ef=12, search_type=2`
* `s_max = 1 − Q25(ep_distances[:,0])/2` (paper 75th-pct top-1 similarity, L2 form).

**8. B → Q_A → Q_B implementation:** `python/hybrid/cachewarmed_treccast.py`,
`main()` benchmark loop (fixed order 0..N-1; per query `timed_baseline` then
`timed_qlr(cfg_a)` then `timed_qlr(cfg_b)`).

**9. Benchmark-02 elements preserved:** one process, index loaded once,
per-query [B, Q_A, Q_B], fixed query order, `N=6980 / warmup=300 / reps=3`,
Q_B = cache-warmed position 2, per-query latency arrays, pooled stats +
BEST/BASELINE json, mandatory cache-warmed `RESULT_LABEL.txt`.

**10. Native / 500k elements deliberately excluded:** native_qlr modules,
`NATIVE_EXPORT_DIR`, `NATIVE_MODULE_DIR`, native 500k doc IDs / EP table /
ground truth, native ef=50 baseline, TREC-CAsT conversational queries.

**11. New config variables:** Benchmark 08 adds **none** (reuses
`HYBRID_PYTHON`, `HYBRID_DOC_INDEX`, `DEV_QUERY_DIR`,
`PCA_QL_DIR`/`PCA_MODEL`/`ROUTER_INDEX`, `QLR_ARTIFACT_DIR`, `EXACT_DIR`).
Preprocessing adds: `PREPROC_OUTPUT_ROOT`, `PREPROC_DOC_EMB_DIR`,
`PREPROC_TRAIN_QUERY_DIR`, `PREPROC_FLAT_INDEX` (and reuses `HYBRID_DOC_INDEX`,
`DEV_QUERY_DIR`).

**12. Preprocessing producers copied (path-adjusted):**
`preprocessing/scripts/{build_index.py, build_query_log_pca.py,
build_ep_table.py, flat_index_search_acc.py}`.

**13. Preprocessing helpers copied (byte-identical):**
`preprocessing/src/{data_loading.py, indexing.py}` + generated `__init__.py`.

**14. Recommended preprocessing order:** `build_index.py` →
`build_query_log_pca.py` → `build_ep_table.py` → `flat_index_search_acc.py`
(steps 2 and 4 independent; step 3 needs step 1's doc index). See
`preprocessing/README.md`.

**15. Supported generated artifacts:** full document HNSW index; PCA model +
router index + qmax; EP table (`ep_indices`, `ep_distances`); exact ground
truth (`exact_scores`, `exact_indices`).

**16. Missing producers:** exact flat index (`treccast_flat.index`); upstream
embedding pipeline (doc/query parquet shards); extracted faithful PCA arrays
(derived from the joblib); native export (out of scope). All marked
`MISSING AUTHORITATIVE PRODUCER` — no replacement algorithm written.

**17. CHECK_ONLY (01–08 + RUN_ALL):** all PASS —
`validation/step09_check_only_each.log` (01,05,06,07,08),
`validation/step09_native_check_only.log` (02,03,04). `CHECK_ONLY=1 ./RUN_ALL.sh`
reports **eight** workflows.

**18. Preprocessing validation:** 14 PASS / 0 WARN / 0 FAIL —
`preprocessing/validation/VALIDATION_REPORT.md`.

**19. Benchmarks 01–07 unchanged:** every 01–07 producer + wrapper is
byte-identical to the pre-extension baseline —
`validation/step14_bm01_07_unchanged.log` (0 differing hashes). Only
package-level registration files changed (`RUN_ALL.sh`, `VERIFY_CODE_PACKAGE.sh`,
`common/common_env.sh`, `common/verify_paths.sh`) plus manifests/README/config.

**20. Originals + index unchanged:** `SUBMISSION_CODE_PACKAGE`,
`SUBMISSION_REPRO_PACKAGE`, faithful source dir, and the four preprocessing
source files all 0-hash-diff; 158 GB index size + mtime unchanged —
`validation/step11_13_upstream_integrity.log`.

**21. Package size:** 116 files, 3.1M (excluding `outputs/`, `dist/`, `preprocessing/_generated/`).

**22. Archives:** `dist/toploc_paper_2.{tar.gz,zip}` — authoritative SHA256 in
`dist/toploc_paper_2.tar.gz.sha256` and `dist/toploc_paper_2.zip.sha256`
(printed in the terminal at build time).

---


## 1. Exact destination path

```
/home/toploc1/Datasets/toploc1/toploc_paper_2
```

Placed at the same filesystem level as `HNSW`, `Data Exploration`, `Exact_Search`.
Not nested under `HNSW/msmarco_HNSW`.

## 2. Exact Experiment 7 path

```
/home/toploc1/Datasets/toploc1/toploc_paper_2/benchmarks/07_faithful_adaptive_depth/
```

Contains `RUN.sh`, `README.md`, `expected_protocol.json`.

## 3. Exact faithful producer / helper files copied

| Destination | Kind | Provenance |
|---|---|---|
| `python/faithful/runner.py`       | path-adjusted producer (Experiment 7) | `paper2_final_track/optimization_search/paper2_faithful_20260718_231400/runner.py` |
| `python/faithful/faithful_qlr.py` | verbatim algorithm module              | `paper2_final_track/optimization_search/paper2_faithful_20260718_231400/faithful_qlr.py` |
| `python/faithful/__init__.py`     | generated (explicit-package init)      | new |

Reused (already bundled by the source package, shared with 1/5/6):

* `python/hybrid/src/data_loading.py` — `load_embeddings_from_parquets`, `l2_normalize`
* `python/hybrid/src/indexing.py` — `load_index` (`faiss.read_index`)
* `python/hybrid/src/__init__.py`

SHA256 pairs (original vs packaged) are in `manifests/COPY_MANIFEST.tsv`.
The unified diff for `runner.py` is appended to `manifests/PATH_ONLY_DIFFS.patch`.

## 4. Exact baseline configuration (Experiment 7)

* Type: ordinary `faiss.IndexHNSWFlat.search(q, 10)` — no seeding, no candidate
  union, no router.
* `efSearch` used: `cfg.ef_default` — per-config in `{32, 48, 64}`.
* Interleave: per-query B-Q (baseline first, then QLR) with the same loaded
  index, same warm-up (30 queries), same query order.
* Additional full-sample sweep: `ef ∈ [16, 24, 32, 40, 48, 64, 96, 128, 160, 200]`
  measured on all 6980 queries (2 reps) whenever a config is promoted to full,
  used for equal-accuracy comparison (paper Table 1 methodology).

## 5. Exact adaptive-depth logic and parameters (Experiment 7)

Formula from `python/faithful/faithful_qlr.py::FaithfulQLR.adaptive_ef`:

```
if s < th:              return HnswSearch(q, ef_default)   # fallback (not adaptive)
if s > s_max:           return int(ef_min)
if s_max - th <= 0:     return int(ef_default)
ef' = ef_min + (ef_default - ef_min) * (s_max - s) / (s_max - th)
ef' = clip(ef', ef_min, ef_default)
return int(round(ef'))
```

Parameter sources:

* `s`         = top-1 router similarity of the historical query I_Q (per-query).
* `s_max`     = `1.0 - Q25(ep_distances[:,0]) / 2.0` — paper's 75th percentile
  of top-1 doc similarity, computed at run time from
  `${QLR_ARTIFACT_DIR}/ep_distances.npy` in
  `python/faithful/runner.py::compute_s_max`.
* `th`, `ef_min`, `ef_default` — per calibration config: 12 rows enumerated in
  `benchmarks/07_faithful_adaptive_depth/expected_protocol.json`.

## 6. Every new config variable added

| Variable | Purpose |
|---|---|
| `FAITHFUL_PCA_DIR`         | dir with `pca_mean_1024.npy` + `pca_components_256x1024.npy` |
| `FAITHFUL_PCA_MEAN` (opt)  | override for `pca_mean_1024.npy` |
| `FAITHFUL_PCA_COMPONENTS` (opt) | override for `pca_components_256x1024.npy` |
| `FAITHFUL_DOC_INDEX_SHM` (opt)  | optional `/dev/shm/...` fast path for the doc index (mmap) |
| `FAITHFUL_QUERY_INDEX_SHM` (opt)| optional `/dev/shm/...` fast path for the router index |

Every other external variable Experiment 7 needs is already declared for the
hybrid track: `HYBRID_PYTHON`, `HYBRID_DOC_INDEX`, `DEV_QUERY_DIR`,
`PCA_QL_DIR` (only `ROUTER_INDEX` from it), `ROUTER_INDEX`, `QLR_ARTIFACT_DIR`,
`EXACT_DIR`.

Documented in: `config/paths.env.example`, `config/paths.env` (reference
values), `config/README.md`, `manifests/EXTERNAL_RESOURCES.yaml`.

## 7. Every copied package file modified to register Experiment 7

| File | Change |
|---|---|
| `README.md`                            | 6→7 workflows in headline, structure diagram, table, RUN_ALL order, "how to run" list, per-benchmark metric section |
| `RUN_ALL.sh`                           | +`07_faithful_adaptive_depth` in the summary block, in the CHECK_ONLY loop, and as Stage 7 in the sequential runner |
| `VERIFY_CODE_PACKAGE.sh`               | +`verify_faithful_paths` + `verify_faithful_import` calls; +07 in the per-wrapper CHECK_ONLY loop |
| `common/common_env.sh`                 | +`FAITHFUL_PY_SCRIPT` export |
| `common/verify_paths.sh`               | +`verify_faithful_paths` + `verify_faithful_import` functions |
| `config/paths.env.example`             | title change; +faithful-specific block (5 new vars) |
| `config/paths.env`                     | title change; +faithful-specific block (reference-machine `FAITHFUL_PCA_DIR`) |
| `config/README.md`                     | +5 faithful rows in the variable table |
| `.gitignore`                           | +`dist/` + `validation/*.log` exclusions (and their `.gitkeep` preservation) |
| `manifests/COPY_MANIFEST.tsv`          | +3 rows for the faithful bundle |
| `manifests/PATH_REWRITE_MAP.tsv`       | +11 rows for the faithful runner rewrites |
| `manifests/PATH_ONLY_DIFFS.patch`      | appended unified diff for `runner.py` |
| `manifests/BEHAVIOUR_PRESERVATION_REPORT.md` | +§`faithful_qlr.py` + §`runner.py` (every row IDENTICAL) |
| `manifests/DEPENDENCY_GRAPH.md`        | +§Experiment 7 chain |
| `manifests/DEPENDENCY_CLASSIFICATION.tsv` | +19 rows for Experiment 7 |
| `manifests/EXTERNAL_RESOURCES.yaml`    | version 1 → 2, +7 blocks for faithful resources |
| `manifests/PYTHON_DEPENDENCIES.txt`    | annotated HYBRID section (which deps are shared, which are 1/5/6-only) |
| `manifests/PACKAGE_TREE.txt`           | regenerated (112 entries) |
| `manifests/FINAL_FILE_LIST.tsv`        | regenerated (86 rows + header) |
| `manifests/FINAL_SHA256SUMS.txt`       | regenerated (86 rows) |
| `manifests/POST_SUBMISSION_REPRO_SHA256SUMS.txt` | regenerated (351 rows) |
| `manifests/ORIGINAL_INTEGRITY_REPORT.md` | added toploc_paper_2 integrity assertions |
| `manifests/SUBMISSION_READINESS_CHECKLIST.md` | 21 → 28 rows (added 7 for Experiment 7) |
| `FINAL_SUBMISSION_CODE_PACKAGE_REPORT.md` | this file |

No producer script for Experiments 1–6 was modified.

## 8. Results of all seven CHECK_ONLY runs

| Wrapper | CHECK_ONLY result |
|---|---|
| `01_safe_hybrid`             | PASS |
| `02_cachewarmed_best`        | PASS |
| `03_native_equal_accuracy`   | PASS |
| `04_native_canonical_v3`     | PASS |
| `05_aggressive_hybrid`       | PASS |
| `06_stage2_bounded_pareto`   | PASS |
| `07_faithful_adaptive_depth` | PASS |
| `RUN_ALL.sh`                 | PASS |

Evidence: `validation/step8_check_only_each.log` (7/7 wrappers + RUN_ALL) and
`validation/step9_external_copy.log` (external copy of 01 + 07 with cleared
`PYTHONPATH` — both PASS).

## 9. Results of shell, Python, native, and faithful validation

| Check | Result |
|---|---|
| `bash -n` on every shell script (17/17)                              | PASS |
| Python `compileall` under `HYBRID_PYTHON` + `NATIVE_PYTHON`          | PASS |
| Faithful import test (`verify_faithful_import`)                      | PASS |
| Native import test (`verify_native_import`, `v3 ntotal=500000 iq=808731 dim=1024 ep=118295`) | PASS |
| Forbidden absolute path scan in executed code                        | PASS |
| Symlink escape scan                                                  | PASS |
| Oversized file scan (>50 MB)                                         | PASS |
| Secret scan                                                          | PASS |

## 10. Did Experiments 1–6 remain unchanged?

YES. `validation/step14_experiments_1_6_unchanged.log` shows every producer
subtree (`python/hybrid`, `python/native`, `benchmarks/{01..06}`, `native/`,
`build/`) IDENTICAL between the source `SUBMISSION_CODE_PACKAGE` and
`toploc_paper_2`. The only non-`.pyc` file-level "diffs" are files
intentionally modified to register Experiment 7 (README, RUN_ALL,
VERIFY_CODE_PACKAGE, `common/*.sh`, `config/*`, `.gitignore`, manifests),
each individually enumerated in section 7 above.

## 11. Did the original package remain byte-identical?

YES.

* Source `SUBMISSION_CODE_PACKAGE`: 90 non-output files re-hashed, hash-only
  diff against baseline = **0 lines** →
  `validation/step10_source_pkg_integrity.log`.
* `SUBMISSION_REPRO_PACKAGE`: 351 files re-hashed, hash-only diff against
  `manifests/ORIGINAL_SUBMISSION_REPRO_SHA256SUMS.txt` = **0 lines** →
  `validation/step11_repro_pkg_integrity.log`.
* Faithful source directory `paper2_faithful_20260718_231400/`: 22 files
  re-hashed, hash-only diff against baseline = **0 lines** →
  `validation/step12_faithful_source_integrity.log`.
* 158 GB `treccast_hnsw_M32.index`: size 168 768 858 058 bytes, mtime
  `2026-05-13 15:26:03`, both unchanged from Phase-0 baseline →
  `validation/step13_index_integrity.log`.

## 12. Package file count and total size

* Directory (excluding `dist/` and `outputs/`):        **86 files, 2.7 MB**
* Directory (with regenerated `dist/` archives):       **91 files, 2.8 MB**

## 13. ZIP and tar.gz paths

* `dist/toploc_paper_2.tar.gz` — 95 tar entries, single top-level directory `toploc_paper_2/`
* `dist/toploc_paper_2.zip`    — 94 zip entries, single top-level directory `toploc_paper_2/`

Absolute:
* `/home/toploc1/Datasets/toploc1/toploc_paper_2/dist/toploc_paper_2.tar.gz`
* `/home/toploc1/Datasets/toploc1/toploc_paper_2/dist/toploc_paper_2.zip`

## 14. Archive SHA256 values

The authoritative SHA256 for each shipped archive lives next to it:

* `dist/toploc_paper_2.tar.gz.sha256`
* `dist/toploc_paper_2.zip.sha256`

Read those files after the final `zip`/`tar` invocation for the exact bytes.
Because these `.sha256` files, `PACKAGE_TREE.txt`, `FINAL_FILE_LIST.tsv`, and
this very report all end up inside the archive, the hash-of-an-archive that
lists its own hash forms a fixed-point that cannot be embedded in the archive
without a further rebuild. The `dist/*.sha256` files are the last write in
the build, so they are always the correct answer.

Also written to `dist/toploc_paper_2.tar.gz.sha256` and
`dist/toploc_paper_2.zip.sha256`.

Excluded from the archives: `config/paths.env` (machine-specific), `outputs/*`
(runtime), `dist/*` (avoid nesting), `validation/*.log`, all `__pycache__`
directories and `*.pyc`/`*.pyo`. All `.gitkeep` files preserved.

## 15. Remaining portability limitations

* Prebuilt `.so` binaries are `cpython-310-x86_64-linux-gnu`; any other ABI
  requires `build/BUILD_NATIVE.sh`.
* Native builds default to `-march=znver2` for v2/v3; automatic fallback to
  `-march=native` is present when the compiler lacks Zen 2 support.
* `HYBRID_PYTHON` requires FAISS 1.9 to expose `search_level_0` with the
  9-argument C++ signature used by the faithful producer.
* Full-benchmark memory footprint (158 GB TREC-CAsT index) requires ~200 GB
  usable RAM + page-cache — this is not a portability issue of the code but
  of the environment.
* Experiment 7's `/dev/shm/qlr_indexes/` fast path is preferred if present
  but not required; the wrapper transparently falls back to
  `HYBRID_DOC_INDEX` + `ROUTER_INDEX` when the tmpfs files do not exist.

## 16. Blockers

None.

## 17. Exact commands the GitHub-uploading friend should use

```bash
cd /home/toploc1/Datasets/toploc1/toploc_paper_2

# 1. Configure paths for this machine
cp config/paths.env.example config/paths.env
${EDITOR:-vi} config/paths.env

# 2. Verify everything resolves
./VERIFY_CODE_PACKAGE.sh

# 3. Dry-run every wrapper (no benchmark launched)
CHECK_ONLY=1 ./RUN_ALL.sh

# 4. Run any single benchmark
./benchmarks/01_safe_hybrid/RUN.sh
./benchmarks/02_cachewarmed_best/RUN.sh
./benchmarks/03_native_equal_accuracy/RUN.sh
./benchmarks/04_native_canonical_v3/RUN.sh
./benchmarks/05_aggressive_hybrid/RUN.sh
./benchmarks/06_stage2_bounded_pareto/RUN.sh
./benchmarks/07_faithful_adaptive_depth/RUN.sh
```

For GitHub upload:

```bash
cd /home/toploc1/Datasets/toploc1/toploc_paper_2
git init
git add .
git commit -m "Initial commit: toploc_paper_2"
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin master
```

`.gitignore` already excludes `config/paths.env`, `outputs/*`, `dist/*`,
`validation/*.log`, `__pycache__/`, `*.pyc`, and editor caches.
