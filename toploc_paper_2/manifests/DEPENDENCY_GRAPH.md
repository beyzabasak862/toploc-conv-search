# DEPENDENCY_GRAPH.md

Full dependency chain for every RUN.sh in this package, resolved against the
two authoritative execution environments discovered in Phase 1:

* `HYBRID_PYTHON = /home/toploc1/miniforge3/envs/toploc-cpp/bin/python`  (Python 3.11.15)
* `NATIVE_PYTHON = /home/fatemeh/anaconda3/envs/zeroec/bin/python`       (Python 3.10.19)

External resources (large datasets/indexes not bundled) are enumerated in
`manifests/EXTERNAL_RESOURCES.yaml`. Every source file that IS bundled has a
row in `manifests/COPY_MANIFEST.tsv` with SHA256 before/after the copy.

## Result 1 — 01_safe_hybrid, and Result 5 — 05_aggressive_hybrid (same run)

### Shell chain
```
benchmarks/01_safe_hybrid/RUN.sh
├── source config/load_config.sh    (via common/common_env.sh)
├── source common/timestamp_helpers.sh
├── source common/verify_paths.sh
└── exec taskset -c $CORE  env OUTPUT_ROOT=<sandbox>  $HYBRID_PYTHON -u python/hybrid/rescue_full_run.py
```

### Python chain
```
python/hybrid/rescue_full_run.py
├── numpy 1.26.4              (external — HYBRID env)
├── faiss 1.9.0               (external — HYBRID env; provides IndexHNSWFlat, search_level_0)
├── joblib 1.5.3              (external — HYBRID env; loads the pickled PCA)
├── threadpoolctl 3.6.0       (external — HYBRID env; wraps timed loops)
├── pandas 3.0.3              (transitive via src/data_loading; parquet reader)
├── scikit-learn 1.9.0        (required only for the pickled PCA class definition)
├── from src.data_loading import load_embeddings_from_parquets, l2_normalize
│    └── python/hybrid/src/data_loading.py   (bundled; SHA256 in COPY_MANIFEST.tsv)
└── from src.indexing import load_index
     └── python/hybrid/src/indexing.py       (bundled; SHA256 in COPY_MANIFEST.tsv)
```

### External runtime paths (config/paths.env)
```
DEV_QUERY_DIR      -> parquet shards with dev-query embeddings (~43 MB)
HYBRID_DOC_INDEX   -> 158 GB TREC-CAsT FAISS HNSW M=32 index
PCA_QL_DIR         -> pca_1024_to_256.joblib (~1 MB) + train_query_pca256_hnsw.faiss (~1 GB) + qmax_pca256.npy
QLR_ARTIFACT_DIR   -> ep_indices.npy (~32 MB) + ep_distances.npy (~32 MB)
EXACT_DIR          -> exact_indices.npy (~546 KB) — ground-truth top-10
```

## Result 6 — 06_stage2_bounded_pareto

Same shell + Python chain as Result 1 with `python/hybrid/rescue_stage2_accuracy.py`
as the producer.  Adds one runtime path:

```
PCA_QMAX  (defaults to PCA_QL_DIR/qmax_pca256.npy) — scalar float32, 132 bytes
```

## Results 2, 3, 4 — native track

### Shell chain
```
benchmarks/{02,03,04}/RUN.sh
├── source config/load_config.sh   (via common/common_env.sh)
├── source common/timestamp_helpers.sh
├── source common/verify_paths.sh
└── exec taskset -c $CORE  [env OUTPUT_ROOT=<sandbox>]  $NATIVE_PYTHON -u python/native/<producer>.py [--out ...]
```

### Python chain
```
python/native/benchmark_native.py    (Result 3)
python/native/canonical_final.py     (Result 4)
python/native/final_validate.py      (Result 2)
├── numpy 1.26.4                (external — NATIVE env)
└── sys.path.insert(0, os.environ["NATIVE_MODULE_DIR"])
    ├── import native_qlr       (from native/prebuilt/*.so or native/build/*.so)
    ├── import native_qlr_v2    (v2 + v3 not imported by benchmark_native.py)
    └── import native_qlr_v3
```

Result 4 additionally runs `benchmarks/04_native_canonical_v3/aggregate_results.py`
(byte-identical copy of the original, no path edits).

### Native module ABI

Each `.so` is a self-contained pybind11 module compiled with g++ 12 for
CPython 3.10 x86-64 (cpython-310-x86_64-linux-gnu):

| Module | -march flags used | -mavx2 | -mfma | -mf16c | -flto |
|---|---|---|---|---|---|
| native_qlr     (v1) | native   | ✓ | ✓ | – | –  |
| native_qlr_v2       | znver2   | ✓ | ✓ | – | ✓  |
| native_qlr_v3       | znver2   | ✓ | ✓ | ✓ | ✓  |

`ldd` on each `.so`:

```
libstdc++.so.6   (system)
libm.so.6        (system)
libgcc_s.so.1    (system)
libc.so.6        (system)
ld-linux-x86-64.so.2 (system)
```

No RPATH or RUNPATH is present — the modules do not carry non-portable path
information.

### External runtime paths (config/paths.env)
```
NATIVE_MODULE_DIR  -> directory containing the .so files (prebuilt or freshly built)
NATIVE_EXPORT_DIR  -> 4.2 GB flat binary export (doc_vecs.f32, iq_vecs.{f32,f16},
                       pca_mean.f32, pca_components.f32, ep_ids.i32,
                       dev_embs.f32, dev_gt_top10.i32, and neighbour arrays)
FAITH_ROOT         -> ground_truth/dev_small_query_embs.npy (~28 MB)
                    + ground_truth/dev_small_exact_top10_ids.npy (~279 KB)
                    + ep_table/ep_scores.npy (~207 MB)
```

## What is NOT loaded by the executed chain (excluded from the bundle)

* `verify_current_result.py` — audits the frozen artefact directory of the
  paper; not called by any RUN.sh in this package. Retained only for
  provenance sha256 recording in the original SUBMISSION_REPRO_PACKAGE.
* `src/evaluation.py`, `src/search.py`, `src/hnsw_msmarco_query_dnm.ipynb`,
  `src/hnsw_train_context_small.faiss` (436 MB) — not imported by either
  hybrid producer. Excluded.
* `checkpoints/verified_1.092x/*.json` and other benchmark evidence JSONs —
  outputs, not code.
* Snapshotted build.sh copies in `native/src/original_build*.sh` are kept
  for provenance but the executed rebuild path is `build/BUILD_NATIVE.sh`.

## Experiment 7 — 07_faithful_adaptive_depth

### Shell chain
```
benchmarks/07_faithful_adaptive_depth/RUN.sh
├── source config/load_config.sh    (via common/common_env.sh)
├── source common/timestamp_helpers.sh
├── source common/verify_paths.sh
├── verify_faithful_paths           (asserts every external + bundled path)
└── exec taskset -c $CORE  env OUTPUT_ROOT=<sandbox>  $HYBRID_PYTHON -u python/faithful/runner.py
```

### Python chain
```
python/faithful/runner.py
├── numpy 1.26.4              (external — HYBRID env; same as Results 1, 5, 6)
├── faiss 1.9.0               (external — HYBRID env; IndexHNSWFlat + search_level_0 9-arg)
├── threadpoolctl 3.6.0       (external — HYBRID env; wraps timed loops)
├── pandas 3.0.3              (transitive via src/data_loading; parquet reader)
├── from src.data_loading import load_embeddings_from_parquets, l2_normalize
│    └── python/hybrid/src/data_loading.py   (bundled; SHA256 in COPY_MANIFEST.tsv)
├── from src.indexing import load_index
│    └── python/hybrid/src/indexing.py       (bundled; SHA256 in COPY_MANIFEST.tsv)
└── from faithful_qlr import FaithfulQLR, QLRConfig, QLRHandles
     └── python/faithful/faithful_qlr.py     (bundled; byte-identical to source)
```

Note: `joblib` and `scikit-learn` are NOT imported by Experiment 7 — the
faithful runner reads the PCA arrays as raw `.npy` files
(`FAITHFUL_PCA_DIR/pca_mean_1024.npy` and `pca_components_256x1024.npy`),
not the pickled `PCA_QL_DIR/pca_1024_to_256.joblib`.

### External runtime paths (config/paths.env)
```
DEV_QUERY_DIR      -> parquet shards with dev-query embeddings (~43 MB)          (shared with 1, 5, 6)
HYBRID_DOC_INDEX   -> 158 GB TREC-CAsT FAISS HNSW M=32 index                     (shared with 1, 5, 6)
PCA_QL_DIR         -> directory holding train_query_pca256_hnsw.faiss            (only ROUTER_INDEX is used by 7; PCA joblib is not consulted)
ROUTER_INDEX (opt) -> override for train_query_pca256_hnsw.faiss                 (shared with 1, 5, 6)
QLR_ARTIFACT_DIR   -> ep_indices.npy (~32 MB) + ep_distances.npy (~32 MB; s_max) (shared with 1, 5, 6)
EXACT_DIR          -> exact_indices.npy (~546 KB) + exact_scores.npy (~546 KB)   (extended over 1/5/6: exact_scores.npy is asserted only by 7)
FAITHFUL_PCA_DIR   -> pca_mean_1024.npy (~4 KB) + pca_components_256x1024.npy (~1 MB)  (Experiment 7 only)
FAITHFUL_PCA_MEAN (opt), FAITHFUL_PCA_COMPONENTS (opt)   -> individual overrides
FAITHFUL_DOC_INDEX_SHM (opt), FAITHFUL_QUERY_INDEX_SHM (opt)   -> optional /dev/shm fast paths
```

## Benchmark 08 — 08_cachewarmed_treccast (hybrid full TREC-CAsT cache-warmed)

### Shell chain
```
benchmarks/08_cachewarmed_treccast/RUN.sh
├── source config/load_config.sh    (via common/common_env.sh)
├── source common/timestamp_helpers.sh
├── source common/verify_paths.sh
├── verify_cachewarmed_treccast_paths   (hybrid assets; NO native assets)
└── exec taskset -c $CORE  env OUTPUT_ROOT=<sandbox>  $HYBRID_PYTHON -u python/hybrid/cachewarmed_treccast.py \
        --n 6980 --warmup 300 --reps 3 --baseline_ef 64 --cfg_a ... --cfg_b ... --out_dir ... --core $CORE
```

### Python chain
```
python/hybrid/cachewarmed_treccast.py
├── numpy 1.26.4              (external — HYBRID env)
├── faiss 1.9.0              (external — HYBRID env; IndexHNSWFlat + search_level_0)
├── joblib 1.5.3            (external — HYBRID env; loads the pickled PCA)
├── threadpoolctl 3.6.0     (external — HYBRID env; wraps timed loop)
├── pandas 3.0.3            (transitive via src/data_loading; parquet reader)
├── scikit-learn 1.9.0      (required only to unpickle pca_1024_to_256.joblib)
├── from src.data_loading import load_embeddings_from_parquets, l2_normalize
│    └── python/hybrid/src/data_loading.py   (bundled; byte-identical)
├── from src.indexing import load_index
│    └── python/hybrid/src/indexing.py       (bundled; byte-identical)
└── from faithful_qlr import FaithfulQLR, QLRConfig, QLRHandles
     └── python/faithful/faithful_qlr.py     (bundled; byte-identical to Benchmark 07)
```

Imports **no** `native_qlr*` module and reads **no** `NATIVE_EXPORT_DIR` /
`NATIVE_MODULE_DIR` (asserted by `verify_cachewarmed_treccast_import`).

### External runtime paths (config/paths.env — all reused hybrid vars)
```
DEV_QUERY_DIR      -> parquet shards with dev-query embeddings (~43 MB)          (shared with 1, 5, 6, 7)
HYBRID_DOC_INDEX   -> 158 GB TREC-CAsT FAISS HNSW M=32 index                     (shared with 1, 5, 6, 7)
PCA_QL_DIR         -> pca_1024_to_256.joblib (PCA_MODEL) + train_query_pca256_hnsw.faiss (ROUTER_INDEX)
QLR_ARTIFACT_DIR   -> ep_indices.npy + ep_distances.npy (ep_distances = s_max source)
EXACT_DIR          -> exact_indices.npy — acc@10 ground truth
```

Benchmark 08 introduces **no new config variable**. The PCA is read from the
`PCA_MODEL` joblib (`pca.mean_`, `pca.components_`) — identical non-whitening
transform to `rescue_full_run.py::bare_pca` — and fed to `FaithfulQLR`.

## Preprocessing (artifact generation)

The `preprocessing/` directory bundles the producers that generate the hybrid
artifacts consumed above. Full graph in
`preprocessing/manifests/DEPENDENCY_GRAPH.{md,json}`. Summary:

```
preprocessing/scripts/build_index.py            -> HYBRID_DOC_INDEX
preprocessing/scripts/build_query_log_pca.py    -> PCA_MODEL + ROUTER_INDEX (+ qmax)
preprocessing/scripts/build_ep_table.py         -> QLR_ARTIFACT_DIR (ep_indices, ep_distances)
preprocessing/scripts/flat_index_search_acc.py  -> EXACT_DIR (exact ground truth)
```

All four depend only on `preprocessing/src/{data_loading,indexing}.py`
(byte-identical to `python/hybrid/src/`).

## Producer / snapshot equivalence

Every executed producer script under `python/` derives from the original at
its canonical location. `manifests/COPY_MANIFEST.tsv` records the original
SHA256 (byte-identical to the copies in the source
`SUBMISSION_REPRO_PACKAGE/scripts_snapshot/`) and the SHA256 of the
path-adjusted copy. The unified diff between the two is in
`manifests/PATH_ONLY_DIFFS.patch`; the row-by-row behavioural equivalence
tables are in `manifests/BEHAVIOUR_PRESERVATION_REPORT.md`.
