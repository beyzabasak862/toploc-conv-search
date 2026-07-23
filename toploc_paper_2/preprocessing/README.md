# toploc_paper_2 / preprocessing

Portable copies of the **authoritative code used to generate the benchmark
artifacts** consumed by the hybrid TREC-CAsT benchmarks (01, 05, 06, 07, 08).
This directory preserves the producers so a reviewer can regenerate the
document index, PCA/router assets, EP table, and exact ground truth from
source — it does **not** ship any generated artifact.

Nothing here is executed during packaging. Run a producer only when you
deliberately want to regenerate an artifact.

## Layout

```
preprocessing/
├── README.md                    (this file)
├── VERIFY_PREPROCESSING.sh      static validator (never starts a job)
├── scripts/                     the 4 path-adjusted producers
├── src/                         data_loading.py + indexing.py (byte-identical) + __init__.py
├── manifests/                   dependency + copy + path-rewrite evidence
└── validation/                  VERIFY_PREPROCESSING.sh output + VALIDATION_REPORT.md
```

All producers share the two local helpers in `src/`, byte-identical to the
copies under `python/hybrid/src/`. Producers use the same central configuration
(`config/paths.env`) as the benchmarks — there is no second path system.

## Producers

| Script | Purpose | Artifact | Expected input | Output (under `PREPROC_OUTPUT_ROOT`) | Format | Approx size | Benchmark consumers | Config variables | Example command |
|---|---|---|---|---|---|---|---|---|---|
| `scripts/build_index.py` | build full document HNSW index | TREC-CAsT document HNSW index | doc embedding parquet shards (`PREPROC_DOC_EMB_DIR`) | `index_artifacts/train_query_full_hnsw.faiss` + `_ids.npy` | FAISS `IndexHNSWFlat` (M=32, ip) | ~158 GB (deployed) | `HYBRID_DOC_INDEX` (01,05,06,07,08); `build_ep_table.py` | `PREPROC_DOC_EMB_DIR`, `PREPROC_OUTPUT_ROOT` | `python scripts/build_index.py` |
| `scripts/build_query_log_pca.py` | fit PCA + build router I_Q index + qmax | PCA model, router index, qmax | train/query-log embedding parquets (`PREPROC_TRAIN_QUERY_DIR`) | `querylog_pca256/{train_query_pca256.npy, _ids.npy, train_query_pca256_hnsw.faiss, pca_1024_to_256.joblib, qmax_pca256.npy, pca_meta.json}` | npy / FAISS / joblib / json | ~1.8 GB | `PCA_MODEL` + `ROUTER_INDEX` (01,05,06,07,08); faithful PCA arrays (07) | `PREPROC_TRAIN_QUERY_DIR`, `PREPROC_OUTPUT_ROOT` | `python scripts/build_query_log_pca.py` |
| `scripts/build_ep_table.py` | search doc index with query log → EP table | EP table (`ep_indices`, `ep_distances`) | train/query-log parquets (`PREPROC_TRAIN_QUERY_DIR`) + doc index (`HYBRID_DOC_INDEX`) | `qlr_artifacts_full/{ep_indices.npy, ep_distances.npy, train_query_ids.npy}` | npy (int32 / float32) | ~69 MB | `QLR_ARTIFACT_DIR` (01,05,06,07,08); `ep_distances.npy` = s_max source | `PREPROC_TRAIN_QUERY_DIR`, `HYBRID_DOC_INDEX`, `PREPROC_OUTPUT_ROOT` | `python scripts/build_ep_table.py` |
| `scripts/flat_index_search_acc.py` | exact flat-index top-10 search | exact ground truth (acc@10) | dev queries (`DEV_QUERY_DIR`) + exact flat index (`PREPROC_FLAT_INDEX`) | `exact_results_full/{exact_scores.npy, exact_indices.npy, dev_query_ids.npy, exact_meta.csv}` | npy / csv | ~1 MB | `EXACT_DIR` (01,05,06,07,08) | `DEV_QUERY_DIR`, `PREPROC_FLAT_INDEX`, `PREPROC_OUTPUT_ROOT` | `python scripts/flat_index_search_acc.py` |

## Recommended execution order

1. **`build_index.py`** — full document HNSW index. **REQUIRED · EXPENSIVE · HIGH-RAM** (the deployed index is ~158 GB).
2. **`build_query_log_pca.py`** — PCA model + router index + qmax. **REQUIRED.**
3. **`build_ep_table.py`** — EP table; needs step 1's doc index (or the deployed `HYBRID_DOC_INDEX`). **REQUIRED · EXPENSIVE.**
4. **`flat_index_search_acc.py`** — exact ground truth; needs an exact flat index (`PREPROC_FLAT_INDEX`). **REQUIRED · EXPENSIVE.**

**Stage categories**

* **Required** — all four produce artifacts every hybrid benchmark consumes.
* **Optional branches** — step 2 and step 4 are independent of each other.
* **Corpus-specific** — `build_index` (document corpus) and `build_ep_table`
  (document + query-log corpus) are TREC-CAsT-corpus specific.
* **Expensive** — steps 1, 3, 4 scan the full corpus / query log.
* **GPU** — none of the four require a GPU (FAISS-CPU + scikit-learn).
* **High-RAM** — `build_index` (holds all document vectors + builds the HNSW graph).

## Config variables

Add these to `config/paths.env` (documented in `config/paths.env.example`):

| Variable | Purpose |
|---|---|
| `PREPROC_OUTPUT_ROOT` | where regenerated artifacts are written (default `preprocessing/_generated/`) |
| `PREPROC_DOC_EMB_DIR` | document embedding parquet shards (input to `build_index`) |
| `PREPROC_TRAIN_QUERY_DIR` | train / query-log embedding parquets (input to `build_query_log_pca` + `build_ep_table`) |
| `PREPROC_FLAT_INDEX` | exact flat index (`treccast_flat.index`) for exact ground truth |
| `HYBRID_DOC_INDEX` (reused) | full TREC-CAsT doc index, searched by `build_ep_table` |
| `DEV_QUERY_DIR` (reused) | 6,980 dev queries, searched by `flat_index_search_acc` |

Producers write into per-artifact subdirectories of `PREPROC_OUTPUT_ROOT`
(`index_artifacts/`, `querylog_pca256/`, `qlr_artifacts_full/`,
`exact_results_full/`), so regeneration never clobbers the benchmark input
directories unless you point `PREPROC_OUTPUT_ROOT` at them yourself.

## Missing authoritative producers

* **Exact flat index** (`treccast_flat.index`, consumed by
  `flat_index_search_acc.py`): **MISSING AUTHORITATIVE PRODUCER** — no builder
  exists under `msmarco_HNSW/scripts/`.
* **Document / query embedding parquet shards**: **MISSING AUTHORITATIVE
  PRODUCER** — produced by an upstream embedding pipeline not present in
  `scripts/`.
* **Extracted faithful PCA arrays** (`pca_mean_1024.npy`,
  `pca_components_256x1024.npy` for Benchmark 07): **MISSING AUTHORITATIVE
  PRODUCER** — they are `pca.mean_` / `pca.components_` extracted from the
  `pca_1024_to_256.joblib` produced by `build_query_log_pca.py`; no standalone
  extraction script exists.
* **Native binary export** (`native_export/`): a native-track artifact
  (Benchmarks 02/03/04), out of scope for hybrid TREC-CAsT preprocessing.

No replacement algorithm has been written for any missing producer.

## Validate (never runs a job)

```bash
./preprocessing/VERIFY_PREPROCESSING.sh
```

It runs `bash -n`, `compileall`, producer import + local-dependency resolution,
a forbidden-path scan, symlink/oversize/secret scans, config-variable schema
checks, and prints the exact per-producer commands — without starting any
index build, PCA fit, EP-table build, or exact search.

## Preservation

Every producer is a **path-only** rewrite of its original under
`msmarco_HNSW/scripts/` — the FAISS index type, HNSW `M`, `efConstruction`,
`efSearch`, PCA dimensions, `random_state`, normalization, top-k, batching,
metric, document order, and output schema are unchanged. Evidence:
`manifests/COPY_MANIFEST.tsv`, `manifests/PATH_REWRITE_MAP.tsv`,
`manifests/PATH_ONLY_DIFFS.patch`, `manifests/DEPENDENCY_GRAPH.{md,json}`.
