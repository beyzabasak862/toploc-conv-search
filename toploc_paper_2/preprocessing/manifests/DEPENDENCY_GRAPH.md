# preprocessing / DEPENDENCY_GRAPH.md

Dependency chain for every bundled preprocessing producer. All four producers
share the same two local helpers (`src/data_loading.py`, `src/indexing.py`),
which are byte-identical to the copies bundled under `python/hybrid/src/`.

## Common import chain (all producers)

```
preprocessing/scripts/<producer>.py
├── PROJECT_ROOT = Path(__file__).resolve().parents[1]   -> preprocessing/
├── sys.path.append(PROJECT_ROOT)
├── from src.data_loading import load_embeddings_from_parquets, l2_normalize
│    └── preprocessing/src/data_loading.py    (byte-identical; needs pandas, numpy)
└── from src.indexing import ...
     └── preprocessing/src/indexing.py         (byte-identical; needs faiss, numpy)
```

Third-party packages: `numpy`, `pandas` (parquet), `faiss` (index build/search),
`scikit-learn` (PCA — build_query_log_pca only), `joblib` (PCA dump — same).
No native module, no tokenizer/model, no subprocess, no dynamic import.

## build_index.py — full document HNSW index

```
inputs : PREPROC_DOC_EMB_DIR                (document embedding parquet shards)
uses   : src.data_loading.load_embeddings_from_parquets + l2_normalize
         src.indexing.build_hnsw_index + save_index + save_ids
params : M=32, efConstruction=500, efSearch=64, metric=ip, normalize=True
output : ${PREPROC_OUTPUT_ROOT}/index_artifacts/train_query_full_hnsw.faiss
         ${PREPROC_OUTPUT_ROOT}/index_artifacts/train_query_full_ids.npy
consumer: the full TREC-CAsT document index (HYBRID_DOC_INDEX) consumed by
          Benchmarks 01, 05, 06, 07, 08 and by build_ep_table.py
note   : the original writes the index under the name train_query_full_hnsw.faiss;
         the deployed benchmark index is treccast_hnsw_M32.index — same builder,
         M=32 HNSW over the normalized document embeddings.
```

## build_query_log_pca.py — PCA model + router index + qmax

```
inputs : PREPROC_TRAIN_QUERY_DIR            (train / query-log embedding parquets)
uses   : src.data_loading.* ; src.indexing.build_hnsw_index/save_index/save_ids
         sklearn.decomposition.PCA ; joblib
params : N_COMPONENTS=256, svd_solver=randomized, random_state=42, normalize=True,
         router HNSW M=32/efC=500/efS=64/metric=ip, QMAX_QUANTILE=0.75
output : ${PREPROC_OUTPUT_ROOT}/querylog_pca256/
           train_query_pca256.npy, train_query_pca256_ids.npy,
           train_query_pca256_hnsw.faiss  (router I_Q index),
           pca_1024_to_256.joblib          (PCA model — mean_, components_),
           qmax_pca256.npy, pca_meta.json
consumer: PCA_MODEL + ROUTER_INDEX consumed by Benchmarks 01, 05, 06, 07, 08;
          the extracted pca_mean_1024.npy / pca_components_256x1024.npy used by
          Benchmark 07 derive from this joblib.
```

## build_ep_table.py — EP table (entry points per historical query)

```
inputs : PREPROC_TRAIN_QUERY_DIR            (train / query-log embedding parquets)
         HYBRID_DOC_INDEX                   (the full TREC-CAsT doc HNSW index)
uses   : src.data_loading.* ; src.indexing.load_index
params : TOPK_EP=10, normalize=True
output : ${PREPROC_OUTPUT_ROOT}/qlr_artifacts_full/
           ep_indices.npy   (int32 [nQL, 10]),
           ep_distances.npy (float32 [nQL, 10]) — s_max source,
           train_query_ids.npy
consumer: QLR_ARTIFACT_DIR consumed by Benchmarks 01, 05, 06, 07, 08
```

## flat_index_search_acc.py — exact ground truth (acc@10 reference)

```
inputs : DEV_QUERY_DIR                      (reused hybrid var; 6,980 dev queries)
         PREPROC_FLAT_INDEX                 (exact flat index, e.g. treccast_flat.index)
uses   : src.data_loading.* ; src.indexing.load_index ; pandas
params : TOPK=10, BATCH_SIZE=512, normalize=True
output : ${PREPROC_OUTPUT_ROOT}/exact_results_full/
           exact_scores.npy, exact_indices.npy (int64 [6980,10]),
           dev_query_ids.npy, exact_meta.csv
consumer: EXACT_DIR consumed by Benchmarks 01, 05, 06, 07, 08
```

## Recommended execution order

1. `build_index.py`          — build the full document HNSW index (very high RAM;
                                the deployed index is ~158 GB).                [REQUIRED, EXPENSIVE, HIGH-RAM]
2. `build_query_log_pca.py`  — fit PCA + build the router I_Q index + qmax.     [REQUIRED]
3. `build_ep_table.py`       — search the doc index with the query log to get
                                the EP table. Depends on step 1's doc index.    [REQUIRED, EXPENSIVE]
4. `flat_index_search_acc.py`— exact top-10 ground truth from the flat index.
                                Independent of steps 1–3, but needs the exact
                                flat index (PREPROC_FLAT_INDEX) which is NOT
                                produced by any bundled script.                 [REQUIRED, EXPENSIVE]

Steps 2 and 4 are independent of each other. Step 3 requires step 1's doc index
(or the deployed HYBRID_DOC_INDEX). Step 4 additionally requires an exact flat
index that must be built separately (see "Missing authoritative producers").

## Missing authoritative producers

* **Exact flat index** (`treccast_flat.index`, consumed by
  flat_index_search_acc.py as `PREPROC_FLAT_INDEX`): no builder script exists
  under `msmarco_HNSW/scripts/`. → `MISSING AUTHORITATIVE PRODUCER`.
* **Document / query embedding generation** (the Snowflake Arctic-Embed parquet
  shards feeding `PREPROC_DOC_EMB_DIR` / `PREPROC_TRAIN_QUERY_DIR` /
  `DEV_QUERY_DIR`): produced by an upstream embedding pipeline not present in
  `scripts/`. → `MISSING AUTHORITATIVE PRODUCER`.
* **Native binary export** (`native_export/`): this is a native-track artifact
  (Benchmarks 02/03/04); no hybrid preprocessing producer builds it, and it is
  out of scope for the hybrid TREC-CAsT preprocessing bundled here.
  → not applicable to hybrid preprocessing.
* **Extracted faithful PCA arrays** (`pca_mean_1024.npy`,
  `pca_components_256x1024.npy` used by Benchmark 07): derived from the
  `pca_1024_to_256.joblib` produced by `build_query_log_pca.py`; the extraction
  step itself has no standalone script under `scripts/`.
  → `MISSING AUTHORITATIVE PRODUCER` (trivially `pca.mean_` / `pca.components_`).
