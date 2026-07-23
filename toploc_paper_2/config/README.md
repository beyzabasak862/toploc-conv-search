# config/ — one place to point the package at your files

Every path the package needs at runtime lives in `paths.env`. Nothing in the
Python or shell code has an absolute filesystem path baked in — copy the
example, edit it once, and every wrapper picks up the values automatically.

## First-time setup

```bash
cp config/paths.env.example config/paths.env
${EDITOR:-vi} config/paths.env
./VERIFY_CODE_PACKAGE.sh
```

`VERIFY_CODE_PACKAGE.sh` reports every missing or unreadable path with the
variable name that controls it. Fix any red rows, re-run, and only proceed to
`RUN_ALL.sh` when the verifier is clean.

## What is in `paths.env`

| Category | Variable | Purpose |
|---|---|---|
| Interpreter | `HYBRID_PYTHON`   | FAISS 1.9 / Py 3.11 env for Results 1, 5, 6 |
| Interpreter | `NATIVE_PYTHON`   | Py 3.10 env matching the native `.so` ABI (Results 2, 3, 4) |
| Interpreter | `REPORT_PYTHON`   | numpy-only helper interpreter used by wrappers |
| Hybrid data | `HYBRID_DOC_INDEX` | 158 GB TREC-CAsT HNSW index |
| Hybrid data | `DEV_QUERY_DIR`    | dev-query parquet shards |
| Hybrid data | `PCA_QL_DIR`       | query-log PCA + FAISS router directory |
| Hybrid data | `PCA_MODEL` (opt)  | override for `pca_1024_to_256.joblib` |
| Hybrid data | `ROUTER_INDEX` (opt) | override for `train_query_pca256_hnsw.faiss` |
| Hybrid data | `PCA_QMAX` (opt)   | override for `qmax_pca256.npy` |
| Hybrid data | `QLR_ARTIFACT_DIR` | `ep_indices.npy`, `ep_distances.npy` |
| Hybrid data | `EXACT_DIR`        | `exact_indices.npy` (ground-truth top-10) |
| Native data | `FAITH_ROOT`       | faithful MS MARCO-v1 workspace (ground-truth + ep-table) |
| Native code | `NATIVE_MODULE_DIR`| directory containing the `native_qlr*.so` files |
| Native data | `NATIVE_EXPORT_DIR`| 4.2 GB flat binary export used by the native backends |
| Faithful   | `FAITHFUL_PCA_DIR` | directory with `pca_mean_1024.npy` + `pca_components_256x1024.npy` (Experiment 7) |
| Faithful   | `FAITHFUL_PCA_MEAN` (opt)       | override for `pca_mean_1024.npy` |
| Faithful   | `FAITHFUL_PCA_COMPONENTS` (opt) | override for `pca_components_256x1024.npy` |
| Faithful   | `FAITHFUL_DOC_INDEX_SHM` (opt)  | optional `/dev/shm/...` fast-path for the doc index |
| Faithful   | `FAITHFUL_QUERY_INDEX_SHM` (opt)| optional `/dev/shm/...` fast-path for the router index |
| Preproc    | `PREPROC_OUTPUT_ROOT` | where regenerated preprocessing artifacts go (default: `preprocessing/_generated/`) |
| Preproc    | `PREPROC_DOC_EMB_DIR` | document embedding parquets (input to `build_index.py`) |
| Preproc    | `PREPROC_TRAIN_QUERY_DIR` | train/query-log embedding parquets (`build_query_log_pca.py`, `build_ep_table.py`) |
| Preproc    | `PREPROC_FLAT_INDEX` | exact flat index for ground truth (`flat_index_search_acc.py`) |
| Runtime    | `OUTPUT_ROOT`      | where fresh outputs go (default: `<package>/outputs/`) |
| Runtime    | `CORE`             | CPU core index for `taskset` (default: 21) |

Benchmark 08 (`08_cachewarmed_treccast`) introduces **no new variable** — it
reuses the hybrid variables (`HYBRID_PYTHON`, `HYBRID_DOC_INDEX`,
`DEV_QUERY_DIR`, `PCA_QL_DIR`/`PCA_MODEL`/`ROUTER_INDEX`, `QLR_ARTIFACT_DIR`,
`EXACT_DIR`). The `PREPROC_*` variables are only needed if you regenerate
artifacts via `preprocessing/`; the benchmarks never read them.

See `manifests/EXTERNAL_RESOURCES.yaml` for the expected file format,
dimensions, and approximate size of each external item.

## How the values propagate

* `config/load_config.sh` is sourced by every shell wrapper. It exports every
  key from `paths.env` and then sets the fixed single-thread env vars
  (`OMP_NUM_THREADS=1`, etc.).
* Every Python producer reads the same environment variables via
  `os.environ` at import time and fails with an actionable message if a
  required variable is unset.

## What NOT to change

* Don't rename variables. The producers look them up by exact name.
* Don't relax `OMP_NUM_THREADS=1` etc. — timings become non-comparable.
* Don't change `CORE` unless you understand the thermal characteristics of
  your machine; the reference measurements were taken on core 21.

The example values in `paths.env.example` are the placeholders the supervisor
edits — they intentionally look like `/path/to/…` to prevent an accidental
run against a wrong dataset.
