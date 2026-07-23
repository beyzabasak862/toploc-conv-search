# preprocessing / VALIDATION_REPORT.md

Static validation of the bundled preprocessing producers. No preprocessing job
(index build, PCA fit, embedding generation, EP-table build, exact search) was
started. Re-run with `./preprocessing/VERIFY_PREPROCESSING.sh`.

## Result: PASS (14 PASS / 0 WARN / 0 FAIL)

| # | Step | Verdict |
|---|---|---|
| 1  | `bash -n` on shell scripts | PASS |
| 2  | Python `compileall` (HYBRID_PYTHON) | PASS |
| 3+4 | Producer imports + recursive local-dependency resolution | PASS (every `from src.*` symbol resolves for all 4 producers) |
| 5  | Forbidden absolute-path scan (executed code) | PASS (no original-repository path) |
| 6  | Symlink-escape scan | PASS |
| 7  | Oversized-file scan (>50 MB) | PASS |
| 8  | Secret scan | PASS |
| 9  | Config-variable validation | PASS (schema documents all 6 required variables) |
| 10 | Command-printing validation | PASS (no job started) |

## Copied producers (path-adjusted)

| Producer | Original SHA256 | Packaged SHA256 |
|---|---|---|
| `scripts/build_index.py`            | `82b2bc39…415b3` | `8f0221dc…deba` |
| `scripts/build_query_log_pca.py`    | `5adaba37…405db` | `22f9015b…3406` |
| `scripts/build_ep_table.py`         | `c82f5f67…db121` | `fd89b0d7…6dee` |
| `scripts/flat_index_search_acc.py`  | `04b826a4…31dd9` | `0af2173f…c57f` |

Full rows in `manifests/COPY_MANIFEST.tsv`; diffs in
`manifests/PATH_ONLY_DIFFS.patch`.

## Copied helpers (byte-identical)

| Helper | SHA256 (before == after) |
|---|---|
| `src/data_loading.py` | `56eb99e5…a3b69` |
| `src/indexing.py`     | `a96be20e…b0a0be` |
| `src/__init__.py`     | generated (`f686e4a0…ea53a`) |

## Third-party dependencies

`numpy`, `pandas` (parquet), `faiss` (index build/search), `scikit-learn`
(PCA — `build_query_log_pca` only), `joblib` (PCA dump — same). No native
module, tokenizer, model, subprocess, or dynamic import.

## External datasets (never bundled)

* document embedding parquets (`PREPROC_DOC_EMB_DIR`)
* train / query-log embedding parquets (`PREPROC_TRAIN_QUERY_DIR`)
* dev query parquets (`DEV_QUERY_DIR`, reused)
* full TREC-CAsT doc index (`HYBRID_DOC_INDEX`, reused)
* exact flat index (`PREPROC_FLAT_INDEX`)

## Intentionally excluded artifacts

No generated artifact is bundled: no `.faiss` index, no `.joblib`, no `.npy`
EP/exact arrays, no embeddings, no `_generated/` output.

## Missing authoritative producers

* Exact flat index (`treccast_flat.index`) — `MISSING AUTHORITATIVE PRODUCER`.
* Document / query embedding parquet shards — `MISSING AUTHORITATIVE PRODUCER`.
* Extracted faithful PCA arrays (Benchmark 07) — `MISSING AUTHORITATIVE PRODUCER`
  (derived from `pca_1024_to_256.joblib`).
* Native binary export — native-track artifact, out of scope for hybrid
  preprocessing.

No replacement algorithm was written for any missing producer.
