# VALIDATION_REPORT.md — Benchmark 08 + preprocessing extension

Validation of `toploc_paper_2` after adding Benchmark 08
(`08_cachewarmed_treccast`) and the `preprocessing/` directory. Each row cites
the log that produced the verdict.

| # | Step | Verdict | Log |
|---|---|---|---|
| 1  | `bash -n` on every shell script (19/19)                              | PASS | `step01_bash_n.log` |
| 2  | `compileall` under HYBRID_PYTHON + NATIVE_PYTHON (python + benchmarks + preprocessing) | PASS | `step02_compileall.log` |
| 3  | Benchmark 08 backend proof — hybrid FAISS, no native refs in executable code | PASS | `step08_bench08_backend.log` |
| 4  | Forbidden absolute-path scan in executed code (0 real matches)       | PASS | `step04_forbidden_paths.log` |
| 5  | Symlink-escape scan                                                  | PASS | `step05_07_scans.log` |
| 6  | Oversized-file scan (>50 MB)                                         | PASS | `step05_07_scans.log` |
| 7  | Secret scan                                                          | PASS | `step05_07_scans.log` |
| 8  | Benchmark 08 CHECK_ONLY                                              | PASS | `step09_check_only_each.log` |
| 9  | CHECK_ONLY for hybrid wrappers 01, 05, 06, 07, 08                    | PASS | `step09_check_only_each.log` |
| 10 | CHECK_ONLY for native wrappers 02, 03, 04                            | PASS | `step09_native_check_only.log` |
| 11 | External-copy independence (Benchmark 08 + preprocessing, PYTHONPATH unset) | PASS | `step10_external_copy.log` |
| 12 | Benchmarks 01–07 producers + wrappers byte-identical to baseline     | PASS | `step14_bm01_07_unchanged.log` |
| 13 | Source `SUBMISSION_CODE_PACKAGE` unchanged (0 hash diffs)            | PASS | `step11_13_upstream_integrity.log` |
| 14 | `SUBMISSION_REPRO_PACKAGE` unchanged (0 hash diffs)                  | PASS | `step11_13_upstream_integrity.log` |
| 15 | Faithful authoritative source dir unchanged (0 hash diffs)          | PASS | `step11_13_upstream_integrity.log` |
| 16 | Preprocessing authoritative source unchanged (0 hash diffs)         | PASS | `step11_13_upstream_integrity.log` |
| 17 | 158 GB TREC-CAsT index size + mtime unchanged                       | PASS | `step11_13_upstream_integrity.log` |
| 18 | Preprocessing static validation (14 PASS / 0 FAIL; no job started)  | PASS | `../preprocessing/validation/VALIDATION_REPORT.md` |

## Benchmark 08 — verification highlights

* **Backend is hybrid FAISS, not native.** `python/hybrid/cachewarmed_treccast.py`
  imports `FaithfulQLR` + `src.data_loading` + `src.indexing`, reads
  `HYBRID_DOC_INDEX`, and contains **zero** `native_qlr` / `NATIVE_EXPORT_DIR` /
  `NATIVE_MODULE_DIR` references (asserted by `verify_cachewarmed_treccast_import`).
* **Corpus + workload.** Full TREC-CAsT document corpus (`HYBRID_DOC_INDEX`,
  ~38.6M docs) × all 6,980 MS MARCO v1 dev.small queries (producer asserts
  `n_total == 6980`).
* **B → Q_A → Q_B order** and the cache-warmed position-2 label are preserved
  (`RESULT_LABEL.txt`; `BEHAVIOUR_PRESERVATION_REPORT.md` §cachewarmed_treccast).
* **Q_A/Q_B mapped 1:1** from Benchmark 02 (see
  `benchmarks/08_cachewarmed_treccast/expected_protocol.json`).

## Not re-executed

No full benchmark (01–08) and no preprocessing job was run. `CHECK_ONLY=1`
proves the path/import/interpreter/config chain resolves; it does not reproduce
speedup numbers or generate any artifact.
