# ORIGINAL_INTEGRITY_REPORT.md

**Result**: `UNCHANGED` (updated 2026-07-23 during the toploc_paper_2 repackage build).

The pre-build SHAs recorded when the original `SUBMISSION_CODE_PACKAGE` was
built (2026-07-22T18:13:01Z) and the post-build re-scan taken while producing
`toploc_paper_2` (2026-07-23) both agree with the baseline snapshots stored in
`manifests/ORIGINAL_SUBMISSION_REPRO_SHA256SUMS.txt`. Every content hash below
was reverified.

## SUBMISSION_REPRO_PACKAGE

* Pre-build manifest:  `manifests/ORIGINAL_SUBMISSION_REPRO_SHA256SUMS.txt` (351 files)
* Post-build manifest: `manifests/POST_SUBMISSION_REPRO_SHA256SUMS.txt`     (351 files)
* diff outcome:       (byte-identical)

## 158 GB TREC-CAsT index

    lrwxrwxrwx 1 1040 1043 77 Jul 22 00:58 /home/toploc1/Datasets/toploc1/indexes/HNSW_indexes/treccast_hnsw_M32.index -> /home/toploc1/Datasets/toploc1/indexes/Snowflake/HNSW/treccast_hnsw_M32.index
    -rw-r--r-- 1 1040 1043 168768858058 May 13 15:26 /home/toploc1/Datasets/toploc1/indexes/Snowflake/HNSW/treccast_hnsw_M32.index

## Canonical producer scripts (unchanged during build)

    a5321206c449523d828c3e6609713a4e015cc1809ff9546e936f9c664a77c2dd  /home/toploc1/Datasets/toploc1/HNSW/msmarco_HNSW/claude_qlr_diagnostics/rescue_full_run.py
    e3e0cf506a803289528169f7ffcfecedd1550c060e470f8e2309f63c82e40a79  /home/toploc1/Datasets/toploc1/HNSW/msmarco_HNSW/claude_qlr_diagnostics/rescue_stage2_accuracy.py
    a95fff52e5dd6fd9c37c55666c7b62bc25ef761d2212c7061e8d9ab5f93325c3  /home/toploc1/Datasets/toploc1/HNSW/msmarco_HNSW/claude_qlr_diagnostics/paper2_final_track/current_verified_proof/verify_current_result.py
    3746e31a514443f9ba7aa3e16e9a24cb9951d600bffa2535989820b46e5176d6  /home/toploc1/Datasets/toploc1/HNSW/msmarco_HNSW/claude_qlr_diagnostics/paper2_final_track/faithful_msmarco_v1_20260719_042528/native_qlr_optimization/20260719_190628/python/benchmark_native.py
    6ece4bfdab46aa0326fcee88389d7e13d1617d693686b91f64708007286c294a  /home/toploc1/Datasets/toploc1/HNSW/msmarco_HNSW/claude_qlr_diagnostics/paper2_final_track/faithful_msmarco_v1_20260719_042528/native_qlr_optimization/20260719_190628/python/canonical_final.py
    2b158d80ae0dd1933c04895ec7d85074a7aed9a26b3d8b323bbc3acd19db9796  /home/toploc1/Datasets/toploc1/HNSW/msmarco_HNSW/claude_qlr_diagnostics/paper2_final_track/faithful_msmarco_v1_20260719_042528/native_qlr_optimization/20260719_190628/python/final_validate.py

Every SHA256 above matches the row for the same original path in `manifests/COPY_MANIFEST.tsv`. Verified 2026-07-22T18:13:01Z (source-package build) and 2026-07-23T02:55Z (toploc_paper_2 build).

## Additional integrity assertions covering the toploc_paper_2 build

* Source `SUBMISSION_CODE_PACKAGE`: 90 non-output files re-hashed; hash-only
  diff against baseline: **0 lines** → byte-identical
  (`validation/step10_source_pkg_integrity.log`).
* `SUBMISSION_REPRO_PACKAGE`: 351 files re-hashed; hash-only diff against
  `manifests/ORIGINAL_SUBMISSION_REPRO_SHA256SUMS.txt`: **0 lines** →
  byte-identical (`validation/step11_repro_pkg_integrity.log`).
* Faithful source directory
  `paper2_final_track/optimization_search/paper2_faithful_20260718_231400/`:
  22 files re-hashed; hash-only diff against baseline: **0 lines** →
  byte-identical (`validation/step12_faithful_source_integrity.log`).
* 158 GB TREC-CAsT index: size and mtime unchanged
  (`validation/step13_index_integrity.log`).
* Experiments 1–6 producer scripts and wrappers: every subtree in the source
  package matches the corresponding subtree in `toploc_paper_2` byte-for-byte
  (`validation/step14_experiments_1_6_unchanged.log`).

## Faithful bundle (Experiment 7)

* `python/faithful/faithful_qlr.py`: `042958a516bcf0fb5c0a73a1ec0d17627fd20e9fc2c36d36c9d4f3769867d6aa` (byte-identical copy of source).
* `python/faithful/runner.py`: original
  `4b3cac280ec441904b36a2a5077a1f92896bbcd7589c547b0710ec3787d81c1d` →
  packaged `623d3a06d5dea6fde560a50410b58ce464454516e5c9ff48f310b99617bd2bb8`
  (path-only rewrite; diff in `manifests/PATH_ONLY_DIFFS.patch`; equivalence
  table in `manifests/BEHAVIOUR_PRESERVATION_REPORT.md`).
* `python/faithful/__init__.py`:
  `918d9e4f9567222b4e1e73c0c9e52dde399b559123ceb74d250698c7dbbad9f8`
  (generated explicit-package marker).
