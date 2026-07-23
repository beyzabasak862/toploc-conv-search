# SUBMISSION_READINESS_CHECKLIST.md

Mechanical acceptance-criteria checklist for `toploc_paper_2`. Each row cites
the artefact that proves the claim. Rows 1–20 are inherited unchanged from the
validated `SUBMISSION_CODE_PACKAGE`; rows 22–28 cover the additions specific
to Experiment 7 and this repackage.

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Original project unchanged                                 | PASS | `validation/step12_source_pkg_integrity.log` (SUBMISSION_CODE_PACKAGE = byte-identical) |
| 2 | SUBMISSION_REPRO_PACKAGE byte-identical                    | PASS | `validation/step14_repro_pkg_integrity.log` (unchanged) |
| 3 | All source dependency chains accounted for                 | PASS | `manifests/DEPENDENCY_GRAPH.md` + `manifests/DEPENDENCY_CLASSIFICATION.tsv` |
| 4 | Every required code dependency bundled                     | PASS | `manifests/COPY_MANIFEST.tsv` (12 verbatim + 6 path-adjusted + 2 generated + 3 helpers) |
| 5 | Native source + build instructions bundled                 | PASS | `native/src/*.cpp`, `build/BUILD_NATIVE.sh`, `manifests/NATIVE_BUILD_MANIFEST.md` |
| 6 | No required code loaded from original repository           | PASS | `validation/step10_external_copy.log` (external copy passes CHECK_ONLY with PYTHONPATH cleared) |
| 7 | No bundled symlink escapes package                         | PASS | `validation/step4_symlink_scan.log` |
| 8 | No large datasets/indexes bundled                          | PASS | `validation/step5_size_scan.log` + `manifests/EXTERNAL_RESOURCES.yaml` (all datasets are external) |
| 9 | No generated benchmark outputs bundled                     | PASS | `outputs/` contains only `.gitkeep` — see `manifests/PACKAGE_TREE.txt` |
| 10 | External paths in one config file                         | PASS | `config/paths.env.example` documents every variable |
| 11 | No hard-coded original absolute path in executed code     | PASS | `validation/step3b_forbidden_paths.log` |
| 12 | Algorithm + benchmark behaviour preserved (1–6)           | PASS | `manifests/BEHAVIOUR_PRESERVATION_REPORT.md` — every row IDENTICAL |
| 13 | Seven RUN.sh pass CHECK_ONLY                              | PASS | `validation/step9_check_only_each.log` |
| 14 | RUN_ALL.sh passes CHECK_ONLY (reports 7 workflows)        | PASS | `validation/step9_check_only_each.log` |
| 15 | Shell syntax checks pass                                  | PASS | `validation/step1_bash_n.log` |
| 16 | Python compilation checks pass                            | PASS | `validation/step2_compileall.log` (both interpreters) |
| 17 | Native import/build check passes                          | PASS | VERIFY_CODE_PACKAGE.sh log: `v3 ntotal=500000 iq=808731 dim=1024` |
| 18 | Security + cleanliness pass                               | PASS | `validation/SECURITY_AND_CLEANLINESS_REPORT.md` |
| 19 | Package can be copied outside repo and still validate     | PASS | `validation/step10_external_copy.log` |
| 20 | ZIP + tar.gz produced                                     | PASS | `dist/toploc_paper_2.tar.gz` + `dist/toploc_paper_2.zip` (SHA256 in `dist/*.sha256`) |
| 21 | Every claim supported by a manifest / validation output   | PASS | (this table) |
| 22 | Experiment 7 wrapper + producer + algorithm module bundled | PASS | `benchmarks/07_faithful_adaptive_depth/RUN.sh` + `python/faithful/runner.py` + `python/faithful/faithful_qlr.py` |
| 23 | Experiment 7 faithful imports resolve under HYBRID_PYTHON  | PASS | `validation/step5b_faithful_import.log` (verify_faithful_import) |
| 24 | Experiment 7 adaptive-depth semantics are the paper formula| PASS | `python/faithful/faithful_qlr.py::FaithfulQLR.adaptive_ef` — byte-identical to source; `benchmarks/07_faithful_adaptive_depth/expected_protocol.json` documents the exact formula |
| 25 | Experiment 7 baseline is ordinary HNSW at cfg.ef_default   | PASS | `benchmarks/07_faithful_adaptive_depth/expected_protocol.json` + `python/faithful/runner.py::timed_baseline` |
| 26 | Experiment 7 path-only diff recorded                       | PASS | `manifests/PATH_ONLY_DIFFS.patch` (appended runner.py hunk) |
| 27 | Experiment 7 behaviour equivalence recorded                | PASS | `manifests/BEHAVIOUR_PRESERVATION_REPORT.md` §runner.py — every row IDENTICAL |
| 28 | Faithful authoritative source directory unchanged          | PASS | `validation/step15_faithful_source_integrity.log` (byte-identical) |
| 29 | Benchmark 08 wrapper + hybrid producer bundled             | PASS | `benchmarks/08_cachewarmed_treccast/RUN.sh` + `python/hybrid/cachewarmed_treccast.py` |
| 30 | Benchmark 08 backend is hybrid FAISS (no native modules)  | PASS | `validation/step08_bench08_backend.log` (no `native_qlr` / `NATIVE_EXPORT_DIR` / `NATIVE_MODULE_DIR` in producer) |
| 31 | Benchmark 08 uses HYBRID_DOC_INDEX + all 6,980 dev queries | PASS | `benchmarks/08_cachewarmed_treccast/expected_protocol.json` + producer `assert n_total == 6980` |
| 32 | Benchmark 08 B→Q_A→Q_B order + cache-warmed label preserved | PASS | `manifests/BEHAVIOUR_PRESERVATION_REPORT.md` §cachewarmed_treccast; `RESULT_LABEL.txt` |
| 33 | Benchmark 08 Q_A/Q_B mapped 1:1 from Benchmark 02          | PASS | `benchmarks/08_cachewarmed_treccast/expected_protocol.json` parameter-mapping table |
| 34 | Benchmark 08 CHECK_ONLY passes                             | PASS | `validation/step09_check_only_each.log` |
| 35 | Preprocessing producers + helpers bundled + validated     | PASS | `preprocessing/validation/VALIDATION_REPORT.md` (14 PASS / 0 FAIL) |
| 36 | Preprocessing is path-only vs originals                   | PASS | `preprocessing/manifests/PATH_ONLY_DIFFS.patch` + `COPY_MANIFEST.tsv` |
| 37 | Benchmarks 01–07 producers/wrappers unchanged             | PASS | `validation/step14_bm01_07_unchanged.log` (SHA256 identical) |
| 38 | Original repos + 158 GB index unchanged                   | PASS | `validation/step11_*` … `step13_index_integrity.log` |
