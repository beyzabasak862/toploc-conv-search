# SECURITY_AND_CLEANLINESS_REPORT.md

Scan of `toploc_paper_2` after the Benchmark 08 + preprocessing extension.

| Check | Result | Evidence |
|---|---|---|
| Symlink-escape scan (no target leaves the package) | PASS | `step05_07_scans.log` |
| Oversized file scan (no file > 50 MB, excl dist/) | PASS | `step05_07_scans.log` |
| Secret scan (private keys / tokens / passwords) | PASS | `step05_07_scans.log` |
| Forbidden-path scan in executed code | PASS | `step04_forbidden_paths.log` |
| Benchmark 08 imports no native module / native export | PASS | `step08_bench08_backend.log` |
| No `__pycache__` / `*.pyc` shipped | PASS | cleaned after every compile/CHECK_ONLY |
| No generated preprocessing artifacts bundled | PASS | `preprocessing/` is code-only; `preprocessing/_generated/` is git-ignored |
| No credentials in `config/paths.env` (local paths only) | PASS | manual review |

## Allowed, non-dependency occurrences of the original repo path

The string `/home/toploc1/Datasets/toploc1/HNSW/msmarco_HNSW` appears only in:

1. `native/src/original_build{,_v2,_v3}.sh` — preserved reference build recipes
   (`native_reference` in `manifests/COPY_MANIFEST.tsv`), never invoked by
   executed code.
2. `VERIFY_CODE_PACKAGE.sh` and `preprocessing/VERIFY_PREPROCESSING.sh` — the
   `FORBIDDEN='…'` scanner-pattern definitions (the validators search *for* that
   path; the literal is the needle, not a dependency).

Both are documented and expected. No executed producer, wrapper, helper, or
config file contains an original-repository path.
