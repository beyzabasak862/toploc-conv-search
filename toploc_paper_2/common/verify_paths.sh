#!/bin/bash
# common/verify_paths.sh — read-only sanity checks. Source; do not execute.
# Every message names the environment variable the caller must set in
# config/paths.env.

assert_file() {
    if [ ! -f "$1" ]; then
        echo "[verify_paths] MISSING FILE: $2 (env $3) = $1" >&2
        return 1
    fi
}
assert_dir() {
    if [ ! -d "$1" ]; then
        echo "[verify_paths] MISSING DIR: $2 (env $3) = $1" >&2
        return 1
    fi
}
assert_executable() {
    if [ ! -x "$1" ]; then
        echo "[verify_paths] NOT EXECUTABLE: $2 (env $3) = $1" >&2
        return 1
    fi
}

# --- hybrid track paths (Results 1, 5, 6) -----------------------------------
verify_hybrid_paths() {
    local rc=0
    assert_file "$HYBRID_PY_SCRIPT"     "producer script"        SUBMISSION_CODE_PKG_ROOT || rc=1
    assert_executable "$HYBRID_PY"       "hybrid python"          HYBRID_PYTHON            || rc=1
    assert_executable "$REPORT_PY"       "report python"          REPORT_PYTHON            || rc=1
    assert_dir  "$DEV_QUERY_DIR"         "dev query parquet dir"  DEV_QUERY_DIR            || rc=1
    assert_file "$HYBRID_DOC_INDEX"      "158 GB TREC-CAsT index" HYBRID_DOC_INDEX         || rc=1
    assert_dir  "$PCA_QL_DIR"            "PCA + router dir"       PCA_QL_DIR               || rc=1
    assert_dir  "$QLR_ARTIFACT_DIR"      "QLR EP table dir"       QLR_ARTIFACT_DIR         || rc=1
    assert_dir  "$EXACT_DIR"             "exact top-10 dir"       EXACT_DIR                || rc=1
    # optional overrides
    local pca_model="${PCA_MODEL:-${PCA_QL_DIR}/pca_1024_to_256.joblib}"
    local router="${ROUTER_INDEX:-${PCA_QL_DIR}/train_query_pca256_hnsw.faiss}"
    assert_file "$pca_model"             "PCA joblib"             PCA_MODEL                || rc=1
    assert_file "$router"                "router HNSW index"      ROUTER_INDEX             || rc=1
    return $rc
}

# --- stage-2 also needs qmax --------------------------------------------------
verify_hybrid_stage2_paths() {
    verify_hybrid_paths || return 1
    local qmax="${PCA_QMAX:-${PCA_QL_DIR}/qmax_pca256.npy}"
    if [ ! -f "$qmax" ]; then
        echo "[verify_paths] MISSING FILE: qmax (env PCA_QMAX) = $qmax" >&2
        return 1
    fi
    return 0
}

# --- native track paths (Results 2, 3, 4) -----------------------------------
verify_native_paths() {
    local rc=0
    assert_file "$NATIVE_BENCHMARK_SCRIPT"    "benchmark_native.py"    SUBMISSION_CODE_PKG_ROOT || rc=1
    assert_file "$NATIVE_CANONICAL_SCRIPT"    "canonical_final.py"     SUBMISSION_CODE_PKG_ROOT || rc=1
    assert_file "$NATIVE_INTERLEAVED_SCRIPT"  "final_validate.py"      SUBMISSION_CODE_PKG_ROOT || rc=1
    assert_executable "$NATIVE_PY"            "native python"          NATIVE_PYTHON            || rc=1
    assert_dir  "$NATIVE_MODULE_DIR"          "native module dir"      NATIVE_MODULE_DIR        || rc=1
    assert_dir  "$NATIVE_EXPORT_DIR"          "native_export dir"      NATIVE_EXPORT_DIR        || rc=1
    assert_dir  "$FAITH_ROOT"                 "faithful root"          FAITH_ROOT               || rc=1
    assert_file "$FAITH_ROOT/ground_truth/dev_small_query_embs.npy"     "dev query embs"    FAITH_ROOT || rc=1
    assert_file "$FAITH_ROOT/ground_truth/dev_small_exact_top10_ids.npy" "dev GT ids"       FAITH_ROOT || rc=1
    assert_file "$FAITH_ROOT/ep_table/ep_scores.npy"                    "ep_scores.npy"     FAITH_ROOT || rc=1
    assert_file "$NATIVE_SO_V1"               ".so v1"                 NATIVE_MODULE_DIR        || rc=1
    assert_file "$NATIVE_SO_V2"               ".so v2"                 NATIVE_MODULE_DIR        || rc=1
    assert_file "$NATIVE_SO_V3"               ".so v3"                 NATIVE_MODULE_DIR        || rc=1
    return $rc
}

# --- faithful adaptive-depth track paths (Experiment 7) --------------------
verify_faithful_paths() {
    local rc=0
    # Producer + bundled algorithm module
    assert_file "$FAITHFUL_PY_SCRIPT"                        "faithful runner.py"    SUBMISSION_CODE_PKG_ROOT || rc=1
    assert_file "${SUBMISSION_CODE_PKG_ROOT}/python/faithful/faithful_qlr.py" \
                                                              "faithful_qlr.py"       SUBMISSION_CODE_PKG_ROOT || rc=1
    # Uses the same hybrid interpreter (needs FAISS 1.9 + numpy + threadpoolctl + pandas)
    assert_executable "$HYBRID_PY"                            "hybrid python"         HYBRID_PYTHON            || rc=1
    # Shared external assets already required by the hybrid track
    assert_dir  "$DEV_QUERY_DIR"                              "dev query parquet dir" DEV_QUERY_DIR            || rc=1
    assert_file "$HYBRID_DOC_INDEX"                           "158 GB TREC-CAsT idx"  HYBRID_DOC_INDEX         || rc=1
    assert_dir  "$PCA_QL_DIR"                                 "PCA + router dir"      PCA_QL_DIR               || rc=1
    local router="${ROUTER_INDEX:-${PCA_QL_DIR}/train_query_pca256_hnsw.faiss}"
    assert_file "$router"                                     "router HNSW index"     ROUTER_INDEX             || rc=1
    assert_dir  "$QLR_ARTIFACT_DIR"                           "QLR EP table dir"      QLR_ARTIFACT_DIR         || rc=1
    assert_dir  "$EXACT_DIR"                                  "exact top-10 dir"      EXACT_DIR                || rc=1
    # Faithful-specific PCA arrays (paper-faithful shape: (1024,) and (256, 1024))
    local pca_dir="${FAITHFUL_PCA_DIR}"
    if [ -z "$pca_dir" ]; then
        echo "[verify_paths] MISSING VAR: FAITHFUL_PCA_DIR is not set" >&2
        rc=1
    else
        assert_dir  "$pca_dir"                                "faithful PCA dir"      FAITHFUL_PCA_DIR         || rc=1
        local mean="${FAITHFUL_PCA_MEAN:-${pca_dir}/pca_mean_1024.npy}"
        local comps="${FAITHFUL_PCA_COMPONENTS:-${pca_dir}/pca_components_256x1024.npy}"
        assert_file "$mean"                                   "faithful pca_mean"     FAITHFUL_PCA_MEAN        || rc=1
        assert_file "$comps"                                  "faithful pca_comps"    FAITHFUL_PCA_COMPONENTS  || rc=1
    fi
    return $rc
}

# --- faithful import test under HYBRID_PY -----------------------------------
verify_faithful_import() {
    if [ ! -x "$HYBRID_PY" ]; then
        echo "[verify_paths] HYBRID_PYTHON not executable" >&2
        return 1
    fi
    "$HYBRID_PY" - <<'PY'
import sys, os
pkg = os.environ["SUBMISSION_CODE_PKG_ROOT"]
sys.path.insert(0, os.path.join(pkg, "python", "faithful"))
sys.path.insert(0, os.path.join(pkg, "python", "hybrid"))
try:
    import faithful_qlr
    from faithful_qlr import FaithfulQLR, QLRConfig, QLRHandles
    # runner imports faiss and threadpoolctl at module load
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_probe_runner", os.path.join(pkg, "python", "faithful", "runner.py"))
    # Only compile, do not execute main (which would load 168 GB index)
    with open(spec.origin) as f:
        code = compile(f.read(), spec.origin, "exec")
    print("[verify_paths] faithful import OK:", faithful_qlr.__file__)
    print("[verify_paths] runner.py compiles OK:", spec.origin)
except Exception as e:
    print("[verify_paths] FAITHFUL IMPORT FAIL:", type(e).__name__, e, file=sys.stderr)
    sys.exit(2)
PY
}

# --- Benchmark 08: hybrid full TREC-CAsT cache-warmed track -----------------
# Same external assets as the hybrid track (Benchmarks 1/5/6) PLUS the bundled
# faithful algorithm module. Uses the joblib PCA (PCA_MODEL) — NOT the faithful
# extracted arrays and NOT any native asset.
verify_cachewarmed_treccast_paths() {
    local rc=0
    # Producer + bundled algorithm module
    assert_file "$CACHEWARMED_TRECCAST_SCRIPT"               "cachewarmed_treccast.py" SUBMISSION_CODE_PKG_ROOT || rc=1
    assert_file "${SUBMISSION_CODE_PKG_ROOT}/python/faithful/faithful_qlr.py" \
                                                              "faithful_qlr.py"       SUBMISSION_CODE_PKG_ROOT || rc=1
    assert_file "${SUBMISSION_CODE_PKG_ROOT}/python/hybrid/src/data_loading.py" \
                                                              "src/data_loading.py"   SUBMISSION_CODE_PKG_ROOT || rc=1
    assert_file "${SUBMISSION_CODE_PKG_ROOT}/python/hybrid/src/indexing.py" \
                                                              "src/indexing.py"       SUBMISSION_CODE_PKG_ROOT || rc=1
    # Interpreter (FAISS 1.9 + numpy + joblib + threadpoolctl + pandas + sklearn)
    assert_executable "$HYBRID_PY"                            "hybrid python"         HYBRID_PYTHON            || rc=1
    assert_executable "$REPORT_PY"                            "report python"         REPORT_PYTHON            || rc=1
    # Hybrid external assets (identical to Benchmarks 1/5/6)
    assert_dir  "$DEV_QUERY_DIR"                              "dev query parquet dir" DEV_QUERY_DIR            || rc=1
    assert_file "$HYBRID_DOC_INDEX"                           "158 GB TREC-CAsT idx"  HYBRID_DOC_INDEX         || rc=1
    assert_dir  "$PCA_QL_DIR"                                 "PCA + router dir"      PCA_QL_DIR               || rc=1
    local pca_model="${PCA_MODEL:-${PCA_QL_DIR}/pca_1024_to_256.joblib}"
    assert_file "$pca_model"                                  "PCA joblib"            PCA_MODEL                || rc=1
    local router="${ROUTER_INDEX:-${PCA_QL_DIR}/train_query_pca256_hnsw.faiss}"
    assert_file "$router"                                     "router HNSW index"     ROUTER_INDEX             || rc=1
    assert_dir  "$QLR_ARTIFACT_DIR"                           "QLR EP table dir"      QLR_ARTIFACT_DIR         || rc=1
    assert_file "$QLR_ARTIFACT_DIR/ep_indices.npy"            "ep_indices.npy"        QLR_ARTIFACT_DIR         || rc=1
    assert_file "$QLR_ARTIFACT_DIR/ep_distances.npy"          "ep_distances.npy"      QLR_ARTIFACT_DIR         || rc=1
    assert_dir  "$EXACT_DIR"                                  "exact top-10 dir"      EXACT_DIR                || rc=1
    assert_file "$EXACT_DIR/exact_indices.npy"                "exact_indices.npy"     EXACT_DIR                || rc=1
    return $rc
}

# --- Benchmark 08 import test under HYBRID_PY (no native modules loaded) -----
verify_cachewarmed_treccast_import() {
    if [ ! -x "$HYBRID_PY" ]; then
        echo "[verify_paths] HYBRID_PYTHON not executable" >&2
        return 1
    fi
    "$HYBRID_PY" - <<'PY'
import sys, os
pkg = os.environ["SUBMISSION_CODE_PKG_ROOT"]
sys.path.insert(0, os.path.join(pkg, "python", "faithful"))
sys.path.insert(0, os.path.join(pkg, "python", "hybrid"))
try:
    from src.data_loading import load_embeddings_from_parquets, l2_normalize
    from src.indexing import load_index
    from faithful_qlr import FaithfulQLR, QLRConfig, QLRHandles
    # Compile the Benchmark 08 producer without executing main (would load 158 GB).
    prod = os.path.join(pkg, "python", "hybrid", "cachewarmed_treccast.py")
    with open(prod) as f:
        compile(f.read(), prod, "exec")
    # Guard: the producer must NOT reference native modules or native export.
    txt = open(prod).read()
    for forbidden in ("native_qlr", "NATIVE_EXPORT_DIR", "NATIVE_MODULE_DIR"):
        assert forbidden not in txt, f"forbidden native reference in producer: {forbidden}"
    print("[verify_paths] Benchmark 08 import OK (hybrid FAISS; no native refs)")
except Exception as e:
    print("[verify_paths] BENCH08 IMPORT FAIL:", type(e).__name__, e, file=sys.stderr)
    sys.exit(2)
PY
}

# --- import test: try to load the three native .so files under NATIVE_PY ----
verify_native_import() {
    if [ ! -x "$NATIVE_PY" ]; then
        echo "[verify_paths] NATIVE_PYTHON not executable" >&2
        return 1
    fi
    "$NATIVE_PY" - <<PY
import sys, os
sys.path.insert(0, os.environ["NATIVE_MODULE_DIR"])
try:
    import native_qlr, native_qlr_v2, native_qlr_v3
    ix = native_qlr_v3.NativeQLR(os.environ["NATIVE_EXPORT_DIR"])
    print("[verify_paths] native import OK. v3 ntotal=", ix.doc_ntotal(),
          "iq=", ix.iq_ntotal(), "dim=", ix.dim(), "ep=", ix.doc_entry_point())
except Exception as e:
    print("[verify_paths] IMPORT FAIL:", type(e).__name__, e, file=sys.stderr)
    sys.exit(2)
PY
}
