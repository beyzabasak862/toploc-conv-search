#!/bin/bash
# benchmarks/08_cachewarmed_treccast/RUN.sh
#
# Benchmark 08: FULL TREC-CAsT DOCUMENT CORPUS / FULL MS MARCO V1 DEV.SMALL
# QUERY WORKLOAD / HYBRID FAISS QLR / MULTI-CONFIG CACHE-WARMED B-Q_A-Q_B
# POSITION-2 OBSERVATION.
#
# Canonical producer: python/hybrid/cachewarmed_treccast.py (hybrid FAISS).
#   * doc index  : HYBRID_DOC_INDEX (158 GB TREC-CAsT HNSW, ~38.6M docs)
#   * queries    : DEV_QUERY_DIR (ALL 6,980 MS MARCO v1 dev.small)
#   * PCA        : PCA_MODEL joblib (same asset as Benchmarks 1/5/6)
#   * router     : ROUTER_INDEX (train_query_pca256_hnsw.faiss)
#   * EP table   : QLR_ARTIFACT_DIR/ep_{indices,distances}.npy
#   * exact GT   : EXACT_DIR/exact_indices.npy
#   * QLR algo   : python/faithful/faithful_qlr.py::FaithfulQLR (Benchmark 07)
#
# Benchmark 02 contributes ONLY the cache-warmed protocol + reporting style.
# This wrapper NEVER touches native modules, NATIVE_EXPORT_DIR, or the 500k
# native document corpus. The search backend and corpus are hybrid TREC-CAsT.
#
# Per-query call order (fixed query order 0..N-1):
#   B  -> ordinary hybrid HNSW baseline (efSearch = --baseline_ef, default 64)
#   Q_A -> first hybrid QLR configuration
#   Q_B -> target hybrid QLR configuration (position-2, CACHE-WARMED)
#
# CHECK_ONLY=1 ./RUN.sh
#   Verifies every dependency, prints B/Q_A/Q_B configs, corpus and query
#   count, prints the exact command, and exits BEFORE loading the 158 GB index.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_CODE_PKG_ROOT="$(cd "${HERE}/../.." && pwd)"
export SUBMISSION_CODE_PKG_ROOT
# shellcheck source=../../common/common_env.sh
source "${SUBMISSION_CODE_PKG_ROOT}/common/common_env.sh"
# shellcheck source=../../common/timestamp_helpers.sh
source "${SUBMISSION_CODE_PKG_ROOT}/common/timestamp_helpers.sh"
# shellcheck source=../../common/verify_paths.sh
source "${SUBMISSION_CODE_PKG_ROOT}/common/verify_paths.sh"

verify_cachewarmed_treccast_paths

# --- fixed protocol defaults (Benchmark-02 cache-warmed philosophy) -----------
N_ARG="${N:-6980}"
WARMUP_ARG="${WARMUP:-300}"
REPS_ARG="${REPS:-3}"
BASELINE_EF_ARG="${BASELINE_EF:-64}"      # ordinary hybrid HNSW baseline (Benchmarks 1/5/6)

# --- exact hybrid Q_A / Q_B configs (mapped 1:1 from Benchmark 02) -------------
#   Native 02 Q_A: kp=20,kep=10,th=0.35,ef=112,ef_min=10,rEF=16,backend=v2
#   Native 02 Q_B: kp=20,kep=10,th=0.32,ef=112,ef_min=10,rEF=12,backend=v2
#   backend=v2 (pooled beam) -> search_type=2 (st=2) in the hybrid FaithfulQLR.
CFG_A="kp=20,kep=10,th=0.35,ef=112,ef_min=10,rEF=16,st=2"
CFG_B="kp=20,kep=10,th=0.32,ef=112,ef_min=10,rEF=12,st=2"

# --- output directory ---------------------------------------------------------
if [ -n "${CAMPAIGN_DIR:-}" ]; then
    OUT_DIR="${CAMPAIGN_DIR}/08_cachewarmed_treccast_$(now_stamp)"
    mkdir -p "$OUT_DIR"
else
    OUT_DIR="$(make_result_dir "$SUBMISSION_CODE_PKG_ROOT" "08_cachewarmed_treccast")"
fi

CANONICAL_CMD=(
    taskset -c "${CORE}"
    env "OUTPUT_ROOT=${OUT_DIR}"
    "$HYBRID_PY" -u "$CACHEWARMED_TRECCAST_SCRIPT"
        --n "$N_ARG"
        --warmup "$WARMUP_ARG"
        --reps "$REPS_ARG"
        --baseline_ef "$BASELINE_EF_ARG"
        --cfg_a "$CFG_A"
        --cfg_b "$CFG_B"
        --out_dir "$OUT_DIR"
        --core "$CORE"
)
{
    echo "# canonical command (cachewarmed_treccast.py — hybrid FAISS TREC-CAsT)"
    printf '%q ' "${CANONICAL_CMD[@]}"
    echo
} > "${OUT_DIR}/command.txt"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[08_cachewarmed_treccast][CHECK_ONLY] paths + interpreter + imports OK"
    echo "[08_cachewarmed_treccast][CHECK_ONLY] document corpus : full TREC-CAsT (HYBRID_DOC_INDEX)"
    echo "[08_cachewarmed_treccast][CHECK_ONLY]   index path     : ${HYBRID_DOC_INDEX}"
    echo "[08_cachewarmed_treccast][CHECK_ONLY] query workload  : full MS MARCO v1 dev.small"
    echo "[08_cachewarmed_treccast][CHECK_ONLY]   expected count : 6980 (from ${DEV_QUERY_DIR})"
    echo "[08_cachewarmed_treccast][CHECK_ONLY] backend         : hybrid FAISS (FaithfulQLR); NO native modules"
    echo "[08_cachewarmed_treccast][CHECK_ONLY] per-query order : B -> Q_A -> Q_B  (fixed query order 0..N-1)"
    echo "[08_cachewarmed_treccast][CHECK_ONLY]   B (baseline)  : ordinary hybrid HNSW  efSearch=${BASELINE_EF_ARG}  topk=10"
    echo "[08_cachewarmed_treccast][CHECK_ONLY]   Q_A           : ${CFG_A}"
    echo "[08_cachewarmed_treccast][CHECK_ONLY]   Q_B           : ${CFG_B}   <- position 2, CACHE-WARMED"
    echo "[08_cachewarmed_treccast][CHECK_ONLY]   s_max source  : ${QLR_ARTIFACT_DIR}/ep_distances.npy (25th pct sq-L2; paper defn)"
    echo "[08_cachewarmed_treccast][CHECK_ONLY]   N=${N_ARG} WARMUP=${WARMUP_ARG} REPS=${REPS_ARG} CORE=${CORE}"
    echo "[08_cachewarmed_treccast][CHECK_ONLY] command that WOULD run:"
    cat "${OUT_DIR}/command.txt"
    verify_cachewarmed_treccast_import
    exit 0
fi

STDOUT_LOG="${OUT_DIR}/stdout.log"
STDERR_LOG="${OUT_DIR}/stderr.log"
COMBINED_LOG="${OUT_DIR}/combined.log"

LOAD_BEFORE="$(cat /proc/loadavg 2>/dev/null || echo unavailable)"
emit_environment_snapshot "$OUT_DIR" > /dev/null
emit_pip_versions "$HYBRID_PY" "${OUT_DIR}/pip_versions.txt"

SRC_SHA="$(sha256_or_missing "$CACHEWARMED_TRECCAST_SCRIPT")"
FAITHFUL_QLR_SHA="$(sha256_or_missing "${SUBMISSION_CODE_PKG_ROOT}/python/faithful/faithful_qlr.py")"

echo "[08_cachewarmed_treccast] host=$(hostname) load=${LOAD_BEFORE}"
echo "[08_cachewarmed_treccast] producer:     $CACHEWARMED_TRECCAST_SCRIPT (sha256=${SRC_SHA})"
echo "[08_cachewarmed_treccast] faithful_qlr: sha256=${FAITHFUL_QLR_SHA}"
echo "[08_cachewarmed_treccast] doc index:    $HYBRID_DOC_INDEX (full TREC-CAsT)"
echo "[08_cachewarmed_treccast] B: ordinary hybrid HNSW ef=${BASELINE_EF_ARG}"
echo "[08_cachewarmed_treccast] Q_A: $CFG_A"
echo "[08_cachewarmed_treccast] Q_B: $CFG_B  <- position 2, cache-warmed"
echo "[08_cachewarmed_treccast] N=$N_ARG WARMUP=$WARMUP_ARG REPS=$REPS_ARG CORE=$CORE"
echo "[08_cachewarmed_treccast] expected wall-clock at low load: several hours (158 GB index, 6980 x 3 reps x 3 methods)"

t0=$(date +%s)
set +e
"${CANONICAL_CMD[@]}" > >(tee "$STDOUT_LOG" > "$COMBINED_LOG") 2> >(tee "$STDERR_LOG" >&2)
RC=$?
set -e
t1=$(date +%s)
DUR=$(( t1 - t0 ))

FRESH_DIR="$(find "$OUT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'cachewarmed_treccast_*' -printf '%p\n' | sort | tail -1)"
if [ -n "$FRESH_DIR" ] && [ -d "$FRESH_DIR" ]; then
    echo "$FRESH_DIR" > "${OUT_DIR}/canonical_output_source_path.txt"
else
    echo "MISSING" > "${OUT_DIR}/canonical_output_source_path.txt"
fi

cat > "${OUT_DIR}/RESULT_LABEL.txt" <<'EOF'
============================================================
RESULT LABEL (mandatory when reporting Q_B):
    FULL TREC-CAsT DOCUMENT CORPUS / FULL MS MARCO V1 DEV.SMALL
    QUERY WORKLOAD / HYBRID FAISS QLR /
    MULTI-CONFIG CACHE-WARMED B-Q_A-Q_B POSITION-2 OBSERVATION

Q_B (position 2) inherits Q_A's freshly warmed PCA weights, router
index, EP rows and doc-index caches; the same-run baseline does not
share that exact cache state. Never present Q_B as an isolated
speedup or a paper-comparable result.
============================================================
EOF

"$REPORT_PY" - "$OUT_DIR" "$FRESH_DIR" "$SRC_SHA" "$FAITHFUL_QLR_SHA" "$RC" "$DUR" "$LOAD_BEFORE" \
    "$CFG_A" "$CFG_B" "$BASELINE_EF_ARG" <<'PY'
import sys, os, json
(out_dir, fresh_dir, src_sha, algo_sha, rc, dur, load_before,
 cfg_a, cfg_b, baseline_ef) = sys.argv[1:11]
meta = {
    "result_id": "08_cachewarmed_treccast",
    "protocol_class": "F. HYBRID FULL TREC-CAsT MULTI-CONFIG CACHE-WARMED (B-Q_A-Q_B)",
    "protocol_label": ("FULL TREC-CAsT DOCUMENT CORPUS / FULL MS MARCO V1 DEV.SMALL "
                       "QUERY WORKLOAD / HYBRID FAISS QLR / MULTI-CONFIG CACHE-WARMED "
                       "B-Q_A-Q_B POSITION-2 OBSERVATION"),
    "backend": "hybrid_faiss_treccast",
    "document_corpus": "full TREC-CAsT (HYBRID_DOC_INDEX)",
    "query_workload": "full MS MARCO v1 dev.small (all 6,980)",
    "canonical_script": os.environ.get("CACHEWARMED_TRECCAST_SCRIPT"),
    "canonical_script_sha256": src_sha,
    "faithful_qlr_module_sha256": algo_sha,
    "python_interpreter": os.environ.get("HYBRID_PY"),
    "cpu_core": int(os.environ.get("CORE", "21")),
    "thread_env": {k: os.environ.get(k) for k in
                    ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS","PYTHONDONTWRITEBYTECODE"]},
    "baseline": {"backend": "hybrid_hnsw", "ef": int(baseline_ef), "topk": 10},
    "cfg_a": cfg_a,
    "cfg_b": cfg_b,
    "per_query_call_order": ["baseline(hybrid HNSW, ef=%s)" % baseline_ef, "Q_A", "Q_B"],
    "query_order": "fixed 0..N-1",
    "one_process": True,
    "return_code": int(rc),
    "wall_clock_seconds": int(dur),
    "loadavg_before": load_before.strip(),
    "loadavg_after": open("/proc/loadavg").read().strip() if os.path.exists("/proc/loadavg") else "unknown",
    "canonical_output_dir": fresh_dir or "MISSING",
    "package_output_dir": out_dir,
    "uses_native_modules": False,
    "uses_native_export_dir": False,
    "external_paths_used": {
        "DEV_QUERY_DIR": os.environ.get("DEV_QUERY_DIR"),
        "HYBRID_DOC_INDEX": os.environ.get("HYBRID_DOC_INDEX"),
        "PCA_QL_DIR": os.environ.get("PCA_QL_DIR"),
        "PCA_MODEL": os.environ.get("PCA_MODEL"),
        "ROUTER_INDEX": os.environ.get("ROUTER_INDEX"),
        "QLR_ARTIFACT_DIR": os.environ.get("QLR_ARTIFACT_DIR"),
        "EXACT_DIR": os.environ.get("EXACT_DIR"),
    },
    "warning": "Q_B (position 2) is a CACHE-WARMED observation. Do not present as isolated.",
}
with open(os.path.join(out_dir, "wrapper_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
PY

echo "[08_cachewarmed_treccast] DONE rc=${RC} elapsed=${DUR}s output=${OUT_DIR}"
exit "$RC"
