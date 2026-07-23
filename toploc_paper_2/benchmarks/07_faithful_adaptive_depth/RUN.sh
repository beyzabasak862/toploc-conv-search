#!/bin/bash
# benchmarks/07_faithful_adaptive_depth/RUN.sh
#
# Experiment 7: paper-faithful QLR (Algorithm 1) with query-dependent
# adaptive HNSW search depth, compared against ordinary HNSW baseline
# on the TREC-CAsT track (158 GB doc index, 6980 dev queries).
#
# Canonical producer: python/faithful/runner.py (path-only rewrite of the
#   original paper2_faithful_20260718_231400/runner.py — SHA256 pair in
#   manifests/COPY_MANIFEST.tsv, unified diff in manifests/PATH_ONLY_DIFFS.patch).
# Local algorithm import: python/faithful/faithful_qlr.py (byte-identical copy).
#
# CHECK_ONLY=1 ./RUN.sh
#   Verifies faithful paths + Python interpreter + faithful/hybrid Python imports +
#   prints the exact final command AND the adaptive-ef parameters that the
#   producer will use.  Does NOT load the 158 GB index or start a benchmark.
#
# Adaptive-depth semantics (from python/faithful/faithful_qlr.py::FaithfulQLR.adaptive_ef):
#   1. Router returns top-1 similarity s of the historical query I_Q[q, :1].
#   2. If s <  th             -> full HNSW fallback at ef = ef_default.
#   3. If s >  s_max          -> ef' = ef_min           (very confident routed).
#   4. Otherwise              -> ef' = ef_min + (ef_default - ef_min) * (s_max - s) / (s_max - th)
#                                clamped to [ef_min, ef_default].
#   s_max is the 75th percentile of top-1 doc similarity, derived from
#   QLR_ARTIFACT_DIR/ep_distances.npy at run time (paper definition).
#
# See ./expected_protocol.json for the exact baseline + adaptive parameters
# that runner.py bakes in.

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

verify_faithful_paths

# --- output directory ---------------------------------------------------------
if [ -n "${CAMPAIGN_DIR:-}" ]; then
    OUT_DIR="${CAMPAIGN_DIR}/07_faithful_adaptive_depth_$(now_stamp)"
    mkdir -p "$OUT_DIR"
else
    OUT_DIR="$(make_result_dir "$SUBMISSION_CODE_PKG_ROOT" "07_faithful_adaptive_depth")"
fi

# --- construct the exact canonical command -----------------------------------
# runner.py accepts no CLI arguments; every knob is baked into the script:
#   SEED=20260718, N_CALIB=500, N_HOLDOUT=1500, N_WARMUP=30,
#   N_REPS_{CALIB,HOLDOUT,FULL}=(2,3,3),
#   BASELINE_EF_SWEEP=[16,24,32,40,48,64,96,128,160,200],
#   ACC_FLOOR=0.952, ACC_TOL_VS_BASE=0.005, SPEEDUP_TARGET=1.40,
#   12 calibration configs (kp in {5,10,20}, kep in {5,10},
#   th in {0.30,0.40,0.42,0.50}, ef_min in {8,10,16},
#   ef_default in {32,48,64}, search_type in {1,2}).
CANONICAL_CMD=(
    taskset -c "${CORE}"
    env "OUTPUT_ROOT=${OUT_DIR}"
    "$HYBRID_PY" -u "$FAITHFUL_PY_SCRIPT"
)
{
    echo "# canonical command (runner.py takes no CLI args)"
    printf '%q ' "${CANONICAL_CMD[@]}"
    echo
} > "${OUT_DIR}/command.txt"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[07_faithful_adaptive_depth][CHECK_ONLY] paths + interpreter + imports OK"
    echo "[07_faithful_adaptive_depth][CHECK_ONLY] command that WOULD run:"
    cat "${OUT_DIR}/command.txt"
    echo
    echo "[07_faithful_adaptive_depth][CHECK_ONLY] adaptive-ef parameters:"
    echo "  threshold_th          : per-config (0.30, 0.40, 0.42, 0.50)"
    echo "  ef_min                : per-config (8, 10, or 16)"
    echo "  ef_default            : per-config (32, 48, or 64)"
    echo "  s_max                 : computed at run time from"
    echo "                          \${QLR_ARTIFACT_DIR}/ep_distances.npy"
    echo "                          (25th percentile of squared-L2; paper definition)"
    echo "  formula source        : python/faithful/faithful_qlr.py::FaithfulQLR.adaptive_ef"
    echo "  baseline used         : ordinary HNSW at cfg.ef_default (same run, interleaved B-Q)"
    echo "[07_faithful_adaptive_depth][CHECK_ONLY] see expected_protocol.json for full details"
    "$HYBRID_PY" - <<'PY'
import importlib, sys, os
for m in ["numpy","faiss","joblib","threadpoolctl","pandas"]:
    importlib.import_module(m)
pkg = os.environ["SUBMISSION_CODE_PKG_ROOT"]
sys.path.insert(0, os.path.join(pkg, "python", "faithful"))
sys.path.insert(0, os.path.join(pkg, "python", "hybrid"))
import faithful_qlr
print("[07_faithful_adaptive_depth][CHECK_ONLY] hybrid Python imports OK on",
      sys.version.split()[0])
print("[07_faithful_adaptive_depth][CHECK_ONLY] faithful module:",
      faithful_qlr.__file__)
# Compile runner.py without executing it (would load 168 GB index)
with open(os.path.join(pkg, "python", "faithful", "runner.py")) as f:
    compile(f.read(), "runner.py", "exec")
print("[07_faithful_adaptive_depth][CHECK_ONLY] runner.py compiles OK")
PY
    exit 0
fi

STDOUT_LOG="${OUT_DIR}/stdout.log"
STDERR_LOG="${OUT_DIR}/stderr.log"
COMBINED_LOG="${OUT_DIR}/combined.log"

LOAD_BEFORE="$(cat /proc/loadavg 2>/dev/null || echo unavailable)"
emit_environment_snapshot "$OUT_DIR" > /dev/null
emit_pip_versions "$HYBRID_PY" "${OUT_DIR}/pip_versions.txt"

SRC_SHA="$(sha256_or_missing "$FAITHFUL_PY_SCRIPT")"
FAITHFUL_QLR_SHA="$(sha256_or_missing "${SUBMISSION_CODE_PKG_ROOT}/python/faithful/faithful_qlr.py")"

echo "[07_faithful_adaptive_depth] host=$(hostname) load=${LOAD_BEFORE}"
echo "[07_faithful_adaptive_depth] hybrid python:      $HYBRID_PY"
echo "[07_faithful_adaptive_depth] faithful runner:    $FAITHFUL_PY_SCRIPT (sha256=${SRC_SHA})"
echo "[07_faithful_adaptive_depth] faithful_qlr.py:    sha256=${FAITHFUL_QLR_SHA}"
echo "[07_faithful_adaptive_depth] output dir:         $OUT_DIR"
echo "[07_faithful_adaptive_depth] expected wall-clock at low load: several hours (calib + holdout + full 6980 x 3 reps + baseline ef sweep on full 6980 x 2 reps)"

t0=$(date +%s)
set +e
"${CANONICAL_CMD[@]}" > >(tee "$STDOUT_LOG" > "$COMBINED_LOG") 2> >(tee "$STDERR_LOG" >&2)
RC=$?
set -e
t1=$(date +%s)
DUR=$(( t1 - t0 ))

# --- locate the fresh faithful_<TS>/ that runner.py wrote inside OUT_DIR -----
FRESH_DIR="$(find "$OUT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'faithful_*' -printf '%p\n' | sort | tail -1)"

if [ -n "$FRESH_DIR" ] && [ -d "$FRESH_DIR" ]; then
    echo "$FRESH_DIR" > "${OUT_DIR}/canonical_output_source_path.txt"
else
    echo "[07_faithful_adaptive_depth] WARNING: producer did not create a faithful_* directory." >&2
    echo "MISSING" > "${OUT_DIR}/canonical_output_source_path.txt"
fi

# --- wrapper metadata --------------------------------------------------------
"$REPORT_PY" - "$OUT_DIR" "$FRESH_DIR" "$SRC_SHA" "$FAITHFUL_QLR_SHA" "$RC" "$DUR" "$LOAD_BEFORE" <<'PY'
import sys, os, json
out_dir, fresh_dir, src_sha, algo_sha, rc, dur, load_before = sys.argv[1:8]
meta = {
    "result_id": "07_faithful_adaptive_depth",
    "protocol_class": "D. FULL-RUN MEASURED (paper-faithful adaptive-depth TREC-CAsT track)",
    "canonical_script": os.environ.get("FAITHFUL_PY_SCRIPT"),
    "canonical_script_sha256": src_sha,
    "faithful_qlr_module_sha256": algo_sha,
    "python_interpreter": os.environ.get("HYBRID_PY"),
    "cpu_core": int(os.environ.get("CORE", "21")),
    "thread_env": {k: os.environ.get(k) for k in
                    ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS","PYTHONDONTWRITEBYTECODE"]},
    "canonical_output_dir": fresh_dir or "MISSING",
    "package_output_dir": out_dir,
    "return_code": int(rc),
    "wall_clock_seconds": int(dur),
    "loadavg_before": load_before.strip(),
    "loadavg_after": open("/proc/loadavg").read().strip() if os.path.exists("/proc/loadavg") else "unknown",
    "notes": (
        "runner.py accepts no CLI args; the 12 calibration configs + 1500-query "
        "holdout + full 6980 x 3 reps + baseline ef sweep on full 6980 x 2 reps "
        "are baked in.  Adaptive ef' is per-query, computed from router top-1 "
        "similarity s via FaithfulQLR.adaptive_ef.  Baseline in each config is "
        "ordinary HNSW at cfg.ef_default, interleaved B-Q with the QLR run "
        "(same loaded index, same query order, same warmup)."
    ),
    "external_paths_used": {
        "DEV_QUERY_DIR": os.environ.get("DEV_QUERY_DIR"),
        "HYBRID_DOC_INDEX": os.environ.get("HYBRID_DOC_INDEX"),
        "PCA_QL_DIR": os.environ.get("PCA_QL_DIR"),
        "ROUTER_INDEX": os.environ.get("ROUTER_INDEX"),
        "QLR_ARTIFACT_DIR": os.environ.get("QLR_ARTIFACT_DIR"),
        "EXACT_DIR": os.environ.get("EXACT_DIR"),
        "FAITHFUL_PCA_DIR": os.environ.get("FAITHFUL_PCA_DIR"),
        "FAITHFUL_PCA_MEAN": os.environ.get("FAITHFUL_PCA_MEAN"),
        "FAITHFUL_PCA_COMPONENTS": os.environ.get("FAITHFUL_PCA_COMPONENTS"),
        "FAITHFUL_DOC_INDEX_SHM": os.environ.get("FAITHFUL_DOC_INDEX_SHM"),
        "FAITHFUL_QUERY_INDEX_SHM": os.environ.get("FAITHFUL_QUERY_INDEX_SHM"),
    },
}
with open(os.path.join(out_dir, "wrapper_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
PY

echo "[07_faithful_adaptive_depth] DONE rc=${RC} elapsed=${DUR}s output=${OUT_DIR}"
exit "$RC"
