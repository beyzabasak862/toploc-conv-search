#!/bin/bash
# benchmarks/01_safe_hybrid/RUN.sh
#
# Result 1: Safe hybrid QLR full 6980-query benchmark.
# Canonical producer: python/hybrid/rescue_full_run.py (path-only rewrite of
#   the original claude_qlr_diagnostics/rescue_full_run.py — SHA256 pair in
#   manifests/COPY_MANIFEST.tsv, unified diff in manifests/PATH_ONLY_DIFFS.patch).
#
# CHECK_ONLY=1 ./RUN.sh
#   Verifies paths + Python interpreter + Python import graph + prints the
#   exact final command. Does NOT load the 158 GB index or start a benchmark.
#
# The same rescue_full_run.py execution produces the aggressive endpoint used
# by Result 5; the RUN_ALL.sh orchestrator exports SHARE_HYBRID_OUTPUT so
# Result 5 reuses this output.

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

verify_hybrid_paths

# --- output directory ---------------------------------------------------------
if [ -n "${CAMPAIGN_DIR:-}" ]; then
    OUT_DIR="${CAMPAIGN_DIR}/01_safe_hybrid_$(now_stamp)"
    mkdir -p "$OUT_DIR"
else
    OUT_DIR="$(make_result_dir "$SUBMISSION_CODE_PKG_ROOT" "01_safe_hybrid")"
fi

# --- construct the exact canonical command -----------------------------------
# rescue_full_run.py accepts no CLI arguments; every knob is baked into the
# script (RS=0.5, EF_DEFAULT=64, ROUTER_EF=16, NPROBE=3, SEEDED_EFS=[32,16],
# N_REPS=2, N_WARMUP=50, seed_mode=recompute_l2, SEED=20260717, ACC_FLOOR=0.952).
CANONICAL_CMD=(
    taskset -c "${CORE}"
    env "OUTPUT_ROOT=${OUT_DIR}"
    "$HYBRID_PY" -u "$HYBRID_PY_SCRIPT"
)
{
    echo "# canonical command (rescue_full_run.py takes no CLI args)"
    printf '%q ' "${CANONICAL_CMD[@]}"
    echo
} > "${OUT_DIR}/command.txt"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[01_safe_hybrid][CHECK_ONLY] paths + interpreter + imports OK"
    echo "[01_safe_hybrid][CHECK_ONLY] command that WOULD run:"
    cat "${OUT_DIR}/command.txt"
    "$HYBRID_PY" - <<'PY'
import importlib, sys
for m in ["numpy","faiss","joblib","threadpoolctl","pandas"]:
    importlib.import_module(m)
print("[01_safe_hybrid][CHECK_ONLY] hybrid Python imports OK on", sys.version.split()[0])
PY
    exit 0
fi

STDOUT_LOG="${OUT_DIR}/stdout.log"
STDERR_LOG="${OUT_DIR}/stderr.log"
COMBINED_LOG="${OUT_DIR}/combined.log"

LOAD_BEFORE="$(cat /proc/loadavg 2>/dev/null || echo unavailable)"
emit_environment_snapshot "$OUT_DIR" > /dev/null
emit_pip_versions "$HYBRID_PY" "${OUT_DIR}/pip_versions.txt"

SRC_SHA="$(sha256_or_missing "$HYBRID_PY_SCRIPT")"

echo "[01_safe_hybrid] host=$(hostname) load=${LOAD_BEFORE}"
echo "[01_safe_hybrid] hybrid python: $HYBRID_PY"
echo "[01_safe_hybrid] hybrid script: $HYBRID_PY_SCRIPT (sha256=${SRC_SHA})"
echo "[01_safe_hybrid] output dir:    $OUT_DIR"
echo "[01_safe_hybrid] expected wall-clock at low load: 1-4 hours"

t0=$(date +%s)
set +e
"${CANONICAL_CMD[@]}" > >(tee "$STDOUT_LOG" > "$COMBINED_LOG") 2> >(tee "$STDERR_LOG" >&2)
RC=$?
set -e
t1=$(date +%s)
DUR=$(( t1 - t0 ))

# --- locate the fresh full_<TS>/ that rescue_full_run.py wrote inside OUT_DIR --
FRESH_DIR="$(find "$OUT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'full_*' -printf '%p\n' | sort | tail -1)"

if [ -n "$FRESH_DIR" ] && [ -d "$FRESH_DIR" ]; then
    echo "$FRESH_DIR" > "${OUT_DIR}/canonical_output_source_path.txt"
else
    echo "[01_safe_hybrid] WARNING: producer did not create a full_* directory." >&2
    echo "MISSING" > "${OUT_DIR}/canonical_output_source_path.txt"
fi

# --- wrapper metadata --------------------------------------------------------
"$REPORT_PY" - "$OUT_DIR" "$FRESH_DIR" "$SRC_SHA" "$RC" "$DUR" "$LOAD_BEFORE" <<'PY'
import sys, os, json
out_dir, fresh_dir, src_sha, rc, dur, load_before = sys.argv[1:7]
meta = {
    "result_id": "01_safe_hybrid",
    "protocol_class": "A. FULL-RUN MEASURED (hybrid TREC-CAsT track)",
    "canonical_script": os.environ.get("HYBRID_PY_SCRIPT"),
    "canonical_script_sha256": src_sha,
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
        "rescue_full_run.py accepts no CLI args; config is baked in "
        "(RS=0.5, EF_DEFAULT=64, ROUTER_EF=16, NPROBE=3, "
        "SEEDED_EFS=[32,16], N_REPS=2, N_WARMUP=50, seed_mode=recompute_l2). "
        "This SAME run also produces the aggressive (ef16) endpoint used by "
        "Result 5 (05_aggressive_hybrid)."
    ),
    "external_paths_used": {
        "DEV_QUERY_DIR": os.environ.get("DEV_QUERY_DIR"),
        "HYBRID_DOC_INDEX": os.environ.get("HYBRID_DOC_INDEX"),
        "PCA_QL_DIR": os.environ.get("PCA_QL_DIR"),
        "QLR_ARTIFACT_DIR": os.environ.get("QLR_ARTIFACT_DIR"),
        "EXACT_DIR": os.environ.get("EXACT_DIR"),
    },
}
with open(os.path.join(out_dir, "wrapper_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
PY

echo "[01_safe_hybrid] DONE rc=${RC} elapsed=${DUR}s output=${OUT_DIR}"
exit "$RC"
