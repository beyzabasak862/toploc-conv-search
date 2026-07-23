#!/bin/bash
# benchmarks/03_native_equal_accuracy/RUN.sh
#
# Result 3: Native equal-accuracy checkpoint (~1.092x) replay.
# Canonical producer: python/native/benchmark_native.py.
#
# CLI (verified from source):
#   --n           default 6980
#   --warmup      default 100  (we set 200 per task spec)
#   --reps        default 3
#   --baseline_ef default sweep [10 20 30 40 50 64 80 100 130]
#   --out         default $OUTPUT_ROOT/native_500k.json
#
# CHECK_ONLY=1 ./RUN.sh — verifies paths, imports the native modules, prints
#   the exact final command, exits before running the benchmark.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_CODE_PKG_ROOT="$(cd "${HERE}/../.." && pwd)"
export SUBMISSION_CODE_PKG_ROOT
source "${SUBMISSION_CODE_PKG_ROOT}/common/common_env.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/timestamp_helpers.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/verify_paths.sh"

verify_native_paths

if [ -n "${CAMPAIGN_DIR:-}" ]; then
    OUT_DIR="${CAMPAIGN_DIR}/03_native_equal_accuracy_$(now_stamp)"
    mkdir -p "$OUT_DIR"
else
    OUT_DIR="$(make_result_dir "$SUBMISSION_CODE_PKG_ROOT" "03_native_equal_accuracy")"
fi

STDOUT_LOG="${OUT_DIR}/stdout.log"
STDERR_LOG="${OUT_DIR}/stderr.log"
COMBINED_LOG="${OUT_DIR}/combined.log"
OUT_JSON="${OUT_DIR}/native_500k_replay.json"

N_ARG="${N:-6980}"
WARMUP_ARG="${WARMUP:-200}"
REPS_ARG="${REPS:-3}"

CANONICAL_CMD=(
    taskset -c "${CORE}"
    "$NATIVE_PY" -u "$NATIVE_BENCHMARK_SCRIPT"
        --n "$N_ARG"
        --warmup "$WARMUP_ARG"
        --reps "$REPS_ARG"
        --out "$OUT_JSON"
)
{
    echo "# canonical command (benchmark_native.py --out set to our sandbox)"
    printf '%q ' "${CANONICAL_CMD[@]}"
    echo
} > "${OUT_DIR}/command.txt"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[03_native_equal_accuracy][CHECK_ONLY] paths OK"
    echo "[03_native_equal_accuracy][CHECK_ONLY] command that WOULD run:"
    cat "${OUT_DIR}/command.txt"
    verify_native_import || true
    exit 0
fi

emit_environment_snapshot "$OUT_DIR" > /dev/null
emit_pip_versions "$NATIVE_PY" "${OUT_DIR}/pip_versions.txt"

SRC_SHA="$(sha256_or_missing "$NATIVE_BENCHMARK_SCRIPT")"
SO_V1_SHA="$(sha256_or_missing "$NATIVE_SO_V1")"

LOAD_BEFORE="$(cat /proc/loadavg 2>/dev/null || echo unavailable)"
echo "[03_native_equal_accuracy] host=$(hostname) load=${LOAD_BEFORE}"
echo "[03_native_equal_accuracy] script: $NATIVE_BENCHMARK_SCRIPT (sha256=${SRC_SHA})"
echo "[03_native_equal_accuracy] N=$N_ARG WARMUP=$WARMUP_ARG REPS=$REPS_ARG CORE=$CORE"
echo "[03_native_equal_accuracy] expected wall-clock at low load: 15-45 minutes"

t0=$(date +%s)
set +e
"${CANONICAL_CMD[@]}" > >(tee "$STDOUT_LOG" > "$COMBINED_LOG") 2> >(tee "$STDERR_LOG" >&2)
RC=$?
set -e
t1=$(date +%s)
DUR=$(( t1 - t0 ))

"$REPORT_PY" - "$OUT_DIR" "$OUT_JSON" "$SRC_SHA" "$SO_V1_SHA" "$RC" "$DUR" "$LOAD_BEFORE" <<'PY'
import sys, os, json
out_dir, out_json, src_sha, so_v1_sha, rc, dur, load_before = sys.argv[1:8]
meta = {
    "result_id": "03_native_equal_accuracy",
    "protocol_class": "C. ISOLATED / PAPER-COMPARABLE - Faithful MS MARCO-v1 500k (Class J)",
    "canonical_script": os.environ.get("NATIVE_BENCHMARK_SCRIPT"),
    "canonical_script_sha256": src_sha,
    "native_so_v1_sha256": so_v1_sha,
    "python_interpreter": os.environ.get("NATIVE_PY"),
    "cpu_core": int(os.environ.get("CORE", "21")),
    "thread_env": {k: os.environ.get(k) for k in
                    ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS","PYTHONDONTWRITEBYTECODE"]},
    "n_queries": int(os.environ.get("N", "6980")),
    "warmup": int(os.environ.get("WARMUP", "200")),
    "reps": int(os.environ.get("REPS", "3")),
    "output_json": out_json,
    "return_code": int(rc),
    "wall_clock_seconds": int(dur),
    "loadavg_before": load_before.strip(),
    "loadavg_after": open("/proc/loadavg").read().strip() if os.path.exists("/proc/loadavg") else "unknown",
    "notes": "Replay of the 1.092x equal-accuracy checkpoint using benchmark_native.py.",
}
with open(os.path.join(out_dir, "wrapper_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
PY

echo "[03_native_equal_accuracy] DONE rc=${RC} elapsed=${DUR}s output=${OUT_DIR}"
exit "$RC"
