#!/bin/bash
# benchmarks/04_native_canonical_v3/RUN.sh
#
# Result 4: Native canonical v3 3-run average.
# Canonical producer: python/native/canonical_final.py (path-only rewritten;
#   the original writes to $WS/results/canonical_final.json but the packaged
#   version writes to $OUTPUT_ROOT/canonical_final.json so each iteration is
#   captured without clobbering).
#
# The wrapper invokes canonical_final.py NUM_RUNS times (default 3), copies
# each run's canonical_final.json to a unique canonical_final_run<i>.json
# under the timestamped output dir, and then aggregates via aggregate_results.py.
#
# CHECK_ONLY=1 ./RUN.sh — verifies imports + prints the command, then exits.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_CODE_PKG_ROOT="$(cd "${HERE}/../.." && pwd)"
export SUBMISSION_CODE_PKG_ROOT
source "${SUBMISSION_CODE_PKG_ROOT}/common/common_env.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/timestamp_helpers.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/verify_paths.sh"

verify_native_paths

if [ -n "${CAMPAIGN_DIR:-}" ]; then
    OUT_DIR="${CAMPAIGN_DIR}/04_native_canonical_v3_$(now_stamp)"
    mkdir -p "$OUT_DIR"
else
    OUT_DIR="$(make_result_dir "$SUBMISSION_CODE_PKG_ROOT" "04_native_canonical_v3")"
fi

STDOUT_LOG="${OUT_DIR}/stdout.log"
COMBINED_LOG="${OUT_DIR}/combined.log"

# canonical_final.py hard-codes the output file NAME but reads OUTPUT_ROOT.
CANON_OUT="${OUT_DIR}/canonical_final.json"
NUM_RUNS="${NUM_RUNS:-3}"

CANONICAL_CMD=(
    taskset -c "${CORE}"
    env "OUTPUT_ROOT=${OUT_DIR}"
    "$NATIVE_PY" -u "$NATIVE_CANONICAL_SCRIPT"
)
{
    echo "# canonical command (canonical_final.py takes no CLI args)"
    printf '%q ' "${CANONICAL_CMD[@]}"
    echo
    echo "# invoked ${NUM_RUNS} times consecutively"
} > "${OUT_DIR}/command.txt"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[04_native_canonical_v3][CHECK_ONLY] paths OK"
    echo "[04_native_canonical_v3][CHECK_ONLY] command that WOULD run ${NUM_RUNS} times:"
    cat "${OUT_DIR}/command.txt"
    verify_native_import || true
    exit 0
fi

emit_environment_snapshot "$OUT_DIR" > /dev/null
emit_pip_versions "$NATIVE_PY" "${OUT_DIR}/pip_versions.txt"

SRC_SHA="$(sha256_or_missing "$NATIVE_CANONICAL_SCRIPT")"
SO_V1_SHA="$(sha256_or_missing "$NATIVE_SO_V1")"
SO_V2_SHA="$(sha256_or_missing "$NATIVE_SO_V2")"
SO_V3_SHA="$(sha256_or_missing "$NATIVE_SO_V3")"
LOAD_BEFORE="$(cat /proc/loadavg 2>/dev/null || echo unavailable)"

echo "[04_native_canonical_v3] host=$(hostname) load=${LOAD_BEFORE}"
echo "[04_native_canonical_v3] script: $NATIVE_CANONICAL_SCRIPT (sha256=${SRC_SHA})"
echo "[04_native_canonical_v3] NUM_RUNS=$NUM_RUNS CORE=$CORE"
echo "[04_native_canonical_v3] expected wall-clock at low load: 45-120 minutes total"

RUN_JSONS=()
overall_t0=$(date +%s)
RC=0
for i in $(seq 1 "$NUM_RUNS"); do
    echo
    echo "== 04_native_canonical_v3 run $i / $NUM_RUNS =="
    ri_stamp="$(now_stamp)"
    run_stdout="${OUT_DIR}/run${i}_stdout.log"
    run_stderr="${OUT_DIR}/run${i}_stderr.log"

    t0=$(date +%s)
    set +e
    "${CANONICAL_CMD[@]}" > >(tee "$run_stdout" >> "$COMBINED_LOG") \
                         2> >(tee "$run_stderr" >&2)
    RC=$?
    set -e
    t1=$(date +%s)

    if [ "$RC" -ne 0 ]; then
        echo "[04_native_canonical_v3] run $i FAILED rc=$RC" >&2
        break
    fi
    if [ ! -f "$CANON_OUT" ]; then
        echo "[04_native_canonical_v3] ERROR: canonical_final.json not produced at $CANON_OUT" >&2
        RC=99; break
    fi
    DST="${OUT_DIR}/canonical_final_run${i}_${ri_stamp}.json"
    cp -a "$CANON_OUT" "$DST"
    RUN_JSONS+=("$DST")
    echo "[04_native_canonical_v3] run $i wrote $DST  (elapsed $((t1 - t0))s)"
done
TOTAL_DUR=$(( $(date +%s) - overall_t0 ))

AGG_JSON="${OUT_DIR}/AGGREGATE.json"
if [ "${#RUN_JSONS[@]}" -ge 1 ]; then
    "$NATIVE_PY" "$HERE/aggregate_results.py" "${RUN_JSONS[@]}" --out "$AGG_JSON"
else
    echo "[04_native_canonical_v3] no successful runs; skipping aggregation" >&2
fi

"$REPORT_PY" - "$OUT_DIR" "$SRC_SHA" "$SO_V1_SHA" "$SO_V2_SHA" "$SO_V3_SHA" "$RC" "$TOTAL_DUR" "$LOAD_BEFORE" <<PY
import os, json
out_dir, src_sha, sv1, sv2, sv3, rc, dur, load_before = "${OUT_DIR}", "${SRC_SHA}", "${SO_V1_SHA}", "${SO_V2_SHA}", "${SO_V3_SHA}", ${RC}, ${TOTAL_DUR}, """${LOAD_BEFORE}"""
meta = {
    "result_id": "04_native_canonical_v3",
    "protocol_class": "A. FULL-RUN MEASURED - Faithful MS MARCO-v1 500k (Class J), 3-canonical-run averaged",
    "canonical_script": "${NATIVE_CANONICAL_SCRIPT}",
    "canonical_script_sha256": src_sha,
    "native_so_v1_sha256": sv1,
    "native_so_v2_sha256": sv2,
    "native_so_v3_sha256": sv3,
    "python_interpreter": "${NATIVE_PY}",
    "cpu_core": ${CORE},
    "n_queries": 6980,
    "reps_per_run": 3,
    "canonical_runs": ${NUM_RUNS},
    "warmup_baseline": 200,
    "warmup_qlr": 150,
    "output_dir": out_dir,
    "final_return_code": int(rc),
    "wall_clock_seconds": int(dur),
    "loadavg_before": load_before.strip(),
    "loadavg_after": open("/proc/loadavg").read().strip() if os.path.exists("/proc/loadavg") else "unknown",
}
with open(os.path.join(out_dir, "wrapper_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
PY

echo "[04_native_canonical_v3] DONE rc=${RC} elapsed=${TOTAL_DUR}s output=${OUT_DIR}"
exit "$RC"
