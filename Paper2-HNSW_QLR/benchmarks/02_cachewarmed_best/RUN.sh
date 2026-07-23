#!/bin/bash
# benchmarks/02_cachewarmed_best/RUN.sh
#
# Result 2: MULTI-CONFIG CACHE-WARMED B-Q_A-Q_B POSITION-2 OBSERVATION.
# Canonical producer: python/native/final_validate.py.
# Protocol:
#   • one Python process
#   • per-query call order [Baseline, Q_A, Q_B], fixed query order (0..N-1)
#   • N=6980, warmup=300, reps=3
#   • baseline backend = v2, ef=50, single-core taskset -c $CORE
#   • the second QLR call (Q_B) inherits the first QLR call's cache warmth
#
# The Q_B row is NOT an isolated speedup; it is a cache-warmed observation.
# RESULT_LABEL.txt is written into every output dir accordingly.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_CODE_PKG_ROOT="$(cd "${HERE}/../.." && pwd)"
export SUBMISSION_CODE_PKG_ROOT
source "${SUBMISSION_CODE_PKG_ROOT}/common/common_env.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/timestamp_helpers.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/verify_paths.sh"

verify_native_paths

if [ -n "${CAMPAIGN_DIR:-}" ]; then
    OUT_DIR="${CAMPAIGN_DIR}/02_cachewarmed_best_$(now_stamp)"
    mkdir -p "$OUT_DIR"
else
    OUT_DIR="$(make_result_dir "$SUBMISSION_CODE_PKG_ROOT" "02_cachewarmed_best")"
fi

# --- fixed protocol overrides (defaults ARE the canonical values) -------------
N_ARG="${N:-6980}"
WARMUP_ARG="${WARMUP:-300}"
REPS_ARG="${REPS:-3}"

# --- exact Q_A / Q_B configs (per the task specification) ---------------------
CFG_A="kp=20,kep=10,th=0.35,ef=112,ef_min=10,rEF=16,backend=v2"
CFG_B="kp=20,kep=10,th=0.32,ef=112,ef_min=10,rEF=12,backend=v2"

CANONICAL_CMD=(
    taskset -c "${CORE}"
    "$NATIVE_PY" -u "$NATIVE_INTERLEAVED_SCRIPT"
        --n "$N_ARG"
        --warmup "$WARMUP_ARG"
        --reps "$REPS_ARG"
        --baseline_ef 50
        --baseline_backend v2
        --cfg_a "$CFG_A"
        --cfg_b "$CFG_B"
        --out_dir "$OUT_DIR"
        --core "$CORE"
)
{
    echo "# canonical command (final_validate.py with --out_dir set to our sandbox)"
    printf '%q ' "${CANONICAL_CMD[@]}"
    echo
} > "${OUT_DIR}/command.txt"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[02_cachewarmed_best][CHECK_ONLY] paths + interpreter OK"
    echo "[02_cachewarmed_best][CHECK_ONLY] command that WOULD run:"
    cat "${OUT_DIR}/command.txt"
    verify_native_import || true
    exit 0
fi

STDOUT_LOG="${OUT_DIR}/stdout.log"
STDERR_LOG="${OUT_DIR}/stderr.log"
COMBINED_LOG="${OUT_DIR}/combined.log"

emit_environment_snapshot "$OUT_DIR" > /dev/null
emit_pip_versions "$NATIVE_PY" "${OUT_DIR}/pip_versions.txt"

SRC_SHA="$(sha256_or_missing "$NATIVE_INTERLEAVED_SCRIPT")"
SO_V1_SHA="$(sha256_or_missing "$NATIVE_SO_V1")"
SO_V2_SHA="$(sha256_or_missing "$NATIVE_SO_V2")"

LOAD_BEFORE="$(cat /proc/loadavg 2>/dev/null || echo unavailable)"
echo "[02_cachewarmed_best] host=$(hostname) load=${LOAD_BEFORE}"
echo "[02_cachewarmed_best] script: $NATIVE_INTERLEAVED_SCRIPT (sha256=${SRC_SHA})"
echo "[02_cachewarmed_best] Q_A: $CFG_A"
echo "[02_cachewarmed_best] Q_B: $CFG_B  <- position 2, cache-warmed"
echo "[02_cachewarmed_best] N=$N_ARG WARMUP=$WARMUP_ARG REPS=$REPS_ARG CORE=$CORE"
echo "[02_cachewarmed_best] expected wall-clock at low load: 25-60 minutes"

t0=$(date +%s)
set +e
"${CANONICAL_CMD[@]}" > >(tee "$STDOUT_LOG" > "$COMBINED_LOG") 2> >(tee "$STDERR_LOG" >&2)
RC=$?
set -e
t1=$(date +%s)
DUR=$(( t1 - t0 ))

cat > "${OUT_DIR}/RESULT_LABEL.txt" <<'EOF'
============================================================
RESULT LABEL (mandatory when reporting Q_B):
    MULTI-CONFIG CACHE-WARMED B-Q_A-Q_B POSITION-2 OBSERVATION

Never present this as an isolated speedup or a paper-comparable
result. Q_B inherits Q_A's freshly warmed PCA weights, iq_vecs,
ep_ids and doc_vecs caches; the same-run baseline does not share
that exact cache state.
============================================================
EOF

"$REPORT_PY" - "$OUT_DIR" "$SRC_SHA" "$SO_V1_SHA" "$SO_V2_SHA" "$RC" "$DUR" "$LOAD_BEFORE" <<'PY'
import sys, os, json
out_dir, src_sha, so1_sha, so2_sha, rc, dur, load_before = sys.argv[1:8]
meta = {
    "result_id": "02_cachewarmed_best",
    "protocol_class": "E. MULTI-CONFIG CACHE-WARMED (Faithful MS MARCO-v1 500k, class J)",
    "protocol_label": "MULTI-CONFIG CACHE-WARMED B-Q_A-Q_B POSITION-2 OBSERVATION",
    "canonical_script": os.environ.get("NATIVE_INTERLEAVED_SCRIPT"),
    "canonical_script_sha256": src_sha,
    "native_so_v1_sha256": so1_sha,
    "native_so_v2_sha256": so2_sha,
    "python_interpreter": os.environ.get("NATIVE_PY"),
    "cpu_core": int(os.environ.get("CORE", "21")),
    "thread_env": {k: os.environ.get(k) for k in
                    ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS","PYTHONDONTWRITEBYTECODE"]},
    "cfg_a": "kp=20,kep=10,th=0.35,ef=112,ef_min=10,rEF=16,backend=v2",
    "cfg_b": "kp=20,kep=10,th=0.32,ef=112,ef_min=10,rEF=12,backend=v2",
    "baseline_backend": "v2", "baseline_ef": 50,
    "n_queries": int(os.environ.get("N", "6980")),
    "warmup": int(os.environ.get("WARMUP", "300")),
    "reps": int(os.environ.get("REPS", "3")),
    "query_order": "fixed 0..N-1",
    "per_query_call_order": ["baseline(v2, ef=50)", "Q_A (v2)", "Q_B (v2)"],
    "one_process": True,
    "return_code": int(rc),
    "wall_clock_seconds": int(dur),
    "loadavg_before": load_before.strip(),
    "loadavg_after": open("/proc/loadavg").read().strip() if os.path.exists("/proc/loadavg") else "unknown",
    "output_dir": out_dir,
    "warning": "Q_B (position 2) is a CACHE-WARMED observation. Do not present as isolated.",
}
with open(os.path.join(out_dir, "wrapper_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
PY

echo "[02_cachewarmed_best] DONE rc=${RC} elapsed=${DUR}s output=${OUT_DIR}"
exit "$RC"
