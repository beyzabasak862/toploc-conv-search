#!/bin/bash
# RUN_ALL.sh — sequential, dedup-aware execution of the six reproduction results.
#
# ORDER (mirrors the validated overnight campaign in the source repro package,
#         with Experiment 7 appended after Experiment 6 as per Experiment-7 spec):
#   1) 01_safe_hybrid             (rescue_full_run.py - safe+aggressive both measured)
#   2) 05_aggressive_hybrid       (dedup: extracts aggressive from Result 1's output)
#   3) 06_stage2_bounded_pareto   (rescue_stage2_accuracy.py - bounded 500-query grid)
#   4) 03_native_equal_accuracy   (benchmark_native.py - v1)
#   5) 04_native_canonical_v3     (canonical_final.py x 3 + aggregation)
#   6) 02_cachewarmed_best        (final_validate.py - B-Q_A-Q_B cache-warmed)
#   7) 07_faithful_adaptive_depth (runner.py - paper Algorithm 1 with adaptive HNSW ef)
#   8) 08_cachewarmed_treccast    (cachewarmed_treccast.py - HYBRID full TREC-CAsT B-Q_A-Q_B)
#
# CHECK_ONLY=1 ./RUN_ALL.sh
#   Runs CHECK_ONLY=1 for every wrapper. No benchmark is launched.
#
# YES=1 ./RUN_ALL.sh
#   Skips the interactive "RUN ALL EIGHT" confirmation gate.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_CODE_PKG_ROOT="$HERE"
export SUBMISSION_CODE_PKG_ROOT

source "${SUBMISSION_CODE_PKG_ROOT}/common/common_env.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/timestamp_helpers.sh"

CAMPAIGN_DIR="$(make_campaign_dir "$SUBMISSION_CODE_PKG_ROOT")"
export CAMPAIGN_DIR
CAMPAIGN_LOG="${CAMPAIGN_DIR}/RUN_ALL.log"

{
    echo "===================================================="
    echo "SUBMISSION_CODE_PACKAGE / RUN_ALL"
    echo "===================================================="
    echo "campaign_dir: $CAMPAIGN_DIR"
    echo "hostname: $(hostname)"
    echo "user: $(whoami)"
    echo "date_utc: $(date -u)"
    echo "loadavg: $(cat /proc/loadavg 2>/dev/null || echo unavailable)"
    echo "CORE (taskset): ${CORE}"
    echo
    echo "This run will execute (in order):"
    echo "  1) 01_safe_hybrid             — rescue_full_run.py"
    echo "  2) 05_aggressive_hybrid       — DEDUP from Result 1"
    echo "  3) 06_stage2_bounded_pareto   — rescue_stage2_accuracy.py"
    echo "  4) 03_native_equal_accuracy   — benchmark_native.py"
    echo "  5) 04_native_canonical_v3     — canonical_final.py x 3"
    echo "  6) 02_cachewarmed_best        — final_validate.py"
    echo "  7) 07_faithful_adaptive_depth — runner.py (paper Alg 1 + adaptive ef)"
    echo "  8) 08_cachewarmed_treccast    — cachewarmed_treccast.py (HYBRID TREC-CAsT B-Q_A-Q_B)"
} | tee -a "$CAMPAIGN_LOG"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[RUN_ALL] CHECK_ONLY=1 - running every wrapper in check-only mode" | tee -a "$CAMPAIGN_LOG"
    for rid in 01_safe_hybrid 05_aggressive_hybrid 06_stage2_bounded_pareto \
               03_native_equal_accuracy 04_native_canonical_v3 02_cachewarmed_best \
               07_faithful_adaptive_depth 08_cachewarmed_treccast; do
        echo | tee -a "$CAMPAIGN_LOG"
        echo "== CHECK_ONLY: $rid ==" | tee -a "$CAMPAIGN_LOG"
        CHECK_ONLY=1 CAMPAIGN_DIR="$CAMPAIGN_DIR" \
            bash "${SUBMISSION_CODE_PKG_ROOT}/benchmarks/${rid}/RUN.sh" 2>&1 | tee -a "$CAMPAIGN_LOG"
    done
    echo | tee -a "$CAMPAIGN_LOG"
    echo "[RUN_ALL] CHECK_ONLY complete" | tee -a "$CAMPAIGN_LOG"
    exit 0
fi

if [ "${YES:-0}" = "1" ]; then
    echo "[RUN_ALL] YES=1 - confirmation gate skipped."
else
    echo
    echo "Type exactly:  RUN ALL EIGHT"
    echo -n "> "
    IFS= read -r CONFIRM
    if [ "$CONFIRM" != "RUN ALL EIGHT" ]; then
        echo "[RUN_ALL] confirmation not entered. Aborting."
        exit 1
    fi
fi
echo "[RUN_ALL] starting at $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$CAMPAIGN_LOG"

run_stage() {
    local label="$1" runsh="$2" stagelog="$3"; shift 3
    local extra_env=("$@")
    echo | tee -a "$CAMPAIGN_LOG"
    echo "== STAGE: $label   ==   $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$CAMPAIGN_LOG"
    local t0; t0=$(date +%s)
    (env "${extra_env[@]}" bash "$runsh") 2>&1 | tee "$stagelog"
    local rc=${PIPESTATUS[0]}
    echo "== STAGE $label rc=$rc elapsed=$(($(date +%s) - t0))s ==" | tee -a "$CAMPAIGN_LOG"
    return "$rc"
}

# Stage 1 -----------------------------------------------------------------
run_stage "01_safe_hybrid" "$SUBMISSION_CODE_PKG_ROOT/benchmarks/01_safe_hybrid/RUN.sh" \
          "${CAMPAIGN_DIR}/stage_01_safe_hybrid.log"

STAGE1_OUT_DIR="$(find "$CAMPAIGN_DIR" -maxdepth 1 -type d -name '01_safe_hybrid_*' | sort | tail -1)"
STAGE1_FULL_JSON="$(find "$STAGE1_OUT_DIR" -maxdepth 3 -name full_run.json -print -quit || true)"
if [ -n "$STAGE1_FULL_JSON" ] && [ -f "$STAGE1_FULL_JSON" ]; then
    STAGE1_SHARE_DIR="$(dirname "$STAGE1_FULL_JSON")"
    echo "[RUN_ALL] Result 5 will dedup from: $STAGE1_SHARE_DIR" | tee -a "$CAMPAIGN_LOG"
else
    STAGE1_SHARE_DIR=""
    echo "[RUN_ALL] WARNING: Result 1 did not produce full_run.json; Result 5 will run standalone." | tee -a "$CAMPAIGN_LOG"
fi

# Stage 2 -----------------------------------------------------------------
if [ -n "$STAGE1_SHARE_DIR" ]; then
    run_stage "05_aggressive_hybrid (DEDUP)" \
        "$SUBMISSION_CODE_PKG_ROOT/benchmarks/05_aggressive_hybrid/RUN.sh" \
        "${CAMPAIGN_DIR}/stage_05_aggressive_hybrid.log" \
        "SHARE_HYBRID_OUTPUT=$STAGE1_SHARE_DIR"
else
    run_stage "05_aggressive_hybrid (STANDALONE)" \
        "$SUBMISSION_CODE_PKG_ROOT/benchmarks/05_aggressive_hybrid/RUN.sh" \
        "${CAMPAIGN_DIR}/stage_05_aggressive_hybrid.log"
fi

# Stage 3 -----------------------------------------------------------------
run_stage "06_stage2_bounded_pareto" \
    "$SUBMISSION_CODE_PKG_ROOT/benchmarks/06_stage2_bounded_pareto/RUN.sh" \
    "${CAMPAIGN_DIR}/stage_06_stage2_bounded_pareto.log"

# Stage 4 -----------------------------------------------------------------
run_stage "03_native_equal_accuracy" \
    "$SUBMISSION_CODE_PKG_ROOT/benchmarks/03_native_equal_accuracy/RUN.sh" \
    "${CAMPAIGN_DIR}/stage_03_native_equal_accuracy.log"

# Stage 5 -----------------------------------------------------------------
run_stage "04_native_canonical_v3" \
    "$SUBMISSION_CODE_PKG_ROOT/benchmarks/04_native_canonical_v3/RUN.sh" \
    "${CAMPAIGN_DIR}/stage_04_native_canonical_v3.log"

# Stage 6 -----------------------------------------------------------------
run_stage "02_cachewarmed_best" \
    "$SUBMISSION_CODE_PKG_ROOT/benchmarks/02_cachewarmed_best/RUN.sh" \
    "${CAMPAIGN_DIR}/stage_02_cachewarmed_best.log"

# Stage 7 -----------------------------------------------------------------
run_stage "07_faithful_adaptive_depth" \
    "$SUBMISSION_CODE_PKG_ROOT/benchmarks/07_faithful_adaptive_depth/RUN.sh" \
    "${CAMPAIGN_DIR}/stage_07_faithful_adaptive_depth.log"

# Stage 8 -----------------------------------------------------------------
run_stage "08_cachewarmed_treccast" \
    "$SUBMISSION_CODE_PKG_ROOT/benchmarks/08_cachewarmed_treccast/RUN.sh" \
    "${CAMPAIGN_DIR}/stage_08_cachewarmed_treccast.log"

echo | tee -a "$CAMPAIGN_LOG"
echo "===================================================="
echo "[RUN_ALL] DONE  campaign_dir=$CAMPAIGN_DIR"
echo "===================================================="
