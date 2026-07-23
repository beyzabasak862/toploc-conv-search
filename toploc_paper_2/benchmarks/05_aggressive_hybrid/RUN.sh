#!/bin/bash
# benchmarks/05_aggressive_hybrid/RUN.sh
#
# Result 5: Aggressive hybrid QLR trade-off endpoint (SEEDED_EF=16).
#
# The aggressive endpoint is measured IN THE SAME rescue_full_run.py execution
# as Result 1 (SEEDED_EFS=[32, 16] both measured in one run).
#
# Dedup: if SHARE_HYBRID_OUTPUT (path to an existing Result-1 canonical output
# directory containing full_run.json) is set, extract the aggressive row from
# that file instead of running the 1-4h benchmark again. RUN_ALL.sh sets it.
#
# Standalone: if neither SHARE_HYBRID_OUTPUT nor REUSE_HYBRID_JSON is set,
# this wrapper invokes rescue_full_run.py itself (produces both Result 1 and
# Result 5 outputs; the label written into RESULT_LABEL.txt makes this clear).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_CODE_PKG_ROOT="$(cd "${HERE}/../.." && pwd)"
export SUBMISSION_CODE_PKG_ROOT
source "${SUBMISSION_CODE_PKG_ROOT}/common/common_env.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/timestamp_helpers.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/verify_paths.sh"

verify_hybrid_paths

if [ -n "${CAMPAIGN_DIR:-}" ]; then
    OUT_DIR="${CAMPAIGN_DIR}/05_aggressive_hybrid_$(now_stamp)"
    mkdir -p "$OUT_DIR"
else
    OUT_DIR="$(make_result_dir "$SUBMISSION_CODE_PKG_ROOT" "05_aggressive_hybrid")"
fi
echo "[05_aggressive_hybrid] output directory: $OUT_DIR"

SRC_SHA="$(sha256_or_missing "$HYBRID_PY_SCRIPT")"
LOAD_BEFORE="$(cat /proc/loadavg 2>/dev/null || echo unavailable)"

FULL_JSON=""; SOURCE_KIND=""

if [ -n "${SHARE_HYBRID_OUTPUT:-}" ]; then
    _try="${SHARE_HYBRID_OUTPUT}/full_run.json"
    if [ ! -f "$_try" ]; then
        _try="$(find "$SHARE_HYBRID_OUTPUT" -maxdepth 3 -name full_run.json -print -quit)"
    fi
    if [ -n "$_try" ] && [ -f "$_try" ]; then
        FULL_JSON="$_try"; SOURCE_KIND="dedup_from_result_1"
        echo "[05_aggressive_hybrid] dedup: reusing $FULL_JSON"
    else
        echo "[05_aggressive_hybrid] SHARE_HYBRID_OUTPUT set but no full_run.json under $SHARE_HYBRID_OUTPUT" >&2
        exit 2
    fi
elif [ -n "${REUSE_HYBRID_JSON:-}" ] && [ -f "${REUSE_HYBRID_JSON}" ]; then
    FULL_JSON="${REUSE_HYBRID_JSON}"; SOURCE_KIND="dedup_from_manual_override"
    echo "[05_aggressive_hybrid] dedup: reusing $FULL_JSON"
fi

CANONICAL_CMD=(
    taskset -c "${CORE}"
    env "OUTPUT_ROOT=${OUT_DIR}"
    "$HYBRID_PY" -u "$HYBRID_PY_SCRIPT"
)
{
    if [ -n "$FULL_JSON" ]; then
        echo "# dedup source (no benchmark invoked here): $FULL_JSON"
    else
        echo "# canonical command (rescue_full_run.py takes no CLI args)"
        printf '%q ' "${CANONICAL_CMD[@]}"
        echo
    fi
} > "${OUT_DIR}/command.txt"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[05_aggressive_hybrid][CHECK_ONLY] paths OK"
    echo "[05_aggressive_hybrid][CHECK_ONLY] plan:"
    cat "${OUT_DIR}/command.txt"
    exit 0
fi

emit_environment_snapshot "$OUT_DIR" > /dev/null
emit_pip_versions "$HYBRID_PY" "${OUT_DIR}/pip_versions.txt"

if [ -z "$FULL_JSON" ]; then
    STDOUT_LOG="${OUT_DIR}/stdout.log"
    STDERR_LOG="${OUT_DIR}/stderr.log"
    COMBINED_LOG="${OUT_DIR}/combined.log"
    echo "[05_aggressive_hybrid] standalone run - launching rescue_full_run.py"
    echo "[05_aggressive_hybrid] expected wall-clock at low load: 1-4 hours"
    t0=$(date +%s)
    set +e
    "${CANONICAL_CMD[@]}" > >(tee "$STDOUT_LOG" > "$COMBINED_LOG") 2> >(tee "$STDERR_LOG" >&2)
    RC=$?
    set -e
    DUR=$(( $(date +%s) - t0 ))
    FRESH_DIR="$(find "$OUT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'full_*' -printf '%p\n' | sort | tail -1)"
    if [ -n "$FRESH_DIR" ] && [ -d "$FRESH_DIR" ] && [ -f "$FRESH_DIR/full_run.json" ]; then
        FULL_JSON="$FRESH_DIR/full_run.json"; SOURCE_KIND="standalone_run"
        echo "$FRESH_DIR" > "${OUT_DIR}/canonical_output_source_path.txt"
    else
        echo "[05_aggressive_hybrid] ERROR: could not identify newly created full_ dir" >&2
        exit 3
    fi
else
    RC=0
    DUR=0
fi

AGG_JSON="${OUT_DIR}/aggressive_extracted.json"
"$REPORT_PY" - "$FULL_JSON" "$AGG_JSON" "$SOURCE_KIND" <<'PY'
import sys, json
full_path, out_path, kind = sys.argv[1:4]
j = json.load(open(full_path))
b = j["results"]["baseline"]
a = j["results"]["variants"]["recompute_l2_np3_ef16"]
out = {
    "result_id": "05_aggressive_hybrid",
    "result_label": "AGGRESSIVE ACCURACY-SPEED TRADE-OFF",
    "source_full_run_json": full_path,
    "source_kind": kind,
    "n_queries": j["n_queries"],
    "baseline_ef64": {
        "acc10": b["acc10"],
        "mean_us": b["lat_us"]["mean"],
        "median_us": b["lat_us"]["median"],
        "p95_us": b["lat_us"]["p95"],
    },
    "aggressive_ef16": {
        "acc10": a["acc10"],
        "mean_us": a["lat_us"]["mean"],
        "median_us": a["lat_us"]["median"],
        "p95_us": a["lat_us"]["p95"],
        "speedup_mean":   a["speedup_mean"],
        "speedup_median": a["speedup_median"],
        "acc_delta_vs_baseline": a["acc_delta_vs_baseline"],
        "acc_safe": a["acc_safe"],
    },
    "warning": (
        "Trade-off endpoint. Acc@10 drops below the 0.952 safety floor. "
        "Do not present as an equal-accuracy speedup or as the primary safe headline."
    ),
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"[05_aggressive_hybrid] wrote {out_path}")
print(f"  baseline acc={b['acc10']:.6f} mean_us={b['lat_us']['mean']:.3f}")
print(f"  aggressive acc={a['acc10']:.6f} mean_us={a['lat_us']['mean']:.3f}")
print(f"  aggressive spd_mean={a['speedup_mean']:.4f} spd_median={a['speedup_median']:.4f}")
PY

cat > "${OUT_DIR}/RESULT_LABEL.txt" <<'EOF'
============================================================
RESULT LABEL (mandatory):
    AGGRESSIVE ACCURACY-SPEED TRADE-OFF

Acc@10 is BELOW the 0.952 safety floor. This is a trade-off
endpoint, not a headline. Never present it as equal-accuracy
or as the primary safe result. Report only alongside Result 1.
============================================================
EOF

"$REPORT_PY" - "$OUT_DIR" "$SRC_SHA" "$SOURCE_KIND" "$LOAD_BEFORE" <<'PY'
import sys, os, json
out_dir, src_sha, source_kind, load_before = sys.argv[1:5]
meta = {
    "result_id": "05_aggressive_hybrid",
    "protocol_class": "A. FULL-RUN MEASURED - hybrid corpus (I), AGGRESSIVE endpoint",
    "canonical_script": os.environ.get("HYBRID_PY_SCRIPT"),
    "canonical_script_sha256": src_sha,
    "source_kind": source_kind,
    "cpu_core": int(os.environ.get("CORE", "21")),
    "warning": "AGGRESSIVE ACCURACY-SPEED TRADE-OFF (below the 0.952 safety floor).",
}
with open(os.path.join(out_dir, "wrapper_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
PY

echo "[05_aggressive_hybrid] DONE output=${OUT_DIR}"
exit 0
