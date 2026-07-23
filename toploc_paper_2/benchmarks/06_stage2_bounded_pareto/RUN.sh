#!/bin/bash
# benchmarks/06_stage2_bounded_pareto/RUN.sh
#
# Result 6: BOUNDED 500-QUERY STAGE-2 PARETO OPERATING POINT.
# Canonical producer: python/hybrid/rescue_stage2_accuracy.py.
# Config (baked in): N=500, SEED=20260717, N_REPS=3, N_WARMUP=50,
# RS=0.5, EF_DEFAULT=64, ROUTER_EF=16, SEED_MODES=[cached,recompute_l2],
# NPROBES=[3,5,10], SEEDED_EFS=[16,32,64,128] — 24 configs, target row
# recompute_l2_np3_ef16.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_CODE_PKG_ROOT="$(cd "${HERE}/../.." && pwd)"
export SUBMISSION_CODE_PKG_ROOT
source "${SUBMISSION_CODE_PKG_ROOT}/common/common_env.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/timestamp_helpers.sh"
source "${SUBMISSION_CODE_PKG_ROOT}/common/verify_paths.sh"

verify_hybrid_stage2_paths

if [ -n "${CAMPAIGN_DIR:-}" ]; then
    OUT_DIR="${CAMPAIGN_DIR}/06_stage2_bounded_pareto_$(now_stamp)"
    mkdir -p "$OUT_DIR"
else
    OUT_DIR="$(make_result_dir "$SUBMISSION_CODE_PKG_ROOT" "06_stage2_bounded_pareto")"
fi

CANONICAL_CMD=(
    taskset -c "${CORE}"
    env "OUTPUT_ROOT=${OUT_DIR}"
    "$HYBRID_PY" -u "$HYBRID_STAGE2_SCRIPT"
)
{
    echo "# canonical command (rescue_stage2_accuracy.py takes no CLI args)"
    printf '%q ' "${CANONICAL_CMD[@]}"
    echo
} > "${OUT_DIR}/command.txt"

if [ "${CHECK_ONLY:-0}" = "1" ]; then
    echo "[06_stage2_bounded_pareto][CHECK_ONLY] paths OK"
    echo "[06_stage2_bounded_pareto][CHECK_ONLY] command that WOULD run:"
    cat "${OUT_DIR}/command.txt"
    "$HYBRID_PY" - <<'PY'
import importlib
for m in ["numpy","faiss","joblib","threadpoolctl","pandas"]:
    importlib.import_module(m)
print("[06_stage2_bounded_pareto][CHECK_ONLY] hybrid Python imports OK")
PY
    exit 0
fi

STDOUT_LOG="${OUT_DIR}/stdout.log"
STDERR_LOG="${OUT_DIR}/stderr.log"
COMBINED_LOG="${OUT_DIR}/combined.log"

emit_environment_snapshot "$OUT_DIR" > /dev/null
emit_pip_versions "$HYBRID_PY" "${OUT_DIR}/pip_versions.txt"

SRC_SHA="$(sha256_or_missing "$HYBRID_STAGE2_SCRIPT")"
LOAD_BEFORE="$(cat /proc/loadavg 2>/dev/null || echo unavailable)"

echo "[06_stage2_bounded_pareto] host=$(hostname) load=${LOAD_BEFORE}"
echo "[06_stage2_bounded_pareto] script sha256=$SRC_SHA"
echo "[06_stage2_bounded_pareto] output dir: $OUT_DIR"
echo "[06_stage2_bounded_pareto] expected wall-clock at low load: 30-90 minutes"
echo "  (dominated by the 158 GB TREC-CAsT doc-index load; the 24-config"
echo "   grid itself is under 1 minute per config on the 500-query subset.)"

t0=$(date +%s)
set +e
"${CANONICAL_CMD[@]}" > >(tee "$STDOUT_LOG" > "$COMBINED_LOG") 2> >(tee "$STDERR_LOG" >&2)
RC=$?
set -e
DUR=$(( $(date +%s) - t0 ))

FRESH_DIR="$(find "$OUT_DIR" -maxdepth 1 -mindepth 1 -type d -name 'stage2_*' -printf '%p\n' | sort | tail -1)"
if [ -n "$FRESH_DIR" ] && [ -d "$FRESH_DIR" ]; then
    echo "$FRESH_DIR" > "${OUT_DIR}/canonical_output_source_path.txt"
else
    echo "MISSING" > "${OUT_DIR}/canonical_output_source_path.txt"
fi

cat > "${OUT_DIR}/RESULT_LABEL.txt" <<'EOF'
============================================================
RESULT LABEL (mandatory when reporting Result 6):
    BOUNDED 500-QUERY STAGE-2 PARETO OPERATING POINT

- bounded 500-query result (N=500, SEED=20260717)
- NOT a full 6980-query validation
- NOT equal-accuracy (Acc@10 below the 0.952 safety floor)
- MUST NOT be confused with Result 5 (aggressive full hybrid)
============================================================
EOF

"$REPORT_PY" - "$OUT_DIR" "$FRESH_DIR" "$SRC_SHA" "$RC" "$DUR" "$LOAD_BEFORE" <<'PY'
import sys, os, json
out_dir, fresh_dir, src_sha, rc, dur, load_before = sys.argv[1:7]
meta = {
    "result_id": "06_stage2_bounded_pareto",
    "protocol_class": "B. BOUNDED MEASURED - hybrid corpus (Class I), Stage-2 Pareto grid",
    "canonical_script": os.environ.get("HYBRID_STAGE2_SCRIPT"),
    "canonical_script_sha256": src_sha,
    "python_interpreter": os.environ.get("HYBRID_PY"),
    "cpu_core": int(os.environ.get("CORE", "21")),
    "n_queries": 500,
    "seed": 20260717,
    "n_reps": 3,
    "n_warmup": 50,
    "canonical_output_dir": fresh_dir or "MISSING",
    "package_output_dir": out_dir,
    "return_code": int(rc),
    "wall_clock_seconds": int(dur),
    "loadavg_before": load_before.strip(),
    "target_config_key": "recompute_l2_np3_ef16",
    "warning": "BOUNDED 500-QUERY STAGE-2 PARETO OPERATING POINT.",
}
with open(os.path.join(out_dir, "wrapper_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)
PY

echo "[06_stage2_bounded_pareto] DONE rc=${RC} elapsed=${DUR}s output=${OUT_DIR}"
exit "$RC"
