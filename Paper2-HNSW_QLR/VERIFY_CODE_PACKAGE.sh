#!/bin/bash
# VERIFY_CODE_PACKAGE.sh — check that the package can execute on this host.
#
# Runs:
#   1. bash -n on every shell script
#   2. Python compileall on every bundled .py
#   3. paths.env presence + required-variable schema check
#   4. External-path existence (fails loudly with the exact env variable)
#   5. Native module import test (imports native_qlr, _v2, _v3)
#   6. Interpreter version + minimal package availability check
#   7. CHECK_ONLY=1 for every RUN.sh (verifies wrapper-level chain works)
#
# Exit code is non-zero if any hard check fails.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_CODE_PKG_ROOT="$HERE"
export SUBMISSION_CODE_PKG_ROOT

VAL_DIR="${SUBMISSION_CODE_PKG_ROOT}/validation"
mkdir -p "$VAL_DIR"
REPORT="${VAL_DIR}/VERIFY_CODE_PACKAGE_$(date -u +%Y%m%dT%H%M%SZ).log"

pass=0; warn=0; fail=0
_p() { echo "PASS: $*" | tee -a "$REPORT"; pass=$((pass+1)); }
_w() { echo "WARN: $*" | tee -a "$REPORT"; warn=$((warn+1)); }
_f() { echo "FAIL: $*" | tee -a "$REPORT"; fail=$((fail+1)); }

echo "== VERIFY_CODE_PACKAGE  $(date -u)  host=$(hostname)  pkg=$SUBMISSION_CODE_PKG_ROOT" | tee "$REPORT"

# ---------------------------------------------------------------- 1. bash -n
echo | tee -a "$REPORT"; echo "-- bash -n --" | tee -a "$REPORT"
while IFS= read -r -d '' sh; do
    if bash -n "$sh" 2>>"$REPORT"; then
        _p "bash -n $sh"
    else
        _f "bash -n $sh"
    fi
done < <(find "$SUBMISSION_CODE_PKG_ROOT" -type f -name '*.sh' -print0)

# ---------------------------------------------------------------- 2. compileall
echo | tee -a "$REPORT"; echo "-- python compileall --" | tee -a "$REPORT"
if [ -f "${SUBMISSION_CODE_PKG_ROOT}/config/paths.env" ]; then
    source "${SUBMISSION_CODE_PKG_ROOT}/config/load_config.sh"
fi
for py in "${HYBRID_PYTHON:-/usr/bin/python3}" "${NATIVE_PYTHON:-/usr/bin/python3}"; do
    [ -x "$py" ] || continue
    if PYTHONDONTWRITEBYTECODE=1 "$py" -m compileall -q -b "${SUBMISSION_CODE_PKG_ROOT}/python" \
                             "${SUBMISSION_CODE_PKG_ROOT}/benchmarks" 2>>"$REPORT"; then
        _p "compileall ok under $py"
    else
        _f "compileall failed under $py"
    fi
done

# ---------------------------------------------------------------- 3. paths.env
echo | tee -a "$REPORT"; echo "-- config/paths.env --" | tee -a "$REPORT"
if [ ! -f "${SUBMISSION_CODE_PKG_ROOT}/config/paths.env" ]; then
    _f "config/paths.env missing (copy config/paths.env.example and edit)"
else
    _p "config/paths.env present"
fi

# ---------------------------------------------------------------- 4. external paths + verify_paths.sh helpers
echo | tee -a "$REPORT"; echo "-- external path existence --" | tee -a "$REPORT"
if [ -f "${SUBMISSION_CODE_PKG_ROOT}/config/paths.env" ]; then
    source "${SUBMISSION_CODE_PKG_ROOT}/common/common_env.sh"
    source "${SUBMISSION_CODE_PKG_ROOT}/common/verify_paths.sh"
    if verify_hybrid_stage2_paths 2>>"$REPORT"; then
        _p "hybrid + stage2 paths OK"
    else
        _f "hybrid + stage2 paths FAIL — see log for the exact env variable"
    fi
    if verify_native_paths 2>>"$REPORT"; then
        _p "native paths OK"
    else
        _f "native paths FAIL — see log"
    fi
    if verify_faithful_paths 2>>"$REPORT"; then
        _p "faithful paths OK"
    else
        _f "faithful paths FAIL — see log for the exact env variable"
    fi
    if verify_cachewarmed_treccast_paths 2>>"$REPORT"; then
        _p "benchmark 08 (hybrid TREC-CAsT) paths OK"
    else
        _f "benchmark 08 paths FAIL — see log for the exact env variable"
    fi
fi

# ---------------------------------------------------------------- 5. native module import
echo | tee -a "$REPORT"; echo "-- native module import --" | tee -a "$REPORT"
if [ -f "${SUBMISSION_CODE_PKG_ROOT}/config/paths.env" ]; then
    if verify_native_import 2>>"$REPORT"; then
        _p "native_qlr / _v2 / _v3 import OK"
    else
        _w "native module import failed (may need build/BUILD_NATIVE.sh)"
    fi
    if verify_faithful_import 2>>"$REPORT"; then
        _p "faithful (faithful_qlr + runner.py compile) import OK"
    else
        _f "faithful import FAIL — see log"
    fi
    if verify_cachewarmed_treccast_import 2>>"$REPORT"; then
        _p "benchmark 08 (hybrid FAISS; no native refs) import OK"
    else
        _f "benchmark 08 import FAIL — see log"
    fi
fi

# ---------------------------------------------------------------- 6. minimal pypackage availability
echo | tee -a "$REPORT"; echo "-- python package availability --" | tee -a "$REPORT"
if [ -x "${HYBRID_PYTHON:-}" ]; then
    "${HYBRID_PYTHON}" - <<'PY' 2>&1 | tee -a "$REPORT"
import importlib
required = ["numpy", "faiss", "joblib", "threadpoolctl", "pandas", "sklearn"]
missing = []
for m in required:
    try:
        importlib.import_module(m)
    except ImportError as e:
        missing.append((m, str(e)))
if missing:
    print("HYBRID_PYTHON missing packages:", missing)
    raise SystemExit(2)
print("HYBRID_PYTHON package check OK")
PY
    [ ${PIPESTATUS[0]} -eq 0 ] && _p "hybrid python pkg check" || _f "hybrid python pkg check"
fi
if [ -x "${NATIVE_PYTHON:-}" ]; then
    "${NATIVE_PYTHON}" - <<'PY' 2>&1 | tee -a "$REPORT"
import importlib
required = ["numpy"]
for m in required: importlib.import_module(m)
print("NATIVE_PYTHON package check OK")
PY
    [ ${PIPESTATUS[0]} -eq 0 ] && _p "native python pkg check" || _f "native python pkg check"
fi

# ---------------------------------------------------------------- 7. CHECK_ONLY runs
echo | tee -a "$REPORT"; echo "-- CHECK_ONLY per RUN.sh --" | tee -a "$REPORT"
for rid in 01_safe_hybrid 02_cachewarmed_best 03_native_equal_accuracy \
           04_native_canonical_v3 05_aggressive_hybrid 06_stage2_bounded_pareto \
           07_faithful_adaptive_depth 08_cachewarmed_treccast; do
    if CHECK_ONLY=1 bash "${SUBMISSION_CODE_PKG_ROOT}/benchmarks/${rid}/RUN.sh" >>"$REPORT" 2>&1; then
        _p "CHECK_ONLY $rid"
    else
        _f "CHECK_ONLY $rid"
    fi
done

# ---------------------------------------------------------------- 8. preprocessing (static; no job started)
echo | tee -a "$REPORT"; echo "-- preprocessing static validation --" | tee -a "$REPORT"
if [ -x "${SUBMISSION_CODE_PKG_ROOT}/preprocessing/VERIFY_PREPROCESSING.sh" ]; then
    if bash "${SUBMISSION_CODE_PKG_ROOT}/preprocessing/VERIFY_PREPROCESSING.sh" >>"$REPORT" 2>&1; then
        _p "preprocessing VERIFY_PREPROCESSING.sh (no job started)"
    else
        _f "preprocessing VERIFY_PREPROCESSING.sh"
    fi
else
    _w "preprocessing/VERIFY_PREPROCESSING.sh missing or not executable"
fi

echo | tee -a "$REPORT"
echo "=== SUMMARY  PASS=$pass  WARN=$warn  FAIL=$fail ===" | tee -a "$REPORT"
if [ "$fail" -gt 0 ]; then
    echo "(inspect $REPORT for details)"
    exit 1
fi
exit 0
