#!/bin/bash
# common/common_env.sh — shared environment configuration for every RUN.sh.
# Source; do not execute. No absolute paths here — everything is derived from
# config/paths.env via config/load_config.sh.

# --- resolve package root and load config ------------------------------------
_COMMON_ENV_SH="${BASH_SOURCE[0]:-$0}"
SUBMISSION_CODE_PKG_ROOT="$(cd "$(dirname "$_COMMON_ENV_SH")/.." && pwd)"
export SUBMISSION_CODE_PKG_ROOT
# shellcheck source=../config/load_config.sh
. "${SUBMISSION_CODE_PKG_ROOT}/config/load_config.sh"

# --- Aliases used throughout the wrappers (readable names) -------------------
# The Python producers themselves look up HYBRID_PYTHON / NATIVE_PYTHON via
# os.environ. These aliases give the shell code a stable, short handle.
export HYBRID_PY="$HYBRID_PYTHON"
export NATIVE_PY="$NATIVE_PYTHON"
export REPORT_PY="$REPORT_PYTHON"

# --- Producer script paths (all local to this package) -----------------------
export HYBRID_PY_SCRIPT="${SUBMISSION_CODE_PKG_ROOT}/python/hybrid/rescue_full_run.py"
export HYBRID_STAGE2_SCRIPT="${SUBMISSION_CODE_PKG_ROOT}/python/hybrid/rescue_stage2_accuracy.py"
export NATIVE_BENCHMARK_SCRIPT="${SUBMISSION_CODE_PKG_ROOT}/python/native/benchmark_native.py"
export NATIVE_CANONICAL_SCRIPT="${SUBMISSION_CODE_PKG_ROOT}/python/native/canonical_final.py"
export NATIVE_INTERLEAVED_SCRIPT="${SUBMISSION_CODE_PKG_ROOT}/python/native/final_validate.py"
export FAITHFUL_PY_SCRIPT="${SUBMISSION_CODE_PKG_ROOT}/python/faithful/runner.py"
export CACHEWARMED_TRECCAST_SCRIPT="${SUBMISSION_CODE_PKG_ROOT}/python/hybrid/cachewarmed_treccast.py"

# --- Optional: native .so files that will be imported by the native scripts --
# NATIVE_MODULE_DIR is set in paths.env. The verify helper below inspects it.
export NATIVE_SO_V1="${NATIVE_MODULE_DIR}/native_qlr.cpython-310-x86_64-linux-gnu.so"
export NATIVE_SO_V2="${NATIVE_MODULE_DIR}/native_qlr_v2.cpython-310-x86_64-linux-gnu.so"
export NATIVE_SO_V3="${NATIVE_MODULE_DIR}/native_qlr_v3.cpython-310-x86_64-linux-gnu.so"

unset _COMMON_ENV_SH
