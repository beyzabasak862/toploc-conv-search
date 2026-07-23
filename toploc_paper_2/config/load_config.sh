#!/bin/bash
# config/load_config.sh — the single, package-local shell config loader.
# Source (do not execute) from every launcher in SUBMISSION_CODE_PACKAGE.
#
# 1. Resolves SUBMISSION_CODE_PKG_ROOT from this script's own path.
# 2. Sources config/paths.env (fails loudly if missing).
# 3. Exports every variable defined there so subprocesses (Python producers,
#    build tools) can see them.
# 4. Sets the immutable single-thread environment used by every measured
#    benchmark (OMP_NUM_THREADS, MKL_NUM_THREADS, etc.).
#
# It contains no absolute paths of its own.

_LOAD_CONFIG_SH="${BASH_SOURCE[0]:-$0}"
SUBMISSION_CODE_PKG_ROOT="$(cd "$(dirname "$_LOAD_CONFIG_SH")/.." && pwd)"
export SUBMISSION_CODE_PKG_ROOT

_ENV_FILE="${SUBMISSION_CODE_PKG_ROOT}/config/paths.env"
if [ ! -f "$_ENV_FILE" ]; then
    echo "[load_config] ERROR: config/paths.env is missing." >&2
    echo "              cp \"${SUBMISSION_CODE_PKG_ROOT}/config/paths.env.example\" \\" >&2
    echo "                 \"${_ENV_FILE}\"" >&2
    echo "              and edit \"${_ENV_FILE}\" to point at your files." >&2
    return 1 2>/dev/null || exit 1
fi

# Load KEY=VALUE lines from paths.env. Use `set -a` so every assignment auto-exports.
# The dot ( . ) source keeps interpolation of ${SUBMISSION_CODE_PKG_ROOT} inside paths.env working.
set -a
# shellcheck source=./paths.env.example
. "$_ENV_FILE"
set +a

# --- Immutable single-thread env (must not be relaxed for benchmark runs) ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

# --- Default CORE ------------------------------------------------------------
export CORE="${CORE:-21}"

# --- Default OUTPUT_ROOT (inside the package if not overridden) -------------
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SUBMISSION_CODE_PKG_ROOT}/outputs}"

# --- Sanity: verify variables that every workflow needs are non-empty --------
_REQUIRED_ALWAYS=(HYBRID_PYTHON NATIVE_PYTHON REPORT_PYTHON CORE)
_missing=""
for v in "${_REQUIRED_ALWAYS[@]}"; do
    if [ -z "${!v:-}" ]; then _missing="${_missing} ${v}"; fi
done
if [ -n "$_missing" ]; then
    echo "[load_config] WARNING: unset variables (edit config/paths.env):${_missing}" >&2
fi
unset _missing _REQUIRED_ALWAYS _ENV_FILE _LOAD_CONFIG_SH
