#!/bin/bash
# build/BUILD_NATIVE.sh — portable rebuild of the three native pybind11 modules.
#
# Compiles native_qlr, native_qlr_v2, native_qlr_v3 from native/src/*.cpp
# into native/build/*.so using the interpreter named in NATIVE_PYTHON (from
# config/paths.env). The default -march for v2/v3 was the original AMD Zen 2
# (znver2); this wrapper falls back to -march=native when znver2 support is
# absent, which is more portable but produces the same numerical results.
#
# The prebuilt native/prebuilt/*.so binaries were compiled on an AMD Zen 2
# machine with cpython-310 x86-64. If your CPU differs or the Python ABI
# differs, rebuild here.
#
# Usage:
#   ./build/BUILD_NATIVE.sh              # builds all three .so files
#   NATIVE_MARCH=native ./build/BUILD_NATIVE.sh   # override -march for v2/v3
#   V=v3 ./build/BUILD_NATIVE.sh         # only build a specific module

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_CODE_PKG_ROOT="$(cd "${HERE}/.." && pwd)"
export SUBMISSION_CODE_PKG_ROOT

# Load config for NATIVE_PYTHON
if [ ! -f "${SUBMISSION_CODE_PKG_ROOT}/config/paths.env" ]; then
    echo "[BUILD_NATIVE] config/paths.env missing. cp config/paths.env.example config/paths.env and edit." >&2
    exit 1
fi
# shellcheck source=../config/load_config.sh
source "${SUBMISSION_CODE_PKG_ROOT}/config/load_config.sh"

PY="${NATIVE_PYTHON:?NATIVE_PYTHON must be set in config/paths.env}"
if [ ! -x "$PY" ]; then
    echo "[BUILD_NATIVE] NATIVE_PYTHON not executable: $PY" >&2
    exit 2
fi

BUILD_DIR="${SUBMISSION_CODE_PKG_ROOT}/native/build"
SRC_DIR="${SUBMISSION_CODE_PKG_ROOT}/native/src"
mkdir -p "$BUILD_DIR"

PYINC=$("$PY" -c "import sysconfig; print(sysconfig.get_path('include'))")
PYSUFFIX=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYBIND11INC=$("$PY" -c "import pybind11; print(pybind11.get_include())" 2>/dev/null || echo "")
if [ -z "$PYBIND11INC" ]; then
    echo "[BUILD_NATIVE] pybind11 not available in $PY" >&2
    echo "              install with:  $PY -m pip install pybind11" >&2
    exit 3
fi

echo "[BUILD_NATIVE] python: $PY"
echo "[BUILD_NATIVE] extension suffix: $PYSUFFIX"
echo "[BUILD_NATIVE] pybind11 include: $PYBIND11INC"

# Detect Zen 2 support in g++
_march_v23="znver2"
if ! echo | g++ -march=znver2 -E - > /dev/null 2>&1; then
    _march_v23="native"
    echo "[BUILD_NATIVE] compiler lacks -march=znver2, using -march=native for v2/v3"
fi
_march_v23="${NATIVE_MARCH:-$_march_v23}"

build_one () {
    local ver="$1" march="$2" extra="$3"
    local out="$BUILD_DIR/native_qlr${ver}${PYSUFFIX}"
    local src="$SRC_DIR/native_qlr${ver}.cpp"
    if [ ! -f "$src" ]; then
        echo "[BUILD_NATIVE] source missing: $src" >&2
        return 1
    fi
    echo "[BUILD_NATIVE] building native_qlr${ver} -> $out"
    # -march + optional -flto for v2/v3 mirror the original build*.sh scripts
    # (see native/src/original_build*.sh).
    g++ -O3 -march="$march" $extra -DNDEBUG -std=c++17 -Wall -Wextra -fPIC -shared \
        -fvisibility=hidden \
        -I"$PYINC" -I"$PYBIND11INC" \
        "$src" -o "$out"
    ls -la "$out"
}

_only="${V:-}"
if [ -z "$_only" ] || [ "$_only" = "v1" ]; then
    build_one ""    "native"       "-mavx2 -mfma"
fi
if [ -z "$_only" ] || [ "$_only" = "v2" ]; then
    build_one "_v2" "$_march_v23"  "-mavx2 -mfma -flto -mtune=$_march_v23"
fi
if [ -z "$_only" ] || [ "$_only" = "v3" ]; then
    build_one "_v3" "$_march_v23"  "-mavx2 -mfma -mf16c -flto -mtune=$_march_v23"
fi

echo
echo "[BUILD_NATIVE] to use the fresh build:"
echo "    edit config/paths.env"
echo "    set NATIVE_MODULE_DIR=\${SUBMISSION_CODE_PKG_ROOT}/native/build"
echo "    then re-run ./VERIFY_CODE_PACKAGE.sh"
