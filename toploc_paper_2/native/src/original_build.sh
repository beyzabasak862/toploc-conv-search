#!/bin/bash
set -euo pipefail
# ==================== CLAUDE IMPROVEMENT START ====================
# Build the native_qlr pybind11 module.
# ==================== CLAUDE IMPROVEMENT END ====================
WS="/home/toploc1/Datasets/toploc1/HNSW/msmarco_HNSW/claude_qlr_diagnostics/paper2_final_track/faithful_msmarco_v1_20260719_042528/native_qlr_optimization/20260719_190628"
PY=/home/fatemeh/anaconda3/envs/zeroec/bin/python
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_path('include'))")
PYSUFFIX=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYBIND11INC=$($PY -c "import pybind11; print(pybind11.get_include())")

echo "python include: $PYINC"
echo "extension suffix: $PYSUFFIX"
echo "pybind11 include: $PYBIND11INC"

OUT="$WS/build/native_qlr${PYSUFFIX}"
echo "Building -> $OUT"

g++ -O3 -march=native -mavx2 -mfma -DNDEBUG -std=c++17 -Wall -Wextra -fPIC -shared \
    -fvisibility=hidden \
    -I"$PYINC" -I"$PYBIND11INC" \
    "$WS/src/native_qlr.cpp" \
    -o "$OUT"

echo "Build succeeded: $(ls -la $OUT | awk '{print $5, $NF}')"
