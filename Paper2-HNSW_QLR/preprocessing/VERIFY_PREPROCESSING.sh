#!/bin/bash
# preprocessing/VERIFY_PREPROCESSING.sh
#
# Static validation of the bundled preprocessing producers. It NEVER starts a
# preprocessing job (no index build, no PCA fit, no embedding generation, no
# EP-table build, no exact search, no large dataset load).
#
# Steps:
#   1. bash -n on every shell script under preprocessing/
#   2. Python compileall on every bundled .py
#   3. producer import tests (src helpers import; producers compile)
#   4. recursive local-dependency verification (every `from src...` resolves)
#   5. forbidden absolute-path scan (executed code)
#   6. symlink-escape scan
#   7. oversized-file scan (>50 MB)
#   8. secret scan
#   9. config-variable validation (PREPROC_* schema present)
#  10. command-printing validation (prints the exact command per producer)

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${HERE}/.." && pwd)"
PRE="$HERE"
VAL_DIR="${PRE}/validation"
mkdir -p "$VAL_DIR"
REPORT="${VAL_DIR}/VERIFY_PREPROCESSING_$(date -u +%Y%m%dT%H%M%SZ).log"

pass=0; warn=0; fail=0
_p() { echo "PASS: $*" | tee -a "$REPORT"; pass=$((pass+1)); }
_w() { echo "WARN: $*" | tee -a "$REPORT"; warn=$((warn+1)); }
_f() { echo "FAIL: $*" | tee -a "$REPORT"; fail=$((fail+1)); }

echo "== VERIFY_PREPROCESSING  $(date -u)  host=$(hostname)  pre=$PRE" | tee "$REPORT"

# Try to load the central config (for HYBRID_PYTHON + PREPROC vars). Optional.
HYBRID_PY_FALLBACK="/usr/bin/python3"
if [ -f "${PKG_ROOT}/config/paths.env" ]; then
    # shellcheck source=/dev/null
    . "${PKG_ROOT}/config/load_config.sh" 2>/dev/null || true
fi
PY="${HYBRID_PYTHON:-$HYBRID_PY_FALLBACK}"
[ -x "$PY" ] || PY="$HYBRID_PY_FALLBACK"

PRODUCERS=(build_index build_query_log_pca build_ep_table flat_index_search_acc)

# ---------------------------------------------------------------- 1. bash -n
echo | tee -a "$REPORT"; echo "-- 1. bash -n --" | tee -a "$REPORT"
found_sh=0
while IFS= read -r -d '' sh; do
    found_sh=1
    if bash -n "$sh" 2>>"$REPORT"; then _p "bash -n $sh"; else _f "bash -n $sh"; fi
done < <(find "$PRE" -type f -name '*.sh' -print0)
[ "$found_sh" = "1" ] || _p "bash -n (this script only)"

# ---------------------------------------------------------------- 2. compileall
echo | tee -a "$REPORT"; echo "-- 2. python compileall ($PY) --" | tee -a "$REPORT"
if PYTHONDONTWRITEBYTECODE=1 "$PY" -m compileall -q -b "$PRE/scripts" "$PRE/src" 2>>"$REPORT"; then
    _p "compileall ok under $PY"
else
    _f "compileall failed under $PY"
fi

# ---------------------------------------------------------------- 3+4. imports & local deps
echo | tee -a "$REPORT"; echo "-- 3+4. producer imports + local-dependency resolution --" | tee -a "$REPORT"
if PYTHONDONTWRITEBYTECODE=1 "$PY" - "$PRE" <<'PY' 2>>"$REPORT"
import sys, os
pre = sys.argv[1]
sys.path.insert(0, pre)  # so `import src` resolves to preprocessing/src
# src helpers have no side effects at import
from src.data_loading import load_embeddings_from_parquets, l2_normalize
from src.indexing import build_hnsw_index, save_index, save_ids, load_index
# each producer must compile and its `from src...` targets must exist
import ast
producers = ["build_index", "build_query_log_pca", "build_ep_table", "flat_index_search_acc"]
for name in producers:
    path = os.path.join(pre, "scripts", name + ".py")
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    compile(open(path).read(), path, "exec")
    # verify every `from src.X import a,b` symbol resolves
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            mod = __import__(node.module, fromlist=[a.name for a in node.names])
            for a in node.names:
                assert hasattr(mod, a.name), f"{name}: {node.module}.{a.name} missing"
print("producer imports + local deps OK for:", producers)
PY
then
    _p "producer imports + local-dependency resolution"
else
    _f "producer imports / local-dependency resolution"
fi

# ---------------------------------------------------------------- 5. forbidden path scan
echo | tee -a "$REPORT"; echo "-- 5. forbidden absolute-path scan (executed code) --" | tee -a "$REPORT"
FORBIDDEN='/home/toploc1/Datasets/toploc1/HNSW/msmarco_HNSW'
hits="$(grep -rnF "$FORBIDDEN" "$PRE/scripts" "$PRE/src" --include='*.py' 2>/dev/null || true)"
if [ -z "$hits" ]; then
    _p "no original-repository path in executed preprocessing code"
else
    echo "$hits" >> "$REPORT"; _f "original-repository path found in executed code"
fi

# ---------------------------------------------------------------- 6. symlink escape
echo | tee -a "$REPORT"; echo "-- 6. symlink-escape scan --" | tee -a "$REPORT"
sl="$(find "$PRE" -type l -printf '%p %l\n' | awk '$2 ~ /^\// {print}')"
if [ -z "$sl" ]; then _p "no escaping symlinks"; else echo "$sl" >>"$REPORT"; _f "escaping symlink(s)"; fi

# ---------------------------------------------------------------- 7. oversized files
echo | tee -a "$REPORT"; echo "-- 7. oversized-file scan (>50 MB) --" | tee -a "$REPORT"
big="$(find "$PRE" -type f -size +50M -printf '%s %p\n')"
if [ -z "$big" ]; then _p "no files over 50 MB"; else echo "$big" >>"$REPORT"; _f "file(s) over 50 MB"; fi

# ---------------------------------------------------------------- 8. secret scan
echo | tee -a "$REPORT"; echo "-- 8. secret scan --" | tee -a "$REPORT"
sec="$(grep -rnE '(-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[bpar]-|ghp_|glpat-|AKIA[0-9A-Z]{16}|password[[:space:]]*[:=])' "$PRE" 2>/dev/null | grep -v /validation/ || true)"
if [ -z "$sec" ]; then _p "no secrets"; else echo "$sec" >>"$REPORT"; _f "possible secret(s)"; fi

# ---------------------------------------------------------------- 9. config-variable validation
echo | tee -a "$REPORT"; echo "-- 9. config-variable validation --" | tee -a "$REPORT"
EXAMPLE="${PKG_ROOT}/config/paths.env.example"
REQ_VARS=(PREPROC_OUTPUT_ROOT PREPROC_DOC_EMB_DIR PREPROC_TRAIN_QUERY_DIR PREPROC_FLAT_INDEX HYBRID_DOC_INDEX DEV_QUERY_DIR)
for v in "${REQ_VARS[@]}"; do
    if grep -qE "^[#]?${v}=" "$EXAMPLE" 2>/dev/null; then
        _p "config schema documents $v"
    else
        _f "config schema missing $v in config/paths.env.example"
    fi
done

# ---------------------------------------------------------------- 10. command printing
echo | tee -a "$REPORT"; echo "-- 10. command-printing validation (NO job started) --" | tee -a "$REPORT"
{
    echo "Recommended order + exact commands (env from config/paths.env):"
    echo "  1) $PY $PRE/scripts/build_index.py            # needs PREPROC_DOC_EMB_DIR, PREPROC_OUTPUT_ROOT"
    echo "  2) $PY $PRE/scripts/build_query_log_pca.py    # needs PREPROC_TRAIN_QUERY_DIR, PREPROC_OUTPUT_ROOT"
    echo "  3) $PY $PRE/scripts/build_ep_table.py         # needs PREPROC_TRAIN_QUERY_DIR, HYBRID_DOC_INDEX, PREPROC_OUTPUT_ROOT"
    echo "  4) $PY $PRE/scripts/flat_index_search_acc.py  # needs DEV_QUERY_DIR, PREPROC_FLAT_INDEX, PREPROC_OUTPUT_ROOT"
} | tee -a "$REPORT"
_p "command-printing validation (no preprocessing job started)"

echo | tee -a "$REPORT"
echo "=== SUMMARY  PASS=$pass  WARN=$warn  FAIL=$fail ===" | tee -a "$REPORT"
[ "$fail" -gt 0 ] && { echo "(inspect $REPORT)"; exit 1; }
exit 0
