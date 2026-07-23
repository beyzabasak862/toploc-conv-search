#!/bin/bash
# common/timestamp_helpers.sh — timestamp + output-dir helpers. Source; do not execute.
# No absolute paths.

now_stamp() { date -u +%Y%m%dT%H%M%SZ; }

# make_result_dir <PKG_ROOT> <result_id>
#   Creates <OUTPUT_ROOT>/<result_id>/<STAMP>/ and echoes it. Refuses to overwrite.
make_result_dir() {
    local pkg_root="$1" rid="$2"
    local stamp; stamp="$(now_stamp)"
    local root="${OUTPUT_ROOT:-${pkg_root}/outputs}"
    local out="${root}/${rid}/${stamp}"
    if [ -e "$out" ]; then
        echo "[timestamp_helpers] ERROR: output dir already exists: $out" >&2
        return 1
    fi
    mkdir -p "$out"
    echo "$out"
}

# make_campaign_dir <PKG_ROOT>
#   Creates <OUTPUT_ROOT>/campaign_<STAMP>/ used by RUN_ALL.sh.
make_campaign_dir() {
    local pkg_root="$1"
    local stamp; stamp="$(now_stamp)"
    local root="${OUTPUT_ROOT:-${pkg_root}/outputs}"
    local out="${root}/campaign_${stamp}"
    if [ -e "$out" ]; then
        echo "[timestamp_helpers] ERROR: campaign dir already exists: $out" >&2
        return 1
    fi
    mkdir -p "$out"
    echo "$out"
}

# emit_environment_snapshot <out_dir>
#   Writes environment.txt into the given directory.
emit_environment_snapshot() {
    local outd="$1"
    local envf="${outd}/environment.txt"
    {
        echo "# environment.txt — captured $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "hostname: $(hostname 2>/dev/null || echo unknown)"
        echo "user: $(whoami)"
        echo "cwd: $(pwd)"
        echo "date_utc: $(date -u)"
        echo "date_local: $(date)"
        echo "uname: $(uname -a)"
        echo
        echo "# --- loadavg ---"
        cat /proc/loadavg 2>/dev/null || echo "unavailable"
        echo
        echo "# --- memory ---"
        free -h 2>/dev/null || echo "unavailable"
        echo
        echo "# --- CPU (top of lscpu) ---"
        lscpu 2>/dev/null | head -25 || echo "unavailable"
        echo
        echo "# --- CPU affinity (of this shell) ---"
        (taskset -p $$ 2>/dev/null) || echo "unknown"
        echo
        echo "# --- Thread env vars ---"
        for v in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS \
                 PYTHONDONTWRITEBYTECODE CORE; do
            echo "$v=${!v:-<unset>}"
        done
        echo
        echo "# --- Python interpreters ---"
        for name in HYBRID_PY NATIVE_PY REPORT_PY; do
            val="${!name}"
            echo "$name=$val"
            if [ -x "$val" ]; then
                "$val" --version 2>&1 | sed 's/^/  version: /'
                "$val" -c 'import sys; print("  exe:", sys.executable)'
            fi
        done
    } > "$envf"
    echo "$envf"
}

# emit_pip_versions <python_exe> <out_file>
emit_pip_versions() {
    local py="$1" outf="$2"
    if [ ! -x "$py" ]; then
        echo "python interpreter not executable: $py" > "$outf"
        return 0
    fi
    "$py" - <<'PY' > "$outf" 2>&1 || true
import importlib, sys
print("python:", sys.version.split()[0], "exe:", sys.executable)
for mod in ["numpy","faiss","joblib","threadpoolctl","pyarrow","torch","pybind11","pandas","sklearn"]:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"{mod}: {v}")
    except Exception as e:
        print(f"{mod}: NOT AVAILABLE ({e.__class__.__name__})")
PY
}

# sha256_or_missing <file>
sha256_or_missing() {
    if [ -e "$1" ]; then sha256sum "$1" | awk '{print $1}'; else echo "MISSING"; fi
}
