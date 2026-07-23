# build/ — native rebuild

The three native pybind11 modules can either be used as-is from `native/prebuilt/`
or rebuilt from source in `native/src/`.

## When to rebuild

Rebuild if any of the following holds:

* The `NATIVE_PYTHON` interpreter is not CPython 3.10 (the shipped `.so` files
  are `cpython-310-x86_64-linux-gnu.so`).
* The CPU is not x86-64.
* The CPU is x86-64 but does not implement AVX2 + FMA (v1 needs AVX2 + FMA;
  v2 and v3 additionally target AMD Zen 2 — v3 additionally uses F16C).
* You want to observe timing differences between machines using host-specific
  `-march=native`.

Otherwise the prebuilt binaries in `native/prebuilt/` are functionally
identical (SHA256 recorded in `manifests/NATIVE_BUILD_MANIFEST.md`).

## How

```bash
# Configure NATIVE_PYTHON and NATIVE_MODULE_DIR (recommended value below).
edit config/paths.env    # set NATIVE_MODULE_DIR=${SUBMISSION_CODE_PKG_ROOT}/native/build

# Install the tiny build-time requirement in the native env:
"${NATIVE_PYTHON}" -m pip install pybind11

# Build:
./build/BUILD_NATIVE.sh
```

`BUILD_NATIVE.sh` writes `native_qlr{,_v2,_v3}<EXT_SUFFIX>.so` into
`native/build/`. Point `NATIVE_MODULE_DIR` there and re-run
`./VERIFY_CODE_PACKAGE.sh` — it will import the fresh modules and confirm
`doc_ntotal`, `iq_ntotal`, `dim`, and `doc_entry_point` are consistent with
the `NATIVE_EXPORT_DIR` you configured.

## Compiler flags reproduced from the original build

The shipped `native/src/original_build{,_v2,_v3}.sh` scripts are byte-identical
copies of the original `build/build*.sh` scripts (SHA256 pairs in
`manifests/COPY_MANIFEST.tsv`). `BUILD_NATIVE.sh` reproduces the flag matrix:

| Module | -O | -march | -mavx2 | -mfma | -mf16c | -flto | -std |
|---|---|---|---|---|---|---|---|
| native_qlr    (v1) | 3 | native   | yes | yes | -   | -   | c++17 |
| native_qlr_v2      | 3 | znver2\* | yes | yes | -   | yes | c++17 |
| native_qlr_v3      | 3 | znver2\* | yes | yes | yes | yes | c++17 |

\* When the compiler lacks `-march=znver2`, `BUILD_NATIVE.sh` transparently
falls back to `-march=native`; override with `NATIVE_MARCH=native ./BUILD_NATIVE.sh`.

The prebuilt `.so` files have no RPATH/RUNPATH and depend only on
`libstdc++`, `libm`, `libgcc_s`, `libc`, and the dynamic loader (checked via
`ldd`).

## Build only one module

```bash
V=v3 ./build/BUILD_NATIVE.sh
```

## Cleanup

`native/build/` is just a directory of `.so` files. Delete it to force a
full clean rebuild.
