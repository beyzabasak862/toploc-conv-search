# NATIVE_BUILD_MANIFEST.md

Everything needed to rebuild the three native pybind11 modules on a fresh
machine.

## Sources

| Module | Source file (bundled) | SHA256 | Size |
|---|---|---|---|
| native_qlr (v1) | `native/src/native_qlr.cpp`    | `f413e1bce62a4fc19549d3fa616b2ab797f6beee1fb210816bab98be8a42c5b6` | 26 863 B |
| native_qlr_v2   | `native/src/native_qlr_v2.cpp` | `d7c8fedc5eedbe8f94bf17fceadce2f6e9e66cc5a68b77bad37574a5e50865e0` | 30 527 B |
| native_qlr_v3   | `native/src/native_qlr_v3.cpp` | `afc98a0efefd02cd8f9ea9f2c99fd29791f0c86f37bac98c165e6619a172caa5` | 33 702 B |

## Original build scripts (bundled for provenance)

| Original | Bundled copy | SHA256 |
|---|---|---|
| `build/build.sh`    | `native/src/original_build.sh`    | `6a2d4d4826fd16c900d49a45db0d2ae5f99997a72cca3d76636d7d8ef6a3051d` |
| `build/build_v2.sh` | `native/src/original_build_v2.sh` | `96ca3974f9665714e6f8649c75dbb3e79043c0bd729d3de7f9b94118daea14a2` |
| `build/build_v3.sh` | `native/src/original_build_v3.sh` | `d235021b42add555f59297565a6948f27bfb44a3032a97565ec2b5c0515648c3` |

## Portable rebuild

`build/BUILD_NATIVE.sh` reproduces the same compiler-flag matrix but sources
paths from `config/paths.env` instead of hard-coded workspace absolutes.

## Compiler + build-time requirements

| Requirement | Reference value |
|---|---|
| C++ compiler | g++ >= 11 with C++17 support (reference: g++ 12.2.0) |
| CPU features (v1) | AVX2 + FMA |
| CPU features (v2, v3) | AVX2 + FMA (v3 additionally uses F16C) |
| -march (v1) | `native` |
| -march (v2, v3) | `znver2` (falls back to `native` if unsupported) |
| Optimisation | `-O3 -DNDEBUG -flto` (v2, v3 only for -flto) |
| Visibility | `-fvisibility=hidden` |
| Python ABI | CPython 3.10 (`cpython-310-x86_64-linux-gnu`) |
| Include: python | `sysconfig.get_path('include')` — resolved at build time |
| Include: pybind11 | `pybind11.get_include()` — install with `pip install pybind11` |

## Expected output filenames

`native_qlr{,_v2,_v3}<EXT_SUFFIX>` where `EXT_SUFFIX` comes from
`sysconfig.get_config_var('EXT_SUFFIX')` — for CPython 3.10 x86-64:
`.cpython-310-x86_64-linux-gnu.so`.

## Prebuilt binaries shipped

| File | SHA256 | Size |
|---|---|---|
| `native/prebuilt/native_qlr.cpython-310-x86_64-linux-gnu.so`    | `0825c10d7bc370c2471a2470f5fea32a3349a1cf793457543ec7690429c38fa4` | 308 112 B |
| `native/prebuilt/native_qlr_v2.cpython-310-x86_64-linux-gnu.so` | `aa8718820898e701dde5bad45fcc1d960f23b2898f08594eb643aec7b13e1368` | 288 296 B |
| `native/prebuilt/native_qlr_v3.cpython-310-x86_64-linux-gnu.so` | `634400af269916c9c3be7521392691639abfa491b1946da3bf29dc5795de9211` | 288 384 B |

### ldd (system libraries only)

```
libstdc++.so.6
libm.so.6
libgcc_s.so.1
libc.so.6
ld-linux-x86-64.so.2
```

No `RPATH` or `RUNPATH` is set — none of the `.so` files carry a hard-coded
non-portable path. Verified with `readelf -d`.

## Module import test

Run either automatically via `./VERIFY_CODE_PACKAGE.sh`, or manually:

```bash
NATIVE_MODULE_DIR=${SUBMISSION_CODE_PKG_ROOT}/native/prebuilt \
NATIVE_EXPORT_DIR=/path/to/native_export \
"${NATIVE_PYTHON}" - <<'PY'
import os, sys
sys.path.insert(0, os.environ["NATIVE_MODULE_DIR"])
import native_qlr, native_qlr_v2, native_qlr_v3
ix = native_qlr_v3.NativeQLR(os.environ["NATIVE_EXPORT_DIR"])
print("v3 ntotal=", ix.doc_ntotal(), "iq=", ix.iq_ntotal(),
      "dim=", ix.dim(), "ep=", ix.doc_entry_point())
PY
```

Expected on the reference machine: `ntotal=500000 iq=~808k dim=1024 ep=<int>`.
