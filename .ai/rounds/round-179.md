# Round 179 — Cross-Platform Verification (ARM64 NEON+SVE, RISC-V RVV, x86 AVX2)

**Signed**: Header (Planned) | **Date**: 2026-05-29 | **Status**: PENDING

## Objective

Verify all CPU SIMD backends are functional and build correctly. Produce a compatibility matrix showing SIMD level support per architecture, including known issues.

## Architecture Overview

```
tsac-ng CPU SIMD Matrix
┌─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Architecture    │ Baseline │ SIMD-1   │ SIMD-2   │ SIMD-3   │ Notes    │
├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ x86-64          │ scalar   │ SSE4.2   │ AVX+FMA  │ AVX2     │ AVX-512 ⚠️│
│ ARM64 (aarch64) │ scalar   │ NEON     │ SVE      │ —        │          │
│ RISC-V (rv64gc) │ scalar   │ RVV 1.0  │ —        │ —        │          │
└─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

## Task T1: x86-64 SIMD Build + Verification

**Current platform** (native x86-64): Build and test all SIMD levels.

```bash
# Check CPU features
grep -m1 'flags' /proc/cpuinfo | grep -oE 'avx[^ ]*|sse[^ ]*|fma' | sort -u

# Build with all x86-64 SIMD levels
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) 2>&1

# Verify SIMD dispatch works
./build/tsac-ng --help 2>&1 | head -5

# Decode test file with verbose output (shows SIMD level)
./build/tsac-ng -v d /tmp/test_fast.txc /tmp/x86_out.wav 2>&1

# Check which SIMD level is active in cpu_decoder debug output
# (look for "AVX512", "AVX2", "AVX", "scalar" in stderr)
```

**AVX-512 known issue**: Per the state.json and R145 findings:
```
AVX-512 conv kernels broken (10-70× amplification vs scalar). Scalar fallback forced.
```
Verify this is still active:
```bash
# Force AVX-512 path via environment or code (if CPUID bypass exists)
# Compare output of AVX-512 path vs scalar path:
#   conv1d_avx512 (cpu_simd.inc:342-385) vs conv1d_s (cpu_simd.inc:12-23)
# Expected: AVX-512 flagged as ⚠️ with scalar fallback
```

**AVX2 verification**: AVX2 kernels (cpu_simd.inc:200-339) should be functional:
- `conv1d_avx2` (line 200)
- `convt1d_avx2` (line 246)
- `snake_avx2` (line 292)
- `add_avx2` (line 327)

**Exact SIMD level detection code** (cpu_decoder.c:79-90):
```c
typedef enum { SIMD_SCALAR, SIMD_SSE42, SIMD_AVX, SIMD_AVX2, SIMD_AVX512 } SimdLevel;
```

## Task T2: ARM64 NEON+SVE Cross-Compile Verification

Since we are on x86-64 host, use cross-compilation for ARM64 verification.

**Check if ARM64 cross-toolchain is available**:
```bash
# Check for ARM64 cross-compiler
which aarch64-linux-gnu-gcc 2>/dev/null && echo "ARM64 GCC available" || echo "ARM64 GCC NOT available"
which aarch64-linux-gnu-g++ 2>/dev/null && echo "ARM64 G++ available" || echo "ARM64 G++ NOT available"

# Check for QEMU user-mode emulation
which qemu-aarch64 2>/dev/null && echo "QEMU aarch64 available" || echo "QEMU aarch64 NOT available"

# Check for SVE support in QEMU
qemu-aarch64 --version 2>&1 | head -1
```

**If cross-toolchain + QEMU available**:
```bash
# SVE cross-compile
cmake -B build_arm64 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-arm64.cmake \
  -DCMAKE_BUILD_TYPE=Release 2>&1
cmake --build build_arm64 -j$(nproc) 2>&1

# Run in QEMU (with SVE vector length 256)
qemu-aarch64 -cpu max,sve256=on ./build_arm64/tsac-ng --help 2>&1

# Test decode in QEMU
qemu-aarch64 -cpu max,sve256=on ./build_arm64/tsac-ng -v d /tmp/test_fast.txc /tmp/arm64_out.wav 2>&1
```

**If cross-toolchain NOT available**:
```bash
# At minimum, verify ARM64 NEON code compiles by checking preprocessor guards
echo "=== ARM64 NEON code check ==="
gcc -E -P -dM -x c /dev/null | grep -c __aarch64__ || echo "__aarch64__ not defined (expected on x86 host)"

# Verify header guards are correct
grep -n '__aarch64__\|__ARM_FEATURE_SVE\|arm_sve' src/arch/arm/cpu_arm.c
echo "---"
echo "ARM64 NEON functions:"
grep -c 'void.*_neon(' src/arch/arm/cpu_arm.c
echo "ARM64 SVE functions:"
grep -c 'void.*_sve(' src/arch/arm/cpu_arm.c
```

**Expected ARM64 capabilities**:
| Feature | Source | Status |
|---------|--------|--------|
| NEON add | `add_neon()` cpu_arm.c:38-42 | ✅ Built-in ARM64 ISA |
| NEON conv1d | `conv1d_neon()` cpu_arm.c:45-64 | ✅ FMLA inner loop |
| NEON group_norm | `group_norm_neon()` cpu_arm.c:79-107 | ✅ Vectorized |
| SVE add | `add_sve()` cpu_arm.c:113-120 | ✅ `ifdef __ARM_FEATURE_SVE` |
| SVE conv1d | `conv1d_sve()` cpu_arm.c:122-144 | ✅ `svmla_f32_m` fused |
| SVE snake | `snake_sve()` cpu_arm.c:146-154 | ✅ (sinf scalar fallback) |
| Runtime detection | `cpu_arch_init()` cpu_arm.c:22-26 | ✅ `getauxval(AT_HWCAP)` |
| Runtime SVE detect | `cpu_arch_has_sve()` cpu_arm.c:33-35 | ✅ |

## Task T3: RISC-V RVV Cross-Compile Verification

**Check RISC-V cross-toolchain**:
```bash
which riscv64-unknown-linux-gnu-gcc 2>/dev/null && echo "RISC-V GCC available" || echo "RISC-V GCC NOT available"
which qemu-riscv64 2>/dev/null && echo "QEMU riscv64 available" || echo "QEMU riscv64 NOT available"
```

**If available**:
```bash
# RVV cross-compile (requires RVV 1.0 toolchain)
cmake -B build_riscv64 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-riscv64.cmake \
  -DCMAKE_BUILD_TYPE=Release 2>&1
cmake --build build_riscv64 -j$(nproc) 2>&1

# Run in QEMU with RVV
qemu-riscv64 -cpu rv64,v=true,vlen=256 ./build_riscv64/tsac-ng --help 2>&1
```

**If NOT available**: Verify code quality by inspection:
```bash
echo "=== RISC-V RVV code check ==="
# Check RVV function count
grep -c 'void.*_rvv(' src/arch/riscv/cpu_riscv.c
echo "RVV functions  : $(grep -c '__riscv_v' src/arch/riscv/cpu_riscv.c)"

# Check runtime detection
grep -A5 'cpu_arch_init' src/arch/riscv/cpu_riscv.c

# Check dispatch
grep -B2 -A5 'conv1d_riscv' src/arch/riscv/cpu_riscv.c | head -10

# Verify scalar fallback chain
echo "Scalar fallback routes:"
grep -c 'conv1d_scalar' src/arch/riscv/cpu_riscv.c
```

**Expected RISC-V capabilities**:
| Feature | Source | Status |
|---------|--------|--------|
| Scalar fallback | `conv1d_scalar()` cpu_riscv.c:48-62 | ✅ Always available |
| RVV conv1d | `conv1d_rvv()` cpu_riscv.c:90-120 | ✅ `__riscv_vfmacc` vectorized |
| RVV snake | `snake_rvv()` cpu_riscv.c:126-160 | ✅ sinf scalar fallback |
| RVV add | `add_rvv()` cpu_riscv.c:165-177 | ✅ |
| Runtime RVV detect | `cpu_arch_init()` cpu_riscv.c:21-31 | ✅ `/proc/cpuinfo` |
| Runtime RVV check | `riscv_rvv_available()` cpu_riscv.c:40-42 | ✅ |

## Task T4: Build Compatibility Matrix

Create a comprehensive matrix document at `/tmp/r179_compatibility_matrix.md`:

```markdown
# tsac-ng CPU SIMD Compatibility Matrix
Generated: <date>

## Build Results

| Platform | Compiler | Build Status | Runtime | Notes |
|----------|----------|-------------|---------|-------|
| x86-64 (native) | gcc/X.Y | ✅ PASS | ✅ RUNS | Current build host |
| x86-64 (AVX-512) | gcc/X.Y | ✅ BUILDS | ⚠️ SCALAR FALLBACK | conv/convtr kernels 10-70× amplified |
| ARM64 NEON | aarch64-gcc/X.Y | ❓/✅ | ❓/✅ | <cross-compile result> |
| ARM64 SVE | aarch64-gcc/X.Y | ❓/✅ | ❓/✅ | <cross-compile result> |
| RISC-V RVV | riscv64-gcc/X.Y | ❓/✅ | ❓/✅ | <cross-compile result> |

## Kernel Coverage per Platform

### conv1d
| Platform | Kernel | Lines | SIMD Width | FMA | Status |
|----------|--------|-------|-----------|-----|--------|
| x86 scalar | conv1d_s | 12-23 | 1 | no | ✅ |
| x86 AVX | conv1d_avx | 157-199 | 8 | _mm256_fmadd_ps | ✅ |
| x86 AVX2 | conv1d_avx2 | 201-244 | 8 | _mm256_fmadd_ps | ✅ |
| x86 AVX-512 | conv1d_avx512 | 342-385 | 16 | _mm512_fmadd_ps | ⚠️ |
| ARM NEON | conv1d_neon | 45-64 | 4 | vfmaq_f32 | ✅ |
| ARM SVE | conv1d_sve | 122-144 | scalable | svmla_f32_m | ✅ |
| RISC-V RVV | conv1d_rvv | 90-120 | scalable | __riscv_vfmacc | ✅ |

### convt1d (conv transpose)
| Platform | Kernel | Lines | SIMD Width | FMA | Status |
|----------|--------|-------|-----------|-----|--------|
| x86 scalar | convt1d_s | 58-72 | 1 | no | ✅ |
| x86 AVX2 | convt1d_avx2 | 246-290 | 8 | _mm256_fmadd_ps | ✅ |
| x86 AVX-512 | convt1d_avx512 | 387-429 | 16 | _mm512_fmadd_ps | ⚠️ |

### snake activation
| Platform | Kernel | Lines | SIMD Width | sinf intrinsic | Status |
|----------|--------|-------|-----------|---------------|--------|
| x86 scalar | snake_s | 90-97 | 1 | sinf() | ✅ |
| x86 AVX | snake_avx | 292-338 | 8 | scalar fallback | ✅ |
| x86 AVX-512 | snake_avx512 | 431-469 | 16 | scalar fallback | ✅ |
| ARM NEON | snake_neon | 67-76 | scalar | sinf() | ✅ |
| RISC-V RVV | snake_rvv | 126-160 | scalable | scalar fallback | ✅ |

### group_norm
| Platform | Kernel | Lines | SIMD | Status |
|----------|--------|-------|------|--------|
| x86 scalar | gn_s | 74-88 | 1 | ✅ |
| ARM NEON | group_norm_neon | 79-107 | 4 | ✅ |

## CPU Feature Detection
| Platform | Method | Implementation | Status |
|----------|--------|---------------|--------|
| x86-64 | CPUID | cpu_decoder.c:27-30 | ✅ |
| ARM64 | getauxval(AT_HWCAP) | cpu_arm.c:22-26 | ✅ |
| RISC-V | /proc/cpuinfo | cpu_riscv.c:21-31 | ✅ |

## Known Issues
1. **AVX-512 conv1d/convt1d**: 10-70× amplification compared to scalar.
   Root cause: gather16() emulation via scalar tmp buffer is correct, but FMA accumulation
   across kernel elements amplifies when Ci is not 16-aligned. Workaround: scalar fallback.
   See: src/cpu_simd.inc:342-385, src/cpu_simd.inc:387-429
   
2. **sinf in SIMD paths**: No SIMD intrinsic for sinf() exists on any platform.
   All SIMD snake implementations fall back to scalar sinf() per element.
   This is acceptable as snake is not the dominant cost (~5% of decode time).

3. **RISC-V toolchain maturity**: RVV 1.0 C intrinsics require gcc 12+/llvm 15+.
   Older toolchains will fall back to scalar.
   See: src/arch/riscv/cpu_riscv.c:83-84 (`#ifdef __riscv_v`)

## Recommendations
1. Fix AVX-512 conv1d by aligning Ci to 16 in caller (pad input channels)
2. Add ARM64 SVE CI testing via GitHub Actions self-hosted runner
3. Consider RVV length-agnostic optimization for snake sinf
```

## Task T5: Update CMakeLists.txt Cross-Compile Support

Verify cross-compile toolchain files exist:
```bash
ls -la cmake/Toolchain-arm64.cmake 2>/dev/null || echo "MISSING: cmake/Toolchain-arm64.cmake"
ls -la cmake/Toolchain-riscv64.cmake 2>/dev/null || echo "MISSING: cmake/Toolchain-riscv64.cmake"

# If missing, at minimum verify CMakeLists.txt references them correctly
grep -n 'Toolchain' CMakeLists.txt
```

**Cross-compile notes to add to CMakeLists.txt if needed**:
```cmake
# ARM64 with SVE:
# cmake -B build_arm64 -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-arm64.cmake -DCMAKE_C_FLAGS="-march=armv8-a+sve"
#
# RISC-V with RVV:
# cmake -B build_riscv64 -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-riscv64.cmake -DCMAKE_C_FLAGS="-march=rv64gcv"
```

## Success Criteria

- [ ] R179-T1: x86-64 build passes all SIMD levels
- [ ] R179-T1: AVX-512 known issue re-verified and documented
- [ ] R179-T2: ARM64 NEON codes compile-checked (cross or native)
- [ ] R179-T2: ARM64 SVE codes compile-checked
- [ ] R179-T3: RISC-V RVV codes compile-checked
- [ ] R179-T4: Compatibility matrix created at `/tmp/r179_compatibility_matrix.md`
- [ ] R179-T5: Cross-compile toolchain files verified
- [ ] No code regressions on native x86-64 build
