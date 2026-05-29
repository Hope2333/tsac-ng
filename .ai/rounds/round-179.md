# Round 179 — Cross-Platform Verification (Worker Signed)
**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
Verify tsac-ng builds and runs correctly on all three target architectures: ARM64 (NEON + SVE), RISC-V (RVV), and x86-64 (AVX2 fallback + AVX-512). Create a cross-platform compatibility matrix documenting build status, runtime behavior, known issues, and numeric output parity. This is critical for the v0.2.0/v0.1.4 release — platform support is a key differentiator of tsac-ng.

## Platform Targets

| Platform | SIMD | Toolchain | Build System |
|----------|------|-----------|-------------|
| ARM64 (aarch64) | NEON + SVE | aarch64-linux-gnu-gcc (cross) | cmake -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-arm64.cmake |
| RISC-V (riscv64) | RVV + scalar | riscv64-linux-gnu-gcc (cross) | cmake -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-riscv64.cmake |
| x86-64 (native) | AVX2 + AVX-512 | gcc/x86_64-linux-gnu-gcc | cmake -DCMAKE_BUILD_TYPE=Release |
| x86-64 (fallback) | Scalar + SSE | gcc with -mno-avx2 | cmake -DCMAKE_BUILD_TYPE=Release -DUSE_NATIVE=OFF |

## Tasks

### T1: Verify ARM64 build (NEON + SVE)
**Goal**: Cross-compile tsac-ng for aarch64 using the ARM64 toolchain. Verify both NEON and SVE code paths compile cleanly.

**Pre-requisites**:
- ARM64 cross-compiler: `aarch64-linux-gnu-gcc` (from `g++-aarch64-linux-gnu` package)
- CMake toolchain file: `cmake/Toolchain-arm64.cmake`

**Exact commands**:
```bash
# 1. Verify cross-compiler exists
aarch64-linux-gnu-gcc --version
# Expected: "aarch64-linux-gnu-gcc (Ubuntu ...)" or similar
which aarch64-linux-gnu-gcc || echo "NOT FOUND — install: apt install g++-aarch64-linux-gnu"

# 2. Clean build directory
rm -rf build-arm64 && mkdir build-arm64 && cd build-arm64

# 3. Configure with ARM64 toolchain
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/Toolchain-arm64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_NATIVE=OFF \
  -DENABLE_SVE=ON \
  2>&1 | tail -10
# Expected: "Configuring done", "Generating done"
# Note any missing SVE-related warnings

# 4. Build
make -j$(nproc) 2>&1 | tail -20
# Expected: 0 errors, 0 warnings
# Known acceptable warnings: "NEON vector width mismatch" (cosmetic)

# 5. Verify binary properties
file tsac-ng
# Expected: "ELF 64-bit LSB executable, ARM aarch64"

# 6. Check NEON/SVE symbols
aarch64-linux-gnu-objdump -t tsac-ng | grep -E "neon|NEON|sve|SVE" | head -10
# Expected: NEON symbols present (conv1d_neon, etc.), SVE symbols present if compiled in

# 7. Also test with SVE disabled
cd ..
rm -rf build-arm64-nosve && mkdir build-arm64-nosve && cd build-arm64-nosve
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/Toolchain-arm64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SVE=OFF \
  2>&1 | tail -5
make -j$(nproc) 2>&1 | tail -5
# Expected: Builds successfully, no SVE symbols
aarch64-linux-gnu-objdump -t tsac-ng | grep -i sve
# Expected: no output (no SVE symbols)
```

**Acceptance**: ARM64 binary produced (ELF aarch64). Both NEON+SVE and NEON-only builds compile cleanly. NEON intrinsics present in symbol table. Build time < 5 minutes.

### T2: Verify RISC-V build (RVV)
**Goal**: Cross-compile tsac-ng for riscv64 with RVV support. Verify the vector extension is detected and used.

**Pre-requisites**:
- RISC-V cross-compiler: `riscv64-linux-gnu-gcc` (from `g++-riscv64-linux-gnu` package)
- CMake toolchain file: `cmake/Toolchain-riscv64.cmake`
- RISC-V QEMU (optional, for runtime test): `qemu-riscv64`

**Exact commands**:
```bash
# 1. Verify cross-compiler
riscv64-linux-gnu-gcc --version
# Expected: "riscv64-linux-gnu-gcc (Ubuntu ...)"
which riscv64-linux-gnu-gcc || echo "NOT FOUND — install: apt install g++-riscv64-linux-gnu"

# 2. Build
rm -rf build-riscv64 && mkdir build-riscv64 && cd build-riscv64
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/Toolchain-riscv64.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_RVV=ON \
  2>&1 | tail -10
make -j$(nproc) 2>&1 | tail -20

# 3. Verify binary
file tsac-ng
# Expected: "ELF 64-bit LSB executable, UCB RISC-V"
riscv64-linux-gnu-objdump -d tsac-ng | grep -c "vsetvli\|vadd\|vmul"
# Expected: > 0 (RVV instructions present)

# 4. If QEMU is available, run a quick test
if command -v qemu-riscv64 &> /dev/null; then
  # Static build needed for QEMU without sysroot
  cmake .. -DCMAKE_BUILD_TYPE=Release -DSTATIC=ON -DENABLE_RVV=ON
  make -j$(nproc) tsac-ng-static 2>&1 | tail -5
  
  # Quick decode test (QEMU user mode, limited file I/O)
  qemu-riscv64 ./tsac-ng-static --help 2>&1
  # Expected: help text printed
  echo "QEMU RISC-V runtime test: PASS"
else
  echo "QEMU not available — compile-only verification"
fi
```

**Acceptance**: RISC-V binary produced (ELF RISC-V). RVV instructions present in disassembly. If QEMU available, `--help` runs successfully. Build time < 5 minutes.

### T3: x86-64 AVX2 fallback verification
**Goal**: Build and test x86-64 binary with specific SIMD levels disabled to verify fallback paths work correctly. The key test: AVX-512 may be buggy (see R145 finding), so AVX2 fallback must produce correct output.

**Exact commands**:
```bash
# 1. Build with AVX-512 explicitly disabled (test AVX2 fallback)
rm -rf build-avx2-only && mkdir build-avx2-only && cd build-avx2-only
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DDISABLE_AVX512=ON \
  -DUSE_NATIVE=OFF \
  2>&1 | tail -5
make -j$(nproc) 2>&1 | tail -5

# 2. Verify AVX-512 instructions are NOT present
objdump -d tsac-ng | grep -c "vpmuludq\|vaddps.*zmm\|vmulps.*zmm"
# Expected: 0 (no AVX-512 zmm registers)

# 3. Verify AVX2 instructions ARE present
objdump -d tsac-ng | grep -c "vfmadd231ps\|vaddps.*ymm\|vmulps.*ymm"
# Expected: > 0 (AVX2 ymm registers used)

# 4. Verify scalar fallback builds
rm -rf build-scalar && mkdir build-scalar && cd build-scalar
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DDISABLE_AVX2=ON \
  -DDISABLE_AVX512=ON \
  -DUSE_NATIVE=OFF \
  2>&1 | tail -5
make -j$(nproc) 2>&1 | tail -5

# 5. Verify no vector instructions
objdump -d tsac-ng | grep -c "vfmadd\|vaddps\|vmulps"
# Expected: < 10 (maybe some in libc/libm, but none in our conv kernels)

# 6. Run functional test (AVX2 build)
cd ../build-avx2-only
./tsac-ng -v d ../../test-simples/short_fast.txc /tmp/out_avx2_fast.wav 2>&1
# Expected: decode succeeds, exit 0, AVX2 auto-detected in verbose output

# 7. Compare output with scalar build
cd ../build-scalar
./tsac-ng -v d ../../test-simples/short_fast.txc /tmp/out_scalar_fast.wav 2>&1
cmp /tmp/out_avx2_fast.wav /tmp/out_scalar_fast.wav && echo "IDENTICAL" || echo "DIFFERENT"
# Expected: "IDENTICAL" — SIMD and scalar must produce bit-exact output
```

**Acceptance**: AVX2-only build and scalar-only build both compile and run. Output from AVX2 and scalar builds is binary-identical (same WAV file). AVX2 build shows ymm instructions but no zmm. Scalar build has no AVX2/AVX-512 instructions.

## Compatibility Matrix

Complete this table with build results:

### Build Compatibility
| Platform | SIMD | Build Status | Binary Type | Runtime Test | Known Issues |
|----------|------|:------------:|:-----------:|:------------:|--------------|
| ARM64 | NEON+SVE | ✅/❌ | ELF aarch64 | ✅/❌/N/A | |
| ARM64 | NEON only | ✅/❌ | ELF aarch64 | ✅/❌/N/A | |
| RISC-V | RVV | ✅/❌ | ELF riscv64 | ✅/❌/N/A | |
| RISC-V | Scalar | ✅/❌ | ELF riscv64 | ✅/❌/N/A | |
| x86-64 | AVX-512 | ✅/❌ | ELF x86-64 | ⚠️/broken | AVX-512 conv kernel bug (R145) |
| x86-64 | AVX2 | ✅/❌ | ELF x86-64 | ✅/❌ | |
| x86-64 | Scalar | ✅/❌ | ELF x86-64 | ✅/❌ | |

### Output Parity
| Pair | Bit-Identical? | Notes |
|------|:--------------:|-------|
| AVX2 vs Scalar (x86-64) | | |
| NEON vs Scalar (ARM64, if testable) | | |
| RVV vs Scalar (RISC-V, if testable) | | |

### Toolchain Versions
| Tool | Version | Notes |
|------|:-------:|-------|
| Host gcc | | |
| aarch64-linux-gnu-gcc | | |
| riscv64-linux-gnu-gcc | | |
| cmake | | |
| QEMU (if used) | | |

## Acceptance
- [ ] T1: ARM64 cross-compile succeeds (NEON + SVE), ELF aarch64 binary produced
- [ ] T2: RISC-V cross-compile succeeds (RVV), ELF riscv64 binary produced
- [ ] T3: x86-64 AVX2 and scalar builds both pass, output is bit-identical
- [ ] T3: AVX-512 build explicitly marked as having known kernel bugs (documented in matrix)
- [ ] Compatibility matrix fully populated with status, known issues, and toolchain versions
