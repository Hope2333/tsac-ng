# Round 170 — HIP + Scalar Fallback Comparison (Phase 4C)

**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Strategy

Create a cross-backend comparison matrix for HIP (AMD GPU), scalar fallback (no SIMD), and the standard CPU (AVX2) backend. This validates that all backends produce consistent output and identifies any backend-specific numerical divergence. The scalar fallback path is especially critical as it's the correctness baseline for all SIMD implementations.

**Prerequisites**: `build/tsac-ng` (CPU), optionally `build_cuda_hip/tsac-ng` with HIP enabled. For scalar fallback, need a build with SIMD disabled or use `SIMD_DISABLED=1` env var.

**Key files**:
- `hip/hip_kernels.hip.cpp` — HIP kernel implementations
- `hip/dac_decoder.hip.cpp` — HIP DAC decoder
- `src/cpu_decoder.c` — CPU decoder with SIMD dispatch
- `src/arch/` — SIMD arch-specific implementations

## Tasks

### T1: Run HIP decode (if AMD GPU available)

```bash
mkdir -p /tmp/r170

# Check if HIP build exists and AMD GPU is available
ls -la build_cuda_hip/tsac-ng 2>/dev/null || (
  echo "Building with HIP support..."
  mkdir -p build_cuda_hip && cd build_cuda_hip
  cmake .. -DUSE_HIP=ON -DCMAKE_C_COMPILER=hipcc -DCMAKE_CXX_COMPILER=hipcc 2>&1 | tail -5
  make -j$(nproc) 2>&1 | tail -10
)

# Run HIP decode
./build_cuda_hip/tsac-ng --hip -v d test-simples/short_fast.txc /tmp/r170/hip_output.wav 2>&1 | tee /tmp/r170/hip_decode.log

python3 << 'EOF'
import numpy as np, wave, os

def read_wav(path):
    if not os.path.exists(path):
        return None
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels())

hip = read_wav('/tmp/r170/hip_output.wav')
if hip is not None:
    print(f'HIP decode: {hip.shape}, RMS={np.sqrt(np.mean(hip**2)):.6f}')
else:
    print('HIP decode: SKIP (no AMD GPU or build failure)')
EOF
```

**Acceptance**: HIP decode runs (or documented as SKIP if no AMD GPU available).

### T2: Compare HIP vs CPU output

```bash
# CPU decode for comparison
./build/tsac-ng -v d test-simples/short_fast.txc /tmp/r170/cpu_output.wav 2>&1

# Compare HIP vs CPU
python3 << 'EOF'
import numpy as np, wave, os, hashlib

def read_wav_bytes(path):
    if not os.path.exists(path): return None, None
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        return frames, w.getnframes()

cpu_raw, cpu_n = read_wav_bytes('/tmp/r170/cpu_output.wav')
hip_raw, hip_n = read_wav_bytes('/tmp/r170/hip_output.wav')

if hip_raw is None:
    print('HIP output not available — SKIP comparison')
    exit(0)

cpu_hash = hashlib.md5(cpu_raw).hexdigest()
hip_hash = hashlib.md5(hip_raw).hexdigest()

print(f'CPU MD5: {cpu_hash}')
print(f'HIP MD5: {hip_hash}')

if cpu_hash == hip_hash:
    print('✅ PASS: HIP and CPU outputs are BIT-IDENTICAL')
else:
    cpu_f = np.frombuffer(cpu_raw, dtype=np.int16).astype(np.float32) / 32768.0
    hip_f = np.frombuffer(hip_raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = min(len(cpu_f), len(hip_f))
    corr = np.corrcoef(cpu_f[:n], hip_f[:n])[0,1]
    max_diff = np.max(np.abs(cpu_f[:n] - hip_f[:n]))
    print(f'❌ HIP and CPU DIFFER')
    print(f'   Correlation: {corr:.6f}')
    print(f'   Max diff:    {max_diff:.6f}')
EOF
```

**Acceptance**: HIP output compared with CPU. Bit-identity is ideal; correlation > 0.99 is acceptable.

### T3: Test scalar fallback path (no SIMD) — verify correctness

Build without SIMD or disable SIMD at runtime:

```bash
# Option A: Build scalar-only version
mkdir -p /tmp/r170/build_scalar && cd /tmp/r170/build_scalar
cmake /home/miao/Projects/tsac-ng -DCMAKE_C_FLAGS="-O0 -DTSAC_SCALAR_ONLY" 2>&1 | tail -3
make -j$(nproc) tsac-ng 2>&1 | tail -5
cd /home/miao/Projects/tsac-ng

# Run scalar decode
/tmp/r170/build_scalar/tsac-ng -v d test-simples/short_fast.txc /tmp/r170/scalar_output.wav 2>&1 | tee /tmp/r170/scalar_decode.log

# Compare scalar vs normal CPU
python3 << 'EOF'
import numpy as np, wave, os, hashlib

def read_wav_bytes(path):
    with wave.open(path, 'rb') as w:
        return w.readframes(w.getnframes()), w.getnframes()

scalar_raw, _ = read_wav_bytes('/tmp/r170/scalar_output.wav')
cpu_raw, _ = read_wav_bytes('/tmp/r170/cpu_output.wav')

cpu_hash = hashlib.md5(cpu_raw).hexdigest()
scalar_hash = hashlib.md5(scalar_raw).hexdigest()

print(f'CPU (AVX2) MD5:   {cpu_hash}')
print(f'Scalar-only MD5:  {scalar_hash}')

if cpu_hash == scalar_hash:
    print('✅ PASS: Scalar and AVX2 outputs are BIT-IDENTICAL')
else:
    cpu_f = np.frombuffer(cpu_raw, dtype=np.int16).astype(np.float32) / 32768.0
    sca_f = np.frombuffer(scalar_raw, dtype=np.int16).astype(np.float32) / 32768.0
    n = min(len(cpu_f), len(sca_f))
    corr = np.corrcoef(cpu_f[:n], sca_f[:n])[0,1]
    max_diff = np.max(np.abs(cpu_f[:n] - sca_f[:n]))
    print(f'❌ Scalar and AVX2 DIFFER')
    print(f'   Correlation: {corr:.6f}')
    print(f'   Max diff:    {max_diff:.6f}')
    print(f'   THIS IS A BUG — SIMD kernels produce different results from scalar!')
EOF
```

**Acceptance**: Scalar fallback produces bit-identical output to AVX2 CPU path. If not, document the divergence as a SIMD kernel bug.

### T4: Document backend comparison matrix

```bash
python3 << 'EOF'
import os, json

results = {}

# Collect data
backends = [
    ('CPU (AVX2)', '/tmp/r170/cpu_output.wav', 'build/tsac-ng'),
    ('Scalar only', '/tmp/r170/scalar_output.wav', '/tmp/r170/build_scalar/tsac-ng'),
    ('HIP (AMD GPU)', '/tmp/r170/hip_output.wav', 'build_cuda_hip/tsac-ng'),
]

for name, wav_path, binary_path in backends:
    entry = {'binary_exists': os.path.exists(binary_path), 'wav_exists': os.path.exists(wav_path)}
    if entry['wav_exists']:
        import hashlib
        with open(wav_path, 'rb') as f:
            entry['md5'] = hashlib.md5(f.read()).hexdigest()
        import wave, numpy as np
        with wave.open(wav_path, 'rb') as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            entry['rms'] = round(float(np.sqrt(np.mean(data**2))), 6)
            entry['samples'] = w.getnframes()
            entry['channels'] = w.getnchannels()
    results[name] = entry

# Matrix table
print('=' * 70)
print('R170: Cross-Backend Comparison Matrix')
print('=' * 70)
print()
print(f'| Backend | Binary | WAV | MD5 | RMS | Samples |')
print(f'|---------|--------|-----|-----|-----|---------|')
for name, entry in results.items():
    bin_ok = '✅' if entry['binary_exists'] else '❌'
    wav_ok = '✅' if entry['wav_exists'] else '❌'
    md5 = entry.get('md5', 'N/A')[:16] + '...'
    rms = entry.get('rms', 'N/A')
    samples = entry.get('samples', 'N/A')
    print(f'| {name} | {bin_ok} | {wav_ok} | {md5} | {rms} | {samples} |')

# Bit-identity check
cpu_md5 = results['CPU (AVX2)'].get('md5', None)
print()
print('### Bit-Identity Matrix')
print()
if cpu_md5:
    for name, entry in results.items():
        if 'md5' in entry:
            match = '✅ IDENTICAL' if entry['md5'] == cpu_md5 else '❌ DIFFERENT'
            print(f'  {name:20s}: {match}')

# Save
matrix = f"""# Round 170 — Cross-Backend Comparison Matrix

| Backend | Binary | WAV | MD5 | RMS | Samples |
|---------|--------|-----|-----|-----|---------|
"""
for name, entry in results.items():
    bin_ok = 'YES' if entry['binary_exists'] else 'NO'
    wav_ok = 'YES' if entry['wav_exists'] else 'NO'
    md5 = entry.get('md5', 'N/A')
    rms = entry.get('rms', 'N/A')
    samples = entry.get('samples', 'N/A')
    matrix += f'| {name} | {bin_ok} | {wav_ok} | {md5} | {rms} | {samples} |\n'

os.makedirs('/tmp/r170', exist_ok=True)
with open('/tmp/r170/backend_matrix.md', 'w') as f:
    f.write(matrix)
print(f"\nResults saved to /tmp/r170/backend_matrix.md")
EOF
```

**Acceptance**: `/tmp/r170/backend_matrix.md` created with all backend comparisons.

## Acceptance Criteria

- **AC1**: HIP decode attempted (SKIP documented if no AMD GPU)
- **AC2**: HIP vs CPU comparison measured (if HIP available)
- **AC3**: Scalar fallback path built and tested
- **AC4**: Scalar vs AVX2 comparison verifies SIMD correctness
- **AC5**: Cross-backend matrix saved to `/tmp/r170/backend_matrix.md`
- **AC6**: All backends produce consistent output (or discrepancies documented)

## Expected Output

```
| Backend          | Binary | WAV | MD5              | RMS      | Samples |
|------------------|--------|-----|------------------|----------|---------|
| CPU (AVX2)       | ✅     | ✅  | a1b2c3d4...      | 0.321000 | 7938    |
| Scalar only      | ✅     | ✅  | a1b2c3d4...      | 0.321000 | 7938    |
| HIP (AMD GPU)    | ✅     | ❌  | N/A              | N/A      | N/A     |

### Bit-Identity Matrix
  CPU (AVX2):        REFERENCE
  Scalar only:       ✅ IDENTICAL
  HIP (AMD GPU):     ❌ (no AMD GPU available)
```
