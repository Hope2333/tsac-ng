# Round 169 — CUDA Backend Output vs CPU Comparison (Phase 4C)

**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Strategy

Fix the known CUDA illegal memory access at `cuda_backend.cu:463` (`cudaStreamSynchronize`) and then verify that CUDA decode output is bit-identical to CPU decode output. GPU backends must produce identical numerical results to the CPU reference for the same model and input.

**Prerequisites**: CUDA-capable GPU (SM 8.0+), `build_cuda_hip/tsac-ng` binary with `-DUSE_CUDA=ON`, `build/tsac-ng` CPU binary.

**Key files**:
- `src/cuda/cuda_backend.cu` — CUDA decoder backend (known bug at line 463)
- `src/cpu_decoder.c` — CPU reference decoder
- `src/tsac_codec.c` — backend dispatch logic

## Tasks

### T1: Diagnose CUDA illegal memory access at cuda_backend.cu:463

Reproduce the crash and capture diagnostic info:

```bash
mkdir -p /tmp/r169

# Run CUDA decode to trigger the bug
./build_cuda_hip/tsac-ng --cuda -v d test-simples/short_fast.txc /tmp/r169/cuda_output.wav 2>&1 | tee /tmp/r169/cuda_crash.log

# If CUDA build doesn't exist, verify build setup
ls -la build_cuda_hip/tsac-ng 2>/dev/null || echo "CUDA build not found — need to build with:"
echo "  mkdir -p build_cuda_hip && cd build_cuda_hip"
echo "  cmake .. -DUSE_CUDA=ON -DCUDAToolkit_ROOT=/opt/cuda 2>&1 | tail -5"
echo "  make -j\$(nproc) 2>&1 | tail -10"

# If crash reproduced, get backtrace
cat /tmp/r169/cuda_crash.log
```

**Acceptance**: Crash reproduced (or CUDA build created if missing). Backtrace captured.

### T2: Fix CUDA illegal memory access at cuda_backend.cu:463

The bug at line 463 is `cudaStreamSynchronize(s)`. Investigate root causes:

1. **Buffer overflow**: Previous kernel launched at line 460 (`launch_rvq_lookup`) may write past allocation
2. **Null pointer**: `d_feat`, `b->d_codes`, or `b->d_cb_data` may be null/uninitialized
3. **Stream sync before prior async ops complete**: Add explicit `cudaStreamSynchronize` before sync if stream has pending work

```bash
# Check line 463 context
grep -n "cudaStreamSynchronize" src/cuda/cuda_backend.cu

# Add defensive checks around line 463
# Fix approach — insert validation between kernel launch and sync:
#   if (b->d_codes == NULL) { fprintf(stderr, "[cuda] d_codes null\n"); return TSAC_ERR_BACKEND; }
#   cudaError_t err = cudaGetLastError();
#   if (err != cudaSuccess) { fprintf(stderr, "[cuda] rvq_lookup error: %s\n", cudaGetErrorString(err)); return TSAC_ERR_BACKEND; }
#   CUDA_CHK(cudaStreamSynchronize(s));

# Apply the fix (manual edit of src/cuda/cuda_backend.cu)
# After editing, rebuild
cd build_cuda_hip && make -j$(nproc) 2>&1 | tail -10
```

**Acceptance**: CUDA backend no longer crashes on decode. Exit code 0.

### T3: Compare CUDA output vs CPU output — must be identical

```bash
# Decode same file with CPU backend
./build/tsac-ng -v d test-simples/short_fast.txc /tmp/r169/cpu_output.wav 2>&1

# Decode with CUDA backend (after fix)
./build_cuda_hip/tsac-ng --cuda -v d test-simples/short_fast.txc /tmp/r169/cuda_output.wav 2>&1

# Compare WAVs byte-by-byte
python3 << 'EOF'
import numpy as np, wave, hashlib, os

def read_wav(path):
    with wave.open(path, 'rb') as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
        print(f'  {path}: {params.nchannels}ch, {params.framerate}Hz, {params.nframes}samples')
        return frames, w.getnframes(), w.getnchannels()

print('=== CUDA vs CPU WAV Comparison ===')
cpu_frames, cpu_n, cpu_ch = read_wav('/tmp/r169/cpu_output.wav')
cuda_frames, cuda_n, cuda_ch = read_wav('/tmp/r169/cuda_output.wav')

cpu_hash = hashlib.md5(cpu_frames).hexdigest()
cuda_hash = hashlib.md5(cuda_frames).hexdigest()

print(f'\nCPU MD5:  {cpu_hash}')
print(f'CUDA MD5: {cuda_hash}')

if cpu_hash == cuda_hash:
    print('\n✅ PASS: CUDA and CPU outputs are BIT-IDENTICAL')
    print(f'   Same: {cpu_n} samples, {cpu_ch} channels')
else:
    print('\n❌ FAIL: CUDA and CPU outputs DIFFER')
    # Convert to float for detailed comparison
    cpu_float = np.frombuffer(cpu_frames, dtype=np.int16).astype(np.float32) / 32768.0
    cuda_float = np.frombuffer(cuda_frames, dtype=np.int16).astype(np.float32) / 32768.0
    n = min(len(cpu_float), len(cuda_float))
    diff = np.abs(cpu_float[:n] - cuda_float[:n])
    print(f'   Max sample diff: {np.max(diff):.6f}')
    print(f'   Mean diff:       {np.mean(diff):.6f}')
    print(f'   Correlation:     {np.corrcoef(cpu_float[:n], cuda_float[:n])[0,1]:.6f}')
    print(f'   Samples differ:  {np.sum(diff > 1e-6)} / {n}')
EOF
```

**Acceptance**: CUDA and CPU WAV outputs are bit-identical (MD5 match). If not, max diff < 1e-4 and correlation > 0.9999.

### T4: Additional test files for CUDA verification

```bash
# Test with multiple codebook counts
for q in 6 8 12; do
  echo "=== Testing q=$q ==="
  
  # Encode with CPU
  ./build/tsac-ng c /tmp/r168/stereo_ref.wav /tmp/r169/test_q${q}.txc -q $q -f
  
  # Decode with CPU
  ./build/tsac-ng d /tmp/r169/test_q${q}.txc /tmp/r169/cpu_q${q}.wav
  
  # Decode with CUDA
  ./build_cuda_hip/tsac-ng --cuda d /tmp/r169/test_q${q}.txc /tmp/r169/cuda_q${q}.wav
  
  # Compare
  python3 -c "
import hashlib
cpu = open('/tmp/r169/cpu_q${q}.wav','rb').read()
cuda = open('/tmp/r169/cuda_q${q}.wav','rb').read()
cpu_h = hashlib.md5(cpu).hexdigest()
cuda_h = hashlib.md5(cuda).hexdigest()
print(f'  q={q}: CPU={cpu_h} CUDA={cuda_h} {\"✅\" if cpu_h==cuda_h else \"❌\"}')"
done
```

**Acceptance**: All q values produce bit-identical CPU and CUDA output.

## Acceptance Criteria

- **AC1**: CUDA illegal memory access at line 463 diagnosed and fixed
- **AC2**: CUDA backend builds and runs without crash
- **AC3**: CUDA output WAV bit-identical to CPU output (same MD5)
- **AC4**: Verified across at least 3 q values (6, 8, 12)
- **AC5**: Results documented in `/tmp/r169/cuda_comparison.md`

## Expected Output

```
=== CUDA vs CPU WAV Comparison ===
  /tmp/r169/cpu_output.wav: 2ch, 44100Hz, 7938 samples
  /tmp/r169/cuda_output.wav: 2ch, 44100Hz, 7938 samples

CPU MD5:  a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6
CUDA MD5: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6

✅ PASS: CUDA and CPU outputs are BIT-IDENTICAL
```

## Bug Fix Detail

The illegal memory access at `cuda_backend.cu:463` occurs because:

```
Line 460: launch_rvq_lookup(...)     // Kernel launch (async)
Line 462: cudaGetLastError()          // Checks last error
Line 463: cudaStreamSynchronize(s)    // ❌ Illegal memory access here
```

Root cause: `launch_rvq_lookup` accesses device memory that is either:
- Uninitialized (null pointer in `b->d_codes` or `b->d_cb_offsets`)
- Out of bounds (buffer size mismatch for n_frames × n_codebooks)
- Previously freed (stale pointer from re-initialization)

**Fix**: Add null-pointer checks before kernel launch + separate error check before sync:
```c
if (!b->d_codes || !b->d_cb_data || !b->d_cb_offsets) {
    fprintf(stderr, "[cuda] RVQ GPU buffers not allocated\n");
    return TSAC_ERR_BACKEND;
}
// Then launch and sync
```
