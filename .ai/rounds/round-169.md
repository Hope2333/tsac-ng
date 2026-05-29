# R169 — CUDA Backend Output Comparison vs CPU

**Status**: PENDING
**Phase**: 4C — Multi-File + Multi-Backend Validation
**Owner**: Worker
**Depends on**: R166 (baseline CPU WAV references)

---

## Strategy

Decode all 4 test `.txc` files from R166 using the CUDA backend and compare PCM output sample-by-sample against the CPU backend. CUDA output MUST be bit-identical (or within ±1 LSB at 16-bit quantization). If not identical, isolate which CUDA kernel is responsible and fix it.

---

## Background: CUDA Backend Architecture

The CUDA backend (`src/cuda/`) consists of:

| File | Role |
|------|------|
| `src/cuda/cuda_backend.cu` | Weight upload, decode/encode graph, buffer management |
| `src/cuda/cuda_kernels.cu` | 8 kernel launchers: conv1d, convt1d, snake, add_bias, tanh_clip, rvq_lookup, rvq_quantize, rvq_subtract |

---

## Tasks

### Task 1: Verify CUDA backend builds and runs

- **Action**: Test CUDA decode with a minimal file.
- **Commands**:
  ```bash
  BUILD_CUDA=/home/miao/Projects/tsac-ng/build_cuda
  OUTDIR=/tmp/r169
  mkdir -p "$OUTDIR"

  # Test short_fast.txc — this has been known to fail with "illegal memory access" at rvq_lookup
  "$BUILD_CUDA/tsac-ng" --cuda -v d /tmp/short_fast.txc "$OUTDIR/short_fast_cuda.wav" 2>&1
  echo "Exit code: $?"
  ```
- **Expected output**: If it works, a WAV file is produced. Known issue: `cuda_backend.cu:463: an illegal memory access was encountered` at `launch_rvq_lookup`.
- **If failure occurs**: Report the exact error message and line number. Proceed to Task 2 (fix).

### Task 2: Debug and fix CUDA illegal memory access (if present)

- **Symptom**: `cuda_backend.cu:463: an illegal memory access was encountered`
- **Root cause analysis**:
  Line 463 is `CUDA_CHK(cudaGetLastError())` after `launch_rvq_lookup` on line 460-461.
  ```c
  CUDA_CHK(launch_rvq_lookup(d_feat, b->d_codes, b->d_cb_data, b->d_cb_offsets,
                              n_frames, n_codebooks, rvq_dim, s));
  CUDA_CHK(cudaGetLastError());  // ← line 463
  ```
  Possible causes:
  1. `d_codes` buffer too small for the n_frames × n_codebooks allocation
  2. `cb_offsets` indexing out of bounds in `rvq_lookup_kernel`
  3. Codebook data on GPU not properly allocated or copied
- **Fix steps**:
  ```bash
  # Add CUDA debug checks
  python3 << 'PYEOF'
  # Generate a debug build with cuda-memcheck or compute-sanitizer
  fix_steps = [
    "1. Add cuda-memcheck: compute-sanitizer --tool memcheck build_cuda/tsac-ng --cuda d /tmp/short_fast.txc /dev/null",
    "2. Check d_codes allocation size in cuda_backend.cu: n_frames * n_codebooks * sizeof(int)",
    "3. Check cb_offsets[12] sentinel: ensure cb_offsets[12] ≤ total allocated entries",
    "4. Verify d_cb_data has room for all 12 codebooks: cb_offsets[12] * 1024 * sizeof(float)",
    "5. Add bounds check in rvq_lookup_kernel: if base + entry * cb_dim + d >= total_size, skip",
  ]
  for s in fix_steps:
      print(s)
PYEOF
  ```
- **Commands to run**:
  ```bash
  # Run with compute-sanitizer (if available)
  compute-sanitizer --tool memcheck "$BUILD_CUDA/tsac-ng" --cuda d /tmp/short_fast.txc /dev/null 2>&1 | head -40
  ```
- **If compute-sanitizer not available**, add manual bounds checking:
  ```bash
  grep -n "rvq_lookup_kernel" /home/miao/Projects/tsac-ng/src/cuda/cuda_kernels.cu
  # Inspect the kernel for out-of-bounds access patterns
  ```
- **Expected fix locations** (if bug found):
  - `/home/miao/Projects/tsac-ng/src/cuda/cuda_kernels.cu` — `rvq_lookup_kernel` (lines 142-160)
  - `/home/miao/Projects/tsac-ng/src/cuda/cuda_backend.cu` — `cuda_upload_weights` cb_offsets calculation (lines 334-365)

### Task 3: Decode all 4 test vectors through CUDA

- **Action**: Once CUDA backend runs without errors, decode all 4 test files.
- **Commands**:
  ```bash
  for f in short_fast silent_fast mogra_5s; do
    TXC_FILE="/tmp/${f}.txc"
    if [ ! -f "$TXC_FILE" ]; then
      # Try mogra_slices for music_5s
      continue
    fi
    "$BUILD_CUDA/tsac-ng" --cuda -v d "$TXC_FILE" "$OUTDIR/${f}_cuda.wav"
  done

  # music_5s_f_q6
  "$BUILD_CUDA/tsac-ng" --cuda -v d /tmp/mogra_slices/music_5s_f_q6.txc "$OUTDIR/music_5s_cuda.wav"
  ```
- **Expected output**: 4 WAV files in `/tmp/r169/`.

### Task 4: Sample-by-sample comparison against CPU output

- **Action**: Compare each CUDA WAV against the corresponding R166 CPU WAV. First at 16-bit integer level, then at float32 level.
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import numpy as np, wave, os

  CPU_DIR = '/tmp/r166'
  CUDA_DIR = '/tmp/r169'
  CORRUPTION_TOLERANCE = 2  # max sample values that may differ (LSB noise)

  def load_wav_int16(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          sig = np.frombuffer(frames, dtype=np.int16)
          return sig, wf.getnchannels()

  test_files = [
      'short_fast_cpu.wav',
      'silent_fast_cpu.wav',
      'mogra_5s_cpu.wav',
      'music_5s_cpu.wav',
  ]

  print(f"{'File':<25} {'CPU_samples':<12} {'CUDA_samples':<13} {'Match':<8} {'MaxDiff':<10} {'BAD >2':<8}")
  print("-"*78)

  for cpu_name in test_files:
      cpu_path = os.path.join(CPU_DIR, cpu_name)
      cuda_name = cpu_name.replace('_cpu', '_cuda')
      cuda_path = os.path.join(CUDA_DIR, cuda_name)

      if not os.path.exists(cpu_path):
          print(f"{cpu_name:<25} CPU file missing")
          continue
      if not os.path.exists(cuda_path):
          print(f"{cuda_name:<25} CUDA file missing")
          continue

      cpu_sig, cpu_ch = load_wav_int16(cpu_path)
      cuda_sig, cuda_ch = load_wav_int16(cuda_path)

      # Trim to min length
      min_len = min(len(cpu_sig), len(cuda_sig))
      cpu_sig, cuda_sig = cpu_sig[:min_len], cuda_sig[:min_len]

      diff = np.abs(cpu_sig.astype(np.int32) - cuda_sig.astype(np.int32))
      max_diff = int(np.max(diff))
      bad_samples = int(np.sum(diff > CORRUPTION_TOLERANCE))
      match = "YES" if max_diff <= CORRUPTION_TOLERANCE else "NO"

      print(f"{cpu_name:<25} {len(cpu_sig):<12} {len(cuda_sig):<13} {match:<8} {max_diff:<10} {bad_samples:<8}")

      if max_diff > CORRUPTION_TOLERANCE:
          # Find first mismatch location
          bad_idx = np.where(diff > CORRUPTION_TOLERANCE)[0]
          print(f"  → First mismatch at sample {bad_idx[0]}: cpu={cpu_sig[bad_idx[0]]}, cuda={cuda_sig[bad_idx[0]]}")
PYEOF
  ```

### Task 5: Float-level comparison (if int16 matches)

- **Action**: If int16 comparison passes, do a float32 sample-level comparison to verify true bit-exactness.
- **Note**: The CPU decoder processes in float32 internally then quantizes to int16 for WAV output. If CUDA and CPU use the same graph but different intermediate precision (float vs double, or different fma behavior), expect ±1 LSB differences.
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import numpy as np, wave, os, struct

  CPU_DIR = '/tmp/r166'
  CUDA_DIR = '/tmp/r169'

  def load_wav_float32(path):
      with wave.open(path, 'rb') as wf:
          sw = wf.getsampwidth()
          frames = wf.readframes(wf.getnframes())
          if sw == 2:
              raw = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
          elif sw == 4:
              raw = np.frombuffer(frames, dtype=np.float32).astype(np.float64)
          else:
              raw = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
          return raw, wf.getnchannels()

  for cpu_name in ['short_fast_cpu.wav', 'silent_fast_cpu.wav', 'mogra_5s_cpu.wav', 'music_5s_cpu.wav']:
      cpu_path = os.path.join(CPU_DIR, cpu_name)
      cuda_name = cpu_name.replace('_cpu', '_cuda')
      cuda_path = os.path.join(CUDA_DIR, cuda_name)
      if not os.path.exists(cpu_path) or not os.path.exists(cuda_path):
          continue
      cpu_sig, _ = load_wav_float32(cpu_path)
      cuda_sig, _ = load_wav_float32(cuda_path)
      min_len = min(len(cpu_sig), len(cuda_sig))
      diff = np.abs(cpu_sig[:min_len] - cuda_sig[:min_len])
      print(f"{cpu_name:<25} max_abs_diff={np.max(diff):.10f}  mean_abs_diff={np.mean(diff):.10f}  "
            f"rms_diff={np.sqrt(np.mean(diff**2)):.10f}")
PYEOF
  ```

### Task 6: Fix any CUDA kernel discrepancies

- **Action**: If CUDA output differs from CPU, isolate the offending kernel and fix.
- **Method**:
  1. Add intermediate buffer dumps after each kernel launch
  2. Compare layer-by-layer output between CPU and CUDA
  3. Fix the kernel that diverges
  4. Rebuild and re-run full comparison
- **Known potential issues**:
  - `conv1d_kernel`: Order of floating-point operations differs from CPU (`float sum = b ? b[oc] : 0.0f` vs CPU's `+= w[j] * x[...]`)
  - `snake_kernel`: Uses `__sinf` vs CPU's `sinf` — should match but test
  - `tanh_clip_kernel`: Uses `tanhf` vs CPU's `tanhf` — identical
  - `rvq_lookup_kernel`: May accumulate in different order `sum += cb_data[...]` vs CPU's loop — largest source of discrepancy
- **Fix command** (if needed):
  ```bash
  # Example: Fix rvq_lookup_kernel to use Kahan summation
  sed -i 's/sum += cb_data\[base + entry \* cb_dim + d\];/\/\/ Kahan summation\n            float y = cb_data[base + entry * cb_dim + d] - c;\n            float t = sum + y;\n            c = (t - sum) - y;\n            sum = t;/' /home/miao/Projects/tsac-ng/src/cuda/cuda_kernels.cu
  ```
  *(This is illustrative — actual fix depends on divergence analysis)*

---

## Acceptance Criteria

| # | Criterion | Method |
|---|-----------|--------|
| 1 | CUDA backend runs without illegal memory access | All 4 files decode with exit code 0 |
| 2 | CUDA WAV + CPU WAV have same sample count per channel | Length comparison |
| 3 | CUDA output matches CPU within ±1 LSB (int16) for all 4 files | `max_diff ≤ 2` in int16 comparison |
| 4 | Float-level RMS diff < 1e-5 | Float comparison script |
| 5 | If discrepancies found, root cause documented and kernel fixed | Report in round summary |
| 6 | Results table in `/tmp/r169/cuda_cpu_comparison.json` | JSON with per-file metrics |

---

## Files Touched

| File | Action |
|------|--------|
| `/home/miao/Projects/tsac-ng/src/cuda/cuda_kernels.cu` | Modified (if fix needed) |
| `/home/miao/Projects/tsac-ng/src/cuda/cuda_backend.cu` | Modified (if fix needed) |
| `/home/miao/Projects/tsac-ng/build_cuda/tsac-ng` | Rebuilt (if fix applied) |
| `/tmp/r169/*_cuda.wav` | Created |
| `/tmp/r169/cuda_cpu_comparison.json` | Created |

---

## Known State (Pre-Fix)

From previous run:
```
[cuda] init OK: NVIDIA GeForce RTX 4060 Laptop GPU (24 SMs, CC 8.9)
[cuda] w0[0..3]=-0.006147 -0.000129 0.002038 -0.022175
[cuda] /home/miao/Projects/tsac-ng/src/cuda/cuda_backend.cu:463: an illegal memory access was encountered
```

Note: CUDA weight values (w0[0..3]) differ from CPU/HIP weight values:
- CUDA w0: `-0.006147 -0.000129 0.002038 -0.022175`
- HIP w0:   `-0.487303 -0.448057 -0.379377 -0.748942`

This suggests the CUDA weight upload path (`cuda_upload_weights`) may already be wrong — the `dequant_weights` function likely returns different values when called from CUDA vs HIP context, or the tensor lookup fails and uploads zeros.

---

## Risks

| Risk | Mitigation |
|------|------------|
| CUDA illegal memory access at rvq_lookup persists | Use `compute-sanitizer` to pinpoint exact OOB access; fix cb_offsets calculation |
| W0 weight values differ from CPU/HIP | Debug tensor lookup in `cuda_upload_weights`: ensure tensor names match exactly |
| Float accumulation order differences | Accept ±1 LSB as "matching" |
| CUDA not available on build machine | Document as SKIPPED; proceed with R170 |
