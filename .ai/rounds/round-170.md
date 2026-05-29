# R170 — HIP + Scalar Fallback Comparison & Cross-Backend Matrix

**Status**: PENDING
**Phase**: 4C — Multi-File + Multi-Backend Validation
**Owner**: Worker
**Depends on**: R166 (CPU baseline), R168 (stereo analysis), R169 (CUDA fix)

---

## Strategy

Complete the cross-backend validation picture:
1. Verify HIP backend output matches CPU scalar output (sample-accurate)
2. Document the full cross-backend comparison matrix across all 4 backends (CPU, CUDA, HIP, Scalar fallback)
3. Document any discrepancies per backend pair with root cause

---

## Background: Available Backends

| Backend | Binary Path | Type | Status |
|---------|-------------|------|--------|
| CPU (AVX2) | `build/tsac-ng` | SIMD-optimized CPU | Baseline |
| CPU Scalar | `build_debug/tsac-ng` or forced `TSAC_SIMD=0` | Pure scalar | Fallback |
| CUDA | `build_cuda/tsac-ng` | NVIDIA GPU | R169 fixes |
| HIP | `build_hip/tsac-ng` | AMD GPU | Previously working |

---

## Tasks

### Task 1: Verify HIP backend decodes all test vectors

- **Action**: Decode all 4 test `.txc` files through the HIP backend.
- **Commands**:
  ```bash
  BUILD_HIP=/home/miao/Projects/tsac-ng/build_hip
  OUTDIR=/tmp/r170
  mkdir -p "$OUTDIR"

  # 1. short_fast
  "$BUILD_HIP/tsac-ng" --hip -v d /tmp/short_fast.txc "$OUTDIR/short_fast_hip.wav"

  # 2. silent_fast
  "$BUILD_HIP/tsac-ng" --hip -v d /tmp/silent_fast.txc "$OUTDIR/silent_fast_hip.wav"

  # 3. mogra_5s
  "$BUILD_HIP/tsac-ng" --hip -v d /tmp/mogra_5s.txc "$OUTDIR/mogra_5s_hip.wav"

  # 4. music_5s_f_q6
  "$BUILD_HIP/tsac-ng" --hip -v d /tmp/mogra_slices/music_5s_f_q6.txc "$OUTDIR/music_5s_hip.wav"
  ```
- **Expected output**: 4 WAV files. Each decode should show `[hip] completed: output samples=N` and `Success.`
- **Note**: HIP has previously succeeded on all 4 files. This step re-validates.

### Task 2: Decode with scalar fallback (no SIMD)

- **Action**: Force scalar CPU path by building without SIMD flags or using `build_debug` build.
- **Commands**:
  ```bash
  BUILD_SCALAR=/home/miao/Projects/tsac-ng/build_debug
  if [ ! -f "$BUILD_SCALAR/tsac-ng" ]; then
      # Build a scalar-only version
      mkdir -p /tmp/r170_build
      cd /tmp/r170_build
      cmake /home/miao/Projects/tsac-ng -DCMAKE_C_FLAGS="-O0 -fno-fast-math" -DCMAKE_CXX_FLAGS="-O0 -fno-fast-math"
      make -j$(nproc) tsac-ng
      BUILD_SCALAR=/tmp/r170_build
  fi

  for f in short_fast silent_fast mogra_5s; do
    TXC="/tmp/${f}.txc"
    [ -f "$TXC" ] && "$BUILD_SCALAR/tsac-ng" -v d "$TXC" "$OUTDIR/${f}_scalar.wav"
  done
  "$BUILD_SCALAR/tsac-ng" -v d /tmp/mogra_slices/music_5s_f_q6.txc "$OUTDIR/music_5s_scalar.wav"
  ```
- **Alternative**: Force scalar path by setting `TSAC_SIMD=0` environment variable in the CPU build:
  ```bash
  TSAC_SIMD=0 /home/miao/Projects/tsac-ng/build/tsac-ng -v d /tmp/short_fast.txc "$OUTDIR/short_fast_scalar.wav"
  ```
- **Expected output**: 4 WAV files from scalar path.

### Task 3: HIP vs CPU sample comparison

- **Action**: Compare HIP WAV output against CPU reference WAV sample-by-sample.
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import numpy as np, wave, os

  CPU_DIR = '/tmp/r166'
  HIP_DIR = '/tmp/r170'

  def load_wav_int16(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          return np.frombuffer(frames, dtype=np.int16), wf.getnchannels()

  test_files = [
      ('short_fast_cpu.wav', 'short_fast_hip.wav'),
      ('silent_fast_cpu.wav', 'silent_fast_hip.wav'),
      ('mogra_5s_cpu.wav', 'mogra_5s_hip.wav'),
      ('music_5s_cpu.wav', 'music_5s_hip.wav'),
  ]

  print(f"{'File':<25} {'Samples':<10} {'Match':<8} {'MaxDiff':<10} {'MeanDiff':<10}")
  print("-"*65)

  for cpu_name, hip_name in test_files:
      cpu_path = os.path.join(CPU_DIR, cpu_name)
      hip_path = os.path.join(HIP_DIR, hip_name)
      if not os.path.exists(cpu_path) or not os.path.exists(hip_path):
          print(f"{cpu_name:<25} MISSING FILE")
          continue
      cpu_sig, _ = load_wav_int16(cpu_path)
      hip_sig, _ = load_wav_int16(hip_path)
      min_len = min(len(cpu_sig), len(hip_sig))
      cpu_s, hip_s = cpu_sig[:min_len], hip_sig[:min_len]
      diff = np.abs(cpu_s.astype(np.int32) - hip_s.astype(np.int32))
      max_d = int(np.max(diff))
      mean_d = float(np.mean(diff))
      match = "YES" if max_d <= 2 else "NO"
      print(f"{cpu_name:<25} {min_len:<10} {match:<8} {max_d:<10} {mean_d:<10.4f}")
      if max_d > 2:
          bad = np.where(diff > 2)[0]
          print(f"  → {len(bad)} samples differ >2. First at idx {bad[0]}: cpu={cpu_s[bad[0]]}, hip={hip_s[bad[0]]}")
PYEOF
  ```

### Task 4: Scalar vs CPU (AVX-optimized) comparison

- **Action**: Verify scalar fallback produces identical output to SIMD-optimized CPU path.
- **Method**: Same sample comparison as Task 3, but CPU vs scalar.
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import numpy as np, wave, os

  CPU_DIR = '/tmp/r166'
  SCALAR_DIR = '/tmp/r170'

  def load_wav_int16(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          return np.frombuffer(frames, dtype=np.int16), wf.getnchannels()

  for suffix, label in [('scalar', 'Scalar')]:
      test_files = [
          ('short_fast_cpu.wav', f'short_fast_{suffix}.wav'),
          ('silent_fast_cpu.wav', f'silent_fast_{suffix}.wav'),
          ('mogra_5s_cpu.wav', f'mogra_5s_{suffix}.wav'),
          ('music_5s_cpu.wav', f'music_5s_{suffix}.wav'),
      ]
      print(f"\n=== CPU vs {label} ===")
      print(f"{'File':<25} {'Match':<8} {'MaxDiff':<10}")
      print("-"*45)
      for cpu_name, scalar_name in test_files:
          cpu_path = os.path.join(CPU_DIR, cpu_name)
          scalar_path = os.path.join(SCALAR_DIR, scalar_name)
          if not os.path.exists(cpu_path) or not os.path.exists(scalar_path):
              print(f"{cpu_name:<25} MISSING")
              continue
          cpu_sig, _ = load_wav_int16(cpu_path)
          sca_sig, _ = load_wav_int16(scalar_path)
          min_len = min(len(cpu_sig), len(sca_sig))
          diff = np.abs(cpu_sig[:min_len].astype(np.int32) - sca_sig[:min_len].astype(np.int32))
          max_d = int(np.max(diff))
          match = "YES" if max_d <= 2 else "NO"
          print(f"{cpu_name:<25} {match:<8} {max_d:<10}")
PYEOF
  ```

### Task 5: Build cross-backend comparison matrix

- **Action**: Create a comprehensive matrix comparing all backend pairs across all 4 test files.
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import numpy as np, wave, os, json

  BACKEND_DIRS = {
      'CPU': '/tmp/r166',
      'CUDA': '/tmp/r169',
      'HIP': '/tmp/r170',
      'Scalar': '/tmp/r170',
  }

  def load_wav_int16(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          return np.frombuffer(frames, dtype=np.int16), wf.getnchannels()

  def suffix(name):
      m = {'short_fast': 'short_fast_cpu.wav', 'silent_fast': 'silent_fast_cpu.wav',
           'mogra_5s': 'mogra_5s_cpu.wav', 'music_5s': 'music_5s_cpu.wav'}
      return m.get(name, f'{name}_cpu.wav')

  file_stems = ['short_fast', 'silent_fast', 'mogra_5s', 'music_5s']
  backends = ['CPU', 'CUDA', 'HIP', 'Scalar']

  backend_file_map = {
      'CPU': {s: f'{s}_cpu.wav' for s in file_stems},
      'CUDA': {s: f'{s}_cuda.wav' for s in file_stems},
      'HIP': {s: f'{s}_hip.wav' for s in file_stems},
      'Scalar': {s: f'{s}_scalar.wav' for s in file_stems},
  }

  # Load all signals
  signals = {}
  for backend, dir_path in BACKEND_DIRS.items():
      signals[backend] = {}
      for stem in file_stems:
          fname = backend_file_map[backend][stem]
          fpath = os.path.join(dir_path, fname)
          if os.path.exists(fpath):
              sig, ch = load_wav_int16(fpath)
              signals[backend][stem] = {'sig': sig, 'ch': ch, 'path': fpath}
          else:
              signals[backend][stem] = None

  # Build matrix
  matrix = {}
  for b1 in backends:
      for b2 in backends:
          if b1 >= b2:
              continue  # avoid symmetric duplicates
          pair_key = f"{b1}_vs_{b2}"
          matrix[pair_key] = {}
          for stem in file_stems:
              s1 = signals[b1].get(stem)
              s2 = signals[b2].get(stem)
              if s1 is None or s2 is None:
                  matrix[pair_key][stem] = {'status': 'MISSING'}
                  continue
              sig1, sig2 = s1['sig'], s2['sig']
              min_len = min(len(sig1), len(sig2))
              if min_len == 0:
                  matrix[pair_key][stem] = {'status': 'EMPTY'}
                  continue
              diff = np.abs(sig1[:min_len].astype(np.int32) - sig2[:min_len].astype(np.int32))
              max_diff = int(np.max(diff))
              mean_diff = float(np.mean(diff))
              bad_count = int(np.sum(diff > 2))
              matrix[pair_key][stem] = {
                  'status': 'MATCH' if max_diff <= 2 else 'MISMATCH',
                  'max_diff': max_diff,
                  'mean_diff': round(mean_diff, 4),
                  'bad_samples': bad_count,
                  'total_samples': min_len,
                  'samples_match_pct': round(100.0 * (1.0 - bad_count / max(min_len, 1)), 2),
              }

  # Pretty print
  print("Cross-Backend Comparison Matrix")
  print("=" * 80)
  print(f"{'Pair':<20} {'File':<15} {'Status':<10} {'MaxDiff':<10} {'Match%':<10}")
  print("-" * 80)
  for pair, files in matrix.items():
      for stem, result in files.items():
          status = result.get('status', 'ERROR')
          max_d = result.get('max_diff', 'N/A')
          match_pct = result.get('samples_match_pct', 'N/A')
          print(f"{pair:<20} {stem:<15} {status:<10} {str(max_d):<10} {str(match_pct):<10}")

  # Save
  with open('/tmp/r170/cross_backend_matrix.json', 'w') as fp:
      json.dump(matrix, fp, indent=2)
  print(f"\nMatrix saved to /tmp/r170/cross_backend_matrix.json")
PYEOF
  ```

### Task 6: Document cross-backend discrepancies

- **Action**: For any MISMATCH pair, investigate root cause and document.
- **Format**:
  ```bash
  cat > /tmp/r170/cross_backend_notes.md << 'NOTES'
# Cross-Backend Discrepancy Notes

## Summary
- Backends compared: CPU, CUDA, HIP, Scalar
- Test files: short_fast, silent_fast, mogra_5s, music_5s

## Discrepancies Found

### [pair] vs [pair] on [file]
- **Max diff**: N
- **Samples affected**: N / total (X%)
- **Root cause**: [description]
- **Fix**: [description or "Known limitation — no fix"]

## Overall Assessment
[Summary paragraph]

## Bug Reports / TODOs
1. [ ] [issue]
2. [ ] [issue]
NOTES
  ```

---

## Acceptance Criteria

| # | Criterion | Method |
|---|-----------|--------|
| 1 | HIP backend decodes all 4 files successfully | exit code 0, WAV produced |
| 2 | Scalar fallback decodes all 4 files successfully | exit code 0, WAV produced |
| 3 | HIP output matches CPU within ±2 LSB for all files | max_diff ≤ 2 in int16 comparison |
| 4 | Scalar output matches CPU within ±1 LSB for all files | max_diff ≤ 1 (same code, same math) |
| 5 | Cross-backend matrix JSON written | `/tmp/r170/cross_backend_matrix.json` exists |
| 6 | Discrepancy notes document any MISMATCH | `/tmp/r170/cross_backend_notes.md` exists |
| 7 | Matrix covers all 6 backend pairs × 4 files = 24 entries | JSON has 24 data points |

---

## Expected Results Shape

| Pair | Expected Status | Notes |
|------|----------------|-------|
| CPU vs Scalar | MATCH (±1 LSB) | Same math, different compilation flags |
| CPU vs HIP | MATCH (±2 LSB) | Float accumulation differs on AMD GPU |
| CPU vs CUDA | MATCH (±2 LSB) | After R169 fix |
| HIP vs CUDA | MATCH (±2 LSB) | Both GPU, different architectures |
| Scalar vs HIP | MATCH (±2 LSB) | |
| Scalar vs CUDA | MATCH (±2 LSB) | |

---

## Files Touched

| File | Action |
|------|--------|
| `/home/miao/Projects/tsac-ng/build_hip/tsac-ng` | Execute |
| `/home/miao/Projects/tsac-ng/build_debug/tsac-ng` | Execute (if exists) |
| `/tmp/r170/*_hip.wav` | Created |
| `/tmp/r170/*_scalar.wav` | Created |
| `/tmp/r170/cross_backend_matrix.json` | Created |
| `/tmp/r170/cross_backend_notes.md` | Created |

---

## Risks

| Risk | Mitigation |
|------|------------|
| HIP backend not available (no AMD GPU) | Mark HIP rows as UNAVAILABLE in matrix |
| Scalar build fails | Use `TSAC_SIMD=0` env var with CPU build |
| HIP produces different sample count than CPU | Align by frame count / truncate to min |
| CUDA backend from R169 still broken | Mark CUDA rows as BROKEN, document error |
