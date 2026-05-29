# R166 — Multi-File WAV Testing

**Status**: PENDING
**Phase**: 4C — Multi-File + Multi-Backend Validation
**Owner**: Worker
**Depends on**: R165 (encoder strided conv fix)

---

## Strategy

Systematically decode 4 distinct `.txc` test vectors through the CPU backend, measure per-file RMS (relative to reference WAV) and cross-correlation, then produce a unified results table. This establishes the baseline quality floor before codebook/backend experiments in R167–R170.

Each test vector exercises a different audio characteristic:

| File | Characteristics | Duration | Frames |
|------|----------------|----------|--------|
| `/tmp/short_fast.txc` | Very short burst, 76 bytes | ~0.1s | 1 |
| `/tmp/silent_fast.txc` | Silence, q=6, 661 bytes | ~1s | 2 |
| `/tmp/mogra_5s.txc` | MOGRA DJ set 5s, q=8 | ~5s | 10 |
| `/tmp/mogra_slices/music_5s_f_q6.txc` | Music 5s, q=6 | ~5s | 10 |

---

## Tasks

### Task 1: CPU decode all 4 test vectors

- **Action**: Run CPU backend decode for each `.txc` file.
- **Commands**:
  ```bash
  BUILD=/home/miao/Projects/tsac-ng/build
  OUTDIR=/tmp/r166
  mkdir -p "$OUTDIR"

  # 1. short_fast (76 bytes, ~0.1s, 1 frame, default codebooks)
  "$BUILD/tsac-ng" -v d /tmp/short_fast.txc "$OUTDIR/short_fast_cpu.wav"

  # 2. silent_fast (silence, q=6, 661 bytes)
  "$BUILD/tsac-ng" -v d /tmp/silent_fast.txc "$OUTDIR/silent_fast_cpu.wav"

  # 3. mogra_5s (q=8, 4318 bytes, 5s DJ audio)
  "$BUILD/tsac-ng" -v d /tmp/mogra_5s.txc "$OUTDIR/mogra_5s_cpu.wav"

  # 4. music_5s_f_q6 (q=6, 3241 bytes, 5s music)
  "$BUILD/tsac-ng" -v d /tmp/mogra_slices/music_5s_f_q6.txc "$OUTDIR/music_5s_cpu.wav"
  ```
- **Expected output**: 4 WAV files in `/tmp/r166/`. Each decode prints `bitrate=XX kb/s` and `Success.`
- **Verify**:
  ```bash
  ls -la "$OUTDIR"/*.wav
  ```

### Task 2: Compute per-file RMS (dBFS) and correlation vs reference

- **Action**: For each decoded WAV with a reference, compute residual RMS and cross-correlation.
- **Reference mapping**:
  - `short_fast_cpu.wav` → `/tmp/short_fast_dec.wav` (if exists) or self-RMS
  - `silent_fast_cpu.wav` → `/tmp/silent_fast_dec.wav` (if exists) or self-RMS
  - `mogra_5s_cpu.wav` → `/tmp/mogra_5s_ref.wav` (ground truth)
  - `music_5s_cpu.wav` → `/tmp/mogra_slices/music_5s.wav` (original)
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import numpy as np, wave, os, math

  def load_wav(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          sig = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
          return sig, wf.getnchannels()

  def rms_dbfs_residual(sig, ref):
      min_len = min(len(sig), len(ref))
      diff = sig[:min_len] - ref[:min_len]
      rms = np.sqrt(np.mean(diff**2))
      return 20.0 * math.log10(max(rms, 1e-10))

  def correlation(sig, ref):
      min_len = min(len(sig), len(ref))
      a, b = sig[:min_len], ref[:min_len]
      if np.std(a) < 1e-10 or np.std(b) < 1e-10:
          return 0.0
      return float(np.corrcoef(a, b)[0, 1])

  refs = {
      'short_fast_cpu.wav': '/tmp/short_fast_dec.wav',
      'silent_fast_cpu.wav': '/tmp/silent_fast_dec.wav',
      'mogra_5s_cpu.wav': '/tmp/mogra_5s_ref.wav',
      'music_5s_cpu.wav': '/tmp/mogra_slices/music_5s.wav',
  }
  outdir = '/tmp/r166'

  print(f"{'File':<25} {'RMS(dBFS)':<12} {'Corr':<10} {'Samples':<10} {'Ch':<4}")
  print("-"*65)
  for fname, ref_path in refs.items():
      wpath = os.path.join(outdir, fname)
      if not os.path.exists(wpath):
          print(f"{fname:<25} MISSING")
          continue
      sig, ch = load_wav(wpath)
      if os.path.exists(ref_path):
          ref_sig, _ = load_wav(ref_path)
          r = rms_dbfs_residual(sig, ref_sig)
          c = correlation(sig, ref_sig)
      else:
          r = 20.0 * math.log10(max(np.sqrt(np.mean(sig**2)), 1e-10))
          c = 1.0
      print(f"{fname:<25} {r:<12.2f} {c:<10.4f} {len(sig):<10} {ch:<4}")
PYEOF
  ```

### Task 3: Pairwise correlation matrix across all 4 outputs

- **Action**: Measure cross-file correlation to verify outputs are structurally independent.
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import numpy as np, wave, os

  def load_wav(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          return np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0

  outdir = '/tmp/r166'
  files = sorted([f for f in os.listdir(outdir) if f.endswith('.wav')])
  sigs = {f: load_wav(os.path.join(outdir, f)) for f in files}

  print("Pairwise Correlation Matrix")
  names = list(sigs.keys())
  print(f"{'':<25}", end="")
  for n in names:
      print(f"{n[:24]:<24}", end="")
  print()
  for n1 in names:
      print(f"{n1[:25]:<25}", end="")
      for n2 in names:
          ml = min(len(sigs[n1]), len(sigs[n2]))
          if ml < 10 or np.std(sigs[n1][:ml]) < 1e-10 or np.std(sigs[n2][:ml]) < 1e-10:
              corr = 0.0
          else:
              corr = np.corrcoef(sigs[n1][:ml], sigs[n2][:ml])[0, 1]
          print(f"{corr:<24.4f}", end="")
      print()
PYEOF
  ```

### Task 4: Structured results snapshot

- **Action**: Save results as JSON for programmatic consumption in later rounds.
- **Command**:
  ```bash
  python3 << 'PYEOF'
  import json, wave, os

  outdir = '/tmp/r166'
  summary = {}
  for f in sorted(os.listdir(outdir)):
      if not f.endswith('.wav'): continue
      p = os.path.join(outdir, f)
      with wave.open(p, 'rb') as wf:
          summary[f] = {
              'frames': wf.getnframes(),
              'channels': wf.getnchannels(),
              'sampwidth': wf.getsampwidth(),
              'framerate': wf.getframerate(),
              'size_bytes': os.path.getsize(p),
          }
  with open(f'{outdir}/results.json', 'w') as fp:
      json.dump(summary, fp, indent=2)
  print(json.dumps(summary, indent=2))
PYEOF
  ```

---

## Acceptance Criteria

| # | Criterion | Method |
|---|-----------|--------|
| 1 | All 4 .txc files decode successfully | exit code 0, WAV produced |
| 2 | Each output WAV is valid | `file` command shows RIFF/WAVE |
| 3 | RMS script runs without error | no traceback, table printed |
| 4 | Correlation matrix computed | non-degenerate, no NaN values |
| 5 | Results table written as JSON | `/tmp/r166/results.json` exists and parsable |
| 6 | Table documents: filename, frames, channels, sample rate, RMS(dBFS), correlation | JSON keys match |

---

## Files Touched

| File | Action |
|------|--------|
| `/home/miao/Projects/tsac-ng/build/tsac-ng` | Execute (CPU backend) |
| `/tmp/r166/short_fast_cpu.wav` | Created |
| `/tmp/r166/silent_fast_cpu.wav` | Created |
| `/tmp/r166/mogra_5s_cpu.wav` | Created |
| `/tmp/r166/music_5s_cpu.wav` | Created |
| `/tmp/r166/results.json` | Created |

---

## Risks

| Risk | Mitigation |
|------|------------|
| Reference WAV not found for short/silent | Use self-referenced RMS (signal RMS) |
| All correlations near-zero | Expected — BF8 weight noise known issue. Document as baseline |
| WAV length mismatch | Script truncates to min length |
| Encoder regression from R165 | Rebuild CPU backend before running |
