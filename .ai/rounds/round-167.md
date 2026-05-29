# R167 — Multi-Codebook Comparison

**Status**: PENDING
**Phase**: 4C — Multi-File + Multi-Backend Validation
**Owner**: Worker
**Depends on**: R166 (baseline RMS/comparison values)

---

## Strategy

Encode and then decode a 5s music clip (`music_5s.wav`) at 3 codebook counts: q=6, q=8, q=12. Measure bitrate, output RMS vs original, and correlation for each. Produce a bitrate-vs-quality trade-off curve.

The TSAC DAC model supports `--n_codebooks` (or `-q`) from 1–12 for joint stereo, 1–9 for mono. Each codebook adds ~0.66 kbps to the bitrate and improves reconstruction fidelity.

---

## Tasks

### Task 1: Encode music_5s.wav at q=6, q=8, q=12

- **Action**: Encode `/tmp/mogra_slices/music_5s.wav` at 3 quality levels using CPU backend.
- **Commands**:
  ```bash
  BUILD=/home/miao/Projects/tsac-ng/build
  OUTDIR=/tmp/r167
  mkdir -p "$OUTDIR"

  SRCDIR=/tmp/mogra_slices
  SRC="$SRCDIR/music_5s.wav"

  # Encode at q=6
  "$BUILD/tsac-ng" -v -q 6 c "$SRC" "$OUTDIR/music_5s_q6.txc"

  # Encode at q=8
  "$BUILD/tsac-ng" -v -q 8 c "$SRC" "$OUTDIR/music_5s_q8.txc"

  # Encode at q=12
  "$BUILD/tsac-ng" -v -q 12 c "$SRC" "$OUTDIR/music_5s_q12.txc"
  ```
- **Expected output**: 3 `.txc` files of increasing size as q increases.
- **Verify**:
  ```bash
  ls -la "$OUTDIR"/*.txc
  ```
  Expected sizes: q6 < q8 < q12.

### Task 2: Decode all 3 encoded files back to WAV

- **Action**: Decode each `.txc` back to WAV using CPU backend.
- **Commands**:
  ```bash
  for q in 6 8 12; do
    "$BUILD/tsac-ng" -v d "$OUTDIR/music_5s_q${q}.txc" "$OUTDIR/music_5s_q${q}_decoded.wav"
  done
  ```
- **Expected output**: 3 WAV files.

### Task 3: Measure bitrate and file size

- **Action**: Extract bitrate from verbose output or compute from file size and duration.
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import os, wave, math

  OUTDIR = '/tmp/r167'
  SRC_WAV = '/tmp/mogra_slices/music_5s.wav'

  with wave.open(SRC_WAV, 'rb') as wf:
      duration_s = wf.getnframes() / wf.getframerate()

  for q in [6, 8, 12]:
      txc_path = os.path.join(OUTDIR, f'music_5s_q{q}.txc')
      wav_path = os.path.join(OUTDIR, f'music_5s_q{q}_decoded.wav')
      txc_bytes = os.path.getsize(txc_path) if os.path.exists(txc_path) else 0
      bitrate = (txc_bytes * 8) / duration_s / 1000.0
      wav_bytes = os.path.getsize(wav_path) if os.path.exists(wav_path) else 0
      print(f"q={q:2d} | txc={txc_bytes:>6}B | bitrate={bitrate:>6.2f} kbps | wav={wav_bytes:>8}B")
PYEOF
  ```

### Task 4: Compute per-codebook RMS and correlation vs original

- **Action**: For each decoded WAV, measure RMS residual and correlation against the original `music_5s.wav`.
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import numpy as np, wave, os, math

  def load_wav(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          return np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0

  SRC = '/tmp/mogra_slices/music_5s.wav'
  OUTDIR = '/tmp/r167'
  original = load_wav(SRC)

  print(f"{'Codebook':<10} {'Bitrate(kbps)':<15} {'RMS(dBFS)':<12} {'Corr':<10} {'FileSize(B)':<12}")
  print("-"*60)

  for q in [6, 8, 12]:
      decoded = load_wav(os.path.join(OUTDIR, f'music_5s_q{q}_decoded.wav'))
      txc_path = os.path.join(OUTDIR, f'music_5s_q{q}.txc')
      txc_size = os.path.getsize(txc_path) if os.path.exists(txc_path) else 0

      # duration from original
      with wave.open(SRC, 'rb') as wf:
          dur = wf.getnframes() / wf.getframerate()
      bitrate = (txc_size * 8) / dur / 1000.0

      # Residual RMS
      min_len = min(len(decoded), len(original))
      diff = decoded[:min_len] - original[:min_len]
      rms = 20.0 * math.log10(max(np.sqrt(np.mean(diff**2)), 1e-10))

      # Correlation
      a, b = decoded[:min_len], original[:min_len]
      if np.std(a) < 1e-10 or np.std(b) < 1e-10:
          corr = 0.0
      else:
          corr = np.corrcoef(a, b)[0, 1]

      print(f"q={q:<5}  {bitrate:<15.2f} {rms:<12.2f} {corr:<10.4f} {txc_size:<12}")
PYEOF
  ```

### Task 5: Generate trade-off curve data

- **Action**: Save structured comparison data as JSON.
- **Command**:
  ```bash
  python3 << 'PYEOF'
  import json, os, wave, math, numpy as np

  OUTDIR = '/tmp/r167'
  SRC = '/tmp/mogra_slices/music_5s.wav'

  def load_wav(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          return np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0

  original = load_wav(SRC)
  with wave.open(SRC, 'rb') as wf:
      dur = wf.getnframes() / wf.getframerate()

  results = {}
  for q in [6, 8, 12]:
      decoded = load_wav(os.path.join(OUTDIR, f'music_5s_q{q}_decoded.wav'))
      txc_path = os.path.join(OUTDIR, f'music_5s_q{q}.txc')
      txc_size = os.path.getsize(txc_path)
      bitrate = (txc_size * 8) / dur / 1000.0
      min_len = min(len(decoded), len(original))
      diff = decoded[:min_len] - original[:min_len]
      rms = float(20.0 * math.log10(max(np.sqrt(np.mean(diff**2)), 1e-10)))
      a, b = decoded[:min_len], original[:min_len]
      corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) >= 1e-10 and np.std(b) >= 1e-10 else 0.0
      results[f'q{q}'] = {
          'codebooks': q,
          'txc_size_bytes': txc_size,
          'bitrate_kbps': round(bitrate, 2),
          'rms_dbfs': round(rms, 2),
          'correlation': round(corr, 4),
          'duration_s': round(dur, 2),
      }

  with open(os.path.join(OUTDIR, 'codebook_comparison.json'), 'w') as fp:
      json.dump(results, fp, indent=2)
  print(json.dumps(results, indent=2))
PYEOF
  ```

---

## Acceptance Criteria

| # | Criterion | Method |
|---|-----------|--------|
| 1 | All 3 quality levels encode successfully | exit code 0, .txc files produced |
| 2 | All 3 .txc files decode successfully | exit code 0, WAV files produced |
| 3 | Bitrate increases with codebook count | q6 < q8 < q12 in kbps |
| 4 | RMS improves (lower/less-negative) with more codebooks | q6 RMS > q8 RMS > q12 RMS |
| 5 | Correlation improves with more codebooks | q6 < q8 < q12 |
| 6 | Structured results in `/tmp/r167/codebook_comparison.json` | valid JSON, all 3 entries present |

---

## Expected Results Shape

| Codebook | Bitrate (kbps) | RMS (dBFS) | Correlation |
|----------|----------------|------------|-------------|
| q=6      | ~5.8           | -4.5       | 0.004       |
| q=8      | ~7.8           | -3.9       | 0.005       |
| q=12     | ~11.7          | -3.6       | 0.005       |

*(Values approximate — BF8 weight divergence limits absolute quality)*

---

## Files Touched

| File | Action |
|------|--------|
| `/home/miao/Projects/tsac-ng/build/tsac-ng` | Execute |
| `/tmp/r167/music_5s_q{6,8,12}.txc` | Created (encoded) |
| `/tmp/r167/music_5s_q{6,8,12}_decoded.wav` | Created (decoded) |
| `/tmp/r167/codebook_comparison.json` | Created |

---

## Risks

| Risk | Mitigation |
|------|------------|
| q=12 encode fails (out of memory) | Monitor `max_memory` in verbose output; if > available RAM, skip q=12 |
| Bitrate computation doesn't match printed bitrate | Use printed bitrate from `-v` output instead of computed |
| No measurable quality improvement across q values | BF8 noise may mask codebook improvements — document as finding |
