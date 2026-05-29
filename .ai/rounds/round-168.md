# R168 — Stereo + Mono Validation

**Status**: PENDING
**Phase**: 4C — Multi-File + Multi-Backend Validation
**Owner**: Worker
**Depends on**: R167 (encoded .txc files at various qualities)

---

## Strategy

Encode a stereo music clip in 3 channel modes — joint stereo (default, 2ch), forced mono (1ch), and separate-channels (dual mono) — then compare L/R channel correlation in the decoded output. This validates the channel handling logic in the decoder graph.

The DAC decoder final layer outputs `96 → 2` (channels) via Conv1d. Stereo joint mode uses the full 2-channel output. Mono mode uses `--channels 1` to force single-channel operation. Separate channels (`-s`) encodes each channel independently.

---

## Tasks

### Task 1: Encode music_5s at 3 channel modes

- **Action**: Encode `/tmp/mogra_slices/music_5s.wav` in joint stereo, mono, and separate-channels modes.
- **Commands**:
  ```bash
  BUILD=/home/miao/Projects/tsac-ng/build
  OUTDIR=/tmp/r168
  SRC=/tmp/mogra_slices/music_5s.wav
  mkdir -p "$OUTDIR"

  # 1. Joint stereo (default, 2ch)
  "$BUILD/tsac-ng" -v -q 8 c "$SRC" "$OUTDIR/music_5s_stereo.txc"

  # 2. Mono (force 1ch) — original is stereo, tsac will mix down or take L
  "$BUILD/tsac-ng" -v -q 8 -c 1 c "$SRC" "$OUTDIR/music_5s_mono.txc"

  # 3. Separate channels (dual mono encoding, each ch encoded independently)
  "$BUILD/tsac-ng" -v -q 8 -s c "$SRC" "$OUTDIR/music_5s_sepch.txc"
  ```
- **Expected output**: 3 `.txc` files. Mono should be ~half the size of stereo. Separate channels may be similar to stereo but with different internal layout.
- **Verify**:
  ```bash
  ls -la "$OUTDIR"/*.txc
  ```

### Task 2: Decode all 3 back to WAV

- **Action**: Decode each `.txc` to WAV.
- **Commands**:
  ```bash
  for mode in stereo mono sepch; do
    "$BUILD/tsac-ng" -v d "$OUTDIR/music_5s_${mode}.txc" "$OUTDIR/music_5s_${mode}_decoded.wav"
  done
  ```
- **Expected output**: 3 WAV files. `mono` WAV is 1-channel, others are 2-channel.

### Task 3: Analyze channel count and per-channel RMS

- **Action**: Verify channel count and measure per-channel signal properties.
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import wave, numpy as np, os, math

  OUTDIR = '/tmp/r168'
  SRC = '/tmp/mogra_slices/music_5s.wav'

  def load_wav_channels(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          nch = wf.getnchannels()
          sw = wf.getsampwidth()
          framerate = wf.getframerate()
          raw = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
          # Reshape to (samples, channels)
          raw = raw.reshape(-1, nch) if nch > 1 else raw.reshape(-1, 1)
          return raw, nch, framerate

  for mode in ['stereo', 'mono', 'sepch']:
      path = os.path.join(OUTDIR, f'music_5s_{mode}_decoded.wav')
      if not os.path.exists(path):
          print(f"{mode}: MISSING")
          continue
      sig, nch, rate = load_wav_channels(path)
      rms_per_ch = [20.0 * math.log10(max(np.sqrt(np.mean(sig[:, ch]**2)), 1e-10)) for ch in range(nch)]
      print(f"{mode:>8}: ch={nch}, rate={rate}, frames={sig.shape[0]}, "
            f"RMS_L={rms_per_ch[0]:.2f} dBFS" + (f", RMS_R={rms_per_ch[1]:.2f} dBFS" if nch > 1 else ""))
PYEOF
  ```

### Task 4: Compute L/R channel correlation

- **Action**: For each 2-channel file, compute cross-correlation between L and R channels. Mono file has correlation=1.0 by definition.
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import wave, numpy as np, os

  OUTDIR = '/tmp/r168'

  def load_wav_channels(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          nch = wf.getnchannels()
          sw = wf.getsampwidth()
          raw = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
          raw = raw.reshape(-1, nch) if nch > 1 else raw.reshape(-1, 1)
          return raw, nch

  print(f"{'Mode':<10} {'Channels':<10} {'L/R Corr':<12} {'L RMS(dBFS)':<14} {'R RMS(dBFS)':<14}")
  print("-"*60)

  for mode in ['stereo', 'mono', 'sepch']:
      path = os.path.join(OUTDIR, f'music_5s_{mode}_decoded.wav')
      if not os.path.exists(path):
          print(f"{mode:<10} MISSING")
          continue
      sig, nch = load_wav_channels(path)
      if nch == 2:
          L, R = sig[:, 0], sig[:, 1]
          if np.std(L) > 1e-10 and np.std(R) > 1e-10:
              lr_corr = np.corrcoef(L, R)[0, 1]
          else:
              lr_corr = 0.0
          l_rms = 20.0 * np.log10(max(np.sqrt(np.mean(L**2)), 1e-10))
          r_rms = 20.0 * np.log10(max(np.sqrt(np.mean(R**2)), 1e-10))
          print(f"{mode:<10} {nch:<10} {lr_corr:<12.4f} {l_rms:<14.2f} {r_rms:<14.2f}")
      else:
          sig_1d = sig[:, 0]
          rms_val = 20.0 * np.log10(max(np.sqrt(np.mean(sig_1d**2)), 1e-10))
          print(f"{mode:<10} {nch:<10} {'N/A':<12} {rms_val:<14.2f} {'N/A':<14}")
PYEOF
  ```

### Task 5: Compare decoded output length across modes

- **Action**: Verify all modes produce the same number of output samples (or known ratio for mono).
- **Script**:
  ```bash
  python3 << 'PYEOF'
  import wave, os

  OUTDIR = '/tmp/r168'
  for mode in ['stereo', 'mono', 'sepch']:
      path = os.path.join(OUTDIR, f'music_5s_{mode}_decoded.wav')
      if not os.path.exists(path):
          continue
      with wave.open(path, 'rb') as wf:
          nf = wf.getnframes()
          nch = wf.getnchannels()
          rate = wf.getframerate()
          dur = nf / rate
          print(f"{mode:>8}: {nf:>8} samples, {nch}ch, {rate}Hz, {dur:.2f}s")
PYEOF
  ```

### Task 6: Structured results snapshot

- **Action**: Save all channel analysis as JSON.
- **Command**:
  ```bash
  python3 << 'PYEOF'
  import json, os, wave, numpy as np, math

  OUTDIR = '/tmp/r168'

  def analyze(path):
      with wave.open(path, 'rb') as wf:
          frames = wf.readframes(wf.getnframes())
          nch = wf.getnchannels()
          rate = wf.getframerate()
          sampwidth = wf.getsampwidth()
          raw = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
          raw = raw.reshape(-1, nch) if nch > 1 else raw.reshape(-1, 1)
          result = {
              'channels': nch,
              'sample_rate': rate,
              'sampwidth': sampwidth,
              'frames': wf.getnframes(),
              'duration_s': round(wf.getnframes() / rate, 3),
          }
          if nch == 2:
              L, R = raw[:, 0], raw[:, 1]
              lr_corr = float(np.corrcoef(L, R)[0, 1]) if np.std(L) > 1e-10 and np.std(R) > 1e-10 else 0.0
              result['lr_correlation'] = round(lr_corr, 6)
              result['L_rms_dbfs'] = round(float(20.0 * math.log10(max(np.sqrt(np.mean(L**2)), 1e-10))), 2)
              result['R_rms_dbfs'] = round(float(20.0 * math.log10(max(np.sqrt(np.mean(R**2)), 1e-10))), 2)
          else:
              result['lr_correlation'] = 1.0
              mono = raw[:, 0]
              result['mono_rms_dbfs'] = round(float(20.0 * math.log10(max(np.sqrt(np.mean(mono**2)), 1e-10))), 2)
          return result

  results = {}
  for mode in ['stereo', 'mono', 'sepch']:
      path = os.path.join(OUTDIR, f'music_5s_{mode}_decoded.wav')
      if os.path.exists(path):
          results[mode] = analyze(path)

  with open(os.path.join(OUTDIR, 'stereo_analysis.json'), 'w') as fp:
      json.dump(results, fp, indent=2)
  print(json.dumps(results, indent=2))
PYEOF
  ```

---

## Acceptance Criteria

| # | Criterion | Method |
|---|-----------|--------|
| 1 | Stereo encode produces 2ch .txc | file size > mono, decode gives 2ch WAV |
| 2 | Mono encode produces 1ch .txc | file size ~half of stereo, decode gives 1ch WAV |
| 3 | Separate-channels encode produces 2ch output | decode gives 2ch WAV |
| 4 | L/R correlation measured for stereo and sepch | script prints correlation value |
| 5 | Mono decode has correlation=1.0 (single channel, whatever the definition) or N/A | script reports N/A or 1.0 |
| 6 | All output durations within 1% of original (5.0s) | duration_s in JSON |
| 7 | Structured results in `/tmp/r168/stereo_analysis.json` | valid JSON, 3 entries |

---

## Expected Results Shape

| Mode | Channels | L/R Corr | Expected |
|------|----------|----------|----------|
| stereo | 2 | ~0.95+ | Joint stereo: L/R share codebook, high correlation |
| mono | 1 | N/A | Mixed down to mono |
| sepch | 2 | ~0.5-0.8 | Independent encoding → lower L/R correlation |

---

## Files Touched

| File | Action |
|------|--------|
| `/home/miao/Projects/tsac-ng/build/tsac-ng` | Execute |
| `/tmp/r168/music_5s_{stereo,mono,sepch}.txc` | Created |
| `/tmp/r168/music_5s_{stereo,mono,sepch}_decoded.wav` | Created |
| `/tmp/r168/stereo_analysis.json` | Created |

---

## Risks

| Risk | Mitigation |
|------|------------|
| `-c 1` forces mono but DAC model expects 2ch output | Check model type: use `dac_mono_q8.bin` for mono if `-c 1` fails |
| Separate channels mode produces mismatched channel lengths | Verify L/R frame count match in decode |
| L/R correlation near 1.0 for sepch | If both channels decode identically, codec may ignore `-s` — file bug |
