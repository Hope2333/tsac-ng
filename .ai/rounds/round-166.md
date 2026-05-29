# Round 166 — Multi-File WAV Testing (Phase 4C)

**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Strategy

Validate tsac-ng's fast TXC decode across multiple audio content types using objective metrics (RMS, correlation). This establishes a baseline for Sub-Phase 4C, letting us quantify decode quality on speech, silence, music, and mixed-content samples. Results table goes to `/tmp/r166/results_table.md` for cross-round reference.

**Prerequisites**: `build/tsac-ng` binary exists, test TXC files exist in `test-simples/` or are generated.

**Key files**: `src/cpu_decoder.c` (decode impl), `src/tsac_codec.c` (WAV I/O), `experimental/compare_activations.py` (metrics framework)

## Tasks

### T1: Decode short_fast.txc (9 frames) — measure RMS + correlation

Generate the short test file from WAV if not present, or use existing:

```bash
# Ensure test files exist
ls -la test-simples/*.txc

# Decode short fast TXC (if available)
./build/tsac-ng -v d test-simples/short_fast.txc /tmp/r166/short_fast.wav 2>&1

# If short_fast.txc doesn't exist, create it from a short WAV
# First need a short WAV:
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=0.2" -ar 44100 -ac 2 -acodec pcm_f32le /tmp/r166/short_ref.wav
# Then encode with original tsac (if available): tsac c /tmp/r166/short_ref.wav /tmp/r166/short_fast.txc -f -q 8
# Or use our encoder: ./build/tsac-ng c /tmp/r166/short_ref.wav /tmp/r166/short_fast.txc -q 8 -f

# Measure RMS
python3 -c "
import numpy as np
import wave, struct

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        dtype = np.int16 if w.getsampwidth() == 2 else np.float32
        data = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if dtype == np.int16:
            data /= 32768.0
        return data.reshape(-1, w.getnchannels())

our = read_wav('/tmp/r166/short_fast.wav')
print(f'Shape: {our.shape}, RMS/ch: {np.sqrt(np.mean(our**2, axis=0))}, Max: {np.max(np.abs(our))}')
# If reference exists
if os.path.exists('/tmp/r166/short_ref.wav'):
    ref = read_wav('/tmp/r166/short_ref.wav')
    n = min(len(our), len(ref))
    corr = np.corrcoef(our[:n].flatten(), ref[:n].flatten())[0,1]
    print(f'Correlation with ref: {corr:.6f}')
"
```

**Acceptance**: WAV decodes without error. RMS per channel within [0.0, 1.0]. If reference available, correlation logged.

### T2: Decode silent_fast.txc (87 frames) — measure RMS + correlation

```bash
# Decode silent fast TXC
./build/tsac-ng -v d test-simples/silent_fast.txc /tmp/r166/silent_fast.wav 2>&1

# Measure
python3 << 'EOF'
import numpy as np, wave, os
def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels())
our = read_wav('/tmp/r166/silent_fast.wav')
print(f'Shape: {our.shape}, RMS/ch: {np.sqrt(np.mean(our**2, axis=0))}')
print(f'Silence check - should be near-zero RMS: RMS total = {np.sqrt(np.mean(our**2)):.6f}')
EOF
```

**Acceptance**: Silent file produces near-zero RMS (< 0.01). No clicks/pops in output.

### T3: Decode MOGRA 5s fast TXC — measure RMS + correlation

```bash
# The MOGRA test file
./build/tsac-ng -v d "test-simples/4.8(wed)MOGRA × #DSPM presents ぷらぷらうんじ@b96s.txc" /tmp/r166/mogra_5s_fast.wav 2>&1

# Compare with reference decode
python3 << 'EOF'
import numpy as np, wave, os
def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels())
our = read_wav('/tmp/r166/mogra_5s_fast.wav')
ref_path = "test-simples/4.8(wed)MOGRA × #DSPM presents ぷらぷらうんじ@b96s.txc#tsac.wav"
if os.path.exists(ref_path):
    ref = read_wav(ref_path)
    n = min(len(our), len(ref))
    corr = np.corrcoef(our[:n].flatten(), ref[:n].flatten())[0,1]
    print(f'Reference decode exists: corr={corr:.6f}')
    print(f'Our RMS/ch: {np.sqrt(np.mean(our**2, axis=0))}')
    print(f'Ref RMS/ch:  {np.sqrt(np.mean(ref**2, axis=0))}')
else:
    print(f'No reference WAV found at {ref_path}')
    print(f'Our shape: {our.shape}, RMS/ch: {np.sqrt(np.mean(our**2, axis=0))}')
EOF
```

**Acceptance**: WAV decodes. Correlation with reference logged (expected < 0.5 given known BF8 gap, but must be > 0). RMS non-zero (audio content present).

### T4: Decode music_5s_f_q6.txc — measure RMS + correlation

```bash
# If music_5s_f_q6.txc doesn't exist, create from the short P丸様 test:
# Take 5s from the MOGRA or create a synth music file
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=5" -f lavfi -i "sine=frequency=880:duration=5" \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=first" -ar 44100 -ac 2 /tmp/r166/music_5s_ref.wav
# Encode: ./build/tsac-ng c /tmp/r166/music_5s_ref.wav /tmp/r166/music_5s_f_q6.txc -q 6 -f

# Decode
./build/tsac-ng -v d /tmp/r166/music_5s_f_q6.txc /tmp/r166/music_5s_f_q6.wav 2>&1

# Measure
python3 << 'EOF'
import numpy as np, wave
def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels())
our = read_wav('/tmp/r166/music_5s_f_q6.wav')
ref = read_wav('/tmp/r166/music_5s_ref.wav')
n = min(len(our), len(ref))
corr = np.corrcoef(our[:n].flatten(), ref[:n].flatten())[0,1]
print(f'Music 5s q6: shape={our.shape}, corr={corr:.6f}')
print(f'Our RMS/ch: {np.sqrt(np.mean(our**2, axis=0))}')
print(f'Ref RMS/ch:  {np.sqrt(np.mean(ref**2, axis=0))}')
EOF
```

**Acceptance**: WAV decodes. Correlation measured. RMS levels reasonable for music content.

### T5: Build multi-file results table

```bash
mkdir -p /tmp/r166

# Aggregate all measurements into a table
python3 << 'EOF'
import numpy as np, wave, os, json

def read_wav(path):
    if not os.path.exists(path):
        return None, None
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels()), w.getnframes()

results = []
files = [
    ('short_fast', '/tmp/r166/short_fast.wav', '/tmp/r166/short_ref.wav'),
    ('silent_fast', '/tmp/r166/silent_fast.wav', None),
    ('mogra_5s_fast', '/tmp/r166/mogra_5s_fast.wav',
     'test-simples/4.8(wed)MOGRA × #DSPM presents ぷらぷらうんじ@b96s.txc#tsac.wav'),
    ('music_5s_f_q6', '/tmp/r166/music_5s_f_q6.wav', '/tmp/r166/music_5s_ref.wav'),
]

for name, our_path, ref_path in files:
    our, n = read_wav(our_path)
    if our is None:
        results.append({'file': name, 'status': 'SKIP - no output', 'rms': 'N/A', 'corr': 'N/A'})
        continue
    rms_per_ch = np.sqrt(np.mean(our**2, axis=0)).tolist()
    rms_total = float(np.sqrt(np.mean(our**2)))
    corr = 'N/A'
    if ref_path:
        ref, _ = read_wav(ref_path)
        if ref is not None:
            n_ = min(len(our), len(ref))
            c_ = np.corrcoef(our[:n_].flatten(), ref[:n_].flatten())[0,1]
            corr = f'{c_:.6f}'
    results.append({
        'file': name,
        'status': 'OK',
        'samples': n,
        'rms_per_ch': [f'{r:.6f}' for r in rms_per_ch],
        'rms_total': f'{rms_total:.6f}',
        'correlation': corr
    })

# Print table
print('#' * 80)
print('# Round 166 — Multi-File WAV Test Results')
print('#' * 80)
print()
print(f'| File | Status | Samples | RMS/ch | RMS total | Correlation |')
print(f'|------|--------|---------|--------|-----------|-------------|')
for r in results:
    if r['status'] == 'SKIP - no output':
        print(f"| {r['file']} | SKIP | - | - | - | - |")
    else:
        print(f"| {r['file']} | {r['status']} | {r['samples']} | {r['rms_per_ch']} | {r['rms_total']} | {r['correlation']} |")

# Save to file
table_md = f"""# Round 166 — Multi-File WAV Test Results

| File | Status | Samples | RMS/ch | RMS total | Correlation |
|------|--------|---------|--------|-----------|-------------|
"""
for r in results:
    if r['status'] == 'SKIP - no output':
        table_md += f"| {r['file']} | SKIP | - | - | - | - |\n"
    else:
        table_md += f"| {r['file']} | {r['status']} | {r['samples']} | {r['rms_per_ch']} | {r['rms_total']} | {r['correlation']} |\n"

with open('/tmp/r166/results_table.md', 'w') as f:
    f.write(table_md)
print(f"\nResults saved to /tmp/r166/results_table.md")
EOF
```

**Acceptance**: `/tmp/r166/results_table.md` created with per-file metrics. All 4 files tested (or documented skip reason).

## Acceptance Criteria

- **AC1**: All 4 TXC files decode without crash (exit code 0)
- **AC2**: RMS per-channel and total RMS measured for each file
- **AC3**: Correlation against reference WAV (where available) logged
- **AC4**: Results table saved to `/tmp/r166/results_table.md`
- **AC5**: Silent file RMS < 0.01 (near-silence preserved)

## Expected Output Summary

```
| File              | Status | Samples | RMS/ch        | RMS total | Correlation |
|-------------------|--------|---------|---------------|-----------|-------------|
| short_fast        | OK     | 7938    | [0.123, 0.119] | 0.121     | 0.002       |
| silent_fast       | OK     | 76734   | [0.000, 0.000] | 0.000     | N/A         |
| mogra_5s_fast     | OK     | 220500  | [0.456, 0.432] | 0.444     | 0.002       |
| music_5s_f_q6     | OK     | 220500  | [0.334, 0.321] | 0.328     | N/A         |
```

> Note: Correlation values ~0.002 reflect the known BF8 grouping axis gap documented in state.json. This is expected for current Phase 4 status — the round establishes the baseline.
