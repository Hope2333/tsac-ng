# Round 174 — Normal TXC Multi-File Testing (Phase 4D)

**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Strategy

Extend the normal TXC decode testing from R171 (single file) to multiple content types: MOGRA (mixed music/speech), music_1s (short music clip), and silent_5s (silence with room tone). This validates the Transformer + range coder pipeline across diverse audio content and produces a multi-file results table similar to R166 but for normal mode.

**Prerequisites**: R171 integration completed (normal TXC decode pipeline wired), `build/tsac-ng` binary, normal TXC test files for all 3 content types.

**Key files**: `src/tsac_normal_decode.c`, `src/tsac_transformer.c`, `src/tsac_codec.c`, `src/range_coder.c`.

## Tasks

### T1: Test with MOGRA 5s normal TXC

```bash
mkdir -p /tmp/r174

# If MOGRA normal TXC doesn't exist, create it:
# Using original tsac (tsac c) or our encoder

# Check if test file exists
ls -la test-simples/*.txc 2>/dev/null | grep -i mogra

# The MOGRA file with @b96s suffix is likely fast mode
# Need to create a normal mode version:
# 1. Extract 5s from the Opus source
# 2. Encode with original tsac in normal mode

# Decode MOGRA normal TXC
./build/tsac-ng -v d test-simples/mogra_5s_normal_q6.txc /tmp/r174/mogra_normal_output.wav 2>&1 | tee /tmp/r174/mogra_decode.log

# Check for errors
grep -c "error\|Error\|FAIL\|failed" /tmp/r174/mogra_decode.log

python3 << 'EOF'
import numpy as np, wave, os

def read_wav(path):
    if not os.path.exists(path):
        return None, 0
    try:
        with wave.open(path, 'rb') as w:
            frames = w.readframes(w.getnframes())
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return data.reshape(-1, w.getnchannels()), w.getnframes()
    except:
        return None, 0

our, n = read_wav('/tmp/r174/mogra_normal_output.wav')
if our is not None:
    print(f'MOGRA normal decode: {our.shape}')
    print(f'RMS: {np.sqrt(np.mean(our**2)):.6f}')
    print(f'Max abs: {np.max(np.abs(our)):.6f}')
    
    # Compare with original tsac decode of same file
    ref, _ = read_wav('test-simples/mogra_5s_normal_orig.wav')
    if ref is not None:
        n_ = min(n, len(ref))
        corr = np.corrcoef(our[:n_].flatten(), ref[:n_].flatten())[0,1]
        print(f'Correlation with original: {corr:.6f}')
else:
    print('MOGRA normal decode FAILED — see mogra_decode.log')
    with open('/tmp/r174/mogra_decode.log') as f:
        print(f.read())
EOF
```

**Acceptance**: MOGRA normal TXC decode attempted. Results (success or failure) documented.

### T2: Test with music_1s_normal_q6.txc

```bash
# Create music_1s_normal_q6.txc if needed:
# ffmpeg -t 1 -i some_music.wav -ar 44100 -ac 2 /tmp/r174/music_1s_ref.wav
# tsac c /tmp/r174/music_1s_ref.wav music_1s_normal_q6.txc -q 6

# Decode
./build/tsac-ng -v d test-simples/music_1s_normal_q6.txc /tmp/r174/music_normal_output.wav 2>&1 | tee /tmp/r174/music_decode.log

python3 << 'EOF'
import numpy as np, wave, os

def read_wav(path):
    if not os.path.exists(path):
        return None, 0
    with wave.open(path, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0, w.getnframes()

our, n = read_wav('/tmp/r174/music_normal_output.wav')
if our is not None:
    our = our.reshape(-1, 2) if our.ndim == 1 and len(our) > 0 else our
    print(f'Music normal decode: {our.shape}')
    print(f'RMS: {np.sqrt(np.mean(our**2)):.6f}')
    
    # Reference comparison
    ref_path = '/tmp/r174/music_1s_ref.wav'
    if os.path.exists(ref_path):
        ref, _ = read_wav(ref_path)
        n_ = min(len(our), len(ref))
        corr = np.corrcoef(our[:n_].flatten(), ref[:n_].flatten())[0,1]
        print(f'Correlation with reference: {corr:.6f}')
else:
    print('Music normal decode FAILED')
EOF
```

**Acceptance**: Music normal TXC decode attempted. Results documented.

### T3: Test with silent_5s_normal_q6.txc

```bash
# Create silent_5s_normal_q6.txc if needed:
# ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t 5 -acodec pcm_f32le /tmp/r174/silent_5s_ref.wav
# tsac c /tmp/r174/silent_5s_ref.wav silent_5s_normal_q6.txc -q 6

# Decode
./build/tsac-ng -v d test-simples/silent_5s_normal_q6.txc /tmp/r174/silent_normal_output.wav 2>&1 | tee /tmp/r174/silent_decode.log

python3 << 'EOF'
import numpy as np, wave, os

def read_wav(path):
    if not os.path.exists(path):
        return None
    with wave.open(path, 'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

our = read_wav('/tmp/r174/silent_normal_output.wav')
if our is not None:
    rms = np.sqrt(np.mean(our**2))
    print(f'Silent normal decode: RMS={rms:.6f}')
    print(f'Silence check: {"✅ PASS (< 0.01)" if rms < 0.01 else "❌ FAIL (> 0.01)"}')
else:
    print('Silent normal decode FAILED')
EOF
```

**Acceptance**: Silent normal TXC decode attempted. Near-zero RMS expected for silent input.

### T4: Multi-file normal TXC results table

```bash
python3 << 'EOF'
import numpy as np, wave, os, json

def read_wav(path):
    if not os.path.exists(path):
        return None
    try:
        with wave.open(path, 'rb') as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            ch = w.getnchannels()
            if ch > 1:
                data = data.reshape(-1, ch)
            return data
    except:
        return None

files = [
    ('MOGRA 5s normal', '/tmp/r174/mogra_normal_output.wav', '/tmp/r174/mogra_decode.log'),
    ('Music 1s normal', '/tmp/r174/music_normal_output.wav', '/tmp/r174/music_decode.log'),
    ('Silent 5s normal', '/tmp/r174/silent_normal_output.wav', '/tmp/r174/silent_decode.log'),
]

results = []
for name, wav_path, log_path in files:
    entry = {'file': name}
    
    # Check log for errors
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = f.read()
            entry['has_errors'] = any(e in log.lower() for e in ['error', 'fail', 'abort', 'segmentation'])
    else:
        entry['has_errors'] = True
    
    # Check WAV
    data = read_wav(wav_path)
    if data is None:
        entry['status'] = 'FAIL'
        entry['rms'] = 'N/A'
        entry['shape'] = 'N/A'
    else:
        entry['status'] = 'OK'
        entry['rms'] = f'{np.sqrt(np.mean(data**2)):.6f}'
        entry['shape'] = str(data.shape)
    
    results.append(entry)

# Print table
print('=' * 70)
print('R174: Normal TXC Multi-File Results')
print('=' * 70)
print()
print(f'| File | Status | Shape | RMS | Errors in log |')
print(f'|------|--------|-------|-----|---------------|')
for r in results:
    err_str = 'YES' if r.get('has_errors', True) else 'no'
    print(f"| {r['file']} | {r['status']} | {r['shape']} | {r['rms']} | {err_str} |")

# Save table
table = f"""# Round 174 — Normal TXC Multi-File Results

| File | Status | Shape | RMS | Errors in log |
|------|--------|-------|-----|---------------|
"""
for r in results:
    err_str = 'YES' if r.get('has_errors', True) else 'no'
    table += f"| {r['file']} | {r['status']} | {r['shape']} | {r['rms']} | {err_str} |\n"

os.makedirs('/tmp/r174', exist_ok=True)
with open('/tmp/r174/normal_multi_file_results.md', 'w') as f:
    f.write(table)
print(f'\nResults saved to /tmp/r174/normal_multi_file_results.md')
EOF
```

**Acceptance**: Multi-file results table saved to `/tmp/r174/normal_multi_file_results.md`.

## Acceptance Criteria

- **AC1**: MOGRA 5s normal TXC decode attempted, result documented
- **AC2**: Music 1s normal TXC decode attempted, result documented
- **AC3**: Silent 5s normal TXC decode attempted, result documented
- **AC4**: Multi-file results table saved to `/tmp/r174/normal_multi_file_results.md`
- **AC5**: Error logs reviewed for each file
- **AC6**: Comparison with original tsac output (where available)

## Expected Output

```
| File                 | Status | Shape        | RMS       | Errors in log |
|----------------------|--------|--------------|-----------|---------------|
| MOGRA 5s normal      | OK     | (220500, 2)  | 0.012000  | no            |
| Music 1s normal      | OK     | (44100, 2)   | 0.008000  | no            |
| Silent 5s normal     | OK     | (220500, 2)  | 0.000200  | no            |
```

> Note: RMS values may be very low in early runs if the Transformer + range coder pipeline produces near-constant or zero codebook indices. This is expected for the first multi-file test pass and establishes the baseline.
