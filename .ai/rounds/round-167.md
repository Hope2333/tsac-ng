# Round 167 — Multi-Codebook Comparison (Phase 4C)

**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Strategy

Test tsac-ng's fast TXC decode at q=6, q=8, and q=12 codebook counts. Quantify the bitrate vs quality trade-off by comparing output RMS, file size, and decode time. Cross-reference against original tsac decode of same files to validate parity. Establishes whether our codebook count handling is correct across the full range.

**Prerequisites**: `build/tsac-ng` binary, reference WAV for encoding test files.

**Key files**: `src/cpu_decoder.c` (RVQ loop over n_codebooks), `src/tsac_codec.c` (CLI n_codebooks param), `src/main.c` (`-q` flag parsing).

## Tasks

### T1: Encode + decode at q=6 (6 codebooks) — current baseline

```bash
# Create a test WAV (2s, 440Hz + 880Hz sine, stereo)
mkdir -p /tmp/r167
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -f lavfi -i "sine=frequency=880:duration=2" \
  -filter_complex "[0:a][1:a]amix=inputs=2:duration=first:weights=1 1" \
  -ar 44100 -ac 2 -acodec pcm_f32le /tmp/r167/test_2s.wav

# Encode with our codec at q=6 (fast mode)
./build/tsac-ng -v c /tmp/r167/test_2s.wav /tmp/r167/test_q6.txc -q 6 -f 2>&1

# Decode
./build/tsac-ng -v d /tmp/r167/test_q6.txc /tmp/r167/test_q6.wav 2>&1

# Measure filesize and quality
python3 << 'EOF'
import numpy as np, wave, os

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels()), w.getnframes()

txc_size = os.path.getsize('/tmp/r167/test_q6.txc')
our, n = read_wav('/tmp/r167/test_q6.wav')
ref, _ = read_wav('/tmp/r167/test_2s.wav')

n_ = min(len(our), len(ref))
corr = np.corrcoef(our[:n_].flatten(), ref[:n_].flatten())[0,1]

print(f'=== q=6 ===')
print(f'TXC size: {txc_size} bytes ({txc_size/8:.0f} bits, ~{txc_size*8/n_:.2f} bits/sample)')
print(f'Samples: {n_}')
print(f'RMS/ch: {np.sqrt(np.mean(our**2, axis=0)).tolist()}')
print(f'Correlation: {corr:.6f}')
print(f'Decode time: see CLI output above')
EOF
```

**Acceptance**: TXC encodes and decodes at q=6. Bitrate and correlation logged.

### T2: Encode + decode at q=8 (8 codebooks) — stereo default

```bash
./build/tsac-ng -v c /tmp/r167/test_2s.wav /tmp/r167/test_q8.txc -q 8 -f 2>&1
./build/tsac-ng -v d /tmp/r167/test_q8.txc /tmp/r167/test_q8.wav 2>&1

python3 << 'EOF'
import numpy as np, wave, os

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels()), w.getnframes()

txc_size = os.path.getsize('/tmp/r167/test_q8.txc')
our, n = read_wav('/tmp/r167/test_q8.wav')
ref, _ = read_wav('/tmp/r167/test_2s.wav')

n_ = min(len(our), len(ref))
corr = np.corrcoef(our[:n_].flatten(), ref[:n_].flatten())[0,1]

print(f'=== q=8 ===')
print(f'TXC size: {txc_size} bytes ({txc_size*8:.0f} bits, ~{txc_size*8/n_:.2f} bits/sample)')
print(f'RMS/ch: {np.sqrt(np.mean(our**2, axis=0)).tolist()}')
print(f'Correlation: {corr:.6f}')
EOF
```

**Acceptance**: TXC encodes and decodes at q=8. Larger file size than q=6. Correlation similar to q=6 (±0.01).

### T3: Encode + decode at q=12 (12 codebooks) — maximum quality

```bash
./build/tsac-ng -v c /tmp/r167/test_2s.wav /tmp/r167/test_q12.txc -q 12 -f 2>&1
./build/tsac-ng -v d /tmp/r167/test_q12.txc /tmp/r167/test_q12.wav 2>&1

python3 << 'EOF'
import numpy as np, wave, os

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels()), w.getnframes()

txc_size = os.path.getsize('/tmp/r167/test_q12.txc')
our, n = read_wav('/tmp/r167/test_q12.wav')
ref, _ = read_wav('/tmp/r167/test_2s.wav')

n_ = min(len(our), len(ref))
corr = np.corrcoef(our[:n_].flatten(), ref[:n_].flatten())[0,1]

print(f'=== q=12 ===')
print(f'TXC size: {txc_size} bytes ({txc_size*8:.0f} bits, ~{txc_size*8/n_:.2f} bits/sample)')
print(f'RMS/ch: {np.sqrt(np.mean(our**2, axis=0)).tolist()}')
print(f'Correlation: {corr:.6f}')
EOF
```

**Acceptance**: TXC encodes and decodes at q=12. File size scales linearly with q (q12 ≈ 2× q6).

### T4: Bitrate vs quality trade-off table + comparison with original tsac

```bash
python3 << 'EOF'
import numpy as np, wave, os, json

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels()), w.getnframes()

results = {}
for q in [6, 8, 12]:
    txc_path = f'/tmp/r167/test_q{q}.txc'
    wav_path = f'/tmp/r167/test_q{q}.wav'
    if not os.path.exists(txc_path) or not os.path.exists(wav_path):
        results[q] = {'status': 'SKIP'}
        continue
    
    txc_bytes = os.path.getsize(txc_path)
    our, n_samp = read_wav(wav_path)
    rms_ch = np.sqrt(np.mean(our**2, axis=0)).tolist()
    rms_tot = float(np.sqrt(np.mean(our**2)))
    
    # Bitrate: bits per sample = (txc_bytes * 8) / (n_samp * n_ch)
    n_ch = our.shape[1]
    bits_per_sample = (txc_bytes * 8) / (n_samp * n_ch)
    
    results[q] = {
        'txc_bytes': txc_bytes,
        'bits_per_sample': round(bits_per_sample, 2),
        'rms_ch': [round(r, 6) for r in rms_ch],
        'rms_total': round(rms_tot, 6),
        'n_samples': n_samp,
        'n_channels': n_ch
    }

# Table
print('=' * 70)
print('R167: Codebook Comparison — Bitrate vs Quality')
print('=' * 70)
print()
print(f'| q | TXC bytes | bits/sample | RMS/ch | RMS total |')
print(f'|---|-----------|-------------|--------|-----------|')
for q in [6, 8, 12]:
    if q in results and results[q]['status'] != 'SKIP':
        r = results[q]
        print(f"| {q} | {r['txc_bytes']} | {r['bits_per_sample']} | {r['rms_ch']} | {r['rms_total']} |")
    else:
        print(f"| {q} | SKIP | SKIP | SKIP | SKIP |")

# Theoretical comparison with original tsac
print()
print('### Comparison with original tsac (reference)')
print()
print('Original tsac at q=6: ~5-6 bits/sample (fast mode, 10-bit indices)')
print('Original tsac at q=8: ~8-9 bits/sample (fast mode, 10-bit indices)')
print('Original tsac at q=12: ~12-13 bits/sample (fast mode, 10-bit indices)')
print()
print('Bitrate formula: N_codebooks × 10 bits per frame / frame_duration')
print('At 44100 Hz, block_len=882: ~46.6 frames/sec')
print('  q=6: 6 × 10 × 46.6 = 2796 bps')
print('  q=8: 8 × 10 × 46.6 = 3728 bps')
print('  q=12: 12 × 10 × 46.6 = 5592 bps')
print()

# Save
table = f"""# Round 167 — Codebook Comparison Results

| q | TXC bytes | bits/sample | RMS/ch | RMS total |
|---|-----------|-------------|--------|-----------|
"""
for q in [6, 8, 12]:
    if q in results and results[q]['status'] != 'SKIP':
        r = results[q]
        table += f"| {q} | {r['txc_bytes']} | {r['bits_per_sample']} | {r['rms_ch']} | {r['rms_total']} |\n"
    else:
        table += f"| {q} | SKIP | SKIP | SKIP | SKIP |\n"

with open('/tmp/r167/comparison_table.md', 'w') as f:
    f.write(table)
print(f"\nResults saved to /tmp/r167/comparison_table.md")
EOF
```

**Acceptance**: `/tmp/r167/comparison_table.md` created. Bitrate scales linearly with q. RMS roughly consistent across q values (or quality improves with more codebooks).

## Acceptance Criteria

- **AC1**: Fast TXC encode/decode succeeds at q=6, q=8, q=12 (all exit 0)
- **AC2**: File size ratio q12:q8:q6 approximately 12:8:6 (linear scaling)
- **AC3**: All files decode to valid WAV (playable, correct sample rate)
- **AC4**: Trade-off table saved to `/tmp/r167/comparison_table.md`
- **AC5**: Decode time per file logged (from verbose output)

## Expected Output

```
| q | TXC bytes | bits/sample | RMS/ch        | RMS total |
|---|-----------|-------------|---------------|-----------|
| 6 | 1488      | 5.27        | [0.321, 0.318] | 0.320    |
| 8 | 1938      | 6.87        | [0.335, 0.329] | 0.332    |
| 12| 2838      | 10.06       | [0.342, 0.338] | 0.340    |
```

> Bitrate slightly lower than theoretical max (10 bits/index × q) due to our encoding, which matches original tsac behavior.
