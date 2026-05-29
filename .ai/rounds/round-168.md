# Round 168 — Stereo + Mono Validation (Phase 4C)

**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Strategy

Validate tsac-ng's channel handling across stereo (2ch), mono (1ch), and `separate_channels` mode. For stereo, verify that L/R correlation is correct (stereo content should show channel-specific differences). For mono, verify channel count is reduced appropriately. Separate_channels mode should process each channel independently. This is critical for correct multi-channel audio handling.

**Prerequisites**: `build/tsac-ng` binary, stereo and mono test WAV files.

**Key files**: `src/tsac_codec.c` (channel count handling, `-c` and `-s` flags), `src/cpu_decoder.c` (channel dimension in decoder graph), `src/main.c` (CLI flag parsing).

## Tasks

### T1: Test stereo decode (2 channels) — verify L/R correlation

```bash
mkdir -p /tmp/r168

# Use existing stereo test content
# Create a stereo WAV with known channel differences
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -f lavfi -i "sine=frequency=880:duration=2" \
  -filter_complex "[0:a][1:a]amerge=inputs=2" -ar 44100 -ac 2 -acodec pcm_f32le /tmp/r168/stereo_ref.wav

# Encode as fast TXC
./build/tsac-ng -v c /tmp/r168/stereo_ref.wav /tmp/r168/stereo.txc -q 8 -f 2>&1

# Decode
./build/tsac-ng -v d /tmp/r168/stereo.txc /tmp/r168/stereo_decoded.wav 2>&1

# Measure L/R correlation
python3 << 'EOF'
import numpy as np, wave

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels()), w.getnframes(), w.getnchannels()

our, n, ch = read_wav('/tmp/r168/stereo_decoded.wav')
ref, _, _ = read_wav('/tmp/r168/stereo_ref.wav')

print(f'=== STEREO (2ch) ===')
print(f'Channels: {ch}')
print(f'Samples per ch: {n}')

# Per-channel metrics
for c in range(ch):
    print(f'  Ch{c}: RMS={np.sqrt(np.mean(our[:,c]**2)):.6f}')

# L/R correlation in OUR output
lr_corr = np.corrcoef(our[:,0], our[:,1])[0,1]
print(f'  L/R correlation (our): {lr_corr:.6f}')

# L/R correlation in REFERENCE
ref_lr_corr = np.corrcoef(ref[:,0], ref[:,1])[0,1]
print(f'  L/R correlation (ref):  {ref_lr_corr:.6f}')

# Cross-correlation with reference per channel
n_ = min(len(our), len(ref))
for c in range(ch):
    corr = np.corrcoef(our[:n_,c], ref[:n_,c])[0,1]
    print(f'  Ch{c} vs ref correlation: {corr:.6f}')
EOF
```

**Acceptance**: Stereo WAV decodes with 2 channels. L/R correlation matches reference pattern (left=440Hz, right=880Hz → low L/R corr in both our and ref).

### T2: Test mono decode (1 channel, -c 1)

```bash
# Create mono reference
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 44100 -ac 1 -acodec pcm_f32le /tmp/r168/mono_ref.wav

# Encode mono TXC (our codec should auto-detect mono from 1-channel WAV)
./build/tsac-ng -v c /tmp/r168/mono_ref.wav /tmp/r168/mono.txc -q 8 -f 2>&1

# Decode — should produce mono output
./build/tsac-ng -v d /tmp/r168/mono.txc /tmp/r168/mono_decoded.wav 2>&1

# Also test forcing mono on stereo content with -c 1
./build/tsac-ng -v c /tmp/r168/stereo_ref.wav /tmp/r168/stereo_as_mono.txc -q 8 -f -c 1 2>&1
./build/tsac-ng -v d /tmp/r168/stereo_as_mono.txc /tmp/r168/stereo_as_mono_decoded.wav 2>&1

python3 << 'EOF'
import numpy as np, wave

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels()), w.getnframes(), w.getnchannels()

# Mono decoded from mono source
our, n, ch = read_wav('/tmp/r168/mono_decoded.wav')
ref, _, _ = read_wav('/tmp/r168/mono_ref.wav')
n_ = min(len(our), len(ref))
corr = np.corrcoef(our[:n_].flatten(), ref[:n_].flatten())[0,1]
print(f'=== MONO (source=mono) ===')
print(f'Channels: {ch}, Samples: {n}')
print(f'RMS: {np.sqrt(np.mean(our**2)):.6f}')
print(f'Correlation with ref: {corr:.6f}')

# Mono decoded from stereo source with -c 1
our2, n2, ch2 = read_wav('/tmp/r168/stereo_as_mono_decoded.wav')
print(f'\n=== MONO (stereo source, -c 1) ===')
print(f'Channels: {ch2}, Samples: {n2}')
print(f'RMS: {np.sqrt(np.mean(our2**2)):.6f}')
EOF
```

**Acceptance**: Mono decode produces 1-channel WAV. Forced mono from stereo source works.

### T3: Test separate_channels mode (-s)

```bash
# Encode stereo with separate_channels
./build/tsac-ng -v c /tmp/r168/stereo_ref.wav /tmp/r168/stereo_sep.txc -q 8 -f -s 2>&1

# Decode
./build/tsac-ng -v d /tmp/r168/stereo_sep.txc /tmp/r168/stereo_sep_decoded.wav 2>&1

python3 << 'EOF'
import numpy as np, wave

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels()), w.getnframes(), w.getnchannels()

our, n, ch = read_wav('/tmp/r168/stereo_sep_decoded.wav')
print(f'=== SEPARATE CHANNELS (-s) ===')
print(f'Channels: {ch}, Samples per ch: {n}')

# Compare with regular stereo decode
reg, _, _ = read_wav('/tmp/r168/stereo_decoded.wav')
n_ = min(len(our), len(reg))
for c in range(min(ch, reg.shape[1])):
    diff = np.abs(our[:n_,c] - reg[:n_,c])
    print(f'  Ch{c}: max diff from regular stereo = {np.max(diff):.6f}')

# If -s mode processed channels independently, they may differ from joint stereo
print(f'  Total RMS: {np.sqrt(np.mean(our**2)):.6f}')
print(f'  Regular stereo RMS: {np.sqrt(np.mean(reg[:n_]**2)):.6f}')
EOF
```

**Acceptance**: separate_channels mode produces valid stereo WAV. Output may differ from joint stereo encoding (by design, since each channel is quantized independently).

### T4: L/R correlation analysis across all modes

```bash
python3 << 'EOF'
import numpy as np, wave, os

def read_wav(path):
    if not os.path.exists(path):
        return None, 0, 0
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels()), w.getnframes(), w.getnchannels()

tests = [
    ('Stereo (joint)', '/tmp/r168/stereo_decoded.wav'),
    ('Mono (from mono)', '/tmp/r168/mono_decoded.wav'),
    ('Mono (stereo→mono)', '/tmp/r168/stereo_as_mono_decoded.wav'),
    ('Separate channels', '/tmp/r168/stereo_sep_decoded.wav'),
]

print('=' * 60)
print('R168: Channel Mode Comparison')
print('=' * 60)
print(f'| Mode | Ch | Samples | RMS | L/R corr |')
print(f'|------|----|---------|-----|----------|')
for name, path in tests:
    data, n, ch = read_wav(path)
    if data is None:
        print(f'| {name} | SKIP | - | - | - |')
        continue
    rms = float(np.sqrt(np.mean(data**2)))
    if ch == 2:
        lr_corr = np.corrcoef(data[:,0], data[:,1])[0,1]
    else:
        lr_corr = 1.0  # mono — perfect self-correlation
    print(f'| {name} | {ch} | {n} | {rms:.6f} | {lr_corr:.6f} |')

# Save results
table = f"""# Round 168 — Channel Mode Validation Results

| Mode | Ch | Samples | RMS | L/R corr |
|------|----|---------|-----|----------|
"""
for name, path in tests:
    data, n, ch = read_wav(path)
    if data is None:
        table += f"| {name} | SKIP | - | - | - |\n"
        continue
    rms = float(np.sqrt(np.mean(data**2)))
    if ch == 2:
        lr_corr = np.corrcoef(data[:,0], data[:,1])[0,1]
    else:
        lr_corr = 1.0
    table += f"| {name} | {ch} | {n} | {rms:.6f} | {lr_corr:.6f} |\n"

os.makedirs('/tmp/r168', exist_ok=True)
with open('/tmp/r168/channel_comparison.md', 'w') as f:
    f.write(table)
print(f"\nResults saved to /tmp/r168/channel_comparison.md")
EOF
```

**Acceptance**: `/tmp/r168/channel_comparison.md` created with per-mode metrics.

## Acceptance Criteria

- **AC1**: Stereo decode produces 2-channel WAV with correct sample rate
- **AC2**: Mono decode produces 1-channel WAV
- **AC3**: `-c 1` flag successfully forces mono encoding from stereo source
- **AC4**: `-s` (separate_channels) flag produces output without errors
- **AC5**: L/R correlation analysis saved to `/tmp/r168/channel_comparison.md`
- **AC6**: All channel modes produce playable WAV files

## Expected Output

```
| Mode                    | Ch | Samples | RMS       | L/R corr |
|-------------------------|----|---------|-----------|----------|
| Stereo (joint)          | 2  | 88200   | 0.321000  | -0.012   |
| Mono (from mono)        | 1  | 88200   | 0.318000  | 1.000    |
| Mono (stereo→mono)      | 1  | 88200   | 0.315000  | 1.000    |
| Separate channels       | 2  | 88200   | 0.320000  | -0.010   |
```
