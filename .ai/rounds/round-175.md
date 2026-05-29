# Round 175 — Fast TXC Encoder Round-Trip

**Status**: PENDING (Header Planned) | **Date**: 2026-05-29
**Predecessor**: round-174
**Priority**: HIGH — Verify encoder/decoder symmetry

## Strategy — WHY this round exists

All previous rounds in Sub-Phase 4D focus on NORMAL TXC decode. But the encoder (fast TXC mode) has been sitting in the codebase since R146-R148 and has never been validated in a full encode→decode round-trip. This round:
1. Encodes WAV → fast TXC using our encoder
2. Decodes fast TXC → WAV using our decoder
3. Compares the round-trip WAV with the original
4. Measures bitrate/filesize vs original tsac

This is critical because:
- If our encoder and decoder are symmetric (decode(encode(x)) ≈ x), the DAC model transformations are correct
- If our TXC files are smaller/larger than original tsac's at the same quality level, our compression efficiency differs
- Round-trip accuracy is the gold standard for any codec

**Note**: This round targets FAST TXC (not normal TXC), because our encoder only supports fast mode. Normal TXC encoding is a future milestone.

## Key files
- `/home/miao/Projects/tsac-ng/src/main.c` — CLI entry point
- `/home/miao/Projects/tsac-ng/src/tsac_codec.c` — tsac_compress_file, tsac_decompress_file
- `/home/miao/Projects/tsac-ng/src/cpu_decoder.c` — encoder implementation
- `/home/miao/Projects/tsac-ng/src/cpu_encoder.inc` — encoder include
- `/home/miao/Projects/tsac-ng/src/txc_format.c` — txc_write (fast TXC writer)
- `/home/miao/Projects/tsac-ng/build/tsac-ng` — our binary
- `/usr/bin/tsac` — original tsac binary (for comparison)

## Dependencies
- R174 complete (decoder works on multiple file types)
- Original `tsac` binary for comparison encoding
- Test WAV files from R174
- `python3` with `numpy`, `scipy.io.wavfile` for metrics

## Tasks

### T1: Encode WAV → fast TXC with our encoder (⬜)

**Encode test files**:
```bash
cmake --build /home/miao/Projects/tsac-ng/build

# Use the MOGRA test file from R174 (or create a new short test)
# Test: 1s 440Hz sine
python3 -c "
import struct, math
sr=44100; dur=1.0; n=int(sr*dur)
samples=[0.3*math.sin(2*math.pi*440*i/sr) for i in range(n)]
with open('/tmp/r175_sine.wav','wb') as f:
    f.write(b'RIFF'); f.write(struct.pack('<I',36+n*2))
    f.write(b'WAVEfmt '); f.write(struct.pack('<I',16))
    f.write(struct.pack('<H',1)); f.write(struct.pack('<H',2))
    f.write(struct.pack('<I',sr)); f.write(struct.pack('<I',sr*4))
    f.write(struct.pack('<H',4)); f.write(struct.pack('<H',16))
    f.write(b'data'); f.write(struct.pack('<I',n*2))
    for s in samples:
        v = max(-32768, min(32767, int(s*32767)))
        f.write(struct.pack('<h',v))  # L
        f.write(struct.pack('<h',v))  # R
print(f'Created /tmp/r175_sine.wav ({n} samples)')
"

# Encode with our tsac-ng
LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build \
    /home/miao/Projects/tsac-ng/build/tsac-ng \
    -v c /tmp/r175_sine.wav /tmp/r175_sine_ours.txc 2>&1

echo "Our TXC size: $(stat -c%s /tmp/r175_sine_ours.txc) bytes"

# Encode with original tsac (if available)
if command -v tsac &>/dev/null; then
    tsac -v c /tmp/r175_sine.wav /tmp/r175_sine_orig.txc 2>&1
    echo "Original TXC size: $(stat -c%s /tmp/r175_sine_orig.txc) bytes"
fi
```

**Check the encoded TXC format**:
```bash
# Verify it's a valid fast TXC (version=0, not version=1)
xxd /tmp/r175_sine_ours.txc | head -4
# Bytes 4-5 should be 0x0000 for fast TXC

# Count frames and codebooks
python3 -c "
with open('/tmp/r175_sine_ours.txc','rb') as f:
    data=f.read()
n_cb=data[7]
# header end detection
for h in range(8,256):
    if (len(data)-h) % n_cb == 0:
        frame_count=(len(data)-h)//n_cb
        print(f'Frames: {frame_count}, Codebooks: {n_cb}, Header: {h}B')
        break
"
```

**Acceptance**: Our encoder produces a valid fast TXC file. File size measured.

---

### T2: Decode our fast TXC and compare with original WAV (⬜)

**Round-trip decode**:
```bash
# Decode our TXC
LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build \
    /home/miao/Projects/tsac-ng/build/tsac-ng \
    -v d /tmp/r175_sine_ours.txc /tmp/r175_sine_roundtrip.wav 2>&1

echo "Round-trip WAV size: $(stat -c%s /tmp/r175_sine_roundtrip.wav) bytes"
```

**Compare round-trip vs original**:
```bash
python3 << 'EOF'
import numpy as np
import struct

def read_wav_samples(path):
    """Read WAV file and return mono float32 samples."""
    with open(path, 'rb') as f:
        raw = f.read()
    if len(raw) < 44:
        return None
    channels = int.from_bytes(raw[22:24], 'little')
    sr = int.from_bytes(raw[24:28], 'little')
    bits_per_sample = int.from_bytes(raw[34:36], 'little')
    data = raw[44:]
    
    if bits_per_sample == 16:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 32:
        # Could be float or int32
        if raw[20] == 3:  # IEEE float
            samples = np.frombuffer(data, dtype=np.float32)
        else:
            samples = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return None
    
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    
    return samples, sr

orig, sr_orig = read_wav_samples('/tmp/r175_sine.wav')
rtrip, sr_rt = read_wav_samples('/tmp/r175_sine_roundtrip.wav')

if orig is None or rtrip is None:
    print("ERROR: Could not read WAV files")
    exit(1)

print(f"Original:  {len(orig)} samples @ {sr_orig}Hz")
print(f"Roundtrip: {len(rtrip)} samples @ {sr_rt}Hz")

# Align
min_len = min(len(orig), len(rtrip))
orig = orig[:min_len]
rtrip = rtrip[:min_len]

# Basic stats
print(f"\nOriginal RMS:  {np.sqrt(np.mean(orig**2)):.6f}")
print(f"Roundtrip RMS: {np.sqrt(np.mean(rtrip**2)):.6f}")

# Correlation
if np.std(orig) > 0 and np.std(rtrip) > 0:
    corr = np.corrcoef(orig, rtrip)[0, 1]
    print(f"Correlation: {corr:.6f}")
else:
    print("Correlation: N/A (one signal is flat)")
    corr = 0.0

# Error metrics
diff = orig - rtrip
mae = np.mean(np.abs(diff))
rmse = np.sqrt(np.mean(diff**2))
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

# SNR
signal_power = np.mean(orig**2)
noise_power = np.mean(diff**2)
if noise_power > 1e-12:
    snr = 10 * np.log10(signal_power / noise_power)
    print(f"SNR:  {snr:.2f} dB")
else:
    print("SNR:  Infinity (perfect reconstruction)")

# Save metrics for results table
metrics = {
    "correlation": float(corr),
    "mae": float(mae),
    "rmse": float(rmse),
    "snr_db": float(snr) if noise_power > 1e-12 else float('inf'),
    "orig_rms": float(np.sqrt(np.mean(orig**2))),
    "rtrip_rms": float(np.sqrt(np.mean(rtrip**2))),
}
import json
with open('/tmp/r175_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to /tmp/r175_metrics.json")
EOF
```

**Acceptance**: Round-trip decode succeeds. Metrics (correlation, SNR) recorded.

---

### T3: Compare with original tsac encode→decode (⬜)

**If original tsac is available**:
```bash
# Encode with original tsac
tsac -v c /tmp/r175_sine.wav /tmp/r175_sine_orig_t.txc 2>&1

# Decode with original tsac
tsac -v d /tmp/r175_sine_orig_t.txc /tmp/r175_sine_orig_rt.wav 2>&1

# Also decode our TXC with original tsac (to isolate encode vs decode issues)
tsac -v d /tmp/r175_sine_ours.txc /tmp/r175_ours_by_orig_decoder.wav 2>&1
```

**Cross-comparison**:
```bash
python3 << 'EOF'
import numpy as np
import struct, json

def read_wav_samples(path):
    with open(path, 'rb') as f:
        raw = f.read()
    if len(raw) < 44: return None, None
    channels = int.from_bytes(raw[22:24], 'little')
    sr = int.from_bytes(raw[24:28], 'little')
    data = raw[44:]
    samples = np.frombuffer(data, dtype=np.float32)
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples, sr

# Load all WAVs
orig, sr = read_wav_samples('/tmp/r175_sine.wav')
our_rt, _ = read_wav_samples('/tmp/r175_sine_roundtrip.wav')
orig_rt, _ = read_wav_samples('/tmp/r175_sine_orig_rt.wav')
cross, _ = read_wav_samples('/tmp/r175_ours_by_orig_decoder.wav')

# Case 1: Our encoder + our decoder vs original (our round-trip quality)
# Case 2: Original encoder + original decoder (original's round-trip quality)  
# Case 3: Our encoder + original decoder (encoder quality isolation)
# Case 4: Original encoder + our decoder (decoder quality isolation - already tested in R166-170)

def compare(label, a, b):
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    corr = np.corrcoef(a, b)[0,1] if np.std(a)>0 and np.std(b)>0 else 0
    rmse = np.sqrt(np.mean((a-b)**2))
    print(f"{label:40s} corr={corr:.6f}  rmse={rmse:.6f}")

print("=== Comparison Matrix ===\n")

if our_rt is not None:
    compare("Our Encode→Decode vs Original WAV:", our_rt, orig)
if orig_rt is not None:
    compare("Orig Encode→Decode vs Original WAV:", orig_rt, orig)
if cross is not None:
    compare("Our Encode→Orig Decode vs Original:", cross, orig)
    compare("Our Encode→Orig Decode vs Our RT:", cross, our_rt)

# Bitrate comparison
import os
our_txc_size = os.path.getsize('/tmp/r175_sine_ours.txc')
orig_txc_size = os.path.getsize('/tmp/r175_sine_orig_t.txc')
duration = len(orig) / sr if orig is not None else 0

print(f"\n=== Bitrate Comparison ===")
print(f"Our TXC:    {our_txc_size} bytes ({our_txc_size*8/duration/1000:.2f} kbps @ {duration:.2f}s)")
print(f"Orig TXC:   {orig_txc_size} bytes ({orig_txc_size*8/duration/1000:.2f} kbps @ {duration:.2f}s)")
print(f"Ratio:      {our_txc_size/orig_txc_size:.2%}")

# Save comparison results
results = {
    "our_encode_our_decode_vs_orig": {
        "correlation": float(np.corrcoef(our_rt[:min(len(our_rt),len(orig))], 
                                          orig[:min(len(our_rt),len(orig))])[0,1]) if our_rt is not None else None,
        "our_txc_bytes": our_txc_size,
        "orig_txc_bytes": orig_txc_size,
        "bitrate_ratio": our_txc_size/orig_txc_size,
    }
}
with open('/tmp/r175_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to /tmp/r175_comparison.json")
EOF
```

**Acceptance**: Cross-comparison matrix completed. Bitrate comparison documented.

---

### T4: Multi-file round-trip test + results table (⬜)

**Test with multiple audio types**:
```bash
# Test files to encode
FILES=(
    "/tmp/r175_sine.wav"
)

# Add files from R174 if they exist
for f in /tmp/r174_mogra_5s.wav /tmp/r174_silent_5s.wav; do
    if [ -f "$f" ]; then
        FILES+=("$f")
    fi
done

for wav in "${FILES[@]}"; do
    base=$(basename "$wav" .wav)
    echo "=== Processing $base ==="
    
    # Encode with our encoder
    LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build \
        /home/miao/Projects/tsac-ng/build/tsac-ng \
        -v c "$wav" "/tmp/r175_${base}_ours.txc" 2>&1
    
    # Encode with original (if available)
    if command -v tsac &>/dev/null; then
        tsac -v c "$wav" "/tmp/r175_${base}_orig.txc" 2>&1
    fi
    
    # Decode our TXC
    LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build \
        /home/miao/Projects/tsac-ng/build/tsac-ng \
        -v d "/tmp/r175_${base}_ours.txc" "/tmp/r175_${base}_rtrip.wav" 2>&1
    
    echo ""
done
```

**Create comprehensive results script `/home/miao/Projects/tsac-ng/experimental/tests/compile_results_175.py`**:
```python
#!/usr/bin/env python3
"""Compile fast TXC encoder round-trip results."""

import numpy as np
import json, os, glob, struct

RESULTS_JSON = "/home/miao/Projects/tsac-ng/docs/FAST_TXC_ROUNDTRIP_R175.json"
RESULTS_MD = "/home/miao/Projects/tsac-ng/docs/FAST_TXC_ROUNDTRIP_R175.md"

def read_wav(path):
    if not os.path.exists(path):
        return None, None, None
    with open(path, 'rb') as f:
        raw = f.read()
    if len(raw) < 44:
        return None, None, None
    channels = int.from_bytes(raw[22:24], 'little')
    sr = int.from_bytes(raw[24:28], 'little')
    fmt = int.from_bytes(raw[20:22], 'little')
    data = raw[44:]
    if fmt == 3:  # IEEE float
        samples = np.frombuffer(data, dtype=np.float32)
    elif fmt == 1:  # PCM
        bits = int.from_bytes(raw[34:36], 'little')
        if bits == 16:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            return None, None, None
    else:
        return None, None, None
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples, sr, channels

def main():
    test_files = []
    for path in sorted(glob.glob("/tmp/r175_*_ours.txc")):
        base = path.replace("_ours.txc", "").replace("/tmp/r175_", "")
        test_files.append({
            "name": base,
            "txc_ours": path,
            "txc_orig": path.replace("_ours.txc", "_orig.txc"),
            "wav_orig": glob.glob(f"/tmp/r175_{base}.wav") + [f"/tmp/r174_{base}.wav"],
            "rtrip_wav": path.replace("_ours.txc", "_rtrip.wav"),
        })
    
    results = []
    for tf in test_files:
        print(f"\n=== {tf['name']} ===")
        
        wav_orig_path = None
        for p in tf['wav_orig']:
            if os.path.exists(p):
                wav_orig_path = p
                break
        
        orig, sr, ch = read_wav(wav_orig_path)
        rtrip, _, _ = read_wav(tf['rtrip_wav'])
        
        our_txc_size = os.path.getsize(tf['txc_ours']) if os.path.exists(tf['txc_ours']) else 0
        orig_txc_size = os.path.getsize(tf['txc_orig']) if os.path.exists(tf['txc_orig']) else 0
        
        entry = {
            "file": tf['name'],
            "our_txc_bytes": our_txc_size,
            "orig_txc_bytes": orig_txc_size,
            "bitrate_ratio": our_txc_size / orig_txc_size if orig_txc_size > 0 else None,
        }
        
        if orig is not None and rtrip is not None:
            min_len = min(len(orig), len(rtrip))
            o, r = orig[:min_len], rtrip[:min_len]
            
            entry["orig_rms"] = float(np.sqrt(np.mean(o**2)))
            entry["rtrip_rms"] = float(np.sqrt(np.mean(r**2)))
            entry["correlation"] = float(np.corrcoef(o, r)[0, 1]) if np.std(o)>0 and np.std(r)>0 else 0.0
            entry["rmse"] = float(np.sqrt(np.mean((o-r)**2)))
            
            signal_power = np.mean(o**2)
            noise_power = np.mean((o-r)**2)
            entry["snr_db"] = float(10*np.log10(signal_power/noise_power)) if noise_power > 1e-12 else float('inf')
            
            print(f"  RMS orig={entry['orig_rms']:.4f} rtrip={entry['rtrip_rms']:.4f}")
            print(f"  Correlation: {entry['correlation']:.6f}")
            print(f"  SNR: {entry['snr_db']:.1f} dB")
        else:
            print(f"  WARNING: Cannot compare (orig={orig is not None}, rtrip={rtrip is not None})")
        
        print(f"  TXC sizes: our={our_txc_size}B, orig={orig_txc_size}B")
        
        results.append(entry)
    
    # Save JSON
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_JSON}")
    
    # Generate markdown
    with open(RESULTS_MD, 'w') as f:
        f.write("# Fast TXC Encoder Round-Trip Results (R175)\n\n")
        f.write("**Date**: 2026-05-29\n\n")
        f.write("| File | Our TXC | Orig TXC | Ratio | Corr | RMSE | SNR |\n")
        f.write("|------|---------|----------|-------|------|------|-----|\n")
        for r in results:
            corr = f"{r.get('correlation', 'N/A'):.4f}" if 'correlation' in r else "N/A"
            rmse = f"{r.get('rmse', 'N/A'):.6f}" if 'rmse' in r else "N/A"
            snr = f"{r.get('snr_db', 'N/A'):.1f}" if 'snr_db' in r else "N/A"
            ratio = f"{r.get('bitrate_ratio', 'N/A'):.2%}" if r.get('bitrate_ratio') else "N/A"
            f.write(f"| {r['file']} | {r['our_txc_bytes']} | {r['orig_txc_bytes']} | {ratio} | {corr} | {rmse} | {snr} |\n")
    
    print(f"Markdown table saved to {RESULTS_MD}")

if __name__ == "__main__":
    main()
```

**Acceptance**: Multi-file round-trip test results table produced. Both JSON and markdown.

---

### T5: Document encoder quality + known issues (⬜)

**Update `/home/miao/Projects/tsac-ng/docs/FAST_TXC_STATUS.md`** (or create if not exists):

```markdown
# Fast TXC Encoder Status (R175)

## Round-Trip Quality
- Our encoder + our decoder: [correlation from results]
- Our encoder + original decoder: [correlation] (encoder quality isolation)
- Original encoder + our decoder: [correlation] (decoder quality isolation, from R166-170)

## Bitrate Comparison
| File | Our bitrate | Original bitrate | Ratio |
|------|-------------|------------------|-------|
| ...  | ... kbps    | ... kbps         | ...   |

## Known Issues
1. [Issue 1]: ...
2. [Issue 2]: ...
```

**Also create summary update for Sub-Phase 4D**:
```markdown
# Sub-Phase 4D Summary (R171-R175)

## Deliverables
- R171: Normal TXC end-to-end integration ✅/⬜
- R172: Transformer output validation ✅/⬜
- R173: Range coder edge case testing ✅/⬜
- R174: Normal TXC multi-file testing ✅/⬜
- R175: Fast TXC encoder round-trip ✅/⬜

## Normal TXC Decode Status
- Pipeline: [wired/partial/broken]
- Index accuracy: [%]
- WAV correlation (best): [value]

## Fast TXC Encoder Status
- Round-trip correlation: [value]
- Bitrate ratio vs original: [value]

## Decision Gate
- [ ] G5: Normal TXC indices match >80% → Yes/No
- [ ] Proceed to Sub-Phase 4E (R176-R180) → Yes/No
```

**Acceptance**: All documentation updated. Sub-Phase 4D completion status recorded for Header.

---

## Acceptance Criteria

- [ ] **T1**: Our encoder produces valid fast TXC for at least sine WAV. File size measured.
- [ ] **T2**: Our encode→decode round-trip succeeds. Correlation > 0 measured (even if low).
- [ ] **T3**: Cross-comparison with original tsac (encode, decode) completed. Matrix documented.
- [ ] **T4**: Multi-file round-trip results table at `docs/FAST_TXC_ROUNDTRIP_R175.md`.
- [ ] **T5**: Status documentation updated. Sub-Phase 4D summary created.
- [ ] `cmake --build build` succeeds.
- [ ] Fast TXC encode + decode command-line verified for all test files.

## Decision Gate (Sub-Phase 4D Completion)

| Condition | Action |
|-----------|--------|
| Our encode→decode correlation > 0.5 | → Sub-Phase 4E (R176-R180): Quality + Release |
| Our encode→decode correlation > 0 | → Proceed to 4E, note quality gap |
| Encode produces invalid TXC | Fix encoder, retry R175 |
| Decode of our TXC fails completely | → R171 (decoder regression) |
