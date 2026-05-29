# Round 174 — Normal TXC Multi-File Testing

**Status**: PENDING (Header Planned) | **Date**: 2026-05-29
**Predecessor**: round-173
**Priority**: MEDIUM — Validate across multiple audio types

## Strategy — WHY this round exists

R171-R173 verified that normal TXC decode works on a single file and that the range coder handles edge cases. But neural audio codecs are notoriously non-robust to out-of-distribution inputs. The Transformer's probability predictions may work well for music but fail for silence, or vice versa.

This round tests normal TXC decode across 3 distinct audio types:
- **MOGRA 5s**: DJ set excerpt — complex polyphonic music, wide frequency range
- **music_1s**: Short music clip — transient-rich, tests frame-level prediction
- **silent_5s**: Silence — tests whether Transformer predicts correct symbol ids for zeros

Each file is decoded with both our pipeline and (if available) the original tsac. We compare:
1. **Decode success**: Does it complete without error?
2. **Output RMS**: Is the amplitude reasonable?
3. **WAV correlation**: How close are we to original tsac output?
4. **File size**: Does our decode produce the same number of samples?

Results are compiled into a standardized results table that feeds into R175 (encoder round-trip) and R176 (quality optimization).

## Key files
- `/home/miao/Projects/tsac-ng/build/tsac-ng` — our decoder binary
- `/home/miao/Projects/tsac-ng/experimental/tests/test_normal_txc.sh` — regression test (from R171)
- `/home/miao/Projects/tsac-ng/docs/NORMAL_TXC_STATUS.md` — status doc (from R171)
- `/home/miao/Projects/tsac-ng/test-simples/` — test files directory

## Test files required

| File | Description | Expected size | Source |
|------|-------------|---------------|--------|
| MOGRA_5s_normal_q6.txc | DJ set, 5s, q6 | ~5 KB | Encode from test-simples MOGRA |
| music_1s_normal_q6.txc | Music, 1s, q6 | ~1 KB | Encode or use existing |
| silent_5s_normal_q6.txc | Silence, 5s, q6 | ~1 KB | Encode from generated silent WAV |

If these specific files don't exist, create them by encoding with the original tsac:
```bash
# Create test WAVs → encode with original tsac normal mode
```

## Dependencies
- R173 complete (range coder verified)
- Original `tsac` binary for encoding test files (if files don't exist yet)
- `python3` with `numpy` and `scipy` for metrics

## Tasks

### T1: Create MOGRA 5s normal TXC test file and decode (⬜)

**If MOGRA normal TXC doesn't exist, create it**:
```bash
# Extract 5s from the Opus file
ffmpeg -y -i "test-simples/4.8(wed)MOGRA × #DSPM presents ぷらぷらうんじ.opus" \
  -t 5 -ar 44100 -ac 2 /tmp/r174_mogra_5s.wav 2>/dev/null || {
  echo "ffmpeg not available, using sox or python"
  # Alternative: use Python with pydub or just use the existing TXC file as-is
}

# Check if existing TXC is normal mode
version=$(xxd -p -l2 -s4 "test-simples/4.8(wed)MOGRA × #DSPM presents ぷらぷらうんじ@A.txc")
echo "Version bytes: $version"
# If version=0001 → it's normal TXC, skip encoding
```

**Decode MOGRA normal TXC**:
```bash
cmake --build /home/miao/Projects/tsac-ng/build

# Determine the TXC file to use
TXC_FILE="test-simples/4.8(wed)MOGRA × #DSPM presents ぷらぷらうんじ@A.txc"

LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build \
    /home/miao/Projects/tsac-ng/build/tsac-ng \
    -v d "$TXC_FILE" /tmp/r174_mogra_ours.wav 2>&1

# If original tsac available, decode for reference
if command -v tsac &>/dev/null; then
    tsac -v d "$TXC_FILE" /tmp/r174_mogra_orig.wav 2>&1
fi
```

**Compare**:
```bash
python3 -c "
import numpy as np

def read_wav(path):
    try:
        with open(path, 'rb') as f:
            f.read(44)
            data = f.read()
        return np.frombuffer(data, dtype=np.float32)
    except: return None

ours = read_wav('/tmp/r174_mogra_ours.wav')
orig = read_wav('/tmp/r174_mogra_orig.wav')

if ours is not None:
    rms = np.sqrt(np.mean(ours**2))
    print(f'MOGRA ours: {len(ours)} samples, RMS={rms:.6f}, max={np.max(np.abs(ours)):.6f}')
    print(f'  zero samples: {np.sum(ours==0)}/{len(ours)}')

if ours is not None and orig is not None and len(ours) == len(orig):
    corr = np.corrcoef(ours, orig)[0,1]
    mae = np.mean(np.abs(ours - orig))
    print(f'MOGRA vs original: corr={corr:.6f}, MAE={mae:.6f}')
elif orig is not None:
    print(f'MOGRA original: {len(orig)} samples')
    print(f'  length diff: {len(ours)-len(orig) if ours is not None else \"N/A\"}')
"
```

**Acceptance**: MOGRA 5s normal TXC decoded. Output WAV created. Metrics recorded.

---

### T2: Create music_1s normal TXC test and decode (⬜)

**Create test file if needed**:
```bash
# Extract 1s from the P丸様 MP3
ffmpeg -y -i "test-simples/P丸様。-自分後回し.mp3" \
  -t 1 -ar 44100 -ac 2 /tmp/r174_music_1s.wav 2>/dev/null && {
    echo "Created /tmp/r174_music_1s.wav"
    tsac c /tmp/r174_music_1s.wav /tmp/r174_music_1s_normal.txc 2>&1
} || {
    echo "ffmpeg not available, using existing TXC file"
    # Check if P丸様 TXC is normal mode
    version=$(xxd -p -l2 -s4 test-simples/P丸様。-自分後回し@A.txc)
    echo "P丸様 TXC version: $version"
}
```

**Decode**:
```bash
TXC_FILE="test-simples/P丸様。-自分後回し@A.txc"
# Check if normal mode
version=$(xxd -p -l2 -s4 "$TXC_FILE")
if [ "$version" = "0001" ]; then
    LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build \
        /home/miao/Projects/tsac-ng/build/tsac-ng \
        -v d "$TXC_FILE" /tmp/r174_music_ours.wav 2>&1
    
    if command -v tsac &>/dev/null; then
        tsac -v d "$TXC_FILE" /tmp/r174_music_orig.wav 2>&1
    fi
fi
```

**Compare**:
```bash
python3 -c "
import numpy as np

def read_wav(path):
    try:
        with open(path, 'rb') as f:
            f.read(44); data = f.read()
        return np.frombuffer(data, dtype=np.float32)
    except: return None

ours = read_wav('/tmp/r174_music_ours.wav')
orig = read_wav('/tmp/r174_music_orig.wav')

if ours is not None:
    print(f'Music ours: {len(ours)} samples, RMS={np.sqrt(np.mean(ours**2)):.6f}')
if ours is not None and orig is not None and len(ours)==len(orig):
    print(f'Music corr: {np.corrcoef(ours,orig)[0,1]:.6f}')
"
```

**Acceptance**: music_1s normal TXC decoded. Metrics recorded.

---

### T3: Create silent_5s normal TXC test and decode (⬜)

**Create silent test file**:
```bash
python3 -c "
import struct
sr=44100; dur=5; n=int(sr*dur)
with open('/tmp/r174_silent_5s.wav','wb') as f:
    f.write(b'RIFF'); f.write(struct.pack('<I',36+n*2))
    f.write(b'WAVEfmt '); f.write(struct.pack('<I',16))
    f.write(struct.pack('<H',1)); f.write(struct.pack('<H',2))
    f.write(struct.pack('<I',sr)); f.write(struct.pack('<I',sr*4))
    f.write(struct.pack('<H',4)); f.write(struct.pack('<H',16))
    f.write(b'data'); f.write(struct.pack('<I',n*2))
    for i in range(n):
        f.write(struct.pack('<h',0))  # left
        f.write(struct.pack('<h',0))  # right (silence)
print(f'Created /tmp/r174_silent_5s.wav')
"

# Encode with original tsac if available
if command -v tsac &>/dev/null; then
    tsac c /tmp/r174_silent_5s.wav /tmp/r174_silent_5s_normal.txc 2>&1
    echo "Encoded silent_5s normal TXC"
fi
```

**Decode**:
```bash
if [ -f /tmp/r174_silent_5s_normal.txc ]; then
    LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build \
        /home/miao/Projects/tsac-ng/build/tsac-ng \
        -v d /tmp/r174_silent_5s_normal.txc /tmp/r174_silent_ours.wav 2>&1
    
    if command -v tsac &>/dev/null; then
        tsac -v d /tmp/r174_silent_5s_normal.txc /tmp/r174_silent_orig.wav 2>&1
    fi
fi
```

**Compare**:
```bash
python3 -c "
import numpy as np

def read_wav(path):
    try:
        with open(path,'rb') as f:
            f.read(44); data=f.read()
        return np.frombuffer(data, dtype=np.float32)
    except: return None

ours = read_wav('/tmp/r174_silent_ours.wav')
orig = read_wav('/tmp/r174_silent_orig.wav')
if ours is not None:
    nz = np.count_nonzero(ours)
    rms = np.sqrt(np.mean(ours**2))
    print(f'Silence ours: {len(ours)} samples, RMS={rms:.6f}, non-zero={nz}')
if ours is not None and orig is not None:
    if len(ours) == len(orig):
        print(f'Silence corr: {np.corrcoef(ours,orig)[0,1]:.6f}')
    else:
        print(f'Silence length mismatch: {len(ours)} vs {len(orig)}')
"
```

**Acceptance**: silent_5s normal TXC decoded. Output should be approximately silence (low RMS).

---

### T4: Compile multi-file results table and update status doc (⬜)

**Create results script `/home/miao/Projects/tsac-ng/experimental/tests/compile_results_174.py`**:

```python
#!/usr/bin/env python3
"""Compile multi-file normal TXC decode results into a table."""

import numpy as np
import json, os, sys, subprocess, glob

RESULTS_FILE = "/home/miao/Projects/tsac-ng/docs/NORMAL_TXC_RESULTS_R174.json"
MARKDOWN_FILE = "/home/miao/Projects/tsac-ng/docs/NORMAL_TXC_RESULTS_R174.md"

def read_wav(path):
    if not os.path.exists(path):
        return None, None
    with open(path, 'rb') as f:
        raw = f.read()
    if len(raw) < 44:
        return None, None
    # Read WAV header to get channels and data
    channels = int.from_bytes(raw[22:24], 'little')
    data = raw[44:]
    samples = np.frombuffer(data, dtype=np.float32)
    # If stereo, average to mono for comparison
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return samples, channels


def compute_metrics(ours, orig):
    """Compute comparison metrics between our and original WAV."""
    metrics = {}
    
    # Basic stats
    metrics["our_rms"] = float(np.sqrt(np.mean(ours**2)))
    metrics["our_max"] = float(np.max(np.abs(ours)))
    metrics["our_nonzero"] = int(np.count_nonzero(ours))
    metrics["our_length"] = len(ours)
    
    if orig is not None:
        metrics["orig_rms"] = float(np.sqrt(np.mean(orig**2)))
        metrics["orig_length"] = len(orig)
        
        # Align lengths
        min_len = min(len(ours), len(orig))
        ours_a = ours[:min_len]
        orig_a = orig[:min_len]
        
        # Correlation
        if np.std(ours_a) > 0 and np.std(orig_a) > 0:
            metrics["correlation"] = float(np.corrcoef(ours_a, orig_a)[0, 1])
        else:
            metrics["correlation"] = 0.0
        
        # Error metrics
        diff = ours_a - orig_a
        metrics["mae"] = float(np.mean(np.abs(diff)))
        metrics["rmse"] = float(np.sqrt(np.mean(diff**2)))
        
        # Signal-to-noise ratio
        signal_power = np.mean(orig_a**2)
        noise_power = np.mean(diff**2)
        if noise_power > 1e-10 and signal_power > 1e-10:
            metrics["snr_db"] = float(10 * np.log10(signal_power / noise_power))
        else:
            metrics["snr_db"] = float('inf') if noise_power < 1e-10 else -float('inf')
    else:
        metrics["orig_available"] = False
    
    return metrics


def main():
    # Define test cases
    test_cases = [
        {
            "name": "MOGRA_5s",
            "desc": "DJ set excerpt, 5s, q6 stereo",
            "our_wav": "/tmp/r174_mogra_ours.wav",
            "orig_wav": "/tmp/r174_mogra_orig.wav",
        },
        {
            "name": "music_1s",
            "desc": "Music clip, 1s, q6 stereo",
            "our_wav": "/tmp/r174_music_ours.wav",
            "orig_wav": "/tmp/r174_music_orig.wav",
        },
        {
            "name": "silent_5s",
            "desc": "Silence, 5s, q6 stereo",
            "our_wav": "/tmp/r174_silent_ours.wav",
            "orig_wav": "/tmp/r174_silent_orig.wav",
        },
    ]
    
    all_results = {}
    
    for tc in test_cases:
        print(f"\n=== {tc['name']}: {tc['desc']} ===")
        
        ours_samples, our_ch = read_wav(tc["our_wav"])
        orig_samples, orig_ch = read_wav(tc["orig_wav"])
        
        if ours_samples is None:
            print(f"  WARNING: Our output not found at {tc['our_wav']}")
            all_results[tc["name"]] = {"status": "NO_OUTPUT"}
            continue
        
        metrics = compute_metrics(ours_samples, orig_samples)
        all_results[tc["name"]] = {
            "status": "OK",
            "metrics": metrics,
            "our_path": tc["our_wav"],
            "orig_path": tc["orig_wav"],
        }
        
        # Print summary
        print(f"  Our RMS:    {metrics['our_rms']:.6f}")
        print(f"  Our length: {metrics['our_length']} samples")
        if "correlation" in metrics:
            print(f"  Correlation: {metrics['correlation']:.6f}")
        if "rmse" in metrics:
            print(f"  RMSE:        {metrics['rmse']:.6f}")
        if "snr_db" in metrics:
            print(f"  SNR:         {metrics['snr_db']:.1f} dB")
    
    # Save JSON
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    
    # Generate markdown table
    with open(MARKDOWN_FILE, 'w') as f:
        f.write("# Normal TXC Multi-File Results (R174)\n\n")
        f.write(f"**Date**: 2026-05-29\n\n")
        f.write("| File | Frames | Codebooks | Status | Our RMS | Correlation | RMSE | SNR(dB) |\n")
        f.write("|------|--------|-----------|--------|---------|-------------|------|--------|\n")
        for name, result in all_results.items():
            if result["status"] == "OK":
                m = result["metrics"]
                corr = f"{m.get('correlation', 'N/A'):.4f}" if 'correlation' in m else "N/A"
                rmse = f"{m.get('rmse', 'N/A'):.6f}" if 'rmse' in m else "N/A"
                snr = f"{m.get('snr_db', 'N/A'):.1f}" if 'snr_db' in m else "N/A"
                f.write(f"| {name} | {m['our_length']//1024}k | 8 | OK | {m['our_rms']:.4f} | {corr} | {rmse} | {snr} |\n")
            else:
                f.write(f"| {name} | - | - | FAIL | - | - | - | - |\n")
        
        f.write("\n## Notes\n\n")
        f.write("- RMS: Root mean square amplitude of decoded audio\n")
        f.write("- Correlation: Pearson correlation between our WAV and original tsac WAV\n")
        f.write("- RMSE: Root mean square error between our WAV and original\n")
        f.write("- SNR: Signal-to-noise ratio in dB\n")
        f.write("\n## Known Issues\n\n")
        f.write("- [ ] (fill from results)\n")
    
    print(f"Markdown table saved to {MARKDOWN_FILE}")

if __name__ == "__main__":
    main()
```

**Run results compilation**:
```bash
python3 /home/miao/Projects/tsac-ng/experimental/tests/compile_results_174.py
```

**Update `/home/miao/Projects/tsac-ng/docs/NORMAL_TXC_STATUS.md`**:
```markdown
# Normal TXC Decode Status

## Current Status (R174)
- Pipeline: wired and compiles
- Multi-file testing: complete (see results below)

## Results Table
(import from docs/NORMAL_TXC_RESULTS_R174.md)

## Remaining Issues
1. ...
2. ...
```

**Acceptance**: Results JSON and markdown table created. Status doc updated with multi-file results.

---

## Acceptance Criteria

- [ ] **T1**: MOGRA 5s normal TXC decoded. Output WAV exists at `/tmp/r174_mogra_ours.wav`. Metrics recorded.
- [ ] **T2**: music_1s normal TXC decoded (from P丸様 or generated). Metrics recorded.
- [ ] **T3**: Silent 5s normal TXC decoded. Output is approximately silent (RMS < 0.01).
- [ ] **T4**: Results table at `docs/NORMAL_TXC_RESULTS_R174.md` with structured metrics for all 3 files.
- [ ] Results JSON valid at `docs/NORMAL_TXC_RESULTS_R174.json`.
- [ ] Status doc `docs/NORMAL_TXC_STATUS.md` updated.
- [ ] Fast TXC decode NOT broken.

## Decision Gate

| Condition | Action |
|-----------|--------|
| All 3 files decode successfully | → R175 (encoder round-trip) |
| 2/3 decode, 1 fails | Document failure, → R175 |
| All 3 fail | → R171 (fundamental pipeline issue) |
