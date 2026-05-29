# Round 171 — Normal TXC End-to-End Integration (Phase 4D)

**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Strategy

Wire the full normal TXC decode pipeline: TXC file → Range Coder → Transformer → Softmax → Codebook indices → DAC Decoder → PCM. This is the first time we attempt to produce actual audio from a normal TXC file. The key integration point is feeding Transformer-predicted probability distributions into the range coder's cumulative frequency decode, then routing the decoded codebook indices into the existing DAC decoder graph.

**Prerequisites**: `build/tsac-ng` binary, `tsac_stereo_q8.bin` (or equivalent with transformer weights), normal TXC test files.

**Key files** (all critical — read and understand before starting):
- `src/tsac_normal_decode.c` — Normal TXC decode pipeline (needs integration wiring)
- `src/tsac_transformer.c` — Transformer forward pass (12L GPT-2)
- `src/tsac_transformer.h` — Transformer API (TF_D_MODEL, TF_VOCAB, tsac_transformer_forward)
- `src/range_coder.c` — `rc_decode_cumul()` function
- `src/range_coder.h` — Range coder API
- `src/tsac_codec.c` — Codec dispatch logic (needs normal TXC path)
- `src/txc_format.c` — TXC format parser (normal vs fast detection)
- `src/cpu_decoder.c` — DAC decoder (codebook indices → PCM)

## Tasks

### T1: Wire Transformer output → range decoder → codebook indices

The current `tsac_normal_decode.c` decodes one frame but the Transformer integration is incomplete:
- Line 60: `tsac_transformer_forward()` is called with `NULL` input_ids
- Line 81: The decoded index is NOT fed back as the next Transformer input

```bash
# Read the current pipeline
cat src/tsac_normal_decode.c

# Read the transformer API
cat src/tsac_transformer.h
```

Required integration:
1. Transformer takes `input_ids` (previously decoded indices) and `position_ids` 
2. Outputs logits for next frame's codebook probabilities
3. Softmax converts logits → probabilities
4. `build_cum_freq()` converts probs → cumulative frequency table
5. `rc_decode_cumul()` decodes one codebook index from the range coder
6. Decoded index is appended to `input_ids` for autoregressive next step

**Acceptance**: `decode_one_frame()` correctly calls Transformer, decodes one index via range coder, and returns the decoded index. Unit test passes.

### T2: Wire codebook indices → DAC decoder → PCM

The second integration step takes the decoded codebook indices (n_frames × n_cb) and routes them through the existing DAC decoder to produce PCM audio:

```bash
# The existing DAC decoder takes codebook_indices → PCM
# This path already works for fast TXC mode
# Need to ensure it works for normal TXC indices too

# grep for how indices flow to DAC decoder
grep -n "codebook_indices\|decode_batch\|cpu_decoder_run" src/tsac_codec.c
```

Integration approach (modify `src/tsac_codec.c`):
1. Detect normal TXC format (version >= 1 + flags & 0x80)
2. Run `tsac_normal_decode()` to get codebook indices
3. Feed indices into existing DAC decoder (same path as fast TXC)
4. Output PCM

**Acceptance**: Normal TXC decode path produces codebook indices that can be fed to DAC decoder without format errors.

### T3: Decode silent_1s_normal_q6.txc — first normal TXC audio

```bash
mkdir -p /tmp/r171

# Check if silent_1s_normal_q6.txc exists in test-simples
ls -la test-simples/ | grep -i silent
ls -la test-simples/ | grep -i normal

# If not found, create one with original tsac:
# tsac c reference_silent.wav silent_1s_normal_q6.txc -q 6

# Attempt decode with our codec
./build/tsac-ng -v d test-simples/silent_1s_normal_q6.txc /tmp/r171/silent_normal_output.wav 2>&1 | tee /tmp/r171/normal_decode.log

# Check the log for errors
grep -i "error\|failed\|not implemented\|corrupt" /tmp/r171/normal_decode.log || echo "No obvious errors"
```

**Acceptance**: Normal TXC decode attempt either:
- ✅ Produces a WAV (even if quality is poor)
- ❌ Fails gracefully with specific error message (not a crash)

### T4: Compare with original tsac normal TXC output

```bash
# Original tsac decode for comparison
# tsac d test-simples/silent_1s_normal_q6.txc /tmp/r171/silent_normal_orig.wav

# Compare
python3 << 'EOF'
import numpy as np, wave, os

def read_wav(path):
    if not os.path.exists(path): return None, 0
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return data.reshape(-1, w.getnchannels()), w.getnframes()

our, n_our = read_wav('/tmp/r171/silent_normal_output.wav')
ref, n_ref = read_wav('/tmp/r171/silent_normal_orig.wav')

if our is None:
    print('Our decode failed — see normal_decode.log')
elif ref is None:
    print(f'Our decode produced WAV: shape={our.shape}, RMS={np.sqrt(np.mean(our**2)):.6f}')
    print('Reference not available — log our output stats only')
else:
    n = min(n_our, n_ref)
    corr = np.corrcoef(our[:n].flatten(), ref[:n].flatten())[0,1]
    print(f'Our: {our.shape} RMS={np.sqrt(np.mean(our**2)):.6f}')
    print(f'Ref: {ref.shape} RMS={np.sqrt(np.mean(ref**2)):.6f}')
    print(f'Correlation: {corr:.6f}')
EOF
```

**Acceptance**: Comparison with original tsac logged. Any correlation > 0 is progress.

### T5: Document normal TXC decode status

```bash
python3 << 'EOF'
import os

log_path = '/tmp/r171/normal_decode.log'
wav_path = '/tmp/r171/silent_normal_output.wav'

status = {
    'normal_txc_decode_attempted': True,
    'binary_exists': os.path.exists('build/tsac-ng'),
    'log_exists': os.path.exists(log_path),
    'wav_produced': os.path.exists(wav_path),
    'errors': []
}

if os.path.exists(log_path):
    with open(log_path) as f:
        content = f.read()
        if 'error' in content.lower() or 'failed' in content.lower() or 'not implemented' in content.lower():
            status['errors'] = [line for line in content.split('\n') if any(e in line.lower() for e in ['error', 'fail', 'not implement'])]

print('=' * 60)
print('R171: Normal TXC Decode Status')
print('=' * 60)
print(f'  Attempted: {status["normal_txc_decode_attempted"]}')
print(f'  Binary:    {"✅" if status["binary_exists"] else "❌"}')
print(f'  WAV out:   {"✅" if status["wav_produced"] else "❌"}')
print(f'  Errors:    {status["errors"] if status["errors"] else "None"}')

# Save
import json
with open('/tmp/r171/normal_decode_status.json', 'w') as f:
    json.dump(status, f, indent=2)
print(f'\nStatus saved to /tmp/r171/normal_decode_status.json')
EOF
```

**Acceptance**: Status document created at `/tmp/r171/normal_decode_status.json`.

## Acceptance Criteria

- **AC1**: Transformer → range decoder → codebook indices integration wired in `tsac_normal_decode.c`
- **AC2**: Codebook indices → DAC decoder → PCM path connected in `tsac_codec.c`
- **AC3**: Normal TXC decode attempt runs without segfault
- **AC4**: Silent normal TXC decode produces WAV (or graceful failure with clear error)
- **AC5**: Comparison with original tsac logged
- **AC6**: Status document saved to `/tmp/r171/normal_decode_status.json`

## Expected Outcome

Most likely:
- The integration produces codebook indices and attempts DAC decode
- Audio may be noise or silence due to residual issues in the Transformer or range coder integration
- First normal TXC audio — even if imperfect — validates the pipeline wiring
- Errors will point to specific integration gaps (autoregressive feedback, softmax scaling, cumulative frequency edge cases)

## Integration Architecture

```
Normal TXC File
      │
      ▼
  txc_format.c ─── parse header (magic, version, n_blocks, CRC32)
      │
      ▼
  tsac_codec.c ─── detect normal mode, dispatch to normal path
      │
      ▼
  tsac_normal_decode.c ─── main decode loop
      │
      ├── RangeCoder init (rc_decoder_init)
      │
      ├── For each frame:
      │   ├── tsac_transformer_forward() → logits [d_model]
      │   ├── softmax() → probabilities [vocab=1024]
      │   ├── build_cum_freq() → cumulative freq table
      │   ├── rc_decode_cumul() → codebook index (0-1023)
      │   └── Append index to input_ids (autoregressive feedback)
      │
      ▼
  Codebook indices [n_frames × n_cb]
      │
      ▼
  cpu_decoder.c ─── DAC decoder (same as fast TXC path)
      │
      ▼
  PCM audio → WAV file
```
