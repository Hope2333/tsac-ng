# Round 171 — Normal TXC End-to-End Integration

**Status**: PENDING (Header Planned) | **Date**: 2026-05-29
**Predecessor**: round-170
**Priority**: HIGH — First normal TXC audio output

## Strategy — WHY this round exists

Currently, `tsac_normal_decode.c` contains the structural skeleton but has never been wired into the codec pipeline. The `decode_one_frame()` function calls `tsac_transformer_forward()` but the logits-to-probability pipeline has a critical error: it uses `pos_logits = logits` (the first element) instead of projecting through the `g` weight matrix. Moreover, the Transformer forward pass output shape is `[seq_len, d_model]` and after `g` projection it produces a scalar logit — not 1024 logits. The codebook index vocabulary is 0-1023, but the current softmax operates on `d_model=512` values, not on 1024 logits.

The normal TXC decode pipeline must be:
1. **txc_read** → detect normal TXC (version>=1, flags&0x80) → extract range-coded payload
2. **RangeCoder init** from payload (skip CRC32 last 4 bytes)
3. **For each frame f=0..n_frames-1**:
   a. Run Transformer forward with `position_ids=[f]` and previous indices as `input_ids`
   b. For each codebook cb=0..n_cb-1:
      - Get logits for codebook cb from Transformer output
      - Softmax → cumulative frequency table (1024 symbols)
      - `rc_decode_cumul` → decode one codebook index
      - Feed index back as input for next codebook
4. **DAC decode** codebook indices → PCM → WAV

This round wires all of these together end-to-end for the first time.

## Key files
- `/home/miao/Projects/tsac-ng/src/tsac_normal_decode.c` — rewrite core logic
- `/home/miao/Projects/tsac-ng/src/tsac_transformer.c` — fix g projection for 1024 logits
- `/home/miao/Projects/tsac-ng/src/tsac_transformer.h` — may need API change
- `/home/miao/Projects/tsac-ng/src/txc_format.c` — normal TXC payload extraction path
- `/home/miao/Projects/tsac-ng/src/tsac_codec.c` — wire tsac_normal_decode into decompress
- `/home/miao/Projects/tsac-ng/src/range_coder.c` — verify rc_decode_cumul works for 1024 syms
- `/home/miao/Projects/tsac-ng/include/tsac.h` — may need public API if normal decode exposed
- `/home/miao/Projects/tsac-ng/models/tsac/tsac_stereo_q8.bin` — transformer weights source

## Dependencies
- Working DAC decoder (fast TXC decode already works)
- Working Transformer forward pass (test_transformer.c compiled and runnable)
- Working range coder with cumul decode (R120 verified)

## Tasks

### T1: Fix Transformer output projection for 1024-way codebook classification (⬜)

**Problem**: `tsac_transformer_forward()` at line 276-283 of `/home/miao/Projects/tsac-ng/src/tsac_transformer.c` computes:
```c
if (tf->g) {
    for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int d = 0; d < D; d++)
            dot += logits[s * D + d] * tf->g[d];
        logits[s * D] = dot;  /* scalar output (next token logit) */
    }
}
```

This produces ONE scalar per position. But for normal TXC codebook decoding we need 1024 logits (vocabulary size). The `g` weight tensor in the model is shape `[512]` — a single vector. This suggests `g` is used differently.

**Root cause analysis**: In the original tsac, the Transformer predicts probabilities for each codebook index 0-1023. With d_model=512, the output per frame is `[512]`. The original must project to 1024 somehow. Either:
- (a) `g` is not `[512]→1` but `[512]→1024` — check tensor shape in model
- (b) The codebook index is decoded from d_model logits using a different projection
- (c) Softmax over 512 dims maps to 1024 via some embedding

**Action**:
1. Inspect the `g` tensor from `tsac_stereo_q8.bin`:
   ```bash
   python3 -c "
   import struct, sys
   with open('/home/miao/Projects/tsac-ng/models/tsac/tsac_stereo_q8.bin','rb') as f:
       data = f.read()
   # Search for 'g' tensor close to name markers
   idx = data.find(b'\x00g\x00')
   if idx >= 0:
       print('g found at offset', idx)
       # Read preceding tensor header
       hdr_start = max(0, idx - 40)
       print('Context bytes:', data[hdr_start:idx+10].hex())
   "
   ```
   Or use the model loader API to inspect:
   ```bash
   cat > /tmp/inspect_g.c << 'EOF'
   #include "dac_model.h"
   #include "model_loader.h"
   #include <stdio.h>
   int main() {
       DACModel *m = dac_model_create();
       model_loader_load("/home/miao/Projects/tsac-ng/models/tsac/tsac_stereo_q8.bin", m);
       DACTensor *g = dac_model_find(m, "g");
       if (g) {
           printf("g tensor: ndims=%d, dims=[", g->ndims);
           for (int i = 0; i < g->ndims; i++) printf("%d%c", g->dims[i], i+1<g->ndims?',':'');
           printf("], data_size=%d, elem_size=%d\n", g->data_size, g->elem_size);
       }
       model_loader_free(m);
       free(m);
       return 0;
   }
   EOF
   gcc -o /tmp/inspect_g /tmp/inspect_g.c \
       -I/home/miao/Projects/tsac-ng/include -I/home/miao/Projects/tsac-ng/src \
       -L/home/miao/Projects/tsac-ng/build -ltsac-ng -lm
   LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build /tmp/inspect_g
   ```

2. Based on findings, fix `tsac_transformer_forward()`:
   - If `g` is `[512, 1024]`: change dot product to matmul `[1, 512] @ [512, 1024] → [1, 1024]`
   - If `g` is `[512]` with unknown 1024 mapping: search for codebook projection layers

3. **Fallback**: If g is [512]→scalar, the 1024 logits may come from a separate per-codebook projection. Search for tensors matching patterns like `h*/codebook*`, `cb*`, `proj*`, `classifier*` in the model.

4. Update `tsac_transformer.h` if API changes.

**Verification**:
```bash
cmake --build /home/miao/Projects/tsac-ng/build
# Run test harness that checks output shape
cat > /tmp/test_tf_logits.c << 'EOF'
#include "tsac_transformer.h"
#include "dac_model.h"
#include "model_loader.h"
#include <stdio.h>
#include <math.h>
int main() {
    DACModel *m = dac_model_create();
    model_loader_load("/home/miao/Projects/tsac-ng/models/tsac/tsac_stereo_q8.bin", m);
    TSACTransformer tf;
    tsac_transformer_load(&tf, m->tensors, m->n_tensors);
    int pos[1] = {0};
    float logits[TF_MAX_SEQ * TF_D_MODEL] = {0};
    int ret = tsac_transformer_forward(&tf, NULL, pos, 1, logits);
    if (ret == 0) {
        printf("First 8 values: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
               logits[0], logits[1], logits[2], logits[3],
               logits[4], logits[5], logits[6], logits[7]);
        float sum_exp = 0;
        for (int i = 0; i < 1024 && i < TF_D_MODEL; i++)
            sum_exp += expf(logits[i]);
        printf("exp sum over first min(1024, d_model): %.6f\n", sum_exp);
    }
    tsac_transformer_free(&tf);
    dac_model_destroy(m);
    return ret;
}
EOF
gcc -o /tmp/test_tf_logits /tmp/test_tf_logits.c \
    -I/home/miao/Projects/tsac-ng/include -I/home/miao/Projects/tsac-ng/src \
    -L/home/miao/Projects/tsac-ng/build -ltsac-ng -lm -lpthread
LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build /tmp/test_tf_logits
```

**Acceptance**: Transformer produces output interpretable as 1024-way logits (exp sum over first 1024 elements > 1.0).

---

### T2: Rewrite `tsac_normal_decode.c` with correct decode loop (⬜)

Current `decode_one_frame()` has these bugs:
- Uses `pos_logits = logits` (line 64) — should use projected logits for the codebook
- Transformer called once per codebook but should be called once per frame (autoregressive across codebooks)
- `input_ids` is NULL — needs previous decoded indices as input tokens
- No codebook-specific logit selection/projection

**New implementation plan**:

Create a new function `tsac_normal_decode_indices()` that:
1. Initializes RangeCoder from compressed payload (offset past 16-byte header, excluding trailing 4-byte CRC)
2. For each frame f=0..n_frames-1:
   a. Build `input_ids` array from previously decoded indices of PREVIOUS frame (auto-regressive across frames)
   b. Run `tsac_transformer_forward()` once per frame → get hidden states
   c. For each codebook cb=0..n_cb-1:
      - Compute logits for this codebook (project hidden through codebook-specific weights if they exist, or through `g`)
      - Softmax over 1024 symbols
      - Build cumulative frequency table
      - Range decode one symbol
      - Store in all_indices[f * n_cb + cb]

**Key model tensor search**: Before coding, identify how the original tsac maps Transformer hidden states to 1024 codebook logits. Search for these tensor name patterns in the .bin model:
```bash
python3 -c "
import sys
with open('/home/miao/Projects/tsac-ng/models/tsac/tsac_stereo_q8.bin', 'rb') as f:
    data = f.read()
# Search for all tensor names containing 'codebook', 'proj', 'cb', 'class', 'head'
import re
for pattern in [b'codebook', b'proj', b'cb[0-9]', b'class', b'head']:
    for m in re.finditer(pattern, data):
        start = max(0, m.start()-20)
        end = min(len(data), m.end()+20)
        print(f'Found {pattern.decode()} at {m.start()}: {data[start:end]}')
"
```

**Implementation files**:
- `/home/miao/Projects/tsac-ng/src/tsac_normal_decode.c` — full rewrite

**Verification**:
```bash
cmake --build /home/miao/Projects/tsac-ng/build
# Use test harness from T3 to call tsac_normal_decode
```

**Acceptance**: `tsac_normal_decode` returns `TSAC_OK` with all indices in [0, 1023].

---

### T3: Wire normal TXC path into `tsac_codec.c` (⬜)

**Modify `/home/miao/Projects/tsac-ng/src/tsac_codec.c`**:

1. Add `#include "tsac_transformer.h"` at top
2. Add `TSACTransformer normal_tf;` and `int normal_tf_loaded;` to `struct TSACContext`
3. In `tsac_init()`: after loading DAC model, call `tsac_transformer_load()` to init transformer
4. In `tsac_decompress()` or `tsac_decompress_file()`:
   ```c
   if (hdr.version >= 1 && (hdr.flags & 0x80U)) {
       // Normal TXC path
       // payload = data + 16, payload_len = txc_size - 16 - 4
       int n_frames = hdr.n_blocks;
       int n_cb = hdr.n_codebooks;
       int *codebook_indices = NULL;
       int out_frames = 0;
       ret = tsac_normal_decode(data + 16, txc_size - 16 - 4,
                                 &ctx->normal_tf,
                                 n_frames, n_cb,
                                 &codebook_indices, &out_frames);
       if (ret == TSAC_OK) {
           ret = dac_model_decode(ctx->model, codebook_indices, out_frames,
                                   n_cb, hdr.block_len, channels,
                                   pcm, n_samples, ctx->n_threads);
           free(codebook_indices);
       }
   }
   ```

5. In `tsac_free()`: call `tsac_transformer_free(&ctx->normal_tf)`

**Verification**:
```bash
cmake --build /home/miao/Projects/tsac-ng/build
LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build \
    /home/miao/Projects/tsac-ng/build/tsac-ng \
    -v d "test-simples/4.8(wed)MOGRA × #DSPM presents ぷらぷらうんじ@A.txc" \
    /tmp/171_output.wav 2>&1
```

**Acceptance**: CLI command produces WAV output without crash. WAV file size > 44 bytes.

---

### T4: Test with silent WAV → normal TXC → decode round-trip (⬜)

**Create test file and decode**:
```bash
# Generate 1s silent WAV
python3 -c "
import struct, math
sr = 44100; dur = 1.0; n = int(sr * dur)
samples = [0.0 for _ in range(n)]  # silence
with open('/tmp/silent_1s.wav', 'wb') as f:
    f.write(b'RIFF'); f.write(struct.pack('<I', 36 + n*2))
    f.write(b'WAVEfmt '); f.write(struct.pack('<I', 16))
    f.write(struct.pack('<H', 1))
    f.write(struct.pack('<H', 1))  # mono
    f.write(struct.pack('<I', sr))
    f.write(struct.pack('<I', sr*2))
    f.write(struct.pack('<H', 2))
    f.write(struct.pack('<H', 16))
    f.write(b'data'); f.write(struct.pack('<I', n*2))
    for s in samples:
        v = 0
        f.write(struct.pack('<h', v))
print(f'Created /tmp/silent_1s.wav ({n} samples)')
"

# Encode with original tsac normal mode (if available)
if command -v tsac &> /dev/null; then
    tsac c /tmp/silent_1s.wav /tmp/silent_1s_normal.txc
    echo "Encoded normal TXC"
fi
```

If no original tsac is available, use the existing test file from test-simples.

**Decode test**:
```bash
LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build \
    /home/miao/Projects/tsac-ng/build/tsac-ng \
    -v d /tmp/silent_1s_normal.txc /tmp/171_silent_out.wav 2>&1

# Check output
python3 -c "
import numpy as np, struct
with open('/tmp/171_silent_out.wav','rb') as f:
    hdr = f.read(44)
    data = f.read()
samples = np.frombuffer(data, dtype=np.float32)
print(f'Output: {len(samples)} samples, RMS={np.sqrt(np.mean(samples**2)):.6f}')
print(f'Max={np.max(np.abs(samples)):.6f}, Min={np.min(samples):.6f}')
print(f'Non-zero samples: {np.count_nonzero(samples)}')
"
```

**Acceptance**: At least one normal TXC file decodes successfully. Output WAV has non-zero PCM samples.

---

### T5: Document status and create regression test script (⬜)

**Create `/home/miao/Projects/tsac-ng/experimental/tests/test_normal_txc.sh`**:
```bash
#!/bin/bash
# Normal TXC decode regression test — R171
# Usage: ./experimental/tests/test_normal_txc.sh

set -e
BUILD="/home/miao/Projects/tsac-ng/build"
BINARY="$BUILD/tsac-ng"
MODEL="/home/miao/Projects/tsac-ng/models/tsac/tsac_stereo_q8.bin"
TESTDIR="${1:-/home/miao/Projects/tsac-ng/test-simples}"

echo "=== Normal TXC Regression Test (R171) ==="
echo "Binary: $BINARY"
echo "Model:  $MODEL"
echo ""

PASS=0
FAIL=0
for txc in "$TESTDIR"/*.txc; do
    version=$(xxd -p -l2 -s4 "$txc" 2>/dev/null || echo "00")
    if [ "$version" = "0001" ]; then
        echo "--- Testing: $(basename "$txc") ---"
        out="/tmp/r171_$(basename "$txc" .txc).wav"
        LD_LIBRARY_PATH="$BUILD" "$BINARY" -v d "$txc" "$out" 2>&1
        if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 44 ]; then
            echo "PASS: $(basename "$txc") → $out ($(stat -c%s "$out") bytes)"
            PASS=$((PASS+1))
        else
            echo "FAIL: $(basename "$txc")"
            FAIL=$((FAIL+1))
        fi
        echo ""
    fi
done

echo "=== Results: $PASS passed, $FAIL failed ==="
exit $FAIL
```

**Create `/home/miao/Projects/tsac-ng/docs/NORMAL_TXC_STATUS.md`**:
```markdown
# Normal TXC Decode Status (R171)

**Date**: 2026-05-29
**Status**: First integration complete

## Files Tested
| File | Frames | Codebooks | Decode OK | Output RMS | Notes |
|------|--------|-----------|-----------|------------|-------|
| ...  | ...    | ...       | ...       | ...        | ...   |

## Known Issues
- [ ] Transformer g projection: [describe findings]
- [ ] Per-codebook logit projection: [identified or not]
- [ ] Index accuracy vs original: (R172)

## Next
Proceed to R172: Transformer output validation.
```

**Acceptance**: Regression script committed. Status document created with initial results table.

---

## Acceptance Criteria

- [ ] **T1**: Transformer output produces interpretable logits (fixed g projection or alternative). Verified by test harness.
- [ ] **T2**: `tsac_normal_decode.c` rewritten with correct autoregressive loop. All returned indices in [0, 1023]. Returns TSAC_OK.
- [ ] **T3**: Normal TXC path wired into `tsac_codec.c`. CLI decode command produces WAV file > 44 bytes.
- [ ] **T4**: At least one normal TXC file decodes end-to-end. Output has non-zero RMS.
- [ ] **T5**: Regression script at `experimental/tests/test_normal_txc.sh`. Status doc at `docs/NORMAL_TXC_STATUS.md`.
- [ ] `cmake --build build` succeeds with no new warnings.
- [ ] Fast TXC decode still works (regression check):
  ```bash
  LD_LIBRARY_PATH=build build/tsac-ng -v d test-simples/P丸様。-自分後回し@A.txc /tmp/r171_regression.wav 2>&1
  ```

## Decision Gate

| Condition | Action |
|-----------|--------|
| Decode succeeds, RMS > 0.001 | → R172 (transformer validation) |
| Decode fails | Fix error, retry R171 |
| Decode succeeds but silence | → R172 (indices wrong but pipeline works) |
