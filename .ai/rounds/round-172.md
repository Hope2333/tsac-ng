# Round 172 — Transformer Output Validation

**Status**: PENDING (Header Planned) | **Date**: 2026-05-29
**Predecessor**: round-171
**Priority**: HIGH — Verify codebook index correctness

## Strategy — WHY this round exists

R171 wires the pipeline end-to-end but the decoded codebook indices may be completely wrong. The Transformer predicts probability distributions over 1024 codebook entries, and the range coder decodes symbols from those distributions. If the Transformer's predictions don't match the original tsac's Transformer, the decoded indices will be garbage — and the audio will be noise.

This round compares our decoded codebook indices against the original tsac's indices for the same normal TXC file. The comparison is done by:
1. Encoding a known WAV file with the ORIGINAL tsac in normal mode (produces a .txc with range-coded payload)
2. Decoding the indices from that .txc BOTH with our pipeline AND with original tsac (via LD_PRELOAD/GDB dump or by decoding the range-coded payload with a reference decoder)
3. Measuring per-index accuracy: what percentage of our decoded indices (0-1023) match the original?

If match rate > 80%, the Transformer is working. If match rate < 20%, something fundamental is wrong (g projection, codebook-specific weights, or range coder state management).

## Key files
- `/home/miao/Projects/tsac-ng/src/tsac_normal_decode.c` — our decoder (from R171)
- `/home/miao/Projects/tsac-ng/src/tsac_transformer.c` — forward pass
- `/home/miao/Projects/tsac-ng/src/range_coder.c` — cumulative decode
- `/home/miao/Projects/tsac-ng/src/txc_format.c` — payload extraction
- `/home/miao/Projects/tsac-ng/experimental/compare_indices.py` — new: comparison script
- `/home/miao/Projects/tsac-ng/experimental/tests/test_normal_txc.sh` — regression test

## Dependencies
- R171 complete (pipeline wired, produces some output)
- Original `tsac` binary available for reference encoding/decoding

## Tasks

### T1: Capture reference codebook indices from original tsac (⬜)

**Approach A — Decode via range coder with original tsac's own Transformer**:
If the original tsac binary is available and can decode normal TXC:
```bash
# Decode with original tsac and capture codebook indices via GDB
# Or use a Python script that opens the same txc and applies the same decode steps
```

**Approach B — Use known test vectors**:
If the original tsac's normal TXC files were created with a fixed seed/deterministic encoder, the indices may be reproducible. Most practical approach:

1. Use the original tsac to decode a normal TXC file and dump internal state:
```bash
# If original tsac has verbose output that shows codebook indices:
tsac -v d input_normal.txc output.wav 2>&1 | head -50
```

2. Or create a Python reference decoder that reads the range-coded payload:
   - Parse the normal TXC header (16 bytes)
   - Extract range-coded payload (bytes 16..end-4)
   - The payload contains arithmetically coded symbols
   - Without the original Transformer's probability model, we cannot decode the arithmetic symbols
   - **Alternative**: If the original tsac stores raw indices somewhere (debug build?), use that

3. **Practical approach**: Use the original tsac binary via LD_PRELOAD to intercept index writes:
```c
// LD_PRELOAD library to hook codebook index writes
// Intercept RVQ codebook lookup or index storage
```

**Most pragmatic approach**: Create a Python harness that:
1. Takes the known bitstream (range-coded payload)
2. Steps through the arithmetic decoder with known probability tables
3. Extracts each symbol

Since we don't have original source, create the comparison at the WAV level instead:
1. Decode with original tsac → reference WAV
2. Decode with our pipeline → our WAV  
3. Compare WAVs for correlation

**For index-level comparison**, add debug output to `tsac_normal_decode.c` that dumps all decoded indices to a binary file:
```c
// Add to decode loop after rc_decode_cumul:
static FILE *dump_fp = NULL;
if (!dump_fp) dump_fp = fopen("/tmp/r172_our_indices.bin", "wb");
fwrite(&sym, sizeof(int), 1, dump_fp);
```

**To capture original tsac indices**, if the original tsac has any debug mode:
```bash
# Search for any debug flags in original tsac
strings $(which tsac) | grep -i debug 2>/dev/null
# Or run with strace to see file operations
strace -e trace=write tsac -v d input.txc /dev/null 2>&1 | grep index
```

**Fallback**: If no original index capture is possible, compare at the WAV level and document the delta.

**Acceptance**: Either (a) reference indices captured to `/tmp/r172_ref_indices.bin`, or (b) methodology for WAV-level comparison documented with clear limitations.

---

### T2: Implement index accuracy measurement script (⬜)

**Create `/home/miao/Projects/tsac-ng/experimental/compare_indices.py`**:

```python
#!/usr/bin/env python3
"""Compare codebook indices from our decoder vs reference (original tsac)."""

import numpy as np
import sys, os, json

def load_indices(path, dtype=np.int32):
    """Load binary array of codebook indices."""
    data = np.fromfile(path, dtype=dtype)
    return data

def compute_accuracy(ours, ref):
    """Compute per-position accuracy metrics."""
    if len(ours) != len(ref):
        # Try to align by truncating
        min_len = min(len(ours), len(ref))
        ours = ours[:min_len]
        ref = ref[:min_len]
        print(f"Warning: length mismatch. Truncated to {min_len}")
    
    exact_match = np.mean(ours == ref)
    
    # Per-codebook accuracy
    n_total = len(ours)
    # Assume n_cb columns
    for n_cb in [6, 8, 12]:
        if n_total % n_cb == 0:
            n_frames = n_total // n_cb
            ours_2d = ours.reshape(n_frames, n_cb)
            ref_2d = ref.reshape(n_frames, n_cb)
            per_cb = np.mean(ours_2d == ref_2d, axis=0)
            print(f"Per-codebook accuracy (n_cb={n_cb}):")
            for cb in range(n_cb):
                print(f"  CB {cb}: {per_cb[cb]*100:.2f}%")
            break
    
    # Error distribution
    abs_diff = np.abs(ours.astype(np.int32) - ref.astype(np.int32))
    print(f"\nExact match rate: {exact_match*100:.2f}%")
    print(f"Mean abs diff:    {np.mean(abs_diff):.2f}")
    print(f"Median abs diff:  {np.median(abs_diff):.2f}")
    print(f"Max abs diff:     {np.max(abs_diff)}")
    print(f"Std of diff:      {np.std(abs_diff):.2f}")
    
    # Distribution of errors
    errors = abs_diff[abs_diff > 0]
    if len(errors) > 0:
        print(f"\nError distribution (non-zero errors only):")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  P{p}: {np.percentile(errors, p):.2f}")
    
    return {
        "exact_match_rate": float(exact_match),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_abs_diff": int(np.max(abs_diff)),
        "n_total": n_total,
        "n_matching": int(np.sum(ours == ref)),
        "per_codebook": {f"cb{i}": float(v) for i, v in enumerate(per_cb)} if 'per_cb' in dir() else {}
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare codebook indices")
    parser.add_argument("--ours", required=True, help="Our decoded indices (.bin)")
    parser.add_argument("--ref", required=True, help="Reference indices (.bin)")
    parser.add_argument("--json", help="Output JSON report path")
    args = parser.parse_args()
    
    ours = load_indices(args.ours)
    ref = load_indices(args.ref)
    
    print(f"Our indices:  {len(ours)} elements")
    print(f"Ref indices:  {len(ref)} elements")
    print(f"Our range:    [{np.min(ours)}, {np.max(ours)}]")
    print(f"Ref range:    [{np.min(ref)}, {np.max(ref)}]")
    
    metrics = compute_accuracy(ours, ref)
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nReport saved to {args.json}")

if __name__ == "__main__":
    main()
```

**Verification**:
```bash
# Create synthetic test data
python3 -c "
import numpy as np
a = np.random.randint(0, 1024, 1000)
a.tofile('/tmp/test_ours.bin')
a.tofile('/tmp/test_ref.bin')
"
python3 experimental/compare_indices.py --ours /tmp/test_ours.bin --ref /tmp/test_ref.bin
# Expected: 100% match rate

# Test with 50% corruption
python3 -c "
import numpy as np
a = np.random.randint(0, 1024, 1000)
b = a.copy()
b[::2] = np.random.randint(0, 1024, 500)  # corrupt half
a.tofile('/tmp/test_ours.bin')
b.tofile('/tmp/test_ref.bin')
"
python3 experimental/compare_indices.py --ours /tmp/test_ours.bin --ref /tmp/test_ref.bin
# Expected: ~50% match rate
```

**Acceptance**: Script runs, produces per-codebook accuracy metrics, handles length mismatches gracefully.

---

### T3: Decode normal TXC and measure index accuracy (⬜)

**Test procedure**:
```bash
# 1. Create a normal TXC file (if not already available)
# Generate short WAV
python3 -c "
import struct, math
sr=44100; dur=0.5; n=int(sr*dur)
samples=[0.3*math.sin(2*math.pi*440*i/sr) for i in range(n)]
with open('/tmp/r172_test.wav','wb') as f:
    f.write(b'RIFF'); f.write(struct.pack('<I',36+n*2))
    f.write(b'WAVEfmt '); f.write(struct.pack('<I',16))
    f.write(struct.pack('<H',1)); f.write(struct.pack('<H',1))
    f.write(struct.pack('<I',sr)); f.write(struct.pack('<I',sr*2))
    f.write(struct.pack('<H',2)); f.write(struct.pack('<H',16))
    f.write(b'data'); f.write(struct.pack('<I',n*2))
    for s in samples: f.write(struct.pack('<h',max(-32768,min(32767,int(s*32767)))))
"

# 2. Encode with original tsac in normal mode
tsac c /tmp/r172_test.wav /tmp/r172_test_normal.txc 2>&1

# 3. Build our modified decoder with index dump feature
cmake --build /home/miao/Projects/tsac-ng/build

# 4. Run our decoder (must dump indices to /tmp/r172_our_indices.bin)
cat > /tmp/r172_dump_indices.c << 'TESTEOF'
// Small harness that calls tsac_normal_decode and dumps indices
#include "tsac.h"
#include "txc_format.h"
#include "dac_model.h"
#include "model_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int tsac_normal_decode(const uint8_t *compressed, size_t comp_len,
                               TSACTransformer *tf,
                               int n_frames, int n_cb,
                               int **indices_out, int *out_frames);

int main(int argc, char **argv) {
    const char *txc_path = argc > 1 ? argv[1] : "/tmp/r172_test_normal.txc";
    const char *model_path = argc > 2 ? argv[2] : 
        "/home/miao/Projects/tsac-ng/models/tsac/tsac_stereo_q8.bin";
    const char *dump_path = argc > 3 ? argv[3] : "/tmp/r172_our_indices.bin";
    
    // Read TXC file
    FILE *f = fopen(txc_path, "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = malloc(fsize);
    fread(data, 1, fsize, f);
    fclose(f);
    
    // Parse header
    TSCHeader hdr;
    int *indices = NULL;
    int n_frames = 0;
    txc_read(data, fsize, &hdr, &indices, &n_frames);
    free(indices);
    
    printf("File: %s, frames=%u, codebooks=%u, version=%u, flags=0x%02x\n",
           txc_path, hdr.n_blocks, hdr.n_codebooks, hdr.version, hdr.flags);
    
    // Load model + transformer
    DACModel *model = dac_model_create();
    model_loader_load(model_path, model);
    
    TSACTransformer tf;
    tsac_transformer_load(&tf, model->tensors, model->n_tensors);
    
    // Extract payload (skip 16-byte header, exclude 4-byte CRC)
    size_t payload_offset = 16;
    size_t payload_len = fsize - payload_offset - 4;
    const uint8_t *payload = data + payload_offset;
    
    printf("Payload: offset=%zu, len=%zu\n", payload_offset, payload_len);
    
    int *decoded = NULL;
    int out_frames = 0;
    int ret = tsac_normal_decode(payload, payload_len, &tf,
                                  hdr.n_blocks, hdr.n_codebooks,
                                  &decoded, &out_frames);
    
    if (ret == 0) {
        int total = out_frames * hdr.n_codebooks;
        printf("Decoded %d indices\n", total);
        
        // Dump to binary file
        FILE *dump = fopen(dump_path, "wb");
        fwrite(decoded, sizeof(int), total, dump);
        fclose(dump);
        printf("Indices dumped to %s\n", dump_path);
        
        // Print summary
        int zeros = 0, max_val = 0;
        for (int i = 0; i < total; i++) {
            if (decoded[i] == 0) zeros++;
            if (decoded[i] > max_val) max_val = decoded[i];
        }
        printf("Zero indices: %d/%d, Max index: %d\n", zeros, total, max_val);
        
        free(decoded);
    } else {
        printf("normal_decode failed: %d\n", ret);
    }
    
    tsac_transformer_free(&tf);
    dac_model_destroy(model);
    free(data);
    return ret;
}
TESTEOF

gcc -o /tmp/r172_dump_indices /tmp/r172_dump_indices.c \
    -I/home/miao/Projects/tsac-ng/include -I/home/miao/Projects/tsac-ng/src \
    -L/home/miao/Projects/tsac-ng/build -ltsac-ng -lm -lpthread
LD_LIBRARY_PATH=/home/miao/Projects/tsac-ng/build \
    /tmp/r172_dump_indices /tmp/r172_test_normal.txc

# 5. Run accuracy comparison
python3 experimental/compare_indices.py \
    --ours /tmp/r172_our_indices.bin \
    --ref /tmp/r172_our_indices.bin \  # placeholder — need actual ref
    --json /tmp/r172_metrics.json
```

**For reference indices**: Since we cannot run the original tsac's Transformer, the best approach is:
- **Self-consistency check**: Encode with our encoder (fast mode), decode with normal mode path → compare
- **Bit-exact test**: Create a known range-coded payload with deterministic symbols, verify our decoder extracts them correctly
- **WAV-level proxy**: Decode with original tsac, decode with ours, compare WAV

**Acceptance**: Index comparison script runs. Self-consistency or WAV-level metrics documented.

---

### T4: Fix identified index prediction errors (⬜)

**Based on T3 findings**, common issues and fixes:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All indices are 0 | Softmax over wrong dimension | Fix logit indexing in decode_one_frame |
| All indices are random (uniform) | g projection wrong | Fix g as [512]→scalar, find real 1024 proj |
| Most indices match, some off-by-1 | Range coder precision | Check cum_freq building, use higher precision |
| First codebook correct, others wrong | Autoregressive state not updated | Fix input_ids for subsequent codebooks in same frame |
| Frame 0 correct, later frames wrong | Position embedding or KV cache issue | Check position_ids, causal masking |

**For each fix**:
1. Make the change in `/home/miao/Projects/tsac-ng/src/tsac_normal_decode.c`
2. Rebuild: `cmake --build /home/miao/Projects/tsac-ng/build`
3. Re-run index comparison
4. Document improvement in metrics

**Acceptance**: At least one identified issue fixed. Index accuracy improved (or documented why it cannot be improved without libnc source).

---

### T5: Add self-consistency test to regression suite (⬜)

**Add to `/home/miao/Projects/tsac-ng/experimental/tests/test_normal_txc.sh`**:

```bash
# Self-consistency test: encode with our fast encoder → decode with normal path
# (If fast and normal paths share the same DAC decoder, this tests index transport only)

# Create known test vector
python3 -c "
import numpy as np
# Generate 5 frames of deterministic indices
np.random.seed(42)
indices = np.random.randint(0, 1024, 5 * 8, dtype=np.int32)
indices.tofile('/tmp/r172_known_indices.bin')
print('Known indices:', indices[:16])
"

# Encode known indices as TXC, then decode with normal path → verify round-trip
```

**Acceptance**: Self-consistency test added to regression script. Round-trip accuracy (encode→decode→compare) documented.

---

## Acceptance Criteria

- [ ] **T1**: Reference indices captured or methodology for comparison established.
- [ ] **T2**: `experimental/compare_indices.py` created, runs on synthetic data with correct results.
- [ ] **T3**: Index accuracy measured for at least one normal TXC file. Metrics documented.
- [ ] **T4**: At least one index prediction bug identified and fixed. Accuracy quantified before/after.
- [ ] **T5**: Self-consistency test added to regression suite.
- [ ] `cmake --build build` succeeds.
- [ ] Fast TXC decode NOT broken.

## Decision Gate

| Accuracy | Action |
|----------|--------|
| > 80% match | → R173 (range coder edge cases) |
| 20-80% match | Fix identified issues in T4, then → R173 |
| < 20% match | → R171 (fundamental architecture issue, redo integration) |
