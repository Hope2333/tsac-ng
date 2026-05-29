# Round 163 — Fix is_ct Detection for ALL Layers (Phase 4B)
**Signed**: Header | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
The `is_ct` flag in `dequant_weights()` (`src/cpu_decoder.c` lines 240-242) determines whether a weight tensor should be interpreted as convtranspose (`[Co][K][Ci]`) or conv1d (`[Ci][K][Co]`). The current heuristic uses bias dimension matching:

```c
int is_ct = (bias && bias->dims[0] == d0 &&
             !(name && strstr(name, "block.4.weight_v"))) ? 1 : 0;
```

**Known misclassifications** (from code comments, line 237-238):
> Exception: encoder strided convs (block.4.weight_v) have bias->dims[0]==d0 but use conv1d layout [Ci, K, Co] with K=4/8/16 stride=K/2.

**Full list of layers requiring correct is_ct**:
- Encoder: 12 conv layers (K=4/8/16 strided — ALL conv1d, is_ct=0)
- Decoder model.0: conv1d K=7, is_ct=0
- Decoder blocks 1-4 block.1: convtranspose K=16, is_ct=1
- Decoder blocks 1-4 block.2: conv1d K=7 dilation=1, is_ct=0
- Decoder blocks 1-4 block.3: conv1d K=7 dilation=3, is_ct=0
- Decoder blocks 1-4 block.4: conv1d K=7 dilation=9, is_ct=0
- Decoder model.6: conv1d K=7, is_ct=0
- RVQ quantizer in_proj: linear K=1, is_ct=0
- RVQ quantizer out_proj: linear K=1, is_ct=1

**The encoder strided convs are particularly dangerous** (R146 fixed them but the fix relied on a fragile string-match exception).

## Round Strategy
1. Audit ALL weight tensors in model for correct is_ct classification
2. Replace fragile heuristic with explicit layer-type table
3. Verify encoder K=4/8/16 convs are correctly classified as conv1d
4. Verify all 22 decoder layers are correctly classified
5. Test dequant output against libnc for every misclassified layer

## Tasks

### T1: Audit all weight tensors for is_ct correctness
**File**: `src/cpu_decoder.c` function `dequant_weights()` lines 235-242

**Action**: Create a comprehensive test that enumerates ALL weight tensors, their is_ct classification, and expected type:

```python
# Full layer audit table
LAYERS = [
    # (name, expected_is_ct, reason)
    # Encoder conv layers
    ("encoder.model.0.block.1.weight_v", 0, "conv1d stride=2 K=4"),
    ("encoder.model.0.block.3.weight_v", 0, "conv1d stride=2 K=4"),
    ("encoder.model.1.block.1.weight_v", 0, "conv1d stride=2 K=8"),
    ("encoder.model.1.block.3.weight_v", 0, "conv1d stride=2 K=8"),
    ("encoder.model.2.block.1.weight_v", 0, "conv1d stride=2 K=16"),
    ("encoder.model.2.block.3.weight_v", 0, "conv1d stride=2 K=16"),
    ("encoder.model.3.block.1.weight_v", 0, "conv1d stride=2 K=16"),
    ("encoder.model.3.block.3.weight_v", 0, "conv1d stride=2 K=16"),
    ("encoder.model.4.block.1.weight_v", 0, "conv1d stride=2 K=16"),
    ("encoder.model.4.block.3.weight_v", 0, "conv1d stride=2 K=16"),
    # ... (22 decoder layers listed explicitly)
]
```

Run `dequant_weights()` for each layer with the GDB-captured tensor, record actual vs expected is_ct.

**Acceptance**: Full audit table with actual vs expected is_ct for every weight tensor in the model.

### T2: Replace is_ct heuristic with explicit type table
**File**: `src/cpu_decoder.c`

**Action**: Replace:
```c
int is_ct = (bias && bias->dims[0] == d0 &&
             !(name && strstr(name, "block.4.weight_v"))) ? 1 : 0;
```

With a name-based lookup:
```c
static int is_conv_transpose_layer(const char *name) {
    if (!name) return 0;
    // Explicit decoder convt layers
    if (strstr(name, "decoder.model.")) {
        // model.0 = conv1d (input projection)
        if (strstr(name, "model.0.")) return 0;
        // model.6 = conv1d (output projection)
        if (strstr(name, "model.6.")) return 0;
        // model.5 = snake, no weights
        // Blocks 1-4: block.1 = convt, block.2/3/4 = inner conv1d residual units
        if (strstr(name, "block.1.")) return 1;   // convtranspose
        if (strstr(name, "block.2.")) return 0;   // inner conv1d
        if (strstr(name, "block.3.")) return 0;   // inner conv1d
        if (strstr(name, "block.4.")) return 0;   // inner conv1d
    }
    // Encoder: ALL convs are conv1d (strided, not transposed)
    if (strstr(name, "encoder.model.")) return 0;
    // Quantizer in_proj: conv1d
    if (strstr(name, "in_proj")) return 0;
    // Quantizer out_proj: convtranspose (K=1, linear up-projection)
    if (strstr(name, "out_proj")) return 1;
    return 0;
}
```

**Acceptance**: Explicit table correctly classifies all 40+ weight tensors in the model.

### T3: Fix encoder strided conv is_ct (block.4 exception)
**File**: `src/cpu_decoder.c` line 241

**Action**: Verify the `block.4` exception is no longer needed with the explicit table. Remove the fragile string-match hack:
```c
// REMOVE: !(name && strstr(name, "block.4.weight_v"))
```

**Verify** by loading encoder model.4.block.1 and block.3 weight_v:
```python
# Check encoder block.4
tensor = load_weight_tensor("encoder.model.4.block.1.weight_v")
# dims: [Ci=512, K=16, Co=512]
assert tensor.dims[0] == 512  # Ci
assert tensor.dims[1] == 16   # K
assert tensor.dims[2] == 512  # Co

# With explicit table, is_ct = 0 (correct: it's a strided conv1d)
# With old heuristic: bias->dims[0] == d0 (= 512) → is_ct = 1 (WRONG!)
```

**Acceptance**: Encoder model.4 block.1/block.3 correctly classified as conv1d without fragile exception.

### T4: Verify quantizer out_proj is_ct classification
**File**: `src/cpu_decoder.c` and model loading in `decode_batch()` lines 422-470

**Action**: The quantizer out_proj K=1 layers have a special layout:
- in_proj: [1024, 1, 8] → [Co=8][Ci=1024][K=1], is_ct=0 (correct)
- out_proj: [8, 1, 1024] → [Co=1024][Ci=8][K=1], is_ct=1 (convt)

R156 already fixed the access pattern to use:
```c
// in_proj: ip_f32[o * ip_Ci + raw]   where o=output_channel, raw=codebook_index
// out_proj: op_f32[d * op_Ci + o]    where d=output_dim, o=input_channel
```

Verify that `dequant_weights()` correctly handles out_proj K=1 with L2 normalization.

**Acceptance**: out_proj L2 normalization produces identical weights to libnc.

### T5: Full decoder test with corrected is_ct
**File**: `experimental/compare_activations.py`

**Action**: After is_ct classification is verified correct, run full decoder comparison:
```bash
cmake --build build
./build/tsac-ng -v -f d /tmp/short_fast.txc /tmp/out.wav
python3 experimental/compare_activations.py
```

**Expected**: If any layers were previously misclassified, their correlation should improve significantly. The worst affected would be encoder convs (if they were treated as convt) — these affect TXC encoding but not decoding.

**Acceptance**: No layers show corr < 0.9 due to is_ct misclassification.

## Acceptance Criteria
- [ ] Full audit table of all 40+ weight tensors with is_ct classification
- [ ] Explicit name-based type table replaces fragile bias-dim heuristic
- [ ] Encoder block.4 exception removed (no longer needed)
- [ ] Quantizer out_proj correctly handled as convt K=1
- [ ] All layers verified: no is_ct misclassifications
- [ ] Decoder output correlation stable or improved after fix
