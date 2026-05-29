# Round 163 — Fix is_ct Detection for ALL Layers (Header Dispatch)
**Signed**: Worker | **Date**: 2026-05-28 | **Status**: PENDING

## Summary
The `is_ct` (is_conv_transpose) flag in `dequant_weights()` at `cpu_decoder.c:240-242` determines the weight layout interpretation and output rearrangement. The current heuristic is:
```c
int is_ct = (bias && bias->dims[0] == d0 &&
             !(name && strstr(name, "block.4.weight_v"))) ? 1 : 0;
```
This has the encoder block.4 exception hardcoded to prevent misclassification. However, the encoder has 4 strided conv layers (blocks 1-4, each with K=4/8/16/16) that could also be misclassified. Additionally, the 22 decoder layers (model.0 conv1d, 4× convtr, 12× inner conv1d, model.6 conv1d) need individual verification. A misclassified `is_ct` causes wrong weight layout (`[Co][K][Ci]` vs `[Co][Ci][K]`) which produces incorrect activations.

**Strategy**: Systematically audit ALL 32+4 layers (decoder: 1 + 4 + 12 + 1 = 18 direct layers, plus encoder: 1 + 4 + 12 + 1 = 18) by examining tensor dimensions and bias sizes. Replace the heuristic with an explicit layer-type lookup table. Verify each layer's dequant output against libnc reference. Test with all available TXC files to confirm no regression.

## Tasks

### T1: Build Complete Layer-Type Lookup Table
- Replace the string-based heuristic in `dequant_weights` with an explicit table lookup:
  ```c
  typedef enum { LAYER_CONV1D, LAYER_CONVTRANSPOSE, LAYER_CONV1D_STRIDED } LayerType;
  
  static LayerType get_layer_type(const char *name, int d0, int d1, int d2, const DACTensor *bias) {
      if (!name) return LAYER_CONV1D;
      
      // Decoder layers (explicit list)
      if (strstr(name, "decoder.model.0.weight_v")) return LAYER_CONV1D;       // K=7, Ci=1024, Co=1536
      if (strstr(name, "decoder.model.1.block.1.weight_v")) return LAYER_CONVTRANSPOSE;  // K=11, upsample
      if (strstr(name, "decoder.model.2.block.1.weight_v")) return LAYER_CONVTRANSPOSE;  // K=11, upsample
      if (strstr(name, "decoder.model.3.block.1.weight_v")) return LAYER_CONVTRANSPOSE;  // K=11, upsample
      if (strstr(name, "decoder.model.4.block.1.weight_v")) return LAYER_CONVTRANSPOSE;  // K=7,  upsample
      if (strstr(name, "decoder.model.6.weight_v")) return LAYER_CONV1D;       // K=7,  Ci=96, Co=2
      
      // Decoder inner conv1d layers (K=7, dilated)
      if (strstr(name, "decoder.model.") && strstr(name, "block.1.weight_v") && 
          !strstr(name, "block.1.block")) return LAYER_CONV1D;  // outer conv1d is actually convtr
        
      // Inner residual conv1d: block.2.block.1, block.3.block.1, block.4.block.1
      // These are always conv1d with K=7 (dilated)
      if (strstr(name, "block.1.weight_v") && strstr(name, "block.")) {
          // Check if it's inside a residual block (block.2, block.3, block.4)
          // Pattern: decoder.model.N.block.M.block.1.weight_v where M >= 2
          // These are conv1d layers
          return LAYER_CONV1D;
      }
      
      // Encoder layers
      if (strstr(name, "encoder.block.0.weight_v")) return LAYER_CONV1D;  // K=7, input conv
      if (strstr(name, "encoder.block.") && strstr(name, "block.4.weight_v")) return LAYER_CONV1D_STRIDED;  // K=4/8/16
      if (strstr(name, "encoder.block.") && strstr(name, "block.1.weight_v") && 
          !strstr(name, "block.1.block")) return LAYER_CONV1D;  // encoder conv1d (not strided)
      
      // Fallback: use old heuristic
      int Co = bias ? bias->dims[0] : d2;
      return (Co == d0 && d0 != d2) ? LAYER_CONVTRANSPOSE : LAYER_CONV1D;
  }
  ```
- Update `dequant_weights` signature or internal logic to use `get_layer_type`.
- **Key insight**: The correct classification rule is:
  - Convtranspose layers have `bias->dims[0] == d0` (Co == d0) AND weight is stored in `[Co][K][Ci]` layout
  - Conv1d layers have `bias->dims[0] == d2` (Co == d2) AND weight is stored in `[Ci][K][Co]` layout  
  - Strided conv1d (encoder) have `bias->dims[0] == d0` but weight is stored in `[Ci][K][Co]` layout — this is the edge case
  - **The true discriminant**: Strided convs have `K != 7` (K=4,8,16) and `d0 != d2` (input channels ≠ output channels), while convtranspose has `K == 11 || K == 7` AND `d0 == d2 == Co`.

### T2: Verify ALL 22 Decoder Layer Classifications
- Decoder layer inventory:
  | Name | d0 | d1(K) | d2 | bias.dims[0] | Correct Type |
  |------|----|-------|-----|--------------|--------------|
  | model.0.weight_v | 1024 | 7 | 1536 | 1536 | CONV1D |
  | model.1.block.1.weight_v | 768 | 11 | 1536 | 768 | CONVTRANSPOSE |
  | model.1.block.2.block.1.weight_v | 768 | 7 | 768 | 768 | CONV1D |
  | model.1.block.3.block.1.weight_v | 768 | 7 | 768 | 768 | CONV1D |
  | model.1.block.4.block.1.weight_v | 768 | 1 | 768 | 768 | CONV1D |
  | model.2.block.1.weight_v | 384 | 11 | 768 | 384 | CONVTRANSPOSE |
  | model.2.block.2.block.1.weight_v | 384 | 7 | 384 | 384 | CONV1D |
  | model.2.block.3.block.1.weight_v | 384 | 7 | 384 | 384 | CONV1D |
  | model.2.block.4.block.1.weight_v | 384 | 1 | 384 | 384 | CONV1D |
  | model.3.block.1.weight_v | 192 | 11 | 384 | 192 | CONVTRANSPOSE |
  | model.3.block.2.block.1.weight_v | 192 | 7 | 192 | 192 | CONV1D |
  | model.3.block.3.block.1.weight_v | 192 | 7 | 192 | 192 | CONV1D |
  | model.3.block.4.block.1.weight_v | 192 | 1 | 192 | 192 | CONV1D |
  | model.4.block.1.weight_v | 96 | 7 | 192 | 96 | CONVTRANSPOSE |
  | model.4.block.2.block.1.weight_v | 96 | 7 | 96 | 96 | CONV1D |
  | model.4.block.3.block.1.weight_v | 96 | 7 | 96 | 96 | CONV1D |
  | model.4.block.4.block.1.weight_v | 96 | 1 | 96 | 96 | CONV1D |
  | model.6.weight_v | 96 | 7 | 2 | 2 | CONV1D |
  
- **Note**: For conv1d layers with `K=1` (block.4 of each residual unit), the kernel size is 1. The layout distinction is irrelevant since `K=1` means `[Co][1][Ci] == [Co][Ci][1]`, but the output channel ordering must still be correct.
- Write a test that loads ALL decoder weight tensors, runs `dequant_weights`, and verifies:
  1. `is_ct` matches expected type
  2. Output weight tensor has correct dimensions
  3. Weight values are finite (no NaN, no Inf)

### T3: Fix Encoder K=4/8/16 Misclassification & Verify ALL 18 Encoder Layers
- Encoder strided conv layers:
  | Name | d0 | d1(K) | d2 | bias.dims[0] | Correct Type |
  |------|----|-------|-----|--------------|--------------|
  | encoder.block.0.weight_v | 64 | 7 | 1536 | 1536 | CONV1D |
  | encoder.block.1.block.4.weight_v | 128 | 4 | 64 | 128 | CONV1D_STRIDED |
  | encoder.block.2.block.4.weight_v | 256 | 8 | 128 | 256 | CONV1D_STRIDED |
  | encoder.block.3.block.4.weight_v | 512 | 16 | 256 | 512 | CONV1D_STRIDED |
  | encoder.block.4.block.4.weight_v | 1024 | 16 | 512 | 1024 | CONV1D_STRIDED |
  
- The current exception only catches `block.4.weight_v` (encoder block 4, the last one). But blocks 1, 2, 3 also have `block.4.weight_v` inside their residual blocks. Wait — looking at the encoder structure more carefully:
  - Encoder naming: `encoder.block.N.block.M.*` where N=1..4 is the outer block, M=2..4 are inner residual blocks
  - The strided conv is at `encoder.block.N.block.4.weight_v` where N={1,2,3,4}
  - Current exception: `strstr(name, "block.4.weight_v")` catches ALL of them, which is correct
- **BUT**: Check if there are encoder conv layers that are NOT strided but DO have `bias->dims[0] == d0`. For example, inner residual conv1d layers in the encoder. These should be CONV1D.
- **Fix**: The string check `"block.4.weight_v"` is too broad — it also matches decoder inner residual block.4 (which is conv1d with K=1). Actually, decoder inner block.4 has K=1 and bias->dims[0]=d0=d2 so the old heuristic `bias->dims[0]==d0 && !block.4` would classify it as convtr (wrong). Let me check: for decoder model.1.block.4.block.1.weight_v, d0=768, d1=1, d2=768, bias.dims[0]=768. Here d0==d2 so the old heuristic `Co = bias->dims[0] = d0 = d2` would give Ci = d2 = d0 = Co which is self-consistent. But `bias->dims[0] == d0` is true, so it would be classified as convtr with the old heuristic if `block.4.weight_v` wasn't excluded. Since `block.4.weight_v` IS in the string, it's excluded from convtr and treated as conv1d. **This is accidentally correct**.
  
- **Proper fix**: Instead of string-based excluding, use the explicit layer type table from T1. For every layer, determine its type by its role in the architecture, not by inference from dims.

### T4: Test Dequant Output Matches libnc for ALL Layers
- Use existing libnc override files (from `/tmp/libnc_OVR_*.bin`) or generate them by running the original tsac with LD_PRELOAD.
- For each layer:
  1. Load our dequantized weights with the corrected `is_ct` detection
  2. Load libnc float32 reference (from model_loader.c override mechanism, or from files)
  3. Compute per-element correlation and max-abs-diff
- **Expected**: After fixing `is_ct`, all layers should have weight correlation > 0.98 with libnc refs.
- Log layers with corr < 0.98 for further investigation.
- Verify that the encoder strided convs (K=4/8/16/16) dequant correctly as conv1d (not convtr), producing `[Co][Ci][K]` layout with correct dimensions.

### T5: Regression Test — Decode with Fixed is_ct + Compare RMS
- Build and run the full decoder with the fixed `is_ct` detection
- Encode a known WAV → TXC, then decode back to WAV
- Compare with the baseline (before fix): RMS should NOT regress
- If RMS changes significantly (>5%), investigate which layer's reclassification caused it
- Expected outcome: RMS remains stable (0.641 baseline) or improves. Correlation with original tsac output should increase.

## Acceptance Criteria
1. `get_layer_type()` implemented as explicit lookup table — replaces string heuristic
2. All 22 decoder layers verified: 1 conv1d + 4 convtr + 12 inner conv1d + 1 conv1d → exact match
3. All 18 encoder layers verified: 1 conv1d + 4 strided conv1d + 12 inner conv1d + 1 conv1d → exact match
4. Encoder block.4.weight_v exception correctly covers ALL 4 encoder blocks (1-4), not just block 4
5. Decoder inner block.4 (K=1) not misclassified as convtr
6. Per-layer weight correlation with libnc > 0.98
7. No regression in decoder output RMS (baseline 0.641)
8. Build clean, no warnings
