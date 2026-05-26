# Round 045 — Divergence Point Identified

**Date**: 2026-05-26
**Status**: Complete

## Key Finding
The RMS divergence between our decoder and libnc's **reverses** through the DAC graph:

```
Layer      libnc_RMS    our_RMS    Ratio
model.0      2.901       0.609     libnc 4.8× LARGER
... (residual blocks + convtr) ...
model.6      0.203       0.641     our 3.2× LARGER
```

The reversal happens in the **residual blocks + convtr upsampling** stages (models 1-4). These stages contain:
- Snake activation
- ConvTranspose1d (upsampling by 8× or 4×)
- 3× residual blocks (each: snake→conv1d K=7→snake→conv1d K=1 + residual connection)

## Root Cause Hypothesis
1. **ConvTranspose1d weights**: These are stored as [Co,K,Ci] with is_ct=1 flag. Our dequant might incorrectly handle the norm axis or weight_g application for convtr tensors.
2. **Residual connections**: Our feed_forward implementation might differ from libnc's nc_add.
3. **Snake activation**: alpha parameter handling may differ.

## Next Steps
- Round 046: Dump model.1 block1 convtr input/output from libnc for direct comparison
- Round 047: Verify convtr weight dequant (is_ct=1 path) produces correct weights
- Round 048: Test with fixed convtr weights
