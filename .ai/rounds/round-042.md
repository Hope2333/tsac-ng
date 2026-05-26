# Round 042 — nc_conv_1d Intercept Breakthrough

**Date**: 2026-05-26
**Status**: Complete

## Achievement
Successfully cracked nc_conv_1d's calling convention and dumped libnc float32 weights from the running original tsac binary. This was the #1 blocker for 20+ rounds.

## What This Unlocks
1. Can dump ANY conv1d weight from original tsac — compare with our dequant
2. Can dump conv1d INPUTS and OUTPUTS — layer-by-layer comparison
3. Can identify exactly which layer first diverges
4. Paves way for full DAC graph verification

## Method
```
LD_PRELOAD nc_conv_1d → read weight via nc_tensor_get_ptr → dump to file
Signature: output_tensor(rdi) + weight_tensor(rsi) + input_tensor(rdx) + ints
```

## Next Steps
- Dump all 32 conv1d weights (not just first)
- Compare each against our dequant
- Dump conv1d inputs/outputs for layer-by-layer RMS tracking
- Identify which layer first diverges → pinpoint exact bug
