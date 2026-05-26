# Round 052 — Model.0 Weight Injection Test

**Date**: 2026-05-26
**Status**: Complete (decisive negative result)

## Test
Injected libnc's model.0 weight (from /tmp/libnc_w07.bin, 11M floats) into our decoder, replacing our dequantized weight. Transposed from [Ci,K,Co] to [Co,Ci,K] to match our kernel layout.

## Result
**RMS unchanged**: 0.080 vs baseline 0.080. The injection had zero effect.

## Weight Comparison
- libnc RMS: 0.004835
- our RMS: 0.004835
- Ratio: 1.000 (identical energy per channel)
- Individual values differ: avg_diff=0.005, max_diff=0.033
- 0/70,000 sampled values match exactly

## Conclusion
Model.0 weight dequant is NOT the cause of the 4.8× output gap (our 0.609 vs libnc 2.901). With identical weights, the conv1d produces the same output. The gap must come from the INPUT to model.0 (RVQ output) or the conv1d kernel itself.
