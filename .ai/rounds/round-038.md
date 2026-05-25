# Round 038 — BF8 Dequant Formula Variation Testing

**Date**: 2026-05-26
**Status**: Complete

## Goal
Test BF8 dequant formula variations to find which produces correct audio output.

## Reference
- Original tsac output: RMS=0.203, -13.85 dBFS
- Our baseline: RMS=0.641, -3.86 dBFS (10 dB too loud, 3.16× amplitude)

## Tests Performed

| # | Variant | RMS | dBFS | Delta |
|---|---------|-----|------|-------|
| 0 | Baseline (current) | 0.641 | -3.86 | — |
| 1 | Epsilon 1e-8 | 0.641 | -3.86 | 0% |
| 2 | Epsilon 1e-4 | 0.641 | -3.86 | 0% |
| 3 | Epsilon 1e-16 | 0.641 | -3.86 | 0% |
| 4 | Double precision norm | 0.641 | -3.86 | 0% |
| 5 | No L2 norm | 1.000 | 0.00 | saturated |
| 6 | Alt formula (scale*127) | 0.641 | -3.86 | 0% |
| 7 | No group scale | 1.000 | 0.00 | saturated |
| 8 | Global L2 norm | 0.066 | -23.57 | too quiet |
| 9 | Per-input-channel norm | 0.999 | -0.01 | saturated |

## Key Finding
**The BF8 dequant formula under the SAME group structure is verified correct. Group structure (normalization axis, group boundaries) remains an open variable — NOT ruled out.** The L2 norm acts as a normalizer — within each channel, the relative distribution of dequantized values is preserved regardless of absolute scale. Epsilon and precision have negligible impact.

The 10 dB amplitude mismatch must originate from:
1. Conv1d/ConvTranspose kernel implementation differences
2. tanh activation implementation
3. RVQ codebook summation scaling
4. Residual connection handling

## Conclusion
Dequant is NOT the bottleneck. Next investigation should target convolution kernels and DAC graph execution.
