# Round 046 — ConvTranspose Weight Comparison

**Date**: 2026-05-26
**Status**: Complete (data collected, comparison performed)

## Achievement
1. Intercepted nc_conv_transpose_1d, dumped model.1.block.1 weight [768,16,1536] = 18.9M floats
2. Compared first 16 values of our dequant vs libnc

## Comparison Results
| # | Our | libnc | Diff |
|---|-----|-------|------|
| 0 | -0.006642 | 0.001577 | 0.008219 |
| 1 | -0.001084 | 0.004188 | 0.005272 |
| 2 | 0.009769 | -0.001480 | 0.011249 |
| 3 | -0.008764 | 0.000314 | 0.009078 |
| 4 | 0.000213 | 0.000235 | 0.000022 |

**RMS**: our=0.00576, libnc=0.00153 → our values are **3.8× LARGER**

## Key Finding
ALL 16 values differ — not a systematic scaling or offset. Signs differ for several values.

## Norm Analysis
- Before fix (per-Co norm for is_ct=1): each norm covers 24576 elements, produces weights 3.8× too large
- Attempted fix (per-Ci norm): each norm covers 12288 elements → weights become larger → output saturates
- Libnc must use an even LARGER norm (more elements per norm) to produce smaller weights

## Evidence
- /tmp/libnc_cvt1.bin — 75MB libnc convtr weight dump
- /tmp/preload_convt2.log — intercept confirms correct dims
