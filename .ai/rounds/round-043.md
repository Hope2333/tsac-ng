# Round 043 — ALL 32 libnc Weights Dumped

**Date**: 2026-05-26
**Status**: Complete

## Achievement
Extended LD_PRELOAD nc_conv_1d intercept to dump ALL 32 conv1d weights from original tsac.

## Data
- 32 weight dump files: /tmp/libnc_w01.bin through /tmp/libnc_w32.bin
- Complete DAC graph confirmed from weight dims:
  - #01-06: [8,1,1024] RVQ out_proj (6 codebooks, 8192 each)
  - #07: [1024,7,1536] model.0 conv1d (11M floats)
  - #08-13: model.1 residual blocks ([768,7,768] K=7 + [768,1,768] K=1)
  - #14-19: model.2 residual blocks ([384,7,384])
  - #20-25: model.3 residual blocks ([192,7,192])
  - #26-31: model.4 residual blocks ([96,7,96])
  - #32: [96,7,2] model.6 output conv1d

## Evidence
- /tmp/preload_all.log — 32-line trace with dims, element counts
- /tmp/libnc_w*.bin — 32 weight dump files
