# Round 044 — libnc Conv1d Output RMS Pipeline

**Date**: 2026-05-26
**Status**: Complete

## Data Collected
Extended LD_PRELOAD to dump conv1d OUTPUT tensors with full RMS.

## libnc Full-Tensor Output RMS
| Call | Layer | RMS | n |
|------|-------|-----|---|
| #01 | RVQ codebook 0 | 1.061 | 9216 |
| #07 | model.0 conv1d | 2.901 | 13824 |
| #14 | model.2 block2 block1 | 0.576 | 221184 |
| #20 | model.3 block2 block1 | 0.278 | 442368 |
| #26 | model.4 block2 block1 | 0.281 | 442368 |
| #32 | model.6 final output | **0.203** | 9216 |

## Our Decoder (captured)
| Layer | Our RMS | n |
|-------|---------|---|
| model.0 conv1d | 0.609 | 13824 |
| Final WAV | 0.641 | 9216 |

## Key Finding
At model.0: libnc RMS=2.901, our RMS=0.609 → libnc is **4.8× LARGER**
At final: libnc RMS=0.203, our RMS=0.641 → our is **3.2× LARGER**

The relationship **REVERSES** through the DAC pipeline. The divergence originates in the residual blocks + convtr upsampling, not in model.0 or RVQ.

## Evidence
- /tmp/libnc_out*.bin — 6 output dumps with full RMS computed
- /tmp/preload_io2.log — RMS trace log
