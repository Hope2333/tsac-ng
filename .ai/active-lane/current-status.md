# Current Status — TSAC Reverse Engineering

## Active Lane: ROUND_104_PLANNED

### Strategy Shift (May 27)
**From**: BF8 dequant formula investigation (R079-R103, 25 rounds)
**To**: Dequant weight output LAYOUT investigation — dequant_weights may produce correct VALUES but in wrong ORDER

### Evidence
- R100: BF8 in_proj bypass — correlation got WORSE (0.002→0.0005)
- RVQ transpose fix: correcting layout made output WORSE (0.641→0.380)
- R103: 13 BF8 formulas all failed (best corr 0.039)
- **New hypothesis**: dequant_weights output ordering mismatches libnc for ALL layers

### Next
R104 — Compare our model.0 dequant output with libnc LD_PRELOAD capture byte-for-byte.
