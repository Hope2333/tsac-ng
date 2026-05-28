# Current Status — TSAC Reverse Engineering

## Active Lane: ROUND_114_COMPLETE → ROUND_117_PLANNED

| Metric | Value | Target |
|--------|-------|--------|
| BF8 correlation | **0.799** (was 0.016) | 1.000 |
| RMS | 0.380 (current) | 0.203 |
| Quality | 86.85 | 87+ |

### 🔥 R114 GDB Breakthrough
GDB session at libnc 0x8990 discovered actual BF8 decode formula:
**gs=32, int8_t signed, scale=uint16→shl 0x10→float32**
Python validation: correlation 0.799 with libnc (50× improvement!).

### R117: Implement this formula in dequant_weights
