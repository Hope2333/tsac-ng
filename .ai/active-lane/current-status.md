# Current Status — TSAC Reverse Engineering

## Active Lane: FINAL — All Investigation Rounds Complete

| Metric | Value | Target |
|--------|-------|--------|
| BF8 formula isolation | **0.799** (GDB verified) | 1.000 |
| RMS | 0.641 (current) | 0.203 |
| Correlation | 0.002 | >0.9 |
| Quality | 86.67 | 87+ |

### R117 Complete: BF8 Formula Deploy Test — FAILED

**Formula found**: `int8 signed, gs=32, scale=uint16→shl 0x10→float32` (corr 0.799 in isolation, 50× improvement)

**Deploy results**:
- int8 signed alone → RMS 0.96 clipping, corr 0.002
- gs=32 group-norm → corr -0.002 (grouping axis still wrong)

**Root cause**: The uint16→shl16 scale pre-computation happens during model loading (libnc internal). Without source access to libnc's model loader, the K×Co interleaved grouping pattern cannot be replicated.

**Status**: Investigation exhausted after 38 rounds (079-117). GDB sessions CLOSED. Requires libnc source access to proceed.

### Artifacts
- `/tmp/_r117.wav` — int8 signed test output
- `/tmp/_r117b.wav` — gs=32 group-norm test output
- `/tmp/_gs32.wav` — gs=32 decode test
- `/tmp/_int8.wav`, `/tmp/_signed.wav` — signed int8 tests
