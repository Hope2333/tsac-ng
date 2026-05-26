# Round 040 — nc_conv_1d Calling Convention Cracked

**Date**: 2026-05-26
**Status**: Complete (BREAKTHROUGH)

## Discovery
Disassembled libnc `nc_conv_1d` at 0x70d40 to determine actual calling convention.

### Actual Signature
```
nc_conv_1d(output_tensor, weight_tensor, input_tensor, stride?, pad?, dilation?, groups?)
   rdi = output    rsi = weight     rdx = input    ecx   r8d    r9d     stack
```

Previously assumed: state in rdi, output in rsi, input in rdx, weight in rcx — ALL WRONG.

### Implementation
- LD_PRELOAD with correct argument positions
- rsi = weight tensor (previously assumed rcx)
- Successfully intercepted 32 calls per batch
- Dumped first weight: [8, 1, 1024] = 8192 floats → `/tmp/libnc_conv1.bin`

## Key Data
- First conv1d uses quantizer out_proj weight
- Int args: [0 0 1 1] for calls #1-6 (snake/group_norm), [3 3 1 1] for conv with stride=3, etc.
- Pattern confirmed: 32 calls = 6 snake + 4×2 conv1d + 4×2 convtr = 30 DAC + 2 RVQ

## Evidence
- `/tmp/libnc_conv1.bin` — 32768 bytes, 8192 float32 values
- `/tmp/preload_correct.log` — full 32-call trace
