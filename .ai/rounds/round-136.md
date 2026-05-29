# Round 136 — GDB Trace: Original tsac ConvTranspose Execution

**Status**: COMPLETED (Header Planned) | **Date**: 2026-05-28
**Predecessor**: round-135 (M1 sign-off) | **Priority**: CRITICAL

## Strategy
Original tsac's convtr (conv_transpose) layers have different BF8 stride patterns.
GDB tracing of nc_conv_transpose_1d calls will reveal:
1. Weight layout for convtr layers ([Co,K,Ci] vs [Ci,Co,K])
2. BF8 grouping axis for K=16/8/4 convtr kernels
3. Input/output tensor shapes and memory access patterns

This is essential for M2 BF8 stride extraction.

## Tasks

### T1: Identify convtr call addresses
- Using docs/evidence/gdb_all_nc.txt, map all nc_conv_transpose_1d calls
- Record address, input/output dims for each of the 4 convtr layers
- Verify against DAC architecture (block.1 of models 1-4)

### T2: GDB breakpoint at first convtr
- Set breakpoint at nc_conv_transpose_1d entry
- Capture input tensor dims and weight tensor dims
- Record rdi (output), rsi (weight), rdx (input) for each call

### T3: Weight stride analysis
- At each convtr breakpoint, dump weight tensor memory layout
- Extract K×Co grouping pattern from raw bytes
- Compare with our dequant_weights convtr path (is_ct=1)

### T4: Input/output tensor comparison
- Capture convtr input and output for layer model.1.block.1
- Compare with our convt1d_parallel output
- Identify divergence source

### T5: Document findings
- Create convtr stride reference document
- Map all 4 convtr layer stride patterns
- Feed into R137 BF8 formula deployment
