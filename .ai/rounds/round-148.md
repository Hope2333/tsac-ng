# Round 148 — GPU Backend: CUDA Implementation

**Status**: PENDING (Header Planned) | **Predecessor**: round-147

## Tasks
### T1: Port dequant_weights to CUDA kernel
### T2: Port conv1d to cuBLAS matmul
### T3: Port snake activation to CUDA
### T4: Port convtr to CUDA
### T5: Benchmark vs CPU (speed comparison table)

## 🔬 Explore Agent Findings (bg_cec1be66)
### CUDA Status
- `cuda_backend.cu` (797L): **COMPLETE** — full DAC decode+encode graph
- `cuda_kernels.cu` (284L): **COMPLETE** — all kernel launchers
- `nc_cuda_device.cu` (541L): **PARTIAL** — 40% tensor ops, matmul/attention/layernorm placeholder
- `cuda_arch.cu` (163L): **STUB** — legacy, not compiled, can be removed

### HIP Status
- `dac_decoder.hip.cpp` (312L): **BROKEN** — `#include`-splicing mid-function won't compile
- 5 `.inc` files need restructuring into proper module
- All decoder/encoder kernels EXIST but can't be built

### Vulkan Status
- Pipeline infra COMPLETE (4 compute shaders: add/conv1d/snake/group_norm)
- `tsac_vk_decode/encode`: **NOT IMPLEMENTED** — stub returns error
- Missing shaders: convtr, tanh_clip (or implement via existing)

### LLVM JIT Status
- 4 JIT functions working (conv1d verified by sanity test)
- `tsac_llvm_decode`: **STUB** — returns TSAC_ERR_BACKEND
