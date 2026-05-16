# tsac-ng — Neural Audio Codec (Multi-Backend)

**tsac-ng v0.1.0** — Clean-room implementation of a neural audio codec,
independently developed from first principles. Compatible with the `.txc`
container format and `.bin` model files used by TSAC.

> Relationship to TSAC: Like Linux to Unix — same ecosystem compatibility,
> built from scratch with zero shared code.

## Features

- **5 CPU SIMD levels** across 3 architectures (x86-64 AVX/AVX2/AVX-512, ARM NEON/SVE, RISC-V RVV)
- **3 GPU backends**: CUDA (NVIDIA), HIP/ROCm (AMD), Vulkan (cross-platform)
- **1 experimental backend**: LLVM JIT
- Runtime CPUID dispatch — auto-selects best SIMD with scalar fallback
- Zero `system()` calls — fully self-contained
- CLI compatible with original `tsac` (2024-04-08)

## Quick Start

```bash
# Build (CPU backend, x86-64)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Decompress
./tsac-ng -v d input.txc output.wav

# With CUDA
cmake .. -DUSE_CUDA=ON -DCUDAToolkit_ROOT=/opt/cuda
./tsac-ng --cuda -v d input.txc output.wav

# With HIP/ROCm
cmake .. -DUSE_HIP=ON -DHIP_PATH=/opt/rocm
./tsac-ng --hip -v d input.txc output.wav
```

## Backend Status

| Backend | Build | Runtime | Notes |
|---------|:-----:|:-------:|-------|
| CPU (x86-64) | ✅ | ✅ | AVX/AVX2/AVX-512 auto-dispatch |
| CPU (ARM64) | ✅ | ✅ | NEON + SVE auto-detect |
| CPU (RISC-V) | ✅ | ✅ | RVV + scalar fallback |
| CUDA | ✅ | ✅ | SM 8.0+, Runtime API |
| HIP/ROCm | ✅ | ✅ | gfx1030+, ROCm 7.x |
| Vulkan | ✅ | ⚠️ | Cross-compile for ARM64 Mali |
| LLVM JIT | ✅ | ⚠️ | Experimental, init hangs on LLVM 22 |

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  .txc file  │───▶│  txc_format  │───▶│ codebook_idx │
└─────────────┘    └──────────────┘    └──────┬───────┘
                                              │
                    ┌─────────────────────────┘
                    ▼
┌──────────┐  RVQ lookup  ┌──────────┐  decode graph  ┌──────┐
│ .bin     │─────────────▶│  1024-d  │───────────────▶│ PCM  │
│ model    │  12 codebooks│ features  │  7-layer DAC  │audio │
└──────────┘              └──────────┘                └──────┘
```

**Decoder graph**: RVQ Codebook → Conv1d(1024→1536) → 4× ResidualBlock
(1536→768→384→192→96) → Snake → Conv1d(96→2) → PCM

## CLI Reference

```
tsac-ng [options] c|d|t infile outfile

Options (compatible with original tsac):
  --cuda, --hip, --vulkan, --llvm   GPU/accelerator backend
  -q, --n_codebooks n    Codebooks (1-12 stereo, 1-9 mono, default=max)
  -T n                   Thread count (default=1)
  -v                     Verbose mode
  -h, --help             Show help
  -s, --separate_channels  Stereo as dual mono
  -c, --channels n       Force channel count
  -f, --fast             Fast mode (no transformer)
  -m, --model path       Model file path
  -M, --trf_model path   Transformer model path
  --batch_size n         Batch size (default=auto)
```

## Cross-Compilation

```bash
# ARM64 (Termux, Raspberry Pi 5, etc.)
cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-arm64.cmake

# RISC-V (experimental)
cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-riscv64.cmake
```

## Project Structure

```
tsac-ng/
├── src/
│   ├── cpu_decoder.c      # CPU decoder + x86 SIMD dispatch
│   ├── tsac_codec.c       # Codec API + WAV I/O
│   ├── txc_format.c       # .txc container parser
│   ├── model_loader.c     # .bin model loader (BF8/float32)
│   ├── main.c             # CLI (compatible with original tsac)
│   ├── cuda/              # CUDA backend
│   │   ├── cuda_kernels.cu
│   │   └── cuda_backend.cu
│   ├── llvm/              # LLVM JIT backend (experimental)
│   ├── vulkan/            # Vulkan compute backend
│   ├── arch/arm/          # ARM NEON + SVE
│   └── arch/riscv/        # RISC-V RVV
├── hip/                   # HIP/ROCm backend
├── include/               # Public headers
├── cmake/                 # Toolchain files
└── experimental/          # Experimental code
```

## Compatibility

- `.txc` container format (original TSAC)
- `.bin` model files (DAC stereo/mono, q8)
- Original `tsac` CLI arguments

## License

MIT

---

```
tsac-ng v0.1.0 — Copyright (c) 2026 Hope2333 (幽零小喵)
```
