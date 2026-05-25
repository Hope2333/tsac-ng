# tsac-ng — Neural Audio Codec (Multi-Backend)

**tsac-ng v0.1.1** — Clean-room reimplementation of the TSAC neural audio codec.
Compatible with the `.txc` container format and `.bin` model files.

> Relationship to TSAC: Like Linux to Unix — same ecosystem compatibility,
> built from scratch with zero shared code.

---

## Compatibility Status (Honest Assessment)

| Feature | Status | Notes |
|---------|:------:|-------|
| **Our own fast TXC encode/decode** | ✅ | Raw uint8 format, works correctly |
| **Original tsac fast TXC decode** | ⚠️ | 10-bit indices 100% correct (GDB verified). Audio output wrong: RMS -3.4dB. Root cause identified (libnc BF8 dequant + weight_norm fused op differs from ours) |
| **Original tsac normal TXC decode** | ❌ | Transformer + range coder not implemented |
| **Verbose output parity** | ✅ | batch_size, progress %, bitrate, AVG_BITS table — all match original |
| **CRC32 validation** | ✅ | Fully reversed (polynomial 0x04C11DB7), implemented |
| **CPU decoder (DAC)** | ✅ | 32 conv1d/29 snake/4 convtr verified via GDB |
| **CPU encoder** | 🔧 | Architecture correct, needs strided convs |

### What We Know (23 Rounds of Deep Reverse Engineering)

- **Fast TXC format**: 10-bit fixed-width bit packing (NOT arithmetic range coding). Algorithm: `bswap + shr(22-(bp&7)) + and 0x3FF`. Verified 54/54 indices against original GDB ground truth.
- **Normal TXC format**: n_blocks in BE uint32 at bytes 8-11, payload at byte 16, CRC32 at end.
- **Transformer model**: 12-layer decoder-only, d_model=512, n_head=4, RoPE positional encoding.
- **Range coder (arith.c)**: get_freq (15-bit adaptive probability) is used in normal mode. get_bit is dead code. Fast mode does NOT call any arith.c functions — range coder is inline.
- **RMS -3.4dB root cause**: libnc's `nc_convert_from_old_bf` (0x61370) combines BF8 dequant + L2 norm + weight scale into one fused operation. Our separate steps produce slightly different weights.
- **BF8 bug**: Fixed double-division by 127 (commit 6a42865).

---

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

# Decompress our own fast TXC files
./tsac-ng -v d input.txc output.wav

# Decompress original tsac fast TXC files (produces audio, but not bit-accurate yet)
./tsac-ng -v d original_fast.txc output.wav

# With CUDA
cmake .. -DUSE_CUDA=ON -DCUDAToolkit_ROOT=/opt/cuda
./tsac-ng --cuda -v d input.txc output.wav
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
| LLVM JIT | ✅ | ⚠️ | Experimental |

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
(1536→768→384→192→96) → Snake → Conv1d(96→2) → tanh → PCM

## Project Structure

```
tsac-ng/
├── src/
│   ├── cpu_decoder.c      # CPU decoder + encoder + BF8 dequant
│   ├── range_coder.c      # get_freq adaptive range coder (arith.c RE)
│   ├── txc_format.c       # .txc parser (10-bit bitpacking + CRC32)
│   ├── tsac_codec.c       # Codec API + WAV I/O + bitrate display
│   ├── model_loader.c     # .bin model loader (BF8/float32 auto-detect)
│   ├── main.c             # CLI (compatible with original tsac)
│   ├── cuda/              # CUDA backend (kernels + backend)
│   ├── llvm/              # LLVM JIT backend (experimental)
│   ├── vulkan/            # Vulkan compute backend
│   ├── arch/arm/          # ARM NEON + SVE
│   └── arch/riscv/        # RISC-V RVV
├── hip/                   # HIP/ROCm backend
├── include/               # Public headers
├── docs/evidence/         # GDB ground truth + libnc disassembly
├── cmake/                 # Toolchain files
└── experimental/          # Experimental code
```

## CLI Reference

```
tsac-ng [options] c|d|t infile outfile

Options (compatible with original tsac):
  --cuda, --hip, --vulkan, --llvm   GPU/accelerator backend
  -q, --n_codebooks n    Codebooks (1-12 stereo, 1-9 mono, default=max)
  -T n                   Thread count (default=1)
  -v                     Verbose mode (batch_size, progress, bitrate, AVG_BITS)
  -h, --help             Show help
  -s, --separate_channels  Stereo as dual mono
  -c, --channels n       Force channel count
  -f, --fast             Fast mode (no transformer)
  -m, --model path       Model file path (directory or direct .bin path)
  -M, --trf_model path   Transformer model path
  --batch_size n         Batch size (default=auto)
```

## Known Limitations

- **Original fast TXC audio**: Decoded indices are 100% correct, but audio output has -3.4dB RMS error due to BF8 dequantization mismatch with libnc's fused operation.
- **Normal TXC**: Transformer + range coder path not yet implemented.
- **Encoder**: Strided convolutions missing for proper temporal encoding.

## Roadmap

See [.ai/ROADMAP.md](.ai/ROADMAP.md) for detailed milestone planning.
Current phase: **ROUND_023_COMPLETE** (~60% overall completion).

## License

MIT

---

```
tsac-ng v0.1.1 — Copyright (c) 2026 Hope2333 (幽零小喵)
```
