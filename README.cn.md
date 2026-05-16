# tsac-ng — 神经音频编解码器（多后端）

**tsac-ng v0.1.0** — 从第一性原理独立开发的神经音频编解码器。
兼容 `.txc` 容器格式和 `.bin` 模型文件。

> 与 TSAC 的关系：如同 Linux 之于 Unix — 生态系统兼容，从零构建，零共享代码。

## 特性

- **5 级 CPU SIMD**，覆盖 3 种架构（x86-64 AVX/AVX2/AVX-512、ARM NEON/SVE、RISC-V RVV）
- **3 种 GPU 后端**：CUDA (NVIDIA)、HIP/ROCm (AMD)、Vulkan（跨平台）
- **1 种实验性后端**：LLVM JIT
- 运行时 CPUID 自动调度 — 自动选择最优 SIMD，无 SIMD 时回退标量
- 零 `system()` 调用 — 完全自包含
- CLI 兼容原始 `tsac`（2024-04-08）

## 快速开始

```bash
# 构建（CPU 后端，x86-64）
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 解压
./tsac-ng -v d input.txc output.wav

# 使用 CUDA
cmake .. -DUSE_CUDA=ON -DCUDAToolkit_ROOT=/opt/cuda
./tsac-ng --cuda -v d input.txc output.wav

# 使用 HIP/ROCm
cmake .. -DUSE_HIP=ON -DHIP_PATH=/opt/rocm
./tsac-ng --hip -v d input.txc output.wav
```

## 后端状态

| 后端 | 构建 | 运行 | 备注 |
|---------|:-----:|:-------:|-------|
| CPU (x86-64) | ✅ | ✅ | AVX/AVX2/AVX-512 自动调度 |
| CPU (ARM64) | ✅ | ✅ | NEON + SVE 自动检测 |
| CPU (RISC-V) | ✅ | ✅ | RVV + 标量回退 |
| CUDA | ✅ | ✅ | SM 8.0+, Runtime API |
| HIP/ROCm | ✅ | ✅ | gfx1030+, ROCm 7.x |
| Vulkan | ✅ | ⚠️ | ARM64 Mali 交叉编译 |
| LLVM JIT | ✅ | ⚠️ | 实验性，LLVM 22 上 init 挂起 |

## 架构

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  .txc 文件   │───▶│  txc_format  │───▶│ codebook_idx │
└─────────────┘    └──────────────┘    └──────┬───────┘
                                              │
                    ┌─────────────────────────┘
                    ▼
┌──────────┐  RVQ 查表  ┌──────────┐  解码图  ┌──────┐
│ .bin     │───────────▶│  1024 维  │─────────▶│ PCM  │
│ 模型     │  12 个码本  │  特征    │  7 层 DAC │ 音频 │
└──────────┘            └──────────┘           └──────┘
```

**解码器图**：RVQ 码本 → Conv1d(1024→1536) → 4× 残差块
(1536→768→384→192→96) → Snake → Conv1d(96→2) → PCM

## CLI 参考

```
tsac-ng [选项] c|d|t 输入文件 输出文件

选项（兼容原始 tsac）：
  --cuda, --hip, --vulkan, --llvm   GPU/加速器后端
  -q, --n_codebooks n    码本数（立体声 1-12，单声道 1-9，默认=最大值）
  -T n                   线程数（默认=1）
  -v                     详细模式
  -h, --help             显示帮助
  -s, --separate_channels  立体声分离为双单声道
  -c, --channels n       强制声道数
  -f, --fast             快速模式（无 Transformer）
  -m, --model 路径        模型文件路径
  -M, --trf_model 路径    Transformer 模型路径
  --batch_size n         批大小（默认=自动）
```

## 交叉编译

```bash
# ARM64（Termux、树莓派 5 等）
cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-arm64.cmake

# RISC-V（实验性）
cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-riscv64.cmake
```

## 项目结构

```
tsac-ng/
├── src/
│   ├── cpu_decoder.c      # CPU 解码器 + x86 SIMD 调度
│   ├── tsac_codec.c       # 编解码 API + WAV I/O
│   ├── txc_format.c       # .txc 容器解析器
│   ├── model_loader.c     # .bin 模型加载器（BF8/float32）
│   ├── main.c             # CLI（兼容原始 tsac）
│   ├── cuda/              # CUDA 后端
│   ├── llvm/              # LLVM JIT 后端（实验性）
│   ├── vulkan/            # Vulkan 计算后端
│   ├── arch/arm/          # ARM NEON + SVE
│   └── arch/riscv/        # RISC-V RVV
├── hip/                   # HIP/ROCm 后端
├── include/               # 公共头文件
├── cmake/                 # 工具链文件
└── experimental/          # 实验性代码
```

## 兼容性

- `.txc` 容器格式（原始 TSAC）
- `.bin` 模型文件（DAC 立体声/单声道, q8）
- 原始 `tsac` CLI 参数

## 许可证

MIT

---

```
tsac-ng v0.1.0 — Copyright (c) 2026 Hope2333（幽零小喵）
```
