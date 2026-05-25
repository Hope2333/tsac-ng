# HANDOFF: Round 005 → Round 006

**Created**: 2026-05-23T08:50:00+08:00
**By**: Hephaestus

## Round 005 Achievements

### Decoder ✅
- Architecture matches DAC (Snake→dilated Conv7→Snake→Conv1→residual)
- Self-consistent: deterministic, different entries→different output, NaN=0
- CUDA backend synced (dilation, full ResidualUnit, block.2/block.3 weights)
- Weight dequant correct: BF8 group scale + weight_norm L2
- ConvTranspose stride K/2, output Ti*stride, tanh final

### Encoder 🔧
- Tensor naming fixed: `encoder.block.X` (was `encoder.model.X`)
- Channel counts fixed: 64→128→256→512→1024 (was 96→192→384→768→1536)
- is_ct detection fixed: bias->dims[0]==d0 (was K-based, failed for K=3/4/8/16)
- Architecture limitation: stride-1 convs can't encode 320-sample blocks
- Full temporal encoder designed but computationally infeasible without strides

### TXC Parser 📦
- Fast-mode (version=0, flags&0x80) uint8 path implemented
- Normal-mode correctly fail-fasts
- Discovery: original fast TXC uses non-raw index encoding (format unknown)

### .ai Framework 🤖
- Created `.ai/AGENTS.md` with self-driving workflow instructions
- Hephaestus behavior pattern documented in COT-020
- Comprehensive RE database in `.ai/memories/reverse-engineering.md`
- 28 tasks tracked, 15 ISS items, 9 backends

## Open Blockers

| # | Blocker | Impact | Next Step |
|---|---------|--------|-----------|
| 1 | Encoder needs strided conv1d | Cannot encode audio | Implement conv1d_strided kernel |
| 2 | Fast TXC format unknown | Cannot compare decoders | Reverse original binary's fread/fopen |
| 3 | Normal .txc needs Transformer | Cannot read original files | Implement GPT-NeoX + arithmetic coder |

## Round 006 Priority Actions

1. **Strided conv1d** — add stride parameter to conv1d_s, let encoder progressively reduce 320→1
2. **Fast TXC format RE** — use GDB on fopen/fread to find exact payload offset
3. **Decoder comparison** — once format is known, feed same indices to both decoders
4. **HIP decoder sync** — port full ResidualUnit from CUDA

## Quick Start
```bash
cd /home/miao/Projects/tsac-ng
cat .ai/AGENTS.md          # Self-driving workflow
cat .ai/state.json         # Current state
cmake --build build         # CPU build
cmake --build build_cuda_hip  # CUDA+HIP build
```
