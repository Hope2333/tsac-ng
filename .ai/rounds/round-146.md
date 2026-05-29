# Round 146 — Encoder Strided Convolution Implementation

**Status**: PENDING (Header Planned) | **Date**: 2026-05-28
**Predecessor**: round-145 (M2 sign-off) | **Priority**: HIGH

## Strategy
Complete the encoder by implementing strided convolutions (K=4/8/16). Current encoder uses regular conv1d which doesn't downsample — strided convs are needed for proper temporal encoding.

## Tasks

### T1: Research encoder architecture
- Analyze original tsac encoder via GDB: identify which layers use stride > 1
- Map encoder tensor dims from tsac_stereo_q8.bin
- Identify strided conv patterns (K=4/8/16, stride=K/2)

### T2: Implement strided conv1d kernel
- Add stride parameter to conv1d_parallel / SIMD kernels
- Handle downsampling: output has fewer frames (n_frames / stride)

### T3: Integrate with encoder framework
- Modify cpu_encoder.inc to use strided convs where appropriate
- Update encoder tensor loading (enc: prefixed tensors)

### T4: Test encoder→decoder round-trip
- Encode with our encoder, decode with our decoder
- Compare codebook indices with original tsac encoder
- Measure bitrate and audio quality

### T5: Document encoder status
CFOF

cat > .ai/rounds/round-147.md << 'CEOF'
# Round 147 — Encoder Full Pipeline + CLI Integration

**Status**: PENDING (Header Planned) | **Predecessor**: round-146

## Tasks
### T1: Add encode path to CLI (tsac-ng c input.wav output.txc)
### T2: Test fast TXC encode → our decode
### T3: Test fast TXC encode → original tsac decode
### T4: Measure encode quality (PESQ/STOI vs original tsac encoder)
### T5: Document encode status in README

## 🔬 Explore Agent Findings (bg_8498a938)
### Critical Issues Discovered
1. **CPU tensor naming mismatch**: Uses `encoder.block.X` but model has `encoder.model.X` — loads NO actual weights
2. **dequant_weights is_ct bug**: K=4/8/16 encoder convs mis-classified as convtranspose
3. **conv1d_strided_s deprecated**: Must migrate to conv1d_dilated_s
4. **No SIMD strided convs**: Scalar-only, no AVX/NEON variants

### GPU Encoder Issues
5. **CUDA/HIP: stride=1 throughout** — no temporal compression
6. **conv1d_strided_kernel exists but NOT called** — just needs wiring
