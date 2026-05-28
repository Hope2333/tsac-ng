# TSAC-ng Roadmap

## Current Phase: ROUND_054_COMPLETE

## Completed (Rounds 001–036)
- ✅ Full DAC decoder architecture (32 conv1d/29 snake/4 convtr)
- ✅ 10-bit bitpacking decoder — 54/54 indices verified against GDB ground truth
- ✅ CRC32 fully reversed and implemented
- ✅ Fast TXC format: 10-bit fixed-width fields, bswap+shr, data@byte 8
- ✅ in_proj+out_proj dual projection RVQ lookup implemented
- ✅ get_freq adaptive range coder (15-bit probability)
- ✅ Normal TXC format documented (state→freq table→binary search→index)
- ✅ 5 CPU SIMD backends (AVX/AVX2/AVX-512, NEON/SVE, RVV)
- ✅ 3 GPU backends (CUDA, HIP, Vulkan)
- ✅ Code quality baseline: 69.55/100

## Blocked
- **BF8 dequant RMS fix**: libnc nc_convert_from_old_bf fused op differs from our separate steps; direct comparison blocked by libnc's thread-heavy initialization

## Next Milestones

### v0.2.0 — RMS Fix + Normal TXC (~Round 037–050)
- [ ] Fix BF8 dequant to match libnc fused operation (mathematical approach or GDB memory dump)
- [ ] Implement normal TXC range coder state→frequency table initialization
- [ ] Implement binary search index decoder (get_freq adaptive)
- [ ] Implement Transformer autoregressive decoder (12-layer, d_model=512, n_head=4, RoPE)

### v0.3.0 — Production Quality (~Round 051–070)
- [ ] Full RMS −0.1dB fidelity
- [ ] CPU encoder with strided convs
- [ ] Code quality ≥85/100
- [ ] Comprehensive test suite


## Phase 3 (R126-R155): Production Release

See [ROADMAP_PHASE3.md](ROADMAP_PHASE3.md) for detailed 30-round plan.

**Milestones:**
1. Normal TXC end-to-end decode (R126-R135, 45 tasks)
2. Fast TXC bit-accuracy via BF8 stride fix (R136-R145, 45 tasks)
3. Production readiness: encoder, GPU, quality 87+ (R146-R155, 44 tasks)

**Target**: v0.3.0 with fast TXC corr > 0.95, normal TXC decode, GPU backends.
