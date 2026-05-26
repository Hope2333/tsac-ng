# Development Methodology — tsac-ng

**How one person + AI rebuilt a closed-source neural audio codec from scratch.**

---

## The Setup

- **Developer**: 1 human (Hope2333 / 幽零小喵)
- **AI Assistant**: Claude (Anthropic), accessed via OpenCode orchestration framework
- **Target**: TSAC — a closed-source neural audio codec by Fabrice Bellard (2024)
- **Binary**: Stripped x86-64 ELF, linked against libnc.so
- **Time**: ~51 investigation rounds over ~10 days (May 2026)

## The Workflow

### Phase 1: Extract Ground Truth (Human-Led)

The human does what AI cannot: interact with a running binary.

| Technique | Tool | Example Output |
|-----------|------|---------------|
| **Hardware watchpoints** | GDB `watch` | Located inline 10-bit decoder at 0x40578d |
| **PLT tracing** | LD_PRELOAD, objdump | Mapped 32 conv1d calls per batch |
| **Callgrind profiling** | valgrind | 174KB call graph for fast + normal modes |
| **Memory dumping** | GDB `dump memory` | Captured 54/54 codebook indices from original |
| **WAV comparison** | Custom C tool | Tracked RMS error per layer (-3.4dB → -13.85dB ref) |
| **Disassembly analysis** | objdump -d | CRC32 polynomial (0x04C11DB7), nc_conv_1d signature |
| **Weight interception** | LD_PRELOAD nc_conv_1d | Dumped 32 libnc float32 weight tensors |

### Phase 2: Specify the Format (Human-Led)

The human translates raw binary observations into structured specifications:

```
TXC Format (Fast):
  Bytes 0-3:   Magic "FBAZ"
  Bytes 4-5:   Version (BE uint16)
  Byte 6:      Flags
  Byte 7:      n_codebooks
  Bytes 8+:    10-bit bit-packed codebook indices
  Algorithm:   bswap(*(uint32_t*)(data+8+bp/8)) >> (22-(bp&7)) & 0x3FF
  Verification: 54/54 indices match GDB ground truth (R016)

DAC Graph:
  RVQ(12×codebook) → Conv1d(1024→1536,K=7) → 4× ResBlock(Snake→Convtr→3×ResUnit) → Conv1d(96→2,K=7) → tanh
  Verification: 32 nc_conv_1d calls per batch match callgrind (GDB confirmed)
```

### Phase 3: Generate Implementation (AI-Augmented)

The human writes prompts. The AI writes code.

**Example prompt for 10-bit decoder**:
> "Implement a function that reads a TXC byte buffer. The data starts at byte 8.
> For each frame × codebook, extract a 10-bit big-endian value using:
> `bswap(*(uint32_t*)(data+8+bp/8)) >> (22-(bp&7)) & 0x3FF`.
> The codebook count comes from the TXC header byte 7.
> Return an array of int indices. Handle buffer boundaries."

**Example prompt for AVX-512 conv1d kernel**:
> "Write a conv1d kernel using AVX-512 intrinsics. Signature:
> `void conv1d_avx512(float *out, const float *in, const float *w, const float *bias, int Ti, int K, int Ci, int Co)`.
> The weight layout is [Co, Ci, K]. Use _mm512_loadu_ps, _mm512_fmadd_ps.
> Prefer 16-wide vectorization over Co. Handle remainder with scalar fallback."

### Phase 4: Verify (Human-Led)

The human tests. If it fails, the human diagnoses and writes a better prompt.

| Check | Method | Example |
|-------|--------|---------|
| **Build** | `cmake --build build` | Must compile with 0 errors |
| **Indices** | Compare against GDB dump | 54/54 must match (R016) |
| **Audio RMS** | Compare WAV output against original | Target: 0.203 dBFS |
| **Weight values** | LD_PRELOAD dump vs our dequant | Per-element diff < 1e-7 |
| **Layer outputs** | RMS per conv1d call | Track divergence through DAC graph |
| **CRC32** | Verify against known test vectors | Must match 0x04C11DB7 |

### Phase 5: Iterate (Loop)

When verification fails → diagnose → fix prompt → regenerate → verify again.

51 rounds because:
- **13 rounds** (R001-013): Figuring out the TXC format (10-bit bitpacking was not obvious)
- **15 rounds** (R014-028): Chasing the RMS -3.4dB error (tried 6 different hypotheses)
- **8 rounds** (R029-036): RVQ lookup method (direct → weighted → in_proj+out_proj)
- **6 rounds** (R037-042): Cracking nc_conv_1d calling convention for weight comparison
- **6 rounds** (R043-048): Dumping all 32 weights, finding divergence reversal
- **3 rounds** (R049-051): Fixing is_ct false positive, first actual code fix

## Why AI Augmentation Works for This

### What AI does well (in this project)
- Generating SIMD intrinsics for 5 architectures (the patterns are well-documented)
- Writing CMake build systems (boilerplate with clear rules)
- Implementing standard DSP operations (conv1d, tanh, snake activation)
- Creating GPU kernel variants (CUDA → HIP → Vulkan porting is mechanical)
- Formatting documentation (structured markdown, tables, ASCII diagrams)

### What AI does poorly (in this project)
- Understanding that a stripped binary's dispatch path at 0x4044d0 is NOT fast-mode
- Realizing that `bias->dims[0] == d0` can be true for non-convtr tensors when Ci=Co
- Debugging why RMS is -3.4dB when all 54 indices match ground truth
- Figuring out that nc_conv_1d takes `(output, weight, input, ...)` not `(state, output, input, weight, ...)`
- Knowing when to stop investigating a dead end (Round 012: tautological test)

### The human's irreplaceable role
1. **Design the verification strategy** — what to test, how to test it, what constitutes "correct"
2. **Extract ground truth from the binary** — GDB, objdump, LD_PRELOAD are not promptable
3. **Judge AI output** — "this looks right but the RMS is wrong, so something is off"
4. **Decide what to do next** — when 6 hypotheses fail, pick the 7th based on evidence weight
5. **Write the prompts** — garbage in, garbage out; precision matters

## The Numbers

| Metric | Value |
|--------|-------|
| Investigation rounds | 51 |
| Git commits | 21 |
| Round docs | 51 × .ai/rounds/round-NNN.md |
| Lines of C code | ~4,500 (src/) |
| SIMD variants | 5 (AVX-512, AVX2, SSE, NEON, RVV) |
| GPU backends | 3 (CUDA, HIP, Vulkan) |
| GDB ground truth captures | 8 (indices, weights, RVQ output, conv1d outputs) |
| LD_PRELOAD intercepts | 4 (nc_find_param, nc_conv_1d, nc_conv_transpose_1d, nc_load_param) |
| Weeks of work | ~2 |

## Takeaway

You don't need a team to reverse-engineer a production neural codec anymore.
You need:
1. Deep knowledge of one domain (audio/signal processing, in this case)
2. Willingness to spend 50+ rounds in a tight verify→prompt→test loop
3. An AI assistant that can generate competent C, SIMD, and GPU code from precise specs
4. The humility to admit when you (or the AI) are wrong, and try again

The bottleneck is no longer "can I write this code?" — it's "do I know what correct looks like?"
