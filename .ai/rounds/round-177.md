# Round 177 — Memory + Performance Profiling

**Signed**: Header (Planned) | **Date**: 2026-05-29 | **Status**: PENDING

## Objective

Profile tsac-ng for memory usage and decode speed. Establish benchmarks vs original `tsac` binary.

## Test Setup

**Hardware**: x86-64 (current build host), 1 thread (`-T 1`)
**Model**: `/usr/share/tsac/dac_stereo_q8.bin`
**Test files** (use existing assets):

| File | Description | Size | Expected |
|------|-------------|------|----------|
| `/tmp/test_fast.txc` | Fast TXC, 30s stereo | ~240 KB | Fast decode |
| `/tmp/test_normal_q6.txc` | Normal TXC, 30s stereo, q6 | ~180 KB | Transformer decode |
| `/tmp/silent_1s_normal_q6.txc` | 1s silent, normal | ~6 KB | Quick test |
| `/tmp/music_1s_normal_q6.txc` | 1s music, normal | ~8 KB | Quick test |

**Build**: `build/` (CPU, Release mode)

## Task T1: Peak Memory — Fast TXC vs Normal TXC

**Objective**: Compare peak memory usage between fast and normal decode paths.

**Exact commands**:
```bash
# Build with memory tracking instrumentation
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Fast TXC: measure peak RSS via /usr/bin/time
echo "=== FAST TXC ==="
/usr/bin/time -v ./build/tsac-ng -v -T 1 d /tmp/test_fast.txc /tmp/fast_out.wav 2>&1 | grep -E 'Maximum resident|User time|System time|Elapsed|Minor page faults'

# Normal TXC: measure peak RSS
echo "=== NORMAL TXC ==="
/usr/bin/time -v ./build/tsac-ng -v -T 1 d /tmp/test_normal_q6.txc /tmp/normal_out.wav 2>&1 | grep -E 'Maximum resident|User time|System time|Elapsed|Minor page faults'

# Compare file sizes
echo "=== FILE SIZES ==="
ls -la /tmp/test_fast.txc /tmp/test_normal_q6.txc

# Output WAV sizes
ls -la /tmp/fast_out.wav /tmp/normal_out.wav
```

**Expected output format** (approximate):
```
Fast TXC:  Peak RSS = XX MB, User time = Y.YYs, Wall = Z.Zs
Normal TXC: Peak RSS = XX MB, User time = Y.YYs, Wall = Z.Zs
Fast vs Normal memory ratio: 1:X
```

**Success criteria**: tabulate in report format:
| Mode | Peak RSS (MB) | User Time (s) | Wall Time (s) | Output WAV Size |
|------|--------------|---------------|---------------|-----------------|
| Fast TXC | | | | |
| Normal TXC | | | | |

## Task T2: Speed vs Original tsac Binary

**Objective**: Compare decode speed of tsac-ng vs original `/usr/bin/tsac` on same input files.

**Original tsac must be installed at `/usr/bin/tsac`**.
If not available, build comparison from tsac-ng's own benchmarks.

**Exact commands**:
```bash
# Check original tsac availability
which tsac && tsac --version 2>&1 | head -3

# Original tsac fast decode
if [ -x /usr/bin/tsac ]; then
    echo "=== ORIGINAL TSAC FAST ==="
    /usr/bin/time -v /usr/bin/tsac -v -T 1 d /tmp/test_fast.txc /tmp/ref_fast.wav 2>&1 | grep -E 'Maximum resident|User time|System time|Elapsed'
fi

# Compare outputs (RMS difference)
echo "=== RMS DIFFERENCE ==="
python3 -c "
import struct, math, sys

def read_wav(path):
    with open(path, 'rb') as f:
        data = f.read()
    # Parse WAV: skip 44-byte header, read as float32
    pcm = struct.unpack('<' + 'f' * ((len(data) - 44) // 4), data[44:])
    return pcm

if len(sys.argv) >= 3:
    a = read_wav(sys.argv[1])
    b = read_wav(sys.argv[2])
    n = min(len(a), len(b))
    if n == 0: print('Empty WAV'); sys.exit(1)
    mse = sum((a[i] - b[i])**2 for i in range(n)) / n
    rms = math.sqrt(mse)
    corr_num = sum(a[i]*b[i] for i in range(n))
    corr_den = math.sqrt(sum(a[i]**2 for i in range(n)) * sum(b[i]**2 for i in range(n)))
    corr = corr_num / corr_den if corr_den > 0 else 0
    print(f'Samples compared: {n}')
    print(f'RMS difference: {rms:.6f}')
    print(f'RMS dB: {20*math.log10(rms+1e-10):.2f} dBFS')
    print(f'Correlation: {corr:.6f}')
" /tmp/ref_fast.wav /tmp/fast_out.wav
```

**Speed comparison table** (expected):
| Implementation | Wall Time (s) | User Time (s) | Peak RSS (MB) | Ratio |
|---------------|---------------|---------------|---------------|-------|
| tsac-ng (fast) | | | | 1.0x |
| tsac-ng (normal) | | | | Nx |
| Original tsac (fast) | | | | Nx |

**Note**: tsac-ng normal TXC decode has partial Transformer+range coder implementation.
If end-to-end does not yet produce audio, measure up to the point of decoder dispatch.

## Task T3: Top-3 Optimization Targets

**Objective**: Using profiling data, identify top-3 functions/locations for optimization.

**Commands for profiling**:

```bash
# Option A: perf (if available)
perf record -g ./build/tsac-ng -v -T 1 d /tmp/test_fast.txc /tmp/perf_out.wav 2>&1
perf report --stdio -g flat | head -50

# Option B: gprof
# Rebuild with -pg flag
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_FLAGS="-pg -O2"
cmake --build build -j$(nproc)
./build/tsac-ng -v -T 1 d /tmp/test_fast.txc /tmp/gprof_out.wav 2>&1
gprof ./build/tsac-ng gmon.out > /tmp/gprof_report.txt 2>&1
head -80 /tmp/gprof_report.txt

# Option C: valgrind callgrind
valgrind --tool=callgrind --callgrind-out-file=/tmp/callgrind.out ./build/tsac-ng -v -T 1 d /tmp/test_fast.txc /tmp/cg_out.wav 2>&1
# Analyze with callgrind_annotate
callgrind_annotate /tmp/callgrind.out --auto=yes --threshold=95 | head -40
```

**Expected top-3 hotspots** (based on code analysis):

| Rank | Function | File | Estimated % | Reason |
|------|----------|------|------------|--------|
| 1 | `dequant_weights` | cpu_decoder.c | ~35% | O(n³) amortized, called per-batch for all 32 conv layers |
| 2 | `conv1d_dilated_s` | cpu_simd.inc (inline) | ~20% | Scalar inner blocks (T=6656/T=26624), no SIMD for dilated variant |
| 3 | `decode_batch` | cpu_decoder.c | ~15% | Orchestration overhead: 7-layer DAC loop, per-layer dispatch |

**Validate/update this list with actual profiling data.**

## Task T4: Optimization Recommendation Report

For each of the top-3 hotspots, produce:

```
### Hotspot #1: [function name]
- **File**: src/cpu_decoder.c:XXX
- **% Time**: XX%
- **Current cost**: XX calls, XX ms total
- **Root cause**: [specific code issue]
- **Recommendation**: [specific fix]
- **Estimated speedup**: Nx
- **Risk**: Low/Medium/High
- **Prerequisites**: [e.g., GDB confirmation of weight layout]
```

## Deliverables

1. **`/tmp/r177_memory_report.md`** — Memory profiling data (T1+T2)
2. **`/tmp/r177_perf_report.txt`** — Perf/callgrind raw output (T3)
3. **`/tmp/r177_optimization_targets.md`** — Top-3 recommendation report (T4)
4. **Update to `state.json`** — Add profiling baseline data
5. **Append to `decision.log`** — R177 decisions and findings

## Success Criteria

- [ ] R177-T1: Fast TXC peak RSS measured and documented
- [ ] R177-T1: Normal TXC peak RSS measured and documented  
- [ ] R177-T1: Memory comparison table complete
- [ ] R177-T2: Speed comparison table against original tsac (or noted if unavailable)
- [ ] R177-T3: Top-3 optimization targets identified with profiling evidence
- [ ] R177-T4: Each hotspot has actionable recommendation
- [ ] Profiling raw data saved to `/tmp/r177_*`
- [ ] No regressions in decode quality (RMS/corr stable)

## Risks

| Risk | Mitigation |
|------|------------|
| Original tsac not installed | Note in report, proceed with self-comparison only |
| valgrind/perf not available | Use gprof or built-in timing (`/usr/bin/time`) |
| Profiling huge datasets too slow | Use 5s/30s test files, not full-length audio |
