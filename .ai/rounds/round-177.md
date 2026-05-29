# Round 177 — Memory + Performance Profiling (Worker Signed)
**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
Profile and document peak memory usage and decode speed for both fast TXC (DAC-only) and normal TXC (Transformer + DAC) decode paths. Compare against original tsac binary. Identify and rank top-3 optimization targets with estimated ROI. This round produces actionable optimization guidance for v0.2.0/v0.1.4 release.

## Key Questions
1. How much memory does each decode path peak at (heap + stack + mmap)?
2. How fast is tsac-ng vs original tsac on identical inputs?
3. What are the top-3 bottlenecks with highest performance ROI?

## Tasks

### T1: Profile peak memory for fast TXC decode (DAC only)
**Goal**: Measure peak RSS (Resident Set Size) and heap usage for decoding a fast TXC file through the DAC-only decoder path.

**Method**: Use `valgrind --tool=massif` for heap profiling and `/usr/bin/time -v` for peak RSS.

**Test files**:
- `test-simples/short_fast.txc` (9 frames, ~375 bytes) — minimal case
- `test-simples/music_5s_f_q6.txc` (5 seconds, ~30K frames) — typical case
- `test-simples/MOGRA_30s_fast.txc` (30 seconds, ~180K frames) — worst case

**Exact commands**:
```bash
# 1. Build with debug symbols and no optimization (for accurate valgrind traces)
mkdir -p build_profile && cd build_profile
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_NATIVE=OFF
make -j$(nproc) tsac-ng

# 2. Peak RSS via /usr/bin/time (Linux)
/usr/bin/time -v ./tsac-ng -v d test-simples/short_fast.txc /tmp/out_fast_short.wav 2>&1 | grep -E "Maximum resident|Elapsed"
/usr/bin/time -v ./tsac-ng -v d test-simples/music_5s_f_q6.txc /tmp/out_fast_5s.wav 2>&1 | grep -E "Maximum resident|Elapsed"

# 3. Heap profiling via massif (small file only, massif is slow)
valgrind --tool=massif --massif-out-file=/tmp/massif_fast_short.out ./tsac-ng -v d test-simples/short_fast.txc /tmp/out_fast_short_m.wav
# Analyze peak
ms_print /tmp/massif_fast_short.out | head -40

# 4. Also profile with massif for 5s file (may take longer, set timeout 300s)
valgrind --tool=massif --massif-out-file=/tmp/massif_fast_5s.out --time-unit=B ./tsac-ng -v d test-simples/music_5s_f_q6.txc /tmp/out_fast_5s_m.wav 2>&1
ms_print /tmp/massif_fast_5s.out | grep -E "peak|MB|snapshot" | head -10
```

**Data to capture** (fill this table):
| Test File | Frames | Peak RSS | Heap Peak | Stack Peak | Time |
|-----------|:------:|:--------:|:---------:|:----------:|:----:|
| short_fast.txc | 9 | | | | |
| music_5s_f_q6.txc | ~154K | | | | |
| MOGRA_30s_fast.txc | ~900K | | | | |

**Acceptance**: Peak RSS and heap for all 3 test files measured and recorded. Massif snapshot showing top allocation site identified.

### T2: Profile peak memory for normal TXC decode (Transformer + DAC)
**Goal**: Measure peak memory for normal TXC decode path which loads both Transformer (293 additional layers) and DAC decoder. The Transformer adds ~190 tensors of varying sizes (d_model=512, 12 layers, FFN 2048-dim).

**Method**: Same as T1 but with normal TXC input files.

**Test files**:
- `test-simples/silent_1s_normal_q6.txc` (1 second, ~378 bytes) — minimal case
- Create if not exists: encode 5s normal TXC from MOGRA slice

**Exact commands**:
```bash
# 1. Check if normal TXC test files exist
ls -la test-simples/*normal*.txc 2>/dev/null || echo "No normal TXC test files found"

# If no normal TXC files, encode one (requires Transformer model)
# ./build/tsac-ng -v c --model /usr/share/tsac/tsac_stereo_q8.bin \
#   --trf_model /usr/share/tsac/tsac_stereo_q8.bin \
#   test-simples/30s.wav /tmp/normal_30s.txc 2>&1

# 2. Profile RSS
/usr/bin/time -v ./build/tsac-ng -v d test-simples/silent_1s_normal_q6.txc /tmp/out_normal_1s.wav 2>&1 | grep -E "Maximum resident|Elapsed"

# 3. Heap profile (small file only due to valgrind overhead)
valgrind --tool=massif --massif-out-file=/tmp/massif_normal_1s.out ./build/tsac-ng -v d test-simples/silent_1s_normal_q6.txc /tmp/out_normal_1s_m.wav 2>&1
ms_print /tmp/massif_normal_1s.out | head -40
```

**Data to capture**:
| Test File | Frames | Peak RSS | Heap Peak | Transformer overhead | Time |
|-----------|:------:|:--------:|:---------:|:-------------------:|:----:|
| silent_1s_normal_q6.txc | ~29 | | | | |
| (5s normal TXC) | ~145 | | | | |

**Key comparison**: Normal TXC vs fast TXC memory delta = Transformer overhead.
Expected: Transformer adds ~2-5MB (190 tensors × d_model=512 × float32).

**Acceptance**: Memory delta between fast and normal TXC quantified. Transformer memory footprint isolated.

### T3: Profile decode speed vs original tsac
**Goal**: Compare tsac-ng decode speed against the original `tsac` binary on identical inputs. Measure wall-clock time, CPU time, and frames/second.

**Original tsac binary**: Located at `/usr/bin/tsac` or `/usr/local/bin/tsac`.

**Test protocol** (all runs ×3, report median):
```bash
# 1. Original tsac — fast decode
# Single-thread, no GPU
hyperfine --warmup 1 --runs 3 \
  '/usr/bin/tsac -v d test-simples/short_fast.txc /tmp/ref_fast_short.wav' \
  2>&1 | tee /tmp/perf_tsac_fast.txt

# 2. tsac-ng — fast decode (CPU scalar fallback, single-thread)
hyperfine --warmup 1 --runs 3 \
  './build/tsac-ng -v d -T 1 test-simples/short_fast.txc /tmp/ng_fast_short.wav' \
  2>&1 | tee /tmp/perf_ng_fast_scalar.txt

# 3. tsac-ng — fast decode (CPU auto-dispatch, single-thread)
hyperfine --warmup 1 --runs 3 \
  './build/tsac-ng -v d -T 1 test-simples/music_5s_f_q6.txc /tmp/ng_fast_5s.wav' \
  2>&1 | tee /tmp/perf_ng_fast_5s.txt

# 4. Original tsac — fast decode 5s
hyperfine --warmup 1 --runs 3 \
  '/usr/bin/tsac -v d test-simples/music_5s_f_q6.txc /tmp/ref_fast_5s.wav' \
  2>&1 | tee /tmp/perf_tsac_fast_5s.txt

# 5. Frames/second computation (from verbose output)
./build/tsac-ng -v d test-simples/short_fast.txc /tmp/ng_rate.wav 2>&1 | grep -oP '\d+/\d+' | tail -1
# Expected output format: "75/75" — frames decoded / total frames
```

**Data to capture**:
| Decoder | Input | Wall Time | CPU Time | Frames/s | vs tsac ratio |
|---------|:-----:|:---------:|:--------:|:--------:|:-------------:|
| original tsac | short_fast.txc | | | | 1.0× |
| tsac-ng scalar | short_fast.txc | | | | |
| tsac-ng auto | music_5s_f_q6.txc | | | | |
| original tsac | music_5s_f_q6.txc | | | | 1.0× |

**Acceptance**: Speed comparison table populated. Performance ratio (tsac-ng / original tsac) calculated per test case.

### T4: Identify and rank top-3 optimization targets
**Goal**: Using profiling data from T1-T3 plus static analysis, identify the three highest-ROI optimization targets with estimated effort and impact.

**Method**:
1. From massif output: identify top heap allocation sites
2. From `perf record`/`perf report`: identify hottest functions
3. From source analysis: identify algorithmic inefficiencies (e.g., O(n²) patterns, repeated allocations)
4. From callgrind (if available): identify call count hotspots

**Exact commands**:
```bash
# perf profiling (requires root or perf_event_paranoid adjustment)
perf record -g ./build/tsac-ng -v d test-simples/music_5s_f_q6.txc /tmp/perf_out.wav 2>&1
perf report --stdio -g flat | head -30

# Alternative: simple timing of individual decode phases by adding minimal instrumentation
# Use strace -c for syscall-level analysis
strace -c ./build/tsac-ng -v d test-simples/short_fast.txc /tmp/strace_out.wav 2>&1 | tail -15
# Expected: top syscalls by time (mmap, read, write, etc.)
```

**Top-3 candidate targets** (hypotheses to verify):
1. **Conv1d dilated inner blocks** (scalar only, no SIMD): `conv1d_dilated_s` at blocks 3 (T=6656) and 4 (T=26624) from R100 finding. Expected: 10-50× speedup with SIMD.
2. **dequant_weights L2 normalization**: O(n²) per-layer L2 norm computation for all 322 tensors at model load. Could cache or pre-compute.
3. **Range coder binary search in normal decode**: `rc_decode_cumul()` uses linear binary search over 1024-entry table. Could use O(1) alias table or direct lookup.

**Ranking template**:
| Rank | Target | File | Current Cost | Est. Speedup | Effort | ROI |
|:----:|--------|------|:------------:|:------------:|:------:|:---:|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

**Acceptance**: Top-3 targets identified with quantified current cost, estimated speedup, estimated effort, and ROI score. Each target has a clear optimization strategy (e.g., "SIMD-ize conv1d_dilated_s inner loop over T dimension using AVX2 gather").

## Data Collection Template

### Memory Profile Results Table
| Test File | Path | Frames | Peak RSS (KB) | Heap Peak (KB) | Transformers | Conv Only |
|-----------|------|:------:|:-------------:|:--------------:|:------------:|:---------:|
| short_fast | test-simples/short_fast.txc | 9 | | | N/A | |
| music_5s_f | test-simples/music_5s_f_q6.txc | ~154K | | | N/A | |
| MOGRA_30s_f | test-simples/MOGRA_30s_fast.txc | ~900K | | | N/A | |
| silent_1s_n | test-simples/silent_1s_normal_q6.txc | ~29 | | | | |
| (5s normal) | (to create) | ~145 | | | | |

### Speed Profile Results Table
| Backend | Input | Frames | Wall (ms) | CPU (ms) | FPS | vs tsac |
|---------|:-----:|:------:|:---------:|:--------:|:---:|:-------:|
| orig tsac | short_fast | 9 | | | | 1.00× |
| tsac-ng scalar | short_fast | 9 | | | | |
| tsac-ng auto | short_fast | 9 | | | | |
| orig tsac | music_5s_f | ~154K | | | | 1.00× |
| tsac-ng auto | music_5s_f | ~154K | | | | |

## Acceptance
- [ ] T1: Peak RSS and heap for fast TXC decode measured across 3 test files
- [ ] T2: Peak RSS and heap for normal TXC decode measured, Transformer overhead quantified
- [ ] T3: Speed comparison vs original tsac completed with 3-run median
- [ ] T4: Top-3 optimization targets identified, ranked by ROI, with specific strategies
