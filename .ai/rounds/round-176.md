# Round 176 — Quality Optimization Push (Worker Signed)
**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
Target quality score 87+ from current ~85.53 baseline. Focus on the two highest-CC files identified in R150 analysis: `model_loader.c` (CC=28) and `tsac_codec.c` (CC=56). Also fix error_handling issues in `range_coder.c` and `tsac_normal_decode.c`. The Phase 4 roadmap decision gate G4 (R166 WAV corr < 0.3 → R176 quality+accept) means this round may proceed regardless of WAV correlation status — quality is a standalone deliverable.

## Current Quality Baseline (from state.json)
| Metric | Current | Target |
|--------|:-------:|:------:|
| fuck-u-code score | 85.53 | 87+ |
| Previous best | 86.67 | 87+ |
| Cyclomatic Complexity | ~10% | <15% |
| Parameter Count | ~26% | <20% |
| Comment Ratio | ~29% | >30% |

## Tasks

### T1: Refactor model_loader.c — Extract helpers, reduce CC from 28 to <15
**Problem**: `model_loader_load()` is a long monolithic function (160 lines) with high cyclomatic complexity due to nested tensor-type detection, dimension parsing, and format dispatch.

**Actions**:
1. Extract tensor header parsing into `static bool parse_tensor_header(FILE *fp, TensorHeader *hdr)`
2. Extract BF8 vs float32 detection into `static bool is_bf8_quantized(const char *name, size_t data_size, int dims_product)`
3. Extract weight retrieval into `static Tensor *find_tensor(ModelState *state, const char *name_pattern)`
4. Keep `model_loader_load()` as orchestration only (≈60 lines)

**Exact commands**:
```bash
# After refactoring, verify build and CC
cmake --build build 2>&1 | tail -5
# Expected: "Build SUCCESSFUL" or "0 errors"

# Verify no functional change
git diff --stat src/model_loader.c
# Expected: "-160 +~160" lines (same length, restructured)

# Run quality tool
fuck-u-code_analyze src/model_loader.c 2>&1 | grep -E "CC=|complexity|score"
# Expected: CC < 15, complexity contribution < 5%
```

**Verification**: `model_loader_load()` function length < 80 lines. All three extracted functions have CC ≤ 5. Build passes with zero warnings. Existing test files decode identically (bit-exact output via `sha256sum` before and after).

### T2: Refactor tsac_codec.c — Reduce CC from 56 to <30
**Problem**: `tsac_codec.c` (149 lines) has high complexity due to `tsac_init()` which handles codec parameter validation, backend selection, and decoder/encoder initialization in one function.

**Actions**:
1. Extract backend validation into `static int validate_backend_params(TsacContext *ctx)`
2. Extract WAV I/O setup into `static int setup_wav_io(TsacContext *ctx, const char *outfile)`
3. Extract bitrate display into `static void display_bitrate(const TsacContext *ctx, int n_frames)`
4. Simplify `tsac_init()` to call extracted helpers sequentially

**Exact commands**:
```bash
# After refactoring
cmake --build build 2>&1 | tail -5

# Run full decode test to verify no regression
./build/tsac-ng -v d test-simples/short_fast.txc /tmp/test_r176.wav 2>&1
# Expected: decode succeeds, bitrate displayed, exit code 0

# Compare output with baseline (if prior WAV exists)
# sha256sum /tmp/test_r176.wav vs baseline
```

**Verification**: `tsac_init()` < 50 lines. Each extracted helper ≤ 20 lines. Build clean. Decode produces identical output (same sha256sum as before refactoring).

### T3: Fix error_handling in range_coder.c and tsac_normal_decode.c
**Problem**: Range coder functions lack proper error propagation — buffer underflow/overflow returns silently incorrect values instead of error codes. `normal_decode` similarly lacks input validation.

**Actions**:
**range_coder.c** (119 lines):
- Add `rc_valid()` check at entry of `rc_decode_cumul()`, `rc_decode_bits()`, `rc_decode_get_freq()`
- Return `-1` or set `rc->error` flag on underflow (total < n_syms, freq out of range)
- Add `TSAC_ERR_RANGE_CODER` error code to `tsac_codec.h`
- Ensure `rc_normalize()` detects buffer exhaustion and sets error flag

**tsac_normal_decode.c** (114 lines):
- Validate `softmax()` output sums to approx 1.0 (within 1e-3 tolerance)
- Validate cumulative frequency table is monotonic in `build_cum_freq()`
- Add null checks for `rc` and `state` parameters

**Exact commands**:
```bash
# After changes, create a corrupted TXC to test error handling
printf '\x46\x42\x41\x52\x00\x01\x81\x06' > /tmp/corrupt.txc
# 8-byte header only, no payload — should trigger range coder error

./build/tsac-ng -v d /tmp/corrupt.txc /tmp/out.wav 2>&1
# Expected: "Error: Range coder buffer underflow" (non-zero exit)

# Verify normal decode still works (fast TXC regression test)
./build/tsac-ng -v d test-simples/short_fast.txc /tmp/regress_test.wav
# Expected: exit 0, bitrate displayed

# Quality tool
fuck-u-code_analyze src/range_coder.c 2>&1 | grep -E "error_handling|score"
# Expected: error_handling score ≥ 0.9 (was ~1.2)
```

**Verification**: All range coder functions return proper error codes on invalid input. Corrupted TXC triggers descriptive error message (not segfault or silent wrong output). `build_cum_freq()` validates monotonicity. Error code `TSAC_ERR_RANGE_CODER` defined and returned.

### T4: Improve comment_ratio — Add documentation to key functions
**Problem**: Several key functions lack header comments explaining parameters, return values, and edge cases. Comment ratio is ~29% (target >30%).

**Target files and functions**:
- `model_loader.c`: Add doc comment to `model_loader_load()` explaining .bin format, tensor layout, file offset computation
- `tsac_codec.c`: Add doc comment to `tsac_init()` explaining backend dispatch logic and parameter constraints
- `range_coder.c`: Add doc comments to `rc_create()`, `rc_decode_cumul()`, `rc_decode_bits()` explaining range coder algorithm (15-bit frequency, normalization, carry-less variant)
- `tsac_normal_decode.c`: Add doc comment to `tsac_normal_decode_frame()` explaining the Transformer→softmax→range decode→DAC pipeline
- `cpu_decoder.c`: Add doc comment to `dequant_weights()` explaining BF8 format, gs=32 grouping, and layout detection (is_ct)

**Acceptance**: Each target function has a block comment specifying:
1. Purpose (1 sentence)
2. Parameters (name, type, meaning)
3. Return value (0=success, negative=error code)
4. Edge cases (null inputs, zero-length, buffer exhaustion)

**Exact commands**:
```bash
# Verify comment ratio improved
fuck-u-code_analyze src/model_loader.c 2>&1 | grep "comment_ratio"
# Expected: comment_ratio >= 0.30

# Aggregate score check
fuck-u-code_aggregate 2>&1
# Expected: overall score >= 87
```

### T5: Run fuck-u-code — Verify score 87+ across all source files
**Actions**:
1. Run `fuck-u-code` aggregate on entire `src/` directory
2. If score < 87, identify bottom-3 files and iterate T1-T4 fixes
3. Re-run until score ≥ 87 or marginal improvement exhausted
4. Document any remaining low-scoring files with rationale for not fixing

**Exact commands**:
```bash
# Full quality run
fuck-u-code 2>&1 | tee /tmp/quality_r176.txt
# Expected: Overall Quality Score: ≥ 87.00

# Score breakdown per file
grep -E "Overall|score|SCORE" /tmp/quality_r176.txt
# Expected: each major source file scores ≥ 70

# If score < 87, iterate:
# 1. Identify worst file: grep -E "score|%" /tmp/quality_r176.txt | sort -t: -k2 -n | head -3
# 2. Fix worst issues in bottom file
# 3. Re-run
```

**Acceptance criteria**:
- Overall fuck-u-code score ≥ 87.00
- No individual file scores < 60
- All sub-metrics ≥ 0.7 (CC, param_count, comment_ratio, error_handling)
- `cmake --build build` passes with 0 errors, 0 warnings
- Regression: fast TXC decode output is binary-identical to pre-refactor build

## Risk Assessment
| Risk | Impact | Mitigation |
|------|--------|------------|
| Refactoring introduces subtle bug in model loading | High — all models fail to load | Keep original functions alongside (guard with `#if 0`), run decode comparison test |
| Error handling changes break happy path | Medium — decode returns -1 incorrectly | Exhaustive test with valid + invalid inputs |
| Comment-only changes inflate score without substance | Low — score requires 30%+ ratio | Focus on meaningful documentation, not padding |
| 87+ target may be unachievable without invasive changes | Medium — architectural issues may limit score | Document remaining gaps with rationale; accept 86+ as close enough |

## Dependencies
- R150 quality baseline data (already captured)
- R166 decision gate G4 (if WAV corr < 0.3, quality round proceeds)
- No dependency on R156-R175 (can run in parallel with earlier sub-phases)

## Acceptance
- [ ] T1: model_loader.c CC reduced from 28 to <15, build passes
- [ ] T2: tsac_codec.c CC reduced from 56 to <30, build passes, decode identical
- [ ] T3: Error handling in range_coder.c and tsac_normal_decode.c returns proper error codes
- [ ] T4: Comment ratio ≥ 30% across all target files
- [ ] T5: fuck-u-code aggregate score ≥ 87.00
