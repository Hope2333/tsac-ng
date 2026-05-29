# Round 176 — Quality Push (85.53→87+)

**Signed**: Header (Planned) | **Date**: 2026-05-29 | **Status**: PENDING

## Objective

Push `fuck-u-code` quality score from **85.53 → 87.0+** (target: ≥87.0). Primary targets:

1. **`src/model_loader.c`** — Cyclomatic complexity CC=28. Refactor to CC≤15.
2. **`src/tsac_codec.c`** — Cyclomatic complexity CC=56. Refactor to CC≤25.
3. **`error_handling`** — Add consistent error propagation across all public API functions.
4. **Run `fuck-u-code`** full analysis to verify score improvement.

## Baseline (Current State)

| Metric | Current | Target |
|--------|---------|--------|
| Overall Score | 85.53 | ≥87.0 |
| model_loader.c CC | 28 | ≤15 |
| tsac_codec.c CC | 56 | ≤25 |
| model_loader.c weight | ~28% | ≤15% |
| tsac_codec.c weight | ~26% | ≤15% |
| Parameter Count | 26.37% | <20% |
| Comment Ratio | 28.62% | >30% |

## Task T1: Refactor model_loader.c (CC=28→≤15)

**Current problems** (from reading source):

1. **`model_loader_load()`** (lines 18-149): Single monolithic 131-line function doing:
   - File I/O (fopen/fseek/ftell/read/fclose)
   - Header validation
   - Tensor offset scanning (realloc loop)
   - Per-tensor parsing (name, dims, data)
   - Weight_v elem_size detection (3 branches)
   - LibNC override path (80 lines nested inside tensor loop)
   - Error path inconsistencies (free paths differ per exit)

2. **LibNC override** (lines 108-143): Deeply nested inside `for` loop, uses `strstr` on every tensor, builds path, opens file, reads, replaces data.

**Refactoring plan**:

```
model_loader.c breakdown:
├── model_loader_load()          ← orchestrator (~30 lines)
│   ├── model_read_file()        ← new: file I/O, returns buf+size
│   ├── model_verify_header()    ← new: magic+type validation
│   ├── model_scan_tensors()     ← new: returns offset array
│   ├── model_parse_tensor()     ← new: parse single tensor header
│   ├── model_detect_elem_size() ← new: weight_v vs float32 logic
│   ├── model_try_libnc_ovr()    ← new: libnc override (separate file?)
│   └── model_free_buf()         ← cleanup helper
├── model_loader_free()          ← already simple, leave as-is
```

**Exact commands**:
```bash
# Before refactor: measure baseline complexity
fuck-u-code analyze src/model_loader.c --format json | grep cyclomatic

# Refactor: extract helper functions
# See detailed code changes below

# After refactor: verify CC reduction
fuck-u-code analyze src/model_loader.c --top 5
```

**Expected**: CC drops from 28 to ≤15. File weight drops from ~28% to ≤15%.

## Task T2: Refactor tsac_codec.c (CC=56→≤25)

**Current problems**:

1. **`tsac_init()`** (lines 25-91): 66-line function with:
   - 5 backend init branches (CUDA/HIP/Vulkan/LLVM/fallback) — each 6-10 lines, nearly identical
   - `#include "tsac_last.inc"` inside function body — code smell
   - Missing model load path (tsac_last.inc has model_loader_load but in wrong scope)
   - Function returns `TSAC_OK` at line 87 without reaching actual init complete path
   - Line 91 references undeclared `ret` — **potential UB/compiIation error**

2. **`tsac_decompress_file()`** (lines 95-139): 44-line function doing:
   - File I/O + TXC header parse + decompress + bitrate display
   - Overlapping responsibilities

**Refactoring plan**:

```
tsac_codec.c breakdown:
├── tsac_init()                 ← slim orchestrator (~20 lines)
│   ├── tsac_init_common()     ← new: alloc ctx, set defaults
│   ├── tsac_init_backend()    ← new: backend dispatch (switch)
│   └── tsac_load_model()      ← new: model path resolution + load
├── tsac_decompress_file()     ← slim orchestrator (~15 lines)
│   ├── tsac_read_file()       ← new: read file to buffer
│   ├── tsac_parse_header()    ← new: extract sample rate
│   └── tsac_write_wav()       ← new: WAV file writer (extracted from tsac_io.inc)
├── tsac_compress_file()       ← from tsac_last.inc
├── tsac_free()                ← from tsac_last.inc
└── tsac_free_buffer() / tsac_version() ← leave as-is
```

**Key fix**: The broken `tsac_init()` model load path must be corrected. Currently:
```c
// Line 76-88: THIS IS BROKEN
if (model_path) {
    // ... builds dac_path
    #include "tsac_last.inc"  // contains rest of tsac_init AND tsac_free AND tsac_compress_file
    // tsac_last.inc has } closing the if, then ctx->initialized=1; return ctx;
    // But then line 88-91 are dead code or UB
}
return ret;  // ret undeclared here!
```

Fix by extracting the include contents into proper function organization.

**Exact commands**:
```bash
# Before refactor: measure baseline
fuck-u-code analyze src/tsac_codec.c --format json | grep cyclomatic

# After refactor: verify
fuck-u-code analyze src/tsac_codec.c --top 5
```

**Expected**: CC drops from 56 to ≤25. File weight drops from ~26% to ≤15%.

## Task T3: Error Handling Fix

**Current state**: Error codes exist but error reporting is inconsistent.

**Problems to fix**:
1. `model_loader_load()` — Same error printed differently per path; some paths skip free(buf)
2. `tsac_init()` — Backend init warnings go to stderr, but no structured error return
3. `tsac_decompress_file()` — Returns TSAC_ERR_FILE for missing file but doesn't differentiate: file not found vs read error vs format error
4. Missing error string translation — `TSAC_ERR_BACKEND = -7` has no `tsac_strerror()` function

**Implementation**:
```c
// Add to include/tsac.h
const char *tsac_strerror(int err);

// Add to src/tsac_codec.c
const char *tsac_strerror(int err) {
    switch (err) {
        case TSAC_OK:           return "success";
        case TSAC_ERR_MEMORY:   return "out of memory";
        case TSAC_ERR_FILE:     return "file I/O error";
        case TSAC_ERR_FORMAT:   return "invalid format";
        case TSAC_ERR_MODEL:    return "model load failed";
        case TSAC_ERR_CODEC:    return "codec error";
        case TSAC_ERR_PARAM:    return "invalid parameter";
        case TSAC_ERR_BACKEND:  return "backend not available";
        case TSAC_ERR_INTERNAL: return "internal error";
        default:                return "unknown error";
    }
}
```

## Task T4: Run fuck-u-code Full Analysis

**Exact commands**:
```bash
# Full analysis with JSON output for parsing
fuck-u-code analyze . --format json > /tmp/fuc_before.json 2>&1

# Build verification
cmake --build build 2>&1 | tail -20

# After refactoring
cmake --build build 2>&1
fuck-u-code analyze . --format json > /tmp/fuc_after.json 2>&1

# Verify score improvement
python3 -c "
import json
with open('/tmp/fuc_after.json') as f:
    d = json.load(f)
print(f'Score: {d.get(\"score\", \"N/A\")}')
print(f'Files improved: {d.get(\"files_improved\", \"N/A\")}')
"
```

**Success criteria**:
- `fuck-u-code` overall score ≥87.0
- `cmake --build build` compiles with 0 errors, 0 warnings
- `model_loader.c` CC ≤15
- `tsac_codec.c` CC ≤25
- All existing functionality preserved (decode/encode still works)

## Risks

| Risk | Mitigation |
|------|------------|
| Refactor introduces functional regression | Test decode of `/tmp/test_*.txc` before/after |
| `.inc` file extraction breaks include chain | Re-verify all `#include` directives, check `tsac_wrap.inc` and `tsac_io.inc` |
| LibNC override path breaks | Test override: `touch /tmp/libnc_OVR_test.bin && ./tsac-ng -v d /tmp/test.txc /dev/null` |
| Score improves but build size increases | Check `strip` size: `strip build/tsac-ng && ls -la build/tsac-ng` |

## Verification Checklist

- [ ] R176-T1: model_loader.c CC ≤15 (measured via `fuck-u-code`)
- [ ] R176-T2: tsac_codec.c CC ≤25 (measured via `fuck-u-code`)
- [ ] R176-T3: tsac_strerror() added to public API
- [ ] R176-T3: All model_loader.c error paths cleaned (no missing free())
- [ ] R176-T3: tsac_init() model loading path functional (no UB on `ret`)
- [ ] R176-T4: Overall score ≥87.0
- [ ] R176-T4: `cmake --build build` passes (0 errors, 0 warnings)
- [ ] Regression: `./tsac-ng -v d /tmp/test_fast.txc /tmp/test_out.wav` produces valid WAV
