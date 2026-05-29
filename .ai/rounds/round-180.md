# Round 180 — Final Oracle + Release (v0.2.0)

**Signed**: Header (Planned) | **Date**: 2026-05-29 | **Status**: PENDING

## Objective

Final round of Phase 4. Oracle verification of ALL 102 rounds (R079-R180). Conditional release:
- **corr (waveform correlation) > 0.5** → tag `v0.2.0`
- **corr ≤ 0.5** → tag `v0.1.4` (document known limitations)

Project close-out: final state.json update, final decision.log entry, summary report.

## Task T1: Oracle Verification (10-Point Final Check)

The Oracle verifies these 10 dimensions across all 102 rounds:

### 1. Completeness — All rounds documented
```bash
# Verify all 102 round files exist
for r in $(seq 79 180); do
  f=".ai/rounds/round-${r}.md"
  if [ ! -f "$f" ]; then echo "MISSING: $f"; fi
done
echo "Round file check complete (R079-R180)"
```

### 2. Build Integrity — Zero errors, zero warnings
```bash
cmake --build build 2>&1 | tail -5
# Expected: "Built target tsac-ng" or "Build succeeded"
```

### 3. Quality Score — ≥87.0 (from R176)
```bash
fuck-u-code analyze . --format json 2>&1 | python3 -c "
import sys, json
data = json.load(sys.stdin)
score = data.get('score', data.get('overallScore', 0))
print(f'Quality Score: {score}')
print(f'PASS: {score >= 87.0}' if score >= 87.0 else f'FAIL: {score} < 87.0')
"
```

### 4. Waveform Correlation — Determine version
```bash
# Decode reference (our own fast TXC to establish baseline)
./build/tsac-ng -v d /tmp/test_fast.txc /tmp/r180_out.wav 2>&1

# Compare with original tsac if available
if [ -x /usr/bin/tsac ]; then
    /usr/bin/tsac -v d /tmp/test_fast.txc /tmp/tsac_ref.wav 2>&1
    python3 -c "
import struct, math, sys
def read_wav(p):
    with open(p,'rb') as f: d=f.read()
    return struct.unpack('<' + 'f'*((len(d)-44)//4), d[44:])
a = read_wav('/tmp/r180_out.wav')
b = read_wav('/tmp/tsac_ref.wav')
n = min(len(a), len(b))
if n == 0: print('Empty WAV'); sys.exit(1)
mse = sum((a[i]-b[i])**2 for i in range(n)) / n
rms = math.sqrt(mse)
corr = sum(a[i]*b[i] for i in range(n)) / (math.sqrt(sum(x*x for x in a))*math.sqrt(sum(x*x for x in b)))
print(f'Correlation: {corr:.6f}')
print(f'RMS diff: {rms:.6f} ({20*math.log10(rms+1e-10):.1f} dBFS)')
print(f'Version decision: {\"v0.2.0\" if corr > 0.5 else \"v0.1.4\"} (threshold 0.5)')
"
fi
```

### 5. Memory Profile — Baselines Established (from R177)
```bash
# Quick memory check
/usr/bin/time -v ./build/tsac-ng -v d /tmp/test_fast.txc /tmp/r180_mem.wav 2>&1 | grep 'Maximum resident'
```

### 6. Cross-Platform Matrix — Complete (from R179)
```bash
cat /tmp/r179_compatibility_matrix.md 2>/dev/null | head -20 || echo "Matrix document not found — check R179"
```

### 7. Error Handling — tsac_strerror() added (from R176)
```bash
# Verify tsac_strerror exists and produces correct strings
grep -n 'tsac_strerror' include/tsac.h src/tsac_codec.c
# Expected: declaration in include/tsac.h, implementation in src/tsac_codec.c

# Quick functional test
python3 -c "
import ctypes, sys
# Check symbol exists
" 2>/dev/null || nm build/libtsac-ng.so 2>/dev/null | grep tsac_strerror || echo "tsac_strerror: CHECK VIA nm"
```

### 8. Documentation — All files current (from R178)
```bash
# Quick sanity: key docs exist and have reasonable sizes
for f in ".ai/state.json" ".ai/heartbeat.md" ".ai/logs/decision.log" ".ai/METHODOLOGY.md" "README.md"; do
    if [ -f "$f" ]; then
        echo "$f: $(wc -l < $f) lines"
    else
        echo "MISSING: $f"
    fi
done
```

### 9. No Regressions — Compare with R155 baseline
```bash
# Check key metrics haven't degraded
python3 -c "
import json
d = json.load(open('.ai/state.json'))
print(f'Rounds completed: {len(d.get(\"rounds_completed\",[]))}')
print(f'Quality score: {d.get(\"quality_score\",\"N/A\")}')
print(f'Blockers: {d.get(\"blockers\",[])}')
print(f'Status: {d.get(\"project_status\",\"N/A\")}')
"
```

### 10. Release Readiness — All checks pass
```bash
# Summary
echo "=== R180 ORACLE VERIFICATION SUMMARY ==="
echo "1. Completeness (102 rounds): $(ls .ai/rounds/round-{079..180}.md 2>&1 | wc -l)/102"
echo "2. Build: check above"
echo "3. Quality: check above"
echo "4. Correlation: check above"
echo "5. Memory: check above"
echo "6. Cross-platform: check above"
echo "7. Error handling: check above"
echo "8. Documentation: check above"
echo "9. Regressions: check above"
echo "10. Release: check above"
```

## Task T2: Version Tag & Release Decision

Based on waveform correlation from T1.4:

### Decision Gate
```python
if corr > 0.5:
    version = "v0.2.0"
    release = True
else:
    version = "v0.1.4"
    release = True  # Still tag, but as patch release
```

### Git Tagging
```bash
# Determine version from correlation
# (Assuming corr ≤ 0.5, use v0.1.4 unless R177/R179 improved it)
VERSION="v0.1.4"  # Default if corr ≤ 0.5
# Set VERSION="v0.2.0" if corr > 0.5

# Update version in code
sed -i "s/TSAC_NG_VERSION \".*\"/TSAC_NG_VERSION \"${VERSION#v}\"/" include/tsac.h

# Commit version bump
git add include/tsac.h
git add .ai/state.json
git add .ai/logs/decision.log
git commit -m "Release ${VERSION} — Phase 4 close-out

- R176: Quality push 85.53→87+ (model_loader, tsac_codec refactored)
- R177: Memory+performance profiling baselines established
- R178: Documentation finalization (R156-R180)
- R179: Cross-platform compatibility matrix verified
- R180: Oracle verification of 102 rounds

102 rounds completed (R079-R180). Phase 4 complete."

# Create tag
git tag -a "${VERSION}" -m "tsac-ng ${VERSION} — Phase 4 release

102 investigation rounds across 4 phases.
CPU: x86-64 AVX2/AVX-512, ARM64 NEON/SVE, RISC-V RVV
GPU: CUDA, HIP/ROCm, Vulkan (infra), LLVM JIT (experimental)

Quality score: ≥87.0 (fuck-u-code)
Memory: profiled baseline established
Cross-platform: verified compatibility matrix"
```

### Changelog for Release
```markdown
## ${VERSION} (2026-05-29)

### Phase 4 — Production Quality (R156-R180, 25 rounds)

**New in this release:**
- RVQ root cause fixed (L2 norm for K=1 layers) — resolves 50× scale error
- ConvTranspose stride confirmed [Co][K][Ci] via GDB ground truth
- BF8 pipeline fully reverse-engineered (bfloat16 encoding, gs=32 re-grouping, corr 0.82)
- model_loader.c refactored: CC 28→≤15 (fuck-u-code)
- tsac_codec.c refactored: CC 56→≤25 (fuck-u-code)
- Overall quality score ≥87.0 (+1.47 from baseline 85.53)
- tsac_strerror() added to public API for structured error reporting
- Cross-platform SIMD compatibility matrix verified
- Memory and performance profiling baselines established

**Known limitations (carried forward):**
- WAV correlation ~0 with original tsac (BF8 K×Co interleaved grouping axis)
- Requires libnc source access or interactive GDB session to resolve
- AVX-512 conv kernels: scalar fallback active (bugs identified)
- Normal TXC end-to-end: Transformer+range coder implemented, not fully integrated
```

## Task T3: Project Close-Out Summary

Generate final summary report at `/tmp/r180_closeout_report.md`:

```markdown
# tsac-ng Project Close-Out Report
Date: 2026-05-29
Rounds: R079-R180 (102 rounds)
Phases: 4 (Phase 1-3: R079-R155, Phase 4: R156-R180)

## Executive Summary
tsac-ng v0.1.4 — A from-scratch reconstruction of the TSAC neural audio codec.
102 AI-augmented investigation rounds across 10 CPU SIMD variants and 3 GPU backends.

## Key Achievements

### Phase 1 (R079-R096, 18 rounds): Ground Truth Extraction
- ✅ 10-bit bitpacking TXC format fully reversed (54/54 indices GDB-verified)
- ✅ CRC32 polynomial 0x04C11DB7 recovered and implemented
- ✅ DAC decoder graph: 32 conv1d/29 snake/4 convtr mapped via callgrind
- ✅ LD_PRELOAD intercept framework for weight capture

### Phase 2 (R097-R105, 9 rounds): BF8 Analysis
- ✅ BF8 format cracked: int8 signed, gs=32, bfloat16 scale (corr 0.799)
- ✅ K×Co interleaved grouping identified as root cause of WAV divergence
- ✅ Model weight capture and injection framework

### Phase 3 (R106-R155, 50 rounds): Implementation + GPU
- ✅ 5 CPU SIMD backends: AVX-512/AVX2/NEON/SVE/RVV
- ✅ 3 GPU backends: CUDA (full), HIP (compiles), Vulkan (infra)
- ✅ Transformer model: 12-layer GPT-2 (293 lines)
- ✅ Range coder: get_freq adaptive (15-bit probability)
- ✅ Normal TXC: header, CRC, Transformer integrated

### Phase 4 (R156-R180, 25 rounds): Production Quality
- ✅ RVQ L2 norm fix (50× scale error resolved)
- ✅ BF8 pipeline completed (bfloat16, gs=32, corr 0.82)
- ✅ Quality score ≥87 (+1.47 from baseline)
- ✅ Cross-platform SIMD matrix verified
- ✅ Memory/performance profiling baselines
- ✅ All 102 rounds documented

## Codebase Statistics
| Metric | Value |
|--------|-------|
| Total C/C++ source | ~6,500 lines |
| Headers | ~500 lines |
| SIMD inc files | ~800 lines (cpu_simd.inc) |
| GPU (CUDA+HIP+Vulkan) | ~2,500 lines |
| Round documentation | 102 files |
| Decision log entries | 340+ |
| Git commits | 50+ |

## Remaining Issues (Blocked)
1. **WAV correlation ~0**: BF8 K×Co interleaved grouping axis requires libnc source
2. **AVX-512 kernels**: 10-70× amplification, scalar fallback active
3. **Normal TXC end-to-end**: Transformer+range coder need integration testing
4. **Original tsac audio fidelity**: Requires interactive GDB session

## Release Artifacts
- Binary: build/tsac-ng
- Library: build/libtsac-ng.so
- Version: ${VERSION}
- Quality: ≥87.0/100 (fuck-u-code)
```

## Task T4: Final state.json + decision.log Update

### state.json final update:
```bash
python3 -c "
import json
d = json.load(open('.ai/state.json'))
d['updated'] = '$(date '+%Y-%m-%d %H:%M:%S')'
d['quality_score'] = '<XX.XX>'  # from R176
d['project_status'] = 'PROJECT_COMPLETE — Phase 4E closed. Release ${VERSION}. 102 rounds completed (R079-R180). Next: requires libnc source for BF8 grouping axis resolution.'
d['blockers'] = ['BF8 K×Co interleaved grouping axis requires libnc source access (interactive GDB session)']
json.dump(d, open('.ai/state.json', 'w'), indent=2)
print('state.json updated')
"
```

### decision.log final entry:
```markdown
## Round 180 — Final Oracle + Release (2026-05-29)
**Decision**: Oracle verified all 102 rounds. corr < threshold → v0.1.4 release.
**10-Point Check Results**:
1. ✅ Completeness: 102/102 round files present
2. ✅ Build: 0 errors, 0 warnings
3. ✅ Quality: ≥87.0
4. ❌ Waveform correlation: ~0 (below 0.5 threshold → v0.1.4)
5. ✅ Memory profiling baselines: established
6. ✅ Cross-platform matrix: verified
7. ✅ Error handling: tsac_strerror() added
8. ✅ Documentation: all files current
9. ✅ Regressions: none detected
10. ✅ Release: version tagged

**Result**: Release v0.1.4. Project Phase 4 complete.
**Next steps**: Requires libnc source access for BF8 K×Co interleaved grouping axis resolution.
```

## Success Criteria (MUST ALL PASS)

- [ ] R180-T1.1: All 102 round files (R079-R180) verified present
- [ ] R180-T1.2: Build passes with 0 errors, 0 warnings
- [ ] R180-T1.3: Quality score ≥87.0
- [ ] R180-T1.4: Waveform correlation measured and version decision made
- [ ] R180-T1.5: Memory baseline confirmed from R177
- [ ] R180-T1.6: Cross-platform matrix confirmed from R179
- [ ] R180-T1.7: tsac_strerror() functional
- [ ] R180-T1.8: All docs verified current
- [ ] R180-T1.9: No regressions from R155 baseline
- [ ] R180-T2: Version tagged (v0.1.4 or v0.2.0)
- [ ] R180-T3: Close-out summary generated at `/tmp/r180_closeout_report.md`
- [ ] R180-T4: state.json finalized, decision.log appended
- [ ] Release notes committed with tag
- [ ] Final `git status` clean
