# Round 165 — Full Re-Verification & Residual Documentation (Phase 4B)
**Signed**: Header | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
After R157-R164 fixes, perform comprehensive re-verification of ALL 32+4 decoder layers against libnc GDB reference. This is the Phase 4 closure round — measure per-layer improvement, identify any remaining residuals, and decide whether WAV correlation has improved from ~0 to acceptable.

**Starting state** (R156 closure):
| Metric | R156 Value | Target |
|--------|-----------|--------|
| RVQ RMS | 0.0644 | ~0.03-0.06 |
| m6_pre_tanh RMS | 1.81 | <3 |
| WAV clipping | 10.6% | 0% |
| WAV corr | ~0 | >0.95 |

**Expected end state** after all R157-R164 fixes:
| Metric | Target | Measurement |
|--------|--------|-------------|
| RVQ RMS | ~0.03-0.06 | TBD |
| m6_pre_tanh RMS | <3 | TBD |
| Per-layer corr (all) | >0.95 | TBD |
| WAV corr | >0.95 | TBD |
| WAV clipping | 0% | TBD |

## Round Strategy
1. Re-build with all fixes active and AVX-512 re-enabled
2. Run full decoder comparison for all 36 layers
3. Compute end-to-end WAV correlation against original tsac
4. Compare WAV file with bit-exact or near-bit-exact reference
5. Document remaining residuals and prioritize for Phase 5
6. Update project status and close Phase 4

## Tasks

### T1: Re-build with all fixes and re-run full comparison
**Files**: `src/cpu_decoder.c`, `src/cpu_simd.inc`, `src/cpu_blocks.inc`, `src/cpu_tail.inc`

**Action**: Clean build with all fixes:
```bash
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_AVX512=ON
make -j$(nproc)
```

Verify build configuration:
```bash
./build/tsac-ng --help
# Check: SIMD should show "AVX-512" for supported hardware
```

**Acceptance**: Clean build with all fixes, no warnings, all SIMD levels detected.

### T2: Per-layer correlation — full 36-layer comparison
**File**: `experimental/compare_activations.py`

**Action**: Run full comparison:
```bash
# Run decoder
./build/tsac-ng -v -f d /tmp/short_fast.txc /tmp/out_after_fixes.wav

# Compare all layers
python3 experimental/compare_activations.py

# Generate comprehensive report
```

**Produce final comparison table**:
```
Layer                corr(R156)  corr(after fixes)  RMSE(after)  improvement
rvq_out              0.9999      ?.????             ?.????        ?
m0_conv1d            ?.????      ?.????             ?.????        ?
block1_convt         ?.????      ?.????             ?.????        ?
block2_convt         ?.????      ?.????             ?.????        ?
block3_convt         ?.????      ?.????             ?.????        ?
block4_convt         ?.????      ?.????             ?.????        ?
m5_snake             ?.????      ?.????             ?.????        ?
m6_pre_tanh          ?.????      ?.????             ?.????        ?
```

**Acceptance**: Per-layer correlation table shows progress from R156 baseline.

### T3: End-to-end WAV correlation
**File**: `experimental/compare_wav.py` (new script)

**Action**: Compare decoder output WAV with original tsac:
```python
import numpy as np
import wave

def read_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

# Our output
ours = read_wav('/tmp/out_after_fixes.wav')

# Original tsac output
# Need to generate: /usr/lib/tsac/tsac -v -f d /tmp/short_fast.txc /tmp/libnc_output.wav
ref = read_wav('/tmp/libnc_output.wav')

# Trim to match (original may produce slightly different length)
n = min(len(ours), len(ref))
ours, ref = ours[:n], ref[:n]

# Metrics
corr = np.corrcoef(ours, ref)[0, 1]
rmse = np.sqrt(np.mean((ours - ref)**2))
snr = 20 * np.log10(np.sqrt(np.mean(ref**2)) / (rmse + 1e-30))
max_diff = np.max(np.abs(ours - ref))

print(f"=== WAV Comparison ===")
print(f"Correlation: {corr:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"SNR: {snr:.2f} dB")
print(f"Max sample diff: {max_diff:.6f}")
print(f"Clipping: ours={np.sum(np.abs(ours)>0.99)/len(ours)*100:.1f}% ref={np.sum(np.abs(ref)>0.99)/len(ref)*100:.1f}%")
```

**Acceptance**: WAV correlation measured. If < 0.95, residual analysis determines why.

### T4: Residual analysis — isolate remaining divergence
**File**: `experimental/residual_analysis.py` (new)

**Action**: If WAV corr < 0.95, perform structured decomposition:

```python
# 1. Scale-only residual: find α that minimizes |ours - α*ref|
from scipy.optimize import minimize_scalar
def scale_error(alpha):
    return np.sqrt(np.mean((ours - alpha * ref)**2))
res = minimize_scalar(scale_error, bounds=(0.5, 2.0), method='bounded')
alpha_opt = res.x
print(f"Optimal scale factor: {alpha_opt:.4f}")
print(f"RMSE after scale correction: {res.fun:.6f} (was {rmse:.6f})")
if res.fun < rmse * 0.5:
    print("→ Residual is primarily SCALE error (single global scale factor)")

# 2. Frequency-domain residual
from scipy import fft
our_spec = np.abs(fft.fft(ours))
ref_spec = np.abs(fft.fft(ref))
spec_corr = np.corrcoef(our_spec, ref_spec)[0, 1]
print(f"Spectral correlation: {spec_corr:.4f}")
if spec_corr < 0.9:
    print("→ Residual is frequency-dependent (not just scale)")

# 3. Per-channel residual
for ch in range(2):
    ch_ours = ours[ch::2]
    ch_ref = ref[ch::2]
    ch_corr = np.corrcoef(ch_ours, ch_ref)[0, 1]
    ch_rmse = np.sqrt(np.mean((ch_ours - ch_ref)**2))
    print(f"Channel {ch}: corr={ch_corr:.4f} rmse={ch_rmse:.6f}")
```

**Acceptance**: Residual categorized as: scale-only, frequency-dependent, channel-specific, or random.

### T5: Document all residuals and Phase 5 recommendations
**File**: `.ai/rounds/round-165.md` (this file — append findings)

**Action**: Create final Phase 4 closure document with:
1. Summary of all fixes applied (R157-R164)
2. Per-layer improvement table (R156 baseline vs post-fix)
3. WAV correlation result and interpretation
4. Residual error decomposition
5. Recommended Phase 5 work items prioritized:
   - P0: Any remaining WAV corr gap > 0.05
   - P1: Encoder parity
   - P2: Normal TXC decode
   - P3: GPU backend parity
   - P4: Cross-platform validation

**Template for findings section**:
```markdown
## Phase 4 Results
| Metric | R156 | Post-Fix | Target | Status |
|--------|------|----------|--------|--------|
| RVQ RMS | 0.0644 | ? | ~0.03-0.06 | ? |
| m6_pre_tanh RMS | 1.81 | ? | <3 | ? |
| WAV corr | ~0 | ? | >0.95 | ? |
| WAV clipping | 10.6% | ? | 0% | ? |
| Per-layer best corr | ? | ? | >0.99 | ? |
| Per-layer worst corr | ? | ? | >0.95 | ? |

## Residual Breakdown
- Scale error: ?%
- Frequency error: ?%
- Channel-specific: ?%
- Random/uncorrelated: ?%

## Phase 5 Recommendations
1. ...
2. ...
```

**Acceptance**: Phase 4 closure report complete with all metrics, residual breakdown, and Phase 5 plan.

## Acceptance Criteria
- [ ] Clean build with all R157-R164 fixes
- [ ] 36-layer per-layer correlation table with R156 baseline comparison
- [ ] WAV correlation measured (absolute and relative to R156)
- [ ] Residual error categorized by type (scale/frequency/channel)
- [ ] Phase 4 closure document with metrics and recommendations
- [ ] Decision: proceed to Phase 5 or iterate on remaining residuals
- [ ] `.ai/state.json` updated to reflect Phase 4 completion
