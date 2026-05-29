# Round 160 — Conv1d Kernel Comparison (Scalar vs AVX2 vs AVX-512) (Phase 4B)
**Signed**: Header | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
Round 159 likely identified one or more kernel implementations with divergent output. This round systematically tests the conv1d kernel at three SIMD levels (scalar, AVX2, AVX-512) against identical input+weights, comparing each against the libnc GDB reference.

**Known bug** (from `src/cpu_decoder.c` lines 204-208):
> NOTE: convt1d_avx512 and conv1d_avx512 both have suspected FMA bugs producing 27× and 70× larger activations respectively. Using scalar kernels for correctness until SIMD bugs are fixed.

**Root cause identified in R154-R155 analysis**: `gather16()` at `src/cpu_simd.inc:141` uses a scalar temp array with `_mm512_loadu_ps(tmp)` instead of `_mm512_mask_i32gather_ps`. This means:
1. 16 individual scalar loads + store to temp → load from temp
2. Cache-line splitting + no hardware gather optimization
3. No masking for out-of-bounds channels

Additionally, `-ffast-math` FMA reassociation may produce different rounding than libnc's sequential FMA.

## Round Strategy
1. Build test harness that runs conv1d_s, conv1d_avx2, and conv1d_avx512 on identical input
2. Use R158's GDB-captured input+weights from libnc as ground truth
3. Compare each kernel's output against libnc reference output
4. Measure: are AVX kernels the same as scalar? Does either match libnc?
5. Determine if the fix is SIMD-level (AVX-512 implementation) or structural (access pattern)

## Tasks

### T1: Create `experimental/test_conv1d_kernels.py` — SIMD comparison harness
**File**: `experimental/test_conv1d_kernels.py` (new)

**Action**: Write Python script that:
1. Loads GDB reference input + weights from `docs/evidence/`
2. Generates C test file that includes all three conv1d implementations
3. Compiles with each SIMD target (`-mavx512f -mavx2 -mfma`)
4. Runs each kernel on identical data
5. Dumps outputs to `/tmp/conv1d_scalar.bin`, `/tmp/conv1d_avx2.bin`, `/tmp/conv1d_avx512.bin`
6. Compares all outputs against each other and against libnc reference

```python
def generate_c_test(T=9, K=7, Ci=1024, Co=1536):
    """Generate standalone C file testing all conv1d variants."""
    code = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

// Include the actual kernel implementations
// (copy relevant functions from cpu_simd.inc)

// Scalar reference
void conv1d_s(float *o, const float *x, const float *w, const float *b,
              int T, int K, int Ci, int Co) {
    int P = K/2;
    for (int oc = 0; oc < Co; oc++) {
        for (int oi = 0; oi < T; oi++) {
            float sum = b ? b[oc] : 0.0f;
            for (int ic = 0; ic < Ci; ic++) {
                for (int j = 0; j < K; j++) {
                    int ii = oi + j - P;
                    if (ii >= 0 && ii < T)
                        sum += x[ic*T+ii] * w[oc*Ci*K + ic*K + j];
                }
            }
            o[oc*T+oi] = sum;
        }
    }
}

// Run all kernels
int main(int argc, char **argv) {
    // Load input, weights, bias from binary files
    // Allocate outputs
    // Call each kernel
    // Dump outputs
    return 0;
}
'''
```

**Acceptance**: Test harness compiles and runs all three kernel variants.

### T2: Load GDB-captured reference data
**File**: `docs/evidence/gdb_model0_input.bin`, `docs/evidence/gdb_m0_weighs.bin` (need to capture)

**Note**: R158 only captured activations, NOT the weight tensors. We need separate GDB capture for model.0 weight_v/weight_g/bias.

**Action**: Extend GDB capture to also dump weights:
```gdb
break nc_load if $rdi != 0
commands
  silent
  set $name = *(long*)((char*)$rdi + 0x20)
  if $name != 0
    set $name_str = $name
    # Check if this is decoder.model.0.weight_v
  end
  continue
end
```

Alternative: Use existing `libnc_model0_f32.bin` from `docs/evidence/` (if available). Check:
```bash
ls -la docs/evidence/libnc_model0_f32.bin
python3 -c "import numpy as np; d=np.fromfile('docs/evidence/libnc_model0_f32.bin',dtype=np.float32); print(f'model0 weights: {d.shape} = {d.size*4} bytes')"
```

**Acceptance**: Model.0 input tensor, weight tensor (1024, 7, 1536), bias (1536), and output tensor (1536, 9) available as float32 binary dumps.

### T3: Run kernel comparison — scalar vs AVX2 vs AVX-512
**File**: `/tmp/conv1d_test_output/` — outputs from test harness

**Action**: Execute all three kernels, capture statistics:

```bash
./build/test_conv1d_kernels
python3 -c "
import numpy as np
for name in ['scalar', 'avx2', 'avx512']:
    d = np.fromfile(f'/tmp/conv1d_{name}.bin', dtype=np.float32)
    ref = np.fromfile('docs/evidence/gdb_m0_conv1d.bin', dtype=np.float32)
    n = min(len(d), len(ref))
    corr = np.corrcoef(d[:n], ref[:n])[0,1]
    rmse = np.sqrt(np.mean((d[:n]-ref[:n])**2))
    print(f'{name:10s}: corr={corr:.6f} rmse={rmse:.6f}')
    # Also compare avx2 vs scalar, avx512 vs scalar
"
```

**Expected results table**:
| Kernel | corr vs libnc | rmse vs libnc | max_abs | Status |
|--------|---------------|---------------|---------|--------|
| scalar | TBD | TBD | TBD | Baseline |
| AVX2 | TBD | TBD | TBD | Should match scalar |
| AVX-512 | TBD | TBD | TBD | Known 70× bug |

**Acceptance**: Quantitative comparison reveals which kernels match libnc.

### T4: FMA reassociation analysis with -ffast-math
**File**: `experimental/test_fma_reassoc.c` (new)

**Action**: Test whether `-ffast-math` causes FMA reassociation differences:
```c
// Test FMA associativity
float test_fma() {
    float a = 1.0f / 3.0f;
    float b = 2.0f / 7.0f;
    float c = 3.0f / 11.0f;
    float r1 = a*b + c;          // (a*b) + c
    float r2 = fmaf(a, b, c);    // fused multiply-add
    float r3 = c + a*b;          // c + (a*b) — different rounding
    return r1 + r2 + r3;         // force all paths
}
```

Compile with and without `-ffast-math`, compare binary output.

**Acceptance**: Document whether `-ffast-math` changes FMA output for our specific weight ranges.

### T5: Isolate the AVX-512 gather16 bug
**File**: `src/cpu_simd.inc` lines 141-150 (`gather16` function)

**Action**: Create minimal reproducer:
```c
#include <immintrin.h>
#include <stdio.h>

// Current buggy implementation
__m512 gather16_scalar(const float *x, int ic_base, int ii, int T, int Ci) {
    float tmp[16];
    for (int k = 0; k < 16 && (ic_base + k) < Ci; k++)
        tmp[k] = x[(ic_base + k) * T + ii];
    for (int k = (Ci - ic_base); k < 16; k++)
        tmp[k] = 0.0f;
    return _mm512_loadu_ps(tmp);
}

// Correct implementation using hardware gather
__m512 gather16_hw(const float *x, int ic_base, int ii, int T, int Ci) {
    __m512i indices = _mm512_set_epi32(
        ((ic_base+15) * T + ii) * 4,  // byte offsets
        ((ic_base+14) * T + ii) * 4,
        ...
    );
    __mmask16 mask = (ic_base + 16 <= Ci) ? 0xFFFF : ((1 << (Ci - ic_base)) - 1);
    return _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, indices, x, 4);
}
```

Compare outputs of both on real decoder data.

**Acceptance**: Bug root cause confirmed. `gather16` scalar temp array approach produces identical results to hardware gather (confirming the bug is elsewhere) OR hardware gather produces different (correct) results (confirming gather is the bug).

## Acceptance Criteria
- [ ] Test harness compiles and runs scalar, AVX2, AVX-512 conv1d
- [ ] GDB reference input+weights+output captured for model.0
- [ ] Quantitative comparison table: scalar vs AVX2 vs AVX-512 vs libnc
- [ ] FMA reassociation tested with/without `-ffast-math`
- [ ] `gather16` bug root cause confirmed or ruled out
- [ ] Clear go/no-go decision for R161 (fix AVX-512 kernel)
