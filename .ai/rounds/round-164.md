# Round 164 — Align Snake/Tanh with libnc (Phase 4B)
**Signed**: Header | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
The `snake` activation function used in the DAC decoder has two known formulations in the literature:
1. **sin² formulation**: `snake(x) = x + sin²(α*x) / α` (used in original DAC paper)
2. **Sigmoid formulation**: `snake(x) = x + sigmoid(α*x) * sigmoid(-α*x) / α`

Our current implementation in `src/cpu_simd.inc` (`snake_s` at line 90, `snake_avx` at line 248, `snake_avx512` at line 432) uses:
```c
float sa = sinf(al * v);
o[i] = v + sa * sa / al;
```

Which is the sin² formulation. The question is whether libnc uses the same formulation or the sigmoid variant. R135-R145 GDB traces should have confirmed, but this was never explicitly verified.

**Additionally**: The final `tanhf` at line 61 of `src/cpu_tail.inc` may differ from libnc's tanh implementation. libnc might use a fast-approximation tanh (e.g., Pade approximant) rather than the C library's `tanhf`.

**Potential divergence sources**:
1. Snake: sin² vs sigmoid formulation
2. Snake: alpha clamping (`al < 1e-6f ? 1e-6f : al`) may differ
3. Snake: floating-point evaluation order (FMA vs sequential)
4. Tanh: `tanhf` vs libnc's custom fast-tanh
5. Tanh: clipping to [-1, 1] after tanh (lines 62-63) — libnc might not clip

## Round Strategy
1. GDB-capture snake output at model.5 (the final snake before model.6)
2. GDB-capture tanh output at model.6
3. Compare our snake output with libnc at identical input
4. Test sin² vs sigmoid formulation
5. Test tanhf vs fast-approximation alternatives
6. Align implementation with libnc

## Tasks

### T1: GDB capture snake output at model.5
**File**: `docs/evidence/gdb_m5_snake.bin` (new capture)

**Action**: Extend GDB capture to dump:
- model.5 input (= block.4 output): `(96, 144)` dims
- model.5 alpha tensor: `(96,)` dims
- model.5 output (= model.6 input): `(96, 144)` dims

GDB approach:
```gdb
# Break at nc_tensor or nc_inference to find model.5
# model.5 input should have dims (96, 144)
# model.5 output also has dims (96, 144) but should differ in value

# Set watchpoint on key activations
break nc_inference_tensor if *(long*)($rdi+0x10) == 96  # dims[0]=96
commands
  silent
  set $d1 = *(long*)($rdi + 0x18)
  if $d1 == 144
    set $data = *(long*)($rdi + 0x40)
    if $data > 0x100000
      set $tag = $_
      dump binary memory /tmp/gdb_m5_snake_$tag.bin $data $data+96*144*4
    end
  end
  continue
end
```

**Acceptance**: Model.5 input, alpha, and output captured as float32 dumps.

### T2: GDB capture tanh output at model.6
**File**: `docs/evidence/gdb_m6_tanh.bin` (new capture)

**Action**: Capture model.6 pre-tanh input and post-tanh output:
- model.6 input (pre-tanh): `(2, 144)` dims
- model.6 output (post-tanh/PCM): `(2, 144)` dims

The PCM output after tanh should have values in [-1, 1]. Compare our `tanhf` output with libnc's at identical input.

GDB approach:
```gdb
# The output PCM buffer after tanh
break nc_inference_tensor if *(long*)($rdi+0x10) == 2  # dims[0]=2
commands
  silent
  set $d1 = *(long*)($rdi + 0x18)
  if $d1 == 144
    set $data = *(long*)($rdi + 0x40)
    if $data > 0x100000
      dump binary memory /tmp/gdb_m6_output.bin $data $data+2*144*4
    end
  end
  continue
end
```

**Acceptance**: Model.6 pre-tanh and post-tanh outputs captured.

### T3: Compare our snake output with libnc at model.5
**File**: `experimental/test_snake_alignment.py` (new)

**Action**: Compare our snake implementation:
```python
import numpy as np

# Load libnc reference
x = np.fromfile('docs/evidence/gdb_block4_convt.bin', dtype=np.float32).flatten()
alpha = np.fromfile('docs/evidence/gdb_m5_alpha.bin', dtype=np.float32)
ref = np.fromfile('docs/evidence/gdb_m5_snake.bin', dtype=np.float32)

# Our snake_s implementation
def our_snake(x, alpha, C):
    o = np.zeros_like(x)
    for i in range(len(x)):
        al = alpha[i % C]
        if al < 1e-6:
            al = 1e-6
        sa = np.sin(al * x[i])
        o[i] = x[i] + sa * sa / al
    return o

our = our_snake(x.flatten(), alpha, len(alpha))
corr = np.corrcoef(our.flatten(), ref.flatten())[0, 1]
rmse = np.sqrt(np.mean((our - ref)**2))
print(f'Model.5 snake: corr={corr:.6f} rmse={rmse:.6f}')

# Test alternative formulation: sigmoid-based
def sigmoid_snake(x, alpha, C):
    o = np.zeros_like(x)
    for i in range(len(x)):
        al = alpha[i % C]
        if al < 1e-6:
            al = 1e-6
        sx = 1.0 / (1.0 + np.exp(-al * x[i]))  # sigmoid(al*x)
        snx = 1.0 - sx  # sigmoid(-al*x)
        o[i] = x[i] + sx * snx / al
    return o

our_sig = sigmoid_snake(x.flatten(), alpha, len(alpha))
corr_sig = np.corrcoef(our_sig.flatten(), ref.flatten())[0, 1]
print(f'Model.5 snake (sigmoid): corr={corr_sig:.6f}')

# Test without alpha clamping
def our_snake_noclamp(x, alpha, C):
    o = np.zeros_like(x)
    for i in range(len(x)):
        al = alpha[i % C]  # no clamping
        sa = np.sin(al * x[i])
        o[i] = x[i] + sa * sa / al
    return o

our_nc = our_snake_noclamp(x.flatten(), alpha, len(alpha))
corr_nc = np.corrcoef(our_nc.flatten(), ref.flatten())[0, 1]
print(f'Model.5 snake (no clamp): corr={corr_nc:.6f}')
```

**Acceptance**: Formulation match or mismatch identified quantitatively.

### T4: Test tanhf vs libnc's tanh
**File**: `experimental/test_tanh_alignment.py` (new)

**Action**: Compare `tanhf` against libnc's tanh:

```python
import numpy as np

# Load libnc pre-tanh and post-tanh
pre = np.fromfile('docs/evidence/gdb_m6_pre_tanh.bin', dtype=np.float32)
ref = np.fromfile('docs/evidence/gdb_m6_post_tanh.bin', dtype=np.float32)

# Our tanhf
our = np.tanh(pre)

corr = np.corrcoef(our.flatten(), ref.flatten())[0, 1]
rmse = np.sqrt(np.mean((our - ref)**2))
max_diff = np.max(np.abs(our - ref))
print(f'tanhf vs libnc: corr={corr:.6f} rmse={rmse:.6f} max_diff={max_diff:.6f}')

# Test Pade approximant for tanh (used in some DSP libs)
def tanh_pade(x):
    # [3/3] Pade approximant for tanh
    x2 = x * x
    return x * (1.0 + x2 / 9.0) / (1.0 + 4.0 * x2 / 9.0 + x2 * x2 / 63.0)

our_pade = tanh_pade(pre)
corr_p = np.corrcoef(our_pade.flatten(), ref.flatten())[0, 1]
print(f'tanh Pade vs libnc: corr={corr_p:.6f}')

# Check if libnc clips
clipped = np.clip(our, -1.0, 1.0)
print(f'After clipping: max_diff from ref={np.max(np.abs(clipped-ref)):.6f}')
print(f'Ref in [-1,1]? min={ref.min():.4f} max={ref.max():.4f}')
```

**Acceptance**: libnc's tanh behavior characterized (exact tanhf, approximant, or with/without clipping).

### T5: Fix snake/tanh implementation to match libnc
**File**: `src/cpu_simd.inc` (snake functions), `src/cpu_tail.inc` (tanh)

**Action**: Based on T3-T4 findings:
1. If snake formulation differs (sin² vs sigmoid): update `snake_s`, `snake_avx`, `snake_avx512` to match
2. If alpha clamping differs: update clamping threshold
3. If tanh differs: either use `tanhf` (if match confirmed) or add custom tanh implementation
4. If clipping differs: add/remove `[-1, 1]` clipping after tanh

**Example fix for snake formulation change**:
```c
// If libnc uses sigmoid formulation:
static inline float snake_elem(float v, float al) {
    if (al < 1e-6f) al = 1e-6f;
    float ex = expf(-al * v);
    float sig = 1.0f / (1.0f + ex);
    return v + sig * (1.0f - sig) / al;
}
```

**Acceptance**: Snake and tanh outputs match libnc with corr > 0.999.

## Acceptance Criteria
- [ ] GDB captures for model.5 snake (input + alpha + output)
- [ ] GDB captures for model.6 (pre-tanh + post-tanh)
- [ ] Snake formulation verified (sin² vs sigmoid vs other)
- [ ] Tanh implementation verified (tanhf vs approximant)
- [ ] Alpha clamping and clipping behavior documented
- [ ] Snake/tanh fixed to match libnc with corr > 0.999
- [ ] Final decoder output correlation improves accordingly
