# Round 164 — Align Snake & Tanh with libnc Implementation (Header Dispatch)
**Signed**: Worker | **Date**: 2026-05-28 | **Status**: PENDING

## Summary
The DAC decoder uses two activation functions: `snake(x) = x + sin²(αx)/α` (29 snake layers throughout the decoder) and `tanh` (at model.6 output). The current `snake_s` implementation at `cpu_simd.inc:90-97` uses `sinf()` which is standard math library. However, the original tsac (libnc) may use a different snake formulation — possibly `snake(x) = x * sigmoid(x)` (also known as Swish/SiLU) or a polynomial approximation of `sin²`. Additionally, `tanhf` is used at the output but libnc may clip differently or use a tanh approximation. The model.6 output values directly affect PCM samples. According to the roadmap, "we need to compare tanh at model.6 output" to ensure the final PCM values are correct.

**Strategy**: GDB-capture the original tsac's snake and tanh outputs for 3 representative layers (model.0→snake region, model.5 snake, model.6→tanh). Compare our snake output element-by-element with libnc. If divergence > 1%, investigate libnc's snake formula (possible x*sigmoid(x) variant). Also check tanh behavior: does libnc use fast tanh approximation or standard `tanhf`? Align activation functions accordingly.

## Tasks

### T1: GDB-Capture libnc Snake Output for 3 Layers
- Use the LD_PRELOAD approach (see docs/evidence/libnc_preload.so) or GDB to capture snake outputs:
  1. **model.1.block.0.alpha** — first snake after model.0 conv1d (input: 1536-dim activations)
  2. **model.3.block.2.block.0.alpha** — inner snake in block 3 (input: 384-dim activations)
  3. **model.5.alpha** — final snake before model.6 conv1d (input: 96-dim activations)
- For each layer:
  1. Run original tsac on a short (5-frame) TXC file
  2. Set breakpoint after snake output (e.g., at the return of `nc_snake_1d`)
  3. Dump output buffer to file: `/tmp/libnc_snake_N_layer.bin`
  4. Also dump input buffer to isolate the snake transform itself
- **Alternative**: If GDB breakpoint at snake is difficult, capture both input and output of the entire block and compute snake as `output_block - non_snake_contributions`. Simpler: capture model.5 snake input and output directly since model.5 is just snake (no other computation).

### T2: Compare Our snake(x) = x + sin²(αx)/α with libnc
- Build a comparison program that:
  1. Loads libnc snake input and output from T1 captures
  2. Computes our expected snake output: `o[i] = x[i] + sin²(α[i%C]*x[i]) / α[i%C]`
  3. Compares element-by-element with libnc output
- **If correlation > 0.995**: Our snake formula matches libnc. Close task.
- **If correlation < 0.9 but functional form matches**: Investigate alpha parameter loading — are we reading `.alpha` tensor correctly? Check `dims` and `data` for the alpha tensor.
- **If correlation < 0.5**: libnc uses a different snake formula. Possible alternatives:
  - `snake(x) = x * sigmoid(αx)` — Swish/SiLU variant:
    ```c
    o[i] = x[i] / (1.0f + expf(-alpha[i%C] * x[i]));
    ```
  - `snake(x) = x * sigmoid(x)` — no alpha parameter:
    ```c
    o[i] = x[i] / (1.0f + expf(-x[i]));
    ```
  - `snake(x) = x + (sin(αx))²/α` with different constant for small-alpha guard:
    Current: `if (al < 1e-6f) al = 1e-6f;`
    Libnc may use: `if (al < 1e-10f) al = 1e-10f;` or `al = fmaxf(al, 1e-10f);`
  - Polynomial approximation (no sinf):
    ```c
    float sx = alpha * x[i];
    float sx2 = sx * sx;
    // sin²(x) ≈ x² - x⁴/3 + 2x⁶/45 - ...
    float sin2_approx = sx2 - sx2*sx2/3.0f + 2.0f*sx2*sx2*sx2/45.0f;
    o[i] = x[i] + sin2_approx / alpha;
    ```
- **Implementation to test if sigmoid variant is correct**:
  ```c
  // Test: snake(x) = x * sigmoid(x)  (Swish/SiLU, no alpha)
  void snake_sigmoid_s(float *o, const float *x, const float *a, int n, int C) {
      (void)a; (void)C;  // alpha is unused in this formulation
      for (int i = 0; i < n; i++) {
          float v = x[i];
          o[i] = v / (1.0f + expf(-v));  // x * sigmoid(x)
      }
  }
  
  // Test: snake(x) = x * sigmoid(alpha*x)  (learned Swish)
  void snake_swish_s(float *o, const float *x, const float *a, int n, int C) {
      for (int i = 0; i < n; i++) {
          float v = x[i], al = a[i % C];
          o[i] = v / (1.0f + expf(-al * v));  // x * sigmoid(alpha*x)
      }
  }
  ```

### T3: Compare tanh at model.6 Output with libnc
- Capture model.6 output (before tanh) and final PCM (after tanh) from original tsac via GDB:
  - Breakpoint after `nc_conv_1d` for model.6 weights → dump conv1d output
  - Breakpoint after `tanh` application → dump final PCM
- Compare our pipeline:
  1. Our model.6 conv1d output (before tanh)
  2. Our tanh application (`tanhf` at cpu_tail.inc:61)
- **Check for differences**:
  - `tanhf` vs `tanh` (C vs C99 — same on most platforms)
  - Does libnc apply tanh per-channel or per-element? (Our code per-element at line 61)
  - Does libnc clip before tanh? Our code has clipping at lines 62-63 (`if (val > 1.0f) val = 1.0f`)
  - Does libnc scale output before tanh? (Our code doesn't, but maybe libnc does normalization)
- **Comparison**:
  ```python
  # Python comparison script
  import numpy as np
  our_pre_tanh = np.fromfile('/tmp/act_m6_pre_tanh.bin', dtype=np.float32)
  libnc_pre_tanh = np.fromfile('/tmp/libnc_m6_pre_tanh.bin', dtype=np.float32)
  our_post_tanh = np.tanh(our_pre_tanh)
  libnc_post_tanh = libnc_pre_tanh  # already has tanh applied
  corr = np.corrcoef(our_post_tanh, libnc_post_tanh)[0,1]
  max_diff = np.max(np.abs(our_post_tanh - libnc_post_tanh))
  ```

### T4: Align Snake Formula with libnc Implementation
- Once the correct snake formula is determined (from T2):
  1. If it's the current `sin²/α` formula but with different parameters: update `snake_s`, `snake_avx`, `snake_avx512`, `snake_neon`, `snake_sve`, `snake_riscv`
  2. If it's `x*sigmoid(x)` or `x*sigmoid(αx)`: rewrite ALL snake kernels
  3. Update `snake_parallel` in `cpu_threads.inc` to call the new kernel
  4. Update HIP snake kernel at `hip/dac_decoder.hip.cpp:33-40`:
     ```c
     __global__ void snake_k(float *o, const float *x, const float *a,
                              int n, int C) {
         int i = blockIdx.x * BLK + threadIdx.x;
         if (i >= n) return;
         float v = x[i];
         float al = a[i % C];
         // Current: o[i] = v + __sinf(al*v)*__sinf(al*v) / fmaxf(al, 1e-6f);
         // Option A: o[i] = v / (1.0f + expf(-al * v));  // Swish variant
         // Option B: o[i] = v;  // No activation (if snake is just identity)
     }
     ```
- **Important**: The alpha tensor might not be used in the sigmoid formulation. If the `alpha` tensors exist in the model file but the activation doesn't use them, we'd need to handle this gracefully (pass NULL or ignore).
- **Verification**: After aligning snake, re-run entire decoder and compare:
  - Per-layer activation correlation with libnc (from R156-160 GDB captures)
  - Final WAV correlation with original tsac output

## Acceptance Criteria
1. libnc snake formula determined: either `sin²/α` confirmed or `sigmoid` variant identified
2. Our snake kernel matches libnc snake with per-element correlation > 0.995
3. model.6 tanh behavior aligned: tanhf used, no unnecessary clipping, output within [-1, 1]
4. All snake kernel variants updated: scalar, AVX, AVX2, AVX-512, NEON, SVE, RVV, HIP
5. After alignment, model.5 output correlation with libnc > 0.99
6. After alignment, model.6 final output (PCM) correlation with original tsac improved
7. Build clean, no new warnings
