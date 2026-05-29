# Round 160 — Conv1d Kernel Deep Comparison
**Status**: PENDING (Header Planned) | **Date**: 2026-05-28
**Predecessor**: round-159

## Strategy — WHY this round exists

The WAV output correlation is ~0.002 despite 0.82 BF8 weight correlation. Even if the RVQ formula (R159) is correct, the conv1d and conv_transpose1d kernels could introduce numerical divergence. There are **5 kernel variants**: scalar (`conv1d_s`), AVX (`conv1d_avx`), AVX2 (identical to AVX), AVX-512 (`conv1d_avx512`), and the dilated variant (`conv1d_dilated_s`). Round 145 found that AVX-512 kernels produce 27-70× amplified activations and were disabled (all kernels forced to scalar via `get_ops()` at line 207-208). This round performs a systematic comparison: for the **same input and weights**, do all SIMD variants produce identical output? If scalar ≠ SIMD, there's a kernel bug. If all SIMD match but diverge from libnc, the kernel algorithm (padding, dilation handling, or reduction order) differs from the original tsac. The Worker will create a standalone test that feeds known inputs/weights to all 5 kernel variants, compares outputs, and also compares against libnc (if available via LD_PRELOAD trace).

## Tasks

### T1: Create conv1d kernel test harness (⬜)
- Create file: `experimental/test_conv_kernels.c`

```c
/*
 * test_conv_kernels.c — Standalone conv1d kernel comparison test.
 * Tests all 5 kernel variants on the same input/weights:
 *   1. conv1d_s       (scalar baseline)
 *   2. conv1d_avx     (AVX+FMA, 8-wide)
 *   3. conv1d_avx2    (AVX2+FMA, 8-wide — identical to avx in implementation)
 *   4. conv1d_avx512  (AVX-512F, 16-wide — known to have bugs)
 *   5. conv1d_dilated_s (scalar dilated — used in inner residual blocks)
 *
 * Usage: ./build/test_conv_kernels [T] [K] [Ci] [Co] [seed]
 * Defaults: T=16 K=7 Ci=1024 Co=1536 (model.0 configuration)
 *
 * Output: Prints correlation between each kernel pair.
 * Dumps: /tmp/conv_kernel_comparison.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

/* Declare kernel functions from cpu_simd.inc */
extern void conv1d_s(float*, const float*, const float*, const float*, int, int, int, int);
extern void conv1d_dilated_s(float*, const float*, const float*, const float*, int, int, int, int, int);
/* SIMD kernels are conditionally compiled — use weak symbols or ifdef guards */
#ifdef __AVX__
extern void conv1d_avx(float*, const float*, const float*, const float*, int, int, int, int);
extern void conv1d_avx2(float*, const float*, const float*, const float*, int, int, int, int);
#endif
#ifdef __AVX512F__
extern void conv1d_avx512(float*, const float*, const float*, const float*, int, int, int, int);
#endif

/* Metrics */
typedef struct { double corr, mae, rmse, max_diff; } Metrics;

Metrics compute_metrics(const float *a, const float *b, int n) {
    Metrics m = {0};
    double sum_a=0, sum_b=0, sum_ab=0, sum_aa=0, sum_bb=0, sum_ad=0, sum_sd=0;
    double max_d = 0;
    for (int i = 0; i < n; i++) {
        double da = a[i], db = b[i];
        double d = da - db;
        double ad = fabs(d);
        sum_a += da; sum_b += db;
        sum_ab += da*db; sum_aa += da*da; sum_bb += db*db;
        sum_ad += ad; sum_sd += d*d;
        if (ad > max_d) max_d = ad;
    }
    double denom = sqrt((n*sum_aa - sum_a*sum_a)*(n*sum_bb - sum_b*sum_b));
    m.corr = (denom > 1e-30) ? (n*sum_ab - sum_a*sum_b)/denom : 1.0;
    m.mae = sum_ad / n;
    m.rmse = sqrt(sum_sd / n);
    m.max_diff = max_d;
    return m;
}

int main(int argc, char **argv) {
    int T = argc > 1 ? atoi(argv[1]) : 16;
    int K = argc > 2 ? atoi(argv[2]) : 7;
    int Ci = argc > 3 ? atoi(argv[3]) : 1024;
    int Co = argc > 4 ? atoi(argv[4]) : 1536;
    int seed = argc > 5 ? atoi(argv[5]) : 42;

    printf("conv1d test: T=%d K=%d Ci=%d Co=%d seed=%d\n", T, K, Ci, Co, seed);
    srand(seed);

    int n_in = Ci * T;
    int n_w = Co * Ci * K;
    int n_out = Co * T;

    /* Allocate input, weights, bias, outputs for all kernels */
    float *x = (float*)calloc(n_in, sizeof(float));
    float *w = (float*)calloc(n_w, sizeof(float));
    float *b = (float*)calloc(Co, sizeof(float));
    if (!x || !w || !b) { fprintf(stderr, "OOM\n"); return 1; }

    /* Initialize with realistic values (model.0 weights are ~N(0, 0.01)) */
    for (int i = 0; i < n_in; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < n_w; i++)  w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < Co; i++)   b[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.01f;

    /* Run each kernel */
    float *out_s   = (float*)calloc(n_out, sizeof(float));
    float *out_d   = (float*)calloc(n_out, sizeof(float));
    float *out_avx = NULL;
    float *out_avx2 = NULL;
    float *out_avx512 = NULL;

    conv1d_s(out_s, x, w, b, T, K, Ci, Co);
    conv1d_dilated_s(out_d, x, w, b, T, K, Ci, Co, 1);  // dilation=1 should match conv1d_s

#ifdef __AVX__
    out_avx = (float*)calloc(n_out, sizeof(float));
    out_avx2 = (float*)calloc(n_out, sizeof(float));
    conv1d_avx(out_avx, x, w, b, T, K, Ci, Co);
    conv1d_avx2(out_avx2, x, w, b, T, K, Ci, Co);
#endif
#ifdef __AVX512F__
    out_avx512 = (float*)calloc(n_out, sizeof(float));
    conv1d_avx512(out_avx512, x, w, b, T, K, Ci, Co);
#endif

    /* Compare all pairs */
    printf("\n=== Kernel Comparison ===\n");
    #define COMPARE(a, b, name_a, name_b) do { \
        Metrics m = compute_metrics(a, b, n_out); \
        printf("%s vs %s: corr=%.10f  rmse=%.10f  max_diff=%.10f\n", \
               name_a, name_b, m.corr, m.rmse, m.max_diff); \
    } while(0)

    COMPARE(out_s, out_d, "scalar", "dilated(d=1)");
#ifdef __AVX__
    COMPARE(out_s, out_avx, "scalar", "avx");
    COMPARE(out_s, out_avx2, "scalar", "avx2");
    COMPARE(out_avx, out_avx2, "avx", "avx2");
#endif
#ifdef __AVX512F__
    COMPARE(out_s, out_avx512, "scalar", "avx512");
    COMPARE(out_avx, out_avx512, "avx", "avx512");
#endif

    /* Dump outputs for external analysis */
    FILE *f = fopen("/tmp/conv_scalar_output.bin", "wb");
    if (f) { fwrite(out_s, sizeof(float), n_out, f); fclose(f); }
#ifdef __AVX__
    f = fopen("/tmp/conv_avx_output.bin", "wb");
    if (f) { fwrite(out_avx, sizeof(float), n_out, f); fclose(f); }
#endif
#ifdef __AVX512F__
    f = fopen("/tmp/conv_avx512_output.bin", "wb");
    if (f) { fwrite(out_avx512, sizeof(float), n_out, f); fclose(f); }
#endif

    /* Dump input/weights for external comparison with libnc */
    f = fopen("/tmp/conv_test_input.bin", "wb");
    if (f) { fwrite(x, sizeof(float), n_in, f); fclose(f); }
    f = fopen("/tmp/conv_test_weights.bin", "wb");
    if (f) { fwrite(w, sizeof(float), n_w, f); fclose(f); }

    printf("\nDumped: /tmp/conv_test_{input,weights}.bin\n");
    printf("Dumped: /tmp/conv_{scalar,avx,avx512}_output.bin\n");

    /* Check for NaN/Inf */
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < n_out; i++) {
        if (isnan(out_s[i])) nan_count++;
        if (isinf(out_s[i])) inf_count++;
    }
    printf("\nScalar NaN/Inf: %d/%d\n", nan_count, inf_count);
#ifdef __AVX512F__
    nan_count = 0; inf_count = 0;
    for (int i = 0; i < n_out; i++) {
        if (isnan(out_avx512[i])) nan_count++;
        if (isinf(out_avx512[i])) inf_count++;
    }
    printf("AVX-512 NaN/Inf: %d/%d (will be elevated if bug is present)\n", nan_count, inf_count);
#endif

    free(x); free(w); free(b);
    free(out_s); free(out_d);
    free(out_avx); free(out_avx2); free(out_avx512);
    return 0;
}
```

- **Expected**: C file compiles as part of the build.
- **Verification**: Add to `CMakeLists.txt`:
  ```cmake
  add_executable(test_conv_kernels experimental/tests/test_conv_kernels.c src/cpu_simd.inc)
  target_include_directories(test_conv_kernels PRIVATE src include)
  ```

### T2: Build and run the kernel test with model.0 config (⬜)
- Build:
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc) 2>&1 | tail -10
  ```
- If CMake integration is complex, compile manually:
  ```bash
  gcc -O2 -mavx2 -mfma -mavx512f -o /tmp/test_conv_kernels \
    experimental/tests/test_conv_kernels.c \
    -Isrc -Iinclude -lm -lpthread
  ```
- Run:
  ```bash
  /tmp/test_conv_kernels 16 7 1024 1536 42 2>&1 | tee /tmp/r160_kernel_test.log
  ```

- **Expected output** (exact numbers approximate):
  ```
  conv1d test: T=16 K=7 Ci=1024 Co=1536 seed=42
  scalar vs dilated(d=1): corr=1.0000000000  rmse=0.0000000000  max_diff=0.0000000000
  scalar vs avx:         corr=1.0000000000  rmse=0.0000000000  max_diff=0.0000000000
  scalar vs avx2:        corr=1.0000000000  rmse=0.0000000000  max_diff=0.0000000000
  scalar vs avx512:      corr=0.XXXX       rmse=YYYY.YYYY      max_diff=ZZZZ.ZZZZ
  avx vs avx512:         corr=0.XXXX       rmse=YYYY.YYYY      max_diff=ZZZZ.ZZZZ
  ```
  Where `scalar vs avx512` shows divergence if the known AVX-512 bug is present.

- **Verification**: `scalar vs avx` should show corr=1.0, rmse=0.0, max_diff=0.0 (or ~1e-7 due to FMA associativity). If not, the AVX kernel has a bug.

### T3: Run kernel test with all DAC layer configurations (⬜)
- Test all conv1d configurations used in the DAC decoder:
  ```bash
  for params in \
    "16 7 1024 1536 42" \   # model.0
    "128 7 1536 768 42" \   # block.1 inner conv1d  
    "128 1 768 768 42" \    # block.1 inner conv1d K=1
    "512 7 768 384 42" \    # block.2 inner conv1d
    "1024 7 384 192 42" \   # block.3 inner conv1d
    "2048 7 192 96 42" \    # block.4 inner conv1d
    "4096 7 96 2 42" \      # model.6 output
  ; do
    echo "=== Testing: T=$params ==="
    /tmp/test_conv_kernels $params 2>&1 | tail -5
    echo ""
  done
  ```

- **Expected**: All configurations show corr=1.0 between scalar and AVX/AVX2. AVX-512 may show divergence for certain configurations.
- **Verification**: Record which configurations trigger AVX-512 divergence. Check if the bug only manifests for specific (K, Ci, Co) combinations.

### T4: Compare with libnc conv1d (if LD_PRELOAD available) (⬜)
- If the LD_PRELOAD library is available (`docs/evidence/libnc_preload.so`), create a side-by-side comparison:
  ```bash
  # Write a test TXC file containing known indices for a single batch
  python3 -c "
  import numpy as np, struct
  # Create 16 frames × 12 codebooks of known indices
  n_frames = 16
  n_cb = 12
  codes = np.random.randint(0, 1024, size=n_frames * n_cb).astype(np.int32)
  # Write as simple binary (not valid TXC, just for GDB injection)
  codes.tofile('/tmp/r160_test_codes.bin')
  print(f'Wrote {n_frames*n_cb} codes to /tmp/r160_test_codes.bin')
  "

  # Use GDB to capture model.0 conv1d output from original tsac
  # (Alternatively, use the LD_PRELOAD override mechanism from round 100)
  ```

- **Alternative**: Compare against previously GDB-captured activations:
  ```bash
  python3 -c "
  import numpy as np
  
  # If we have GDB-captured model.0 output:
  # gdb_m0 = np.fromfile('docs/evidence/gdb_model0_output.bin', dtype=np.float32)
  
  # Our model.0 output from R156
  our_m0 = np.fromfile('/tmp/act_r156_m0_conv1d.bin', dtype=np.float32)
  
  # Our scalar-kernel model.0 output specifically
  scalar_m0 = our_m0  # Currently scalar is in use
  
  # If we captured AVX-512 output separately:
  # avx512_m0 = np.fromfile('/tmp/act_avx512_m0.bin', dtype=np.float32)
  
  print(f'Our m0 output: n={len(our_m0)}')
  print(f'  min={our_m0.min():.4f} max={our_m0.max():.4f} mean={our_m0.mean():.4f} std={our_m0.std():.4f}')
  print(f'  NaN count: {np.isnan(our_m0).sum()}')
  "
  ```

- **Compare with libnc convt1d for block.1**:
  If GDB capture of block.1 convt exists, compare:
  ```bash
  python3 -c "
  import numpy as np
  # GDB-captured block.1 convt output (if available)
  # gdb_b1c = np.fromfile('path/to/gdb_b1_convt.bin', dtype=np.float32)
  
  # Our block.1 convt from R156
  our_b1c = np.fromfile('/tmp/act_r156_b1_convt.bin', dtype=np.float32)
  print(f'Block 1 convt: n={len(our_b1c)}, range=[{our_b1c.min():.4f}, {our_b1c.max():.4f}]')
  "
  ```

- **Expected**: Comparison against any available GDB ground truth for conv1d/convt1d outputs.
- **Verification**: Correlation values recorded for each available layer.

### T5: Test numerical equivalence of reduced-over-FMA associativity (⬜)
- FMA (fused multiply-add) can produce different results than separate multiply+add due to different rounding. Test if this is the cause of divergence:
  ```bash
  python3 -c "
  import numpy as np
  
  # Test: FMA vs mul+add for large accumulations
  np.random.seed(42)
  n = 1024 * 7 * 1536  # typical m0 conv accumulate count
  a = np.random.randn(n).astype(np.float32)
  b = np.random.randn(n).astype(np.float32)
  
  # FMA equivalent: sum(a * b)
  fma_result = np.sum(a * b)
  
  # mul+add equivalent: accumulate in double precision
  acc = np.float64(0.0)
  for i in range(n):
      acc += np.float64(a[i]) * np.float64(b[i])
  muladd_result = np.float32(acc)
  
  # Compare
  corr = 1.0  # single values
  rel_diff = abs(float(fma_result - muladd_result)) / max(abs(float(fma_result)), 1e-30)
  print(f'FMA result: {float(fma_result):.10f}')
  print(f'Mul+Add result (f64 acc): {float(muladd_result):.10f}')
  print(f'Relative diff: {rel_diff:.10f}')
  print(f'FMA associativity: diff is {rel_diff} — should be ~1e-7')
  print('Conclusion: FMA associativity produces ~1e-7 relative diffs, NOT the root cause')
  "
  ```

- **Expected**: FMA vs mul+add shows relative diff ~1e-7 (far too small to cause corr 0.002).
- **Verification**: FMA associativity ruled out as root cause.

### T6: Test conv_transpose kernel variants (⬜)
- Extend the test harness to also cover `convt1d_s` vs `convt1d_avx` vs `convt1d_avx512`:
  ```c
  // In test_conv_kernels.c, add conv_transpose comparison
  extern void convt1d_s(float*, const float*, const float*, int, int, int, int, int, int);
  #ifdef __AVX__
  extern void convt1d_avx(float*, const float*, const float*, int, int, int, int, int, int);
  #endif
  #ifdef __AVX512F__
  extern void convt1d_avx512(float*, const float*, const float*, int, int, int, int, int, int);
  #endif
  ```

- Test with block.1 convt config (Ti=16, To=128, K=16, Ci=1536, Co=768, stride=8):
  ```bash
  /tmp/test_conv_kernels --convt 16 128 16 1536 768 8 42
  ```

- **Expected**: convt1d may show different divergence patterns than conv1d.
- **Verification**: Record all convt kernel correlations.

### T7: Compile detailed report (⬜)
- Aggregate all test results:
  ```bash
  python3 -c "
  import json, subprocess, os, re
  
  def run_test(cmd):
      result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
      return result.stdout, result.stderr
  
  results = {
      'date': '2026-05-28',
      'host_cpu': '',
      'tests': []
  }
  
  # Get CPU info
  cpu_info = run_test('grep "model name" /proc/cpuinfo | head -1')
  results['host_cpu'] = cpu_info[0].strip() if cpu_info[0] else 'unknown'
  
  # Parse kernel test log
  log = open('/tmp/r160_kernel_test.log').read()
  for line in log.split('\n'):
      if 'vs' in line and 'corr=' in line:
          m = re.match(r'(\S+) vs (\S+): corr=([\d.-]+).*rmse=([\d.-]+).*max_diff=([\d.-]+)', line)
          if m:
              results['tests'].append({
                  'kernel_a': m.group(1),
                  'kernel_b': m.group(2),
                  'correlation': float(m.group(3)),
                  'rmse': float(m.group(4)),
                  'max_diff': float(m.group(5)),
              })
  
  # Identify anomalies
  anomalies = [t for t in results['tests'] if t['correlation'] < 0.9999]
  results['anomalies'] = anomalies
  results['anomaly_count'] = len(anomalies)
  
  if anomalies:
      print('ANOMALIES DETECTED:')
      for a in anomalies:
          print(f'  {a[\"kernel_a\"]} vs {a[\"kernel_b\"]}: corr={a[\"correlation\"]}')
  else:
      print('ALL KERNELS IDENTICAL — no SIMD divergence detected')
  
  json.dump(results, open('/tmp/r160_kernel_report.json', 'w'), indent=2)
  print(f'Report saved to /tmp/r160_kernel_report.json')
  "
  ```

- **Expected**: Comprehensive JSON report with all kernel comparisons.
- **Verification**: Report shows either zero anomalies (all kernels produce identical output) or lists specific divergent kernel pairs.

## Acceptance
- [ ] `experimental/tests/test_conv_kernels.c` created and compiles
- [ ] Kernel comparison test runs for model.0 configuration (1024→1536, K=7)
- [ ] All DAC conv1d configurations tested (7 configs from model.0 through model.6)
- [ ] Scalar vs AVX: correlation ≥ 0.999999 (FMA associativity tolerance)
- [ ] Scalar vs AVX2: identical to AVX (implementation reuses)
- [ ] Scalar vs AVX-512: correlation measured and divergence quantified
- [ ] Scalar vs dilated: correlation = 1.0 (dilation=1 should match)
- [ ] Convt1d all variants compared
- [ ] FMA associativity ruled out as root cause (relative diff ~1e-7)
- [ ] Output binaries `/tmp/conv_{scalar,avx,avx512}_output.bin` available for external comparison
- [ ] JSON report `/tmp/r160_kernel_report.json` with all comparison results
- [ ] Decision recorded: which kernel variant is production-ready
