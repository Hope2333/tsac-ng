# Round 157 — Compare DAC Layers 0–2 (Init Divergence Search)
**Status**: PENDING (Header Planned) | **Date**: 2026-05-28
**Predecessor**: round-156

## Strategy — WHY this round exists

The WAV output correlation is ~0.002, but individual BF8 weights show 0.82 correlation. This means the divergence compounds through the DAC decoder graph but originates at some specific layer. By instrumenting the first three decoder stages (model.0 conv1d → model.1 block: snake+convt+inner_blocks → model.2 block: snake+convt+inner_blocks) with activation dumps and comparing against GDB captures of the original tsac, we can identify the first layer where correlation drops below 0.95. R156 built the comparison framework; R157 executes it on layers 0-2. The Worker must systematically enable `DEBUG_DECODER=1`, rebuild, run on the same input, then run the comparison script, and report which layer is the first divergence point.

## Tasks

### T1: Instrument all layer-0 and layer-1 activations (⬜)
- Edit `src/cpu_decoder.c`:
  - At line 396 (RVQ loop exit), after `DUMP_ACT(rvq_out, 1024*ctx_frames, "rvq_out");`, add:
    ```c
    DUMP_ACT(rvq_out, 1024*ctx_frames, "r156_rvq_out");
    ```
  - At line 468, after `DUMP_ACT(buf0, m0_Co*ctx_frames, "m0_conv1d");`, add:
    ```c
    DUMP_ACT(buf0, m0_Co*ctx_frames, "r156_m0_conv1d");
    ```
- Ensure `DEBUG_DECODER` is set to `1` at line 42:
  ```c
  #define DEBUG_DECODER 1
  ```
- Verify: `grep -n "DEBUG_DECODER" src/cpu_decoder.c` shows `#define DEBUG_DECODER 1`

- **Expected**: After rebuild and decode, `/tmp/act_r156_rvq_out.bin` and `/tmp/act_r156_m0_conv1d.bin` exist with non-zero sizes.
- **Verification**: `ls -la /tmp/act_r156_*.bin`

### T2: Instrument block.1 and block.2 (layer-1 and layer-2) inner activations (⬜)
- In `src/cpu_blocks.inc`, add activation dumps inside the block loop:
  
  After the snake (line 21), add:
  ```c
  char sn_dump[64]; snprintf(sn_dump, sizeof(sn_dump), "r156_b%d_snake", block);
  DUMP_ACT(current, current_C*cur_frames, sn_dump);
  ```
  
  After the convt (line 59), add:
  ```c
  char ct_dump[64]; snprintf(ct_dump, sizeof(ct_dump), "r156_b%d_convt", block);
  DUMP_ACT(next_buf, target_C*n_frames_out, ct_dump);
  ```
  
  Inside inner block loop (after inner conv1d at line 110), add:
  ```c
  char in_dump[64]; snprintf(in_dump, sizeof(in_dump), "r156_b%d_inner%d_conv1d", block, inner);
  DUMP_ACT(conv_out, ic_Co*cur_frames, in_dump);
  ```
  
  After residual add (line 150), add:
  ```c
  char res_dump[64]; snprintf(res_dump, sizeof(res_dump), "r156_b%d_inner%d_after_add", block, inner);
  DUMP_ACT(current, current_C*cur_frames, res_dump);
  ```

- **Expected**: Files for blocks 1 and 2 with names like `r156_b1_snake.bin`, `r156_b1_convt.bin`, `r156_b1_inner2_conv1d.bin`, etc.
- **Verification**: `ls /tmp/r156_b*.bin | sort` shows 10+ files covering all sub-layers of blocks 1 and 2.

### T3: Rebuild with DEBUG_DECODER=1 and run decode (⬜)
- Build:
  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j$(nproc)
  ```
- Decode one short test file:
  ```bash
  ./build/tsac-ng -v -f d test-simples/short_fast.txc /tmp/r157_out.wav 2>&1 | tee /tmp/r157_build.log
  ```
- **Expected**: Build succeeds with no errors. Decoder runs to completion.
- **Verification**: `grep -c "error" /tmp/r157_build.log` = 0. Also `ls /tmp/act_r156_*.bin | wc -l` ≥ 12.

### T4: Run comparison framework on layers 0-2 captures (⬜)
- Run comparison script:
  ```bash
  python3 experimental/compare_activations.py --our-prefix /tmp/act_r156_ --output /tmp/r157_comparison.json
  ```
- If framework doesn't yet support `--our-prefix`, create a wrapper:
  ```bash
  python3 -c "
  import json, numpy as np
  # Load each /tmp/act_r156_*.bin and compute stats
  import glob, os
  files = sorted(glob.glob('/tmp/act_r156_*.bin'))
  results = {}
  for f in files:
      name = os.path.basename(f).replace('.bin','')
      data = np.fromfile(f, dtype=np.float32)
      results[name] = {
          'shape': list(data.shape),
          'size': len(data),
          'min': float(data.min()),
          'max': float(data.max()),
          'mean': float(data.mean()),
          'std': float(data.std()),
          'nan_count': int(np.isnan(data).sum()),
          'inf_count': int(np.isinf(data).sum()),
      }
  json.dump(results, open('/tmp/r157_activation_stats.json','w'), indent=2)
  print(f'Captured {len(files)} activations')
  "
  ```

- **Expected**: JSON report with stats for all captured layers.
- **Verification**: `python3 -c "import json; d=json.load(open('/tmp/r157_activation_stats.json')); [print(k,v['shape'],v['nan_count']) for k,v in d.items()]"`
- Check for NaN/Inf: **Zero NaN/Inf values expected** (any NaN indicates a bug in conv kernel or dequant).

### T5: Validate layer 0 (RVQ output) against GDB ground truth (⬜)
- Load GDB ground truth:
  ```bash
  python3 -c "
  import numpy as np
  gdb = np.fromfile('docs/evidence/gdb_model0_input_9216f32.bin', dtype=np.float32).reshape(1024, -1)
  print(f'GDB model0 input shape: {gdb.shape}')
  print(f'Min: {gdb.min():.4f} Max: {gdb.max():.4f} Mean: {gdb.mean():.4f} Std: {gdb.std():.4f}')
  "
  ```
- Load our RVQ output:
  ```bash
  python3 -c "
  import numpy as np
  ours = np.fromfile('/tmp/act_r156_rvq_out.bin', dtype=np.float32)
  print(f'Our RVQ shape: {ours.shape}')
  print(f'Min: {ours.min():.4f} Max: {ours.max():.4f} Mean: {ours.mean():.4f} Std: {ours.std():.4f}')
  "
  ```
- Compare shapes: GDB is `(1024, 9)` but our shape depends on `ctx_frames`. If mismatch, the context padding differs.
- If shapes match, compute correlation:
  ```bash
  python3 -c "
  import numpy as np
  gdb = np.fromfile('docs/evidence/gdb_model0_input_9216f32.bin', dtype=np.float32)
  ours = np.fromfile('/tmp/act_r156_rvq_out.bin', dtype=np.float32)
  if len(gdb) == len(ours):
      corr = np.corrcoef(gdb.flatten(), ours.flatten())[0,1]
      rmse = np.sqrt(np.mean((gdb-ours)**2))
      print(f'RVQ: corr={corr:.6f} rmse={rmse:.6f}')
      print(f'REPORT: rvq_correlation={corr}')
  else:
      print(f'Shape mismatch: GDB={gdb.shape} Ours={ours.shape}')
      # If context padding differs, slice to match
      min_len = min(len(gdb), len(ours))
      gdb = gdb[:min_len]
      ours = ours[:min_len]
      corr = np.corrcoef(gdb.flatten(), ours.flatten())[0,1]
      print(f'Trimmed RVQ: corr={corr:.6f} (len={min_len})')
  "
  ```

- **Expected**: RVQ correlation should be **high (≈0.95+)** — this is the simplest layer (codebook lookup + matmul).
- **Verification**: Record the correlation value. If < 0.95, file an immediate issue — the RVQ implementation has a bug.

### T6: Identify first divergence layer in 0-2 (⬜)
- Create a chain of comparisons from layer 0 through block 2's inner residual blocks:
  ```python
  # Save as /tmp/r157_divergence_check.py
  import numpy as np, json, sys
  
  layers = [
      "r156_rvq_out",
      "r156_m0_conv1d",
      "r156_b1_snake",
      "r156_b1_convt",
      "r156_b1_inner2_conv1d",
      "r156_b1_inner2_after_add",
      "r156_b1_inner3_conv1d",
      "r156_b1_inner3_after_add",
      "r156_b1_inner4_conv1d",
      "r156_b1_inner4_after_add",
      "r156_b2_snake",
      "r156_b2_convt",
      "r156_b2_inner2_conv1d",
      "r156_b2_inner2_after_add",
      "r156_b2_inner3_conv1d",
      "r156_b2_inner3_after_add",
      "r156_b2_inner4_conv1d",
      "r156_b2_inner4_after_add",
  ]
  
  results = {}
  first_div = None
  threshold = 0.5  # Could be 0.95 for GDB comparison, but we're comparing to self
  
  for layer in layers:
      path = f"/tmp/act_{layer}.bin"
      try:
          data = np.fromfile(path, dtype=np.float32)
          nan_count = int(np.isnan(data).sum())
          inf_count = int(np.isinf(data).sum())
          has_nan = nan_count > 0 or inf_count > 0
          rms = float(np.sqrt(np.mean(data**2)))
          max_abs = float(np.max(np.abs(data)))
          
          results[layer] = {
              "shape": list(data.shape),
              "n": len(data),
              "rms": rms,
              "max_abs": max_abs,
              "nan": nan_count,
              "inf": inf_count,
              "has_nan": has_nan,
          }
          
          if has_nan and first_div is None:
              first_div = f"{layer} (NaN/Inf detected)"
          elif rms > 1e6 and first_div is None:
              first_div = f"{layer} (RMS exploded to {rms:.2f})"
              
      except FileNotFoundError:
          results[layer] = {"error": f"File not found: {path}"}
  
  json.dump(results, open("/tmp/r157_divergence_report.json", "w"), indent=2)
  print(f"First divergence: {first_div or 'None detected in layers 0-2'}")
  if first_div:
      sys.exit(1)
  ```
- Run: `python3 /tmp/r157_divergence_check.py`
- **Expected**: Either "None detected" (no NaN, no RMS explosion) or a specific first-divergence layer identified.
- **Verification**: Exit code 0 = no divergence in R157 scope. Exit code 1 = divergence found, recorded in JSON.

### T7: Visualize activation magnitudes across layers 0-2 (⬜)
- Create magnitude bar chart:
  ```bash
  python3 -c "
  import json, matplotlib.pyplot as plt
  import numpy as np
  d = json.load(open('/tmp/r157_divergence_report.json'))
  names = [k for k in d if 'error' not in d[k]]
  rms_vals = [d[k].get('rms',0) for k in names]
  max_vals = [d[k].get('max_abs',0) for k in names]
  
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
  x = range(len(names))
  ax1.bar(x, rms_vals)
  ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=45, ha='right')
  ax1.set_ylabel('RMS'); ax1.set_title('Activation RMS per Layer')
  
  ax2.bar(x, max_vals, color='orange')
  ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=45, ha='right')
  ax2.set_ylabel('Max Abs'); ax2.set_title('Activation Max Abs per Layer')
  
  plt.tight_layout()
  plt.savefig('/tmp/r157_magnitude_chart.png', dpi=150)
  print('Chart saved')
  "
  ```
- **Expected**: Bar chart saved showing activation magnitude progression through the decoder graph.
- **Verification**: Chart shows smooth magnitude flow (no sudden 1000× jumps).

## Acceptance
- [ ] All layer 0-2 activations captured to `/tmp/act_r156_*.bin` (≥12 files)
- [ ] Zero NaN/Inf values across all captured activations
- [ ] RVQ output correlation with GDB ground truth measured (even if shape mismatch documented)
- [ ] Divergence report `/tmp/r157_divergence_report.json` produced
- [ ] First divergence layer in 0-2 identified (or confirmed none)
- [ ] Magnitude chart `/tmp/r157_magnitude_chart.png` shows monotonic progression
- [ ] `experimental/compare_activations.py` can consume the r156_* dumps and produce structured output
