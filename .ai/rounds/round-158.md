# Round 158 — Compare DAC Layers 3–6 (Complete Heatmap)
**Status**: PENDING (Header Planned) | **Date**: 2026-05-28
**Predecessor**: round-157

## Strategy — WHY this round exists

R157 covered layers 0-2 (model.0 → block 1 → block 2). This round extends the activation dump and comparison to layers 3-6 (block 3 → block 4 → model.5 snake → model.6 output conv1d → tanh output). The goal is to produce a **complete correlation heatmap** for all 32 DAC sub-layers and identify the first layer where correlation with GDB ground truth drops below 0.95. If R157 found divergence in layers 0-2, this round will confirm its progression. If R157 found no divergence, this round pinpoints where it first appears. The Worker will also compare the final PCM output (after tanh) with the original tsac's WAV output to confirm the ~0.002 correlation and trace which layer caused it.

## Tasks

### T1: Instrument layers 3-6 with activation dumps (⬜)
- **In `src/cpu_blocks.inc`**: Activation dumps for blocks 3 and 4 are already covered if R157 instrumentation was general (uses `block` variable). Verify:
  ```bash
  grep -c "r156_b3_" /tmp/act_*.bin 2>/dev/null || echo "No block 3 dumps yet"
  grep -c "r156_b4_" /tmp/act_*.bin 2>/dev/null || echo "No block 4 dumps yet"
  ```
- If missing, add instrumentation for blocks 3 and 4 using the same pattern as R157 T2. The blocks loop in `cpu_blocks.inc` covers blocks 1-4, so if R157 instrumentation used `block` variable, blocks 3-4 are already covered. Verify in source:
  ```bash
  grep -n "dump_name\|sn_dump\|ct_dump\|in_dump\|res_dump" src/cpu_blocks.inc
  ```

- **Instrument model.5 snake in `src/cpu_tail.inc`**:
  After line 12 (snake call), add:
  ```c
  DUMP_ACT(current, current_C*cur_frames, "r158_m5_snake");
  ```

- **Instrument model.6 conv1d output (pre-tanh)** in `src/cpu_tail.inc`:
  After line 32 (DUMP_ACT for m6_pre_tanh already exists), verify:
  ```bash
  grep -n "m6_pre_tanh" src/cpu_tail.inc
  ```
  If not present, add after line 29:
  ```c
  DUMP_ACT(output, m6_Co*cur_frames, "r158_m6_pre_tanh");
  ```

- **Instrument final tanh output**:
  After tanh clamp loop (around line 66 in cpu_tail.inc), add:
  ```c
  DUMP_ACT(pcm, n_samples*ch, "r158_final_pcm");
  ```
  Note: This will dump the full PCM output, not just the current batch. Be careful about overwriting.

- **Expected**: All layers 3-6 activations dump to `/tmp/act_r158_*.bin` files.
- **Verification**: `ls /tmp/act_r158_*.bin 2>/dev/null | sort`

### T2: Build and decode with extended instrumentation (⬜)
- Set `DEBUG_DECODER 1` in `src/cpu_decoder.c` (line 42), rebuild:
  ```bash
  cmake --build build -j$(nproc) 2>&1 | tail -5
  ```
- Decode a short audio clip that exercises all layers:
  ```bash
  ./build/tsac-ng -v -f d test-simples/short_fast.txc /tmp/r158_out.wav 2>&1 | tee /tmp/r158_build.log
  ```
- If short_fast.txc doesn't exist, use any available `.txc` test file:
  ```bash
  ls test-simples/*.txc 2>/dev/null | head -3
  ```
  Fallback: encode a silent WAV first:
  ```bash
  python3 -c "
  import struct, math
  sr, dur, ch = 44100, 1, 2
  ns = int(sr*dur)
  with open('/tmp/silent1s.wav','wb') as f:
      data_size = ns*ch*2
      f.write(b'RIFF'); f.write(struct.pack('<I',36+data_size))
      f.write(b'WAVEfmt '); f.write(struct.pack('<I',16))
      f.write(struct.pack('<H',1)); f.write(struct.pack('<H',ch))
      f.write(struct.pack('<I',sr)); f.write(struct.pack('<I',sr*ch*2))
      f.write(struct.pack('<H',ch*2)); f.write(struct.pack('<H',16))
      f.write(b'data'); f.write(struct.pack('<I',data_size))
      for _ in range(ns*ch): f.write(struct.pack('<h',0))
  "
  ./build/tsac-ng -v -f e /tmp/silent1s.wav /tmp/silent_fast.txc --model /usr/share/tsac/dac_stereo_q8.bin 2>&1 | tee -a /tmp/r158_build.log
  ./build/tsac-ng -v -f d /tmp/silent_fast.txc /tmp/r158_out.wav 2>&1 | tee -a /tmp/r158_build.log
  ```

- **Expected**: Decoder runs to completion, all layer dumps created.
- **Verification**: `ls /tmp/act_r158_*.bin /tmp/act_r156_*.bin 2>/dev/null | wc -l` — count should be ≥ 30 (all 32 sub-layers).

### T3: Run activation analysis on all 32 layers (⬜)
- Run comprehensive analysis:
  ```bash
  python3 -c "
  import numpy as np, json, glob, os
  
  files = sorted(glob.glob('/tmp/act_r156_*.bin') + glob.glob('/tmp/act_r158_*.bin'))
  results = {}
  nan_layers = []
  inf_layers = []
  zero_rms_layers = []
  max_abs_vals = {}
  
  for f in files:
      name = os.path.basename(f).replace('.bin','')
      data = np.fromfile(f, dtype=np.float32)
      n = len(data)
      nan_c = int(np.isnan(data).sum())
      inf_c = int(np.isinf(data).sum())
      rms = float(np.sqrt(np.mean(data**2))) if n > 0 else 0.0
      max_abs = float(np.max(np.abs(data))) if n > 0 else 0.0
      mean_v = float(np.mean(data)) if n > 0 else 0.0
      
      results[name] = {
          'n': n, 'rms': rms, 'max_abs': max_abs, 'mean': mean_v,
          'nan': nan_c, 'inf': inf_c
      }
      max_abs_vals[name] = max_abs
      
      if nan_c > 0: nan_layers.append((name, nan_c))
      if inf_c > 0: inf_layers.append((name, inf_c))
      if rms == 0: zero_rms_layers.append(name)
  
  print(f'Total activations: {len(files)}')
  print(f'NaN layers: {nan_layers}')
  print(f'Inf layers: {inf_layers}')
  print(f'Zero RMS layers: {zero_rms_layers}')
  
  # Sort by max_abs to find explosion points
  sorted_by_abs = sorted(max_abs_vals.items(), key=lambda x: -x[1])
  print('\\nTop 5 by max_abs:')
  for name, val in sorted_by_abs[:5]:
      print(f'  {name}: {val:.4f}')
  print('\\nBottom 5 by max_abs:')
  for name, val in sorted_by_abs[-5:]:
      print(f'  {name}: {val:.6f}')
  
  # Detect sudden jumps: consecutive layers > 10x
  prev_name = None
  prev_max = None
  jumps = []
  for name, val in sorted_by_abs:
      if prev_max and val > 0:
          ratio = val / prev_max if prev_max > val else prev_max / val
          if ratio > 10:
              jumps.append((prev_name, name, ratio))
      prev_name = name
      prev_max = val
  
  print('\\nSudden jumps (>10x):')
  for a,b,r in jumps:
      print(f'  {a} -> {b}: {r:.1f}x')
  
  json.dump(results, open('/tmp/r158_full_activation_report.json', 'w'), indent=2)
  "
  ```

- **Expected**: Comprehensive report with all 32 layer activations, NaN detection, and jump analysis.
- **Verification**: Zero NaN/Inf expected. Any sudden max_abs jump > 10x indicates a numerical stability issue.

### T4: Compare our final PCM with original tsac output (⬜)
- Get original tsac reference output:
  ```bash
  /usr/bin/tsac -v -f d /tmp/silent_fast.txc /tmp/r158_ref.wav 2>&1
  ```
  (If no original tsac available, use a previously captured reference from `docs/evidence/`)

- Compute WAV-level comparison:
  ```bash
  python3 -c "
  import numpy as np, wave
  
  def read_wav(path):
      w = wave.open(path, 'rb')
      n = w.getnframes()
      ch = w.getnchannels()
      data = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, ch).T
      return data.astype(np.float32) / 32768.0
  
  ours = read_wav('/tmp/r158_out.wav')
  ref = read_wav('/tmp/r158_ref.wav')
  
  min_len = min(ours.shape[1], ref.shape[1])
  ours = ours[:, :min_len]
  ref = ref[:, :min_len]
  
  print(f'Samples: {min_len}, Channels: {ours.shape[0]}')
  
  for ch in range(ours.shape[0]):
      o = ours[ch].flatten()
      r = ref[ch].flatten()
      corr = np.corrcoef(o, r)[0,1]
      rmse = np.sqrt(np.mean((o-r)**2))
      mae = np.mean(np.abs(o-r))
      orms = np.sqrt(np.mean(o**2))
      rrms = np.sqrt(np.mean(r**2))
      snr = 20*np.log10(rrms/rmse) if rmse > 1e-10 else 99.9
      print(f'Ch{ch}: corr={corr:.6f} rmse={rmse:.6f} mae={mae:.6f} snr={snr:.2f}dB')
      print(f'  Ours RMS={orms:.4f} Ref RMS={rrms:.4f}')
  
  # Full comparison
  full_corr = np.corrcoef(ours.flatten(), ref.flatten())[0,1]
  print(f'Full corr={full_corr:.6f}')
  "
  ```

- **Expected**: Correlation ~0.002 (confirmed WAV divergence persists).
- **Verification**: The correlation value should match prior findings (around 0.00176).

### T5: Generate complete per-layer correlation heatmap (⬜)
- For each layer where GDB ground truth exists, compute correlation:
  ```bash
  python3 -c "
  import numpy as np, json, glob, os
  
  # GDB ground truth files
  gdb_files = {
      'model0_input': 'docs/evidence/gdb_model0_input_9216f32.bin',
  }
  libnc_files = {
      'inproj_cb0': 'docs/evidence/libnc_inproj_cb0_f32.bin',
      'inproj_cb1': 'docs/evidence/libnc_inproj_cb1_f32.bin',
      'inproj_cb2': 'docs/evidence/libnc_inproj_cb2_f32.bin',
      'inproj_cb3': 'docs/evidence/libnc_inproj_cb3_f32.bin',
      'inproj_cb4': 'docs/evidence/libnc_inproj_cb4_f32.bin',
      'inproj_cb5': 'docs/evidence/libnc_inproj_cb5_f32.bin',
  }
  
  # Our activation files
  our_files = glob.glob('/tmp/act_r156_*.bin') + glob.glob('/tmp/act_r158_*.bin')
  our_data = {}
  for f in our_files:
      name = os.path.basename(f).replace('.bin','')
      our_data[name] = np.fromfile(f, dtype=np.float32)
  
  # Build correlation matrix for all layers
  all_names = sorted(list(our_data.keys()))
  n = len(all_names)
  corr_matrix = np.eye(n)
  
  for i, ni in enumerate(all_names):
      for j, nj in enumerate(all_names):
          if i >= j: continue
          a = our_data[ni]
          b = our_data[nj]
          min_len = min(len(a), len(b))
          if min_len > 1:
              corr = np.corrcoef(a[:min_len], b[:min_len])[0,1]
              corr_matrix[i,j] = corr_matrix[j,i] = corr
          else:
              corr_matrix[i,j] = corr_matrix[j,i] = 0
  
  # Plot heatmap
  try:
      import matplotlib.pyplot as plt
      fig, ax = plt.subplots(figsize=(20, 16))
      im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
      ax.set_xticks(range(n))
      ax.set_yticks(range(n))
      ax.set_xticklabels(all_names, rotation=90, fontsize=6)
      ax.set_yticklabels(all_names, fontsize=6)
      plt.colorbar(im, label='Correlation')
      plt.title('Per-Layer Activation Correlation Matrix (All 32 DAC Layers)')
      plt.tight_layout()
      plt.savefig('/tmp/r158_full_heatmap.png', dpi=150)
      print('Heatmap saved: /tmp/r158_full_heatmap.png')
  except ImportError:
      print('matplotlib not available — saving correlation matrix as JSON')
      corr_dict = {
          'names': all_names,
          'matrix': corr_matrix.tolist(),
      }
      json.dump(corr_dict, open('/tmp/r158_correlation_matrix.json', 'w'))
  
  # Flag layers with > 10x RMS jump
  rms_vals = [float(np.sqrt(np.mean(our_data[n]**2))) for n in all_names]
  for i in range(1, len(rms_vals)):
      if rms_vals[i-1] > 0 and rms_vals[i] / rms_vals[i-1] > 10:
          print(f'WARNING: RMS jump {all_names[i-1]}->{all_names[i]}: {rms_vals[i-1]:.4f}->{rms_vals[i]:.4f} ({rms_vals[i]/rms_vals[i-1]:.1f}x)')
  "
  ```

- **Expected**: Correlation heatmap PNG showing relationships between all 32 sub-layers.
- **Verification**: Heatmap shows clear block structure (blocks 1-4 should each have high intra-block correlation, lower inter-block).

### T6: Decision gate — divergence location (⬜)
- Based on R157 and R158 findings, determine:
  ```python
  # Save as /tmp/r158_decision_gate.py
  import json

  conclusions = {
      "layers_0_2_divergence": None,   # Set from R157
      "layers_3_6_divergence": None,   # Set from R158
      "first_divergence_layer": None,
      "first_divergence_type": None,   # "nan", "rms_explosion", "corr_drop"
      "has_nan_any_layer": False,
      "final_pcm_correlation": None,   # Set from T4
      "recommendation": "",
  }
  
  # Fill from reports
  try:
      r157 = json.load(open('/tmp/r157_divergence_report.json'))
      nan_layers = [k for k,v in r157.items() if v.get('has_nan') or v.get('nan',0) > 0]
      if nan_layers:
          conclusions["layers_0_2_divergence"] = nan_layers[0]
          conclusions["has_nan_any_layer"] = True
  except: pass
  
  try:
      r158 = json.load(open('/tmp/r158_full_activation_report.json'))
      nan_found = [k for k,v in r158.items() if v.get('nan',0) > 0]
      inf_found = [k for k,v in r158.items() if v.get('inf',0) > 0]
      if nan_found:
          conclusions["layers_3_6_divergence"] = nan_found[0]
          conclusions["has_nan_any_layer"] = True
  except: pass
  
  # Determine recommendation
  if conclusions["has_nan_any_layer"]:
      conclusions["recommendation"] = "NaN/Inf detected — probable conv kernel bug. Proceed to R160 (conv1d deep comparison)."
  else:
      conclusions["recommendation"] = "No NaN/Inf. If final corr ≈ 0.002, divergence is cumulative numerical mismatch. Proceed to R159 (RVQ formula variants)."
  
  print(f"First divergence: {conclusions['first_divergence_layer']}")
  print(f"Has NaN: {conclusions['has_nan_any_layer']}")
  print(f"Recommendation: {conclusions['recommendation']}")
  
  json.dump(conclusions, open('/tmp/r158_decision.json', 'w'), indent=2)
  ```
- Run: `python3 /tmp/r158_decision_gate.py`
- **Expected**: Clear decision on which R159/R160 branch to take next.
- **Verification**: Decision JSON written to `/tmp/r158_decision.json`.

## Acceptance
- [ ] All 32 DAC sub-layers captured to activation dump files (R156+R158 combined)
- [ ] Zero NaN/Inf across all layers
- [ ] Complete correlation heatmap `/tmp/r158_full_heatmap.png` (or JSON matrix)
- [ ] Final PCM correlation with original tsac measured and documented
- [ ] Divergence layer identified (first layer with NaN, RMS explosion, or corr drop)
- [ ] Decision gate output `/tmp/r158_decision.json` guides R159/R160 selection
- [ ] Activation magnitude report `/tmp/r158_full_activation_report.json` shows smooth progression (no >10x jumps between consecutive layers)
