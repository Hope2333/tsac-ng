# Round 157 — Per-Layer Activation Comparison (Phase 4A)
**Signed**: Header | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
Build on R156 `compare_activations.py` framework to dump DAC decoder layer outputs (model.0 → model.6) and compare with libnc GDB captures. Target: identify the first layer where correlation drops below 0.9, pinpointing where our graph diverges from ground truth.

R156 root cause findings carried forward:
- RVQ in_proj/out_proj access pattern fixed [Ci][Co][K] → [Co][Ci][K]
- L2 normalization added for K=1 layers (quantizer only)
- RVQ RMS improved 0.0012 → 0.0644 (target ~0.03-0.06)
- m6_pre_tanh RMS improved 447 → 1.81 (target <3)

**Open question**: After RVQ fix, per-layer activation correlation is unknown. Need to measure systematically to guide remaining fixes (conv kernel, convt access, snake/tanh alignment).

## Round Strategy
1. Run decoder with `DEBUG_DECODER=1` (already enabled in `src/cpu_decoder.c:42`) to dump activations at all 7 key layer points
2. Ensure GDB reference captures exist for at least model.0 input (RVQ output) from original libnc
3. Run `experimental/compare_activations.py` to compute per-layer correlation metrics
4. Identify first divergence layer (corr < 0.9) — this is the priority fix target
5. Generate correlation heatmap to visualize hotspot layers

## Tasks

### T1: Activate full debug dumping for all 32+4 DAC layers
**File**: `src/cpu_decoder.c` lines 472, 494, and `src/cpu_blocks.inc` line 58-59, `src/cpu_tail.inc` line 32

Currently only 7 layer points are dumped: `rvq_out`, `m0_conv1d`, `block{1,2,3,4}_convt`, `m6_pre_tanh`. The residual unit inner blocks (block.2, block.3, block.4 inside each of 4 main blocks = 12 inner layers) are NOT dumped.

**Action**: Add `DUMP_ACT()` calls in `src/cpu_blocks.inc` for:
- After each snake (block.{1-4}.block.{2,3,4}.block.0.alpha) — 12 dumps
- After each inner conv1d (block.{1-4}.block.{2,3,4}.block.1) — 12 dumps
- After each skip-add (block.{1-4}.block.{2,3,4} output) — 4 dumps
- After model.5 snake — in tail.inc line 11

**Naming scheme**: `b{1,2,3,4}_snake{2,3,4}`, `b{1,2,3,4}_conv_dil{1,3,9}`, `b{1,2,3,4}_add{2,3,4}`, `m5_snake`

**Acceptance**: Running decoder produces 30+ activation dump files in `/tmp/act_*.bin`

### T2: Capture libnc GDB reference for model.1 through model.6 layers
**File**: `docs/evidence/gdb_capture_activations.gdb` (create new GDB script)

Existing GDB captures only have model.0 input (`docs/evidence/gdb_model0_input_9216f32.bin`). Need reference dumps for ALL layers.

**Action**: Create `docs/evidence/gdb_capture_activations.gdb`:
```gdb
set pagination off
set disable-randomization on
break nc_inference_tensor if $rdi != 0
commands
  silent
  printf "=== TENSOR at %p ===\n", $rdi
  set $nd = *(int*)((char*)$rdi + 0x64)
  set $d0 = *(long*)((char*)$rdi + 0xa8)
  set $d1 = ($nd >= 2) ? *(long*)((char*)$rdi + 0xb0) : 1
  set $d2 = ($nd >= 3) ? *(long*)((char*)$rdi + 0xb8) : 1
  set $data = *(long*)((char*)$rdi + 0x40)
  printf "dims=[%ld,%ld,%ld] nd=%d data=0x%lx\n", $d0, $d1, $d2, $nd, $data
  set $name = *(long*)((char*)$rdi + 0x20)
  if $name != 0
    printf "name=%s\n", $name
  end
  if $data > 0x100000
    dump binary memory /tmp/gdb_act_$d0_$d1_$d2.bin $data $data+$d0*$d1*$d2*4
  end
  continue
end
run -v -f d /tmp/short_fast.txc /dev/null
```

**Key layers to identify by dim signature**:
- model.0 input: (1024, 9) → `/tmp/act_rvq_out.bin`
- model.0 output: (1536, 9) → `/tmp/act_m0_conv1d.bin`
- block.1 convt output: (768, 18) → `/tmp/act_block1_convt.bin`
- block.2 convt output: (384, 36) → `/tmp/act_block2_convt.bin`
- block.3 convt output: (192, 72) → `/tmp/act_block3_convt.bin`
- block.4 convt output: (96, 144) → `/tmp/act_block4_convt.bin`
- model.5 snake output: (96, 144) → `/tmp/act_m5_snake.bin`
- model.6 output: (2, 144) → `/tmp/act_m6_pre_tanh.bin`

Post-process with Python to rename into `docs/evidence/gdb_act_*.bin`.

**Acceptance**: All 8 key layer dumps from libnc exist in `docs/evidence/`

### T3: Run compare_activations.py with full dataset
**File**: `experimental/compare_activations.py`

**Action**: Execute:
```bash
# Build with DEBUG_DECODER=1
cmake --build build && ./build/tsac-ng -v -f d /tmp/short_fast.txc /tmp/out.wav

# Copy GDB captures to docs/evidence/
# (from T2 output)
cp /tmp/gdb_act_*.bin docs/evidence/

# Run comparison
python3 experimental/compare_activations.py
```

**Expected analysis output**:
```
Layer                corr      RMSE      SNR(dB)
rvq_out              0.9999    0.0012    68.3
m0_conv1d            ?.????    ?.????    ?.?     ← first suspect
block1_convt         ?.????    ?.????    ?.?
...
m6_pre_tanh          0.0001    1.8123    0.8
```

**Acceptance**: JSON report generated at `experimental/activation_comparison.json` with per-layer metrics. First divergence layer identified.

### T4: Generate correlation heatmap and scatter plots
**File**: `experimental/compare_activations.py` functions `generate_heatmap()` (line 55) and `generate_scatter()` (line 87)

**Action**: Ensure matplotlib is available, run the comparison script which auto-generates:
- `experimental/correlation_heatmap.png` — heatmap of correlation/MAE/RMSE/SNR across all layers
- `experimental/scatter_{layer}.png` — scatter plots for each layer

**Acceptance**: Heatmap PNG saved. Scatter plots for all compared layers. Visual identification of divergence hotspot.

### T5: Document findings in `experimental/activation_comparison.json`
**File**: `experimental/activation_comparison.json`

**Action**: Review JSON report for:
- `first_divergence_layer` — the layer with corr < 0.95 (or highest corr drop)
- `recommendation` — which component to fix next

**Acceptance**: JSON file contains actionable recommendation for R158-R165 prioritization.

## Acceptance Criteria
- [ ] 30+ activation dump files produced from our decoder (inner layers included)
- [ ] GDB captures for all 8 key layers from original libnc
- [ ] `compare_activations.py` runs end-to-end without errors
- [ ] Correlation heatmap generated at `experimental/correlation_heatmap.png`
- [ ] First divergence layer identified and documented in JSON report
- [ ] Concrete evidence guiding R158 priority (GDB infra, conv kernel, or convt fix)
