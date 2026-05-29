# Round 156 — Create Python Activation Comparison Framework
**Status**: PENDING (Header Planned) | **Date**: 2026-05-28
**Predecessor**: round-155

## Strategy — WHY this round exists

We have two sources of truth — GDB-captured activation dumps from the original `tsac` binary (stored in `docs/evidence/gdb_model0_input_9216f32.bin`, plus LD_PRELOAD captures at `docs/evidence/libnc_inproj_cb*_f32.bin`) and our own decoder's layer outputs (written to `/tmp/act_*.bin` by `DUMP_ACT()` when `DEBUG_DECODER=1`). Currently there is no structured framework to compare these across all 32 DAC layers. This round builds a standalone Python script (`experimental/compare_activations.py`) that loads both our activations and libnc GDB captures, identifies corresponding layers by name/dimension, computes per-layer correlation/MAE/RMSE metrics, generates a correlation heatmap, and produces a structured JSON report. The Worker must not need to manually inspect each layer's numbers — this script will be the single source of truth for pinpointing the first divergence layer in R157-R158.

## Tasks

### T1: Create comparison framework script (⬜)
- Create file: `experimental/compare_activations.py` with the following requirements:

**Imports**:
```python
import numpy as np
import json, os, sys, glob, struct
from pathlib import Path
```

**Function 1: `load_activation(path, n, dtype=np.float32)`**
- Read `n` floats from a binary `.bin` file.
- Return `np.array(dtype=dtype)` of shape `(n,)`.
- If file doesn't exist, return `None`.
- Expected: `gdb_model0_input_9216f32.bin` holds 9216 float32 values (= 1024×9).
- Verification: print shape and first 4 values.

**Function 2: `compute_metrics(ours, reference)`**
- Inputs: two 1D numpy arrays of same length.
- Output: dict with keys `correlation` (np.corrcoef), `mae` (mean abs error), `rmse` (sqrt(mean sq error)), `max_abs_diff`, `snr_db` (20*log10(ref_rms / rmse)).
- Handle NaN/Inf gracefully: use `np.nan_to_num` before computation.
- Expected: if arrays identical, correlation=1.0, mae=0.0, rmse=0.0.

**Function 3: `load_libnc_inproj(codebook_idx)`**
- Load `docs/evidence/libnc_inproj_cb{idx}_f32.bin`.
- Each file is 8192 float32 values (= 1024 entries × 8 dims).
- Reshape to (1024, 8).
- Return `None` if file not found.

**Function 4: `load_gdb_model0_input()`**
- Load `docs/evidence/gdb_model0_input_9216f32.bin`.
- Shape: (1024, 9) — 9 frames of 1024-dim RVQ features.
- Return numpy array.

**Function 5: `load_our_activation(layer_name)`**
- Load `/tmp/act_{layer_name}.bin` — our decoder's DUMP_ACT output.
- Infer dimension from file size: `file_size / 4` gives element count.
- Return `None` if file not found or size is 0.

**Main function:**
```python
def main():
    # 1. Define layer registry: list of (name, shape_expected, gdb_source_or_none)
    LAYERS = [
        ("rvq_out", (1024, 16), "gdb_model0_input_9216f32.bin"),  # model.0 input is RVQ output
        ("m0_conv1d", (1536, 16), None),   # No GDB capture yet for m0
        ("block1_convt", (768, 128), None),
        ("block2_convt", (384, 512), None),
        ("block3_convt", (192, 1024), None),
        ("block4_convt", (96, 2048), None),
        ("m6_pre_tanh", (2, 4096), None),
    ]
    
    # 2. For each layer, load ours & reference, compute metrics
    # 3. Print formatted table of correlation, RMSE, SNR_dB
    # 4. Generate heatmap as PNG: experimental/correlation_heatmap.png
    # 5. Save JSON report: experimental/activation_comparison.json
    # 6. Highlight first layer where correlation drops below 0.95
```

- **Expected output**: `experimental/compare_activations.py` file created and runs without import errors.
- **Verification**: `python3 -c "import sys; sys.path.insert(0,'experimental'); from compare_activations import *; print('OK')"`

### T2: Implement heatmap generation with matplotlib (⬜)
- Add `generate_heatmap(metrics_dict, output_path)` function:
  - Use `matplotlib` (import guard: try/except ImportError → print `"Install matplotlib: pip install matplotlib"` and skip).
  - Heatmap size: 10×6 inches.
  - X-axis: layer names (short: "RVQ", "M0", "B1C", "B2C", "B3C", "B4C", "M6").
  - Y-axis: metrics ("Correlation", "MAE", "RMSE", "SNR(dB)").
  - Annotate each cell with value formatted to 3 decimal places.
  - Colormap: `RdYlGn` for correlation (green=1.0, red=0.0), `viridis` for others.
  - Save to `experimental/correlation_heatmap.png` with 150 DPI.
  - Print `"Heatmap saved to experimental/correlation_heatmap.png"`.

- **Expected output**: heatmap PNG created when matplotlib is available.
- **Verification**: `python3 -c "import matplotlib; print('matplotlib available')"` — if true, heatmap generates.

### T3: Add per-element scatter plot for divergence analysis (⬜)
- Add `generate_scatter(ours, reference, layer_name, max_points=10000)`:
  - Downsample to `max_points` if array larger (random sample).
  - Create scatter plot: ours vs reference (reference on x-axis, ours on y-axis).
  - Add y=x line in red dashed.
  - Title: `f"{layer_name}: corr={corr:.4f}"`.
  - Save to `experimental/scatter_{layer_name}.png`.
  - Print outliers: points where `|ours - reference| > 3 * rmse`.

- **Expected output**: scatter plots for layers with GDB reference data.
- **Verification**: `ls experimental/scatter_*.png` shows files.

### T4: Add structured JSON output for downstream consumption (⬜)
- After computing all metrics, write `experimental/activation_comparison.json`:
```json
{
  "date": "2026-05-28",
  "n_layers": 7,
  "layers": {
    "rvq_out": {
      "correlation": 0.999,
      "mae": 0.001,
      "rmse": 0.002,
      "snr_db": 45.0,
      "shape": [1024, 16],
      "first_divergence": false
    }
  },
  "first_divergence_layer": null,
  "first_divergence_at_correlation_lt": 0.95,
  "recommendation": "All layers above threshold — no divergence detected."
}
```
- This format is parseable by R157-R160 for decision gates.
- **Expected output**: valid JSON file with all layers.
- **Verification**: `python3 -c "import json; d=json.load(open('experimental/activation_comparison.json')); print(f'{len(d[\"layers\"])} layers')"`

### T5: Create test fixtures and smoke test (⬜)
- Generate synthetic test data:
  - `python3 -c "import numpy as np; np.random.seed(42); a=np.random.randn(1000); a.tofile('/tmp/test_ours.bin'); a.tofile('/tmp/test_ref.bin')"`
- Run framework on synthetic data:
  - `python3 experimental/compare_activations.py --ours /tmp/test_ours.bin --ref /tmp/test_ref.bin --n 1000`
- Expected: correlation ≈ 1.0 (same random seed), mae ≈ 0.0.
- Then test with flipped sign: `python3 -c "a=np.random.randn(1000); a.tofile('/tmp/test_ours.bin'); (-a).tofile('/tmp/test_ref.bin')"`
- Expected: correlation ≈ -1.0.

- **Verification**: Both smoke tests pass.
- **Verification**: `python3 experimental/compare_activations.py --help` prints usage.

## Acceptance
- [ ] `experimental/compare_activations.py` exists and is importable without errors
- [ ] `python3 experimental/compare_activations.py` runs without error (skips layers with no data)
- [ ] `experimental/activation_comparison.json` produced with valid JSON structure
- [ ] `experimental/correlation_heatmap.png` produced (or gracefully skipped if no matplotlib)
- [ ] At least one scatter plot (`experimental/scatter_*.png`) generated from any available data
- [ ] Synthetic test fixtures pass (corr ≈ ±1.0 for same/flipped arrays)
- [ ] JSON report includes `first_divergence_layer` field with null or layer name
- [ ] Framework correctly reports "No comparison data available" when no /tmp/act_*.bin files exist yet
