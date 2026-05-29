# Round 158 — GDB Capture Infrastructure Refinement (Phase 4A)
**Signed**: Header | **Date**: 2026-05-29 | **Status**: PENDING

## Summary
R156 noted that GDB scripting for `nc_tensor` data dump needs refinement. The existing scripts in `docs/evidence/gdb_trace_sqr.py` and `gdb_trace_nc_reduce_sum_sqr.py` work for specific internal function tracing but don't generalize to arbitrary tensor captures. R157 requires clean GDB captures of decoder layer activations — this round builds a reusable, robust GDB capture infrastructure.

**Problem**: Original libnc (`/usr/lib/tsac/libnc.so`) exposes tensors through `nc_inference_tensor()` which returns a `DACTensor*`. The struct layout at runtime differs from our struct definition — field offsets must be correct (verified in R120-R125: ndims at +0x64, dims[] at +0xa8, name at +0x20, data at +0x40, type at +0x38).

**Current blockers**:
1. GDB breakpoints on `nc_inference_tensor` fire 300+ times — need dim-based filtering
2. `nc_tensor` type is opaque in GDB — must use raw memory access
3. No automated dim↔layer mapping — manual dim lookup required per dump

## Round Strategy
1. Create reusable GDB command file with named tensor capture using dim+name matching
2. Add Python post-processing to auto-rename dumps by dim signature
3. Validate captures for model.0 input (compare with existing `gdb_model0_input_9216f32.bin`)
4. Document field offsets and struct layout as authoritative reference
5. Build Makefile target for one-command GDB capture

## Tasks

### T1: Create `docs/evidence/gdb_capture_tensor.py` — robust tensor capture
**File**: `docs/evidence/gdb_capture_tensor.py` (new)

**Action**: Write Python GDB script with:
- Breakpoint on `nc_inference_tensor` with conditional filtering
- Capture all tensors to a circular buffer (keep last N)
- On model exit or signal, dump unique tensors by dim signature
- Auto-map dims to layer names using lookup table:

```python
DIM_MAP = {
    (1024, 9): "gdb_model0_input",
    (1536, 9): "gdb_m0_conv1d",
    (768, 18): "gdb_block1_convt",
    (384, 36): "gdb_block2_convt",
    (192, 72): "gdb_block3_convt",
    (96, 144): "gdb_block4_convt",
    (2, 144): "gdb_m6_pre_tanh",
}
```

**GDB struct field offsets** (verified during R120-R125):
```
+0x00: int32 reference_count    (4 bytes)
+0x04: int32 tensor_type        (4 bytes) — 0=float32, 4=uint8, 9=BF8
+0x08: int32 elem_size          (4 bytes) — 0=libnc_override, 4=float32, 1=uint8
+0x0c: int32 ndims              (4 bytes) — 2 or 3
+0x10: int64 dims[4]            (32 bytes) — starts at +0x10 (NOT +0xa8!)

CORRECTION: Earlier code used +0x64 for ndims and +0xa8 for dims — these were from a different struct or libnc version. Re-verify with:
```gdb
print *(int*)((char*)$rdi + 0x0c)  # ndims
print *(long*)((char*)$rdi + 0x10)  # dims[0]
print *(long*)((char*)$rdi + 0x18)  # dims[1]
```

**Acceptance**: GDB script dumps uniquely identified tensors to `/tmp/gdb_act_*.bin`

### T2: Write Makefile target for automated GDB capture
**File**: `Makefile` (or build script alias)

**Action**: Add target:
```makefile
gdb-capture:
	gdb -batch -x docs/evidence/gdb_capture_tensor.py \
	  -ex "run -v -f d /tmp/short_fast.txc /dev/null" \
	  /usr/lib/tsac/tsac 2>&1 | tee /tmp/gdb_capture.log
	mkdir -p docs/evidence/
	python3 docs/evidence/gdb_rename_captures.py
```

**Acceptance**: `make gdb-capture` works in one command, produces named `.bin` files.

### T3: Validate against existing reference
**File**: `/tmp/gdb_act_model0_input.bin` vs `docs/evidence/gdb_model0_input_9216f32.bin`

**Action**: Byte-level comparison:
```bash
python3 -c "
import numpy as np
old = np.fromfile('docs/evidence/gdb_model0_input_9216f32.bin', dtype=np.float32)
new = np.fromfile('/tmp/gdb_act_model0_input.bin', dtype=np.float32)
print(f'old: {len(old)} floats, new: {len(new)} floats')
if len(old) == len(new):
    corr = np.corrcoef(old, new)[0,1]
    print(f'Correlation: {corr:.6f}')
    print(f'Max diff: {np.max(np.abs(old-new)):.8f}')
"
```

**Acceptance**: Correlation > 0.9999 between old and new capture. Validates struct offsets.

### T4: Capture all decoder layer activations
**File**: `docs/evidence/gdb_act_*.bin`

**Action**: Run full capture, verify all expected files exist:
```bash
ls -la docs/evidence/gdb_act_*.bin
for f in docs/evidence/gdb_act_*.bin; do
    python3 -c "import numpy as np; d=np.fromfile('$f',dtype=np.float32); print(f'$f: {d.shape} {d.size*4} bytes'); done"
done
```

**Expected files**:
| File | Dims | Elements | Description |
|------|------|----------|-------------|
| `gdb_model0_input.bin` | (1024, 9) | 9216 | RVQ output → conv1d input |
| `gdb_m0_conv1d.bin` | (1536, 9) | 13824 | model.0 conv1d output |
| `gdb_block1_convt.bin` | (768, 18) | 13824 | block.1 convt output |
| `gdb_block2_convt.bin` | (384, 36) | 13824 | block.2 convt output |
| `gdb_block3_convt.bin` | (192, 72) | 13824 | block.3 convt output |
| `gdb_block4_convt.bin` | (96, 144) | 13824 | block.4 convt output |
| `gdb_m6_pre_tanh.bin` | (2, 144) | 288 | model.6 conv1d output (pre-tanh) |

**Acceptance**: All 7 files present, total ~80KB of float32 reference data.

### T5: Document GDB field offsets in `docs/evidence/GDB_TENSOR_LAYOUT.md`
**File**: `docs/evidence/GDB_TENSOR_LAYOUT.md` (new)

**Action**: Document the `nc_tensor` / `DACTensor` struct layout as validated:
- Field-by-field memory layout with hex offsets
- `nc_inference_tensor` call convention
- Dim-based layer identification table
- Example `dump binary memory` commands per layer

**Acceptance**: Reference document usable by future rounds without re-deriving offsets.

## Acceptance Criteria
- [ ] `gdb_capture_tensor.py` captures all decoder tensors in one pass
- [ ] `make gdb-capture` works as one-command workflow
- [ ] New model.0 capture matches existing reference (corr > 0.9999)
- [ ] All 7 decoder layer captures saved to `docs/evidence/`
- [ ] Field offset document created for team reference
- [ ] R157-ready: captures feed directly into `compare_activations.py`
