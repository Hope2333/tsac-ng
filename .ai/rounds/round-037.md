# Round 037 — libnc Weight Dump Attempt (New GDB/LD_PRELOAD Approach)

**Date**: 2026-05-26
**Status**: Complete (blocked)

## Goal
Dump libnc-converted float32 weights from original tsac's memory for comparison with our dequant output.

## Approaches Attempted

### LD_PRELOAD nc_find_param + nc_tensor_get_ptr
- ✅ nc_find_param intercept works (322 calls logged)
- ✅ Found decoder.model.0.weight_v at tensor ptr 0x2f443140
- ❌ **nc_tensor_get_ptr returns NULL** — old-BF8 tensor hasn't been converted yet
- Root cause: nc_convert_from_old_bf is called INTERNALLY by libnc (not through PLT), and nc_find_param is called BEFORE conversion

### LD_PRELOAD nc_conv_1d
- ✅ Intercept works, 32 calls per batch confirmed
- ❌ nc_conv_1d calling convention unclear — weight not in expected register position
- First 6 calls have weight=NULL (snake/group_norm ops)
- Remaining calls show small integers (0x3, 0x9, 0x1b) in weight arg — not pointers

### GDB state walk
- ✅ Found state→model→params chain
- ❌ params hash table at params+0x28 has all-zero entries
- params+0x20 count=1 (expected 322)

### Key Finding
**nc_tensor_get_ptr returns NULL for non-converted old-BF8 tensors.** The conversion happens inside libnc (nc_load_param→nc_convert_from_old_bf, internal call), making it impossible to intercept with LD_PRELOAD or GDB inferior calls.

## Conclusion
Direct libnc weight comparison is infeasible without modifying the binary or using kernel-level tracing. Pivot to mathematical reverse-engineering.
