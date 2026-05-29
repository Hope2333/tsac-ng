# Round 172 — Transformer Output Validation (Phase 4D)

**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Strategy

Quantitatively compare our Transformer's predicted codebook indices with those produced by the original tsac. The key metric is **index accuracy %**: for each frame and codebook, what fraction of our decoded indices (0-1023) match the original's? This is the most direct test of Transformer correctness, independent of the DAC decoder. High index accuracy (>80%) indicates the Transformer + range coder pipeline is working correctly.

**Prerequisites**: `build/tsac-ng` binary, `tsac_stereo_q8.bin` transformer weights, normal TXC test files with original tsac reference decodes.

**Key files**: `src/tsac_transformer.c`, `src/tsac_transformer.h`, `src/tsac_normal_decode.c`, `src/range_coder.c`.

## Tasks

### T1: Capture original tsac codebook indices from normal TXC decode

```bash
mkdir -p /tmp/r172

# If we have GDB infrastructure to dump original tsac's decoded indices:
# gdb -batch -ex "b *tsac_decode+0x1234" -ex "run d input.txc output.wav" -ex "dump binary memory /tmp/r172/ref_indices.bin \$rsi \$rdi" ./tsac 2>&1

# Alternative: decode with original tsac and compare WAV shapes to infer indices
# Simplest: if we have tsac source patched to dump indices, use that
# For now, create a Python test harness 

# First, decode a normal TXC with our codec and dump the codebook indices
# Add debug output to tsac_normal_decode.c to log indices
# Or use GDB to capture our indices:
# gdb -batch \
#   -ex "b tsac_normal_decode if frame_n==0 && cb==0" \
#   -ex "run d test.txc out.wav" \
#   -ex "p all_indices[0]" \
#   ./build/tsac-ng 2>&1

echo "See T4 for index analysis framework"
```

**Acceptance**: Framework for capturing reference indices established.

### T2: Implement codebook index dumping in our decoder

Add a debug flag to `tsac_normal_decode.c` to dump decoded indices:

```c
// In tsac_normal_decode.c, add after decode loop:
// #ifdef DEBUG_DUMP_INDICES
//     FILE *f = fopen("/tmp/r172/our_indices.bin", "wb");
//     fwrite(all_idx, sizeof(int), n_frames * n_cb, f);
//     fclose(f);
// #endif
```

Or use the existing activation dump infrastructure:

```bash
# Rebuild with -DDEBUG_DECODER=1
cd build && cmake .. -DCMAKE_C_FLAGS="-DDEBUG_DECODER=1" && make -j$(nproc) 2>&1 | tail -5
```

**Acceptance**: Our decoder dumps codebook indices to `/tmp/r172/our_indices.bin`.

### T3: Test with multiple normal TXC files

```bash
# Encode test files if needed, or use existing normal TXC files
# List what's available
find . -name "*normal*.txc" -o -name "*_n_*.txc" 2>/dev/null

# If none exist, create minimal test:
# 1. Use original tsac to encode
# 2. Decode with our codec

# For each normal TXC file found, decode and capture indices
for txc in $(find . -name "*normal*.txc" 2>/dev/null); do
    echo "=== Decoding: $txc ==="
    base=$(basename "$txc" .txc)
    ./build/tsac-ng -v d "$txc" "/tmp/r172/${base}_output.wav" 2>&1
    # Captured indices should be at /tmp/r172/our_indices.bin
done
```

**Acceptance**: At least 2 normal TXC files decoded with index dumps.

### T4: Measure codebook index accuracy (0-1023 range)

```bash
python3 << 'EOF'
import numpy as np, os

def load_indices(path, n_frames=None, n_cb=6):
    """Load int32 indices from binary dump."""
    if not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.int32)
    if n_frames:
        expected = n_frames * n_cb
        if len(data) >= expected:
            data = data[:expected]
        return data.reshape(-1, n_cb)
    # Try to infer shape — n_cb is typically 6-12
    for nc in [6, 8, 12]:
        if len(data) % nc == 0:
            return data.reshape(-1, nc)
    return data

# Load our indices
our = load_indices('/tmp/r172/our_indices.bin')
if our is None:
    print('Our indices not found at /tmp/r172/our_indices.bin')
    print('Run decoder with index dumping first (T2)')
    exit(1)

print(f'Our indices shape: {our.shape}')
print(f'Index range: [{our.min()}, {our.max()}]')
print(f'Expected range: [0, 1023]')

# Check that indices are in valid range
in_range = (our >= 0) & (our < 1024)
pct_valid = in_range.mean() * 100
print(f'Indices in [0,1023]: {pct_valid:.2f}%')
if pct_valid < 100:
    invalid = our[~in_range]
    print(f'  Invalid indices: {invalid[:20]}...')

# Per-codebook statistics
print(f'\nPer-codebook stats:')
for cb in range(our.shape[1]):
    col = our[:, cb]
    unique = np.unique(col)
    print(f'  CB{cb}: range=[{col.min()},{col.max()}], unique={len(unique)}, most_common={np.bincount(col.astype(int), minlength=1024).max()}')

# If reference indices exist, compare
ref = load_indices('/tmp/r172/ref_indices.bin')
if ref is not None:
    n_frames = min(len(our), len(ref))
    matches = (our[:n_frames] == ref[:n_frames])
    accuracy = matches.mean() * 100
    print(f'\n=== Index Accuracy ===')
    print(f'Overall: {accuracy:.2f}%')
    for cb in range(our.shape[1]):
        cb_acc = matches[:, cb].mean() * 100
        print(f'  CB{cb}: {cb_acc:.2f}%')
    
    # Per-frame accuracy
    frame_acc = matches.all(axis=1).mean() * 100
    print(f'  Per-frame (all CBs correct): {frame_acc:.2f}%')
else:
    print(f'\nNo reference indices available for comparison.')
    print('Logged our index distribution for analysis.')
EOF
```

**Acceptance**: Index accuracy measured. Valid range [0,1023] confirmed for all indices.

### T5: Fix any index prediction errors

If accuracy < 80%, investigate root causes:

```bash
# Check specific failure patterns
python3 << 'EOF'
import numpy as np

our = np.fromfile('/tmp/r172/our_indices.bin', dtype=np.int32)
# Analyze patterns
print('Index value distribution:')
hist, bins = np.histogram(our, bins=32, range=(0, 1024))
for i in range(32):
    print(f'  [{bins[i]:4.0f},{bins[i+1]:4.0f}): {hist[i]}')

# Check for systematic bias (all zeros, all same value, etc.)
print(f'\nMost common indices:')
unique, counts = np.unique(our, return_counts=True)
top5 = np.argsort(counts)[-5:][::-1]
for i in top5:
    print(f'  Index {unique[i]}: {counts[i]} times ({counts[i]/len(our)*100:.1f}%)')
EOF

# If all indices are 0: Transformer returning zero logits or softmax broken
# If indices are uniformly random: range coder not using Transformer output properly
# If indices match but ~0 correlation: quantization noise in probability table
```

**Acceptance**: Root cause of inaccuracies identified and fix applied (or documented as known limitation).

## Acceptance Criteria

- **AC1**: Index dump infrastructure works (outputs `/tmp/r172/our_indices.bin`)
- **AC2**: All decoded indices are in valid range [0, 1023]
- **AC3**: Index accuracy measured and reported as percentage
- **AC4**: Tested with 2+ normal TXC files
- **AC5**: Errors diagnosed if accuracy < 80%

## Expected Output

```
Our indices shape: (86, 6)
Index range: [0, 1023]
Indices in [0,1023]: 100.00%

Per-codebook stats:
  CB0: range=[0,1023], unique=1024, most_common=1
  CB1: range=[0,1023], unique=1024, most_common=1
  ...

If reference available:
  Overall: 0.12%  (expected low — Transformer output not yet correctly wired)
  
Most common indices:
  Index 512: 86 times (16.7%)  ← systematic bias toward middle of range
```

> Note: Early runs will likely show near-random index selection (~0.1% accuracy = random chance for 1024-class problem). This is expected for the first integration attempt. The task prioritizes establishing the measurement framework.
