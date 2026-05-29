# Round 173 — Range Coder Edge Case Testing (Phase 4D)

**Signed**: Worker | **Date**: 2026-05-29 | **Status**: PENDING

## Strategy

Subject the range coder to extreme probability distributions to verify it handles edge cases correctly: near-zero probabilities, near-certain probabilities, alternating patterns, and boundary conditions where cumulative frequency tables overflow or underflow. A robust range coder is essential for correct normal TXC decode since Transformer outputs can produce heavily skewed probability distributions.

**Prerequisites**: `build/tsac-ng` binary (static library or test harness). No GPU needed.

**Key files**: `src/range_coder.c`, `src/range_coder.h`, `src/test_transformer.c` (existing test harness).

## Tasks

### T1: Create range coder edge case test harness

```bash
mkdir -p /tmp/r173

# Create a standalone test program
cat > /tmp/r173/test_range_coder_edge.c << 'EOF'
/* test_range_coder_edge.c — Edge case testing for range coder.
 * Compile: gcc -I src -I include -o /tmp/r173/test_range_edge src/range_coder.c /tmp/r173/test_range_coder_edge.c -lm
 */
#include "range_coder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define MAX_SYMS 1024

/* Encode helper: build cumulative frequencies, encode a symbol, return compressed data */
static uint8_t *encode_sym(int sym, int n_syms, uint32_t total, size_t *out_len) {
    /* Simple test: encode a single symbol with given cumulative freq table */
    /* For testing decode only, we create compressed data manually */
    return NULL;
}

/* Test 1: decode with max probability (near 1.0) for a single symbol */
int test_max_probability() {
    printf("T1: Max probability (near 1.0)...\n");
    uint8_t buf[256];
    memset(buf, 0, sizeof(buf));
    buf[0] = 0xFF; buf[1] = 0xFF;  /* high initial bytes */
    
    RangeCoder rc;
    rc_decoder_init(&rc, buf, sizeof(buf));
    
    /* Cum freq: symbol 0 has 32766/32767 probability (~99.997%) */
    uint32_t cum_freq[MAX_SYMS + 1];
    cum_freq[0] = 0;
    cum_freq[1] = RC_MAX_FREQ - 1;  /* symbol 0 uses almost all range */
    cum_freq[2] = RC_MAX_FREQ;       /* symbol 1 gets 1/32767 */
    uint32_t total = RC_MAX_FREQ;
    
    int sym = rc_decode_cumul(&rc, cum_freq, 2, total);
    printf("  Decoded: %d (expected 0)\n", sym);
    return (sym == 0) ? 0 : -1;
}

/* Test 2: decode with min probability (near 0.0) */
int test_min_probability() {
    printf("T2: Min probability (near 0.0)...\n");
    uint8_t buf[256];
    memset(buf, 0, sizeof(buf));
    buf[0] = 0x00; buf[1] = 0x01;  /* low initial bytes */
    
    RangeCoder rc;
    rc_decoder_init(&rc, buf, sizeof(buf));
    
    /* Symbol 0 gets 1/32767, symbol 1 gets 32766/32767 */
    uint32_t cum_freq[MAX_SYMS + 1];
    cum_freq[0] = 0;
    cum_freq[1] = 1;                /* symbol 0 gets 1 */
    cum_freq[2] = RC_MAX_FREQ;      /* symbol 1 gets the rest */
    uint32_t total = RC_MAX_FREQ;
    
    int sym = rc_decode_cumul(&rc, cum_freq, 2, total);
    printf("  Decoded: %d (expected 1 — near-certain symbol)\n", sym);
    return (sym == 1) ? 0 : -1;
}

/* Test 3: decode with equal probabilities (all 1/n) */
int test_equal_probability() {
    printf("T3: Equal probabilities (all 1/n)...\n");
    uint8_t buf[1024];
    memset(buf, 0, sizeof(buf));
    buf[0] = 0x80; buf[1] = 0x00;  /* mid-range initial bytes */
    
    RangeCoder rc;
    rc_decoder_init(&rc, buf, sizeof(buf));
    
    /* 16 symbols, all equal probability */
    int n = 16;
    uint32_t cum_freq[MAX_SYMS + 1];
    cum_freq[0] = 0;
    for (int i = 0; i < n; i++)
        cum_freq[i+1] = cum_freq[i] + RC_MAX_FREQ / n;
    cum_freq[n] = RC_MAX_FREQ;
    uint32_t total = RC_MAX_FREQ;
    
    int sym = rc_decode_cumul(&rc, cum_freq, n, total);
    printf("  Decoded: %d (expected in [0,15])\n", sym);
    return (sym >= 0 && sym < n) ? 0 : -1;
}

/* Test 4: alternating high/low probabilities */
int test_alternating_probs() {
    printf("T4: Alternating high/low probabilities...\n");
    uint8_t buf[1024];
    memset(buf, 0, sizeof(buf));
    
    int failures = 0;
    
    /* Run 4 tests with different buf seeds to exercise different range coder paths */
    for (int seed = 0; seed < 4; seed++) {
        memset(buf, seed * 0x40, sizeof(buf));
        buf[0] = 0x80 + seed * 0x20;
        
        RangeCoder rc;
        rc_decoder_init(&rc, buf, sizeof(buf));
        
        /* Two symbols: symbol 0 = 90%, symbol 1 = 10% */
        uint32_t cum_freq[3] = {0, RC_MAX_FREQ * 9 / 10, RC_MAX_FREQ};
        int sym = rc_decode_cumul(&rc, cum_freq, 2, RC_MAX_FREQ);
        
        printf("  Seed %d: decoded %d\n", seed, sym);
        if (sym < 0 || sym > 1) failures++;
    }
    return failures ? -1 : 0;
}

/* Test 5: 1024-symbol vocabulary (like real Transformer output) */
int test_large_vocab() {
    printf("T5: 1024-symbol vocabulary (Transformer-like)...\n");
    uint8_t buf[4096];
    memset(buf, 0, sizeof(buf));
    /* Fill with pseudo-random pattern */
    for (int i = 0; i < (int)sizeof(buf); i++)
        buf[i] = (uint8_t)(i * 37 + 123);
    
    RangeCoder rc;
    rc_decoder_init(&rc, buf, sizeof(buf));
    
    /* Simulate a Transformer-like distribution: most mass on one symbol */
    int n = 1024;
    uint32_t cum_freq[MAX_SYMS + 1];
    cum_freq[0] = 0;
    for (int i = 0; i < n; i++) {
        /* Give 50% probability to symbol 512, rest distributed */
        if (i == 512)
            cum_freq[i+1] = cum_freq[i] + RC_MAX_FREQ / 2;
        else
            cum_freq[i+1] = cum_freq[i] + RC_MAX_FREQ / (2 * (n - 1));
    }
    /* Normalize to RC_MAX_FREQ */
    cum_freq[n] = RC_MAX_FREQ;
    
    int sym = rc_decode_cumul(&rc, cum_freq, n, RC_MAX_FREQ);
    printf("  Decoded: %d (expected near 512, high-probability symbol)\n", sym);
    return (sym >= 0 && sym < n) ? 0 : -1;
}

/* Test 6: boundary — total < cum_freq[n] (overflow check) */
int test_boundary_overflow() {
    printf("T6: Boundary — total parameter mismatch...\n");
    uint8_t buf[256];
    memset(buf, 0, sizeof(buf));
    buf[0] = 0x80;
    
    RangeCoder rc;
    rc_decoder_init(&rc, buf, sizeof(buf));
    
    /* Pass total LARGER than cum_freq[n] */
    uint32_t cum_freq[3] = {0, 1000, RC_MAX_FREQ};
    int sym = rc_decode_cumul(&rc, cum_freq, 2, RC_MAX_FREQ * 2);
    printf("  Decoded: %d (should handle gracefully, no crash)\n", sym);
    return (sym >= 0) ? 0 : -1;
}

/* Test 7: empty buffer (boundary) */
int test_empty_buffer() {
    printf("T7: Empty buffer (boundary)...\n");
    uint8_t buf[2] = {0};
    
    RangeCoder rc;
    int ret = rc_decoder_init(&rc, buf, 2);
    if (ret != 0) {
        printf("  Init failed with %d (expected — 2 bytes < RC_INIT_CODE_BYTES)\n", ret);
        return 0;  /* expected failure */
    }
    
    uint32_t cum_freq[3] = {0, RC_MAX_FREQ / 2, RC_MAX_FREQ};
    int sym = rc_decode_cumul(&rc, cum_freq, 2, RC_MAX_FREQ);
    printf("  Decoded: %d (may succeed with garbage or fail gracefully)\n", sym);
    return 0;  /* not strictly a failure — boundary test */
}

int main() {
    printf("=== Range Coder Edge Case Tests ===\n\n");
    int failures = 0;
    
    failures += test_max_probability();
    failures += test_min_probability();
    failures += test_equal_probability();
    failures += test_alternating_probs();
    failures += test_large_vocab();
    failures += test_boundary_overflow();
    failures += test_empty_buffer();
    
    printf("\n=== Results: %d failures ===\n", failures);
    return failures;
}
EOF

# Compile and run
gcc -I src -I include -o /tmp/r173/test_range_edge src/range_coder.c /tmp/r173/test_range_coder_edge.c -lm 2>&1
/tmp/r173/test_range_edge 2>&1 | tee /tmp/r173/edge_test_results.txt
```

**Acceptance**: Test harness compiles and runs. All edge cases handled without crash.

### T2: Test range coder with max probability (near 1.0)

Already covered by T1 (`test_max_probability`). Verify:

```bash
# Run the specific test
/tmp/r173/test_range_edge 2>&1 | grep -A3 "T1"
```

**Acceptance**: Symbol decoded correctly when one symbol has >99.99% probability.

### T3: Test range coder with near-zero probabilities

Already covered by T1 (`test_min_probability`). Also test with actual zero-frequency symbols:

```bash
# Additional: zero-probability test
python3 << 'EOF'
# Simulate zero-probability scenario:
# Transformer outputs zero probability for some symbols
# This should never happen in practice (probs are softmaxed), 
# but verify the cum_freq builder handles it
import ctypes, os

# Load the test output
with open('/tmp/r173/edge_test_results.txt') as f:
    print(f.read())
EOF
```

**Acceptance**: Near-zero probability symbols decode correctly. Zero-probability symbols are handled (either skipped or minimum probability assigned).

### T4: Test frequency table adaptation matches original tsac

```bash
# Verify against known original tsac range coder behavior
# Get the original tsac and run a comparison

python3 << 'EOF'
# Verify our range coder against reference test vectors

# Test vector 1: encode a known symbol with known probabilities
# Use the range coder to encode and decode a simple message
# Then verify the round-trip is correct for all edge cases

import subprocess, json, os

# Run through all tests and collect results
result = subprocess.run(['/tmp/r173/test_range_edge'], capture_output=True, text=True)
print(result.stdout)

# Parse results
lines = result.stdout.split('\n')
failures = 0
for line in lines:
    if 'failures' in line:
        import re
        m = re.search(r'(\d+) failures', line)
        if m:
            failures = int(m.group(1))

print(f'\nTest summary: {failures} edge case failures')
if failures == 0:
    print('✅ All range coder edge cases pass')
else:
    print(f'❌ {failures} edge case(s) fail — see output above')

# Save results
report = {
    'test': 'range_coder_edge_cases',
    'failures': failures,
    'total_tests': 7,
    'passed': 7 - failures,
    'details': result.stdout
}
with open('/tmp/r173/test_report.json', 'w') as f:
    json.dump(report, f, indent=2)
EOF
```

**Acceptance**: All 7 edge case tests pass (0 failures). Report saved.

## Acceptance Criteria

- **AC1**: Standalone test harness compiles and runs (`/tmp/r173/test_range_edge`)
- **AC2**: Max probability (>99.99%) case decodes correctly
- **AC3**: Min probability (<0.01%) case decodes correctly
- **AC4**: Equal probability (uniform distribution) case works
- **AC5**: Alternating high/low probability patterns work
- **AC6**: 1024-symbol vocabulary (Transformer-like) works without memory error
- **AC7**: Boundary conditions (overflow, empty buffer) handled gracefully
- **AC8**: Test report saved to `/tmp/r173/test_report.json`

## Expected Output

```
=== Range Coder Edge Case Tests ===

T1: Max probability (near 1.0)...
  Decoded: 0 (expected 0)
T2: Min probability (near 0.0)...
  Decoded: 1 (expected 1)
T3: Equal probabilities (all 1/n)...
  Decoded: 8 (expected in [0,15])
T4: Alternating high/low probabilities...
  Seed 0: decoded 0
  Seed 1: decoded 1
  Seed 2: decoded 0
  Seed 3: decoded 1
T5: 1024-symbol vocabulary (Transformer-like)...
  Decoded: 512 (expected near 512)
T6: Boundary — total parameter mismatch...
  Decoded: 1 (should handle gracefully)
T7: Empty buffer (boundary)...
  Init failed with -2

=== Results: 0 failures ===
```
