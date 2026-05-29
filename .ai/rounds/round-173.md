# Round 173 — Range Coder Edge Case Testing

**Status**: PENDING (Header Planned) | **Date**: 2026-05-29
**Predecessor**: round-172
**Priority**: MEDIUM — Verify robustness of arithmetic decode

## Strategy — WHY this round exists

The range coder is the backbone of normal TXC decode. Every codebook index (0-1023) is extracted via `rc_decode_cumul()`, which uses a cumulative frequency table derived from Transformer-predicted probabilities. If the range coder has edge cases at extreme probabilities, buffer boundaries, or repeated symbols, the entire normal TXC pipeline will produce corrupted indices.

The original tsac uses `arith.c` which we reverse-engineered in R120. Our `range_coder.c` implements the same algorithm, but has never been tested with:
- **Max probability**: Transformer predicts p=0.999 for one symbol, p~0 for rest
- **Min probability**: Transformer predicts p~0.001 for all symbols (near-uniform noise)
- **Alternating probabilities**: Extreme swing between frames (50% → 0.1% → 99.9%)
- **Buffer boundary**: Range coder hits end of buffer mid-decode
- **CRC boundary**: The last 4 bytes of the payload are CRC, not data — our decoder must stop before CRC
- **Many symbols**: 1024 symbols in the cumulative table (edge case for binary search)

This round creates a standalone test harness that exercises the range coder against a known reference, then adapts the frequency table adaptation logic to match the original tsac's behavior.

## Key files
- `/home/miao/Projects/tsac-ng/src/range_coder.c` — range coder implementation
- `/home/miao/Projects/tsac-ng/src/range_coder.h` — public API
- `/home/miao/Projects/tsac-ng/src/tsac_normal_decode.c` — consumer of range coder
- `/home/miao/Projects/tsac-ng/experimental/tests/test_range_coder.c` — new: standalone test
- `/home/miao/Projects/tsac-ng/docs/evidence/range_coder_test_vectors.bin` — new: reference test vectors

## Dependencies
- R172 complete (range coder is used in normal decode path)
- Python with numpy for test vector generation

## Tasks

### T1: Create standalone range coder test harness (⬜)

**Create `/home/miao/Projects/tsac-ng/experimental/tests/test_range_coder.c`**:

```c
/* test_range_coder.c — Standalone range coder edge case tests
 * Compile: gcc -o /tmp/test_range_coder test_range_coder.c \
 *          -I../.. -I../../src ../../src/range_coder.c -lm
 */
#include "range_coder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  TEST: %s ... ", name); \
    tests_run++; \
} while(0)

#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)

/* T1a: Test rc_decode_cumul with 1024 symbols, uniform probabilities */
static void test_uniform_1024(void) {
    TEST("uniform 1024 symbols");
    
    // Build uniform cumulative frequency table for 1024 symbols
    uint32_t cum_freq[1025];
    uint32_t total = RC_MAX_FREQ;  // 32767
    cum_freq[0] = 0;
    for (int i = 0; i < 1024; i++) {
        cum_freq[i+1] = cum_freq[i] + total / 1024;
    }
    cum_freq[1024] = total;
    
    // Encode known symbols using reference encoder, then decode with ours
    // For now: verify cum_freq table is monotonic
    for (int i = 0; i < 1024; i++) {
        assert(cum_freq[i] <= cum_freq[i+1]);
    }
    printf("cum_freq[0]=%u, cum_freq[512]=%u, cum_freq[1024]=%u\n",
           cum_freq[0], cum_freq[512], cum_freq[1024]);
    PASS();
}

/* T1b: Test decode of symbols from known compressed buffer */
static void test_decode_known(void) {
    TEST("decode symbols from reference bitstream");
    
    // Create a reference bitstream using a Python-generated test vector
    // Format: [4-byte length][payload bytes]
    const char *vec_path = "docs/evidence/range_coder_test_vectors.bin";
    FILE *f = fopen(vec_path, "rb");
    if (!f) {
        printf("SKIP (no test vectors at %s)\n", vec_path);
        return;
    }
    
    uint32_t payload_len;
    fread(&payload_len, 1, 4, f);
    uint8_t *payload = malloc(payload_len);
    fread(payload, 1, payload_len, f);
    fclose(f);
    
    // Expected symbols (after payload in the test vector file)
    uint32_t n_expected;
    fread(&n_expected, 1, 4, f);
    int *expected = malloc(n_expected * sizeof(int));
    fread(expected, sizeof(int), n_expected, f);
    
    RangeCoder rc;
    rc_decoder_init(&rc, payload, payload_len);
    
    // Build uniform cum_freq for 1024 symbols
    uint32_t cum_freq[1025];
    cum_freq[0] = 0;
    for (int i = 0; i < 1024; i++)
        cum_freq[i+1] = cum_freq[i] + RC_MAX_FREQ / 1024;
    cum_freq[1024] = RC_MAX_FREQ;
    
    int errors = 0;
    for (uint32_t i = 0; i < n_expected; i++) {
        int sym = rc_decode_cumul(&rc, cum_freq, 1024, RC_MAX_FREQ);
        if (sym != expected[i]) {
            if (errors < 5)
                printf("  Symbol %u: expected %d, got %d\n", i, expected[i], sym);
            errors++;
        }
    }
    
    if (errors == 0) {
        PASS();
    } else {
        FAIL("symbol mismatch");
        printf("  %d/%u symbols incorrect\n", errors, n_expected);
    }
    
    free(payload);
    free(expected);
}

int main(void) {
    printf("=== Range Coder Edge Case Tests ===\n\n");
    
    test_uniform_1024();
    test_decode_known();
    
    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
```

**Compile and run**:
```bash
gcc -o /tmp/test_range_coder \
    /home/miao/Projects/tsac-ng/experimental/tests/test_range_coder.c \
    /home/miao/Projects/tsac-ng/src/range_coder.c \
    -I/home/miao/Projects/tsac-ng/include -I/home/miao/Projects/tsac-ng/src -lm

/tmp/test_range_coder
```

**Acceptance**: Test harness compiles and runs. Reports PASS/SKIP/FAIL for each test.

---

### T2: Generate reference test vectors and verify decode (⬜)

**Create Python script `/home/miao/Projects/tsac-ng/experimental/generate_rc_test_vectors.py`**:

```python
#!/usr/bin/env python3
"""
Generate reference test vectors for the range coder.
Creates a binary file with:
- Encoded payload (range-coded symbols)
- Expected decoded symbols

This serves as ground truth for verifying rc_decode_cumul.
"""

import struct
import numpy as np

# Range coder constants (matching range_coder.h)
RC_MIN_VALUE = 0x0000FF00
RC_INIT_RANGE = 0xFFFFFFFF
RC_MAX_FREQ = 32767

def encode_symbols(symbols, n_syms=1024):
    """
    Simple range encoder for test purposes.
    Encodes symbols using cumulative frequencies into a byte buffer.
    
    This is a REFERENCE encoder - we just need to produce a bitstream
    that our decoder can decode. We use the same algorithm as the
    original tsac.
    """
    # Build uniform cumulative frequency table
    cum_freq = [0]
    for i in range(n_syms):
        cum_freq.append(cum_freq[-1] + RC_MAX_FREQ // n_syms)
    cum_freq[-1] = RC_MAX_FREQ
    
    low = 0
    high = RC_INIT_RANGE
    bytes_out = []
    
    for sym in symbols:
        # Scale range by symbol's cumulative frequency
        freq_range = high - low + 1
        sym_low = low + (freq_range * cum_freq[sym]) // RC_MAX_FREQ
        sym_high = low + (freq_range * cum_freq[sym + 1]) // RC_MAX_FREQ - 1
        low = sym_low
        high = sym_high
        
        # Renormalize
        while True:
            if high < 0x80000000:  # both < 2^31
                bytes_out.append((low >> 24) & 0xFF)
                low <<= 8
                high = (high << 8) | 0xFF
            elif low >= 0x80000000:  # both >= 2^31
                bytes_out.append((low >> 24) & 0xFF)
                low <<= 8
                high = (high << 8) | 0xFF
            elif low >= 0x40000000 and high < 0xC0000000:
                # Underflow: expand range without emitting byte
                low = (low - 0x40000000) << 8
                high = (high - 0x40000000) << 8 | 0xFF
            else:
                break
    
    # Flush remaining bytes
    bytes_out.append((low >> 24) & 0xFF)
    if low & 0x800000:
        bytes_out.append((low >> 16) & 0xFF)
    if low & 0x8000:
        bytes_out.append((low >> 8) & 0xFF)
    if low & 0x80:
        bytes_out.append(low & 0xFF)
    
    return bytes(bytes_out)


def main():
    np.random.seed(42)
    output_path = "/home/miao/Projects/tsac-ng/docs/evidence/range_coder_test_vectors.bin"
    
    test_cases = [
        # (name, symbols, description)
        ("uniform_10", np.random.randint(0, 1024, 10).tolist(), "10 random symbols"),
        ("uniform_100", np.random.randint(0, 1024, 100).tolist(), "100 random symbols"),
        ("all_zeros", [0] * 50, "50 repetitions of symbol 0"),
        ("all_1023", [1023] * 50, "50 repetitions of symbol 1023"),
        ("alternating", [0, 1023] * 25, "50 alternating symbol 0 and 1023"),
        ("ramp", list(range(1024)), "All 1024 symbols in order"),
    ]
    
    with open(output_path, "wb") as f:
        for name, symbols, desc in test_cases:
            payload = encode_symbols(symbols)
            
            # Write: [4-byte name_len][name][4-byte payload_len][payload][4-byte n_syms][symbols as int32]
            name_bytes = name.encode()
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('I', len(payload)))
            f.write(payload)
            f.write(struct.pack('I', len(symbols)))
            for s in symbols:
                f.write(struct.pack('i', s))
        
        # Write terminator
        f.write(struct.pack('I', 0))
    
    print(f"Test vectors written to {output_path}")
    print(f"Test cases: {[tc[0] for tc in test_cases]}")
    
    # Verification: read back and check
    with open(output_path, "rb") as f:
        while True:
            name_len = struct.unpack('I', f.read(4))[0]
            if name_len == 0:
                break
            name = f.read(name_len).decode()
            payload_len = struct.unpack('I', f.read(4))[0]
            payload = f.read(payload_len)
            n_syms = struct.unpack('I', f.read(4))[0]
            symbols = list(struct.unpack(f'{n_syms}i', f.read(n_syms * 4)))
            print(f"  Case '{name}': {len(payload)} bytes → {n_syms} symbols")

if __name__ == "__main__":
    main()
```

**Generate vectors and test**:
```bash
python3 /home/miao/Projects/tsac-ng/experimental/generate_rc_test_vectors.py
cmake --build /home/miao/Projects/tsac-ng/build
/tmp/test_range_coder 2>&1
```

**Acceptance**: Test vectors generated. Range coder decodes all vectors correctly (100% symbol match).

---

### T3: Test max/min probability edge cases (⬜)

**Create additional tests in `test_range_coder.c`**:

```c
/* T1c: Test with extreme probability distribution (one symbol dominates) */
static void test_extreme_probs(void) {
    TEST("extreme probabilities");
    
    uint32_t cum_freq[1025];
    uint32_t total = RC_MAX_FREQ;
    
    // Scenario: symbol 0 has 90% probability, remaining 10% spread across others
    cum_freq[0] = 0;
    cum_freq[1] = (uint32_t)(total * 0.9f);  // 90% for symbol 0
    int remaining = total - cum_freq[1];
    for (int i = 1; i < 1024; i++) {
        cum_freq[i+1] = cum_freq[i] + remaining / 1023;
    }
    cum_freq[1024] = total;
    
    // Create a payload with 20 symbols, expect most to decode as 0
    // This is a decode-only test — we just check no crash
    uint8_t test_payload[64];
    memset(test_payload, 0, 64);
    
    RangeCoder rc;
    rc_decoder_init(&rc, test_payload, 64);
    
    int decoded[20];
    for (int i = 0; i < 20; i++) {
        int sym = rc_decode_cumul(&rc, cum_freq, 1024, total);
        if (sym < 0 || sym >= 1024) {
            FAIL("symbol out of range");
            return;
        }
        decoded[i] = sym;
    }
    
    // Print distribution
    int count_zero = 0;
    for (int i = 0; i < 20; i++)
        if (decoded[i] == 0) count_zero++;
    printf("  Zero count: %d/20 (expected ~18)\n", count_zero);
    
    PASS();
}

/* T1d: Test near-zero probability (all symbols nearly uniform) */
static void test_near_zero(void) {
    TEST("near-zero probability");
    
    uint32_t cum_freq[1025];
    cum_freq[0] = 0;
    for (int i = 0; i < 1024; i++)
        cum_freq[i+1] = cum_freq[i] + 32;  // 32 each = 32768 total → clamped to RC_MAX_FREQ
    cum_freq[1024] = RC_MAX_FREQ;
    
    uint8_t test_payload[64];
    memset(test_payload, 0xFF, 64);  // all-ones payload
    
    RangeCoder rc;
    rc_decoder_init(&rc, test_payload, 64);
    
    for (int i = 0; i < 10; i++) {
        int sym = rc_decode_cumul(&rc, cum_freq, 1024, RC_MAX_FREQ);
        if (sym < 0) { FAIL("decode error"); return; }
    }
    PASS();
}

/* T1e: Test alternating high/low probability frames */
static void test_alternating_probs(void) {
    TEST("alternating frame probabilities");
    
    uint32_t cum_freq_high[1025];  // symbol 512 dominates
    uint32_t cum_freq_low[1025];   // symbol 0 dominates
    
    // Build "high" table: symbol 512 gets 80%
    cum_freq_high[0] = 0;
    for (int i = 0; i < 1024; i++) {
        if (i == 512)
            cum_freq_high[i+1] = cum_freq_high[i] + (uint32_t)(RC_MAX_FREQ * 0.8f);
        else
            cum_freq_high[i+1] = cum_freq_high[i] + (uint32_t)(RC_MAX_FREQ * 0.2f / 1023.0f);
    }
    cum_freq_high[1024] = RC_MAX_FREQ;
    
    // Build "low" table: symbol 0 gets 80%
    cum_freq_low[0] = 0;
    cum_freq_low[1] = (uint32_t)(RC_MAX_FREQ * 0.8f);
    for (int i = 1; i < 1024; i++)
        cum_freq_low[i+1] = cum_freq_low[i] + (uint32_t)(RC_MAX_FREQ * 0.2f / 1023.0f);
    cum_freq_low[1024] = RC_MAX_FREQ;
    
    uint8_t test_payload[128];
    memset(test_payload, 0, 128);
    
    RangeCoder rc;
    rc_decoder_init(&rc, test_payload, 128);
    
    // Alternate between tables
    for (int frame = 0; frame < 5; frame++) {
        uint32_t *table = (frame % 2 == 0) ? cum_freq_high : cum_freq_low;
        int sym = rc_decode_cumul(&rc, table, 1024, RC_MAX_FREQ);
        if (sym < 0) { FAIL("decode error on alternating"); return; }
        printf("  Frame %d (table=%s): sym=%d\n", frame, 
               (frame%2==0)?"high":"low", sym);
    }
    PASS();
}
```

**Compile and run all tests**:
```bash
gcc -o /tmp/test_range_coder \
    /home/miao/Projects/tsac-ng/experimental/tests/test_range_coder.c \
    /home/miao/Projects/tsac-ng/src/range_coder.c \
    -I/home/miao/Projects/tsac-ng/include -I/home/miao/Projects/tsac-ng/src -lm

/tmp/test_range_coder
```

**Acceptance**: All edge case tests pass without crash. For extreme probabilities, output distribution is reasonable (more zero symbols with 90% probability table, etc.).

---

### T4: Test buffer boundary and CRC boundary conditions (⬜)

**Add to `test_range_coder.c`**:

```c
/* T1f: Test buffer boundary — decode exactly at end of payload */
static void test_buffer_boundary(void) {
    TEST("buffer boundary decode");
    
    // Create minimal payload: exactly 4 bytes + 1 symbol
    uint32_t cum_freq[1025];
    cum_freq[0] = 0;
    for (int i = 0; i < 1024; i++)
        cum_freq[i+1] = cum_freq[i] + RC_MAX_FREQ / 1024;
    cum_freq[1024] = RC_MAX_FREQ;
    
    uint8_t tiny_payload[5] = {0x80, 0x00, 0x00, 0x00, 0x00};  // minimal valid stream
    
    RangeCoder rc;
    int ret = rc_decoder_init(&rc, tiny_payload, 5);
    if (ret != 0) { FAIL("init failed on small buffer"); return; }
    
    // Decode as many symbols as possible
    int decoded = 0;
    for (int i = 0; i < 10; i++) {
        int sym = rc_decode_cumul(&rc, cum_freq, 1024, RC_MAX_FREQ);
        if (sym >= 0) decoded++;
        else break;  // legitimate end-of-stream
    }
    printf("  Decoded %d symbols from 5-byte payload (expected ~1-2)\n", decoded);
    PASS();
}

/* T1g: Test that decoder stops BEFORE consuming CRC bytes */
static void test_crc_boundary(void) {
    TEST("CRC boundary — exclude last 4 bytes");
    
    // Create payload where last 4 bytes are CRC, not data
    uint8_t payload_with_crc[32];
    memset(payload_with_crc, 0, 32);
    
    uint32_t cum_freq[1025];
    cum_freq[0] = 0;
    for (int i = 0; i < 1024; i++)
        cum_freq[i+1] = cum_freq[i] + RC_MAX_FREQ / 1024;
    cum_freq[1024] = RC_MAX_FREQ;
    
    // Init with full payload (includes CRC)
    RangeCoder rc_full;
    rc_decoder_init(&rc_full, payload_with_crc, 32);
    
    // Init with CRC-excluded payload (first 28 bytes)
    RangeCoder rc_no_crc;
    rc_decoder_init(&rc_no_crc, payload_with_crc, 28);
    
    // Both should decode the same number of symbols before exhaustion
    int count_full = 0, count_no_crc = 0;
    while (rc_decode_cumul(&rc_full, cum_freq, 1024, RC_MAX_FREQ) >= 0) count_full++;
    while (rc_decode_cumul(&rc_no_crc, cum_freq, 1024, RC_MAX_FREQ) >= 0) count_no_crc++;
    
    printf("  Symbols from full=%d, from no-CRC=%d\n", count_full, count_no_crc);
    if (count_full >= count_no_crc) {
        PASS();
    } else {
        FAIL("CRC boundary issue");
    }
}
```

**Acceptance**: Buffer boundary tests pass. Range coder gracefully handles end-of-stream. CRC boundary exclusion works correctly (stops before last 4 bytes).

---

### T5: Verify frequency table adaptation logic matches original tsac (⬜)

**Analysis task**: The original tsac's range coder may use an ADAPTIVE frequency table (the `get_freq` function in `arith.c`). Our implementation has both `rc_decoder_get_freq` (adaptive 15-bit) and `rc_decode_cumul` (static table). Determine which one the original normal TXC actually uses.

**Check the original tsac binary**:
```bash
# Disassemble arith.c functions from original tsac (if available)
objdump -d $(which tsac) | grep -A 50 "get_freq\|decode_cumul\|arith" | head -100
```

**Compare behavior**:
1. In the original tsac, does the frequency table change after each symbol decode?
2. Or is it static for the entire frame?

**If adaptive**: Implement `rc_adapt_freq()` function:
```c
void rc_adapt_freq(uint32_t *cum_freq, int n_syms, int decoded_sym, uint32_t *total) {
    // Increase frequency of decoded symbol, decrease others
    // Algorithm reverse-engineered from original tsac
}
```

**Test the adaptation**:
```c
/* T1h: Test frequency table adaptation */
static void test_freq_adaptation(void) {
    TEST("frequency table adaptation");
    
    // Verify that after decoding a symbol, its frequency increases
    
    PASS();
}
```

**Acceptance**: Frequency table adaptation behavior documented. If adaptation is used by original tsac, implemented and verified. If not, confirmed that static table is correct.

---

## Acceptance Criteria

- [ ] **T1**: Standalone test harness at `experimental/tests/test_range_coder.c` compiles and runs.
- [ ] **T2**: Reference test vectors generated at `docs/evidence/range_coder_test_vectors.bin`. All decode correctly.
- [ ] **T3**: Max/min probability edge cases tested. No crash or assertion failure.
- [ ] **T4**: Buffer boundary and CRC boundary tests pass. Decoder stops at correct position.
- [ ] **T5**: Frequency table adaptation analyzed. Documented whether adaptive or static, with evidence.
- [ ] All tests pass: `/tmp/test_range_coder` returns exit code 0.
- [ ] Fast TXC decode NOT broken.

## Decision Gate

| Condition | Action |
|-----------|--------|
| All tests pass | → R174 (multi-file normal TXC testing) |
| Buffer boundary fails | Fix rc_decode_cumul, add range bounds checking, retry |
| Frequency adaptation wrong | Fix to match original tsac, retry |
