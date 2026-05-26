# Reverse Engineering Notes — original tsac binary

**Binary**: /usr/lib/tsac/tsac (stripped x86-64 ELF, BuildID b822853e, GCC 8.5.0)
**Version**: 2024-04-08, Copyright Fabrice Bellard
**Library**: libnc.so (257 nc_* symbols)

## CRC32 (FULLY REVERSED)

| Property | Value |
|----------|-------|
| Function address | 0x42a610 (file offset ~0x36610) |
| Table address | 0x43dda0 |
| Polynomial | 0x04C11DB7 |
| Initial value | 0xFFFFFFFF |
| Algorithm | Shift-left (big-endian), byte-at-a-time, non-reflected |
| Coverage | TXC payload data (size in dwords × 4) |
| Comparison address | 0x405d68 |
| Patch offset | 0x5D6A (NOP `jne` to bypass) |
| Error message | "Invalid CRC32" @ 0x43109b |

```
crc32(data, len):
    crc = 0xFFFFFFFF
    for each byte:
        idx = (crc >> 24) ^ byte
        crc = (crc << 8) ^ TABLE[idx]
    return crc
```

## Decoder Batch Structure

| Property | Value |
|----------|-------|
| nc_conv_1d per batch | 32 (= 30 DAC + 2 RVQ) |
| Batch duration | ~1 second audio |
| Frames per batch | ~86-98 (86 for silence) |
| Verified files | silence1s=32, s15d6=192, clip8s=224, s15d10=288 |

## Source Files (from binary strings)

- arith.c — Range/arithmetic coder
- transformer.c — RWKV-style transformer
- nc_block.c — Neural network block operations
- cmdopt.c — CLI parsing
- cutils.c — Utilities
- grammar.c — Grammar-constrained decoding
- gpt2_tokenizer.c — GPT-2 tokenizer
- libregexp.c — Regex engine
- libunicode.c — Unicode tables

## Key PLT Addresses

| Symbol | PLT |
|--------|-----|
| nc_conv_1d | 0x4039e0 |
| nc_conv_transpose_1d | 0x403f00 |
| nc_snake | 0x403700 |
| nc_group_norm | 0x403db0 |
| nc_convert_from_old_bf | 0x403af0 |
| nc_find_param | 0x403ad0 |
| nc_load_param_header | 0x403e80 |
| nc_load_param | 0x4039b0 |
| nc_tensor_get_ptr | 0x403790 |

## Range Coder — arith.c (FULLY REVERSED Round 007-008)

### get_freq — Adaptive Probability Decoder
| Property | Value |
|----------|-------|
| Function address | 0x42bbe0 |
| Algorithm | `range0 = (range * freq) >> 15` (15-bit probability) |
| Normalization | `cmp eax, 0xFF00` (NOT 0x01000000) |
| State layout | low@+0x14, range@+0x18, buf_pos@+0x10, buf_size@+0x08 |
| Source file | arith.c (confirmed via assertion strings) |
| callers | 0x42aae3 (binary search decoder), 0x404d3f (tensor processing) |
| Frequency source | `vec_sum_f32` float table → `(32767 * partial / total)`, clamped [1,32767] |
| tsac-ng status | ✅ Implemented in `src/range_coder.c` |

### get_bit — DEAD CODE
| Property | Value |
|----------|-------|
| Function address | 0x42bd30 |
| Algorithm | `range0 = range >> 1` (50/50 fixed) |
| Status | **Never called in tsac binary** — zero references found |

### Fast-Mode TXC I/O Trace (Round 008)
| Property | Value |
|----------|-------|
| fgetc calls | 4 (version×2 + flags + nc) |
| fread calls | 2 (magic 4B + payload 68B) |
| Total I/O | **6 calls** for entire TXC file |
| Conclusion | Range coder is **in-memory** — processes 68B fread buffer internally |
| Evidence | `/tmp/gdb_fgetc_count.log`, `/tmp/callgrind_fast.out` (174KB) |

### Related Functions
| Function | Address | Purpose |
|----------|---------|---------|
| put_bit | 0x42b7d0 | Encoder main |
| put_bit_raw | 0x42b960 | Encoder raw output |
| put_bit_flush | 0x42b9xx | Encoder flush |
| range_decoder_init | 0x42bb00 | Decoder init |
| get_freq (adaptive) | 0x42bbe0 | **USED** — 15-bit probability decode |
| get_bit (fixed) | 0x42bd30 | **DEAD CODE** — never called |

## Transformer Model (tsac_stereo_q8.bin)
| Property | Value |
|----------|-------|
| File size | 47.3 MB (190 tensors) |
| Architecture | 12-layer decoder-only |
| d_model | 512, n_head=4, d_key=128 |
| Positional | RoPE (rotary_embed=true) |
| Data type | BF16, quantized BF8 |
| Codebook decoder | 3 layers, codebook_dim=1024, vocab_size=1024 |

## Normal Mode TXC Header
| Field | Bytes | Type |
|-------|-------|------|
| Magic | 0-3 | "FBAZ" |
| Version | 4-5 | BE uint16 |
| Flags | 6 | uint8 (0x80=compressed, 0x01=stereo) |
| n_codebooks | 7 | uint8 |
| n_blocks | 8-11 | BE uint32 |
| Parameter | 12-15 | BE uint32 |
| Payload | 16+ | state(N bytes) + range-coded data |
| CRC32 | last 4 | BE uint32 |

State size varies: silence=4B, music=16B (q6 normal mode)

## nc_conv_1d Signature (CRACKED Round 040)
- **Function address**: 0x70d40 in libnc.so
- **Signature**: `nc_conv_1d(output(rdi), weight(rsi), input(rdx), stride(ecx), pad(r8d), dilation(r9d), groups(stack))`
- **Weight dims**: [Ci, K, Co] = [dim0, dim1, dim2]
- **Calls per batch**: 32 (6 snake/group_norm + 26 conv/convtr)
- **Int args pattern**: [0 0 1 1] = snake, [3 3 1 1] = conv stride=3, etc.
- **Evidence**: /tmp/preload_correct.log, /tmp/libnc_conv1.bin
