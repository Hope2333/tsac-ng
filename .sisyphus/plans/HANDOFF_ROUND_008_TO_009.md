# HANDOFF Round 008 → Round 009

**Date**: 2026-05-25
**From**: Sisyphus (Round 008)
**To**: Round 009

## Primary Deliverable
**Fast-mode range coder is in-memory** — not I/O-based. TXC read: 2 fread + 4 fgetc = 6 total I/O calls for the entire file.

### Evidence
- `/tmp/gdb_fgetc_count.log` — fgetc=4, fread=2 (GDB Python breakpoint counter)
- 4 fgetc = version(2 bytes) + flags(1) + n_codebooks(1)
- 2 fread = magic "FBAZ"(4 bytes) + payload(68 bytes)
- 68-byte fread buffer processed entirely in memory

### Callgrind Profiles
- `/tmp/callgrind_fast.out` (174KB) — fast mode decode profile
- `/tmp/callgrind_norm.out` (174KB) — normal mode decode profile

## Critical Context for Round 009
- get_freq/vec_sum_f32 NOT called in fast mode (confirmed rounds 007+008)
- Range coder operates in-memory on the 68-byte fread buffer
- Index decoding happens between TXC fclose (0x4055fe) and decoder dispatch (0x405d3c)
- The range coder function is an internal (non-PLT, non-exported) function
- 190 PLT entries catalogued in `/tmp/plt_list.txt`

## Next Steps (Round 009)
1. Identify the exact in-memory range coder function address using callgrind hot paths
2. Determine probability model (adaptive vs direct, frequency table source)
3. Fix audio output correctness
