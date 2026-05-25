# HANDOFF Round 020 → Round 021

## Finding
Weighted blending (dominant=index/128, blend=(index%128)/128) no improvement (-3.4dB).

## Discovery
Model uses BF8 encoding. GDB shows dequantized float32. Our BF8 formula may differ from libnc nc_convert_from_old_bf.

## Next
Verify BF8 dequantization against libnc. Compare dequantized weights byte-by-byte.
