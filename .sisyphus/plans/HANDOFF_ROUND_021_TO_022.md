# HANDOFF Round 021 → Round 022

## Finding
libnc's nc_convert_from_old_bf (@0x61370) combines BF8 dequant + L2 norm + weight scale into ONE operation. Our separate approach produces different results.

## Next
Re-implement dequant_weights to match libnc's combined operation.
