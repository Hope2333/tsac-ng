#!/bin/bash
# 对比 tsac-ng 和原版 tsac 的解码输出

set -e

TXC_FILE="${1:-test-simples/P丸様。-自分後回し@A.txc}"
MODEL_FILE="${2:-/usr/share/tsac/dac_stereo_q8.bin}"
REF_WAV="/tmp/tsac_ref_$$.wav"
TEST_WAV="/tmp/tsac_ng_test_$$.wav"

# 检查文件
if [ ! -f "$TXC_FILE" ]; then
    echo "错误: 找不到 TXC 文件: $TXC_FILE"
    exit 1
fi

echo "===== tsac-ng 解码对比测试 ====="
echo "TXC 文件: $TXC_FILE"
echo "模型文件: $MODEL_FILE"
echo ""

# 1. 使用原版 tsac 解码生成参考
echo "[1/3] 使用原版 tsac 解码..."
/usr/bin/tsac -v d "$TXC_FILE" "$REF_WAV" 2>&1 | tail -5
echo ""

# 2. 使用 tsac-ng 解码
echo "[2/3] 使用 tsac-ng 解码..."
if [ -f "./build/tsac-ng" ]; then
    LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH ./build/tsac-ng -v d "$TXC_FILE" "$TEST_WAV" 2>&1 | tail -10
elif [ -f "./tsac-ng" ]; then
    LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./tsac-ng -v d "$TXC_FILE" "$TEST_WAV" 2>&1 | tail -10
else
    echo "错误: 找不到 tsac-ng 可执行文件"
    exit 1
fi
echo ""

# 3. 对比输出
echo "[3/3] 对比解码结果..."
if [ -f "./wav_compare" ]; then
    ./wav_compare "$REF_WAV" "$TEST_WAV" 10000
else
    echo "警告: 找不到 wav_compare，跳过详细对比"
    ls -la "$REF_WAV" "$TEST_WAV"
fi

# 清理临时文件
rm -f "$REF_WAV" "$TEST_WAV"
