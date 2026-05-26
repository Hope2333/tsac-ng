/*
 * wav_compare.c — 对比两个 WAV 文件的差异
 * 用于验证 tsac-ng 解码输出与原版 tsac 的一致性
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    int sample_rate;
    int channels;
    int bits_per_sample;
    int data_size;
    int num_samples;
    int16_t *data;  // 统一转换为 int16
} WAVFile;

static int read_wav(const char *path, WAVFile *wav) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "无法打开: %s\n", path); return -1; }
    
    char riff[4], wave[4], fmt[4], data[4];
    fread(riff, 1, 4, f);
    fread(&wav->data_size, 4, 1, f);
    fread(wave, 1, 4, f);
    
    if (memcmp(riff, "RIFF", 4) != 0 || memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "不是有效的 WAV 文件: %s\n", path);
        fclose(f);
        return -1;
    }
    
    // 查找 fmt chunk
    int fmt_size, audio_format;
    while (1) {
        fread(fmt, 1, 4, f);
        if (memcmp(fmt, "fmt ", 4) == 0) break;
        int skip_size;
        fread(&skip_size, 4, 1, f);
        fseek(f, skip_size, SEEK_CUR);
    }
    
    fread(&fmt_size, 4, 1, f);
    fread(&audio_format, 2, 1, f);
    fread(&wav->channels, 2, 1, f);
    fread(&wav->sample_rate, 4, 1, f);
    fseek(f, 4, SEEK_CUR); // byte_rate
    fseek(f, 2, SEEK_CUR); // block_align
    fread(&wav->bits_per_sample, 2, 1, f);
    fseek(f, fmt_size - 16, SEEK_CUR);
    
    // 查找 data chunk
    while (1) {
        fread(data, 1, 4, f);
        if (memcmp(data, "data", 4) == 0) break;
        int skip_size;
        fread(&skip_size, 4, 1, f);
        fseek(f, skip_size, SEEK_CUR);
    }
    
    fread(&wav->data_size, 4, 1, f);
    
    int bytes_per_sample = wav->bits_per_sample / 8;
    wav->num_samples = wav->data_size / (wav->channels * bytes_per_sample);
    wav->data = (int16_t *)malloc(wav->num_samples * wav->channels * sizeof(int16_t));
    
    if (audio_format == 1) {  // PCM int16
        if (wav->bits_per_sample == 16) {
            fread(wav->data, 2, wav->num_samples * wav->channels, f);
        } else if (wav->bits_per_sample == 24) {
            for (int i = 0; i < wav->num_samples * wav->channels; i++) {
                uint8_t b[3];
                fread(b, 1, 3, f);
                int32_t v = (b[2] & 0x80) ? (0xFF000000 | (b[2] << 16) | (b[1] << 8) | b[0])
                                          : ((b[2] << 16) | (b[1] << 8) | b[0]);
                wav->data[i] = (int16_t)(v >> 8);
            }
        }
    } else if (audio_format == 3) {  // IEEE float
        for (int i = 0; i < wav->num_samples * wav->channels; i++) {
            float f32;
            fread(&f32, 4, 1, f);
            int v = (int)(f32 * 32768.0f);
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            wav->data[i] = (int16_t)v;
        }
    }
    
    fclose(f);
    return 0;
}

static void free_wav(WAVFile *wav) {
    free(wav->data);
    wav->data = NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("用法: %s <reference.wav> <test.wav> [max_samples]\n", argv[0]);
        printf("对比两个 WAV 文件的差异\n");
        return 1;
    }
    
    WAVFile ref = {0}, test = {0};
    
    if (read_wav(argv[1], &ref) != 0) return 1;
    if (read_wav(argv[2], &test) != 0) { free_wav(&ref); return 1; }
    
    int max_samples = (argc > 3) ? atoi(argv[3]) : ref.num_samples;
    if (max_samples > ref.num_samples) max_samples = ref.num_samples;
    if (max_samples > test.num_samples) max_samples = test.num_samples;
    
    printf("===== WAV 对比报告 =====\n");
    printf("参考文件: %s\n", argv[1]);
    printf("测试文件: %s\n", argv[2]);
    printf("\n文件信息:\n");
    printf("  参考: %d Hz, %d ch, %d bits, %d samples\n", 
           ref.sample_rate, ref.channels, ref.bits_per_sample, ref.num_samples);
    printf("  测试: %d Hz, %d ch, %d bits, %d samples\n",
           test.sample_rate, test.channels, test.bits_per_sample, test.num_samples);
    
    if (ref.sample_rate != test.sample_rate || ref.channels != test.channels) {
        printf("\n错误: 格式不匹配!\n");
        free_wav(&ref); free_wav(&test);
        return 1;
    }
    
    // 计算差异统计
    double sum_sq_diff = 0;
    double sum_sq_ref = 0;
    int max_diff = 0;
    int max_diff_idx = 0;
    int zero_crossings = 0;
    int sign_diffs = 0;
    
    for (int i = 0; i < max_samples * ref.channels; i++) {
        int diff = abs(ref.data[i] - test.data[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        sum_sq_diff += (double)diff * diff;
        sum_sq_ref += (double)ref.data[i] * ref.data[i];
        
        if ((ref.data[i] >= 0) != (test.data[i] >= 0)) sign_diffs++;
        if (i > 0 && (ref.data[i] >= 0) != (ref.data[i-1] >= 0)) zero_crossings++;
    }
    
    double mse = sum_sq_diff / (max_samples * ref.channels);
    double rmse = sqrt(mse);
    double snr = (sum_sq_ref > 0) ? 10.0 * log10(sum_sq_ref / sum_sq_diff) : 0;
    
    printf("\n差异统计 (前 %d samples):\n", max_samples);
    printf("  RMSE: %.2f\n", rmse);
    printf("  最大差异: %d (sample %d)\n", max_diff, max_diff_idx / ref.channels);
    printf("  SNR: %.2f dB\n", snr);
    printf("  符号不一致: %d / %d (%.2f%%)\n", sign_diffs, max_samples * ref.channels,
           100.0 * sign_diffs / (max_samples * ref.channels));
    
    // 样本对比
    printf("\n前 20 个样本对比:\n");
    printf("  Idx    参考        测试        差异\n");
    for (int i = 0; i < 20 && i < max_samples * ref.channels; i++) {
        int diff = ref.data[i] - test.data[i];
        printf("  %4d:  %6d  vs  %6d  =  %6d %s\n", 
               i, ref.data[i], test.data[i], diff,
               abs(diff) > 1000 ? "***" : abs(diff) > 100 ? "**" : abs(diff) > 10 ? "*" : "");
    }
    
    // 最大差异点附近的样本
    if (max_diff > 100) {
        printf("\n最大差异点 (sample %d) 附近:\n", max_diff_idx / ref.channels);
        int start = (max_diff_idx / ref.channels - 5) * ref.channels;
        if (start < 0) start = 0;
        for (int i = start; i < start + 20 && i < max_samples * ref.channels; i += ref.channels) {
            printf("  %4d:  ", i / ref.channels);
            for (int c = 0; c < ref.channels; c++) {
                int idx = i + c;
                int diff = ref.data[idx] - test.data[idx];
                printf("[%6d vs %6d = %6d]  ", ref.data[idx], test.data[idx], diff);
            }
            printf("\n");
        }
    }
    
    // 评估结果
    printf("\n===== 评估 =====\n");
    if (max_diff == 0) {
        printf("✓ 完全一致!\n");
    } else if (max_diff <= 1) {
        printf("✓ 基本正确 (舍入误差)\n");
    } else if (max_diff <= 10) {
        printf("⚠ 轻微差异 (可接受)\n");
    } else if (max_diff <= 100) {
        printf("⚠ 明显差异 (需检查)\n");
    } else {
        printf("✗ 严重差异 (解码错误)\n");
    }
    
    free_wav(&ref);
    free_wav(&test);
    return (max_diff > 100) ? 2 : (max_diff > 10) ? 1 : 0;
}
