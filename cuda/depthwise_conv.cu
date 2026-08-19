#include "include/common.cuh"

namespace mnv2 {

/*
 * ============================================================
 * Depthwise Conv2d 3×3
 * ============================================================
 *
 * 功能：
 *   对每个输入通道独立做 3×3 卷积（groups = C_in），
 *   不跨通道累加，是 MobileNetV2 IRB 中最耗时的算子。
 *
 * 输入张量（NCHW，连续）：
 *   input   [N, C, H, W]
 *   weight  [C, 1, 3, 3]   每通道独立一组 9 个权重
 *   bias    [C]             可为 nullptr（无 bias 时跳过）
 *
 * 输出张量：
 *   output  [N, C, H_out, W_out]
 *   H_out = (H + 2*padding - 3) / stride + 1
 *   W_out = (W + 2*padding - 3) / stride + 1
 *
 * 参数：
 *   stride  = 1 或 2（MobileNetV2 中仅这两种）
 *   padding = 1（保持 same 尺寸时）
 *
 * ------------------------------------------------------------
 * 你需要实现什么：
 *
 * 1. __global__ kernel 函数
 *    - 推荐 thread 映射：每个 thread 负责输出的一个 (n, c, oh, ow)
 *    - block 维度建议：(32, 4, 1) 或 (16, 8, 1)，先跑正确再调
 *    - grid 维度：覆盖所有 (N, C, H_out, W_out)
 *
 * 2. 边界处理
 *    - 输入坐标 ih = oh*stride - padding + kh，需判断是否越界
 *    - 越界时按 0 处理（zero padding）
 *
 * 3. 可选优化（先把朴素版跑正确）
 *    - 将 weight 放入 __constant__ memory（9 个 float × C，适合小 C）
 *    - 用 shared memory 预加载输入 tile，减少 global memory 访问
 *
 * 4. host 侧包装函数（下方 stub）
 *    - 计算 H_out, W_out
 *    - 配置 grid/block
 *    - 调用 kernel，传 stream
 *
 * 验证方式：
 *   用 PyTorch 的 F.conv2d(input, weight, bias, stride, padding, groups=C)
 *   对比输出，max diff < 1e-4 即为正确
 * ============================================================
 */

void depthwise_conv3x3_fp32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int channels,
    int height,
    int width,
    int stride,
    int padding,
    cudaStream_t stream) {
  // TODO: 在这里实现
  (void)input;
  (void)weight;
  (void)bias;
  (void)output;
  (void)batch;
  (void)channels;
  (void)height;
  (void)width;
  (void)stride;
  (void)padding;
  (void)stream;
}

}  // namespace mnv2
