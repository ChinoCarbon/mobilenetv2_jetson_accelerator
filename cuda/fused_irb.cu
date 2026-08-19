#include "include/common.cuh"

namespace mnv2 {

/*
 * ============================================================
 * Fused Inverted Residual Block（IRB）
 * ============================================================
 *
 * MobileNetV2 IRB 结构：
 *
 *   input [N, C_in, H, W]
 *     │
 *     ├─ 1×1 Expand Conv  →  [N, C_in*t, H, W]      (t = expansion factor, 通常 6)
 *     │    + BN + ReLU6
 *     │
 *     ├─ 3×3 Depthwise Conv  →  [N, C_in*t, H', W']  (stride 1 或 2)
 *     │    + BN + ReLU6
 *     │
 *     └─ 1×1 Project Conv  →  [N, C_out, H', W']     (线性，无激活)
 *          + BN
 *     │
 *     └─ residual add（仅当 stride==1 且 C_in==C_out 时）
 *
 * 注意：BN 在推理时已 fold 进 weight/bias，不需要单独处理。
 *
 * ------------------------------------------------------------
 * 分两个阶段实现，先做阶段一跑正确，再做阶段二：
 *
 * 阶段一：三个算子各自独立 kernel，串行调用
 *   kernel_expand_relu6   : 1×1 conv + ReLU6
 *   kernel_depthwise_relu6: 3×3 depthwise + ReLU6  (复用 depthwise_conv.cu 的逻辑)
 *   kernel_project        : 1×1 conv（无激活）+ residual add
 *
 *   好处：逐步验证，每步都能和 PyTorch 对齐
 *
 * 阶段二：融合 kernel（创新点核心）
 *   将 Expand + Depthwise + ReLU6 合并为一个 kernel
 *   目标：减少中间 feature map 对 global memory 的读写次数
 *
 *   思路（供参考）：
 *   - 一个 thread block 负责输出的一个 spatial tile
 *   - Expand 结果写入 shared memory，不落 global memory
 *   - 直接在 shared memory 上做 depthwise，结果写出
 *
 * ------------------------------------------------------------
 * 你需要实现什么（阶段一优先）：
 *
 * 1. kernel_expand_relu6
 *    - 本质是 1×1 GEMM：output[n,c_out,h,w] = sum_k(input[n,k,h,w] * w[c_out,k])
 *    - 每个 thread 计算一个输出元素，或用 tiled GEMM
 *    - ReLU6：out = min(max(out + bias, 0), 6)
 *
 * 2. kernel_depthwise_relu6
 *    - 和 depthwise_conv.cu 一致，额外加 ReLU6 输出
 *    - 可以直接复用，加一行 out = min(max(out, 0.f), 6.f)
 *
 * 3. kernel_project
 *    - 同 1×1 GEMM，无激活
 *    - 若 use_residual==true，加上 residual add
 *
 * 4. host 侧 fused_irb_fp32 函数（下方 stub）
 *    - 按顺序调用三个 kernel
 *    - 管理中间 buffer（expand 输出、dw 输出）
 *    - 传同一个 stream 保证顺序
 *
 * 验证方式：
 *   用 PyTorch 跑一个 InvertedResidual block（torchvision 里有），
 *   输入相同权重，对比整个 block 的输出，max diff < 1e-4
 * ============================================================
 */

void fused_irb_fp32(
    const float* input,
    float* output,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream) {
  // TODO: 在这里实现（先实现阶段一三个独立 kernel，再融合）
  (void)input;
  (void)output;
  (void)batch;
  (void)channels;
  (void)height;
  (void)width;
  (void)stream;
}

}  // namespace mnv2
