# 毕设设计文档

**题目方向**：面向工业边缘缺陷检测的 MobileNetV2 CUDA 算子加速与 Jetson 部署  
**任务类型**：图像分类（后续可扩展检测）  
**开发平台**：RTX 4090 → Jetson

---

## 1. 问题定义

工业边缘缺陷检测需要在有限算力与显存下，以低延迟完成在线推理。MobileNetV2 结构轻量、深度可分离卷积占比高，适合作为边缘基线模型，但在 batch=1、小分辨率场景下，通用框架（PyTorch Eager）开销偏大，内存带宽成为瓶颈。

**目标**：针对 MobileNetV2 热点算子实现 CUDA 加速与算子融合，在精度不变的前提下降低单张推理延迟与峰值显存，并在 Jetson 上验证边缘部署价值。

**约束**

| 维度 | 要求 |
|------|------|
| 输入 | 224×224（主实验），可扩展 128/320 |
| Batch | 1（边缘主场景），可选 8/16 |
| 精度 | 相对 FP32 PyTorch 基线，F1 下降 < 0.5% |
| 部署 | 同一套代码，4090 调优后移植 Jetson |

---

## 2. 技术路线

```
数据集与分类基线 → 算子 Profiling → CUDA 算子实现
        → 推理运行时 → 4090 验证 → Jetson 移植 → 对比与消融
```

**原则**：Profiling 驱动优化，不做全网络重写；4090 上保证数值正确性与性能，再迁移 Jetson。

---

## 3. 创新点

### 3.1 主创新：倒残差块（IRB）算子融合 CUDA 实现

MobileNetV2 核心单元：

```
1×1 Expand → 3×3 Depthwise → 1×1 Project (+ residual)
```

**工作**：
- 将 Expand + Depthwise + ReLU6 融合为单 kernel，减少中间 feature map 读写
- 针对小通道数、小 spatial size（工业 ROI 常见）设计 thread/block 映射
- 可选进一步融合 Project 与 residual add

**与 TensorRT 差异**：黑盒自动融合 vs 可解释、可消融、可针对缺陷检测输入尺寸手工调参。

### 3.2 次创新：显存/带宽感知推理调度

Jetson 瓶颈常在 DRAM 带宽而非算力。

**工作**：
- 静态内存池，推理过程零 `cudaMalloc`
- 层间 buffer 复用
- 4090 / Jetson 分平台 launch config 表

### 3.3 应用创新：缺陷检测场景评测协议

不仅报告 FPS，还包括：
- 固定召回率下的延迟（如 recall ≥ 95%）
- 精度–延迟 Pareto 曲线
- Jetson 功耗与能效（FPS/W）

---

## 4. 实现方案

### 阶段 0：分类基线（当前）

- 数据集：NEU-DET 等，按 ImageFolder 格式组织（见 `data/README.md`）
- 模型：`models/mobilenetv2.py`，ImageNet 预训练 + 微调
- 基线推理：PyTorch Eager，记录 accuracy / F1 / latency

### 阶段 1：Profiling

工具：`torch.profiler`、Nsight Systems

预期热点：
1. Depthwise Conv3×3（~40–60%）
2. Pointwise Conv1×1（~20–30%）
3. ReLU6 / Add

产出：算子耗时饼图，确定优化优先级。

### 阶段 2：CUDA 算子

| 优先级 | 算子 | 要点 |
|--------|------|------|
| P0 | Depthwise Conv3×3 | direct conv 或 im2col；groups=C |
| P0 | Fused IRB | Expand+DW+ReLU6 融合 |
| P1 | Pointwise 1×1 | GEMM 化 |
| P2 | ReLU6 + Add | element-wise 融合 |

集成：`torch.utils.cpp_extension`，见 `bindings/` 与 `setup.py`。

### 阶段 3：推理运行时

轻量算子图执行器：层配置 → CUDA kernel 调度 → 内存池 → 输出对齐验证（误差 < 1e-4）。

### 阶段 4：Jetson 移植

- 对齐 JetPack CUDA 版本
- FP16 为主实验精度；INT8 作对比
- 分平台 launch config

---

## 5. 对比策略

### 5.1 对比对象

| 方法 | 角色 |
|------|------|
| PyTorch Eager | 易用性上界 |
| ONNX Runtime (CUDA) | 通用推理框架 |
| TensorRT FP32/FP16 | 工业强基线 |
| **Ours（CUDA Runtime）** | 主方法 |

### 5.2 指标

**性能**：p50/p95/p99 延迟（batch=1）、吞吐（batch=8）、峰值显存、Jetson 功耗

**精度**：Accuracy、F1、与 FP32 数值误差

**公平性**：同一权重、同一输入尺寸、warmup 100 轮、`cudaEvent` 计时

### 5.3 结果表示例

| 方法 | 平台 | F1(%) | Latency p50(ms) | 显存(MB) |
|------|------|-------|-----------------|----------|
| PyTorch | 4090 | — | — | — |
| TensorRT FP16 | 4090 | — | — | — |
| Ours | 4090 | — | — | — |
| TensorRT FP16 | Jetson | — | — | — |
| Ours | Jetson | — | — | — |

---

## 6. 消融实验

### 6.1 算子替换

| ID | 配置 |
|----|------|
| A0 | 全 PyTorch 基线 |
| A1 | 仅 Depthwise → CUDA |
| A2 | Depthwise + Pointwise → CUDA |
| A3 | Fused IRB（无跨层融合） |
| A4 | Fused IRB（完整融合，主方法） |

### 6.2 融合粒度

无融合 → 部分融合（Expand+DW） → 完全融合（Expand+DW+ReLU6+Project+Add）

### 6.3 精度

FP32 / FP16 / INT8（校准集）对 F1 的影响

### 6.4 输入尺寸

128 / 224 / 320：延迟 vs F1，论证小 ROI 场景优势

### 6.5 内存策略

动态分配 vs 内存池 vs 层间复用

### 6.6 平台迁移

4090 vs Jetson：加速比、瓶颈是否从 compute 转为 memory

---

## 7. 论文结构

1. 绪论：工业边缘缺陷检测 + 轻量模型 + 自定义加速动机
2. 相关工作：MobileNet、CUDA 算子优化、TensorRT、边缘推理
3. 方法：IRB 融合 kernel、内存调度、推理框架
4. 实验：数据集、环境、对比、消融
5. 总结：贡献与局限

---

## 8. 风险与 Scope

| 风险 | 应对 |
|------|------|
| 自定义 CUDA 慢于 TensorRT | 强调小图/batch=1、内存优势、可解释性 |
| Jetson 调试慢 | 4090 完成正确性后再移植 |
| 检测任务过重 | 主实验分类，检测作扩展 |
| INT8 精度损失 | 主实验 FP16，INT8 作对比 |

---

## 9. 里程碑

| 周次 | 任务 | 产出 |
|------|------|------|
| 1–2 | 数据集 + 分类基线 | `checkpoints/best.pt`、基线指标 |
| 3 | Profiling | 算子耗时报告 |
| 4–7 | CUDA 算子 + 运行时 | 4090 数值对齐 + 加速 |
| 8–9 | Jetson 移植 | 边缘 benchmark |
| 10 | 对比 + 消融 + 论文 | 完整实验表格 |
