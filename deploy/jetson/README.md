# Jetson 部署说明

## 目标平台

- Jetson Orin Nano / Orin NX / Xavier NX（按实际硬件填写）
- JetPack 版本：与 CUDA / cuDNN 对齐

## 移植步骤

1. **4090 上完成**
   - CUDA 算子数值对齐（与 PyTorch diff < 1e-4）
   - benchmark 脚本可复现

2. **Jetson 环境**
   ```bash
   # 在 Jetson 上
   git clone <repo>
   cd mobilenetv2_jetson_accelerator
   pip install -r requirements.txt
   pip install -e .   # 编译 CUDA 扩展
   ```

3. **Benchmark**
   ```bash
   python benchmarks/benchmark_inference.py \
     --checkpoint checkpoints/best.pt \
     --batch-size 1
   ```

4. **对比 TensorRT（可选）**
   - 导出 ONNX → `trtexec` 或 Python TensorRT API
   - 与自定义 runtime 同条件对比

## Jetson 特有注意

| 项 | 说明 |
|----|------|
| FP16 | 优先测试，边缘收益明显 |
| 功耗 | `tegrastats` 记录 FPS/W |
| Launch config | 与 4090 分开调参 |
| 交叉编译 | 可选在 x86 主机 aarch64 交叉编译 |

## 文件规划

```
deploy/jetson/
├── build.sh           # 一键编译（待添加）
├── trt_export.py      # ONNX → TensorRT（待添加）
└── tegrastats_log.sh  # 功耗采集（待添加）
```
