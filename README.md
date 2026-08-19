# MobileNetV2 CUDA 加速器

面向工业边缘缺陷检测的 MobileNetV2 分类推理加速，先在 4090 上开发与验证，最终部署到 Jetson。

## 项目结构

```
mobilenetv2_jetson_accelerator/
├── configs/           # 训练与 benchmark 配置
├── cuda/              # 自定义 CUDA 算子
├── bindings/          # PyTorch C++ 扩展入口
├── data/              # 数据集目录（见 data/README.md）
├── docs/              # 设计文档与实验记录
├── models/            # MobileNetV2 分类模型
├── train/             # 训练脚本
├── benchmarks/        # 性能测试
├── deploy/jetson/     # Jetson 部署说明
└── checkpoints/       # 模型权重（git 忽略）
```

## 快速开始

### 1. 环境

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 准备数据集

将分类数据集按 ImageFolder 格式放入 `data/`（详见 [data/README.md](data/README.md)）：

```
data/
└── neu_det_cls/          # 示例名称，可自定义
    ├── train/
    │   ├── class_a/
    │   └── class_b/
    └── val/
        ├── class_a/
        └── class_b/
```

### 3. 训练基线

```bash
python train/train_classifier.py \
  --data-dir data/neu_det_cls \
  --epochs 30 \
  --batch-size 64
```

### 4. 推理 Benchmark

```bash
python benchmarks/benchmark_inference.py \
  --checkpoint checkpoints/best.pt \
  --input-size 224
```

### 5. 编译 CUDA 扩展（开发中）

```bash
pip install -e .
```

## 文档

- [毕设设计文档](docs/design.md)：创新点、实现路线、对比与消融实验设计

## 硬件路线

| 阶段 | 平台 | 用途 |
|------|------|------|
| 开发 | RTX 4090 | 算子实现、正确性验证、性能调优 |
| 部署 | Jetson | 边缘延迟、显存、功耗评测 |
