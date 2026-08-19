# 数据集目录

本目录用于存放分类数据集，**不纳入 git**（大文件请自行管理）。

## 数据集准备

### 方案一：CIFAR-10（快速验证，32px 自动 resize）

```bash
python data/prepare_cifar10.py
# 输出: data/cifar10_imagefolder/   10 类，train 50000 / val 10000
```

训练：
```bash
python train/train_classifier.py --data-dir data/cifar10_imagefolder
```

---

### 方案二：ImageNette（原生大图，推荐替代 CIFAR-10）

```bash
python data/prepare_imagenette.py --size 320
# 下载约 600 MB，输出: data/imagenette/   10 类，train ~9500 / val ~3900
```

训练：
```bash
python train/train_classifier.py --data-dir data/imagenette
```

默认下载 320px 版本（训练时再 resize 到 224），`--size full` 可下载原图。

---

### 方案三：NEU-DET（钢材缺陷，毕设正式实验）

NEU-DET 需手动下载（[官方页面](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html)），
下载解压后执行：

```bash
python data/prepare_neu_det.py --raw-dir /path/to/NEU-DET
# 输出: data/neu_det_cls/   6 类，按 8:2 自动划分 train/val
```

训练：
```bash
python train/train_classifier.py --data-dir data/neu_det_cls
```

## 目录格式（ImageFolder）

```
data/
└── <dataset_name>/
    ├── train/
    │   ├── <class_0>/
    │   │   ├── 00001.png
    │   │   └── ...
    │   └── <class_1>/
    └── val/
        ├── <class_0>/
        └── ...
```

## 其他数据集

### ImageNette（大图，自动下载）

```bash
python data/prepare_imagenette.py          # 下载 320px 版，约 600 MB
python data/prepare_imagenette.py --size full  # 原始大图，约 1.4 GB
```

输出至 `data/imagenette/`，10 类，train ~9400 张，val ~3900 张。

### NEU-DET（钢材缺陷，毕设正式实验）

1. 前往 [NEU-DET 官网](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) 下载压缩包并解压
2. 运行整理脚本：

```bash
python data/prepare_neu_det.py --raw-dir /path/to/NEU-DET
```

输出至 `data/neu_det_cls/`，6 类（crazing / inclusion / patches / pitted / rolled / scratches），共 1800 张，8:2 划分。

---

切换数据集只需修改 `configs/default.yaml` 中的 `data.dir`，无需改训练脚本。

| 数据集 | 配置 `data.dir` | 类别数 | 用途 |
|--------|-----------------|--------|------|
| CIFAR-10 | `data/cifar10_imagefolder` | 10 | 快速调通 |
| ImageNette | `data/imagenette` | 10 | 大图验证 |
| NEU-DET | `data/neu_det_cls` | 6 | 毕设正式实验 |
