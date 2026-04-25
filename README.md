# trash-division

一个基于卷积神经网络的垃圾分类识别系统

> 同济大学 Python 人工智能程序设计课程小组作业

基于自定义 ResNet 风格 Bottleneck 架构的 CNN 模型（约 80M 参数），将生活垃圾分为厨余垃圾、可回收物、其他垃圾、有害垃圾四个类别，输入为 256×256 RGB 图像。

---

## 目录

- [项目特点](#项目特点)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [文件说明](#文件说明)
- [目录结构](#目录结构)
- [训练细节](#训练细节)
- [许可证](#许可证)

---

## 项目特点

- **四类垃圾分类**：厨余垃圾（1）、可回收物（2）、其他垃圾（3）、有害垃圾（4）
- **自定义 ResNet Bottleneck 架构**：约 80M 参数，50 层深度残差网络
- **数据增强**：训练时使用随机裁剪、水平翻转、旋转、色彩抖动
- **Macro-F1 评估**：采用宏平均 F1 分数作为主要评估指标，兼顾各类别表现
- **类别加权损失**：自动计算类别权重，缓解类别不平衡问题
- **余弦退火学习率调度**：使用 CosineAnnealingLR 平滑调整学习率
- **断点续训**：自动检测 `best_model.pth` 并加载继续训练
- **多设备支持**：自动选择 CUDA > Intel XPU > CPU

## 模型架构

模型基于残差网络（ResNet）的 Bottleneck 构建块设计。

### Bottleneck 块

每个 Bottleneck 块包含三个卷积层：

| 层 | 卷积 | 作用 |
|---|---|---|
| 1x1 Conv | 降维 | 减少通道数，降低计算量 |
| 3x3 Conv | 特征提取 | 核心卷积操作 |
| 1x1 Conv | 升维 (x4) | 恢复通道数至输入的 4 倍 |

### 网络结构

| 阶段 | 块数 | 输出通道数 | 说明 |
|---|---|---|---|
| 初始层 | - | 64 | 7x7 Conv, stride=2 + MaxPool |
| Stage 1 | 3 | 256 | 第一个残差阶段 |
| Stage 2 | 4 | 512 | - |
| Stage 3 | 14 | 1024 | 最深阶段（比 ResNet-50 加深） |
| Stage 4 | 3 | 2048 | 最终残差阶段 |
| 分类头 | - | 4 | 全局平均池化 + 全连接层 |

## 数据集

本项目使用 [tany0699/garbage265](https://modelscope.cn/datasets/tany0699/garbage265) 中文生活垃圾分类数据集，包含 265 个子类别的生活垃圾图片。

通过 `Merge_classes.py` 脚本将 265 个子类别合并为 4 个顶级类别：

```
厨余垃圾 -> 1
可回收物 -> 2
其他垃圾 -> 3
有害垃圾 -> 4
```

数据集预期放置在 `../trash_division_data/`（与项目根目录平级的兄弟目录）。

## 环境要求

本项目无 `requirements.txt`，需手动安装以下依赖：

- Python 3.8+
- PyTorch（推荐 1.10+）
- torchvision
- tqdm
- matplotlib
- pandas
- Pillow
- torchsummary

## 快速开始

1. **数据预处理**：将 265 个子类别合并为 4 个顶级类别

   ```bash
   python Merge_classes.py
   ```

2. **训练模型**：

   ```bash
   python Train.py
   ```

> **注意**：
> - 数据目录默认为 `../trash_division_data/ultimate_4_class/`，需先运行合并脚本
> - Windows 系统需将 `num_workers` 设为 `0`（参见 `Dataloader.py` 和 `Train.py`）
> - 训练会自动从 `best_model.pth` 断点续训（若存在）

## 文件说明

| 文件 | 功能 |
|---|---|
| `Train.py` | 训练主脚本，包含训练循环、验证、评估 |
| `Dataloader.py` | 数据加载模块，包含 RobustImageFolder 和 DataLoader 创建 |
| `Model.py` | 模型定义，Bottleneck 残差块 + Net 主模型 |
| `Merge_classes.py` | 数据集预处理，265 类合并为 4 类 |
| `best_model.pth` | 训练好的最佳模型权重（约 125 MB） |
| `AGENTS.md` | AI 助手指南（开发辅助） |
| `THIRD_PARTY_LICENSES.md` | 第三方数据集许可证声明 |

## 目录结构

```
trash-division/
├── AGENTS.md               # AI 助手指南
├── best_model.pth           # 最佳模型权重
├── Dataloader.py            # 数据加载模块
├── .gitattributes           # Git 属性配置
├── LICENSE                  # MIT 许可证
├── Merge_classes.py         # 数据集预处理脚本
├── Model.py                 # 模型定义
├── README.md                # 项目说明（本文件）
├── THIRD_PARTY_LICENSES.md  # 第三方许可证声明
└── Train.py                 # 训练主脚本
```

## 训练细节

| 配置项 | 说明 |
|---|---|
| 输入尺寸 | 256 x 256 RGB |
| 优化器 | SGD（momentum=0.9, weight_decay=1e-4） |
| 初始学习率 | 0.001 |
| 学习率调度 | CosineAnnealingLR |
| 损失函数 | 类别加权 CrossEntropyLoss |
| 评估指标 | Macro-F1（宏平均 F1 分数） |
| 批量大小 | 默认 16（可通过参数调整） |
| 训练轮数 | 默认 20（可通过参数调整） |
| 设备选择优先级 | CUDA > Intel XPU > CPU |
| 断点续训 | 自动检测 best_model.pth 并加载 |

训练时数据增强管线：RandomResizedCrop(256, scale=(0.8, 1.0)) + RandomHorizontalFlip(p=0.5) + RandomRotation(+-15 deg) + ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

## 许可证

本项目主代码采用 [MIT 许可证](LICENSE)。

本项目包含的数据集 `tany0699/garbage265` 采用 [Apache License 2.0](THIRD_PARTY_LICENSES.md)，详情请参阅 `THIRD_PARTY_LICENSES.md` 文件。
