# 无条件生成 Pipeline - Med-D3CG Unconditional

## 📌 概述

本项目基于Med-D3CG框架，实现了一个完整的**无条件生成（Unconditional Generation）pipeline**。相比原有的条件生成（依赖CT-MRI等医学图像对），无条件生成直接在小波域学习生成特定医学图像的分布，无需输入条件。

### 核心特性

- **小波变换加速**：使用离散小波变换（DWT）直接在小波域进行扩散，大幅加速生成过程
- **多种小波类型**：支持Haar、DB4、Coif3等多种小波基函数
- **灵活的配置系统**：支持单层/多层小波变换，可自定义所有超参数
- **独立的训练和生成脚本**：完全独立于条件生成流程

## 🏗️ 架构差异

### 条件生成 vs 无条件生成

|  | 条件生成 | 无条件生成 |
|--|---------|----------|
| **输入** | 2个图像 (条件+目标) | 1个图像 (目标) |
| **模型输入** | [d_t, 条件系数] | [d_t] |
| **差异域** | 是 | 否 |
| **使用场景** | 医学图像转换 (CT→MRI) | 特定医学图像生成 |
| **小波系数** | 4倍通道数 (差异) | 原始通道数 (图像) |

## 📁 新增文件

```
Med-D3CG/
├── unconditional_train.py          # 无条件训练脚本
├── unconditional_generation.py     # 无条件生成脚本
└── UNCONDITIONAL_GUIDE.md         # 本文档
```

## 🚀 快速开始

### 1. 数据准备

创建一个包含医学图像的数据目录，支持两种格式：

#### 格式 A：直接文件
```
data/training_images/
├── image_001.png
├── image_002.png
├── image_003.png
└── ...
```

#### 格式 B：分文件夹
```
data/training_images/
├── sample_001/
│   └── image.png
├── sample_002/
│   └── image.png
└── ...
```

### 2. 训练模型

#### 基础训练（使用预定义的无条件D3CG模型）

```bash
python unconditional_train.py \
    --model_name D3CG_uncond_db4 \
    --dataset_dir ./data/training_images \
    --val_dataset_dir ./data/validation_images \
    --out_name D3CG_uncond_db4_model \
    --batch_size 4 \
    --n_epochs 1000 \
    --image_size 256
```

#### 自定义配置训练

```bash
python unconditional_train.py \
    --model_name D3CG_custom_uncond \
    --dataset_dir ./data/training_images \
    --wave_type coif3 \
    --transform_levels 2 \
    --batch_size 8 \
    --n_epochs 1500 \
    --image_size 512 \
    --lr 1e-5
```

#### RGB 图像训练

```bash
python unconditional_train.py \
    --model_name D3CG_uncond_db4 \
    --dataset_dir ./data/training_images_rgb \
    --is_rgb \
    --batch_size 2 \
    --image_size 256
```

### 3. 生成新图像

#### 基础生成

```bash
python unconditional_generation.py \
    --model_name D3CG_uncond_db4 \
    --model_weight_path ./results/unconditional/D3CG_uncond_db4/best_model_epoch_0500.pt \
    --num_samples 10 \
    --image_size 256 \
    --output_dir ./results/generated_images/
```

#### RGB 图像生成

```bash
python unconditional_generation.py \
    --model_name D3CG_uncond_db4 \
    --model_weight_path ./results/unconditional/D3CG_uncond_db4/best_model_epoch_0500.pt \
    --is_rgb \
    --num_samples 20 \
    --output_dir ./results/generated_rgb_images/
```

#### 自定义配置生成

```bash
python unconditional_generation.py \
    --model_name D3CG_custom_uncond \
    --wave_type coif3 \
    --transform_levels 2 \
    --num_samples 15 \
    --image_size 512
```

## 📊 模型配置

### 预定义的无条件D3CG模型

| 模型名称 | 小波类型 | 变换层数 | 描述 |
|---------|--------|--------|------|
| D3CG_uncond_haar | Haar | 1 | 快速、内存轻量 |
| D3CG_uncond_db4 | DB4 | 1 | 平衡性能和质量 ⭐ |
| D3CG_uncond_coif3 | Coif3 | 1 | 高质量生成 |
| D3CG_uncond_twice_haar | Haar | 2 | 深层特征提取 |
| D3CG_uncond_twice_db4 | DB4 | 2 | 高级深层处理 |
| D3CG_custom_uncond | 可配置 | 可配置 | 自定义配置 |

### 小波类型对比

| 小波 | 平滑度 | 处理速度 | 推荐场景 |
|-----|-------|--------|--------|
| Haar | 低 | 最快 | 粗粒度结构 |
| DB4 | 中 | 快 | 细节结构 (推荐) |
| Coif3 | 高 | 中等 | 光滑区域 |
| Bior2.2 | 高 | 中等 | 精细纹理 |

## 🔧 参数详解

### 训练参数

```bash
python unconditional_train.py \
    # 数据配置
    --dataset_dir ./data/images              # 训练数据路径
    --val_dataset_dir ./data/val_images      # 验证数据路径 (可选)
    --is_rgb                                 # RGB模式标志
    --image_size 256                         # 输入图像大小
    
    # 模型配置
    --model_name D3CG_custom_uncond          # 模型名称
    --wave_type db4                          # 小波类型
    --transform_levels 1                     # 变换层数
    --ch 128                                 # 基础通道数
    --ch_mult 1 2 3 4                        # 通道乘数
    --num_res_blocks 2                       # 残差块数量
    --dropout 0.3                            # Dropout比例
    
    # 训练参数
    --batch_size 4                           # 批次大小
    --n_epochs 1000                          # 总训练轮数
    --lr 1e-5                                # 学习率
    --beta_1 1e-4                            # 扩散起始噪声
    --beta_T 0.02                            # 扩散结束噪声
    --T 1000                                 # 扩散步数
    --grad_clip 1.0                          # 梯度裁剪
    
    # 保存和验证
    --save_weight_dir ./results/             # 权重保存路径
    --save_interval 50                       # 每N个epoch保存一次
    --val_start_epoch 100                    # 验证开始epoch
    --val_num 5                              # 生成样本数量
```

### 生成参数

```bash
python unconditional_generation.py \
    # 模型配置
    --model_name D3CG_custom_uncond          # 模型名称
    --model_weight_path ./path/to/weights.pt # 权重文件路径
    --wave_type db4                          # 小波类型
    --transform_levels 1                     # 变换层数
    
    # 生成参数
    --num_samples 10                         # 生成样本数
    --batch_size 1                           # 批处理大小
    --image_size 256                         # 生成图像大小
    --is_rgb                                 # RGB模式标志
    
    # 扩散参数
    --T 1000                                 # 扩散步数
    --beta_1 1e-4                            # 起始噪声
    --beta_T 0.02                            # 结束噪声
    
    # 输出
    --output_dir ./results/generated/        # 输出目录
```

## 💾 训练输出

训练完成后，在 `--save_weight_dir` 目录下会生成：

```
results/unconditional/D3CG_uncond_db4/
├── training_log.log                    # 训练日志
├── best_model_epoch_0500.pt           # 最优模型
├── model_epoch_0100.pt                # 周期保存点 1
├── model_epoch_0200.pt                # 周期保存点 2
...
├── epoch_0100_samples/                # 验证生成的样本
│   ├── sample_000.png
│   ├── sample_001.png
│   └── ...
└── epoch_0200_samples/
    ├── sample_000.png
    └── ...
```

## 📈 性能指标

### 训练监控

在训练过程中，可以监控以下指标：

- **Loss**: 均方误差损失（MSE），应该逐渐下降
- **生成样本**: 每个验证间隔生成的样本质量
- **模型参数**: 使用Profiling工具测量

### 推荐的训练参数

| 数据集大小 | 批次大小 | 学习率 | 推荐轮数 | 内存需求 |
|----------|---------|-------|--------|--------|
| < 500 | 2-4 | 1e-5 | 500-1000 | 8GB |
| 500-2000 | 4-8 | 1e-5 | 1000-2000 | 16GB |
| > 2000 | 8-16 | 5e-6 | 2000+ | 24GB+ |

## 🔄 无条件生成的工作流

```
┌─────────────────────┐
│  训练医学图像集合    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  DWT (小波变换)      │  
│  图像 → 小波系数     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   扩散过程           │
│ (1000步逐步加噪)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  UNet模型学习        │
│ 去噪小波系数         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   逆向扩散生成       │
│ (噪声→去噪→图像)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  IDWT (逆小波变换)   │
│  小波系数 → 图像     │
└─────────────────────┘
```

## 🐛 故障排除

### 内存不足

```bash
# 减小批次大小
--batch_size 1

# 减小图像分辨率
--image_size 128

# 减少通道数
--ch 64
```

### 生成质量差

```bash
# 增加扩散步数
--T 2000

# 改用更高阶小波
--wave_type coif3

# 添加多层变换
--transform_levels 2

# 增加训练轮数
--n_epochs 2000
```

### 模型训练缓慢

```bash
# 增加学习率（谨慎）
--lr 5e-5

# 增加批次大小
--batch_size 8

# 检查GPU使用情况
nvidia-smi
```

## 📚 参考文献

- **DDPM**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **小波变换**: Discrete Wavelet Transforms in Data Compression
- **Med-D3CG**: 原始条件生成框架

## 🤝 如何使用该Pipeline

### 场景1：生成特定类型的医学图像

```bash
# 1. 准备数据（仅需目标图像）
mkdir -p data/ct_images
# 将CT图像放入 data/ct_images/

# 2. 训练模型
python unconditional_train.py \
    --model_name D3CG_uncond_db4 \
    --dataset_dir ./data/ct_images \
    --out_name CT_generator \
    --n_epochs 1000

# 3. 生成新CT图像
python unconditional_generation.py \
    --model_name D3CG_uncond_db4 \
    --model_weight_path ./results/unconditional/CT_generator/best_model_epoch_*.pt \
    --num_samples 50 \
    --output_dir ./results/synthetic_ct/
```

### 场景2：数据增强

```bash
# 使用无条件模型生成合成医学图像用于数据增强
python unconditional_generation.py \
    --model_name D3CG_uncond_db4 \
    --num_samples 1000 \
    --output_dir ./data/synthetic_augmentation/
```

### 场景3：RGB医学图像生成

```bash
python unconditional_train.py \
    --model_name D3CG_uncond_coif3 \
    --dataset_dir ./data/rgb_medical_images \
    --is_rgb \
    --image_size 512

python unconditional_generation.py \
    --model_name D3CG_uncond_coif3 \
    --is_rgb \
    --image_size 512 \
    --num_samples 100
```

## ✅ 最佳实践

1. **数据预处理**: 确保所有输入图像尺寸一致
2. **小波选择**: 对于边界较多的医学图像选择DB4，对于光滑区域选择Coif3
3. **学习率调度**: 如果损失震荡，逐步降低学习率
4. **验证集**: 总是准备一个单独的验证集
5. **模型检查点**: 定期保存模型，不仅保存最优值
6. **结果分析**: 生成的样本应该与训练集有相同的统计特性

## 📞 技术细节

### 小波域扩散的优势

1. **计算效率**: 小波系数通常为1/4大小，加快计算
2. **多分辨率表示**: DWT自然提供多尺度特征
3. **信息保留**: 相比传统方法保留更多细节信息
4. **频率分解**: 分别处理不同频率的特征

### 无条件 vs 条件生成

无条件生成学习的是 $p(x)$（数据的真实分布）
条件生成学习的是 $p(x|y)$（给定条件下的分布）

## 许可证

该方案基于原Med-D3CG框架开发，遵循相同的许可证。

## 贡献指南

欢迎提交问题和改进建议！
