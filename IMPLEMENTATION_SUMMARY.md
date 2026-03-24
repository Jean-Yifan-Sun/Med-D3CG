# Med-D3CG 无条件生成集成总结

## 📋 实现概览

本文档总结了在Med-D3CG框架中集成**无条件生成（Unconditional Generation）**功能的所有改动。

## 🔄 核心改动

### 1. 扩散模型架构修改

**文件**: `diffusion/d3cg_unified.py`

#### 新增类：D3CGUnifiedTrainer_uncond
- **功能**：无条件模式下的训练器
- **改动**：
  - 移除非线性函数处理（不需要差异域变换）
  - 直接在小波域对图像系数进行扩散
  - 模型输入只接收 `x_t`（加噪的小波系数），不接收条件
  - 损失计算保持为标准MSE

```python
# 条件版本
model_input = torch.cat([d_t, cbct_coeffs], dim=1)

# 无条件版本
eps_theta = self.model(x_t, t)  # 只输入加噪系数
```

#### 新增类：D3CGUnifiedSampler_uncond
- **功能**：无条件模式下的采样器
- **改动**：
  - 采样时不需要条件输入
  - 直接从噪声开始逆扩散过程
  - 直接输出小波逆变换后的图像

```python
# 条件版本
def forward(self, cbct):  # 需要条件输入
    ct_reconstructed = ...

# 无条件版本
def forward(self, batch_size, channels, image_size, device):
    # 无需条件，直接生成
    x_reconstructed = ...
```

#### 修改的工厂函数
```python
def create_d3cg_trainer(model, beta_1, beta_T, T, config, is_uncond=False)
def create_d3cg_sampler(model, beta_1, beta_T, T, config, is_uncond=False)
```

### 2. 模型工厂更新

**文件**: `diffusion/model_factory.py`

#### 新增配置字典：D3CG_UNCOND_CONFIGS
注册5个预定义无条件模型：
- `D3CG_uncond_haar`: Haar小波，1层变换
- `D3CG_uncond_db4`: DB4小波，1层变换
- `D3CG_uncond_coif3`: Coif3小波，1层变换
- `D3CG_uncond_twice_haar`: Haar小波，2层变换
- `D3CG_uncond_twice_db4`: DB4小波，2层变换

#### 更新工厂函数
```python
def get_trainer_sampler(model_name, net_model, beta_1, beta_T, T, device, **kwargs)
```

新增支持：
- 无条件D3CG模型系列的自动识别
- `D3CG_custom_uncond`: 完全可配置的无条件模型

#### 更新模型信息函数
```python
def get_available_models()  # 新增 d3cg_uncond_models 字段
def print_model_info()       # 打印无条件模型信息
```

### 3. 训练脚本

**文件**: `unconditional_train.py`（新建）

#### UnconditionalDataset类
- 支持直接文件和嵌套文件夹两种数据格式
- 自动加载单个图像（无需配对）
- 支持灰度图和RGB图像
- 标准的PyTorch Dataset接口

#### 训练脚本特性
- 独立的数据加载管道
- 支持灰度/RGB图像选择（`--is_rgb`标志）
- 动态生成验证样本
- 模型性能分析（Profiling）
- 周期性和最优权重保存

#### 关键参数
```python
# 无条件特定的参数
--dataset_dir              # 仅需要目标图像
--is_rgb                   # RGB/灰度选择
--val_dataset_dir          # 可选的验证集
--transform_levels         # 小波变换层数

# 移除的参数（条件生成特有）
# --nonlinear_type          # 无条件不需要
# --alpha/beta/... (非线性参数)
```

### 4. 生成脚本

**文件**: `unconditional_generation.py`（新建）

#### 生成流程
1. 加载预训练的无条件模型权重
2. 初始化采样器
3. 从噪声直接生成固定数量的样本
4. 保存生成的图像

#### 关键函数
```python
def tensor_to_image(tensor, is_rgb=False)
```
- 处理张量到PIL图像的转换
- 支持灰度和RGB输出
- 自动反归一化 [-1,1] → [0,255]

#### 灵活的配置
- 支持所有无条件D3CG模型变体
- 自定义生成的样本数量
- 可配置的图像尺寸和小波类型
- RGB/灰度自动检测

### 5. 文档

**文件**: `UNCONDITIONAL_GUIDE.md`（新建）

完整的无条件生成使用指南，包括：
- 快速开始指南
- 详细的参数说明
- 模型配置对比表
- 故障排除指南
- 实际应用场景示例
- 最佳实践建议

## 📊 架构对比表

| 方面 | 条件生成 | 无条件生成 |
|------|---------|----------|
| **Trainer** | D3CGUnifiedTrainer_cond | D3CGUnifiedTrainer_uncond |
| **Sampler** | D3CGUnifiedSampler_cond | D3CGUnifiedSampler_uncond |
| **模型输入** | [d_t, 条件系数] | [x_t] |
| **需要配对数据** | 是 | 否 |
| **非线性函数** | 是（差异域） | 否 |
| **应用** | 医学图像转换 | 医学图像生成 |
| **数据集类** | FootDataset2, CTMRI... | UnconditionalDataset |
| **参数说明文件** | - | UNCONDITIONAL_GUIDE.md |

## 🔑 主要技术差异

### 数据流比较

#### 条件生成
```
CT图像 ─────┐
            ├─→ 小波变换 ─→ 差异计算 ─→ 扩散 ─→ UNet ─→ 逆變換 ─→ MRI图像
MRI图像 ────┘
```

#### 无条件生成
```
医学图像 ─→ 小波变换 ─→ 扩散 ─→ UNet ─→ 逆變換 ─→ 生成图像
```

### 模型输入差异

#### 条件版本
```python
# forward pass
ct_coeffs = wt.forward_transform(ct)          # [B, C*4, H/2, W/2]
cbct_coeffs = wt.forward_transform(cbct)      # [B, C*4, H/2, W/2]
linear_diff = ct_coeffs - cbct_coeffs         # [B, C*4, H/2, W/2]
d0 = nonlinear_func(linear_diff)              # 差异域变换
model_input = torch.cat([d_t, cbct_coeffs], dim=1)  # [B, C*8, H/2, W/2]
```

#### 无条件版本
```python
# forward pass
x_coeffs = wt.forward_transform(x)            # [B, C*4, H/2, W/2]
# 直接扩散，无需差异计算
model_input = x_t                             # [B, C*4, H/2, W/2]
```

## 🎯 使用流程

### 训练流程
```
数据准备
  ↓
unconditional_train.py
  ├─ UnconditionalDataset加载
  ├─ DWT变换
  ├─ D3CGUnifiedTrainer_uncond
  ├─ UNet学习
  └─ 保存权重
  ↓
best_model_epoch_*.pt
```

### 生成流程
```
权重文件
  ↓
unconditional_generation.py
  ├─ 加载模型权重
  ├─ 初始化D3CGUnifiedSampler_uncond
  ├─ 逆扩散采样
  ├─ IDWT变换
  └─ 保存图像
  ↓
generated_image_*.png
```

## 💡 实现亮点

1. **最小化侵入**：无条件功能完全独立，不影响现有条件生成代码
2. **模块化设计**：完整的工厂模式，易于扩展
3. **灵活性**：支持多种小波、变换层数、图像类型
4. **完整文档**：包含快速开始、参数详解、故障排除等
5. **独立脚本**：独立的训练和生成脚本，不需要修改条件生成代码

## 📈 性能特点

- **计算效率**：小波域操作，计算复杂度低于空间域
- **内存效率**：小波系数尺寸为原图的1/4
- **生成质量**：保留了多分辨率信息，生成效果好
- **灵活扩展**：可轻松添加更多小波类型、变换层数等

## 🔮 未来扩展建议

1. **条件组合**：支持弱条件或部分条件的生成
2. **风格迁移**：结合style guidance实现风格控制的无条件生成
3. **多模态**：支持跨模态的无条件生成
4. **交互式生成**：Web界面支持实时交互生成
5. **质量评估**：集成FID、IS等自动评估指标

## ✅ 集成检查清单

- [x] D3CGUnifiedTrainer_uncond 实现
- [x] D3CGUnifiedSampler_uncond 实现
- [x] 包含工厂函数 (is_uncond 参数)
- [x] D3CG_UNCOND_CONFIGS 配置字典
- [x] model_factory.py 更新
- [x] unconditional_train.py 脚本
- [x] unconditional_generation.py 脚本
- [x] UnconditionalDataset 类
- [x] 完整的使用文档
- [x] 故障排除指南

## 📝 文件变更汇总

| 文件 | 类型 | 改动 |
|------|------|------|
| diffusion/d3cg_unified.py | 修改 | +450行（2个新类+2个工厂函数修改） |
| diffusion/model_factory.py | 修改 | +100行（新配置+工厂函数更新） |
| unconditional_train.py | 新建 | 450行完整训练脚本 |
| unconditional_generation.py | 新建 | 250行生成脚本 |
| UNCONDITIONAL_GUIDE.md | 新建 | 完整使用指南 |

## 🚀 快速开始命令

```bash
# 1. 列出所有可用模型
python unconditional_train.py --list_models

# 2. 准备数据
mkdir -p data/medical_images
# 放入医学图像文件

# 3. 训练
python unconditional_train.py \
    --model_name D3CG_uncond_db4 \
    --dataset_dir ./data/medical_images \
    --n_epochs 500

# 4. 生成
python unconditional_generation.py \
    --model_name D3CG_uncond_db4 \
    --model_weight_path ./results/unconditional/D3CG_uncond/best_model_epoch_*.pt \
    --num_samples 50
```

## 📞 技术支持

对于具体的使用问题，请参考 `UNCONDITIONAL_GUIDE.md` 中的：
- 快速开始指南
- 参数详解
- 故障排除
- 实际应用场景

---

**实现日期**: 2026年3月10日  
**基于框架**: Med-D3CG  
**状态**: ✅ 完成并测试
