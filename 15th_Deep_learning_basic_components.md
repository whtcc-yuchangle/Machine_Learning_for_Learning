# 深度学习基础组件（Deep Learning Basic Components）教程

## 概述

本 notebook 系统梳理了深度学习模型的**五大基础组件**——**权重初始化、激活函数、归一化层、正则化策略、完整训练流程**。从 `nn.Linear` 的参数初始化策略（Xavier / Kaiming）出发，通过可视化理解各激活函数的数学形态，对比 BatchNorm / LayerNorm / InstanceNorm / GroupNorm 四种归一化的计算维度与适用场景，最终以一个带 L1/L2 正则化的 MLP 二分类训练案例收尾，为后续 CNN、RNN、Transformer 等复杂架构的学习奠定组件基础。

本教程是 [13th 神经网络基础](13th_Neural_Network_Basics.md) 的延续，相比 13th 侧重"如何构建和训练一个网络"，本教程更侧重于**网络内部每个组件的工作原理与选型依据**。

---

## 学习目标

通过学习本教程，您将掌握：

1. **权重初始化原理**：理解 Xavier 和 Kaiming 初始化的数学推导与适用激活函数，能根据激活函数选择正确的初始化策略
2. **激活函数选型**：掌握 Sigmoid、Tanh、ReLU、Leaky ReLU、ELU、Mish 的公式、图像特征与优缺点
3. **归一化方法对比**：清晰区分 BatchNorm（特征通道维度）、LayerNorm（样本内全维度）、InstanceNorm（单样本单通道）、GroupNorm（分组通道）的计算维度与适用场景
4. **正则化实践**：理解 L1/L2 正则化的作用与 PyTorch 实现方式
5. **完整训练流程**：综合以上组件，使用 PyTorch 完成一个二分类 MLP 的训练

---

## 内容结构

### 第一章：权重初始化（Weight Initialization）

#### 1.1 为什么需要精心初始化

神经网络训练的起点是随机初始化的权重。初始化方式直接影响梯度传播的质量：

| 初始化状态 | 后果 |
|:---|:---|
| **权重全零** | 所有神经元学习到相同的特征（对称性问题），网络退化为单一神经元 |
| **权重过大** | 激活值进入饱和区（Sigmoid/Tanh 两端），梯度趋近于 0（**梯度消失**） |
| **权重过小** | 信号逐层衰减，深层网络几乎学不到任何东西 |
| **合适的初始化** | 保持前向传播时各层激活值方差稳定，反向传播时各层梯度方差稳定 |

#### 1.2 PyTorch 默认初始化

```python
from torch import nn

# 创建一个全连接层（默认使用 Kaiming Uniform 初始化）
model = nn.Linear(in_features=16, out_features=128)
print(model.weight)
```

`nn.Linear` 的默认初始化策略为 `kaiming_uniform_`，适合 ReLU 族激活函数。

#### 1.3 Xavier（Glorot）初始化

**公式**：权重从均匀分布 $U[-a, a]$ 中采样，其中：

$$a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$$

| 参数 | 含义 |
|:---|:---|
| `fan_in` | 输入神经元数量 |
| `fan_out` | 输出神经元数量 |
| `gain` | 增益系数，根据激活函数调整 |

**适用激活函数**：Sigmoid、Tanh（及其变体）、Linear

```python
import torch.nn as nn

# Xavier Uniform 初始化
nn.init.xavier_uniform_(model.weight, gain=nn.init.calculate_gain('tanh'))

# Xavier Normal 初始化
nn.init.xavier_normal_(model.weight, gain=nn.init.calculate_gain('sigmoid'))
```

**`nn.init.calculate_gain` 常用增益值**：

| 激活函数 | gain 值 | 说明 |
|:---|:---|:---|
| `'sigmoid'` | 1.0 | |
| `'tanh'` | $\frac{5}{3} \approx 1.667$ | |
| `'relu'` | $\sqrt{2} \approx 1.414$ | |
| `'leaky_relu'` | $\sqrt{\frac{2}{1+\alpha^2}}$ | $\alpha$ 为负斜率（默认 0.01 时 ≈ 1.414） |
| `'linear'` | 1.0 | 无激活函数时使用 |

> **核心思想**：Xavier 初始化假设激活函数在零点附近**近似线性**，因此特别适合 Sigmoid 和 Tanh。它通过保持前向和反向传播的方差一致，来缓解梯度消失/爆炸。

#### 1.4 Kaiming（He）初始化

**公式**：权重从均匀分布 $U[-a, a]$ 中采样，其中：

$$a = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}$$

| 参数 | 含义 |
|:---|:---|
| `mode='fan_in'` | 使用前向传播的方差推导（**默认，推荐**） |
| `mode='fan_out'` | 使用反向传播的方差推导 |
| `a`（`nonlinearity='leaky_relu'`） | Leaky ReLU 的负斜率 |

**适用激活函数**：ReLU、Leaky ReLU、ELU、PReLU

```python
# Kaiming Uniform 初始化（默认 mode='fan_in'）
nn.init.kaiming_uniform_(model.weight, a=0, mode='fan_in', nonlinearity='relu')

# Kaiming Normal 初始化
nn.init.kaiming_normal_(model.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
```

> **核心思想**：Xavier 假设激活函数在零点附近是线性的，但 **ReLU 将负半轴输出归零**，破坏了方差计算的前提。Kaiming 修正了这一假设，乘以 $\sqrt{2}$ 来补偿被 ReLU 丢弃的一半激活值。

#### 1.5 Xavier vs Kaiming 对比

| 对比维度 | Xavier (Glorot) | Kaiming (He) |
|:---|:---|:---|
| **核心公式** | $a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$ | $a = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}$ |
| **分母** | fan_in + fan_out（调和） | fan_in 或 fan_out（单一） |
| **激活函数假设** | 零点附近线性（对称激活） | 负半轴为零（ReLU 特性） |
| **适用激活函数** | Sigmoid、Tanh、Linear | ReLU、Leaky ReLU、ELU、Mish |
| **ReLU 场景** | ❌ 方差会逐渐衰减 | ✅ 正确补偿了 ReLU 的信息损失 |
| **Sigmoid/Tanh 场景** | ✅ 表现良好 | 可用但非最优 |
| **PyTorch 中的默认** | 需要手动调用 | `nn.Linear` 和 `nn.Conv2d` 的**默认初始化** |

> **实践建议**：
> - 使用 ReLU 族激活函数 → **Kaiming 初始化**（无需额外操作，PyTorch 已默认）
> - 使用 Sigmoid/Tanh 激活函数 → **手动调用 Xavier 初始化**
> - 不确定时 → **Kaiming 初始化**（更通用，现代网络首选）

---

### 第二章：激活函数（Activation Functions）

#### 2.1 为什么需要激活函数

如果没有激活函数（即 $y = x$，线性激活），多层网络等价于单层：

$$W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2) = W'x + b'$$

**激活函数引入非线性**，使神经网络能够逼近任意复杂函数。

#### 2.2 Sigmoid

**公式**：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**图像特征**：
- 输出范围：$(0, 1)$
- 以 $(0, 0.5)$ 为中心对称
- 当 $|x| > 5$ 时，梯度几乎为 0（饱和区）

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 1000)
y = 1 / (1 + np.exp(-x))
plt.plot(x, y)
plt.show()
```

| 优点 | 缺点 |
|:---|:---|
| 输出有界 $(0,1)$，适合概率输出 | **梯度消失**：两端饱和区梯度 ≈ 0 |
| 平滑可导、单调递增 | **非零中心输出**：导致 zig-zag 梯度更新 |
| 历史最悠久的激活函数 | 计算 $e^{-x}$ 较慢 |

> **使用场景**：二分类任务的**最后一层**（输出概率），隐藏层已基本不再使用。

#### 2.3 Tanh（双曲正切）

**公式**：

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**与 Sigmoid 的关系**：$\tanh(x) = 2\sigma(2x) - 1$

**图像特征**：
- 输出范围：$(-1, 1)$
- 以 $(0, 0)$ 为中心对称（**零中心**，优于 Sigmoid）

```python
y = np.tanh(x)
plt.plot(x, y)
plt.show()
```

| 优点 | 缺点 |
|:---|:---|
| **零中心输出**，梯度更新更稳定 | 仍然存在**梯度消失**问题 |
| 比 Sigmoid 收敛更快 | 计算量大于 ReLU |

> **使用场景**：RNN / LSTM 的隐藏状态（`tanh` 是标准组件），以及某些需要 $(-1,1)$ 输出的生成模型。

#### 2.4 ReLU（Rectified Linear Unit）

**公式**：

$$\text{ReLU}(x) = \max(0, x)$$

**图像特征**：
- $x > 0$：$y = x$（线性，梯度 = 1）
- $x \leq 0$：$y = 0$（梯度 = 0）

```python
y = np.maximum(0, x)
plt.plot(x, y)
plt.show()
```

| 优点 | 缺点 |
|:---|:---|
| **解决梯度消失**：正区间梯度恒为 1 | **Dying ReLU**：负区间梯度为 0，神经元可能永久失活 |
| 计算极快（仅需判断阈值） | 非零中心输出 |
| 稀疏激活（约 50% 神经元激活），正则化效果 | 不可导于 $x=0$（实践中不影响） |

> **ReLU 是隐层的默认首选激活函数**。绝大多数 CNN 和 MLP 都使用 ReLU 或其变体。

##### Dying ReLU 问题详解

当神经元的权重更新到使得所有输入都 $\leq 0$ 时，该神经元输出恒为 0，梯度也为 0——该神经元**永久死亡**，后续训练中不再更新。

解决方法：
- 使用 **Leaky ReLU** 或 **ELU**（给负区间一个非零斜率）
- 使用较小的学习率
- 使用 **Kaiming 初始化**（确保初始时正负各半）

#### 2.5 Leaky ReLU

**公式**：

$$\text{LeakyReLU}(x) = \max(\alpha x, x)$$

其中 $\alpha$ 为负斜率（默认 $\alpha = 0.01$）。

```python
y = np.maximum(0.01 * x, x)
plt.plot(x, y)
plt.show()
```

| 优点 | 缺点 |
|:---|:---|
| **解决 Dying ReLU**：负区间有微弱梯度 | $\alpha$ 需要手动设定 |
| 保留 ReLU 的所有优点 | 实际效果不一定优于 ReLU |

**PyTorch 使用**：

```python
import torch.nn.functional as F

# 默认 α = 0.01
output = F.leaky_relu(x, negative_slope=0.01)

# 作为网络层（可学习的 α）
layer = nn.LeakyReLU(negative_slope=0.01)
```

#### 2.6 ELU（Exponential Linear Unit）

**公式**：

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

```python
alpha = 1.0
y = np.where(x > 0, x, alpha * (np.exp(x) - 1))
plt.plot(x, y)
plt.show()
```

| 优点 | 缺点 |
|:---|:---|
| 负区间输出接近零中心（**比 Leaky ReLU 更平滑**） | 计算 $e^x$ 比 ReLU 慢 |
| 对噪声更鲁棒 | 需要设置 $\alpha$ |

> **ELU vs Leaky ReLU**：ELU 在负区间是平滑曲线而非直线，这使得梯度更新更稳定，但计算代价更高。

#### 2.7 Mish

**公式**：

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

```python
import torch.nn.functional as F

def mish(x):
    return x * torch.tanh(F.softplus(x))

output = mish(x)
```

| 优点 | 缺点 |
|:---|:---|
| 平滑非单调，允许微小负梯度（自正则化） | 计算代价最高 |
| 在许多 benchmark 上优于 ReLU | 非 PyTorch 原生提供（需自定义） |

> Mish 由 Misra (2019) 提出，在 YOLOv4 等架构中表现优异。

#### 2.8 激活函数全景对比

| 激活函数 | 公式 | 输出范围 | 梯度消失 | 零中心 | 计算速度 | 推荐场景 |
|:---|:---|:---|:---|:---|:---|:---|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | $(0,1)$ | 严重 | ✗ | 慢 | 二分类输出层 |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $(-1,1)$ | 严重 | ✓ | 慢 | RNN 隐藏状态 |
| **ReLU** | $\max(0,x)$ | $[0,\infty)$ | 无(正区间) | ✗ | 极快 | **隐层默认首选** |
| **Leaky ReLU** | $\max(\alpha x, x)$ | $(-\infty,\infty)$ | 无 | ✗ | 极快 | ReLU 的 Dying 问题替代 |
| **ELU** | $x>0?x:\alpha(e^x-1)$ | $(-\alpha,\infty)$ | 无 | ≈✓ | 中等 | 需要平滑负梯度的场景 |
| **Mish** | $x\cdot\tanh(\ln(1+e^x))$ | $(\approx-0.31,\infty)$ | 无 | ✗ | 较慢 | 追求极致性能的 CNN |

**PyTorch 调用速查**：

| 激活函数 | `torch.nn` 层 | `torch.nn.functional` |
|:---|:---|:---|
| Sigmoid | `nn.Sigmoid()` | `F.sigmoid(x)` |
| Tanh | `nn.Tanh()` | `F.tanh(x)` |
| ReLU | `nn.ReLU()` | `F.relu(x)` |
| Leaky ReLU | `nn.LeakyReLU(0.01)` | `F.leaky_relu(x, 0.01)` |
| ELU | `nn.ELU(alpha=1.0)` | `F.elu(x, alpha=1.0)` |
| Mish | 无（需自定义） | `x * F.tanh(F.softplus(x))` |

> **选择原则**：
> 1. **隐藏层默认选 ReLU**（快速验证）
> 2. 遇到 Dying ReLU → 换 **Leaky ReLU** 或 **ELU**
> 3. RNN/LSTM → 隐藏状态用 **Tanh**
> 4. 二分类输出层 → **Sigmoid**
> 5. 多分类输出层 → 不需要额外激活（CrossEntropyLoss 内置 Softmax）
> 6. 追求 SOTA → 尝试 **Mish** 或 **GELU**（Transformer 常用）

---

### 第三章：PyTorch 层与激活函数实战

本章演示如何在 PyTorch 中将 `nn.Linear` 与各种激活函数组合使用。

#### 3.1 基础层的前向计算

```python
import torch
from torch import nn

# 定义一个全连接层：输入维度 16 → 输出维度 5
layer = nn.Linear(in_features=16, out_features=5)

# 模拟一个 batch 数据：batch_size=8，特征维度=16
x = torch.randn(size=(8, 16))

# 前向计算
layer_output = layer(x)
print(layer_output.size())  # torch.Size([8, 5])
```

#### 3.2 在层输出上应用激活函数

```python
import torch.nn.functional as F

# Sigmoid
output = F.sigmoid(layer_output)    # 输出范围 (0, 1)

# ReLU
output = F.relu(layer_output)       # 输出范围 [0, ∞)

# Leaky ReLU
output = F.leaky_relu(layer_output) # 输出范围 (-∞, ∞)

# Mish（自定义）
def mish(x):
    return x * torch.tanh(F.softplus(x))
output = mish(layer_output)
```

#### 3.3 层 + 激活函数的组合方式

```python
# 方式一：函数式（灵活，推荐用于前向传播中）
class Model(nn.Module):
    def forward(self, x):
        x = F.relu(self.layer_1(x))  # 函数式调用
        return self.layer_2(x)

# 方式二：层封装（简洁，适合 Sequential）
model = nn.Sequential(
    nn.Linear(16, 128),
    nn.ReLU(),                       # 作为层注册
    nn.Linear(128, 5)
)
```

---

### 第四章：归一化层（Normalization Layers）

归一化层是深度学习中的"稳定器"——通过在中间层对激活值进行标准化，加速收敛、允许更大学习率、降低对初始化的敏感性。

#### 4.1 数据格式：NCHW vs NHWC

在讨论归一化之前，需要理解张量的两种内存布局：

```python
output_from_pre_layer = torch.randn(size=(8, 224, 224, 16))

# NHWC
# N: batch size  |  H: height  |  W: width  |  C: channel
#       8               224           224           16
#
# NCHW（PyTorch 默认）
# N: batch size  |  C: channel  |  H: height  |  W: width
#       8                16            224           224
```

| 格式 | 全称 | 内存布局 | 常用框架 |
|:---|:---|:---|:---|
| **NHWC** | Batch-Height-Width-Channel | 通道维在最后 | TensorFlow（默认） |
| **NCHW** | Batch-Channel-Height-Width | 通道维在最前 | **PyTorch（默认）** |

```python
# NHWC → NCHW
tensor_nchw = tensor_nhwc.permute(0, 3, 1, 2)
# (8, 224, 224, 16) → (8, 16, 224, 224)
```

#### 4.2 Batch Normalization（批归一化）

**计算维度**：在 **N（batch）维度** 上求均值和方差，每个通道独立计算。

**公式**：
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

其中 $\mu_B$ 和 $\sigma_B^2$ 是当前 mini-batch 的均值和方差，$\gamma$ 和 $\beta$ 是可学习的缩放和偏移参数。

```python
bn_norm = nn.BatchNorm2d(num_features=16)  # input shape: (N, C, H, W)
norm_out = bn_norm(output_from_pre_layer.permute(0, 3, 1, 2))
print(norm_out.shape)  # torch.Size([8, 16, 224, 224])
```

| 优点 | 缺点 |
|:---|:---|
| 加速收敛（可用更大学习率） | **对 batch size 敏感**（小 batch 时统计不稳定） |
| 降低对初始化的敏感性 | **不适合变长数据**（Text、Speech，batch 中各样本长度不同） |
| 轻微正则化效果（batch 噪声） | 训练/推理行为不同（需 `model.eval()` 切换） |

> **使用场景**：CNN 图像分类任务（batch size 较大且稳定），如 ResNet、VGG。

#### 4.3 Layer Normalization（层归一化）

**计算维度**：在 **单个样本的所有特征维度（H, W, C）** 上求均值和方差。

**对比 BatchNorm**：BN 是"跨样本"归一化，LN 是"样本内"归一化。

```python
ln_norm = nn.LayerNorm([224, 224, 16])  # input shape: [N, *, *, *]
norm_out = ln_norm(output_from_pre_layer)
print(norm_out.shape)  # torch.Size([8, 224, 224, 16])
```

| 优点 | 缺点 |
|:---|:---|
| **对 batch size 不敏感** | 在 CNN 上效果通常不如 BN |
| **适合变长序列数据**（NLP） | 对视觉任务不是最优 |
| 训练/推理行为一致 | |

> **使用场景**：
> - **Transformer 的标准归一化**（每个 Self-Attention 和 FFN 后使用 LN）
> - RNN/LSTM 等序列模型
> - NLP 任务（文本长度不一，BN 难以统计跨样本均值）

#### 4.4 Instance Normalization（实例归一化）

**计算维度**：在**单个样本的单个通道（H, W）** 上求均值和方差。每个样本、每个通道独立归一化。

```python
in_norm = nn.InstanceNorm2d(16)  # input shape: (N, C, H, W)
norm_out = in_norm(output_from_pre_layer.permute(0, 3, 1, 2))
print(norm_out.shape)  # torch.Size([8, 16, 224, 224])
```

| 优点 | 缺点 |
|:---|:---|
| 样本间完全独立（适合生成任务） | 不适用于分类任务（丢掉了通道间的对比信息） |
| 对 batch size 完全不敏感 | |

> **使用场景**：**GAN（生成对抗网络）的风格迁移任务**。Image-to-Image translation（如 CycleGAN、Pix2Pix）中，IN 帮助去除原始图像的对比度信息，保留内容结构。

#### 4.5 Group Normalization（分组归一化）

**计算维度**：将通道分成若干组，在**每个样本的每组通道（group_size × H × W）** 上求均值和方差。

```python
# 16 个通道分成 4 组，每组 4 个通道
gn_norm = nn.GroupNorm(num_groups=4, num_channels=16)
norm_out = gn_norm(output_from_pre_layer.permute(0, 3, 1, 2))
print(norm_out.shape)  # torch.Size([8, 16, 224, 224])
```

| 优点 | 缺点 |
|:---|:---|
| **对 batch size 完全不敏感** | `num_groups` 需要精心调参 |
| 小 batch 下性能优于 BN | 大 batch 下理论上略逊于 BN |
| 兼具 IN 和 LN 的特性 | |

**GN 的退化情况**：

| `num_groups` | 等价于 | 说明 |
|:---|:---|:---|
| `num_groups = 1` | **LayerNorm** | 所有通道为一组 → 对整个样本归一化 |
| `num_groups = C`（通道数） | **InstanceNorm** | 每组一个通道 → 逐通道归一化 |
| `1 < num_groups < C` | **GroupNorm** | GN 的一般形式 |

> **使用场景**：**目标检测和分割任务**（Mask R-CNN、FCOS 等）。由于显存限制，这些任务通常只能使用较小的 batch size（2~4），BN 在这种情况下表现很差，GN 是最佳替代。

#### 4.6 四种归一化全景对比

```
BatchNorm                     LayerNorm                    InstanceNorm               GroupNorm
在每个通道上、跨 batch 计算    在每个样本上、跨所有维度计算   在每个样本每个通道上计算     在每个样本每个通道组上计算

  N维度                                N维度                          N维度                      N维度
  ┌─────────────────┐                 ┌─────────────────┐            ┌─────────────────┐        ┌─────────────────┐
  │ ████████████████ │ ← C₁所有样本    │ ████            │ ← 样本1    │ ██              │ ← C₁    │ ████            │ ← C₁₋₂组
  │ ████████████████ │                │ ████            │            │ ██              │         │ ████            │
  │ ████████████████ │                │ ████            │            │                │         │                │
  │ ████████████████ │                │ ████            │            │                │         │ ████            │ ← C₃₋₄组
  └─────────────────┘                 └─────────────────┘            └─────────────────┘        └─────────────────┘
 在每个通道的 (N,H,W) 上              在每个样本的 (C,H,W) 上           在每个 (H,W) 上             在每组通道的 (H,W) 上
 求 E(x), Var(x)                     求 E(x), Var(x)                 求 E(x), Var(x)             求 E(x), Var(x)
```

| 归一化类型 | 统计维度 | PyTorch API | 对 batch size 敏感度 | 主要应用 |
|:---|:---|:---|:---|:---|
| **BatchNorm** | N, H, W（每通道跨样本） | `nn.BatchNorm2d(C)` | **高** | CNN 图像分类（ResNet, VGG） |
| **LayerNorm** | C, H, W（每样本内所有特征） | `nn.LayerNorm([C,H,W])` | **无** | Transformer, RNN, NLP |
| **InstanceNorm** | H, W（每样本每通道独立） | `nn.InstanceNorm2d(C)` | **无** | GAN 风格迁移 |
| **GroupNorm** | G, H, W（每样本每组通道） | `nn.GroupNorm(G, C)` | **无** | 目标检测/分割（小 batch） |

> **选型决策树**：
> ```
> 是否有充足的 batch size（≥16）且数据定长？
>   ├── 是 → BatchNorm（CNN 视觉任务首选）
>   └── 否 → 是否为序列/变长数据（NLP / RNN）？
>             ├── 是 → LayerNorm
>             └── 否 → 是否为生成/风格迁移任务（GAN）？
>                       ├── 是 → InstanceNorm
>                       └── 否 → GroupNorm（检测/分割小 batch 场景，G=32 常见起点）
> ```

---

### 第五章：正则化 — L1 与 L2

#### 5.1 为什么需要正则化

深度学习模型参数众多，容易**过拟合**——在训练集上表现完美，在测试集上表现很差。正则化通过在损失函数中增加对权重大小的惩罚项，抑制模型复杂度。

#### 5.2 L1 正则化（Lasso）

$$L = L_{\text{original}} + \lambda_1 \sum_i |w_i|$$

- 对所有权重的绝对值求和
- 倾向产生**稀疏权重**（部分权重变为 0）
- 等价于**特征选择**

```python
lambda_l1 = 0.01
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = criteria(y_hat, y) + lambda_l1 * l1_norm
```

#### 5.3 L2 正则化（Ridge / Weight Decay）

$$L = L_{\text{original}} + \lambda_2 \sum_i w_i^2$$

- 对所有权重的平方和
- 倾向产生**小且分散的权重**
- 防止权重过大，平滑决策边界

```python
lambda_l2 = 0.0001

# 方式一：手动计算 L2 范数
l2_norm = torch.norm(all_weights, p=2)  # 即 sqrt(sum(w_i²))
loss = criteria(y_hat, y) + lambda_l2 * l2_norm

# 方式二：通过优化器的 weight_decay 参数（推荐）
opt = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
```

> **推荐方案**：使用优化器的 `weight_decay` 参数实现 L2 正则化，代码更简洁且 PyTorch 对其有优化。

#### 5.4 L1 vs L2 对比

| 对比维度 | L1 正则化 | L2 正则化 |
|:---|:---|:---|
| **惩罚项** | $\sum \vert w_i \vert$ | $\sum w_i^2$ |
| **几何形状** | 菱形（各方向等概率收缩到 0） | 圆形（均匀收缩） |
| **产生的解** | **稀疏解**（部分权重=0） | 小权重解（均匀缩小） |
| **特征选择** | ✅ 自动特征选择 | ✗ 不产生稀疏性 |
| **梯度** | 常数 $\pm \lambda$ | $2\lambda w$（权重越小梯度越小） |
| **常用场景** | 高维稀疏特征 | 深度学习默认选择 |

---

### 第六章：完整训练案例

本章综合以上所有组件，构建一个带正则化的多层感知机，完成二分类任务。

#### 6.1 数据准备

```python
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_item = 1000
n_feature = 2

torch.manual_seed(123)
data_x = torch.randn(size=(n_item, n_feature)).float()

# 决策规则：0.5*x₁ - 1.5*x₂ + 0.02 > 0 → 类别 0，否则 → 类别 1
data_y = torch.where(
    torch.subtract(data_x[:, 0] * 0.5, data_x[:, 1] * 1.5) + 0.02 > 0,
    0, 1
).long()

data_y = F.one_hot(data_y)  # [n_item, 2]
```

#### 6.2 模型定义

```python
class BinaryClassificationModel(nn.Module):
    def __init__(self, in_feature):
        super(BinaryClassificationModel, self).__init__()
        # 多层感知机（MLP）
        self.layer_1 = nn.Linear(in_features=in_feature, out_features=128, bias=True)
        self.layer_2 = nn.Linear(in_features=128, out_features=512, bias=True)
        self.layer_final = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x):
        layer_1_output = F.sigmoid(self.layer_1(x))
        layer_2_output = F.sigmoid(self.layer_2(layer_1_output))
        output = F.sigmoid(self.layer_final(layer_2_output))
        return output
```

**模型结构**：`[2] → Linear+Sigmoid → [128] → Linear+Sigmoid → [512] → Linear+Sigmoid → [2]`

#### 6.3 训练配置

```python
learning_rate = 0.01

model = BinaryClassificationModel(n_feature).to(device)
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
criteria = nn.BCELoss()  # 二分类交叉熵（需 Sigmoid 输出）
```

#### 6.4 带 L2 正则化的训练循环

```python
lambda_l2 = 0.0001

for epoch in range(1000):
    for step in range(n_item):
        x = data_x[step].to(device)
        y = data_y[step].to(device)

        # ① 梯度清零
        opt.zero_grad()

        # ② 前向计算
        y_hat = model(x.unsqueeze(0))  # [1, 2]

        # ③ 计算 L2 正则化项
        layer_1_w = torch.cat([x.view(-1) for p in model.layer_1.parameters()])
        layer_2_w = torch.cat([x.view(-1) for p in model.layer_2.parameters()])
        layer_final_w = torch.cat([x.view(-1) for p in model.layer_final.parameters()])
        all_weights = torch.cat([layer_1_w, layer_2_w, layer_final_w], dim=-1)
        l2_norm = torch.norm(all_weights, p=2)

        # ④ 损失计算 = 原始损失 + L2 惩罚
        loss = criteria(y_hat, y.unsqueeze(0).float()) + lambda_l2 * l2_norm

        # ⑤ 反向传播
        loss.backward()

        # ⑥ 参数更新
        opt.step()

    if epoch % 50 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss.cpu().item():.3f}')
```

**训练结果**：

| Epoch | Loss |
|:---|:---|
| 0 | 0.360 |
| 50 | 0.008 |
| 100 | 0.001 |
| 150+ | ≈0.000（收敛） |

#### 6.5 训练技巧：L2 正则化的优化器写法

```python
# 更简洁的写法：直接在优化器中设置 weight_decay
opt = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-4  # L2 正则化系数
)

# 训练循环中无需手动计算 L2 范数
for epoch in range(epochs):
    for step in range(n_item):
        opt.zero_grad()
        y_hat = model(x.unsqueeze(0))
        loss = criteria(y_hat, y.unsqueeze(0).float())  # 不需要手动加 L2
        loss.backward()
        opt.step()
```

---

### 第七章：组件组合最佳实践

#### 7.1 经典组件组合姿势

```
① CNN 图像分类
   Conv2d → BatchNorm2d → ReLU → ... → Linear → Softmax
   初始化：Kaiming（默认）

② Transformer / NLP
   Linear → LayerNorm → Self-Attention → LayerNorm → FFN
   初始化：Xavier（通常框架默认）

③ GAN 生成器
   ConvTranspose2d → InstanceNorm2d → ReLU → ... → Tanh
   初始化：Kaiming

④ 目标检测（小 batch）
   Conv2d → GroupNorm(G=32) → ReLU → ... → Head
   初始化：Kaiming
```

#### 7.2 组件选择速查表

| 场景 | 激活函数 | 归一化 | 初始化 | 正则化 |
|:---|:---|:---|:---|:---|
| CNN 图像分类（大 batch） | ReLU | BatchNorm | Kaiming | L2 (weight_decay) |
| Transformer / NLP | GELU | LayerNorm | Xavier | Dropout |
| RNN / LSTM | Tanh | LayerNorm | Xavier | Dropout |
| GAN 生成器 | ReLU | InstanceNorm | Kaiming | — |
| GAN 判别器 | Leaky ReLU(0.2) | — | Kaiming | — |
| 小 batch 检测/分割 | ReLU | GroupNorm | Kaiming | L2 |
| MLP（通用） | ReLU | BatchNorm/LayerNorm | Kaiming | L2 + Dropout |

---

## 环境要求

| 依赖包 | 用途 |
|:---|:---|
| Python 3.8+ | 编程语言 |
| PyTorch ≥ 1.10 | nn.Module、nn.init、归一化层 |
| NumPy | 激活函数可视化数据生成 |
| Matplotlib | 激活函数图像绘制 |

```bash
pip install torch numpy matplotlib
```

---

## 使用方法

1. 启动 Jupyter Notebook：
   ```bash
   jupyter notebook
   ```
2. 在浏览器中打开 `15th_Deep_learning_basic_components.ipynb`
3. 按顺序运行每个代码单元格（`Shift + Enter`）
4. 观察 Xavier 和 Kaiming 初始化对权重分布的影响
5. 对比 5 种激活函数的图像特征
6. 理解 4 种归一化方法分别从哪些维度计算统计量
7. 运行完整训练案例，观察 L2 正则化对 Loss 收敛的影响

---

## 学习建议

1. **对比初始化效果**：修改 Xavier 和 Kaiming 的 `gain` 参数，观察权重分布的变化，加深对初始化目的的理解
2. **可视化激活函数**：将 5 种激活函数绘制在同一张图上，直观对比它们的差异
3. **手动计算归一化统计量**：对一个小的张量手动计算 BN 和 LN 的结果，验证对"计算维度"的理解
4. **实验 Dying ReLU**：将激活函数改为 ReLU，使用过大的学习率（如 lr=10），观察 Loss 不下降的情况
5. **对比归一化对小 batch 的影响**：将 batch size 设为 2 和 32，分别测试 BN 和 GN 的性能差异
6. **尝试不同归一化组合**：在 MLP 训练案例中加入 BatchNorm / LayerNorm，观察收敛速度的变化
7. **L1 vs L2 稀疏性实验**：分别用 L1 和 L2 训练同一个网络，统计每层权重中接近 0 的权重数量，验证 L1 的稀疏性

---

## 常见问题

### Q1: 为什么 PyTorch 的 `nn.Linear` 默认使用 Kaiming 初始化而不是 Xavier？

因为 ReLU 已经成为隐层激活函数的实际标准，而 Kaiming 初始化是专门为 ReLU 设计的——它乘以 $\sqrt{2}$ 来补偿 ReLU 丢弃的负半轴信息。如果使用 Xavier 初始化 + ReLU，前向信号的方差会逐层衰减。

### Q2: BatchNorm 的 `num_features` 参数应该填什么？

填**输入张量的通道数**。例如：
- 输入 shape `(N, 16, 224, 224)` → `nn.BatchNorm2d(16)`
- 输入 shape `(N, 32, 64, 64)` → `nn.BatchNorm2d(32)`

### Q3: LayerNorm 和 BatchNorm 可以互相替代吗？

不完全能。在 CNN 视觉任务中，BN 通常是更好的选择（因为通道维度的统计量有意义）。在 NLP/Transformer 中，LN 更好（因为文本序列长度不一，且 batch 维度上的统计量没有物理意义）。GroupNorm 在视觉任务小 batch 场景下是 BN 的良好替代。

### Q4: 什么时候用 `F.relu(x)` 而不是 `nn.ReLU()`？

| 方式 | 适用场景 |
|:---|:---|
| `F.relu(x)` | 在 `forward()` 中灵活调用，不需要学习参数 |
| `nn.ReLU()` | 作为层注册到 `nn.Sequential` 中，或在 `__init__` 中需要显式管理 |

如果激活函数**没有可学习参数**（ReLU、Sigmoid、Tanh），两种方式等价。但如果有（如 PReLU 的 $\alpha$），必须使用 `nn` 层方式。

### Q5: 为什么归一化层有 `affine=True` 参数？

`affine=True`（默认）意味着归一化层包含可学习的缩放（$\gamma$）和偏移（$\beta$）参数。这让网络可以在训练过程中学习是否真的需要归一化——如果不需要，它可以学到 $\gamma \approx 1, \beta \approx 0$。

### Q6: 训练和推理时 BatchNorm 的行为有何不同？

| 阶段 | 统计量来源 |
|:---|:---|
| **训练** (`model.train()`) | 当前 mini-batch 的均值和方差 |
| **推理** (`model.eval()`) | 训练期间累积的**全局移动平均**均值和方差 |

> 切换模式通过 `model.train()` 和 `model.eval()` 完成。忘记切换到 `eval()` 会导致推理结果依赖 batch 组成，这是常见 bug。

### Q7: 如何打印模型参数的梯度和权重？

```python
# 查看权重
print(model.layer_1.weight)

# 查看梯度（必须在 backward() 之后）
print(model.layer_1.weight.grad)

# 查看所有参数名称和 shape
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
```

---

## 下一步

完成本教程后，建议进入深度学习后续内容：
- **CNN（卷积神经网络）**：将 `nn.Linear` 替换为 `nn.Conv2d`，结合 BatchNorm + ReLU 构建图像分类网络
- **RNN/LSTM**：结合 LayerNorm 处理序列数据
- **Transformer**：深入理解 LayerNorm + Residual Connection 的组合模式
- **GAN**：练习 InstanceNorm 在生成模型中的使用
- 或返回 [13th_Neural_Network_Basics.md](13th_Neural_Network_Basics.md) 回顾神经网络训练基础

---

**最后更新**：2026-06-21
