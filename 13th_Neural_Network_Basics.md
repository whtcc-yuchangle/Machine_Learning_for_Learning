# 神经网络基础（Neural Network Basics）教程

## 概述

本 notebook 是深度学习入门的第一个教程，从"手写逻辑回归"开始，逐步过渡到 PyTorch 的 `nn.Module` 构建多层感知机（MLP）。通过**手动实现梯度下降**和**PyTorch 框架训练**两条路径，帮助您理解神经网络的核心训练机制——**前向传播、损失计算、反向传播、参数更新**。

---

## 学习目标

通过学习本教程，您将掌握：

1. **手动实现逻辑回归**：从零手写前向计算、二分类交叉熵损失、梯度下降更新，理解 PyTorch 的 `requires_grad` 和自动微分机制
2. **感知机模型**：理解单层感知机与多层感知机（MLP）的结构差异
3. **PyTorch 建模流程**：掌握 `nn.Module`、`nn.Linear`、`nn.BCELoss`、`torch.optim.SGD` 的标准用法
4. **GPU 训练**：了解 `torch.device` 和 `.to(device)` 的 GPU 迁移方法
5. **训练循环五步法**：梯度清零 → 前向计算 → 损失计算 → 反向传播 → 参数更新

---

## 内容结构

### 第一章：神经网络基础概念

#### 1.1 从逻辑回归到神经网络

逻辑回归可以看作**只有一个神经元的神经网络**：

```
逻辑回归：                         单层感知机：
                                    
 x₁ ──→ w₁ ─╮                       x₁ ──→ w₁₁ ─╮
             ├─→ Σ → Sigmoid → ŷ                 ├─→ Σ₁ → Sigmoid → ŷ₁
 x₂ ──→ w₂ ─╯                       x₂ ──→ w₁₂ ─╯
                                                 
 决策函数:                            x₁ ──→ w₂₁ ─╮
 ŷ = σ(w₁x₁ + w₂x₂ + b)                           ├─→ Σ₂ → Sigmoid → ŷ₂
                                     x₂ ──→ w₂₂ ─╯
                                     
                                     多层感知机（MLP）：
                                     
                                     x₁ ──→ [隐藏层1] → [隐藏层2] → ... → ŷ
                                     x₂ ──→ [ 128神经元]→[ 512神经元]→ [输出层]
```

| 模型 | 层数 | 表达能力 |
|:---|:---|:---|
| 逻辑回归 | 1 层（输入→输出） | 只能解决**线性可分**问题 |
| 单层感知机 | 1 层（输入→输出，多神经元） | 多分类线性可分 |
| **多层感知机（MLP）** | ≥2 层（含隐藏层） | 解决**非线性**问题（万能逼近定理） |

#### 1.2 神经网络训练四步曲

```
for epoch in range(epochs):
    for each sample (x, y):
        ① 前向传播    y_hat = model(x)        # 从输入计算预测值
        ② 损失计算    loss = loss_fn(y_hat, y) # 衡量预测与真实的差距
        ③ 反向传播    loss.backward()          # 计算每个参数的梯度 ∂loss/∂w
        ④ 参数更新    w = w - lr * w.grad      # 沿梯度负方向更新
```

---

### 第二章：手动实现逻辑回归（PyTorch 手写版）

> 本章通过**不使用 `nn.Module`** 的方式手写逻辑回归，目的在于理解 PyTorch 的**自动微分（Autograd）**机制和**梯度下降**的底层实现。

#### 2.1 数据准备

```python
import torch
import torch.nn.functional as F

n_item = 1000        # 样本容量
n_feature = 2        # 特征维度
learning_rate = 0.001
epochs = 100

# 构造假数据
torch.manual_seed(123)
data_x = torch.randn(size=(n_item, n_feature)).float()

# 标签规则：0.5*x₁ - 1.5*x₂ > 0 → 标签1，否则标签0
data_y = torch.where(
    torch.subtract(data_x[:, 0]*0.5, data_x[:, 1]*1.5) > 0,
    1., 0.
).float()
```

#### 2.2 手动构建模型类

```python
class LogisticRegressionManually(object):
    def __init__(self):
        # 构造待估参数（requires_grad=True 表示需要计算梯度）
        self.w = torch.randn(size=(n_feature, 1), requires_grad=True)
        self.b = torch.zeros(size=(1, 1), requires_grad=True)
    
    # 前向计算：ŷ = σ(wᵀx + b)
    def forward(self, x):
        y_hat = F.sigmoid(
            torch.matmul(self.w.transpose(0, 1), x) + self.b
        )
        return y_hat
    
    # 二分类交叉熵损失：L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    @staticmethod
    def loss_func(y_hat, y):
        return -(torch.log(y_hat)*y + (1-y)*torch.log(1-y_hat))
    
    # 训练过程
    def train(self):
        for epoch in range(epochs):
            for step in range(n_item):
                # ① 前向计算
                y_hat = self.forward(data_x[step])
                y = data_y[step]
                # ② 损失计算
                loss = self.loss_func(y_hat, y)
                # ③ 反向传播（自动求导）
                loss.backward()
                # ④ 参数更新（torch.no_grad() 上下文禁止梯度跟踪）
                with torch.no_grad():
                    self.w.data -= learning_rate * self.w.grad.data
                    self.b.data -= learning_rate * self.b.grad.data
                # ⑤ 梯度清零
                self.w.grad.data.zero_()
                self.b.grad.data.zero_()
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
```

执行结果（Loss 持续下降，说明模型在正常学习）：

```
Epoch:  0, Loss: 0.562
Epoch: 10, Loss: 0.476
Epoch: 20, Loss: 0.436
Epoch: 30, Loss: 0.409
...
Epoch: 90, Loss: 0.320
```

#### 2.3 关键知识点

##### requires_grad — PyTorch 自动微分的开关

| 设置 | 效果 |
|:---|:---|
| `requires_grad=True` | PyTorch 会追踪该张量的所有运算，`backward()` 时自动计算 $\partial L / \partial w$ |
| `requires_grad=False` | 该张量不参与梯度计算（如输入数据、标签） |

##### torch.no_grad() — 为什么参数更新需要它？

```python
# ❌ 错误：不带 no_grad()，更新操作也会被记录到计算图中
self.w.data -= lr * self.w.grad.data  # 会构建新的计算图！

# ✅ 正确：带 no_grad()，更新操作不会被追踪
with torch.no_grad():
    self.w.data -= lr * self.w.grad.data
```

> **原因**：参数更新是**优化算法**的一部分，不是模型前向计算的一部分。如果更新操作也被记录到计算图中，`backward()` 会尝试对更新公式本身求导，导致计算图膨胀甚至报错。

##### 梯度清零 — 为什么必须手动清零？

PyTorch 的 `.backward()` 执行的是**梯度累加**（gradient accumulation），而非覆盖。如果不清零，`w.grad` 会累积多轮的值：

```python
# 每轮训练后必须清零
self.w.grad.data.zero_()
self.b.grad.data.zero_()
```

| 清零时机 | 说明 |
|:---|:---|
| 应在每次 `.backward()` 后 | 防止梯度累积 |
| 或者在下一轮前向传播前 | 但放在 `.step()` 后更清晰 |

---

### 第三章：PyTorch nn.Module 构建神经网络

> 本章使用 PyTorch 的 `nn.Module` 标准 API，构建单层感知机和多层感知机（MLP），并完成完整的 GPU 训练流程。

#### 3.1 GPU 设备配置

```python
import torch
from torch import nn
from torch.nn import functional as F

# 自动选择可用设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

> `cuda` 是 NVIDIA GPU 的并行计算平台。通过 `.to(device)` 可以将张量和模型迁移到 GPU，加速训练。

#### 3.2 数据准备（含独热编码）

```python
n_item = 1000
n_feature = 2

torch.manual_seed(123)
data_x = torch.randn(size=(n_item, n_feature)).float()
data_y = torch.where(
    torch.subtract(data_x[:, 0]*0.5, data_x[:, 1]*1.5) + 0.02 > 0,
    0, 1
).long()

# 标签独热编码：0 → [1, 0]，1 → [0, 1]
data_y = F.one_hot(data_y)
# shape: [1000, 2]
```

| `F.one_hot` | 说明 |
|:---|:---|
| 输入 | 整数标签张量，shape `[N]` |
| 输出 | 独热编码张量，shape `[N, num_classes]` |
| 示例 | `[0, 1, 1, 0]` → `[[1,0],[0,1],[0,1],[1,0]]` |

#### 3.3 构建二分类模型

```python
class BinaryClassificationModel(nn.Module):
    def __init__(self, in_feature):
        super(BinaryClassificationModel, self).__init__()
        
        # 多层感知机（MLP）
        self.layer_1 = nn.Linear(
            in_features=in_feature,   # 输入维度 = 2
            out_features=128,         # 隐藏层1：128个神经元
            bias=True                 # 包含偏置
        )
        self.layer_2 = nn.Linear(
            in_features=128,          # 输入 = 上一层输出
            out_features=512,         # 隐藏层2：512个神经元
            bias=True
        )
        self.layer_final = nn.Linear(
            in_features=512,
            out_features=2,           # 输出层：二分类 → 2个神经元
            bias=True
        )
    
    def forward(self, x):
        layer_1_output = F.sigmoid(self.layer_1(x))
        layer_2_output = F.sigmoid(self.layer_2(layer_1_output))
        output = F.sigmoid(self.layer_final(layer_2_output))
        return output
```

#### 3.4 模型层次结构

```
输入层 (2)         隐藏层1 (128)        隐藏层2 (512)         输出层 (2)
                                                          
 x₁ ──────┬──→ [h₁₁] ────sigmoid──→ [h₂₁] ────sigmoid──→ [ŷ₁]
           │     ×128                   ×512                 ×2
 x₂ ──────┴──→ [h₁₁₂₈] ──sigmoid──→ [h₂₅₁₂] ──sigmoid──→ [ŷ₂]
```

| 组件 | API | 说明 |
|:---|:---|:---|
| 全连接层 | `nn.Linear(in, out, bias=True)` | $y = xW^T + b$，可学习参数 $W$(out×in) 和 $b$(out) |
| Sigmoid 激活 | `F.sigmoid(x)` | $\sigma(x) = \frac{1}{1+e^{-x}}$，输出 ∈ (0,1) |
| 模型基类 | `nn.Module` | 必须继承，重写 `__init__` 和 `forward` |

##### 单层感知机 vs 多层感知机

```python
# 单层感知机（线性二分类）
self.layer_1 = nn.Linear(in_features=in_feature, out_features=2, bias=True)

# 多层感知机（非线性二分类）
self.layer_1 = nn.Linear(in_features=in_feature, out_features=128)    # ← 第一隐藏层
self.layer_2 = nn.Linear(in_features=128, out_features=512)           # ← 第二隐藏层
self.layer_final = nn.Linear(in_features=512, out_features=2)         # ← 输出层
```

| 对比 | 单层感知机 | 多层感知机（MLP） |
|:---|:---|:---|
| 层数 | 1（输入→输出） | ≥2（含隐藏层） |
| 激活函数 | 可选 | **必须有**（否则多层退化为一层） |
| 可解决问题 | 线性可分 | 非线性 |
| 参数量 | 较少 | 更多（128×2 + 512×128 + 2×512 ≈ 66k） |

#### 3.5 训练循环（五步法）

```python
# 超参数
learning_rate = 0.01
epochs = 100

# 实例化模型并迁移到 GPU
model = BinaryClassificationModel(n_feature).to(device)

# 优化器（SGD）
opt = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 损失函数（二分类交叉熵）
criteria = nn.BCELoss()

# 训练循环
for epoch in range(epochs):
    for step in range(n_item):
        x = data_x[step].to(device)
        y = data_y[step].to(device)
        
        # ① 梯度清零
        opt.zero_grad()
        
        # ② 前向计算
        y_hat = model(x.unsqueeze(0))  # [1, 2]
        
        # ③ 损失计算
        loss = criteria(y_hat, y.unsqueeze(0).float())
        
        # ④ 反向传播
        loss.backward()
        
        # ⑤ 参数更新
        opt.step()
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss.cpu().item():.3f}')
```

训练结果（Loss 快速收敛）：

```
Epoch: 000, Loss: 0.323
Epoch: 010, Loss: 0.136
Epoch: 020, Loss: 0.055
Epoch: 030, Loss: 0.026
Epoch: 040, Loss: 0.014
...
Epoch: 090, Loss: 0.001
```

#### 3.6 优化器与损失函数速查

##### torch.optim 常用优化器

| 优化器 | 导入 | 适用场景 |
|:---|:---|:---|
| **SGD** | `torch.optim.SGD(params, lr)` | 基础随机梯度下降 |
| SGD + Momentum | `torch.optim.SGD(params, lr, momentum=0.9)` | 加速收敛 |
| **Adam** | `torch.optim.Adam(params, lr)` | **默认首选**（自适应学习率） |
| RMSprop | `torch.optim.RMSprop(params, lr)` | RNN 常用 |

##### nn 常用损失函数

| 损失函数 | API | 适用任务 |
|:---|:---|:---|
| **BCELoss** | `nn.BCELoss()` | 二分类（需 Sigmoid 输出） |
| **BCEWithLogitsLoss** | `nn.BCEWithLogitsLoss()` | 二分类（内置 Sigmoid，数值更稳定） |
| **CrossEntropyLoss** | `nn.CrossEntropyLoss()` | 多分类（内置 Softmax） |
| MSELoss | `nn.MSELoss()` | 回归任务 |

---

### 第四章：手写版 vs nn.Module 版对比

| 对比维度 | 手写逻辑回归 | nn.Module + 优化器 |
|:---|:---|:---|
| **参数定义** | 手动 `torch.randn(requires_grad=True)` | `nn.Linear` 自动管理 |
| **前向计算** | 手动 `torch.matmul(w.T, x) + b` | `model(x)` 自动调用 `forward` |
| **参数更新** | 手动 `w.data -= lr * w.grad.data` | `opt.step()` 自动更新 |
| **梯度清零** | 手动 `w.grad.data.zero_()` | `opt.zero_grad()` 一键清零 |
| **代码量** | 多（核心逻辑清晰可见） | 少（框架封装） |
| **灵活性** | 最高 | 中（受框架约束） |
| **学习目的** | 理解底层机制 | 实际项目开发 |

> **建议**：学习时先手写一遍，理解自动微分和梯度下降的底层机制；开发时使用 `nn.Module` + `optim`，代码简洁且不易出错。

---

### 第五章：训练循环五步法速记

```
┌─────────────────────────────────────────────────┐
│  ① opt.zero_grad()     梯度清零                  │
│          ↓                                      │
│  ② y_hat = model(x)    前向计算                  │
│          ↓                                      │
│  ③ loss = criteria(y_hat, y)  损失计算           │
│          ↓                                      │
│  ④ loss.backward()     反向传播（计算梯度）       │
│          ↓                                      │
│  ⑤ opt.step()          参数更新                  │
└─────────────────────────────────────────────────┘
```

> **记忆口诀**："清零-前向-损失-反向-更新"，缺一不可。

---

### 第六章：Notebook 实战总结

| 步骤 | 内容 | 核心技术点 |
|:---|:---|:---|
| ① 导入 | `torch` + `torch.nn.functional` | PyTorch 基础 |
| ② 数据准备 | 1000 样本 × 2 特征 + 线性决策规则 | `torch.randn` + `torch.where` |
| ③ 手写逻辑回归 | 自定义类 + 手动前向/反向/更新 | `requires_grad`、`backward`、`no_grad` |
| ④ 执行训练 | 100 轮 × 1000 样本 = 10 万次迭代 | Loss 从 0.562 降至 0.320 |
| ⑤ nn.Module 导包 | `nn` + `functional` + `device` | GPU 配置 |
| ⑥ 数据独热编码 | `F.one_hot(data_y)` | 标签格式转换 |
| ⑦ MLP 模型 | 2 → 128 → 512 → 2 | `nn.Linear` + `F.sigmoid` |
| ⑧ 优化器+损失 | `SGD(lr=0.01)` + `BCELoss` | 标准配置 |
| ⑨ 训练循环 | 100 轮 × 1000 样本 | Loss 从 0.323 降至 0.001 |

---

## 环境要求

| 依赖包 | 用途 |
|:---|:---|
| Python 3.8+ | 编程语言 |
| PyTorch ≥ 1.10 | 自动微分、nn.Module、优化器、损失函数 |

安装命令：

```bash
# CPU 版
pip install torch

# CUDA 11.8 版（有 NVIDIA GPU 时推荐）
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 使用方法

1. 启动 Jupyter Notebook：
   ```bash
   jupyter notebook
   ```
2. 在浏览器中打开 `13th_Neural_Network_Basics.ipynb`
3. 按顺序运行每个代码单元格（`Shift + Enter`）
4. 观察手写逻辑回归的 Loss 下降过程
5. 对比单层感知机和多层感知机的代码差异
6. 如有 GPU，观察 `.to(device)` 的加速效果

---

## 学习建议

1. **先手写再框架**：亲手写完一遍手写逻辑回归的 4 个步骤，再使用 `nn.Module`，理解框架在背后做了什么
2. **理解 `requires_grad`**：在任意张量上测试 `.requires_grad` 的行为，打印 `.grad` 观察梯度变化
3. **调试梯度**：在 `backward()` 前后打印 `model.layer_1.weight.grad`，观察梯度的产生和变化
4. **对比优化器**：将 `SGD` 换成 `Adam`，观察 Loss 下降速度的差异
5. **修改网络结构**：修改隐藏层的神经元数量（128→64、512→256），观察对 Loss 收敛的影响
6. **尝试不同损失函数**：将 `BCELoss` 替换为 `BCEWithLogitsLoss`（需移除 Sigmoid），理解数值稳定性
7. **可视化决策边界**：对 2 维数据绘制模型的决策边界，直观感受 MLP 的非线性分类能力

---

## 常见问题

### Q1: 为什么要 `loss.backward()` 之前先 `zero_grad()`？

PyTorch 的梯度是**累加**的。如果不清零，当前 batch 的梯度会叠加到之前的梯度上，导致参数更新错误：

```python
# ❌ 错误：梯度累加导致 w.grad 越来越大
for x, y in data:
    y_hat = model(x)
    loss = criteria(y_hat, y)
    loss.backward()       # 梯度累加！
    opt.step()

# ✅ 正确
for x, y in data:
    opt.zero_grad()       # 先清零
    y_hat = model(x)
    loss = criteria(y_hat, y)
    loss.backward()
    opt.step()
```

> **例外**：当显存不足以支持大 batch size 时，可以故意不清零，用多次小 batch 的梯度累加模拟大 batch。

### Q2: 手写版中为何要用 `with torch.no_grad()`？

```python
# ❌ 不带 no_grad
self.w.data -= lr * self.w.grad.data  # 这个减法操作会被记录到计算图！

# ✅ 带 no_grad
with torch.no_grad():
    self.w.data -= lr * self.w.grad.data  # 更新操作不记录，不干扰下次 backward
```

参数更新是**优化算法**的操作，不是模型计算图的一部分。`no_grad()` 隔离了更新操作和计算图。

### Q3: 什么时候用 Sigmoid，什么时候不用？

| 场景 | 建议 |
|:---|:---|
| 二分类 + `BCELoss` | 最后一层需要**手动加 Sigmoid** |
| 二分类 + `BCEWithLogitsLoss` | **不需要**手动加 Sigmoid（Loss 内置） |
| 多分类 + `CrossEntropyLoss` | **不需要** Softmax（Loss 内置） |
| MLP 隐藏层 | 需要激活函数（Sigmoid / ReLU / Tanh） |

> **推荐**：二分类用 `BCEWithLogitsLoss` 代替 `BCELoss` + Sigmoid，数值更稳定。

### Q4: 单层感知机和逻辑回归有什么区别？

| 对比 | 逻辑回归 | 单层感知机 |
|:---|:---|:---|
| 输出层神经元数 | 1（直接输出概率） | 可多个（每个类别一个神经元） |
| 激活函数 | Sigmoid | Sigmoid（但也可不用） |
| 损失函数 | BCELoss | BCELoss / CrossEntropyLoss |

> 实质上，输出层只有 1 个神经元 + Sigmoid 的单层感知机**就是**逻辑回归。

### Q5: 多层感知机为什么隐藏层需要激活函数？

如果没有激活函数（或只用线性激活），多层线性变换的组合仍然是线性的：

$$W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2) = W'x + b'$$

多层等价于一层！**激活函数引入非线性**，使 MLP 能够逼近任意复杂函数。

### Q6: 如何确认 PyTorch 在使用 GPU？

```python
# 检查 CUDA 是否可用
torch.cuda.is_available()  # True/False

# 查看张量所在设备
x.device  # cpu / cuda:0

# 模型参数所在设备
next(model.parameters()).device
```

---

## 下一步

完成本教程后，建议进入深度学习后续内容：
- **CNN（卷积神经网络）**：图像特征提取
- **RNN/LSTM**：序列数据处理
- 或返回 [12th_EnsembleLearning.md](12th_EnsembleLearning.md) 回顾集成学习方法

---

**最后更新**：2026-06-20
