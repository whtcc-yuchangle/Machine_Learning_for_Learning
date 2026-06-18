# SVM 支持向量机（Support Vector Machine）教程

## 概述

本 notebook 从两个层次介绍支持向量机算法：**① 使用 sklearn 高层 API** 快速构建 SVM 分类器，在鸢尾花数据集上进行实战；**② 使用 PyTorch 手动实现**线性 SVM，包括 Hinge Loss（合页损失）和 SGD 梯度下降参数更新，帮助理解底层数学原理。此外，前半部分通过 2D/3D 可视化直观演示了**最大间隔分类超平面**的核心概念。

---

## 学习目标

通过学习本教程，您将掌握：

1. **SVM 核心思想**：理解最大间隔分类器、支持向量、决策边界的概念
2. **线性可分与非线性可分**：理解为什么需要核函数（Kernel Trick）将低维数据映射到高维
3. **sklearn SVC**：使用 `SVC` 完成多分类任务，掌握 `kernel`、`C` 等核心参数
4. **Hinge Loss**：理解合页损失函数的数学含义及其与最大间隔目标的关系
5. **PyTorch 手动实现**：从零构建线性 SVM，理解前向传播、合页损失、反向传播和 SGD
6. **实战案例**：鸢尾花分类 + 模拟数据二分类

---

## 内容结构

### 第一章：SVM 算法原理

#### 1.1 什么是支持向量机？

SVM 的核心目标是找到一个**最优超平面（Optimal Hyperplane）**，使不同类别的样本之间的**间隔（Margin）最大化**：

- **超平面**：在 $d$ 维空间中，超平面是一个 $d-1$ 维的线性判定面，方程为：

$$\mathbf{w}^T\mathbf{x} + b = 0$$

- **支持向量**：距离超平面最近的训练样本点，这些点"支撑"了超平面的位置
- **间隔**：两类支持向量到超平面的距离之和，即 $\text{margin} = \frac{2}{\|\mathbf{w}\|}$

最大化间隔等价于最小化 $\|\mathbf{w}\|$，同时满足所有样本被正确分类的约束：

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \ \forall i$$

> **直观理解**：SVM 不只关心把所有样本分对，而是寻找距离所有样本都"最远"的那条分界线——就像在两类之间画一条"最宽的马路"，支持向量就是路边上的"路灯"。

#### 1.2 软间隔（Soft Margin）与参数 C

现实数据往往不是严格线性可分的。SVM 通过引入**松弛变量 $\xi_i$** 允许部分样本被误分类：

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\xi_i$$

| 参数 | 说明 |
|:---|:---|
| $C$ 很大 | 惩罚误分类严重 → 硬间隔，可能过拟合 |
| $C$ 很小 | 容忍更多误分类 → 软间隔，泛化能力更强 |

$$
\text{支撑向量：} \quad
\begin{cases}
\alpha_i = 0 & \text{正确分类且远离边界} \\
0 < \alpha_i < C & \text{正确分类且恰在边界上（支持向量）} \\
\alpha_i = C & \text{误分类或落入间隔内部}
\end{cases}
$$

> 在 sklearn 中，`C=1.0` 是默认值。C 越大，模型越"较真"于每一个样本，边界越复杂。

#### 1.3 核函数（Kernel Trick）

对于非线性可分数据，SVM 通过核函数将数据**隐式映射**到高维空间，在高维空间中寻找线性超平面：

| 核函数 | sklearn 参数 | 公式 | 适用场景 |
|:---|:---|:---|:---|
| **线性核** | `kernel='linear'` | $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T\mathbf{x}_j$ | 特征维度高、线性可分 |
| **多项式核** | `kernel='poly'` | $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma\mathbf{x}_i^T\mathbf{x}_j + r)^d$ | 中等非线性 |
| **RBF 核** | `kernel='rbf'`（默认） | $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma\|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | 通用，最常用 |
| **Sigmoid 核** | `kernel='sigmoid'` | $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma\mathbf{x}_i^T\mathbf{x}_j + r)$ | 类似神经网络激活 |

**为什么需要核函数？**

```
在二维平面中，两类样本可能无法用一条直线分开。
但将它们映射到三维空间后（如添加 z = x² + y²），
可能就变得线性可分了——核函数帮你做这件事，且不显式计算高维坐标。
```

#### 1.4 Hinge Loss（合页损失）

从损失函数角度看，SVM 等价于使用**合页损失 + L2 正则化**：

$$L(\mathbf{w}, b) = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{n}\max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b))$$

| $y \cdot f(x)$ 的值 | Hinge Loss | 含义 |
|:---|:---|:---|
| $\geq 1$ | $0$ | 正确分类且在安全间隔外，无损失 ✅ |
| $0 \sim 1$ | $1 - yf(x) > 0$ | 正确分类但在间隔内，有损失 ⚠️ |
| $< 0$ | $\geq 1$ | 误分类，损失较大 ❌ |

> **对比**：逻辑回归用交叉熵损失（处处可导），SVM 用合页损失（只有"不够好"的样本才有梯度）。这使得 SVM 的解更"稀疏"——只依赖于支持向量。

---

### 第二章：Notebook 上半部分 — 可视化理解 SVM

本章对应 notebook 中前 3 个代码单元格，通过 2D/3D 散点图直观演示超平面和间隔的概念。

#### 2.1 单元格 1 — 2D 线性可分数据与决策边界

**生成数据**：生成两类样本（Class 1 中心在 $(-2,-2)$，Class 2 中心在 $(2,2)$），绘制散点图和一条手动放置的分隔线。

```python
np.random.seed(123)
class1 = np.random.randn(50, 2) - [2, 2]
class2 = np.random.randn(50, 2) + [2, 2]

plt.scatter(class1[:, 0], class1[:, 1], marker='o', label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], marker='x', label='Class 2')
plt.plot([-2, 4], [4, -2], color='red', linestyle='dashed', linewidth=2, label='Decision Boundary')
```

> **说明**：红色虚线代表分离超平面（在这里是二维平面中的一条直线）。SVM 的目标本质就是自动学习出这样一条"最优"的分界线——使得两类样本到这条线的距离最大。

生成的图片保存为 `09th_img/img1.png`。

#### 2.2 单元格 2 — 3D 线性可分数据与分离平面

**生成 3D 数据**：两类样本在三维空间中分布，叠加一个灰色半透明分离平面。

```python
class_0 = np.random.rand(num_samples, 3) * 5
class_1 = np.random.rand(num_samples, 3) * 5 + 6   # 偏移 6 个单位

# 分离平面：法向量 [1, -1, 1]，过点 [5, 5, 5]
normal_vector = np.array([1, -1, 1])
point_on_plane = np.array([5, 5, 5])
```

> **说明**：在三维空间中，分离面是一个**二维平面**（而非直线），这就是"超平面（Hyperplane）"的来源——在 d 维空间中，超平面是 d-1 维的。核函数的核心思想就是将低维中不可分的数据映射到像这样的高维空间。

生成的图片保存为 `09th_img/img2.png`。

#### 2.3 单元格 3 — 3D 随机样本分布（对比）

**目的**：展示随机分布、没有明显分离趋势的数据在 3D 空间中的样子，与前面线性可分的情况形成对比。

```python
class_0 = np.random.rand(num_samples, 3) * 10
class_1 = np.random.rand(num_samples, 3) * 10
```

> **说明**：当两类样本完全混合时，即使在三维空间中也找不到一个平面将它们分开。此时需要 RBF 核等非线性方法，将数据映射到更高维度来寻找分离超平面。

生成的图片保存为 `09th_img/img3.png`。

---

### 第三章：Notebook 下半部分 — sklearn SVC 实战（鸢尾花分类）

#### 3.1 数据集介绍

使用 sklearn 自带的 **Iris Dataset**（鸢尾花数据集）：

| 属性 | 说明 |
|:---|:---|
| 样本数量 | 150 个 |
| 特征数量 | 4 个（花萼长度、花萼宽度、花瓣长度、花瓣宽度） |
| 类别数量 | 3 个（Setosa / Versicolour / Virginica） |
| 类别分布 | 均衡（每类 50 个） |
| 数据形状 | `(150, 4)` |

#### 3.2 单元格 4 详解 — sklearn SVC 训练与评估

**步骤一：导入模块**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False
```

**步骤二：加载并划分数据集**

```python
iris = load_iris()
x = iris.data   # (150, 4)
y = iris.target # (150,)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    x, y, test_size=0.3, random_state=42
)
```

> 训练集 105 条，测试集 45 条。`random_state=42` 固定随机种子。

**步骤三：构建并训练 SVM 模型**

```python
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(Xtrain, Ytrain)
```

> **参数说明**：
> - `kernel='linear'`：使用线性核，此时 SVM 退化为线性最大间隔分类器
> - `C=1.0`：正则化参数，控制对误分类的惩罚强度
> - `random_state=42`：固定随机种子

**步骤四：预测与评估**

```python
y_pred = clf.predict(Xtest)
accuracy_score(Ytest, y_pred)        # 1.0 → 100% 准确率
print(classification_report(Ytest, y_pred))
```

输出结果：

```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        13
           2       1.00      1.00      1.00        13

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45
```

> **说明**：线性核 SVM 在鸢尾花数据集上达到了 **100% 准确率**。这是因为鸢尾花的类别之间线性可分程度很高。每个类别都在 1.0 的精确率和召回率，说明模型完美区分了三种鸢尾花。

#### 3.3 SVC 核心参数速查

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `kernel` | `'rbf'` | 核函数类型：`'linear'`, `'poly'`, `'rbf'`, `'sigmoid'` |
| `C` | `1.0` | 正则化参数，越大惩罚越重（越不容忍错误） |
| `gamma` | `'scale'` | RBF/多项式/sigmoid 核的系数，`'scale'` 自动计算为 `1/(n_features * X.var())` |
| `degree` | `3` | 多项式核的阶数（仅 `kernel='poly'` 有效） |
| `probability` | `False` | 是否启用概率估计（会降低速度） |
| `class_weight` | `None` | 类别权重，`'balanced'` 自动按频率反比加权 |
| `decision_function_shape` | `'ovr'` | 多分类策略：`'ovr'`（一对多）或 `'ovo'`（一对一） |

---

### 第四章：Notebook 下半部分 — PyTorch 手动实现 SVM

下半部分用 PyTorch **从零手动构建**线性 SVM 模型，帮助理解 Hinge Loss 和梯度下降的底层细节。这里做的是**二分类**（用模拟数据），使用**合页损失（Hinge Embedding Loss）**和 SGD 优化器。

#### 4.1 单元格 5 详解 — PyTorch SVM 实现

**步骤一：导入 PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

> **说明**：自动检测 GPU，如果可用则将数据和模型移至 GPU 训练。

**步骤二：定义 SVM 模型类**

```python
class SVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)
```

> **说明**：线性 SVM 本质上就是一个**线性层（全连接层）**。`nn.Linear(2, 1)` 等价于 $\mathbf{w}^T\mathbf{x} + b$，其中 $\mathbf{w}$ 是 2 维权重向量，$b$ 是偏置。

**步骤三：构造训练数据**

```python
# 4 个样本，2 个特征，模仿 XOR 简化版
x_train = torch.tensor([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]]).to(device)
y_train = torch.tensor([1., -1., -1., -1.]).to(device)
```

> **说明**：训练数据的标签为 $\{1, -1\}$（而非 $\{0, 1\}$），这是配合 `HingeEmbeddingLoss` 的要求。`HingeEmbeddingLoss` 定义如下：
>
> $$L_n = \begin{cases} x_n & \text{if } y_n = 1 \\ \max(0, \Delta - x_n) & \text{if } y_n = -1 \end{cases}$$
>
> 其中 $\Delta$ 默认为 1。

**步骤四：定义损失函数和优化器**

```python
svm = SVM(input_size=2, num_classes=1).to(device)
criterion = nn.HingeEmbeddingLoss()
optimizer = optim.SGD(svm.parameters(), lr=0.01)
```

| 组件 | PyTorch 实现 | 数学含义 |
|:---|:---|:---|
| **模型** | `nn.Linear(2, 1)` | $f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b$ |
| **损失** | `nn.HingeEmbeddingLoss()` | $L = \max(0, 1 - y \cdot f(\mathbf{x}))$ |
| **优化** | `optim.SGD(lr=0.01)` | $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L$ |

**步骤五：训练循环**

```python
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = svm(x_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

训练过程输出：

```
Epoch 0, Loss: 0.4347
Epoch 100, Loss: 0.1105
Epoch 200, Loss: -0.0562
Epoch 300, Loss: -0.2226
Epoch 400, Loss: -0.3895
Epoch 500, Loss: -0.5562
Epoch 600, Loss: -0.6840
Epoch 700, Loss: -0.8090
Epoch 800, Loss: -0.9340
Epoch 900, Loss: -1.0590
```

> **说明**：HingeEmbeddingLoss 允许损失为**负值**，这表示样本被正确分类且输出值已超过间隔要求（$y \cdot f(x) > 1$）。损失持续下降说明模型在不断提升分类置信度。

**步骤六：测试模型**

```python
x_test = torch.tensor([[2., 2.], [-2., 2.], [-2., -2.], [2., -2.]])
outputs = svm(x_test)
predicted = torch.sign(outputs).squeeze().cpu()
# tensor([-1., 1., 1., 1.])
```

> **说明**：使用 `torch.sign()` 将模型输出映射为类别标签（$\geq 0 \to +1, < 0 \to -1$）。测试数据中只有 $(-2, -2)$ 对应的输出 $\geq 0$，预测为 $+1$ 类；其余三个预测为 $-1$ 类。

#### 4.2 PyTorch 训练循环 5 个关键步骤

| 步骤 | 代码 | 含义 |
|:---|:---|:---|
| **Zero grad** | `optimizer.zero_grad()` | 清零上一步的梯度（PyTorch 默认累加） |
| **Forward** | `outputs = svm(x_train)` | 前向计算 $f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b$ |
| **Loss** | `criterion(outputs, y_train)` | 计算合页损失 |
| **Backward** | `loss.backward()` | 自动求 $\frac{\partial L}{\partial \mathbf{w}}$ 和 $\frac{\partial L}{\partial b}$ |
| **Update** | `optimizer.step()` | SGD 更新参数 $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L$ |

---

### 第五章：sklearn vs PyTorch 手动实现对比

| 对比维度 | sklearn（上半部分） | PyTorch 手动（下半部分） |
|:---|:---|:---|
| 抽象层级 | 高层 API，一行 `fit()` | 底层实现，手动前向/反向传播 |
| 损失函数 | 内部自动选择（Hinge + L2） | 显式使用 `nn.HingeEmbeddingLoss()` |
| 优化算法 | SMO / 二次规划求解 | SGD（随机梯度下降） |
| 分类类型 | 多分类（OvR/OvO） | 二分类 |
| 核函数 | 支持 linear/poly/rbf/sigmoid | 仅线性 |
| 代码量 | ~5 行核心代码 | ~25 行核心代码 |
| 学习价值 | 快速应用、调参实战 | 理解 Hinge Loss 和梯度下降 |

---

### 第六章：SVM 算法特点

#### 6.1 优点

| 优点 | 说明 |
|:---|:---|
| **理论上最优** | 凸优化问题，解是全局最优（不像神经网络可能陷入局部最优） |
| **核技巧强大** | 通过核函数高效处理非线性问题，无需显式计算高维映射 |
| **对高维数据有效** | 在高维空间（特征数 > 样本数）中依然表现良好 |
| **泛化能力强** | 最大间隔策略天然具有正则化效果 |
| **稀疏性** | 模型只依赖支持向量，预测时计算量与支持向量数成正比 |

#### 6.2 缺点

| 缺点 | 说明 |
|:---|:---|
| **大数据集效率低** | 训练时间复杂度 $O(n^2) \sim O(n^3)$，不适合超大样本 |
| **核函数选择困难** | RBF 核通常效果不错，但最优核函数需要经验或调参 |
| **概率输出需额外处理** | 原生不输出概率，需启用 `probability=True`（内部使用 Platt Scaling） |
| **对参数敏感** | C 和 gamma 的选择对结果影响很大，需要仔细调参 |
| **可解释性弱于决策树** | 决策边界难以直观向非技术人员解释 |

#### 6.3 适用场景

- **文本分类 / 高维稀疏数据**：如垃圾邮件检测、情感分析
- **图像分类**：手写数字识别、人脸检测
- **生物信息学**：基因表达分类、蛋白质结构预测
- **中小规模数据集**：样本数在几千到几万的场景
- **需要强泛化**：当训练数据有限但需要模型在新数据上表现稳定

---

## 环境要求

| 依赖包 | 用途 |
|:---|:---|
| Python 3.7+ | 编程语言 |
| numpy | 数值计算 |
| pandas | 数据处理 |
| matplotlib | 数据可视化 |
| scikit-learn | SVM 模型、数据集、评估指标 |
| PyTorch | 手动实现 SVM（下半部分） |

安装命令：

```bash
pip install numpy pandas matplotlib scikit-learn torch
```

---

## 使用方法

1. 启动 Jupyter Notebook：
   ```bash
   jupyter notebook
   ```
2. 在浏览器中打开 `09th_SVM.ipynb`
3. 按顺序运行每个代码单元格（`Shift + Enter`）

---

## 学习建议

1. **先看可视化部分**：通过 2D/3D 图理解"超平面分割数据"的直觉
2. **再跑 sklearn SVC**：用一行 `fit()` 感受 SVM 的强大，尝试修改 `kernel`（`'linear'` vs `'rbf'`）和 `C` 值观察效果
3. **深入 PyTorch 实现**：理解 Hinge Loss 怎样驱动参数更新，体会 SVM 的损失函数设计
4. **动手调参**：修改学习率 `lr`、训练轮数 `num_epochs`，观察收敛速度和最终损失
5. **横向对比**：在鸢尾花数据集上对比 SVM、逻辑回归（08th）、KNN（04th）、决策树（05th）的表现
6. **尝试不同核函数**：将 `kernel='linear'` 改为 `kernel='rbf'`，观察准确率变化和训练时间差异

---

## 常见问题

### Q1: SVM 和逻辑回归有什么区别？

| 对比维度 | SVM | 逻辑回归 |
|:---|:---|:---|
| 损失函数 | Hinge Loss | 交叉熵（Cross-Entropy） |
| 优化目标 | 最大化间隔 | 最大化似然 |
| 输出 | 决策值（需额外步骤得概率） | 天然输出概率 |
| 稀疏性 | 是（只依赖支持向量） | 否（所有样本都有影响） |
| 核函数 | 支持通过核技巧处理非线性 | 通常只处理线性 |
| 大样本 | 较慢 | 较快 |

**结论**：当只关心分类结果且数据量适中时，SVM 通常表现更好；当需要概率输出或数据量很大时，逻辑回归更合适。

### Q2: kernel 选 linear 还是 rbf？

| kernel | 适用场景 | 特点 |
|:---|:---|:---|
| `'linear'` | 特征维度很高（如文本 TF-IDF），或数据线性可分 | 参数少，训练快，不易过拟合 |
| `'rbf'`（默认） | 非线性数据，样本数不太大 | 最通用，需要调 `gamma` |

> 经验法则：如果特征数量 > 样本数量，用 `linear`；否则先用 `rbf` 试。

### Q3: C 和 gamma 怎么调？

| 参数 | 含义 | 偏大 | 偏小 |
|:---|:---|:---|:---|
| **C** | 误分类惩罚 | 低偏差、高方差（过拟合风险） | 高偏差、低方差（欠拟合风险） |
| **gamma** | RBF 核的影响半径 | 每个样本影响范围小（过拟合风险） | 每个样本影响范围大（过于平滑） |

> C 和 gamma 通常是**互相独立地调优**：先粗略确定数量级（如 $10^{-3}, 10^{-2}, ..., 10^3$），再用 GridSearchCV 精细搜索。

### Q4: SVM 如何处理多分类问题？

sklearn 的 `SVC` 默认使用 **OvO（One-vs-One）**策略：
- 训练 $\frac{K(K-1)}{2}$ 个二分类器（每两类之间一个）
- 预测时投票决定最终类别

也可通过 `decision_function_shape='ovr'` 切换为 **OvR（One-vs-Rest）**：
- 训练 K 个二分类器（每类 vs 其余）
- 选决策函数值最大的类别

> OvO 分类器更多但每个分类器只在两类子集上训练，训练更快；OvR 分类器少但每个需要全量数据。实践中两者差异通常不大。

### Q5: 为什么 PyTorch 实现中的损失会变为负数？

`nn.HingeEmbeddingLoss` 定义为：

$$
L_n = \begin{cases} x_n & \text{if } y_n = 1 \\ \max(0, \Delta - x_n) & \text{if } y_n = -1 \end{cases}
$$

当标签为 $y_n = 1$ 时，损失直接等于模型输出 $x_n$。如果模型输出为**负值**（意味着预测错误），损失就是负的——这在数值上可能看起来奇怪，但实际上是该损失的数学特性。`HingeEmbeddingLoss` 旨在衡量 $y=1$ 时输出应尽量大（正），$y=-1$ 时输出应尽量小（负），使得损失趋于 $-\infty$。

> 实际中更常用 `torch.nn.functional.hinge_embedding_loss` 或者自己实现 `max(0, 1 - y*f(x))` 形式的合页损失来避免此歧义。

### Q6: SVM 需要特征标准化吗？

**强烈建议标准化**。SVM 基于距离/间隔，如果特征量纲不一致（如一个特征范围 0-1，另一个 0-10000），大数值特征会主导间隔计算。使用 `StandardScaler`：

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clf = make_pipeline(StandardScaler(), SVC(kernel='rbf'))
clf.fit(Xtrain, Ytrain)
```

---

**最后更新**：2026-06-18
