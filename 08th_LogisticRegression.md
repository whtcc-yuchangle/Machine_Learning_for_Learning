# 逻辑回归（Logistic Regression）教程

## 概述

本 notebook 从两个层次介绍逻辑回归算法：**① 使用 sklearn 高层 API** 快速构建多分类模型，在鸢尾花数据集上进行实战；**② 使用 PyTorch 手动实现**二分类逻辑回归，包括 Sigmoid 函数、交叉熵损失函数、梯度下降参数更新，帮助理解底层数学原理。

---

## 学习目标

通过学习本教程，您将掌握：

1. **逻辑回归原理**：理解 Sigmoid 函数将线性输出映射为概率
2. **交叉熵损失**：理解二分类交叉熵损失函数的推导与含义
3. **sklearn 逻辑回归**：使用 `LogisticRegression` 完成多分类任务
4. **多分类策略**：理解 OvR（One-vs-Rest）多分类机制
5. **PyTorch 手动实现**：从零构建逻辑回归，理解前向传播、反向传播和梯度下降
6. **实战案例**：鸢尾花分类 + 模拟数据二分类

---

## 内容结构

### 第一章：逻辑回归算法原理

#### 1.1 为什么叫"回归"却是做分类？

逻辑回归的名字有历史原因——它在**线性回归的基础上**套了一个 Sigmoid 函数，将 $(-\infty, +\infty)$ 的线性输出压缩到 $(0, 1)$ 区间，从而表示概率：

$$z = \mathbf{w}^T\mathbf{x} + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

| $z$ 的取值 | $\sigma(z)$ 的概率含义 |
|:---|:---|
| $z \to +\infty$ | $\sigma(z) \to 1$（大概率正类） |
| $z = 0$ | $\sigma(z) = 0.5$（决策边界） |
| $z \to -\infty$ | $\sigma(z) \to 0$（大概率负类） |

最终的分类决策：

$$\text{预测类别} = \begin{cases} 1 & \text{if } \sigma(z) \ge 0.5 \\ 0 & \text{if } \sigma(z) < 0.5 \end{cases}$$

#### 1.2 损失函数：二分类交叉熵（Binary Cross-Entropy）

逻辑回归使用**交叉熵损失函数**（而非 MSE），因为交叉熵对概率输出的梯度更合理：

$$L(\hat{y}, y) = -\big[y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y})\big]$$

| $y$ | $\hat{y}$ 接近 | 损失 |
|:---|:---|:---|
| 1 | 1（预测正确且自信） | $-\log(1) \approx 0$ ✅ |
| 1 | 0（预测错误且自信） | $-\log(0) \to +\infty$ ❌ |
| 0 | 0（预测正确且自信） | $-\log(1) \approx 0$ ✅ |
| 0 | 1（预测错误且自信） | $-\log(0) \to +\infty$ ❌ |

> 直观理解：当模型对正确答案很"自信"时损失小，对错误答案"自信"时损失极大。

#### 1.3 多分类：One-vs-Rest（OvR）

逻辑回归原生是二分类模型。对于 K 类多分类问题，sklearn 默认使用 **OvR 策略**：

1. 训练 K 个二分类器，第 k 个分类器将"类别 k"作为正类，其余作为负类
2. 预测时，取 K 个分类器中输出概率最大的类别

| 属性 | 形状 | 含义 |
|:---|:---|:---|
| `coef_` | `(n_classes, n_features)` | 每个类别的权重向量 |
| `intercept_` | `(n_classes,)` | 每个类别的截距 |
| `predict_proba()` | `(n_samples, n_classes)` | 每个样本属于各类别的概率（和为 1） |

---

### 第二章：Notebook 上半部分 — sklearn 逻辑回归（鸢尾花分类）

#### 2.1 数据集介绍

使用 sklearn 自带的 **Iris Dataset**（鸢尾花数据集）：

| 属性 | 说明 |
|:---|:---|
| 样本数量 | 150 个 |
| 特征数量 | 4 个（花萼长度、花萼宽度、花瓣长度、花瓣宽度） |
| 类别数量 | 3 个（Setosa / Versicolour / Virginica） |
| 类别分布 | 均衡（每类 50 个） |
| 数据形状 | `(150, 4)` |

#### 2.2 Notebook 单元格详解

**上半部分 — sklearn 高层实现**

**单元格 1：导入模块**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
```

> **说明**：`plt.style.use('ggplot')` 设置图表风格为 R 语言 ggplot2 风格。

**单元格 2：加载鸢尾花数据集**

```python
iris_data = load_iris()
x = iris_data.data
y = iris_data.target

x.shape   # (150, 4)
y.shape   # (150,)
```

**单元格 3：数据探索**

```python
pd.DataFrame(x).head()
pd.DataFrame(y).value_counts()
```

输出：每类 50 个样本，类别均衡。

| 特征索引 | 特征名称 | 含义 |
|:---|:---|:---|
| 0 | sepal length (cm) | 花萼长度 |
| 1 | sepal width (cm) | 花萼宽度 |
| 2 | petal length (cm) | 花瓣长度 |
| 3 | petal width (cm) | 花瓣宽度 |

**单元格 4：数据集划分**

```python
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    x, y, test_size=0.3, random_state=123
)
Xtrain.shape   # (105, 4)
Xtest.shape    # (45, 4)
```

> **说明**：`random_state=123` 固定随机种子。训练集 105 条，测试集 45 条。

**单元格 5：构建并训练逻辑回归模型**

```python
# 构建逻辑回归模型
clf = LogisticRegression()

# 模型训练
clf = clf.fit(Xtrain, Ytrain)
```

> **说明**：默认参数下，`LogisticRegression` 使用：
> - `penalty='l2'`：L2 正则化
> - `C=1.0`：正则化强度的倒数（越小正则化越强）
> - `multi_class='ovr'`：One-vs-Rest 多分类策略（旧版默认；新版默认 `'auto'` 在二分类时等价）
> - `solver='lbfgs'`：拟牛顿法优化器

**单元格 6：预测并对比结果**

```python
y_pred = clf.predict(Xtest)
y_pred, Ytest
```

> **说明**：对比预测值和真实值，可以看出模型在大多数样本上预测正确。

**单元格 7：模型评估**

```python
accuracy_score(Ytest, y_pred)   # 0.9333
```

> **说明**：准确率 **93.33%**，45 个测试样本中约 42 个被正确分类（仅 3 个预测错误）。

**单元格 8：查看预测概率**

```python
clf.predict_proba(Xtest)
```

> **说明**：返回形状 `(45, 3)` 的概率矩阵。每行是 3 个类别的概率（和为 1）。概率最大值所在的列即为预测类别。例如某行 `[0.98, 0.02, 3.5e-08]` 表示模型以 98% 的置信度预测该样本属于类别 0。

**单元格 9：查看模型参数**

```python
clf.coef_        # shape (3, 4)
clf.intercept_   # shape (3,)
```

> **说明**：
> - `coef_` 是一个 3×4 的矩阵——因为 OvR 策略训练了 3 个二分类器（每类一个），每个分类器有 4 个特征权重
> - `intercept_` 是 3 个截距值（每个二分类器一个）
> - 含义：第 k 个分类器的决策函数为 $z_k = \mathbf{w}_k^T\mathbf{x} + b_k$

---

### 第三章：Notebook 下半部分 — PyTorch 手动实现二分类逻辑回归

下半部分用 PyTorch **从零手动构建**逻辑回归模型，帮助理解底层算法细节。这里只做**二分类**（用模拟数据），使用**随机梯度下降（SGD）**逐样本更新参数。

#### 3.1 模拟数据生成

**单元格 10：导入 PyTorch**

```python
import torch
import torch.nn.functional as F
```

**单元格 11：构造模拟数据**

```python
n_item = 1000        # 样本容量
n_feature = 2        # 特征维度
learning_rate = 0.001 # 学习率
epochs = 100          # 训练轮数

# 固定随机种子
torch.manual_seed(123)

# 生成 1000×2 的随机特征数据（标准正态分布）
data_x = torch.randn(size=(n_item, n_feature)).float()

# 构造标签：如果 0.5*x0 - 1.5*x1 > 0 则为 1，否则为 0
data_y = torch.where(
    torch.subtract(data_x[:, 0] * 0.5, data_x[:, 1] * 1.5) > 0,
    1., 0.
).float()
```

> **说明**：真实的决策边界为 $0.5x_0 - 1.5x_1 = 0$，即 $x_1 = \frac{1}{3}x_0$。模型的目标是从数据中学习出这个边界：
> - 将 $0.5x_0 - 1.5x_1 > 0$ 变形为 $\mathbf{w}^T\mathbf{x} > 0$，可得真实权重 $\mathbf{w} = [0.5, -1.5]$
> - 模型训练完成后，学到的权重应接近 $[0.5, -1.5]$

**单元格 12：手动实现逻辑回归类**

```python
class LogisticRegressionManually(object):
    def __init__(self):
        # 待估参数列向量 w（2×1），开启梯度追踪
        self.w = torch.randn(size=(n_feature, 1), requires_grad=True)
        # 偏置 b（1×1），开启梯度追踪
        self.b = torch.zeros(size=(1, 1), requires_grad=True)

    # 前向计算：sigmoid(w^T x + b)
    def forward(self, x):
        y_hat = F.sigmoid(
            torch.matmul(self.w.transpose(0, 1), x) + self.b
        )
        return y_hat

    # 损失函数：二分类交叉熵
    @staticmethod
    def loss_func(y_hat, y):
        return -(torch.log(y_hat) * y + (1 - y) * torch.log(1 - y_hat))

    # 训练过程：SGD 逐样本更新
    def train(self):
        for epoch in range(epochs):
            for step in range(n_item):
                # 1. 前向计算
                y_hat = self.forward(data_x[step])
                y = data_y[step]
                # 2. 损失计算
                loss = self.loss_func(y_hat, y)
                # 3. 反向传播（自动求导）
                loss.backward()
                # 4. 梯度下降更新参数（关闭梯度追踪）
                with torch.no_grad():
                    self.w.data -= learning_rate * self.w.grad.data
                    self.b.data -= learning_rate * self.b.grad.data
                # 5. 梯度清零（否则会累加）
                self.w.grad.data.zero_()
                self.b.grad.data.zero_()
            if epoch % 10 == 0:
                print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))
```

> **手动实现的 5 个关键步骤详解**：

| 步骤 | 代码 | 含义 |
|:---|:---|:---|
| **Forward** | `F.sigmoid(w^T x + b)` | 计算预测概率 $\hat{y}$ |
| **Loss** | `-[y·log(ŷ) + (1-y)·log(1-ŷ)]` | 计算交叉熵损失 |
| **Backward** | `loss.backward()` | 自动求 $\frac{\partial L}{\partial w}$ 和 $\frac{\partial L}{\partial b}$ |
| **Update** | `w -= lr * w.grad` | 沿负梯度方向更新参数 |
| **Zero grad** | `w.grad.zero_()` | 清除梯度（PyTorch 默认累加） |

**单元格 13：执行训练**

```python
if __name__ == '__main__':
    lrm = LogisticRegressionManually()
    lrm.train()
```

输出（损失持续下降）：

```
Epoch: 0,  Loss: 0.5621
Epoch: 10, Loss: 0.4762
Epoch: 20, Loss: 0.4364
Epoch: 30, Loss: 0.4089
Epoch: 40, Loss: 0.3875
Epoch: 50, Loss: 0.3699
Epoch: 60, Loss: 0.3548
Epoch: 70, Loss: 0.3417
Epoch: 80, Loss: 0.3300
Epoch: 90, Loss: 0.3195
```

> **说明**：损失从 0.56 下降至 0.32，说明模型正在逐步学习。经过 100 轮 × 1000 样本 = 10 万次参数更新后，模型权重应接近真实值 $[0.5, -1.5]$。

---

### 第四章：模型评估指标补充

除了准确率，二分类问题还有更多评估维度：

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(Ytest, y_pred, target_names=iris_data.target_names))
print(confusion_matrix(Ytest, y_pred))
```

| 指标 | 含义 |
|:---|:---|
| **Precision（精确率）** | 预测为正的样本中真正为正的比例 |
| **Recall（召回率）** | 真正为正的样本中被正确找出的比例 |
| **F1-score** | 精确率与召回率的调和平均 |
| **Confusion Matrix** | 行=真实类别，列=预测类别，对角线为正确数 |

---

### 第五章：sklearn vs PyTorch 手动实现对比

| 对比维度 | sklearn（上半部分） | PyTorch 手动（下半部分） |
|:---|:---|:---|
| 抽象层级 | 高层 API，一行 `fit()` | 底层实现，手动前向/反向传播 |
| 优化算法 | LBFGS（拟牛顿法） | SGD（随机梯度下降） |
| 参数更新 | 自动批量求解 | 逐样本更新 |
| 分类类型 | 多分类（OvR） | 二分类 |
| 代码量 | ~5 行核心代码 | ~30 行核心代码 |
| 学习价值 | 快速应用 | 理解底层原理 |

---

### 第六章：逻辑回归算法特点

#### 6.1 优点

| 优点 | 说明 |
|:---|:---|
| **可解释性强** | 权重系数直接反映特征对分类的影响方向和大小 |
| **输出概率** | 天然输出概率值，便于设定不同阈值 |
| **训练速度快** | 相比复杂模型，计算开销小 |
| **正则化支持** | L1/L2 正则化防止过拟合 |
| **小样本有效** | 在样本量较少时也表现稳定 |

#### 6.2 缺点

| 缺点 | 说明 |
|:---|:---|
| **只能处理线性可分** | 无法直接处理非线性决策边界 |
| **对异常值敏感** | 极端值可能影响决策边界 |
| **特征独立性假设** | 多重共线性影响系数稳定性 |
| **多分类需额外策略** | OvR 或 Multinomial 增加模型复杂度 |

#### 6.3 适用场景

- **二分类问题**：垃圾邮件检测、疾病诊断、信用评估
- **需要概率输出的场景**：风险评分、转化率预测
- **基线模型**：作为复杂模型的对比基准
- **特征重要性分析**：通过系数大小判断哪些特征最重要

---

## 环境要求

| 依赖包 | 用途 |
|:---|:---|
| Python 3.7+ | 编程语言 |
| numpy | 数值计算 |
| pandas | 数据处理 |
| matplotlib | 数据可视化 |
| scikit-learn | 逻辑回归模型、数据集、评估 |
| PyTorch | 手动实现逻辑回归（下半部分） |

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
2. 在浏览器中打开 `08th_LogisticRegression.ipynb`
3. 按顺序运行每个代码单元格（`Shift + Enter`）

---

## 学习建议

1. **先看懂 sklearn 部分**：快速建立对逻辑回归的直观认识
2. **再深入 PyTorch 部分**：理解 forward、loss、backward、update 的完整流程
3. **对比两种实现**：思考"调用 `.fit()` "背后到底发生了什么
4. **修改学习率和轮数**：在 PyTorch 部分改 `learning_rate` 和 `epochs`，观察收敛速度
5. **尝试其他多分类策略**：设置 `multi_class='multinomial'` 对比效果
6. **与之前算法对比**：在同一数据集（如 iris）上对比逻辑回归与 KNN、决策树的表现

---

## 常见问题

### Q1: 逻辑回归是"回归"还是"分类"？

**分类算法**。"回归"一词是历史遗留——它在线性回归的输出上套了一个 Sigmoid 函数，将实数映射为概率，本质上做的是分类。可以理解为"用回归的思路做分类"。

### Q2: Sigmoid 函数的作用是什么？

- 将 $(-\infty, +\infty)$ 的线性输出**压缩**到 $(0, 1)$
- 输出可解释为**概率**
- 在 $z=0$ 附近接近线性，在两端趋于饱和

图形特征：S 形曲线，中心点为 $(0, 0.5)$。

### Q3: 为什么分类不用 MSE 而用交叉熵？

| | MSE | 交叉熵 |
|:---|:---|:---|
| 梯度形状 | 当 $\hat{y}$ 接近 0 或 1 时梯度趋于 0（梯度消失） | 梯度始终较大，收敛快 |
| 概率解释 | 无 | 来自最大似然估计 |
| 凸性 | 非凸（有局部最优） | 凸函数（可找到全局最优） |

**结论**：交叉熵 + Sigmoid 是"天作之合"，数学上梯度简洁、优化路径凸性良好。

### Q4: OvR 和 Multinomial 多分类有什么区别？

| 策略 | sklearn 参数 | 训练方式 | 概率输出 |
|:---|:---|:---|:---|
| **OvR** | `multi_class='ovr'` | K 个独立二分类器 | 各自独立（不一定和为 1） |
| **Multinomial** | `multi_class='multinomial'` | 一个统一的 Softmax 模型 | Softmax 归一化，和为 1 |

> 一般推荐 `multinomial`，因为它直接优化多分类目标，概率更准确。但 OvR 每个分类器独立，可并行训练。

### Q5: 手动实现中为什么需要 `w.grad.zero_()`？

PyTorch 的梯度默认**累加**（不自动清零），每次 `.backward()` 会将新梯度加到 `.grad` 上。如果不清零，梯度会越来越大，导致参数更新错误。因此每次更新后必须手动清零。

---

**最后更新**：2026-06-18
