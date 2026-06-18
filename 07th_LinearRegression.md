# 线性回归（Linear Regression）教程

## 概述

本 notebook 详细介绍了线性回归算法的原理、实现和应用。线性回归是机器学习中最基础、最经典的回归算法，通过拟合一条直线（或超平面）来描述特征与目标变量之间的线性关系。本教程使用 sklearn 的 `LinearRegression` 对人工生成的模拟数据进行回归实战，完整演示从数据生成到模型评估的全流程。

---

## 学习目标

通过学习本教程，您将掌握：

1. **线性回归原理**：理解最小二乘法的数学推导
2. **sklearn 线性回归**：使用 `LinearRegression` 拟合模型
3. **模型系数解读**：理解回归系数（`coef_`）和截距（`intercept_`）的含义
4. **模型评估指标**：均方误差（MSE）和决定系数（R²）
5. **实战案例**：从生成模拟数据到完整建模流程

---

## 内容结构

### 第一章：线性回归算法原理

#### 1.1 算法思想

线性回归假设目标变量 $y$ 与特征 $x$ 之间存在**线性关系**：

$$y = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

- **一元线性回归**（一个特征）：$y = wx + b$（一条直线）
- **多元线性回归**（多个特征）：$y = \mathbf{w}^T\mathbf{x} + b$（一个超平面）

本教程先演示**一元线性回归**，真实关系设定为：

$$y = 2x - 5 + \varepsilon$$

其中 $\varepsilon \sim \mathcal{N}(0, 1)$ 是高斯噪声。

#### 1.2 最小二乘法（Ordinary Least Squares, OLS）

线性回归的训练目标是找到一组参数 $(\mathbf{w}, b)$，使预测值与真实值之间的**均方误差**最小：

$$\min_{\mathbf{w}, b} \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \min_{\mathbf{w}, b} \frac{1}{n}\sum_{i=1}^{n}(y_i - \mathbf{w}^T\mathbf{x}_i - b)^2$$

**解析解**（闭式解）：

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

> sklearn 的 `LinearRegression` 内部使用 SVD（奇异值分解）求解，数值稳定性更好。

#### 1.3 截距项的处理

线性回归模型包含两部分：

| 组成部分 | sklearn 属性 | 含义 |
|:---|:---|:---|
| **回归系数** | `coef_` | 每个特征对目标的贡献权重 |
| **截距** | `intercept_` | 当所有特征为 0 时的预测基准值 |

sklearn 的 `LinearRegression` 默认 `fit_intercept=True`，会自动学习截距项。本 notebook 中手动添加了一列全 1 的列来演示截距的处理方式（教学目的）。

---

### 第二章：模型评估指标

#### 2.1 均方误差（Mean Squared Error, MSE）

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- MSE 越小越好，最小为 0（完美预测）
- 单位是目标变量单位的平方，对异常值敏感（平方放大误差）

#### 2.2 决定系数（R² Score）

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

| R² 值 | 含义 |
|:---|:---|
| $R^2 = 1$ | 完美预测 |
| $R^2 = 0$ | 模型等于用均值预测（最差基准） |
| $R^2 < 0$ | 模型比用均值预测还差 |
| $R^2 \approx 0.91$ | 模型解释了 91% 的目标变量方差 |

> R² 衡量的是模型解释了目标变量方差的比例，越接近 1 越好。

---

### 第三章：Notebook 实战 — 模拟数据线性回归

本章对应 notebook 中的实际代码单元格，逐步演示线性回归的完整建模流程。

#### 3.1 数据集说明

本教程使用**人工生成的模拟数据**，而非真实数据集。这样做的好处是：
- 已知真实的参数值（斜率=2，截距=-5），可以验证模型是否学到了正确的规律
- 数据和噪声可控，便于理解模型行为

| 属性 | 说明 |
|:---|:---|
| 样本数量 | 100 个 |
| 特征数量 | 1 个（一维 x） |
| 真实关系 | $y = 2x - 5 + \varepsilon$，$\varepsilon \sim \mathcal{N}(0, 1)$ |
| 数据范围 | $x \in [0, 5)$ |

#### 3.2 Notebook 单元格详解

**单元格 1：导入模块**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LinearRegression
from IPython.core.interactiveshell import InteractiveShell

# 设置全行输出
InteractiveShell.ast_node_interactivity = "all"

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False
```

> **说明**：`LinearRegression` 是 sklearn 中普通最小二乘线性回归的实现，无需设置超参数。

**单元格 2：生成模拟数据**

```python
# 设置随机种子
rng = np.random.RandomState(1)

# 100 个 [0, 5) 的随机数
x = 5 * rng.rand(100)

# 真实规律的标签取值（含噪声）
y = 2 * x - 5 + rng.randn(100)

x, y
```

> **说明**：
> - `np.random.RandomState(1)` 创建一个独立的随机数生成器，设置种子确保每次运行结果一致
> - `rng.rand(100)` 生成 100 个 [0, 1) 均匀分布随机数，乘以 5 后范围为 [0, 5)
> - 真实关系 `y = 2x - 5` 加上标准正态噪声 `randn(100)`
> - 训练完成后，模型的系数应该接近 `[2.0]`，截距接近 `-5.0`

**单元格 3：构造 DataFrame**

```python
# 构造为 DataFrame
X = pd.DataFrame(x)
Y = pd.DataFrame(y)

X, Y
```

> **说明**：将 numpy 数组转为 DataFrame，便于后续使用 `pd.concat` 拼接和列索引操作。

**单元格 4 & 5：手动添加截距列**

```python
# 添加一列全为 1 的列，表示截距
ex = pd.DataFrame(np.ones([100, 1]))
ex

# 轴 1 方向构造最终"数据集"
data = pd.concat([ex, X, Y], axis=1)
data
```

> **说明**：
> - 手动添加了一列全为 1 的列（截距列），构成设计矩阵
> - 最终 `data` 有 3 列：`[全1列, X特征, Y标签]`
> - **注意**：sklearn 的 `LinearRegression` 默认 `fit_intercept=True`，会自动添加截距项。这里手动添加全 1 列是为了教学演示，帮助理解"截距"在数学上的表达方式

**单元格 6：训练线性回归模型**

```python
# 创建线性回归器对象
reg = LinearRegression()

# 训练线性回归模型（前两列为特征，最后一列为标签）
reg.fit(data.iloc[:, :-1].values, data.iloc[:, -1].values)

reg.coef_       # array([0.        , 1.93698502])
reg.intercept_  # -4.763042745851091
```

> **说明**：
> - `data.iloc[:, :-1]` 取前两列（全 1 列 + X）作为特征，`data.iloc[:, -1]` 取最后一列作为标签
> - `coef_` 返回 `[0., 1.93698502]`：
>   - 第一个系数 ≈ 0 对应手动添加的全 1 列（因为 sklearn 已自动处理截距）
>   - 第二个系数 **1.937** 对应真实的 X 特征——接近真实斜率 **2.0**
> - `intercept_` 为 **-4.763**——接近真实截距 **-5.0**
>
> **结论**：模型成功从带噪声的数据中恢复了近似的真实线性关系：
>
> $$\hat{y} = 1.937x - 4.763 \quad \text{（真实：} y = 2x - 5 \text{）}$$

**单元格 7：模型评估**

```python
from sklearn.metrics import mean_squared_error, r2_score

# 获取预测值
yhat = reg.predict(data.iloc[:, :-1])

# 获取标签观察值
y = data.iloc[:, -1].values

# 计算 MSE
mean_squared_error(y, yhat)   # 0.7998

# 计算 R²
r2_score(y, yhat)             # 0.9104
```

> **说明**：
> - **MSE = 0.80**：预测值与真实值的均方误差约为 0.8。考虑到添加的噪声方差为 1（`randn`），这个误差是合理的
> - **R² = 0.91**：模型解释了目标变量 **91%** 的方差，拟合效果非常好
>   - 未解释的 9% 主要来自添加的随机噪声 $\varepsilon \sim \mathcal{N}(0, 1)$

---

### 第四章：回归结果可视化

建议在 notebook 中追加以下代码来可视化回归效果：

```python
# 绘制散点和回归直线
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, alpha=0.6, label='观测数据（含噪声）')

# 生成排序后的 x 用于画线（避免折线交叉）
x_line = np.linspace(0, 5, 100).reshape(-1, 1)
# 构造与训练时相同的特征矩阵（全 1 列 + X）
x_line_with_ones = np.column_stack([np.ones_like(x_line), x_line])
y_line = reg.predict(x_line_with_ones)

# 绘制真实关系（无噪声）
y_true = 2 * x_line.flatten() - 5

plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'拟合直线: y={reg.coef_[1]:.2f}x+({reg.intercept_:.2f})')
plt.plot(x_line, y_true, 'g--', linewidth=2, label='真实关系: y=2x-5')
plt.xlabel('x')
plt.ylabel('y')
plt.title('一元线性回归 — 拟合结果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### 第五章：线性回归算法特点

#### 5.1 优点

| 优点 | 说明 |
|:---|:---|
| **简单直观** | 模型结构和参数含义清晰，容易解释 |
| **训练速度快** | 有解析解，无需迭代优化 |
| **可解释性强** | 系数大小和符号直接反映特征的影响力 |
| **无超参数** | 普通最小二乘无需调参 |
| **小样本有效** | 在特征数远小于样本数时表现稳定 |
| **可做统计推断** | 可计算置信区间和 p 值（需 statsmodels） |

#### 5.2 缺点

| 缺点 | 说明 |
|:---|:---|
| **只能拟合线性关系** | 对非线性数据拟合效果差 |
| **对异常值敏感** | 最小二乘会被极端值拉偏（平方放大误差） |
| **特征独立性假设** | 多重共线性导致系数估计不稳定 |
| **不能处理缺失值** | 需要预处理填充或删除 |
| **外推能力差** | 在训练数据范围之外的预测不可靠 |

#### 5.3 适用场景

- **趋势分析**：如销售额随时间的变化
- **房价预测**：基于面积、地段等特征的房价估算
- **经济建模**：GDP 与各种经济指标的线性关系
- **基线模型**：作为复杂模型的性能对比基准
- **因果推断**：理解各因素对目标的影响方向和程度

---

### 第六章：线性回归的变体与扩展

| 方法 | sklearn 类 | 特点 |
|:---|:---|:---|
| **普通最小二乘** | `LinearRegression` | 标准线性回归（本教程使用） |
| **岭回归** | `Ridge` | L2 正则化，处理多重共线性 |
| **Lasso 回归** | `Lasso` | L1 正则化，自动做特征选择 |
| **弹性网络** | `ElasticNet` | L1 + L2 混合正则化 |
| **多项式回归** | `PolynomialFeatures` + `LinearRegression` | 拟合非线性曲线 |
| **SGD 回归** | `SGDRegressor` | 随机梯度下降，适合大数据集 |

---

## 环境要求

| 依赖包 | 用途 |
|:---|:---|
| Python 3.7+ | 编程语言 |
| numpy | 数值计算、随机数生成 |
| pandas | DataFrame 构造与操作 |
| matplotlib | 数据可视化 |
| scikit-learn | 线性回归模型和评估指标 |

安装命令：

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 使用方法

1. 启动 Jupyter Notebook：
   ```bash
   jupyter notebook
   ```
2. 在浏览器中打开 `07th_LinearRegression.ipynb`
3. 按顺序运行每个代码单元格（`Shift + Enter`）

---

## 学习建议

1. **修改真实参数**：将 `y = 2 * x - 5` 改为其他值，观察模型能否恢复正确的参数
2. **改变噪声水平**：增大或减小 `randn` 的系数，观察 MSE 和 R² 的变化
3. **对比含截距与无截距**：设置 `LinearRegression(fit_intercept=False)` 观察差异
4. **尝试多项式回归**：生成 $y = 3x^2 + 2x + 1 + noise$ 的非线性数据，观察线性回归的表现
5. **交叉验证**：使用 `cross_val_score` 对模型进行 K 折交叉验证

---

## 常见问题

### Q1: 为什么手动加了一列全 1，模型的 coef_[0] 还是 ≈ 0？

因为 sklearn 的 `LinearRegression` 默认 `fit_intercept=True`，会自动学习截距。手动加的全 1 列与自动截距**功能重复**，所以该特征的系数趋近于 0，实际的截距值存储在 `intercept_` 中。

如果想完全手动处理截距，应设置 `fit_intercept=False`：

```python
reg = LinearRegression(fit_intercept=False)
reg.fit(data.iloc[:, :-1].values, data.iloc[:, -1].values)
# 此时 coef_ 会包含截距：[b, w] 两个值都非零
```

### Q2: R² 为 0.91 算好吗？

| R² 范围 | 评价 |
|:---|:---|
| > 0.9 | 优秀 |
| 0.7 ~ 0.9 | 良好 |
| 0.5 ~ 0.7 | 一般 |
| < 0.5 | 较差 |

0.91 属于优秀水平。但因为这是模拟数据，噪声是人为加入的（$\sigma=1$），所以 R² 并不等于 1。真实场景中 R² > 0.8 通常就比较理想了。

### Q3: MSE 和 R² 有什么区别？

| 对比 | MSE | R² |
|:---|:---|:---|
| 含义 | 绝对误差 | 相对拟合优度 |
| 范围 | [0, +∞) | (-∞, 1] |
| 受量纲影响 | 是（单位平方） | 否（比例） |
| 跨数据集比较 | 不可以 | 可以 |

实际中通常**同时使用两者**：MSE 反映绝对误差大小，R² 反映模型解释能力。

### Q4: 线性回归和之前学的 KNN/决策树有什么不同？

| 对比维度 | 线性回归 | KNN | 决策树 |
|:---|:---|:---|:---|
| 任务类型 | 回归（也有分类变体） | 分类 + 回归 | 分类 + 回归 |
| 模型类型 | 参数模型 | 非参数模型 | 非参数模型 |
| 拟合能力 | 仅线性 | 任意形状 | 阶梯状 |
| 可解释性 | 高（系数直接解释） | 低 | 高（可视化） |
| 训练速度 | 快（解析解） | 无训练 | 快 |
| 外推能力 | 强（可外推） | 无外推能力 | 无外推能力 |

---

**最后更新**：2026-06-18
