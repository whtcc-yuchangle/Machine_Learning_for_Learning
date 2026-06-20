# 数据预处理（Data Preprocessing）教程

## 概述

本 notebook 系统地介绍机器学习中数据预处理的七大核心环节，使用 sklearn 的 wine、iris、breast_cancer、digits 等经典数据集进行实战演示。数据预处理是机器学习流程中**最耗时但也最关键**的步骤——**"垃圾进，垃圾出"（Garbage In, Garbage Out）**，高质量的数据预处理直接决定模型效果的上限。

---

## 学习目标

通过学习本教程，您将掌握：

1. **缺失值检测与填充**：使用 `isnull()`、`missingno` 可视化缺失值，`SimpleImputer` 完成填充
2. **标准化与归一化**：理解 `StandardScaler`（Z-Score）与 `MinMaxScaler` 的原理、区别及适用场景
3. **特征编码**：掌握独热编码（One-Hot）、标签编码（Label Encoding）、序列编码（Ordinal Encoding）三种方式
4. **数据分箱**：等宽分箱（`pd.cut`）、等频分箱（`pd.qcut`）、基于决策树的分箱
5. **特征选择**：方差过滤、卡方检验、皮尔逊相关系数三种过滤式特征选择方法
6. **PCA 降维**：主成分分析原理、`n_components` 选择策略、PCA 降噪实战
7. **全流程串联**：理解这些预处理步骤在实际 ML 项目中的组合顺序

---

## 内容结构

### 第一章：缺失值检测与处理

#### 1.1 缺失值检测

缺失值是现实数据中最常见的问题之一。pandas 提供了便捷的检测方法：

```python
# 检测每个特征是否存在缺失值
df.isnull().any()     # 返回布尔 Series，True 表示该列有缺失

# 统计每个特征的缺失值数量
df.isnull().sum()     # 返回每列缺失值计数

# 计算缺失比例
df.isnull().mean() * 100   # 返回每列缺失百分比
```

#### 1.2 缺失值可视化 — missingno

`missingno` 库提供了直观的缺失值可视化工具：

```python
import missingno as ms

# 柱状图展示每列缺失值数量
ms.bar(df)

# 矩阵图展示缺失值分布模式
ms.matrix(df)

# 热力图展示缺失值相关性
ms.heatmap(df)
```

| missingno 函数 | 用途 |
|:---|:---|
| `ms.bar(df)` | 柱状图，显示每列非缺失值和缺失值数量 |
| `ms.matrix(df)` | 矩阵图，白线表示缺失值，可观察缺失模式 |
| `ms.heatmap(df)` | 相关性热力图，展示列间缺失值共现关系 |
| `ms.dendrogram(df)` | 层次聚类树状图，按缺失模式对列分组 |

#### 1.3 缺失值填充 — SimpleImputer

sklearn 的 `SimpleImputer` 提供四种填充策略：

```python
from sklearn.impute import SimpleImputer

# 最频值填充
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data_filled = imp.fit_transform(data)
```

| 参数 | 取值 | 说明 |
|:---|:---|:---|
| `missing_values` | `np.nan`（默认） | 缺失值标记符号 |
| `strategy` | `'mean'` | 用该列的**均值**填充（仅适用于数值型） |
| | `'median'` | 用该列的**中位数**填充（对异常值鲁棒） |
| | `'most_frequent'` | 用该列的**众数**填充（适用于数值和分类型） |
| | `'constant'` | 用 `fill_value` 指定的常数填充 |

```python
# 均值填充
SimpleImputer(strategy='mean').fit_transform(data)

# 中位数填充
SimpleImputer(strategy='median').fit_transform(data)

# 常数填充
SimpleImputer(strategy='constant', fill_value=0).fit_transform(data)
```

| 策略 | 适用场景 | 注意事项 |
|:---|:---|:---|
| `mean` | 数据近似正态分布，无极端异常值 | 异常值会严重偏斜均值 |
| `median` | 数据有偏态或存在异常值 | 对偏态比均值更稳健 |
| `most_frequent` | 分类特征、多峰分布 | 可能引入偏差 |

---

### 第二章：标准化与归一化

#### 2.1 为什么需要标准化/归一化？

机器学习算法依赖距离或梯度计算时，不同特征的**量纲差异**会导致模型被大数值特征主导：

> **例子**：年龄 (0-100) vs 收入 (0-100000)，欧氏距离几乎完全由收入决定。

**必须标准化/归一化的算法**：KNN、SVM、逻辑回归、神经网络、K-Means、PCA

**不需要的算法**：决策树、随机森林、GBDT（基于树模型对特征缩放不敏感）

#### 2.2 标准化（Standardization / Z-Score）

将数据转换为**均值为 0、标准差为 1** 的分布：

$$x' = \frac{x - \mu}{\sigma}$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)
```

| 属性 | 说明 |
|:---|:---|
| 转换后分布 | 均值 ≈ 0，标准差 ≈ 1 |
| 对异常值敏感度 | 较敏感（均值和标准差受异常值影响） |
| 是否改变分布形状 | 否（线性变换，保持原始分布形态） |

```python
# 验证：标准化后均值≈0，标准差≈1
pd.DataFrame(data_standardized).describe()
# mean ≈ 0.0, std ≈ 1.0
```

#### 2.3 归一化（Normalization / Min-Max Scaling）

将数据线性映射到 **[0, 1]** 区间：

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)
```

| 属性 | 说明 |
|:---|:---|
| 转换后分布 | 所有值 ∈ [0, 1] |
| 对异常值敏感度 | **非常敏感**（min/max 由极端值决定） |
| 是否保持稀疏性 | 是（0 值保持不变） |

```python
# 验证：归一化后 min=0，max=1
pd.DataFrame(data_normalized).describe()
# min=0.0, max=1.0
```

#### 2.4 标准化 vs 归一化 对比

| 对比维度 | StandardScaler | MinMaxScaler |
|:---|:---|:---|
| **公式** | $(x-\mu)/\sigma$ | $(x-x_{min})/(x_{max}-x_{min})$ |
| **输出范围** | 无界（以 0 为中心） | [0, 1] |
| **异常值影响** | 中等 | **严重** |
| **适用算法** | PCA、SVM、逻辑回归 | 神经网络（需要 0-1 输入） |
| **保持稀疏性** | 否（0 值会被移动） | 是 |

> **选择建议**：
> - 默认优先用 `StandardScaler`（对异常值更鲁棒）
> - 数据需严格限定在 [0,1] 时用 `MinMaxScaler`（如神经网络 sigmoid/tanh 激活，或图像像素值）
> - 存在大量异常值时，考虑 `RobustScaler`（用中位数和四分位距替代均值和标准差）

---

### 第三章：特征编码

分类特征必须转换为数值才能输入 ML 模型。三种主流编码方式：

#### 3.1 独热编码（One-Hot Encoding）

为分类特征的每个取值创建一个新的二值列（0/1），**适用于无序分类变量**：

```python
# 方法一：pandas get_dummies
pd.get_dummies(df, columns=['education'])

# 方法二：sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['education']])
# 返回稀疏矩阵，.toarray() 可转为密集数组
```

| 对比 | `pd.get_dummies()` | `OneHotEncoder` |
|:---|:---|:---|
| 返回值 | DataFrame（列名自动生成） | 稀疏矩阵（内存高效） |
| 记忆训练集类别 | **否**（新数据可能缺类别） | **是**（`handle_unknown` 参数） |
| 适用场景 | 探索性数据分析 | **生产环境 ML pipeline** |

```python
# OneHotEncoder 关键参数
OneHotEncoder(
    sparse_output=True,       # 返回稀疏矩阵（默认，节省内存）
    handle_unknown='error',   # 测试集出现未知类别时：'error' 报错 / 'ignore' 全零编码
    drop=None                 # 是否丢弃一列避免共线性
)
```

| 方法 | 优点 | 缺点 |
|:---|:---|:---|
| One-Hot | 不引入虚假的顺序关系 | 类别多时**特征爆炸**（维数灾难） |
| One-Hot | 适合大多数 ML 算法 | 树模型对高维稀疏特征效率低 |

> **独热编码 vs 哑变量编码**：独热编码为 k 个类别创建 k 个列；哑变量（Dummy）创建 k-1 个列（丢弃一列做基准）。线性模型建议用 k-1 列以避免多重共线性。

#### 3.2 标签编码（Label Encoding）

将分类值转换为**整数标签**，**适用于有序分类变量（有天然顺序）或树模型**：

```python
from sklearn.preprocessing import LabelEncoder

df['country_label'] = LabelEncoder().fit_transform(df['country'])
```

| 编码结果 | country |
|:---|:---|
| 0 | China |
| 1 | Japan |
| 2 | Korea |
| 3 | UK |
| 4 | USA |

> **警告**：`LabelEncoder` 按字母顺序编码（China < Japan < Korea < UK < USA）。对于无序类别，这个 0 < 1 < 2 < 3 < 4 的顺序关系是**虚假的**，不应直接用于线性模型或距离类模型。

#### 3.3 序列编码（Ordinal Encoding）

**手动指定**类别间的顺序关系，适用于**有序分类变量**：

```python
# 方法一：pandas map 手动映射
df['education'] = df['education'].map({
    'Bachelor': 1,
    'Master': 2,
    'PHD': 3
})

# 方法二：sklearn OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[['Bachelor', 'Master', 'PHD']])
encoded = encoder.fit_transform(df[['education']])
```

#### 3.4 编码方式选择决策

```
                    分类特征
                       │
                ┌ 有序关系？ ─┐
                │ 是          │ 否
                ▼             ▼
          序列编码       ┌ 类别数量？ ─┐
        (Ordinal)       │ 少(k<10)  │ 多(k≥10)
                        ▼           ▼
                    独热编码    考虑目标编码
                  (One-Hot)    (Target Encoding)
                              或特征哈希
```

| 编码方式 | 适用场景 | 不适用场景 |
|:---|:---|:---|
| **One-Hot** | 无序类别、线性模型、距离模型 | 高基数类别（>100 类）、树模型 |
| **Label** | 树模型、神经网络 Embedding | 线性模型、距离模型（引入虚假顺序） |
| **Ordinal** | 有序类别（学历、评级） | 无序类别 |
| **Target** | 高基数类别（用目标变量均值编码） | 样本量小（易过拟合） |

---

### 第四章：数据分箱（Binning / Discretization）

将连续特征离散化为有限的区间（箱），能**降低异常值影响、捕捉非线性关系**。

#### 4.1 等宽分箱（Equal-Width Binning）

将数据范围均匀划分为等宽区间：

```python
# 将 'ash' 特征等宽分为 4 个箱
value, cutoff = pd.cut(data['ash'], bins=4, retbins=True)

# cutoff 返回各箱边界值
# value 返回每个样本所属的箱
```

| `pd.cut` 参数 | 说明 |
|:---|:---|
| `x` | 要分箱的数据 |
| `bins` | 整数（箱数）或列表（自定义边界） |
| `labels` | 箱的标签（默认用区间表示） |
| `retbins` | 是否返回箱的边界值 |
| `precision` | 边界值的显示精度（默认 3） |

> **特点**：实现简单，但对异常值敏感——极端值会撑宽区间，导致中间区间的样本很少。

#### 4.2 等频分箱（Equal-Frequency Binning）

使每个箱包含**大致相同数量**的样本：

```python
# 等频分为 4 个箱
value, cutoff = pd.qcut(data['ash'], q=4, retbins=True)
```

| `pd.qcut` 参数 | 说明 |
|:---|:---|
| `x` | 要分箱的数据 |
| `q` | 箱数（或分位数列表 [0, 0.25, 0.5, 0.75, 1.0]） |
| `duplicates` | 边界值重复时的处理：`'raise'` 报错 / `'drop'` 丢弃重复边界 |

```python
# 边界值重复时的处理
pd.qcut(x=s1, q=3, duplicates='drop', retbins=True)
# duplicates='drop' 丢弃重复边界值，减少箱数
```

| 对比维度 | 等宽分箱 (cut) | 等频分箱 (qcut) |
|:---|:---|:---|
| 划分依据 | 值域范围 | 样本数量 |
| 每箱宽度 | 相同 | 可能不同 |
| 每箱样本数 | 可能悬殊 | 大致相同 |
| 异常值影响 | **大**（拉伸区间） | **小**（异常值归入边缘箱） |
| 适用场景 | 数据均匀分布 | 数据分布不均、有偏态 |

#### 4.3 基于决策树的分箱

利用决策树寻找**最优分割点**进行分箱，分割点基于信息增益/基尼系数确定：

```python
from sklearn.tree import DecisionTreeClassifier

# 训练决策树（深度限制为预剪枝）
clf = DecisionTreeClassifier(
    criterion='entropy',   # 用信息增益（ID3）
    max_depth=3            # 预剪枝：限制最大深度
).fit(X, y)

# 获取所有叶子节点的分割阈值
cut_points = clf.tree_.threshold[
    np.where(clf.tree_.children_left > -1)
]
```

> **特点**：分箱边界**由目标变量驱动**（有监督分箱），分割点比等宽/等频更具业务含义。

| 分箱方法 | 监督/无监督 | 优点 | 缺点 |
|:---|:---|:---|:---|
| 等宽分箱 | 无监督 | 简单直观 | 对异常值敏感 |
| 等频分箱 | 无监督 | 每箱样本均衡 | 可能把相近值分到不同箱 |
| 决策树分箱 | **有监督** | 分割点有预测意义 | 可能过拟合 |

---

### 第五章：特征选择（Feature Selection）

#### 5.1 方差过滤（VarianceThreshold）

删除**方差低于阈值**的特征——方差极低的特征携带的信息量很少：

```python
from sklearn.feature_selection import VarianceThreshold

# 删除方差 < 0.1 的特征
selector = VarianceThreshold(threshold=0.1)
X_filtered = selector.fit_transform(X)
```

| 参数 | 说明 |
|:---|:---|
| `threshold` | 方差阈值，低于此值的特征被删除（默认 0，删除所有值相同的特征） |

> **适用场景**：作为特征选择的**第一步**，快速去除常量特征或近常量特征（如 95% 样本取值相同的特征）。

#### 5.2 卡方检验过滤（Chi-Square Test）

衡量**分类特征与目标变量**之间的统计相关性：

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 根据卡方分数选择 top 2 特征
selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)

# 查看各特征的卡方分数和 p 值
selector.scores_     # 卡方统计量（越大越相关）
selector.pvalues_    # p 值（越小越显著）
```

| 重要限制 | 说明 |
|:---|:---|
| 输入必须非负 | `chi2` 要求特征值 ≥ 0 |
| 适用于分类特征 | 连续特征需先离散化 |
| 衡量线性相关性 | 无法捕捉非线性依赖 |

#### 5.3 皮尔逊相关系数过滤

衡量特征与目标变量之间的**线性相关性**：

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

$$r \in [-1, 1]：1 = 完全正相关，-1 = 完全负相关，0 = 无线性相关$$

```python
from scipy.stats import pearsonr

# 计算每个特征与目标的皮尔逊相关系数及 p 值
r, p_value = pearsonr(feature_column, target_column)

# 通过 SelectKBest 批量选择
from sklearn.feature_selection import SelectKBest

def pearson_selector(X, Y):
    # 返回 (相关系数数组, p值数组) 的二元组
    return tuple(map(
        tuple,
        np.array(list(map(
            lambda x: pearsonr(x, Y), X.T
        ))).T
    ))

selector = SelectKBest(pearson_selector, k=3)
X_new = selector.fit_transform(X, y)
```

| 皮尔逊相关系数 r | 相关程度 |
|:---|:---|
| 0.8 ~ 1.0 | 强正相关 |
| 0.5 ~ 0.8 | 中等正相关 |
| 0.3 ~ 0.5 | 弱正相关 |
| 0 ~ 0.3 | 极弱/无相关 |
| < 0 | 负相关（绝对值同上） |

#### 5.4 三种过滤方法对比

| 方法 | 适用特征类型 | 适用目标类型 | 衡量什么 | 局限 |
|:---|:---|:---|:---|:---|
| **方差过滤** | 任意 | 不需要目标变量 | 特征自身的变化程度 | 高方差≠高预测力 |
| **卡方检验** | 非负、分类/离散 | 分类 | 特征与目标的统计独立性 | 不能处理连续特征 |
| **皮尔逊相关** | 连续数值 | 连续数值 | 线性相关性 | 无法捕捉非线性关系 |

---

### 第六章：PCA 降维（Principal Component Analysis）

#### 6.1 PCA 原理概述

主成分分析（PCA）是最经典的**线性降维**方法，核心思想是**将原始特征空间投影到方差最大的正交方向上**：

$$
\begin{aligned}
&\text{第一主成分：} & \max_w \; \text{Var}(Xw), \quad \|w\| = 1 \\
&\text{第二主成分：} & \max_w \; \text{Var}(Xw), \quad \|w\| = 1, \; w \perp w_1 \\
&\text{依此类推...}
\end{aligned}
$$

数学上，PCA 等价于对协方差矩阵 $X^T X$ 做特征值分解（EVD）或对 $X$ 做奇异值分解（SVD）：

$$X = U \Sigma V^T$$

其中 $V$ 的列就是主成分方向，$\Sigma^2 / (n-1)$ 的对角元素是各主成分的方差。

#### 6.2 PCA 基本使用

```python
from sklearn.decomposition import PCA

# 降到 2 维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

```python
# 鸢尾花 4D → 2D 降维可视化
pca = PCA(n_components=2)
x_dr = pca.fit_transform(iris.data)

# 绘制三个类别的降维散点图
plt.scatter(x_dr[y==0, 0], x_dr[y==0, 1], c='red', label='setosa')
plt.scatter(x_dr[y==1, 0], x_dr[y==1, 1], c='black', label='versicolor')
plt.scatter(x_dr[y==2, 0], x_dr[y==2, 1], c='blue', label='virginica')
```

#### 6.3 n_components 选择策略

| 策略 | 设置方式 | 说明 |
|:---|:---|:---|
| **指定维数** | `n_components=2` | 直接指定降到几维 |
| **MLE 自动选择** | `n_components='mle'` | 用极大似然估计自动确定维数 |
| **保留信息比例** | `n_components=0.97` | 保留 97% 的信息量 |
| **累计方差比选** | 绘制累计方差曲线 | 手动观察拐点选择 |

```python
# 策略一：MLE 自动选择
pca_mle = PCA(n_components='mle')
pca_mle.fit(X)
print(pca_mle.n_components_)  # 自动确定的维数

# 策略二：保留 97% 信息量
pca = PCA(n_components=0.97, svd_solver='full')
X_reduced = pca.fit_transform(X)

# 策略三：绘制累计可解释方差曲线，手动选拐点
pca_full = PCA().fit(X)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
# 在"拐点"处选择 n_components
```

| 参数 | 说明 |
|:---|:---|
| `explained_variance_ratio_` | 各主成分的方差贡献率（总和 = 1） |
| `components_` | 主成分方向（特征向量矩阵），shape = (n_components, n_features) |
| `explained_variance_` | 各主成分的方差值 |
| `n_components_` | 实际保留的维数 |

#### 6.4 PCA 降噪实战 — 手写数字去噪

PCA 的逆变换可将降维后的数据**重建回原始空间**，此过程天然滤除方差小的噪声成分：

```python
# 1. 加载手写数字数据（8×8=64维）
from sklearn.datasets import load_digits
digits = load_digits()

# 2. 添加噪声
noisy = np.random.RandomState(50).normal(digits.data, 2)

# 3. PCA 降维（保留 70% 信息，64D → ~13D）
pca = PCA(0.7, svd_solver='full').fit(noisy)
x_reduced = pca.transform(noisy)

# 4. 逆变换重建（13D → 64D）
restored = pca.inverse_transform(x_reduced)
```

| PCA 降噪原理 | 说明 |
|:---|:---|
| 信号 | 集中在**前几个高方差主成分** |
| 噪声 | 分散在**所有方向**，尤其是低方差方向 |
| 降噪机制 | 丢弃低方差主成分 → 滤除噪声分量 |

---

### 第七章：PCA 核心属性与数学

#### 7.1 主成分矩阵 `components_`

```python
pca = PCA(2).fit(X)
pca.components_
# shape: (2, 4)，每行是一个主成分方向
# 每列对应原始特征在该主成分上的权重
```

主成分是原始特征的**线性组合**：

$$PC_1 = w_{11}x_1 + w_{12}x_2 + ... + w_{1d}x_d$$

其中 $(w_{11}, w_{12}, ..., w_{1d})$ 就是 `components_[0, :]`。

#### 7.2 PCA 与 t-SNE 的区别

| 对比维度 | PCA | t-SNE |
|:---|:---|:---|
| 类型 | 线性降维 | 非线性降维 |
| 目标 | 保留全局方差结构 | 保留局部近邻关系 |
| 全局距离 | **保持** | **不保持**（只保近邻） |
| 可解释性 | 强（主成分是原始特征的线性组合） | 弱（无显式映射函数） |
| 可逆性 | 可逆（`inverse_transform`） | 不可逆 |
| 用途 | 降维、去噪、特征提取 | **纯可视化** |
| 计算速度 | 快 | 慢 |

---

### 第八章：全流程速查 — 预处理步骤推荐顺序

典型 ML 项目的数据预处理流水线：

```
原始数据
   │
   ▼
① 缺失值处理（SimpleImputer）
   │
   ▼
② 异常值处理（IQR、截尾、分箱化）
   │
   ▼
③ 特征编码（OneHot / Ordinal / Label）
   │
   ▼
④ 特征选择 / 降维（VarianceThreshold → SelectKBest → PCA）
   │
   ▼
⑤ 标准化 / 归一化（StandardScaler / MinMaxScaler）
   │
   ▼
⑥ 数据划分（train_test_split）
   │
   ▼
⑦ 建模
```

> **关键原则**：
> - 标准化/归一化**必须放在 train_test_split 之后**（在训练集上 fit，在测试集上 transform，防止数据泄露）
> - 特征选择和降维也**必须在训练集上 fit**，避免使用测试集的信息
> - 使用 `sklearn.pipeline.Pipeline` 将这些步骤串联，防止数据泄露并简化代码

---

## 环境要求

| 依赖包 | 用途 |
|:---|:---|
| Python 3.7+ | 编程语言 |
| numpy | 数值计算 |
| pandas | 数据处理与分箱 |
| matplotlib | 可视化 |
| seaborn | 统计可视化（分布图） |
| scikit-learn | StandardScaler、MinMaxScaler、编码器、分箱、特征选择、PCA |
| scipy | 皮尔逊相关系数计算 |
| missingno | 缺失值可视化 |

安装命令：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy missingno
```

---

## 使用方法

1. 启动 Jupyter Notebook：
   ```bash
   jupyter notebook
   ```
2. 在浏览器中打开 `11th_data_preprocess.ipynb`
3. 按顺序运行每个代码单元格（`Shift + Enter`）
4. 观察标准化/归一化前后数据的统计描述变化
5. 对比不同编码方式的结果
6. 运行 PCA 降维可视化，观察 Iris 数据的分类分布

---

## 学习建议

1. **理解"为什么"**：先搞清楚每种预处理方法的数学原理和适用场景，避免盲目套用
2. **动手实验**：对同一份数据分别用 `StandardScaler` 和 `MinMaxScaler`，观察对模型精度的影响
3. **掌握 Pipeline**：学习用 `sklearn.pipeline.Pipeline` 串联预处理步骤，防止数据泄露
4. **PCA 曲线**：亲手绘制累计方差贡献率曲线，体会"用 2 维保留 95% 信息"的含义
5. **编码选择**：对同一份分类数据分别用 One-Hot 和 Label 编码训练同一个模型，观察精度差异
6. **分箱调参**：修改 `pd.cut` 的 `bins` 和 `pd.qcut` 的 `q` 值，观察不同分箱粒度对特征分布的影响
7. **特征选择组合**：先方差过滤，再相关系数过滤，观察两次过滤后保留的特征是否一致

---

## 常见问题

### Q1: 什么时候用标准化，什么时候用归一化？

| 场景 | 推荐 |
|:---|:---|
| 数据近似正态分布 | `StandardScaler` |
| 数据有异常值 | `RobustScaler`（基于中位数和 IQR） |
| 神经网络（sigmoid/tanh 输出层） | `MinMaxScaler`（[0,1] 范围匹配激活函数） |
| 需要保持稀疏性 | `MinMaxScaler` |
| 不确定 | **`StandardScaler`（通用默认选择）** |

### Q2: 训练集和测试集的标准化怎么处理？

```python
# ✅ 正确做法
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # 在训练集上 fit
X_test_scaled = scaler.transform(X_test)          # 只用 transform！

# ❌ 错误做法 — 数据泄露
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)  # 信息从测试集泄露到训练集
```

### Q3: One-Hot 编码后特征爆炸怎么办？

- **特征哈希**（Feature Hashing / HashingVectorizer）：用哈希函数将高基数类别映射到固定维度
- **目标编码**（Target Encoding）：用目标变量的均值替代类别标签
- **嵌入**（Embedding）：用神经网络学习低维稠密向量表示（深度学习常用）
- **降维**：One-Hot 后接 PCA 或 TruncatedSVD

### Q4: PCA 的 n_components 到底该怎么选？

三条经验法则：

| 策略 | 方法 | 适用 |
|:---|:---|:---|
| 保留 95% 方差 | `n_components=0.95` | 通用，数据压缩 |
| 降维可视化 | `n_components=2` 或 `3` | 数据探索 |
| MLE 自动 | `n_components='mle'` | 不确定时自动确定 |

### Q5: PCA 的特征需要先标准化吗？

**必须！** PCA 基于方差最大化，如果特征量纲不一致，方差大的特征会主导主成分方向。

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipeline = make_pipeline(StandardScaler(), PCA(n_components=0.95))
X_reduced = pipeline.fit_transform(X)
```

### Q6: 卡方检验为什么要求特征值非负？

卡方统计量的计算基于**观测频数与期望频数的比较**，频数天然非负。若特征值有负数，可考虑：
- 用 `MinMaxScaler` 将特征缩放到 [0,1]
- 改用其他不要求非负的特征选择方法（如 `f_classif`、`mutual_info_classif`）

### Q7: 如何处理新测试集中的未知类别？

```python
# OneHotEncoder 设置 handle_unknown='ignore'
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X_train[['category']])
encoder.transform(X_test[['category']])  # 未知类别 → 全零行
```

---

**最后更新**：2026-06-20
