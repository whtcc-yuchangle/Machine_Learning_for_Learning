# KNN（K近邻算法）教程

## 概述

本 notebook 详细介绍了 KNN（K-Nearest Neighbors，K近邻算法）的原理、实现和应用。KNN 是一种简单而强大的监督学习算法，广泛应用于分类和回归任务。本教程通过手动实现和 sklearn 库两种方式，帮助学习者深入理解 KNN 算法的核心思想。

---

## 学习目标

通过学习本教程，您将掌握：

1. **KNN 算法原理**：理解距离度量、K 值选择、投票机制
2. **距离计算**：欧氏距离的原理和实现
3. **手动实现 KNN**：从零开始编写 KNN 分类器
4. **sklearn KNN**：使用 scikit-learn 的 KNeighborsClassifier
5. **模型评估**：准确率计算和模型性能评估
6. **实战应用**：红酒分类和乳腺癌检测案例

---

## 内容结构

### 第一章：KNN 算法原理

#### 1.1 算法思想

KNN 是一种**基于实例的学习**（Instance-based Learning）算法，其核心思想非常简单：

> **"近朱者赤，近墨者黑"**

- 对于一个新样本，找到它在训练数据中的 K 个最近邻居
- 根据这 K 个邻居的类别，通过投票机制决定新样本的类别

#### 1.2 算法流程

```
1. 输入：训练数据集、新样本、K 值
2. 计算新样本与所有训练样本的距离
3. 选择距离最近的 K 个邻居
4. 根据 K 个邻居的类别进行投票
5. 返回投票结果作为预测类别
```

#### 1.3 距离度量

**欧氏距离（Euclidean Distance）**：最常用的距离度量方法

$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

**曼哈顿距离（Manhattan Distance）**：

$$d(x, y) = \sum_{i=1}^{n}|x_i - y_i|$$

**闵可夫斯基距离（Minkowski Distance）**：

$$d(x, y) = \left(\sum_{i=1}^{n}|x_i - y_i|^p\right)^{1/p}$$

其中 $p=2$ 时为欧氏距离，$p=1$ 时为曼哈顿距离。

#### 1.4 K 值选择

| K 值 | 效果 |
|:---|:---|
| **K 过小** | 模型过于复杂，容易过拟合 |
| **K 过大** | 模型过于简单，容易欠拟合 |
| **K 为奇数** | 避免投票平局 |

---

### 第二章：红酒数据集探索

#### 2.1 数据集介绍

本教程使用一个简单的红酒数据集，包含以下特征：

| 特征 | 说明 |
|:---|:---|
| 颜色深度 | 红酒的颜色深浅程度 |
| 酒精浓度 | 红酒的酒精含量 |
| 品种 | 红酒类别（0 或 1） |

#### 2.2 数据准备

```python
import numpy as np
import pandas as pd

# 数据准备
data = [
    [14.13, 5.64, 0],
    [13.20, 4.28, 0],
    [13.16, 5.68, 0],
    [14.27, 4.80, 0],
    [13.24, 4.22, 0],
    [12.07, 2.76, 1],
    [12.43, 3.94, 1],
    # ... 更多数据
]

# 构建为 dataframe 对象
df = pd.DataFrame(data, columns=['颜色深度', '酒精浓度', '品种'])
```

#### 2.3 数据可视化

```python
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制散点图
plt.figure(figsize=(10, 6))
colors = ['red', 'blue']
labels = ['品种0', '品种1']

for i in range(2):
    subset = df[df['品种'] == i]
    plt.scatter(subset['颜色深度'], subset['酒精浓度'], 
                c=colors[i], label=labels[i], s=100)

plt.xlabel('颜色深度')
plt.ylabel('酒精浓度')
plt.title('红酒数据散点图')
plt.legend()
plt.grid(True)
```

---

### 第三章：距离计算实现

#### 3.1 欧氏距离计算

```python
import math

def euclidean_distance(x1, x2):
    """计算两个向量之间的欧氏距离"""
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return math.sqrt(distance)

# 测试
point1 = [12.03, 4.1]
point2 = [14.13, 5.64]
distance = euclidean_distance(point1, point2)
```

#### 3.2 向量化距离计算（使用 NumPy）

```python
import numpy as np

def euclidean_distance_np(x1, x2):
    """使用 NumPy 计算欧氏距离"""
    return np.sqrt(np.sum((np.array(x1) - np.array(x2)) ** 2))
```

---

### 第四章：手动实现 KNN

#### 4.1 KNN 分类器实现

```python
import math
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        """训练模型"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def _distance(self, x1, x2):
        """计算欧氏距离"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _vote(self, neighbors):
        """投票机制"""
        return Counter(neighbors).most_common(1)[0][0]
    
    def predict(self, X):
        """预测新样本"""
        X = np.array(X)
        predictions = []
        
        for x in X:
            # 计算距离
            distances = [self._distance(x, x_train) for x_train in self.X_train]
            
            # 获取 K 个最近邻居的索引
            nearest_indices = np.argsort(distances)[:self.k]
            
            # 获取 K 个邻居的标签
            nearest_labels = self.y_train[nearest_indices]
            
            # 投票
            predictions.append(self._vote(nearest_labels))
        
        return predictions
```

#### 4.2 使用自定义 KNN

```python
# 创建分类器
knn = KNNClassifier(k=3)

# 准备数据
X = df[['颜色深度', '酒精浓度']].values
y = df['品种'].values

# 训练模型
knn.fit(X, y)

# 预测
new_data = [[12.03, 4.1]]
prediction = knn.predict(new_data)
```

---

### 第五章：使用 sklearn 的 KNN

#### 5.1 导入模块

```python
from sklearn.neighbors import KNeighborsClassifier
```

#### 5.2 数据准备

```python
import pandas as pd

# 数据准备
data = [
    [14.13, 5.64, 0],
    [13.20, 4.28, 0],
    # ... 更多数据
]

df = pd.DataFrame(data, columns=['颜色深度', '酒精浓度', '品种'])

# 分离特征和标签
X = df[['颜色深度', '酒精浓度']]
y = df['品种']
```

#### 5.3 创建并训练模型

```python
# 创建 KNN 分类器
clf = KNeighborsClassifier(n_neighbors=3)

# 训练模型
clf.fit(X, y)
```

#### 5.4 预测

```python
# 预测单个样本
new_sample = [[12.03, 4.1]]
prediction = clf.predict(new_sample)

# 预测多个样本
new_samples = [[12.03, 4.1], [13.5, 5.0]]
predictions = clf.predict(new_samples)
```

#### 5.5 模型评估

```python
# 计算准确率
accuracy = clf.score(X, y)
print(f"准确率: {accuracy:.2f}")
```

---

### 第六章：乳腺癌检测实战

#### 6.1 导入数据集

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 导入数据集
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
```

#### 6.2 数据集划分

```python
# 划分训练集和测试集（30% 作为测试集）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

#### 6.3 构建模型

```python
# 创建 KNN 分类器，指定 K 值为 4
clf = KNeighborsClassifier(n_neighbors=4)

# 训练模型
clf.fit(X_train, y_train)
```

#### 6.4 模型评估

```python
# 在训练集上评估
train_accuracy = clf.score(X_train, y_train)
print(f"训练集准确率: {train_accuracy:.4f}")

# 在测试集上评估
test_accuracy = clf.score(X_test, y_test)
print(f"测试集准确率: {test_accuracy:.4f}")
```

---

### 第七章：KNN 进阶功能

#### 7.1 查找最近邻居

```python
# 查找测试集中指定样本的 K 个邻居
# 返回距离和邻居索引
distances, indices = clf.kneighbors(X_test.iloc[[20, 30], :], return_distance=True)

# distances: 距离数组
# indices: 邻居在训练集中的索引
```

**注意事项**：当 `X_test` 是 Pandas DataFrame 时，必须使用 `.iloc` 进行位置索引：

```python
# 正确写法
clf.kneighbors(X_test.iloc[[20, 30], :], return_distance=True)

# 错误写法（会导致 InvalidIndexError）
# clf.kneighbors(X_test[[20, 30], :], return_distance=True)
```

#### 7.2 权重设置

```python
# 默认：uniform（均匀权重）
clf = KNeighborsClassifier(n_neighbors=3, weights='uniform')

# distance（距离加权）：距离越近权重越大
clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
```

#### 7.3 距离度量设置

```python
# 欧氏距离（默认）
clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

# 曼哈顿距离
clf = KNeighborsClassifier(n_neighbors=3, metric='manhattan')

# 闵可夫斯基距离
clf = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=3)
```

---

### 第八章：KNN 算法特点

#### 8.1 优点

| 优点 | 说明 |
|:---|:---|
| **简单易懂** | 算法原理直观，易于理解和实现 |
| **无需训练** | 不需要训练过程，直接存储数据 |
| **多分类支持** | 天然支持多分类任务 |
| **适用于非线性数据** | 能够处理非线性关系 |

#### 8.2 缺点

| 缺点 | 说明 |
|:---|:---|
| **计算量大** | 预测时需要计算与所有训练样本的距离 |
| **内存开销大** | 需要存储所有训练数据 |
| **对 K 值敏感** | K 值选择不当会影响模型性能 |
| **对特征缩放敏感** | 需要进行特征标准化 |
| **对噪声敏感** | 容易受离群点影响 |

#### 8.3 适用场景

- 数据量较小的分类任务
- 多分类问题
- 需要解释性强的模型
- 特征维度较低的场景

---

### 第九章：K 值选择策略

#### 9.1 交叉验证选择 K

```python
from sklearn.model_selection import cross_val_score

# 尝试不同的 K 值
k_values = range(1, 21)
scores = []

for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(clf, X, y, cv=5).mean()
    scores.append(score)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.plot(k_values, scores, marker='o')
plt.xlabel('K 值')
plt.ylabel('交叉验证准确率')
plt.title('K 值选择')
plt.grid(True)
```

#### 9.2 经验法则

- 通常尝试 K = $\sqrt{n}$，其中 n 是训练样本数量
- K 值通常选择奇数，避免投票平局
- 通过交叉验证选择最佳 K 值

---

## 环境要求

| 依赖包 | 版本要求 | 用途 |
|:---|:---|:---|
| Python | 3.7+ | 编程语言 |
| numpy | - | 数值计算 |
| pandas | - | 数据处理 |
| matplotlib | - | 数据可视化 |
| scikit-learn | 0.22+ | 机器学习库 |

---

## 安装依赖

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 使用方法

1. **启动 Jupyter Notebook**：
   ```bash
   jupyter notebook
   ```

2. **打开文件**：在浏览器中打开 `04th_KNN.ipynb`

3. **运行代码**：按顺序运行每个代码单元格（Shift+Enter）

---

## 学习建议

1. **理解原理**：先理解 KNN 的核心思想，再看代码实现
2. **动手实现**：尝试自己实现 KNN 算法，加深理解
3. **比较实现**：对比手动实现和 sklearn 实现的差异
4. **调参实践**：尝试不同的 K 值和距离度量方法
5. **特征缩放**：学习数据标准化对 KNN 的影响

---

## Notebook 单元格结构参考

| 单元格序号 | 内容主题 | 知识点 |
|:---|:---|:---|
| 1 | 环境设置 | 中文显示、交互模式 |
| 2 | 红酒数据准备 | DataFrame 构建 |
| 4 | 数据可视化 | 散点图绘制 |
| 5 | 距离计算 | 欧氏距离手动实现 |
| 7 | K 值投票 | Counter 投票机制 |
| 10 | KNN 类实现 | 完整算法封装 |
| 11 | sklearn KNN | 导入和环境设置 |
| 12 | sklearn 数据准备 | 特征和标签分离 |
| 14 | 模型训练 | fit 方法 |
| 15 | 模型预测 | predict 方法 |
| 16 | 模型评估 | score 方法 |
| 18 | 乳腺癌数据集 | load_breast_cancer |
| 19 | 数据集探索 | DataFrame 格式化 |
| 20 | 数据集划分 | train_test_split |
| 21-23 | 模型构建评估 | 完整流程 |
| 24 | kneighbors 方法 | 查找最近邻居 |

---

## 常见问题

### Q1: K 值如何选择？

```python
# 使用交叉验证选择最佳 K
from sklearn.model_selection import cross_val_score

best_k = None
best_score = 0

for k in range(1, 21):
    clf = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(clf, X, y, cv=5).mean()
    if score > best_score:
        best_score = score
        best_k = k

print(f"最佳 K 值: {best_k}, 准确率: {best_score:.4f}")
```

### Q2: 特征需要标准化吗？

**是的，非常重要！** KNN 对特征尺度敏感。

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Q3: kneighbors 方法报错 InvalidIndexError？

当 `X` 是 DataFrame 时，需要使用 `.iloc` 索引：

```python
# 正确写法
clf.kneighbors(X_test.iloc[[20, 30], :], return_distance=True)

# 错误写法
# clf.kneighbors(X_test[[20, 30], :], return_distance=True)  # 报错！
```

---

**最后更新**：2026-06-18

**文件版本**：v1.0
