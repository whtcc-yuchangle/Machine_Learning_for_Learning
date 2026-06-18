# 决策树（Decision Tree）教程

## 概述

本 notebook 详细介绍了决策树算法的原理、实现和应用。决策树是一种强大的监督学习算法，既可以用于分类任务，也可以用于回归任务。本教程通过 sklearn 库实现决策树分类器，并使用红酒数据集（Wine Dataset）进行实战演示。

---

## 学习目标

通过学习本教程，您将掌握：

1. **决策树算法原理**：理解决策树的构建过程、节点分裂策略
2. **特征选择准则**：理解基尼系数（Gini）和信息增益（Entropy）
3. **sklearn 决策树**：使用 `DecisionTreeClassifier` 构建模型
4. **模型评估**：使用 `score()` 方法计算准确率
5. **决策树可视化**：使用 graphviz 导出和渲染决策树结构
6. **实战应用**：红酒品种分类案例

---

## 内容结构

### 第一章：决策树算法原理

#### 1.1 算法思想

决策树是一种**树形结构**的分类模型，其核心思想是通过递归地划分特征空间来进行分类：

- **根节点**：包含所有训练数据
- **内部节点**：表示特征测试
- **分支**：表示测试结果
- **叶节点**：表示分类结果

决策树的构建过程类似于人类做决策的过程：

```
例如：判断一杯红酒的类别
1. 酒精浓度 > 13%？
   - 是 → 继续判断颜色深度
   - 否 → 继续判断其他特征
2. 颜色深度 > 5.0？
   - 是 → 类别 A
   - 否 → 类别 B
```

#### 1.2 算法流程

```
1. 输入：训练数据集、特征集
2. 选择最优特征进行分裂
3. 递归地对每个子节点进行分裂
4. 当满足停止条件时停止分裂
5. 返回决策树
```

#### 1.3 特征选择准则

**1. 基尼系数（Gini Impurity）**

衡量数据集中的不纯度：

$$Gini(p) = 1 - \sum_{i=1}^{c}p_i^2$$

其中 $p_i$ 是类别 $i$ 在节点中的比例。基尼系数越小，表示数据越纯净。

CART 决策树默认使用基尼系数作为分裂准则（对应 `criterion="gini"`）。

**2. 信息增益（Information Gain）**

基于信息熵的特征选择：

$$Entropy(S) = -\sum_{i=1}^{c}p_i \log_2(p_i)$$

$$InformationGain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)$$

可通过 `criterion="entropy"` 切换为基于信息增益的分裂策略。

| 准则 | sklearn 参数 | 特点 |
|:---|:---|:---|
| 基尼系数 | `criterion="gini"`（默认） | 计算效率高，适合大数据集 |
| 信息增益 | `criterion="entropy"` | 对纯度变化更敏感 |

#### 1.4 剪枝策略

决策树容易过拟合，需要通过剪枝来优化：

| 剪枝类型 | 说明 |
|:---|:---|
| **预剪枝** | 在构建过程中提前停止分裂 |
| **后剪枝** | 构建完整树后，从叶节点向上剪枝 |

**常用预剪枝参数**：
- `max_depth`：最大深度
- `min_samples_split`：分裂所需的最小样本数
- `min_samples_leaf`：叶节点最小样本数
- `max_features`：分裂时考虑的最大特征数

---

### 第二章：Notebook 实战 — 红酒分类

本章对应 notebook 中的实际代码单元格，逐步演示如何使用决策树对红酒数据集进行分类。

#### 2.1 数据集介绍

使用 sklearn 自带的 **Wine Dataset**（红酒检测数据集）：

| 属性 | 说明 |
|:---|:---|
| 样本数量 | 178 个 |
| 特征数量 | 13 个（酒精、苹果酸、灰分等化学成分） |
| 类别数量 | 3 个（三种不同产地的红酒品种） |
| 数据形状 | `(178, 13)` |

#### 2.2 Notebook 单元格详解

**单元格 1：导入模块**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus'] = False
```

> **说明**：导入所有需要的库和模块。`plt.rcParams` 设置中文字体以支持图表中的中文标注。

**单元格 2：设置 Notebook 全行输出**

```python
from IPython.core.interactiveshell import InteractiveShell

# 设置该对象 ast_node_interactivity 属性的值为 all，
# 表示在 notebook 下每一行有输出的代码全部输出运算结果
InteractiveShell.ast_node_interactivity = "all"
```

> **说明**：默认情况下 Jupyter 只显示每个单元格最后一行的输出。设置 `ast_node_interactivity = "all"` 后，单元格中每一行有输出的代码都会显示结果，方便同时查看多个变量的值。

**单元格 3：加载并查看数据集**

```python
# 加载 sklearn 自带红酒检测数据集
wine = load_wine()

# 获取特征数据
wine.data

# 获取数据集形状
wine.data.shape

# 获取标签
wine.target
```

输出结果：
- `wine.data`：形状为 `(178, 13)` 的二维数组
- `wine.data.shape`：`(178, 13)`
- `wine.target`：长度为 178 的一维数组，值为 0、1、2（三类红酒）

**单元格 4 & 5：构建 DataFrame 并展示数据**

```python
# 拼接 data 和 target，轴 1 方向拼接（列方向）
wine_pd = pd.concat([
    pd.DataFrame(wine.data),
    pd.DataFrame(wine.target)
], axis=1).head()

# 为标签字段添加列索引名称
wine.feature_names.append("类别")

# 设置标签的列索引名称
wine_pd.columns = wine.feature_names
wine_pd
```

> **说明**：将特征数据和标签数据横向拼接成一个 DataFrame，并添加列名。注意这里用 `wine.feature_names.append("类别")` 在原始特征名列表后追加了一个"类别"列名，所以修改后 `wine.feature_names` 包含 14 个元素。

**13 个原始特征名称**：

| 英文名 | 中文含义 |
|:---|:---|
| alcohol | 酒精 |
| malic_acid | 苹果酸 |
| ash | 灰分 |
| alcalinity_of_ash | 灰的碱性 |
| magnesium | 镁 |
| total_phenols | 总酚 |
| flavanoids | 类黄酮 |
| nonflavanoid_phenols | 非黄烷类酚类 |
| proanthocyanins | 花青素 |
| color_intensity | 颜色强度 |
| hue | 色调 |
| od280/od315_of_diluted_wines | OD280/OD315 稀释葡萄酒 |
| proline | 脯氨酸 |

**单元格 6：划分训练集和测试集**

```python
# 划分数据集为训练集和测试集
Xtrain, Xtest, Ytrain, ytest = train_test_split(
    wine.data, wine.target,
    test_size=0.3,
    random_state=420
)

# 查看训练集和测试集形状
Xtrain.shape   # (124, 13)
Xtest.shape    # (54, 13)
```

> **说明**：将数据集按 7:3 的比例划分为训练集（124 条）和测试集（54 条）。`random_state=420` 固定随机种子，确保每次划分结果一致。

**单元格 7：构建并训练决策树模型**

```python
# 使用 DecisionTreeClassifier 决策树分类器建立决策树模型，
# criterion 参数通过指定 gini 值表示采用 CART 算法构建决策树模型
clf = tree.DecisionTreeClassifier(criterion="gini")

# 调用估计器训练该决策树模型，传入训练集数据即可
clf = clf.fit(Xtrain, Ytrain)

# 模型训练完毕后，返回预测的准确度
clf.score(Xtest, ytest)
```

输出结果：`0.9444444444444444`（约 94.44%）

> **说明**：使用默认参数的 CART 决策树，在测试集上达到了约 94.4% 的准确率。`clf.score()` 是 sklearn 内置的评估方法，内部调用 `predict()` + `accuracy_score()`，使用更方便。

**单元格 8：决策树可视化**

```python
import graphviz

# 将英文特征名翻译为中文，便于可视化阅读
feature_name = [
    '酒精', '苹果酸', '灰', '灰的碱性', '镁',
    '总酚', '类黄酮', '非黄烷类酚类', '花青素',
    '颜色强度', '色调', 'od280/od315 稀释葡萄酒', '脯氨酸'
]

dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_name,
    class_names=["琴酒", "雪莉", "贝尔摩德"],
    filled=True,
    rounded=True
)

graph = graphviz.Source(dot_data)
graph
```

> **说明**：使用 `export_graphviz()` 将训练好的决策树导出为 DOT 格式的数据，再用 graphviz 渲染为可视化树形图。
>
> **参数说明**：
> - `filled=True`：节点填充颜色（颜色深浅反映类别纯度）
> - `rounded=True`：圆角节点
> - `feature_names`：特征的中文名称列表
> - `class_names`：三类红酒品种的别名（琴酒 / 雪莉 / 贝尔摩德）
>
> **注意**：可视化需要安装 graphviz 软件（不仅仅是 pip 包）：
> ```bash
> # 1. 安装 Python 包
> pip install graphviz
> # 2. 下载安装 graphviz 软件：https://graphviz.org/download/
> # 3. 将 graphviz 的 bin 目录添加到系统 PATH 环境变量
> ```

---

### 第三章：模型评估与参数调优

#### 3.1 评估指标

| 指标 | 说明 | sklearn 用法 |
|:---|:---|:---|
| 准确率 | 预测正确的比例 | `clf.score(Xtest, ytest)` 或 `accuracy_score(y_true, y_pred)` |
| 精确率 | TP / (TP + FP) | `precision_score(y_true, y_pred, average='macro')` |
| 召回率 | TP / (TP + FN) | `recall_score(y_true, y_pred, average='macro')` |
| F1 分数 | 精确率和召回率的调和平均 | `f1_score(y_true, y_pred, average='macro')` |

示例代码：

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ypred = clf.predict(Xtest)

print(f"准确率: {accuracy_score(ytest, ypred):.4f}")
print(f"精确率: {precision_score(ytest, ypred, average='macro'):.4f}")
print(f"召回率: {recall_score(ytest, ypred, average='macro'):.4f}")
print(f"F1分数: {f1_score(ytest, ypred, average='macro'):.4f}")
```

#### 3.2 过拟合问题

决策树如果不加约束，会一直生长直到每个叶节点只包含一个样本，导致在训练集上达到 100% 准确率但测试集表现很差。

**检查是否过拟合**：

```python
train_score = clf.score(Xtrain, Ytrain)
test_score = clf.score(Xtest, ytest)
print(f"训练集准确率: {train_score:.4f}")
print(f"测试集准确率: {test_score:.4f}")
# 如果训练集远高于测试集 → 过拟合
```

**解决方法**——限制树的生长：

```python
clf = tree.DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,             # 限制最大深度
    min_samples_split=5,     # 节点至少 5 个样本才继续分裂
    min_samples_leaf=2,      # 叶节点至少 2 个样本
    random_state=420
)
clf.fit(Xtrain, Ytrain)
print(f"剪枝后测试集准确率: {clf.score(Xtest, ytest):.4f}")
```

#### 3.3 使用 GridSearchCV 调参

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=tree.DecisionTreeClassifier(random_state=420),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(Xtrain, Ytrain)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳参数在测试集上评估
best_clf = grid_search.best_estimator_
print(f"调优后测试集准确率: {best_clf.score(Xtest, ytest):.4f}")
```

---

### 第四章：特征重要性分析

决策树可以输出每个特征对分类的贡献程度：

```python
# 获取特征重要性
feature_importance = pd.DataFrame({
    'feature': feature_name,
    'importance': clf.feature_importances_
}).sort_values(by='importance', ascending=False)

print(feature_importance)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('重要性')
plt.title('决策树特征重要性排序')
plt.gca().invert_yaxis()  # 重要特征在上
plt.show()
```

---

### 第五章：决策树算法特点

#### 5.1 优点

| 优点 | 说明 |
|:---|:---|
| **易于理解和解释** | 决策树直观，可视化后非专业人士也能理解 |
| **无需数据预处理** | 不需要特征标准化或归一化 |
| **处理非线性关系** | 天然处理非线性可分数据 |
| **多分类支持** | 天然支持多分类任务 |
| **可处理缺失值** | 某些实现支持缺失值处理 |

#### 5.2 缺点

| 缺点 | 说明 |
|:---|:---|
| **容易过拟合** | 尤其是在深度较大、不剪枝时 |
| **对噪声敏感** | 少量噪声数据可能导致完全不同的树结构 |
| **不稳定性** | 数据的微小变化可能导致树结构的巨大变化 |
| **偏向于多数类** | 对不平衡数据处理不佳 |
| **忽略特征间相关性** | 假设各特征相互独立 |

#### 5.3 适用场景

- 数据具有明显的层次决策结构
- 需要可解释性强、可审计的模型
- 数据预处理成本高的场景
- 中小型数据集（大数据集推荐集成方法）

---

### 第六章：决策树变体

| 算法 | 提出者 | 分裂方式 | 分裂准则 | 适用任务 |
|:---|:---|:---|:---|:---|
| **CART** | Breiman (1984) | 二叉树 | 基尼系数 | 分类 + 回归 |
| **ID3** | Quinlan (1986) | 多叉树 | 信息增益 | 仅分类 |
| **C4.5** | Quinlan (1993) | 多叉树 | 信息增益比 | 分类（支持连续值） |
| **C5.0** | Quinlan | 多叉树 | 信息增益比 | 分类（C4.5 的优化版） |

sklearn 的 `DecisionTreeClassifier` 默认使用 **CART** 算法。

---

## 环境要求

| 依赖包 | 用途 |
|:---|:---|
| Python 3.7+ | 编程语言 |
| numpy | 数值计算 |
| pandas | 数据处理 |
| matplotlib | 数据可视化 |
| scikit-learn | 机器学习库（提供决策树和数据集） |
| graphviz | 决策树可视化（需额外安装系统软件） |

安装命令：

```bash
pip install numpy pandas matplotlib scikit-learn graphviz
```

> **注意**：graphviz 除了 pip 安装 Python 包外，还需要在系统层面安装 graphviz 软件并配置 PATH 环境变量。详见 [graphviz 官网](https://graphviz.org/download/)。

---

## 使用方法

1. 启动 Jupyter Notebook：
   ```bash
   jupyter notebook
   ```
2. 在浏览器中打开 `05th_DecisionTree.ipynb`
3. 按顺序运行每个代码单元格（`Shift + Enter`）

---

## 学习建议

1. **先理解原理**：搞清楚基尼系数和信息增益的计算方式
2. **手动调参**：修改 `criterion`、`max_depth`、`random_state` 等参数，观察模型变化
3. **可视化分析**：通过决策树图理解模型每一步的决策逻辑
4. **对比实验**：分别用 `criterion="gini"` 和 `criterion="entropy"` 训练，比较结果
5. **横向对比**：将决策树与 KNN（04th）在同一数据集上的表现进行对比

---

## 常见问题

### Q1: criterion 选 gini 还是 entropy？

| 参数值 | 特点 | 推荐场景 |
|:---|:---|:---|
| `"gini"`（默认） | 计算效率高，不涉及对数运算 | 大多数情况，大数据集 |
| `"entropy"` | 对纯度变化更敏感，可能生成更平衡的树 | 需要更精细分裂时 |

实际应用中两者差异通常很小，默认 `"gini"` 即可。

### Q2: 决策树过拟合怎么办？

1. **限制深度**：`max_depth=5`
2. **增加分裂门槛**：`min_samples_split=5`
3. **增加叶节点最小值**：`min_samples_leaf=2`
4. **后剪枝**：使用 `ccp_alpha`（成本复杂度剪枝）

### Q3: 如何处理不平衡数据？

```python
# 使用 class_weight 参数自动平衡类别权重
clf = tree.DecisionTreeClassifier(class_weight='balanced')
```

### Q4: 决策树和 KNN 有什么区别？

| 对比维度 | 决策树 | KNN |
|:---|:---|:---|
| 学习方式 |  eager learning（先训练模型） | lazy learning（无训练过程） |
| 可解释性 | 高（树结构直观） | 低（黑盒投票） |
| 预测速度 | 快（O(log n)） | 慢（需要计算所有距离） |
| 对异常值 | 相对不敏感 | 敏感 |
| 特征缩放 | 不需要 | 需要（距离度量） |

---

**最后更新**：2026-06-18
