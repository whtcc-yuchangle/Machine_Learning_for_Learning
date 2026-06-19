# 聚类算法（Cluster Algorithm）教程

## 概述

本 notebook 系统地介绍机器学习中四大类聚类算法，并通过 sklearn 模拟数据和 `pyclust` 库的 K-Medoids 进行实战演示。聚类属于**无监督学习（Unsupervised Learning）**，目标是在没有标签的情况下将数据划分成有意义的组（簇），使得**簇内相似度高、簇间相似度低**。

---

## 学习目标

通过学习本教程，您将掌握：

1. **聚类 vs 分类**：理解无监督学习与有监督学习的本质区别
2. **四大聚类范式**：基于原型、层次、密度、图论的聚类方法及其适用场景
3. **K-Means 原理**：掌握最经典的原型聚类算法，理解质心迭代更新机制
4. **K-Medoids 实战**：使用 `pyclust` 的 `KMedoids` 对不同 k 值进行聚类并可视化
5. **t-SNE 降维可视化**：理解高维数据的低维嵌入及其在聚类可视化中的作用
6. **DBSCAN 密度聚类**：了解无需预设簇数的密度聚类方法

---

## 内容结构

### 第一章：聚类算法概述与分类体系

#### 1.1 什么是聚类？

聚类将数据集划分为若干个**簇（Cluster）**，使得：

- **簇内高内聚**：同一簇内的样本尽可能相似
- **簇间低耦合**：不同簇的样本尽可能不同

与分类（Classification）不同，聚类没有标签 $y$，完全基于数据本身的特征分布进行分组：

| 对比维度 | 分类（有监督） | 聚类（无监督） |
|:---|:---|:---|
| 训练数据 | 有标签 $(X, y)$ | 无标签（只有 $X$） |
| 目标 | 学习 $X \to y$ 的映射 | 发现数据内在结构 |
| 评估 | 准确率、精确率、召回率 | 轮廓系数、CH 指数、DB 指数 |
| 典型算法 | KNN、决策树、SVM | K-Means、DBSCAN、层次聚类 |

> **直观理解**：分类是老师告诉你每道题的正确答案后再让你判卷；聚类是你自己根据题目的相似性把它们分组，没人告诉你标准答案。

#### 1.2 聚类算法四大类

Notebook 中列出了四类主流聚类算法：

```
聚类算法
├── 基于原型（划分）聚类
│   ├── K-Means —— 最经典，质心为簇代表
│   ├── K-Medoids —— 用实际样本点作为簇中心（对噪声更鲁棒）
│   └── GMM（高斯混合模型）—— 概率软聚类，每个簇是一个高斯分布
│
├── 基于层次聚类
│   ├── AGNES —— 自底向上聚合，逐步合并最近的簇
│   ├── BIRCH —— 平衡迭代归约，适合大数据集
│   └── CURE —— 使用代表点聚类，能发现任意形状的簇
│
├── 基于密度聚类
│   ├── DBSCAN —— 基于密度连通性，能发现任意形状簇 + 识别噪声
│   └── OPTICS —— DBSCAN 的改进版，对参数不敏感
│
└── 基于图论聚类
    └── 谱聚类 —— 基于图拉普拉斯矩阵特征向量进行聚类
```

---

### 第二章：基于原型的聚类算法

#### 2.1 K-Means 算法

K-Means 是**最经典、最常用**的聚类算法，核心思想是迭代优化簇内平方误差：

**目标函数（SSE — Sum of Squared Errors）：**

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

其中 $\mu_k$ 是第 $k$ 个簇的质心（该簇所有样本的均值）。

**算法流程：**

| 步骤 | 操作 | 说明 |
|:---|:---|:---|
| **① 初始化** | 随机选择 $K$ 个样本作为初始质心 $\mu_1, \mu_2, ..., \mu_K$ | K 需预先指定 |
| **② 分配** | 将每个样本 $x_i$ 分配到最近的质心：$C_k = \{x_i : \|x_i - \mu_k\| \leq \|x_i - \mu_j\|, \forall j\}$ | 形成 K 个簇 |
| **③ 更新** | 重新计算每个簇的质心：$\mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i$ | 质心移动到簇的几何中心 |
| **④ 收敛判断** | 重复 ②③，直到质心不再变化或达到最大迭代次数 | 保证收敛到局部最优 |

```python
# sklearn K-Means 示例
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
kmeans.fit(X)
labels = kmeans.labels_       # 每个样本的簇标签
centers = kmeans.cluster_centers_  # 质心坐标
```

| sklearn 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `n_clusters` | `8` | 簇的数量 $K$ |
| `init` | `'k-means++'` | 初始化方法，k-means++ 能选更好的初始质心 |
| `n_init` | `'auto'` | 运行次数，取 SSE 最小的结果 |
| `max_iter` | `300` | 单次运行的最大迭代次数 |
| `random_state` | `None` | 随机种子 |
| `algorithm` | `'lloyd'` | 优化算法：`'lloyd'`（经典）或 `'elkan'`（用三角不等式加速） |

**优点与局限：**

| 优点 | 局限 |
|:---|:---|
| 简单、快速，$O(n \cdot K \cdot d \cdot t)$ | 需预知 K 值 |
| 可解释性强，质心即为簇代表 | 对初始质心敏感（用 k-means++ 缓解） |
| 对凸形（球形）簇效果好 | 对非凸形状簇效果差 |
| 大规模数据可扩展（Mini-Batch K-Means） | 对噪声和离群点敏感 |

#### 2.2 K-Medoids（K-中心点）算法

K-Medoids 是 K-Means 的变体，核心区别在于**用实际样本点（Medoid）作为簇中心**，而非质心（样本均值）：

| 对比维度 | K-Means | K-Medoids |
|:---|:---|:---|
| **簇中心** | 质心（可以是虚拟点） | Medoid（必须是实际样本） |
| **目标函数** | 最小化簇内平方误差 | 最小化簇内相异度之和 |
| **对噪声鲁棒性** | 敏感（异常值会拉偏质心） | 鲁棒（Medoid 是真实样本） |
| **计算复杂度** | $O(nKdt)$ | $O(n^2)$（更高） |
| **距离度量** | 通常用欧氏距离 | 支持任意相异度矩阵 |

> **选择建议**：数据干净、有噪声少 → K-Means（更快）；数据可能有异常值 → K-Medoids（更鲁棒）。

本 notebook 使用 `pyclust` 库的 `KMedoids` 进行实战：

```python
from pyclust import KMedoids
kmedoids = KMedoids(n_clusters=3, distance='euclidean', max_iter=1000)
labels = kmedoids.fit_predict(data)
```

| pyclust 参数 | 说明 |
|:---|:---|
| `n_clusters` | 簇的数量 |
| `distance` | 距离度量：`'euclidean'`（默认） |
| `max_iter` | 最大迭代次数 |

**安装 pyclust**：

```bash
pip install pyclust
```

> **注意**：`pyclust` 是第三方库，非 sklearn 原生支持。sklearn 中没有 K-Medoids，可以用 `sklearn_extra.cluster.KMedoids`（需安装 `scikit-learn-extra`）。

#### 2.3 GMM（高斯混合模型）

GMM 是一种**软聚类**（Soft Clustering）方法，假设数据由多个高斯分布混合生成：

$$p(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x|\mu_k, \Sigma_k)$$

其中 $\pi_k$ 是第 $k$ 个高斯分量的权重（$\sum \pi_k = 1$），$\mathcal{N}(x|\mu_k, \Sigma_k)$ 是多元高斯分布。

| 对比维度 | K-Means | GMM |
|:---|:---|:---|
| 聚类方式 | 硬聚类（每个样本属于一个簇） | 软聚类（每个样本有属于各簇的概率） |
| 簇形状 | 各向同性（球形） | 任意椭圆形（由协方差矩阵决定） |
| 优化方法 | EM 算法的特例 | 完整 EM 算法 |
| 概率输出 | 否 | 是（`predict_proba()`） |

---

### 第三章：基于层次的聚类算法

#### 3.1 AGNES（AGglomerative NESting）

自底向上的聚合层次聚类，初始每个样本为一个簇，逐步合并最近的簇：

**簇间距离度量（Linkage）：**

| Linkage | 公式 | 特点 |
|:---|:---|:---|
| **Single Link** | $d(C_i, C_j) = \min_{a \in C_i, b \in C_j} d(a,b)$ | 能处理非椭圆形状，但对噪声敏感 |
| **Complete Link** | $d(C_i, C_j) = \max_{a \in C_i, b \in C_j} d(a,b)$ | 倾向紧凑球形簇 |
| **Average Link** | $d(C_i, C_j) = \frac{1}{|C_i||C_j|}\sum_{a \in C_i}\sum_{b \in C_j} d(a,b)$ | 折中方案，较稳定 |
| **Ward** | 最小化合併后的 SSE 增量 | 倾向均匀大小的簇 |

```python
from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = agg.fit_predict(X)
```

**层次聚类可视化 — 树状图（Dendrogram）：**

```python
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(X, method='ward')
dendrogram(Z)
plt.show()
```

树状图将整个聚类过程可视化——横轴是样本，纵轴是距离，可以在任意高度"切一刀"得到不同数量的簇。

#### 3.2 BIRCH 与 CURE（简要）

| 算法 | 核心思路 | 适用场景 |
|:---|:---|:---|
| **BIRCH** | 构建 CF 树（Clustering Feature Tree）增量聚类 | 大规模数据，内存受限 |
| **CURE** | 选择多个代表点表示簇，向质心方向收缩 | 能发现非球形、大小不均的簇 |

---

### 第四章：基于密度的聚类算法

#### 4.1 DBSCAN

DBSCAN 将簇定义为**密度相连的点的最大集合**，不需要预先指定簇数：

| 参数 | 含义 |
|:---|:---|
| **$\varepsilon$ (eps)** | 邻域半径 |
| **MinPts** | 核心点所需的最小邻域内样本数 |

**点的三种类型：**

```
核心点（Core Point）  ：邻域内样本数 ≥ MinPts
边界点（Border Point）：邻域内样本数 < MinPts，但在核心点的邻域内
噪声点（Noise Point） ：既非核心点也非边界点
```

**算法流程**：从任一核心点开始，通过密度相连关系不断扩展簇，直到所有密度可达的点都被包含。

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)  # labels=-1 表示噪声点
```

**优点 vs 局限：**

| 优点 | 局限 |
|:---|:---|
| 不需要指定 K（簇数） | 参数 $\varepsilon$ 和 MinPts 难以选择 |
| 能发现**任意形状**的簇 | 密度差异大的数据效果差 |
| 能识别噪声点 | 高维数据效果不佳（维度灾难） |
| 对数据输入顺序不敏感 | 计算复杂度 $O(n \log n)$（空间索引）~ $O(n^2)$ |

> **调参建议**：先用 k-distance 图选择合适的 $\varepsilon$（找拐点）；MinPts 一般取 $\geq D+1$（D 为特征维数）。

#### 4.2 OPTICS

OPTICS 是 DBSCAN 的改进版，对 $\varepsilon$ 参数不敏感，通过**可达距离**生成数据点的排序，然后在排序图上识别簇。

```python
from sklearn.cluster import OPTICS

optics = OPTICS(min_samples=5)
labels = optics.fit_predict(X)
```

---

### 第五章：谱聚类（Spectral Clustering）

谱聚类基于图论，将数据点视为图的节点，边权重表示相似度，通过图的拉普拉斯矩阵特征向量进行聚类。

**核心步骤：**

| 步骤 | 操作 |
|:---|:---|
| ① 构建相似图 | 计算样本间的相似度矩阵 $W$（常用 RBF 核：$W_{ij} = \exp(-\gamma\|x_i - x_j\|^2)$） |
| ② 计算拉普拉斯矩阵 | $L = D - W$（非归一化）或 $L_{sym} = D^{-1/2}LD^{-1/2}$（归一化） |
| ③ 特征分解 | 求 $L$ 最小的 $k$ 个特征值对应的特征向量，构成 $n \times k$ 矩阵 |
| ④ K-Means | 在特征向量矩阵的行上运行 K-Means |

```python
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=4, affinity='rbf', gamma=1.0)
labels = sc.fit_predict(X)
```

> **特点**：擅长发现非凸形状的簇，适合图像分割；但对大规模数据计算开销大。

---

### 第六章：Notebook 实战 — t-SNE + K-Medoids

#### 6.1 单元格 1-2：导入库与数据生成

**导入模块**：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# 中文显示配置
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Simhei']
```

**使用 `make_blobs` 创建模拟数据**：

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

# 按颜色展示真实标签
color = ["red", "pink", "orange", "gray"]
for i in range(4):
    plt.scatter(X[y==i, 0], X[y==i, 1], marker='o', s=8, c=color[i])
```

| `make_blobs` 参数 | 值 | 说明 |
|:---|:---|:---|
| `n_samples` | `500` | 总样本数 |
| `n_features` | `2` | 特征维度（方便 2D 绘图） |
| `centers` | `4` | 簇数量 |
| `random_state` | `1` | 固定随机种子 |

> `make_blobs` 创建的每个簇服从各向同性的高斯分布，是**天然适合 K-Means 的球形数据**。实际数据往往更复杂（非球形、密度不均），这引出了 DBSCAN 和 GMM 等方法的必要性。

#### 6.2 单元格 3-5：高维数据与 t-SNE 降维

**构造高维聚类数据**（10 维）：

```python
data1 = np.random.normal(0, 0.9, (1000, 10))   # 簇1：中心在0
data2 = np.random.normal(1, 0.9, (1000, 10))   # 簇2：中心在1
data3 = np.random.normal(2, 0.9, (1000, 10))   # 簇3：中心在2
data4 = np.random.normal(3, 0.9, (1000, 10))   # 簇4：中心在3
data5 = np.random.normal(50, 0.9, (50, 10))    # 簇5：中心在50（离群簇）
data = np.concatenate((data1, data2, data3, data4, data5))
# data.shape → (4050, 10)
```

> **数据特点**：4 个主簇（各 1000 样本，中心从 0 到 3） + 1 个远距离小簇（50 样本，中心在 50）。这模拟了实际数据中的**离群群体**——它们与主流数据分布很远。

**t-SNE 降维**（10D → 2D）：

```python
from sklearn.manifold import TSNE

data_TSNE = TSNE(learning_rate=100).fit_transform(data)
# data_TSNE.shape → (4050, 2)
```

**t-SNE（t-Distributed Stochastic Neighbor Embedding）**是一种非线性降维方法，专门用于高维数据的**可视化**：

| TSNE 参数 | 说明 |
|:---|:---|
| `n_components` | 目标维度（默认 2） |
| `perplexity` | 平衡局部与全局的困惑度（默认 30，范围 5-50） |
| `learning_rate` | 学习率（默认 200，范围 10-1000） |
| `random_state` | 随机种子 |

> **重要警告**：t-SNE 仅保留局部近邻关系，**簇间距离和全局结构可能失真**。因此 t-SNE 结果适合**可视化定性判断**，不能基于 t-SNE 图中的距离做定量聚类评估。

#### 6.3 单元格 6：K-Medoids 聚类与可视化

```python
from pyclust import KMedoids

plt.figure(figsize=(12, 8))
for i in range(2, 6):    # 尝试 k = 2, 3, 4, 5
    k = KMedoids(n_clusters=i, distance='euclidean', max_iter=1000).fit_predict(data)
    colors = ([['red', 'blue', 'black', 'yellow', 'green'][i] for i in k])
    plt.subplot(219 + i)
    plt.scatter(data_TSNE[:, 0], data_TSNE[:, 1], c=colors, s=10)
    plt.title(f'K-medoids Result of {i}')
```

> **实验目的**：对比 $k=2,3,4,5$ 时的聚类结果。由于数据有 5 个生成簇（其中 1 个极小的离群簇），$k=5$ 时理论上应能分离出所有簇，而 $k=4$ 时可能会将离群簇合并到最近的簇中。

---

### 第七章：聚类评估指标

由于聚类没有真实标签（无监督），评估比分类更难。常用指标：

#### 7.1 内部指标（无需真实标签）

| 指标 | 公式/含义 | 越优 |
|:---|:---|:---|
| **轮廓系数（Silhouette Score）** | $s = \frac{b - a}{\max(a,b)}$，$a$=簇内平均距离，$b$=到最近簇的平均距离 | 接近 1 |
| **Calinski-Harabasz Index（CH 指数）** | $\frac{\text{tr}(B_k)}{\text{tr}(W_k)} \times \frac{n-K}{K-1}$，方差比率 | 越大越好 |
| **Davies-Bouldin Index（DB 指数）** | $\frac{1}{K}\sum_{i=1}^{K} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}$ | 越小越好 |
| **簇内 SSE（Elbow Method）** | $\sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$ | 找拐点（Elbow） |

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

silhouette_score(X, labels)            # 轮廓系数 → 接近1好
calinski_harabasz_score(X, labels)     # CH指数 → 越大越好
davies_bouldin_score(X, labels)        # DB指数 → 越小越好
```

#### 7.2 Elbow Method（手肘法）选择 K

```python
sse = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X)
    sse.append(km.inertia_)

plt.plot(range(1, 10), sse, marker='o')
plt.xlabel('K')
plt.ylabel('SSE')
# K=拐点处即为推荐的簇数
```

#### 7.3 外部指标（有真实标签时）

| 指标 | 含义 |
|:---|:---|
| **ARI（Adjusted Rand Index）** | 衡量聚类与真实标签的一致性，值域 [-1, 1] |
| **AMI（Adjusted Mutual Information）** | 互信息调整版本，值域 [0, 1] |
| **NMI（Normalized Mutual Information）** | 归一化互信息 |
| **同质性/完整性/V-Measure** | 三个互补维度 |

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
```

---

### 第八章：聚类算法速查对比表

| 算法 | 类型 | 需指定 K | 簇形状 | 噪声识别 | 大规模适用 | 复杂度 |
|:---|:---|:---|:---|:---|:---|:---|
| **K-Means** | 原型 | 是 | 球形 | 否 | 是（MiniBatch） | $O(nKdt)$ |
| **K-Medoids** | 原型 | 是 | 球形 | 比 K-Means 好 | 中等 | $O(n^2)$ |
| **GMM** | 原型 | 是 | 椭圆形 | 否 | 中等 | $O(nKd^2t)$ |
| **AGNES** | 层次 | 可后选 | 取决于 Linkage | 否 | 否 | $O(n^2 \log n)$ |
| **BIRCH** | 层次 | 可后选 | 球形 | 部分 | **是** | $O(n)$ |
| **DBSCAN** | 密度 | **否** | **任意** | **是** | 中等（空间索引） | $O(n \log n)$ |
| **OPTICS** | 密度 | **否** | **任意** | **是** | 中等 | $O(n \log n)$ |
| **谱聚类** | 图论 | 是 | 任意 | 否 | 否 | $O(n^3)$ |

---

### 第九章：算法选择决策流程

```
                        开始
                         │
                         ▼
                  ┌─ 已知 K 值？ ─┐
                  │ 是            │ 否
                  ▼               ▼
           ┌─ 大规模数据？ ─┐  ┌─ 需识别噪声？ ─┐
           │ 是   │ 否     │  │ 是   │ 否      │
           ▼      ▼        │  ▼      ▼         │
       MiniBatch  ┌ 簇形状？┐ DBSCAN  ┌ 数据量？┐
       K-Means    │球形│任意│ OPTICS │小 │ 大  │
                  ▼    ▼    │        ▼    ▼    │
              K-Means  GMM  │    谱聚类  BIRCH  │
              (或Medoids)   │                   │
                            └───────────────────┘
```

> **一般经验**：先用 K-Means 快速试（不指定 K 时用 Elbow Method），如果效果不理想再考虑 DBSCAN（能发现任意形状和噪声）或 GMM（需要概率软聚类时）。

---

## 环境要求

| 依赖包 | 用途 |
|:---|:---|
| Python 3.7+ | 编程语言 |
| numpy | 数值计算与随机数据生成 |
| pandas | 数据处理（备用） |
| matplotlib | 数据可视化 |
| scikit-learn | `make_blobs`、`TSNE`、内置聚类算法 |
| pyclust | K-Medoids 实现 |

安装命令：

```bash
pip install numpy pandas matplotlib scikit-learn pyclust
```

---

## 使用方法

1. 启动 Jupyter Notebook：
   ```bash
   jupyter notebook
   ```
2. 在浏览器中打开 `10th_ClusterAlgorithm.ipynb`
3. 按顺序运行每个代码单元格（`Shift + Enter`）
4. 观察 t-SNE 可视化结果，对比不同 k 值下的聚类效果

---

## 学习建议

1. **先理解分类体系**：搞清楚四大类聚类范式的核心区别和适用场景
2. **深入 K-Means**：手动推演一轮 K-Means 迭代过程，理解质心更新
3. **动手调参**：修改 notebook 中 KMedoids 的 `n_clusters`、`distance` 参数，观察聚类结果变化
4. **尝试其他算法**：用 sklearn 的 `DBSCAN`、`AgglomerativeClustering` 替代 K-Medoids，对比结果
5. **学习评估指标**：对聚类结果计算轮廓系数，实践 Elbow Method 选择最优 K
6. **注意 t-SNE 局限**：理解 t-SNE 只能可视化的限制，不能替代定量评估
7. **横向对比**：在同一数据上用 K-Means、DBSCAN、GMM 三种方法，体验不同范式的差异

---

## 常见问题

### Q1: K-Means 的 K 怎么选？

三种主流方法：
1. **Elbow Method（手肘法）**：绘制 SSE-K 曲线，找拐点
2. **轮廓系数**：选使平均轮廓系数最大的 K
3. **业务知识**：根据实际问题含义确定合理的簇数

> 没有普适的最佳方法，建议结合多种指标综合判断。

### Q2: K-Means 和 K-Medoids 选哪个？

| 场景 | 推荐 |
|:---|:---|
| 数据量大、计算效率重要 | K-Means |
| 存在异常值或极端值 | K-Medoids |
| 需要可解释的簇代表 | K-Medoids（Medoid 是真实样本） |
| 均值有实际意义（如身高、价格均值） | K-Means |

### Q3: DBSCAN 的 eps 怎么调？

**k-distance 图法**：
1. 对每个点计算到第 k 个最近邻的距离（$k$ 通常取 `min_samples`）
2. 将所有点的 k-distance 排序后绘图
3. 拐点处的值就是合适的 `eps`

```python
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X)
distances, _ = neighbors_fit.kneighbors(X)
distances = np.sort(distances[:, -1])
plt.plot(distances)
# eps = 拐点处的y值
```

### Q4: 数据需要标准化吗？

**聚类算法几乎都依赖距离——必须标准化！** 如果特征量纲不一致（如年龄 0-100 vs 收入 0-100000），距离计算会被大数值特征主导。

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

clf = make_pipeline(StandardScaler(), KMeans(n_clusters=4))
clf.fit(X)
```

### Q5: t-SNE 图中簇的大小和距离可信吗？

**不可信**。t-SNE 只保持局部近邻关系，图中：
- 簇的大小可能不代表实际样本数量
- 簇间距离可能不代表实际相似度
- 同一数据不同 `perplexity` 值可能产生差异很大的图

> **正确用法**：t-SNE 用于"大致看看数据有没有分组趋势"，不做定量分析。

### Q6: 如何确定聚类结果的好坏？

1. **内部指标**（无标签）— 轮廓系数、CH 指数、DB 指数
2. **外部指标**（有标签）— ARI、NMI、V-Measure
3. **可视化验证** — 降维图 + 原始特征两两散点图
4. **业务验证** — 聚类结果在业务上是否有意义

> 最好的验证是在多个指标上一致表现良好，并且聚类结果能从业务角度解释。

---

**最后更新**：2026-06-19
