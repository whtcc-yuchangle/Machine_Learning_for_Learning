# 集成学习（Ensemble Learning）教程

## 概述

本 notebook 系统地介绍机器学习中两大主流集成学习算法——**随机森林（Random Forest）**和 **AdaBoost**，通过 sklearn 的 wine、iris、diabetes 等数据集进行实战对比。集成学习的核心思想是**"三个臭皮匠，顶个诸葛亮"**——将多个弱学习器组合成一个强学习器，从而获得比单一模型更优的泛化性能。

---

## 学习目标

通过学习本教程，您将掌握：

1. **集成学习思想**：理解 Bagging 与 Boosting 两大范式及其核心区别
2. **随机森林原理**：掌握 Bootstrap 采样、特征随机选择、OOB 估计、投票机制
3. **随机森林调参**：通过交叉验证学习曲线找到最优 `n_estimators`
4. **AdaBoost 原理**：理解自适应增强、样本权重更新、弱学习器加权组合
5. **单树 vs 集成对比**：用交叉验证量化集成学习相对单模型的性能提升
6. **OOB 与包外估计**：理解不用额外测试集也能评估模型的 OOB Score
7. **回归任务扩展**：掌握 `RandomForestRegressor` 的使用

---

## 内容结构

### 第一章：集成学习概述

#### 1.1 什么是集成学习？

集成学习（Ensemble Learning）通过构建并结合**多个基学习器（Base Learner）**来完成学习任务：

```
单个模型：  Data → [ 模型 ] → 预测
集成学习：  Data → [ 模型₁ ] → 投票/平均 → 最终预测
                  [ 模型₂ ]
                  [ ...   ]
                  [ 模型ₙ ]
```

**为什么集成有效？**

> 扔一枚硬币，正面朝上的概率为 0.5。但如果找 25 枚硬币一起扔，多数投票决定结果，错误的概率是多少？
>
> $$P(\text{多数错误}) = \sum_{i=13}^{25} \binom{25}{i} \cdot 0.2^{i} \cdot (1-0.2)^{25-i} \approx 0.00037$$

即使每棵树的错误率高达 20%，25 棵树投票后的整体错误率也仅为 **0.037%**——远低于单棵树的错误率。

#### 1.2 两大集成范式

```
集成学习
├── Bagging（并行）
│   ├── 核心：Bootstrap 采样 + 并行训练 + 投票/平均
│   ├── 代表：随机森林（Random Forest）
│   └── 特点：降低方差（Variance），防止过拟合
│
└── Boosting（串行）
    ├── 核心：序列训练 + 样本权重调整 + 加权组合
    ├── 代表：AdaBoost、GBDT、XGBoost
    └── 特点：降低偏差（Bias），逐步修正错误
```

| 对比维度 | Bagging | Boosting |
|:---|:---|:---|
| **训练方式** | 各学习器**并行**独立训练 | 学习器**串行**依赖训练 |
| **样本使用** | Bootstrap 自助采样（约 63% 样本） | 全量样本，权重动态调整 |
| **结合方式** | 简单投票（分类）/ 平均（回归） | **加权**投票/平均 |
| **核心优势** | 降**方差**，防过拟合 | 降**偏差**，修正误差 |
| **对噪声敏感度** | 较鲁棒 | **敏感**（会过度关注噪声样本） |
| **代表算法** | 随机森林 | AdaBoost、GBDT、XGBoost |

---

### 第二章：随机森林（Random Forest）

#### 2.1 核心原理

随机森林 = **Bagging + 决策树 + 特征随机选择**

**双重随机性**（使森林中的树"各不相同"）：

| 随机性来源 | 方法 | 作用 |
|:---|:---|:---|
| **样本随机** | Bootstrap 采样（有放回抽取 n 个样本） | 每棵树看到的样本不同 |
| **特征随机** | 每次分裂时只考虑 $m$ 个随机特征子集（$m \ll d$） | 每棵树的分裂选择不同 |

> **关键参数**：`max_features` — 每次分裂时随机考虑的特征数
> - 分类默认：$m = \sqrt{d}$
> - 回归默认：$m = d/3$

**算法流程：**

| 步骤 | 操作 |
|:---|:---|
| ① Bootstrap | 从原始数据集有放回抽取 n 个样本，形成 n 个自助样本集 |
| ② 建树 | 对每个自助样本集，使用特征随机选择建一棵决策树（不剪枝） |
| ③ 重复 | 重复 ①② 共 $T$ 次，得到 $T$ 棵决策树 |
| ④ 投票 | 分类：多数投票；回归：平均值 |

```python
from sklearn.ensemble import RandomForestClassifier

# 基本使用
rfc = RandomForestClassifier(n_estimators=25, random_state=0)
rfc.fit(X_train, y_train)
rfc.score(X_test, y_test)
```

#### 2.2 单树 vs 随机森林 — 交叉验证对比

Notebook 使用 wine 数据集对比单一决策树与随机森林的性能：

```python
# 单树
clf = DecisionTreeClassifier(random_state=0)
clf_s = cross_val_score(clf, wine.data, wine.target, cv=10)

# 随机森林（25棵树）
rfc = RandomForestClassifier(n_estimators=25)
rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)

# 可视化对比
plt.plot(range(1, 11), rfc_s, label='Random Forest')
plt.plot(range(1, 11), clf_s, label='Decision Tree')
```

**10 组 10 折交叉验证**（100 次评估）进一步消除单次实验的随机性：

```python
rfc_scores = []
clf_scores = []
for i in range(10):
    rfc_scores.append(cross_val_score(
        RandomForestClassifier(n_estimators=25),
        wine.data, wine.target, cv=10).mean())
    clf_scores.append(cross_val_score(
        DecisionTreeClassifier(),
        wine.data, wine.target, cv=10).mean())
```

> **结论**：随机森林在几乎所有轮次中都稳定优于单决策树，且方差更小。

#### 2.3 关键参数速查

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `n_estimators` | `100` | 森林中树的数量（越大越稳定，但收益递减） |
| `criterion` | `'gini'` | 分裂标准：`'gini'` / `'entropy'` |
| `max_depth` | `None` | 树的最大深度（默认不限制，完全生长） |
| `min_samples_split` | `2` | 内部节点再分裂的最小样本数 |
| `min_samples_leaf` | `1` | 叶节点的最小样本数 |
| `max_features` | `'sqrt'` | 每次分裂的特征数：`'sqrt'` / `'log2'` / `None`（全特征） |
| `bootstrap` | `True` | 是否使用 Bootstrap 采样 |
| `oob_score` | `False` | 是否计算 OOB Score |
| `random_state` | `None` | 随机种子（控制 Bootstrap 和特征选择的随机性） |
| `n_jobs` | `None` | 并行作业数（`-1` 使用所有核心） |

#### 2.4 n_estimators 学习曲线

```python
superpa = []
for i in range(200):
    rfc = RandomForestClassifier(n_estimators=i+1)
    superpa.append(cross_val_score(rfc, wine.data, wine.target, cv=10).mean())

plt.plot(range(1, 201), superpa)
# 取最高分对应的 n_estimators
best_n = superpa.index(max(superpa)) + 1
```

**学习曲线特点**：
- 初期（树少）：分数随 `n_estimators` 增加快速上升
- 中期：增长趋缓，曲线逐渐平稳
- 后期：趋于饱和，继续增加树数量**收益递减**

#### 2.5 Bootstrap 与 OOB（Out-of-Bag）

**Bootstrap 采样**：从 n 个样本中有放回抽取 n 次

每个样本被抽中的概率：
$$P(\text{被抽中}) = 1 - \left(1 - \frac{1}{n}\right)^n \approx 1 - e^{-1} \approx 63.2\%$$

**约 36.8% 的样本从未被抽中**——这些就是**包外样本（OOB）**，可天然用作验证集！

```python
# 启用 OOB Score（不需要手动划分测试集！）
rfc = RandomForestClassifier(n_estimators=25, oob_score=True)
rfc.fit(wine.data, wine.target)  # 直接使用全部数据
print(rfc.oob_score_)             # OOB 准确率 ≈ 0.972
```

| OOB Score 优势 | 说明 |
|:---|:---|
| 免测试集划分 | 不需要 `train_test_split` |
| 无偏估计 | 每棵树的 OOB 样本从未参与该树的训练 |
| 数据利用率高 | 全部数据既用于训练又用于验证（不同角度） |

#### 2.6 随机森林的误差分析

Notebook 给出了一个重要的理论可视化：

```python
import numpy as np
from scipy.special import comb

x = np.linspace(0, 1, 20)
y = []
for epsilon in np.linspace(0, 1, 20):
    E = np.array([comb(25, i) * (epsilon**i) *
          ((1-epsilon)**(25-i)) for i in range(13, 26)]).sum()
    y.append(E)

# 对比：基学习器各不相同时 vs 基学习器全部相同时
plt.plot(x, y, "o-", label="when estimators are different")
plt.plot(x, x, "--", color="red", label="if all estimators are same")
```

| 曲线 | 含义 |
|:---|:---|
| 蓝线（各基学习器不同） | 随机森林的真实误差 — 随基学习器错误率上升**缓慢增长** |
| 红线（所有基学习器相同） | 全相同学习器的误差 = 单个错误率 — **线性增长** |

> **核心洞察**：随机森林的优势**不在于每棵树有多准**，而在于**树与树之间的多样性（Diversity）**。如果所有树输出相同，集成没有意义。

验证森林中树的多样性 — 每棵树的 `random_state` 都不同：

```python
rfc = RandomForestClassifier(n_estimators=20, random_state=2)
rfc.fit(Xtrain, Ytrain)
for i in range(len(rfc.estimators_)):
    print(rfc.estimators_[i].random_state)
# 输出 20 个不同的随机种子 → 每棵树在特征/样本选择上各不相同
```

#### 2.7 随机森林回归（RandomForestRegressor）

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
regressor = RandomForestRegressor(
    n_estimators=100, random_state=0, oob_score=True
)
cross_val_score(regressor, diabetes.data, diabetes.target,
                cv=10, scoring="neg_mean_squared_error")
```

| RandomForestRegressor 参数 | 说明 |
|:---|:---|
| `n_estimators` | 树数量 |
| `oob_score` | OOB R² 分数 |
| `criterion` | `'squared_error'`（MSE）/ `'absolute_error'` / `'friedman_mse'` |

---

### 第三章：AdaBoost（自适应增强）

#### 3.1 核心原理

AdaBoost（Adaptive Boosting）通过**串行训练弱学习器**，每次训练时**增大前一轮被错误分类样本的权重**，使后续学习器更关注"难"样本：

**算法流程：**

| 步骤 | 操作 |
|:---|:---|
| ① 初始化 | 所有样本权重相同：$w_i = 1/N$ |
| ② 训练弱学习器 | 在当前权重分布下训练一个基学习器 $G_t(x)$ |
| ③ 计算误差 | $e_t = \sum w_i \cdot \mathbb{1}[G_t(x_i) \neq y_i]$（加权错误率） |
| ④ 计算权重 | $\alpha_t = \frac{1}{2} \ln\frac{1-e_t}{e_t}$（学习器在最终投票中的权重） |
| ⑤ 更新样本权重 | $w_i \leftarrow w_i \cdot \exp(-\alpha_t y_i G_t(x_i))$，然后归一化 |
| ⑥ 重复 | 重复 ②-⑤ 共 $T$ 次 |
| ⑦ 最终模型 | $G(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t G_t(x)\right)$ |

> **直观理解**：AdaBoost 是一个"纠错"过程——后面的学习器**专门学习前面学不好的样本**。学习器权重 $\alpha_t$ 越大，说明它在最终决策中的"话语权"越大。

#### 3.2 AdaBoost 实战

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# 定义弱学习器（max_depth=5 的 CART 决策树）
weak_clf = DecisionTreeClassifier(max_depth=5)

# 构建 AdaBoost（sklearn 1.4+ API）
clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),  # 弱学习器
    n_estimators=300,                                 # 迭代轮数
    learning_rate=0.4                                 # 学习率（权重缩减）
).fit(Xtrain, Ytrain)

# 预测与评估
y_pred = clf.predict(Xtest)
print(accuracy_score(Ytest, y_pred))

# 查看特征重要性
clf.feature_importances_

# 查看每轮弱学习器的投票权重 α_t
clf.estimator_weights_
```

#### 3.3 关键参数速查

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `estimator` | `DecisionTreeClassifier(max_depth=1)` | 基学习器（sklearn 1.4+ 改名，原 `base_estimator`） |
| `n_estimators` | `50` | 弱学习器数量（迭代轮数） |
| `learning_rate` | `1.0` | 学习率/步长，缩小每轮的贡献以防止过拟合 |
| `random_state` | `None` | 随机种子 |

> **sklearn 1.4+ API 变更**：
> - `base_estimator` → `estimator`
> - `algorithm` 参数已移除（自动选择：基学习器支持 `predict_proba` 时用 SAMME.R）
>
> SAMME.R（Stagewise Additive Modeling using a Multi-class Exponential loss function — Real）是 SAMME 的概率改进版，收敛更快。

#### 3.4 弱学习器 vs AdaBoost 对比

Notebook 使用 iris 数据集进行对比：

```python
# 弱学习器单独预测
weak_clf = DecisionTreeClassifier(max_depth=5).fit(Xtrain, Ytrain)
dtc_y_pred = weak_clf.predict(Xtest)
print('CART分类树预测的准确率：{}'.format(
    accuracy_score(Ytest, dtc_y_pred)))

# AdaBoost 集成预测
clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=300,
    learning_rate=0.4
).fit(Xtrain, Ytrain)
clf_y_pred = clf.predict(Xtest)
print('Adaboost自适应增强分类模型预测的准确率：{}'.format(
    accuracy_score(Ytest, clf_y_pred)))
```

> **结果**：AdaBoost 在 iris 数据集上通常能达到 95%+ 的准确率。

---

### 第四章：随机森林 vs AdaBoost 全面对比

| 对比维度 | 随机森林 (Bagging) | AdaBoost (Boosting) |
|:---|:---|:---|
| **训练方式** | 并行独立训练 | 串行依赖训练 |
| **弱学习器** | 完全生长的大树（高方差、低偏差） | 浅树/树桩（高偏差、低方差） |
| **样本权重** | Bootstrap 等权采样 | 动态调整，错分样本权重大 |
| **学习器权重** | 等权投票 | 加权投票（$\alpha_t$） |
| **核心降低** | **方差**（Variance） | **偏差**（Bias） |
| **过拟合风险** | 低（树越多越稳定） | 中（学习率小时较低） |
| **对噪声敏感度** | **低**（Bootstrap 稀释噪声） | **高**（噪声样本权重持续增大） |
| **并行能力** | 天然可并行 | 不可并行（依赖前一轮结果） |
| **可解释性** | 弱（黑箱） | 弱（黑箱），但有权重向量 |
| **特征选择** | `feature_importances_` | `feature_importances_` |
| **OOB 估计** | 支持 | 不支持 |

---

### 第五章：算法选择决策流程

```
                    开始
                     │
                     ▼
            ┌ 数据量很大（≥10万）？ ─┐
            │ 是                      │ 否
            ▼                         ▼
      随机森林              ┌ 数据噪声较多？ ─┐
     （并行训练快）          │ 是              │ 否
                            ▼                 ▼
                      随机森林          ┌ 追求极致精度？ ─┐
                    （对噪声鲁棒）      │ 是              │ 否
                                       ▼                 ▼
                                  AdaBoost          ┌ 需要概率输出？ ─┐
                                （需小心调参）      │ 是              │ 否
                                                    ▼                 ▼
                                              随机森林          随机森林
                                            （有概率 + OOB）   （万能默认）
```

> **一般经验**：**默认先用随机森林**（鲁棒、快、不需要精细调参），如果数据干净且追求最优精度，再尝试 AdaBoost 或 GBDT/XGBoost。

---

### 第六章：Notebook 实战总结

| 步骤 | 内容 | 数据集 | 核心方法 |
|:---|:---|:---|:---|
| ① 导入 | 导包与配置 | — | `InteractiveShell.ast_node_interactivity = "all"` |
| ② 数据准备 | 加载 wine 数据集 | Wine (178×13) | `load_wine()` |
| ③ 单树 vs 森林 | 训练对比 | Wine | `DecisionTreeClassifier` + `RandomForestClassifier` |
| ④ 交叉验证 1 | 10 折 CV 对比 | Wine | `cross_val_score(cv=10)` |
| ⑤ 交叉验证 2 | 10 组 × 10 折 CV | Wine | `cross_val_score` 循环 10 次取均值 |
| ⑥ 学习曲线 | n_estimators 1→200 | Wine | 遍历 + `cross_val_score` |
| ⑦ Bagging 理论 | 二项分布计算 | — | `scipy.special.comb` |
| ⑧ 树多样性 | 查看各树 random_state | Wine | `rfc.estimators_[i].random_state` |
| ⑨ OOB Score | 无测试集评估 | Wine | `RandomForestClassifier(oob_score=True)` |
| ⑩ 误差分析 | 森林误差 vs 单树误差 | — | 二项分布累积概率 |
| ⑪ 回归扩展 | 随机森林回归 | Diabetes | `RandomForestRegressor` + `neg_mean_squared_error` |
| ⑫ AdaBoost | 自适应增强 | Iris | `AdaBoostClassifier(estimator=DecisionTreeClassifier(...))` |

---

## 环境要求

| 依赖包 | 用途 |
|:---|:---|
| Python 3.7+ | 编程语言 |
| numpy | 数值计算 |
| pandas | 数据处理 |
| matplotlib | 可视化 |
| scikit-learn | RandomForest、AdaBoost、交叉验证、数据集 |
| scipy | 组合数计算（`comb`） |

安装命令：

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

---

## 使用方法

1. 启动 Jupyter Notebook：
   ```bash
   jupyter notebook
   ```
2. 在浏览器中打开 `12th_EnsembleLearning.ipynb`
3. 按顺序运行每个代码单元格（`Shift + Enter`）
4. 观察随机森林 vs 单决策树的交叉验证对比图
5. 运行学习曲线，找出 wine 数据集的最佳 `n_estimators`
6. 对比 AdaBoost 与单树的准确率提升

---

## 学习建议

1. **先理解 Bagging 数学**：手算一遍 Bootstrap 概率（为何约 63.2% 的样本被抽中），这是随机森林的基石
2. **调试 n_estimators**：修改学习曲线循环范围（如 1→500），观察收益递减现象
3. **尝试 max_features**：对比 `'sqrt'` / `'log2'` / `None` 的效果差异
4. **OOB 实践**：用 `oob_score=True` 替代手动 `train_test_split`，理解包外估计的优雅之处
5. **手动推演 AdaBoost**：对一个简单数据集（3-5 个样本），手动计算一轮 AdaBoost 的权重更新过程
6. **对比基学习器深度**：修改 AdaBoost 中 `max_depth=1` vs `max_depth=5`，观察集成效果的差异
7. **横向对比**：在同一数据集上用随机森林和 AdaBoost 分别建模，记录精度和训练时间

---

## 常见问题

### Q1: 随机森林的树越多越好吗？

**不是。** n_estimators 存在收益递减：

| 树数量 | 效果 |
|:---|:---|
| 1-50 | 性能快速提升 |
| 50-200 | 提升趋缓 |
| 200+ | 基本饱和，只增加计算成本 |

> **建议**：先用 100 棵树快速试，然后通过交叉验证学习曲线确定适合当前数据集的最佳值（通常在 50-200 之间）。

### Q2: 随机森林需要剪枝吗？

**不需要。** 随机森林的每棵树应**完全生长（不剪枝）**：
- 单棵树的过拟合被森林的**集成投票**天然抵消
- 完全生长的树偏差更低，有利于随机森林整体性能
- Bootstrap 采样和特征随机选择已经提供了足够的正则化

### Q3: 随机森林和 AdaBoost 各适合什么场景？

| 场景 | 推荐 |
|:---|:---|
| 数据量大（≥10万） | **随机森林**（并行训练快） |
| 数据噪声较多 | **随机森林**（对噪声鲁棒） |
| 数据干净、追求极致精度 | **AdaBoost / GBDT / XGBoost** |
| 需要快速 baseline | **随机森林**（默认参数即可） |
| 特征维度很高 | 两者都可以（都有 `feature_importances_`） |
| 需要可解释性 | 单决策树 > 随机森林 ≈ AdaBoost |

### Q4: OOB Score 可靠吗？

**可靠。** OOB Score 已被证明是泛化误差的无偏估计，与相同大小的测试集效果相当。但注意：
- OOB Score 基于**训练数据**内部的包外样本，不能替代真正的**分布外（Out-of-Distribution）测试**
- 如果数据存在时间序列或分组结构，仍需手动按时间/组划分测试集

### Q5: AdaBoost 对异常值为什么敏感？

AdaBoost 会**指数级增大**被错分样本的权重。异常值/噪声样本因为"难以分类"，其权重会在多轮迭代后变得极大，导致模型过度拟合噪声。

> **应对**：增大 `learning_rate`（如 0.5-1.0），或选择对噪声更鲁棒的基学习器。

### Q6: sklearn 1.4+ 中 AdaBoost 的 API 变化是什么？

```python
# ❌ sklearn < 1.2（旧版）
clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=5),
    algorithm='SAMME.R'
)

# ✅ sklearn 1.4+（新版）
clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=5)  # 改名
    # algorithm 参数已移除，自动推断
)
```

### Q7: 随机森林能用于特征选择吗？

**可以。** `feature_importances_` 属性给出每个特征的重要性分数（基于基尼不纯度减少或均方误差减少）。

```python
rfc = RandomForestClassifier().fit(X, y)
importances = rfc.feature_importances_

# 可视化
plt.bar(range(len(importances)), importances)
```

---

**最后更新**：2026-06-20
