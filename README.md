# Machine Learning for Learning

机器学习系统学习项目 — 通过 Jupyter Notebook 实践 + Markdown 详细文档的方式，循序渐进地掌握机器学习的核心概念、算法和工具。

## 项目结构

每个学习主题包含两个互补文件：

| 文件类型 | 用途 |
|:---|:---|
| `.ipynb` | 可运行的 Jupyter Notebook，包含完整代码和输出结果 |
| `.md` | 详细参考文档，包含原理解析、公式推导、API 速查表、最佳实践 |

```
Machine_Learning_for_Learning/
├── 01th_numpy.ipynb              # NumPy 基础教程
├── 01th_numpy.md                 # NumPy 详细文档
├── 02th_pandas.ipynb             # Pandas 数据处理教程
├── 02th_pandas.md                # Pandas 详细文档
├── 02th_grade.csv                # 学生成绩数据（Pandas 练习用）
├── 02th_dongchediData.csv        # 懂车帝车辆评分数据（爬虫实战用）
├── 02th_salary.xlsx              # 薪资数据（单工作表读取练习）
├── 02th_data.xlsx                # 多工作表数据（多 Sheet 读取练习）
├── 03th_matplotlib.ipynb         # Matplotlib 可视化教程
├── 03th_matplotlib.md            # Matplotlib 详细文档
├── 03th_img/                     # Matplotlib 生成的示例图片（28张）
├── 04th_KNN.ipynb                # KNN 算法教程
├── 04th_KNN.md                   # KNN 详细文档
├── 05th_DecisionTree.ipynb       # 决策树算法教程
├── 05th_DecisionTree.md          # 决策树详细文档
├── 06th_NaiveBayes.ipynb         # 朴素贝叶斯算法教程
├── 06th_NaiveBayes.md            # 朴素贝叶斯详细文档
├── 07th_LinearRegression.ipynb   # 线性回归算法教程
├── 07th_LinearRegression.md      # 线性回归详细文档
├── 08th_LogisticRegression.ipynb # 逻辑回归算法教程
├── 08th_LogisticRegression.md    # 逻辑回归详细文档
├── 09th_SVM.ipynb                # SVM 支持向量机教程
├── 09th_SVM.md                   # SVM 详细文档
├── 09th_img/                     # SVM 可视化图片
├── 10th_ClusterAlgorithm.ipynb   # 聚类算法教程
├── 10th_ClusterAlgorithm.md      # 聚类算法详细文档
├── 11th_data_preprocess.ipynb    # 数据预处理教程
├── 11th_data_preprocess.md       # 数据预处理详细文档
├── 12th_EnsembleLearning.ipynb   # 集成学习教程
├── 12th_EnsembleLearning.md      # 集成学习详细文档
├── CLAUDE.md                     # Claude Code 开发指引
├── .gitignore
└── README.md
```

## 学习路径

### 第一阶段：基础工具库

| 编号 | 主题 | 核心内容 |
|:---|:---|:---|
| 01th | **NumPy** | 数组创建与操作、数学运算、索引切片、广播机制、线性代数 |

### 第二阶段：数据处理与可视化

| 编号 | 主题 | 核心内容 |
|:---|:---|:---|
| 02th | **Pandas** | Series/DataFrame、CSV/Excel 读写、数据统计与清洗、索引切片、爬虫实战（懂车帝数据采集） |
| 03th | **Matplotlib** | 折线图、散点图、柱状图、直方图、箱线图、饼图、雷达图、3D 图、热力图、子视图布局、样式定制 |
| 11th | **数据预处理** | 缺失值检测与填充（missingno / SimpleImputer）、标准化与归一化（StandardScaler / MinMaxScaler）、特征编码（独热 / 标签 / 序列）、数据分箱（等宽 / 等频 / 决策树分箱）、特征选择（方差过滤 / 卡方检验 / 皮尔逊相关）、PCA 降维与降噪 |

### 第三阶段：机器学习算法

| 编号 | 主题 | 核心内容 |
|:---|:---|:---|
| 04th | **KNN** | 算法原理、欧氏/曼哈顿距离、手动实现 KNN 分类器、sklearn `KNeighborsClassifier`、模型评估、乳腺癌检测实战 |
| 05th | **决策树** | 树形模型原理、基尼系数与信息增益、sklearn `DecisionTreeClassifier`、graphviz 可视化、剪枝与参数调优、红酒分类实战 |
| 06th | **朴素贝叶斯** | 贝叶斯定理、特征条件独立假设、sklearn `GaussianNB`、混淆矩阵、概率预测、手写数字识别实战 |
| 07th | **线性回归** | 最小二乘法、sklearn `LinearRegression`、MSE 与 R² 评估、模拟数据建模、回归系数解读 |
| 08th | **逻辑回归** | Sigmoid 函数与交叉熵、sklearn `LogisticRegression` 多分类（OvR）、PyTorch 手动实现、鸢尾花分类实战 |
| 09th | **SVM** | 最大间隔分类、支持向量与核函数、sklearn `SVC`（线性核）、Hinge Loss、PyTorch 手动实现、鸢尾花分类实战 |
| 10th | **聚类算法** | K-Means / K-Medoids / GMM（原型聚类）、AGNES / BIRCH / CURE（层次聚类）、DBSCAN / OPTICS（密度聚类）、谱聚类（图论聚类）、t-SNE 降维可视化、pyclust 实战 |
| 12th | **集成学习** | 随机森林（Bootstrap / OOB / 特征随机选择 / 交叉验证 / 学习曲线）、AdaBoost（样本权重更新 / SAMME.R / 弱学习器加权组合）、Bagging vs Boosting 对比、sklearn 1.4+ API 适配 |

**计划中的内容：**
- Scikit-learn 系统梳理（Pipeline、特征工程、模型选择）
- 集成学习进阶（GBDT、XGBoost）
- 无监督学习进阶（GMM、谱聚类）

### 第四阶段：深度学习（计划中）

- PyTorch 基础与自动微分
- 全连接神经网络
- 卷积神经网络（CNN）
- 循环神经网络（RNN/LSTM）

## 环境要求

- **Python**：3.11+
- **核心依赖**：numpy、pandas、matplotlib、scikit-learn
- **辅助工具**：jupyter、openpyxl、requests、lxml、seaborn、graphviz、torch

```bash
pip install numpy pandas matplotlib jupyter openpyxl requests lxml seaborn scikit-learn graphviz torch
```

> **注意**：05th 决策树可视化需要额外安装 [graphviz 系统软件](https://graphviz.org/download/) 并将其 `bin` 目录加入 PATH 环境变量。

## 使用方法

1. 克隆项目：
   ```bash
   git clone <repo-url>
   cd Machine_Learning_for_Learning
   ```
2. 安装依赖：
   ```bash
   pip install numpy pandas matplotlib jupyter openpyxl requests lxml seaborn scikit-learn torch
   ```
3. 启动 Jupyter Notebook：
   ```bash
   jupyter notebook
   ```
4. 按照 `01th → 02th → 03th → 11th → 04th → 05th → 06th → 07th → 08th → 09th → 10th → 12th` 的编号顺序打开并运行各 notebook（11th 数据预处理建议在进入 ML 算法之前学习）
5. 每个 notebook 配套的 `.md` 文档可供查阅原理详解和 API 速查

> **提示**：如果你使用 Claude Code 打开本项目，`CLAUDE.md` 文件包含了本项目的开发指引和约定。

---

**最后更新**：2026-06-20

> 注：12th 编号虽大于 11th，但其内容属于第三阶段（ML 算法），建议在学完 01th-11th 后再学习。
