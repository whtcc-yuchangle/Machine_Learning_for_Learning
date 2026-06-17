# Pandas 基础教程

## 概述

本 notebook 详细介绍了 Python 数据分析库 Pandas 的核心功能和使用方法。Pandas 是数据科学领域最常用的工具之一，提供了高效的数据结构（Series 和 DataFrame）以及强大的数据分析能力。本教程通过大量实战代码示例，帮助学习者系统掌握 Pandas 的各项功能。

---

## 学习目标

通过学习本教程，您将掌握：

1. **Series 数据结构**：创建、索引、操作和缺失值处理
2. **DataFrame 数据结构**：多种方式创建、查看和基本操作
3. **数据读写**：CSV 和 Excel 文件的读取与写入
4. **数据索引与切片**：loc、iloc 等多种索引方式
5. **数据筛选**：布尔索引和条件筛选
6. **数据赋值**：修改和添加数据
7. **实战项目**：爬虫数据采集与 Pandas 结合应用

---

## 内容结构

### 第一章：Series 数据结构

#### 1.1 基础概念

Series 是 Pandas 中的一维带标签数组，可以存储任意数据类型（整数、字符串、浮点数、Python 对象等）。

#### 1.2 创建 Series

**方法一：使用列表创建**
```python
import pandas as pd

l = [1, 2, 3, 4, 5]
s1 = pd.Series(data=l)
```

**方法二：指定索引创建**
```python
s2 = pd.Series(data=l, index=list('ABCDE'), dtype='float32')
```

**方法三：使用字典创建**
```python
dicts = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
s3 = pd.Series(data=dicts, name='pandas_demo')
```

#### 1.3 缺失值处理

Pandas 使用 `NaN`（Not a Number）表示缺失值，支持 NumPy 的 `np.nan` 和 Python 的 `None`：
```python
import numpy as np

l = [0, 1, 7, 9, np.nan, None, 1024, 1024, 512]
s4 = pd.Series(data=l)
```

---

### 第二章：DataFrame 数据结构

#### 2.1 基础概念

DataFrame 是 Pandas 中的二维表格数据结构，类似于 Excel 表格或 SQL 表，包含行索引和列索引。

#### 2.2 创建 DataFrame

**方法一：使用列表创建**
```python
data = [
    ['laoan', 30],
    ['laoli', 29],
    ['lijianbing', 39],
    ['hupeng', 20],
    ['luotianyun', 23]
]
df = pd.DataFrame(data, columns=['name', 'age'])
```

**方法二：使用 NumPy 数组创建**
```python
import numpy as np

df = pd.DataFrame(
    data=np.random.randint(60, 100, (6, 4)),
    dtype='int32',
    columns=['Python', 'Java', 'C++', 'Ruby'],
    index=['laoan', 'laoli', 'lijianbing', 'luotiayun', 'hupeng', 'huxiaodong']
)
```

**方法三：使用字典创建（方式一）**
```python
df = pd.DataFrame(data={
    'Site': ['Google', 'Runoob', 'Wiki'], 
    'Age': [10, 12, 13]
})
```

**方法四：使用字典创建（方式二）**
```python
df = pd.DataFrame([
    {'a': 1, 'b': 2},
    {'a': 5, 'b': 10, 'c': 20}
])
```

#### 2.3 查看 DataFrame 基本信息

| 方法 | 功能说明 |
| :--- | :--- |
| `df.head(n)` | 显示前 n 行数据（默认 5 行） |
| `df.tail(n)` | 显示后 n 行数据（默认 5 行） |
| `df.shape` | 返回数据形状（行数, 列数） |
| `df.dtypes` | 返回各列数据类型 |
| `df.index` | 返回行索引 |
| `df.columns` | 返回列索引 |
| `df.values` | 返回数据的二维 NumPy 数组 |
| `df.describe()` | 数值型列的汇总统计（计数、均值、标准差、最小/最大值、四分位数） |
| `df.info()` | 列索引、数据类型、非空计数和内存信息 |

---

### 第三章：数据读写操作

#### 3.1 CSV 文件操作

**写入 CSV 文件**
```python
df = pd.DataFrame(data=np.random.randint(0, 151, size=(150, 3)), columns=['Python', 'Math', 'En'])
df.to_csv('./02th_grade.csv', sep=',', header=True, index=True)
```

**读取 CSV 文件**
```python
df = pd.read_csv('./02th_dongchediData.csv', sep=',')
```

#### 3.2 Excel 文件操作

**写入 Excel 文件（单工作表）**
```python
df1 = pd.DataFrame(data=np.random.randint(0, 50, size=[50, 5]), columns=['IT', '化工', '生物', '教师', '士兵'])
df1.to_excel(excel_writer='./02th_salary.xlsx', sheet_name='salary', header=True, index=True)
```

**读取 Excel 文件**
```python
df = pd.read_excel('./02th_salary.xlsx', sheet_name=0, header=0)
```

**写入多个工作表**
```python
df1 = pd.DataFrame(data=np.random.randint(0, 50, size=[50, 5]), columns=['IT', '化工', '生物', '教师', '士兵'])
df2 = pd.DataFrame(data=np.random.randint(0, 50, size=[150, 3]), columns=['Python', 'Tensorflow', 'Keras'])

with pd.ExcelWriter('./02th_data.xlsx') as writer:
    df1.to_excel(writer, sheet_name='salary', index=False)
    df2.to_excel(writer, sheet_name='score', index=False)
```

**读取指定工作表**
```python
df = pd.read_excel('./02th_data.xlsx', sheet_name='score')
```

---

### 第四章：数据索引与访问

#### 4.1 列访问

**按列名获取整列**
```python
df['Python']  # 返回 Series
df.Python     # 同样效果
```

**获取多列**
```python
df[['Python', 'Keras']]  # 返回 DataFrame
```

#### 4.2 行访问

**使用 loc（按标签索引）**
```python
df.loc['A']          # 获取单行
df.loc['A':'C']      # 行切片（包含末尾）
df.loc['A':'F':2]    # 行切片，步长为 2
df.loc['A', 'Python']  # 获取指定行列的标量值
```

**使用 iloc（按位置索引）**
```python
df.iloc[0]           # 获取第一行
df.iloc[0:3]         # 行位置切片（不包含末尾）
df.iloc[0, 0]        # 获取第一行第一列的标量值
df.iloc[:, 0]        # 所有行，第一列
df.iloc[0, :]        # 第一行，所有列
```

---

### 第五章：数据筛选

#### 5.1 布尔索引

**单条件筛选**
```python
df[df['Python'] > 100]  # Python 成绩大于 100 分
```

**多条件筛选**
```python
df[(df['Python'] > 50) & (df['Keras'] > 50)]  # 两个条件都满足
```

**检查行是否存在**
```python
df.index.isin(['A', 'C', 'F'])  # 返回布尔数组
```

---

### 第六章：数据赋值与修改

#### 6.1 添加新列
```python
s = pd.Series([1, 2, 3, 4, 5], index=df.index)
df['new_column'] = s
```

#### 6.2 修改数据

**按标签赋值**
```python
df.loc['A', 'Python'] = 99
```

**按位置赋值**
```python
df.iloc[0, 0] = 99
```

**整列重新赋值**
```python
df['Python'] = df['Python'] * 1.1  # 所有 Python 成绩乘以 1.1
```

---

### 第七章：实战案例：爬虫数据采集与处理

本教程包含一个完整的实战案例，展示如何结合爬虫和 Pandas 进行数据采集与处理。

#### 7.1 案例概述

**目标**：采集懂车帝车辆评分数据并保存到 CSV 文件

**技术栈**：
- `requests`：发送 HTTP 请求
- `lxml`：解析 HTML
- `pandas`：数据处理和存储

#### 7.2 核心代码结构

```python
import requests
import pandas as pd
import numpy as np
from lxml import etree

class DongchediSpider(object):
    def __init__(self):
        self.__headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...'
        }
    
    def __get_requests(self, url='', headers=None, method='GET', form_data=''):
        # 发送请求
    
    def __parse_data(self, data):
        # 解析 HTML，提取数据
    
    def __cleaner_data(self, data):
        # 数据清洗
    
    def getDataFrame(self, data):
        # 转换为 DataFrame
    
    def write_to_csv(self, dataframes):
        # 写入 CSV
    
    def display(self):
        # 主入口
```

#### 7.3 数据字段说明

| 字段 | 说明 |
| :--- | :--- |
| 作者id | 评论作者用户名 |
| 总评分 | 车辆总评分 |
| 外观评分 | 外观评分 |
| 内饰评分 | 内饰评分 |
| 配置评分 | 配置评分 |
| 空间评分 | 空间评分 |
| 舒适性评分 | 舒适性评分 |
| 操控评分 | 操控评分 |
| 动力评分 | 动力评分 |

---

## 配套数据文件

| 文件名称 | 说明 | 生成方式 |
| :--- | :--- | :--- |
| `02th_grade.csv` | 学生成绩数据（150行×3列） | Notebook 第10单元格生成 |
| `02th_dongchediData.csv` | 懂车帝车辆评分数据 | 爬虫案例生成 |
| `02th_salary.xlsx` | 薪资数据（50行×5列） | Notebook 第13单元格生成 |
| `02th_data.xlsx` | 多工作表数据文件 | Notebook 第15单元格生成 |

---

## 环境要求

| 依赖包 | 版本要求 | 用途 |
| :--- | :--- | :--- |
| Python | 3.11+ | 编程语言 |
| pandas | 1.5+ | 数据分析 |
| numpy | - | 数值计算 |
| openpyxl | - | Excel 文件读写 |
| requests | - | HTTP 请求 |
| lxml | - | HTML 解析 |

---

## 安装依赖

```bash
pip install pandas numpy openpyxl requests lxml
```

---

## 使用方法

1. **启动 Jupyter Notebook**：
   ```bash
   jupyter notebook
   ```

2. **打开文件**：在浏览器中打开 `02th_pandas.ipynb`

3. **运行代码**：按顺序运行每个代码单元格（Shift+Enter）

---

## 学习建议

1. **先修知识**：建议先学习 `01th_numpy.ipynb`，掌握 NumPy 基础
2. **实践为主**：动手运行每个代码示例，观察输出结果
3. **理解核心概念**：重点理解 Series 和 DataFrame 的区别与联系
4. **掌握索引方式**：熟练使用 `[]`、`loc`、`iloc` 三种索引方式
5. **拓展练习**：尝试修改爬虫案例中的目标网站和数据字段

---

## Notebook 单元格结构参考

| 单元格序号 | 内容主题 | 知识点 |
| :--- | :--- | :--- |
| 1-4 | Series 创建 | 列表、字典、索引、缺失值 |
| 5-8 | DataFrame 创建 | 列表、数组、字典多种方式 |
| 9 | DataFrame 查看 | head、tail、shape、info、describe |
| 10 | CSV 写入 | to_csv 方法 |
| 11 | 爬虫实战 | requests、lxml、数据清洗、写入CSV |
| 12 | CSV 读取 | read_csv 方法 |
| 13 | Excel 写入 | to_excel 方法 |
| 14 | Excel 读取 | read_excel 方法 |
| 15-16 | 多工作表操作 | ExcelWriter、指定工作表读取 |
| 17-19 | 列访问 | `[]` 操作符、属性访问 |
| 20-33 | 行索引 | loc、iloc、切片 |
| 34-37 | 条件筛选 | 布尔索引、isin |
| 38-44 | 数据赋值 | 添加列、修改值 |
| ... | 后续章节 | 更多高级操作 |

---

**最后更新**：2026-06-17

**文件版本**：v1.0
