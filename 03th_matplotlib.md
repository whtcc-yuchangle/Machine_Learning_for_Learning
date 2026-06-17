# Matplotlib 基础教程

## 概述

本 notebook 详细介绍了 Python 数据可视化库 Matplotlib 的核心功能和使用方法。Matplotlib 是 Python 最流行的可视化库之一，能够生成各种静态、动态和交互式的图表。本教程通过丰富的实战示例，帮助学习者系统掌握数据可视化的各项技能。

---

## 学习目标

通过学习本教程，您将掌握：

1. **基础绘图**：创建图表对象、绘制函数图像
2. **图表样式**：颜色、线型、点型、透明度等属性设置
3. **子视图布局**：subplot、axes、gridspec 等多种布局方式
4. **图表元素**：标题、标签、图例、网格、刻度等设置
5. **注释标注**：箭头连接、文本注释、坐标点标注
6. **多种图表类型**：折线图、散点图、柱状图、直方图、箱线图、饼图、雷达图、3D 图、热力图
7. **图片保存**：多种格式导出和高清设置

---

## 内容结构

### 第一章：基础绘图入门

#### 1.1 创建绘图对象

Matplotlib 的绘图通常从创建 `figure`（画布）开始：

```python
import matplotlib.pyplot as plt
import numpy as np

# 创建绘图对象，指定尺寸为 5*3 英寸
plt.figure(figsize=(5, 3))
```

#### 1.2 绘制函数图像

```python
# 定义 x 轴数据
x = np.linspace(0, 2*np.pi)

# 绘制正弦函数
y = np.sin(x)
plt.plot(x, y)

# 在同一个图表中绘制多个函数
m = np.cos(x)
plt.plot(x, m)
```

#### 1.3 设置网格线

```python
plt.grid(
    linestyle='--',  # 网格线样式：'-'实线 '--'虚线 ':'点线
    color='green',   # 网格线颜色
    alpha=0.75       # 透明度（0-1）
)
```

#### 1.4 设置坐标轴范围

```python
plt.axis([-1, 10, -1.5, 1.5])  # [xmin, xmax, ymin, ymax]
```

#### 1.5 保存图片

```python
import os

# 创建保存目录
if not os.path.exists('./03th_img'):
    os.makedirs('./03th_img')

plt.savefig(
    './03th_img/img1.png',  # 文件名：png、jpg、pdf
    dpi=100,                 # 保存图片像素密度
    facecolor='white',       # 视图与边界之间颜色设置
    edgecolor='red',         # 视图边界颜色设置
    bbox_inches='tight'      # 保存图片完整（裁剪多余空白）
)
```

---

### 第二章：图表样式设置

#### 2.1 颜色设置

```python
# 使用颜色名称
plt.plot(x, y, color='indigo')

# 使用十六进制颜色码
plt.plot(x, y, color='#FF00EE')

# 使用 RGB 元组（0-1 之间）
plt.plot(x, y, color=(0.2, 0.7, 0.2))

# 快速颜色代码
plt.plot(x, y, 'b')  # b=blue, r=red, g=green, m=magenta, c=cyan, k=black, w=white
```

#### 2.2 线型设置

```python
# 常用线型
plt.plot(x, y, linestyle='-')   # 实线
plt.plot(x, y, linestyle='--')  # 虚线
plt.plot(x, y, linestyle=':')    # 点线
plt.plot(x, y, linestyle='-.')  # 点划线

# 快速代码
plt.plot(x, y, '--')  # 虚线
```

#### 2.3 点型（标记）设置

```python
# 常用标记样式
plt.plot(x, y, marker='o')   # 圆点
plt.plot(x, y, marker='s')   # 方形
plt.plot(x, y, marker='^')   # 三角形
plt.plot(x, y, marker='*')   # 星形
plt.plot(x, y, marker='p')   # 五边形
plt.plot(x, y, marker='D')   # 菱形

# 标记属性
plt.plot(x, y,
         marker='o',
         markerfacecolor='red',    # 标记填充颜色
         markeredgecolor='green',  # 标记边缘颜色
         markersize=10,             # 标记大小
         markeredgewidth=3          # 标记边缘宽度
        )
```

#### 2.4 线宽和透明度

```python
plt.plot(x, y,
         linewidth=3,   # 线宽
         alpha=0.7      # 透明度（0-1）
        )
```

#### 2.5 快速组合参数

```python
# 格式字符串：[颜色][标记][线型]
plt.plot(x, y, 'bo--')  # 蓝色圆点 + 虚线
plt.plot(x, y, 'r*-')  # 红色星形 + 实线
plt.plot(x, y, 'gs:')   # 绿色方形 + 点线
```

---

### 第三章：函数绘图与刻度

#### 3.1 自定义函数绘图

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义复杂函数
def f(x):
    return np.exp(-x) * np.cos(2*np.pi*x)

x = np.linspace(0, 5, 50)
plt.plot(x, f(x),
         color='purple',
         marker='o',
         ls='--',
         lw=2,
         alpha=0.6,
         markerfacecolor='red',
         markersize=10,
         markeredgecolor='green',
         markeredgewidth=3
        )
```

#### 3.2 设置刻度大小

```python
plt.xticks(size=18)  # 设置 x 轴刻度大小
plt.yticks(size=18)  # 设置 y 轴刻度大小
```

---

### 第四章：子视图布局

子视图（subplot）允许在一个图表中创建多个独立的小图表。

#### 4.1 使用 subplot 创建子视图

**方式一：两行两列布局**

```python
fig = plt.figure(figsize=(5, 3))

# 子视图1：两行两列第一个子视图
ax = plt.subplot(2, 2, 1)
ax.plot(x, y, color='red')
ax.set_facecolor('green')  # 设置子视图背景色

# 子视图2：两行两列第二个子视图
ax = plt.subplot(2, 2, 2)
line, = ax.plot(x, -y)  # 返回绘制对象
line.set_marker('*')
line.set_markerfacecolor('red')
line.set_markeredgecolor('green')
line.set_markersize(10)

# 子视图3：两行一列第二行视图
ax = plt.subplot(2, 1, 2)
plt.sca(ax)  # 设置当前视图
x = np.linspace(-np.pi, np.pi, 200)
plt.plot(x, np.sin(x*x), color='red')
```

**方式二：3x3 网格布局**

```python
fig = plt.figure(figsize=(5, 3))
x = np.linspace(0, 2*np.pi, 50)

# 使用 3 阶方阵返回子视图
for i in range(1, 10):
    ax = plt.subplot(3, 3, i)
    plt.plot(x, np.sin(x + i/3))
    plt.title(f'Plot {i}')

plt.tight_layout()  # 紧凑显示
```

#### 4.2 使用 axes 嵌套子视图

```python
fig = plt.figure(figsize=(5, 3))

# 方式一：使用 plt.axes()
ax = plt.axes([0.2, 0.55, 0.3, 0.3])  # [left, bottom, width, height]
ax.plot(x, y, color='g')

# 方式二：使用 fig.add_axes()
ax = fig.add_axes([0.55, 0.2, 0.3, 0.3])
ax.plot(x, y, color='r')
```

#### 4.3 使用切片方式设置子视图

```python
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

# 通过切片方式访问子视图
ax[0, 0].plot(x, y)
ax[0, 0].set_title('ax1')  # 设置标题
ax[0, 0].set_xlim([0, 10]) # 设置 x 轴范围
ax[0, 0].set_ylim([-2, 2]) # 设置 y 轴范围
ax[0, 0].set_xlabel('X')   # 设置 x 轴标签
ax[0, 0].set_ylabel('Y')   # 设置 y 轴标签
```

#### 4.4 使用 subplot2grid

```python
fig = plt.figure(figsize=(5, 3))

# 使用 subplot2grid 创建不规则布局
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)  # 第一行，前两列
ax2 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)   # 右侧，两行
ax3 = plt.subplot2grid((3, 3), (2, 0), colspan=3) # 底部，一行

ax1.plot(x, y)
ax2.plot(x, -y)
ax3.plot(x, np.sin(2*x))
```

#### 4.5 使用 GridSpec 模块

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(5, 3))

# 将整个视图分成 3x3 布局
gs = gridspec.GridSpec(3, 3)

# 使用切片方式创建子视图
ax1 = plt.subplot(gs[0, :2])  # 第一行，前两列
ax2 = plt.subplot(gs[0, 2])   # 第一行，最后一列
ax3 = plt.subplot(gs[1:, :])  # 后两行
```

---

### 第五章：图表元素设置

#### 5.1 设置标题和标签

```python
# 设置标题
plt.title('My Plot Title', fontsize=16, fontweight='bold')

# 设置坐标轴标签
plt.xlabel('X Axis Label', fontsize=12)
plt.ylabel('Y Axis Label', fontsize=12)

# 设置刻度标签
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['0', 'π/2', 'π', '3π/2', '2π'])
```

#### 5.2 设置字体属性

```python
from matplotlib.font_manager import FontProperties

# 设置字体
font = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=12)

plt.title('中文标题', fontproperties=font)
plt.xlabel('X轴', fontproperties=font)
plt.ylabel('Y轴', fontproperties=font)
```

#### 5.3 设置图例

```python
# 基本图例
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.legend()  # 自动显示图例
plt.legend(loc='upper right')  # 指定位置

# 自定义图例
plt.legend(['sin', 'cos'],
           loc='best',
           frameon=True,
           shadow=True,
           fontsize=10)
```

#### 5.4 设置网格

```python
plt.grid(True)              # 显示网格
plt.grid(color='gray',      # 网格颜色
         linestyle='--',     # 网格线型
         linewidth=1,        # 网格线宽
         alpha=0.5)          # 透明度
```

---

### 第六章：注释和标注

#### 6.1 散点图与文本标注

```python
import numpy as np

# 准备数据
x = np.random.randn(30)
y = np.random.randn(30)
labels = range(30)

# 绘制散点图
plt.scatter(x, y)

# 绘制箭头
for i in range(len(x)):
    plt.annotate(f'{i}',
                  xy=(x[i], y[i]),
                  xytext=(x[i]+0.1, y[i]+0.1),
                  arrowprops=dict(arrowstyle='->', color='red'))

# 添加文本
plt.text(0, 0, 'Origin', fontsize=12, ha='center')
```

#### 6.2 箭头连接样式

```python
# 常用箭头样式
arrow_styles = ['-', '->', '-[', '-|>', '<-', '<->', '<|-', '<|-|']

x = np.linspace(0, 10, 10)
for i, style in enumerate(arrow_styles):
    ax = plt.subplot(4, 2, i+1)
    ax.annotate('Point',
                 xy=(5, 0.5),
                 xytext=(2, 0.8),
                 arrowprops=dict(arrowstyle=style, color='blue'))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.set_title(f"Style: {style}")
```

---

### 第七章：各种图表类型

#### 7.1 折线图（Line Plot）

```python
# 基本折线图
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linewidth=2, markersize=6)

# 多条折线
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='Series 1', marker='o')
plt.plot(x, y2, label='Series 2', marker='s')
plt.plot(x, y3, label='Series 3', marker='^')
plt.legend()
```

#### 7.2 散点图（Scatter Plot）

```python
# 基本散点图
plt.figure(figsize=(8, 6))
plt.scatter(x, y, s=100, c='blue', alpha=0.5, marker='o')

# 气泡图（大小和颜色变化）
plt.scatter(x, y,
            s=sizes,           # 点大小
            c=colors,          # 点颜色
            cmap='viridis',    # 颜色映射
            alpha=0.7)

# 带颜色条的气泡图
plt.colorbar()  # 显示颜色条
```

#### 7.3 柱状图（Bar Plot）

**基本柱状图**

```python
# 数据准备
labels = ['Python', 'Java', 'C++', 'JavaScript']
values = [45, 30, 25, 20]

plt.figure(figsize=(8, 6))
plt.bar(labels, values, color=['blue', 'red', 'green', 'orange'])
plt.xlabel('Programming Language')
plt.ylabel('Popularity Score')
plt.title('Language Popularity')
```

**分组柱状图**

```python
# 数据准备
labels = ['G1', 'G2', 'G3']
men_means = [20, 35, 30]
women_means = [25, 32, 35]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制分组柱状图
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# 设置标签和标题
ax.set_xlabel('Groups')
ax.set_ylabel('Scores')
ax.set_title('Scores by Group and Gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 添加数值标签
for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height}',
                 xy=(rect.get_x() + rect.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom')
```

**堆叠柱状图**

```python
# 数据准备
categories = ['A', 'B', 'C', 'D']
values1 = [10, 20, 30, 40]
values2 = [15, 25, 35, 45]
values3 = [20, 30, 40, 50]

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制堆叠柱状图
ax.bar(categories, values1, label='Group 1')
ax.bar(categories, values2, bottom=values1, label='Group 2')
ax.bar(categories, values3, bottom=[v1+v2 for v1, v2 in zip(values1, values2)], label='Group 3')

ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Stacked Bar Chart')
ax.legend()
```

#### 7.4 直方图（Histogram）

```python
# 数据准备
data = np.random.randn(1000)

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制直方图
n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7)

# 添加概率密度曲线
mu, sigma = 0, 1
pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(bins-mu)**2/(2*sigma**2))
ax.plot(bins, pdf, 'r-', linewidth=2)

ax.set_xlabel('Value')
ax.set_ylabel('Probability Density')
ax.set_title('Histogram with PDF')
```

#### 7.5 箱线图（Box Plot）

```python
# 数据准备
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制箱线图
bp = ax.boxplot(data,
                labels=['Box 1', 'Box 2', 'Box 3'],
                patch_artist=True)

# 设置箱体颜色
colors = ['pink', 'lightblue', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_xlabel('Groups')
ax.set_ylabel('Values')
ax.set_title('Box Plot')
```

#### 7.6 饼图（Pie Chart）

**基本饼图**

```python
# 解决中文字体乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
labels = ['Python', 'Java', 'C++', 'JavaScript']
sizes = [45, 30, 15, 10]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

fig, ax = plt.subplots(figsize=(10, 8))

# 绘制饼图
wedges, texts, autotexts = ax.pie(sizes,
                                    labels=labels,
                                    colors=colors,
                                    autopct='%1.1f%%',  # 显示百分比
                                    shadow=True,        # 显示阴影
                                    startangle=90)       # 起始角度

ax.set_title('Language Distribution')
```

**嵌套饼图**

```python
# 数据准备
# p1: 外部百分比例, p2: 内部百分比例
sizes1 = [15, 30, 45, 10]
sizes2 = [20, 25, 25, 30]

fig, ax = plt.subplots(figsize=(10, 8))

# 绘制内部饼图
wedges1, texts1, autotexts1 = ax.pie(sizes1,
                                       radius=1,
                                       autopct='%1.1f%%',
                                       wedgeprops=dict(width=0.3, edgecolor='w'))

# 绘制外部饼图
wedges2, texts2, autotexts2 = ax.pie(sizes2,
                                       radius=1.3,
                                       autopct='%1.1f%%',
                                       wedgeprops=dict(width=0.3, edgecolor='w'))

# 设置图例
ax.legend(wedges1, ['A', 'B', 'C', 'D'],
          title="Categories",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          frameon=False)
```

#### 7.7 雷达图（Radar Chart）

```python
# 数据准备
categories = ['Math', 'Science', 'English', 'History', 'Art']
values1 = [85, 90, 78, 88, 92]
values2 = [80, 85, 92, 80, 87]

# 将极坐标根据数据长度进行等分
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)

# 使雷达图数据封闭
values1 = np.concatenate((values1, [values1[0]]))
values2 = np.concatenate((values2, [values2[0]]))
angles = np.concatenate((angles, [angles[0]]))

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# 绘制雷达图
ax.plot(angles, values1, 'o-', linewidth=2, label='Student 1')
ax.fill(angles, values1, alpha=0.25)
ax.plot(angles, values2, 'o-', linewidth=2, label='Student 2')
ax.fill(angles, values2, alpha=0.25)

# 设置刻度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

ax.set_title('Student Performance Radar Chart')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
```

#### 7.8 3D 图表

**3D 散点图**

```python
from mpl_toolkits.mplot3d import Axes3D

# 数据准备
n = 100
x = np.random.randn(n)
y = np.random.randn(n)
z = np.random.randn(n)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 散点图
ax.scatter(x, y, z,
           c=z,              # 颜色根据 z 值变化
           cmap='viridis',   # 颜色映射
           s=50,             # 点大小
           alpha=0.6)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Scatter Plot')
```

**3D 曲面图**

```python
from mpl_toolkits.mplot3d import Axes3D

# 数据准备
X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 3D 曲面图
surf = ax.plot_surface(X, Y, Z,
                        cmap='coolwarm',
                        linewidth=0,
                        antialiased=False)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('3D Surface Plot')

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)
```

#### 7.9 热力图（Heatmap）

```python
import seaborn as sns

# 数据准备：每个月 4 周每周都会产生数据
# 三个维度：月、周、销量
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
data = np.random.rand(6, 4) * 100

fig, ax = plt.subplots(figsize=(10, 8))

# 绘制热力图
im = ax.imshow(data, cmap='YlOrRd')

# 设置刻度标签
ax.set_xticks(np.arange(len(weeks)))
ax.set_yticks(np.arange(len(months)))
ax.set_xticklabels(weeks)
ax.set_yticklabels(months)

# 添加数值标签
for i in range(len(months)):
    for j in range(len(weeks)):
        text = ax.text(j, i, f'{data[i, j]:.1f}',
                       ha="center", va="center", color="black")

# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Sales", rotation=-90, va="bottom")

ax.set_title("Monthly Sales Heatmap")
plt.tight_layout()
```

---

### 第八章：高级技巧

#### 8.1 一图多线

```python
fig, ax = plt.subplots(figsize=(10, 6))

# 多个数据集绘制在同一图上
for i in range(5):
    y = np.sin(x + i*0.5)
    ax.plot(x, y, label=f'Line {i+1}')

ax.legend()
ax.set_title('Multiple Lines')
```

#### 8.2 双 Y 轴

```python
fig, ax1 = plt.subplots(figsize=(10, 6))

# 第一个 Y 轴
color = 'tab:red'
ax1.set_xlabel('X')
ax1.set_ylabel('Y1', color=color)
ax1.plot(x, y1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# 第二个 Y 轴
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Y2', color=color)
ax2.plot(x, y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
```

#### 8.3 设置坐标轴样式

```python
# 隐藏坐标轴
plt.axis('off')  # 隐藏所有坐标轴
plt.xticks([])   # 隐藏刻度
plt.yticks([])

# 设置对数坐标
plt.xscale('log')
plt.yscale('log')

# 设置极坐标
ax = plt.subplot(111, projection='polar')
```

---

## 配套文件

| 文件夹 | 说明 |
| :--- | :--- |
| `03th_img/` | 存放教程中生成的所有图片文件 |

| 图片文件 | 对应图表类型 |
| :--- | :--- |
| `img1.png` | 基础正弦余弦函数图 |
| `img2.png` | 颜色线型样式图 |
| `img3.png` | 自定义函数绘图 |
| `img4.png` | 子视图布局 |
| `img5.png` | 嵌套子视图 |
| ... | 更多示例图片 |

---

## 环境要求

| 依赖包 | 版本要求 | 用途 |
| :--- | :--- | :--- |
| Python | 3.7+ | 编程语言 |
| matplotlib | 3.0+ | 数据可视化 |
| numpy | - | 数值计算 |
| pandas | - | 数据处理 |
| seaborn | - | 统计图形（热力图） |

---

## 安装依赖

```bash
pip install matplotlib numpy pandas seaborn
```

---

## 使用方法

1. **启动 Jupyter Notebook**：
   ```bash
   jupyter notebook
   ```

2. **打开文件**：在浏览器中打开 `03th_matplotlib.ipynb`

3. **运行代码**：按顺序运行每个代码单元格（Shift+Enter）

4. **查看图片**：生成的图片将保存在 `03th_img/` 文件夹中

---

## 学习建议

1. **循序渐进**：从基础绘图开始，逐步学习高级图表类型
2. **动手实践**：修改示例代码的参数，观察图表变化
3. **参考文档**：Matplotlib 官方文档提供了丰富的示例
4. **组合学习**：结合 NumPy 和 Pandas 学习数据处理和可视化
5. **颜色选择**：学会使用颜色映射提升图表可读性

---

## Notebook 单元格结构参考

| 单元格序号 | 内容主题 | 知识点 |
| :--- | :--- | :--- |
| 1 | 基础绘图 | figure、正弦余弦、网格、坐标轴、保存图片 |
| 2 | 样式设置 | 颜色、线型、点型、快速参数 |
| 3 | 函数绘图 | 自定义函数、刻度大小 |
| 4 | 子视图布局 | 两行两列、set_facecolor |
| 5 | 嵌套子视图 | axes、add_axes |
| 6 | 3x3 网格 | tight_layout |
| 7 | 切片子视图 | subplots、set_title、xlabel |
| 8 | subplot2grid | 不规则布局 |
| 9 | GridSpec | 复杂布局 |
| 10 | 标题标签 | title、xlabel、ylabel |
| 11 | 字体属性 | FontProperties、中文显示 |
| 12 | 散点图+箭头 | scatter、annotate |
| 13 | 注释标注 | 箭头样式 |
| 14 | 箭头样式 | 多种 arrowstyle |
| 15 | 一图多线 | 循环绑定 |
| 16 | 多图布局 | 子图组合 |
| 17 | 分组柱状图 | bar、图例、注释 |
| 18 | 堆叠柱状图 | bottom 参数 |
| 19 | 折线图 | plot、marker |
| 20 | 折线图验证 | 数据可视化 |
| 21 | 直方图 | hist、概率密度函数 |
| 22 | 箱线图 | boxplot |
| 23 | 散点图 | scatter |
| 24 | 饼图 | pie、中文乱码解决 |
| 25 | 嵌套饼图 | 双层 pie、图例 |
| 26 | 雷达图 | polar 投影 |
| 27 | 3D 散点图 | Axes3D |
| 28 | 热力图 | imshow |

---

## 常见问题

### Q1: 中文显示乱码怎么办？

```python
# 方法一：设置 rcParams
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 方法二：使用 FontProperties
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')
plt.title('中文标题', fontproperties=font)
```

### Q2: 如何导出高清图片？

```python
plt.savefig('output.png', dpi=300, bbox_inches='tight', facecolor='white')
```

### Q3: 如何调整子图间距？

```python
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
# 或使用
plt.tight_layout()  # 自动调整
```

---

**最后更新**：2026-06-17

**文件版本**：v1.0
