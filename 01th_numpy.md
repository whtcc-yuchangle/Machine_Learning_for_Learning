# NumPy 基础教程

## 目录
- [数组创建](#数组创建)
- [数组属性](#数组属性)
- [数组运算](#数组运算)
- [数组索引与切片](#数组索引与切片)
- [数组重塑](#数组重塑)

---

## 数组创建

### 1. 从列表创建数组
```python
import numpy as np
lists = [i for i in range(10)]
new_arr = np.array(lists)
# 输出: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### 2. 创建全1数组
```python
ones_arr = np.ones(10)
# 输出: array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
```

### 3. 创建全0数组
```python
zeros_arr = np.zeros(10)
# 输出: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
```

### 4. 创建指定值的数组
```python
arr3 = np.full(shape=[5,6], fill_value=3.14)
# 创建一个5行6列的数组，所有元素都为3.14
```

### 5. 创建等差数列（指定步长）
```python
arr4 = np.arange(start=0, stop=20, step=2)
# 输出: array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
```

### 6. 创建等差数列（指定元素个数）
```python
arr5 = np.linspace(start=0, stop=15, num=4)
# 输出: array([ 0.,  5., 10., 15.])
```

### 7. 创建随机整数数组
```python
arr6 = np.random.randint(1, 50, size=4)
# 在1-50之间随机生成4个整数
```

### 8. 创建随机浮点数数组（0-1之间）
```python
arr8 = np.random.random(size=6)
# 生成6个0-1之间的随机浮点数
```

### 9. 创建标准正态分布随机数
```python
# 一维数组
arr7 = np.random.randn(3)
# 生成3个标准正态分布的随机数

# 二维数组
arr7 = np.random.randn(3, 3)
# 生成3x3的标准正态分布随机数矩阵

# 三维数组
arr7 = np.random.randn(3, 3, 3)
# 生成3x3x3的标准正态分布随机数张量

# 四维数组
arr7 = np.random.randn(3, 3, 3, 3)
# 生成3x3x3x3的标准正态分布随机数四维张量
```

### 10. 指定数据类型创建数组
```python
arr = np.array([1, 2, 3, 4, 5], dtype='float32')
# 输出: array([1., 2., 3., 4., 5.], dtype=float32)
```

### 11. 使用asarray创建数组
```python
arr = np.asarray([1, 2, 3, 4, 5], dtype='float32')
# 与np.array类似，但如果输入已经是数组，则不会创建副本
```

---

## 数组属性

### 1. 数组维度 (ndim)
```python
arr = np.random.randint(0, 100, size=(3, 4, 5))
arr.ndim  # 输出: 3（三维数组）
```

### 2. 数组形状 (shape)
```python
arr = np.random.randint(0, 100, size=(3, 4, 5))
arr.shape  # 输出: (3, 4, 5)
```

### 3. 数组数据类型 (dtype)
```python
arr = np.random.randint(0, 100, size=(3, 4, 5))
print(arr.dtype)  # 输出: int32
print(type(arr.dtype))  # 输出: <class 'numpy.dtypes.Int32DType'>
```

### 4. 数组元素总数 (size)
```python
arr = np.random.randint(0, 100, size=(3, 4, 5))
arr.size  # 输出: 60 (3×4×5=60)
```

### 5. 单个元素字节大小 (itemsize)
```python
arr = np.random.randint(0, 100, size=(3, 4, 5))
arr.itemsize  # 输出: 4（每个int32占4字节）
```

### 6. 数组内存信息 (flags)
```python
arr = np.random.randint(0, 100, size=(3, 4, 5))
arr.flags  # 返回数组的内存布局信息
print(type(arr.flags))  # <class 'numpy._core.multiarray.flagsobj'>
```

---

## 数组运算

### 数组间矩阵运算
```python
arr1 = np.random.randint(0, 100, size=(2, 2))
arr2 = np.random.randint(0, 100, size=(2, 2))

arr1 + arr2  # 加法
arr1 - arr2  # 减法
arr1 * arr2  # 乘法（对应元素相乘）
arr1 / arr2  # 除法
arr1 ** arr2  # 幂运算
arr1 // arr2  # 整除
arr1 % arr2  # 取余
```

---

## 数组索引与切片

### 一维数组切片
```python
arr1 = np.random.randint(0, 100, size=10)

# 基本切片语法: [start:stop:step]
arr1[1:7:2]  # 索引1开始，到索引7前一位结束，步长为2
arr1[:]      # 没有开头没有结尾，步长默认 - 浅拷贝当前数组对象
arr1[::-1]   # 没有开头没有结尾，步长-1 - 一维数组反序
```

### 二维数组切片
```python
arr2 = np.random.randint(0, 100, size=(8, 8))

# 多维切片语法: [dim1_start:dim1_stop:dim1_step, dim2_start:dim2_stop:dim2_step]
arr2[1:7:2, 2:4:1]  # 第一维：索引1开始，索引7前一位结束，步长2；第二维：索引2开始，索引4前一位结束，步长1
arr2[:, :]          # 第一第二维均没有开头没有结尾，步长默认 - 浅拷贝当前数组对象
arr2[::-1, ::-1]    # 第一第二维均没有开头没有结尾，步长-1 - 二维数组反序
```

### 三维数组切片
```python
arr3 = np.random.randint(0, 100, size=(8, 8, 8))

# 三维切片语法: [dim1:dim1:dim1, dim2:dim2:dim2, dim3:dim3:dim3]
arr3[1:7:2, 2:4:1, 3:8:3]  # 第一维：索引1开始，索引7前一位结束，步长2；第二维：索引2开始，索引4前一位结束，步长1；第三维：索引3开始，索引8前一位结束，步长3
arr3[:, :, :]              # 第一第二维第三维均没有开头没有结尾，步长默认 - 浅拷贝当前数组对象
arr3[::-1, ::-1, ::-1]     # 第一第二维第三维均没有开头没有结尾，步长-1 - 三维数组反序
```

---

## 数组重塑

### reshape方法
```python
# 准备一个三维数组
arr = np.random.randint(0, 10, size=(3, 4, 5))
# 原始形状: (3, 4, 5)，共60个元素

# 重塑为二维数组
arr2 = arr.reshape((12, 5))
# 新形状: (12, 5)，仍然是60个元素
```

**注意事项：**
- reshape后的元素总数必须与原数组相同
- reshape返回的是视图（view），不是副本，修改会影响原数组
- 可以使用-1让NumPy自动计算某个维度的大小

---

## 总结

本教程涵盖了NumPy的核心基础操作：
1. **数组创建**：多种方式创建不同类型和维度的数组
2. **数组属性**：了解数组的基本特征和内存信息
3. **数组运算**：支持元素级的数学运算
4. **索引切片**：灵活的多维数组访问方式
5. **数组重塑**：改变数组形状而不改变数据

NumPy是Python科学计算的基础库，掌握这些基础知识对于后续学习机器学习、深度学习等领域至关重要。