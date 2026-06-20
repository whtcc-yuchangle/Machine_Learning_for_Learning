# 深度学习工程化结构 — 纸币真伪分类

## 项目概述

本项目以**"纸币真伪分类"（Banknote Authentication）**为案例，展示一个**标准化的 PyTorch 深度学习工程目录结构**，完整覆盖从原始数据到可部署模型的六个环节。

**核心目标**：理解从 Jupyter Notebook 实验代码 → 模块化工程代码的范式转变。

---

## 数据集

**Banknote Authentication Dataset**（UCI 机器学习仓库）

| 属性 | 值 |
|:---|:---|
| 样本数 | 1372 |
| 特征数 | 4（小波变换提取：方差、偏度、峰度、熵） |
| 类别数 | 2（`0` = 真钞，`1` = 伪钞） |
| 原始文件 | `data/data_banknote_authentication.txt`（CSV 格式，最后一列为标签） |

---

## 目录结构

```
14th_Engineering_structure/
├── config.py              # 超参数配置中心（所有模块的单一配置来源）
├── preprocess.py          # 数据预处理（加载 → 打乱 → 7:2:1 切分 → 保存）
├── dataset_banknote.py    # PyTorch Dataset 封装（读取磁盘 → 切分 x/y → 转 Tensor）
├── model.py               # MLP 模型定义（4→64→128→64→2，ModuleList 动态构建）
├── trainer.py             # 训练主脚本（训练循环 + 验证评估 + checkpoint + 断点续训）
├── Inference.py            # 推理脚本（加载最优模型 → 测试集评估准确率）
├── data/                  # 数据目录（原始数据 + 切分后的 train/dev/test）
└── model_save/            # 模型存档目录（gitignore 忽略，不提交到仓库）
```

---

## 数据流全景

以下按照**一次完整训练+推理的时序**，展示 6 个模块之间的调用关系和数据流向：

```
① preprocess.py                    ② dataset_banknote.py
   ┌──────────┐                        ┌─────────────────┐
   │ 加载原始TXT │  ──切分──→  train.txt │ BanknoteDataset  │
   │ 随机打乱    │            dev.txt   │ (Dataset子类)    │
   │ 7:2:1 保存  │            test.txt  │ __getitem__切片  │
   └──────────┘                        │ 自动转Tensor/gpu │
                                       └────────┬────────┘
                                                │
                 ③ config.py                     │ DataLoader 包装
                 ┌──────────────┐                │
                 │ Hyperparameter │ ←── 所有模块   │
                 │ 类属性 = 超参  │      import    │
                 └──────────────┘                │
                                                 ▼
④ model.py                     ⑤ trainer.py
   ┌──────────────────┐           ┌──────────────────────────────┐
   │ BanknoteClassifier│           │ train()                       │
   │ Model(nn.Module)  │ ──import→ │  ├─ model.to(cuda)            │
   │ ┌────────────────┐│           │  ├─ DataLoader(batch=64)      │
   │ │ ModuleList逐层  ││           │  ├─ Adam(model.parameters())  │
   │ │ for layer in..: ││           │  ├─ CrossEntropyLoss()        │
   │ │   F.relu(...)   ││           │  ├─ for epoch:               │
   │ └────────────────┘│           │  │    for batch:              │
   └──────────────────┘           │  │      zero_grad → forward  │
                                   │  │      → loss → backward    │
                                   │  │      → step → save/log    │
                                   │  └─────────────────────────── │
                                   │  evaluate() 每 verbose_step   │
                                   │  save_checkpoint() 每 save_step│
                                   └──────────────┬───────────────┘
                                                  │ 生成 .pth 文件
                                                  ▼
⑥ Inference.py                   model_save/
   ┌──────────────────┐           ┌──────────────────────┐
   │ torch.load(.pth)  │ ←──────── │ model_93_1400.pth    │
   │ model.load_state_ │           │ (state_dict + epoch  │
   │ model.eval()      │           │  + optimizer_state)  │
   │ 测试集准确率计算    │           └──────────────────────┘
   └──────────────────┘
```

---

## 模块详解（按数据流顺序）

### ① `config.py` — 全局超参数配置（最先被 import）

**职责**：作为整个项目的**单一配置来源（Single Source of Truth）**，集中定义所有超参数。

```python
class Hyperparameter:
    # ===== Data =====
    device = 'cuda'                                    # 计算设备：cuda(GPU) 或 cpu
    data_dir = './data/'                               # 预处理后数据目录
    data_path = './data/data_banknote_authentication.txt'  # 原始数据集路径
    trainset_path = './data/train.txt'                 # 训练集路径
    devset_path = './data/dev.txt'                     # 验证集路径
    testset_path = './data/test.txt'                   # 测试集路径

    # ===== Input/Output =====
    in_features = 4    # 输入特征维度（4个小波特征）
    out_dim = 2        # 输出类别数（真钞/伪钞）
    seed = 1234        # 全局随机种子（保证可复现）

    # ===== Model Architecture =====
    # 列表 [0]=输入层 [1..-2]=隐藏层 [-1]=输出层
    layer_list = [in_features, 64, 128, 64, out_dim]
    #               ↑输入(4)  ↑隐1(64) ↑隐2(128) ↑隐3(64) ↑输出(2)

    # ===== Training =====
    batch_size = 64      # 每批 64 条样本
    init_lr = 1e-3       # 初始学习率 0.001
    epochs = 100         # 训练总轮数
    verbose_step = 10    # 每 10 步打印一次训练日志
    save_step = 200      # 每 200 步保存一次 checkpoint

HP = Hyperparameter()  # 全局单例，其他模块 import HP 直接使用
```

**设计要点**：

| 要点 | 说明 |
|:---|:---|
| **layer_list 的巧思** | 用一个列表定义整个网络结构，`model.py` 通过 `zip(layer_list[:-1], layer_list[1:])` 自动生成相邻层配对：`[(4,64), (64,128), (128,64), (64,2)]` |
| **device 统一管理** | 所有模块通过 `HP.device` 获取设备信息，切换 CPU/GPU 只需改一处 |
| **类属性而非实例属性** | 所有配置是类级别属性，无需实例化即可访问 `HP.batch_size` |
| **末尾单例** | `HP = Hyperparameter()` 创建一个全局可 import 的实例 |

---

### ② `preprocess.py` — 数据预处理（最先执行）

**职责**：将原始 CSV 数据加载、打乱、按比例切分为训练/验证/测试三份，保存到磁盘。

**完整处理流程**：

```
data_banknote_authentication.txt  (1372 行 × 5 列)
        │
        ▼  np.loadtxt(delimiter=',')
   numpy 数组 [1372, 5]
        │
        ▼  np.random.shuffle()
   随机打乱（固定 seed 可复现）
        │
        ▼  按 7:2:1 切片
   ┌──────────┼──────────┐
   ▼          ▼          ▼
train.txt  dev.txt   test.txt
(960条)    (274条)   (138条)
```

```python
# 关键代码
np.random.seed(HP.seed)                          # ① 固定随机种子
dataset = np.loadtxt(HP.data_path, delimiter=',') # ② 加载数据集
np.random.shuffle(dataset)                        # ③ 随机打乱

n_items = dataset.shape[0]                        # ④ 获取总样本数
trainset_num = int(0.7 * n_items)                 # ⑤ 计算各集合大小
devset_num   = int(0.2 * n_items)
testset_num  = n_items - trainset_num - devset_num  # 剩余归测试集

np.savetxt('train.txt', dataset[:trainset_num], delimiter=',')
np.savetxt('dev.txt',   dataset[trainset_num:trainset_num+devset_num], ...)
np.savetxt('test.txt',  dataset[trainset_num+devset_num:], ...)
```

**执行方式**：

```bash
python preprocess.py
```

> **运行时机**：只需运行一次。此脚本**独立于训练流程**，仅在原始数据首次到位或需要重新划分时执行。

---

### ③ `dataset_banknote.py` — 数据加载层（训练/验证/推理时被调用）

**职责**：将磁盘上的 `.txt` 文件封装为 PyTorch 标准的 `Dataset` 对象，供 `DataLoader` 批量加载。

**为什么需要自定义 Dataset？**

PyTorch 的 `DataLoader` 是实现批量化、打乱、多进程加载的核心。它要求数据源必须实现 `__getitem__`（按索引取一条数据）和 `__len__`（返回数据总数）。自定义 `Dataset` 就是实现这两个接口的适配器。

```python
class BanknoteDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        # 一次性将整个 .txt 加载到内存（数据量小，可行）
        self.dataset = np.loadtxt(data_path, delimiter=',')

    def __getitem__(self, idx):
        item = self.dataset[idx]                        # 取第 idx 行
        x, y = item[:HP.in_features], item[HP.in_features:]  # 切片：前4列=特征，第5列起=标签
        return (
            torch.Tensor(x).float().to(HP.device),      # 特征 → float32 → GPU
            torch.Tensor(y).squeeze().long().to(HP.device)  # 标签 → long → 降维 → GPU
        )

    def __len__(self):
        return self.dataset.shape[0]                    # 返回样本总数
```

**与 DataLoader 的配合**：

```python
# trainer.py 中的使用方式
trainset = BanknoteDataset(HP.trainset_path)            # 实例化 Dataset
train_loader = DataLoader(
    trainset,
    batch_size=HP.batch_size,  # 64 —— 每批取 64 条
    shuffle=True,              # 每 epoch 打乱顺序
    drop_last=True             # 丢弃最后不足 batch_size 的余数
)
# 之后就可以 for batch in train_loader: 逐批取数据
```

**数据流细节**：

| 步骤 | 原始值 | 转换后 |
|:---|:---|:---|
| `np.loadtxt` 读入 | `[0.23, -1.5, 3.2, 0.8, 1]` | numpy 数组 `(5,)` |
| 切片 `x, y` | — | `x=[0.23, -1.5, 3.2, 0.8]`, `y=[1]` |
| `torch.Tensor(x).float()` | `[0.23, -1.5, 3.2, 0.8]` | `tensor([0.2300, -1.5000, 3.2000, 0.8000])` |
| `torch.Tensor(y).squeeze().long()` | `[1]` | `tensor(1, dtype=int64)` |
| `.to(HP.device)` | CPU 张量 | GPU (cuda) 张量 |

> **设计考量**：在 `__getitem__` 中直接 `.to(device)` 将数据送到 GPU，避免了在训练循环中每次都要手动 `.to(device)`。但这也意味着数据在加载时就绑定设备——如果要在 CPU 上跑，改 `config.py` 的 `device='cpu'` 即可。

---

### ④ `model.py` — 模型定义（训练和推理时被 import）

**职责**：定义神经网络结构，继承 `nn.Module` 并实现 `forward`。

**网络结构可视化**：

```
输入层(4)    隐藏层1(64)    隐藏层2(128)    隐藏层3(64)    输出层(2)
   x₁ ─╮
   x₂ ──┼──→ [h₁]──ReLU──→ [h₁]──ReLU──→ [h₁]──ReLU──→ [logit₁]
   x₃ ──┤    ×64             ×128            ×64           ×2
   x₄ ─╯                                              [logit₂]
                                                          ↓
                                                    CrossEntropyLoss
                                                    (内置 Softmax)
```

```python
class BanknoteClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 核心：用列表推导式 + zip 动态构建各层
        self.linear_layer = nn.ModuleList([
            nn.Linear(in_features=in_dim, out_features=out_dim)
            for in_dim, out_dim in zip(HP.layer_list[:-1], HP.layer_list[1:])
        ])
        # zip([4,64,128,64], [64,128,64,2]) → 生成4个 Linear 层:
        #   Linear(4, 64)   → 权重[64,4]  + 偏置[64]
        #   Linear(64, 128) → 权重[128,64] + 偏置[128]
        #   Linear(128, 64) → 权重[64,128] + 偏置[64]
        #   Linear(64, 2)   → 权重[2,64]   + 偏置[2]

    def forward(self, input_x):
        for layer in self.linear_layer:
            input_x = layer(input_x)    # 线性变换 Wx + b
            input_x = F.relu(input_x)   # 非线性激活
        return input_x                  # 返回 logits（未归一化概率）
```

**设计要点解析**：

| 要点 | 详细说明 |
|:---|:---|
| **`nn.ModuleList` vs `nn.Sequential`** | `ModuleList` 只是普通 Python 列表的子类（带自动参数注册），需要手动写 `for` 循环调用；`Sequential` 自动串联，但不够灵活。这里用 `ModuleList` + 循环，可以在层间插入任意操作 |
| **`zip(layer_list[:-1], layer_list[1:])`** | 这是将网络结构列表转换为 `(输入维度, 输出维度)` 元组对的标准技巧。修改 `layer_list = [4, 32, 2]` 即刻变为单隐藏层网络 |
| **ReLU（Rectified Linear Unit）** | $f(x) = \max(0, x)$。比 Sigmoid 更快收敛（无梯度饱和问题），是现代 MLP 的默认激活函数 |
| **输出层不加 Softmax** | `nn.CrossEntropyLoss` 内部自动做 `log_softmax + nll_loss`，如果模型输出加了 Softmax 反而会重复计算，导致精度下降 |
| **`__main__` 测试块** | 文件末尾的测试代码（`torch.randn` 造 16 条假数据跑一遍 forward）验证模型结构和设备迁移正确性 |

**参数统计**：

| 层 | 权重参数 | 偏置参数 | 小计 |
|:---|:---|:---|:---|
| `Linear(4, 64)` | 4×64 = 256 | 64 | 320 |
| `Linear(64, 128)` | 64×128 = 8192 | 128 | 8320 |
| `Linear(128, 64)` | 128×64 = 8192 | 64 | 8256 |
| `Linear(64, 2)` | 64×2 = 128 | 2 | 130 |
| **合计** | | | **17,026** |

---

### ⑤ `trainer.py` — 训练主脚本（核心调度器）

**职责**：整合 model + dataset + optimizer + loss function，执行训练循环，并在训练过程中进行验证、日志打印和模型保存。

`trainer.py` 内含**三个函数**，各自承担独立职责：

#### 5.1 `evaluate(model_, devloader, crit)` — 验证评估

```python
def evaluate(model_, devloader, crit):
    model_.eval()                # ① 切换到评估模式
    sum_loss = 0.
    with torch.no_grad():        # ② 关闭梯度计算（节省显存+加速）
        for batch in devloader:
            x, y = batch
            pred = model_(x)     # ③ 前向计算
            loss = crit(pred, y)
            sum_loss += loss.item()
    model_.train()               # ④ 切回训练模式
    return sum_loss / len(devloader)  # ⑤ 返回平均 loss
```

| 步骤 | 为什么这样做 |
|:---|:---|
| `model_.eval()` | 告诉模型"现在是评估"。虽然此模型没有 Dropout/BatchNorm，但这是标准写法，习惯性调用 |
| `torch.no_grad()` | 不记录计算图。验证不需要反向传播，开启后显存占用大幅减少 |
| `model_.train()` 恢复 | 评估完必须切回训练模式，否则后续训练轮次可能异常 |

#### 5.2 `save_checkpoint(model_, epoch_, optm, checkpoint_path)` — 存档保存

```python
def save_checkpoint(model_, epoch_, optm, checkpoint_path):
    save_dict = {
        'epoch': epoch_,                              # 当前轮数
        'model_state_dict': model_.state_dict(),      # 模型所有参数
        'optimizer_state_dict': optm.state_dict()     # 优化器状态（动量等）
    }
    torch.save(save_dict, checkpoint_path)
```

**为什么同时保存 optimizer 状态？**

Adam 优化器内部维护每个参数的**一阶矩（动量）和二阶矩**。如果只保存模型参数而丢弃优化器状态，从 checkpoint 恢复训练时相当于用一个"失忆"的优化器重新开始，收敛轨迹会完全不同。同时保存两者才能实现**真正的断点续训**。

#### 5.3 `train()` — 主训练逻辑

这是最核心的函数，包含完整的训练流水线：

```python
def train():
    # ===== 1. 命令行参数解析 =====
    parser = ArgumentParser()
    parser.add_argument('--c', default=None, type=str,
                        help='checkpoint path for resume training')
    args = parser.parse_args()
    # python trainer.py                      → 从头训练
    # python trainer.py --c model_40_600.pth → 从存档恢复

    # ===== 2. 模型初始化 =====
    model = BanknoteClassificationModel().to(HP.device)
    criterion = nn.CrossEntropyLoss()               # 多分类损失（内置Softmax）
    opt = optim.Adam(model.parameters(), lr=HP.init_lr)  # Adam自适应学习率

    # ===== 3. 数据加载 =====
    trainset = BanknoteDataset(HP.trainset_path)
    devset   = BanknoteDataset(HP.devset_path)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True,  drop_last=True)
    dev_loader   = DataLoader(devset,   batch_size=64, shuffle=True,  drop_last=False)

    # ===== 4. 断点恢复（可选） =====
    start_epoch, step = 0, 0
    if args.c:
        checkpoint = torch.load(args.c)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    # ===== 5. 训练主循环 =====
    model.train()
    for epoch in range(start_epoch, HP.epochs):      # 外循环：轮次
        for batch in train_loader:                    # 内循环：批次
            x, y = batch
            opt.zero_grad()         # ① 梯度清零
            pred = model(x)         # ② 前向计算
            loss = criterion(pred, y)  # ③ 损失计算
            loss.backward()         # ④ 反向传播
            opt.step()              # ⑤ 参数更新

            if not step % HP.verbose_step:   # 每10步评估打印
                eval_loss = evaluate(model, dev_loader, criterion)
            if not step % HP.save_step:      # 每200步保存一次
                save_checkpoint(model, epoch, opt, f'model_{epoch}_{step}.pth')
            step += 1
```

**训练五步法回顾**：

```
opt.zero_grad()        → 清零。防止上一 batch 的梯度累加
pred = model(x)        → 前向。数据流过网络得到预测
loss = criterion(p, y) → 损失。衡量预测与真实标签的差距
loss.backward()        → 反向。自动计算每个参数的 ∂loss/∂w
opt.step()             → 更新。沿梯度负方向更新 w = w - lr * ∂loss/∂w
```

**关键参数选择理由**：

| 参数 | 值 | 为什么 |
|:---|:---|:---|
| 优化器 | `Adam` | 自适应学习率，无需手动调 `lr` 衰减；收敛快于 SGD |
| 损失函数 | `CrossEntropyLoss` | 多分类标准选择；内置 Softmax，数值稳定 |
| `drop_last=True` (训练) | 丢弃最后不足 64 条的数据 | 避免最后一个 batch 样本数变化导致 BN 不稳定 |
| `drop_last=False` (验证) | 保留所有数据 | 验证需要完整评估，不能丢弃样本 |

---

### ⑥ `Inference.py` — 推理脚本（训练完成后执行）

**职责**：加载训练好的最优 checkpoint，在测试集上评估最终准确率。

```python
# 1. 实例化模型（结构与训练时完全一致）
model = BanknoteClassificationModel()

# 2. 加载最优 checkpoint 的参数
checkpoint = torch.load('./model_save/model_93_1400.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(HP.device)

# 3. 加载测试集
testset = BanknoteDataset(HP.testset_path)
test_loader = DataLoader(testset, batch_size=HP.batch_size, shuffle=True)

# 4. 评估模式 + 无梯度推理
model.eval()
total_cnt = 0
correct_cnt = 0
with torch.no_grad():
    for batch in test_loader:
        x, y = batch
        pred = model(x)                                  # [batch, 2] logits
        total_cnt += pred.size(0)                        # 累加本批样本数
        correct_cnt += (torch.argmax(pred, 1) == y).sum()  # 预测正确的数量

print('Acc: %.3f' % (correct_cnt / total_cnt))  # 输出：Acc: 1.000（100%）
```

**推理关键步骤详解**：

| 步骤 | 代码 | 解释 |
|:---|:---|:---|
| 模型实例化 | `BanknoteClassificationModel()` | 必须用与训练**完全相同的结构**（layer_list 一致），否则 `load_state_dict` 会因 shape 不匹配报错 |
| 加载参数 | `model.load_state_dict(checkpoint['model_state_dict'])` | 只加载权重，不加载优化器状态（推理不需要优化器） |
| 评估模式 | `model.eval()` | 固化 BN 统计量、禁用 Dropout |
| 无梯度 | `torch.no_grad()` | 推理不需要反向传播，关闭自动微分以节省显存 |
| 预测解码 | `torch.argmax(pred, 1)` | 取 logits 最大值对应的列索引作为预测类别（dim=1 表示在类别维度上取 argmax） |

---

## 执行流程

```bash
# Step 0：环境准备（首次）
pip install torch numpy

# Step 1：数据预处理（只需一次，生成 train/dev/test 三个文件）
python preprocess.py

# Step 2：训练模型
python trainer.py
# 输出示例：
# Epoch: [0/100],  Train Loss: 0.51234, Dev Loss: 0.49876
# Epoch: [10/100], Train Loss: 0.02345, Dev Loss: 0.01890
# ...

# Step 3（可选）：从中断处恢复训练
python trainer.py --c model_save/model_40_600.pth

# Step 4：推理，在测试集上评估
python Inference.py
# 输出：Acc: 1.000
```

---

## 设计原则

### 1. 配置与代码分离

> `config.py` 是所有超参数的**单一事实来源（Single Source of Truth）**。切换 CPU/GPU、调整学习率、修改网络结构，全部只需改这一个文件。

### 2. 模块单一职责（SRP）

| 模块 | 只负责一件事 | 不负责的事 |
|:---|:---|:---|
| `config.py` | 定义参数 | 不包含任何业务逻辑 |
| `preprocess.py` | 数据切分+保存 | 不参与训练 |
| `dataset_banknote.py` | 数据加载+格式转换 | 不知道模型结构 |
| `model.py` | 定义网络结构 | 不知道数据来源 |
| `trainer.py` | 编排训练流程 | 不包含模型定义细节 |
| `Inference.py` | 加载模型+预测 | 不修改模型参数 |

### 3. 可复现性

```python
# trainer.py 开头固定所有随机性来源
torch.manual_seed(HP.seed)          # PyTorch CPU 随机
torch.cuda.manual_seed(HP.seed)     # PyTorch GPU 随机
random.seed(HP.seed)                # Python 内置 random
np.random.seed(HP.seed)             # NumPy 随机
```

> 固定种子后，同一份代码在同一台机器上多次运行，结果完全一致。

### 4. 断点续训

> `torch.save(save_dict)` 保存的不只是模型权重，还有 epoch 数和优化器状态。下次 `torch.load` + `load_state_dict` 后可以**完美接续**中断的训练，loss 曲线无缝衔接。

---

## Notebook 开发 vs 工程化结构

| 维度 | Jupyter Notebook | 本目录的工程化方式 |
|:---|:---|:---|
| **代码组织** | 按执行顺序线性排列 cell | 按功能角色分 6 个独立 `.py` 文件 |
| **参数管理** | 分散在多个 cell 中，修改易遗漏 | 集中在 `config.py` 一处 |
| **复用性** | 复制粘贴 cell 到新 notebook | `import model` 即可在新任务中复用 |
| **版本 diff** | 含 base64 图片和 cell output，diff 冗长 | 纯 Python 代码，diff 一目了然 |
| **协作开发** | 顺序依赖强，并行开发易冲突 | 模块独立，可分工并行 |
| **生产部署** | 不适合（需 Jupyter 环境 + 手动执行） | 可直接 `python Inference.py` 或封装成 API |
| **实验灵活性** | ⭐⭐⭐⭐⭐ 即时可视化、修改即运行 | ⭐⭐ 需改代码+重新跑 |

> **最佳实践**：在 Notebook 中做探索性实验（调参、可视化），确定有效方案后移植到工程化结构中做正式训练和部署。

---

## 学习建议

1. **先跑通全流程**：按上面的执行流程完整跑一遍，感受 6 个模块如何串起来
2. **改结构看效果**：把 `layer_list` 改为 `[4, 32, 16, 2]`，重新训练，对比准确率变化
3. **模拟中断恢复**：训练到一半 `Ctrl+C`，然后用 `--c` 恢复，观察 loss 是否连续
4. **换优化器实验**：`Adam` → `SGD(lr=0.01)`，感受收敛速度的巨大差异
5. **阅读每个文件**：打开每个 `.py` 文件，逐行理解，遇到不认识的 API 查 PyTorch 文档
6. **套用为模板**：把整个目录复制一份，只改 `config.py` + `dataset` + `model`，用在自己的新任务上

---

## 参考资料

- [UCI Banknote Authentication Dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
- [PyTorch 官方教程 — 模型与数据集](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch 官方教程 — 保存和加载模型](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [前置教程 — 13th 神经网络基础](../13th_Neural_Network_Basics.ipynb)

---

**最后更新**：2026-06-20
