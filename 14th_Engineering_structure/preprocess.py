import numpy as np
from config import HP
import os

trainset_ratio = 0.7 # 训练集数据比例
devset_ratio = 0.2 # 验证集数据比例
testset_ratio = 0.1 # 测试集数据比例

np.random.seed(HP.seed) # 设置随机种子
dataset = np.loadtxt(HP.data_path, delimiter=',') # 指定路径下加载数据集
np.random.shuffle(dataset) # 数据集重新随机排列

n_items = dataset.shape[0] # 获取样本数量

trainset_num = int(trainset_ratio*n_items) # 指定训练集样本数量
devset_num = int(devset_ratio*n_items) # 指定验证集样本数量
testset_num = n_items - trainset_num - devset_num # 指定测试集样本数量
np.savetxt(os.path.join(HP.data_dir, 'train.txt'), dataset[:trainset_num], delimiter=',') # 生成训练集
np.savetxt(os.path.join(HP.data_dir, 'dev.txt'), dataset[trainset_num:trainset_num+devset_num], delimiter=',') # 生成验证集
np.savetxt(os.path.join(HP.data_dir, 'test.txt'), dataset[trainset_num+devset_num:], delimiter=',') # 生成测试集