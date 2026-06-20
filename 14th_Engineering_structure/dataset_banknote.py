import torch
from torch.utils.data import DataLoader
from config import HP
import numpy as np

class BanknoteDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.dataset = np.loadtxt(data_path, delimiter=',') # 指定数据集路径加载数据集

    # 重载方法
    def __getitem__(self, idx):
        item = self.dataset[idx] # 获取一条数据
        x, y = item[:HP.in_features], item[HP.in_features:] # 切片出数据和标签
        return torch.Tensor(x).float().to(HP.device), torch.Tensor(y).squeeze().long().to(HP.device) # 类型转换，构建张量，也可以方便显示

    # 重载方法
    def __len__(self):
        return self.dataset.shape[0] # 返回数据集的样本个数

# 简单测试一下
if __name__ == '__main__':
    bkdataset = BanknoteDataset(HP.testset_path) # 使用测试集测试一下
    bkdataloader = DataLoader(bkdataset, batch_size=16, shuffle=True, drop_last=True) # 测试集数据加载进来

    for batch in bkdataloader:
        x, y = batch
        print(x)
        print(y)
        break