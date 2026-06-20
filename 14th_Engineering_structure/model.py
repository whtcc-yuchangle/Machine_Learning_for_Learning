import torch
from torch import nn
from torch.nn import functional as F
from config import HP

# 定义一个模型类，继承nn.Module基类，调用基类的初始化方法
class BanknoteClassificationModel(nn.Module):
    def __init__(self):

        # 调用基类的初始化方法，帮助当前派生类完成可复用的前置操作
        super(BanknoteClassificationModel, self).__init__()

        # 定义一个神经网络模型，传入的参数表示神经网络每一层的结构
        self.linear_layer = nn.ModuleList([
            nn.Linear(in_features=in_dim, out_features=out_dim)
            for in_dim, out_dim in zip(HP.layer_list[:-1], HP.layer_list[1:])
        ])

    # 重载fowward函数，完成前向计算
    def forward(self, input_x):
        for layer in self.linear_layer: # for循环逐层执行前向过程
            input_x = layer(input_x) # 逐层前向线性计算
            input_x = F.relu(input_x) # 指定激活函数，这里用的就是relu，赋予非线性能力
        return input_x # 前向计算结果

if __name__ == '__main__':
    model = BanknoteClassificationModel().to(HP.device)
    x = torch.randn(size=(16, HP.in_features)).to(HP.device) # 一次取16条数据，每条数据取4个特征，使用cuda，即GPU计算
    y_pred = model(x) # 完成前向计算，获取output layer
    print(y_pred)
    print(y_pred.size())
