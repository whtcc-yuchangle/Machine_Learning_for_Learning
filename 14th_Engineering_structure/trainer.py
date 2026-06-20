import os
from argparse import ArgumentParser # 导入所需的参数
import torch.optim as optim # 导入优化器
import torch
import random
import numpy as np
import torch.nn as nn # 导入神经网络
from torch.utils.data import DataLoader # 导入数据集加载


from model import BanknoteClassificationModel # 导入我们构建好的模型
from config import HP # 导入配置文件，里面包含超惨
from dataset_banknote import BanknoteDataset # 导入数据集

# seed init: Ensure Reproducible Result 定义初始化随机，seed是刚才准备好的随机种子
torch.manual_seed(HP.seed)
torch.cuda.manual_seed(HP.seed)
random.seed(HP.seed)
np.random.seed(HP.seed)

# 1.定义函数，评估模型在验证集上的表现 - 先声明，后续再实现
def evaluate(model_, devloader, crit):
    model_.eval() # set evaluation flag 模型进入评估验证
    sum_loss = 0. # 总的loss，初值为0
    with torch.no_grad():
        for batch in devloader: # 获取每一条验证集数据
            x, y = batch # 获取每一条验证集数据
            pred = model_(x) # 前向计算，获取预测结果
            loss = crit(pred, y) # 损失函数计算
            sum_loss += loss.item() # 总loss

    model_.train() # back to training mode 返回到训练模式
    return sum_loss / len(devloader)

# 2.定义函数，实现模型每隔多少步要保存一次 - 先声明，后续再实现
def save_checkpoint(model_, epoch_, optm, checkpoint_path):
    save_dict = {
        'epoch': epoch_,
        'model_state_dict': model_.state_dict(),
        'optimizer_state_dict': optm.state_dict()
    }
    if not os.path.exists(os.path.dirname(checkpoint_path)):
        os.makedirs(os.path.dirname(checkpoint_path))
    torch.save(save_dict, checkpoint_path)

# 3.定义函数，先主要实现训练过程
def train():
    parser = ArgumentParser(description="Model Training") # 定义命令行的参数传递
    parser.add_argument(
        '--c', # 传入的参数就是模型的checkpoint，记录点，类似存档的效果
        default=None,
        type=str,
        help='train from scratch or resume training'
    )
    args = parser.parse_args() # 构造好该传参对象

    # new model instance 新建模型的实例
    model = BanknoteClassificationModel()
    model = model.to(HP.device) # 模型送到cuda进行训练

    # loss function (loss.py) 定义损失函数 - 使用的多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # optimizer # 定义adam优化器 - 传入模型和初始的学习率
    opt = optim.Adam(model.parameters(), lr=HP.init_lr)
    # opt = optim.SGD(model.parameters(), lr=HP.init_lr)

    # train dataloader 导入数据集
    trainset = BanknoteDataset(HP.trainset_path) # 导入训练集
    train_loader = DataLoader(trainset, batch_size=HP.batch_size, shuffle=True, drop_last=True)

    # dev datalader(evaluation) 导入验证集，用于模型评估
    devset = BanknoteDataset(HP.devset_path)
    dev_loader = DataLoader(devset, batch_size=HP.batch_size, shuffle=True, drop_last=False)

    start_epoch, step = 0, 0 # 定义开始的轮数和步数，均为0 - 这里需要调的参只有一个，就是step，目的是为了保存得到一个最好的模型，因为当前案例只是展示工程化建模，还没涉及到相关模型
    # 后续的训练日志构建的就是loss与step步数的关系，从最后训练日志的可视化结果中得出最终结论：当步数达到？时，模型loss收敛至0；

    if args.c:# 表示如果模型从某个checkpoint点恢复的话
        checkpoint = torch.load(args.c) # 执行恢复
        model.load_state_dict(checkpoint['model_state_dict']) # 参数更新到model
        opt.load_state_dict(checkpoint['optimizer_state_dict']) # 优化器需要的参数更新到优化器
        start_epoch = checkpoint['epoch'] # 指定当前已到达的轮数
        print('Resume From %s.' % args.c)
    else:
        print('Training From scratch!') # 否则就是从头开始了

    model.train()   # set training flag 模型开始训练

    # main loop 给定一个主循环
    for epoch in range(start_epoch, HP.epochs):
        # print('Start Epoch: %d, Steps: %d' % (epoch, len(train_loader)/HP.batch_size))
        for batch in train_loader:
            x, y = batch    # load data 加载数据
            opt.zero_grad() # gradient clean 梯度归零
            pred = model(x) # forward process 前向计算
            loss = criterion(pred, y)   # loss calc 损失函数计算
            loss.backward() # backward process 反向传播
            opt.step() # 更新模型参数

            if not step % HP.verbose_step:  # evaluate log print
                eval_loss = evaluate(model, dev_loader, criterion)

            if not step % HP.save_step: # model save
                model_path = 'model_%d_%d.pth' % (epoch, step)
                save_checkpoint(model, epoch, opt, os.path.join('model_save', model_path))

            step += 1
        if epoch % HP.verbose_step == 0: # 每隔多少步打印一次日志
            print('Epoch: [%d/%d], Train Loss: %.5f, Dev Loss: %.5f'
                % (epoch, HP.epochs, loss.item(), eval_loss))

if __name__ == '__main__':
    train()
