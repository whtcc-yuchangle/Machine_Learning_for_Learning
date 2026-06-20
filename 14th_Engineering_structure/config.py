# banknote classification config

# 超参配置
# yaml
class Hyperparameter:
    # ################################################################
    #                             Data 数据相关的超参数配置
    # ################################################################
    device = 'cuda'  # cpu 硬件计算类型，要么cpu，要么gpu（cuda）
    data_dir = './data/' # 生成的数据集的目录
    data_path = './data/data_banknote_authentication.txt' # 原始数据集目录
    trainset_path = './data/train.txt' # 训练集数据目录
    devset_path = './data/dev.txt' # 验证集数据目录
    testset_path = './data/test.txt' # 测试集数据目录

    in_features = 4  # input feature dim 输入的特征数量
    out_dim = 2  # output feature dim (classes number) 输出的标签结果数量
    seed = 1234  # random seed 指定的随机种子

    # ################################################################
    #                             Model Structure 模型结构的相关超参数配置
    # ################################################################
    layer_list = [in_features, 64, 128, 64, out_dim] # 多层感知机结构，一个输入层，一个输出层，三个隐藏层，每个隐藏层神经元个数分别为64 128 64
    # ################################################################
    #                             Experiment 训练相关的超参数配置
    # ################################################################
    batch_size = 64 # 一次取64条数据参与训练
    init_lr = 1e-3 # 初始的学习率0.001
    epochs = 100 # 训练100轮
    verbose_step = 10 # 每10步打印一次
    save_step = 200 # 200步保存一次模型


HP = Hyperparameter()
