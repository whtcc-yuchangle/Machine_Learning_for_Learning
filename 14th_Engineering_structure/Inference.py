import torch
from torch.utils.data import DataLoader
from dataset_banknote import BanknoteDataset
from model import BanknoteClassificationModel
from config import HP

# new model instance 新建一个模型实例
model = BanknoteClassificationModel()
checkpoint = torch.load('./model_save/model_93_1400.pth') # 因为模型已经在600步左右就收敛了，所以我们加载的模型就是600步这个
model.load_state_dict(checkpoint['model_state_dict']) # 加载模型的参数
model.to(HP.device)
# test set 导入测试集数据
# dev datalader(evaluation)
testset = BanknoteDataset(HP.testset_path)
test_loader = DataLoader(testset, batch_size=HP.batch_size, shuffle=True, drop_last=False)

model.eval()

total_cnt = 0
correct_cnt = 0

with torch.no_grad(): # 不做梯度的反向传播
    for batch in test_loader: # 拿到每一条测试集数据
        x, y = batch # 取数据
        pred = model(x) # 前向计算，获得预测结果
        # print(pred)
        total_cnt += pred.size(0) # 统计测试了多少条数据
        correct_cnt += (torch.argmax(pred, 1) == y).sum() # 正确分类的样本数量

print('Acc: %.3f' % (correct_cnt/total_cnt)) # 输出预测的准确率
