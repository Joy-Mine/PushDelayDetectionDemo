# author: baiCai
# 1. 导入所需的包
import warnings

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import datetime

from stock_predict import decoder, calculate_map

warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import models

from network.stock_ResNet import resnet50
# from utils.Yolo_Loss import Yolo_Loss
# from utils.My_Dataset import Yolo_Dataset
from utils.stock_Yolo_Loss import Yolo_Loss
from utils.stock_Dataset import Yolo_Dataset


def plot_loss_curves(train_losses, val_losses):
    """
    Plot train loss and validation loss curve.

    Parameter:
    - train_losses: train losses history
    - val_losses: validation losses history
    """
    plt.figure(figsize=(6, 4), dpi=200)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Epochs')
    plt.grid(True)
    plt.savefig("loss.jpg")
    plt.show()


# 2. 定义基本参数
# 获取当前时间
current_time = datetime.datetime.now()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 4  # 根据自己的电脑设定
epochs = 150
lr = 0.01
# file_root = 'VOC2012/JPEGImages/'   # 需要根据的实际路径修改
file_root = 'dataset_midterm/Dataset_stock/JPEGImages_train_2024_4_6/'  # 需要根据的实际路径修改
# 3. 创建模型并继承预训练参数
pytorch_resnet = models.resnet50(pretrained=True)  # 官方的resnet50预训练模型
model = resnet50()  # 创建自己的resnet50，
# 接下来就是让自己的模型去继承官方的权重参数
pytorch_state_dict = pytorch_resnet.state_dict()
model_state_dict = model.state_dict()
for k in pytorch_state_dict.keys():
    # 调试: 看看模型哪些有没有问题
    # print(k)
    # 如果自己的模型和官方的模型key相同，并且不是fc层，则继承过来
    if k in model_state_dict.keys() and not k.startswith('fc'):
        model_state_dict[k] = pytorch_state_dict[k]
# 4. 损失函数，优化器，并将模型、损失函数放入GPU中
loss = Yolo_Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
model.to(device)
loss.to(device)
# 5. 加载数据
# train_dataset = Yolo_Dataset(root=file_root,list_file='./utils/voctrain.txt',train=True,transforms = [T.ToTensor()])
train_dataset = Yolo_Dataset(root=file_root, list_file='./utils/stocktrain.txt', train=True, transforms=[T.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# test_dataset = Yolo_Dataset(root=file_root,list_file='./utils/voctest.txt',train=False,transforms = [T.ToTensor()])
test_dataset = Yolo_Dataset(root=file_root, list_file='./utils/stocktest.txt', train=False, transforms=[T.ToTensor()])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# 6. 训练
# 打印一些基本的信息
print('starting train the model')
print('the train_dataset has %d images' % len(train_dataset))
print('the batch_size is ', batch_size)
# 定义一个最佳损失值
best_test_loss = 0
# 定义一个列表，用于存放训练和验证的损失值
train_losses = []
val_losses = []
log_writer = SummaryWriter()
best_mAP = 0
best_epoch = 0
# 开始训练
for e in range(epochs):
    model.train()
    # 调整学习率
    if e == 50:
        print('change the lr')
        optimizer.param_groups[0]['lr'] /= 10
    if e == 100:
        print('change the lr')
        optimizer.param_groups[0]['lr'] /= 10
    # 进度条显示
    tqdm_tarin = tqdm(train_loader)
    # 定义损失变量
    total_loss = 0.
    for i, (images, target) in enumerate(tqdm_tarin):
        # 将变量放入设备中
        images, target = images.to(device), target.to(device)
        # 训练--损失等
        pred = model(images)
        loss_value = loss(pred, target)
        total_loss += loss_value.item()
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        # 打印一下损失值
        if (i + 1) % 5 == 0:
            tqdm_tarin.desc = 'train epoch[{}/{}] loss:{:.6f}'.format(e + 1, epochs, total_loss / (i + 1))
    # 在每个训练周期结束后添加平均训练损失
    train_losses.append(total_loss / len(train_loader))
    # 添加训练损失到tensorboard
    log_writer.add_scalar('train_loss', total_loss / len(train_loader), e)

    # 启用验证模式
    model.eval()
    validation_loss = 0.0
    tqdm_test = tqdm(test_loader)
    for i, (images, target) in enumerate(tqdm_test):
        images, target = images.cuda(), target.cuda()
        pred = model(images)
        loss_value = loss(pred, target)
        validation_loss += loss_value.item()

    validation_loss /= len(test_loader)
    # 在验证阶段结束后添加平均验证损失
    val_losses.append(validation_loss)
    # 添加验证损失到tensorboard
    log_writer.add_scalar('validation_loss', validation_loss, e)
    # 显示验证集的损失值
    print('In the test step,the average loss is %.6f' % validation_loss)
    # 计算mAP
    mAP = calculate_map(model, dataset_path='dataset_midterm/Dataset_stock/JPEGImages_train_2024_4_6/', annotation_path='utils/stocktest.txt')
    print(f'Epoch {e + 1}/{epochs} - Validation mAP: {mAP}')
    # Optional: Save the best model based on mAP
    if best_mAP < mAP:
        best_mAP = mAP
        best_epoch = e
        filename = current_time.strftime('./save_weights/weight_%b%d_%H-%M-%S_best_mAP.pth')
# # 画出损失值的变化
# plot_loss_curves(train_losses, val_losses)
# 格式化时间字符串为 "月份缩写日期_小时-分钟-秒" 格式
filename = current_time.strftime('./save_weights/weight_%b%d_%H-%M-%S.pth')
print('best mAP:', best_mAP, 'best epoch:', best_epoch)
# 保存模型权重
torch.save(model.state_dict(), filename)
