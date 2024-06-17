# author: baiCai
# 1. 导包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# 2. 损失函数类
class Yolo_Loss(nn.Module):
    def __init__(self,S=7, B=2, l_coord=5, l_noobj=0.5):
        '''
        :param S: Yolov1论文中的S，即划分的网格，默认为7
        :param B: Yolov1论文中的B，即多少个框预测，默认为2
        :param l_coord:  损失函数中的超参数，默认为5
        :param l_noobj:  同上，默认为0.5
        '''
        super(Yolo_Loss, self).__init__()
        # 初始化各个参数
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    # 前向传播
    def forward(self,pred_tensor,target_tensor):
        # 获取batchsize大小
        N = pred_tensor.size()[0]
        # 具有目标标签的索引值，此时shape为[batch,7,7]
        coo_mask = target_tensor[:, :, :, 4] > 0
        # 不具有目标的标签索引值，此时shape为[batch,7,7]
        noo_mask = target_tensor[:, :, :, 4] == 0
        # 将shape变为[batch,7,7,13]
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        # 获取预测值中包含对象的所有点（共7*7个点），并转为[x,13]的形式，其中x表示有多少点中的框包含有对象
        coo_pred = pred_tensor[coo_mask].view(-1, 13)
        # 对上面获取的值进行处理
        # 1. 转为box形式：box[x1,y1,w1,h1,c1]，shape为[2x,5]，因为每个单元格/点有两个预测框
        box_pred = coo_pred[:, :10].contiguous().view(-1, 5)
        # 2. 转为class信息，即13中后面的3个值
        class_pred = coo_pred[:, 10:]
        # 同理，对真实值进行操作，方便对比计算损失值
        coo_target = target_tensor[coo_mask].view(-1, 13)
        box_target = coo_target[:, :10].contiguous().view(-1, 5)
        class_target = coo_target[:, 10:]
        # 同上的操作，获取不包含对象的预测值、真实值
        noo_pred = pred_tensor[noo_mask].view(-1, 13)
        noo_target = target_tensor[noo_mask].view(-1, 13)

        # 不包含物体grid ceil的置信度损失：即图中的D部分
        # 1. 自己创建一个索引
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()).bool()
        noo_pred_mask.zero_() # 将全部元素变为Flase的意思
        # 2. 将其它位置的索引置为0，唯独两个框的置信度位置变为1
        noo_pred_mask[:, 4] = 1
        noo_pred_mask[:, 9] = 1
        # 3. 获取对应的值
        noo_pred_c = noo_pred[noo_pred_mask]  # noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        # 4. 计算损失值：均方误差
        nooobj_loss = F.mse_loss(noo_pred_c, noo_target_c, size_average=False)

        # 计算包含物体的损失值
        # 创建几个全为False/0的变量，用于后期存储值
        coo_response_mask = torch.cuda.ByteTensor(box_target.size()).bool() # 负责预测框
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size()).bool() # 不负责预测的框的索引（因为一个cell两个预测框，而只有IOU最大的负责索引）
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda() # 具体的IOU值存放处
        # 由于一个单元格两个预测框，因此step=2
        for i in range(0, box_target.size()[0], 2):  # choose the best iou box
            # 获取预测值中的两个box
            box1 = box_pred[i:i + 2] # [x,y,w,h,c]
            # 创建一个临时变量，用于存储中左上角+右下角坐标值，因为计算IOU需要
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            # 下面将中心坐标+高宽 转为 左上角+右下角坐标的形式，并归一化
            box1_xyxy[:, :2] = box1[:, :2] / float(self.S) - 0.5 * box1[:, 2:4] # 原本(xc,yc)为7*7 所以要除以7
            box1_xyxy[:, 2:4] = box1[:, :2] / float(self.S) + 0.5 * box1[:, 2:4]
            # 用同样的思路对真实值进行处理，不过不同的是真实值一个对象只有一个框
            box2 = box_target[i].view(-1, 5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:, :2] = box2[:, :2] / float(self.S) - 0.5 * box2[:, 2:4]
            box2_xyxy[:, 2:4] = box2[:, :2] / float(self.S) + 0.5 * box2[:, 2:4]
            # 计算两者的IOU
            iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # 前者shape为[2,4]，后者为[1,4]
            #  获取两者IOU最大的值和索引，因为一个cell有两个预测框，一般而言取IOU最大的作为预测框
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()
            # 将IOU最大的索引设置为1，即表示这个框负责预测
            coo_response_mask[i + max_index] = 1
            # 将不是IOU最大的索引设置为1，即表示这个预测框不负责预测
            coo_not_response_mask[i + 1 - max_index] = 1
            # 获取具体的IOU值
            box_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()
        # 获取负责预测框的值、IOU值和真实框的值
        box_pred_response = box_pred[coo_response_mask].view(-1, 5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1, 5)
        box_target_response = box_target[coo_response_mask].view(-1, 5)
        #  这个对应的是图中的部分C，负责预测框的损失
        contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response_iou[:, 4], size_average=False)
        # 1. 计算坐标损失，即图中的A和B部分
        loc_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], size_average=False) + F.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4]), torch.sqrt(box_target_response[:, 2:4]), size_average=False)
        # 获取不负责预测框的值、真实值
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1, 5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1, 5)
        box_target_not_response[:, 4] = 0 # 将真实值置为0
        # 2. 计算不负责预测框的损失值,即图中的部分C
        not_contain_loss = F.mse_loss(box_pred_not_response[:, 4], box_target_not_response[:, 4], size_average=False)
        # 3. 类别损失，即图中的E部分
        class_loss = F.mse_loss(class_pred, class_target, size_average=False)
        return (self.l_coord * loc_loss +  contain_loss + not_contain_loss + self.l_noobj * nooobj_loss  + class_loss) / N

    # 计算IOU的函数
    def compute_iou(self, box1, box2):
        '''
        :param box1: 预测的box，一般为[2,4]
        :param box2: 真实的box，一般为[1,4]
        :return:
        '''
        # 获取各box个数
        N = box1.size(0)
        M = box2.size(0)
        # 计算两者中左上角左边较大的
        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        # 计算两者右下角左边较小的
        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )
        # 计算两者相交部分的长、宽
        wh = rb - lt  # [N,M,2]
        # 如果长、宽中有小于0的，表示可能没有相交趋于，置为0即可
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        # 计算各个的面积
        # box1的面积
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        # box2的面积
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]
        # IOu值，交集除以并集，其中并集为两者的面积和减去交集部分
        iou = inter / (area1 + area2 - inter)
        return iou