# 1. 导包
import os
import sys
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt
import numpy as np
import warnings

from mAP import mean_average_precision

warnings.filterwarnings('ignore')

from network.stock_ResNet import resnet50

# 2. 定义一些基本的参数
# 类别索引
STOCK_CLASSES = (
    'quote', 'turnover', 'volume')
# 画矩形框的时候用到的颜色变量
Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


# 3. 解码函数
def decoder(pred):
    '''
    :param pred: batchx7x7x13，但是预测的时候一般一张图片一张的放，因此batch=1
    :return: box[[x1,y1,x2,y2]] label[...]
    '''
    # 定义一些基本的参数
    grid_num = 7  # 网格划分标准大小
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num  # 缩放因子
    # 获取一些值
    pred = pred.data  # 预测值的数据：1*7*7*13
    pred = pred.squeeze(0)  # 预测值的数据：7x7x13
    contain1 = pred[:, :, 4].unsqueeze(2)  # 先获取第一个框的置信度，然后升维变为7*7*1
    contain2 = pred[:, :, 9].unsqueeze(2)  # 同上，只是为第二个框
    contain = torch.cat((contain1, contain2), 2)  # 拼接在一起，变为7*7*2
    mask1 = contain > 0.1  # 大于阈值0.1，设置为True
    mask2 = (contain == contain.max())  # 找出置信度最大的，设置为True
    mask = (mask1 + mask2).gt(0)  # 将mask1+mask2，让其中大于0的设置为True
    # 开始迭代每个单元格，即7*7个
    for i in range(grid_num):
        for j in range(grid_num):
            # 迭代两个预测框
            for b in range(2):
                # 如果mask为1，表示这个框是最大的置信度框
                if mask[i, j, b] == 1:
                    # 获取坐标值
                    box = pred[i, j, b * 5:b * 5 + 4]
                    # 获取置信度值
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    # 将7*7的坐标，归一化
                    xy = torch.FloatTensor([j, i]) * cell_size  # cell左上角  up left of cell
                    #
                    box[:2] = box[:2] * cell_size + xy
                    # 将[cx,cy,w,h]转为[x1,xy1,x2,y2]
                    box_xy = torch.FloatTensor(box.size())  # 重新创建一个变量存储值
                    box_xy[:2] = box[:2] - 0.5 * box[2:]  # 这个就是中心坐标加减宽度/高度得到左上角/右下角坐标
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    # 获取最大的概率和类别索引值
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    # 如果置信度 * 类别概率 > 0.1，即说明有一定的可信度
                    # 那么把值加入各个变量列表中
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexs.append(torch.tensor([cls_index.item()]))
                        probs.append(contain_prob * max_prob)
    # 如果boxes为0，表示没有框，返回0
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    # 否则，进行处理，就是简单把原来的列表值[tensor,tensor]转为tensor的形式
    # 里面的值不变
    else:
        boxes = torch.cat(boxes, 0)  # (n,4)
        probs = torch.cat(probs, 0)  # (n,)
        cls_indexs = torch.cat(cls_indexs, 0)  # (n,)
    # 后处理——NMS
    keep = nms(boxes, probs)
    # 返回值
    return boxes[keep], cls_indexs[keep], probs[keep]


# 4. NMS处理
def nms(bboxes, scores, threshold=0.5):
    '''
    :param bboxes:  bboxes(tensor) [N,4]
    :param scores:  scores(tensor) [N,]
    :param threshold: 阈值
    :return: 返回过滤后的框
    '''
    # 获取各个框的坐标值
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    # 计算面积
    areas = (x2 - x1) * (y2 - y1)
    # 将置信度按照降序排序，并获取排序后的各个置信度在这个顺序中的索引
    _, order = scores.sort(0, descending=True)
    keep = []
    # 判断order中的元素个数是否大于0
    while order.numel() > 0:
        # 如果元素个数只剩下一个了，结束循环
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        # 获取最大置信度的索引
        i = order[0]
        keep.append(i)
        # 对后面的元素坐标进行截断处理
        xx1 = x1[order[1:]].clamp(min=x1[i])  # min指的是小于它的设置为它的值，大于它的不管
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        # 此时的xx1，yy1等是排除了目前选中的框的，即假设x1有三个元素，那么xx1只有2个元素
        # 获取排序后的长和宽以及面积，如果小于0则设置为0
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        # 准备更新order、
        # 计算选中的框和剩下框的IOU值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 如果 IOU小于设定的阈值，说明需要保存下来继续筛选（NMS原理）
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


# 5. 预测函数
def predict_single(model, image_path):
    result = []  # 保存结果的变量
    # 打开图片
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    # resize为模型的输入大小，即448*448
    img = cv2.resize(image, (448, 448))
    # 由于我们模型那里定义的颜色模式为RGB，因此这里需要转换
    # 由于我们模型那里定义的颜色模式为RGB，因此这里需要转换
    mean = (123, 117, 104)  # RGB均值
    img = img - np.array(mean, dtype=np.float32)
    # 预处理
    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)
    img = Variable(img[None, :, :, :], volatile=True)
    img = img.cuda()
    # 开始预测
    pred = model(img)  # 1x7x7x13
    pred = pred.cpu()
    # 解码
    boxes, cls_indexs, probs = decoder(pred)
    # 开始迭代每个框
    for i, box in enumerate(boxes):
        # 获取相关坐标，只是需要把原来归一化后的坐标转回去
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        # 获取类别索引、概率等值
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        # 把这些值集中放入一个变量中返回
        result.append([(x1, y1), (x2, y2), STOCK_CLASSES[cls_index], image_path, prob])
    return result

def save_result_image(image_path, result, output_image_path):
    # 打开原始图片
    image = cv2.imread(image_path)
    # 画矩形框和对应的类别信息
    for left_up, right_bottom, class_name, _, prob in result:
        # 获取颜色
        color = Color[STOCK_CLASSES.index(class_name)]
        # 画矩形
        cv2.rectangle(image, left_up, right_bottom, color, 2)
        # 获取类型信息和对应概率，此时为str类型
        label = class_name + str(round(prob, 2))
        # 把类别和概率信息写上，还要为这个信息加上一个矩形框
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
    # 保存结果图片
    cv2.imwrite(output_image_path, image)

if __name__ == '__main__':
    # 从命令行读取参数
    if len(sys.argv) != 4:
        print("Usage: python stock_predict.py <model_path> <image_path> <output_image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    output_image_path = sys.argv[3]

    print(model_path)
    print(image_path)
    print(output_image_path)

    # 创建模型，加载参数
    model = resnet50()
    print('load model...')
    model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print('load model done!')
    model.eval()
    model.cuda()

    # 预测并保存结果图片
    result = predict_single(model, image_path)
    save_result_image(image_path, result, output_image_path)
    print(f'Result saved to {output_image_path}')

