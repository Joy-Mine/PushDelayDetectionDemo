import os
import sys
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import cv2
import numpy as np
import warnings
from network.stock_ResNet import resnet50

warnings.filterwarnings('ignore')

# 类别索引
STOCK_CLASSES = ('quote', 'turnover', 'volume')

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


# 解码函数
def decoder(pred):
    grid_num = 7
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num
    pred = pred.data
    pred = pred.squeeze(0)
    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    mask1 = contain > 0.1
    mask2 = (contain == contain.max())
    mask = (mask1 + mask2).gt(0)
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size
                    box[:2] = box[:2] * cell_size + xy
                    box_xy = torch.FloatTensor(box.size())
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexs.append(torch.tensor([cls_index.item()]))
                        probs.append(contain_prob * max_prob)
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)
        probs = torch.cat(probs, 0)
        cls_indexs = torch.cat(cls_indexs, 0)
    keep = nms(boxes, probs)
    return boxes[keep], cls_indexs[keep], probs[keep]


# NMS处理
def nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        i = order[0]
        keep.append(i)
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


# 预测函数
def predict_single(model, image_path):
    result = []
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: unable to read image at {image_path}")
        sys.exit(1)
    h, w, _ = image.shape
    img = cv2.resize(image, (448, 448))
    mean = (123, 117, 104)
    img = img - np.array(mean, dtype=np.float32)
    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)
    img = Variable(img[None, :, :, :], volatile=True)
    img = img.cuda()
    pred = model(img)
    pred = pred.cpu()
    boxes, cls_indexs, probs = decoder(pred)
    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), STOCK_CLASSES[cls_index], image_path, prob])
    return result


def save_result_image(image_path, result, output_image_path):
    image = cv2.imread(image_path)
    for left_up, right_bottom, class_name, _, prob in result:
        color = Color[STOCK_CLASSES.index(class_name)]
        cv2.rectangle(image, left_up, right_bottom, color, 2)
        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
    cv2.imwrite(output_image_path, image)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python stock_predict.py <model_path> <image_path> <output_image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    output_image_path = sys.argv[3]

    print(model_path)
    print(image_path)
    print(output_image_path)

    model = resnet50()
    print('load model...')

    # Add map_location to load the model on CPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print('load model done!')
    model.eval()

    result = predict_single(model, image_path)
    save_result_image(image_path, result, output_image_path)
    print(f'Result saved to {output_image_path}')
