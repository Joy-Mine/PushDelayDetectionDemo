# author: baiCai
# 1. 导入所需要的包
import torch
import math
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

# 2. 构建block: 不含有1*1
class Base_Block(nn.Module):
    # 用于扩充的变量，表示扩大几倍
    expansion = 1
    def __init__(self,in_planes,out_planes,stride=1,downsample=None):
        '''
        :param in_planes: 输入的通道数
        :param planes: 输出的通道数
        :param stride: 默认步长
        :param downsample: 是否进行下采样
        '''
        super(Base_Block, self).__init__()
        # 定义网络结构 + 初始化参数
        self.conv1 = nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) # inplace为True表示直接改变原始参数值
        self.conv2 = nn.Conv2d(out_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        # 前向传播
        res = x  #  残差
        # 正常传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 判断是否下采样
        if self.downsample is not None:
            res = self.downsample(res)
        # 相加
        out += res
        # 返回结果
        out = self.relu(out)
        return out

# 3. 构建Block：含有1*1
class Senior_Block(nn.Module):
    expansion = 4
    def __init__(self,in_planes,planes,stride=1,downsample=None):
        '''
        :param in_planes: 输入通道数
        :param planes: 中间通道数，最终的输出通道数还需要乘以扩大系数，即expansion
        :param stride: 步长
        :param downsample: 下采样方法
        '''
        super(Senior_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        # 残差
        res = x
        # 前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        # 是否下采样
        if self.downsample is not None:
            res = self.downsample(x)
        # 相加
        out += res
        out = self.relu(out)
        return out

# 4. 构建输出层
class Output_Block(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        '''
        :param in_planes: 输入通道数
        :param planes:  中间通道数
        :param stride: 步长
        :param block_type: block类型，为A表示不需要下采样，为B则需要
        '''
        super(Output_Block, self).__init__()
        # 定义卷积
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        # 判断是否需要下采样，相比于普通的判断方式，多了一个block类型
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # 前向传播
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # 相加与下采样
        out += self.downsample(x)
        out = F.relu(out)
        return out

# 5. 构建ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers):
        '''
        :param block:  即基本的Block块对象
        :param layers:  指的是创建的何种ResNet，以及其对应的各个层的个数，比如ResNet50，传入的就是[3, 4, 6, 3]
        '''
        super(ResNet, self).__init__()
        # 最开始的通道数，为64
        self.inplanes = 64
        # 最开始大家都用到的卷积层和池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 开始定义不同的block块
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 不要忘记我们最后定义的output_block
        self.layer5 = self._make_out_layer(in_channels=2048)
        # 接上最后的卷积层即可，将输出变为13个通道数，shape为7*7*13
        self.avgpool = nn.AvgPool2d(2)  # kernel_size = 2  , stride = 2
        self.conv_end = nn.Conv2d(256, 13, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(13)
        # 进行参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 根据传入的layer个数和block创建
    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        :param block: Block对象
        :param planes:  输入的通道数
        :param blocks: 即需要搭建多少个一样的块
        :param stride: 步长
        '''
        # 初始化下采样变量
        downsample = None
        # 判断是否需要进行下采样，即根据步长或者输入与输出通道数是否匹配
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # 如果需要下采样，目的肯定是残差和输出可以加在一起
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        # 开始创建
        layers = []
        # 第一个block需要特别处理：
        # 比如第一个，传入的channel为512，但是最终的输出为256，那么是需要下采样的
        # 但是对于第二个block块，传入的肯定是第一个的输出即256，而最终输出也为256，因此不需要下采样
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            # 重复指定次数
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)  # *表示解码，即将列表解开为一个个对象

    # 输出层的构建
    def _make_out_layer(self, in_channels):
        layers = []
        # 根据需求，构建出类似与block的即可
        layers.append(Output_Block(in_planes=in_channels, planes=256, block_type='B'))
        layers.append(Output_Block(in_planes=256, planes=256, block_type='A'))
        layers.append(Output_Block(in_planes=256, planes=256, block_type='A'))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 经历共有的卷积和池化层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 经历各个block块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # 经历最终的输出
        x = self.avgpool(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        x = F.sigmoid(x)  # 归一化到0-1
        # 将输出构建为正确的shape
        x = x.permute(0, 2, 3, 1)  # (-1,7,7,13)
        return x

# 6. 构建不同的ResNet函数
# 预训练下载链接
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# 构建ResNet18
def resnet18(pretrained=False, **kwargs):
    model = ResNet(Base_Block, [2, 2, 2, 2], **kwargs)
    # 是否预训练
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
# 构建ResNet34
def resnet34(pretrained=False, **kwargs):
    model = ResNet(Base_Block, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model
# 构建ResNet50
def resnet50(pretrained=False, **kwargs):
    model = ResNet(Senior_Block, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
# 构建ResNet101
def resnet101(pretrained=False, **kwargs):
    model = ResNet(Senior_Block, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model
# 构建ResNet152
def resnet152(pretrained=False, **kwargs):
    model = ResNet(Senior_Block, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model