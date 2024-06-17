# author: baiCai
# 1. 导包
from xml.etree import ElementTree as ET
import os

# 2. 定义一些基本的参数
# 定义所有的类名
STOCK_CLASSES = ('报价', '成交额', '成交量')


# 3. 定义解析xml文件的函数
def parse_rec(filename):
    # 参数：输入xml文件名
    # 创建xml对象
    tree = ET.parse(filename)
    objects = []
    # 迭代读取xml文件中的object节点，即物体信息
    for obj in tree.findall('object'):
        obj_struct = {}
        # difficult属性，即这里不需要那些难判断的对象
        difficult = int(obj.find('difficult').text)
        if difficult == 1:  # 若为1则跳过本次循环
            continue
        # 开始收集信息
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)

    return objects


# 4. 把信息保存入文件中
def write_txt(train_lists, test_lists, train_Annotations, test_Annotations, train_set, test_set):
    # # 生成训练集txt
    count = 0
    for train_list in train_lists:
        count += 1
        # 获取图片名字
        image_name = train_list.split('.')[0] + '.jpg'  # 图片文件名
        # 对他进行解析
        results = parse_rec(train_Annotations + train_list)
        # 如果返回的对象为空，表示张图片难以检测，因此直接跳过
        if len(results) == 0:
            print(train_list)
            continue
        # 否则，则写入文件中
        # 先写入文件名字
        train_set.write(image_name)
        # 接着指定下面写入的格式
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = STOCK_CLASSES.index(class_name)
            train_set.write(' ' + str(bbox[0]) +
                            ' ' + str(bbox[1]) +
                            ' ' + str(bbox[2]) +
                            ' ' + str(bbox[3]) +
                            ' ' + str(class_name))
        train_set.write('\n')
    train_set.close()
    # 生成测试集txt
    # 原理同上面
    for test_list in test_lists:
        count += 1
        image_name = test_list.split('.')[0] + '.jpg'  # 图片文件名
        results = parse_rec(test_Annotations + test_list)
        if len(results) == 0:
            print(test_list)
            continue
        test_set.write(image_name)
        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = STOCK_CLASSES.index(class_name)
            test_set.write(' ' + str(bbox[0]) +
                           ' ' + str(bbox[1]) +
                           ' ' + str(bbox[2]) +
                           ' ' + str(bbox[3]) +
                           ' ' + str(class_name))
        test_set.write('\n')
    test_set.close()


# 5. 运行
# 主函数：遍历所有数据集并生成相应的文件
def main():
    base_dir = '../datasets_finalTerm'

    # 遍历每个数据集大小配置
    dataset_configs = [
        'train_216_test_54',
        'train_432_test_108',
        'train_864_test_216'
    ]

    for config in dataset_configs:
        base_path = os.path.join(base_dir, config)
        train_file = os.path.join(base_path, 'stocktrain.txt')
        test_file = os.path.join(base_path, 'stocktest.txt')

        # 获取训练集和测试集的xml文件路径
        train_annotations = os.path.join(base_path, 'train', 'labels')
        test_annotations = os.path.join(base_path, 'test', 'labels')

        print("Train Annotations Path:", train_annotations)
        print("Test Annotations Path:", test_annotations)

        # 列出训练集和测试集路径中的文件
        print("Files in Train Annotations:")
        print(os.listdir(train_annotations))

        print("Files in Test Annotations:")
        print(os.listdir(test_annotations))

        # 获取所有的xml文件
        train_lists = [f for f in os.listdir(train_annotations) if f.endswith('.xml')]
        test_lists = [f for f in os.listdir(test_annotations) if f.endswith('.xml')]

        write_txt(train_lists, test_lists, train_annotations + '/', test_annotations + '/', open(train_file, 'w'),
                  open(test_file, 'w'))


if __name__ == '__main__':
    main()

print("数据集文件生成完毕")
