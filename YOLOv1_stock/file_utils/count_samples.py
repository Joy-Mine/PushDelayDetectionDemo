# 示例数据读取和处理脚本
from collections import defaultdict

# 文件路径
file_path = '../utils/stocktest.txt'

# 初始化数据结构
software_samples = defaultdict(int)
software_interfaces = defaultdict(lambda: defaultdict(int))
software_interface_trends = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# 读取文件并处理每行
with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(' ')
        filename = parts[0]

        # 提取基本信息
        name_parts = filename.split('_')
        software_name = name_parts[0]
        interface_type = name_parts[1]  # 'self' 或 'grail'
        trend = name_parts[2]  # 'up' 或 'down'

        # 更新总样本数
        software_samples[software_name] += 1

        # 更新界面类型数量
        software_interfaces[software_name][interface_type] += 1

        # 更新涨跌数量
        software_interface_trends[software_name][interface_type][trend] += 1

# 打印结果
print("股票软件样本统计：")
for software, count in software_samples.items():
    print(f"{software}: {count} / 60 个样本")

print("\n界面类型统计：")
for software, interfaces in software_interfaces.items():
    for interface, count in interfaces.items():
        if interface == 'grail':
            print(f"{software} 的 {interface} 界面: {count} / 20 个样本")
        if interface == 'self':
            print(f"{software} 的 {interface} 界面: {count} / 40 个样本")

print("\n涨跌统计：")
for software, interfaces in software_interface_trends.items():
    for interface, trends in interfaces.items():
        for trend, count in trends.items():
            if interface == 'grail':
                print(f"{software} 的 {interface} 界面 {trend}: {count} / 10 个样本")
            if interface == 'self':
                print(f"{software} 的 {interface} 界面 {trend}: {count} / 20 个样本")
