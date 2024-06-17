import torch
print(torch.cuda.current_device())

x = torch.tensor([1, 2, 3])
print(x.device)  # 通常输出为'cpu'

x = x.cuda()
print(x.device)  # 输出为'cuda:0' 如果你的机器有GPU且PyTorch配置了CUDA

print(torch.version.cuda)
