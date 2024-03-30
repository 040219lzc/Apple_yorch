import torch
import torch.nn as nn
import torch.nn.functional as F

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # 定义全连接层
        self.fc1 = nn.Linear(128 * 56 * 56, 1024)  # 注意输入维度需要根据实际情况调整
        self.fc2 = nn.Linear(1024, 10)  # 假设输出类别数为10

    def forward(self, x):
        # 第一个卷积层
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 最大池化层
        # 第二个卷积层
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 最大池化层
        # 扁平化操作
        x = torch.flatten(x, 1)  # 将特征图展平为向量
        # 第一个全连接层
        x = F.relu(self.fc1(x))
        # 第二个全连接层
        x = self.fc2(x)
        return x
