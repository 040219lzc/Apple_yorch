import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import Tudui  # 导入你的模型类

# 定义测试数据集的转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 加载测试数据集
test_data = CIFAR10(root="../data", train=False, transform=transform, download=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# 加载模型
model = Tudui()
model.load_state_dict(torch.load("tudui_9.pth"))  # 加载模型参数

# 设置模型为评估模式
model.eval()

# 定义类别标签
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 在测试数据集上进行推断
correct = 0
total = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
