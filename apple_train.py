import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn
from model import Tudui  # 假设你已经定义了模型类

# 定义数据集的根目录
root_dir = "C:\\Users\\lenovo\\Desktop\\pytorch-tutorial-master\\pytorch-tutorial-master\\src\\"

# 定义图像预处理操作
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 将图像调整为256x256大小
    transforms.CenterCrop(224),      # 中心裁剪为224x224大小
    transforms.ToTensor(),           # 转换为张量
    transforms.Normalize(            # 标准化图像数据
        mean=[0.485, 0.456, 0.406],   # 均值
        std=[0.229, 0.224, 0.225]     # 方差
    )
])

# 创建数据集对象
dataset = ImageFolder(root=root_dir, transform=transform)

# 创建数据加载器
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_step = 0
total_test_step = 0
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    tudui.train()
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/len(dataset)))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/len(dataset), total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui.state_dict(), "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()
