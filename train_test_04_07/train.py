import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from torch.utils.data import random_split
from tqdm import tqdm
from datasets import load_from_disk
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
#设置可以使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

class cat_and_dog(nn.Module):
    def __init__(self):
        super(cat_and_dog, self).__init__()
        self.model = Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2,stride = 2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2 , stride = 2),
            Flatten(),
            Linear(16384,64),
            Linear(64,2),
        )
    def forward(self,x):
        x = self.model(x)
        return x


class cat_dog(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = os.listdir(root_dir)
        self.labels = [1 if "class1" in img else 0 for img in self.imgs]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.imgs[idx])).convert("RGB")#强制转换成RGB格式三通道
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

class cat_dog_undeduplication(Dataset):
    def __init__(self, path, transform=None):
        self.dataset = load_from_disk(path)
        self.transform = transform
        self.imgs = self.dataset['train']["image"]
        self.labels = self.dataset['train']["labels"]
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image = self.imgs[idx].convert("RGB")#强制转换成RGB格式三通道
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

#dataset
full_dataset_deduplication = cat_dog("deduped_images", transform=transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor(),]))
full_dataset = cat_dog_undeduplication("data/cats_vs_dogs", transform=transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor(),]))

def run_training(dataset_name, full_dataset, epochs=30):
    """使用给定数据集运行完整的训练流程并返回准确率历史"""
    
    # 分割数据集
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型和优化器
    model = cat_and_dog().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # 训练记录
    train_accuracies = []
    test_accuracies = []
    
    # 训练循环
    for t in range(epochs):
        print(f"[{dataset_name}] Epoch {t+1}/{epochs}")
        
        # 训练
        model.train()
        correct_train = 0
        total_samples = len(train_loader.dataset)
        
        for images, labels in tqdm(train_loader, desc=f"{dataset_name} epoch", leave=False):
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss = loss_fn(pred, labels)
            correct_train += (pred.argmax(1) == labels).type(torch.float).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 计算训练准确率
        train_accuracy = correct_train / total_samples
        train_accuracies.append(train_accuracy)
        
        # 测试
        model.eval()
        test_loss, correct_test = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images)
                test_loss += loss_fn(pred, labels).item()
                correct_test += (pred.argmax(1) == labels).type(torch.float).sum().item()
        
        # 计算测试准确率
        test_accuracy = correct_test / len(test_loader.dataset)
        test_accuracies.append(test_accuracy)
        
        # 输出当前结果
        print(f"  训练准确率: {(100*train_accuracy):>0.1f}%, 测试准确率: {(100*test_accuracy):>0.1f}%")
    
    # 最终测试结果
    print(f"\n[{dataset_name}] 最终测试准确率: {(100*test_accuracies[-1]):>0.1f}%")
    
    # 保存最终模型
    torch.save(model.state_dict(), f"{dataset_name}_model.pth")
    
    return {
        'train_acc': train_accuracies,
        'test_acc': test_accuracies,
        'name': dataset_name
    }

deduped_results = run_training("deduped", full_dataset_deduplication, epochs=100)
undeduped_results = run_training("undeduped", full_dataset, epochs=100)

epochs = 100
# 绘制训练准确率对比图
plt.figure(figsize=(12, 5))

# 子图1: 训练准确率对比
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), undeduped_results['train_acc'], 'b-', label='未去重训练准确率')
plt.plot(range(1, epochs+1), deduped_results['train_acc'], 'r-', label='去重训练准确率')
plt.xlabel('Epochs')
plt.ylabel('准确率')
plt.title('训练准确率对比')
plt.legend()
plt.grid(True)

# 子图2: 测试准确率对比
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), undeduped_results['test_acc'], 'b-', label='未去重测试准确率')
plt.plot(range(1, epochs+1), deduped_results['test_acc'], 'r-', label='去重测试准确率')
plt.xlabel('Epochs')
plt.ylabel('准确率')
plt.title('测试准确率对比')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300)
plt.show()
#保存模型
#torch.save(model.state_dict(), "cat_and_dog_model.pth")
#writer.close()