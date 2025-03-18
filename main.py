from compute_pretrained_embeddings import get_embeddings
import numpy as np
import torch
import clip
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset, load_from_disk
import os
import requests
from io import BytesIO

# 获取环境变量中的令牌
token = os.getenv("HUGGINGFACE_TOKEN")

# 指定缓存目录
cache_dir = 'data/laion400m_cache'

# 检查数据集是否已经下载并保存
if not os.path.exists(cache_dir):
    # 加载数据集并保存到本地
    ds = load_dataset("laion/laion400m", use_auth_token=token, cache_dir=cache_dir)
    ds.save_to_disk(cache_dir)
else:
    # 从本地加载数据集
    ds = load_from_disk(cache_dir)

# 选择数据集分割，例如 'train'
ds = ds['train']

device = "cuda" if torch.cuda.is_available() else "cpu"
print(ds[0])
# 加载 CLIP ViT-B/16 模型
model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()

# 自定义数据集
class MyImageDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取图像 URL
        image_url = self.dataset[idx]['url']
        
        try:
            # 下载图像并转换为 RGB 格式
            response = requests.get(image_url)
            response.raise_for_status()  # 检查请求是否成功
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            # 应用预处理
            image = self.transform(image)
            
            return image, image_url, idx
        
        except (requests.RequestException, UnidentifiedImageError) as e:
            print(f"Error downloading or opening image {image_url}: {e}")
            return None



# 使用 CLIP 提供的预处理函数
dataset = MyImageDataset(ds, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 定义存储文件路径和数据格式参数
path_str_type = 'S256'
emb_memory_loc = "embeddings.dat"
paths_memory_loc = "paths.dat"
dataset_size = len(dataset)
emb_size = 512  # CLIP ViT-B/16 的输出嵌入维度为 512

# 创建 memmap 文件存储嵌入与路径
emb_array = np.memmap(emb_memory_loc, dtype='float32', mode='w+', shape=(dataset_size, emb_size))
path_array = np.memmap(paths_memory_loc, dtype=path_str_type, mode='w+', shape=(dataset_size,))

# 调用 get_embeddings 计算并保存嵌入
get_embeddings(model, dataloader, emb_array, path_array)
