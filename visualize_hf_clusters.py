import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import yaml
from tqdm import tqdm
import argparse
import io

# 设置参数解析
parser = argparse.ArgumentParser(description='可视化聚类结果 (Huggingface Datasets版本)')
parser.add_argument('--cluster-id', type=int, default=None, help='指定要可视化的簇ID')
parser.add_argument('--num-clusters', type=int, default=5, help='要可视化的簇数量')
parser.add_argument('--images-per-cluster', type=int, default=9, help='每个簇显示的图像数量')
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--max-size', type=int, default=224, help='图像显示的最大尺寸')
parser.add_argument('--output-dir', type=str, default='visualization_results', help='输出目录')
parser.add_argument('--dataset-dir', type=str, required=True, help='HuggingFace数据集目录')
parser.add_argument('--image-column', type=str, default='image', help='数据集中的图像列名')
args = parser.parse_args()

# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)

# 加载配置
config_file = "clustering/configs/openclip/clustering_configs.yaml"
with open(config_file, 'r') as f:
    params = yaml.safe_load(f)

# 获取相关路径
sorted_clusters_folder = params['sorted_clusters_file_loc']
save_folder = params['save_folder']

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

# 加载 HuggingFace 数据集
print(f"正在加载 HuggingFace 数据集: {args.dataset_dir}")
try:
    from datasets import load_from_disk, Dataset, DatasetDict
    
    # 尝试从磁盘加载数据集
    print("正在使用HuggingFace datasets库加载数据集...")
    hf_dataset = load_from_disk(args.dataset_dir)
    
    if isinstance(hf_dataset, Dataset):
        print(f"加载了单个数据集，大小: {len(hf_dataset)}")
        dataset_to_use = hf_dataset
    elif isinstance(hf_dataset, DatasetDict):
        print(f"加载了数据集字典，包含以下分割: {list(hf_dataset.keys())}")
        # 默认使用train分割，如果存在的话
        if 'train' in hf_dataset:
            dataset_to_use = hf_dataset['train']
            print(f"使用'train'分割，大小: {len(dataset_to_use)}")
        else:
            # 使用第一个分割
            first_key = list(hf_dataset.keys())[0]
            dataset_to_use = hf_dataset[first_key]
            print(f"使用'{first_key}'分割，大小: {len(dataset_to_use)}")
    else:
        print(f"未知的数据集类型: {type(hf_dataset)}")
        raise ValueError("不支持的数据集类型")
    
    # 打印数据集结构
    features = dataset_to_use.features
    print(f"数据集特征: {features}")
    
    # 找到图像列
    image_column = None
    for column in dataset_to_use.column_names:
        if column == args.image_column:
            image_column = column
            break
        elif 'image' in column.lower():
            image_column = column
            print(f"使用猜测的图像列: '{column}'")
            break
    
    if not image_column:
        print(f"警告: 未找到图像列。可用列: {dataset_to_use.column_names}")
        # 尝试找到PIL.Image类型的列
        for column in dataset_to_use.column_names:
            if hasattr(features[column], 'feature') and features[column].feature._type == 'PIL.Image':
                image_column = column
                print(f"找到图像类型列: '{column}'")
                break
    
    # 显示样例
    print("数据集示例:")
    sample = dataset_to_use[0]
    for key, value in sample.items():
        if key == image_column:
            if hasattr(value, 'size'):
                print(f"  {key}: <图像, 尺寸 {value.size}>")
            else:
                print(f"  {key}: <图像数据>")
        else:
            print(f"  {key}: {value}")
    
    # 构建索引到图像的映射
    index_to_image = {}
    
    # 检查是否有索引列
    index_column = None
    for col in ['index', 'idx', 'id']:
        if col in dataset_to_use.column_names:
            index_column = col
            break
    
    if index_column:
        print(f"使用数据集中的'{index_column}'列作为索引")
        
        # 如果有索引列，使用它来映射
        for i in tqdm(range(len(dataset_to_use)), desc="加载图像索引"):
            sample = dataset_to_use[i]
            idx = str(sample[index_column])
            if image_column and image_column in sample:
                index_to_image[idx] = sample[image_column]
    else:
        print("使用行号作为索引")
        
        # 如果没有索引列，使用行号
        for i in tqdm(range(len(dataset_to_use)), desc="加载图像索引"):
            sample = dataset_to_use[i]
            if image_column and image_column in sample:
                index_to_image[str(i)] = sample[image_column]
    
    print(f"加载了 {len(index_to_image)} 条图像记录")
    
except Exception as e:
    print(f"无法使用HuggingFace datasets加载数据集: {e}")
    print("尝试查看文件的前几个字节以确定格式...")
    
    # 尝试直接读取文件的头部来确定格式
    try:
        arrow_files = [f for f in os.listdir(args.dataset_dir) if f.endswith('.arrow')]
        if arrow_files:
            file_path = os.path.join(args.dataset_dir, arrow_files[0])
            with open(file_path, 'rb') as f:
                header = f.read(16)  # 读取前16字节
                print(f"文件头部字节: {header.hex()}")
                # 如果是Arrow文件，头部应该以'ARROW1'开头
                if b'ARROW1' in header:
                    print("这看起来像是一个Arrow文件，但格式可能不标准")
                elif b'PK\x03\x04' in header:
                    print("这看起来像是一个ZIP文件，而不是Arrow文件")
                elif b'\x89HDF' in header:
                    print("这看起来像是一个HDF5文件，而不是Arrow文件")
    except Exception as e:
        print(f"无法读取文件头部: {e}")
    
    print("继续使用占位图像...")
    index_to_image = {}

# 创建占位图像
def create_placeholder_image(idx, dist, size=(224, 224)):
    """创建一个占位图像，显示索引和距离"""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 240  # 浅灰背景
    
    # 添加文本
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # 使用默认字体
    font = ImageFont.load_default()
    
    # 绘制文本
    text1 = f"ID: {idx}"
    text2 = f"Distance: {dist:.4f}"
    
    draw.text((10, size[1]//2 - 20), text1, fill=(0, 0, 0), font=font)
    draw.text((10, size[1]//2 + 10), text2, fill=(0, 0, 0), font=font)
    
    return pil_img

def load_image_from_arrow(img_data):
    """从数据集中加载图像"""
    if img_data is None:
        return None
    
    try:
        # 如果已经是PIL图像
        if isinstance(img_data, Image.Image):
            return img_data
        
        # 如果是字节数据
        elif isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data))
        
        # 如果是numpy数组
        elif isinstance(img_data, np.ndarray):
            return Image.fromarray(img_data)
        
        else:
            print(f"未知的图像数据类型: {type(img_data)}")
            return None
    
    except Exception as e:
        print(f"无法加载图像数据: {e}")
        return None

# 加载按大小排序的簇
size_sorted_file = os.path.join(save_folder, "clusters_sorted_by_size.txt")
cluster_sizes = []
with open(size_sorted_file, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            cluster_id, size = int(parts[0]), int(parts[1])
            cluster_sizes.append((cluster_id, size))

print(f"加载了 {len(cluster_sizes)} 个簇的大小信息")

# 根据参数选择要可视化的簇
clusters_to_visualize = []
if args.cluster_id is not None:
    # 如果指定了特定簇ID
    for cluster_id, size in cluster_sizes:
        if cluster_id == args.cluster_id:
            clusters_to_visualize.append((cluster_id, size))
            break
    if not clusters_to_visualize:
        print(f"警告: 没有找到ID为 {args.cluster_id} 的簇")
else:
    # 否则选择最大的几个簇
    clusters_to_visualize = cluster_sizes[:args.num_clusters]

if not clusters_to_visualize:
    print("没有找到要可视化的簇，退出")
    exit(1)

print(f"将可视化 {len(clusters_to_visualize)} 个簇")

# 创建图像网格的函数
def create_image_grid(images, cluster_id, cluster_size, output_path):
    """创建图像网格并保存"""
    n = len(images)
    if n == 0:
        print(f"警告: 簇 {cluster_id} 没有有效图像")
        return False
    
    # 确定网格大小
    rows = 3
    cols = 3
    plt.figure(figsize=(15, 15))
    
    for i, (img, idx, dist) in enumerate(images):
        if i >= args.images_per_cluster:
            break
            
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(f"ID: {idx}\nDist: {dist:.4f}", fontsize=10)
        plt.axis('off')
    
    plt.suptitle(f"cluster {cluster_id} (size: {cluster_size})", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(output_path)
    plt.close()
    print(f"保存图像网格到: {output_path}")
    return True

# 检查簇文件格式
def inspect_cluster_file(file_path):
    """检查簇文件的格式，返回列数和样本行"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            parts = first_line.split('\t')
            return len(parts), first_line
    except Exception as e:
        print(f"无法读取簇文件 {file_path}: {e}")
        return 0, ""

# 可视化每个选定的簇
successful_visualizations = []
for cluster_id, cluster_size in tqdm(clusters_to_visualize, desc="可视化簇"):
    # 读取簇文件
    cluster_file = os.path.join(sorted_clusters_folder, f"sorted_cluster_{cluster_id}.txt")
    if not os.path.exists(cluster_file):
        print(f"警告: 找不到簇文件 {cluster_file}")
        continue
    
    # 检查文件格式
    num_cols, sample_line = inspect_cluster_file(cluster_file)
    print(f"簇文件 {cluster_file} 格式: {num_cols} 列, 样本: {sample_line}")
    
    # 读取文件中的图像索引
    image_data = []
    with open(cluster_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            
            # 根据列数解析
            if num_cols >= 3:
                # 标准格式: 索引、标识符、距离
                idx = parts[0]  # 使用第一列作为索引
                dist = float(parts[2])  # 第三列是距离
            elif num_cols == 2:
                # 简化格式: 索引、距离
                idx = parts[0]
                dist = float(parts[1])
            else:
                continue  # 跳过格式不正确的行
            
            image_data.append((idx, dist))
    
    # 随机选择一些图像显示
    if len(image_data) > args.images_per_cluster:
        # 如果图像太多，随机采样
        sampled_data = random.sample(image_data, args.images_per_cluster)
    else:
        sampled_data = image_data
    
    # 加载图像
    loaded_images = []
    for idx, dist in sampled_data:
        if idx in index_to_image:
            # 从 Arrow 数据加载图像
            img_data = index_to_image[idx]
            image = load_image_from_arrow(img_data)
            
            if image:
                # 调整大小保持比例
                if max(image.size) > args.max_size:
                    ratio = args.max_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.LANCZOS)
                
                loaded_images.append((image, idx, dist))
            else:
                # 加载失败，使用占位图像
                img = create_placeholder_image(idx, dist)
                loaded_images.append((img, idx, dist))
        else:
            # 索引不在映射中，使用占位图像
            img = create_placeholder_image(idx, dist)
            loaded_images.append((img, idx, dist))
    
    # 创建输出路径
    output_path = os.path.join(args.output_dir, f"cluster_{cluster_id}.png")
    
    # 创建并保存图像网格
    if create_image_grid(loaded_images, cluster_id, cluster_size, output_path):
        successful_visualizations.append((cluster_id, cluster_size))

# 创建汇总页面
if successful_visualizations:
    # 创建HTML页面显示所有可视化结果
    html_path = os.path.join(args.output_dir, "index.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n<head>\n")
        f.write("<meta charset='utf-8'>\n")
        f.write("<title>聚类可视化结果</title>\n")
        f.write("<style>\n")
        f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
        f.write("h1 { color: #333; }\n")
        f.write("table { border-collapse: collapse; width: 100%; }\n")
        f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }\n")
        f.write("th { background-color: #f2f2f2; }\n")
        f.write("img { max-width: 100%; height: auto; }\n")
        f.write("</style>\n")
        f.write("</head>\n<body>\n")
        f.write("<h1>聚类可视化结果</h1>\n")
        f.write("<p>数据集目录: " + args.dataset_dir + "</p>\n")
        f.write("<table>\n")
        f.write("<tr><th>簇ID</th><th>大小</th><th>预览</th></tr>\n")
        
        for cluster_id, cluster_size in successful_visualizations:
            image_path = f"cluster_{cluster_id}.png"
            if os.path.exists(os.path.join(args.output_dir, image_path)):
                f.write(f"<tr><td>{cluster_id}</td><td>{cluster_size}</td>")
                f.write(f"<td><a href='{image_path}' target='_blank'><img src='{image_path}' height='200'></a></td></tr>\n")
        
        f.write("</table>\n</body>\n</html>")
    
    print(f"创建了汇总HTML页面: {html_path}")

print("可视化完成！")