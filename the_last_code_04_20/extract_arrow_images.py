import os
import argparse
from tqdm import tqdm
from datasets import load_from_disk, Dataset, DatasetDict

def extract_images_from_arrow(index_file, source_folder, target_folder):
    """
    从Arrow数据集中根据索引提取图像
    
    参数:
    index_file: 包含图片索引的文本文件路径
    source_folder: Arrow数据集文件夹路径
    target_folder: 目标图片文件夹路径
    """
    # 创建目标文件夹
    os.makedirs(target_folder, exist_ok=True)
    print(f"创建目标文件夹: {target_folder}")
    
    # 读取索引文件
    with open(index_file, 'r') as f:
        indices = [int(line.strip()) for line in f if line.strip() and line.strip().isdigit()]
    
    print(f"从 {index_file} 读取了 {len(indices)} 个索引")
    
    # 加载Arrow数据集
    try:
        print(f"加载Arrow数据集: {source_folder}")
        dataset = load_from_disk(source_folder)
        print(f"数据集类型: {type(dataset)}")
        
        # 确定数据集结构和处理方式
        if isinstance(dataset, DatasetDict):
            # 数据集字典，可能有多个分割
            print(f"数据集包含以下分割: {list(dataset.keys())}")
            
            # 尝试确定包含图像的分割
            target_split = None
            for split_name in dataset.keys():
                split = dataset[split_name]
                
                # 检查这个分割是否有足够的样本和图像特征
                if (len(split) >= max(indices, default=0) and 
                    ('image' in split.features or 'img' in split.features)):
                    target_split = split_name
                    print(f"选择分割 '{target_split}' 进行提取，包含 {len(split)} 个样本")
                    break
            
            if target_split is None:
                # 如果没有找到合适的分割，使用第一个分割
                target_split = list(dataset.keys())[0]
                print(f"未找到合适的图像分割，使用 '{target_split}'")
            
            # 获取目标分割
            dataset = dataset[target_split]
        
        # 确定图像字段
        image_field = None
        for field in ['image', 'img', 'pixel_values']:
            if field in dataset.features:
                image_field = field
                print(f"找到图像字段: '{image_field}'")
                break
        
        if not image_field:
            print(f"警告: 无法确定图像字段。可用字段: {list(dataset.features.keys())}")
            # 尝试猜测可能的图像字段
            for field in dataset.features.keys():
                if 'image' in field.lower() or 'img' in field.lower():
                    image_field = field
                    print(f"猜测图像字段: '{image_field}'")
                    break
        
        if not image_field:
            raise ValueError("数据集中没有找到图像字段")
        
        # 提取图像
        success_count = 0
        error_count = 0
        
        print(f"开始提取 {len(indices)} 个图像...")
        for i, idx in enumerate(tqdm(indices, desc="提取图像")):
            if idx >= len(dataset):
                print(f"警告: 索引 {idx} 超出数据集范围 (0-{len(dataset)-1})")
                continue
                
            try:
                # 获取图像
                example = dataset[idx]
                image = example[image_field]
                
                # 为图像生成文件名
                # 如果数据集中有标签，可以包含在文件名中
                label_str = ""
                if 'labels' in example or 'label' in example:
                    label_key = 'labels' if 'labels' in example else 'label'
                    label_str = f"_class{example[label_key]}"
                
                # 保存图像
                # 使用索引作为文件名
                image_filename = f"image_{idx}{label_str}.jpg"
                image_path = os.path.join(target_folder, image_filename)
                
                # 检查image是否是PIL.Image对象
                if hasattr(image, 'save'):
                    image.save(image_path)
                    success_count += 1
                else:
                    print(f"警告: 索引 {idx} 的图像不是PIL.Image对象，类型: {type(image)}")
                    error_count += 1
                    
            except Exception as e:
                print(f"处理索引 {idx} 时出错: {e}")
                error_count += 1
        
        print(f"提取完成! 成功: {success_count}, 失败: {error_count}")
        print(f"图像已保存到: {target_folder}")
        
    except Exception as e:
        print(f"加载或处理数据集时出错: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从Arrow数据集提取图像")
    parser.add_argument("--index-file", type=str, default="deduped_image_paths.txt",
                        help="包含图片索引的文本文件路径")
    parser.add_argument("--source", type=str, required=True,
                        help="Arrow数据集文件夹路径")
    parser.add_argument("--target", type=str, default="deduped_images",
                        help="目标图片文件夹路径")
    
    args = parser.parse_args()
    
    extract_images_from_arrow(
        index_file=args.index_file,
        source_folder=args.source,
        target_folder=args.target
    )