import os
import shutil
from tqdm import tqdm
import argparse

def extract_images(index_file, source_folder, target_folder, file_extension='.jpg', filename_pattern=None):
    """
    根据索引文件提取图片到目标文件夹
    
    参数:
    index_file: 包含图片索引的文本文件路径
    source_folder: 源图片文件夹路径
    target_folder: 目标文件夹路径
    file_extension: 图片文件扩展名（默认为.jpg）
    filename_pattern: 文件名模式，用于格式化索引（例如：'image_{}.jpg'）
    """
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)
    
    # 读取索引文件
    with open(index_file, 'r') as f:
        indices = [line.strip() for line in f if line.strip()]
    
    print(f"从 {index_file} 读取了 {len(indices)} 个索引")
    
    # 获取源文件夹中的所有图片文件
    if os.path.isdir(source_folder):
        # 方式1：源文件夹是常规图片文件夹
        all_files = sorted([f for f in os.listdir(source_folder) 
                           if os.path.isfile(os.path.join(source_folder, f)) and 
                           f.lower().endswith(file_extension)])
        
        print(f"在源文件夹中找到 {len(all_files)} 个{file_extension}文件")
        
        # 检查文件数量与索引数量的关系
        if len(all_files) < max([int(idx) for idx in indices if idx.isdigit()], default=0):
            print(f"警告: 索引最大值超过源文件夹中的文件数量!")
        
        # 提取文件到目标文件夹
        extracted_count = 0
        for idx in tqdm(indices, desc="提取图片"):
            try:
                # 处理不同的索引格式
                if idx.isdigit():
                    # 如果索引是数字
                    idx_num = int(idx)
                    
                    if idx_num < len(all_files):
                        # 直接使用索引获取文件
                        source_file = os.path.join(source_folder, all_files[idx_num])
                        target_file = os.path.join(target_folder, all_files[idx_num])
                    elif filename_pattern:
                        # 使用文件名模式
                        source_filename = filename_pattern.format(idx)
                        source_file = os.path.join(source_folder, source_filename)
                        target_file = os.path.join(target_folder, source_filename)
                    else:
                        # 找不到对应文件，跳过
                        print(f"警告: 找不到索引 {idx} 对应的文件")
                        continue
                else:
                    # 如果索引不是数字，可能是文件名
                    if not idx.lower().endswith(file_extension):
                        idx = idx + file_extension
                    source_file = os.path.join(source_folder, idx)
                    target_file = os.path.join(target_folder, os.path.basename(idx))
                
                # 复制文件
                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
                    extracted_count += 1
                else:
                    print(f"警告: 文件不存在 - {source_file}")
            except Exception as e:
                print(f"处理索引 {idx} 时出错: {e}")
        
        print(f"成功提取 {extracted_count} 个图片到 {target_folder}")
    
    else:
        print(f"错误: 源文件夹不存在 - {source_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据索引文件提取图片")
    parser.add_argument("--index-file", type=str, default="deduped_image_paths.txt",
                        help="包含图片索引的文本文件路径")
    parser.add_argument("--source", type=str, required=True,
                        help="源图片文件夹路径")
    parser.add_argument("--target", type=str, default="deduped_images",
                        help="目标文件夹路径")
    parser.add_argument("--extension", type=str, default=".jpg",
                        help="图片文件扩展名（默认为.jpg）")
    parser.add_argument("--pattern", type=str, 
                        help="文件名模式，例如：'image_{}.jpg'或'{}.png'")
    
    args = parser.parse_args()
    
    extract_images(
        index_file=args.index_file,
        source_folder=args.source,
        target_folder=args.target,
        file_extension=args.extension,
        filename_pattern=args.pattern
    )