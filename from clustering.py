from clustering.clustering import compute_centroids
import argparse
import yaml
import numpy as np
import os
import pathlib
import pprint
import logging
import sys

def get_logger(file_name=None, level=logging.INFO, stdout=True):
    """简化版的logger函数"""
    logger = logging.getLogger("clustering")
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # 清除现有处理器
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 添加文件处理器
    if file_name:
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 添加标准输出处理器
    if stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    return logger

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", 
        type=str, 
        required=True,
        help=".yaml config file path"
    )
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config_file, "r") as f:
        params = yaml.safe_load(f)
    
    # 确保保存目录存在
    save_folder = params["save_folder"]
    os.makedirs(save_folder, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(save_folder, "compute_centroids.log")
    logger = get_logger(file_name=log_file, level=logging.INFO, stdout=True)
    
    # 保存参数到文本文件
    with open(pathlib.Path(save_folder, "clustering_params.txt"), "w") as fout:
        pprint.pprint(params, fout)
    
    # 加载聚类参数
    seed = params["seed"]
    emb_memory_loc = params["emb_memory_loc"]
    dataset_size = params["dataset_size"]
    emb_size = params["emb_size"]
    niter = params["niter"]
    ncentroids = params["ncentroids"]
    Kmeans_with_cos_dist = params["Kmeans_with_cos_dist"]
    
    # 加载嵌入向量
    logger.info(f"Loading embeddings from {emb_memory_loc}")
    data = np.memmap(
        emb_memory_loc, dtype="float32", mode="r", shape=(dataset_size, emb_size)
    )
    
    # 执行聚类
    logger.info("Starting clustering...")
    compute_centroids(
        data,
        ncentroids,
        niter,
        seed,
        Kmeans_with_cos_dist,
        save_folder,
        logger,
        True
    )
    
    logger.info("Clustering completed successfully!")

if __name__ == "__main__":
    main()