import os
import argparse
import yaml
import torch
import multiprocessing
from semdedup import SemDeDupJob
import time
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="运行SemDeDup去重")
    parser.add_argument('--config-file', required=True, help='配置文件路径')
    parser.add_argument('--eps-list', required=True, help='epsilon值列表，用逗号分隔，例如：0.8,0.85,0.9')
    parser.add_argument('--nodes', type=int, default=1, help='要使用的节点数（在本地运行时忽略）')
    parser.add_argument('--tasks-per-node', type=int, default=4, help='每个节点的任务数')
    parser.add_argument('--cpus-per-task', type=int, default=4, help='每个任务的CPU数')
    return parser.parse_args()

def process_clusters(args_dict, start_cluster, end_cluster):
    """处理一个范围内的簇"""
    print(f"处理簇 {start_cluster} 到 {end_cluster}")
    job = SemDeDupJob(args_dict, start_cluster)
    job._process_shard(start_cluster, end_cluster)

def main():
    args = parse_args()
    
    # 加载配置文件
    # 使用utf-8编码明确打开文件
    with open(args.config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 转换为简单字典结构
    args_dict = argparse.Namespace()
    
    # 复制配置参数
    for key, value in config.items():
        setattr(args_dict, key, value)
    
    # 添加命令行参数
    args_dict.eps_list = [float(eps) for eps in args.eps_list.split(',')]
    args_dict.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保保存目录存在
    args_dict.save_loc = args_dict.save_folder
    os.makedirs(os.path.join(args_dict.save_loc, "dataframes"), exist_ok=True)
    
    # 计算集群数量和任务数
    total_clusters = args_dict.num_clusters
    num_tasks = min(args.tasks_per_node, multiprocessing.cpu_count())
    
    print(f"总共 {total_clusters} 个簇")
    print(f"使用 {num_tasks} 个并行任务")
    print(f"使用设备: {args_dict.device}")
    print(f"Epsilon值: {args_dict.eps_list}")
    
    # 计算每个任务处理的簇数量
    clusters_per_job = total_clusters
    args_dict.clusters_per_job = clusters_per_job
    
    clusters_per_task = int(np.ceil(clusters_per_job / num_tasks))
    
    # 为每个任务分配簇范围
    tasks = []
    for i in range(num_tasks):
        start_cluster = i * clusters_per_task
        end_cluster = min((i + 1) * clusters_per_task, total_clusters)
        
        # 跳过空任务
        if start_cluster >= end_cluster:
            continue
            
        tasks.append((args_dict, start_cluster, end_cluster))
    
    # 使用进程池并行处理
    start_time = time.time()
    
    if num_tasks > 1:
        # 多进程处理
        with multiprocessing.Pool(processes=num_tasks) as pool:
            pool.starmap(process_clusters, tasks)
    else:
        # 单进程处理
        for task in tasks:
            process_clusters(*task)
    
    elapsed_time = time.time() - start_time
    print(f"所有任务完成！总耗时: {elapsed_time/60:.2f} 分钟")
    
if __name__ == "__main__":
    main()