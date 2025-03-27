import yaml
import random
import numpy as np
import logging
import os
import time
import sys
import faiss
from tqdm import tqdm

# 设置环境变量以避免OpenMP警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 设置OpenMP线程数
os.environ['OMP_NUM_THREADS'] = '4'  # 设置为你的CPU核心数

# 设置日志级别
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# 配置文件路径
config_file = "clustering/configs/openclip/clustering_configs.yaml"
logger.info(f"加载配置文件: {config_file}")

# 检查文件是否存在
if not os.path.exists(config_file):
    logger.error(f"配置文件不存在: {config_file}")
    exit(1)

# 加载聚类参数
with open(config_file, 'r') as y_file:
    params = yaml.load(y_file, Loader=yaml.FullLoader)

# 设置随机种子
SEED = params['seed']
random.seed(SEED)
np.random.seed(SEED)

# 打印配置信息
logger.info(f"聚类参数: {params}")

# 获取路径和大小参数
emb_memory_loc = params['emb_memory_loc']
paths_memory_loc = params['paths_memory_loc']
dataset_size = params['dataset_size']
emb_size = params['emb_size']
path_str_type = params.get('path_str_dtype', 'S24')  # 兼容不同命名

# 检查文件是否存在
if not os.path.exists(emb_memory_loc):
    logger.error(f"嵌入向量文件不存在: {emb_memory_loc}")
    exit(1)

# 加载嵌入向量和路径
logger.info(f"加载嵌入向量: {emb_memory_loc}")
try:
    emb_memory = np.memmap(emb_memory_loc, dtype='float32', mode='r', shape=(dataset_size, emb_size))
    paths_memory = np.memmap(paths_memory_loc, dtype=path_str_type, mode='r', shape=(dataset_size,))
    logger.info(f"成功加载，嵌入向量形状: {emb_memory.shape}")
except Exception as e:
    logger.error(f"加载数据时出错: {e}")
    exit(1)

# 确保保存目录存在
save_folder = params['save_folder']
sorted_clusters_folder = params['sorted_clusters_file_loc']
os.makedirs(save_folder, exist_ok=True)
os.makedirs(sorted_clusters_folder, exist_ok=True)
logger.info(f"结果将保存到: {save_folder}")

# 直接在内存中运行K-means聚类
def run_kmeans(data, ncentroids, niter, seed, use_cosine=True):
    """运行K-means聚类，适用于CPU版本的FAISS"""
    logger.info(f"准备K-means聚类: {ncentroids}个簇, {niter}次迭代, 使用余弦距离: {use_cosine}")
    
    d = data.shape[1]  # 向量维度
    
    # 创建一个标准的Kmeans对象
    kmeans = faiss.Kmeans(
        d=d,  # 维度
        k=ncentroids,  # 簇数量
        niter=niter,  # 迭代次数
        seed=seed,  # 随机种子
        verbose=True,  # 显示进度
    )
    
    # 如果使用余弦相似度，先归一化数据
    if use_cosine:
        logger.info("使用余弦相似度，正在归一化向量...")
        # 计算每个向量的范数
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        # 防止除以零
        norms[norms == 0] = 1.0
        # 归一化
        normalized_data = data / norms
        # 转换为float32类型
        data_for_clustering = normalized_data.astype(np.float32)
    else:
        data_for_clustering = data.astype(np.float32)
    
    # 进度条
    with tqdm(total=niter, desc="K-means聚类进度") as pbar:
        # 模拟更新进度条的回调
        def update_progress():
            pbar.update(1)
        
        # 保存原始print函数
        original_print = print
        
        # 创建自定义print函数来捕获进度
        def custom_print(*args, **kwargs):
            message = " ".join(map(str, args))
            original_print(*args, **kwargs)
            if "Iteration" in message:
                update_progress()
        
        # 替换print函数
        import builtins
        builtins.print = custom_print
        
        try:
            # 开始计时
            start_time = time.time()
            
            # 训练K-means模型
            kmeans.train(data_for_clustering)
            
            # 计算用时
            elapsed_time = time.time() - start_time
            logger.info(f"K-means训练完成，用时: {elapsed_time:.2f}秒")
            
            # 获取聚类中心
            centroids = kmeans.centroids
            
            # 创建适当的索引用于分配
            if use_cosine:
                index = faiss.IndexFlatIP(d)  # 内积索引（用于余弦相似度）
            else:
                index = faiss.IndexFlatL2(d)  # L2距离索引
            
            # 添加聚类中心到索引
            index.add(centroids)
            
            # 给每个数据点分配最近的聚类中心
            logger.info("分配数据点到聚类中心...")
            distances, assignments = index.search(data_for_clustering, 1)
            
            # 重塑结果为一维数组
            assignments = assignments.reshape(-1)
            distances = distances.reshape(-1)
            
            return centroids, assignments, distances
            
        finally:
            # 恢复原始print函数
            builtins.print = original_print
    
# 执行聚类
try:
    logger.info("开始执行K-means聚类...")
    
    # 运行聚类
    centroids, assignments, distances = run_kmeans(
        data=emb_memory,
        ncentroids=params['ncentroids'],
        niter=params['niter'],
        seed=params['seed'],
        use_cosine=params['Kmeans_with_cos_dist']
    )
    
    # 保存聚类结果
    logger.info("保存聚类结果...")
    centroids_path = os.path.join(save_folder, "centroids.npy")
    assignments_path = os.path.join(save_folder, "assignments.npy")
    distances_path = os.path.join(save_folder, "distances.npy")
    
    np.save(centroids_path, centroids)
    np.save(assignments_path, assignments)
    np.save(distances_path, distances)
    
    logger.info(f"已保存聚类中心到: {centroids_path}")
    logger.info(f"已保存聚类分配到: {assignments_path}")
    logger.info(f"已保存距离到: {distances_path}")
    
    # 计算聚类统计信息
    cluster_counts = np.bincount(assignments)
    max_size = np.max(cluster_counts)
    min_size = np.min(cluster_counts)
    avg_size = np.mean(cluster_counts)
    single_clusters = np.sum(cluster_counts == 1)
    
    logger.info(f"聚类统计: 最大簇大小={max_size}, 最小簇大小={min_size}, 平均簇大小={avg_size:.2f}")
    logger.info(f"单样本聚类数量: {single_clusters}")
    
    # 创建聚类成员映射
    logger.info("组织聚类成员...")
    clusters = {}
    for i, cluster_id in enumerate(assignments):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        
        # 获取路径（处理可能的字节字符串）
        path = paths_memory[i]
        if isinstance(path, bytes):
            path = path.decode('utf-8')
        
        clusters[cluster_id].append((i, path))
    
    # 按大小排序聚类
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    # 保存排序后的聚类
    logger.info(f"保存{len(sorted_clusters)}个排序后的聚类...")
    for i, (cluster_id, members) in enumerate(tqdm(sorted_clusters, desc="保存聚类")):
        cluster_file = os.path.join(sorted_clusters_folder, f"cluster_{i}.txt")
        with open(cluster_file, 'w') as f:
            for idx, path in members:
                f.write(f"{idx}\t{path}\n")
    
    # 创建聚类摘要
    summary_file = os.path.join(save_folder, "cluster_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"总聚类数: {len(sorted_clusters)}\n")
        f.write(f"最大聚类大小: {max_size}\n")
        f.write(f"最小聚类大小: {min_size}\n")
        f.write(f"平均聚类大小: {avg_size:.2f}\n")
        f.write(f"单样本聚类数: {single_clusters}\n")
        f.write("\n聚类大小分布:\n")
        
        # 统计不同大小聚类的数量
        for size in range(1, 11):
            count = np.sum(cluster_counts == size)
            f.write(f"大小为 {size} 的聚类数: {count}\n")
        
        # 统计大聚类数量
        large_clusters = np.sum(cluster_counts > 10)
        f.write(f"大小 > 10 的聚类数: {large_clusters}\n")
    
    logger.info(f"已保存聚类摘要到: {summary_file}")
    logger.info("聚类过程成功完成!")
    
except KeyboardInterrupt:
    logger.info("用户中断了聚类过程")
    sys.exit(1)
except Exception as e:
    logger.error(f"聚类过程失败: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)