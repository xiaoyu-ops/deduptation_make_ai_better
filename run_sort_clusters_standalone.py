import yaml
import numpy as np
import logging
import os
import sys
import faiss
from tqdm import tqdm

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# 配置文件路径
config_file = "clustering/configs/openclip/clustering_configs.yaml"
logger.info(f"加载配置文件: {config_file}")

# 加载配置
with open(config_file, 'r') as f:
    params = yaml.safe_load(f)

# 获取参数
emb_memory_loc = params['emb_memory_loc']
paths_memory_loc = params['paths_memory_loc']
dataset_size = params['dataset_size']
emb_size = params['emb_size']
save_folder = params['save_folder']
sorted_clusters_folder = params['sorted_clusters_file_loc']
sim_metric = params['sim_metric']
keep_hard = params['keep_hard']
path_str_type = params.get('path_str_dtype', 'S24')

# 确保目录存在
os.makedirs(sorted_clusters_folder, exist_ok=True)

# 加载数据
logger.info(f"加载嵌入向量: {emb_memory_loc}")
data = np.memmap(emb_memory_loc, dtype='float32', mode='r', shape=(dataset_size, emb_size))
paths = np.memmap(paths_memory_loc, dtype=path_str_type, mode='r', shape=(dataset_size,))

# 加载聚类中心
centroids_path = os.path.join(save_folder, "centroids.npy")
logger.info(f"加载聚类中心: {centroids_path}")
if not os.path.exists(centroids_path):
    logger.error(f"找不到聚类中心文件: {centroids_path}")
    sys.exit(1)
centroids = np.load(centroids_path)

# 加载或计算聚类分配
assignments_path = os.path.join(save_folder, "assignments.npy")
distances_path = os.path.join(save_folder, "distances.npy")

if os.path.exists(assignments_path) and os.path.exists(distances_path):
    logger.info(f"加载现有分配: {assignments_path}")
    assignments = np.load(assignments_path)
    distances = np.load(distances_path)
else:
    logger.info("计算数据点分配...")
    # 如果使用余弦距离，归一化向量
    if sim_metric == 'cosine':
        logger.info("使用余弦相似度，正在归一化向量...")
        data_norms = np.linalg.norm(data, axis=1, keepdims=True)
        data_norms[data_norms == 0] = 1
        normalized_data = data / data_norms
        
        centroids_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids_norms[centroids_norms == 0] = 1
        normalized_centroids = centroids / centroids_norms
        
        # 创建索引
        index = faiss.IndexFlatIP(emb_size)  # 内积索引
        index.add(normalized_centroids.astype(np.float32))
        
        # 搜索最近邻
        distances, assignments = index.search(normalized_data.astype(np.float32), 1)
    else:
        # L2距离
        index = faiss.IndexFlatL2(emb_size)
        index.add(centroids.astype(np.float32))
        distances, assignments = index.search(data.astype(np.float32), 1)
    
    # 转为一维数组
    assignments = assignments.reshape(-1)
    distances = distances.reshape(-1)
    
    # 保存结果
    logger.info("保存分配结果...")
    np.save(assignments_path, assignments)
    np.save(distances_path, distances)

# 将数据点组织到簇中
logger.info("将数据点组织到簇中...")
clusters = {}
for i, cluster_id in enumerate(tqdm(assignments, desc="处理数据")):
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    
    # 获取文件路径
    path = paths[i]
    if isinstance(path, bytes):
        try:
            path = path.decode('utf-8')
        except:
            path = str(path)
    
    # 添加到簇
    dist = distances[i][0] if distances.ndim > 1 else distances[i]
    clusters[cluster_id].append((i, path, dist))

# 排序每个簇
logger.info("排序簇中的数据点...")
for cluster_id, members in tqdm(clusters.items(), desc="排序簇"):
    # 根据距离排序
    if sim_metric == 'cosine':
        # 余弦相似度越大越相似
        if keep_hard:
            # 保留困难样本（相似度较低的）
            sorted_members = sorted(members, key=lambda x: x[2])
        else:
            # 保留相似度高的样本
            sorted_members = sorted(members, key=lambda x: x[2], reverse=True)
    else:
        # L2距离越小越相似
        if keep_hard:
            # 保留困难样本（距离较大的）
            sorted_members = sorted(members, key=lambda x: x[2], reverse=True)
        else:
            # 保留距离小的样本
            sorted_members = sorted(members, key=lambda x: x[2])
    
    # 保存排序后的簇
    output_path = os.path.join(sorted_clusters_folder, f"sorted_cluster_{cluster_id}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, path, dist in sorted_members:
            f.write(f"{idx}\t{path}\t{dist}\n")

# 按大小排序簇
logger.info("按大小排序所有簇...")
cluster_sizes = [(cluster_id, len(members)) for cluster_id, members in clusters.items()]
sorted_by_size = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)

# 保存大小排序结果
size_sorted_file = os.path.join(save_folder, "clusters_sorted_by_size.txt")
with open(size_sorted_file, 'w') as f:
    for cluster_id, size in sorted_by_size:
        f.write(f"{cluster_id}\t{size}\n")

logger.info(f"排序完成! 结果已保存到: {sorted_clusters_folder}")

# 打印统计信息
total_clusters = len(sorted_by_size)
logger.info(f"总共有 {total_clusters} 个簇")
if total_clusters > 0:
    largest_cluster = sorted_by_size[0]
    logger.info(f"最大的簇 (ID: {largest_cluster[0]}) 包含 {largest_cluster[1]} 个数据点")
    single_point_clusters = sum(1 for _, size in sorted_by_size if size == 1)
    logger.info(f"有 {single_point_clusters} 个簇只包含1个数据点")