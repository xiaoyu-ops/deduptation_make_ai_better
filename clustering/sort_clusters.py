# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time
import numpy as np
import logging
from tqdm import tqdm
import pandas as pd
import os
import pathlib
import yaml
import math
import os
from typing import List
import random
import numpy as np
import submitit
import torch
import pprint
from tqdm import tqdm
import argparse
from typing import List, Tuple, Union
from clustering.utils import get_logger


def assign_and_sort_clusters(
    data: Union[np.memmap, np.ndarray],
    paths_list: Union[np.memmap, np.ndarray],
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    kmeans_with_cos_dist: bool = False,
    save_folder: str = "",
    sorted_clusters_file_loc: str = "",
    cluster_ids=range(5000),
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """
    Assigns data points to clusters and sorts each cluster items based on distance to its centroid.

    Args:
        data (np.memmap): A memory-mapped array containing the data points.
        paths_list (np.memmap): A memory-mapped array containing the paths of the data points.
        sim_metric (str): The similarity metric to use for clustering. Defaults to "cosine".
        keep_hard (bool): When True, we sort cluster items in descending order by the similarity to cluster centroid. Defaults to True.
        kmeans_with_cos_dist (bool): Whether to use cosine distance for K-means clustering. Defaults to False.
        save_folder (str): The location of the K-means centroids file. Defaults to "".
        sorted_clusters_file_loc (str): The location to save the sorted clusters file. Defaults to "".
        logger (logging.Logger): A logger object to log messages. Defaults to None.
        cluster_ids (list): The range of cluster IDs to sort. Defaults to range(5000).

    Returns:
        pd.DataFrame: A DataFrame containing the sorted clusters.
        函数的主要作用是将数据点分配到各个聚类中，并根据数据点与各自簇中心的距离对每个簇内的项进行排序。

        在参数部分，详细说明了各个输入：

        data (np.memmap)：一个内存映射数组，包含了所有的数据点。使用内存映射可以在处理大型数据集时降低内存开销。
        paths_list (np.memmap)：同样是一个内存映射数组，存储每个数据点的路径信息，这可以作为数据点的唯一标识。
        sim_metric (str)：指定用于聚类的相似度度量方法，默认值为 "cosine"，即余弦相似度。
        keep_hard (bool)：当设置为 True 时，会按数据点与簇中心的相似度降序排序，即相似度高的排在前面；默认值为 True。
        kmeans_with_cos_dist (bool)：用于指示是否在 K-means 聚类中采用余弦距离，默认值为 False。
        save_folder (str)：保存 K-means 质心文件的位置，允许用户指定存储路径。
        sorted_clusters_file_loc (str)：排序后簇文件的保存位置。
        logger (logging.Logger)：用于记录程序日志的记录器对象，便于跟踪程序运行情况。
        cluster_ids (list)：指定需要进行排序的聚类 ID 范围，默认值为 range(5000)，即对前 5000 个聚类进行排序。
        返回值部分说明了函数输出一个 Pandas DataFrame,其中包含了排序后的聚类信息。
    """

    assert sim_metric in [
        "l2",
        "cosine",
    ], f"Unsupported similarity metric '{sim_metric}'."
    assert not (
        kmeans_with_cos_dist and sim_metric == "l2"
    ), "Cannot use cosine distance with L2 similarity metric."

    # If Kmeans_with_cos_dist is True, set spherical=True. This is the spherical parameter of faiss kmeans clustering.
    # 如果 Kmeans_with_cos_dist 为 True，则设置 spherical=True。这是 faiss kmeans 聚类的 spherical 参数。
    spherical = kmeans_with_cos_dist

    # Step 3: Sort each class/cluster
    # 步骤 3：对每个类/簇进行排序
    logger.info("Ranking...")
    kmeans_centroids_file_loc = pathlib.Path(save_folder, "centroids.npy")
    dist_to_cent_file_loc = pathlib.Path(save_folder, "dist_to_cent.npy")
    nearest_cent_file_loc = pathlib.Path(save_folder, "nearest_cent.npy")
    kmeans_centroids = np.load(kmeans_centroids_file_loc)
    nearest_cent = np.load(nearest_cent_file_loc)
    dist_to_cent = np.load(dist_to_cent_file_loc)

    start_time = time.time()

    dist_df = pd.DataFrame(
        {
            "paths_list": paths_list,
            "nearest_cent": nearest_cent,
            "dist_to_cent": dist_to_cent,
        }
    )

    sorted_clusters = rank_within_cluster(
        data,
        dist_df,
        kmeans_centroids,
        sim_metric,
        keep_hard,
        spherical,
        cluster_ids,
        sorted_clusters_file_loc,
    )
    logger.info(f"Time for ranking: {(time.time() - start_time) / 60:.2f} mins")
    logger.info("DONE!")

    return sorted_clusters


def rank_within_cluster(
    data: Union[np.memmap, np.ndarray],
    dist_df: pd.DataFrame,
    centroids: np.ndarray,
    sim_metric: str = "cosine",
    keep_hard: bool = True,
    spherical: bool = False,
    cluster_ids: List[int] = range(50000),
    sorted_clusters_file_loc: str = "",
) -> List[List[Tuple[str, int, float, int]]]:
    """
    Sorts each cluster items by the distance to the cluster centroid.
    Cluster is represented as list of tuples. Each tuple has 4 values:
        example_path: unique path to the example/image/text doc, for imagenet it could be something like "n04235860_14959.JPEG",
        example_id_in_dataset: int between 0 and cluster_size-1
        dist_to_cent: cosine distance to cluster centroid
        cluster_id: cluster number (from 0 to number of clusters)

    Arguments:
    data -- the data for which the clusters were created (np.ndarray or np.memmap)
    dist_df -- DataFrame with the distances between the data points and the centroids, nearest centroid for each example, and path to each example.
    centroids -- np.ndarray with the centroids for each cluster.
    sim_metric -- the similarity metric used to compute distances, should be one of ["cosine", "l2"]
    keep_hard -- a boolean ehen True, we sort cluster items in descending order by the similarity to cluster centroid. Defaults to True.
    spherical -- a boolean True means spherical was used for computing centroids (used for cosine similarity).
    cluster_ids -- a list of cluster ids to process. Each slurm job will process part of the clusters.
    sorted_clusters_file_loc -- the location to save the sorted clusters.

    Returns:
    A list of cluster representations, where each representation is a list of tuples with 4 values.
    -- exampel for a cluster (the list bellow is sorted by dist_to_cent in descending order)
        [
          [example_name, example_id_in_dataset, dist_to_cent, cluster_label],
          [example_name, example_id_in_dataset, dist_to_cent, cluster_label],
                                        .
                                        .
                                        .
                                                                    ]
    该函数用于根据数据点到簇中心的距离对每个簇内的项进行排序。每个簇被表示为一个包含多个元组的列表，每个元组包含四个值：

    example_path：示例或图像/文本文档的唯一路径，对于 ImageNet 数据集，可能类似于 "n04235860_14959.JPEG"。
    example_id_in_dataset：数据集中示例的 ID，范围在 0 到 cluster_size-1 之间。
    dist_to_cent：到簇中心的余弦距离。
    cluster_id：簇的编号，从 0 到簇的总数。
    函数的参数包括：

    data：用于创建簇的数据，可以是 NumPy 数组（np.ndarray）或内存映射数组（np.memmap）。
    dist_df：一个 DataFrame，包含数据点与质心之间的距离、每个示例的最近质心以及每个示例的路径。
    centroids：一个 NumPy 数组，包含每个簇的质心。
    sim_metric：用于计算距离的相似度度量方法，应为 ["cosine", "l2"] 之一。
    keep_hard：一个布尔值，当设置为 True 时，会按数据点与簇中心的相似度降序排序，默认值为 True。
    spherical：一个布尔值，指示是否在球面上计算质心（用于余弦相似度），默认值为 False。
    cluster_ids：一个簇 ID 列表，指定要处理的簇，每个 SLURM 作业将处理部分簇。
    sorted_clusters_file_loc：保存排序后簇文件的位置。
    函数返回一个簇表示的列表，每个表示是一个包含多个元组的列表，每个元组包含四个值。
    """

    assert sim_metric in [
        "cosine",
        "l2",
    ], "sim_metric should be one of ['cosine', 'l2']"
    os.makedirs(sorted_clusters_file_loc, exist_ok=True)

    sorted_clusters_list = []
    for cluster_c in tqdm(cluster_ids):
        if os.path.exists(f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"):
            print(f"Cluster {cluster_c} exits, skipping....")
            continue

        cluster_df = dist_df.loc[dist_df["nearest_cent"] == cluster_c]#保留那些最近质心的数据点且等于cluster_c的

        cluster_items = list(cluster_df.index)  ## -- ids of examples in cluster c
        ## -- 提取当前簇中所有数据点的索引并将其转换为一个列表
        if sim_metric == "cosine":
            if spherical:#注意spherical是球面距离
                cluster_dists_to_cent = list(1 - cluster_df["dist_to_cent"])
            else:
                cluster_c_centroid = torch.Tensor(centroids[cluster_c])#将当前簇的质心转换为一个张量
                sim_to_cent = torch.nn.CosineSimilarity(dim=1)(
                    torch.Tensor(data[cluster_items]), cluster_c_centroid
                )#利用pytorch中的模块来计算数据点与簇中心质点的余xuan相似度
                cluster_dists_to_cent = (1 - sim_to_cent).tolist()

        elif sim_metric == "l2":  # -- get l2 distance from "dist_to_cent" array
            # --获得欧氏距离从数组中
            cluster_dists_to_cent = list(cluster_df["dist_to_cent"])

        cluster_label = np.full((len(cluster_df)), cluster_c).tolist()
        example_paths = list(cluster_df["paths_list"])
        sort_descending = keep_hard
        cluster_sorted = sorted(
            zip(example_paths, cluster_items, cluster_dists_to_cent, cluster_label),
            key=lambda x: x[2],
            reverse=sort_descending,
        )  # -- sort_descending = True for descending sort
        ##sort_decending如果是正的那么就降序排序

        sorted_clusters_list.append(
            cluster_sorted
        )  # -- Descending dists. list of of list of tuples (example, dist). The ith list of tuples corresponds to cluster i
        sorted_cluster_file_path = f"{sorted_clusters_file_loc}/cluster_{cluster_c}.npy"
        np.save(sorted_cluster_file_path, cluster_sorted)
    return sorted_clusters_list


class SLURMJob(submitit.helpers.Checkpointable):
    """
    - Each SLURMJob will calculate and save the encodings for a list of shards.
    - Parallelize shards across jobs so that preemption in the middle of an epoch isn't a problem and because we want to
    keep the shard structure anyway.
    - Process more than one shard per job because each shard takes about a minute so we want to amortize overhead.
    - Preempted jobs get resubmitted. Already computed shards get skipped internally.
    每个 SLURM 作业将计算并保存一组分片的编码：每个作业负责处理一组数据分片，计算其编码并将结果保存下来。
    跨作业并行处理分片：将分片分配给不同的作业并行处理，以避免在一个 epoch 中途被抢占时出现问题，并且因为我们希望保留分片结构。
    每个作业处理多个分片：因为处理每个分片大约需要一分钟时间，所以每个作业处理多个分片，以摊薄作业调度和启动的开销。
    被抢占的作业会重新提交：如果作业在执行过程中被抢占，它们会被重新提交。已经计算完成的分片会在内部跳过，不会重复计算。

    """

    def __init__(self, args, cluster_ids: List[str]):
        self.args = args
        self.cluster_ids = cluster_ids
        assert args.ncentroids == len(self.cluster_ids)

    def seed_everything(self, seed=42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def _encode_shard(self, args):
        # Configure logger
        # 日志
        logger = get_logger(
            file_name=f"{args.save_folder}/clustering-logs.log",
            level=logging.INFO,
            stdout=True,
        )
        args.logger = logger

        data = np.memmap(
            args.emb_memory_loc,
            dtype="float32",
            mode="r",
            shape=(args.dataset_size, args.emb_size),
        )
        paths_list = np.memmap(
            args.paths_memory_loc,
            dtype=args.path_str_dtype,
            mode="r",
            shape=(args.dataset_size,),
        )

        assign_and_sort_clusters(
            data,
            paths_list,
            args.sim_metric,
            args.keep_hard,
            args.Kmeans_with_cos_dist,
            args.save_folder,
            args.sorted_clusters_file_loc,
            args.cluster_ids_for_job,
            args.logger,
        )

        return

    def __call__(self):
        self.seed_everything(self.args.seed)

        num_clusters = len(self.cluster_ids)
        print(
            f"There are {num_clusters} clusters: {self.cluster_ids[0]} to  {self.cluster_ids[-1]}"
        )

        job_env = submitit.JobEnvironment()

        print(f"There are {args.num_tasks} tasks in this job")
        print(f"This is the task #{job_env.local_rank}")

        ## devide clusters across jobs (cpus)
        ## 将簇分配到不同的cpu上进行处理（并行化）
        num_clusters_per_job = int(math.ceil(num_clusters / args.num_tasks))
        task_rank = job_env.local_rank
        start = task_rank * num_clusters_per_job
        end = (task_rank + 1) * num_clusters_per_job
        end = min(end, num_clusters)

        cluster_ids_for_job = self.cluster_ids[start:end]
        print(
            f"This job/task will process {len(cluster_ids_for_job)} clusters: cluster {cluster_ids_for_job[0]} to cluster {cluster_ids_for_job[-1]}"
        )

        self.args.cluster_ids_for_job = cluster_ids_for_job

        self._encode_shard(self.args)


def launch_jobs(args):
    """
    Runs the clustering job using the specified configuration file and SLURM parameters.
    使用指定的配置文件和 SLURM 参数运行聚类作业

    """
    confg_file = args.config_file
    ## -- load kmeans clustering parameters from configs file
    ## --加载kmeans相关的参数从configs file中
    with open(confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    with open(pathlib.Path(params["save_folder"], "clustering_params.txt"), "w") as f:
        pprint.pprint(params, f)

    args.save_folder = params["save_folder"]
    args.emb_memory_loc = params["emb_memory_loc"]
    args.paths_memory_loc = params["paths_memory_loc"]
    args.dataset_size = params["dataset_size"]
    args.emb_size = params["emb_size"]
    args.path_str_dtype = params["path_str_dtype"]  # "S24" for LAION
    args.ncentroids = params["ncentroids"]
    args.seed = params["seed"]
    args.sim_metric = params["sim_metric"]
    args.keep_hard = params["keep_hard"]
    args.Kmeans_with_cos_dist = params["Kmeans_with_cos_dist"]
    args.save_folder = params["save_folder"]
    args.sorted_clusters_file_loc = params["sorted_clusters_file_loc"]
    args.cluster_ids_for_job = list(range(args.ncentroids))

    ## -- SLURM CONFIG
    ## -- SLURM 相关的配置
    PARTITION = args.partition
    SLURM_ARRAY_PARALLELISM = 1000
    NODES = 1
    TIMEOUT = args.timeout
    CPUS_PER_TASKS = args.cpus_per_task
    TASKS_PER_NODE = args.num_tasks

    ## -- SUBMIT
    submitit_path = f"{args.save_folder}/clustering-jobs/%j"
    executor = submitit.AutoExecutor(folder=submitit_path, slurm_max_num_timeout=30)
    executor.update_parameters(
        slurm_partition=PARTITION,
        slurm_array_parallelism=SLURM_ARRAY_PARALLELISM,
        nodes=NODES,
        tasks_per_node=TASKS_PER_NODE,
        cpus_per_task=CPUS_PER_TASKS,
        timeout_min=TIMEOUT,
    )

    jobs = []

    ## -- Start a job with <args.num_tasks> task. Each task will process part of the clusters
    with executor.batch():#是submitit 库中的一个上下文管理器，用于批量提交作业。
        exp = SLURMJob(args, list(range(0, args.ncentroids)))
        job = executor.submit(exp)
        jobs.append(job)

    for job in jobs:
        print("Submitted job_id:", job.job_id)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()#argparse.ArgumentParser用于解析命令行参数
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/openclip/paralellized_kmeans_dino_embs_configs.yaml",
    )
    # -- slurm parameters
    parser.add_argument(
        "--partition", type=str, default="scaling_data_pruning", help="partition"
    )
    parser.add_argument("--num-tasks", type=int, default=10, help="number of tasks")
    parser.add_argument(
        "--cpus-per-task", type=int, default=5, help="number of cpus per task"
    )
    parser.add_argument(
        "--timeout", type=int, default=500, help="job timeout in minutes"
    )
    #这些参数包括分区名称、任务数量、每个任务使用的 CPU 数量和作业超时时间。
    args = parser.parse_args()#解析命令行参数生成命名空间对象

    # Submit the job提交聚类job
    launch_jobs(args)
