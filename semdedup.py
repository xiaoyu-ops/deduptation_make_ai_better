# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import numpy as np
import pandas as pd
import submitit
import torch
from tqdm import tqdm
import pickle
import random
import math
import time
import pprint
from constants import DIST_METRIC_INDEX


def init_memmap_embs(
    embs_memory_loc: str, dataset_size: int, emd_size: int = 512, dtype: str = "float32"
) -> np.memmap:
    """
    Initializes a memory-mapped NumPy array to read embeddings of examples.
    初始化一个内存映射的numpy数组以读取示例的嵌入。

    Args:
        embs_memory_loc (str): Path to the memory-mapped file.
        dataset_size (int): Size of the dataset.
        emd_size (int): Dimensionality of the embeddings.
        dtype (str): Data type of the embeddings.

    Returns:
        np.memmap: A memory-mapped NumPy array.
    """
    embs = np.memmap(
        embs_memory_loc, dtype=dtype, mode="r", shape=(dataset_size, emd_size)
    )
    return embs


class SemDeDupJob(submitit.helpers.Checkpointable):
    """
    - Each SLURMJob will run SemDeDup on number of clusters and save dataframe with which examples to keep from each cluster.
    - Parallelize job_start_cluster across jobs so that preemption in the middle of an epoch isn't a problem and because we want to
    keep the shard structure anyway.
    - Process more than one cluster per job=> run multiple taks inside each jobs.
    - Preempted jobs get resubmitted. Already precessed clusters get skipped internally.
    -每个slurmjob都会在簇数量上运行semdedup,并保存数据帧,并使用哪些示例来保留每个群集。
     -跨作业并行化Job_start_cluster
    无论如何，保持碎片结构。
     -处理每个作业中的多个群集=>在每个作业中运行多个TAK。
     -抢先的工作被重新提交。已经进取的簇被内部跳过。
    """

    def __init__(self, args, job_start_cluster: int):
        self.args = args
        self.job_start_cluster = job_start_cluster
        random.seed(args.seed)#初始化类的实例变量

    def _contains_duplicates(self, arr):
        return len(np.unique(arr)) != len(arr)#np.unique 获得arr数组中的唯一元素如果二者不相等说明有重复元素

    def semdedup(self, cluster, cluster_reps, device):
        st = time.time()
        ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
        ## -- 计算簇内项目之间的成对余弦相似度，并将对角线元素替换为零，以忽略自相似性。
        cluster_reps.to(device)
        pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)#矩阵相乘得到余弦相似度
        del cluster_reps
        pair_w_sim_matrix.fill_diagonal_(0.0)#将对角线元素替换为零，这样可以忽略自相似性
        assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]#判断是否为方阵，如果不是，系统会自动抛出一个断言错误

        ## -- get paths to cluster i images
        ## -- 获取到簇i图像的路径
        image_urls = cluster[:, 0]#提取第一列的数据，而第一列的数据是图片的路径

        ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
        ## -- 确保所有路径都是唯一的，这样可以确保重复的确实多次存储在内存中
        assert not self._contains_duplicates(image_urls)

        ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
        ## -- 我们需要上三角矩阵，因为(1)我们不需要查看自相似性(始终=1)(2)我们需要组合而不是排列
        triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

        ## -- if the max sim between one example and any other example is > 1-eps, remove this example
        ## -- 如果一个示例与任何其他示例之间的最大相似度> 1-eps，则删除此示例
        M = torch.max(triu_sim_mat, dim=0)[0].cpu()#dim=0表示沿着列的方向取最大值，返回的是一个元组，第一个元素是最大值，第二个元素是最大值的索引
        print(f"Step time: {time.time()-st}(s)")#st是之前记录的一个时间戳，这样一相互一减就可以得到这个时间段的时间

        return M#返回最大值

    def _process_shard(self, start_cluster: int, end_cluster: int):
        # print("SemDeDup params: ", self.args)
        # 原本是要打印self.args的
        st = time.time()

        embs = init_memmap_embs(
            self.args.embs_memory_loc, self.args.dataset_size, self.args.emd_size
        )#init_memmap_embs函数是一个在前面已经定义的函数，用来初始化一个内存映射的numpy数组以读取示例的嵌入
        #*********注意这里的embs_memory_loc存储了嵌入向量***********

        step_time = []#记录每一步的时间

        for cluster_id in tqdm(range(start_cluster, end_cluster)):
            step_st = time.time()

            df_file_loc = os.path.join(
                self.args.save_loc, f"dataframes/cluster_{cluster_id}.pkl"
            )#self.args.save_loc是一个保存文件的路径，f"dataframes/cluster_{cluster_id}.pkl"是一个文件名

            if os.path.exists(df_file_loc):  # and os.path.exists(dict_file_loc):
                print(f"{df_file_loc} exists, moving on")#检查对于的文件是否存在，如果存在就跳过
                continue

            ## -- load cluster i representations
            ## -- 加载簇i的表示
            cluster_i = np.load(
                os.path.join(
                    self.args.sorted_clusters_path, f"cluster_{cluster_id}.npy"
                )
            )#前后两段会正确地被拼接起来
            # 1) store cluster size
            # 保存簇的大小
            cluster_size = cluster_i.shape[0]
            print("cluster_size: ", cluster_size)

            if cluster_size == 1:
                points_to_remove_df = pd.DataFrame()#创建一个空的DataFrame
                points_to_remove_df["indices"] = [0]#添加一列数据
                for eps in self.args.eps_list:#遍历eps_list
                    ## We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                    ## 当数据点与其他数据点的成对相似度> 1-ebs时，我们需要从数据集中删除一个数据点
                    points_to_remove_df[f"eps={eps}"] = [False]#初始化为Flase
                if self.args.save_loc != "":
                    ## --save df
                    ## 保存df
                    with open(df_file_loc, "wb") as file:#打开这个文件后以二进制写入
                        pickle.dump(points_to_remove_df, file)#将points_to_remove_df以序列化保存到file中
                print("DONE cluster_id ", cluster_id)
                continue

            ## -- By default, we keep hard examples from groups
            ## -- 默认情况下，我们保留组中的困难示例
            clutser_items_indices = list(range(cluster_size))#包含0到cluster_size-1的列表
            ## -- OR: shuffle cluster to keep random example from each group
            ## -- 或者：随机排列集群以保留每个组的随机示例
            if self.args.which_to_keep.lower() == "random":
                random.shuffle(clutser_items_indices)#打乱列表
                cluster_i = cluster_i[clutser_items_indices]#根据打乱后的索引重新排列cluster_i
            ## -- OR: reverse cluster to keep easy examples
            ## -- 或者：反转集群以保留简单示例
            if self.args.which_to_keep.lower() == "easy":
                clutser_items_indices = clutser_items_indices[::-1]#就是反转序列
                cluster_i = cluster_i[clutser_items_indices]

            ## -- indices for cluster items in the dataset
            ## -- 数据集中簇项目的索引
            cluster_ids = cluster_i[:, 1].astype("int32")#从第二列中提取对应的簇id并转成int32类型
            cluster_reps = embs[cluster_ids]#根据cluster_ids提取对应的embedding
            cluster_reps = torch.tensor(cluster_reps)

            M = self.semdedup(cluster_i, cluster_reps, self.args.device)#实例化一个函数

            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = clutser_items_indices

            for eps in self.args.eps_list:
                ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                ## -- 当数据点与其他数据点的成对相似度> 1-ebs时，我们需要从数据集中删除一个数据点
                eps_points_to_remove = M > 1 - eps#这里记录bool值
                points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

            if self.args.save_loc != "":
                ## --save df
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)#重复上面的操作以序列化保存到file中

            step_time.append_cluster(time.time() - step_st)
            print("DONE cluster: ", cluster_id)

        print(
            f"DONE in {((time.time()-st)/60):.2f} minutes, Average Step time {(sum(step_time)/len(step_time)):.2f}(s)"
        )
        return

    def __call__(self):#实例化一个对象时，会调用这个方法
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(self.args))#打印self.args的全部内容
        job_start_cluster = self.job_start_cluster

        print(
            f"This job will process clusters {job_start_cluster} to  {min(self.args.num_clusters, job_start_cluster+self.args.clusters_per_job)}"
        )

        job_env = submitit.JobEnvironment()#获取当前job的环境

        print(f"There are {job_env.num_tasks} tasks in this job")#任务总数

        print(f"I'm the task #{job_env.local_rank} on node {job_env.node}")#当前任务的在本地节点上的编号和node，编号与节点
        print(f"I'm the task #{job_env.global_rank} in the job")#全局中的编号

        ## divide clusters across tasks (cpus)
        ## 在任务之间划分簇
        num_clusters_per_task = int(
            math.ceil(self.args.clusters_per_job / job_env.num_tasks)#获得每个任务处理的簇的数量
        )
        task_rank = job_env.local_rank
        start_cluster = job_start_cluster + task_rank * num_clusters_per_task
        end_cluster = job_start_cluster + (task_rank + 1) * num_clusters_per_task
        end_cluster = min(self.args.num_clusters, end_cluster)
        end_cluster = min(end_cluster, job_start_cluster + self.args.clusters_per_job)
        print(
            f"This task will process {num_clusters_per_task} clusters: cluster {start_cluster} to cluster {end_cluster}"
        )
        print(
            f"This task will process cluster {start_cluster} to cluster {end_cluster}"
        )

        self._process_shard(start_cluster, end_cluster)
