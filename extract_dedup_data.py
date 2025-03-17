# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
这个模块 extract_dedup_data.py 的主要功能是从已排序的簇数据和去重表中提取需要保留或移除的示例，
并将这些示例的路径保存到一个文本文件中。以下是对代码的详细解释
"""
import os
from tqdm import tqdm
import pickle
import numpy as np
from constants import IMAGE_NAME_INDEX

#提取修建的数据
def extract_pruned_data(
    sorted_clusters_path,
    semdedup_pruning_tables_path,
    eps,
    num_clusters,
    output_txt_path,
    retreive_kept_samples=True,
):

    ## -- list of paths to the examples we want to keep/remove.
    ## -- 保存我们想要保留/移除的示例的路径列表。
    example_paths = []

    for cluster_id in tqdm(range(0, num_clusters)):

        cluster_i = np.load(
            os.path.join(sorted_clusters_path, f"cluster_{cluster_id}.npy")
        )
        with open(
            f"{semdedup_pruning_tables_path}/cluster_{cluster_id}.pkl", "rb"
        ) as file:
            semdedup_pruning_tables = pickle.load(file)

        ## -- See which examples to keep/remove from this cluster.
        ## -- 查看要从此群集中保留/移除的示例。
        ## -- Use retreive_kept_samples=True when kept dataset size <= 50%. This will return a smaller output text file,
        ## -- semdedup_pruning_tables contain True values for the examples to be removed.
        ## -- 当保留的数据集大小<=50%时，使用retreive_kept_samples=True。这将返回一个较小的输出文本文件，
        ## -- semdedup_pruning_tables包含要移除的示例的True值。
        images_to_keep_or_remove = semdedup_pruning_tables[f"eps={eps}"][
            semdedup_pruning_tables[f"eps={eps}"] == (not retreive_kept_samples)
        ].index.to_numpy()
        if "indices" in semdedup_pruning_tables.columns:
            cluster_i = cluster_i[semdedup_pruning_tables["indices"]]
        ## -- retrieve only the examples we want and add to the list.
        ## -- 仅检索我们想要的示例并添加到列表中。
        dedup_cluster = cluster_i[images_to_keep_or_remove]
        example_paths += dedup_cluster[:, IMAGE_NAME_INDEX].astype("<U32").tolist()#每个字符串最多包含32个UNicode字符

    with open(output_txt_path, "w") as fp:
        fp.write("\n".join(example_paths))#每个路径之间用换行符分隔

    print(f"DONE saving {len(example_paths)} image paths")

    return
