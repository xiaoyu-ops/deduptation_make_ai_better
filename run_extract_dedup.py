from extract_dedup_data import extract_pruned_data
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="提取去重后的数据")
    parser.add_argument("--sorted-clusters", type=str, 
                        default="clustering/results/sorted_clusters", 
                        help="排序后的簇数据路径")
    parser.add_argument("--pruning-tables", type=str, 
                        default="clustering/results/semdedup_result/dataframes", 
                        help="去重表的路径")
    parser.add_argument("--epsilon", type=float, default=0.1, 
                        help="使用的epsilon值")
    parser.add_argument("--num-clusters", type=int, default=599, 
                        help="簇的数量")
    parser.add_argument("--output", type=str, default="deduped_image_paths.txt", 
                        help="输出文本文件的路径")
    parser.add_argument("--keep", action="store_true", default=True, 
                        help="是否提取保留的样本（默认为True）")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    print(f"排序后的簇路径: {args.sorted_clusters}")
    print(f"去重表路径: {args.pruning_tables}")
    print(f"Epsilon值: {args.epsilon}")
    print(f"簇数量: {args.num_clusters}")
    print(f"输出路径: {args.output}")
    print(f"提取保留的样本: {args.keep}")
    
    extract_pruned_data(
        sorted_clusters_path=args.sorted_clusters,
        semdedup_pruning_tables_path=args.pruning_tables,
        eps=args.epsilon,
        num_clusters=args.num_clusters,
        output_txt_path=args.output,
        retreive_kept_samples=args.keep
    )
    
    print(f"已成功提取数据到 {args.output}")
    
    # 额外统计保留图像数量
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            lines = f.readlines()
        print(f"保留的样本数量: {len(lines)}")
    else:
        print(f"警告: 输出文件 {args.output} 不存在!")

if __name__ == "__main__":
    main()