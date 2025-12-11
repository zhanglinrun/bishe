"""
数据分布可视化工具
用于生成不同 Non-IID 设置下的数据分布热力图
增加了日志保存功能
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

# 导入数据加载逻辑
from dataset.data_loader import get_dataloaders
from utils.model_config import get_dataset_config
# [新增] 导入日志工具
from utils.model_utils import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='数据分布可视化')
    
    parser.add_argument('--dataset', type=str, default='RML2016.10b',
                       choices=['RML2016.10a', 'RML2016.10b', 'RML2018a', 'HisarMod'],
                       help='数据集名称')
    parser.add_argument('--data_snr', type=str, default='100dB',
                       help='SNR')
    parser.add_argument('--num_clients', type=int, default=5,
                       help='客户端数量')
    parser.add_argument('--non_iid_type', type=str, default='extreme',
                       choices=['iid', 'class', 'snr', 'extreme'],
                       help='数据划分类型')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet 参数')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output_dir', type=str, default='./results/data',
                       help='输出目录')
    
    return parser.parse_args()

def plot_distribution(train_loaders, num_classes, args, logger=None):
    """绘制数据分布热力图并记录日志"""
    num_clients = len(train_loaders)
    client_class_counts = np.zeros((num_clients, num_classes))
    
    msg = "统计客户端数据分布..."
    if logger:
        logger.info(msg)
    else:
        print(msg)

    for i, loader in enumerate(train_loaders):
        labels = []
        for _, y in loader:
            labels.extend(y.numpy())
        
        unique, counts = np.unique(labels, return_counts=True)
        
        # 记录并打印详细分布到日志
        dist_str = f"Client {i}: "
        # 填充矩阵
        for cls, count in zip(unique, counts):
            client_class_counts[i, int(cls)] = count
            dist_str += f"[Class {int(cls)}: {count}] "
        
        if logger:
            logger.info(dist_str)
        else:
            print(dist_str)
            
    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(client_class_counts, annot=True, fmt='g', cmap='YlGnBu',
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Client {i}' for i in range(num_clients)])
    
    plt.title(f'Data Distribution ({args.dataset}, {args.non_iid_type}, alpha={args.alpha})')
    plt.xlabel('Modulation Classes')
    plt.ylabel('Clients')
    
    # 保存图片
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"dist_{args.dataset}_{args.non_iid_type}_a{args.alpha}_c{args.num_clients}_s{args.seed}.png"
    if args.non_iid_type == 'extreme':
        filename = f"dist_{args.dataset}_extreme_c{args.num_clients}_s{args.seed}.png"
        
    save_path = os.path.join(args.output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    msg = f"分布图已保存至: {save_path}"
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    plt.close()

def main():
    args = parse_args()
    
    # [新增] 设置日志保存
    os.makedirs(args.output_dir, exist_ok=True)
    # 根据设置生成日志文件名
    log_filename = f"log_{args.dataset}_{args.non_iid_type}_s{args.seed}.txt"
    log_path = os.path.join(args.output_dir, log_filename)
    logger = setup_logger(log_path)
    
    logger.info("=" * 60)
    logger.info(f"开始可视化: {args.dataset}")
    logger.info(f"划分类型: {args.non_iid_type}")
    logger.info(f"Seed: {args.seed}")
    logger.info("=" * 60)

    # 设置随机种子
    import random
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 获取数据加载器
    # 注意：get_dataloaders 内部的 print 仍然会输出到控制台，但具体的统计数据会通过 plot_distribution 写入日志
    train_loaders, _, num_classes = get_dataloaders(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        batch_size=1024, # 大批次加快统计速度
        non_iid_type=args.non_iid_type,
        alpha=args.alpha,
        data_snr=args.data_snr
    )
    
    plot_distribution(train_loaders, num_classes, args, logger)
    
    # 关闭日志 handler 释放文件
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

if __name__ == '__main__':
    main()