"""
联邦学习自动调制识别主程序
支持 FedAvg, FedProx, FedGen 三种算法
"""

import argparse
import torch
import os
import sys
from datetime import datetime

# 导入数据加载器
from dataset.data_loader import get_dataloaders

# 导入模型
from FLAlgorithms.trainmodel.models import get_model
from FLAlgorithms.trainmodel.generator import Generator
from FLAlgorithms.trainmodel.diffusion import DiffusionAligner

# 导入客户端
from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.users.userFedProx import UserFedProx
from FLAlgorithms.users.userpFedGen import UserFedGen
from FLAlgorithms.users.userFedDiff import UserFedDiff
from FLAlgorithms.users.userFDAM import UserFDAM

# 导入服务器
from FLAlgorithms.servers.serveravg import ServerAVG
from FLAlgorithms.servers.serverFedProx import ServerFedProx
from FLAlgorithms.servers.serverpFedGen import ServerFedGen
from FLAlgorithms.servers.serverfeddiff import ServerFedDiff
from FLAlgorithms.servers.serverFDAM import ServerFDAM

# 导入工具函数
from utils.model_config import get_dataset_config
from utils.model_utils import setup_logger, save_logs, save_model


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='联邦学习自动调制识别')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='RML2016.10a',
                       choices=['RML2016.10a', 'RML2016.10b', 'RML2018a', 'HisarMod'],
                       help='数据集名称')
    parser.add_argument('--data_snr', type=str, default='100dB',
                       help='SNR标识（如 10dB, 100dB, highsnr）')
    parser.add_argument('--data_dir', type=str, default='data_processed',
                       help='预处理数据目录')
    
    # 算法参数
    parser.add_argument('--algorithm', type=str, default='FDAM',
                       choices=['FedAvg', 'FedProx', 'FedGen', 'FedDiff', 'FDAM'],
                       help='联邦学习算法')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='CNN1D',
                       choices=['CNN1D', 'ResNet1D', 'MCLDNN'],
                       help='模型架构')
    
    # 联邦学习参数
    parser.add_argument('--num_clients', type=int, default=5,
                       help='客户端数量')
    parser.add_argument('--num_rounds', type=int, default=30,
                       help='训练轮次')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='本地训练轮数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    
    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['sgd', 'adam', 'adamw'],
                       help='优化器类型（默认：adam）')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD 动量参数（仅当 optimizer=sgd 时有效）')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减（L2 正则化）')
    
    # Non-IID 参数
    parser.add_argument('--non_iid_type', type=str, default='class',
                       choices=['iid', 'class', 'snr'],
                       help='数据划分类型')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Dirichlet 参数（用于 class Non-IID）')
    
    # FedProx 参数
    parser.add_argument('--mu', type=float, default=0.01,
                       help='FedProx 近端项系数')
    
    # FedGen 参数
    parser.add_argument('--gen_learning_rate', type=float, default=0.001,
                       help='生成器学习率')
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='潜在向量维度')
                       
    # FedDiff 参数
    parser.add_argument('--pseudo_batch_size', type=int, default=16,
                       help='扩散生成伪样本的批大小')
    parser.add_argument('--distill_steps', type=int, default=5,
                       help='每轮扩散蒸馏步数')
    parser.add_argument('--distill_lr', type=float, default=0.001,
                       help='蒸馏阶段学习率')
    parser.add_argument('--pseudo_start_round', type=int, default=12,
                       help='从第几轮开始使用生成伪样本训练全局模型（FedDiff 稳定性）')
    parser.add_argument('--pseudo_conf_thresh', type=float, default=0.9,
                       help='仅对置信度高于阈值的伪样本进行自训练')
    parser.add_argument('--pseudo_loss_weight', type=float, default=0.1,
                       help='伪样本自训练损失权重，避免破坏真实数据学习')
    parser.add_argument('--pseudo_ramp_rounds', type=int, default=8,
                       help='伪样本蒸馏的热身轮数，逐步增加步数与权重以防早期破坏')
    # parser.add_argument('--diffusion_steps', type=int, default=50,
    #                    help='扩散时间步数')

    # FDAM 参数
    parser.add_argument('--diffusion_steps', type=int, default=10,
                       help='FDAM 扩散步数（用于控制对齐难度的超参，当前实现为单步去噪）')
    parser.add_argument('--align_hidden', type=int, default=256,
                       help='FDAM 对齐模块隐藏维度')
    parser.add_argument('--lambda_diff', type=float, default=0.5,
                       help='FDAM 噪声预测损失权重')
    parser.add_argument('--lambda_align', type=float, default=0.5,
                       help='FDAM 原型对齐损失权重')
    parser.add_argument('--align_beta', type=float, default=0.5,
                       help='FDAM 聚合时基于分布偏移的惩罚系数')
    parser.add_argument('--align_noise_std', type=float, default=0.1,
                       help='FDAM 特征加噪标准差')
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='输出目录')
    
    # 设备参数
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='指定使用的GPU ID')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备 (cuda 或 cpu)')
    
    # 随机种子
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()



def set_seed(seed):
    """设置随机种子"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_users(algorithm, num_clients, model, train_loaders, args):
    """
    创建客户端列表
    
    Args:
        algorithm: 算法名称
        num_clients: 客户端数量
        model: 模型
        train_loaders: 训练数据加载器列表
        args: 参数
        
    Returns:
        客户端列表
    """
    users = []
    
    if algorithm == 'FedAvg':
        for i in range(num_clients):
            user = UserAVG(
                user_id=i,
                model=get_model(args.model, model.num_classes, model.signal_length),
                train_loader=train_loaders[i],
                learning_rate=args.learning_rate,
                device=args.device,
                optimizer_type=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
            users.append(user)
    
    elif algorithm == 'FedProx':
        for i in range(num_clients):
            user = UserFedProx(
                user_id=i,
                model=get_model(args.model, model.num_classes, model.signal_length),
                train_loader=train_loaders[i],
                learning_rate=args.learning_rate,
                mu=args.mu,
                device=args.device,
                optimizer_type=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
            users.append(user)
    
    elif algorithm == 'FedGen':
        for i in range(num_clients):
            # 创建生成器（嵌入维度设为 256，与分类器 fc1 层一致）
            generator = Generator(
                latent_dim=args.latent_dim,
                embedding_dim=256,
                hidden_dim=512
            )
            
            user = UserFedGen(
                user_id=i,
                model=get_model(args.model, model.num_classes, model.signal_length),
                generator=generator,
                train_loader=train_loaders[i],
                learning_rate=args.learning_rate,
                gen_learning_rate=args.gen_learning_rate,
                device=args.device,
                latent_dim=args.latent_dim,
                optimizer_type=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
            users.append(user)

    elif algorithm == 'FedDiff':
        for i in range(num_clients):
            user = UserFedDiff(
                user_id=i,
                model=get_model(args.model, model.num_classes, model.signal_length),
                train_loader=train_loaders[i],
                learning_rate=args.learning_rate,
                device=args.device,
                optimizer_type=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                diffusion_steps=args.diffusion_steps,
                gen_learning_rate=args.distill_lr
            )
            users.append(user)

    elif algorithm == 'FDAM':
        with torch.no_grad():
            dummy = torch.zeros(1, 2, model.signal_length)
            feat_dim = model.extract_features(dummy).shape[-1]
        for i in range(num_clients):
            aligner = DiffusionAligner(
                feature_dim=feat_dim,
                hidden_dim=args.align_hidden,
                diffusion_steps=args.diffusion_steps
            )
            user = UserFDAM(
                user_id=i,
                model=get_model(args.model, model.num_classes, model.signal_length),
                aligner=aligner,
                train_loader=train_loaders[i],
                learning_rate=args.learning_rate,
                device=args.device,
                optimizer_type=args.optimizer,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                lambda_diff=args.lambda_diff,
                lambda_align=args.lambda_align,
                mu=args.mu,
                noise_std=args.align_noise_std
            )
            users.append(user)
    return users


def create_server(algorithm, model, users, args):
    """
    创建服务器
    
    Args:
        algorithm: 算法名称
        model: 全局模型
        users: 客户端列表
        args: 参数
        
    Returns:
        服务器实例
    """
    if algorithm == 'FedAvg':
        server = ServerAVG(
            model=model,
            users=users,
            num_rounds=args.num_rounds,
            device=args.device
        )
    
    elif algorithm == 'FedProx':
        server = ServerFedProx(
            model=model,
            users=users,
            num_rounds=args.num_rounds,
            device=args.device
        )
    
    elif algorithm == 'FedGen':
        # 创建全局生成器
        generator = Generator(
            latent_dim=args.latent_dim,
            embedding_dim=256,
            hidden_dim=512
        )
        
        server = ServerFedGen(
            model=model,
            generator=generator,
            users=users,
            num_rounds=args.num_rounds,
            device=args.device
        )

    elif algorithm == 'FedDiff':
        server = ServerFedDiff(
            model=model,
            users=users,
            num_rounds=args.num_rounds,
            device=args.device,
            pseudo_batch_size=args.pseudo_batch_size,
            distill_steps=args.distill_steps,
            distill_lr=args.distill_lr,
            diffusion_steps=args.diffusion_steps,
            pseudo_start_round=args.pseudo_start_round,
            pseudo_conf_thresh=args.pseudo_conf_thresh,
            pseudo_loss_weight=args.pseudo_loss_weight,
            pseudo_ramp_rounds=args.pseudo_ramp_rounds,
        )

    elif algorithm == 'FDAM':
        with torch.no_grad():
            dummy = torch.zeros(1, 2, model.signal_length)
            feat_dim = model.extract_features(dummy).shape[-1]
        aligner = DiffusionAligner(
            feature_dim=feat_dim,
            hidden_dim=args.align_hidden,
            diffusion_steps=args.diffusion_steps
        )
        server = ServerFDAM(
            model=model,
            aligner=aligner,
            users=users,
            num_rounds=args.num_rounds,
            num_classes=model.num_classes,
            feature_dim=feat_dim,
            align_beta=args.align_beta,
            device=args.device
        )
        
    return server


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置GPU
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        args.device = f'cuda:{args.gpu_id}'
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建临时输出目录（带时间戳）
    timestamp = datetime.now().strftime('%m%d%H%M')
    temp_output_dir = os.path.join(args.output_dir, f'temp_{timestamp}')
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(temp_output_dir, 
                           f"{args.dataset}_{args.algorithm}_log.txt")
    logger = setup_logger(log_file)
    
    logger.info("=" * 80)
    logger.info("联邦学习自动调制识别")
    logger.info("=" * 80)
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"SNR: {args.data_snr}")
    logger.info(f"划分类型: {args.non_iid_type}")
    logger.info(f"狄利克雷参数: {args.alpha}")
    logger.info(f"算法: {args.algorithm}")
    logger.info(f"模型: {args.model}")
    logger.info(f"客户端数量: {args.num_clients}")
    logger.info(f"训练轮次: {args.num_rounds}")
    logger.info(f"本地训练轮数: {args.local_epochs}")
    logger.info(f"批大小: {args.batch_size}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"优化器: {args.optimizer.upper()}")
    if args.optimizer.lower() == 'sgd':
        logger.info(f"  - 动量: {args.momentum}")
    logger.info(f"  - 权重衰减: {args.weight_decay}")
    logger.info(f"Non-IID 类型: {args.non_iid_type}")
    logger.info(f"设备: {args.device}")
    logger.info("=" * 80)
    
    # 加载数据集
    logger.info("加载数据集...")
    train_loaders, test_loader, num_classes = get_dataloaders(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        non_iid_type=args.non_iid_type,
        alpha=args.alpha,
        data_snr=args.data_snr,
        data_dir=args.data_dir
    )
    
    # 获取数据集配置
    config = get_dataset_config(args.dataset)
    signal_length = config['signal_length']
    
    logger.info(f"类别数: {num_classes}")
    logger.info(f"信号长度: {signal_length}")
    
    # 创建全局模型
    logger.info("创建模型...")
    global_model = get_model(args.model, num_classes, signal_length)
    logger.info(f"模型参数数量: {sum(p.numel() for p in global_model.parameters())}")
    
    # 创建客户端
    logger.info("创建客户端...")
    users = create_users(args.algorithm, args.num_clients, global_model, train_loaders, args)
    logger.info(f"已创建 {len(users)} 个客户端")
    
    # 创建服务器
    logger.info("创建服务器...")
    server = create_server(args.algorithm, global_model, users, args)
    
    # 开始训练
    logger.info("=" * 80)
    logger.info("开始训练...")
    logger.info("=" * 80)
    
    server.train(test_loader, args.local_epochs, logger)
    
    # 保存结果
    logger.info("=" * 80)
    logger.info("保存结果...")
    
    # 保存 CSV 日志
    csv_file = os.path.join(temp_output_dir, 
                           f"{args.dataset}_{args.algorithm}_metrics.csv")
    
    # 先写表头
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Round', 'Accuracy', 'Loss'])
    
    # 写入数据
    losses, accuracies = server.get_train_history()
    for round_num in range(len(losses)):
        save_logs(csv_file, round_num + 1, accuracies[round_num], losses[round_num], mode='a')
    
    logger.info(f"训练指标已保存到: {csv_file}")
    
    # 保存模型
    model_file = os.path.join(temp_output_dir, 
                             f"{args.dataset}_{args.algorithm}_model.pt")
    save_model(server.model, model_file)
    logger.info(f"模型已保存到: {model_file}")
    
    # 输出最终结果
    final_acc = accuracies[-1]
    final_loss = losses[-1]
    logger.info("=" * 80)
    logger.info(f"训练完成！")
    logger.info(f"最终测试准确率: {final_acc:.2f}%")
    logger.info(f"最终测试损失: {final_loss:.4f}")
    logger.info("=" * 80)
    
    # 关闭所有日志处理器，释放文件句柄
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # 构建最终文件夹名称并重命名（移除%符号避免Windows问题）
    params_str = f"{args.num_clients}_{args.num_rounds}_{args.local_epochs}_{args.learning_rate}_{args.batch_size}"
    final_dirname = f"{args.dataset}_{args.data_snr}_{args.non_iid_type}_{args.alpha}_{args.algorithm}_{args.model}_{final_acc:.2f}_{params_str}_{timestamp}"
    final_output_dir = os.path.join(args.output_dir, final_dirname)
    
    # 重命名临时目录为最终目录
    try:
        os.rename(temp_output_dir, final_output_dir)
        print(f"结果已保存到: {final_output_dir}")
    except PermissionError as e:
        print(f"警告：无法重命名目录 {temp_output_dir} -> {final_output_dir}")
        print(f"错误信息: {e}")
        print(f"结果保存在临时目录: {temp_output_dir}")
        final_output_dir = temp_output_dir
    
    print("=" * 80)


if __name__ == '__main__':
    main()

