"""
按SNR评估模型性能
在所有单独SNR测试集上评估已训练的模型，并绘制准确率vs.SNR曲线
"""

import argparse
import os
import re
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import csv

# 导入模型和配置
from FLAlgorithms.trainmodel.models import get_model
from utils.model_config import get_dataset_config
from utils.model_utils import setup_logger


class SignalDataset(Dataset):
    """信号数据集类"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def infer_model_from_path(model_path):
    """
    从模型路径推断模型架构
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        模型名称 ('CNN1D', 'ResNet1D' 或 'MCLDNN')，如果无法推断则返回 None
    """
    # 从父文件夹名称中查找模型架构
    parent_dir = os.path.basename(os.path.dirname(model_path))
    
    if 'MCLDNN' in parent_dir or 'mcldnn' in parent_dir.lower():
        return 'MCLDNN'
    elif 'CNN1D' in parent_dir or 'cnn1d' in parent_dir.lower():
        return 'CNN1D'
    elif 'ResNet1D' in parent_dir or 'resnet1d' in parent_dir.lower():
        return 'ResNet1D'
    
    return None


def load_snr_test_data(data_dir, dataset_name):
    """
    加载所有单独SNR的测试数据
    
    Args:
        data_dir: 数据目录
        dataset_name: 数据集名称
        
    Returns:
        snr_data: 字典，键为SNR值，值为(X_test, y_test)
    """
    test_dir = os.path.join(data_dir, dataset_name, 'test')
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"测试数据目录不存在: {test_dir}")
    
    # 获取所有测试文件
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.pkl')]
    
    snr_data = {}
    
    for file in test_files:
        # 提取SNR值（跳过合并文件）
        if file in ['100dB.pkl', 'highsnr.pkl']:
            continue
        
        # 从文件名提取SNR值（如 "-20dB.pkl" -> -20）
        match = re.match(r'(-?\d+)dB\.pkl', file)
        if match:
            snr = int(match.group(1))
            
            # 加载数据
            file_path = os.path.join(test_dir, file)
            with open(file_path, 'rb') as f:
                loaded = pickle.load(f)
                if len(loaded) == 3:
                    X_test, y_test, _ = loaded
                else:
                    X_test, y_test = loaded
            
            snr_data[snr] = (X_test, y_test)
    
    return snr_data


def evaluate_model(model, X, y, device, batch_size=256):
    """
    评估模型在给定数据上的准确率
    
    Args:
        model: PyTorch 模型
        X: 测试数据
        y: 测试标签
        device: 设备
        batch_size: 批大小
        
    Returns:
        accuracy: 准确率（百分比）
    """
    model.eval()
    
    # 创建数据加载器
    dataset = SignalDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def plot_snr_accuracy(snr_list, accuracy_list, output_path, dataset_name, algorithm_name):
    """
    绘制SNR vs. Accuracy曲线
    
    Args:
        snr_list: SNR值列表
        accuracy_list: 准确率列表
        output_path: 输出图像路径
        dataset_name: 数据集名称
        algorithm_name: 算法名称
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制曲线
    plt.plot(snr_list, accuracy_list, marker='o', linewidth=2, markersize=6, 
             label=f'{algorithm_name} on {dataset_name}')
    
    # 设置标签和标题
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Classification Accuracy vs. SNR\n{dataset_name} - {algorithm_name}', 
              fontsize=14, fontweight='bold')
    
    # 设置网格
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 设置图例
    plt.legend(loc='lower right', fontsize=10)
    
    # 设置Y轴范围
    plt.ylim(0, 100)
    
    # 设置X轴范围（根据SNR值）
    if snr_list:
        plt.xlim(min(snr_list) - 2, max(snr_list) + 2)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  图像已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='按SNR评估模型性能')
    
    parser.add_argument('--model_path', type=str, default='results\RML2016.10a_10dB_FedAvg_9.09pct_10_0.001_128_5_10231051\RML2016.10a_FedAvg_model.pt',
                       help='训练好的模型文件路径（.pt）')
    parser.add_argument('--dataset_name', type=str, default='RML2016.10a',
                       choices=['RML2016.10a', 'RML2016.10b', 'RML2018a', 'HisarMod'],
                       help='数据集名称')
    parser.add_argument('--model', type=str, default='MCLDNN',
                       choices=['CNN1D', 'ResNet1D', 'MCLDNN'],
                       help='模型架构（可选，默认从路径推断）')
    parser.add_argument('--data_dir', type=str, default='data_processed',
                       help='预处理数据目录')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='评估批大小')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备 (cuda 或 cpu)')
    
    args = parser.parse_args()
    
    # 推断模型架构（如果未提供）
    if args.model is None:
        args.model = infer_model_from_path(args.model_path)
        if args.model is None:
            raise ValueError("无法从路径推断模型架构，请使用 --model 参数指定")
        print(f"从路径推断模型架构: {args.model}")
    
    # 获取数据集配置
    config = get_dataset_config(args.dataset_name)
    num_classes = config['num_classes']
    signal_length = config['signal_length']
    
    print("=" * 80)
    print("按SNR评估模型性能")
    print("=" * 80)
    print(f"模型路径: {args.model_path}")
    print(f"数据集: {args.dataset_name}")
    print(f"模型架构: {args.model}")
    print(f"类别数: {num_classes}")
    print(f"信号长度: {signal_length}")
    print(f"设备: {args.device}")
    print("=" * 80)
    
    # 创建模型并加载权重
    print("\n加载模型...")
    model = get_model(args.model, num_classes, signal_length)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    model.eval()
    print("模型加载完成")
    
    # 加载测试数据
    print(f"\n加载测试数据从 {args.data_dir}...")
    snr_data = load_snr_test_data(args.data_dir, args.dataset_name)
    print(f"找到 {len(snr_data)} 个SNR测试集")
    
    # 对每个SNR评估
    print("\n开始评估...")
    results = []
    
    for snr in sorted(snr_data.keys()):
        X_test, y_test = snr_data[snr]
        accuracy = evaluate_model(model, X_test, y_test, args.device, args.batch_size)
        results.append((snr, accuracy))
        print(f"  SNR = {snr:3d}dB: Accuracy = {accuracy:.2f}% (Samples: {len(y_test)})")
    
    # 排序结果
    results.sort(key=lambda x: x[0])
    snr_list = [r[0] for r in results]
    accuracy_list = [r[1] for r in results]
    
    # 创建输出目录（在模型路径的父目录下创建test文件夹）
    model_parent_dir = os.path.dirname(args.model_path)
    output_dir = os.path.join(model_parent_dir, 'test')
    os.makedirs(output_dir, exist_ok=True)
    
    # 从模型路径获取基本名称
    parent_dir = os.path.basename(model_parent_dir)
    base_name = parent_dir if parent_dir != 'results' else args.dataset_name
    
    # 保存CSV
    csv_file = os.path.join(output_dir, f'{base_name}_snr_metrics.csv')
    print(f"\n保存结果到 {output_dir}...")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR', 'Accuracy'])
        for snr, acc in results:
            writer.writerow([snr, acc])
    print(f"  CSV已保存到: {csv_file}")
    
    # 保存日志
    log_file = os.path.join(output_dir, f'{base_name}_snr_log.txt')
    logger = setup_logger(log_file, name='SNR-Test')
    logger.info("=" * 80)
    logger.info("按SNR评估模型性能")
    logger.info("=" * 80)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"数据集: {args.dataset_name}")
    logger.info(f"模型架构: {args.model}")
    logger.info("=" * 80)
    logger.info("评估结果:")
    for snr, acc in results:
        logger.info(f"  SNR = {snr:3d}dB: Accuracy = {acc:.2f}%")
    logger.info("=" * 80)
    print(f"  日志已保存到: {log_file}")
    
    # 绘制图像
    # 从父目录名称推断算法名称
    algorithm_name = 'Unknown'
    if 'FedAvg' in parent_dir:
        algorithm_name = 'FedAvg'
    elif 'FedProx' in parent_dir:
        algorithm_name = 'FedProx'
    elif 'FedGen' in parent_dir:
        algorithm_name = 'FedGen'
    
    plot_file = os.path.join(output_dir, f'{base_name}_snr.png')
    plot_snr_accuracy(snr_list, accuracy_list, plot_file, 
                     args.dataset_name, algorithm_name)
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print("评估完成！")
    print(f"平均准确率: {np.mean(accuracy_list):.2f}%")
    print(f"最高准确率: {np.max(accuracy_list):.2f}% (SNR = {snr_list[np.argmax(accuracy_list)]}dB)")
    print(f"最低准确率: {np.min(accuracy_list):.2f}% (SNR = {snr_list[np.argmin(accuracy_list)]}dB)")
    print("=" * 80)


if __name__ == '__main__':
    main()

