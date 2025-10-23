"""
绘图工具函数
用于可视化训练结果
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


def plot_accuracy_curves(logs_dict, title='Accuracy Comparison', save_path=None):
    """
    绘制多个算法的准确率对比曲线
    
    Args:
        logs_dict: 字典，key 为算法名称，value 为准确率列表
        title: 图表标题
        save_path: 保存路径（如果为 None，则显示图表）
    """
    plt.figure(figsize=(10, 6))
    
    for algorithm, accuracies in logs_dict.items():
        rounds = list(range(1, len(accuracies) + 1))
        plt.plot(rounds, accuracies, marker='o', label=algorithm, linewidth=2)
    
    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"准确率曲线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_curves(logs_dict, title='Loss Comparison', save_path=None):
    """
    绘制多个算法的损失对比曲线
    
    Args:
        logs_dict: 字典，key 为算法名称，value 为损失值列表
        title: 图表标题
        save_path: 保存路径（如果为 None，则显示图表）
    """
    plt.figure(figsize=(10, 6))
    
    for algorithm, losses in logs_dict.items():
        rounds = list(range(1, len(losses) + 1))
        plt.plot(rounds, losses, marker='s', label=algorithm, linewidth=2)
    
    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失曲线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_combined_curves(logs_dict_acc, logs_dict_loss, title='Training Metrics', save_path=None):
    """
    在一张图中同时绘制准确率和损失曲线
    
    Args:
        logs_dict_acc: 准确率字典
        logs_dict_loss: 损失字典
        title: 图表标题
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制准确率
    for algorithm, accuracies in logs_dict_acc.items():
        rounds = list(range(1, len(accuracies) + 1))
        ax1.plot(rounds, accuracies, marker='o', label=algorithm, linewidth=2)
    
    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制损失
    for algorithm, losses in logs_dict_loss.items():
        rounds = list(range(1, len(losses) + 1))
        ax2.plot(rounds, losses, marker='s', label=algorithm, linewidth=2)
    
    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Comparison', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"组合曲线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称列表
        title: 图表标题
        save_path: 保存路径（如果为 None，则显示图表）
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 设置图表大小
    plt.figure(figsize=(12, 10))
    
    # 绘制热图
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_data_distribution(client_distributions, dataset_name, save_path=None):
    """
    绘制客户端数据分布
    
    Args:
        client_distributions: 字典，key 为客户端 ID，value 为类别分布
        dataset_name: 数据集名称
        save_path: 保存路径
    """
    num_clients = len(client_distributions)
    
    plt.figure(figsize=(14, 8))
    
    for client_id, dist in client_distributions.items():
        classes = list(dist.keys())
        counts = list(dist.values())
        plt.bar(np.arange(len(classes)) + client_id * 0.1, counts, 
                width=0.1, label=f'Client {client_id}', alpha=0.7)
    
    plt.xlabel('Class Index', fontsize=12)
    plt.ylabel('Sample Count', fontsize=12)
    plt.title(f'Data Distribution Across Clients - {dataset_name}', fontsize=14)
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"数据分布图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

