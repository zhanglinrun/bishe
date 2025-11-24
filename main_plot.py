"""
训练结果可视化
读取训练日志并绘制对比曲线
"""

import argparse
import os
import pandas as pd
from utils.plot_utils import plot_accuracy_curves, plot_loss_curves, plot_combined_curves


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练结果可视化')
    
    parser.add_argument('--dataset', type=str, default='RML2016.10a',
                       help='数据集名称')
    
    parser.add_argument('--metrics_dir', type=str, default='./results',
                       help='指标文件所在目录')
    
    parser.add_argument('--algorithms', type=str, nargs='+', 
                       default=['FedAvg', 'FedProx', 'FedGen'],
                       help='要对比的算法列表')
    
    parser.add_argument('--output_dir', type=str, default='./results/plots',
                       help='图表保存目录')
    
    return parser.parse_args()


def load_metrics(metrics_file):
    """
    加载训练指标
    
    Args:
        metrics_file: CSV 文件路径
        
    Returns:
        rounds: 轮次列表
        accuracies: 准确率列表
        losses: 损失列表
    """
    if not os.path.exists(metrics_file):
        print(f"警告: 文件不存在 {metrics_file}")
        return None, None, None
    
    try:
        df = pd.read_csv(metrics_file)
        rounds = df['Round'].tolist()
        accuracies = df['Accuracy'].tolist()
        losses = df['Loss'].tolist()
        return rounds, accuracies, losses
    except Exception as e:
        print(f"错误: 无法读取文件 {metrics_file}: {e}")
        return None, None, None


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("训练结果可视化")
    print("=" * 80)
    print(f"数据集: {args.dataset}")
    print(f"对比算法: {', '.join(args.algorithms)}")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载所有算法的指标
    acc_dict = {}
    loss_dict = {}
    
    for algorithm in args.algorithms:
        metrics_file = os.path.join(args.metrics_dir, 
                                   f"{args.dataset}_{algorithm}_metrics.csv")
        
        print(f"\n加载 {algorithm} 的指标...")
        rounds, accuracies, losses = load_metrics(metrics_file)
        
        if accuracies is not None and losses is not None:
            acc_dict[algorithm] = accuracies
            loss_dict[algorithm] = losses
            print(f"  ✓ 成功加载 {len(accuracies)} 轮数据")
            print(f"  最终准确率: {accuracies[-1]:.2f}%")
            print(f"  最终损失: {losses[-1]:.4f}")
        else:
            print(f"  ✗ 无法加载数据")
    
    # 检查是否有数据
    if not acc_dict:
        print("\n错误: 没有找到任何有效的训练指标文件")
        print(f"请确保在 {args.metrics_dir} 目录下存在以下格式的文件:")
        print(f"  {args.dataset}_<algorithm>_metrics.csv")
        return
    
    print("\n" + "=" * 80)
    print("开始绘图...")
    print("=" * 80)
    
    # 绘制准确率对比曲线
    acc_plot_path = os.path.join(args.output_dir, 
                                f"{args.dataset}_accuracy_comparison.png")
    print(f"\n绘制准确率对比曲线...")
    plot_accuracy_curves(
        logs_dict=acc_dict,
        title=f'Accuracy Comparison - {args.dataset}',
        save_path=acc_plot_path
    )
    
    # 绘制损失对比曲线
    loss_plot_path = os.path.join(args.output_dir, 
                                 f"{args.dataset}_loss_comparison.png")
    print(f"绘制损失对比曲线...")
    plot_loss_curves(
        logs_dict=loss_dict,
        title=f'Loss Comparison - {args.dataset}',
        save_path=loss_plot_path
    )
    
    # 绘制组合曲线
    combined_plot_path = os.path.join(args.output_dir, 
                                     f"{args.dataset}_combined_comparison.png")
    print(f"绘制组合对比曲线...")
    plot_combined_curves(
        logs_dict_acc=acc_dict,
        logs_dict_loss=loss_dict,
        title=f'Training Metrics Comparison - {args.dataset}',
        save_path=combined_plot_path
    )
    
    print("\n" + "=" * 80)
    print("绘图完成！")
    print("=" * 80)
    print(f"图表已保存到: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

