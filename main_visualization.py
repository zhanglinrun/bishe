import argparse
import os
import torch
import numpy as np
import random
import sys
from datetime import datetime
from dataset.data_loader import load_preprocessed_data, split_data_non_iid_class
from utils.distribution_vis import visualize_client_data_distribution
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_seed(seed=42):
    """固定随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"随机种子已设置为: {seed}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='RML2016.10a')
    parser.add_argument('--data_dir', type=str, default='data_processed')
    parser.add_argument('--snr', type=str, default='100dB')
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.1, help="Dirichlet alpha for Non-IID")
    parser.add_argument('--seed', type=int, default=42, help="随机种子") # 新增种子参数
    parser.add_argument('--load_diffusion', type=str, default='pretrained_diffusion.pt', help="Path to pre-trained diffusion model")
    args = parser.parse_args()

    # --- 1. 创建带时间戳的输出目录 ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('results', 'data', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 2. 设置日志记录到文件 ---
    log_file = os.path.join(output_dir, 'data_stats.txt')
    # 将 stdout 重定向到 Logger，同时输出到控制台和文件
    sys.stdout = Logger(log_file, sys.stdout)

    print("=" * 60)
    print(f"实验时间: {timestamp}")
    print(f"输出目录: {output_dir}")
    print(f"数据集: {args.dataset}, SNR: {args.snr}")
    print(f"Non-IID 设置: Dirichlet (alpha={args.alpha})")
    print(f"随机种子: {args.seed}")
    print("=" * 60)

    # --- 关键步骤：设置随机种子 ---
    # 必须在调用任何涉及随机性的数据处理函数之前设置
    set_seed(args.seed)

    # --- 3. 加载与划分数据 ---
    print("\n[Step 1] 加载原始数据...")
    # 注意：这里使用的是 args.snr 而不是 args.data_snr
    X_train, y_train, snr_train, _, _, _, num_classes = load_preprocessed_data(
        args.dataset, args.snr, args.data_dir
    )
    
    print(f"\n[Step 2] 执行数据划分 (Clients={args.num_clients})...")
    # split_data_non_iid_class 内部会使用 np.random.dirichlet
    # 因为我们在前面调用了 set_seed(args.seed)，这里的划分结果将是确定的
    client_data = split_data_non_iid_class(
        X_train, y_train, snr_train, 
        num_clients=args.num_clients, 
        alpha=args.alpha
    )
    
    # --- 4. 打印详细统计信息 ---
    print("\n" + "=" * 60)
    print("各客户端数据分布统计")
    print("=" * 60)

    # 统计每个类别的总样本数，用于计算全局比例
    global_cls_counts = np.zeros(num_classes, dtype=int)

    for i, (X_c, y_c, snr_c) in enumerate(client_data):
        unique_labels, label_counts = np.unique(y_c, return_counts=True)
        unique_snrs, snr_counts = np.unique(snr_c, return_counts=True)
        
        # 将 unique_labels 和 label_counts 转为字典以便快速查找
        label_dict = dict(zip(unique_labels, label_counts))
        snr_dict = dict(zip(unique_snrs, snr_counts))
        
        # 更新全局计数
        for cls, cnt in zip(unique_labels, label_counts):
            if cls < num_classes:
                global_cls_counts[cls] += cnt

        print(f"\n>>> 客户端 {i}:")
        print(f"  总样本数: {len(y_c)}")
        print(f"  类别分布 (全部):")
        
        # 按类别索引顺序遍历打印 (0 到 num_classes-1)
        for cls in range(num_classes):
            cnt = label_dict.get(cls, 0)
            pct = cnt / len(y_c) * 100 if len(y_c) > 0 else 0
            # 即使数量为0也打印，这样更直观看到缺失情况
            print(f"    类别 {cls:2d}: {cnt:5d} ({pct:5.1f}%)")
            
        print(f"  SNR分布 (全部):")
        # 按 SNR 值排序打印
        sorted_snrs = sorted(unique_snrs)
        for snr in sorted_snrs:
            cnt = snr_dict.get(snr, 0)
            pct = cnt / len(y_c) * 100 if len(y_c) > 0 else 0
            print(f"    {int(snr):3d}dB: {cnt:5d} ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print("全局类别分布检查")
    print("=" * 60)
    for cls in range(num_classes):
        print(f"类别 {cls:2d}: 总数 {global_cls_counts[cls]}")

    # --- 5. 绘制热力图 ---
    print(f"\n[Step 3] 绘制分布热力图...")
    visualize_client_data_distribution(client_data, num_classes, args.dataset, args.alpha, save_dir=output_dir)
    print(f"热力图已保存至: {output_dir}")

    # --- 6. 扩散模型加载测试 ---
    print("\n" + "=" * 60)
    print("[Step 4] 扩散模型加载测试 (可选)")
    print("=" * 60)
    
    if args.load_diffusion and os.path.exists(args.load_diffusion):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在加载扩散模型: {args.load_diffusion}...")
        
        try:
            # 实例化模型 (参数需与训练时一致，这里假设是默认配置)
            # 如果你在 test_diffusion_gen.py 改了参数，这里也要对应改
            diffusion = DiffusionGenerator(
                num_classes=num_classes,
                signal_length=128,
                base_channels=64,
                channel_mults=(1, 2, 4),
                beta_schedule="cosine"
            ).to(device)
            
            diffusion.load_state_dict(torch.load(args.load_diffusion, map_location=device))
            diffusion.eval()
            print("模型加载成功！")
            
            # 尝试生成
            print("正在生成测试样本 (Batch Size=16)...")
            labels = torch.randint(0, num_classes, (16,), device=device)
            with torch.no_grad():
                # 使用 dynamic_threshold 防止数值爆炸
                fake_data = diffusion.sample(16, labels, device=device, dynamic_threshold=True)
            
            print(f"生成数据形状: {fake_data.shape}")
            print(f"数据统计: Mean={fake_data.mean():.4f}, Std={fake_data.std():.4f}, Max={fake_data.max():.4f}, Min={fake_data.min():.4f}")
            
            # 简单的数值检查
            if torch.isnan(fake_data).any():
                print("警告: 生成数据包含 NaN!")
            else:
                print("生成数据数值正常。")
                
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("请检查模型结构参数是否与训练脚本 (test_diffusion_gen.py) 一致。")
    else:
        if args.load_diffusion:
            print(f"错误: 找不到模型文件 {args.load_diffusion}")
        else:
            print("未指定扩散模型路径 (--load_diffusion)，跳过加载测试。")

    print(f"\n所有任务完成。日志和结果已保存在: {output_dir}")

if __name__ == "__main__":
    main()