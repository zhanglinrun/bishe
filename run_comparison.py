import subprocess
import re
import numpy as np
import time
import sys

# ================= 配置区域 =================
# 你想要对比的随机种子
SEEDS = [21, 1234, 42, 666, 1004, 2025]
# 你想要对比的 Alpha 值
ALPHAS = [0.5, 0.1, 1.0]

# 固定参数
COMMON_ARGS = [
    "--dataset", "RML2016.10a",
    "--data_snr", "100dB",
    "--algorithm", "FedAvg",
    "--model", "MCLDNN",
    "--num_clients", "5",
    "--num_rounds", "40",
    "--local_epochs", "5",
    "--non_iid_type", "class",
    "--gpu_id", "2"  # 根据你的实际情况修改 GPU ID
]
# ===========================================

def run_experiment(alpha, seed):
    """运行单次实验并解析准确率"""
    # 注意：这里我们只设置 --seed (训练种子)，不设置 --data_seed。
    # 如果你想固定数据划分，可以在这里加上 ["--data_seed", "2024"] 之类的固定值
    cmd = ["python", "main.py"] + COMMON_ARGS + ["--alpha", str(alpha), "--seed", str(seed)]
    print(f"\n[开始实验] Alpha={alpha}, Seed={seed}")
    print(f"指令: {' '.join(cmd)}")
    
    start_time = time.time()
    
    # 运行命令并捕获输出
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='ignore'  # 忽略可能的编码错误
    )
    
    final_acc = None
    
    # 实时读取输出，同时寻找结果
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            # 打印关键日志以显示进度（可选，避免刷屏太快）
            if "Round" in line or "最终" in line or "Error" in line:
                print(line.strip())
            
            # [修改点] 更新正则以匹配 "(Best)"
            # 兼容两种格式：
            # 1. "最终测试准确率: 53.72%"
            # 2. "最终测试准确率 (Best): 53.72%"
            # .* 表示匹配 "准确率" 和 ":" 之间的任意字符（比如 " (Best)"）
            match = re.search(r"最终测试准确率.*:\s*([\d\.]+)%", line)
            if match:
                final_acc = float(match.group(1))
    
    duration = time.time() - start_time
    print(f"[实验结束] 耗时: {duration:.1f}s, 结果: {final_acc}%")
    return final_acc

def main():
    print("==================================================")
    print(f"开始自动化对比实验")
    print(f"Seeds: {SEEDS}")
    print(f"Alphas: {ALPHAS}")
    print("==================================================\n")

    results = {alpha: [] for alpha in ALPHAS}

    for alpha in ALPHAS:
        for seed in SEEDS:
            acc = run_experiment(alpha, seed)
            if acc is not None:
                results[alpha].append(acc)
            else:
                print(f"[警告] Alpha={alpha}, Seed={seed} 未找到准确率结果，可能运行出错。")

    print("\n\n")
    print("==================================================")
    print("                 最终实验报告")
    print("==================================================")
    print(f"{'Alpha':<10} | {'Seeds Results':<40} | {'Mean Acc':<10} | {'Std Dev':<10}")
    print("-" * 80)

    for alpha in ALPHAS:
        accs = results[alpha]
        if accs:
            mean_acc = np.mean(accs)
            std_dev = np.std(accs)
            # 格式化列表字符串
            acc_str = ", ".join([f"{a:.2f}" for a in accs])
            print(f"{alpha:<10} | {acc_str:<40} | {mean_acc:.2f}%     | ±{std_dev:.2f}")
        else:
            print(f"{alpha:<10} | {'No Data':<40} | N/A        | N/A")
    
    print("==================================================")

if __name__ == "__main__":
    main()