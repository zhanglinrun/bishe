import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def visualize_client_data_distribution(client_data_map, num_classes, dataset_name, alpha, save_dir="results/plots"):
    """
    绘制联邦学习客户端数据分布热力图 (Bubble Chart / Heatmap style)
    这种图在联邦学习论文中非常常见。
    
    Args:
        client_data_map: dict or list, 客户端数据索引或标签
                         如果是 list, 每个元素是 (X, y) 或 (X, y, snr)
        num_classes: 类别总数
        alpha: Dirichlet 参数
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_clients = len(client_data_map)
    heatmap_data = np.zeros((num_classes, num_clients))

    # 统计每个客户端的每个类别的样本数量
    for i in range(num_clients):
        # 兼容你的 dataloader 返回格式 (X, y) 或 (X, y, snr)
        if isinstance(client_data_map, list):
            data_tuple = client_data_map[i]
            if len(data_tuple) >= 2:
                y_local = data_tuple[1]
            else:
                continue
        else:
            # 假设是字典格式
            y_local = client_data_map[i][1]
            
        unique, counts = np.unique(y_local, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < num_classes:
                heatmap_data[cls, i] = count

    # 归一化 (可选，为了颜色更好看)
    # heatmap_data_norm = heatmap_data / (heatmap_data.sum(axis=0, keepdims=True) + 1e-5)

    plt.figure(figsize=(12, 8))
    
    # 使用 Seaborn 绘制热力图
    # Y轴: 类别 (Modulation Types)
    # X轴: 客户端 ID
    sns.heatmap(heatmap_data, annot=True, fmt='g', cmap='Blues', cbar_kws={'label': 'Number of Samples'})
    
    plt.title(f'Data Distribution across Clients\nDataset: {dataset_name}, Non-IID (Dirichlet alpha={alpha})', fontsize=14)
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Modulation Class ID', fontsize=12)
    
    filename = f"{dataset_name}_Dirichlet_{alpha}_distribution.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 数据分布图已保存至: {save_path}")