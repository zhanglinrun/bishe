import os
import pickle
import numpy as np
import random
from sklearn.model_selection import train_test_split


def generate_dataset_from_RML2016a():
    """
    RML2016.10a数据分割逻辑
    """
    # 设置随机种子确保可复现性
    np.random.seed(42)
    random.seed(42)
    
    # 创建输出目录结构
    base_dir = r"D:\数据集\RML2016A_dB_Semi"
    os.makedirs(os.path.join(base_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test"), exist_ok=True)

    # 加载原始数据
    xd = pickle.load(open(r"D:\数据集\RML2016.10a\RML2016.10a_dict.pkl", 'rb'), encoding='iso-8859-1')

    # 获取所有调制方式和SNR值
    mods = sorted(list(set([k[0] for k in xd.keys()])))
    snrs = sorted(list(set([k[1] for k in xd.keys()])))
    
    print(f"调制方式: {mods}")
    print(f"SNR值: {snrs}")
    
    # 对每个SNR独立进行训练/测试分割
    for snr in snrs:
        print(f"\n处理 SNR = {snr}dB 的数据")
        
        # 收集该SNR下所有调制的数据
        all_samples = []
        all_labels = []
        
        for mod in mods:
            mod_idx = mods.index(mod)
            samples = xd[(mod, snr)]
            
            # 确保数据形状为 (N, 2, 128)
            if len(samples.shape) == 2:
                samples = samples.reshape(-1, 2, 128)
            
            all_samples.extend(samples)
            all_labels.extend([mod_idx] * len(samples))
        
        all_samples = np.array(all_samples)
        all_labels = np.array(all_labels)
        
        print(f"总样本数: {len(all_samples)}")
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"原始分布: {dict(zip(unique, counts))}")
        
        # 分层抽样确保训练集和测试集的类别分布一致
        # 对于RML2016.10a，这已经足够确保平衡，无需额外的balance_classes
        train_samples, test_samples, train_labels, test_labels = train_test_split(
            all_samples, all_labels, 
            test_size=0.4, 
            stratify=all_labels,  # 分层抽样
            random_state=42
        )
        
        # 验证分割后的类别分布
        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        print(f"训练集分布: {dict(zip(train_unique, train_counts))}")
        print(f"测试集分布: {dict(zip(test_unique, test_counts))}")
        
        # 可选：额外打乱（虽然train_test_split已经打乱了）
        train_indices = np.random.permutation(len(train_samples))
        train_samples = train_samples[train_indices]
        train_labels = train_labels[train_indices]
        
        test_indices = np.random.permutation(len(test_samples))
        test_samples = test_samples[test_indices]
        test_labels = test_labels[test_indices]
        
        # 保存数据
        save_dataset(base_dir, f"{snr}dB", train_samples, train_labels, 
                    test_samples, test_labels)

    return snrs


def save_dataset(base_dir, name, train_samples, train_labels, test_samples, test_labels):
    """保存数据集并显示统计信息"""
    # 保存训练集
    train_path = os.path.join(base_dir, "train", f"{name}.pkl")
    with open(train_path, 'wb') as f:
        pickle.dump((train_samples, train_labels), f)
    
    # 保存测试集
    test_path = os.path.join(base_dir, "test", f"{name}.pkl")
    with open(test_path, 'wb') as f:
        pickle.dump((test_samples, test_labels), f)
    
    print(f"保存完成 - 训练集: {train_samples.shape} | 测试集: {test_samples.shape}")


def generate_combined_dataset(base_dir, snrs, name):
    """生成合并数据集（所有SNR或高SNR合并）"""
    print(f"\n生成合并数据集: {name}")
    
    train_samples = []
    train_labels = []
    test_samples = []
    test_labels = []
    
    for snr in snrs:
        # 加载训练集
        train_path = os.path.join(base_dir, "train", f"{snr}dB.pkl")
        with open(train_path, 'rb') as f:
            train_data, train_label = pickle.load(f)
            train_samples.extend(train_data)
            train_labels.extend(train_label)
        
        # 加载测试集
        test_path = os.path.join(base_dir, "test", f"{snr}dB.pkl")
        with open(test_path, 'rb') as f:
            test_data, test_label = pickle.load(f)
            test_samples.extend(test_data)
            test_labels.extend(test_label)
    
    # 保存合并数据集
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    test_samples = np.array(test_samples)
    test_labels = np.array(test_labels)
    
    # 显示合并后的统计信息
    train_unique, train_counts = np.unique(train_labels, return_counts=True)
    test_unique, test_counts = np.unique(test_labels, return_counts=True)
    print(f"合并训练集分布: {dict(zip(train_unique, train_counts))}")
    print(f"合并测试集分布: {dict(zip(test_unique, test_counts))}")
    
    save_dataset(base_dir, name, train_samples, train_labels, test_samples, test_labels)


if __name__ == '__main__':
    print("开始生成RML2016.10a数据集")
    
    # 生成单个SNR数据集
    snrs = generate_dataset_from_RML2016a()
    
    # 生成合并数据集
    base_dir = r"D:\数据集\RML2016A_dB_Semi"
    
    # 所有SNR合并
    generate_combined_dataset(base_dir, snrs, "100dB")
    
    # 高SNR合并（SNR > 0）
    high_snrs = [snr for snr in snrs if snr > 0]
    if high_snrs:
        generate_combined_dataset(base_dir, high_snrs, "highsnr")
    
    print(f"\n✅ 数据集生成完成！")
    print(f"生成了 {len(snrs)} 个单SNR数据集 + 2个合并数据集")
