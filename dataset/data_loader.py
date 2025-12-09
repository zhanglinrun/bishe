"""
数据加载器
支持 RML2016.10a, RML2016.10b, RML2018a, HisarMod 数据集
修复：增加了强制 Z-Score 归一化，确保数据适配扩散模型
"""

import numpy as np
import pickle
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_config import get_dataset_config


class SignalDataset(Dataset):
    """信号数据集类"""
    
    def __init__(self, X, y):
        """
        Args:
            X: 信号数据，shape = (N, 2, L)
            y: 标签，shape = (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_preprocessed_data(dataset_name, data_snr, data_dir='data_processed'):
    """
    加载预处理的数据并进行归一化
    """
    # 构造文件路径
    train_file = os.path.join(data_dir, dataset_name, 'train', f'{data_snr}.pkl')
    test_file = os.path.join(data_dir, dataset_name, 'test', f'{data_snr}.pkl')
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练集文件不存在: {train_file}")
    with open(train_file, 'rb') as f:
        X_train, y_train, snr_train = pickle.load(f)
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试集文件不存在: {test_file}")
    with open(test_file, 'rb') as f:
        X_test, y_test, snr_test = pickle.load(f)
    
    # [关键修复] 强制归一化 (Z-Score Normalization)
    # 扩散模型对数据尺度非常敏感，必须归一化到 Std=1 左右
    mean = np.mean(X_train)
    std = np.std(X_train)
    
    # 防止除以零
    if std < 1e-6:
        std = 1.0
        
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    # 获取类别数
    config = get_dataset_config(dataset_name)
    num_classes = config['num_classes']
    
    print(f"从预处理文件加载数据:")
    print(f"  训练集: {train_file}")
    print(f"  测试集: {test_file}")
    print(f"  训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    print(f"  数据归一化: Mean={mean:.6f}, Std={std:.6f} -> 归一化后 Std=1.0")
    print(f"  数据形状: {X_train.shape}")
    
    return X_train, y_train, snr_train, X_test, y_test, snr_test, num_classes


def split_data_iid(X, y, snr, num_clients):
    """IID 数据划分"""
    num_samples = len(y)
    indices = np.random.permutation(num_samples)
    split_indices = np.array_split(indices, num_clients)
    
    client_data = []
    for client_idx in split_indices:
        client_data.append((X[client_idx], y[client_idx], snr[client_idx]))
    return client_data


def split_data_non_iid_class(X, y, snr, num_clients, alpha=0.5):
    """按调制类型 Non-IID 划分"""
    num_classes = len(np.unique(y))
    client_data = [[] for _ in range(num_clients)]
    
    for class_idx in range(num_classes):
        class_indices = np.where(y == class_idx)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, proportions)
        
        for client_id, idx in enumerate(split_indices):
            client_data[client_id].extend(idx.tolist())
    
    result = []
    for client_id in range(num_clients):
        indices = np.array(client_data[client_id])
        np.random.shuffle(indices)
        result.append((X[indices], y[indices], snr[indices]))
    return result


def split_data_non_iid_snr(X, y, snr, num_clients, alpha=0.5):
    """按 SNR Non-IID 划分"""
    unique_snrs = np.sort(np.unique(snr))
    client_data = [[] for _ in range(num_clients)]
    
    for snr_val in unique_snrs:
        snr_indices = np.where(snr == snr_val)[0]
        np.random.shuffle(snr_indices)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (np.cumsum(proportions) * len(snr_indices)).astype(int)[:-1]
        split_indices = np.split(snr_indices, proportions)
        
        for client_id, idx in enumerate(split_indices):
            client_data[client_id].extend(idx.tolist())
    
    result = []
    for client_id in range(num_clients):
        indices = np.array(client_data[client_id])
        np.random.shuffle(indices)
        result.append((X[indices], y[indices], snr[indices]))
    return result


def get_dataloaders(dataset_name, num_clients, batch_size, non_iid_type='iid', 
                   alpha=0.5, data_snr='100dB', data_dir='data_processed'):
    """获取数据加载器"""
    X_train, y_train, snr_train, X_test, y_test, snr_test, num_classes = load_preprocessed_data(
        dataset_name, data_snr, data_dir
    )
    
    if non_iid_type == 'iid':
        client_data = split_data_iid(X_train, y_train, snr_train, num_clients)
    elif non_iid_type == 'class':
        client_data = split_data_non_iid_class(X_train, y_train, snr_train, num_clients, alpha)
    elif non_iid_type == 'snr':
        client_data = split_data_non_iid_snr(X_train, y_train, snr_train, num_clients, alpha)
    else:
        raise ValueError(f"未知的 Non-IID 类型: {non_iid_type}")
    
    train_loaders = []
    for X_client, y_client, snr_client in client_data:
        dataset = SignalDataset(X_client, y_client)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        train_loaders.append(loader)
    
    test_dataset = SignalDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"数据集: {dataset_name}")
    print(f"SNR: {data_snr}")
    print(f"客户端数量: {num_clients}, 划分类型: {non_iid_type}")
    
    return train_loaders, test_loader, num_classes