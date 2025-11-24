"""
数据加载器
支持 RML2016.10a, RML2016.10b, RML2018a, HisarMod 数据集
支持 IID 和 Non-IID（按类别、按SNR）数据划分
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


def load_RML2016_10a(data_dir='dataset/RML2016.10a'):
    """
    加载 RML2016.10a 数据集
    
    Returns:
        X: shape = (N, 2, 128)
        y: shape = (N,)
        snr: shape = (N,)
    """
    file_path = os.path.join(data_dir, 'RML2016.10a_dict.pkl')
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # 提取数据
    X_list, y_list, snr_list = [], [], []
    
    # 调制类型到标签的映射
    mod_classes = sorted(list(set([key[0] for key in data.keys()])))
    mod_to_label = {mod: idx for idx, mod in enumerate(mod_classes)}
    
    for (mod_type, snr_val), samples in data.items():
        X_list.append(samples)
        y_list.extend([mod_to_label[mod_type]] * len(samples))
        snr_list.extend([snr_val] * len(samples))
    
    X = np.vstack(X_list)  # (N, 2, 128)
    y = np.array(y_list)
    snr = np.array(snr_list)
    
    return X, y, snr


def load_RML2016_10b(data_dir='dataset/RML2016.10b'):
    """
    加载 RML2016.10b 数据集
    
    Returns:
        X: shape = (N, 2, 128)
        y: shape = (N,)
        snr: shape = (N,)
    """
    file_path = os.path.join(data_dir, 'RML2016.10b.dat')
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # 提取数据（格式与 10a 相同）
    X_list, y_list, snr_list = [], [], []
    
    mod_classes = sorted(list(set([key[0] for key in data.keys()])))
    mod_to_label = {mod: idx for idx, mod in enumerate(mod_classes)}
    
    for (mod_type, snr_val), samples in data.items():
        X_list.append(samples)
        y_list.extend([mod_to_label[mod_type]] * len(samples))
        snr_list.extend([snr_val] * len(samples))
    
    X = np.vstack(X_list)  # (N, 2, 128)
    y = np.array(y_list)
    snr = np.array(snr_list)
    
    return X, y, snr


def load_RML2018a(data_dir='dataset/RML2018a'):
    """
    加载 RML2018a 数据集
    
    Returns:
        X: shape = (N, 2, 1024)
        y: shape = (N,)
        snr: shape = (N,)
    """
    file_path = os.path.join(data_dir, 'GOLD_XYZ_OSC.0001_1024.hdf5')
    
    with h5py.File(file_path, 'r') as f:
        X = f['X'][:]  # (N, 1024, 2)
        Y = f['Y'][:]  # (N, 24) one-hot
        Z = f['Z'][:]  # (N, 1) SNR
    
    # 转置 X: (N, 1024, 2) -> (N, 2, 1024)
    X = np.transpose(X, (0, 2, 1))
    
    # 将 one-hot 标签转换为整数标签
    y = np.argmax(Y, axis=1)
    
    # 提取 SNR
    snr = Z.flatten()
    
    return X, y, snr


def load_HisarMod(data_dir='dataset/HisarMod'):
    """
    加载 HisarMod 数据集
    
    Returns:
        X: shape = (N, 2, 1024)
        y: shape = (N,)
        snr: shape = (N,)
    """
    train_path = os.path.join(data_dir, 'HisarMod2019train.h5')
    test_path = os.path.join(data_dir, 'HisarMod2019test.h5')
    
    X_train, y_train, snr_train = _load_hisar_file(train_path)
    X_test, y_test, snr_test = _load_hisar_file(test_path)
    
    # 合并训练集和测试集
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    snr = np.concatenate([snr_train, snr_test])
    
    return X, y, snr


def _load_hisar_file(file_path):
    """加载单个 HisarMod h5 文件"""
    with h5py.File(file_path, 'r') as f:
        X = f['samples'][:]  # 假设 shape = (N, 2, 1024)
        y = f['labels'][:]   # (N,)
        snr = f['snr'][:]    # (N,)
    
    # 确保维度正确
    if len(X.shape) == 2:
        # 如果是 (N, 2048)，重塑为 (N, 2, 1024)
        X = X.reshape(-1, 2, X.shape[1] // 2)
    
    return X, y.astype(np.int64), snr


def split_data_iid(X, y, snr, num_clients):
    """
    IID 数据划分：随机打乱后均匀分配
    
    Args:
        X: 数据
        y: 标签
        snr: SNR 值
        num_clients: 客户端数量
        
    Returns:
        client_data: 列表，每个元素是 (X_client, y_client)
    """
    num_samples = len(y)
    indices = np.random.permutation(num_samples)
    
    # 划分索引
    split_indices = np.array_split(indices, num_clients)
    
    client_data = []
    for client_idx in split_indices:
        client_data.append((X[client_idx], y[client_idx]))
    
    return client_data


def split_data_non_iid_class(X, y, snr, num_clients, alpha=0.5):
    """
    按调制类型 Non-IID 划分：使用 Dirichlet 分布
    
    Args:
        X: 数据
        y: 标签
        snr: SNR 值
        num_clients: 客户端数量
        alpha: Dirichlet 分布参数（越小越不均衡）
        
    Returns:
        client_data: 列表，每个元素是 (X_client, y_client)
    """
    num_classes = len(np.unique(y))
    client_data = [[] for _ in range(num_clients)]
    
    # 为每个类别分配样本到客户端
    for class_idx in range(num_classes):
        # 获取该类别的所有样本索引
        class_indices = np.where(y == class_idx)[0]
        np.random.shuffle(class_indices)
        
        # 使用 Dirichlet 分布生成分配比例
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        
        # 根据比例分配样本
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        split_indices = np.split(class_indices, proportions)
        
        # 将样本分配给各个客户端
        for client_id, idx in enumerate(split_indices):
            client_data[client_id].extend(idx.tolist())
    
    # 转换为 (X, y) 元组
    result = []
    for client_id in range(num_clients):
        indices = np.array(client_data[client_id])
        np.random.shuffle(indices)
        result.append((X[indices], y[indices]))
    
    return result


def split_data_non_iid_snr(X, y, snr, num_clients):
    """
    按 SNR 范围 Non-IID 划分：不同客户端获得不同 SNR 范围的数据
    
    Args:
        X: 数据
        y: 标签
        snr: SNR 值
        num_clients: 客户端数量
        
    Returns:
        client_data: 列表，每个元素是 (X_client, y_client)
    """
    # 获取 SNR 的范围
    unique_snrs = np.sort(np.unique(snr))
    
    # 将 SNR 划分为 num_clients 段
    snr_splits = np.array_split(unique_snrs, num_clients)
    
    client_data = []
    for snr_range in snr_splits:
        # 找到该 SNR 范围内的所有样本
        mask = np.isin(snr, snr_range)
        indices = np.where(mask)[0]
        client_data.append((X[indices], y[indices]))
    
    return client_data


def load_preprocessed_data(dataset_name, data_snr, data_dir='data_processed'):
    """
    加载预处理的数据
    
    Args:
        dataset_name: 数据集名称
        data_snr: SNR标识（如 '10dB', '100dB', 'highsnr'）
        data_dir: 预处理数据目录
        
    Returns:
        X_train: 训练数据
        y_train: 训练标签
        X_test: 测试数据
        y_test: 测试标签
        num_classes: 类别数
    """
    # 构造文件路径
    train_file = os.path.join(data_dir, dataset_name, 'train', f'{data_snr}.pkl')
    test_file = os.path.join(data_dir, dataset_name, 'test', f'{data_snr}.pkl')
    
    # 加载训练集
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练集文件不存在: {train_file}")
    with open(train_file, 'rb') as f:
        X_train, y_train = pickle.load(f)
    
    # 加载测试集
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试集文件不存在: {test_file}")
    with open(test_file, 'rb') as f:
        X_test, y_test = pickle.load(f)
    
    # 获取类别数
    config = get_dataset_config(dataset_name)
    num_classes = config['num_classes']
    
    print(f"从预处理文件加载数据:")
    print(f"  训练集: {train_file}")
    print(f"  测试集: {test_file}")
    print(f"  训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")
    print(f"  数据形状: {X_train.shape}")
    
    return X_train, y_train, X_test, y_test, num_classes


def get_dataloaders(dataset_name, num_clients, batch_size, non_iid_type='iid', 
                   alpha=0.5, data_snr='100dB', data_dir='data_processed'):
    """
    获取数据加载器
    
    Args:
        dataset_name: 数据集名称
        num_clients: 客户端数量
        batch_size: 批大小
        non_iid_type: 数据划分类型 ('iid', 'class', 'snr')
        alpha: Dirichlet 参数（用于 class Non-IID）
        data_snr: SNR标识（如 '10dB', '100dB', 'highsnr'）
        data_dir: 预处理数据目录
        
    Returns:
        train_loaders: 列表，每个元素是一个客户端的 DataLoader
        test_loader: 全局测试集 DataLoader
        num_classes: 类别数
    """
    # 加载预处理的数据
    X_train, y_train, X_test, y_test, num_classes = load_preprocessed_data(
        dataset_name, data_snr, data_dir
    )
    
    # 为 Non-IID SNR 划分创建虚拟 SNR 数组（用于兼容性）
    # 注意：如果使用 'snr' Non-IID 类型，需要实际的 SNR 值
    # 这里我们创建一个均匀分布的虚拟 SNR 用于划分
    snr_train = np.linspace(-20, 20, len(y_train))
    
    # 划分客户端数据
    if non_iid_type == 'iid':
        client_data = split_data_iid(X_train, y_train, snr_train, num_clients)
    elif non_iid_type == 'class':
        client_data = split_data_non_iid_class(X_train, y_train, snr_train, num_clients, alpha)
    elif non_iid_type == 'snr':
        client_data = split_data_non_iid_snr(X_train, y_train, snr_train, num_clients)
    else:
        raise ValueError(f"未知的 Non-IID 类型: {non_iid_type}")
    
    # 创建客户端 DataLoader
    train_loaders = []
    for X_client, y_client in client_data:
        dataset = SignalDataset(X_client, y_client)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        train_loaders.append(loader)
    
    # 创建全局测试集 DataLoader
    test_dataset = SignalDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"数据集: {dataset_name}")
    print(f"SNR: {data_snr}")
    print(f"客户端数量: {num_clients}, 划分类型: {non_iid_type}")
    print(f"每个客户端平均样本数: {len(y_train) // num_clients}")
    
    return train_loaders, test_loader, num_classes

