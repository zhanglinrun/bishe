"""
数据集预处理脚本
将原始数据集按SNR分割为训练集和测试集
支持 RML2016.10a, RML2016.10b, RML2018a, HisarMod
"""

import os
import pickle
import numpy as np
import argparse
import gc

# 尝试导入 h5py（仅在处理 RML2018a/HisarMod 时需要）
try:
    import h5py  # type: ignore
except ModuleNotFoundError:
    h5py = None

# 尝试导入 sklearn 的 train_test_split，不存在时提供简易分层实现
try:
    from sklearn.model_selection import train_test_split  # type: ignore
except ModuleNotFoundError:
    def train_test_split(X, y, test_size=0.3, stratify=None, random_state=None):
        """
        简易版 train_test_split，当 sklearn 不可用时使用。
        仅支持按比例划分，可选分层。
        """
        rng = np.random.default_rng(seed=random_state)
        indices = np.arange(len(y))

        if stratify is not None:
            train_idx = []
            test_idx = []
            for cls in np.unique(stratify):
                cls_indices = indices[stratify == cls]
                rng.shuffle(cls_indices)
                split = int(len(cls_indices) * (1 - test_size))
                train_idx.append(cls_indices[:split])
                test_idx.append(cls_indices[split:])
            train_idx = np.concatenate(train_idx)
            test_idx = np.concatenate(test_idx)
        else:
            rng.shuffle(indices)
            split = int(len(indices) * (1 - test_size))
            train_idx = indices[:split]
            test_idx = indices[split:]

        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def load_RML2016_10a(input_dir):
    """
    加载 RML2016.10a 数据集
    
    Returns:
        X: shape = (N, 2, 128)
        y: shape = (N,)
        snr: shape = (N,)
        mods: 调制类型列表
    """
    file_path = os.path.join(input_dir, 'RML2016.10a', 'RML2016.10a_dict.pkl')
    
    print(f"加载 {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # 提取数据
    X_list, y_list, snr_list = [], [], []
    
    # 调制类型到标签的映射
    mod_classes = sorted(list(set([key[0] for key in data.keys()])))
    mod_to_label = {mod: idx for idx, mod in enumerate(mod_classes)}
    
    print(f"调制类型: {mod_classes}")
    
    for (mod_type, snr_val), samples in data.items():
        X_list.append(samples)
        y_list.extend([mod_to_label[mod_type]] * len(samples))
        snr_list.extend([snr_val] * len(samples))
    
    X = np.vstack(X_list)  # (N, 2, 128)
    y = np.array(y_list)
    snr = np.array(snr_list)
    
    print(f"总样本数: {len(X)}, 形状: {X.shape}")
    
    return X, y, snr, mod_classes


def load_RML2016_10b(input_dir):
    """
    加载 RML2016.10b 数据集
    
    Returns:
        X: shape = (N, 2, 128)
        y: shape = (N,)
        snr: shape = (N,)
        mods: 调制类型列表
    """
    file_path = os.path.join(input_dir, 'RML2016.10b', 'RML2016.10b.dat')
    
    print(f"加载 {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # 提取数据（格式与 10a 相同）
    X_list, y_list, snr_list = [], [], []
    
    mod_classes = sorted(list(set([key[0] for key in data.keys()])))
    mod_to_label = {mod: idx for idx, mod in enumerate(mod_classes)}
    
    print(f"调制类型: {mod_classes}")
    
    for (mod_type, snr_val), samples in data.items():
        X_list.append(samples)
        y_list.extend([mod_to_label[mod_type]] * len(samples))
        snr_list.extend([snr_val] * len(samples))
    
    X = np.vstack(X_list)  # (N, 2, 128)
    y = np.array(y_list)
    snr = np.array(snr_list)
    
    print(f"总样本数: {len(X)}, 形状: {X.shape}")
    
    return X, y, snr, mod_classes


def load_RML2018a(input_dir):
    """
    加载 RML2018a 数据集
    
    Returns:
        X: shape = (N, 2, 1024)
        y: shape = (N,)
        snr: shape = (N,)
        mods: 调制类型列表
    """
    file_path = os.path.join(input_dir, 'RML2018a', 'GOLD_XYZ_OSC.0001_1024.hdf5')

    if h5py is None:
        raise ImportError("缺少 h5py，请安装后处理 RML2018a 数据集")
    
    print(f"加载 {file_path}...")
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
    
    # 调制类型（RML2018a 固定24类）
    mods = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
            '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
            '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
            'FM', 'GMSK', 'OQPSK']
    
    print(f"总样本数: {len(X)}, 形状: {X.shape}")
    print(f"调制类型数: {len(mods)}")
    
    return X, y, snr, mods


def load_HisarMod(input_dir):
    """
    加载 HisarMod 数据集
    
    Returns:
        X: shape = (N, 2, 1024)
        y: shape = (N,)
        snr: shape = (N,)
        mods: 调制类型列表
    """
    train_path = os.path.join(input_dir, 'HisarMod', 'HisarMod2019train.h5')
    test_path = os.path.join(input_dir, 'HisarMod', 'HisarMod2019test.h5')

    if h5py is None:
        raise ImportError("缺少 h5py，请安装后处理 HisarMod 数据集")
    
    print(f"加载 {train_path} 和 {test_path}...")
    
    X_train, y_train, snr_train = _load_hisar_file(train_path)
    X_test, y_test, snr_test = _load_hisar_file(test_path)
    
    # 合并训练集和测试集
    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    snr = np.concatenate([snr_train, snr_test])
    
    # HisarMod 固定26类
    mods = ['2FSK', '2PSK', '4FSK', '4PSK', '8FSK', '8PSK', '16PSK', '32PSK',
            '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-DSB', 'AM-SSB',
            'BPSK', 'CPFSK', 'GFSK', 'MSK', 'OQPSK', 'PAM4', 'QAM16', 'QAM64',
            'QPSK', 'WBFM', 'GMSK']
    
    print(f"总样本数: {len(X)}, 形状: {X.shape}")
    print(f"调制类型数: {len(mods)}")
    
    return X, y, snr, mods


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


def split_and_save_by_snr(X, y, snr, output_dir, dataset_name, test_size=0.3, random_state=42):
    """
    按SNR分割数据集并立即保存
    
    Args:
        X: 数据
        y: 标签
        snr: SNR值
        output_dir: 输出目录
        dataset_name: 数据集名称
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        snr_info: 字典，键为SNR值，值为 (train_shape, test_shape)
    """
    unique_snrs = np.sort(np.unique(snr))
    print(f"\nSNR值范围: {unique_snrs}")
    print(f"SNR数量: {len(unique_snrs)}")
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, dataset_name, 'train')
    test_dir = os.path.join(output_dir, dataset_name, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\n保存数据到 {os.path.join(output_dir, dataset_name)}")
    
    snr_info = {}
    
    for snr_val in unique_snrs:
        print(f"\n处理 SNR = {snr_val}dB")
        
        # 提取该SNR的所有数据
        mask = (snr == snr_val)
        X_snr = X[mask]
        y_snr = y[mask]
        
        print(f"  总样本数: {len(X_snr)}")
        
        # 检查类别分布
        unique, counts = np.unique(y_snr, return_counts=True)
        print(f"  类别分布: {dict(zip(unique, counts))}")
        
        # 分层抽样划分
        train_X, test_X, train_y, test_y = train_test_split(
            X_snr, y_snr,
            test_size=test_size,
            stratify=y_snr,
            random_state=random_state
        )
        
        print(f"  训练集: {len(train_X)}, 测试集: {len(test_X)}")
        
        # 立即保存，包含原始 SNR，后续 Non-IID SNR 划分直接使用真实值
        train_file = os.path.join(train_dir, f"{int(snr_val)}dB.pkl")
        with open(train_file, 'wb') as f:
            pickle.dump((train_X, train_y, np.full(len(train_X), snr_val)), f, protocol=4)
        
        test_file = os.path.join(test_dir, f"{int(snr_val)}dB.pkl")
        with open(test_file, 'wb') as f:
            pickle.dump((test_X, test_y, np.full(len(test_X), snr_val)), f, protocol=4)
        
        print(f"  已保存 {int(snr_val)}dB")
        
        # 保存形状信息用于后续合并
        snr_info[snr_val] = (train_X.shape, test_X.shape)
        
        # 释放内存
        del X_snr, y_snr, train_X, test_X, train_y, test_y
        gc.collect()
    
    return snr_info




def generate_combined_datasets(snr_info, output_dir, dataset_name):
    """
    生成合并数据集（从已保存的文件加载）
    
    Args:
        snr_info: SNR信息字典，键为SNR值，值为 (train_shape, test_shape)
        output_dir: 输出目录
        dataset_name: 数据集名称
    """
    print("\n生成合并数据集...")
    
    train_dir = os.path.join(output_dir, dataset_name, 'train')
    test_dir = os.path.join(output_dir, dataset_name, 'test')
    
    # 1. 所有SNR合并 (100dB)
    print("\n  生成 100dB (所有SNR)...")
    all_train_X, all_train_y, all_train_snr = [], [], []
    all_test_X, all_test_y, all_test_snr = [], [], []
    
    for snr_val in sorted(snr_info.keys()):
        # 从文件加载
        train_file = os.path.join(train_dir, f"{int(snr_val)}dB.pkl")
        test_file = os.path.join(test_dir, f"{int(snr_val)}dB.pkl")
        
        with open(train_file, 'rb') as f:
            loaded = pickle.load(f)
            if len(loaded) == 2:
                train_X, train_y = loaded
                train_snr = np.full(len(train_X), snr_val)
            else:
                train_X, train_y, train_snr = loaded
            all_train_X.append(train_X)
            all_train_y.append(train_y)
            all_train_snr.append(train_snr)
        
        with open(test_file, 'rb') as f:
            loaded = pickle.load(f)
            if len(loaded) == 2:
                test_X, test_y = loaded
                test_snr = np.full(len(test_X), snr_val)
            else:
                test_X, test_y, test_snr = loaded
            all_test_X.append(test_X)
            all_test_y.append(test_y)
            all_test_snr.append(test_snr)
        
        # 立即释放
        del train_X, train_y, test_X, test_y
        gc.collect()
    
    all_train_X = np.vstack(all_train_X)
    all_train_y = np.concatenate(all_train_y)
    all_train_snr = np.concatenate(all_train_snr)
    all_test_X = np.vstack(all_test_X)
    all_test_y = np.concatenate(all_test_y)
    all_test_snr = np.concatenate(all_test_snr)
    
    # 保存 100dB
    with open(os.path.join(train_dir, '100dB.pkl'), 'wb') as f:
        pickle.dump((all_train_X, all_train_y, all_train_snr), f, protocol=4)
    with open(os.path.join(test_dir, '100dB.pkl'), 'wb') as f:
        pickle.dump((all_test_X, all_test_y, all_test_snr), f, protocol=4)
    
    print(f"    训练集: {all_train_X.shape}, 测试集: {all_test_X.shape}")
    
    # 释放内存
    del all_train_X, all_train_y, all_test_X, all_test_y
    gc.collect()
    
    # 2. 高SNR合并 (highsnr: SNR > 0)
    print("\n  生成 highsnr (SNR > 0)...")
    high_train_X, high_train_y, high_train_snr = [], [], []
    high_test_X, high_test_y, high_test_snr = [], [], []
    
    for snr_val in sorted(snr_info.keys()):
        if snr_val > 0:
            # 从文件加载
            train_file = os.path.join(train_dir, f"{int(snr_val)}dB.pkl")
            test_file = os.path.join(test_dir, f"{int(snr_val)}dB.pkl")
            
            with open(train_file, 'rb') as f:
                loaded = pickle.load(f)
                if len(loaded) == 2:
                    train_X, train_y = loaded
                    train_snr = np.full(len(train_X), snr_val)
                else:
                    train_X, train_y, train_snr = loaded
                high_train_X.append(train_X)
                high_train_y.append(train_y)
                high_train_snr.append(train_snr)
            
            with open(test_file, 'rb') as f:
                loaded = pickle.load(f)
                if len(loaded) == 2:
                    test_X, test_y = loaded
                    test_snr = np.full(len(test_X), snr_val)
                else:
                    test_X, test_y, test_snr = loaded
                high_test_X.append(test_X)
                high_test_y.append(test_y)
                high_test_snr.append(test_snr)
            
            # 立即释放
            del train_X, train_y, test_X, test_y
            gc.collect()
    
    if high_train_X:
        high_train_X = np.vstack(high_train_X)
        high_train_y = np.concatenate(high_train_y)
        high_train_snr = np.concatenate(high_train_snr)
        high_test_X = np.vstack(high_test_X)
        high_test_y = np.concatenate(high_test_y)
        high_test_snr = np.concatenate(high_test_snr)
        
        # 保存 highsnr
        with open(os.path.join(train_dir, 'highsnr.pkl'), 'wb') as f:
            pickle.dump((high_train_X, high_train_y, high_train_snr), f, protocol=4)
        with open(os.path.join(test_dir, 'highsnr.pkl'), 'wb') as f:
            pickle.dump((high_test_X, high_test_y, high_test_snr), f, protocol=4)
        
        print(f"    训练集: {high_train_X.shape}, 测试集: {high_test_X.shape}")
        
        # 释放内存
        del high_train_X, high_train_y, high_test_X, high_test_y
        gc.collect()
    else:
        print("    没有SNR > 0的数据")


def process_dataset(dataset_name, input_dir, output_dir):
    """
    处理单个数据集
    
    Args:
        dataset_name: 数据集名称
        input_dir: 输入目录
        output_dir: 输出目录
    """
    print("=" * 80)
    print(f"处理数据集: {dataset_name}")
    print("=" * 80)
    
    # 加载数据
    if dataset_name == 'RML2016.10a':
        X, y, snr, mods = load_RML2016_10a(input_dir)
    elif dataset_name == 'RML2016.10b':
        X, y, snr, mods = load_RML2016_10b(input_dir)
    elif dataset_name == 'RML2018a':
        X, y, snr, mods = load_RML2018a(input_dir)
    elif dataset_name == 'HisarMod':
        X, y, snr, mods = load_HisarMod(input_dir)
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    # 按SNR分割并立即保存
    snr_info = split_and_save_by_snr(X, y, snr, output_dir, dataset_name)
    
    # 释放原始数据
    del X, y, snr, mods
    gc.collect()
    
    # 生成合并数据集
    generate_combined_datasets(snr_info, output_dir, dataset_name)
    
    print(f"\n[完成] {dataset_name} 处理完成！")


def main():
    parser = argparse.ArgumentParser(description='数据集预处理：按SNR分割')
    
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['RML2016.10a', 'RML2016.10b', 'RML2018a', 'HisarMod', 'all'],
                       help='要处理的数据集（默认：all，处理所有数据集）')
    parser.add_argument('--input_dir', type=str, default='dataset',
                       help='原始数据集目录（默认：dataset）')
    parser.add_argument('--output_dir', type=str, default='data_processed',
                       help='输出目录（默认：data_processed）')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("数据集预处理 - 按SNR分割")
    print("=" * 80)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)
    
    # 处理数据集
    if args.dataset == 'all':
        datasets = ['RML2016.10a', 'RML2016.10b', 'RML2018a', 'HisarMod']
        for dataset in datasets:
            try:
                process_dataset(dataset, args.input_dir, args.output_dir)
            except Exception as e:
                print(f"\n[错误] 处理 {dataset} 时出错: {e}")
                print("跳过此数据集，继续处理下一个...\n")
    else:
        process_dataset(args.dataset, args.input_dir, args.output_dir)
    
    print("\n" + "=" * 80)
    print("所有数据集处理完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

