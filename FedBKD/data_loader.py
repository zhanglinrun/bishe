import os
import pickle
import numpy as np
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict
from typing import Dict, Tuple, List, Union, Optional


def load_federated_data(data_path=r'/home/lrzhang/datasets/RML2016.10a_dict.pkl',
                        num_clients=5, 
                        num_classes=10, 
                        mods_per_client=2, 
                        selected_mods=None,
                        snr_range=None,
                        return_global=False, 
                        seed=42) -> Union[Tuple[Dict, Dict], Tuple[Dict, Dict, np.ndarray, np.ndarray, Dict]]:
    """
    加载RML2016.10a数据集，并划分为联邦学习客户端数据（支持非IID + 类别不重叠 + 模型异构）
    
    Args:
        data_path (str): pkl路径
        num_clients (int): 客户端数量
        num_classes (int): 所选调制类型数量
        mods_per_client (int): 每个客户端的调制类型数（默认不重叠）
        selected_mods (list): 明确指定所用调制类型
        snr_range (list): 所选SNR列表，如 [-8, 0, 8]
        return_global (bool): 是否返回全局数据（用于CVAE）
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)

    # 加载数据
    with open(data_path, 'rb') as f:
        raw_data = pickle.load(f, encoding='latin1')

    all_mods = sorted(list(set([k[0] for k in raw_data.keys()])))
    snr_levels = sorted(list(set([k[1] for k in raw_data.keys()])))

    if selected_mods is None:
        selected_mods = all_mods[:num_classes]
    else:
        for mod in selected_mods:
            if mod not in all_mods:
                raise ValueError(f"调制类型 {mod} 不在数据集中")

    num_classes = len(selected_mods)
    label_map = {mod: idx for idx, mod in enumerate(selected_mods)}

    print(f"\n[调制类型选择]: {selected_mods}")
    print(f"[SNR范围]: {min(snr_levels)}dB ~ {max(snr_levels)}dB")

    # 收集训练/测试数据
    train_data, test_data = defaultdict(list), defaultdict(list)
    available_mods = []

    for mod in selected_mods:
        collected = False
        for snr in snr_levels:
            if snr_range is not None and snr not in snr_range:
                continue
            key = (mod, snr)
            if key not in raw_data:
                continue
            samples = raw_data[key]
            X_train, X_test = train_test_split(samples, train_size=0.7, test_size=0.3, random_state=seed)
            train_data[mod].append(X_train)
            test_data[mod].append(X_test)
            collected = True
        if collected:
            available_mods.append(mod)

    train_data = {mod: np.vstack(train_data[mod]) for mod in train_data}
    test_data = {mod: np.vstack(test_data[mod]) for mod in test_data}

    mod_pool = available_mods.copy()
    random.shuffle(mod_pool)

    # 强制类别不重叠划分
    total_needed = num_clients * mods_per_client
    assert len(mod_pool) >= total_needed, \
        f"[错误] 可用调制类型数不足，无法满足 {num_clients} 个客户端 × {mods_per_client} 类的配置"

    client_class_map = {
        i: mod_pool[i * mods_per_client:(i + 1) * mods_per_client]
        for i in range(num_clients)
    }

    print("\n[客户端调制类型分配]:")
    for cid, mods in client_class_map.items():
        print(f"Client {cid}: {mods}")

    # 分配客户端数据
    clients_train, clients_test = {}, {}

    for cid, mod_list in client_class_map.items():
        X_train = np.vstack([train_data[mod] for mod in mod_list])
        y_train = np.hstack([[label_map[mod]] * len(train_data[mod]) for mod in mod_list])
        X_test = np.vstack([test_data[mod] for mod in mod_list])
        y_test = np.hstack([[label_map[mod]] * len(test_data[mod]) for mod in mod_list])
        clients_train[cid] = (X_train, y_train)
        clients_test[cid] = (X_test, y_test)

    if return_global:
        full_x = np.vstack([train_data[mod] for mod in available_mods])
        full_y = np.hstack([[label_map[mod]] * len(train_data[mod]) for mod in available_mods])
        return clients_train, clients_test, full_x, full_y, label_map
    else:
        return clients_train, clients_test
