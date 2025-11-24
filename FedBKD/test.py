import os
import torch
import pandas as pd
from openpyxl import Workbook, load_workbook
import argparse
import numpy as np
import pickle

from config import get_config
from models.global_model import get_global_model
from utils import evaluate_model
from torch.utils.data import DataLoader, TensorDataset

# Custom to_tensor_loader from main.py
def to_tensor_loader(X, y, batch_size=64, shuffle=False):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X).float()
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y).long()

    # 自动适配维度
    if len(X.shape) == 3:  # (B, 2, 128)
        X_tensor = X.unsqueeze(1)
    elif len(X.shape) == 4:  # (B, 1, 2, 128)
        X_tensor = X
    else:
        raise ValueError(f"Unexpected input tensor shape: {X.shape}")
    y_tensor = y
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

def load_test_data_by_snr(data_dir, snr):
    """
    加载指定SNR的测试数据
    """
    with open(data_dir, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # 调整数据格式以匹配原始加载器
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], data.keys())))), [1, 0])
    
    X_test_all, y_test_all = [], []
    
    for mod in mods:
        if (mod, snr) in data:
            X_test_all.append(data[(mod, snr)])
            y_test_all.append(np.array([mods.index(mod)] * len(data[(mod, snr)])))

    if not X_test_all:
        return None, None

    X_test = np.vstack(X_test_all)
    y_test = np.concatenate(y_test_all)
    
    return X_test, y_test

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = get_global_model(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"模型已从 {args.model_path} 加载")

    acc_list, snr_list = [], []
    for cur_snr in range(-20, 20, 2):
        # 加载特定信噪比的数据
        X_test, y_test = load_test_data_by_snr(args.data_dir, cur_snr)

        if X_test is None:
            print(f"SNR {cur_snr}dB - 没有找到测试数据，跳过...")
            continue
            
        test_loader = to_tensor_loader(X_test, y_test, args.batch_size, shuffle=False)

        # 修正：将 device 转为字符串类型传递给 evaluate_model
        val_acc = evaluate_model(model, test_loader, str(device))
        print(f"SNR {cur_snr}dB - 准确率: {val_acc:.4f}")

        acc_list.append(val_acc)
        snr_list.append(cur_snr)

    # 保存结果到Excel
    df = pd.DataFrame({
        "SNR": snr_list,
        "Accuracy": acc_list
    })
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    file_path = os.path.join(results_dir, "snr_accuracy.xlsx")
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]

    if not os.path.exists(file_path):
        df.to_excel(file_path, index=False, sheet_name=model_name)
    else:
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:  # type: ignore
            df.to_excel(writer, index=False, sheet_name=model_name)
    
    print(f"测试完成，结果已保存到 {file_path} 的 sheet '{model_name}' 中")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FedBKD Model Evaluation")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model state_dict")
    
    # 从 config.py 获取默认参数
    default_args = get_config()
    parser.add_argument('--num_classes', type=int, default=default_args.num_classes)
    parser.add_argument('--data_dir', type=str, default=default_args.data_dir)
    parser.add_argument('--batch_size', type=int, default=default_args.batch_size)

    args = parser.parse_args()
    main(args) 