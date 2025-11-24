"""
模型工具函数
包括参数提取、加载、日志记录等功能
"""

import torch
import logging
import os
import csv
from collections import OrderedDict
import copy


def get_model_params(model):
    """
    提取模型参数
    
    Args:
        model: PyTorch 模型
        
    Returns:
        OrderedDict 格式的模型参数
    """
    return copy.deepcopy(model.state_dict())


def set_model_params(model, params):
    """
    设置模型参数
    
    Args:
        model: PyTorch 模型
        params: 模型参数（OrderedDict）
    """
    model.load_state_dict(copy.deepcopy(params))


def average_weights(weights_list, weights=None):
    """
    对多个模型参数进行加权平均
    
    Args:
        weights_list: 模型参数列表
        weights: 权重列表（如果为 None，则均匀加权）
        
    Returns:
        平均后的模型参数
    """
    if weights is None:
        weights = [1.0 / len(weights_list)] * len(weights_list)
    
    # 归一化权重
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # 初始化平均参数字典
    avg_params = OrderedDict()
    
    # 对每个参数进行加权平均
    for key in weights_list[0].keys():
        avg_params[key] = sum([weights[i] * weights_list[i][key] for i in range(len(weights_list))])
    
    return avg_params


def save_logs(log_file, round_num, accuracy, loss, mode='a'):
    """
    保存训练日志到 CSV 文件
    
    Args:
        log_file: CSV 文件路径
        round_num: 训练轮次
        accuracy: 准确率
        loss: 损失值
        mode: 文件打开模式 ('a' 追加, 'w' 覆写)
    """
    file_exists = os.path.exists(log_file)
    
    with open(log_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 如果是新文件或覆写模式，写入表头
        if not file_exists or mode == 'w':
            writer.writerow(['Round', 'Accuracy', 'Loss'])
        
        writer.writerow([round_num, accuracy, loss])


def setup_logger(log_file, name='FL-AMR'):
    """
    配置日志记录器
    
    Args:
        log_file: 日志文件路径
        name: 日志记录器名称
        
    Returns:
        配置好的 logger 对象
    """
    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers
    logger.handlers = []
    
    # 创建文件 handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加 handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_model(model, save_path):
    """
    保存模型权重
    
    Args:
        model: PyTorch 模型
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def load_model(model, load_path, device='cpu'):
    """
    加载模型权重
    
    Args:
        model: PyTorch 模型
        load_path: 加载路径
        device: 设备 ('cpu' 或 'cuda')
    """
    model.load_state_dict(torch.load(load_path, map_location=device))
    return model

