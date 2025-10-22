"""
联邦学习优化器
封装常用的 PyTorch 优化器
"""

import torch.optim as optim


def get_optimizer(optimizer_name, model_parameters, learning_rate, **kwargs):
    """
    获取优化器实例
    
    Args:
        optimizer_name: 优化器名称 ('sgd', 'adam', 'adamw')
        model_parameters: 模型参数
        learning_rate: 学习率
        **kwargs: 其他优化器参数
        
    Returns:
        优化器实例
    """
    if optimizer_name.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        weight_decay = kwargs.get('weight_decay', 0.0)
        return optim.SGD(model_parameters, lr=learning_rate, 
                        momentum=momentum, weight_decay=weight_decay)
    
    elif optimizer_name.lower() == 'adam':
        weight_decay = kwargs.get('weight_decay', 0.0)
        return optim.Adam(model_parameters, lr=learning_rate, 
                         weight_decay=weight_decay)
    
    elif optimizer_name.lower() == 'adamw':
        weight_decay = kwargs.get('weight_decay', 0.01)
        return optim.AdamW(model_parameters, lr=learning_rate, 
                          weight_decay=weight_decay)
    
    else:
        raise ValueError(f"未知优化器: {optimizer_name}. 支持的优化器: sgd, adam, adamw")

