"""
客户端基类
定义联邦学习客户端的基本接口
"""

import torch
import torch.nn as nn
import copy


class User:
    """
    联邦学习客户端基类
    """
    
    def __init__(self, user_id, model, train_loader, learning_rate, device='cpu',
                 optimizer_type='adam', momentum=0.9, weight_decay=1e-4):
        """
        Args:
            user_id: 客户端 ID
            model: 本地模型
            train_loader: 训练数据加载器
            learning_rate: 学习率
            device: 设备 ('cpu' 或 'cuda')
            optimizer_type: 优化器类型 ('sgd', 'adam', 'adamw')
            momentum: SGD 动量参数（仅用于 sgd）
            weight_decay: 权重衰减（L2 正则化）
        """
        self.id = user_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        self.device = device
        
        # 根据类型创建优化器
        if optimizer_type.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练样本数
        self.num_samples = len(train_loader.dataset)
    
    def set_parameters(self, params):
        """
        设置本地模型参数
        
        Args:
            params: 模型参数（OrderedDict）
        """
        self.model.load_state_dict(copy.deepcopy(params))
    
    def get_parameters(self):
        """
        获取本地模型参数
        
        Returns:
            模型参数（OrderedDict）
        """
        return copy.deepcopy(self.model.state_dict())
    
    def get_num_samples(self):
        """
        获取训练样本数
        
        Returns:
            样本数
        """
        return self.num_samples
    
    def train(self, epochs):
        """
        本地训练（抽象方法，由子类实现）
        
        Args:
            epochs: 训练轮数
            
        Returns:
            训练损失
        """
        raise NotImplementedError("子类必须实现 train 方法")
    
    def test(self, test_loader):
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
            
        Returns:
            accuracy: 准确率
            loss: 损失值
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # 前向传播
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                
                # 统计
                total_loss += loss.item() * len(y)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        
        return accuracy, avg_loss

