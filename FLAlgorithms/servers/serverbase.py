"""
服务器基类
定义联邦学习服务器的基本接口
"""

import torch
import torch.nn as nn
import copy


class Server:
    """
    联邦学习服务器基类
    """
    
    def __init__(self, model, users, num_rounds, device='cpu'):
        """
        Args:
            model: 全局模型
            users: 客户端列表
            num_rounds: 训练轮次
            device: 设备
        """
        self.model = model.to(device)
        self.users = users
        self.num_rounds = num_rounds
        self.device = device
        self.num_clients = len(users)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 记录训练历史
        self.train_losses = []
        self.train_accuracies = []
    
    def send_parameters(self):
        """
        将全局模型参数发送给所有客户端
        """
        global_params = self.model.state_dict()
        for user in self.users:
            user.set_parameters(global_params)
    
    def aggregate_parameters(self):
        """
        聚合客户端模型参数（抽象方法，由子类实现）
        """
        raise NotImplementedError("子类必须实现 aggregate_parameters 方法")
    
    def evaluate(self, test_loader):
        """
        在测试集上评估全局模型
        
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
    
    def train(self):
        """
        训练流程（抽象方法，由子类实现）
        """
        raise NotImplementedError("子类必须实现 train 方法")
    
    def get_train_history(self):
        """
        获取训练历史
        
        Returns:
            losses: 损失列表
            accuracies: 准确率列表
        """
        return self.train_losses, self.train_accuracies

