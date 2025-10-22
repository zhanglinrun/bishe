"""
FedProx 客户端
实现带近端项的联邦学习客户端
"""

import torch
import copy
from FLAlgorithms.users.userbase import User


class UserFedProx(User):
    """
    FedProx 客户端
    在损失函数中添加近端项：L = CE_loss + (mu/2) * ||w - w_global||^2
    """
    
    def __init__(self, user_id, model, train_loader, learning_rate, mu, device='cpu',
                 optimizer_type='adam', momentum=0.9, weight_decay=1e-4):
        """
        Args:
            user_id: 客户端 ID
            model: 本地模型
            train_loader: 训练数据加载器
            learning_rate: 学习率
            mu: 近端项系数
            device: 设备
            optimizer_type: 优化器类型
            momentum: SGD 动量参数
            weight_decay: 权重衰减
        """
        super(UserFedProx, self).__init__(user_id, model, train_loader, learning_rate, device,
                                         optimizer_type, momentum, weight_decay)
        self.mu = mu
        self.global_params = None
    
    def set_parameters(self, params):
        """
        设置参数并保存全局参数
        
        Args:
            params: 模型参数
        """
        super().set_parameters(params)
        # 保存全局参数用于计算近端项
        self.global_params = copy.deepcopy(params)
    
    def train(self, epochs):
        """
        带近端项的本地训练
        
        Args:
            epochs: 训练轮数
            
        Returns:
            平均训练损失
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(X)
                ce_loss = self.criterion(outputs, y)
                
                # 计算近端项：(mu/2) * ||w - w_global||^2
                proximal_term = 0.0
                if self.global_params is not None:
                    for name, param in self.model.named_parameters():
                        proximal_term += torch.sum((param - self.global_params[name].to(self.device)) ** 2)
                    proximal_term = (self.mu / 2) * proximal_term
                
                # 总损失 = 交叉熵损失 + 近端项
                loss = ce_loss + proximal_term
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss

