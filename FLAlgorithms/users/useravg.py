"""
FedAvg 客户端
实现标准的联邦平均算法客户端
"""

import torch
from FLAlgorithms.users.userbase import User


class UserAVG(User):
    """
    FedAvg 客户端
    执行标准的本地 SGD 训练
    """
    
    def __init__(self, user_id, model, train_loader, learning_rate, device='cpu',
                 optimizer_type='adam', momentum=0.9, weight_decay=1e-4):
        super(UserAVG, self).__init__(user_id, model, train_loader, learning_rate, device,
                                     optimizer_type, momentum, weight_decay)
    
    def train(self, epochs):
        """
        本地训练
        
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
                loss = self.criterion(outputs, y)
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss

