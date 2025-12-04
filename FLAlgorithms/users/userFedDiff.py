import torch
import copy
from FLAlgorithms.users.userbase import User

class UserFedDiff(User):
    """
    FedDiff 客户端
    
    注意：在 FedDiff 论文的标准实现中，客户端不需要训练生成器。
    生成器是部署在服务器端，利用客户端上传的模型作为 Teacher 进行对抗训练的。
    因此，客户端代码只需负责本地分类模型的训练（SGD）。
    """
    
    def __init__(self, user_id, model, train_loader, learning_rate, device='cpu',
                 optimizer_type='adam', momentum=0.9, weight_decay=1e-4,
                 diffusion_steps=50, gen_learning_rate=1e-3):
        # 注意：diffusion_steps 和 gen_learning_rate 在客户端不再使用，保留参数是为了兼容接口
        super().__init__(user_id, model, train_loader, learning_rate, device,
                        optimizer_type, momentum, weight_decay)
        
    def train(self, epochs):
        """
        本地模型训练
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
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def get_generator_parameters(self):
        """兼容接口，返回空或None"""
        return None

    def set_generator_parameters(self, params):
        """兼容接口，不做任何操作"""
        pass