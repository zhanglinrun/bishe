import torch
import copy
from FLAlgorithms.users.userbase import User
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator

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
        super().__init__(
            user_id,
            model,
            train_loader,
            learning_rate,
            device,
            optimizer_type,
            momentum,
            weight_decay,
        )
        self.generator = DiffusionGenerator(
            num_classes=model.num_classes,
            signal_length=model.signal_length,
            timesteps=diffusion_steps,
        ).to(device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=gen_learning_rate)
        self.last_gen_loss = 0.0
        
    def train(self, epochs):
        """
        本地模型训练：FedDDPM 思路，分类器 + 扩散模型同时优化
        """
        self.model.train()
        self.generator.train()
        
        total_cls_loss = 0.0
        total_gen_loss = 0.0
        num_cls_batches = 0
        num_gen_batches = 0
        
        for epoch in range(epochs):
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
                
                total_cls_loss += loss.item()
                num_cls_batches += 1

                # 扩散模型噪声预测训练
                self.gen_optimizer.zero_grad()
                diff_loss = self.generator.training_loss(X, y)
                diff_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.gen_optimizer.step()

                total_gen_loss += diff_loss.item()
                num_gen_batches += 1
        
        avg_cls_loss = total_cls_loss / num_cls_batches if num_cls_batches > 0 else 0.0
        avg_gen_loss = total_gen_loss / num_gen_batches if num_gen_batches > 0 else 0.0
        self.last_gen_loss = avg_gen_loss
        return {"cls_loss": avg_cls_loss, "gen_loss": avg_gen_loss}

    def get_generator_parameters(self):
        """返回扩散模型参数，供服务器聚合"""
        return copy.deepcopy(self.generator.state_dict())

    def set_generator_parameters(self, params):
        """从服务器下发的全局扩散模型参数"""
        if params is not None:
            self.generator.load_state_dict(copy.deepcopy(params))
