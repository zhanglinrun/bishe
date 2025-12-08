import torch
import torch.nn as nn
import copy
from FLAlgorithms.users.userbase import User
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator

class UserFedDiff(User):
    def __init__(self, user_id, model, train_loader, learning_rate, device='cpu',
                 optimizer_type='adam', momentum=0.9, weight_decay=1e-4,
                 diffusion_steps=1000, gen_learning_rate=2e-4):
        super().__init__(user_id, model, train_loader, learning_rate, device,
                        optimizer_type, momentum, weight_decay)
        
        self.generator = DiffusionGenerator(
            num_classes=model.num_classes,
            signal_length=model.signal_length,
            in_channels=2,
            base_channels=64,
            channel_mults=(1, 2, 4),
            timesteps=diffusion_steps,
            beta_schedule="cosine"
        ).to(device)
        
        self.gen_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=gen_learning_rate, weight_decay=weight_decay)
        self.cfg_prob = 0.1
        
        # 初始学习率记录
        self.initial_lr = learning_rate
        self.initial_gen_lr = gen_learning_rate
    
    def set_generator_parameters(self, params):
        self.generator.load_state_dict(copy.deepcopy(params))

    def get_generator_parameters(self):
        return copy.deepcopy(self.generator.state_dict())

    def update_learning_rate(self, round_num):
        """简单的学习率衰减策略"""
        if round_num > 0:
            # 每 10 轮衰减 0.8
            decay = 0.95 ** round_num
            
            # 更新分类器学习率
            new_lr = self.initial_lr * decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
                
            # 更新生成器学习率
            new_gen_lr = self.initial_gen_lr * decay
            for param_group in self.gen_optimizer.param_groups:
                param_group['lr'] = new_gen_lr

    def train(self, epochs, round_num=0):
        # 更新学习率
        self.update_learning_rate(round_num)
        
        self.model.train()
        self.generator.train()
        
        total_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # --- 1. 训练分类器 ---
                self.optimizer.zero_grad()
                outputs = self.model(X)
                cls_loss = self.criterion(outputs, y)
                cls_loss.backward()
                self.optimizer.step()
                
                # --- 2. 训练扩散生成器 ---
                self.gen_optimizer.zero_grad()
                gen_loss = self.generator.training_loss(X, y, cfg_prob=self.cfg_prob)
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.gen_optimizer.step()
                
                total_loss += cls_loss.item()
                total_gen_loss += gen_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss