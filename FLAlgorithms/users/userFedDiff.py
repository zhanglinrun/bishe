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
        
        # 初始化新的 U-Net 扩散生成器
        # 参数需与 test_diffusion_gen.py 中保持一致
        self.generator = DiffusionGenerator(
            num_classes=model.num_classes,
            signal_length=model.signal_length,
            in_channels=2,
            base_channels=64,
            channel_mults=(1, 2, 4),
            timesteps=diffusion_steps,
            beta_schedule="cosine"
        ).to(device)
        
        # 生成器通常使用 AdamW 优化器
        self.gen_optimizer = torch.optim.AdamW(self.generator.parameters(), lr=gen_learning_rate, weight_decay=1e-4)
        
        # 用于 CFG 训练的 dropout 概率
        self.cfg_prob = 0.1 
    
    def set_generator_parameters(self, params):
        """接收全局生成器参数"""
        self.generator.load_state_dict(copy.deepcopy(params))

    def get_generator_parameters(self):
        """上传本地生成器参数"""
        return copy.deepcopy(self.generator.state_dict())

    def train(self, epochs):
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
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                # --- 2. 训练扩散生成器 ---
                self.gen_optimizer.zero_grad()
                # 使用支持 CFG 的 loss 计算方法
                gen_loss = self.generator.training_loss(X, y, cfg_prob=self.cfg_prob)
                gen_loss.backward()
                
                # 梯度裁剪，防止生成器梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.gen_optimizer.step()
                
                total_loss += loss.item()
                total_gen_loss += gen_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_gen_loss = total_gen_loss / num_batches if num_batches > 0 else 0.0
        
        # 返回元组，包含 (分类器损失, 生成器损失)
        # 注意：Server端接收返回值时需要做相应修改，或者仅打印日志
        # 为了兼容性，这里我们打印日志，主要返回分类器损失
        # print(f"Client {self.id} | Cls Loss: {avg_loss:.4f} | Gen Loss: {avg_gen_loss:.4f}")
        
        return avg_loss