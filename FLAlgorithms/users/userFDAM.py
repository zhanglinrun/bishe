"""
FDAM 客户端 (重构版)
使用与FedDiff相同的DiffusionGenerator进行信号空间扩散生成
"""

import torch
import torch.nn as nn
import copy
from FLAlgorithms.users.userbase import User
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator


class UserFDAM(User):
    def __init__(self, user_id, model, train_loader, learning_rate, device='cpu',
                 optimizer_type='adam', momentum=0.9, weight_decay=1e-4,
                 diffusion_steps=1000, gen_learning_rate=2e-4):
        super(UserFDAM, self).__init__(user_id, model, train_loader, learning_rate, device,
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
        
        self.gen_optimizer = torch.optim.AdamW(
            self.generator.parameters(), 
            lr=gen_learning_rate, 
            weight_decay=weight_decay
        )
        self.cfg_prob = 0.1
        self.initial_lr = learning_rate
        self.initial_gen_lr = gen_learning_rate

    def set_generator_parameters(self, params):
        self.generator.load_state_dict(copy.deepcopy(params))

    def get_generator_parameters(self):
        return copy.deepcopy(self.generator.state_dict())

    def update_learning_rate(self, round_num):
        if round_num > 0:
            decay = 0.95 ** round_num
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.initial_lr * decay
            for param_group in self.gen_optimizer.param_groups:
                param_group['lr'] = self.initial_gen_lr * decay

    def train(self, epochs, round_num=0):
        self.update_learning_rate(round_num)
        self.model.train()
        self.generator.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X)
                cls_loss = self.criterion(outputs, y)
                cls_loss.backward()
                self.optimizer.step()
                
                self.gen_optimizer.zero_grad()
                gen_loss = self.generator.training_loss(X, y, cfg_prob=self.cfg_prob)
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.gen_optimizer.step()
                
                total_loss += cls_loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
