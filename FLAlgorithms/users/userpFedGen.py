"""
FedGen 客户端
实现基于数据自由知识蒸馏的联邦学习客户端
根据论文 "Data-Free Knowledge Distillation for Heterogeneous Federated Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from FLAlgorithms.users.userbase import User


class UserFedGen(User):
    """
    FedGen 客户端
    使用生成器进行数据自由知识蒸馏
    
    训练流程：
    1. 训练生成器：使生成的伪数据在分类器上产生与真实数据相似的输出分布
    2. 训练分类器：使用真实数据 + 生成的伪数据
    """
    
    def __init__(self, user_id, model, generator, train_loader, learning_rate, 
                 gen_learning_rate, device='cpu', latent_dim=100, feature_dim=256,
                 optimizer_type='adam', momentum=0.9, weight_decay=1e-4):
        """
        Args:
            user_id: 客户端 ID
            model: 本地分类模型
            generator: 生成器模型
            train_loader: 训练数据加载器
            learning_rate: 分类器学习率
            gen_learning_rate: 生成器学习率
            device: 设备
            latent_dim: 潜在向量维度
            optimizer_type: 分类器优化器类型
            momentum: SGD 动量参数
            weight_decay: 权重衰减
        """
        super(UserFedGen, self).__init__(user_id, model, train_loader, learning_rate, device,
                                        optimizer_type, momentum, weight_decay)
        
        self.generator = generator.to(device)
        self.feature_dim = feature_dim
        # 生成器始终使用 Adam（生成模型的标准做法）
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=gen_learning_rate)
        self.latent_dim = latent_dim
        self.num_classes = model.num_classes
        # 生成器输出到类别空间的投影，用于蒸馏与伪标签
        self.gen_projection = nn.Linear(self.generator.embedding_dim, self.num_classes).to(device)
        
        # 用于知识蒸馏的温度参数
        self.temperature = 2.0
    
    def set_parameters(self, params):
        """设置分类器参数"""
        super().set_parameters(params)
    
    def set_generator_parameters(self, gen_params):
        """
        设置生成器参数
        
        Args:
            gen_params: 生成器参数
        """
        self.generator.load_state_dict(copy.deepcopy(gen_params))

    def set_projection_parameters(self, proj_params):
        self.gen_projection.load_state_dict(copy.deepcopy(proj_params))
    
    def get_generator_parameters(self):
        """
        获取生成器参数
        
        Returns:
            生成器参数
        """
        return copy.deepcopy(self.generator.state_dict())

    def get_projection_parameters(self):
        return copy.deepcopy(self.gen_projection.state_dict())
    
    def train(self, epochs, gen_ratio=0.5, gen_epochs=1):
        """
        FedGen 训练流程
        
        Args:
            epochs: 分类器训练轮数
            gen_ratio: 生成数据与真实数据的比例
            gen_epochs: 生成器训练轮数
            
        Returns:
            平均训练损失
        """
        total_loss = 0.0
        num_batches = 0
        
        # 第一阶段：训练生成器
        gen_loss = self.train_generator(gen_epochs)
        
        # 第二阶段：训练分类器（使用真实数据 + 生成数据）
        cls_loss = self.train_classifier(epochs, gen_ratio)
        
        return cls_loss
    
    def train_generator(self, epochs):
        """
        训练生成器
        目标：使生成的伪数据在分类器上的输出与真实数据的输出分布相似
        
        Args:
            epochs: 训练轮数
            
        Returns:
            平均损失
        """
        self.generator.train()
        self.gen_projection.train()
        self.model.eval()  # 固定分类器
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                batch_size = X.size(0)
                
                # 清零梯度
                self.gen_optimizer.zero_grad()
                
                # 生成伪嵌入（从分类器倒数第二层）
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_embeddings = self.generator(z)
                
                # 获取真实数据的特征
                with torch.no_grad():
                    real_outputs = self.model(X)
                
                # 生成器预测的 logits，通过投影到类别空间
                fake_logits = self.gen_projection(fake_embeddings)
                
                # 知识蒸馏损失（使用 softmax 温度缩放）
                soft_real = F.softmax(real_outputs / self.temperature, dim=1)
                soft_fake = F.log_softmax(fake_logits / self.temperature, dim=1)
                
                loss = F.kl_div(soft_fake, soft_real, reduction='batchmean') * (self.temperature ** 2)
                
                # 反向传播
                loss.backward()
                self.gen_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train_classifier(self, epochs, gen_ratio=0.5):
        """
        训练分类器（使用真实数据 + 生成数据）
        
        Args:
            epochs: 训练轮数
            gen_ratio: 生成数据的比例
            
        Returns:
            平均损失
        """
        self.model.train()
        self.generator.eval()  # 固定生成器
        self.gen_projection.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                batch_size = X.size(0)
                
                # 清零梯度
                self.optimizer.zero_grad()
                
                # 真实数据的损失
                real_outputs = self.model(X)
                real_loss = self.criterion(real_outputs, y)
                
                # 生成伪特征并蒸馏到分类器（仅更新分类头）
                gen_batch_size = int(batch_size * gen_ratio)
                kd_loss = torch.tensor(0.0, device=self.device)
                if gen_batch_size > 0:
                    with torch.no_grad():
                        z = torch.randn(gen_batch_size, self.latent_dim, device=self.device)
                        fake_embeddings = self.generator(z)
                        teacher_logits = self.gen_projection(fake_embeddings)

                    # 使用生成的伪特征直接通过分类头，提升模型对未见分布的鲁棒性
                    student_logits = self.model.classify_from_features(fake_embeddings)
                    kd_loss = F.kl_div(
                        F.log_softmax(student_logits / self.temperature, dim=1),
                        F.softmax(teacher_logits / self.temperature, dim=1),
                        reduction='batchmean'
                    ) * (self.temperature ** 2)
                
                # 总损失 = 真实数据交叉熵 + 蒸馏损失
                loss = real_loss + gen_ratio * kd_loss
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

