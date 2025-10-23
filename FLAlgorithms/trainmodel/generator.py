"""
生成器模型
用于 FedGen 算法的数据自由知识蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    生成器网络
    用于 FedGen，生成伪嵌入或伪 logits
    
    根据论文 "Data-Free Knowledge Distillation for Heterogeneous Federated Learning"
    生成器生成的是特征嵌入或 logits，而不是原始数据
    """
    
    def __init__(self, latent_dim=100, embedding_dim=256, hidden_dim=512):
        """
        Args:
            latent_dim: 潜在向量维度
            embedding_dim: 输出嵌入维度（通常与分类器的倒数第二层维度相同）
            hidden_dim: 隐藏层维度
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        
        # 生成器网络
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Tanh()  # 归一化输出
        )
    
    def forward(self, z):
        """
        前向传播
        
        Args:
            z: 潜在向量 [B, latent_dim]
            
        Returns:
            伪嵌入 [B, embedding_dim]
        """
        return self.model(z)
    
    def sample_latent(self, batch_size, device='cpu'):
        """
        采样潜在向量
        
        Args:
            batch_size: 批大小
            device: 设备
            
        Returns:
            随机潜在向量 [batch_size, latent_dim]
        """
        return torch.randn(batch_size, self.latent_dim, device=device)


class ConditionalGenerator(nn.Module):
    """
    条件生成器
    可以根据类别标签生成特定类别的伪数据
    """
    
    def __init__(self, latent_dim=100, num_classes=11, embedding_dim=256, hidden_dim=512):
        """
        Args:
            latent_dim: 潜在向量维度
            num_classes: 类别数
            embedding_dim: 输出嵌入维度
            hidden_dim: 隐藏层维度
        """
        super(ConditionalGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # 生成器网络（输入是潜在向量 + 标签嵌入）
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        """
        前向传播
        
        Args:
            z: 潜在向量 [B, latent_dim]
            labels: 类别标签 [B]
            
        Returns:
            伪嵌入 [B, embedding_dim]
        """
        # 获取标签嵌入
        label_emb = self.label_embedding(labels)
        
        # 拼接潜在向量和标签嵌入
        gen_input = torch.cat([z, label_emb], dim=1)
        
        return self.model(gen_input)
    
    def sample_latent(self, batch_size, device='cpu'):
        """采样潜在向量"""
        return torch.randn(batch_size, self.latent_dim, device=device)

