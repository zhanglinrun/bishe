import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# =============================
# 客户端到云端蒸馏模块（Client-to-Cloud Distillation）
# - 客户端上传 logits
# - 云端模型进行 KL 散度蒸馏 + CrossEntropy（论文公式 15~17）
# =============================

def average_logits(all_logits):
    """平均所有客户端的logits"""
    return torch.stack(all_logits).mean(dim=0)

class ClientWeightManager:
    def __init__(self, num_clients, init_weight=1.0, momentum=0.9):
        self.num_clients = num_clients
        self.weights = np.ones(num_clients) * init_weight
        self.momentum = momentum
        self.history = {i: [] for i in range(num_clients)}
        
    def update_weights(self, accuracies, losses):
        """
        基于模型性能和损失更新客户端权重
        Args:
            accuracies: 每个客户端的准确率列表
            losses: 每个客户端的损失列表
        """
        # 归一化准确率和损失
        acc_norm = (accuracies - np.min(accuracies)) / (np.max(accuracies) - np.min(accuracies) + 1e-8)
        loss_norm = 1 - (losses - np.min(losses)) / (np.max(losses) - np.min(losses) + 1e-8)
        
        # 综合评分
        scores = 0.7 * acc_norm + 0.3 * loss_norm
        
        # 更新权重
        new_weights = self.momentum * self.weights + (1 - self.momentum) * scores
        self.weights = new_weights / np.sum(new_weights)  # 归一化
        
        # 记录历史
        for i in range(self.num_clients):
            self.history[i].append(self.weights[i])
            
    def get_weights(self):
        """获取当前权重"""
        return torch.tensor(self.weights, dtype=torch.float32)

def train_global_model(model, optimizer, dataloader, teacher_logits_list, client_weights=None, 
                      temperature=1.0, alpha=0.5, device='cpu'):
    """
    训练全局模型（学生模型）
    Args:
        model: 全局模型
        optimizer: 优化器
        dataloader: 数据加载器
        teacher_logits_list: 教师模型logits列表
        client_weights: 客户端权重
        temperature: 蒸馏温度
        alpha: 知识蒸馏损失权重
        device: 设备
    """
    model.train()
    total_loss = 0
    total_kd = 0
    total_ce = 0
    batch_count = 0
    
    # 如果没有提供权重，使用均匀权重
    if client_weights is None:
        client_weights = torch.ones(len(teacher_logits_list)) / len(teacher_logits_list)
    client_weights = client_weights.to(device)
    
    for (x, y), teacher_logits in zip(dataloader, zip(*teacher_logits_list)):
        x, y = x.to(device), y.to(device)
        teacher_logits = [logit.to(device) for logit in teacher_logits]
        
        # 确保teacher_logit的维度正确
        teacher_logits = [logit.unsqueeze(0) if len(logit.shape) == 1 else logit for logit in teacher_logits]
        
        # 前向传播
        student_logit = model(x)
        
        # 检查NaN
        if torch.isnan(student_logit).any():
            print("[WARNING] NaN detected in student_logit, skipping batch")
            continue
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(student_logit, y)
        
        # 检查NaN
        if torch.isnan(ce_loss):
            print("[WARNING] NaN detected in ce_loss, skipping batch")
            continue
        
        # 计算加权知识蒸馏损失
        kd_loss = torch.tensor(0.0, device=device)
        valid_kd = 0
        for i, teacher_logit in enumerate(teacher_logits):
            # 检查teacher_logit是否包含NaN
            if torch.isnan(teacher_logit).any():
                print(f"[WARNING] NaN detected in teacher_logit {i}, skipping")
                continue
                
            p_s = F.log_softmax(student_logit / temperature, dim=-1)
            p_t = F.softmax(teacher_logit / temperature, dim=-1)
            
            # 检查概率分布是否有效
            if torch.isnan(p_s).any() or torch.isnan(p_t).any():
                print(f"[WARNING] NaN detected in probability distributions, skipping")
                continue
                
            kd_batch = F.kl_div(p_s, p_t, reduction='batchmean') * (temperature ** 2)
            
            if not torch.isnan(kd_batch):
                kd_loss += client_weights[i] * kd_batch
                valid_kd += 1
        
        # 如果没有有效的KD损失，跳过这个batch
        if valid_kd == 0:
            print("[WARNING] No valid KD loss, using only CE loss")
            kd_loss = torch.tensor(0.0, device=device)
        
        # 检查KD损失
        if torch.isnan(kd_loss):
            print("[WARNING] NaN detected in kd_loss, setting to 0")
            kd_loss = torch.tensor(0.0, device=device)
        
        # 使用配置的损失权重
        ce_weight = 1 - alpha
        kd_weight = alpha
        
        # 总损失
        loss = ce_weight * ce_loss + kd_weight * kd_loss
        
        # 检查总损失
        if torch.isnan(loss):
            print("[WARNING] NaN detected in total loss, skipping batch")
            continue
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_kd += kd_loss.item()
        total_ce += ce_loss.item()
        batch_count += 1
    
    if batch_count == 0:
        print("[ERROR] No valid batches processed")
        return 0.0, 0.0, 0.0
    
    return total_loss / batch_count, total_kd / batch_count, total_ce / batch_count
