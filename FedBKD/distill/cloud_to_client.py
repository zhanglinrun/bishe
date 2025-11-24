import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================
# 云端到客户端蒸馏模块（Cloud-to-Client Distillation）
# - 云端提供 soft label（教师）
# - 客户端对本地样本进行蒸馏训练（论文公式 20~22）
# =============================

class AdaptiveDistillWeight:
    def __init__(self, num_clients, init_weight=0.5, momentum=0.9):
        self.num_clients = num_clients
        self.weights = np.ones(num_clients) * init_weight
        self.momentum = momentum
        self.history = {i: [] for i in range(num_clients)}
        self.model_complexity = np.ones(num_clients)  # 模型复杂度评分
        
    def update_model_complexity(self, model_sizes):
        """
        更新模型复杂度评分
        Args:
            model_sizes: 每个客户端模型的参数量列表
        """
        # 归一化模型大小
        sizes_norm = (model_sizes - np.min(model_sizes)) / (np.max(model_sizes) - np.min(model_sizes) + 1e-8)
        self.model_complexity = sizes_norm
        
    def update_weights(self, accuracies, losses, confidence_scores):
        """
        基于模型性能、损失和预测置信度更新蒸馏权重
        Args:
            accuracies: 每个客户端的准确率列表
            losses: 每个客户端的损失列表
            confidence_scores: 每个客户端的预测置信度列表
        """
        # 归一化各项指标
        acc_norm = (accuracies - np.min(accuracies)) / (np.max(accuracies) - np.min(accuracies) + 1e-8)
        loss_norm = 1 - (losses - np.min(losses)) / (np.max(losses) - np.min(losses) + 1e-8)
        conf_norm = (confidence_scores - np.min(confidence_scores)) / (np.max(confidence_scores) - np.min(confidence_scores) + 1e-8)
        
        # 综合评分（考虑模型复杂度）
        scores = (0.4 * acc_norm + 0.3 * loss_norm + 0.3 * conf_norm) * (1 + 0.2 * self.model_complexity)
        
        # 更新权重
        new_weights = self.momentum * self.weights + (1 - self.momentum) * scores
        self.weights = new_weights / np.sum(new_weights)  # 归一化
        
        # 记录历史
        for i in range(self.num_clients):
            self.history[i].append(self.weights[i])
            
    def get_weights(self):
        """获取当前权重"""
        return torch.tensor(self.weights, dtype=torch.float32)

def train_local_with_distill(model, optimizer, batch_data, teacher_logits, 
                           distill_weight=None, temperature=1.0, beta=0.5, device='cpu'):
    """
    使用知识蒸馏训练本地模型
    Args:
        model: 本地模型
        optimizer: 优化器
        batch_data: (x, y) 元组
        teacher_logits: 教师模型的logits
        distill_weight: 蒸馏权重
        temperature: 蒸馏温度
        beta: 蒸馏损失权重
        device: 设备
    """
    model.train()
    x, y = batch_data
    x, y = x.to(device), y.to(device)
    teacher_logits = teacher_logits.to(device)
    
    # 前向传播
    student_logits = model(x)
    
    # 检查NaN
    if torch.isnan(student_logits).any():
        print("[WARNING] NaN detected in student_logits")
        return 0.0, 0.0, 0.0, 0.0
    
    # 计算交叉熵损失
    ce_loss = F.cross_entropy(student_logits, y)
    
    # 检查NaN
    if torch.isnan(ce_loss):
        print("[WARNING] NaN detected in ce_loss")
        return 0.0, 0.0, 0.0, 0.0
    
    # 计算知识蒸馏损失
    p_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    
    # 检查概率分布
    if torch.isnan(p_s).any() or torch.isnan(p_t).any():
        print("[WARNING] NaN detected in probability distributions")
        kd_loss = torch.tensor(0.0, device=device)
    else:
        kd_loss = F.kl_div(p_s, p_t, reduction='batchmean') * (temperature ** 2)
        
        # 检查KD损失
        if torch.isnan(kd_loss):
            print("[WARNING] NaN detected in kd_loss")
            kd_loss = torch.tensor(0.0, device=device)
    
    # 使用自适应权重
    if distill_weight is not None:
        beta = distill_weight
    
    # 总损失
    loss = (1 - beta) * ce_loss + beta * kd_loss
    
    # 检查总损失
    if torch.isnan(loss):
        print("[WARNING] NaN detected in total loss")
        return 0.0, 0.0, 0.0, 0.0
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # 计算预测置信度
    with torch.no_grad():
        probs = F.softmax(student_logits, dim=-1)
        confidence = torch.max(probs, dim=1)[0].mean().item()
        
        # 检查置信度
        if np.isnan(confidence):
            confidence = 0.0
    
    return loss.item(), ce_loss.item(), kd_loss.item(), confidence
