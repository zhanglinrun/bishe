"""
FDAM 客户端
将扩散对齐模块应用于特征空间，对齐后再进行分类
"""

import torch
import torch.nn as nn
import copy
from FLAlgorithms.users.userbase import User


class UserFDAM(User):
    def __init__(self, user_id, model, aligner, train_loader, learning_rate, device='cpu',
                 optimizer_type='adam', momentum=0.9, weight_decay=1e-4,
                 lambda_diff=0.5, lambda_align=0.5, mu=0.0, noise_std=0.1):
        super(UserFDAM, self).__init__(user_id, model, train_loader, learning_rate, device,
                                      optimizer_type, momentum, weight_decay)
        self.aligner = aligner.to(device)
        self.lambda_diff = lambda_diff
        self.lambda_align = lambda_align
        self.noise_std = noise_std
        self.mu = mu
        self.global_params = copy.deepcopy(model.state_dict())
        self.global_prototypes = None  # [num_classes, feat_dim]

        # 统一优化器，包含模型 + 对齐模块
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.aligner.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.mse = nn.MSELoss()
        self.num_classes = None  # 将在首个 batch 中推断
        self.feature_dim = None

        self.prototype_sums = None
        self.prototype_counts = None

    def set_parameters(self, params):
        super().set_parameters(params)
        self.global_params = copy.deepcopy(params)

    def set_aligner_parameters(self, params):
        self.aligner.load_state_dict(copy.deepcopy(params))

    def get_aligner_parameters(self):
        return copy.deepcopy(self.aligner.state_dict())

    def set_global_prototypes(self, prototypes):
        """设置全局原型: Tensor[num_classes, feat_dim]"""
        self.global_prototypes = prototypes.to(self.device) if prototypes is not None else None

    def _init_prototype_buffers(self, feat_dim, num_classes):
        self.prototype_sums = torch.zeros(num_classes, feat_dim, device=self.device)
        self.prototype_counts = torch.zeros(num_classes, device=self.device)

    def reset_prototypes(self):
        """在新一轮联邦开始时重置本地原型统计"""
        self.prototype_sums = None
        self.prototype_counts = None
        self.feature_dim = None
        self.num_classes = None

    def get_local_prototypes(self):
        if self.prototype_sums is None:
            return None, None
        return self.prototype_sums.detach().cpu(), self.prototype_counts.detach().cpu()

    def _proximal_term(self):
        if self.mu <= 0 or self.global_params is None:
            return 0.0
        prox = 0.0
        for name, param in self.model.named_parameters():
            prox += torch.sum((param - self.global_params[name].to(self.device)) ** 2)
        return (self.mu / 2.0) * prox

    def train(self, epochs):
        self.model.train()
        self.aligner.train()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)

                # 推断维度/类别并初始化原型缓冲区
                if self.num_classes is None:
                    self.num_classes = int(y.max().item() + 1)
                if self.feature_dim is None:
                    with torch.no_grad():
                        feat_sample = self.model.extract_features(X[:1])
                        self.feature_dim = feat_sample.shape[-1]
                        self._init_prototype_buffers(self.feature_dim, self.num_classes)

                self.optimizer.zero_grad()

                # 特征提取与加噪
                feats = self.model.extract_features(X)
                noise = torch.randn_like(feats) * self.noise_std
                noisy_feats = feats + noise

                # 对齐/去噪
                aligned_feats, pred_noise = self.aligner(noisy_feats)

                # 分类
                logits = self.model.classify_from_features(aligned_feats)
                cls_loss = self.criterion(logits, y)

                # 扩散噪声预测损失
                diff_loss = self.mse(pred_noise, noise)

                # 原型对齐损失
                align_loss = torch.tensor(0.0, device=self.device)
                if self.global_prototypes is not None:
                    target_proto = self.global_prototypes[y]
                    align_loss = self.mse(aligned_feats, target_proto)

                prox = self._proximal_term()

                loss = cls_loss + self.lambda_diff * diff_loss + self.lambda_align * align_loss + prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.aligner.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 更新局部原型统计
                with torch.no_grad():
                    for c in y.unique():
                        mask = (y == c)
                        if mask.any():
                            self.prototype_sums[c] += aligned_feats[mask].sum(dim=0)
                            self.prototype_counts[c] += mask.sum()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
