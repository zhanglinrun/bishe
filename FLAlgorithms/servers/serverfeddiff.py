"""
FedDiff 服务器：基于扩散模型的知识蒸馏与服务器端校正。
优化版 v4：
1. [新增] 星座图可视化 (Constellation Diagram): 在 visualize_pseudo_data 中增加 Scatter Plot，
   以便直观判断生成的 IQ 信号是否具有调制特征（如 QPSK 的 4 个簇）。
2. 保持 Best Model Reloading, Soft Correction 和 Gradient Clipping。
"""
from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

from FLAlgorithms.servers.serveravg import ServerAVG
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator
from utils.model_utils import average_weights

class ServerFedDiff(ServerAVG):
    """
    集成扩散模型生成与服务器端校正的 FL 服务器。
    """

    def __init__(
        self,
        model,
        users,
        num_rounds: int,
        device: str = 'cuda',
        pseudo_batch_size: int = 128,
        distill_steps: int = 5,
        distill_lr: float = 1e-4, 
        diffusion_steps: int = 1000,
        pseudo_start_round: int = 0,
        guidance_scale: float = 3.0,
        correction_alpha: float = 0.7,
    ):
        super().__init__(model, users, num_rounds, device)
        self.num_classes = model.num_classes
        self.signal_length = model.signal_length
        self.pseudo_batch_size = pseudo_batch_size
        self.distill_steps = distill_steps
        self.pseudo_start_round = pseudo_start_round
        self.guidance_scale = guidance_scale
        self.distill_lr = distill_lr
        self.correction_alpha = correction_alpha

        # 初始化扩散生成器
        self.generator = DiffusionGenerator(
            num_classes=self.num_classes,
            signal_length=self.signal_length,
            in_channels=2,
            base_channels=64,
            channel_mults=(1, 2, 4),
            timesteps=diffusion_steps,
            beta_schedule="cosine"
        ).to(device)

        # 损失函数
        self.correction_criterion = nn.CrossEntropyLoss()

    def load_generator(self, path):
        """加载预训练好的扩散模型权重"""
        if not os.path.exists(path):
            print(f"[Server] Warning: Pretrained generator not found at {path}. Starting from scratch.")
            return
            
        print(f"[Server] Loading Pre-trained Diffusion Model from {path}...")
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.generator.load_state_dict(state_dict)
            self.generator.eval()
            print("[Server] Generator loaded successfully.")
        except Exception as e:
            print(f"[Server] Error loading generator: {e}")

    def aggregate_parameters(self):
        """
        聚合 全局模型 和 生成器模型
        """
        client_params: List[dict] = []
        client_weights: List[float] = []
        gen_params: List[dict] = []
        
        total_samples = 0
        
        for user in self.users:
            client_params.append(user.get_parameters())
            gen_params.append(user.get_generator_parameters())
            
            num_samples = user.get_num_samples()
            client_weights.append(num_samples)
            total_samples += num_samples

        client_weights = [w / total_samples for w in client_weights]
        
        global_params = average_weights(client_params, client_weights)
        self.model.load_state_dict(global_params)
        
        global_gen_params = average_weights(gen_params, client_weights)
        self.generator.load_state_dict(global_gen_params)
        print("[Server] Aggregated Classifier and Generator parameters.")

    def visualize_pseudo_data(self, data, labels, round_num):
        """调试：保存生成的伪数据图片（时域图 + 星座图）"""
        save_dir = "results/debug_vis"
        os.makedirs(save_dir, exist_ok=True)
        
        data = data.cpu().numpy() # [B, 2, 128]
        labels = labels.cpu().numpy()
        
        # 选取前 3 个不同类别的样本进行展示
        unique_labels = np.unique(labels)
        selected_indices = []
        for l in unique_labels[:3]: # 最多展示3个类别
            idx = np.where(labels == l)[0][0]
            selected_indices.append(idx)
            
        num_plots = len(selected_indices)
        if num_plots == 0:
            return

        plt.figure(figsize=(5 * num_plots, 8)) # 增加高度以容纳两行图
        
        for i, idx in enumerate(selected_indices):
            # 1. 时域波形图 (Top Row)
            plt.subplot(2, num_plots, i + 1)
            plt.plot(data[idx, 0, :], label='I', alpha=0.8)
            plt.plot(data[idx, 1, :], label='Q', alpha=0.8)
            plt.title(f"Class {labels[idx]} - Time Domain")
            plt.legend(loc='upper right', fontsize='small')
            plt.grid(True, alpha=0.3)

            # 2. 星座图 (Bottom Row)
            plt.subplot(2, num_plots, i + 1 + num_plots)
            # Scatter plot: I (x-axis) vs Q (y-axis)
            plt.scatter(data[idx, 0, :], data[idx, 1, :], alpha=0.6, s=10)
            plt.title(f"Class {labels[idx]} - Constellation")
            plt.xlabel("In-Phase (I)")
            plt.ylabel("Quadrature (Q)")
            plt.grid(True, alpha=0.3)
            # 保持横纵比例一致，否则圆会变成椭圆
            plt.axis('equal') 

        plt.tight_layout()
        plt.savefig(f"{save_dir}/round_{round_num}_pseudo.png")
        plt.close()

    def server_correction(self, round_num):
        """
        服务器端校正（软更新 + 梯度裁剪）
        """
        if self.distill_steps <= 0:
            return 0.0

        # 备份聚合后的原始模型参数
        aggregated_state = copy.deepcopy(self.model.state_dict())

        self.model.train()
        self.generator.eval()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.distill_lr)
        total_correction_loss = 0.0
        
        # 构造均衡标签
        labels = []
        samples_per_class = self.pseudo_batch_size // self.num_classes
        for c in range(self.num_classes):
            labels.append(torch.full((samples_per_class,), c, dtype=torch.long))
        remaining = self.pseudo_batch_size - len(labels) * samples_per_class
        if remaining > 0:
            labels.append(torch.randint(0, self.num_classes, (remaining,)))
        labels = torch.cat(labels).to(self.device)
        labels = labels[torch.randperm(labels.size(0))]

        for step in range(self.distill_steps):
            optimizer.zero_grad()
            
            with torch.no_grad():
                pseudo_data = self.generator.sample(
                    self.pseudo_batch_size, 
                    labels, 
                    device=self.device, 
                    guidance_scale=self.guidance_scale, 
                    dynamic_threshold=True
                )
            
            # 第一步保存调试图片
            if step == 0 and round_num % 5 == 0:
                self.visualize_pseudo_data(pseudo_data, labels, round_num)

            outputs = self.model(pseudo_data.detach())
            loss = self.correction_criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_correction_loss += loss.item()

        # 软更新
        corrected_state = self.model.state_dict()
        soft_state = {}
        for k in aggregated_state.keys():
            soft_state[k] = (1 - self.correction_alpha) * aggregated_state[k] + \
                            self.correction_alpha * corrected_state[k]
        
        self.model.load_state_dict(soft_state)

        avg_loss = total_correction_loss / self.distill_steps
        return avg_loss

    def train(self, test_loader, local_epochs, logger=None):
        best_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict()) 
        
        for round_num in range(1, self.num_rounds + 1):
            global_params = self.model.state_dict()
            gen_params = self.generator.state_dict()
            for user in self.users:
                user.set_parameters(global_params)
                user.set_generator_parameters(gen_params) 

            # 客户端训练
            local_losses = []
            for user in self.users:
                loss = user.train(local_epochs, round_num=round_num)
                local_losses.append(loss)
            avg_local_loss = sum(local_losses) / len(local_losses)

            self.aggregate_parameters()

            correction_loss = 0.0
            if round_num >= self.pseudo_start_round:
                if round_num == self.pseudo_start_round:
                    print(f"[Server] Start Soft Correction (Scale={self.guidance_scale}, Alpha={self.correction_alpha})...")
                correction_loss = self.server_correction(round_num)

            accuracy, test_loss = self.evaluate(test_loader)
            self.train_losses.append(test_loss)
            self.train_accuracies.append(accuracy)
            
            # 追踪并保存最佳模型参数
            if accuracy > best_acc:
                best_acc = accuracy
                best_model_weights = copy.deepcopy(self.model.state_dict())
                acc_msg = f"{accuracy:.2f}% (*)" 
            else:
                acc_msg = f"{accuracy:.2f}%"

            message = (
                f"Round {round_num}/{self.num_rounds} | "
                f"Local Loss: {avg_local_loss:.4f} | "
                f"Test Acc: {acc_msg} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Correct Loss: {correction_loss:.4f}"
            )

            if logger:
                logger.info(message)
            else:
                print(message)
        
        # 训练结束后，将模型恢复为最佳状态
        print(f"\n[Server] Training finished. Reloading best model with Accuracy: {best_acc:.2f}%")
        self.model.load_state_dict(best_model_weights)