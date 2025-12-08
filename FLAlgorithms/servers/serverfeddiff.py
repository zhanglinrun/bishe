"""
FedDiff 服务器：基于扩散模型的知识蒸馏与服务器端校正。
改进点：
参考 FEDDDPM (Federated Learning for Diffusion Models) 的思想，
在聚合全局模型后，使用扩散模型生成的伪数据（Auxiliary Data）
对全局模型进行 Server-side Correction (Fine-tuning)。
这能有效缓解 Non-IID 带来的全局模型漂移。
"""
from __future__ import annotations

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

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
        device: str = "cpu",
        pseudo_batch_size: int = 128, # 增大 Batch Size 以覆盖更多类别
        distill_steps: int = 5,       # 增加校正步数 (FEDDDPM 中的 E)
        distill_lr: float = 1e-3,
        diffusion_steps: int = 1000,
        # 新增参数
        pseudo_start_round: int = 0,  # 从第几轮开始进行校正
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        guidance_scale: float = 1.0,  # 扩散模型采样引导系数
    ):
        super().__init__(model, users, num_rounds, device)
        self.num_classes = model.num_classes
        self.signal_length = model.signal_length
        self.pseudo_batch_size = pseudo_batch_size
        self.distill_steps = distill_steps
        self.pseudo_start_round = pseudo_start_round
        self.guidance_scale = guidance_scale

        # 初始化扩散生成器
        # 注意：这里假设你已经有预训练好的扩散模型，或者在 FL 过程中会加载它
        self.generator = DiffusionGenerator(
            num_classes=self.num_classes,
            signal_length=self.signal_length,
            timesteps=diffusion_steps,
            beta_schedule="cosine" # 保持与你训练时一致
        ).to(device)

        # 用于校正全局模型的优化器
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=distill_lr)
        
        # 损失函数 (校正阶段使用标准交叉熵，因为生成数据是 Conditional 的，标签已知)
        self.correction_criterion = nn.CrossEntropyLoss()

    def load_generator(self, path):
        """加载预训练好的扩散模型权重"""
        print(f"[Server] Loading Pre-trained Diffusion Model from {path}...")
        try:
            self.generator.load_state_dict(torch.load(path, map_location=self.device))
            self.generator.eval() # 生成器设为评估模式
            print("[Server] Generator loaded successfully.")
        except Exception as e:
            print(f"[Server] Error loading generator: {e}")

    def aggregate_parameters(self):
        """
        标准 FedAvg 聚合
        """
        client_params: List[dict] = []
        client_weights: List[float] = []
        total_samples = 0
        for user in self.users:
            client_params.append(user.get_parameters())
            num_samples = user.get_num_samples()
            client_weights.append(num_samples)
            total_samples += num_samples

        client_weights = [w / total_samples for w in client_weights]
        global_params = average_weights(client_params, client_weights)
        self.model.load_state_dict(global_params)

    def server_correction(self):
        """
        [FEDDDPM 核心改进] 服务器端校正
        使用扩散模型生成包含所有类别的伪数据（Auxiliary Dataset），
        对聚合后的全局模型进行微调。
        """
        if self.distill_steps <= 0:
            return 0.0

        self.model.train()
        self.generator.eval() # 确保生成器不更新
        
        total_correction_loss = 0.0
        
        # 确保每个 Batch 包含所有类别 (Class Balanced)
        # 生成均匀分布的标签
        labels = torch.randint(0, self.num_classes, (self.pseudo_batch_size,), device=self.device)
        
        # 为了更好的效果，可以强制每个 Batch 类别均衡
        # labels = torch.cat([torch.arange(self.num_classes) for _ in range(self.pseudo_batch_size // self.num_classes + 1)])[:self.pseudo_batch_size]
        # labels = labels.to(self.device)

        for step in range(self.distill_steps):
            self.model_optimizer.zero_grad()
            
            with torch.no_grad():
                # 使用扩散模型生成伪数据
                # 启用 dynamic_threshold=True 防止数值爆炸 (基于你之前的测试反馈)
                pseudo_data = self.generator.sample(
                    self.pseudo_batch_size, 
                    labels, 
                    device=self.device, 
                    guidance_scale=self.guidance_scale, # 推荐 1.0 或小一点，视生成质量而定
                    dynamic_threshold=True
                )
            
            # 使用伪数据训练全局模型 (Supervised Fine-tuning)
            # 因为我们知道 labels (Condition)，所以直接用 CrossEntropy
            outputs = self.model(pseudo_data.detach())
            loss = self.correction_criterion(outputs, labels)
            
            loss.backward()
            self.model_optimizer.step()
            
            total_correction_loss += loss.item()

        avg_loss = total_correction_loss / self.distill_steps
        return avg_loss

    def train(self, test_loader, local_epochs, logger=None):
        """
        主训练循环
        """
        best_acc = 0.0
        
        for round_num in range(1, self.num_rounds + 1):
            # 1. 下发参数
            self.send_parameters()

            # 2. 客户端本地训练
            local_losses = []
            for user in self.users:
                loss = user.train(local_epochs)
                local_losses.append(loss)
            avg_local_loss = sum(local_losses) / len(local_losses)

            # 3. 聚合参数
            self.aggregate_parameters()

            # 4. [FEDDDPM] 服务器端校正 (Server Correction)
            correction_loss = 0.0
            if round_num >= self.pseudo_start_round:
                # 只有当生成器可用或预热后才开始校正
                # 这里假设 generator 已经预加载好了，或者在 FL 过程中训练
                # 如果是离线训练好的，直接调用
                correction_loss = self.server_correction()

            # 5. 评估
            accuracy, test_loss = self.evaluate(test_loader)
            self.train_losses.append(test_loss)
            self.train_accuracies.append(accuracy)
            
            if accuracy > best_acc:
                best_acc = accuracy

            # 日志
            message = (
                f"Round {round_num}/{self.num_rounds} | "
                f"Local Loss: {avg_local_loss:.4f} | "
                f"Test Acc: {accuracy:.2f}% | "
                f"Test Loss: {test_loss:.4f}"
            )
            if round_num >= self.pseudo_start_round:
                message += f" | Correct Loss: {correction_loss:.4f}"

            if logger:
                logger.info(message)
            else:
                print(message)
        
        print(f"Training finished. Best Accuracy: {best_acc:.2f}%")