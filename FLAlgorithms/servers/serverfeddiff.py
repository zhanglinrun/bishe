"""FedDDPM 思路的联邦扩散服务器：聚合扩散生成器并用伪样本自训练分类器。"""
from __future__ import annotations

from typing import List
import torch
import torch.nn as nn

from FLAlgorithms.servers.serveravg import ServerAVG
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator
from utils.model_utils import average_weights


class ServerFedDiff(ServerAVG):
    """
    FedDDPM 风格：
    - 客户端在本地同时训练分类器与扩散生成器（噪声预测目标）。
    - 服务器聚合分类器与扩散生成器参数。
    - 使用聚合后的生成器合成伪样本，对全局分类器进行轻量自训练/正则。
    """

    def __init__(
        self,
        model,
        users,
        num_rounds: int,
        device: str = "cpu",
        pseudo_batch_size: int = 64,
        distill_steps: int = 50,
        distill_lr: float = 1e-3,
        diffusion_steps: int = 20,
        pseudo_start_round: int = 5,
        pseudo_conf_thresh: float = 0.6,
        pseudo_loss_weight: float = 0.5,
        pseudo_ramp_rounds: int = 5,
    ):
        super().__init__(model, users, num_rounds, device)
        self.num_classes = model.num_classes
        self.signal_length = model.signal_length
        self.pseudo_batch_size = pseudo_batch_size
        self.distill_steps = distill_steps
        self.pseudo_start_round = pseudo_start_round
        self.pseudo_conf_thresh = pseudo_conf_thresh
        self.pseudo_loss_weight = pseudo_loss_weight
        self.pseudo_ramp_rounds = pseudo_ramp_rounds
        
        self.generator = DiffusionGenerator(
            num_classes=self.num_classes,
            signal_length=self.signal_length,
            timesteps=diffusion_steps,
        ).to(device)

        self.server_optimizer = torch.optim.Adam(self.model.parameters(), lr=distill_lr)
        self.ce_loss = nn.CrossEntropyLoss()

    @staticmethod
    def _set_batchnorm_mode(model: nn.Module, train: bool) -> None:
        """Freeze/unfreeze BatchNorm running stats while keeping affine learnable."""
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train(mode=train)

    def send_parameters(self):
        """
        下发全局分类器与扩散生成器参数
        """
        global_params = self.model.state_dict()
        global_gen_params = self.generator.state_dict()
        for user in self.users:
            user.set_parameters(global_params)
            if hasattr(user, "set_generator_parameters"):
                user.set_generator_parameters(global_gen_params)

    def aggregate_parameters(self):
        """
        聚合分类器与扩散生成器参数（按样本数加权）
        """
        client_params: List[dict] = []
        client_gen_params: List[dict] = []
        client_weights: List[float] = []
        total_samples = 0
        for user in self.users:
            client_params.append(user.get_parameters())
            client_gen_params.append(user.get_generator_parameters())
            num_samples = user.get_num_samples()
            client_weights.append(num_samples)
            total_samples += num_samples

        client_weights = [w / total_samples for w in client_weights]
        global_params = average_weights(client_params, client_weights)
        global_gen_params = average_weights(client_gen_params, client_weights)
        self.model.load_state_dict(global_params)
        self.generator.load_state_dict(global_gen_params)

    def train(self, test_loader, local_epochs, logger=None):
        """
        联邦训练流程：
        1) 下发最新分类器与扩散生成器。
        2) 客户端本地训练（分类器 + 扩散模型）。
        3) 聚合两类参数。
        4) 使用聚合后的生成器合成伪样本，对全局分类器做少量自训练。
        """
        for round_num in range(1, self.num_rounds + 1):
            self.send_parameters()

            cls_losses = []
            gen_losses = []
            for user in self.users:
                metrics = user.train(local_epochs)
                if isinstance(metrics, dict):
                    cls_losses.append(metrics.get("cls_loss", 0.0))
                    gen_losses.append(metrics.get("gen_loss", 0.0))
                else:
                    cls_losses.append(metrics)
                    gen_losses.append(0.0)

            avg_cls_loss = sum(cls_losses) / max(len(cls_losses), 1)
            avg_gen_loss = sum(gen_losses) / max(len(gen_losses), 1)

            self.aggregate_parameters()

            # 服务器利用全局扩散模型进行伪样本自训练
            pseudo_loss = 0.0
            pseudo_updates = 0
            ramp = 0.0
            if self.pseudo_ramp_rounds > 0:
                ramp = max(0.0, round_num - self.pseudo_start_round + 1) / float(self.pseudo_ramp_rounds)
                ramp = min(ramp, 1.0)
            elif round_num >= self.pseudo_start_round:
                ramp = 1.0

            effective_steps = int(self.distill_steps * ramp) if ramp > 0 else 0
            loss_weight = self.pseudo_loss_weight * ramp

            if effective_steps > 0 and loss_weight > 0:
                self.model.train()
                # 避免伪数据污染 BN 统计量
                self._set_batchnorm_mode(self.model, train=False)
                self.generator.eval()
                for _ in range(effective_steps):
                    labels = torch.randint(0, self.num_classes, (self.pseudo_batch_size,), device=self.device)
                    with torch.no_grad():
                        pseudo_data = self.generator.sample(self.pseudo_batch_size, labels, device=self.device)
                        pseudo_data = torch.clamp(pseudo_data, -3.0, 3.0)

                    logits = self.model(pseudo_data)
                    probs = torch.softmax(logits, dim=1)
                    max_conf, preds = probs.max(dim=1)
                    # 仅在模型对伪样本高置信度且预测类别与条件标签一致时更新
                    mask = (max_conf > self.pseudo_conf_thresh) & (preds == labels)
                    if mask.any():
                        self.server_optimizer.zero_grad()
                        pseudo_targets = preds[mask].detach()
                        loss = self.ce_loss(logits[mask], pseudo_targets)
                        loss = loss * loss_weight
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.server_optimizer.step()

                        pseudo_loss += loss.item()
                        pseudo_updates += 1

                pseudo_loss = pseudo_loss / max(pseudo_updates, 1)
                # 恢复 BN 统计的正常更新
                self._set_batchnorm_mode(self.model, train=True)

            accuracy, test_loss = self.evaluate(test_loader)
            self.train_losses.append(test_loss)
            self.train_accuracies.append(accuracy)

            message = (
                f"Round {round_num}/{self.num_rounds} | "
                f"Local CE: {avg_cls_loss:.4f} | "
                f"Diff Loss: {avg_gen_loss:.4f} | "
                f"Pseudo CE: {pseudo_loss:.4f} (steps={pseudo_updates}/{effective_steps}) | "
                f"Test Acc: {accuracy:.2f}% | Test Loss: {test_loss:.4f}"
            )

            if logger:
                logger.info(message)
            else:
                print(message)

            torch.cuda.empty_cache()
