"""FedDiff 服务器：基于扩散模型的知识蒸馏联邦学习。"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from FLAlgorithms.servers.serveravg import ServerAVG
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator
from utils.model_utils import average_weights


class ServerFedDiff(ServerAVG):
    """使用扩散生成器进行数据自由知识迁移的服务器。"""

    def __init__(
        self,
        model,
        users,
        num_rounds: int,
        device: str = "cpu",
        pseudo_batch_size: int = 32,
        distill_steps: int = 1,
        distill_lr: float = 1e-3,
        diffusion_steps: int = 50,
    ):
        super().__init__(model, users, num_rounds, device)
        self.num_classes = model.num_classes
        self.signal_length = model.signal_length
        self.pseudo_batch_size = pseudo_batch_size
        self.distill_steps = distill_steps

        self.generator = DiffusionGenerator(
            num_classes=self.num_classes,
            signal_length=self.signal_length,
            timesteps=diffusion_steps,
        ).to(device)

        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=distill_lr)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=distill_lr)
        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def aggregate_parameters(self):
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

    def _collect_teacher_probs(self, pseudo_data: torch.Tensor) -> torch.Tensor:
        teacher_logits = []
        with torch.no_grad():
            for user in self.users:
                user.model.eval()
                teacher_logits.append(user.model(pseudo_data))
        stacked = torch.stack([torch.softmax(logits, dim=1) for logits in teacher_logits], dim=0)
        return torch.mean(stacked, dim=0)

    def _distill_once(self) -> tuple[float, float]:
        labels = torch.randint(0, self.num_classes, (self.pseudo_batch_size,), device=self.device)
        pseudo_data = self.generator.sample(self.pseudo_batch_size, labels, device=self.device)

        teacher_probs = self._collect_teacher_probs(pseudo_data)
        student_logits = self.model(pseudo_data.detach())
        kd_loss = self.kldiv(torch.log_softmax(student_logits, dim=1), teacher_probs)

        self.model_optimizer.zero_grad()
        kd_loss.backward()
        self.model_optimizer.step()

        for param in self.model.parameters():
            param.requires_grad = False
        pseudo_for_generator = self.generator.sample(self.pseudo_batch_size, labels, device=self.device)
        teacher_probs_g = self._collect_teacher_probs(pseudo_for_generator)
        student_logits_g = self.model(pseudo_for_generator)
        generator_loss = -self.kldiv(
            torch.log_softmax(student_logits_g, dim=1), teacher_probs_g
        )

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        self.generator_optimizer.step()
        for param in self.model.parameters():
            param.requires_grad = True

        return kd_loss.item(), generator_loss.item()

    def train(self, test_loader, local_epochs, logger=None):  # type: ignore[override]
        for round_num in range(1, self.num_rounds + 1):
            self.send_parameters()

            local_losses = []
            for user in self.users:
                loss = user.train(local_epochs)
                local_losses.append(loss)

            avg_local_loss = sum(local_losses) / len(local_losses)
            self.aggregate_parameters()

            kd_losses = []
            gen_losses = []
            if self.distill_steps > 0:
                self.model.train()
                for _ in range(self.distill_steps):
                    kd_loss, gen_loss = self._distill_once()
                    kd_losses.append(kd_loss)
                    gen_losses.append(gen_loss)

            accuracy, test_loss = self.evaluate(test_loader)
            self.train_losses.append(test_loss)
            self.train_accuracies.append(accuracy)

            message = (
                f"Round {round_num}/{self.num_rounds} | "
                f"Local Loss: {avg_local_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Accuracy: {accuracy:.2f}%"
            )
            if kd_losses:
                message += f" | KD Loss: {sum(kd_losses) / len(kd_losses):.4f}"
            if gen_losses:
                message += f" | Gen Loss: {sum(gen_losses) / len(gen_losses):.4f}"

            if logger:
                logger.info(message)
            else:
                print(message)
