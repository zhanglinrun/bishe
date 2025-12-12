"""
FDAM 服务器 (重构版)
参考FedDiff，使用DiffusionGenerator进行伪样本生成和服务器端校正
"""

import torch
import torch.nn as nn
import copy
from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator
from utils.model_utils import average_weights


class ServerFDAM(Server):
    def __init__(self, model, users, num_rounds, device='cpu',
                 pseudo_batch_size=128, distill_steps=5, distill_lr=1e-4,
                 diffusion_steps=1000, pseudo_start_round=5, 
                 guidance_scale=3.0, correction_alpha=0.7):
        super(ServerFDAM, self).__init__(model, users, num_rounds, device)
        self.num_classes = model.num_classes
        self.signal_length = model.signal_length
        self.pseudo_batch_size = pseudo_batch_size
        self.distill_steps = distill_steps
        self.distill_lr = distill_lr
        self.pseudo_start_round = pseudo_start_round
        self.guidance_scale = guidance_scale
        self.correction_alpha = correction_alpha

        self.generator = DiffusionGenerator(
            num_classes=self.num_classes,
            signal_length=self.signal_length,
            in_channels=2,
            base_channels=64,
            channel_mults=(1, 2, 4),
            timesteps=diffusion_steps,
            beta_schedule="cosine"
        ).to(device)

        self.correction_criterion = nn.CrossEntropyLoss()

    def send_parameters(self):
        global_params = self.model.state_dict()
        gen_params = self.generator.state_dict()
        for user in self.users:
            user.set_parameters(global_params)
            user.set_generator_parameters(gen_params)

    def aggregate_parameters(self):
        client_params = []
        gen_params = []
        client_weights = []
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

    def server_correction(self, round_num):
        if self.distill_steps <= 0:
            return 0.0

        aggregated_state = copy.deepcopy(self.model.state_dict())
        self.model.train()
        self.generator.eval()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.distill_lr)
        total_correction_loss = 0.0
        
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
                    self.pseudo_batch_size, labels, 
                    device=self.device, guidance_scale=self.guidance_scale,
                    dynamic_threshold=True
                )
            outputs = self.model(pseudo_data.detach())
            loss = self.correction_criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_correction_loss += loss.item()

        corrected_state = self.model.state_dict()
        soft_state = {}
        for k in aggregated_state.keys():
            soft_state[k] = (1 - self.correction_alpha) * aggregated_state[k] + \
                            self.correction_alpha * corrected_state[k]
        self.model.load_state_dict(soft_state)

        return total_correction_loss / self.distill_steps

    def train(self, test_loader, local_epochs, logger=None):
        best_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())

        for round_num in range(1, self.num_rounds + 1):
            self.send_parameters()

            local_losses = []
            for user in self.users:
                loss = user.train(local_epochs, round_num=round_num)
                local_losses.append(loss)
            avg_local_loss = sum(local_losses) / len(local_losses)

            self.aggregate_parameters()

            correction_loss = 0.0
            if round_num >= self.pseudo_start_round:
                correction_loss = self.server_correction(round_num)

            accuracy, test_loss = self.evaluate(test_loader)
            self.train_losses.append(test_loss)
            self.train_accuracies.append(accuracy)

            if accuracy > best_acc:
                best_acc = accuracy
                best_model_weights = copy.deepcopy(self.model.state_dict())
                acc_msg = f"{accuracy:.2f}% (*)"
            else:
                acc_msg = f"{accuracy:.2f}%"

            client_accs = self.evaluate_clients(test_loader)
            client_acc_str = " | ".join([f"C{i}:{acc:.1f}%" for i, acc in enumerate(client_accs)])

            msg = f"Round {round_num}/{self.num_rounds} | Local Loss: {avg_local_loss:.4f} | Global: {acc_msg} | Test Loss: {test_loss:.4f} | Correct: {correction_loss:.4f} | {client_acc_str}"
            if logger:
                logger.info(msg)
            else:
                print(msg)

        if logger:
            logger.info(f"训练结束。恢复最佳模型参数，准确率: {best_acc:.2f}%")
        self.model.load_state_dict(best_model_weights)
