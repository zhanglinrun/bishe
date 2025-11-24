"""
FDAM 服务器
负责扩散对齐模型、分类器与类原型的聚合
"""

import torch
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import average_weights


class ServerFDAM(Server):
    def __init__(self, model, aligner, users, num_rounds, num_classes, feature_dim, align_beta=0.5, device='cpu'):
        super(ServerFDAM, self).__init__(model, users, num_rounds, device)
        self.aligner = aligner.to(device)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.align_beta = align_beta

        self.prototype_sums = torch.zeros(num_classes, feature_dim, device=device)
        self.prototype_counts = torch.zeros(num_classes, device=device)

    def send_parameters(self):
        global_params = self.model.state_dict()
        aligner_params = self.aligner.state_dict()
        prototypes = self.get_global_prototypes()

        for user in self.users:
            user.set_parameters(global_params)
            user.set_aligner_parameters(aligner_params)
            user.set_global_prototypes(prototypes)
            user.reset_prototypes()

    def get_global_prototypes(self):
        counts = self.prototype_counts.clamp(min=1e-6).unsqueeze(-1)
        return self.prototype_sums / counts

    def aggregate_prototypes(self, client_prototypes):
        # client_prototypes: list of (sum_tensor, count_tensor)
        total_sum = torch.zeros_like(self.prototype_sums)
        total_count = torch.zeros_like(self.prototype_counts)

        for proto_sum, proto_count in client_prototypes:
            if proto_sum is None or proto_count is None:
                continue
            total_sum += proto_sum.to(self.device)
            total_count += proto_count.to(self.device)

        self.prototype_sums = total_sum
        self.prototype_counts = total_count

    def aggregate_parameters(self):
        client_params = []
        align_params = []
        client_weights = []
        prototype_reports = []

        total_samples = 0
        # 计算基于原型偏移的权重
        for user in self.users:
            proto_sum, proto_count = user.get_local_prototypes()
            prototype_reports.append((proto_sum, proto_count))

            num_samples = user.get_num_samples()
            delta = 0.0
            if proto_sum is not None and proto_count is not None:
                with torch.no_grad():
                    global_proto = self.get_global_prototypes().to(self.device)
                    proto_sum_dev = proto_sum.to(self.device)
                    proto_count_dev = proto_count.to(self.device)
                    valid = proto_count_dev > 0
                    if valid.any():
                        local_proto = proto_sum_dev / proto_count_dev.clamp(min=1e-6).unsqueeze(-1)
                        delta = torch.mean(torch.norm(local_proto[valid] - global_proto[valid], dim=1)).item()
            weight = num_samples / (1 + self.align_beta * delta)
            client_weights.append(weight)
            total_samples += weight

            client_params.append(user.get_parameters())
            align_params.append(user.get_aligner_parameters())

        client_weights = [w / total_samples for w in client_weights]

        # 聚合模型
        global_params = average_weights(client_params, client_weights)
        self.model.load_state_dict(global_params)

        # 聚合对齐模块
        global_align = average_weights(align_params, client_weights)
        self.aligner.load_state_dict(global_align)

        # 聚合原型
        self.aggregate_prototypes(prototype_reports)

    def train(self, test_loader, local_epochs, logger=None):
        for round_num in range(1, self.num_rounds + 1):
            self.send_parameters()

            local_losses = []
            for user in self.users:
                loss = user.train(local_epochs)
                local_losses.append(loss)

            avg_local_loss = sum(local_losses) / len(local_losses)

            self.aggregate_parameters()

            accuracy, test_loss = self.evaluate(test_loader)
            self.train_losses.append(test_loss)
            self.train_accuracies.append(accuracy)

            msg = f"[FDAM] Round {round_num}/{self.num_rounds} | Local Loss: {avg_local_loss:.4f} | Test Loss: {test_loss:.4f} | Acc: {accuracy:.2f}%"
            if logger:
                logger.info(msg)
            else:
                print(msg)
