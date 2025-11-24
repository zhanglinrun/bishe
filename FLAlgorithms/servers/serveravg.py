"""
FedAvg 服务器
实现联邦平均算法的服务器端
"""

import torch
import copy
from collections import OrderedDict
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import average_weights


class ServerAVG(Server):
    """
    FedAvg 服务器
    对所有客户端的模型参数进行加权平均
    """
    
    def __init__(self, model, users, num_rounds, device='cpu'):
        super(ServerAVG, self).__init__(model, users, num_rounds, device)
    
    def aggregate_parameters(self):
        """
        聚合客户端参数（加权平均）
        权重为各客户端的样本数比例
        """
        # 收集所有客户端的参数和样本数
        client_params = []
        client_weights = []
        
        total_samples = 0
        for user in self.users:
            client_params.append(user.get_parameters())
            num_samples = user.get_num_samples()
            client_weights.append(num_samples)
            total_samples += num_samples
        
        # 归一化权重
        client_weights = [w / total_samples for w in client_weights]
        
        # 加权平均
        global_params = average_weights(client_params, client_weights)
        
        # 更新全局模型
        self.model.load_state_dict(global_params)
    
    def train(self, test_loader, local_epochs, logger=None):
        """
        FedAvg 训练流程
        
        Args:
            test_loader: 测试数据加载器
            local_epochs: 本地训练轮数
            logger: 日志记录器
        """
        for round_num in range(1, self.num_rounds + 1):
            # 发送全局模型给客户端
            self.send_parameters()
            
            # 客户端本地训练
            local_losses = []
            for user in self.users:
                loss = user.train(local_epochs)
                local_losses.append(loss)
            
            avg_local_loss = sum(local_losses) / len(local_losses)
            
            # 聚合客户端模型
            self.aggregate_parameters()
            
            # 在测试集上评估
            accuracy, test_loss = self.evaluate(test_loader)
            
            # 记录
            self.train_losses.append(test_loss)
            self.train_accuracies.append(accuracy)
            
            # 日志输出
            message = f"Round {round_num}/{self.num_rounds} | " \
                     f"Local Loss: {avg_local_loss:.4f} | " \
                     f"Test Loss: {test_loss:.4f} | " \
                     f"Test Accuracy: {accuracy:.2f}%"
            
            if logger:
                logger.info(message)
            else:
                print(message)

