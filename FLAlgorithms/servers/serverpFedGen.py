"""
FedGen 服务器
实现基于数据自由知识蒸馏的联邦学习服务器端
根据论文 "Data-Free Knowledge Distillation for Heterogeneous Federated Learning"
"""

import torch
import copy
from FLAlgorithms.servers.serverbase import Server
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.model_utils import average_weights


class ServerFedGen(Server):
    """
    FedGen 服务器
    需要同时聚合分类器和生成器
    """
    
    def __init__(self, model, generator, users, num_rounds, device='cpu'):
        """
        Args:
            model: 全局分类模型
            generator: 全局生成器
            users: 客户端列表
            num_rounds: 训练轮次
            device: 设备
        """
        super(ServerFedGen, self).__init__(model, users, num_rounds, device)
        self.generator = generator.to(device)
    
    def send_parameters(self):
        """
        发送全局模型和生成器参数给所有客户端
        """
        global_params = self.model.state_dict()
        gen_params = self.generator.state_dict()
        
        for user in self.users:
            user.set_parameters(global_params)
            user.set_generator_parameters(gen_params)
    
    def aggregate_parameters(self):
        """
        聚合客户端的分类器和生成器参数
        """
        # 聚合分类器参数
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
        
        # 加权平均分类器
        global_params = average_weights(client_params, client_weights)
        self.model.load_state_dict(global_params)
        
        # 聚合生成器参数
        gen_params = []
        for user in self.users:
            gen_params.append(user.get_generator_parameters())
        
        # 加权平均生成器
        global_gen_params = average_weights(gen_params, client_weights)
        self.generator.load_state_dict(global_gen_params)
    
    def train(self, test_loader, local_epochs, logger=None):
        """
        FedGen 训练流程
        
        Args:
            test_loader: 测试数据加载器
            local_epochs: 本地训练轮数
            logger: 日志记录器
        """
        best_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())

        for round_num in range(1, self.num_rounds + 1):
            # 发送全局模型和生成器给客户端
            self.send_parameters()
            
            # 客户端本地训练（包括生成器和分类器）
            local_losses = []
            for user in self.users:
                loss = user.train(local_epochs)
                local_losses.append(loss)
            
            avg_local_loss = sum(local_losses) / len(local_losses)
            
            # 聚合客户端模型和生成器
            self.aggregate_parameters()
            
            # 在测试集上评估
            accuracy, test_loss = self.evaluate(test_loader)
            
            # 记录
            self.train_losses.append(test_loss)
            self.train_accuracies.append(accuracy)
            
            # --- 追踪最佳模型 ---
            if accuracy > best_acc:
                best_acc = accuracy
                best_model_weights = copy.deepcopy(self.model.state_dict())
                acc_msg = f"{accuracy:.2f}% (*)" 
            else:
                acc_msg = f"{accuracy:.2f}%"
            # -------------------

            # 评估各客户端准确率
            client_accs = self.evaluate_clients(test_loader)
            client_acc_str = " | ".join([f"C{i}:{acc:.1f}%" for i, acc in enumerate(client_accs)])
            
            # 日志输出
            message = f"Round {round_num}/{self.num_rounds} | " \
                     f"Local Loss: {avg_local_loss:.4f} | " \
                     f"Test Loss: {test_loss:.4f} | " \
                     f"Global: {acc_msg} | {client_acc_str}"
            
            if logger:
                logger.info(message)
            else:
                print(message)

        # 训练结束后，将模型恢复为最佳状态
        final_msg = f"训练结束。恢复最佳模型参数，准确率: {best_acc:.2f}%"
        if logger:
            logger.info(final_msg)
        else:
            print(f"\n[Server] {final_msg}")
        
        self.model.load_state_dict(best_model_weights)