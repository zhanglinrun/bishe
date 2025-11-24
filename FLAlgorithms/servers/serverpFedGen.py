"""
FedGen 服务器
实现基于数据自由知识蒸馏的联邦学习服务器端
根据论文 "Data-Free Knowledge Distillation for Heterogeneous Federated Learning"
"""

import torch
import copy
from FLAlgorithms.servers.serverbase import Server
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
        self.gen_projection = torch.nn.Linear(generator.embedding_dim, model.num_classes).to(device)
        # 初始化投影层，需在首次发送前设置
    
    def send_parameters(self):
        """
        发送全局模型和生成器参数给所有客户端
        """
        global_params = self.model.state_dict()
        gen_params = self.generator.state_dict()
        proj_params = self.gen_projection.state_dict() if self.gen_projection is not None else None
        
        for user in self.users:
            user.set_parameters(global_params)
            user.set_generator_parameters(gen_params)
            if proj_params is not None:
                user.set_projection_parameters(proj_params)
    
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
        proj_params = []
        for user in self.users:
            gen_params.append(user.get_generator_parameters())
            proj_params.append(user.get_projection_parameters())
        
        # 加权平均生成器与投影
        global_gen_params = average_weights(gen_params, client_weights)
        self.generator.load_state_dict(global_gen_params)

        if proj_params:
            if self.gen_projection is None:
                # 以第一个客户端的形状初始化
                sample_proj = proj_params[0]
                # 找到 weight/bias 推断尺寸
                weight = sample_proj[[k for k in sample_proj.keys() if 'weight' in k][0]]
                in_features, out_features = weight.shape[1], weight.shape[0]
                self.gen_projection = torch.nn.Linear(in_features, out_features).to(self.device)
            global_proj_params = average_weights(proj_params, client_weights)
            self.gen_projection.load_state_dict(global_proj_params)
    
    def train(self, test_loader, local_epochs, logger=None):
        """
        FedGen 训练流程
        
        Args:
            test_loader: 测试数据加载器
            local_epochs: 本地训练轮数
            logger: 日志记录器
        """
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
            
            # 日志输出
            message = f"Round {round_num}/{self.num_rounds} | " \
                     f"Local Loss: {avg_local_loss:.4f} | " \
                     f"Test Loss: {test_loss:.4f} | " \
                     f"Test Accuracy: {accuracy:.2f}%"
            
            if logger:
                logger.info(message)
            else:
                print(message)

