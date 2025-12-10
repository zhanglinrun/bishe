import torch
import copy
from collections import OrderedDict
from FLAlgorithms.servers.serverbase import Server
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
        增加：追踪并保存最佳模型参数
        """
        best_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())

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

            # --- 追踪最佳模型 ---
            if accuracy > best_acc:
                best_acc = accuracy
                best_model_weights = copy.deepcopy(self.model.state_dict())
                acc_msg = f"{accuracy:.2f}% (*)" # 标记为最佳
            else:
                acc_msg = f"{accuracy:.2f}%"
            # -------------------
            
            # 日志输出
            message = f"Round {round_num}/{self.num_rounds} | " \
                     f"Local Loss: {avg_local_loss:.4f} | " \
                     f"Test Loss: {test_loss:.4f} | " \
                     f"Test Accuracy: {acc_msg}"
            
            if logger:
                logger.info(message)
            else:
                print(message)
        
        # 训练结束后，将模型恢复为最佳状态
        if logger:
            logger.info(f"训练结束。恢复最佳模型参数，准确率: {best_acc:.2f}%")
        else:
            print(f"训练结束。恢复最佳模型参数，准确率: {best_acc:.2f}%")
        
        self.model.load_state_dict(best_model_weights)