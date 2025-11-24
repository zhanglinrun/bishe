"""
FedProx 服务器
实现 FedProx 算法的服务器端
服务器端聚合逻辑与 FedAvg 相同，差异在客户端训练
"""

from FLAlgorithms.servers.serveravg import ServerAVG


class ServerFedProx(ServerAVG):
    """
    FedProx 服务器
    服务器端的聚合策略与 FedAvg 完全相同
    FedProx 的特殊之处在于客户端训练时的近端项
    """
    
    def __init__(self, model, users, num_rounds, device='cpu'):
        super(ServerFedProx, self).__init__(model, users, num_rounds, device)
    
    # 继承 ServerAVG 的所有方法
    # aggregate_parameters() 和 train() 与 FedAvg 完全相同

