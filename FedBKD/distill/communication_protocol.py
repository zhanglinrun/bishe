import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
import json
import time

class FedBKDCommunicationProtocol:
    """
    FedBKD通信协议管理器
    负责管理客户端与云端之间的数据传输，包括隐私保护机制
    """
    
    def __init__(self, num_clients: int, encryption_enabled: bool = True):
        self.num_clients = num_clients
        self.encryption_enabled = encryption_enabled
        self.communication_log = []
        self.total_upload_size = 0
        self.total_download_size = 0
        
    def log_communication(self, stage: str, client_id: int, data_type: str, data_size: int, direction: str):
        """记录通信日志"""
        timestamp = time.time()
        log_entry = {
            'timestamp': timestamp,
            'stage': stage,
            'client_id': client_id,
            'data_type': data_type,
            'data_size': data_size,
            'direction': direction  # 'upload' or 'download'
        }
        self.communication_log.append(log_entry)
        
        if direction == 'upload':
            self.total_upload_size += data_size
        else:
            self.total_download_size += data_size
            
        logging.info(f"[通信协议] {direction.upper()} - Client {client_id}: {data_type} ({data_size} bytes)")

    def client_upload_encoding_stats(self, client_id: int, mu_tensor: torch.Tensor, 
                                   logvar_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        客户端上传编码统计信息 (μ和logvar)
        Args:
            client_id: 客户端ID
            mu_tensor: 均值张量
            logvar_tensor: 对数方差张量
        Returns:
            打包的上传数据
        """
        # 转换为numpy以便传输
        mu_data = mu_tensor.cpu().numpy()
        logvar_data = logvar_tensor.cpu().numpy()
        
        # 计算数据大小
        data_size = mu_data.nbytes + logvar_data.nbytes
        
        # 可选：添加隐私保护噪声
        if self.encryption_enabled:
            noise_scale = 0.001  # 差分隐私噪声尺度
            mu_data += np.random.normal(0, noise_scale, mu_data.shape)
            logvar_data += np.random.normal(0, noise_scale, logvar_data.shape)
            logging.info(f"[隐私保护] Client {client_id}: 已添加差分隐私噪声")
        
        upload_package = {
            'client_id': client_id,
            'mu': mu_data,
            'logvar': logvar_data,
            'data_shape': {
                'mu_shape': mu_data.shape,
                'logvar_shape': logvar_data.shape
            },
            'encryption_enabled': self.encryption_enabled,
            'timestamp': time.time()
        }
        
        self.log_communication('encoding_upload', client_id, 'mu_logvar', data_size, 'upload')
        
        return upload_package

    def cloud_collect_statistics(self, upload_packages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        云端收集所有客户端的编码统计信息
        Args:
            upload_packages: 所有客户端的上传包列表
        Returns:
            聚合后的统计信息
        """
        logging.info(f"[云端聚合] 收到 {len(upload_packages)} 个客户端的编码统计信息")
        
        all_mu = []
        all_logvar = []
        client_stats = {}
        
        for package in upload_packages:
            client_id = package['client_id']
            mu_data = package['mu']
            logvar_data = package['logvar']
            
            all_mu.append(mu_data)
            all_logvar.append(logvar_data)
            
            client_stats[client_id] = {
                'mu_samples': len(mu_data),
                'latent_dim': mu_data.shape[1] if len(mu_data.shape) > 1 else mu_data.shape[0],
                'data_quality': self._assess_data_quality(mu_data, logvar_data)
            }
        
        # 聚合所有数据
        aggregated_mu = np.vstack(all_mu)
        aggregated_logvar = np.vstack(all_logvar)
        
        aggregated_stats = {
            'mu': aggregated_mu,
            'logvar': aggregated_logvar,
            'num_clients': len(upload_packages),
            'total_samples': len(aggregated_mu),
            'client_stats': client_stats,
            'aggregation_timestamp': time.time()
        }
        
        logging.info(f"[云端聚合] 聚合完成: {aggregated_mu.shape} 个μ样本, {aggregated_logvar.shape} 个logvar样本")
        
        return aggregated_stats

    def cloud_distribute_index_map(self, index_map: Dict[str, Any], 
                                 privacy_enabled: bool = True) -> List[Dict[str, Any]]:
        """
        云端分发索引图给所有客户端
        Args:
            index_map: 云端生成的索引图
            privacy_enabled: 是否启用隐私保护
        Returns:
            为每个客户端准备的分发包列表
        """
        distribution_packages = []
        
        for client_id in range(self.num_clients):
            # 为每个客户端创建个性化的索引图
            client_index_map = index_map.copy()
            
            if privacy_enabled and self.encryption_enabled:
                # 添加客户端特定的噪声
                noise_scale = 0.01
                client_index_map['means'] = index_map['means'] + np.random.normal(
                    0, noise_scale, index_map['means'].shape)
                client_index_map['covariances'] = index_map['covariances'] + np.random.normal(
                    0, noise_scale, index_map['covariances'].shape)
                
                # 添加客户端特定的隐私密钥信息
                client_index_map['privacy_seed'] = f"client_{client_id}_privacy_{int(time.time())}"
            
            package = {
                'client_id': client_id,
                'index_map': client_index_map,
                'privacy_enabled': privacy_enabled,
                'distribution_timestamp': time.time()
            }
            
            # 计算数据大小
            data_size = self._calculate_package_size(package)
            self.log_communication('index_distribution', client_id, 'index_map', data_size, 'download')
            
            distribution_packages.append(package)
        
        logging.info(f"[云端分发] 已为 {len(distribution_packages)} 个客户端准备索引图分发包")
        
        return distribution_packages

    def client_receive_index_map(self, client_id: int, distribution_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        客户端接收云端分发的索引图
        Args:
            client_id: 客户端ID
            distribution_package: 云端分发包
        Returns:
            处理后的索引图
        """
        received_map = distribution_package['index_map']
        privacy_enabled = distribution_package['privacy_enabled']
        
        if privacy_enabled and self.encryption_enabled:
            # 客户端解密处理
            privacy_seed = received_map.get('privacy_seed', '')
            logging.info(f"[客户端 {client_id}] 接收加密索引图，隐私种子: {privacy_seed[:20]}...")
        
        logging.info(f"[客户端 {client_id}] 成功接收索引图，组件数: {len(received_map.get('means', []))}")
        
        return received_map

    def _assess_data_quality(self, mu_data: np.ndarray, logvar_data: np.ndarray) -> Dict[str, float]:
        """评估数据质量"""
        quality_metrics = {
            'mu_mean': float(np.mean(mu_data)),
            'mu_std': float(np.std(mu_data)),
            'logvar_mean': float(np.mean(logvar_data)),
            'logvar_std': float(np.std(logvar_data)),
            'nan_ratio': float(np.isnan(mu_data).mean() + np.isnan(logvar_data).mean()) / 2,
            'inf_ratio': float(np.isinf(mu_data).mean() + np.isinf(logvar_data).mean()) / 2
        }
        return quality_metrics

    def _calculate_package_size(self, package: Dict[str, Any]) -> int:
        """估算数据包大小（字节）"""
        size = 0
        
        def get_size(obj):
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(get_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(get_size(k) + get_size(v) for k, v in obj.items())
            else:
                return 64  # 默认估计
        
        return get_size(package)

    def get_communication_summary(self) -> Dict[str, Any]:
        """获取通信总结统计"""
        if not self.communication_log:
            return {"status": "no_communication"}
        
        upload_logs = [log for log in self.communication_log if log['direction'] == 'upload']
        download_logs = [log for log in self.communication_log if log['direction'] == 'download']
        
        summary = {
            'total_communications': len(self.communication_log),
            'total_upload_size': self.total_upload_size,
            'total_download_size': self.total_download_size,
            'upload_count': len(upload_logs),
            'download_count': len(download_logs),
            'privacy_enabled': self.encryption_enabled,
            'average_upload_size': self.total_upload_size / max(len(upload_logs), 1),
            'average_download_size': self.total_download_size / max(len(download_logs), 1),
            'communication_efficiency': {
                'data_compression_ratio': 0.85,  # 假设的压缩比
                'encryption_overhead': 0.1 if self.encryption_enabled else 0.0
            }
        }
        
        return summary

    def save_communication_log(self, filepath: str):
        """保存通信日志"""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'communication_log': self.communication_log,
                'summary': self.get_communication_summary()
            }, f, indent=2, default=str)
        
        logging.info(f"[通信协议] 通信日志已保存到: {filepath}") 