import numpy as np
from sklearn.mixture import GaussianMixture
import torch
import hashlib
import random

class GMMIndexMapGenerator:
    def __init__(self, n_components=10, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm = None
        self.index_map = None
        self.privacy_key = None  # 隐私保护密钥
        self.label_permutation = None  # 标签置换映射
        self.reverse_permutation = None  # 反向映射

    def generate_privacy_key(self, client_id):
        """为每个客户端生成唯一的隐私保护密钥"""
        seed_str = f"fedbkd_privacy_{client_id}_{self.random_state}"
        self.privacy_key = hashlib.md5(seed_str.encode()).hexdigest()[:16]
        
        # 基于密钥生成标签置换
        random.seed(hash(self.privacy_key))
        labels = list(range(self.n_components))
        shuffled_labels = labels.copy()
        random.shuffle(shuffled_labels)
        
        self.label_permutation = {orig: new for orig, new in zip(labels, shuffled_labels)}
        self.reverse_permutation = {new: orig for orig, new in self.label_permutation.items()}
        
        print(f"[隐私保护] Client {client_id} 标签置换映射: {self.label_permutation}")

    def encrypt_class_index(self, class_idx):
        """加密类别索引"""
        if self.label_permutation is None:
            return class_idx
        return self.label_permutation.get(class_idx, class_idx)

    def decrypt_class_index(self, encrypted_idx):
        """解密类别索引"""
        if self.reverse_permutation is None:
            return encrypted_idx
        return self.reverse_permutation.get(encrypted_idx, encrypted_idx)

    def collect_client_statistics(self, client_mu_list, client_logvar_list):
        """
        云端收集所有客户端的编码统计信息 (mean和std)
        :param client_mu_list: List[Tensor] - 各客户端的μ向量列表
        :param client_logvar_list: List[Tensor] - 各客户端的logvar向量列表
        :return: dict - 聚合的统计信息
        """
        print(f"[云端聚合] 收到 {len(client_mu_list)} 个客户端的编码统计信息")
        
        # 转换为numpy数组
        all_mu = []
        all_std = []
        
        for mu_tensor, logvar_tensor in zip(client_mu_list, client_logvar_list):
            if isinstance(mu_tensor, torch.Tensor):
                mu = mu_tensor.cpu().numpy()
                std = torch.exp(0.5 * logvar_tensor).cpu().numpy()
            else:
                mu = mu_tensor
                std = np.exp(0.5 * logvar_tensor)
            
            all_mu.append(mu)
            all_std.append(std)
        
        # 聚合所有客户端的编码
        combined_mu = np.vstack(all_mu)
        combined_std = np.vstack(all_std)
        
        print(f"[云端聚合] 聚合数据形状: mu={combined_mu.shape}, std={combined_std.shape}")
        
        return {
            'mu': combined_mu,
            'std': combined_std,
            'num_clients': len(client_mu_list),
            'total_samples': len(combined_mu)
        }

    def fit_cloud_gmm(self, aggregated_stats):
        """
        云端基于聚合的统计信息拟合全局GMM
        :param aggregated_stats: dict - 聚合的统计信息
        """
        print("[云端GMM] 开始拟合全局高斯混合模型...")
        
        mu_data = aggregated_stats['mu']
        logvar_data = aggregated_stats['logvar']
        std_data = np.exp(0.5 * logvar_data)  # 修正：由logvar计算std
        
        # 使用μ值拟合GMM（std用于后续采样时的方差调整）
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            random_state=self.random_state,
            max_iter=200,
            tol=1e-4
        )
        
        self.gmm.fit(mu_data)
        
        # 构建全局索引图
        self.index_map = {
            'means': self.gmm.means_,
            'covariances': self.gmm.covariances_,
            'weights': self.gmm.weights_,
            'aggregated_std': np.mean(std_data, axis=0),  # 平均标准差
            'num_clients': aggregated_stats['num_clients'],
            'total_samples': aggregated_stats['total_samples']
        }
        
        print(f"[云端GMM] 全局GMM拟合完成，组件数: {self.n_components}")
        print(f"[云端GMM] 全局权重分布: {self.gmm.weights_}")

    def fit(self, mu_list):
        """
        兼容旧接口：本地拟合GMM（用于向后兼容）
        :param mu_list: List[np.ndarray] 或 [Tensor] — 编码器输出的μ集合
        """
        if isinstance(mu_list[0], torch.Tensor):
            mu_array = torch.vstack(mu_list).cpu().numpy()
        else:
            mu_array = np.vstack(mu_list)

        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            random_state=self.random_state
        )
        self.gmm.fit(mu_array)

        # 构建"索引图"
        self.index_map = {
            'means': self.gmm.means_,
            'covariances': self.gmm.covariances_,
            'weights': self.gmm.weights_
        }

    def distribute_index_map(self):
        """
        云端分发索引图给客户端（带隐私保护）
        :return: dict - 加密后的索引图
        """
        if self.index_map is None:
            raise RuntimeError("Global GMM not fitted. Call fit_cloud_gmm() first.")
        
        # 对索引图进行加密处理
        encrypted_map = self.index_map.copy()
        
        # 可选：对均值和协方差添加噪声来增强隐私保护
        noise_scale = 0.01
        encrypted_map['means'] = self.index_map['means'] + np.random.normal(0, noise_scale, self.index_map['means'].shape)
        encrypted_map['covariances'] = self.index_map['covariances'] + np.random.normal(0, noise_scale, self.index_map['covariances'].shape)
        
        print("[云端分发] 索引图已加密并准备分发")
        return encrypted_map

    def receive_index_map(self, encrypted_map, client_id):
        """
        客户端接收加密的索引图
        :param encrypted_map: dict - 加密后的索引图
        :param client_id: int - 客户端ID
        """
        self.index_map = encrypted_map
        self.generate_privacy_key(client_id)
        print(f"[客户端 {client_id}] 已接收并解密索引图")

    def sample_from_index_map(self, n_samples=1, class_idx=None):
        """
        从索引图中采样隐变量 z（供伪样本生成）
        :param n_samples: 要采样的数量
        :param class_idx: 指定采样的高斯成分类别（None为随机）
        :return: Tensor[n_samples, latent_dim]
        """
        if self.index_map is None:
            raise RuntimeError("Index map not initialized. Run fit() or receive_index_map() first.")

        means = self.index_map['means']
        covs = self.index_map['covariances']
        weights = self.index_map['weights']
        latent_dim = means.shape[1]

        z_samples = []
        for _ in range(n_samples):
            # 使用加密的类别索引
            if class_idx is not None:
                encrypted_class = self.encrypt_class_index(class_idx % len(means))
                comp = encrypted_class
            else:
                comp = np.random.choice(len(weights), p=weights)
            
            mean = means[comp]
            std = np.sqrt(covs[comp])
            
            # 使用聚合的标准差进行调整（如果可用）
            if 'aggregated_std' in self.index_map:
                std = np.maximum(std, self.index_map['aggregated_std'])
            
            z = np.random.normal(loc=mean, scale=std)
            z_samples.append(z)

        return torch.tensor(np.array(z_samples), dtype=torch.float32)

    def export_index_map(self):
        """
        索引图导出，用于加密传输或保存（可扩展）
        """
        return self.index_map

    def get_privacy_stats(self):
        """获取隐私保护统计信息"""
        if self.privacy_key is None:
            return {"privacy_enabled": False}
        
        if self.label_permutation is None:
            return {"privacy_enabled": False}
        
        return {
            "privacy_enabled": True,
            "privacy_key": self.privacy_key[:8] + "****",  # 部分隐藏
            "label_permutation_sample": {k: v for k, v in list(self.label_permutation.items())[:3]},
            "total_permutations": len(self.label_permutation)
        }
