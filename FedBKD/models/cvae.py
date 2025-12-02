import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gmm_index_map import GMMIndexMapGenerator

class CVAE(nn.Module):
    def __init__(self, input_shape=(2, 128), latent_dim=128, num_classes=10, n_components=10):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.n_components = n_components

        # 添加变形等效结构的参数
        self.deformation_A = nn.Parameter(torch.eye(latent_dim) + 0.1 * torch.randn(latent_dim, latent_dim))
        self.deformation_b = nn.Parameter(torch.zeros(latent_dim))

        # 编码器：提取输入特征
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 3), stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.LeakyReLU(0.2)
        )

        self._get_flattened_size()  # 自动计算展平大小

        # 均值与方差映射层
        self.fc_mu = nn.Linear(self._to_linear, latent_dim)
        self.fc_logvar = nn.Linear(self._to_linear, latent_dim)

        # 解码器输入层（z + onehot）
        self.decoder_fc = nn.Linear(latent_dim + num_classes, self._to_linear)

        # 解码器
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(1, 3), stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 3), stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=(2, 3), stride=1, padding=1),
            nn.Tanh()
        )

        self._init_weights()

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *self.input_shape)
            x = self.encoder_conv(dummy)
            self._to_linear = x.view(1, -1).shape[1]
            self.encoder_output_shape = x.shape[1:]
            print(f"[DEBUG] Encoder output shape: {x.shape}")
            print(f"[DEBUG] Flattened size: {self._to_linear}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        # 检查y是否已经是one-hot编码
        if len(y.shape) == 2 and y.shape[1] == self.num_classes:
            # y已经是one-hot编码
            y_onehot = y
        else:
            # y是索引，需要转换为one-hot编码
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        
        zy = torch.cat([z, y_onehot], dim=1)
        x = self.decoder_fc(zy)
        x = x.view(x.size(0), *self.encoder_output_shape)  # 自动适应维度
        x = self.decoder_deconv(x)
        return x

    def forward(self, x, y):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # 重构损失 - 使用更稳定的L1损失
        recon_loss = F.l1_loss(recon_x, x, reduction='mean')
        
        # KL散度损失 - 调整权重
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 总损失 - 平衡重构和正则化
        total_loss = recon_loss + 0.01 * kld  # 降低KL散度权重
        
        return total_loss

    def generate(self, class_idx, num_samples, device='cpu'):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            y = torch.full((num_samples,), class_idx, dtype=torch.long).to(device)
            samples = self.decode(z, y)
        return samples

    def fit_gmm_index_map(self, mu_list):
        self.index_gen = GMMIndexMapGenerator(n_components=self.n_components)
        # 如果传入的是单个tensor，转换为列表
        if isinstance(mu_list, torch.Tensor):
            mu_list = [mu_list]
        self.index_gen.fit(mu_list)

    def apply_deformation(self, z):
        """
        应用变形等效结构：z' = Az + b
        Args:
            z: 原始潜变量
        Returns:
            变形后的潜变量
        """
        # 确保A是正交矩阵，保持分布特性
        A = self.deformation_A
        try:
            # Gram-Schmidt正交化过程
            U, S, V = torch.svd(A)
            A_ortho = torch.mm(U, V.t())
        except:
            # 如果SVD失败，使用原始矩阵的归一化版本
            A_ortho = A / (torch.norm(A, dim=1, keepdim=True) + 1e-8)
        
        # 应用变形：z' = Az + b
        z_deformed = torch.mm(z, A_ortho.t()) + self.deformation_b.unsqueeze(0)
        return z_deformed

    def update_deformation_parameters(self, learning_rate=0.01):
        """
        自适应更新变形参数，增强多样性
        """
        with torch.no_grad():
            # 添加小幅度随机扰动
            noise_A = torch.randn_like(self.deformation_A) * learning_rate
            noise_b = torch.randn_like(self.deformation_b) * learning_rate
            
            self.deformation_A.add_(noise_A)
            self.deformation_b.add_(noise_b)
            
            # 保持A接近正交矩阵
            A_norm = torch.norm(self.deformation_A, dim=1, keepdim=True)
            self.deformation_A.div_(A_norm + 1e-8)

    def generate_pseudo_sample_from_index(self, n_samples, class_idx, device='cpu', apply_deformation=True):
        if not hasattr(self, 'index_gen'):
            raise RuntimeError("GMM index generator not initialized. Call fit_gmm_index_map() first.")

        self.eval()  # 确保模型在评估模式
        with torch.no_grad():  # 不计算梯度
            z = self.index_gen.sample_from_index_map(n_samples=n_samples, class_idx=class_idx).to(device)
            
            # 应用变形等效结构（增强伪样本多样性）
            if apply_deformation:
                z = self.apply_deformation(z)
            
            y = torch.zeros(n_samples, self.num_classes, device=device)
            y[torch.arange(n_samples), class_idx] = 1.0
            x_fake = self.decode(z, y)
            
            # 清理临时变量
            del z, y
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return x_fake

    def collect_encoding_statistics(self, dataloader, device='cpu'):
        """
        客户端收集编码统计信息用于上传到云端
        :param dataloader: 数据加载器
        :param device: 设备
        :return: tuple - (mu_list, logvar_list)
        """
        self.eval()
        mu_list = []
        logvar_list = []
        
        print(f"[客户端编码] 开始收集编码统计信息...")
        
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(device)
                mu, logvar = self.encode(x)
                mu_list.append(mu.cpu())
                logvar_list.append(logvar.cpu())
                
                # 定期清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 合并所有批次的结果
        all_mu = torch.cat(mu_list)
        all_logvar = torch.cat(logvar_list)
        
        print(f"[客户端编码] 收集完成，μ形状: {all_mu.shape}, logvar形状: {all_logvar.shape}")
        
        return all_mu, all_logvar


def train_cvae(model, dataloader, optimizer, device='cpu', epochs=50):
    model.train()
    best_loss = float('inf')
    patience = 5
    no_improve = 0

    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4

    for epoch in range(epochs):
        total_loss, total_recon, total_kld = 0, 0, 0
        batch_count = 0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            recon, mu, logvar = model(x, y)
            loss = model.loss_function(recon, x, mu, logvar)

            if torch.isnan(loss):
                print("[WARNING] NaN loss detected, skipping batch.")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()

            total_loss += loss.item()
            total_recon += F.mse_loss(recon, x, reduction='mean').item()
            total_kld += (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())).item()
            batch_count += 1
            
            # 清理中间变量以节省内存
            del recon, mu, logvar, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if batch_count == 0:
            print(f"[WARNING] No valid batches in epoch {epoch+1}")
            continue

        avg_loss = total_loss / batch_count
        avg_recon = total_recon / batch_count
        avg_kld = total_kld / batch_count

        print(f"[CVAE] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[CVAE] Early stopping at epoch {epoch+1}")
                break


def get_cvae_model(input_shape=(2, 128), latent_dim=128, num_classes=10, n_components=10):
    return CVAE(input_shape=input_shape, latent_dim=latent_dim, num_classes=num_classes, n_components=n_components)
