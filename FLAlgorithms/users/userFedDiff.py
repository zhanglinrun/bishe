import torch
import torch.nn as nn
import copy
from FLAlgorithms.users.userbase import User
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator


class UserFedDiff(User):
    
    def __init__(self, user_id, model, train_loader, learning_rate, device='cpu',
                 optimizer_type='adam', momentum=0.9, weight_decay=1e-4,
                 diffusion_steps=50, gen_learning_rate=1e-3):
        super().__init__(user_id, model, train_loader, learning_rate, device,
                        optimizer_type, momentum, weight_decay)
        
        self.generator = DiffusionGenerator(
            num_classes=model.num_classes,
            signal_length=model.signal_length,
            timesteps=diffusion_steps,
        ).to(device)
        
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=gen_learning_rate)
    
    def train(self, epochs):
        self.model.train()
        self.generator.train()
        
        total_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                self.gen_optimizer.zero_grad()
                gen_loss = self._train_generator_step(X, y)
                gen_loss.backward()
                self.gen_optimizer.step()
                
                total_loss += loss.item()
                total_gen_loss += gen_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _train_generator_step(self, x, labels):
        batch_size = x.size(0)
        t = torch.randint(0, self.generator.timesteps, (batch_size,), device=self.device)
        
        noise = torch.randn_like(x)
        alpha_cumprod_t = self.generator.alphas_cumprod[t].view(-1, 1, 1)
        x_noisy = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        noise_pred = self.generator.predict_noise(x_noisy, t, labels)
        loss = nn.MSELoss()(noise_pred, noise)
        return loss
    
    def get_generator_parameters(self):
        return copy.deepcopy(self.generator.state_dict())
    
    def set_generator_parameters(self, params):
        self.generator.load_state_dict(copy.deepcopy(params))
