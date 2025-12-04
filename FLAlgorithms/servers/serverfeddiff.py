"""FedDiff 服务器：基于扩散模型的知识蒸馏联邦学习（智能集成版）。"""
from __future__ import annotations

from typing import List
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from FLAlgorithms.servers.serveravg import ServerAVG
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator
from utils.model_utils import average_weights

class ServerFedDiff(ServerAVG):
    """
    使用扩散生成器进行数据自由知识迁移的服务器。
    智能集成版 V4：
    1. Smart Ensemble: 针对特定类别，只聚合预测正确的教师模型Logits，剔除噪声。
    2. BN Protection: 严格的Eval模式保护。
    3. Gradient Stability: 优化生成器训练稳定性。
    """

    def __init__(
        self,
        model,
        users,
        num_rounds: int,
        device: str = "cpu",
        pseudo_batch_size: int = 64,
        distill_steps: int = 100, # 建议增加步数，因为现在训练更稳定了
        distill_lr: float = 1e-3, # 恢复正常的学习率
        diffusion_steps: int = 20,
    ):
        super().__init__(model, users, num_rounds, device)
        self.num_classes = model.num_classes
        self.signal_length = model.signal_length
        self.pseudo_batch_size = pseudo_batch_size
        self.distill_steps = distill_steps
        
        self.generator = DiffusionGenerator(
            num_classes=self.num_classes,
            signal_length=self.signal_length,
            timesteps=diffusion_steps,
        ).to(device)

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=distill_lr)
        self.student_optimizer = torch.optim.Adam(self.model.parameters(), lr=distill_lr)
        
        self.kldiv_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def aggregate_parameters(self):
        client_params: List[dict] = []
        client_weights: List[float] = []
        total_samples = 0
        for user in self.users:
            client_params.append(user.get_parameters())
            num_samples = user.get_num_samples()
            client_weights.append(num_samples)
            total_samples += num_samples

        client_weights = [w / total_samples for w in client_weights]
        global_params = average_weights(client_params, client_weights)
        self.model.load_state_dict(global_params)

    def _get_smart_teacher_logits(self, x, labels, teacher_models):
        """
        智能获取教师Logits：
        对于每个样本，只聚合那些【预测类别 == 目标标签】的教师的Logits。
        如果所有教师都预测错，则选择置信度最高的那个。
        """
        batch_size = x.size(0)
        final_logits = torch.zeros(batch_size, self.num_classes, device=self.device)
        
        # 1. 获取所有教师的预测
        all_teacher_logits = [] # Shape: [Num_Teachers, B, Classes]
        with torch.no_grad():
            for m in teacher_models:
                m.eval()
                all_teacher_logits.append(m(x))
        
        all_teacher_logits = torch.stack(all_teacher_logits) # [T, B, C]
        all_teacher_probs = F.softmax(all_teacher_logits, dim=2)
        all_teacher_preds = torch.argmax(all_teacher_probs, dim=2) # [T, B]
        
        # 2. 逐样本筛选
        for b in range(batch_size):
            target = labels[b]
            
            # 找出预测正确的老师索引
            correct_teachers = (all_teacher_preds[:, b] == target).nonzero(as_tuple=True)[0]
            
            if len(correct_teachers) > 0:
                # 如果有老师预测对，只平均这些老师的 logits
                valid_logits = all_teacher_logits[correct_teachers, b, :]
                final_logits[b] = torch.mean(valid_logits, dim=0)
            else:
                # 如果没人预测对，选对目标类别置信度最高的那个老师
                # (即谁认为它是target的可能性最大，就听谁的)
                target_confs = all_teacher_probs[:, b, target]
                best_teacher_idx = torch.argmax(target_confs)
                final_logits[b] = all_teacher_logits[best_teacher_idx, b, :]
                
        return final_logits

    def train(self, test_loader, local_epochs, logger=None):
        # 预热10轮即可
        warmup_rounds = 10
        
        for round_num in range(1, self.num_rounds + 1):
            self.send_parameters()

            local_losses = []
            round_teacher_models = [] 
            
            for user in self.users:
                loss = user.train(local_epochs)
                local_losses.append(loss)
                
                if round_num > warmup_rounds:
                    teacher_model = copy.deepcopy(user.model)
                    teacher_model.eval()
                    for param in teacher_model.parameters():
                        param.requires_grad = False
                    round_teacher_models.append(teacher_model)

            avg_local_loss = sum(local_losses) / len(local_losses)
            self.aggregate_parameters()

            # === 服务器端蒸馏 ===
            distill_loss_g = 0.0 
            distill_loss_s = 0.0 
            valid_updates = 0
            
            if self.distill_steps > 0 and round_num > warmup_rounds:
                self.model.eval() # 保护BN
                self.generator.train()
                
                for step in range(self.distill_steps):
                    try:
                        # 1. 采样标签
                        labels = torch.randint(0, self.num_classes, (self.pseudo_batch_size,), device=self.device)
                        
                        # 2. 生成数据 (保留梯度)
                        gen_data = self.generator.sample(self.pseudo_batch_size, labels, device=self.device)
                        gen_data = torch.clamp(gen_data, -3.0, 3.0) # 截断
                        
                        # 3. 智能集成获取目标 Logits (No Grad)
                        target_logits = self._get_smart_teacher_logits(gen_data, labels, round_teacher_models)
                        target_probs = F.softmax(target_logits, dim=1)
                        
                        # 4. 更新生成器
                        # 目标：让 Generator 生成的数据，能让 Student (作为代理) 预测出 Labels
                        # 这样 Generator 就会努力生成符合 Labels 特征的数据
                        self.gen_optimizer.zero_grad()
                        
                        student_out_for_gen = self.model(gen_data)
                        # L2: Semantic Loss
                        loss_gen = self.ce_loss(student_out_for_gen, labels)
                        
                        loss_gen.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                        self.gen_optimizer.step()
                        distill_loss_g += loss_gen.item()
                        
                        # 5. 更新学生模型 (Student)
                        # 目标：让 Student 逼近智能集成的 Teacher Logits
                        # 只有当生成器生成的还算靠谱时才更新 (Loss < 2.0 ≈ 15% acc)
                        if loss_gen.item() < 2.5:
                            self.student_optimizer.zero_grad()
                            
                            # 重新计算 (Detach Generator)
                            student_out_detached = self.model(gen_data.detach())
                            student_log_probs = F.log_softmax(student_out_detached, dim=1)
                            
                            loss_student = self.kldiv_loss(student_log_probs, target_probs)
                            
                            loss_student.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.student_optimizer.step()
                            
                            distill_loss_s += loss_student.item()
                            valid_updates += 1
                            
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                        else:
                            raise e
                
                avg_d_g = distill_loss_g / self.distill_steps
                avg_d_s = distill_loss_s / max(valid_updates, 1)
                
            else:
                avg_d_g = 0.0
                avg_d_s = 0.0

            # 5. 评估
            accuracy, test_loss = self.evaluate(test_loader)
            self.train_losses.append(test_loss)
            self.train_accuracies.append(accuracy)

            status = "[Warmup]" if round_num <= warmup_rounds else f"[Distill|Upd:{valid_updates}]"
            
            message = (
                f"Round {round_num}/{self.num_rounds} {status} | "
                f"Local Loss: {avg_local_loss:.4f} | "
                f"Test Acc: {accuracy:.2f}% | "
                f"G Loss: {avg_d_g:.4f} | S Loss: {avg_d_s:.4f}"
            )

            if logger:
                logger.info(message)
            else:
                print(message)
                
            del round_teacher_models
            torch.cuda.empty_cache()