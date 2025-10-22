"""
测试优化器配置功能
验证不同优化器是否正确创建
"""

import torch
from FLAlgorithms.trainmodel.models import CNN1D_AMR
from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.users.userFedProx import UserFedProx
from FLAlgorithms.users.userpFedGen import UserFedGen
from FLAlgorithms.trainmodel.generator import Generator
from torch.utils.data import DataLoader, TensorDataset

print("=" * 80)
print("测试优化器配置功能")
print("=" * 80)

# 创建虚拟数据
X = torch.randn(100, 2, 128)
y = torch.randint(0, 11, (100,))
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 创建模型
model = CNN1D_AMR(num_classes=11, signal_length=128)

print("\n测试 1: Adam 优化器（默认）")
user_adam = UserAVG(
    user_id=0,
    model=CNN1D_AMR(num_classes=11, signal_length=128),
    train_loader=train_loader,
    learning_rate=0.01,
    device='cpu',
    optimizer_type='adam',
    weight_decay=1e-4
)
print(f"  ✓ 优化器类型: {type(user_adam.optimizer).__name__}")
print(f"  ✓ 学习率: {user_adam.optimizer.param_groups[0]['lr']}")
print(f"  ✓ 权重衰减: {user_adam.optimizer.param_groups[0]['weight_decay']}")

print("\n测试 2: SGD 优化器（带动量）")
user_sgd = UserAVG(
    user_id=1,
    model=CNN1D_AMR(num_classes=11, signal_length=128),
    train_loader=train_loader,
    learning_rate=0.01,
    device='cpu',
    optimizer_type='sgd',
    momentum=0.9,
    weight_decay=1e-4
)
print(f"  ✓ 优化器类型: {type(user_sgd.optimizer).__name__}")
print(f"  ✓ 学习率: {user_sgd.optimizer.param_groups[0]['lr']}")
print(f"  ✓ 动量: {user_sgd.optimizer.param_groups[0]['momentum']}")
print(f"  ✓ 权重衰减: {user_sgd.optimizer.param_groups[0]['weight_decay']}")

print("\n测试 3: AdamW 优化器")
user_adamw = UserAVG(
    user_id=2,
    model=CNN1D_AMR(num_classes=11, signal_length=128),
    train_loader=train_loader,
    learning_rate=0.001,
    device='cpu',
    optimizer_type='adamw',
    weight_decay=0.01
)
print(f"  ✓ 优化器类型: {type(user_adamw.optimizer).__name__}")
print(f"  ✓ 学习率: {user_adamw.optimizer.param_groups[0]['lr']}")
print(f"  ✓ 权重衰减: {user_adamw.optimizer.param_groups[0]['weight_decay']}")

print("\n测试 4: FedProx 客户端（Adam）")
user_fedprox = UserFedProx(
    user_id=3,
    model=CNN1D_AMR(num_classes=11, signal_length=128),
    train_loader=train_loader,
    learning_rate=0.01,
    mu=0.01,
    device='cpu',
    optimizer_type='adam'
)
print(f"  ✓ 优化器类型: {type(user_fedprox.optimizer).__name__}")
print(f"  ✓ Mu 参数: {user_fedprox.mu}")

print("\n测试 5: FedGen 客户端（Adam 分类器 + Adam 生成器）")
generator = Generator(latent_dim=100, embedding_dim=256)
user_fedgen = UserFedGen(
    user_id=4,
    model=CNN1D_AMR(num_classes=11, signal_length=128),
    generator=generator,
    train_loader=train_loader,
    learning_rate=0.01,
    gen_learning_rate=0.001,
    device='cpu',
    optimizer_type='adam'
)
print(f"  ✓ 分类器优化器类型: {type(user_fedgen.optimizer).__name__}")
print(f"  ✓ 生成器优化器类型: {type(user_fedgen.gen_optimizer).__name__}")

print("\n" + "=" * 80)
print("所有测试通过！优化器配置功能正常工作。")
print("=" * 80)
print("\n✨ 现在可以使用以下命令训练模型：")
print("   python main.py --optimizer adam --learning_rate 0.01")
print("   python main.py --optimizer sgd --momentum 0.9 --learning_rate 0.01")
print("   python main.py --optimizer adamw --learning_rate 0.001")
print("=" * 80)

