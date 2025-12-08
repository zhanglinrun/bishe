import argparse
import pickle
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 引用更新后的 DiffusionGenerator
from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator
from FLAlgorithms.trainmodel.models import MCLDNN_AMR

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(data_dir, snr):
    # 路径兼容性处理
    train_file = os.path.join(data_dir, 'train', f'{snr}.pkl')
    test_file = os.path.join(data_dir, 'test', f'{snr}.pkl')
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练数据文件未找到: {train_file}")
        
    with open(train_file, "rb") as f:
        X_train, y_train, _ = pickle.load(f)
    with open(test_file, "rb") as f:
        X_test, y_test, _ = pickle.load(f)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    mean = X_train.mean()
    std = X_train.std() + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f"Data Normalized: Mean={mean:.4f}, Std={std:.4f}")
    return X_train, y_train, X_test, y_test

def build_loaders(X_train, y_train, X_test, y_test, batch_size=256):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def evaluate_classifier(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return correct / total, all_preds, all_labels

def train_classifier(model, train_loader, test_loader, epochs=30, device="cuda", lr=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
        val_acc, _, _ = evaluate_classifier(model, test_loader, device)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            print(f"[Classifier] Epoch {epoch+1}: Val Acc {val_acc:.4f}")
            
    model.load_state_dict(best_state)
    return best_acc

def train_diffusion(diffusion, train_loader, epochs=100, device="cuda", lr=1e-4, cfg_prob=0.1, save_path="diffusion_model.pt"):
    """
    训练扩散模型并保存权重
    """
    diffusion.to(device)
    diffusion.train()
    optimizer = optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = diffusion.training_loss(x, y, cfg_prob=cfg_prob)
            loss.backward()
            nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # 保存最佳 Loss 模型（可选，这里简单起见，每10轮或结束时保存）
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 可以在这里保存最佳检查点
            # torch.save(diffusion.state_dict(), save_path.replace('.pt', '_best.pt'))

        if (epoch + 1) % 10 == 0:
            print(f"[Diffusion] Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # 训练结束后保存最终模型
    torch.save(diffusion.state_dict(), save_path)
    print(f"Diffusion model saved to {save_path}")

def evaluate_generated_quality(classifier, diffusion, num_classes, samples_per_class=200, device="cuda", guidance_scale=3.0):
    classifier.eval()
    diffusion.eval()
    
    all_preds = []
    all_labels = []
    all_gen_data = []
    
    print(f"\n开始生成评估样本 (Guidance Scale={guidance_scale})...")
    batch_size = 100
    
    for c in range(num_classes):
        n_generated = 0
        while n_generated < samples_per_class:
            current_batch = min(batch_size, samples_per_class - n_generated)
            labels = torch.full((current_batch,), c, device=device, dtype=torch.long)
            
            with torch.no_grad():
                # 启用 dynamic_thresholding 防止数值爆炸
                gen_data = diffusion.sample(current_batch, labels, device=device, guidance_scale=guidance_scale, dynamic_threshold=True)
                
                logits = classifier(gen_data)
                preds = logits.argmax(1)
                
            all_gen_data.append(gen_data.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            n_generated += current_batch
            
    all_gen_data = torch.cat(all_gen_data)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # 统计信息检查
    print(f"  [Stats] Mean: {all_gen_data.mean():.4f}, Std: {all_gen_data.std():.4f}, Max: {all_gen_data.max():.4f}, Min: {all_gen_data.min():.4f}")
    
    acc = (all_preds == all_labels).float().mean().item()
    print(f"  生成数据分类准确率 (CAS): {acc:.4f}")
    
    return acc, all_preds.numpy(), all_labels.numpy()

def plot_confusion_matrices(real_preds, real_labels, gen_preds, gen_labels, class_names=None, save_path="confusion_comparison.png"):
    """绘制真实数据与生成数据的混淆矩阵对比图"""
    if class_names is None:
        class_names = [str(i) for i in range(11)]
        
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Real Data CM
    cm_real = confusion_matrix(real_labels, real_preds, normalize='true')
    sns.heatmap(cm_real, annot=True, fmt='.2f', cmap='Blues', ax=axes[0], 
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Real Data Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    # Generated Data CM
    cm_gen = confusion_matrix(gen_labels, gen_preds, normalize='true')
    sns.heatmap(cm_gen, annot=True, fmt='.2f', cmap='Blues', ax=axes[1], 
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("Generated Data Confusion Matrix (Diffusion)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n混淆矩阵对比图已保存至: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_processed/RML2016.10a")
    parser.add_argument("--snr", default="highsnr")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--cfg_prob", type=float, default=0.1, help="训练时丢弃条件的概率")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", default="pretrained_diffusion.pt", help="扩散模型权重保存路径")
    args = parser.parse_args()

    set_seed(42)
    
    # 类别名称
    class_names = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    
    print("=== 加载数据 ===")
    X_train, y_train, X_test, y_test = load_data(args.data_dir, args.snr)
    train_loader, test_loader = build_loaders(X_train, y_train, X_test, y_test)
    
    num_classes = 11
    signal_len = 128
    
    print("\n=== 1. 训练评估器 (Oracle Classifier) ===")
    classifier = MCLDNN_AMR(num_classes=num_classes, signal_length=signal_len)
    real_acc = train_classifier(classifier, train_loader, test_loader, epochs=50, device=args.device)
    print(f"分类器在真实测试集上的准确率: {real_acc:.4f}")
    
    # 获取真实数据的预测结果用于绘图
    _, real_preds, real_labels = evaluate_classifier(classifier, test_loader, args.device)
    
    print("\n=== 2. 训练改进版扩散模型 (U-Net + CFG + AdaGN) ===")
    diffusion = DiffusionGenerator(
        num_classes=num_classes,
        signal_length=signal_len,
        base_channels=64,
        channel_mults=(1, 2, 4),
        beta_schedule="cosine"
    )
    
    train_diffusion(diffusion, train_loader, epochs=args.epochs, device=args.device, 
                   lr=args.lr, cfg_prob=args.cfg_prob, save_path=args.save_path)
    
    print("\n=== 3. 评估生成质量与绘图 ===")
    # 选取效果最好的 Scale 进行绘图，通常 2.0 - 4.0 之间
    scales = [1.0, 3.0]
    best_gen_acc = 0
    best_gen_preds = []
    best_gen_labels = []
    
    for s in scales:
        acc, preds, labels = evaluate_generated_quality(classifier, diffusion, num_classes, 
                                                      samples_per_class=200, device=args.device, guidance_scale=s)
        if acc > best_gen_acc:
            best_gen_acc = acc
            best_gen_preds = preds
            best_gen_labels = labels
            
    # 绘制混淆矩阵
    if len(best_gen_preds) > 0:
        plot_confusion_matrices(real_preds, real_labels, best_gen_preds, best_gen_labels, 
                              class_names=class_names, save_path="confusion_matrix_comparison.png")

if __name__ == "__main__":
    main()