import argparse
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from FLAlgorithms.trainmodel.diffusion_generator import DiffusionGenerator
from FLAlgorithms.trainmodel.models import MCLDNN_AMR


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_dir, snr):
    """Load one SNR slice and normalize it with train statistics."""
    train_file = f"{data_dir}/train/{snr}.pkl"
    test_file = f"{data_dir}/test/{snr}.pkl"
    with open(train_file, "rb") as f:
        X_train, y_train, _ = pickle.load(f)
    with open(test_file, "rb") as f:
        X_test, y_test, _ = pickle.load(f)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    # Normalize per channel to stabilize both classifier and diffusion training.
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    stats = {"mean": mean, "std": std}
    return X_train, y_train, X_test, y_test, stats


def build_loaders(X_train, y_train, X_test, y_test, batch_size=256):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def evaluate_classifier(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total, total_loss / total


def cosine_similarity(a, b, eps=1e-6):
    a_norm = a / (a.norm(dim=1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=1)


def loss_ccf(feats, labels, eps=1e-6):
    """Eq.(17) 类共性特征约束，近似为对数正负比。"""
    B = feats.size(0)
    sim_matrix = torch.mm(feats, feats.t())
    norms = feats.norm(dim=1, keepdim=True)
    sim_matrix = sim_matrix / (norms @ norms.t() + eps)

    loss = 0.0
    count = 0
    for i in range(B):
        pos_mask = labels == labels[i]
        pos_mask[i] = False
        neg_mask = ~pos_mask
        pos_sim = sim_matrix[i][pos_mask]
        neg_sim = sim_matrix[i][neg_mask]
        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            continue
        numerator = torch.clamp(pos_sim.mean(), min=eps, max=1 - eps)
        denominator = torch.clamp(neg_sim.mean(), min=eps, max=1 - eps)
        ratio = numerator / (denominator + eps)
        ratio = torch.clamp(ratio, min=eps, max=1 / eps)
        loss = loss - torch.log(ratio)
        count += 1
    if count == 0:
        return feats.new_tensor(0.0)
    return loss / count


def loss_bf(feats, labels, margin=0.2, eps=1e-6):
    """Eq.(18) 平衡特征约束，拉近同类，推远异类。"""
    B = feats.size(0)
    dists = torch.cdist(feats, feats, p=2)
    mask = torch.ones_like(dists, dtype=torch.bool)
    mask.fill_diagonal_(False)

    same = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    pos_term = (dists * same * mask).sum()
    neg_term = (dists * (1 - same) * mask).sum()
    num_pairs = mask.sum()
    if num_pairs == 0:
        return feats.new_tensor(0.0)
    term = (pos_term - neg_term) / (num_pairs + eps)
    return -(term + margin)


def train_classifier(
    model,
    train_loader,
    test_loader,
    epochs=40,
    device="cuda",
    lr=1e-3,
    weight_decay=1e-4,
    patience=5,
    lambda_ccf=0.0,
    lambda_bf=0.0,
    margin=0.2,
):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            feats = model.extract_features(x)
            logits = model.classify_from_features(feats)
            ce_loss = criterion(logits, y)
            ccf_loss = loss_ccf(feats, y) if lambda_ccf > 0 else 0.0
            bf_loss = loss_bf(feats, y, margin=margin) if lambda_bf > 0 else 0.0
            loss = ce_loss + lambda_ccf * ccf_loss + lambda_bf * bf_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        test_acc, test_loss = evaluate_classifier(model, test_loader, device)
        scheduler.step(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Classifier Epoch {epoch+1}/{epochs}, "
            f"Train Loss: {total_loss/total:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Acc: {test_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


def classwise_feature_alignment_loss(x_real, x_pred, labels, classifier):
    """类均值对齐，用于 KM 对齐（无 UKM 数据时的近似）。"""
    with torch.no_grad():
        feats_real = classifier.extract_features(x_real)
    feats_fake = classifier.extract_features(x_pred)
    loss = 0.0
    classes = labels.unique()
    for c in classes:
        mask = labels == c
        if mask.sum() == 0:
            continue
        mean_real = feats_real[mask].mean(dim=0)
        mean_fake = feats_fake[mask].mean(dim=0)
        loss = loss + torch.norm(mean_real - mean_fake, p=2)
    return loss / max(len(classes), 1)


def train_diffusion(
    diffusion,
    train_loader,
    epochs=100,
    device="cuda",
    lr=1e-3,
    classifier=None,
    lambda_dist=0.0,
):
    diffusion.to(device)
    diffusion.train()
    optimizer = optim.Adam(diffusion.parameters(), lr=lr)
    use_amp = torch.cuda.is_available() and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device)
            noise = torch.randn_like(x)

            with torch.amp.autocast("cuda", enabled=use_amp):
                x_noisy = diffusion.q_sample(x, t, noise)
                noise_pred = diffusion.predict_noise(x_noisy, t, y)
                loss = loss_fn(noise_pred, noise)

                if classifier is not None and lambda_dist > 0:
                    was_train = classifier.training
                    classifier.train()  # 允许 RNN 反向传播
                    x0_pred = diffusion.predict_x0(x_noisy, t, noise_pred)
                    dist_loss = classwise_feature_alignment_loss(x, x0_pred, y, classifier)
                    loss = loss + lambda_dist * dist_loss
                    classifier.train(was_train)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Diffusion Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def evaluate_generated(classifier, diffusion, num_classes=11, samples_per_class=100, device="cuda", guidance_scale=0.0):
    classifier.eval()
    diffusion.eval()

    all_preds = []
    all_labels = []

    grad_flag = guidance_scale > 0
    with torch.set_grad_enabled(grad_flag):
        for c in range(num_classes):
            labels = torch.full((samples_per_class,), c, device=device, dtype=torch.long)
            if grad_flag:
                was_train = classifier.training
                classifier.train()  # 允许 RNN 反向传播以获得引导梯度
            generated = diffusion.sample(
                samples_per_class,
                labels=labels,
                device=device,
                classifier=classifier if guidance_scale > 0 else None,
                classifier_scale=guidance_scale,
            )
            if grad_flag:
                classifier.train(was_train)
            preds = classifier(generated).argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()
    print(f"\n生成数据识别准确率: {acc:.4f}")

    print("\n每类准确率:")
    for c in range(num_classes):
        mask = all_labels == c
        c_acc = (all_preds[mask] == all_labels[mask]).float().mean().item()
        print(f"  Class {c}: {c_acc:.4f}")

    return acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier and diffusion generator on a specific SNR slice.")
    parser.add_argument("--data_dir", type=str, default="data_processed/RML2016.10a", help="Root directory of preprocessed data.")
    parser.add_argument("--snr", type=str, default="highsnr", help="SNR split to train/test on, e.g. 6dB / 10dB / highsnr / 100dB.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for both classifier and diffusion.")
    parser.add_argument("--classifier_epochs", type=int, default=50, help="Training epochs for MCLDNN classifier.")
    parser.add_argument("--diffusion_epochs", type=int, default=120, help="Training epochs for the diffusion model.")
    parser.add_argument("--classifier_lr", type=float, default=1e-3, help="Learning rate for the classifier.")
    parser.add_argument("--diffusion_lr", type=float, default=1e-3, help="Learning rate for the diffusion model.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for classifier optimizer.")
    parser.add_argument("--lambda_ccf", type=float, default=0.0, help="Weight for CCF loss (Eq.17).")
    parser.add_argument("--lambda_bf", type=float, default=0.0, help="Weight for BF loss (Eq.18).")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for BF loss.")
    parser.add_argument("--timesteps", type=int, default=1000, help="Diffusion timesteps (paper uses 1000).")
    parser.add_argument("--beta_min", type=float, default=1e-4, help="Minimum beta for linear schedule.")
    parser.add_argument("--beta_max", type=float, default=2e-2, help="Maximum beta for linear schedule.")
    parser.add_argument("--base_channels", type=int, default=128, help="Base channels for the diffusion denoiser.")
    parser.add_argument("--lambda_dist", type=float, default=0.1, help="Feature alignment weight during diffusion training.")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Classifier guidance scale during sampling.")
    parser.add_argument("--samples_per_class", type=int, default=200, help="Number of generated samples per class for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    print(f"Using device: {device}")

    X_train, y_train, X_test, y_test, stats = load_data(args.data_dir, args.snr)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train mean/std per channel: {stats['mean'].flatten()}, {stats['std'].flatten()}")

    train_loader, test_loader = build_loaders(X_train, y_train, X_test, y_test, batch_size=args.batch_size)

    num_classes = 11
    signal_length = 128

    print("\n=== 训练MCLDNN分类器 ===")
    classifier = MCLDNN_AMR(num_classes=num_classes, signal_length=signal_length)
    best_acc = train_classifier(
        classifier,
        train_loader,
        test_loader,
        epochs=args.classifier_epochs,
        device=device,
        lr=args.classifier_lr,
        weight_decay=args.weight_decay,
        lambda_ccf=args.lambda_ccf,
        lambda_bf=args.lambda_bf,
        margin=args.margin,
    )
    print(f"分类器在真实测试集上的最佳准确率: {best_acc:.4f}")

    for param in classifier.parameters():
        param.requires_grad = False
    classifier.eval()
    print("\n分类器已冻结")

    print("\n=== 训练扩散模型 ===")
    diffusion = DiffusionGenerator(
        num_classes=num_classes,
        signal_length=signal_length,
        in_channels=2,
        base_channels=args.base_channels,
        timesteps=args.timesteps,
        beta_start=args.beta_min,
        beta_end=args.beta_max,
    )
    train_diffusion(
        diffusion,
        train_loader,
        epochs=args.diffusion_epochs,
        device=device,
        lr=args.diffusion_lr,
        classifier=classifier,
        lambda_dist=args.lambda_dist,
    )

    print("\n=== 评估生成数据质量 ===")
    evaluate_generated(
        classifier,
        diffusion,
        num_classes=num_classes,
        samples_per_class=args.samples_per_class,
        device=device,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
