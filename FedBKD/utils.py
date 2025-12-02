import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import numpy as np

# =============================
# 工具函数：评估、可视化、保存
# =============================

def evaluate_model(model, dataloader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def plot_tsne(features, labels, save_path=None):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter)
    plt.title("t-SNE of Features")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_accuracy_curve(acc_list, save_path=None):
    plt.figure()
    plt.plot(range(1, len(acc_list)+1), acc_list, marker='o')
    plt.xlabel('Round')
    plt.ylabel('Average Accuracy')
    plt.title('Accuracy Curve')
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(preds, labels, save_path=None, normalize='true'):
    cm = confusion_matrix(labels, preds, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Normalized Confusion Matrix")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model