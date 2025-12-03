import numpy as np
import sys
sys.path.append('.')
from dataset.data_loader import load_preprocessed_data, split_data_non_iid_class, split_data_non_iid_snr

np.random.seed(42)

X_train, y_train, snr_train, X_test, y_test, snr_test, num_classes = load_preprocessed_data(
    'RML2016.10a', '100dB', 'data_processed'
)

client_data = split_data_non_iid_class(X_train, y_train, snr_train, num_clients=5, alpha=0.5)

print("\n" + "=" * 60)
print("各客户端数据分布")
print("=" * 60)

for i, (X_c, y_c, snr_c) in enumerate(client_data):
    unique_labels, label_counts = np.unique(y_c, return_counts=True)
    unique_snrs, snr_counts = np.unique(snr_c, return_counts=True)
    
    print(f"\n客户端 {i}:")
    print(f"  总样本数: {len(y_c)}")
    print(f"  类别分布:")
    for cls, cnt in zip(unique_labels, label_counts):
        pct = cnt / len(y_c) * 100
        print(f"    类别 {cls:2d}: {cnt:5d} ({pct:5.1f}%)")
    print(f"  SNR分布:")
    for snr, cnt in zip(unique_snrs, snr_counts):
        pct = cnt / len(y_c) * 100
        print(f"    {int(snr):3d}dB: {cnt:5d} ({pct:5.1f}%)")

print("\n" + "=" * 60)
print("各类别在客户端间的分布")
print("=" * 60)

for cls in range(num_classes):
    print(f"类别 {cls:2d}:", end="")
    for i, (_, y_c, _) in enumerate(client_data):
        cnt = np.sum(y_c == cls)
        print(f"  客户端{i}:{cnt:5d}", end="")
    print()

print("\n\n" + "=" * 60)
print("SNR Non-IID 划分测试")
print("=" * 60)

client_data_snr = split_data_non_iid_snr(X_train, y_train, snr_train, num_clients=5, alpha=0.5)

for i, (X_c, y_c, snr_c) in enumerate(client_data_snr):
    unique_labels, label_counts = np.unique(y_c, return_counts=True)
    unique_snrs, snr_counts = np.unique(snr_c, return_counts=True)
    
    print(f"\n客户端 {i}:")
    print(f"  总样本数: {len(y_c)}")
    print(f"  类别分布:")
    for cls, cnt in zip(unique_labels, label_counts):
        pct = cnt / len(y_c) * 100
        print(f"    类别 {cls:2d}: {cnt:5d} ({pct:5.1f}%)")
    print(f"  SNR分布:")
    for snr, cnt in zip(unique_snrs, snr_counts):
        pct = cnt / len(y_c) * 100
        print(f"    {int(snr):3d}dB: {cnt:5d} ({pct:5.1f}%)")
