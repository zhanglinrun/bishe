import argparse
import torch

# =============================
# 配置参数：使用 argparse 管理实验配置
# =============================

def get_config():
    parser = argparse.ArgumentParser(description='FedBKD for Modulation Classification')

    # 通用设置
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')

    # 数据设置
    parser.add_argument('--data_dir', type=str, default=r'/home/lrzhang/datasets/RML2016.10a/RML2016.10a_dict.pkl', help='data directory')
    parser.add_argument('--num_clients', type=int, default=5, help='number of clients')
    parser.add_argument('--mods_per_client', type=int, default=2, help='modulation classes per client')
    parser.add_argument('--num_classes', type=int, default=10, help='total modulation classes')
    parser.add_argument('--input_shape', type=tuple, default=(2, 128), help='input shape')

    # 蒸馏超参数
    parser.add_argument('--T1', type=float, default=4.0, help='temperature for client-to-cloud distillation')
    parser.add_argument('--T2', type=float, default=2.0, help='temperature for cloud-to-client distillation')
    parser.add_argument('--alpha', type=float, default=0.3, help='weight for kd vs ce in cloud update')
    parser.add_argument('--beta', type=float, default=0.1, help='weight for ce vs kd in client update')

    # 训练参数
    parser.add_argument('--local_epochs', type=int, default=1, help='epochs per local training')
    parser.add_argument('--rounds', type=int, default=20, help='total distillation rounds')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=4, help='lr decay step')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='lr decay factor')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='client warmup epochs')

    # CVAE 参数
    parser.add_argument('--latent_dim', type=int, default=128, help='latent dimension of CVAE')
    parser.add_argument('--cvae_epochs', type=int, default=10, help='epochs to train CVAE')
    parser.add_argument('--cvae_batch', type=int, default=64, help='batch size for CVAE')
    parser.add_argument('--cvae_lr', type=float, default=5e-4, help='learning rate for CVAE')

    parser.add_argument('--n_components', type=int, default=20, help='number of GMM components')

    # 数据生成
    parser.add_argument('--synthetic_per_class', type=int, default=200, help='num of synthetic samples per class')

    # 路径设置
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='model save path')
    parser.add_argument('--log_path', type=str, default='./logs/', help='log save path')
    parser.add_argument('--model_path', type=str, default="./logs/exp_radioml_FedBKD_0619_2253/best_global_model.pth", help='results save path')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_config()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
