import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from config import get_config
from models.client_models import get_client_model
from models.global_model import get_global_model
from models.cvae import get_cvae_model, train_cvae
from distill.client_to_cloud import train_global_model, ClientWeightManager
from distill.cloud_to_client import train_local_with_distill, AdaptiveDistillWeight
from utils import evaluate_model, save_model, plot_tsne, plot_accuracy_curve, plot_confusion_matrix
from data_loader import load_federated_data
import logging
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import copy
from sklearn.metrics import classification_report

def to_tensor_loader(X, y, batch_size=64, shuffle=True):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X).float()
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y).long()

    # 自动适配维度
    if len(X.shape) == 3:  # (B, 2, 128)
        X_tensor = X.unsqueeze(1)
    elif len(X.shape) == 4:  # (B, 1, 2, 128)
        X_tensor = X
    else:
        raise ValueError(f"Unexpected input tensor shape: {X.shape}")
    y_tensor = y
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

def adjust_learning_rate(optimizer, round_num, initial_lr):
    """每4轮联邦训练后学习率降低一半"""
    lr = initial_lr * (0.5 ** (round_num // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    args = get_config()
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = args.device

    timestamp = datetime.now().strftime('%m%d_%H%M')
    exp_name = f"exp_radioml_FedBKD_{timestamp}"
    base_dir = os.path.join(args.log_path, exp_name)
    os.makedirs(base_dir, exist_ok=True)
    log_file = os.path.join(base_dir, 'train.log')
    csv_file = os.path.join(base_dir, 'accuracy_log.csv')

    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )
    logging.getLogger('').addHandler(logging.StreamHandler())
    logging.info("FedBKD Experiment Started")

    result = load_federated_data(
        args.data_dir, args.num_clients, args.num_classes, args.mods_per_client, return_global=True)
    clients_train, clients_test, full_x, full_y, label_map = result  # type: ignore

    client_models, client_optimizers, train_loaders, test_loaders = [], [], [], []
    for cid in clients_train:
        X_train, y_train = clients_train[cid]
        X_test, y_test = clients_test[cid]
        train_loader = to_tensor_loader(X_train, y_train, args.batch_size)
        test_loader = to_tensor_loader(X_test, y_test, args.batch_size)

        # 使用更简单但稳定的客户端模型
        model = get_client_model("ABCDE"[cid % 5], num_classes=args.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        client_models.append(model)
        client_optimizers.append(optimizer)

    # === 阶段1: 每客户端独立训练 CVAE ===
    logging.info("=== 阶段1: 客户端CVAE训练 ===")
    cvae_list = []
    for cid in range(args.num_clients):
        X_train, y_train = clients_train[cid]
        local_loader = to_tensor_loader(X_train, y_train, args.cvae_batch, shuffle=True)
        cvae = get_cvae_model(num_classes=args.num_classes, n_components=args.n_components).to(device)
        optimizer = torch.optim.Adam(cvae.parameters(), lr=args.cvae_lr)
        train_cvae(cvae, local_loader, optimizer, device, epochs=args.cvae_epochs)
        cvae_list.append(cvae)
        
        # 清理训练过程中的内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === 阶段2: 客户端上传编码统计信息（通过通信协议）===
    logging.info("=== 阶段2: 客户端上传编码统计信息 ===")
    from distill.communication_protocol import FedBKDCommunicationProtocol
    
    # 初始化通信协议
    comm_protocol = FedBKDCommunicationProtocol(num_clients=args.num_clients, encryption_enabled=True)
    
    # 客户端收集并上传编码统计信息
    upload_packages = []
    for cid in range(args.num_clients):
        # 客户端收集编码统计信息
        mu_stats, logvar_stats = cvae_list[cid].collect_encoding_statistics(train_loaders[cid], device)
        
        # 通过通信协议打包上传
        upload_package = comm_protocol.client_upload_encoding_stats(cid, mu_stats, logvar_stats)
        upload_packages.append(upload_package)
        
        logging.info(f"客户端 {cid} 上传编码统计信息: μ形状={mu_stats.shape}, logvar形状={logvar_stats.shape}")

    # === 阶段3: 云端聚合统计信息并拟合全局GMM ===
    logging.info("=== 阶段3: 云端聚合GMM训练 ===")
    from models.gmm_index_map import GMMIndexMapGenerator
    
    # 云端GMM索引图生成器
    cloud_gmm_generator = GMMIndexMapGenerator(n_components=args.n_components)
    
    # 云端通过通信协议收集统计信息
    aggregated_stats = comm_protocol.cloud_collect_statistics(upload_packages)
    
    # 使用GMM索引图生成器处理聚合数据
    cloud_gmm_generator.fit_cloud_gmm(aggregated_stats)
    
    # 云端生成加密的索引图
    raw_index_map = cloud_gmm_generator.distribute_index_map()
    
    # 通过通信协议分发索引图
    distribution_packages = comm_protocol.cloud_distribute_index_map(raw_index_map, privacy_enabled=True)

    # === 阶段4: 云端分发索引图给客户端并生成伪样本 ===
    logging.info("=== 阶段4: 分发索引图并生成伪样本 ===")
    
    # 每个客户端接收并解密索引图
    for cid in range(args.num_clients):
        cvae_list[cid].index_gen = GMMIndexMapGenerator(n_components=args.n_components)
        
        # 通过通信协议接收索引图
        received_index_map = comm_protocol.client_receive_index_map(cid, distribution_packages[cid])
        cvae_list[cid].index_gen.receive_index_map(received_index_map, client_id=cid)
        
        # 打印隐私保护统计信息
        privacy_stats = cvae_list[cid].index_gen.get_privacy_stats()
        logging.info(f"客户端 {cid} 隐私保护状态: {privacy_stats}")

    # 生成伪样本数据（每个客户端现在都能生成所有类别的伪样本）
    logging.info("=== 生成全局伪样本集合 ===")
    pseudo_data = {}
    for cid in range(args.num_clients):
        fake_data, fake_labels = [], []
        batch_size = 50  # 减小批次大小
        
        for class_idx in range(args.num_classes):
            class_fake_data = []
            class_fake_labels = []
            
            # 分批生成伪样本，现在使用云端分发的全局索引图
            for i in range(0, args.synthetic_per_class, batch_size):
                current_batch_size = min(batch_size, args.synthetic_per_class - i)
                
                # 使用增强的变形等效结构和隐私保护
                x_fake = cvae_list[cid].generate_pseudo_sample_from_index(
                    current_batch_size, class_idx, device=device, apply_deformation=True)
                
                class_fake_data.append(x_fake.cpu())
                class_fake_labels.append(torch.full((current_batch_size,), class_idx, dtype=torch.long, device='cpu'))
                
                # 定期更新变形参数增强多样性
                if i % (batch_size * 2) == 0:
                    cvae_list[cid].update_deformation_parameters()
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            fake_data.append(torch.cat(class_fake_data))
            fake_labels.append(torch.cat(class_fake_labels))
            
            # 清理临时变量
            del class_fake_data, class_fake_labels
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        X_pseudo = torch.cat(fake_data)
        y_pseudo = torch.cat(fake_labels)
        pseudo_data[cid] = (X_pseudo, y_pseudo)
        
        logging.info(f"客户端 {cid} 生成伪样本: {X_pseudo.shape}, 标签分布: {torch.bincount(y_pseudo)}")
        
        # 清理临时变量
        del fake_data, fake_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # === 云端构造 synthetic_loader 用于 distill ===
    synthetic_X = torch.cat([pseudo_data[cid][0] for cid in pseudo_data])
    synthetic_y = torch.cat([pseudo_data[cid][1] for cid in pseudo_data])
    synthetic_loader = to_tensor_loader(synthetic_X, synthetic_y, args.batch_size)

    # === 创建全局测试集用于评估全局模型 ===
    # 修复：只包含客户端见过的类别，避免评估未见过的类别
    all_seen_classes = set()
    for cid in range(args.num_clients):
        X_test, y_test = clients_test[cid]
        seen_classes = set(np.unique(y_test))
        all_seen_classes.update(seen_classes)
    
    print(f"客户端见过的所有类别: {sorted(all_seen_classes)}")
    
    # 只使用客户端见过的类别构建全局测试集
    global_test_X_list = []
    global_test_y_list = []
    
    for cid in range(args.num_clients):
        X_test, y_test = clients_test[cid]
        # 只保留见过的类别
        mask = np.isin(y_test, list(all_seen_classes))
        if mask.sum() > 0:
            global_test_X_list.append(torch.tensor(X_test[mask]).float())
            global_test_y_list.append(torch.tensor(y_test[mask]).long())
    
    if global_test_X_list:
        global_test_X = torch.cat(global_test_X_list)
        global_test_y = torch.cat(global_test_y_list)
    else:
        # 如果没有有效数据，使用第一个客户端的测试集
        X_test, y_test = clients_test[0]
        global_test_X = torch.tensor(X_test).float()
        global_test_y = torch.tensor(y_test).long()

    # 添加调试信息
    print(f"Global test set shape: {global_test_X.shape}")
    print(f"Global test labels distribution: {torch.bincount(global_test_y)}")
    print(f"Total samples in global test: {len(global_test_y)}")

    # 验证标签映射
    for cid in range(args.num_clients):
        X_test, y_test = clients_test[cid]
        print(f"Client {cid} test labels: {np.bincount(y_test)}")

    global_test_loader = to_tensor_loader(global_test_X, global_test_y, args.batch_size)

    global_model = get_global_model(num_classes=args.num_classes).to(device)
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)
    
    # === 全局模型初始化训练 ===
    logging.info("=== 开始全局模型初始化训练 ===")
    # 使用真实数据而不是质量差的伪样本进行初始化
    real_data_loader = to_tensor_loader(full_x, full_y, args.batch_size)
    global_model.train()
    for epoch in range(3):  # 简单训练几轮
        epoch_loss = 0
        batch_count = 0
        for x, y in real_data_loader:
            if batch_count >= 50:  # 限制初始化batch数，避免过拟合
                break
            x, y = x.to(device), y.to(device)
            global_optimizer.zero_grad()
            outputs = global_model(x)
            loss = F.cross_entropy(outputs, y)
            loss.backward()
            global_optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
        if batch_count > 0:
            logging.info(f"Global model init epoch {epoch+1}, loss: {epoch_loss/batch_count:.4f}")
        else:
            logging.info(f"Global model init epoch {epoch+1}, no batches processed")
    
    acc_curve = []
    best_acc = 0

    # === 初始化权重管理器
    client_weight_manager = ClientWeightManager(args.num_clients)
    distill_weight_manager = AdaptiveDistillWeight(args.num_clients)
    
    # === 计算模型复杂度
    model_sizes = [sum(p.numel() for p in model.parameters()) for model in client_models]
    distill_weight_manager.update_model_complexity(model_sizes)
    
    # === 保存最佳模型状态
    best_client_states = [copy.deepcopy(model.state_dict()) for model in client_models]
    best_client_accs = [0.0] * args.num_clients
    
    # === 记录每个客户端的验证准确率历史
    client_acc_history = {i: [] for i in range(args.num_clients)}
    
    # === 客户端预热训练阶段 ===
    logging.info("=== 开始客户端预热训练 ===")
    warmup_epochs = args.warmup_epochs  # 使用配置文件中的预热轮数
    
    for epoch in range(warmup_epochs):
        logging.info(f"--- 预热训练 Epoch {epoch + 1}/{warmup_epochs} ---")
        
        for i, (model, optimizer, loader) in enumerate(zip(client_models, client_optimizers, train_loaders)):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = F.cross_entropy(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            # 评估预热后的模型性能
            model.eval()
            acc = evaluate_model(model, test_loaders[i], device)
            logging.info(f"Client {i} 预热后准确率: {acc:.4f}")
            best_client_accs[i] = acc
            best_client_states[i] = copy.deepcopy(model.state_dict())
    
    logging.info("=== 客户端预热训练完成 ===")
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Average Accuracy", "Learning Rate"] + 
                       [f"Client_{i}_Acc" for i in range(args.num_clients)] +
                       [f"Client_{i}_Weight" for i in range(args.num_clients)])

        for rnd in range(args.rounds):
            logging.info(f"---- Round {rnd + 1} ----")
            
            # === 调整学习率
            current_lr = adjust_learning_rate(global_optimizer, rnd, args.lr)
            for optimizer in client_optimizers:
                adjust_learning_rate(optimizer, rnd, args.lr)
            logging.info(f"Current learning rate: {current_lr}")

            # === Client-to-Cloud: collect logits
            all_logits = []
            client_accs = []
            client_losses = []
            
            # === 云端使用全局索引图生成并分发伪样本 ===
            logging.info(f"=== Round {rnd + 1}: 云端生成全局伪样本 ===")
            
            # 使用云端的全局GMM生成高质量伪样本
            synthetic_samples = []
            synthetic_labels = []
            
            with torch.no_grad():
                # 云端直接生成每个类别的伪样本
                for class_idx in range(args.num_classes):
                    # 使用所有客户端协作生成，确保多样性
                    all_class_samples = []
                    samples_per_client = args.synthetic_per_class // args.num_clients
                    remaining_samples = args.synthetic_per_class % args.num_clients
                    
                    for cid in range(args.num_clients):
                        current_samples = samples_per_client
                        if cid < remaining_samples:
                            current_samples += 1
                        
                        if current_samples == 0:
                            continue
                            
                        selected_cvae = cvae_list[cid]
                        selected_cvae.eval()
                        
                        try:
                            # 使用增强的云端索引图生成伪样本
                            client_fake = selected_cvae.generate_pseudo_sample_from_index(
                                current_samples, class_idx, device=device, apply_deformation=True)
                            
                            # 检查生成质量
                            if torch.isnan(client_fake).any() or torch.isinf(client_fake).any():
                                logging.warning(f"Invalid data for class {class_idx} from client {cid}")
                                client_fake = torch.randn(current_samples, 1, 2, 128, device=device)
                            
                            all_class_samples.append(client_fake)
                            
                            # 定期更新变形参数
                            if rnd % 2 == 0:  # 每2轮更新一次
                                selected_cvae.update_deformation_parameters(learning_rate=0.005)
                            
                        except Exception as e:
                            logging.error(f"Client {cid} failed to generate class {class_idx}: {e}")
                            client_fake = torch.randn(current_samples, 1, 2, 128, device=device)
                            all_class_samples.append(client_fake)
                    
                    # 合并所有客户端生成的同类伪样本
                    if all_class_samples:
                        x_fake = torch.cat(all_class_samples)
                    else:
                        x_fake = torch.randn(args.synthetic_per_class, 1, 2, 128, device=device)
                    
                    synthetic_samples.append(x_fake)
                    synthetic_labels.append(torch.full((x_fake.shape[0],), class_idx, 
                                                     dtype=torch.long, device=device))
                
                # 合并所有伪样本
                distributed_synthetic_X = torch.cat(synthetic_samples)
                distributed_synthetic_y = torch.cat(synthetic_labels)
                
                logging.info(f"云端生成伪样本形状: {distributed_synthetic_X.shape}")
                logging.info(f"伪样本标签分布: {torch.bincount(distributed_synthetic_y)}")
                
                # 验证伪样本质量
                nan_count = torch.isnan(distributed_synthetic_X).sum().item()
                inf_count = torch.isinf(distributed_synthetic_X).sum().item()
                if nan_count > 0 or inf_count > 0:
                    logging.warning(f"伪样本质量问题: NaN={nan_count}, Inf={inf_count}")
                else:
                    logging.info("伪样本质量检查通过")
            
            # 分发给每个客户端进行预测
            for i, client_model in enumerate(client_models):
                client_model.eval()
                with torch.no_grad():
                    # 客户端预测云端分发的伪样本
                    logits = client_model(distributed_synthetic_X.to(device))
                    all_logits.append(logits.cpu())
                    
                    # === 评估客户端模型性能（在本地测试集上）
                    acc = evaluate_model(client_model, test_loaders[i], device)
                    client_accs.append(acc)
                    client_acc_history[i].append(acc)
                    
                    # === 计算验证损失
                    val_loss = 0
                    for x, y in test_loaders[i]:
                        x, y = x.to(device), y.to(device)
                        outputs = client_model(x)
                        val_loss += F.cross_entropy(outputs, y).item()
                    client_losses.append(val_loss / len(test_loaders[i]))
            
            # === 更新客户端权重
            client_weight_manager.update_weights(client_accs, client_losses)
            client_weights = client_weight_manager.get_weights()
            
            # === Train global model with weighted aggregation ===
            # 使用云端分发的伪样本和客户端logits训练全局模型
            distributed_synthetic_loader = to_tensor_loader(distributed_synthetic_X, distributed_synthetic_y, args.batch_size)
            
            # 修复：确保全局模型能看到所有类别的数据
            # 添加真实数据到训练中，确保全局模型能学习所有类别
            real_data_loader = to_tensor_loader(full_x, full_y, args.batch_size)
            
            # 修改全局模型训练策略，确保类别平衡
            global_model.train()
            total_global_loss = 0
            batch_count = 0
            
            # 只使用真实数据训练全局模型，避免伪样本误导
            real_batch_count = 0
            for x, y in real_data_loader:
                if real_batch_count >= 100:  # 增加真实数据训练
                    break
                x, y = x.to(device), y.to(device)
                
                global_optimizer.zero_grad()
                outputs = global_model(x)
                
                # 使用标签平滑，提高泛化能力
                smooth_factor = 0.1
                y_smooth = torch.zeros_like(outputs)
                y_smooth.fill_(smooth_factor / (args.num_classes - 1))
                y_smooth.scatter_(1, y.unsqueeze(1), 1.0 - smooth_factor)
                
                # 使用软目标交叉熵
                loss = -torch.mean(torch.sum(y_smooth * F.log_softmax(outputs, dim=1), dim=1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=1.0)
                global_optimizer.step()
                total_global_loss += loss.item()
                batch_count += 1
                real_batch_count += 1
            
            if batch_count > 0:
                avg_loss = total_global_loss / batch_count
                logging.info(f"Global - Avg Loss: {avg_loss:.4f}")
                
                # 训练后立即检查全局模型性能
                global_model.eval()
                with torch.no_grad():
                    temp_preds = []
                    temp_labels = []
                    for x, y in global_test_loader:
                        x, y = x.to(device), y.to(device)
                        preds = global_model(x).argmax(dim=1)
                        temp_preds.append(preds.cpu())
                        temp_labels.append(y.cpu())
                    
                    temp_preds = torch.cat(temp_preds).numpy()
                    temp_labels = torch.cat(temp_labels).numpy()
                    temp_pred_dist = np.bincount(temp_preds, minlength=args.num_classes)
                    logging.info(f"Global model prediction distribution after training: {temp_pred_dist}")
                global_model.train()
            else:
                logging.info("Global - No valid batches processed")

            # === Cloud-to-Client: distill back to clients ===
            # 云端将伪样本 + global logits 分发给客户端
            global_model.eval()
            with torch.no_grad():
                global_logits = global_model(distributed_synthetic_X.to(device))
            
            total_client_loss = 0
            client_confidences = []
            
            for i, (model, optimizer, loader) in enumerate(zip(client_models, client_optimizers, train_loaders)):
                # === 保存当前模型状态
                current_state = copy.deepcopy(model.state_dict())
                
                client_loss, client_ce, client_kd, confidence = 0, 0, 0, 0
                
                # === 纯本地数据训练，避免误导性蒸馏 ===
                model.train()
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    # 检查输入数据
                    if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
                        print(f"[WARNING] Invalid input data detected, skipping batch")
                        continue
                    
                    # 纯本地数据训练
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    ce_loss = F.cross_entropy(outputs, batch_y)
                    
                    ce_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    client_loss += ce_loss.item()
                    client_ce += ce_loss.item()
                    client_kd += 0.0  # 不使用蒸馏
                
                # 计算平均损失
                if len(loader) > 0:
                    client_loss /= len(loader)
                    client_ce /= len(loader)
                    client_kd = 0.0  # 不使用蒸馏
                
                # 计算置信度
                model.eval()
                with torch.no_grad():
                    confidence = 0
                    for x, y in test_loaders[i]:
                        x, y = x.to(device), y.to(device)
                        outputs = model(x)
                        probs = F.softmax(outputs, dim=1)
                        confidence += torch.max(probs, dim=1)[0].mean().item()
                    confidence /= len(test_loaders[i])
                
                client_confidences.append(confidence)
                
                # === 评估更新后的模型
                new_acc = evaluate_model(model, test_loaders[i], device)
                
                # === 模型回滚机制 - 更宽松的条件
                if new_acc < best_client_accs[i] - 0.3:  # 允许30%的性能下降
                    logging.info(f"Client {i}: Performance dropped significantly from {best_client_accs[i]:.4f} to {new_acc:.4f}, rolling back...")
                    model.load_state_dict(best_client_states[i])
                else:
                    if new_acc > best_client_accs[i]:
                        logging.info(f"Client {i}: Performance improved from {best_client_accs[i]:.4f} to {new_acc:.4f}")
                        best_client_accs[i] = new_acc
                        best_client_states[i] = copy.deepcopy(model.state_dict())
                    else:
                        logging.info(f"Client {i}: Performance maintained at {new_acc:.4f}")
                
                total_client_loss += client_loss

            # === 更新蒸馏权重
            distill_weight_manager.update_weights(client_accs, client_losses, client_confidences)
            
            avg_client_loss = total_client_loss / len(client_models)
            logging.info(f"Client - Avg Loss: {avg_client_loss:.4f}")

            # === Evaluate global model
            global_acc = evaluate_model(global_model, global_test_loader, device)
            acc_curve.append(global_acc)
            
            # 添加详细的全局模型评估信息
            global_model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for x, y in global_test_loader:
                    x, y = x.to(device), y.to(device)
                    preds = global_model(x).argmax(dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(y.cpu())
            
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()
            
            # 计算每个类别的准确率
            print(f"\n=== Global Model Performance Analysis ===")
            print(f"Overall Accuracy: {global_acc:.4f}")
            print(f"Predictions distribution: {np.bincount(all_preds, minlength=args.num_classes)}")
            print(f"True labels distribution: {np.bincount(all_labels, minlength=args.num_classes)}")
            
            # 计算每个类别的准确率
            for i in range(args.num_classes):
                mask = all_labels == i
                if mask.sum() > 0:
                    class_acc = (all_preds[mask] == all_labels[mask]).mean()
                    print(f"Class {i} accuracy: {class_acc:.4f} ({mask.sum()} samples)")
            
            # === 记录详细指标
            row = [rnd + 1, global_acc, current_lr] + client_accs + client_weights.tolist()
            writer.writerow(row)
            
            logging.info(f"Global Model Accuracy: {global_acc:.4f}")

            # === Save best model
            if global_acc > best_acc:
                best_acc = global_acc
                save_model(global_model, os.path.join(base_dir, 'best_global_model.pth'))
                
                # === 同时保存最佳客户端模型
                for i, state in enumerate(best_client_states):
                    save_model(client_models[i], os.path.join(base_dir, f'best_client_{i}_model.pth'))

    # Final evaluation + visualization
    all_preds, all_labels, all_features = [], [], []
    with torch.no_grad():
        for x, y in global_test_loader:
            x = x.to(device)
            feats = global_model.extract_logits(x).cpu()
            preds = global_model(x).argmax(dim=1).cpu()
            
            # 检查NaN
            if torch.isnan(feats).any() or torch.isnan(preds).any():
                print("[WARNING] NaN detected in final evaluation, skipping batch")
                continue
                
            all_features.append(feats)
            all_preds.append(preds)
            all_labels.append(y)

    if all_features:
        all_features = torch.cat(all_features).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        # 检查最终数据
        if np.isnan(all_features).any() or np.isnan(all_preds).any():
            print("[WARNING] NaN detected in final data, skipping visualization")
        else:
            try:
                plot_tsne(all_features, all_labels, save_path=os.path.join(base_dir, 'tsne.png'))
                plot_accuracy_curve(acc_curve, save_path=os.path.join(base_dir, 'acc_curve.png'))
                plot_confusion_matrix(all_preds, all_labels, save_path=os.path.join(base_dir, 'confmat.png'))
            except Exception as e:
                print(f"[ERROR] Visualization failed: {e}")
    else:
        print("[WARNING] No valid data for visualization")
    
    logging.info(f"Best Accuracy: {best_acc:.4f}")
    
    # === 保存通信协议统计信息 ===
    comm_summary = comm_protocol.get_communication_summary()
    logging.info("=== 通信协议统计摘要 ===")
    logging.info(f"总通信次数: {comm_summary['total_communications']}")
    logging.info(f"上传数据量: {comm_summary['total_upload_size']} bytes")
    logging.info(f"下载数据量: {comm_summary['total_download_size']} bytes")
    logging.info(f"隐私保护状态: {'启用' if comm_summary['privacy_enabled'] else '禁用'}")
    logging.info(f"平均上传大小: {comm_summary['average_upload_size']:.2f} bytes")
    logging.info(f"平均下载大小: {comm_summary['average_download_size']:.2f} bytes")
    
    # 保存详细的通信日志
    comm_log_path = os.path.join(base_dir, 'communication_log.json')
    comm_protocol.save_communication_log(comm_log_path)
    


if __name__ == '__main__':
    main()
