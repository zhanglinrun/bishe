# 联邦学习自动调制识别系统 (Federated Learning for Automatic Modulation Recognition)

## 1. 项目简介 (Introduction)

### 项目目标

本项目是一个基于**联邦学习**的**无线信号自动调制识别**系统。主要解决以下问题：

- **自动调制识别 (AMR)**：识别无线信号的调制类型（如 BPSK、QPSK、QAM 等），这在频谱监测、认知无线电等领域非常重要。
- **隐私保护**：通过联邦学习框架，多个客户端可以在不共享原始数据的情况下协同训练模型，保护数据隐私。
- **Non-IID 数据挑战**：模拟真实场景中不同客户端数据分布不均衡的情况（如不同 SNR 环境、不同调制类型分布）。

### 核心技术

- **深度学习模型**：
  - **CNN1D**：一维卷积神经网络，适合处理 I/Q 信号序列
  - **ResNet1D**：一维残差网络，更深的网络结构，提升特征提取能力

- **联邦学习算法**：
  - **FedAvg**：联邦平均算法，最经典的联邦学习方法
  - **FedProx**：联邦近端算法，通过添加近端项缓解数据异构问题
  - **FedGen**：基于生成器的联邦学习，使用数据自由知识蒸馏

### 框架

- **深度学习框架**：PyTorch 1.10+
- **支持的数据集**：
  - RML2016.10a (11 类调制，128 采样点)
  - RML2016.10b (11 类调制，128 采样点)
  - RML2018a (24 类调制，1024 采样点)
  - HisarMod (26 类调制，1024 采样点)

---

## 2. 项目框架 (Project Structure)

### 目录树

```
.
├── dataset/                          # 数据集目录
│   ├── RML2016.10a/                  # RML2016.10a 原始数据
│   ├── RML2016.10b/                  # RML2016.10b 原始数据
│   ├── RML2018a/                     # RML2018a 原始数据
│   ├── HisarMod/                     # HisarMod 原始数据
│   ├── data_loader.py                # 数据加载和划分脚本
│   └── datasplit.py                  # 数据预处理脚本
│
├── FLAlgorithms/                     # 联邦学习算法实现
│   ├── trainmodel/                   # 模型定义
│   │   ├── models.py                 # CNN1D 和 ResNet1D 模型
│   │   └── generator.py              # FedGen 生成器模型
│   ├── servers/                      # 服务器端实现
│   │   ├── serverbase.py             # 服务器基类
│   │   ├── serveravg.py              # FedAvg 服务器
│   │   ├── serverFedProx.py          # FedProx 服务器
│   │   └── serverpFedGen.py          # FedGen 服务器
│   └── users/                        # 客户端实现
│       ├── userbase.py               # 客户端基类
│       ├── useravg.py                # FedAvg 客户端
│       ├── userFedProx.py            # FedProx 客户端
│       └── userpFedGen.py            # FedGen 客户端
│
├── utils/                            # 工具函数
│   ├── model_config.py               # 数据集配置信息
│   ├── model_utils.py                # 模型工具（保存、加载、参数聚合等）
│   └── plot_utils.py                 # 可视化工具（绘制训练曲线等）
│
├── results/                          # 训练结果保存目录
│   └── plots/                        # 可视化图表
│
├── main.py                           # 主训练脚本
├── main_plot.py                      # 结果可视化脚本
├── test_snr.py                       # SNR 性能评估脚本
│
├── requirements.txt                  # Python 依赖列表
└── README.md                         # 项目说明文档
```

---

### 文件功能详解

#### **数据处理模块 (dataset/)**

##### 1. `dataset/datasplit.py` - 数据预处理脚本

**主要作用：**
- 将原始数据集按 SNR（信噪比）划分为训练集和测试集
- 支持处理 4 种数据集格式（RML2016.10a/10b, RML2018a, HisarMod）
- 生成多种 SNR 组合（单独 SNR、所有 SNR、高 SNR）

**关键函数：**
- `load_RML2016_10a()`: 加载 RML2016.10a 数据集，返回 (X, y, snr, mods)
- `load_RML2016_10b()`: 加载 RML2016.10b 数据集
- `load_RML2018a()`: 加载 RML2018a 数据集（HDF5 格式）
- `load_HisarMod()`: 加载 HisarMod 数据集
- `split_by_snr()`: 按 SNR 值分层抽样，划分训练/测试集
- `save_split_data()`: 保存单独 SNR 的数据到 `data_processed/` 目录
- `generate_combined_datasets()`: 生成合并数据集（100dB = 所有 SNR，highsnr = SNR > 0）

**使用方法：**
```bash
# 处理所有数据集
python dataset/datasplit.py --dataset all

# 处理单个数据集
python dataset/datasplit.py --dataset RML2016.10a --input_dir dataset --output_dir data_processed
```

**输出结构：**
```
data_processed/
├── RML2016.10a/
│   ├── train/
│   │   ├── -20dB.pkl    # 单独 SNR 训练集
│   │   ├── -18dB.pkl
│   │   ├── ...
│   │   ├── 100dB.pkl    # 所有 SNR 合并
│   │   └── highsnr.pkl  # SNR > 0 合并
│   └── test/
│       └── (同上)
```

---

##### 2. `dataset/data_loader.py` - 数据加载器

**主要作用：**
- 加载预处理后的数据
- 实现 IID 和 Non-IID 数据划分
- 创建 PyTorch DataLoader 供训练使用

**关键类：**
- `SignalDataset`: PyTorch Dataset 类，封装信号数据 (X, y)

**关键函数：**
- `load_preprocessed_data()`: 从 `data_processed/` 加载预处理数据
- `split_data_iid()`: IID 划分，随机打乱后均匀分配给各客户端
- `split_data_non_iid_class()`: 按调制类型 Non-IID 划分，使用 Dirichlet 分布（α 越小越不均衡）
- `split_data_non_iid_snr()`: 按 SNR 范围 Non-IID 划分，不同客户端获得不同 SNR 区间的数据
- `get_dataloaders()`: 核心函数，返回客户端训练 DataLoaders 和全局测试 DataLoader

**与其他文件交互：**
- `main.py` 调用 `get_dataloaders()` 获取训练/测试数据
- 使用 `utils/model_config.py` 获取数据集配置（类别数、信号长度）

---

#### **模型定义模块 (FLAlgorithms/trainmodel/)**

##### 3. `FLAlgorithms/trainmodel/models.py` - 分类模型

**主要作用：**
- 定义自动调制识别的深度学习模型

**包含的类：**

1. **`CNN1D_AMR`**: 一维 CNN 模型
   - 输入：`[B, 2, L]`（B=批大小，2=I/Q 两通道，L=信号长度）
   - 结构：3 个卷积块（Conv1D + BatchNorm + ReLU + MaxPool）+ 2 个全连接层
   - 输出：`[B, num_classes]` 分类 logits

2. **`ResNet1D_AMR`**: 一维 ResNet 模型
   - 输入：`[B, 2, L]`
   - 结构：初始卷积层 + 3 个残差层 + 全局平均池化 + 全连接层
   - 残差块：`ResidualBlock1D`，实现快捷连接

3. **`get_model()`**: 工厂函数
   - 根据模型名称（'CNN1D' 或 'ResNet1D'）返回实例

**与其他文件交互：**
- `main.py` 通过 `get_model()` 创建全局模型和客户端模型

---

##### 4. `FLAlgorithms/trainmodel/generator.py` - 生成器模型

**主要作用：**
- 为 FedGen 算法提供数据自由知识蒸馏的生成器

**包含的类：**

1. **`Generator`**: 无条件生成器
   - 输入：潜在向量 `z` (形状: `[B, latent_dim]`)
   - 输出：伪嵌入 (形状: `[B, embedding_dim]`)，用于模拟分类器的倒数第二层特征
   - 结构：3 层全连接网络 + BatchNorm + ReLU/Tanh

2. **`ConditionalGenerator`**: 条件生成器（可选）
   - 额外输入类别标签，生成特定类别的伪数据

**与其他文件交互：**
- `main.py` 为 FedGen 算法创建生成器实例
- `FLAlgorithms/users/userpFedGen.py` 和 `FLAlgorithms/servers/serverpFedGen.py` 使用生成器进行知识蒸馏

---

#### **服务器端模块 (FLAlgorithms/servers/)**

##### 5. `FLAlgorithms/servers/serverbase.py` - 服务器基类

**主要作用：**
- 定义联邦学习服务器的通用接口和功能

**关键方法：**
- `send_parameters()`: 将全局模型参数分发给所有客户端
- `aggregate_parameters()`: 聚合客户端模型（抽象方法，由子类实现）
- `evaluate()`: 在测试集上评估全局模型，返回准确率和损失
- `train()`: 训练主循环（抽象方法，由子类实现）

---

##### 6. `FLAlgorithms/servers/serveravg.py` - FedAvg 服务器

**主要作用：**
- 实现 FedAvg（联邦平均）算法的服务器端

**核心功能：**
- `aggregate_parameters()`: 
  - 收集所有客户端的模型参数
  - 按样本数加权平均：`θ_global = Σ(n_i / N) * θ_i`
  - 更新全局模型

- `train()`:
  1. 发送全局模型给客户端
  2. 客户端本地训练
  3. 聚合客户端模型
  4. 在测试集上评估
  5. 记录训练历史

**与其他文件交互：**
- `main.py` 创建 ServerAVG 实例并调用 `train()` 方法
- 使用 `utils/model_utils.py` 的 `average_weights()` 进行参数聚合

---

##### 7. `FLAlgorithms/servers/serverFedProx.py` - FedProx 服务器

**主要作用：**
- 实现 FedProx 算法的服务器端

**核心区别：**
- 聚合方式与 FedAvg 相同
- 客户端训练时添加近端项（由客户端实现）

---

##### 8. `FLAlgorithms/servers/serverpFedGen.py` - FedGen 服务器

**主要作用：**
- 实现 FedGen 算法的服务器端
- 维护全局生成器

**核心功能：**
- 聚合客户端的分类器和生成器
- 使用生成器生成伪数据进行知识蒸馏

---

#### **客户端模块 (FLAlgorithms/users/)**

##### 9. `FLAlgorithms/users/userbase.py` - 客户端基类

**主要作用：**
- 定义联邦学习客户端的通用接口

**关键属性：**
- `model`: 本地模型
- `train_loader`: 本地训练数据
- `optimizer`: 优化器（支持 SGD、Adam、AdamW）
- `criterion`: 损失函数（交叉熵）

**关键方法：**
- `set_parameters()`: 接收服务器的全局模型参数
- `get_parameters()`: 返回本地模型参数（用于上传到服务器）
- `train()`: 本地训练（抽象方法，由子类实现）
- `test()`: 在测试集上评估本地模型

**优化器支持：**
- 构造函数接受 `optimizer_type` 参数（'sgd', 'adam', 'adamw'）
- SGD 支持动量（momentum）和权重衰减（weight_decay）
- Adam/AdamW 支持权重衰减

---

##### 10. `FLAlgorithms/users/useravg.py` - FedAvg 客户端

**主要作用：**
- 实现 FedAvg 算法的客户端

**核心功能：**
- `train(epochs)`: 
  - 在本地数据上训练 `epochs` 轮
  - 使用标准的交叉熵损失
  - 返回平均训练损失

---

##### 11. `FLAlgorithms/users/userFedProx.py` - FedProx 客户端

**主要作用：**
- 实现 FedProx 算法的客户端

**核心区别：**
- 损失函数添加近端项：`L = L_CE + (μ/2) * ||θ - θ_global||²`
- `μ` 控制正则化强度，缓解数据异构问题

---

##### 12. `FLAlgorithms/users/userpFedGen.py` - FedGen 客户端

**主要作用：**
- 实现 FedGen 算法的客户端
- 同时训练分类器和生成器

**核心功能：**
- 训练分类器：使用真实数据 + 生成的伪数据
- 训练生成器：使用知识蒸馏损失

---

#### **工具模块 (utils/)**

##### 13. `utils/model_config.py` - 数据集配置

**主要作用：**
- 集中管理所有数据集的配置信息

**配置字典 `DATASET_CONFIG`：**
```python
{
    'RML2016.10a': {
        'num_classes': 11,         # 调制类型数
        'signal_length': 128,      # 信号长度
        'classes': [...]           # 类别名称列表
    },
    ...
}
```

**关键函数：**
- `get_dataset_config(dataset_name)`: 返回指定数据集的配置字典

**与其他文件交互：**
- `data_loader.py`, `main.py` 等调用此函数获取数据集参数

---

##### 14. `utils/model_utils.py` - 模型工具

**主要作用：**
- 提供模型操作、日志记录等通用工具函数

**关键函数：**

1. **模型参数操作：**
   - `get_model_params(model)`: 提取模型参数（深拷贝）
   - `set_model_params(model, params)`: 设置模型参数
   - `average_weights(weights_list, weights)`: 对多个模型参数进行加权平均

2. **日志记录：**
   - `setup_logger(log_file)`: 创建 logger，同时输出到文件和控制台
   - `save_logs(log_file, round_num, accuracy, loss)`: 保存训练指标到 CSV 文件

3. **模型保存/加载：**
   - `save_model(model, save_path)`: 保存模型权重
   - `load_model(model, load_path, device)`: 加载模型权重

**与其他文件交互：**
- `serveravg.py` 使用 `average_weights()` 聚合参数
- `main.py` 使用 `setup_logger()` 和 `save_model()`

---

##### 15. `utils/plot_utils.py` - 可视化工具

**主要作用：**
- 提供训练结果可视化功能

**关键函数：**

1. **训练曲线：**
   - `plot_accuracy_curves(logs_dict, title, save_path)`: 绘制准确率对比曲线
   - `plot_loss_curves(logs_dict, title, save_path)`: 绘制损失对比曲线
   - `plot_combined_curves(logs_dict_acc, logs_dict_loss, ...)`: 绘制组合曲线（1 行 2 列）

2. **其他可视化：**
   - `plot_confusion_matrix(y_true, y_pred, classes, ...)`: 绘制混淆矩阵
   - `plot_data_distribution(client_distributions, ...)`: 绘制客户端数据分布

**与其他文件交互：**
- `main_plot.py` 调用这些函数生成对比图表

---

#### **主脚本**

##### 16. `main.py` - 主训练脚本

**主要作用：**
- 联邦学习训练的主入口
- 解析命令行参数、创建模型、执行训练、保存结果

**工作流程：**

1. **解析参数** (`parse_args()`):
   - 数据集参数：`--dataset`, `--data_snr`, `--data_dir`
   - 算法参数：`--algorithm` (FedAvg/FedProx/FedGen)
   - 模型参数：`--model` (CNN1D/ResNet1D)
   - 联邦学习参数：`--num_clients`, `--num_rounds`, `--local_epochs`
   - 训练参数：`--batch_size`, `--learning_rate`
   - 优化器参数：`--optimizer`, `--momentum`, `--weight_decay`
   - Non-IID 参数：`--non_iid_type`, `--alpha`

2. **设置随机种子** (`set_seed()`):
   - 确保实验可复现

3. **加载数据** (`get_dataloaders()`):
   - 从 `data_processed/` 加载预处理数据
   - 按指定方式划分客户端数据
   - 返回训练/测试 DataLoaders

4. **创建模型**:
   - 创建全局模型 (`get_model()`)

5. **创建客户端** (`create_users()`):
   - 根据算法类型创建对应的客户端实例（UserAVG/UserFedProx/UserFedGen）
   - 每个客户端有独立的模型副本和本地数据

6. **创建服务器** (`create_server()`):
   - 根据算法类型创建对应的服务器实例（ServerAVG/ServerFedProx/ServerFedGen）

7. **训练** (`server.train()`):
   - 执行联邦学习训练循环
   - 每轮：发送参数 → 客户端训练 → 聚合参数 → 评估 → 记录

8. **保存结果**:
   - 训练指标 CSV：`{dataset}_{algorithm}_metrics.csv`
   - 训练日志 TXT：`{dataset}_{algorithm}_log.txt`
   - 模型权重 PT：`{dataset}_{algorithm}_model.pt`
   - 结果文件夹名称包含准确率和参数信息

**命令行示例：**
```bash
python main.py \
    --dataset RML2016.10a \
    --data_snr 100dB \
    --algorithm FedAvg \
    --model CNN1D \
    --num_clients 10 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --optimizer adam \
    --non_iid_type iid
```

---

##### 17. `main_plot.py` - 结果可视化脚本

**主要作用：**
- 读取训练日志并绘制对比曲线
- 比较不同算法的性能

**工作流程：**

1. **解析参数**:
   - `--dataset`: 数据集名称
   - `--metrics_dir`: 指标文件所在目录
   - `--algorithms`: 要对比的算法列表
   - `--output_dir`: 图表保存目录

2. **加载指标** (`load_metrics()`):
   - 从 CSV 文件读取训练历史（Round, Accuracy, Loss）

3. **绘图**:
   - 准确率对比曲线：`{dataset}_accuracy_comparison.png`
   - 损失对比曲线：`{dataset}_loss_comparison.png`
   - 组合曲线：`{dataset}_combined_comparison.png`

**命令行示例：**
```bash
python main_plot.py \
    --dataset RML2016.10a \
    --metrics_dir ./results \
    --algorithms FedAvg FedProx FedGen \
    --output_dir ./results/plots
```

---

##### 18. `test_snr.py` - SNR 性能评估脚本

**主要作用：**
- 评估训练好的模型在不同 SNR 下的性能
- 绘制准确率 vs. SNR 曲线

**工作流程：**

1. **加载模型**:
   - 从 `.pt` 文件加载训练好的模型权重
   - 自动推断模型架构（从路径名称）

2. **加载测试数据** (`load_snr_test_data()`):
   - 从 `data_processed/{dataset}/test/` 加载所有单独 SNR 的测试集

3. **评估** (`evaluate_model()`):
   - 对每个 SNR 计算准确率

4. **保存结果**:
   - CSV 文件：`{base_name}_snr_metrics.csv`
   - 日志文件：`{base_name}_snr_log.txt`
   - 图像：`{base_name}_snr.png`（准确率 vs. SNR 曲线）

**命令行示例：**
```bash
python test_snr.py \
    --model_path ./results/RML2016.10a_FedAvg_model.pt \
    --dataset_name RML2016.10a \
    --model CNN1D \
    --data_dir data_processed
```

---

## 3. 环境配置 (Environment Setup)

### 系统要求

- **操作系统**：Windows / Linux / macOS
- **Python 版本**：Python 3.7 或更高版本
- **硬件要求**：
  - GPU（推荐）：支持 CUDA 的 NVIDIA GPU（显存 ≥ 4GB）
  - CPU：Intel i5 或更高
  - 内存：≥ 8GB

### 依赖库列表

本项目的所有依赖已列在 `requirements.txt` 文件中：

```txt
torch>=1.10.0              # PyTorch 深度学习框架
torchvision>=0.11.0        # PyTorch 视觉库
numpy>=1.21.0              # 数值计算库
h5py>=3.6.0                # HDF5 文件读取
matplotlib>=3.5.0          # 绘图库
scikit-learn>=1.0.0        # 机器学习工具（混淆矩阵、数据划分等）
tqdm>=4.62.0               # 进度条显示
pandas>=1.3.0              # 数据处理（CSV 读写）
```

### 安装步骤

#### 步骤 1：克隆或下载项目

```bash
# 如果使用 Git
git clone <项目地址>
cd bishe

# 或直接下载压缩包并解压
```

#### 步骤 2：创建虚拟环境（推荐使用 conda）

```bash
# 创建名为 fl-amr 的虚拟环境
conda create -n fl-amr python=3.9

# 激活虚拟环境
conda activate fl-amr
```

**或使用 venv（Python 自带）：**

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

#### 步骤 3：安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt
```

**如果需要 GPU 加速（推荐）：**

请根据您的 CUDA 版本安装对应的 PyTorch：

```bash
# CUDA 11.3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113

# CUDA 11.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu116

# CUDA 11.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

# CPU 版本（无 GPU）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**检查 CUDA 版本：**

```bash
# Windows/Linux
nvidia-smi

# 查看 PyTorch 是否识别 GPU
python -c "import torch; print(torch.cuda.is_available())"
```

#### 步骤 4：验证安装

```bash
python -c "import torch; import numpy; import h5py; import matplotlib; print('所有依赖安装成功！')"
```

---

## 4. 使用说明 (Usage)

### 4.1 数据准备 (Data Preparation)

#### 数据集下载

您需要先下载以下数据集（任选其一或多个）：

1. **RML2016.10a**
   - 下载链接：[https://www.deepsig.ai/datasets](https://www.deepsig.ai/datasets)
   - 文件名：`RML2016.10a_dict.pkl`
   - 放置位置：`dataset/RML2016.10a/RML2016.10a_dict.pkl`

2. **RML2016.10b**
   - 下载链接：[https://www.deepsig.ai/datasets](https://www.deepsig.ai/datasets)
   - 文件名：`RML2016.10b.dat`
   - 放置位置：`dataset/RML2016.10b/RML2016.10b.dat`

3. **RML2018a**
   - 下载链接：[https://www.deepsig.ai/datasets](https://www.deepsig.ai/datasets)
   - 文件名：`GOLD_XYZ_OSC.0001_1024.hdf5`
   - 放置位置：`dataset/RML2018a/GOLD_XYZ_OSC.0001_1024.hdf5`

4. **HisarMod**
   - 下载链接：[https://ieee-dataport.org/open-access/hisarmod](https://ieee-dataport.org/open-access/hisarmod)
   - 文件名：`HisarMod2019train.h5`, `HisarMod2019test.h5`
   - 放置位置：`dataset/HisarMod/`

#### 数据集目录结构

下载完成后，您的 `dataset/` 目录应该如下所示：

```
dataset/
├── RML2016.10a/
│   └── RML2016.10a_dict.pkl
├── RML2016.10b/
│   └── RML2016.10b.dat
├── RML2018a/
│   └── GOLD_XYZ_OSC.0001_1024.hdf5
└── HisarMod/
    ├── HisarMod2019train.h5
    └── HisarMod2019test.h5
```

#### 数据预处理

在训练之前，必须先运行数据预处理脚本，将原始数据按 SNR 分割：

**处理所有数据集：**

```bash
python dataset/datasplit.py --dataset all --input_dir dataset --output_dir data_processed
```

**处理单个数据集（以 RML2016.10a 为例）：**

```bash
python dataset/datasplit.py --dataset RML2016.10a --input_dir dataset --output_dir data_processed
```

**预处理脚本会：**
- 加载原始数据集
- 按 SNR 分层抽样，划分训练集（80%）和测试集（20%）
- 生成单独 SNR 的数据文件（如 `-20dB.pkl`, `0dB.pkl`, `18dB.pkl` 等）
- 生成合并数据集：
  - `100dB.pkl`：所有 SNR 合并
  - `highsnr.pkl`：SNR > 0 的数据合并

**预处理后的目录结构：**

```
data_processed/
├── RML2016.10a/
│   ├── train/
│   │   ├── -20dB.pkl
│   │   ├── -18dB.pkl
│   │   ├── ...
│   │   ├── 18dB.pkl
│   │   ├── 100dB.pkl      # 所有 SNR 合并
│   │   └── highsnr.pkl    # SNR > 0 合并
│   └── test/
│       └── (同上结构)
└── (其他数据集...)
```

---

### 4.2 训练流程 (Training Process)

#### 基本训练命令

训练使用 `main.py` 脚本，以下是一个完整的训练示例：

```bash
python main.py \
    --dataset RML2016.10a \
    --data_snr 100dB \
    --algorithm FedAvg \
    --model CNN1D \
    --num_clients 10 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --optimizer adam \
    --weight_decay 1e-4 \
    --non_iid_type iid \
    --output_dir ./results \
    --device cuda
```

#### 参数详解

**数据集参数：**

- `--dataset`：数据集名称
  - 可选值：`RML2016.10a`, `RML2016.10b`, `RML2018a`, `HisarMod`
  - 说明：选择要使用的数据集

- `--data_snr`：SNR 标识
  - 可选值：`100dB`（所有 SNR）, `highsnr`（SNR > 0）, 或单独 SNR（如 `10dB`, `0dB`）
  - 说明：选择训练数据的 SNR 范围

- `--data_dir`：预处理数据目录
  - 默认值：`data_processed`
  - 说明：存放预处理数据的目录

**算法参数：**

- `--algorithm`：联邦学习算法
  - 可选值：`FedAvg`（联邦平均）, `FedProx`（联邦近端）, `FedGen`（生成器联邦学习）
  - 说明：选择要使用的联邦学习算法

**模型参数：**

- `--model`：模型架构
  - 可选值：`CNN1D`, `ResNet1D`
  - 说明：选择深度学习模型

**联邦学习参数：**

- `--num_clients`：客户端数量
  - 默认值：`10`
  - 说明：模拟的客户端（设备）数量

- `--num_rounds`：训练轮次
  - 默认值：`100`
  - 说明：联邦学习的全局迭代轮数

- `--local_epochs`：本地训练轮数
  - 默认值：`5`
  - 说明：每个客户端在本地数据上训练的 epoch 数

**训练参数：**

- `--batch_size`：批大小
  - 默认值：`128`
  - 说明：训练时的 mini-batch 大小

- `--learning_rate`：学习率
  - 默认值：`0.01`
  - 说明：优化器的学习率

**优化器参数：**

- `--optimizer`：优化器类型
  - 可选值：`sgd`, `adam`, `adamw`
  - 默认值：`adam`
  - 说明：选择优化器

- `--momentum`：动量参数
  - 默认值：`0.9`
  - 说明：仅用于 SGD 优化器

- `--weight_decay`：权重衰减
  - 默认值：`1e-4`
  - 说明：L2 正则化系数

**Non-IID 参数：**

- `--non_iid_type`：数据划分类型
  - 可选值：
    - `iid`：独立同分布，数据随机均匀分配
    - `class`：按调制类型 Non-IID，使用 Dirichlet 分布
    - `snr`：按 SNR 范围 Non-IID
  - 说明：模拟真实场景中的数据异构

- `--alpha`：Dirichlet 参数
  - 默认值：`0.5`
  - 说明：仅用于 `--non_iid_type class`，α 越小数据越不均衡

**FedProx 参数：**

- `--mu`：近端项系数
  - 默认值：`0.01`
  - 说明：仅用于 FedProx 算法，控制正则化强度

**FedGen 参数：**

- `--gen_learning_rate`：生成器学习率
  - 默认值：`0.001`
  - 说明：仅用于 FedGen 算法

- `--latent_dim`：潜在向量维度
  - 默认值：`100`
  - 说明：仅用于 FedGen 算法

**输出参数：**

- `--output_dir`：输出目录
  - 默认值：`./results`
  - 说明：保存训练结果的目录

- `--device`：设备
  - 可选值：`cuda`, `cpu`
  - 默认值：自动检测（有 GPU 则用 cuda）
  - 说明：训练设备

**其他参数：**

- `--seed`：随机种子
  - 默认值：`42`
  - 说明：确保实验可复现

---

#### 训练示例

**示例 1：FedAvg + CNN1D + IID 数据**

```bash
python main.py \
    --dataset RML2016.10a \
    --data_snr 100dB \
    --algorithm FedAvg \
    --model CNN1D \
    --num_clients 10 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --optimizer adam \
    --non_iid_type iid
```

**示例 2：FedProx + ResNet1D + Non-IID（按类别）**

```bash
python main.py \
    --dataset RML2016.10a \
    --data_snr 100dB \
    --algorithm FedProx \
    --model ResNet1D \
    --num_clients 10 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --optimizer adam \
    --non_iid_type class \
    --alpha 0.3 \
    --mu 0.01
```

**示例 3：FedGen + CNN1D + Non-IID（按 SNR）**

```bash
python main.py \
    --dataset RML2016.10a \
    --data_snr 100dB \
    --algorithm FedGen \
    --model CNN1D \
    --num_clients 10 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --gen_learning_rate 0.001 \
    --latent_dim 100 \
    --optimizer adam \
    --non_iid_type snr
```

**示例 4：使用 SGD 优化器**

```bash
python main.py \
    --dataset RML2016.10a \
    --data_snr 100dB \
    --algorithm FedAvg \
    --model CNN1D \
    --num_clients 10 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --optimizer sgd \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --non_iid_type iid
```

---

#### 训练输出

训练过程中会在控制台实时输出日志：

```
================================================================================
联邦学习自动调制识别
================================================================================
数据集: RML2016.10a
SNR: 100dB
算法: FedAvg
模型: CNN1D
客户端数量: 10
训练轮次: 100
本地训练轮数: 5
批大小: 128
学习率: 0.01
优化器: ADAM
  - 权重衰减: 0.0001
Non-IID 类型: iid
设备: cuda
================================================================================
加载数据集...
类别数: 11
信号长度: 128
创建模型...
模型参数数量: 1234567
创建客户端...
已创建 10 个客户端
创建服务器...
================================================================================
开始训练...
================================================================================
Round 1/100 | Local Loss: 2.3456 | Test Loss: 2.2345 | Test Accuracy: 25.67%
Round 2/100 | Local Loss: 1.8765 | Test Loss: 1.7654 | Test Accuracy: 38.92%
...
```

**训练完成后，结果保存在 `results/` 目录：**

```
results/
└── RML2016.10a_100dB_FedAvg_65.43%_10_0.01_128_5_10221530/
    ├── RML2016.10a_FedAvg_log.txt         # 训练日志
    ├── RML2016.10a_FedAvg_metrics.csv     # 训练指标（Round, Accuracy, Loss）
    └── RML2016.10a_FedAvg_model.pt        # 模型权重
```

**文件夹名称说明：**
- `RML2016.10a`：数据集名称
- `100dB`：SNR 标识
- `FedAvg`：算法名称
- `65.43%`：最终测试准确率
- `10_0.01_128_5`：参数（客户端数_学习率_批大小_本地轮数）
- `10221530`：时间戳（月日时分）

---

### 4.3 可视化训练结果 (Visualizing Training Results)

训练完成后，可以使用 `main_plot.py` 绘制对比曲线。

#### 可视化命令

```bash
python main_plot.py \
    --dataset RML2016.10a \
    --metrics_dir ./results \
    --algorithms FedAvg FedProx FedGen \
    --output_dir ./results/plots
```

#### 参数说明

- `--dataset`：数据集名称
- `--metrics_dir`：指标文件所在目录（包含 `*_metrics.csv` 文件）
- `--algorithms`：要对比的算法列表（空格分隔）
- `--output_dir`：图表保存目录

**注意：** 
- 确保 `metrics_dir` 中存在对应的 CSV 文件（格式：`{dataset}_{algorithm}_metrics.csv`）
- 如果某个算法的文件不存在，脚本会跳过并提示

#### 输出

脚本会生成 3 张对比图：

1. **准确率对比曲线**：`RML2016.10a_accuracy_comparison.png`
2. **损失对比曲线**：`RML2016.10a_loss_comparison.png`
3. **组合曲线（准确率 + 损失）**：`RML2016.10a_combined_comparison.png`

**示例输出：**

```
================================================================================
训练结果可视化
================================================================================
数据集: RML2016.10a
对比算法: FedAvg, FedProx, FedGen
================================================================================

加载 FedAvg 的指标...
  ✓ 成功加载 100 轮数据
  最终准确率: 65.43%
  最终损失: 0.9876

加载 FedProx 的指标...
  ✓ 成功加载 100 轮数据
  最终准确率: 67.21%
  最终损失: 0.9234

加载 FedGen 的指标...
  ✓ 成功加载 100 轮数据
  最终准确率: 68.56%
  最终损失: 0.8912

================================================================================
开始绘图...
================================================================================

绘制准确率对比曲线...
准确率曲线已保存到: ./results/plots/RML2016.10a_accuracy_comparison.png
绘制损失对比曲线...
损失曲线已保存到: ./results/plots/RML2016.10a_loss_comparison.png
绘制组合对比曲线...
组合曲线已保存到: ./results/plots/RML2016.10a_combined_comparison.png

================================================================================
绘图完成！
================================================================================
图表已保存到: ./results/plots
================================================================================
```

---

### 4.4 SNR 性能评估 (SNR Performance Evaluation)

使用 `test_snr.py` 可以评估训练好的模型在不同 SNR 下的性能，并绘制准确率 vs. SNR 曲线。

#### 评估命令

```bash
python test_snr.py \
    --model_path ./results/RML2016.10a_100dB_FedAvg_65.43%_10_0.01_128_5_10221530/RML2016.10a_FedAvg_model.pt \
    --dataset_name RML2016.10a \
    --model CNN1D \
    --data_dir data_processed \
    --batch_size 256 \
    --device cuda
```

#### 参数说明

- `--model_path`：训练好的模型文件路径（`.pt` 文件）
- `--dataset_name`：数据集名称
- `--model`：模型架构（可选，默认从路径推断）
- `--data_dir`：预处理数据目录
- `--batch_size`：评估批大小（默认 256）
- `--device`：设备（cuda 或 cpu）

#### 输出

脚本会：
1. 加载所有单独 SNR 的测试集（如 `-20dB.pkl`, `0dB.pkl`, `18dB.pkl` 等）
2. 对每个 SNR 计算准确率
3. 保存结果到 `results/test/` 目录：
   - CSV 文件：`*_snr_metrics.csv`（包含 SNR 和 Accuracy 两列）
   - 日志文件：`*_snr_log.txt`
   - 图像：`*_snr.png`（准确率 vs. SNR 曲线）

**示例输出：**

```
================================================================================
按SNR评估模型性能
================================================================================
模型路径: ./results/.../RML2016.10a_FedAvg_model.pt
数据集: RML2016.10a
模型架构: CNN1D
类别数: 11
信号长度: 128
设备: cuda
================================================================================

加载模型...
模型加载完成

加载测试数据从 data_processed...
找到 20 个SNR测试集

开始评估...
  SNR = -20dB: Accuracy = 23.45% (Samples: 4400)
  SNR = -18dB: Accuracy = 28.67% (Samples: 4400)
  ...
  SNR =  16dB: Accuracy = 82.34% (Samples: 4400)
  SNR =  18dB: Accuracy = 84.56% (Samples: 4400)

保存结果到 results/test...
  CSV已保存到: results/test/RML2016.10a_100dB_FedAvg_..._snr_metrics.csv
  日志已保存到: results/test/RML2016.10a_100dB_FedAvg_..._snr_log.txt
  图像已保存到: results/test/RML2016.10a_100dB_FedAvg_..._snr.png

================================================================================
评估完成！
平均准确率: 62.45%
最高准确率: 84.56% (SNR = 18dB)
最低准确率: 23.45% (SNR = -20dB)
================================================================================
```

---

### 4.5 完整工作流示例

以下是一个从零开始的完整示例：

```bash
# 1. 激活虚拟环境
conda activate fl-amr

# 2. 数据预处理（首次使用时必须执行）
python dataset/datasplit.py --dataset RML2016.10a

# 3. 训练 FedAvg 模型
python main.py \
    --dataset RML2016.10a \
    --data_snr 100dB \
    --algorithm FedAvg \
    --model CNN1D \
    --num_clients 10 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --optimizer adam \
    --non_iid_type iid

# 4. 训练 FedProx 模型
python main.py \
    --dataset RML2016.10a \
    --data_snr 100dB \
    --algorithm FedProx \
    --model CNN1D \
    --num_clients 10 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --optimizer adam \
    --non_iid_type iid \
    --mu 0.01

# 5. 训练 FedGen 模型
python main.py \
    --dataset RML2016.10a \
    --data_snr 100dB \
    --algorithm FedGen \
    --model CNN1D \
    --num_clients 10 \
    --num_rounds 100 \
    --local_epochs 5 \
    --batch_size 128 \
    --learning_rate 0.01 \
    --gen_learning_rate 0.001 \
    --optimizer adam \
    --non_iid_type iid

# 6. 绘制对比曲线
python main_plot.py \
    --dataset RML2016.10a \
    --metrics_dir ./results \
    --algorithms FedAvg FedProx FedGen \
    --output_dir ./results/plots

# 7. 评估 SNR 性能（以 FedAvg 为例）
python test_snr.py \
    --model_path ./results/RML2016.10a_100dB_FedAvg_<准确率>_<参数>_<时间戳>/RML2016.10a_FedAvg_model.pt \
    --dataset_name RML2016.10a \
    --model CNN1D
```

**注意：**
- 第 7 步的 `--model_path` 需要替换为实际的模型路径（训练完成后会显示）
- 如果没有 GPU，在所有命令中添加 `--device cpu`

---

## 5. 常见问题 (FAQ)

### Q1: 数据集下载缓慢或无法访问怎么办？

**A:** DeepSig 数据集需要注册账号才能下载。如果无法访问，可以：
- 使用学术网络或 VPN
- 联系项目作者获取数据集

### Q2: 训练时显示 CUDA out of memory 怎么办？

**A:** GPU 显存不足，可以尝试：
- 减小 `--batch_size`（如改为 64 或 32）
- 减少 `--num_clients`（如改为 5）
- 使用 CPU 训练（添加 `--device cpu`）

### Q3: 如何调整超参数以提高准确率？

**A:** 可以尝试：
- 增加 `--num_rounds`（如 200）
- 增加 `--local_epochs`（如 10）
- 调整学习率 `--learning_rate`（尝试 0.001 ~ 0.1）
- 使用 ResNet1D 模型（`--model ResNet1D`）
- 调整 `--alpha` 参数（Non-IID class 场景）

### Q4: 如何对比不同优化器的效果？

**A:** 分别使用不同优化器训练：

```bash
# SGD
python main.py --algorithm FedAvg --optimizer sgd --momentum 0.9 ...

# Adam
python main.py --algorithm FedAvg --optimizer adam ...

# AdamW
python main.py --algorithm FedAvg --optimizer adamw ...
```

### Q5: 训练中断后如何继续？

**A:** 当前版本不支持断点续训。建议：
- 减少 `--num_rounds` 进行快速实验
- 使用后台运行或 tmux/screen 防止中断

### Q6: 如何在多个数据集上批量实验？

**A:** 可以编写批处理脚本（Windows 用 `.bat`，Linux 用 `.sh`）：

**Windows (run_experiments.bat)：**

```batch
@echo off
python main.py --dataset RML2016.10a --algorithm FedAvg ...
python main.py --dataset RML2016.10a --algorithm FedProx ...
python main.py --dataset RML2016.10b --algorithm FedAvg ...
...
```

**Linux (run_experiments.sh)：**

```bash
#!/bin/bash
python main.py --dataset RML2016.10a --algorithm FedAvg ...
python main.py --dataset RML2016.10a --algorithm FedProx ...
python main.py --dataset RML2016.10b --algorithm FedAvg ...
...
```

---

## 6. 进阶使用

### 6.1 自定义模型

如果您想使用自己的模型架构：

1. 在 `FLAlgorithms/trainmodel/models.py` 中定义新模型类
2. 在 `get_model()` 函数中添加模型选项
3. 使用 `--model <你的模型名称>` 训练

### 6.2 添加新的联邦学习算法

1. 在 `FLAlgorithms/servers/` 中创建新的服务器类（继承 `Server`）
2. 在 `FLAlgorithms/users/` 中创建新的客户端类（继承 `User`）
3. 在 `main.py` 的 `create_users()` 和 `create_server()` 中添加逻辑

### 6.3 自定义数据集

1. 在 `utils/model_config.py` 的 `DATASET_CONFIG` 中添加配置
2. 在 `dataset/datasplit.py` 中添加数据加载函数
3. 在 `dataset/data_loader.py` 中添加加载逻辑

---

## 7. 总结 (Conclusion)

本项目实现了一个完整的**联邦学习自动调制识别系统**，具有以下特点：

### 项目优势

- ✅ **模块化设计**：服务器、客户端、模型、数据加载器分离，易于扩展
- ✅ **多算法支持**：FedAvg、FedProx、FedGen 三种经典联邦学习算法
- ✅ **多模型支持**：CNN1D、ResNet1D 两种网络架构
- ✅ **多数据集支持**：RML2016.10a/10b、RML2018a、HisarMod
- ✅ **Non-IID 模拟**：支持 IID、按类别、按 SNR 三种数据划分
- ✅ **灵活的优化器**：支持 SGD、Adam、AdamW
- ✅ **完整的工作流**：数据预处理 → 训练 → 可视化 → SNR 评估
- ✅ **详细的日志**：训练过程、指标、模型权重自动保存

### 应用场景

- 频谱监测：自动识别无线信号调制类型
- 认知无线电：智能频谱感知
- 通信对抗：信号识别与分类
- 联邦学习研究：测试新算法、研究 Non-IID 问题

### 未来优化方向

1. **性能优化**：
   - 添加学习率调度器（如 CosineAnnealingLR）
   - 实现早停机制（Early Stopping）
   - 支持混合精度训练（AMP）

2. **功能扩展**：
   - 添加客户端采样策略（部分客户端参与训练）
   - 实现断点续训功能
   - 添加更多联邦学习算法（如 SCAFFOLD、FedOpt）

3. **可视化增强**：
   - 添加 TensorBoard 支持
   - 绘制混淆矩阵
   - 可视化客户端数据分布

4. **部署**：
   - 提供预训练模型
   - 实现模型导出（ONNX）
   - 添加推理脚本

---

## 8. 致谢与参考

### 数据集来源

- **RML2016.10a/10b**：DeepSig Inc.
- **RML2018a**：DeepSig Inc.
- **HisarMod**：IEEE DataPort

### 参考文献

1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
2. Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx)
3. Zhu et al., "Data-Free Knowledge Distillation for Heterogeneous Federated Learning" (FedGen)

---

## 9. 联系方式

如有问题或建议，欢迎通过以下方式联系：

- GitHub Issues：（请填写项目地址）
- Email：（请填写邮箱）

---

**祝您使用愉快！**

