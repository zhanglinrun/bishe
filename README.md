# 联邦学习自动调制识别（FedAvg/FedProx/FedGen & FedBKD）

## 1. 项目简介 (Introduction)

* **项目目标:** 面向无线电自动调制识别（Automatic Modulation Recognition, AMR）的联邦学习实验平台，支持经典联邦算法（FedAvg、FedProx、FedGen）以及基于知识蒸馏的联邦蒸馏方案（FedBKD）。提供数据预处理、联邦训练、指标可视化和按信噪比测试的完整流程。
* **核心技术:** 基于 1D CNN、1D ResNet 和 MCLDNN 的基线分类器，配合 FedAvg/FedProx/FedGen 联邦优化；另含 CVAE+GMM 伪样本生成与双向知识蒸馏的 FedBKD 实验管线。
* **框架:** 全部基于 PyTorch 实现，训练与数据管线均使用 PyTorch / scikit-learn / matplotlib 等开源组件。

## 2. 项目框架 (Project Structure)

* **目录树:**
  ```
  .
  ├── dataset/                    # 原始数据与预处理脚本
  │   ├── datasplit.py            # 按 SNR 划分并生成 train/test 及合并数据
  │   └── data_loader.py          # 预处理数据加载与客户端划分
  ├── data_processed/             # 运行 datasplit 后生成的预处理数据（需手动生成）
  ├── FLAlgorithms/               # 联邦学习核心实现
  │   ├── servers/                # 服务器端逻辑（FedAvg/FedProx/FedGen）
  │   ├── users/                  # 客户端训练逻辑
  │   ├── trainmodel/             # 分类模型与生成器
  │   └── optimizers/             # 优化器封装
  ├── FedBKD/                     # 基于蒸馏的联邦学习实验管线
  │   ├── distill/                # 蒸馏流程与通信协议
  │   ├── models/                 # 客户端/全局模型、CVAE、GMM 索引
  │   └── utils.py                # 评估与可视化工具
  ├── utils/                      # 通用工具（配置、日志、绘图）
  ├── results/                    # 训练与测试输出（运行后生成）
  ├── main.py                     # FedAvg/FedProx/FedGen 主训练入口
  ├── main_plot.py                # 训练指标可视化脚本
  ├── test_snr.py                 # 按 SNR 分档测试脚本
  ├── run_full_experiment.bat     # Windows 批处理示例（全流程）
  └── FedBKD/test.py              # FedBKD 模型按 SNR 评估
  ```

* **文件功能详解（逐个 .py 文件）：**

  - `main.py`
    - **作用:** 联邦学习主入口，封装参数解析、数据加载、模型/客户端/服务器创建以及训练、保存日志与模型。
    - **关键内容:** `parse_args` 定义全部超参；`set_seed` 固定随机性；`create_users`/`create_server` 根据算法实例化用户与服务器；`main` 组织训练流程并输出 CSV/模型。
    - **交互:** 调用 `dataset.data_loader.get_dataloaders` 读数据，`FLAlgorithms.trainmodel.models.get_model` 构建模型，用户/服务器类完成本地与聚合训练，`utils.model_utils` 负责日志与存储。

  - `main_plot.py`
    - **作用:** 读取 `results/<dataset>_<algorithm>_metrics.csv` 绘制准确率/损失对比图。
    - **关键内容:** `load_metrics` 读取 CSV，`plot_accuracy_curves`/`plot_loss_curves`/`plot_combined_curves` 生成三类图。
    - **交互:** 使用 `utils.plot_utils` 完成绘图，可与 `main.py` 生成的指标文件配合。

  - `test_snr.py`
    - **作用:** 逐个 SNR 切片评估已训练模型，输出 CSV/日志/折线图。
    - **关键内容:** `infer_model_from_path` 自动推断模型类型；`load_snr_test_data` 读取 `data_processed/<dataset>/test/*.pkl`；`evaluate_model` 逐批计算准确率；`plot_snr_accuracy` 绘制 SNR-Accuracy 曲线。
    - **交互:** 需加载 `main.py` 输出的 `.pt` 权重和 `data_processed` 测试集。

  - `dataset/datasplit.py`
    - **作用:** 预处理原始数据，按 SNR 分割生成训练/测试，并合并出 `100dB`（全量）与 `highsnr`（SNR>0）集合。
    - **关键内容:** 针对四个数据集的加载函数（`load_RML2016_10a/10b/2018a/HisarMod`），`split_and_save_by_snr` 按 SNR 分层划分并即时落盘，`generate_combined_datasets` 合并多 SNR 数据，`process_dataset`/`main` 批处理入口。
    - **交互:** 输出到 `data_processed/<dataset>/train|test/*.pkl`，供 `dataset/data_loader.py` 与训练脚本使用。

  - `dataset/data_loader.py`
    - **作用:** 从预处理文件加载数据，按 IID/类别 Non-IID/SNR Non-IID 划分客户端数据，并返回 PyTorch `DataLoader`。
    - **关键内容:** `load_preprocessed_data` 读取 `data_processed`；三种划分函数 `split_data_iid`、`split_data_non_iid_class`、`split_data_non_iid_snr`；`SignalDataset` 包装张量；`get_dataloaders` 综合加载与划分。
    - **交互:** `main.py` 调用以构建联邦客户端与测试集。

  - `utils/model_config.py`
    - **作用:** 定义各数据集的类别数、信号长度及类别名；`get_dataset_config` 提供查询。
    - **交互:** 被数据加载与模型构建调用以确定输出维度。

  - `utils/model_utils.py`
    - **作用:** 通用模型操作与日志：参数提取/设定、加权平均、日志文件写入、模型保存/加载、logger 配置。
    - **交互:** 服务器聚合与训练脚本均依赖其中的平均与日志工具。

  - `utils/plot_utils.py`
    - **作用:** 绘制准确率/损失曲线、组合图、混淆矩阵及客户端数据分布。
    - **交互:** 被 `main_plot.py` 等分析脚本调用。

  - `FLAlgorithms/trainmodel/models.py`
    - **作用:** 定义三种 AMR 分类模型：`CNN1D_AMR`、`ResNet1D_AMR`、`MCLDNN_AMR`；`get_model` 返回实例。
    - **交互:** 训练入口与客户端类均通过 `get_model` 构建模型实例。

  - `FLAlgorithms/trainmodel/generator.py`
    - **作用:** FedGen 的伪特征生成器（`Generator`、`ConditionalGenerator`），负责从潜在向量生成嵌入或条件嵌入。
    - **交互:** FedGen 客户端/服务器在蒸馏或聚合时共享生成器参数。

  - `FLAlgorithms/optimizers/fedoptimizer.py`
    - **作用:** 统一获取 SGD/Adam/AdamW 优化器的简化工厂函数。
    - **交互:** 供自定义拓展时使用，核心客户端目前直接在 `userbase.py` 中创建优化器。

  - `FLAlgorithms/servers/serverbase.py`
    - **作用:** 服务器基类，定义参数下发、聚合接口、全局评估与训练记录。
    - **交互:** 所有服务器子类（FedAvg/FedProx/FedGen）继承并实现聚合与训练逻辑。

  - `FLAlgorithms/servers/serveravg.py`
    - **作用:** FedAvg 服务器实现，加权平均客户端参数并循环训练。
    - **交互:** 通过 `average_weights` 聚合用户模型，供 `main.py` 驱动。

  - `FLAlgorithms/servers/serverFedProx.py`
    - **作用:** FedProx 服务器（聚合同 FedAvg，差异在客户端近端项）。
    - **交互:** 与 `UserFedProx` 配合实现近端正则。

  - `FLAlgorithms/servers/serverpFedGen.py`
    - **作用:** FedGen 服务器，负责同时聚合分类器与生成器参数。
    - **交互:** 调用用户的分类器与生成器权重，实现数据自由知识蒸馏。

  - `FLAlgorithms/users/userbase.py`
    - **作用:** 客户端基类，封装优化器、损失、参数同步与测试。
    - **交互:** 被各具体算法客户端继承并覆盖 `train` 方法。

  - `FLAlgorithms/users/useravg.py`
    - **作用:** FedAvg 客户端，本地交叉熵训练。
    - **交互:** 与 `ServerAVG` 搭配执行标准联邦平均。

  - `FLAlgorithms/users/userFedProx.py`
    - **作用:** FedProx 客户端，在损失中加入近端项约束全局参数。
    - **交互:** 与 `ServerFedProx` 联合实现 FedProx。

  - `FLAlgorithms/users/userpFedGen.py`
    - **作用:** FedGen 客户端，先训练生成器，再用真实数据（可选伪数据）训练分类器，维护生成器投影与蒸馏温度。
    - **交互:** 与 `ServerFedGen` 同步分类器/生成器参数。

  - `FedBKD/main.py`
    - **作用:** 基于 CVAE+GMM 的联邦蒸馏主流程：数据加载、客户端 CVAE 训练、云端聚合并生成伪样本、双向蒸馏、日志与模型存储。
    - **关键组件:** `to_tensor_loader` 统一数据形状；`adjust_learning_rate` 调度 LR；`ClientWeightManager`/`AdaptiveDistillWeight` 调整蒸馏权重；主循环中生成伪样本、训练全局与客户端模型并记录性能。
    - **交互:** 依赖 `FedBKD/data_loader.py` 划分客户端数据，`FedBKD/models/*` 提供模型与生成器，`FedBKD/distill/*` 提供蒸馏与通信协议，`FedBKD/utils.py` 评估与可视化。

  - `FedBKD/data_loader.py`
    - **作用:** 从 RML2016.10a pkl 加载并按“每客户端限定若干调制类型”的非重叠划分生成联邦数据。
    - **关键函数:** `load_federated_data` 支持指定调制类别、SNR 范围、客户端/类别数并返回 train/test 字典，亦可返回全局数据。

  - `FedBKD/config.py`
    - **作用:** argparse 配置，包括数据路径、客户端数量、蒸馏温度、LR、CVAE 参数、日志/权重路径等。
    - **交互:** `FedBKD/main.py` 和 `FedBKD/test.py` 读取默认值，可通过命令行覆盖。

  - `FedBKD/utils.py`
    - **作用:** 评估准确率、t-SNE 可视化、混淆矩阵绘制、模型保存/加载。

  - `FedBKD/distill/client_to_cloud.py`
    - **作用:** 客户端→云端蒸馏：平均 logits、基于准确率/损失更新权重、`train_global_model` 通过 KL + CE 训练全局模型。

  - `FedBKD/distill/cloud_to_client.py`
    - **作用:** 云端→客户端蒸馏：自适应蒸馏权重管理（`AdaptiveDistillWeight`），`train_local_with_distill` 将 soft label 蒸馏到本地模型。

  - `FedBKD/distill/communication_protocol.py`
    - **作用:** 模拟隐私保护的通信协议：记录上传/下载、噪声注入、索引图分发与统计摘要。

  - `FedBKD/models/global_model.py`
    - **作用:** 统一的轻量级全局学生模型，`extract_logits` 提供蒸馏特征。

  - `FedBKD/models/client_models.py`
    - **作用:** 多种异构客户端卷积网络（1~5 与 A~E），`get_client_model` 按字母编号返回模型。

  - `FedBKD/models/cvae.py`
    - **作用:** 条件变分自编码器，带变形增强与 GMM 索引采样；`train_cvae` 训练函数；`collect_encoding_statistics` 供云端聚合。

  - `FedBKD/models/gmm_index_map.py`
    - **作用:** GMM 索引图生成器，支持隐私标签置换与采样；`fit_cloud_gmm`/`distribute_index_map` 等完成云端拟合与客户端接收。

  - `FedBKD/test.py`
    - **作用:** 对保存的 FedBKD 全局模型按不同 SNR 评估并写入 Excel。
    - **交互:** 需指定 `--model_path` 与数据 pkl，调用 `evaluate_model`。

  - `run_full_experiment.bat`
    - **作用:** Windows 平台一键执行 FedAvg/FedProx/FedGen 训练、绘图及按 SNR 测试的示例脚本。

## 3. 环境配置 (Environment Setup)

* **依赖列表（核心）：**
  - `torch`, `torchvision`
  - `numpy`, `h5py`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`（绘图混淆矩阵用）
  - `tqdm`（可选进度条）、`openpyxl`（FedBKD/test 写入 Excel）
* **创建虚拟环境（推荐 Conda）：**
  ```bash
  conda create -n fl-amr python=3.9 -y
  conda activate fl-amr
  ```
* **安装依赖：**
  ```bash
  # 使用已提供的依赖清单
  pip install -r requirements.txt

  # 若缺少绘图/Excel相关依赖，请补充安装
  pip install seaborn openpyxl
  ```
* **硬件要求:** 推荐具备 GPU 与匹配的 CUDA（`torch.cuda.is_available()` 为 True 可自动使用）；CPU 也可运行但训练时间更长。

## 4. 使用说明 (Usage)

### 4.1 数据准备 (Data Preparation)

1. **获取原始数据集**：
   - 将 `RML2016.10a_dict.pkl`、`RML2016.10b.dat` 等原始文件放入 `dataset/RML2016.10a/`、`dataset/RML2016.10b/`（或对应子目录）。
2. **按 SNR 预处理并生成 `data_processed/`：**
   ```bash
   # 处理所有支持的数据集，默认输出到 data_processed/
   python dataset/datasplit.py --dataset all --input_dir dataset --output_dir data_processed

   # 若只处理 RML2016.10a 示例
   python dataset/datasplit.py --dataset RML2016.10a --input_dir dataset --output_dir data_processed
   ```
   预处理完成后目录示例：`data_processed/RML2016.10a/train/100dB.pkl`、`data_processed/RML2016.10a/test/highsnr.pkl` 等。

### 4.2 训练流程 (Training Process)

**联邦训练主入口（FedAvg/FedProx/FedGen）**

```bash
python main.py \
  --dataset RML2016.10a \        # 数据集名称（见 choices）
  --data_snr 100dB \             # 选择使用的 SNR 子集（10dB/100dB/highsnr 等）
  --data_dir data_processed \    # 预处理数据所在目录
  --algorithm FedAvg \           # FedAvg | FedProx | FedGen
  --model MCLDNN \               # CNN1D | ResNet1D | MCLDNN
  --num_clients 10 \             # 客户端数量
  --num_rounds 40 \              # 全局训练轮次
  --local_epochs 5 \             # 每轮本地训练 epoch 数
  --batch_size 128 \             # 批大小
  --learning_rate 0.001 \        # 学习率
  --optimizer adamw \            # sgd/adam/adamw
  --non_iid_type class \         # iid | class | snr 划分方式
  --alpha 0.5 \                  # class Non-IID 的 Dirichlet 参数
  --mu 0.01 \                    # FedProx 近端系数（仅 FedProx 使用）
  --latent_dim 100 \             # FedGen 生成器潜变量维度（仅 FedGen 使用）
  --output_dir ./results         # 训练输出目录
```

参数含义：`--dataset/--data_snr/--data_dir` 决定加载的数据切片；`--algorithm` 选择联邦算法；`--model` 选择分类器架构；`--num_clients/--num_rounds/--local_epochs` 控制联邦迭代；`--optimizer/momentum/weight_decay` 控制优化器；`--non_iid_type/--alpha` 控制数据划分；FedProx/FedGen 相关参数在对应算法时生效。

输出：
* `results/temp_<timestamp>/` 中包含训练日志（`*_log.txt`）、指标 CSV（`*_metrics.csv`）、模型权重（`*_model.pt`）。脚本末尾会尝试根据准确率重命名目录。

### 4.3 测试/推理流程 (Testing/Inference Process)

**按 SNR 评估已训练模型（主联邦线路）**

```bash
python test_snr.py \
  --model_path results/RML2016.10a_100dB_FedAvg_xx.xx.../RML2016.10a_FedAvg_model.pt \
  --dataset_name RML2016.10a \
  --model MCLDNN \         # 若路径无法自动推断模型架构，请显式指定
  --data_dir data_processed \
  --batch_size 256 \
  --device cuda             # 或 cpu
```

输出：`<model_dir>/test/` 下生成 SNR-准确率 CSV、日志与曲线图。

**训练日志可视化**

```bash
python main_plot.py --dataset RML2016.10a --metrics_dir results --algorithms FedAvg FedProx FedGen --output_dir results/plots
```

**FedBKD 蒸馏管线（可选高级实验）**

1) 运行 FedBKD 主实验：
```bash
python FedBKD/main.py \
  --data_dir /path/to/RML2016.10a_dict.pkl \   # 注意默认值为绝对路径，需根据实际数据位置修改
  --num_clients 5 --mods_per_client 2 --num_classes 10 \
  --rounds 20 --batch_size 32 --lr 1e-3 --warmup_epochs 5 \
  --synthetic_per_class 200 --log_path ./FedBKD/logs
```
   该脚本自动完成：客户端 CVAE 训练 → 云端聚合 GMM → 伪样本生成 → 双向蒸馏 → 记录日志/模型（`logs/exp_radioml_FedBKD_<time>/`）。

2) 使用保存的 FedBKD 全局模型按 SNR 测试：
```bash
python FedBKD/test.py \
  --model_path FedBKD/logs/exp_radioml_FedBKD_xxxx/best_global_model.pth \
  --data_dir /path/to/RML2016.10a_dict.pkl \
  --num_classes 10 --batch_size 64
```
   结果写入 `results/snr_accuracy.xlsx` 的新工作表。

## 5. 总结 (Conclusion)

本项目提供从数据预处理、联邦训练到按 SNR 细粒度评估的全链路自动调制识别实验框架，并扩展了基于 CVAE+GMM 的数据自由知识蒸馏方案（FedBKD）。未来可考虑：补充更多联邦算法（如 FedAvgM/FedNova）、完善 FedGen 蒸馏投影层与伪数据训练、在 GPU 多进程环境下加速大规模客户端模拟，以及将预处理与训练流程封装为一键脚本或 Notebook，进一步降低上手门槛。
