@echo off
REM 联邦学习自动调制识别 - 示例运行脚本

echo ================================================================================
echo 联邦学习自动调制识别 - 快速示例
echo ================================================================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python
    pause
    exit /b 1
)

echo [1/4] 运行 FedAvg (IID)...
python main.py --dataset RML2016.10a --algorithm FedAvg --num_rounds 10 --num_clients 5 --local_epochs 3

echo.
echo [2/4] 运行 FedProx (Non-IID by class)...
python main.py --dataset RML2016.10a --algorithm FedProx --num_rounds 10 --num_clients 5 --local_epochs 3 --non_iid_type class --alpha 0.5 --mu 0.01

echo.
echo [3/4] 运行 FedGen (Non-IID by class)...
python main.py --dataset RML2016.10a --algorithm FedGen --num_rounds 10 --num_clients 5 --local_epochs 3 --non_iid_type class --alpha 0.5

echo.
echo [4/4] 生成对比图...
python main_plot.py --dataset RML2016.10a --algorithms FedAvg FedProx FedGen

echo.
echo ================================================================================
echo 示例运行完成！
echo 结果保存在 results/ 目录
echo 图表保存在 results/plots/ 目录
echo ================================================================================
pause

