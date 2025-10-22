@echo off
REM 联邦学习自动调制识别 - 完整实验脚本

echo ================================================================================
echo 联邦学习自动调制识别 - 完整实验
echo 警告: 此脚本将运行 100 轮训练，可能需要较长时间
echo ================================================================================
echo.

set /p confirm="是否继续? (y/n): "
if /i not "%confirm%"=="y" (
    echo 已取消
    exit /b 0
)

echo.
echo [1/3] 运行 FedAvg (100 轮)...
python main.py --dataset RML2016.10a --algorithm FedAvg --num_rounds 100 --num_clients 10 --local_epochs 5 --non_iid_type class --alpha 0.5

echo.
echo [2/3] 运行 FedProx (100 轮)...
python main.py --dataset RML2016.10a --algorithm FedProx --num_rounds 100 --num_clients 10 --local_epochs 5 --non_iid_type class --alpha 0.5 --mu 0.01

echo.
echo [3/3] 运行 FedGen (100 轮)...
python main.py --dataset RML2016.10a --algorithm FedGen --num_rounds 100 --num_clients 10 --local_epochs 5 --non_iid_type class --alpha 0.5

echo.
echo 生成对比图...
python main_plot.py --dataset RML2016.10a --algorithms FedAvg FedProx FedGen

echo.
echo ================================================================================
echo 完整实验完成！
echo 结果保存在 results/ 目录
echo ================================================================================
pause

