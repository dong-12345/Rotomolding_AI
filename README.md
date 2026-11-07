# 滚塑仿真与 AI 训练

本仓库包含滚塑仿真程序和 AI 训练代码：先使用 [simulation](simulation/README.md) 导入模具并生成仿真结果，再按 [training](training/README.md) 中的说明准备数据并训练模型。

- `simulation/`：基于 WebGPU 的滚塑仿真、桌面程序和命令行工具。
- `training/`：仿真数据预处理、网格神经网络训练和预测代码。

## 使用流程

1. 按[仿真说明](simulation/README.md)配置环境，使用 STEP/STP 模具运行仿真。
2. 按[训练说明](training/README.md)整理彩色 OBJ 和对应的工况转速参数。
3. 运行数据预处理与训练，使用训练好的模型进行预测。

训练集、验证集、测试集、CAD 文件、仿真结果、日志、训练产物及模型权重保存在本地，不包含在仓库中。训练前请核对转速参数与仿真工况是否一致。
