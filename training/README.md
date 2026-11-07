# 滚塑仿真 AI 训练

本项目把滚塑仿真输出的彩色 OBJ 转换成网格与标签，训练网格神经网络预测厚度相关的标量。先使用 `rotomolding_webgpu` 仿真，再在本项目中准备数据并训练。

## 环境

需要 Python、NumPy 和 PyTorch。示例（PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如需 CUDA，请按本机环境安装相应的 PyTorch 版本。

## 整理仿真数据

`tools/prepare_native_dataset.py` 需要以下本地目录结构：

```text
dataset/
  rotation_parameters_01_to_25.json
  training_set/<模具名>/simulation_results/simulation_01.obj ...
  validation_set/<模具名>/simulation_results/simulation_01.obj ...
```

训练集和验证集按模具分开，OBJ 必须包含顶点颜色。参数文件必须来自实际仿真工况，并且编号与每个 OBJ 一致。其格式为：

```json
{"results": {"01": {"rotationSpeed": 0.8, "rotationSpeed2": 0.3}}}
```

上面的数值仅为格式示例，实际文件需包含所有使用的工况。当前本地仿真结果目录和 AI 数据目录未发现该文件：请根据对应仿真运行记录导出并核对参数，不能仅凭编号推断。不同批次如果编号对应的转速不同，需要分别准备数据或调整转换流程。

预处理从训练模具中再划出内部验证集；`validation_set` 中的模具作为测试集。默认从颜色映射恢复归一化标量，并非自动得到物理厚度；要使用物理单位，需要提供经过校准的 `--color-map`。

## 预处理与训练

在项目根目录运行：

```powershell
python tools/train_native_dataset.py --dataset-root dataset --device auto
```

默认将预处理结果写入 `dataset_prepared/`，训练结果写入 `runs/native_full/`。已有预处理结果默认会复用；数据或参数改变后使用 `--force-preprocess` 重建。可用 `--preprocess-only` 只准备数据，或通过 `--help` 查看训练与推理选项：

```powershell
python tools/train_native_dataset.py --help
python tools/predict_native.py --help
```

`tools/train_colored_dataset.py` 提供基于 CSV 的另一条数据导入路径。数据集、预处理缓存、日志、输出、检查点与模型文件均仅保留在本地，不提交 Git。
