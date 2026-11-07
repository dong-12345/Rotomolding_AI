# 滚塑仿真与 AI 训练

本项目使用 WebGPU 和物质点法（MPM）进行滚塑过程仿真，支持 STEP/STP 模具导入、可视化和批量工况扫描。仿真结果可整理后交给 `Rotomolding_AI` 项目训练网格神经网络。

## 环境和启动

- Node.js、npm 和支持 WebGPU 的显卡及驱动。
- Windows 桌面程序使用 Electron；构建 Windows CLI 还需 .NET 10 SDK。

在本目录执行：

```powershell
npm ci
npm run dev       # 浏览器开发服务器
# 或 npm run desktop  # 构建并启动桌面程序
npm run dist:win  # 生成 Windows 桌面程序和 CLI，输出到 release/
```

## 先仿真，再训练

准备本地训练和验证用 STEP/STP 文件，然后使用构建后的 CLI；GPU 编号以本机查询结果为准：

```powershell
.\release\RotomoldingSimulation-cli.exe --list-gpus
.\release\RotomoldingSimulation-cli.exe sweep `
  --training-input "D:\molds\training" `
  --validation-input "D:\molds\validation" `
  --output "D:\simulation-results" `
  --gpu-index 0
```

查看 `CLI.md` 和 `--help` 了解批处理与参数。现有工作区的 `run_webgpu_dataset.ps1` 使用固定显卡编号、路径和旧版 `release_17`，并非通用启动命令。

将仿真结果按 AI 项目 README 所述布局整理到本地，确认转速参数 JSON 与每个 OBJ 工况一一对应，再运行 AI 项目的预处理和训练。仿真结果、原始 CAD、训练/验证/测试数据和训练模型均保存在本地，不纳入 Git。

## 代码结构

- `src/`：仿真、渲染和 WebGPU 着色器。
- `electron/`：桌面程序与 Windows CLI。
- `public/`：运行所需的 OpenCascade JavaScript/WASM 依赖。
- `TECHNICAL_DESIGN.md`：技术设计。
