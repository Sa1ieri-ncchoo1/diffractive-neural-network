# D²NN 手写数字识别：损失函数优化与模型规模对比

中文 | [English](README.en.md)

基于 **PyTorch 和角谱传播法（ASM）** 的纯相位衍射深度神经网络项目，以 MNIST 手写数字分类为任务，研究损失函数、归一化方式及相位网格大小对分类性能的影响。

项目包括可运行的训练与评估入口、五组历史模型权重、明确的实验配置、混淆矩阵和结果核验记录，并单独记录尚未成功的 FDTD 验证尝试。

## 主要结果

以下五组权重已于 **2026-10-12** 在完整的 **10,000 张 MNIST 测试图像**上重新评估。

| 配置 | 历史记录 | 保存权重复测 | 可训练相位参数 |
|---|---:|---:|---:|
| 5 层，200×200，One-hot MSE + Sum 归一化 | 89.47% | **89.47%** | 200,000 |
| 5 层，200×200，Contrast Loss + Sum 归一化 | 97.56% | **97.56%** | 200,000 |
| 5 层，100×100，Contrast Loss，探测器环半径 14 px | 97.30% | **97.30%** | 50,000 |
| 5 层，100×100，Contrast Loss，探测器环半径 20 px | 97.47% | **97.47%** | 50,000 |
| 5 层，100×100，Pixel-MSE + 光强 Max 归一化 | 91.20% | **91.20%** | 50,000 |

- 损失与归一化方案调整后，准确率由 **89.47% 提升至 97.56%**，相差 **8.09 个百分点**。
- 相位网格由 200×200 缩小到 100×100，可训练参数减少 **75%**；环半径 14 px 的模型准确率降低 **0.26 个百分点**。
- 这次验证是**保存权重的推理复测**。训练入口已通过小样本检查，尚未完成完整从头训练复现。

![97.56% 模型在完整 MNIST 测试集上的混淆矩阵](validation/contrast_200_verified/confusion_matrix.png)

压缩实验还改变了物理口径和探测器几何，不能解释成仅改变参数量的单变量实验。详细证据和配置差异见[实验结果说明](docs/RESULTS.zh-CN.md)，更多图片见[结果图集](docs/FIGURES.md)。

## 方法

```mermaid
flowchart LR
    A[MNIST 灰度图] --> B[缩放与平方根振幅编码]
    B --> C[角谱传播 ASM]
    C --> D[可训练纯相位调制]
    D --> E[重复多层传播与调制]
    E --> F[末端传播与光强计算]
    F --> G[环形排列的圆形探测器]
    G --> H[类别响应与损失函数]
```

每层相位采用 `π·tanh(parameter)` 参数化。传播过程使用正交归一化 FFT、2 倍零填充和传播后裁剪，保持原始数值定义。

- **One-hot MSE**：对 Sum 归一化后的类别响应计算均方误差。
- **Contrast Loss**：约束正确类别相对最大错误类别的响应优势，margin 为 0.1。
- **Pixel-MSE**：对图像级 Max 归一化的输出光强与目标探测器掩膜计算均方误差。

当前公开入口使用振幅编码；后续相位输入实验和不完整的 FDTD 脚本未纳入支持范围。

## 快速开始

已验证的 Python 版本为 **3.10.20**。在项目根目录创建并激活独立环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
source .venv/bin/activate
```

安装依赖并进行不需要数据集的 CPU 功能检查：

```bash
python -m pip install -r requirements.txt
python -m d2nn.cli smoke --config configs/current_56.json --device cpu --output runs/smoke
python -m unittest discover -s tests -v
```

`requirements.txt` 只包含 **torch、torchvision、numpy、matplotlib** 四项直接依赖。GPU 用户需要安装适合硬件的 torch/torchvision 组合；用途、版本和间接依赖见[依赖说明](docs/DEPENDENCIES.md)。当前未做全新环境安装验证。

### 评估已有权重

```bash
python -m d2nn.cli evaluate --config configs/contrast_200.json --data-root data --download --output runs/contrast200
```

首次下载 MNIST 需要联网；已有数据时省略 `--download`。没有可用 GPU 时添加 `--device cpu`。默认评估批大小为 32，可用 `--batch-size 16` 降低内存需求。每次运行需指定新的输出目录，避免覆盖结果。

运行后会保存配置、权重 SHA-256、样本数、准确率、混淆矩阵 CSV 和 PNG。其他模型通过切换 `configs/` 中的 JSON 文件评估。

### 训练新模型

```bash
python -m d2nn.cli train --config configs/contrast_100.json --data-root data --download --epochs 15 --output runs/new_training
python -m d2nn.cli evaluate --config runs/new_training/config.json --checkpoint runs/new_training/model.pt --data-root data --output runs/new_training_eval
```

新训练会生成 `history.csv`、训练曲线、模型权重和评估结果。历史训练增强配置未完整保存，因此这些重建配置不保证从头得到表中的准确率。完整操作说明见[复现指南](docs/REPRODUCIBILITY.zh-CN.md)。

## 项目目录

```text
README.md                 中文首页
README.en.md              English README
requirements.txt          四项直接运行依赖
d2nn/                     数值核心、命令入口、绘图
configs/                  评估与新训练配置
checkpoints/              五组已核验权重
tests/                    传播、梯度和配置检查
validation/               已执行评估与功能测试的记录
assets/reported/          精选历史样本图、FDTD 诊断图
docs/                     复现、依赖、结果和局限说明
```

数据集通过 torchvision 获取，未随仓库分发。

## 验证范围与局限

| 项目 | 状态 |
|---|---|
| 五组权重完整测试集推理 | 已完成，结果与历史记录一致 |
| 传播、梯度、配置检查 | 5 项测试通过 |
| CPU 合成输入检查 | 前向、反向与参数更新通过 |
| 小样本训练与权重重载 | 已完成；64 张训练图、64 张测试图、2 轮 |
| 完整从头训练、多随机种子统计 | 未完成 |
| 论文所述约 4 倍训练提速 | 未受控复测，不作为已验证指标 |
| Lumerical FDTD 物理验证 | 未成功，本次未重新运行 |

FDTD 历史结果中的约 11% 对应 **9 个诊断样本中判对 1 个**，不是完整测试集准确率。不同实验还改变了输入方式、层数或材料参数，不能直接与主模型 97.56% 计算统一精度差。详见[FDTD 验证与局限](docs/FDTD_LIMITATIONS.zh-CN.md)。

未找到历史逐轮训练曲线，故不展示推测的收敛过程。后续新训练可通过命令入口产生真实曲线。历史开发中使用了测试集反馈，这些结果不属于全新未接触过的留出集基准。

## 来源

数值核心从原始研究代码提取，传播、调制、探测器和损失定义保持一致；权重和历史图片保留原始字节，来源及哈希见[代码与数据来源说明](docs/PROVENANCE.md)。

当前尚未指定开源许可证；依赖库各自的许可条件仍适用。
