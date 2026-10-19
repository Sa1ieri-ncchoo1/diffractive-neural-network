# 运行环境与复现指南

[English](REPRODUCIBILITY.md) · [依赖说明](DEPENDENCIES.md) · [返回首页](../README.md)

## 环境

完整测试集评估使用 Windows、Python 3.10.20、torch 2.11.0.dev20260203+cu128、torchvision 0.25.0.dev20260205+cu128、NumPy 2.0.1、Matplotlib 3.9.1，GPU 为 NVIDIA GeForce RTX 5070 Laptop GPU。这是已安装环境的记录，未验证在全新环境中安装同一套 nightly 构建。

按首页创建环境，安装四项直接依赖。从项目根目录执行以下命令。`python` 应指向当前环境；所有输出目录必须尚不存在。

## 无数据集功能检查

```bash
python -m d2nn.cli smoke --config configs/current_56.json --device cpu --output runs/smoke
python -m unittest discover -s tests -v
```

前者检查合成输入的前向、反向和参数更新；后者执行五项传播/梯度/配置测试。都不需要联网或下载 MNIST。

## 五组历史权重评估

```bash
python -m d2nn.cli evaluate --config configs/baseline_200.json --data-root data --download --output runs/baseline
python -m d2nn.cli evaluate --config configs/contrast_200.json --data-root data --output runs/contrast200
python -m d2nn.cli evaluate --config configs/contrast_100.json --data-root data --output runs/contrast100
python -m d2nn.cli evaluate --config configs/contrast_100_ring20.json --data-root data --output runs/ring20
python -m d2nn.cli evaluate --config configs/pixelmse_100.json --data-root data --output runs/pixelmse
```

MNIST 由 torchvision 获取，标准训练集 60,000 张、测试集 10,000 张。第一次下载使用 `--download`；之后可复用 `--data-root` 指定的缓存。未加下载开关且数据缺失时，程序报错而不会下载。

评估默认选择可用 GPU，否则使用 CPU。可显式指定 `--device cpu`，但历史完整测试集结果是在 CUDA 上测得。默认批大小 32，内存不足时减少 `--batch-size`。不要添加 `--limit` 来生成正式准确率；该选项只取前 N 张图像，适合功能检查。

每次输出包括 `config.json`、`metrics.json`、`confusion_matrix.csv` 和 `confusion_matrix.png`。检查样本数是否为 10,000、checkpoint SHA-256 是否匹配，并保留实际环境记录。

## 新训练与重载

先用小样本走通流程：

```bash
python -m d2nn.cli train --config configs/current_56.json --data-root data --epochs 2 --limit 64 --batch-size 16 --output runs/train_smoke
python -m d2nn.cli evaluate --config runs/train_smoke/config.json --checkpoint runs/train_smoke/model.pt --data-root data --limit 64 --batch-size 16 --output runs/reload_smoke
```

完整新训练可使用：

```bash
python -m d2nn.cli train --config configs/contrast_100.json --data-root data --epochs 15 --batch-size 256 --output runs/train100
```

程序保存最后一轮的权重、逐轮 `history.csv`、训练曲线及混淆矩阵。批大小影响优化，降低批大小后应明确记录。新配置保留随机种子 42、原始数值模型和损失；历史增强配置未完整保存，因此不承诺重现原权重的最终准确率。

当前训练每轮评估测试集，沿用原研究开发方式；未来选择超参数时应另划验证集。程序不保存优化器状态，不支持精确断点续训。跨设备、跨软件版本的完全一致性也未保证。

## 权重与配置检查

加载使用 `weights_only=True` 并严格匹配参数。程序还核验传播缓冲区和历史探测器掩膜，防止更改几何参数后仍误认为在评估同一个模型。请使用与权重配套的 JSON；不要为了消除报错关闭检查。

## 已知问题和验证边界

- 五组保存权重已完成完整测试集评估；未完成完整从头训练、多随机种子对比或受控提速测试。
- 本地曾在数值计算后绘图时出现重复 OpenMP 运行库冲突。当前先保存数值记录，再在独立进程绘图，没有启用忽略冲突的环境变量。
- 若仅绘图失败，修复环境后可执行 `python -m d2nn.plotting runs/contrast200` 重绘已有记录。
- 小样本检查记录仅证明训练、保存、重载和绘图功能可用，不支持性能或收敛结论。
- FDTD 代码依赖不完整，未包含在公开版；详见单独的局限文档。
