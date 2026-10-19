# 依赖说明 / Dependencies

公开版只运行 ASM / MNIST 实验，不包含 FDTD 执行代码。`requirements.txt` 按 `d2nn/` 和 `tests/` 的实际导入整理，只有四项直接第三方依赖。

| 包 | 实际用途 | 使用位置 |
|---|---|---|
| `torch` | 复数张量、FFT、自动微分、优化器、权重加载 | `core.py`、`cli.py`、测试 |
| `torchvision` | MNIST 下载/读取、Resize、RandomAffine、ToTensor | `cli.py` |
| `numpy` | 随机种子、混淆矩阵计数、CSV 存取 | `core.py`、`cli.py`、`plotting.py` |
| `matplotlib` | 保存混淆矩阵和新训练曲线 | `plotting.py` |

`argparse`、`csv`、`hashlib`、`json`、`math`、`pathlib`、`platform`、`random`、`subprocess`、`sys`、`time`、`typing`、`unittest` 都属于 Python 标准库，不需要另行安装。测试使用 `unittest`，无需 pytest。

## 为什么没有其他包

- **Pillow**：项目没有直接调用 PIL；torchvision 和 matplotlib 的依赖声明会安装所需 Pillow，所以不重复列为直接依赖。它仍可能出现在安装后的环境中，这是正常的间接依赖。
- **SciPy、scikit-learn、tqdm**：公开入口不使用。混淆矩阵由 NumPy 统计，无需 scikit-learn。
- **h5py、hdf5storage、lumapi**：服务于未纳入公开版的 MAT/FDTD 链路，不应要求 MNIST 用户安装。
- **Jupyter、pandas、OpenCV、文档处理库**：代码不使用，不列入。

## 安装与版本边界

```bash
python -m pip install -r requirements.txt
```

这是**直接依赖清单，不是精确版本锁文件**。torch 与 torchvision 应作为匹配组合安装；CUDA 版本取决于硬件和所选 wheel，因此这里不强制所有用户安装本机的 CUDA nightly 构建。没有 GPU 时可使用 CPU 环境。

2026-10-12 完成数值核验的现有环境：

| 组件 | 实测版本 |
|---|---|
| Python | 3.10.20 |
| torch | 2.11.0.dev20260203+cu128 |
| torchvision | 0.25.0.dev20260205+cu128 |
| numpy | 2.0.1 |
| matplotlib | 3.9.1 |

版本记录用于追溯结果，不表示旧 nightly wheel 始终可下载，也不保证任意新版本都已测试。未验证全新环境安装；安装后应先执行 README 中的 smoke 和测试命令。`validation/environment.json` 保留原核验环境的历史快照，本公开版快照仅列出四项直接依赖。

## English summary

Only four direct runtime dependencies are declared: **torch, torchvision, numpy and matplotlib**. Tests use the standard-library `unittest`. Pillow is installed transitively by torchvision/matplotlib; scientific, MAT-export and Lumerical packages used only by excluded historical scripts are not required. This file is a dependency inventory, not a lockfile. Install a matching torch/torchvision pair for your CPU/CUDA platform, then run the smoke and unit tests. A fresh-environment installation has not been verified.
