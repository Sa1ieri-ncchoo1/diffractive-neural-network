# 代码与实验来源 / Provenance

数值模型来自本科 D²NN 研究代码。公开版将原始 `train.py` 中的八个定义提取到 `d2nn/core.py`，保持函数与类的语法树一致：随机种子、ASM 传播、圆形探测器、纯相位网络、分类器，以及三种损失。没有改变传播 FFT、零填充、裁剪、相位参数化或探测器积分。

来源文件哈希和提取定义列表见 `core_provenance.json`。新增部分包括命令行入口、显式配置、严格权重检查、记录保存、独立进程绘图和功能测试。历史增强设置未完全恢复，因此训练配置标为重建配置。

五组模型权重和精选历史图片只改了文件名，原字节保持不变；来源、目标路径及 SHA-256 见 `source_manifest.json`。`validation/` 中保存的是实际执行记录，`assets/reported/` 是历史图，不应混为本次新生成结果。

公开版不包含 MNIST 数据、完整论文、个人材料、重复历史脚本或不完整 FDTD 实现。旧 FDTD 代码引用的辅助模块来源尚不完整，未随仓库分发。

直接使用的第三方库为 PyTorch、torchvision、NumPy 和 Matplotlib。MNIST 通过 torchvision 获取。第三方库和数据集的许可条件分别适用；当前仓库尚未指定开源许可证。

## English

Eight numerical definitions were extracted unchanged from the original research `train.py`; their Python ASTs matched during review. The CLI, configuration validation, logging, plotting isolation and tests are new orchestration code. Historical training augmentation is not fully recoverable.

Original checkpoint and selected figure bytes are preserved. See `source_manifest.json` and `core_provenance.json` for hashes and source mapping. Recorded evaluation outputs and historical figures are labeled separately. Personal documents, datasets, duplicate scripts and incomplete FDTD implementations are excluded. No repository license has been assigned; third-party terms remain applicable.
