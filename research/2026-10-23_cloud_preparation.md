# Qwen 扩展实验：云端准备记录

2026-10-23 已完成环境配置、三个模型下载和实际前向检查。本轮没有运行正式的 BBQ 方法比较，也没有修改论文结果。

## 服务器与目录

- GPU：`nvidia-smi` 报告 NVIDIA GeForce RTX 4090，49,140 MiB 显存，驱动 550.163.01。
- Conda 环境：`/root/conda-envs/egus`，从原基础 Conda 环境克隆，Python 3.12.9。
- 代码：`/root/mechanism`。
- 模型：`/hy-tmp/models`。
- 日志与检查记录：`/root/mechanism/logs`。
- 准备结束时模型盘剩余约 8.62 GB，系统盘剩余约 25.32 GB。

进入环境：

```bash
cd /root/mechanism
source scripts/cloud/env.sh
```

核心依赖为 PyTorch 2.6.0+cu124、Transformers 5.18.0、scikit-learn 1.9.1、NumPy 2.5.3、SciPy 1.18.1。CUDA 矩阵运算通过，`pip check` 没有报告依赖缺失或冲突。完整解析版本保存在云端 `logs/environment.lock.txt`，本地副本位于 `artifacts/cloud_setup_20261023/`。

服务器驱动支持 CUDA 12.4，因此这里没有直接使用本地的 PyTorch 2.14.1+cu132。Python 版本也与本地的 3.13 不同。云端通过 PYTHONPATH 导入项目源码；未绕过本地项目安装元数据中的 Python 3.13 要求。

## 模型与固定版本

| 官方仓库 | 固定 revision | 权重大小（十进制 GB） |
|---|---|---:|
| Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 | 16.38 |
| Qwen/Qwen3.5-4B | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | 9.32 |
| Qwen/Qwen3.5-9B | c202236235762e1c871ad0ccb60c8ee5ba337b9a | 19.31 |

模型文件从 hf-mirror 下载。服务器上部分配置和 tokenizer 下载不稳定，因此这部分文件先在本地从 hf-mirror 下载，再上传至服务器。完整模型没有量化。每个模型目录均有 `download_verified.json`；权重通过上游 LFS SHA256 校验，其余文件记录实际 SHA256 并检查预期大小。总计 39 个模型文件。

## 实际模型检查

使用两个短文本输入、FP16 和 SDPA，逐模型验证文本前向、最后提示 token 的 block 输出读取、零向量干预、非零向量干预及批内样本隔离。另比较左填充批处理与单独推理的选项分数。这里的随机向量只检查干预通路，不表示去偏效果。

| 模型 | 层数 | 检查位置（零起始） | 隐藏维度 | 短输入检查显存峰值（GiB） | 结果 |
|---|---:|---:|---:|---:|---|
| Qwen3-8B | 36 | 24 | 4096 | 15.27 | 通过 |
| Qwen3.5-4B | 32 | 21 | 2560 | 8.60 | 通过 |
| Qwen3.5-9B | 32 | 21 | 4096 | 17.59 | 通过 |

零干预保持输出完全一致；非零干预改变被编辑样本的选项分数，未编辑样本的分数完全一致。填充与单独推理的最大选项 logit 差异分别为 0.015625、0.015625、0。

Qwen3.5 使用原生 `Qwen3_5ForConditionalGeneration` 加载，文本 decoder 的 block 位于 `model.model.language_model.layers`；Qwen3 位于 `model.model.layers`。检查仅输入文本。Qwen3.5 的线性注意力当前使用正确的 PyTorch 参考实现，未安装可选的 flash-linear-attention 加速内核。以上显存值来自短输入检查，不代表完整实验的峰值。

## 代码与数据

已上传研究源码、现有配置和测试文件。BBQ 数据保留五个拟合抽样（每个 2,200 题）、352 题开发集和 704 题评价集，并重新验证家族隔离。服务器检查了 223 个源码及数据文件的 SHA256，与本地清单一致。旧模型路径和冻结配置保留原样；本次准备不会覆盖历史结果。

云端 `logs/preparation_report.json` 汇总环境、模型版本和检查结果，`logs/preparation_complete.json` 确认三个模型检查均通过。本地副本位于 `artifacts/cloud_setup_20261023/`。

下一轮需要单独建立 BBQ 云端实验入口，将 Qwen3.5 的加载及 hook 接口接入现有拟合和评价流程，再冻结新配置。现有 `run_evidence_validation.py` 含历史数据准备和 Windows 路径，不能直接作为本轮启动器。
