# EEG Track 2 — BCI 解码比赛

跨 Session / 跨被试的脑机接口解码(PROTEUS 数据集:MI / 心算减法 / 词语想象,三分类)。
本仓库包含**基线代码、评测脚本和实验结果**。

- 当前进展与数字 → [STATUS.md](STATUS.md)

## 仓库结构

```
scripts/     数据构建、基线评测(eval_baselines.py)、诊断脚本
results/     评测结果 JSON(大文件如 *.npz 不入库)
STATUS.md    阶段进展
```

## 约定

- **不提交**:PDF、数据集、模型权重、`.venv`、大于几 MB 的文件。数据放本地 `data/`(已忽略)。
- 统一用 `xrun` / `xsess` / `xsubj` 三套验证协议比较方法,指标为 balanced accuracy(随机水平 0.333)。

## 环境

Python 3.12,依赖见 [requirements-core.txt](requirements-core.txt):

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements-core.txt
```

NeuralBench 需要一个本地的 `neuralbench_config.json`(已加入 `.gitignore`),把 `DATA_DIR` / `CACHE_DIR` / `SAVE_DIR` 指到自己机器上的目录。数据获取与复现命令见 [STATUS.md](STATUS.md) 末尾。
