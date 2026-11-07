# EEG Track 2 — BCI 解码比赛

跨 Session / 跨被试的脑机接口解码(PROTEUS 数据集:MI / 心算减法 / 词语想象,三分类)。
本仓库汇集团队的**文献库、方法调研、基线代码和实验结果**。

- 当前进展与数字 → [STATUS.md](STATUS.md)
- 论文库(组员在这里加论文)→ [papers/](papers/README.md) · [索引](papers/INDEX.md)
- 方法调研(每人一个方法的实验方案)→ [research/](research/README.md)

## 仓库结构

```
papers/      文献库:一篇论文一个 .md,INDEX.md 自动生成
research/    方法调研:每人一个方法 + 下周实验计划
scripts/     数据构建、基线评测、诊断脚本
results/     评测结果 JSON(大文件如 *.npz 不入库)
STATUS.md    阶段进展,开会用
```

## 组员加论文(最常用)

1. `git clone` 本仓库(或 GitHub 网页上直接操作)
2. `cp papers/_TEMPLATE.md papers/<年份>_<作者姓>_<关键词>.md`,按模板填写
3. `python3 scripts/build_paper_index.py --check` 检查格式
4. 新开分支 → 提交 → 开 Pull Request;CI 通过后合并,索引自动更新

详细规则见 [papers/README.md](papers/README.md)。

## 协作约定

- **不直接推 `main`**,都走 Pull Request(分支名 `paper/<名字>-<关键词>`、`exp/<名字>-<关键词>`)。
- **不提交**:PDF、数据集、模型权重、`.venv`、大于几 MB 的文件。数据放本地 `data/`(已忽略)。
- 论文结论要回到原文核对;论文的提升不等于在我们数据上的提升,需用统一的验证协议(`xrun` / `xsess` / `xsubj`)重新验证。
- 每个方法都回答:能不能一个窗口一个窗口地做?能不能写进单个 `submission.py`?

## 环境(跑代码才需要;只加论文不用装)

Python 3.12,依赖见 [requirements-core.txt](requirements-core.txt):

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements-core.txt
```

NeuralBench 需要一个本地的 `neuralbench_config.json`(已加入 `.gitignore`,每人路径不同),把 `DATA_DIR` / `CACHE_DIR` / `SAVE_DIR` 指到自己机器上的目录。数据获取与复现命令见 [STATUS.md](STATUS.md) 末尾。
