# papers/ — 文献库

一篇论文一个 `.md` 文件。**索引由脚本自动生成,不要手改** → [INDEX.md](INDEX.md)

## 怎么加一篇论文(5 分钟)

```bash
git pull
git checkout -b paper/<你的名字>-<关键词>
cp papers/_TEMPLATE.md papers/2020_he_euclidean-alignment.md   # 改成你的文件名
# 编辑文件,填 frontmatter + 各个小节
python scripts/build_paper_index.py          # 本地检查格式并更新 INDEX.md(只用标准库)
git add papers && git commit -m "papers: add He 2020 euclidean alignment"
git push -u origin HEAD                      # 然后在 GitHub 上开 Pull Request
```

不熟 git 的话:直接在 GitHub 网页上 **Add file → Create new file**,路径填 `papers/xxx.md`,粘贴模板内容也行,选 "Create a new branch" 提 PR。

## 规则

- **PDF 不进仓库**(体积大、版权)。写 `link:`(arXiv / DOI / 官网)就行。
- **数字回到原文核对**,标明表/图编号和页码,不要抄摘要。
- 一个人 = 一个文件 = 一篇论文,避免多人改同一个文件(会冲突)。
- 想补充别人的笔记 → 在文末加一节 `## 补充(你的名字, 日期)`,不要改他的原文。
- frontmatter 不合格,CI 会报红,PR 合不了。本地跑 `python scripts/build_paper_index.py --check` 可提前发现。

## 常用标签

| 标签 | 含义 |
|---|---|
| `alignment` | 欧氏 / 黎曼对齐、重中心化 |
| `domain-adaptation` | 跨 Session / 跨被试迁移 |
| `riemannian` | 协方差 + 切空间方法 |
| `deep-learning` | EEGNet、Shallow、Transformer 等 |
| `foundation-model` | REVE、LaBraM、CBraMod 等预训练模型 |
| `augmentation` | 数据增强 |
| `channel-mismatch` | 通道数/位置不一致 |
| `online` | 在线 / 流式 / 因果推理 |
| `dataset` | 数据集或评测基准 |
| `survey` | 综述 |

需要新标签直接用,并回来在上表加一行。

## 和 `research/` 的关系

- `papers/` = **读了什么**:一篇论文一个文件,偏事实记录。
- `research/` = **准备做什么**:每人一个方法的实验方案,依据 `papers/` 里的论文。
