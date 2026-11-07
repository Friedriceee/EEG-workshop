---
title: 论文完整标题
authors: 第一作者 et al.
year: 2024
venue: 期刊/会议名 或 arXiv
link: https://arxiv.org/abs/xxxx.xxxxx
added_by: 你的名字
date_added: 2026-11-08
tags: [alignment, domain-adaptation]
status: read
relevance: medium
---

<!--
复制本文件,改名为  <年份>_<第一作者姓>_<关键词>.md   例:2020_he_euclidean-alignment.md
文件名只用小写字母、数字、连字符、下划线。以 _ 开头的文件不会进入索引。
frontmatter(上面 --- 之间)每个字段都要填:
  tags       用 [a, b] 写法,常用标签见 papers/README.md
  status     to-read(还没读) | skimmed(只扫了摘要/图) | read(读完) | reproduced(复现过)
  relevance  high | medium | low   —— 对我们这次比赛的价值
下面的 HTML 注释是提示,写完可以删。
-->

## 一句话结论
<!-- 这篇论文做了什么、结论是什么,一句话。 -->

## 论文的设定
| 项目 | 内容 |
|---|---|
| 数据集 | |
| 被试数 / 通道数 / 采样率 | |
| 任务 | |
| 评测协议(同 Session / 跨 Session / 跨被试) | |
| 对比的基线 | |

## 方法(自己的话,3~5 句)

## 报告的结果
<!-- 抄原文数字,注明表/图编号和页码。不要抄摘要。 -->
| 指标 | 数值 | 出处 |
|---|---|---|
| | | Table ? / p.? |

## 和我们的差别 / 对我们有什么用
<!-- 人数、电极数、任务是否一致?结论在我们的 10 人 / 2655 段数据上还成立吗? -->

## 能不能用于比赛提交
- 能一个窗口一个窗口地做吗(因果 / 流式)?
- 能写进单个 `submission.py`,只用 numpy/torch/scipy 吗?参数量多大?
- 处理电极数不一致(训练 41 通道 vs 评测 47 通道)吗?

## 疑问 / 没看懂的地方
