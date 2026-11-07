# Track 2 (BCI decoding) — 阶段 1 状态,供会议使用

最后更新:2026-11-07。所有数字都是本地结果,单 seed(33),不是官方榜单成绩。

## 一页结论

| 协议      | 含义                                           | Riemannian+LR | EEGNet |
| ------- | -------------------------------------------- | ------------- | ------ |
| `xrun`  | 同一 Session 内,前 4 个 run 训练、后 2 个 run 测试       | **0.714**     | 0.425  |
| `xsess` | 同一被试跨 Session(仅 P04、P10 有 3 个 Session,共 6 折) | **0.526**     | 0.526  |
| `xsubj` | 留一被试(10 折)                                   | 0.458         | 0.431  |

指标:各 (被试, Session) 单元的 balanced accuracy 再取平均;三分类随机水平 = 0.333。

1. **Session shift 在这份数据里很明显**:Riemannian 同 Session 0.71 → 跨 Session 0.53 → 跨被试 0.46。比赛真正要解决的是中间这一档。
2. **官方 NeuralBench 的 EEGNet baseline 在随机水平**:`neuralbench eeg _motor_imagery_stream -m eegnet`,3 个 seed 的 `test/bal_acc_stream_mean` = 0.333 / 0.325 / 0.343。
   已排查(`scripts/diag_official_eegnet.py`),**没有发现 bug,更像是任务本身难**:
   - 标签和窗口与我们一致(`Stimulus` 事件的 code 0/1/2 = mi/sub/word;提示后 1.0 s 起 4 s),预处理(120 Hz、0.1–75 Hz、50/60 Hz 陷波、µV)一致。官方 EEGNet 吃的是**未标准化的原始 µV**(`scaler: null, clamp: null`)。
   - 按官方配方(原始 µV、AdamW lr 1e-4 + OneCycle、40 epoch、patience 5)在我们的 6/2/2 被试划分上复现:3 个划分平均 0.378(0.399 / 0.380 / 0.355),略高于随机但接近噪声。
   - 单独改一个因素:lr 1e-3 → 0.400;RobustScaler+clamp → 0.357;逐 trial z-score + lr 1e-4 → 0.327;z-score + lr 1e-3 → 0.372。**没有哪个因素能稳定把它拉出随机水平**,标准化不是主因。
   - 自写 EEGNet 在 `xsubj` 拿到 0.431,是因为训练被试 9 人(官方 6 人);测试集只有 2 个人时,标准误约 ±0.02–0.04,0.33 与 0.40 的差别主要是噪声。
   - 结论:6 人训练的跨被试 EEGNet 在这份数据上信号很弱,官方 baseline 在随机水平附近是合理的。**不要把它当"我们做错了"**,也不要把它当强基线。
3. **经典方法(协方差 + 切空间)明显强于 EEGNet**(同 Session 0.71 vs 0.43)。数据只有约 2655 个 trial,小数据下深度模型吃亏。
4. **REVE 能在这台 Mac(MPS)上跑通 `--debug`**(69.5M 参数,debug 测试 bal_acc 0.349,仅 1–2 epoch,无意义)。完整微调的耗时和显存还没测。

## 数据事实(PROTEUS 训练集)

- 10 名被试,**14 个 Session**:P04、P10 各 3 个,其余 8 人各 1 个。**能做真正跨 Session 验证的只有 2 个人**,`xsess` 方差很大,不要过度解读。
- 41 通道,原始 500 Hz,官方预处理重采样到 120 Hz;窗口 = 提示后 1.0–5.0 s(4 s,480 点)。
- 带标签 trial 共 2655 个:MI 885 / 减法 886 / 词语 884,类别均衡;每个 Session 约 190 个 trial。
- 每个 Session 6 个有标签 run:AcquisitionGraz/BH(假反馈)+ OnlineRawGraz/BH ×2(真实反馈)。Baseline 睁眼/闭眼 run 无标签。
- 比赛评测集:20 人 × 6 个 Session、47 通道(训练集 41 通道)。隐藏的是后面的 Session,所以**通道数不一致的问题必须在模型里处理**。

## 阶段 1 清单对照

已完成:
- [x] 获取 PROTEUS 数据集并核对结构、通道、采样率、标签(`results/proteus_epochs.npz` 为预处理后的 trial)
- [x] 配置 Python 3.12 + PyTorch 2.14 + NeuralBench 0.4.0 + MOABB + braindecode + pyriemann(`requirements-core.txt`)
- [x] 确认计算资源:MacBook Air(Apple 芯片),MPS 可训练;无 CUDA
- [x] 跑通官方 Baseline(`neuralbench ... -m eegnet`,完整 3 seed)
- [x] 尝试运行 REVE(`--debug` 通过;完整微调未测)
- [x] 建立 session-wise / subject-wise / run-wise 三套验证划分(`scripts/eval_baselines.py`)
- [x] 保存配置与结果(`results/*.json`,`neuralbench_config.json`);已 `git init`(未提交,未推送)

部分完成 / 需要注意:
- [~] "复现 EEGNet,记录本地 BA":官方 NeuralBench 复现结果在随机水平(见上),原因待查;我们自己的 EEGNet 见上表。
- [~] "实现与官方一致的评分函数":我用 sklearn `balanced_accuracy_score`,按 (被试, Session) 单元求平均。**没有对照官方 Codabench 评分代码**,需要读 Participant Guide / benchopt 套件后确认(官方还要求按流式逐窗口推理,我没有实现)。

## 下一步建议

1. 查清 NeuralBench EEGNet 为什么在随机水平(对比其预处理与我们的逐 trial 标准化)。
2. 先用 Riemannian 做强 baseline,重点做**跨 Session 适配**:逐 Session 的协方差重对齐(Riemannian/Euclidean alignment)、每个被试单独模型(规则允许)。
3. 读 Submission Guide 的流式推理要求,评估重对齐能否在线实现。
4. 测 REVE 完整微调的速度/显s s存,再决定值不值得投入。

## 复现命令

```bash
./.venv/bin/python -I scripts/build_dataset.py data/Dreyer2026Proteus/download/nm000290 results/proteus_epochs.npz
./.venv/bin/python -I scripts/eval_baselines.py xrun,xsess,xsubj riem,eegnet run
```
