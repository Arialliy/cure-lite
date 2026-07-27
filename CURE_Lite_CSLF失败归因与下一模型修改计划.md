# CURE-Lite：PFCR 负结果诊断与 CSLF 下一模型修改计划

> 审查日期：2026-07-27  
> 仓库：`Arialliy/cure-lite`  
> 基础 Git 快照：`main@70178d461386052d4bf7b0b66c0258b3a187b9c7`；本地 observability 实现由 r1/r2 `config.json` 的逐文件 SHA256 绑定，尚未形成新的 commit-bound snapshot  
> 当前正式状态：**HOLD_FORMAL_TRAINING**  
> 当前模型状态：CSLF 核心、真实 \(D_R\) 入口和训练链已闭合；v15 与 v15A
> bounded-400 均已完整结束并封存为负结果；v15A 已证明 completion-rooted
> crossing 有效，但 compact support 与 factual stability 未通过；下一模型为
> v15B support-oriented response root  
> 本报告性质：代码审查、已实现协议与后续实施规范；不声称远端仓库已经同步这些本地修改

## 2026-07-27 v15A 终态更新（优先于本文全部旧进度）

v15A 已按照运行前冻结协议完成唯一一次 \(D_R\)-only bounded-400：

```text
decision = BOUNDED_COMPLETION_ROOTED_CSLF_GATE_FAIL
failed_check = candidate_zero_level_gates
formal_800_authorized = false
D_V_accessed = false
D_T_accessed = false
```

它不是执行失败。三个 objective 均完成 400 updates，17 个产物、全部 receipt
连接和 checkpoint 均已闭合。相对父 v15 response-joint：

```text
added-target negative: 6/149 -> 125/149
all-negative clean pairs: 0/16 -> 8/16
component match: 3/16 -> 15/16
response sign: 1396/1396 -> 1396/1396
compact exact: 0/16 -> 1/16
target-outside new completion: 8 -> 43
factual gate: 15/16 -> 13/16
factual target negative: 321/335 -> 289/335
```

因此完成端锚定的作用已经得到真实训练证据，但全域 minus-root 导致支撑外扩，并
加剧了部分 synthetic/natural 共享参数竞争。v15A 必须冻结，不能追加步数、调阈值
或创建 r2。

下一版只允许一个新的空间定向根：

\[
A=Y_-\setminus Y_+,
\qquad
e_{\mathrm{root}}=A e_-+(1-A)e_+,
\]

\[
\mathcal L_{\mathrm{v15B}}
=
\mathcal E_\mu
\left(
e_{\mathrm{root}},\,
e_- - e_+
\right).
\]

这仍然是同一 scalar field、同一二维 pair energy，不增加模块、第三项 loss、
权重、margin、阈值或训练量。完整结果、源码闭合和 v15B 实施顺序见：

[`CURE_Lite_CSLF_v15A_bounded400正式结果与v15B模型决定.md`](CURE_Lite_CSLF_v15A_bounded400正式结果与v15B模型决定.md)。

在 v15B 重新通过 dataset-free 与同一 \(D_R\) bounded 门槛前，Formal-800、
Full CURE、其他 backbone 和三数据集仍不授权。

## 2026-07-27 历史执行状态（已被上节终态更新取代）

本计划的模型主线与创新边界没有改变，但工程状态已经向前推进：

- expanded dataset-free gate 已通过；
- dataset-free 的 17 项门禁已改为从原始 evidence 重算，不能只信任汇总布尔值；
- 真实 \(D_R\) scalar cache 已通过严格可复用入口重建，cache fingerprint
  仍为
  `569b0fb97d819cf1281ca1d148227bc1c5e229b8301065cb536656b5e578e645`；
- bounded 16-per-role population、400-update schedule、角色/来源暴露门禁、
  fixed-zero evaluator 和 prerequisite-bound authorization 已实现；
- 三目标训练已经接入一次性设备常驻 cache，update loop 内不再重复构造几何或
 传输 payload；
- 当前 `tests_v15` 为 `138 passed, 1 skipped`；默认 skipped 项是耗时真实
  \(D_R\) create-only integration，而该入口已另外实际运行成功；
- 真实 bounded-400 仍未执行，Formal-800 仍未授权，未读取新的 \(D_V/D_T\)。

因此，第 24 节的 55% 旧估计已被当前约 62% 的估计替代。完整、较短且按当前
代码同步的模型说明与结果见
[`CURE_Lite_CSLF_v15_完整模型设计与阶段结果.md`](CURE_Lite_CSLF_v15_完整模型设计与阶段结果.md)。

---

## 0. 证据边界

本报告直接审查了当前仓库中的：

- `cure_lite/coverage_state_level_set.py`
- `cure_lite/coverage_state_sobolev.py`
- `cure_lite/coverage_state_raw_catalog.py`
- `cure_lite/coverage_state_observability.py`
- `cure_lite/experiment/coverage_state_raw_catalog.py`
- `cure_lite/experiment/coverage_state_observability_protocol.py`
- `tests_v15/test_coverage_state_level_set.py`
- `tests_v15/test_coverage_state_raw_catalog.py`
- `tests_v15/test_coverage_state_observability.py`
- `tools/audit_coverage_state_observability.py`
- `protocols/IRSTD-1K/coverage_state_observability_v1/config.json`
- observability r1/r2 的 `config/raw_catalog/observability/decision/COMPLETE`
  五件套
- PFCR 正式 \(D_V\) 结果与严格 \(D_R\)-only 失败诊断
- 当前提交中的工程文件树、测试配置和历史结果文档

以下历史运行结果按冻结状态接受：

```text
CSLF 核心测试：19 passed
PFCR 正式评估回归：21 passed
全仓测试：1066 passed, 39 failed
256² 单像素 raw-field 探索：3 seeds 精确恢复、无额外负岛
```

本轮已经在本地真实 \(D_R\) 上完成 representation-neutral raw catalog 与
scalar/phase observability 的 create-only r1/r2 重放；两次全部结果文件
逐字节一致，正式决策为 `AUTHORIZE_SCALAR_CSLF`。定向测试当前为
76 passed；真实 \(D_R\) cache 集成与 GPU0 三目标两步代码烟测也已完成，
具体人口、门禁和指纹见第 9.4 与 11.5 节。

因此，关于 39 个当前失败的逐 node-id 归因仍须由本地 JUnit/pytest JSON
形成正式清单。在 inventory 完成前，不把它们整体预判为同一种历史绑定失败。
本报告可以确认的是：

1. 当前本地工作树已经增加 CSLF 核心场、representation-neutral raw
   population、真实 \(D_R\) adapter、observability 协议/工具和定向测试；
2. 训练几何 cache、fused step、确定性 schedule 与三类 objective runner 已
   存在；正式 cache/schedule artifact、CSLF 评估和训练结果封存文件尚不存在；
3. `pyproject.toml` 的默认 `testpaths` 仍只有 `tests`，所以普通 `python -m pytest -q` 不会自动覆盖 `tests_v13`、`tests_v14`、`tests_v15`；
4. 当前总结果文档顶部仍记录旧源码基线 `538660...`，而本次审查的仓库提交是 `70178d...`，状态文档和实际提交之间存在需要修复的基线不一致。

---

# 1. 结论先行

## 1.1 当前失败的准确对象

已经正式失败的是：

> **PFCR-v2 relation + bounded evidence-v3 + 既定三分支训练目标 + 既定 Formal-800 协议。**

PFCR 两个 seed 均完整训练，但分别比最强固定比较器少：

```text
seed 42：-3 TP / -3 recovered misses
seed 43：-2 TP / -2 recovered misses
```

严格 \(D_R\)-only 诊断又显示：

```text
factual miss：只有 2/32 具有局部 occupancy 支持
synthetic target：只有 8/206 具有局部 occupancy 支持
```

同时，两个 seed 在 32/32 factual misses 上都形成了超过背景 q99.9
的目标峰值，但只有 24/32、25/32 达到完整
`target-min > background-max`。这些结果支持以下结构诊断：

> 冻结 decoder 可以从当前输入形成目标峰值，但 PFCR 的局部乘性
> relation 在大多数正目标处没有局部 occupancy 条件；同时，严格
> target/background 极值分离仍不完整。

该诊断与正式 \(D_V\) 负结果一致，但不构成唯一因果归因。尤其不能仅由
decoder 输出反推出“冻结 feature 单独已经充分”，也不能把
`target-min <= background-max` 直接等同于某个具体组件级错误。

## 1.2 CSLF 当前不是正式失败，而是未闭合

CSLF 目前已经完成：

- 单一非饱和标量场；
- 固定零水平集；
- occupancy hard exclusion；
- truncated signed-distance target；
- rooted \(W^{1,4}\) 型场能量；
- response-coordinate joint energy 与 identity-coordinate joint energy 公式；
- 19 项核心单元测试；
- 单像素可学习性探索；
- representation-neutral 的真实 \(D_R\) raw population；
- scene-complete absolute target 与 per-target loss focus 的单值契约；
- scalar/phase actual-input、duplicate-target 与 RF observability 审计；
- r1/r2 create-only、逐字节一致的正式证据包；
- \(D_R\) observability 层面的 `scalar_max` representation authorization
  （不包含训练授权）；
- scene-field / loss-focus 双定义域的预计算 target geometry；
- scalar-hidden component-null 的 diagnostic-only 分流；
- 固定 \(4+4+2+2=12\) 状态的一次 forward/一次 backward/一次 update；
- response-joint、identity-joint、separable-endpoint 共初始化、共日程 runner；
- 800×40 确定性 schedule 与 target/source exposure gate 代码；
- 真实 \(D_R\) 上三目标各 2 updates 的 GPU0 代码烟测。

但它尚未完成：

- create-only 的正式 cache/schedule artifact；
- expanded dataset-free gate；
- 正式 \(D_R\)-only bounded 三目标运行；
- fixed-zero calibration contract；
- \(D_R\)-only bounded gate；
- Formal-800；
- 新 \(D_V\) 一次性揭示；
- CSLF 训练与评估的 result/decision/COMPLETE 证据链
  （observability 证据链已完成）。

所以不能把 PFCR 的负结果直接转写成“CSLF 失败”，也不能因为单像素探索成功就写成“CSLF 成功”。

## 1.3 推荐主路线

推荐的下一步不是立即再改一套方程，也不是先把全部时间投入历史凭据整理，
而是让模型开发与仓库闭合并行：

```text
主模型线：
[DONE] 冻结 PFCR 负结果与 CSLF-v15-core
  -> [DONE] 建立 representation-neutral 的 raw D_R population
  -> [DONE] 运行 scalar/phase observability r1/r2
  -> [DECIDED] AUTHORIZE_SCALAR_CSLF；PP NOT_AUTHORIZED
  -> [DONE] 预计算 scalar CSLF cache
  -> [DONE] fused step、确定性 schedule 与三类 matched objectives
  -> [NEXT] create-only cache/schedule receipt 与 dataset-free 扩展门槛
  -> D_R-only bounded 门槛

并行工程线：
39-failure inventory
  -> 历史 snapshot binding
  -> 正式测试矩阵

两线在 Formal authorization 前汇合
  -> 800×40、seed 42/43 开发重复
  -> 封存后一次性读取新 D_V
  -> 若逐 seed 通过，再做冻结后的独立确认
```

其中：

- **CSLF-v15R1**：保持现有核心方程逐字节不变，只补齐模型工程和正式证据链；
- **PP-CSLF-v16**：只在 scalar actual-input target contract 失败、而 phase contract
  通过时启用，唯一核心修改是把 scalar max-projected occupancy 改为
  phase-preserving occupancy basis；
- PP/RF 分支只作为协议或 population 发生变化后的新版本 contingency，不再是
  当前主线的未决分支；
- 不在同一版本中同时改 normalization、occupancy basis、target field、Sobolev 阶数和网络深度。

这里的“可观测性失败”必须进一步分型。PP-CSLF 只处理 scalar projection
造成的信息丢失，不处理 phase representation 下仍超出空间感受域的 target
response。

---

# 2. PFCR 负结果的结构诊断

## 2.1 不是工程失败

PFCR 已经完成：

- 两个 seed 的 800 epochs；
- 每个 seed 32,000 updates；
- create-only artifact；
- 严格加载；
- 一次性 \(D_V\) 揭示；
- Base retention；
- pixel FA、raw-background FA、component FA 预算检查。

因此 PFCR 的负结果不能归因于：

- 训练中断；
- checkpoint/resume；
- 非有限状态；
- 结果伪造；
- Base 被破坏；
- 误报预算直接越界。

## 2.2 局部二值 relation 在多数正目标处无状态支撑

PFCR 的核心作用依赖 target feature cell 周围的 occupancy basis。正式归因显示，大多数 factual/synthetic 正目标的该 basis 为零。

这意味着在这些位置，PFCR 实际退化为：

\[
z \approx \text{baseline}(F_b)+\text{feature evidence}(F_b),
\]

而不是它所声称的 feature–coverage relation。

所以 PFCR 的主要结构性缺口不是“完全没有 target feature”，而是：

> coverage relation 没有稳定参与正目标排序。

## 2.3 冻结 decoder 形成了峰值，但完整极值分离不足

PFCR 的 factual target max 在 32/32 上都超过背景 q99.9，说明冻结 feature 中有可利用信号。

但 `target-min > background-max` 只有：

```text
seed 42：24/32
seed 43：25/32
```

这些结果与以下输出形态一致：

- 目标内部只有局部高峰；
- 目标其他像素不够高；
- 同图极端背景仍较强；
- residual component 更碎；
- component FA 偏高。

因此，下一结构可以把“完整 residual support”作为直接监督对象，而不是只输出
无结构像素 evidence 后依赖阈值截断。CSLF 的水平集方向是针对该诊断缺口提出的
候选解法；在真实 \(D_R\) 和新 \(D_V\) 结果出现前，不能写成已经证明有效的替代方程。
---

# 3. 当前 CSLF 的完整机制

## 3.1 输入与输出

当前模型输入仍为通用接口：

\[
\left(\operatorname{sg}(F_b),O\right),
\]

其中：

- \(F_b\)：冻结 Base detector 的中间特征；
- \(O\)：冻结 Base 的二值 occupancy；
- `sg`：stop-gradient。

模型输出一个原始、非饱和标量场：

\[
\phi_\theta(F_b,O)\in\mathbb R^{H\times W}.
\]

固定语义：

\[
\phi<0
\quad\Longleftrightarrow\quad
\text{residual target support},
\]

\[
\phi\ge0
\quad\Longleftrightarrow\quad
\text{no residual support}.
\]

最终：

\[
Y_{\mathrm{res}}
=
\mathbf1[\phi<0]\land\neg O,
\]

\[
Y_{\mathrm{final}}
=
O\lor Y_{\mathrm{res}}.
\]

没有独立 object head、null head、proposal tree、attention branch、transport solver 或后处理分支。

## 3.2 网络

当前代码执行：

\[
\widehat F
=
\frac{\operatorname{sg}(F_b)}
{\max(\operatorname{RMS}(F_b),10^{-6})},
\]

\[
\widehat O
=
\operatorname{MaxProject}(O),
\]

\[
S=[\widehat F,\widehat O],
\]

\[
H_0=\operatorname{SiLU}(W_{3\times3}S),
\]

\[
H=H_0+\operatorname{SiLU}(W^{dw}_{3\times3}H_0),
\]

\[
\phi
=
\operatorname{PixelShuffle}(W_{1\times1}H+b).
\]

关键属性：

- 无 sigmoid/tanh 饱和；
- 无 normalization layer；
- occupancy 在第一层就与 feature 联合；
- 两个 \(3\times3\) 卷积给出 5×5 feature-grid receptive field；
- phase projection 初始 weight 为 0；
- phase bias 固定初始化为 \(+0.9\)；
- 初始场因此精确为空状态。

参数量公式：

\[
N
=
(C+1)\cdot32\cdot9
+
32\cdot9
+
32s^2+s^2.
\]

在 \(C=64,s=4\) 时：

\[
N=19,536.
\]

它比 PFCR 的 8,705 参数更大，但仍是单一轻量场，不是模块堆叠。

## 3.3 目标水平集

目标 mask \(Y\) 转为截断 chessboard signed-distance field：

\[
\phi^\star(Y)
\in[-0.9,0.9].
\]

属性：

- target pixel 严格负；
- valid background 严格正；
- empty target 为全 \(+0.9\)；
- invalid 区设为 \(+0.9\)，loss measure 排除；
- truncation radius 固定为 feature stride。

当 stride 为 4 时，一像素目标的中心值仅为：

\[
-\frac{0.9}{4}=-0.225.
\]

这是正确的零水平集编码，但其符号裕量较小，后续必须作为正式诊断记录。

## 3.4 Sobolev 风险

自然状态误差：

\[
e=\phi-\phi^\star.
\]

目标、外部距离带、远背景获得等质量 measure。

绝对状态风险同时测量：

\[
\|e\|_{L^4_\mu}
\quad\text{和}\quad
\|\nabla e\|_{L^4_\mu}.
\]

pair 中定义：

\[
e^+=\phi^+-\phi^{\star+},
\]

\[
\Delta\phi=\phi^--\phi^+,
\]

\[
\Delta\phi^\star
=
\phi^{\star-}-\phi^{\star+},
\]

\[
e^\Delta
=
\Delta\phi-\Delta\phi^\star.
\]

response-coordinate joint energy 使用：

\[
(e^+,e^\Delta),
\]

identity-coordinate joint energy 使用：

\[
(e^+,e^-).
\]

二者的精确最优解相同：

\[
e^+=0,\qquad e^-=0.
\]

两者都通过同一个 joint vector \(L^4\) energy 计算，因此 identity-coordinate
版本并不是可分离的 endpoint ERM。它仍包含 endpoint 交叉项，并继续消费由真实
pair 构造的 focus measure。当前 candidate 的准确研究问题只是：

> coverage finite-response 坐标是否在固定预算下提供更好的优化归纳偏置和真实漏检恢复。

为了判断配对关系本身是否必要，还必须增加第三类 decisive comparator：

\[
L_{\mathrm{sep}}
=
\frac12 L_{\mathrm{abs}}(e^+)
+
\frac12 L_{\mathrm{abs}}(e^-),
\]

其中两个 endpoint 分别使用各自的自然状态 target/measure，优化目标不消费
端点配对关系。response-joint、identity-joint 和 separable-endpoint 三者必须
使用相同模型、相同 endpoints、相同 forward 数、相同初始化和相同训练日程。

---

# 4. 当前 CSLF 设计的优点

## 4.1 单一结构化预测对象

empty、single component、multiple components 和形状边界都由同一个场表达。它避免了：

- 先预测分数再做组件提议；
- 独立 null classifier；
- 多 head 协调；
- threshold grid 产生额外自由度。

## 4.2 非饱和 raw field

PFCR 使用概率/logit release 时容易在错误峰值上形成碎片。当前 field 不经过 sigmoid/tanh，空间误差和梯度可以直接作用于场值。

## 4.3 固定零水平集

推理规则不需要 residual threshold 搜索：

\[
\tau_{\mathrm{res}}=0.
\]

这使模型主张更清楚，也减少在 \(D_V\) 上过拟合阈值的自由度。

## 4.4 目标和平滑结构共同监督

离散 \(W^{1,4}\)-type 风险同时约束场值和场的空间变化，因此为完整、平滑的
correction support 提供直接监督。它不自动保证连通性、紧致性或无额外负岛，
这些性质仍须通过拓扑指标和 matched loss 对照验证。

## 4.5 component-null 有局部 focus measure

删除具有 lineage receipt、且未匹配任何 GT 的 Base prediction component 时，
measure 会关注被删区域，而不是让一个局部错误被 \(256^2\) 背景完全稀释。

## 4.6 已有同几何坐标对照，但尚无真正 separable endpoint 对照

identity-coordinate joint control 已经在核心文件中实现，并且使用完全相同的：

- target fields；
- focus support；
- integration measure；
- valid mask。

这是隔离坐标变换 \(T\) 与 identity 坐标的良好基础，但不能据此声称已经隔离
paired learning。正式 runner 还必须实现：

- `separable_endpoint_erm`：不消费 pair relation；
- `pair_shuffle`：保留 endpoint 边缘分布但打乱同源关系，只作机制诊断；
- `feature_only`、`occupancy_only` 与普通 SDF/segmentation loss 对照。

这些属于实验对照，不是向最终模型增加模块。

---

# 5. 当前必须解决的风险

## 5.1 P0：全仓 39 个当前失败尚未归类和闭合

不能使用“CSLF 19 项通过”替代仓库级闭合。

当前观察到的失败集中于历史实现绑定和一个旧 receipt 源码哈希，但在正式
inventory 生成前，不能把 39 项全部预先定性为历史 fingerprint 问题。对确认为
历史绑定的失败，通常有两种可能：

1. 当前测试错误地拿历史 receipt 去校验当前可变源码；
2. 历史绑定文件确实被改写或丢失。

正确处理不是刷新旧 receipt 的 SHA，而是：

- 保存历史 receipt 原文；
- 恢复其对应的历史源码/产物快照；
- 让历史测试验证 snapshot，而不是验证当前 module；
- 若原始字节无法恢复，必须明确记录 `HISTORICAL_ARTIFACT_INVALID`，不能伪造新 hash 让测试变绿。

该闭合是 Formal-800 的硬前提，但不是构建 raw \(D_R\) population、
observability 和 fused-step 单元测试的科学前提。历史闭合与模型代码应并行推进。

## 5.2 P0：默认测试命令不覆盖 CSLF 核心测试

`pyproject.toml` 当前为：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

所以 README 中普通：

```bash
python -m pytest -q
```

不会自动执行 `tests_v15`。

必须新增正式测试矩阵 runner，并由 authorization 绑定其 receipt。

## 5.3 P0：状态文档源码基线不一致

当前总结果文档顶部仍写：

```text
current source baseline = 538660...
```

而当前提交为：

```text
70178d461386...
```

该问题不应通过覆盖历史章节解决。应新增：

```text
CSLF_CURRENT_STATUS.md
cslf_status_receipt.json
```

明确：

- 当前 commit；
- 工作树 dirty manifest；
- PFCR frozen result；
- CSLF core files；
- 当前测试矩阵；
- 未授权阶段。

## 5.4 P1：scalar max projection 丢失 phase-level occupancy

当前模型把：

\[
O\in\{0,1\}^{H\times W}
\]

max-project 为：

\[
\widehat O\in\{0,1\}^{h\times w}.
\]

一个 stride cell 内，只要还有任一 positive pixel，删除另一个 positive pixel 后：

\[
\widehat O^+=\widehat O^-.
\]

此时模型接收到完全相同的输入状态，无法产生 coverage response。

当前测试对此不是修复，而是直接拒绝该 pair：

```text
test_pair_batch_rejects_a_full_grid_change_hidden_by_projection
```

因此必须在真实 population 构建前统计：

```text
full_grid_changed_pair_count
projected_changed_pair_count
hidden_by_projection_pair_count
hidden_by_projection_target_pixels
```

不能通过悄悄丢弃大量 pair 来宣称模型利用了完整 occupancy。

## 5.5 P1：pair target response 必须分别审计 scalar-RF 与 phase-RF

模型中 occupancy 到输出的 receptive radius 为 2 个 feature cells。

对于每个 pair，必须分别检查 scalar representation 和 phase representation：

\[
\operatorname{supp}(\Delta\phi^\star)
\subseteq
\operatorname{RF}^{\mathrm{scalar}}_2
\quad\text{或}\quad
\operatorname{RF}^{\mathrm{phase}}_2.
\]

其中：

- scalar changed cells 来自 max-projected occupancy 的差异；
- phase changed cells 来自 PixelUnshuffle 后任一 phase channel 的差异；
- 两者都按实际网络的 feature-grid radius 2 扩张，再映射回原输出 phase。

若 \(\Delta\phi^\star\) 只超出 scalar-RF、但位于 phase-RF 内，则 PP-CSLF 可以
修复该可观测性缺口。若它仍超出 phase-RF，则 PP 也不可满足，必须重构
state/target 或另立扩大 RF 的版本。

当前 19 项测试只验证一个小型 pair，不足以证明真实 206 个 synthetic states 均满足此条件。

## 5.6 P1：empty-state 小负岛仍有分辨率稀释

当前测试本身证明：

```text
64×64 单负岛 loss
/
256×256 单负岛 loss
≈ 2
```

这符合均匀 \(L^4\) 风险的面积缩放。它比 mean BCE 稀释更慢，但不是分辨率不变。

所以三 seed 单像素探索没有额外负岛是积极信号，但不能替代：

- 多 clutter feature；
- 多目标；
- 边界；
- 多负岛；
- component-null；
- 256² 空图 stress population。

在修改 objective 前，先把“额外负岛数”设为正式门禁。如果 Sobolev 不能稳定满足，再建立独立新版本加入 zero-level worst-excursion guard。

## 5.7 P1：phase projection 零初始化导致首 step 上游无梯度

当前：

```python
phase_projection.weight = 0
phase_projection.bias = +0.9
```

首个 backward 中：

- phase weight/bias 有梯度；
- hidden 的梯度乘以零 phase weight；
- `input_projection` 和 `spatial_mixing` 首 step 梯度为零。

这不是 bug，因为第二 step 起 phase weight 非零后可恢复。梯度审计必须区分：

- required trainable parameter 的 `grad is None`：计算图接线错误；
- finite、但逐元素全零的 gradient tensor：单个 update 上合法；
- `NaN/Inf` gradient：任何 update 上都非法。

正式门禁只约束初始化后的早期 gradient latency：

```text
phase_projection.weight/bias：
    first_nonzero_gradient_update = 0
input_projection.weight/spatial_mixing.weight：
    first_nonzero_gradient_update <= 2
```

某个参数在通过上述早期 latency 门禁后，于后续任意单步再次出现 finite zero
gradient，不构成失败，也不得据此重启或改变 schedule。

不要为了首 step 非零梯度而破坏精确空状态初始化，除非该 latency gate 失败。

## 5.8 P1：global RMS 需要真实分布审计

全样本 RMS 保留同图相对峰值，这是当前设计的明确目的。但它也意味着：

- 一个极强 clutter peak 会缩放整图；
- 极小 RMS 样本受 epsilon 分支影响；
- occupancy 取值为 0/1，而 feature 的局部幅度分布可能跨样本差异很大。

正式 cache 必须记录：

```text
sample_rms min / q01 / median / q99 / max
target-cell normalized feature RMS
background q99.9 normalized RMS
epsilon-clamped sample count
```

在没有真实异常证据前，不应同时改 normalization 和 occupancy basis。

## 5.9 P1：固定目标场对一像素目标的 margin 较小

stride 4 时，一像素 target 的目标场中心为 \(-0.225\)。

因此正式评估必须同时报告：

- zero-level binary correctness；
- target minimum field；
- background minimum positive field；
- target/background sign margin；
- 对小目标尺寸分层的 margin。

不能只看 Sobolev loss。

## 5.10 P1：response-joint 与 identity-joint 的最优解相同

response-joint 坐标：

\[
(e^+,e^--e^+)
\]

相对 identity-joint：

\[
(e^+,e^-)
\]

是可逆线性变换。对单像素坐标，其变换矩阵条件数约为：

\[
2.618.
\]

因此：

- response-joint 不增加表达能力；
- response-joint 不改变精确解集合；
- 它只改变优化几何；
- 必须和 identity-joint、真正 separable endpoint ERM 在相同模型、相同初始化、相同 batch schedule、相同计算预算下比较。

此外，当前 identity-joint 并非 independent endpoint ERM：

\[
\left(
\frac{(e^+)^2+(e^-)^2}{2}
\right)^2
=
\frac14
\left[
(e^+)^4+2(e^+)^2(e^-)^2+(e^-)^4
\right].
\]

若 response-joint 不能稳定优于 identity-joint 和 separable-endpoint，则
“finite-response coordinate”不应成为创新主张。

## 5.11 P1：CSLF 不应执行 residual threshold calibration

CSLF 的定义就是：

\[
\phi<0.
\]

若在 \(D_V\) 上搜索：

\[
\phi<\tau,
\]

则模型不再是固定零水平集版本。

因此需要实现“校准入口”，但其职责应是：

```text
验证 Base threshold 已冻结
验证 CSLF threshold 固定为 0
记录 no_search=true
生成 calibration-contract receipt
```

而不是 grid search。

---

# 6. 推荐版本关系

## 6.1 保留当前核心

当前文件保持为：

```text
CSLF-v15-core
```

不得原地把 projection、loss 或 target policy 改成另一语义后继续沿用同一 fingerprint。

## 6.2 建立工程闭合版本

新版本：

```text
CSLF-v15R1
```

含义：

- 核心 decoder 和 Sobolev 方程不变；
- 新增正式 population/cache；
- 新增 fused step；
- 新增三类 matched objective runner；
- 新增可观测性审计；
- 新增评估与 artifact；
- 不读取新 \(D_V\)。

## 6.3 条件式结构版本

若真实可观测性 receipt 满足：

```text
outside_phase_rf_pixels = 0
phase_duplicate_input_target_conflicts = 0
并且满足以下至少一项：
  scalar_duplicate_input_target_conflicts > 0
  outside_scalar_rf_pixels > 0
```

则当前 scalar-projection CSLF 不得训练，建立：

> **PP-CSLF-v16：Phase-Preserving Coverage-State Level Set**

若：

```text
outside_phase_rf_pixels > 0
```

则 PP-CSLF 同样不授权。此时只能：

- 修正错误的 state/target contract；
- 或建立只改变感受域的独立 RF 版本。

PP-CSLF 的唯一核心变化是 occupancy state basis。

不同时修改：

- global RMS；
- hidden width；
- field amplitude；
- truncation radius；
- Sobolev \(p=4\)；
- depth；
- output rule。

---

# 7. P0：并行的仓库级闭合修改

## 7.1 生成 39-failure inventory

新增：

```text
tools/collect_repository_failure_inventory.py
```

输出：

```text
protocols/IRSTD-1K/cslf_v15/
  repository_failure_inventory.json
```

每个失败必须记录：

```json
{
  "node_id": "...",
  "test_file": "...",
  "failure_class": "historical_source_binding",
  "expected_sha256": "...",
  "actual_sha256": "...",
  "referenced_artifact": "...",
  "disposition": "restore_snapshot"
}
```

不得把 39 个失败整体描述为“无关”，也不得在 inventory 前把它们全部标成
同一 failure class。该工作与第 8–11 节的模型代码并行，在正式授权前汇合。

## 7.2 建立历史 snapshot verifier

新增：

```text
cure_lite/historical_integrity.py
tools/verify_historical_integrity.py
protocols/history/source_snapshots/
protocols/history/historical_binding_manifest.json
```

manifest 记录：

```json
{
  "schema_version": "cure-lite-historical-binding.v1",
  "entries": [
    {
      "historical_version": "v5",
      "repository_commit": "...",
      "path": "source_snapshots/v5/paired_loss.py",
      "sha256": "...",
      "bound_receipt": "..."
    }
  ]
}
```

旧测试改为：

```text
receipt -> historical snapshot
```

而不是：

```text
receipt -> current cure_lite module
```

## 7.3 旧 paired receipt 哈希不匹配

处理顺序：

1. 从 receipt 中读取原预期 SHA；
2. 查找对应 Git blob 或 sealed artifact；
3. 若可恢复，恢复精确字节到历史 snapshot；
4. 若路径语义变化，创建 locator amendment；
5. locator amendment 只能改变“去哪里找旧字节”，不能改变旧 SHA；
6. 若旧字节无法恢复，正式记录历史 artifact invalid，不能重算一个新 SHA 冒充旧结果。

## 7.4 正式测试矩阵 runner

新增：

```text
tools/run_cslf_repository_test_matrix.py
```

由于部分历史 runner 要求独立进程，不建议简单把所有目录写进 `pytest.testpaths`。

正式 runner 应：

- 每个需要隔离的文件/目录使用新 Python 进程；
- 收集 stdout、JUnit 和 exit code；
- 记录 Python/Torch/CUDA；
- 记录测试文件 SHA；
- 拒绝 skip/xfail；
- 生成单一 matrix receipt。

授权门槛：

```text
failed = 0
error = 0
skipped = 0
unclassified_historical_failure = 0
```

---

# 8. 正式 population 与 cache

## 8.1 数据边界

新 CSLF 只允许读取：

```text
冻结 Base 的 D_R cache
```

不得读取：

```text
新 D_V
D_T
其他 detector
其他数据集
```

## 8.2 新数据结构：先建立 representation-neutral raw catalog

raw catalog 必须在当前 scalar-specific `CoverageStatePairBatch.validate()`
之前建立。它只验证 source、lineage、full-grid occupancy、target 和 valid domain，
不得因为 scalar projection 不可见而删除 pair。

新增：

```python
@dataclass(frozen=True)
class CoverageStateNaturalRecord:
    record_id: str
    sample_id: str
    state_kind: Literal["factual_miss", "factual_no_miss"]
    feature_ref: str
    occupancy_ref: str
    target_ref: str
    valid_mask_ref: str
    target_ids: tuple[str, ...]

@dataclass(frozen=True)
class CoverageStatePairRecord:
    pair_id: str
    sample_id: str
    pair_kind: Literal[
        "clean_positive",
        "component_null",
        "identity_null",
    ]
    feature_ref: str
    occupancy_plus_ref: str
    occupancy_minus_ref: str
    target_plus_ref: str
    target_minus_ref: str
    valid_mask_ref: str
    removed_component_ids: tuple[str, ...]
    target_ids_added: tuple[str, ...]
```

在 raw record 之上新增表示感知审计结果，而不是把 scalar/phase 条件写死在
lineage record：

```python
@dataclass(frozen=True)
class CoverageStateRepresentationAudit:
    pair_id: str
    representation: Literal["scalar_max", "phase_preserving"]
    encoded_feature_sha256: str
    occupancy_plus_sha256: str
    occupancy_minus_sha256: str
    input_plus_sha256: str
    input_minus_sha256: str
    target_plus_sha256: str
    target_minus_sha256: str
    changed_feature_cells: tuple[int, ...]
    target_response_pixels: int
    target_response_outside_rf_pixels: int
    duplicate_input_target_conflict: bool
```

`sample_id`、`pair_id`、endpoint role、target 和 valid mask 不得进入 actual-input
fingerprint；否则跨记录的相同模型输入会被元数据伪装成不同输入。target/valid
必须单独绑定并用于 population-level conflict 分组。

## 8.3 状态语义

### Factual miss

同一 source 的绝对 completion field 必须是 scene-complete 单值目标：

\[
Y_{\mathrm{scene}}(F_b,O)
=
\bigcup_{g\in\mathcal M_{\mathrm{eligible}}}
\left(GT_g\land\neg O\right).
\]

若一个图像含多个 eligible factual misses，仍为每个 \(g\) 保留一条独立
focus/exposure record，但这些 records 的 actual input、\(Y_{\mathrm{scene}}\)
和 field-valid domain 完全相同；只允许 integration measure / loss focus
随 \(g\) 改变。即：

\[
\phi^\star_g=\phi^\star(Y_{\mathrm{scene}}),\qquad
\mu_g\neq\mu_{g'}\ \text{可以成立},
\]

但禁止：

\[
\phi^\star_g\neq\phi^\star_{g'}
\quad\text{同时}\quad
(F_b,O)_g=(F_b,O)_{g'}.
\]

此前把单个 focus target 同时当作绝对场目标，会让同一模型输入对应多个
target fields。observability 首次重放因此发现 7 个 scalar/phase 共同冲突
key；scene-complete target contract 将它们全部消除。

### Factual no-miss

目标场为空：

\[
Y_{\mathrm{completion}}=\varnothing.
\]

### Clean positive

同一 source、同一 feature：

\[
O^-=O^+\setminus C_g,
\]

\[
Y^- = Y^+\cup GT_g.
\]

必须证明：

- component lineage 唯一；
- 只新增目标 \(g\)；
- 不破坏其他 target；
- plus/minus feature 完全相同；
- field valid domain 与 image-valid domain 一致。

target response 是否对 occupancy change 可达属于 observability gate，不得写入
raw population inclusion 条件。

### Component null

删除无关或假阳性 component：

\[
Y^-=Y^+.
\]

### Identity null

\[
O^-=O^+,\qquad Y^-=Y^+.
\]

用于验证 deterministic exact equality。

## 8.4 几何预计算

cache 构建时一次性保存：

- target field；
- integration measure；
- focus support；
- focus support field；
- pair target response；
- projected occupancy；
- phase occupancy diagnostics；
- occupancy receptive support；
- zero-level target metrics。

训练 step 不得每次重新计算 distance transform 或 state catalog。

## 8.5 Cache fingerprint

至少绑定：

```text
source Base cache fingerprint
Base threshold
feature tensor name/channel/stride
CSLF target policy
scene-complete target policy
per-target focus-measure policy
CSLF objective policy
population row identities
pair lineage
precomputed tensor SHA256
observability receipt
```

---

# 9. 训练前的状态可观测性门禁

新增：

```text
cure_lite/coverage_state_observability.py
tools/audit_coverage_state_observability.py
```

## 9.1 Phase grid

```python
def occupancy_to_phase_grid(
    occupancy: torch.Tensor,
    *,
    stride: int,
) -> torch.Tensor:
    if occupancy.dtype is not torch.bool:
        raise TypeError("occupancy must be bool")
    if occupancy.shape[-2] % stride or occupancy.shape[-1] % stride:
        raise ValueError("occupancy size must be divisible by stride")
    phase = torch.nn.functional.pixel_unshuffle(
        occupancy.to(torch.float32),
        stride,
    )
    return phase.to(torch.bool).contiguous()
```

PyTorch 的 PixelUnshuffle 与 PixelShuffle 为相反的重排，可把一个输出 cell 内的 \(s^2\) 个 occupancy phase 精确保留为 channel。

所有 duplicate-input 判断必须基于模型实际输入：

\[
\left[
\widehat F,\widehat O
\right]
\quad\text{或}\quad
\left[
\widehat F,U_s(O)
\right],
\]

其中 \(\widehat F\) 是应用 samplewise global RMS 后的 FP32 tensor。不得只比较
raw feature 引用，因为两个不同 raw tensors 可能在归一化后映射为相同输入。

## 9.2 必算指标

```text
full_grid_changed_pairs
phase_changed_pairs
scalar_projected_changed_pairs
hidden_by_scalar_projection_pairs
hidden_target_pixels
clean_positive_hidden_pairs
component_null_hidden_pairs
target_response_pixels
target_response_outside_scalar_rf_pixels
target_response_outside_phase_rf_pixels
target_response_hidden_only_by_scalar_pixels
identity_null_nonidentical_count
scalar_duplicate_input_target_conflicts
phase_duplicate_input_target_conflicts
```

## 9.3 表示选择与授权门槛

```text
identity_null_nonidentical_count = 0
phase_duplicate_input_target_conflicts = 0
target_response_outside_phase_rf_pixels = 0
```

决策顺序固定为：

```text
若 phase_duplicate_input_target_conflicts > 0：
    STATE_TARGET_CONTRACT_UNREALIZABLE
    停止 scalar 与 PP

若 target_response_outside_phase_rf_pixels > 0：
    PHASE_RF_UNREACHABLE
    停止 scalar 与 PP
    重构 state/target 或另立 RF 版本

若 phase 可达，且 scalar_duplicate_input_target_conflicts > 0
或 target_response_outside_scalar_rf_pixels > 0：
    授权 PP-CSLF-v16

若 phase 可达，且 scalar target conflict = 0
且 target_response_outside_scalar_rf_pixels = 0：
    授权 CSLF-v15R1
```

仅有 scalar-hidden 的 `component_null` 且其 target response 为 0，不构成
target contract 失败，不能单独授权 PP-CSLF。它应作为表示诊断保留，但模型
选择必须由 actual-input target conflict 或非零 target response 的 RF 缺口触发。

不得通过删除失败 rows 后继续把 population 称为完整真实 population。只有
lineage 或 full-grid state contract 本身预先判定为非法的 row，才能以明确
reason code 排除。

## 9.4 已完成的正式 \(D_R\) 结果

create-only r1/r2 已完成，两个目录中的 `raw_catalog.json`、
`observability.json`、`decision.json`、`config.json` 与 `COMPLETE.json`
全部逐字节一致。

| 指标 | 正式结果 |
|---|---:|
| natural records | 167 |
| clean-positive pairs | 206 |
| component-null pairs | 17 |
| identity-null pairs | 160 |
| full-grid changed pairs | 223 |
| phase changed pairs | 223 |
| scalar changed pairs | 222 |
| scalar-hidden pairs | 1 |
| scalar-hidden clean pairs | 0 |
| scalar-hidden component-null pairs | 1 |
| target response pixels | 19,722 |
| response outside scalar-RF | 0 |
| response outside phase-RF | 0 |
| scalar duplicate-input target conflicts | 0 |
| phase duplicate-input target conflicts | 0 |
| identity-null nonidentical | 0 |

唯一 scalar-hidden pair 为 `XDU792 / component_null / pred_id=2`，其 target
response 为 0。它证明 phase representation 更完整，但不构成 scalar target
contract 失败，因此不触发 PP。

正式决策：

```text
decision = AUTHORIZE_SCALAR_CSLF
selected_representation = scalar_max
PP-CSLF-v16 = NOT_AUTHORIZED / NOT_NEEDED
training_authorized = false
next_route = build_scalar_cslf_cache_and_fused_step
```

关键指纹：

```text
raw_catalog_fingerprint
= 15eaf6d5482d908f2c0f9899e4495eaff329ccb4d5c2aae648c4bba98ef79a24

observability_receipt_fingerprint
= 0c603611582534ca686fb177ac5786db4fee3f0c03f434b994062312e97ec214

complete_fingerprint
= 9baaf7b08959ac47d25ed8d725fdf2ffd58289685cd4cd453d9833eef7fbe7b3
```

---

# 10. 条件结构修改：PP-CSLF-v16

仅当 phase actual-input 无 target conflict、phase-RF 完整可达，且 scalar
存在 target conflict 或非零 response 的 RF 缺口时，新增独立文件：

```text
cure_lite/phase_preserving_coverage_state_level_set.py
```

不要覆盖 `coverage_state_level_set.py`。

## 10.1 方程

\[
U_s(O)=\operatorname{PixelUnshuffle}_s(O)
\in\{0,1\}^{s^2\times h\times w}.
\]

状态：

\[
S=[\widehat F,U_s(O)].
\]

其余保持：

\[
H_0=\operatorname{SiLU}(W_{3\times3}S),
\]

\[
H=H_0+\operatorname{SiLU}(W^{dw}_{3\times3}H_0),
\]

\[
\phi
=
\operatorname{PixelShuffle}(W_{1\times1}H+b).
\]

仍然只有一个场。

## 10.2 代码骨架

以下骨架只展示结构差异。正式实现必须与 scalar CSLF 共享同等级的 config、
FP32、shape、device、finite、参数量和输出规则检查，不能把下述最小骨架直接当作
完成实现。

```python
class CURELitePhasePreservingCSLF(nn.Module):
    def __init__(self, config: PhasePreservingCSLFConfig) -> None:
        super().__init__()
        self.config = config
        phase_channels = config.feature_stride ** 2
        self.input_projection = nn.Conv2d(
            config.feature_channels + phase_channels,
            config.width,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.spatial_mixing = nn.Conv2d(
            config.width,
            config.width,
            kernel_size=3,
            padding=1,
            groups=config.width,
            bias=False,
        )
        self.phase_projection = nn.Conv2d(
            config.width,
            phase_channels,
            kernel_size=1,
            bias=True,
        )
        self.pixel_shuffle = nn.PixelShuffle(config.feature_stride)
        self._reset_parameters()

        actual = sum(p.numel() for p in self.parameters())
        if actual != config.expected_parameter_count:
            raise AssertionError("PP-CSLF parameter count changed")

    def _reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.input_projection.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.kaiming_normal_(
            self.spatial_mixing.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        nn.init.zeros_(self.phase_projection.weight)
        nn.init.constant_(
            self.phase_projection.bias,
            self.config.initial_field_value,
        )

    def forward(self, feature: Tensor, occupancy: Tensor) -> Tensor:
        self._validate_inputs(feature, occupancy)
        frozen = feature.detach()
        rms = frozen.square().mean((1, 2, 3), keepdim=True).sqrt()
        encoded = frozen / rms.clamp_min(1.0e-6)
        phase_occupancy = F.pixel_unshuffle(
            occupancy.to(encoded.dtype),
            self.config.feature_stride,
        )
        hidden = F.silu(
            self.input_projection(
                torch.cat((encoded, phase_occupancy), dim=1)
            )
        )
        hidden = hidden + F.silu(self.spatial_mixing(hidden))
        field = self.pixel_shuffle(self.phase_projection(hidden))
        self._validate_field(field, occupancy)
        return field

    def predict_completion(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        return ((self(feature, occupancy) < 0.0) & ~occupancy).contiguous()

    def predict_union(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> Tensor:
        return (occupancy | self.predict_completion(
            feature,
            occupancy,
        )).contiguous()
```

还必须新建 representation-aware pair validator。不得继续调用当前
scalar-specific validator 去拒绝 PP 正要处理的 phase-visible pair。

## 10.3 参数量

\[
N_{\mathrm{PP}}
=
(C+s^2)\cdot32\cdot9
+
32\cdot9
+
32s^2+s^2.
\]

在 \(C=64,s=4\)：

```text
current scalar CSLF：19,536
PP-CSLF：23,856
增加：4,320
```

仍然没有新增 head 或分支。

## 10.4 必须证明

- `PixelShuffle(PixelUnshuffle(O)) == O`；
- 同一 feature 下，任何 full-grid occupancy change 都改变模型输入；
- 当前 scalar model 无法区分、PP model 可区分的 toy pair；
- hard union 不变；
- target/loss/objective 不变；
- 参数增加只来自 occupancy basis。
- 初始 field 逐像素精确为 \(+0.9\)；
- `feature/model/geometry` 全程 FP32，autocast 关闭；
- phase-visible、scalar-hidden pair 能通过新的 representation-aware validator。

---

# 11. 单次 fused training step

新增：

```text
cure_lite/train/coverage_state_fused_step.py
```

## 11.1 固定 batch

沿用 12-state 预算：

```text
factual miss：4 states
factual no-miss：4 states
clean-positive：1 pair × 2 endpoints
component-null：1 pair × 2 endpoints
合计：12 states
```

identity-null 不进入优化，只在每个正式检查点执行 exact-equality 诊断。原因是同输入
时 response-joint 使用 \((e,0)\)，identity-joint 使用 \((e,e)\)，若把
identity-null 加入训练，会给两类 objective 不同的重复端点权重。

## 11.2 一次 forward

```python
def coverage_state_fused_step(
    model,
    optimizer,
    factual_miss,
    factual_no_miss,
    pair_batch,
    *,
    pair_objective: Literal[
        "response_joint",
        "identity_joint",
        "separable_endpoint",
    ],
):
    feature = torch.cat(
        (
            factual_miss.feature,
            factual_no_miss.feature,
            pair_batch.feature,
            pair_batch.feature,
        ),
        dim=0,
    )
    occupancy = torch.cat(
        (
            factual_miss.occupancy,
            factual_no_miss.occupancy,
            pair_batch.occupancy_plus,
            pair_batch.occupancy_minus,
        ),
        dim=0,
    )

    field = model(feature, occupancy)  # exactly one model call

    n_miss = factual_miss.batch_size
    n_no = factual_no_miss.batch_size
    n_pair = pair_batch.batch_size

    phi_miss, phi_no, phi_plus, phi_minus = torch.split(
        field,
        (n_miss, n_no, n_pair, n_pair),
        dim=0,
    )

    miss_loss = coverage_state_absolute_sobolev_loss_from_targets(
        phi_miss,
        factual_miss.targets,
        config=factual_miss.sobolev_config,
    ).loss

    no_loss = coverage_state_absolute_sobolev_loss_from_targets(
        phi_no,
        factual_no_miss.targets,
        config=factual_no_miss.sobolev_config,
    ).loss

    if pair_objective == "response_joint":
        pair_loss = coverage_state_pair_sobolev_loss_from_targets(
            phi_plus,
            phi_minus,
            pair_batch.targets,
            config=pair_batch.sobolev_config,
        ).loss
    elif pair_objective == "identity_joint":
        pair_loss = coverage_state_independent_endpoint_loss_from_targets(
            phi_plus,
            phi_minus,
            pair_batch.targets,
            config=pair_batch.sobolev_config,
        ).loss
    else:
        plus_loss = coverage_state_absolute_sobolev_loss_from_targets(
            phi_plus,
            pair_batch.absolute_targets_plus,
            config=pair_batch.sobolev_config,
        ).loss
        minus_loss = coverage_state_absolute_sobolev_loss_from_targets(
            phi_minus,
            pair_batch.absolute_targets_minus,
            config=pair_batch.sobolev_config,
        ).loss
        pair_loss = 0.5 * (plus_loss + minus_loss)

    total = miss_loss + no_loss + pair_loss

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    audit_gradients(...)
    optimizer.step()
    audit_parameters_buffers_and_optimizer_state(...)
    return ...
```

`separable_endpoint` 的 plus/minus geometry 必须分别使用：

```text
field_valid = image_valid
loss_valid_plus = image_valid & ~occupancy_plus
loss_valid_minus = image_valid & ~occupancy_minus
focus_plus = target_plus
focus_minus = target_minus
```

它不能复用 pair-derived focus measure，否则仍然消费 pair relation。

## 11.3 结构门禁

每 update：

```text
model forward calls = 1
decoder states = 12
backward calls = 1
optimizer steps = 1
pair feature identity preserved
pair endpoint order fixed
pair kinds = one clean-positive + one component-null
clean-positive source != component-null source
identity-null training count = 0
autocast enabled = false
all field/target/measure dtype = float32
fresh optimizer per objective/seed/run = true
pre-update finite audit = 1 per run
post-step finite audit = 1 per update
```

其中 12 个 state 的角色组成必须逐 update 精确保持为
`4 factual-miss + 4 factual-no-miss + 1 clean pair × 2 endpoints
+ 1 component-null pair × 2 endpoints`，不能只在整段运行的平均意义上满足。

总 Formal-800：

```text
32,000 forwards / model
384,000 states / model
32,000 backwards / model
32,000 optimizer steps / model
32,001 finite-state audits / model
```

## 11.4 上游梯度 latency

日志必须记录每个参数张量的：

```text
first_nonzero_gradient_update
```

冻结的早期 latency 门禁：

```text
phase_projection.weight：update 0
phase_projection.bias：update 0
input_projection.weight：不晚于 update 2
spatial_mixing.weight：不晚于 update 2
```

该门禁不要求每个 update 的 gradient 都非零。required trainable parameter 的
finite zero gradient tensor 在任意单步均合法；只有 `grad is None`、非有限梯度，
或从未在上述 deadline 前出现非零梯度才失败。

## 11.5 当前实现与真实 \(D_R\) 代码烟测

当前已新增：

```text
cure_lite/coverage_state_precomputed_cache.py
cure_lite/coverage_state_batches.py
cure_lite/coverage_state_schedule.py
cure_lite/train/coverage_state_fused_step.py
cure_lite/experiment/coverage_state_training.py
```

真实 \(D_R\) cache 集成结果：

```text
natural = 167
pairs = 383
clean-positive optimization-eligible = 206
component-null total = 17
component-null optimization-eligible = 16
component-null diagnostic-only = 1
identity-null diagnostic = 160
scalar cache fingerprint
= 569b0fb97d819cf1281ca1d148227bc1c5e229b8301065cb536656b5e578e645
```

GPU0 上使用同一初始化和同一两步 schedule 完成三个 objective 的代码烟测：

| objective | updates | forwards | logical states | trunk 首次非零梯度 |
|---|---:|---:|---:|---:|
| response-joint | 2 | 2 | 24 | update 1 |
| identity-joint | 2 | 2 | 24 | update 1 |
| separable-endpoint | 2 | 2 | 24 | update 1 |

三者的共同初始模型指纹为：

```text
25b467cae0c23a4ce55a5d1153a85a4345999acb9cbd4ab498a970978423aaf1
```

schedule 指纹为：

```text
8c64722d5440d9e86cde93d6faaa93b6d19be80642c3689ee316c512874b5f6a
```

matched smoke result 指纹为：

```text
c0646c6142121c4b1f7f32c80ab4104ea51d76c3a54e76a8d71a4819a950f351
```

这只证明真实张量、GPU、梯度传播和三目标公平 runner 已接通。2 updates 不构成
收敛、机制或检测性能证据，`training_authorized` 与
`formal_training_authorized` 仍为 false。

---

# 12. 三类 matched objective runner

新增：

```text
cure_lite/experiment/coverage_state_real_training.py
cure_lite/experiment/coverage_state_real_runner.py
cure_lite/experiment/coverage_state_real_artifacts.py
tools/train_coverage_state_real.py
```

## 12.1 三类 objective

| 项目 | Response-joint candidate | Identity-joint control | Separable-endpoint ERM |
|---|---|---|---|
| decoder | 相同 | 相同 | 相同 |
| initial state dict | 相同 | 相同 | 相同 |
| feature/occupancy endpoints | 相同 | 相同 | 相同 |
| natural targets | 相同 | 相同 | 相同 |
| schedule | 相同 | 相同 | 相同 |
| optimizer config | 相同 | 相同 | 相同 |
| update/forward/state budget | 相同 | 相同 | 相同 |
| pair objective | \((e^+,e^- - e^+)\) joint norm | \((e^+,e^-)\) joint norm | \(\frac12L(e^+)+\frac12L(e^-)\) |
| pair-derived focus measure | 使用 | 使用 | 不使用 |
| 是否消费 endpoint pairing | 是 | 是 | 否 |

identity-joint 只回答“response 坐标是否优于 identity 坐标”；separable-endpoint
才回答“消费 pair relation 是否优于独立 endpoint ERM”。另做 pair-shuffle
诊断，以检查增益是否依赖正确的同源 pairing。

这里的“公平”不是要求三个 objective 使用完全相同的坐标表达和 integration
measure。以下差异是预先声明的处理变量，因此允许且必须保留：

- response-joint 使用 \((e^+,e^- - e^+)\)；
- identity-joint 使用 \((e^+,e^-)\)；
- separable-endpoint 使用两个自然 endpoint risk，并分别使用预声明的
  endpoint-specific natural measure，不得消费 pair-derived focus measure。

公平性要求的是：除上述 pair coordinate 与相应预声明 measure 之外，模型、
初始参数、输入 endpoints、角色 schedule、optimizer 配置、update/forward/state
预算和数值策略全部相同。不得把“measure 必须逐张量相同”误写成公平条件，
否则 separable control 会被迫消费它本应排除的 pair relation。

## 12.2 初始化公平性

每个 seed 先生成一次初始模型：

```text
initial_state_dict_seed42.safetensors
initial_state_dict_seed43.safetensors
```

三类 objective 都从相同文件 create-only 加载。

每个 `(seed, objective, run)` 必须在加载同一初始模型后创建一个 fresh
optimizer。不得从另一个 objective 或先前尝试加载、复制或继续使用 optimizer
state；三个 optimizer 的超参数和实现必须相同。

runner 在 update 0 前核对：

```text
response-joint initial fingerprint
==
identity-joint initial fingerprint
==
separable-endpoint initial fingerprint
```

## 12.3 Schedule 公平性

提前生成：

```text
formal_schedule_seed42.json
formal_schedule_seed43.json
```

每个 update 固定：

- factual miss IDs；
- no-miss IDs；
- pair IDs；
- endpoint order；
- branch batch sizes。

三类 objective 共享同一 schedule SHA。每步 schedule 还必须固定：

```text
4 factual-miss IDs
4 factual-no-miss IDs
1 clean-positive pair ID
1 component-null pair ID
identity-null IDs = none
clean-positive source != component-null source
logical states = 4 + 4 + 2 + 2 = 12
```

`clean-positive source != component-null source` 必须对每一个 update 成立，并由
冻结 raw catalog 中的 canonical logical source ID 判定。该规则是三个
objective 共同使用的公平/去耦约束：避免同一 update 的正 response pair 与
null-response pair 共享 source-specific 背景状态。它不是独立性证明、性能主张，
也不得根据后续 exposure 或 \(D_V\) 结果开关。

同时模拟完整 32,000-step 日程并报告：

- target exposure 与 source-image exposure；
- target/source ESS；
- 最大单 target/source 暴露比例；
- top-k concentration；
- 零暴露率。

### 12.3.1 读取新 real exposure 前冻结的分角色门禁

以下规则必须先写入 config、生成指纹并封存，之后才允许生成或读取任何新的
real schedule exposure receipt。不得根据真实 exposure 结果修改阈值。

对任一被门禁的 eligible universe \(\mathcal I_r\)，令
\[
p_i=\frac{c_i}{\sum_{j\in\mathcal I_r}c_j},\qquad
\operatorname{ESS}_r=\frac{1}{\sum_i p_i^2},\qquad
N_r=|\mathcal I_r|.
\]

record-level 必须按角色分别计算，不能合并：

```text
factual-miss record
factual-no-miss record
clean-positive pair record
component-null pair record
```

每个 record role 均必须满足：

\[
\operatorname{ESS}_{\mathrm{record},r}\geq0.90N_{\mathrm{record},r},
\qquad
\max_i p_i\leq\frac{2}{N_{\mathrm{record},r}},
\qquad
\mathrm{zero\ exposure}_{\mathrm{record},r}=0.
\]

positive-target exposure 必须拆成两个独立门禁：

```text
factual-focus target：来自 factual-miss record 的冻结 focus target
clean-added target：来自 clean-positive pair 的冻结 added target
```

两类 target 分别满足：

\[
\operatorname{ESS}_{\mathrm{target},r}\geq0.90N_{\mathrm{target},r},
\qquad
\max_i p_i\leq\frac{2}{N_{\mathrm{target},r}},
\qquad
\mathrm{zero\ exposure}_{\mathrm{target},r}=0.
\]

不能用 factual-focus 的充分暴露抵消 clean-added 的集中，也不能反向抵消。
将两者合并得到的 `combined_positive_target` ESS、最大份额或 top-k concentration
只能作为描述性统计，不参与 authorization。

source exposure 也必须按上述四个 record role 分别计算。一次 factual record
选择记该 role 的 logical source 一次；一次 pair 选择记对应 pair role 的
logical source 一次，不能因为有两个 endpoint 而重复计为两次。每个 source
role 均必须满足：

\[
\operatorname{ESS}_{\mathrm{source},r}\geq0.50N_{\mathrm{source},r},
\qquad
\max_i p_i\leq\frac{4}{N_{\mathrm{source},r}},
\qquad
\mathrm{zero\ exposure}_{\mathrm{source},r}=0.
\]

跨角色汇总的 `global_logical_source` exposure 仅作描述，因为 4/4/1/1 的固定
角色预算会机械改变全局 source 权重；它不得覆盖任一分角色 source gate，也
不得参与 authorization。

完整 32,000-step 模拟还必须逐 update 断言：

```text
factual-miss records = 4
factual-no-miss records = 4
clean-positive pairs = 1
component-null pairs = 1
clean source != component-null source
decoder states = 12
identity-null optimization exposure = 0
diagnostic-only component-null optimization exposure = 0
```

## 12.4 不允许的差异

禁止：

- 为 response-joint 单独调 learning rate；
- 为任一 control 单独调 batch；
- 为 response-joint 增加 update；
- 使用不同 negative population；
- response-joint 使用 PP-CSLF、control 使用 scalar CSLF；
- response-joint 有 zero-level guard、control 没有；
- 在两个 runner 中使用不同 target field cache。
- 将 identity-null 放入任一 objective 的优化 batch；
- 只为某一 objective 开启 AMP、gradient clipping 或 weight decay。
- 在 objectives 之间复用 optimizer state；
- 读取 real exposure 后调整 ESS、最大份额、零暴露或不同-source 约束。

允许的 objective 差异严格限于第 12.1 节预声明的 pair coordinates，以及由
该语义决定的 pair-derived 或 endpoint-specific integration measure。它们是
受检验的处理变量，不是 schedule/compute unfairness。

---

# 13. Dataset-free 扩展门禁

当前 19 项核心测试和单像素探索应升级为正式 population，而不是继续作为临时脚本。

## 13.1 至少包含

```text
1px target
3px compact target
two disconnected targets
target at image edge
target near invalid barrier
empty state
single false negative island
multiple false islands
component-null deletion
identity-null
same feature cell multiple occupancy phases
full-grid deletion hidden by scalar projection
clean pair with natural miss already present
clutter feature peak without target
low-RMS feature
high dynamic-range feature
```

分辨率至少：

```text
64²
256²
```

seed：

```text
42 / 43 / 44
```

## 13.2 门禁

```text
all fields finite
target zero-level exact
empty target negative-component count = 0
component-null added negative components = 0
identity-null field difference = 0
hard union exact
all required parameter tensors meet the frozen early first-nonzero latency gate
chosen representation has no hidden legal pair
target response is fully inside chosen phase/scalar RF
response-joint, identity-joint and separable-endpoint all computationally learnable
```

梯度项只检查第 11.4 节的早期 `first_nonzero_gradient_update` latency。通过
latency 后，任意单步 finite zero gradient 合法，不要求每步所有参数非零。

这仍不是 IRSTD 性能结果。

---

# 14. \(D_R\)-only bounded 门禁

正式 32,000-update 之前，使用冻结 \(D_R\) 构建一个新的 bounded population。

推荐沿用历史规模：

```text
16 factual-miss anchors
16 factual-no-miss anchors
16 clean-positive pairs
16 component-null pairs
16 identity-null pairs（仅诊断，不进入优化）
```

但必须由新的 CSLF state contract 重建和重新封存。

## 14.1 运行

```text
response-joint：400 updates
identity-joint：400 updates
separable-endpoint：400 updates
seed：42
D_V read：false
```

## 14.2 共同工程门禁

```text
400/400 updates
400 fused forwards
4,800 states
401 finite-state audits
fresh optimizer created after identical model load
no retry
no resume
single terminal
```

`401 finite-state audits` 精确定义为：update 0 前一次，加上 400 次 optimizer
step 后各一次，即对任意 \(U\)-update run 固定为 \(U+1\) 次。每个 objective
必须使用 fresh optimizer，禁止复用另一个 objective 或先前 attempt 的
optimizer state。

## 14.3 机制门禁

建议冻结以下公共评估：

### Factual miss

```text
target negative-pixel fraction >= 0.95
target-level at least one negative pixel = 16/16
target connected-support recall reported
```

### Factual no-miss

```text
negative component count = 0
negative pixel count = 0
```

### Clean positive

```text
minus target response has correct sign
plus endpoint does not create writable target island
target-response support is compact
```

在读取任何真实 trained checkpoint 前，冻结 clean compact policy 为 `clean_added_target_zero_level_exact_no_spill_v1`：在 valid/writable 域内，clean pair 的 zero-level 新增 completion 必须与 added target 逐像素完全相等，并同时满足 minus added target 全负、plus 不产生 writable 假岛、`new_completion_outside_added=0`、`new_completion_pixels=added_target_pixels`，且二者的 8 连通组件一致；阈值固定为 0，不执行阈值搜索。

### Component null

```text
new negative components after deletion = 0
new negative pixels in removed footprint = 0
```

### Identity null

```text
max |phi_minus - phi_plus| = 0
```

### 三类 objective

bounded 阶段不要求 response-joint 已经优于 controls 的真实 Pd，但要求：

- 三者都可稳定训练；
- 无 objective-specific execution bug；
- 三者使用完全相同 population、endpoints 和 schedule；
- 三者只允许 pair coordinate 与预声明 integration measure 不同；
- identity-null 只诊断 exact equality，训练 exposure 为 0；
- pair-shuffle 诊断不改变 endpoint 边缘曝光。

任何失败都不得进入 Formal-800。

---

# 15. Zero-level evaluation 与“校准”入口

新增：

```text
cure_lite/experiment/coverage_state_zero_level_evaluation.py
cure_lite/experiment/coverage_state_calibration_contract.py
```

## 15.1 Calibration contract

固定输出：

```json
{
  "base_threshold_source": "frozen_cache_manifest",
  "residual_threshold": 0.0,
  "residual_threshold_search_performed": false,
  "selection_data_read": false,
  "output_rule": "field_lt_zero_then_occupancy_exclusion"
}
```

## 15.2 连续诊断

使用：

\[
s=-\phi
\]

只作诊断，不选择阈值。

记录：

```text
target max score
target min score
background q99.9 score
background max score
target-min minus background-max
negative component count
component size distribution
component fragmentation
completion IoU
target-level recovery
```

## 15.3 正式 binary 输出

唯一输出：

```python
completion = (field < 0.0) & ~occupancy
final = occupancy | completion
```

---

# 16. Formal-800 与新 \(D_V\) 揭示

## 16.1 Formal training

只有以下全部通过才授权：

```text
repository closure
implementation closure
population/cache closure
observability gate
dataset-free gate
D_R bounded gate
three-objective fairness gate
```

运行六个主任务：

```text
response-joint seed 42
identity-joint seed 42
separable-endpoint seed 42
response-joint seed 43
identity-joint seed 43
separable-endpoint seed 43
```

每个：

```text
800 epochs
40 updates/epoch
32,000 updates
fresh empty Adam
shared frozen learning rate / betas / eps / weight decay
32,001 finite-state audits = one before update 0 + one after every update
autocast disabled
gradient clipping policy frozen and identical
no checkpoint
no resume
no automatic retry
```

## 16.2 揭示顺序

```text
六个 checkpoint 全部 sealed
  -> external result verifier 全部通过
  -> 创建 D_V reveal authorization
  -> D_V 只物化一次
  -> 固定 zero-level evaluation
  -> decision recomputation
  -> COMPLETE-last
```

## 16.3 性能门禁

建议保留旧项目的逐 seed 严格标准：

\[
TP_{\mathrm{response}}
\ge
TP_{\mathrm{best\ fixed\ comparator}}+2,
\]

\[
R_{\mathrm{response}}
\ge
R_{\mathrm{best\ fixed\ comparator}}+2.
\]

并增加 response-joint 对两个 matched controls 的必要性。对每个 seed 分别要求：

\[
TP_{\mathrm{response}}
>
\max
\left(
TP_{\mathrm{identity}},
TP_{\mathrm{separable}}
\right),
\]

\[
R_{\mathrm{response}}
>
\max
\left(
R_{\mathrm{identity}},
R_{\mathrm{separable}}
\right).
\]

不能用一个 seed 的强提升抵消另一个 seed 的下降。若两 seed 均通过，只说明该
候选获得冻结确认资格，不等于完成独立复现。

同时必须报告并冻结：

- IoU 与 nIoU；
- Pd–FA 曲线上的固定 operating point；
- completion IoU 和 target support completeness；
- 参数量、MACs、峰值显存和推理延迟；
- response-only / identity-only / separable-only 找回目标表。

所有成功规则必须在读取新 \(D_V\) 前冻结，不能事后选择。

## 16.4 误报与保留约束门禁

继续保留：

```text
covered-target retention = 1.0
pixel FA <= 1e-4
raw-background FA <= 1e-4
FP components / MP <= 100
```

新增 CSLF topology ledger：

```text
negative components / image
recovered-target component fragmentation
target support completeness
background extreme margin
```

## 16.5 冻结后的独立确认

seed 42/43 是同一数据划分、同一 Reference Base 和同一 cache 下的 paired
decoder-seed development repeats。两者均通过后，仍须使用预先冻结的额外 seed
或独立划分完成确认，才可把 CURE-Lite 记为设计成功。该确认完成前：

```text
lite_frozen_confirmation = false
full_CURE_authorized = false
cross_backbone_authorized = false
```

---

# 17. 如果 Sobolev 出现额外负岛

不要在 CSLF-v15R1 中临时加 loss。

建立独立：

> **ZM-CSLF-v17：Zero-Level Margin CSLF**

固定最小目标场边界 margin：

\[
m_0=\frac{0.9}{s}.
\]

定义：

\[
V_T
=
\max_{x\in Y}
[\phi(x)+m_0]_+,
\]

\[
V_B
=
\max_{x\in V\setminus Y}
[m_0-\phi(x)]_+.
\]

\[
L_{\mathrm{zero}}
=
\sqrt{V_T^2+V_B^2}.
\]

再与 Sobolev 风险以固定同量纲方式组合。

该分支只在以下失败签名出现时授权：

```text
Sobolev loss下降
target peak存在
但 empty/no-miss/comp-null 仍出现额外 negative islands
```

不能在未观察到该失败前先把 worst-case loss加入当前候选。

---

# 18. 如果 scalar projection 隐藏、但 phase-RF 完整可达

启用 PP-CSLF-v16，而不是：

- 增加 attention；
- 增加第二 decoder；
- 增加 component proposal；
- 改 Base；
- 添加 GT lineage input。

PP-CSLF 仍只消费 detector-accessible：

\[
(F_b,O).
\]

它只是避免在 decoder 内部把完整 \(O\) 过早压成一个 scalar cell bit。

如果 target response 仍超出 phase-RF，本节不适用，PP 不授权。

---

# 19. 如果 response-joint 不优于两个 matched controls

如果：

```text
三类 objective 都训练稳定
response-joint 与 identity-joint / separable-endpoint 的 topology 和性能相同
或任一 control 更好
```

则结论应是：

> 同源 finite-response 坐标在当前单场、数据和预算下没有建立额外机制价值；
> 若 separable-endpoint 同样达到最强结果，则消费 endpoint pairing 的必要性也
> 没有成立。

此时不能只改 loss 权重后重跑同一版本。

可选下一研究问题只有两个：

1. 停止 paired-coordinate 创新主张，把模型定位为普通 frozen-feature level-set completion；
2. 建立一个新的、真正改变 coverage-state representation 的候选。

如果 feature-only 或 separable-endpoint 已达到最强结果，继续增加 paired
机制的科学价值有限。pair-shuffle 若与正确 pairing 相同，也应停止“同源关系”
主张。

---

# 20. 如果 CSLF 仍出现“峰值正确、支持不完整”

失败签名：

```text
target max > background q99.9
但 target min <= background max
目标内部只有部分 phi<0
component fragmentation 偏高
```

下一版本应只修改 target geometry，而不是改网络和 occupancy 同时变化。

候选方向：

> **Adaptive-Radius CSLF**

但 radius 不能从 \(D_V\) 搜索。可从冻结 Base stride 和 \(D_R\) target diameter distribution 预先解析：

\[
r
=
\operatorname{clip}
\left(
\operatorname{median}_{D_R}
(\text{target chessboard radius}),
r_{\min},r_{\max}
\right).
\]

该方向只有在 fixed-radius support incompleteness 被 \(D_R\)-only 证据确认后才授权。

---

# 21. 文件级修改清单

## 21.1 不修改

```text
cure_lite/coverage_state_level_set.py
cure_lite/coverage_state_sobolev.py
PFCR 正式 result/decision/COMPLETE
历史 v3–v7 receipts
已读取的旧 D_V artifacts
```

先冻结它们作为基线。

## 21.2 新增：仓库闭合

```text
cure_lite/historical_integrity.py
tools/collect_repository_failure_inventory.py
tools/verify_historical_integrity.py
tools/run_cslf_repository_test_matrix.py
protocols/history/historical_binding_manifest.json
protocols/history/source_snapshots/...
```

## 21.3 新增：CSLF 数据

```text
cure_lite/coverage_state_real_states.py
cure_lite/coverage_state_raw_catalog.py
cure_lite/coverage_state_real_cache.py
cure_lite/coverage_state_population.py
cure_lite/coverage_state_observability.py
cure_lite/coverage_state_representation.py
cure_lite/coverage_state_batches.py
tools/build_coverage_state_real_cache.py
tools/audit_coverage_state_observability.py
```

## 21.4 新增：训练

```text
cure_lite/train/coverage_state_fused_step.py
cure_lite/experiment/coverage_state_real_training.py
cure_lite/experiment/coverage_state_real_runner.py
tools/train_coverage_state_real.py
```

## 21.5 新增：评估与产物

```text
cure_lite/experiment/coverage_state_calibration_contract.py
cure_lite/experiment/coverage_state_zero_level_evaluation.py
cure_lite/experiment/coverage_state_formal_decision.py
cure_lite/experiment/coverage_state_artifacts.py
cure_lite/experiment/coverage_state_d_v_reveal.py
tools/run_coverage_state_d_v_reveal.py
tools/verify_coverage_state_result.py
```

## 21.6 条件新增

```text
cure_lite/phase_preserving_coverage_state_level_set.py
tests_v16/test_phase_preserving_coverage_state_level_set.py
```

只在 `phase actual-input/RF PASS` 且 scalar target contract 失败后创建。

---

# 22. 新测试矩阵

## 22.1 Core

1. phase projection 初始场精确 \(+0.9\)；
2. first-step upstream gradient 为 0；
3. update 1/2 上游 gradient 恢复，后续单步 finite zero gradient 合法；
4. all-zero feature 有限；
5. RMS below epsilon 有限；
6. positive global scaling invariance（仅限 RMS 未触发 epsilon clamp）；
7. relative peak ratio 保留；
8. hard union 精确；
9. output field 无 sigmoid/tanh；
10. 参数量和参数张量数固定。

## 22.2 Geometry

1. signed-distance empty；
2. 1px/3px/large target；
3. invalid barrier；
4. multiple components；
5. edge target；
6. phase roundtrip；
7. hidden scalar projection；
8. target response within scalar-RF / phase-RF 分别计算；
9. normalized actual-input duplicate target conflict；
10. target field cache与直接重算一致。

## 22.3 Objective

1. exact target zero loss；
2. empty false island gradient；
3. multi-island scaling；
4. component-null focus；
5. identity-null exact；
6. response-joint missing response gradient；
7. identity-joint exact control 与交叉项公式；
8. separable-endpoint ERM 不消费 pair focus/配对关系；
9. finite p4；
10. precomputed/direct exact equality。
11. 三类 objective 的精确最优解一致；
12. pair-shuffle 保持 endpoint 边缘曝光。

## 22.4 Fused step

1. exactly one model call；
2. exactly 12 states；
3. one backward；
4. one step；
5. fresh optimizer，且 finite audits 精确等于 updates + 1；
6. pair feature duplicate identity；
7. split order exact；
8. 三类 objective 只有预声明的 criterion/measure 差异；
9. initial fingerprint identical；
10. schedule fingerprint identical；
11. 每步 pair kinds 固定为 clean-positive + component-null；
12. 每步 clean-positive source != component-null source；
13. identity-null training exposure = 0；
14. exact 4/4/1/1 roles and 12 states per update；
15. FP32/no-AMP contract。

## 22.5 Runner/artifact

沿用 PFCR r2 中已经证明有效的：

- strict schema；
- create-only；
- exact-once；
- COMPLETE-last；
- file SHA inventory；
- raw decision recomputation；
- no self-signed incomplete object；
- no retry/resume；
- D_V import isolation；
- source implementation closure。

## 22.6 Historical closure

39 个当前失败必须全部有：

- node-id；
- failure class；
- fixed evidence；
- no receipt mutation；
- passing verification。

---

# 23. 授权清单

创建 CSLF bounded development-run authorization 前：

```text
[ ] PFCR 正式负结果未修改
[ ] 新 D_V 未读取
[ ] 当前 commit 与 status receipt 一致
[ ] CSLF implementation closure 生成
[x] D_R cache source fingerprint 固定
[x] representation-neutral raw population/catalog fingerprint 固定
[x] scalar/phase observability receipt 完整且 r1/r2 逐字节一致
[x] scalar target-response RF gate 通过
[x] scalar/phase duplicate-input conflict = 0
[x] target/field-valid/loss-valid/occupancy lineage 全部通过
[x] fused step 结构测试通过
[x] response/identity/separable initial state 相同
[x] response/identity/separable schedule 相同
[x] identity-null training exposure = 0
[x] diagnostic-only component-null training exposure = 0
[x] truncation_radius = feature_stride
[x] FP32/no-AMP contract 通过
[ ] exposure-gate config 在读取新 real exposure 前封存
[ ] exact 4/4/1/1 roles、12 states/update 与 clean/component 不同 source
[ ] factual-focus 与 clean-added target 分角色 exposure gate 分别通过
[ ] 四类 record/source exposure gate 分别通过
[ ] fixed zero-level calibration contract 生成
[ ] 结果目录尚不存在
```

创建 Formal-800 authorization 前再增加：

```text
[ ] 39 个全仓失败全部归类并闭合
[ ] 历史 source snapshots 可复核
[ ] 正式测试矩阵 0 failed/error/skipped
[ ] dataset-free expanded population PASS
[ ] D_R bounded response-joint PASS
[ ] D_R bounded identity-joint PASS
[ ] D_R bounded separable-endpoint PASS
[ ] 每个 objective 使用 fresh optimizer
[ ] 每个 \(U\)-update run 的 finite audits 精确为 \(U+1\)
[ ] no extra negative islands
[ ] no component-null response
[ ] identity-null exact
[ ] no D_V import
```

---

# 24. 研究进度重新估算

按“完整 CURE-Lite 模型与分阶段验证闭环”计，当前约 **55%**。

这表示核心场、真实 raw population、scene-complete target contract、表示选择、
预计算训练几何、fused step、确定性 schedule 和三目标公平 runner 已接通；
不表示 bounded training 已通过，更不表示性能主张成立。此前 45% 是
observability 完成但训练代码尚未闭合时的估计，现已过时。

按本计划：

| 阶段 | 累计完成度估计 |
|---|---:|
| 当前核心场、真实 raw population、target contract 与正式 observability | 45% |
| 预计算 cache + fused step + schedule + 三 objective runner | 55% |
| dataset-free 正式门禁 | 60% |
| \(D_R\)-only bounded 三 objective | 65%–70% |
| 六个 Formal-800 artifact sealed | 80%–85% |
| 新 \(D_V\) 一次性开发结果与 decision | 90%–95% |
| 两 seed 全部 PASS | 获得冻结确认资格，不等于 100% |
| 额外 seed 或独立划分冻结确认 PASS | 100%（仅指 CURE-Lite） |

Full CURE、其他 detector 和三数据集不计入上述 Lite 百分比。

---

# 25. 最终研究判断

## 当前方案是否值得继续

**有条件地值得。**

CSLF 针对 PFCR 的两个诊断缺口提出了一个连贯候选：

1. 不再依赖多数正目标处为空的局部乘性 relation；
2. 把完整 residual support 作为带零水平集和空间能量的直接监督对象。

这说明设计动机一致，不等于已经证明 CSLF 可以修复 PFCR 的性能失败。

## 当前能否直接跑 800×40

**不能。**

当前缺失的不是普通脚本，而是决定结果能否被解释的关键层：

- create-only cache/schedule artifact；
- dataset-free expanded gate；
- \(D_R\)-only bounded 三-objective gate；
- fixed-zero evaluation；
- 训练与评估 artifact/decision closure；
- 全仓 39 个当前失败的归类与闭合。

## 是否现在就应修改核心方程

**当前不应修改。**

正式 observability 已经完成，结果是：

- scalar/phase duplicate-input target conflict 均为 0；
- scalar/phase RF 外的 target-response pixel 均为 0；
- 唯一 scalar-hidden 状态是零 target-response 的 component-null；
- 因此正式结果仅授权继续实现 scalar CSLF-v15R1 的
  representation/cache/training pipeline；`training_authorized=false`，尚未授权
  bounded、Formal-800 或性能主张。PP-CSLF-v16 在当前封存
  population/protocol 下 `NOT_AUTHORIZED / NOT_NEEDED`。

下一步必须保持核心方程与标量 occupancy basis 不变，进入正式 cache/schedule
封存、expanded dataset-free gate 和 \(D_R\)-only bounded 三目标验证。只有后续
预先冻结的门禁失败，才按对应失败类型另立版本：

- 额外负岛门禁失败：另建 ZM-CSLF-v17；
- response-joint 不优于 identity-joint 与 separable-endpoint：停止 finite-response 机制主张。

## 下一步最小可执行任务

```text
主模型线：
1. [DONE] 建立 representation-neutral raw D_R population
2. [DONE] 运行 scalar/phase observability 与 RF audit
3. [DONE] 正式选择 scalar CSLF-v15R1
4. [DONE] 为 scalar CSLF 预计算训练几何 cache
5. [DONE] 实现 12-state one-forward fused step
6. [DONE] 实现确定性 schedule 与三目标公平 runner
7. [NEXT] 封存 create-only cache/schedule receipt
8. 通过 expanded dataset-free 与 D_R-only bounded 门槛

并行工程线：
9. 生成 39-failure inventory
10. 修复历史 snapshot binding 和测试矩阵

两线汇合：
11. 创建 Formal-800 authorization
```

这条路线既能让当前 CSLF 真正进入下一阶段，也能确保后续失败具有明确归因，而不是继续在方程、数据、训练和工程同时变化的情况下试错。

## 25.1 2026-07-27 bounded-400 实际结果更新

上述“下一步最小可执行任务”中的 cache、schedule、expanded dataset-free、
设备常驻训练路径和单次真实 \(D_R\) bounded-400 均已完成。正式产物见：

[CURE-Lite CSLF-v15：bounded-400 正式结果](./CURE_Lite_CSLF_v15_bounded400正式结果.md)

结果不是执行失败，而是冻结零水平门禁失败：

```text
decision = BOUNDED_CSLF_GATE_FAIL
failed_top_level_check = zero_level_gates
formal_800_authorized = false
D_V_accessed = false
D_T_accessed = false
```

三个 objective 均完成 400 次更新。response-joint 的关键签名为：

```text
clean-positive response sign = 1396 / 1396
clean-positive added-target negative pixels = 6 / 149
clean-positive compact exact = 0 / 16
factual-miss gate = 15 / 16
factual-no-miss gate = 16 / 16
component-null gate = 17 / 17
identity-null gate = 16 / 16
```

因此当前授权的修改对象不是 occupancy phase、训练时长或检测器接口，而是：

> 在保持同一 coverage-state residual field 的前提下，使已学到的正确相对响应
> 获得充分的绝对零水平锚定，同时继续保持 factual-no-miss 与 null states。

这次签名不等同于第 17 节所定义的“额外负岛”签名，也不能直接授权在那里预写的
ZM-CSLF-v17；它也没有显示 scalar projection 隐藏必要状态，因此不授权
PP-CSLF-v16。下一版本必须先把“锚定不足”写成单一数学修改和 dataset-free
判别实验，不能把 margin、radius、phase representation 和新网络块一起加入。

更新后的主线为：

```text
[DONE] scalar CSLF-v15R1 implementation
[DONE] expanded dataset-free
[DONE/FAIL] D_R bounded-400
[CURRENT] single-change absolute zero-level anchoring revision
[BLOCKED BY GATE] Formal-800
[NOT STARTED] Full CURE / cross-backbone / three-dataset validation
```

---

# 26. ICLR 新颖性边界

当前不能把以下内容单独写成创新：

- level-set/SDF 输出；
- 空间梯度或离散 \(W^{1,p}\)-type loss；
- frozen feature 后接轻量 correction head；
- model-agnostic coarse-mask refinement；
- PixelShuffle/PixelUnshuffle；
- 一般性的 paired/counterfactual supervision。

最窄且仍待验证的创新余量是：

> 在相同冻结 detector evidence 上，只干预 detector coverage state，显式监督该
> 干预引起的 residual completion field response，并以 completion-only hard
> constraint 限制模型只能在 Base 未覆盖区域增加目标。

该主张必须同时通过：

1. response-joint 逐 seed 优于 identity-joint 和 separable-endpoint；
2. 正确 pairing 优于 pair-shuffle；
3. feature-only、occupancy-only 与错误 source-state 对照；
4. 普通 BCE/Dice/SDF residual refiner 的同容量同预算对照；
5. 固定 Pd/FA、IoU/nIoU、拓扑和运行开销证据。

相关工作边界：

- Deep Level Sets，CVPR 2017；
- LevelSet R-CNN，ECCV 2020；
- Learning What Makes a Difference from Counterfactual Examples and Gradient Supervision，ECCV 2020；
- SegRefiner，NeurIPS 2023；
- Efficient Mask Correction，CVPR 2023；
- Counterfactual Segmentation Reasoning，CVPR Findings 2026；
- Seeing Through the Noise，CVPR 2026。

PP-CSLF 是 phase 信息保真的使能表示，不是第二个标题级创新点。

---

# 27. 主要源码与证据

## 仓库快照

- Commit `70178d461386052d4bf7b0b66c0258b3a187b9c7`  
  https://github.com/Arialliy/cure-lite/commit/70178d461386052d4bf7b0b66c0258b3a187b9c7

## CSLF 核心

- Coverage-state level-set  
  https://github.com/Arialliy/cure-lite/blob/70178d461386052d4bf7b0b66c0258b3a187b9c7/cure_lite/coverage_state_level_set.py
- Coverage-state Sobolev objective  
  https://github.com/Arialliy/cure-lite/blob/70178d461386052d4bf7b0b66c0258b3a187b9c7/cure_lite/coverage_state_sobolev.py
- CSLF core tests  
  https://github.com/Arialliy/cure-lite/blob/70178d461386052d4bf7b0b66c0258b3a187b9c7/tests_v15/test_coverage_state_level_set.py

## PFCR 正式结果

- PFCR 真实训练正式结果  
  https://github.com/Arialliy/cure-lite/blob/70178d461386052d4bf7b0b66c0258b3a187b9c7/CURE_Lite_PFCR_%E7%9C%9F%E5%AE%9E%E8%AE%AD%E7%BB%83%E6%AD%A3%E5%BC%8F%E7%BB%93%E6%9E%9C.md
- 全部结果与当前研究结论  
  https://github.com/Arialliy/cure-lite/blob/70178d461386052d4bf7b0b66c0258b3a187b9c7/CURE_Lite_%E5%85%A8%E9%83%A8%E7%BB%93%E6%9E%9C%E4%B8%8E%E5%BD%93%E5%89%8D%E7%A0%94%E7%A9%B6%E7%BB%93%E8%AE%BA.md

## 工程配置

- `pyproject.toml`  
  https://github.com/Arialliy/cure-lite/blob/70178d461386052d4bf7b0b66c0258b3a187b9c7/pyproject.toml
- PyTorch PixelUnshuffle  
  https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelUnshuffle.html
- PyTorch PixelShuffle  
  https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
