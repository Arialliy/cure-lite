# CURE-Lite 全部结果与当前研究结论

> 更新日期：2026-07-27
> 仓库：`/home/md0/ly/cure_lite`
> 当前源码基线：`538660aa7e200bd7acad8964af25121ea56142cf`；NLCC-v12 runner/evidence r2 与正式结果位于当前工作树
> 最新软件验证收据：PFCR 归因、真实 pipeline、artifact、decoder 与 training
> 联合回归 `40 passed`。更早阶段的软件收据保留在历史章节中。
> 最新结果范围：PFCR-v2 + bounded evidence-v3 已完成真实 \(D_R\) 的
> seed 42/43 Formal-800、一次冻结的正式 \(D_V\) 揭示，以及严格
> \(D_R\)-only 失败归因。
> 两个 seed 均通过 Base 保留与误报预算，但均未超过最强固定比较器，正式状态为
> `PFCR_D_V_GATE_FAIL`。\(D_T\) 未读取；Full CURE、三数据集和其他 detector
> 均未获授权。

> **2026-07-27 PFCR 正式 \(D_V\) 状态（优先于下方全部历史阶段）：**
> seed 42 为 151/170、找回 4/23，相对最强比较器为 \(-3/-3\)；seed 43
> 为 150/170、找回 3/23，相对最强比较器为 \(-2/-2\)。两个 seed 的
> retention 均为 1，三项误报约束均通过，选择阈值均为 0.08。完整训练、评估、
> 指纹与判定见
> [CURE-Lite PFCR 真实训练正式结果](CURE_Lite_PFCR_真实训练正式结果.md)。

## 最新追加：PFCR Formal-800、正式 \(D_V\) 揭示与 \(D_R\) 归因

| 项目 | 权威结果 | 当前决定 |
| --- | --- | --- |
| Formal-800 seed 42/43 | 每个 seed 800 epochs、32,000 updates、384,000 states | COMPLETE / COMPLETE |
| 正式揭示完整性 | create-only；COMPLETE-last；严格加载与独立审计通过 | PASS |
| seed 42 | PFCR 151/170、4/23；最佳比较器 154/170、7/23 | \(-3/-3\)，FAIL |
| seed 43 | PFCR 150/170、3/23；最佳比较器 152/170、5/23 | \(-2/-2\)，FAIL |
| 保留率 | 两个 seed 均 147/147 = 1.0 | PASS |
| 误报预算 | 两个 seed 的 pixel FA、raw-background FA、components/MP 均在冻结预算内 | PASS |
| 正式决定 | `PFCR_D_V_GATE_FAIL` | 冻结负结果 |
| \(D_R\)-only attribution r2 | factual 2/32、synthetic 8/206 具有局部 occupancy 支持 | 当前 relation 在绝大多数正目标处不参与 |
| residual peak / full-support | 两个 seed 均为 factual 32/32 peak；完整极值分离为 24/32、25/32 | 稳定产生峰值，但支持不够紧致完整 |
| 后续授权 | confirmation / Full CURE / cross-backbone 均为 false | 继续 CURE-Lite 结构设计 |
| 数据边界 | \(D_V\) 已按冻结协议读取一次；\(D_T\) 未读取 | 不得在 \(D_V\) 上调参 |

这次结果把“PFCR 可学习、可完成真实训练”与“PFCR 具有真实检测增益”明确分开：
前者已经成立，后者没有成立。seed 42 与 U 同为 151/4，但 PFCR 产生更多误报
组件；seed 43 只达到 Base@B 的 150/3，同样产生更多组件。因此当前候选的首要
问题不是 Base 被破坏或预算越界，而是释放证据没有形成稳定优于固定比较器的目标
排序和空间聚集。后续 r2 已严格重放两个冻结 decoder 在 \(D_R\) 的逐状态响应：
30/32 factual miss 和 198/206 synthetic target 的局部 occupancy basis 为零；
与此同时，两个 seed 在 32/32 factual miss 上都形成超过背景 q99.9 的目标峰，
但只有 24/32、25/32 达到完整 target-min 超过 background-max。该证据定位了
“relation 大多不参与、feature 峰值不够紧致完整”的结构性缺口，但仍不构成唯一
因果归因。权威 r2 result fingerprint 为
`e23cf378d28468d884e8a221e9537e4e23e293943fd5087e1f3eb2ebd3af45f8`。

## 历史追加：PFCR-v2 relation-controlled CURE-Lite

| 项目 | 正式结果 | 当前决定 |
| --- | ---: | --- |
| R13-1A exact / abs-value quotient | 0 / 0 conflicts | 单独位置或符号不是首个区分因素 |
| R13-1A signed / unsigned role quotient | 8 / 7 conflict keys；46 / 64 records | 旧输入依赖样本专属数值，继续冻结 |
| PFCR input contract | 32 states；25,088 records；0 conflicts | PASS |
| matched same-geometry relevance | 8 / 8 groups | PASS |
| analytic completion | 0 mismatch pixels | PASS |
| learned seed 42 | mismatch 0；positive min 0.994816601；negative max 0.009262450 | PASS |
| learned seed 43 | mismatch 0；positive min 0.998534203；negative max 0.007652447 | PASS |
| PFCR 参数量（固定预检配置） | 1,089 | 轻量 |
| v14 tests | 27 passed | PASS |
| v13 + v14 tests | 39 passed | PASS |
| 当时的真实数据性能 | NOT RUN | 该历史状态已被本文顶部 Formal-800 与正式 \(D_V\) 结果覆盖 |
| Full CURE | OUT OF CURRENT SCOPE | 当前持久范围只完成 CURE-Lite |

> **2026-07-27 历史补充：** 本文主体保留历史阶段完整记录；NLCC-v12 的
> dataset-free Development 已完整执行并正式 FAIL。该结论只覆盖更早的
> CCFR-v11 阶段，当前最新结论仍以本文顶部 PFCR 正式 \(D_V\) 揭示为准。完整明细见
> [CURE-Lite NLCC-v12 Development 正式负结果](CURE_Lite_NLCC_v12_Development正式负结果.md)。

## 历史追加：NLCC-v12 dataset-free Development

| 项目 | 权威结果 | 当前决定 |
| --- | --- | --- |
| R0 runner/evidence r2 | targeted 43/43、repository 1105/1105；R0-C1～C8 全部通过 | Development 执行资格成立 |
| Development authorization | `authorized=true`；attempt ordinal = 1 | 只允许一次正式启动 |
| Development 执行 | 320/320 updates；960/960 forwards；321/321 finite-state audits | 执行与训练结构完整 |
| 结构门禁 | 25/25 passed | PASS |
| 数值门禁 | 26/76 passed；50 failed | FAIL |
| final groups | 0/8 passed | FAIL |
| population objective | `1.286966323852539`，冻结要求 `<0.1` | FAIL |
| factual-miss target | `0.28760308027267456`，冻结要求 `>0.95` | FAIL |
| factual-miss background | `0.9996875524520874`，冻结要求 `<0.05` | FAIL |
| factual-no-miss | `0.7426276206970215`，冻结要求 `<0.05` | FAIL |
| 正式 decision | `NLCC_V12_DEVELOPMENT_FAIL` | 冻结 v12 负结果，不重跑 |
| 后续授权 | Holdout / \(D_R\) / formal800 / Full CURE 均为 false | 停在 CURE-Lite 模型设计阶段 |

从重新读取的 `result.json` 原始字段独立重算后，完整 gate ledger 与 sealed
`decision.json` 逐值一致。六个适用 clean-D group 的变化方向均正确，且
8/8 normalized-tail 门禁通过；但 paired 响应幅度、绝对端点、matched null、
背景和 factual anchors 没有同时成立。因此本结果否定的是固定
NLCC-v12 + PECO-v10 候选，不支持将失败唯一归因于某一个算子或 loss 项。

正式目录只有 `attempt.json`、`training_started.json`、`result.json`、
`decision.json` 和 `COMPLETE.json`；`result.json` 是唯一 terminal，无
`failure.json` 或 `.incomplete`。运行没有读取数据集、\(D_R\)、\(D_V\) 或
\(D_T\)，所以这不是检测性能结果。

按冻结停止规则：

```text
NLCC_V12_DEVELOPMENT_FAIL
  -> STOP_AND_PRESERVE_EVIDENCE
  -> no automatic retry
  -> no Holdout
  -> no real D_R
  -> no exposure replay or formal800
  -> no Full CURE or cross-detector work
```

## 历史追加：CCFR-v11 dataset-free holdout

| 项目 | 权威结果 | 当前决定 |
| --- | --- | --- |
| development regression | PASS | 获得一次 dataset-free holdout 资格 |
| holdout 执行 | 400/400 updates；1200/1200 forwards；2400/2400 有限非零梯度 | 训练实现与执行契约通过 |
| final groups | 0/8 passed | FAIL |
| population objective | `1.6601788997650146`，冻结要求 `<0.1` | FAIL |
| factual miss target | `0.9955233335494995`，冻结要求 `>0.95` | PASS |
| factual miss background | `0.9990542531013489`，冻结要求 `<0.05` | FAIL |
| factual no-miss | `0.5038354992866516`，冻结要求 `<0.05` | FAIL |
| 正式 decision | `CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_FAIL` | 冻结负结果，不重跑 |
| 真实数据授权 | `real_D_R_authorized=false` | 不运行 IRSTD-1K `D_R` |
| 后续授权 | formal800 / Full CURE / cross-detector 均为 false | 停在 CURE-Lite 模型设计阶段 |

该结果只否定当前 CCFR-v11 状态方程，不否定 CURE 总研究主线。准确的最新
顺序仍是：新 CURE-Lite 核心候选先通过同等级 dataset-free 门禁，再进入真实
`D_R`、32,000-step 暴露重放和 seed 42/43 的 800-epoch 确认；CURE-Lite
冻结前不设计 Full CURE，不开展三数据集或跨 detector 验证。

## 1. 一页结论

| 项目 | 当前结果 | 正式判断 |
| --- | --- | --- |
| Reference Base | 完成 800 epoch；best epoch = 314 | 仅作为冻结证据提供者，不是 CURE 创新点 |
| CURE-Lite v0.1：U | seed 42 不及 F；seed 43 与 F 持平 | 均匀 legal-target selection 未获机制支持 |
| CURE-Lite v0.2：M | 两个 seed 均低于 U 和 F，漏检找回也更少 | 标量硬最近邻 selector 被否定；运行本身完整有效 |
| v0.2 映射分布 | 32 个 factual miss 只落到 16/209 个 legal targets；ESS = 9.4815 | 存在严重集中，不能继续使用硬最近邻路线 |
| P0-v1 | 244 个 native targets 变为 242 个 evaluation targets；存在消失、合并、分裂 | 原 synthetic legal pool 的目标身份不完全成立 |
| geometry-safe P0-v2 A0 | 数据集层面仍观察到 2 次消失、1 次合并、1 次分裂 | A0 是描述性非门禁结果，不要求修改全局标签 |
| geometry-safe P0-v2 A1 | factual 32/32 保留；legal 209→206；A1 = pass | geometry-safe 分析人口构造成功 |
| geometry-safe P0-B | handcrafted 覆盖 2/32；decoder-joint 覆盖 16/32；门槛均为 29/32 | **fail**；两个预声明低维空间均未通过经验支持门禁 |
| geometry-safe P0-C | handcrafted AUC 0.9998；两空间 MMD summary 均高于 legal-vs-legal q95 | **fail**；按冻结规则观察到角色可预测性/状态偏移信号，不作显著性声明 |
| failure attribution | P、`F_background_global` 在全人口为 strong role signal；其余为 mixed/inconclusive；5 个小样本分层探针计算性不确定 | 描述性筛查完成；不能作因果归因，也不授权 transformation |
| 23→16 分解 | 固定旧表示直接删除 3 个目标仍为 23/32；在 legal-only 表示重拟合后变为 16/32 | 覆盖判定对表示拟合人口敏感，不是 3 个目标的直接删除效应 |
| hypothesis review | 五项 transformation 门槛未被任一候选同时满足；outcome = H0 | 当前 occupancy-deletion paradigm 不获 transformation 授权；重新定义 CURE-Lite 核心学习对象 |
| paired CURE-Lite core | 同源离散覆盖响应 \(\Delta_gQ\) 的协议已冻结；catalog、数据结构、paired loss/train step、32,000-step schedule 与 toy learnability 已实现 | 模型训练核心已实现；旧 decoder、model、loss、train step 哈希未变 |
| real \(D_R\) paired preflight | 206 clean-positive、16 component-null、160 identity-null；seed 42/43 各重放 32,000-step 暴露计划 | 两次产物逐字节一致；206 个 clean pair 全部暴露，target ESS 近似 206 |
| matched-control static preflight | DCT control 可构造；source-disjoint target permutation 为 `READY`，206 assignments、42,100 compatible edges、0 fixed points、0 same-source assignments | 静态控制入口通过 |
| bounded \(D_R\) learnability | 两次 GPU0 运行均为 `COMPUTATIONAL_LEARNABILITY_PASS` 且逐字节一致 | 证明固定真实微型人口上的计算可学习性；不是检测性能、自然漏检迁移或机制优越性证据 |
| spatial-tail companion | r2/r3 逐字节一致；component-null 强响应集中在被删组件及投影变化 cell 附近 | 局部捷径风险被显式记录；不是自然漏检或 FA 性能证据 |
| 8-control bounded execution | 全部 controls 完成 400 updates；总计 3,200 updates、9,600 forwards、38,400 states；r1/r2 逐字节一致 | `ENGINEERING_EXECUTION_PASS`；仅证明公平对照可执行 |
| formal schedule/artifact contract | 联合封存 32,000 updates 的 factual/pair identities、完整 exposure ledger、3-forward/12-state 预算和 create-only paired artifact | 实现与严格加载完成 |
| formal Wave A 训练 | seed 42/43 的 `paired_difference` 与 `independent_endpoint` 共四个 800-epoch、32,000-update 任务全部完成；无 checkpoint/resume | 训练产物完整，不能把工程完成解释为机制成功 |
| Wave A seed 42 | paired：147/170 TP、0/23 找回；最佳比较方法：154/170、7/23 | 约束通过，但 `-7/-7`，逐种子门槛失败 |
| Wave A seed 43 | paired：152/170 TP、5/23 找回；最佳比较方法：152/170、5/23 | 约束通过，但 `0/0`，逐种子门槛失败 |
| Wave A 总决定 | `PERFORMANCE_FAIL`; `all_seeds_pass=false`; `next_action=STOP_AND_PRESERVE_EVIDENCE` | 当前 paired 版本停止；不进入 Wave B/C、Full CURE 或跨 backbone |
| NLCC-v12 runner/evidence r2 | R0-C1～C8 通过；43/43 targeted、1105/1105 repository tests | Development 运行结论可由原始字段独立重算 |
| NLCC-v12 Development | 320/320 updates；25/25 structural gates；26/76 numeric gates；0/8 groups | `NLCC_V12_DEVELOPMENT_FAIL`；不重跑，不进入 Holdout |
| NLCC-v12 后续阶段 | Holdout、真实 \(D_R\)、32,000-step exposure、formal800 均未运行 | 均因 Development stop rule 未获授权 |
| PFCR-v2 + evidence-v3 Formal-800 | seed 42/43 均完成 800×40；共 64,000 updates、768,000 states；无 checkpoint/resume | 两个正式训练 artifact 均 COMPLETE |
| PFCR 正式 \(D_V\) seed 42 | 151/170 TP、4/23 找回、147/147 保留；全部误报预算通过 | 相对最佳 154/170、7/23 为 \(-3/-3\)，FAIL |
| PFCR 正式 \(D_V\) seed 43 | 150/170 TP、3/23 找回、147/147 保留；全部误报预算通过 | 相对最佳 152/170、5/23 为 \(-2/-2\)，FAIL |
| PFCR 正式总决定 | `PFCR_D_V_GATE_FAIL`；`all_seeds_pass=false` | 冻结当前候选；不授权 confirmation、Full CURE 或跨 backbone |
| P0-D | 未运行 | B/C 已失败，不能构造 S，也不能进入 exposure replay |
| Full CURE | 未设计、未实现 | 当前 paired 版本未通过，仍停在 CURE-Lite |
| 其他 IRSTD backbone | DNANet、UIUNet、MSHNet、SCTransNet 均未接入 | 目前没有跨 backbone 性能结论 |
| 数据集范围 | 当前正式性能结果只有 IRSTD-1K 的 \(D_V\) | 尚无 NUAA-SIRST、NUDT-SIRST 或 \(D_T\) 正式结果 |

当前最重要的研究结论不是“CURE 总体方向失败”，而是当前 synthetic-state 定义没有满足预声明的边缘重加权前提，因此应操作性地停止该路线：

1. v0.1 的均匀 synthetic 选择没有带来稳定增益；
2. v0.2 的一维硬最近邻只改变 legal targets 的边缘采样频率，且造成样本集中；
3. 原始 209-target synthetic population 含有目标谱系不明确或几何失真的样本；
4. geometry-safe P0-v2 已把分析人口修正为 32 个 factual targets 与 206 个 legal targets；
5. 在该有效人口上，P0-B/C 明确失败；准确含义是两个预声明低维表示均未通过经验支持筛查，不能外推为完整高维状态空间已被证明无共同支持；
6. failure attribution 只发现冻结低维摘要中的角色预测信号：`P` 是与角色定义耦合的 Base selection proxy，`F_background_global` 是唯一在全人口达到 strong 的 decoder-observed 单块；这不确定主要原因，也不直接授权任何干预；
7. 独立 hypothesis review 已完成并得到 H0：代码中的联合状态编辑事实不能被当前证据识别为失败原因，也不能唯一导出一个保持状态的变换；
8. 核心对象重定义已完成：legal before/after 不再作为 factual-like 样本，而用于定义同一 completion operator 的离散 coverage response \(\Delta_gQ\)；
9. 当前仍不得设计或训练边缘分布校正 S；paired-objective protocol、
   additive 训练核心、真实 \(D_R\) preflight、proposed/control bounded
   执行、空间尾部 companion、正式联合 schedule/artifact contract、
   四个 800-epoch 任务与 Wave A 揭示均已完成；
10. Wave A 的两个 seed 均未达到逐种子 `+2 TP/+2 recovered misses`，所以
    当前 paired 版本正式停止并保留；下一步是机制失败归因，而不是 Wave B/C、
    Full CURE、跨 backbone 或增加模块。

### 1.1 bounded learnability 的准确结果边界

冻结微型人口由 16 个 distinct-source clean pairs、16 个 factual-miss
anchors、16 个 factual-no-miss anchors、16 个 component-null 和 16 个
identity-null 构成。400 updates 中每个 clean pair 暴露 50 次，两类 factual
anchor 各暴露 100 次。两次运行的完整目录逐字节一致。

主要观测为：

| 指标 | 结果 | 冻结门槛 |
| --- | ---: | ---: |
| paired loss final/initial | 0.008860 | \(\le 0.5\) |
| paired positive macro mean \(\Delta Q\) | 0.971749 | \(\ge 0.5\) |
| pair fraction with mean \(\Delta Q\ge0.25\) | 1.0 | \(\ge0.75\) |
| paired zero-domain mean \(|\Delta Q|\) | 0.000459 | \(\le0.05\) |
| factual-miss loss final/initial | 0.526834 | \(\le0.75\) |
| factual-no-miss loss final/initial | 0.000771 | \(\le0.75\) |
| component-null macro mean \(|\Delta Q|\) | 0.000249 | \(\le0.05\) |
| identity-null maximum \(|\Delta Q|\) | 0 | \(\le10^{-7}\) |

所有冻结门槛通过，但 component-null 的全人口最大像素差分为
`0.999274`。该值不改变已经预声明的 macro-mean bounded 判定，却说明均值
可能掩盖局部强响应。因此 bounded pass 不能被解释为“null 特异性已经证明”，
更不能替代后续 matched controls、自然漏检恢复和固定 FA/retention 约束。

## 2. 项目和方法边界

CURE 的目标是一个与具体检测器架构解耦的 IRSTD 修正方法。CURE-Lite 是用于建立最小可行机制的第一阶段，不是最终 CURE，也不是 MSHNet 的内部模块。

当前数据流为：

```text
project-owned frozen Reference Base
  -> frozen probability p_b + detached spatial feature F_b
  -> CC8 occupancy / factual states / legal synthetic states
  -> fixed residual decoder
  -> monotone union with Base prediction
```

其中 CC8 表示二值图上的 8 邻域连通域规则。当前 Reference Base 是项目自有 U-Net，仅用于产生冻结的 \((p_b,F_b)\)；它不属于 CURE 方法贡献。

### 2.1 方法条件

| 符号 | 定义 |
| --- | --- |
| A | 冻结 Base 在 anchor threshold 下的结果，无 residual decoder |
| Base@B | 冻结 Base 在相同 FA/retention 约束内重新选择阈值 |
| F | factual-only residual learning |
| F× | 在第三损失槽中重复 factual-positive exposure 的对照 |
| U | factual learning + 均匀选择 decoder-visible legal synthetic targets |
| M | 与 U 完全相同，只把 synthetic target selection 改成基于一维 descriptor 的硬最近邻 |

M 使用：

\[
z(t)=\log\!\left(1+\operatorname{RMS}_{t}(F_b)\right)
\]

并按量化后的绝对距离从整个 legal catalog 中选最近目标。量化尺度为 \(10^6\)，允许多个 factual misses 重复选择同一 legal target。

### 2.2 M 的准确机制解释

当前 factual loss 与 synthetic loss 独立计算并相加：

\[
\mathcal L
=
\mathcal L_{\text{factual}}
+\lambda
\mathbb E_{l_i\sim q}
\left[\mathcal L_{\text{synthetic}}(l_i)\right].
\]

训练目标中没有 \(\mathcal L(m_j,l_i)\) 形式的 factual-miss–legal-target 配对损失，factual miss 信息也没有作为 decoder 条件输入。因此 M 不是 paired correction 或 conditional matching；它的实际作用只是改变：

\[
q_i=P(\text{训练时采样到 legal target }l_i).
\]

后续若继续该方向，准确命名应为：

> Factual-miss-informed support-preserving marginal distribution correction  
> 由 factual miss 引导、保持样本支持的边缘分布校正。

## 3. 数据与实验协议

当前 IRSTD-1K manifest 共 1001 个样本：

| split | 样本数 | 用途 |
| --- | ---: | --- |
| \(D_B\) | 520 | Reference Base 训练与选择 |
| \(D_R\) | 160 | CURE-Lite residual 训练与 P0 |
| \(D_V\) | 120 | 开发阶段阈值校准与机制比较 |
| \(D_T\) | 201 | 后续正式评估；当前未读取 |

\(D_B\) 进一步分为 416 个 fit 样本和 104 个 select 样本。所有当前 Pd、mIoU、nIoU 与 FA 数值都属于 \(D_V\) 开发结果，不能表述成独立测试集泛化结果。

### 3.1 固定约束

| 约束 | 上限或下限 |
| --- | ---: |
| pixel FA | \(\le 1\times10^{-4}\) |
| raw-background FA | \(\le 1\times10^{-4}\) |
| FP components / MP | \(\le 100\) |
| covered-target retention | \(\ge 0.99\) |

v0.1 和 v0.2 的所有比较方法均满足上述约束，所以 v0.2 的负结果不是由某个方法超出 FA 或 retention 约束造成的。

### 3.2 训练契约

| 项目 | 值 |
| --- | ---: |
| epochs | 800 |
| steps / epoch | 40 |
| updates / decoder | 32,000 |
| factual-miss batch | 4 |
| factual-no-miss batch | 4 |
| synthetic batch | 4 |
| optimizer | Adam |
| learning rate | 0.001 |
| weight decay | 0 |
| decoder feature channels | 64 |
| decoder width | 32 |
| groups | 8 |
| \(\lambda_{\text{no-miss}}\) | 1.0 |
| \(\lambda_{\text{synthetic}}\) | 1.0 |

seed 42 与 seed 43 共享数据划分、Reference Base、cache 和实验协议，只改变 residual decoder 初始化及确定性采样种子。它们应称为 **paired decoder-seed repeats**，不是两个独立数据划分或两个独立 Base 实验。

## 4. Reference Base 结果

| 项目 | 结果 |
| --- | ---: |
| 训练 epoch | 800 |
| best epoch | 314 |
| \(D_B\)-select best global mIoU | 0.6459125475 |
| best select loss | 0.3744194347 |
| Base fingerprint | `5f69986b95d11a89c5a5e91d6bdd63add865eda102be8ce486722fee8cd00dce` |
| checkpoint SHA256 | `886721b0985b128b8933a4de125c5b0e0a76cb4995194e70be7e726f056397a5` |

权威产物：

- [Reference Base selection](runs/irstd1k_stage_a_seed42/reference_base_v1/selection.json)
- [Reference Base completion](runs/irstd1k_stage_a_seed42/reference_base_v1/COMPLETE.json)

## 5. v0.1 与 v0.2 的全部正式性能结果

下表把历史 v0.1 的 A/Base@B/F/F×/U 与 v0.2 新增的 M 放在同一协议下。数值来自各 run 的正式 `results.json`；`null` 阈值表示合法的 residual-off 选择，不是缺失结果。

### 5.1 seed 42

| 方法 | 阈值 | Pd ↑ | mIoU ↑ | nIoU ↑ | pixel FA ↓ | raw-bg FA ↓ | FP comp./MP ↓ | retention ↑ | 约束 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 0.72 | 0.864706 | 0.609559 | 0.565328 | 2.110799e-5 | 7.171631e-5 | 3.051758 | 1.000000 | 满足 |
| Base@B | 0.14 | 0.882353 | 0.607629 | 0.564014 | 2.428691e-5 | 8.201599e-5 | 3.433228 | 1.000000 | 满足 |
| F | 0.78 | **0.905882** | 0.594851 | 0.534029 | 3.318787e-5 | 9.167989e-5 | 8.900960 | 1.000000 | 满足 |
| F× | 0.98 | 0.876471 | 0.589170 | 0.528516 | 3.496806e-5 | 9.409587e-5 | 7.756551 | 1.000000 | 满足 |
| U | 1.00 | 0.888235 | 0.597060 | 0.535961 | 3.255208e-5 | 8.583069e-5 | 6.103516 | 1.000000 | 满足 |
| M | 1.00 | 0.882353 | 0.592637 | 0.549957 | 4.590352e-5 | 9.028117e-5 | 6.357829 | 1.000000 | 满足 |

### 5.2 seed 43

| 方法 | 阈值 | Pd ↑ | mIoU ↑ | nIoU ↑ | pixel FA ↓ | raw-bg FA ↓ | FP comp./MP ↓ | retention ↑ | 约束 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 0.72 | 0.864706 | 0.609559 | 0.565328 | 2.110799e-5 | 7.171631e-5 | 3.051758 | 1.000000 | 满足 |
| Base@B | 0.14 | 0.882353 | 0.607629 | 0.564014 | 2.428691e-5 | 8.201599e-5 | 3.433228 | 1.000000 | 满足 |
| F | 0.58 | **0.894118** | 0.597902 | 0.532828 | 4.069010e-5 | 9.841919e-5 | 9.028117 | 1.000000 | 满足 |
| F× | null | 0.864706 | **0.609559** | **0.565328** | 2.110799e-5 | 7.171631e-5 | 3.051758 | 1.000000 | 满足 |
| U | 1.00 | **0.894118** | 0.587288 | 0.540080 | 4.247030e-5 | 9.613037e-5 | 6.484985 | 1.000000 | 满足 |
| M | 1.00 | 0.876471 | 0.605239 | 0.550010 | 3.102620e-5 | 8.430481e-5 | 6.357829 | 1.000000 | 满足 |

### 5.3 两个 paired decoder seeds 的算术均值

这些均值描述 decoder-seed 重复，不代表独立划分均值。

| 方法 | mean Pd ↑ | mean mIoU ↑ | mean nIoU ↑ | mean pixel FA ↓ | mean raw-bg FA ↓ | mean FP comp./MP ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.864706 | 0.609559 | 0.565328 | 2.110799e-5 | 7.171631e-5 | 3.051758 |
| Base@B | 0.882353 | 0.607629 | 0.564014 | 2.428691e-5 | 8.201599e-5 | 3.433228 |
| F | **0.900000** | 0.596376 | 0.533428 | 3.693899e-5 | 9.504954e-5 | 8.964539 |
| F× | 0.870588 | **0.599365** | 0.546922 | 2.803802e-5 | 8.290609e-5 | 5.404154 |
| U | 0.891176 | 0.592174 | 0.538020 | 3.751119e-5 | 9.098053e-5 | 6.294250 |
| M | 0.879412 | 0.598938 | **0.549983** | 3.846486e-5 | 8.729299e-5 | 6.357829 |

主要均值差：

| 比较 | Pd 差值 |
| --- | ---: |
| M − U | −0.011765（−1.1765 个百分点） |
| M − F | −0.020588（−2.0588 个百分点） |
| M − Base@B | −0.002941 |
| M − F× | +0.008824 |

M 的 nIoU 高于 U，但主要门槛预先规定为 Pd 和固定漏检目标找回；次要 IoU 指标不能覆盖主要门槛失败。

## 6. v0.1 结论

v0.1 要求 U 在相同约束内严格超过 Base@B、F 和 F×。

| seed | Pd(F) | Pd(U) | U − F | v0.1 结论 |
| ---: | ---: | ---: | ---: | --- |
| 42 | 0.905882 | 0.888235 | −0.017647 | 不满足 |
| 43 | 0.894118 | 0.894118 | 0 | 不满足严格提升 |

可成立的结论是：

> 在这两个 \(D_V\) 开发重复中，均匀 legal-target selection 没有显示出超过 factual-only learning 的稳定机制信号。

不能据此声称整个 CURE 方向失败。

## 7. v0.2 M 结果

### 7.1 固定漏检目标找回

正式 denominator 为固定的 23 个 \(D_V\) anchor misses。

| seed | F 找回 | U 找回 | M 找回 | M − U | 逐 seed 门槛 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 7/23 | 4/23 | 3/23 | −1 | 失败 |
| 43 | 5/23 | 5/23 | 2/23 | −3 | 失败 |
| 合计 | 12 | 9 | 5 | −4 | 两个 seed 均失败 |

F/U 来自机制诊断，U/M 来自正式 recovery receipt。A、Base@B 和 F× 没有同等级的独立 recovery receipt，因此不把由 Pd 反推的数量作为正式找回证据。

### 7.2 预声明比较与实际差值

M 必须对每个 seed 同时满足：

\[
\operatorname{Pd}(M)>
\max\{
\operatorname{Pd}(\text{Base@B}),
\operatorname{Pd}(F),
\operatorname{Pd}(F\times),
\operatorname{Pd}(U)
\}
\]

且：

\[
\operatorname{RecoveredMisses}(M)>
\operatorname{RecoveredMisses}(U).
\]

| seed | M−Base@B | M−F | M−F× | M−U | M−U 找回数 | mechanism signal |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0 | −0.023529 | +0.005882 | −0.005882 | −1 | false |
| 43 | −0.005882 | −0.017647 | +0.011765 | −0.017647 | −3 | false |

两个 gate 均记录：

- `all_methods_within_constraints = true`；
- `strict_pd_rule_met = false`；
- `recovery_rule_met = false`；
- `mechanism_signal = false`；
- `independent_generalization_claim = false`。

### 7.3 M 映射与集中度

| 项目 | 结果 |
| --- | ---: |
| reachable factual misses mapped | 32/32 |
| legal catalog size | 209 |
| unique selected legal targets | 16 |
| unique selected legal sources | 16 |
| maximum reuse | 7 |
| reuse counts | `[7, 4, 4, 3, 2, 2, 1 × 10]` |
| mean quantized distance | 31,342.1875 |
| maximum quantized distance | 148,544 |
| alignment catalog fingerprint | `f53b65e9962642e2705a6641861d039245c507e6703f8846242376bb43411f73` |
| alignment file SHA256 | `f7d561579b3de0b7966b9a1cf8406890de29a14a28cb856b0282eda9e2acb01e` |

Kish effective sample size 为：

\[
\operatorname{ESS}
=
\frac{32^2}
{7^2+4^2+4^2+3^2+2^2+2^2+10}
=
\frac{1024}{108}
=9.48148.
\]

这说明 32 次 synthetic 选择近似反复使用很少的 legal targets。它支持“硬一维最近邻造成边缘采样集中”的解释，但不能单独证明某种新的平滑方案必然有效。

### 7.4 v0.2 最终结论

v0.2 是一个完整、可复查的负结果，不是程序运行失败：

- 两个 M decoder 都完成 800 个连续 epoch；
- 每个 epoch 40 steps，共 32,000 updates；
- decoder 权重、train log、alignment、calibration、results、gate 和 COMPLETE 均形成文件绑定；
- 历史 A/Base@B/F/F×/U 结果被复用，没有重新训练或修改；
- 两个 M calibration 都选择 residual threshold 1.0。

阈值 1.0 是重要诊断现象，但 receipt 没有记录 max residual logit、饱和像素比例或显式 null 对照。因此不能仅凭阈值断言 residual 完全关闭。

正式判断：

> 停止标量或多维硬最近邻路线，保留 v0.2 作为有效负结果；不要把 M 提升为 CURE。

## 8. CURE-Lite decoder 的效率记录

下列结果只测量增量 residual decoder 的 batch-1、全零形状探针，不包含 Base inference、数据传输、I/O、阈值处理、union、校准或指标计算。

| 项目 | seed 42 环境记录 | seed 43 环境记录 |
| --- | ---: | ---: |
| 参数量 | 21,089 | 21,089 |
| Conv2d MACs | 85,196,800 | 85,196,800 |
| Conv2d FLOPs | 170,393,600 | 170,393,600 |
| median latency | 0.792096 ms | 0.825824 ms |
| p95 latency | 0.809984 ms | 0.854016 ms |
| peak allocated | 3,436,032 bytes | 3,436,032 bytes |
| peak incremental allocated | 2,232,832 bytes | 2,232,832 bytes |

测量设备为 NVIDIA GeForce RTX 3090。这些数值不是端到端 CURE 或 IRSTD 模型速度。

## 9. P0-v1：原 209-target population 的前置诊断

权威执行为：

- `cure_lite_p0_v1_r3`；
- `cure_lite_p0_v1_r4`。

两次目录逐字节一致：

| 项目 | 值 |
| --- | --- |
| COMPLETE SHA256 | `d86db17d3bb95bfd8a25de05152aa9b9da07f0946809d966ec7f7875a17a3fa8` |
| complete fingerprint | `dff2100ff8844747ccc443b79ba812cd1c3aabc3ff5856a6f3ea4dd3b36eddc7` |
| config fingerprint | `fc2364ca94fc5e88d2f416952eff9259610d37f22ebfe9a9f8d1c60e8c2c8c55` |
| P0-A receipt fingerprint | `d5f08de029320bcdaf25e2da4bfd55dd115fd4642c1a7fad8bb907dafcbfc8a5` |
| decision fingerprint | `8cd858ffb4fbd4c4c4b6d42e1b184172a3fef4a4fb1e72cee18f0ce040595b54` |

### 9.1 P0-A 正式结果

| 检查 | 观察值 |
| --- | ---: |
| source images | 160 |
| native targets | 244 |
| evaluation targets | 242 |
| native disappearances | 2 |
| evaluation merges | 1 |
| native splits | 1 |
| legal targets without one-to-one lineage | 2 |
| legal area-ratio failures | 1 |
| legal centroid-shift failures | 0 |
| P0-A | **false** |

具体事件：

- XDU325/native 3：1 像素目标在投影后消失；
- XDU909/native 2：1 像素目标在投影后消失；
- XDU486/evaluation 1：由 native 1 和 native 2 合并；
- XDU526/native 1：分裂为 evaluation 1 和 evaluation 2；
- XDU965/evaluation 1：area ratio = 0.470588，低于冻结下限 0.5。

因此 v1 的唯一正式路线是 `rebuild_synthetic_target_extraction`。

### 9.2 P0-B/C 的旧调试数值

P0-v1 在 P0-A 失败后仍计算了 B/C 数值，但二者的正式值都是 `null`。由于 legal target identity 前提不成立，这些值只能作为旧人口的调试记录，不能写成正式共同支持结论。

#### P0-B：grouped kNN coverage

| 表示空间 | covered factual | required | coverage | 数值诊断 |
| --- | ---: | ---: | ---: | --- |
| handcrafted | 2/32 | 29/32 | 0.0625 | fail |
| decoder joint | 23/32 | 29/32 | 0.71875 | fail |

正式状态：`not_evaluated_due_to_p0_a_failure`，`p0_b_pass = null`。

#### P0-C：grouped distinguishability 与 MMD

| 检查 | 观察值 | 冻结参考 | 数值诊断 |
| --- | ---: | ---: | --- |
| handcrafted grouped OOF AUC | 0.999762 | \(\le 0.70\) | fail |
| handcrafted AUC 95% bootstrap interval | [0.999041, 1.000000] | — | 调试记录 |
| decoder-joint grouped OOF AUC | 0.717361 | \(\le 0.70\) | fail |
| decoder-joint AUC 95% bootstrap interval | [0.594646, 0.831194] | — | 调试记录 |
| handcrafted MMD observed | 0.559901 | legal-vs-legal q95 = 0.017143 | fail |
| decoder-joint MMD observed | 0.124867 | legal-vs-legal q95 = 0.022931 | fail |

正式状态：`not_evaluated_due_to_p0_a_failure`，`p0_c_pass = null`。

P0-D 的正式值也是 `null`；candidate S 没有被构造或训练。

### 9.3 历史 U/M 完整训练暴露回放

P0-D receipt 还保存了对历史 U/M 确定性采样逻辑的完整回放。每个 seed 均模拟：

\[
800\ \text{epochs}\times 40\ \text{steps/epoch}\times 4\ \text{synthetic samples}
=128{,}000\ \text{draws}.
\]

这些数值用于解释历史 M 的集中现象，不是 candidate S 的结果，也不构成正式 P0-D gate。

| seed | 方法 | target 暴露 | target 零暴露 | target ESS | target max share | target Gini | target top-10 share | source 暴露 | source 零暴露 | source ESS | source max share | source Gini | source top-10 share |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | U | 209/209 | 0 | 183.2422 | 0.007227 | 0.201842 | 0.070453 | 150/150 | 0 | 149.8469 | 0.007227 | 0.018007 | 0.071102 |
| 42 | M | 16/209 | 193 | 10.8517 | 0.194555 | 0.948685 | 0.799063 | 16/150 | 134 | 10.8517 | 0.194555 | 0.928501 | 0.799063 |
| 43 | U | 209/209 | 0 | 182.5578 | 0.007164 | 0.205123 | 0.070938 | 150/150 | 0 | 149.8238 | 0.007164 | 0.019302 | 0.071234 |
| 43 | M | 16/209 | 193 | 10.8280 | 0.195305 | 0.948819 | 0.800641 | 16/150 | 134 | 10.8280 | 0.195305 | 0.928688 | 0.800641 |

完整回放进一步确认：M 不仅静态映射的支持很窄，在真实 32,000-step 采样计划中也只暴露 16 个 legal targets 和 16 张 source images；约 80% 的全部暴露集中于 top-10。U 则覆盖全部 209 个 targets 和 150 张 source images。

### 9.4 v1 运行版本边界

以下仅作为历史或无效运行保留，不用于正式结论：

- `cure_lite_p0_v1`；
- `cure_lite_p0_v1_r2`。

权威证据只取逐字节一致的 r3/r4。

## 10. geometry-safe P0-v2：native-first lineage-aware population

P0-v2 没有改动全局 256 标签，也没有通过膨胀、补像素、强制切分或保留最大连通域来“修正”目标。它从 native 512 标签开始建立谱系，只允许满足以下要求的分析目标进入新 population：

1. native 投影非空；
2. native 与 evaluation component 双向一对一；
3. 投影不分裂；
4. 投影 component 与 evaluation component 完全一致；
5. area ratio 位于 \([0.5,2.0]\)；
6. evaluation-grid centroid shift 不超过 1.0 px；
7. synthetic positive 与 evaluation target 一致；
8. 所有排除均有确定的原因码和 receipt。

### 10.1 P0-A0：数据集几何审计，非门禁

| 项目 | 结果 |
| --- | ---: |
| source images | 160 |
| native targets | 244 |
| evaluation targets | 242 |
| disappearances | 2 |
| merges | 1 |
| splits | 1 |
| audit status | `anomalies_present` |
| dataset exact preservation | false |
| downstream gate effect | none |

A0 的具体事件与 P0-v1 一致：

- disappearance：XDU325/native 3、XDU909/native 2；
- merge：XDU486/evaluation 1 ← native [1,2]；
- split：XDU526/native 1 → evaluation [1,2]。

A0 不要求 244 个 native targets 全部进入分析人口。它的作用是完整记录数据表示损失，而不是阻止一个经过明确筛选的 geometry-safe population。

### 10.2 P0-A1：分析人口资格

| 项目 | candidate | eligible | excluded / outside |
| --- | ---: | ---: | ---: |
| factual targets | 32 | 32 | 0 |
| factual groups | 24 | 24 | 0 |
| legal targets | 209 | 206 | 3 |
| legal source images | 150 | 149 | 1 |
| legal groups | 146 | 145 | 1 |
| unreachable factual targets | — | — | 1 |

完整 factual 记账为：

- discovered factual targets：33；
- reachable factual candidates：32；
- unreachable、位于分析人口之外：XDU526/evaluation GT 2；
- 32 个 reachable factual targets 全部 geometry eligible。

被排除的 3 个 legal targets：

| identity | 原因 | 关键值 |
| --- | --- | --- |
| XDU486 / GT 1 / pred 1 | `multiple_native_ancestors` | native ancestors = [1,2] |
| XDU526 / GT 1 / pred 1 | `native_has_multiple_evaluation_descendants` | evaluation descendants = [1,2] |
| XDU965 / GT 1 / pred 1 | `area_ratio_below_minimum` | area ratio = 0.470588 |

A1 的完整性检查：

| 检查 | 结果 |
| --- | --- |
| all candidates classified exactly once | true |
| duplicate candidate identities | 0 |
| duplicate eligible identities | 0 |
| invalid retained targets | 0 |
| unaccounted targets | 0 |
| all reachable factual geometry eligible | true |
| all retained lineage bidirectional one-to-one | true |
| all retained component projections exact | true |
| all retained area ratios within gate | true |
| all retained centroid shifts within gate | true |
| P0-A1 | **pass** |

### 10.3 P0-v2 复现与绑定

权威执行：

- `cure_lite_geometry_safe_p0_v2_r1`；
- `cure_lite_geometry_safe_p0_v2_r2`。

两个目录的全部 receipt 与 COMPLETE 逐字节一致。

| 项目 | 值 |
| --- | --- |
| protocol ID | `irstd1k-dr-geometry-safe-p0-v2` |
| config file SHA256 | `719e956b7c51b2b2c8294699fe26c2d36d5c8190b0d8bb5c1d5665a0f4344558` |
| config fingerprint | `26b35d549faf8e9a4ae151a418acbec888dded5372afa7b4c1e5d25d745cde60` |
| source catalog fingerprint | `936086fcf6d30e6bc1cd3b45b408bceadeb573304bbe4d79c30f4fc47b3a42b2` |
| geometry catalog fingerprint | `16a0a587341d5403614fcc9de90581727f107f6566b195c1b10a857f7d185fa8` |
| eligible catalog fingerprint | `a7f3862e41272edb8cafa50398f104c5fab6a8ae55c78bf90c1f2ef34bf1bcd9` |
| A0 receipt fingerprint | `0fccad2feca25b77c43f477c6c10648e8e595127111dca1025aa94a84b4bdd49` |
| A1 receipt fingerprint | `8ce8d3b00fbc6569a5aa5272a343c9293c0da4b06d592ae2b3c9d2066a93e351` |
| COMPLETE fingerprint | `82520c2c5b936b411ebdb7a6ee1babfc7736fa0562e2dc677b8f7ca91d1655b0` |
| COMPLETE file SHA256 | `84150d26d59093963b0326bb9e85423fc880c520187abc9b22bf751c950627d0` |

geometry-safe view 复用原 catalog 中的 candidate、synthetic example、factual tuple 和 feature 对象；它只新建 view、索引 tuple 与统计元数据，没有复制训练张量，也没有重建一个新训练 catalog。

### 10.4 当前 P0-v2 阶段状态

| 阶段 | 正式值 | 执行状态 |
| --- | --- | --- |
| A0 | `anomalies_present` | completed，非门禁 |
| A1 | true | completed，pass |
| B | false | completed，fail |
| C | false | completed，fail |
| D | null | B/C 已失败，不得构造 S，因此未运行 |
| candidate S | — | not constructed |
| S training | — | not performed |
| new \(D_V\) evaluation | — | not performed |
| Full CURE | — | not authorized |

本轮 follow-on 严格重建并绑定的人口为：

| 项目 | 数量 |
| --- | ---: |
| factual targets / groups | 32 / 24 |
| legal targets / source images / groups | 206 / 149 / 145 |
| factual/legal overlap groups | 14 |
| overlap groups 内 factual / legal targets | 18 / 25 |
| legal-exclusive groups | 131 |

P0-B 的正式结果为：

| 表示空间 | covered factual | coverage | 冻结门槛 | 状态 |
| --- | ---: | ---: | ---: | --- |
| handcrafted | 2/32 | 0.0625 | 至少 29/32 | fail |
| decoder-joint | 16/32 | 0.5000 | 至少 29/32 | fail |

P0-C 的正式结果为：

| 表示空间 | grouped OOF AUC | group-bootstrap interval | MMD observed | legal-vs-legal q95 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| handcrafted | 0.999808 | [0.999167, 1.000000] | 0.559400 | 0.018326 | fail |
| decoder-joint | 0.753330 | [0.629920, 0.858031] | 0.133810 | 0.026651 | fail |

decoder-joint AUC interval 跨越 0.70，因此该 AUC 子项单独为 `inconclusive`；但同一空间的 MMD 明确失败，故 decoder-joint 空间与 P0-C 总门禁仍为 `fail`。这不是用一个统计量替代另一个统计量，而是按预先冻结的三值合取执行。

两个独立输出目录中的同输入文件系统重放 r1/r2 全部文件逐字节一致：

```text
config fingerprint   = 707a9d50d4b707988a90a7220162d2f47c6f219380eee60e7c262a9c11cd8099
COMPLETE fingerprint = de1ad2b460db48c2ec28d21814be0cbc7190e121ee5be5321bcbf178ba6ff997
diff -qr r1 r2       = no differences
```

新 follow-on decision 的准确语义为：

```text
P0-A1 = pass
P0-B  = fail
P0-C  = fail
P0-D  = not_evaluated
all_p0_pass = null
next_route = redesign_synthetic_state
eligible_to_design_candidate_s = false
```

`all_p0_pass = null` 是因为 P0-D 未执行；这不抵消 B/C 已经明确失败。当前可成立的阶段结论是：现有 legal synthetic state 不满足边缘分布校正的前提，因此停止所有仅调整 \(q_i\) 的路线。

### 10.5 \(D_R\)-only synthetic-state failure attribution

本阶段不再重复 P0-B/C 门禁，也不搜索新 descriptor。它只对冻结的 32 个 factual targets 与 206 个 geometry-safe legal targets 进行预声明的描述性筛查，回答哪些低维摘要携带 factual/legal 角色预测信号。

#### 10.5.1 运行与状态契约

| 项目 | 正式值 |
| --- | --- |
| protocol | `irstd1k-dr-synthetic-state-failure-attribution-v1` |
| config SHA256 | `8933113e745ab42119e90a0a3f2b4366290f38a6b523251d94d39bc5665e6161` |
| config fingerprint | `38046ddd0642b2f0dd28f39591cd525d2d6de3b7d224546162cdc7eb8d14b7e8` |
| population fingerprint | `d949fdd388772b83a55da43d688dde2faab311101268fe3d7bcd461c1bd5d22f` |
| r1/r2 COMPLETE fingerprint | `05ef3539f2427f5bc4595e87085ef52eadfec697c6e58a4b6104503d1bcc1939` |
| r1/r2 COMPLETE file SHA256 | `fdc923461ff96c18aaf6774dc3404d8aa1ba2a0aa5715660652de544b0209b09` |
| execution state | `partial_inconclusive` |
| formal P0-B / P0-C / P0-D | `fail / fail / not_evaluated` |
| next stage | `separate_hypothesis_review_required_before_any_transformation_proposal` |

两个独立输出目录中的同输入文件系统重放逐字节一致：

```text
diff -qr cure_lite_synthetic_state_failure_attribution_v1_r1 \
         cure_lite_synthetic_state_failure_attribution_v1_r2
= no differences
```

这证明当前输入、实现和环境下的协议可确定性重放，不是两个统计独立实验，也不提供跨划分泛化证据。

state-contract ledger 对 206 个 legal targets 全部验证：

- 删除后的 occupancy 差分精确等于被冻结的预测组件；
- synthetic feature 与 source frozen feature 的 tensor fingerprint 完全一致；
- 206/206 state-contract ledger rows 均未进入角色分类器；对应的 206 个 legal state factor records 仍作为探针中的 legal 类参与拟合；
- 所有 transformation、candidate S、P0-D、training、calibration、inference、\(D_V\)、\(D_T\)、Full CURE 和其他 backbone 权限均为 `false`。

#### 10.5.2 六个冻结单块探针

各块定义为：

| 块 | 含义 | 维数 | 解释限制 |
| --- | --- | ---: | --- |
| `G_full` | 完整 GT 目标几何摘要 | 6 | 非 decoder 输入 |
| `W` | writable supervision 几何摘要 | 4 | 是 loss support 摘要，不是目标外观 |
| `P` | Base probability 摘要 | 7 | 是角色定义相关的 selection proxy，非 decoder 输入 |
| `F_local` | 目标/背景环局部 feature 矩摘要 | 193 | 与 `F_background_global` 共享背景环信息 |
| `F_background_global` | 背景环与全局 feature 矩摘要 | 256 | 不是独立“背景因果因素” |
| `O` | 删除后 conditioning occupancy 摘要 | 29 | 不是完整 occupancy tensor |

下表中的 AUC interval 是对冻结 OOF scores 的条件 group bootstrap interval，不是重新拟合完整模型得到的置信区间。MMD 只比较 observed summary 是否高于冻结的 legal-vs-legal q95，不是 \(p\) 值或显著性检验。

| 块 | coverage | AUC | 条件 group-bootstrap interval | MMD observed | legal q95 | 冻结筛查状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `G_full` | 27/32 | 0.595734 | [0.462398, 0.720629] | 0.038172 | 0.013626 | mixed/inconclusive |
| `W` | 28/32 | 0.609564 | [0.462062, 0.744312] | 0.089056 | 0.032218 | mixed/inconclusive |
| `P` | 30/32 | 0.996058 | [0.988947, 1.000000] | 0.438820 | 0.024836 | strong role signal |
| `F_local` | 18/32 | 0.684553 | [0.547673, 0.806794] | 0.133807 | 0.026651 | mixed/inconclusive |
| `F_background_global` | 29/32 | 0.862004 | [0.800490, 0.916650] | 0.051349 | 0.016551 | strong role signal |
| `O` | 27/32 | 0.687193 | [0.559597, 0.809778] | 0.100077 | 0.030416 | mixed/inconclusive |

六个块的 observed MMD summary 都高于各自 legal reference q95。不能据此比较不同块的“重要性”，也不能说完整高维状态分布已经被证明不同。

`P` 的强信号首先是一个 sanity signal：factual/legal 角色本来就由同一 Base 输出参与定义。它不进入 decoder，因此不能据此提出“修改 P”。在 decoder-observed 单块中，只有 `F_background_global` 在全人口同时满足冻结 AUC 与 MMD 的 strong 定义；这形成一个待审查假设，但不是“背景导致失败”的因果证明。

#### 10.5.3 预声明分层敏感性

| 块 | 全人口 AUC 状态 | shared manifest groups | selected dual-role sources，中心化后 |
| --- | --- | --- | --- |
| `G_full` | inconclusive | 0.592007，inconclusive | 0.787415，inconclusive |
| `W` | inconclusive | 0.596599，inconclusive | 0.720663，inconclusive |
| `P` | strong | 固定 IRLS 未收敛，计算性 inconclusive | 0.942602，strong |
| `F_local` | inconclusive | 固定 IRLS 未收敛，计算性 inconclusive | 固定 IRLS 未收敛，计算性 inconclusive |
| `F_background_global` | strong | 0.864626，strong | 0.754677，inconclusive |
| `O` | inconclusive | 0.722506，inconclusive | 0.927721，strong |

shared-groups 子集为 14 groups、18 factual、25 legal；selected dual-role-source 子集为 14 sources、18 factual、21 legal。后者是预选择重叠样本上的 transductive sensitivity，不是 source/background effect 已被控制或消除。

两个分层中的 `decoder_input_probe_union` 也因冻结 IRLS 未收敛而记为计算性 `inconclusive`。程序没有更改 L2、迭代上限、fold 或表示后重算。因此 `partial_inconclusive` 描述的是诊断中有 5 个预声明 estimand 未得到，不是方法训练失败。

#### 10.5.4 decoder-input 低维组合探针

全人口 `decoder_input_probe_union = F_local + F_background_global + O` 的 AUC 为 0.807789，条件 interval 为 [0.688933, 0.904504]，cross-fitted log loss 为 1.114610。其 interval 跨越 0.70，故三值状态为 `inconclusive`。

固定 drop-one 点差为：

| probe | log loss | 相对 union 的差值 |
| --- | ---: | ---: |
| union | 1.114610 | — |
| drop `F_local` | 0.498576 | −0.616034 |
| drop `F_background_global` | 1.215033 | +0.100423 |
| drop `O` | 1.178401 | +0.063791 |

这些只是相同 OOF population/fold 下的条件预测 log-loss 点差，没有区间、显著性检验或选优规则。由于三个块相关，且两个 feature 块共享背景环统计，不能把该表写成完整 decoder-input 消融、因果贡献排名或 transformation 选择。

#### 10.5.5 旧 23/32 到 geometry-safe 16/32 的分解

32 个 factual raw states 和保留的 206 个 legal raw states 在旧/新人口间均逐元素完全一致。三个排除目标在旧表示下从未进入任一 factual query 的 group-distinct top-5。

| replay 路径 | legal population | 表示拟合 | covered factual |
| --- | ---: | --- | ---: |
| O | 209 | fit-209 | 23/32 |
| A | 206 | 固定旧 projector、scale、radius | 23/32 |
| A+R | 206 | 固定旧 projector/scale，仅重算 radius | 23/32 |
| B | 206 | 固定旧 projector，重拟合 outer scale/radius | 23/32 |
| C-cross | 209 | fit-206 projector/scale | 16/32 |
| C | 206 | fit-206 | 16/32 |
| D | 206 | 完整 geometry-safe 路径 | 16/32 |

分别删除任一目标或同时删除三个目标、同时固定旧表示与 radius 时，覆盖身份都保持 23/32。覆盖转变在 legal-only feature representation 重拟合时首次出现；fit-209 到 fit-206 改变了 PCA basis、384 个原始维度中的 203 个 robust median 与 277 个 robust scale。七个失去覆盖的 factual targets 为：

```text
XDU309/gt1, XDU486/gt3, XDU543/gt3, XDU731/gt1,
XDU865/gt4, XDU908/gt1, XDU908/gt2
```

因此，准确结论是 decoder-joint coverage 对合法人口上拟合的表示敏感。不能写成“三个被排除目标直接导致七个 factual targets 失去支持”；旧 23/32 本身也来自 identity-invalid population，只能作为分解参照。

#### 10.5.6 本阶段最终判断

failure attribution 没有识别出一个可直接实施的单一原因。它支持以下更窄的判断：

1. 当前 legal-state 构造精确删除 occupancy，但完全保留 source frozen feature；
2. Base selection proxy `P` 强烈携带角色信息，但它不是 decoder 输入；
3. decoder-observed 的 context/global feature 摘要在全人口和 shared groups 中携带 strong 角色信号；occupancy 在 selected-source transductive sensitivity 中携带 strong 信号；
4. `F_local` 未获得稳定 strong 结论，不能据此直接实施 target-feature attenuation；
5. 多个摘要相关，部分小样本拟合不确定，因此当前不能确定“主要原因”。

failure-attribution 阶段结束时的正式决策为：

```text
marginal reweighting route = stopped
transformation proposal    = not authorized
candidate S                = not constructed
P0-D                       = not evaluated
training                   = not performed
new D_V / D_T access       = false
Full CURE                  = not started
attribution-stage handoff  = separate hypothesis review
```

其中最后一行是该阶段的交接状态；独立 review 现已完成并得到 H0，当前路线以第 13 节为准。

## 11. 软件、代码与运行完整性

### 11.1 当前测试

2026-07-24 在冻结上游提交与当前工作树上重新执行。下列数值是本次终端测试记录，不属于科学结果 receipt：

```text
723 passed in 183.17s
```

这说明当前软件路径通过测试；它不替代性能实验。

测试数量的历史演进：

| 快照/阶段 | 测试结果 | 解释 |
| --- | --- | --- |
| 历史公开审计 `b28040f` | 353 passed, 6 skipped | 当时 6 项依赖未公开历史产物或真实适配 |
| P0-v1 完成后 | 376 passed | 增加 P0 诊断覆盖 |
| `a9dcfb2` | 389 passed | 增加 geometry-safe protocol/catalog 测试 |
| Phase 1 工作树 | 442 passed | 增加 P0-B/C protocol、三值决策与 fold-local receipt 测试 |
| failure-attribution 完成时工作树 | 498 passed | 增加 23→16 分解、六块归因、冻结协议、权限边界和 runner 测试 |
| 当前 hypothesis-review 工作树 | 503 passed | 增加 H0 proposal receipt、证据/源码绑定、权限边界和总主线一致性测试 |
| 当前 core-object redefinition 工作树 | 510 passed | 增加 \(\Delta_gQ\) 定义、跨目标干扰反例、证据绑定、阶段权限、性能门槛与文档一致性测试 |
| 当前 additive paired implementation 工作树 | 561 passed | 增加 pair catalog、paired loss/train step、确定性 schedule、preflight writer 与 toy learnability 测试 |
| 当前 paired preflight/control/bounded 工作树 | 609 passed | 增加真实产物重建、matched-control、deterministic bounded execution、产物 loader 与篡改拒绝测试 |
| 当前 formal evidence closure 工作树 | 699 passed | 增加正式 training engine、完整训练人口指纹、control provider、共同 \(D_V\) 协议、无恢复 runner、逐 wave 决策与篡改拒绝测试 |
| 当前 Wave A reveal 工作树 | 723 passed | 增加 receipt-only 历史证据加载、一次性 reveal、原子发布、严格重载与失败停止测试 |

历史阶段中的 `376 passed`、`389 passed`、`442 passed`、`498 passed` 和
`510 passed`、`561 passed`、`609 passed`、`699 passed` 仅表示各自快照；
当前工作树以本次 `723 passed` 为准。

### 11.2 当前新增实现

当前 geometry/P0 代码包括：

- `cure_lite/experiment/p0_geometry.py`
- `cure_lite/experiment/p0_support.py`
- `cure_lite/experiment/exposure_audit.py`
- `cure_lite/experiment/p0_protocol.py`
- `cure_lite/experiment/geometry_catalog_protocol.py`
- `cure_lite/experiment/geometry_safe_catalog.py`
- `cure_lite/experiment/geometry_safe_p0_bc_protocol.py`
- `cure_lite/experiment/coverage_transition.py`
- `cure_lite/experiment/synthetic_state_failure_attribution.py`
- `cure_lite/experiment/synthetic_state_failure_protocol.py`
- `tools/run_p0_diagnostics.py`
- `tools/run_geometry_safe_p0.py`
- `tools/run_geometry_safe_p0_bc.py`
- `tools/run_synthetic_state_failure_attribution.py`
- `protocols/IRSTD-1K/geometry_safe_p0_bc_v1/config.json`
- `protocols/IRSTD-1K/synthetic_state_failure_attribution_v1/config.json`
- `tests/test_geometry_safe_p0_bc_protocol.py`
- `tests/test_geometry_safe_p0_bc_core.py`
- `tests/test_coverage_transition.py`
- `tests/test_synthetic_state_failure_attribution_core.py`
- `tests/test_synthetic_state_failure_protocol.py`
- `tests/test_synthetic_state_failure_runner.py`
- `cure_lite/paired_types.py`
- `cure_lite/paired_losses.py`
- `cure_lite/train/paired_step.py`
- `cure_lite/train/paired_pools.py`
- `cure_lite/experiment/paired_catalog.py`
- `cure_lite/experiment/paired_exposure.py`
- `cure_lite/experiment/paired_preflight.py`
- `tests/test_paired_catalog.py`
- `tests/test_paired_losses.py`
- `tests/test_paired_step.py`
- `tests/test_paired_schedule.py`
- `tests/test_paired_preflight.py`
- `tests/test_paired_toy_overfit.py`
- 对应协议配置与测试。

A0/A1 源码、协议、测试和正式产物位于提交 `a9dcfb2`。P0-B/C follow-on 与 failure-attribution 的实现、测试、正式产物和本 Markdown 位于当前工作树，尚未提交；因此不得把 `a9dcfb2` 单独描述为包含这些结果的代码快照。

### 11.3 不作为证据的运行

以下目录含 `.incomplete` 或已明确归档，不进入任何正式结果：

- `runs/irstd1k_stage_a_seed42/cure_lite_stage_a_v1`
- `runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v2`
- `runs/irstd1k_stage_a_seed42/aborted/cure_lite_stage_a_fx_v1_pre_catalog_20260723`

其中 `cure_lite_stage_a_fx_v2` 因 CUDA 初始化失败而在 0 次参数更新处终止，没有产生可解释的科学结果；对应失败记录位于 `protocols/IRSTD-1K/stage_a_seed42_fx_v2/failure_record.json`。

正式 v0.1 数值只取 `fx_v3` 与 `fx_v3_s43`；正式 v0.2 数值只取 `m_v02_s42` 与 `m_v02_s43`；正式 P0-v1 只取 r3/r4；正式 P0-v2 A0/A1 只取 r1/r2；正式 geometry-safe P0-B/C follow-on 只取 `cure_lite_geometry_safe_p0_bc_v1_r1/r2`；正式 failure attribution 只取 `cure_lite_synthetic_state_failure_attribution_v1_r1/r2`。

## 12. 当前可以和不可以声明的结论

| 声明 | 当前证据 | 状态 |
| --- | --- | --- |
| CURE core 有 \((p_b,F_b)\) 通用 cache 边界 | 代码与 cache contract | 支持软件接口层声明 |
| CURE 已能即插即用到任意 IRSTD | 尚无其他 backbone 实验 | **不能声明** |
| U 优于 factual-only learning | 两个 seed 均未严格超过 F | **不支持** |
| M 优于 U | 两个 seed 的 Pd 和找回数都更低 | **被当前结果否定** |
| M 是配对修正学习 | 配对不进入 loss 或 decoder 条件 | **不成立** |
| 一维硬最近邻导致集中 | 16/209 support，ESS 9.4815 | 支持 |
| 原 209-target population 全部具有稳定目标身份 | P0-v1 发现 merge/split/area failure | **不成立** |
| geometry-safe 分析人口构造成功 | A1：32 factual + 206 legal，完整记账 | 支持 |
| geometry-safe factual/legal 在预声明低维空间通过经验支持筛查 | P0-B：2/32 与 16/32，均低于 29/32 | **未通过**；不能外推到完整高维状态空间 |
| 当前 legal synthetic state 没有冻结探针可观察的角色信号 | handcrafted AUC≈1；两空间 MMD summary 高于 q95 | **不成立**；不作因果或显著性声明 |
| `P` 是失败主因 | `P` 强预测角色，但它参与角色定义且不是 decoder 输入 | **不能声明** |
| 局部 frozen feature 残留是失败主因 | `F_local` 全人口与两个小样本分层均未形成稳定 strong 结论 | **不能声明，也不授权 feature attenuation** |
| context/global feature 摘要携带角色信息 | `F_background_global` 在全人口和 shared groups 为 strong | 支持低维预测信号声明；不支持因果归因 |
| occupancy 是失败主因 | `O` 只在 selected-source transductive sensitivity 为 strong | **不能声明** |
| 23→16 是三个目标被直接删除造成 | 固定旧表示的单独/联合删除均保持 23/32；重拟合表示后才变为 16/32 | **不成立** |
| feature–occupancy 联合状态已被证明是失败原因 | 代码证明 feature 不变而 occupancy 被编辑，但没有 interaction-specific estimand | **不能声明**；只保留结构性假设 |
| 当前存在满足五项门槛的 transformation 假设 | hypothesis review outcome = H0 | **不成立**；当前不授权 transformation protocol |
| factual/legal compatibility score 是新的核心对象 | 若只进入 \(q_i\)，仍是边缘重采样；若不进计算图，只是诊断量 | **不成立** |
| \(\Delta_gQ\) 已被数学定义为同源 coverage response | core-object proposal 定义 sigmoid 后、hard mask 前的 \(Q\)、legal before/after pair、实际标签增量 \(D_g\) 与 clean-pair 非干扰条件 | 支持“概念候选已定义”；不等于机制成立 |
| 旧 P0-B/C 直接否定 \(\Delta_gQ\) | 新候选不要求 deleted endpoint 与 factual miss 可交换 | **不成立**；P0-B/C 仍只否定旧 surrogate/reweighting 路线 |
| paired objective 已被固定并实现为直接消费 \(\Delta_gQ\) 的不可分解目标 | `PairExample/PairBatch`、catalog、loss、train step、32,000-step schedule、真实 preflight、bounded learnability 和四个正式训练任务均完成；全套 723 tests passed | 支持“当前版本已被完整实现和检验”；不支持其性能机制 |
| 当前 `paired_difference` 严格优于独立 endpoint ERM | seed 42 为 `-7 TP/-7 recovered`，seed 43 为 `0/0` | **不成立**；当前版本 Wave A 正式失败 |
| \(\Delta_gQ\) 已排除 occupancy-only 或其他替代解释 | Wave A 已在本波比较 independent-endpoint；后续 controls 所属 Wave B/C 因 Wave A 失败未获授权 | **不能声明**；不能把未运行的后续波写成已否定全部 controls |
| S 可以设计或训练 | P0-B/C 明确失败，P0-D 未运行 | **不能设计、构造或训练** |
| CURE-Lite 已设计成功 | 当前 paired 实例已完成正式 seed 42/43 验证，但两个 seed 均未通过冻结增益门槛 | **不能声明**；否定的是当前实例，不是整个 CURE-Lite 研究空间 |
| Full CURE 已存在 | 尚未设计或实现 | **不能声明** |
| 当前结果足以支持 ICLR 投稿主张 | 无冻结机制、独立划分、\(D_T\)、多数据集或跨 backbone 证据 | **尚不足** |

## 13. 当前阻断点与恢复后的唯一主线

Phase 1、\(D_R\)-only failure attribution、独立 hypothesis review 与 core-object redefinition 均已完成。H0 保持不变；后续重定义阶段保留了一个概念候选：

\[
\Delta_gQ_\theta
=
Q_\theta(F,\Pi(O\setminus C_g))
-
Q_\theta(F,\Pi O).
\]

它表示同一 source、同一 frozen feature 下的离散 coverage response，不表示 deleted endpoint 与 factual miss 可交换。

H0 不否定 CURE 总方向。总主线仍然是：

```text
完成 CURE-Lite 最小核心机制
  -> 冻结确认机制成立
  -> 设计 Full CURE
  -> 跨 IRSTD backbone 与三数据集验证
```

当前执行边界是：

1. 冻结 P0-B/C follow-on、failure-attribution 与 hypothesis-review 的代码/配置、正式产物和解释边界；
2. 停止现有 occupancy-deletion synthetic paradigm 的 transformation 授权，不实施 feature attenuation、context replacement 或 occupancy 修改；
3. \(\Delta_gQ\) 是本次已经受检验的 core-object candidate；条件 feature evidence 与 factual/legal compatibility 只作诊断，不并行形成模块；
4. paired-objective protocol、实现和正式结果全部冻结，不能在原版本上修改 loss、pair、decoder、阈值或选择性重跑；
5. 单元、梯度、hard-mask、batched-forward、toy、真实 \(D_R\) preflight、
   matched-control static preflight、proposed bounded learnability、空间尾部
   companion、8-control bounded engineering execution、四个正式 800-epoch
   训练任务和一次性 Wave A reveal 均已闭合；
6. seed 42 为 `-7 TP/-7 recovered`，seed 43 为 `0/0`，两个 seed 均未满足
   `+2/+2`，故正式决定为 `PERFORMANCE_FAIL`；
7. 按冻结停止规则，不运行 Wave B/C、确认种子或未使用划分，也不开始
   Full CURE 和跨 backbone 验证；
8. 下一步只能先做不改写结果的失败归因。若归因导出新的单一核心机制，它必须
   作为新版本重新定义、冻结和验证；总研究阶段仍保持
   CURE-Lite → 冻结确认 → Full CURE → 跨 detector/三数据集。

## 14. 权威产物索引

### 14.1 v0.1

- seed 42：
  - [results](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3/receipts/results.json)
  - [calibration](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3/receipts/calibration.json)
  - [COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3/COMPLETE.json)
  - [assessment](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3.assessment.json)
- seed 43：
  - [results](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3_s43/receipts/results.json)
  - [calibration](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3_s43/receipts/calibration.json)
  - [COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3_s43/COMPLETE.json)
  - [assessment](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3_s43.assessment.json)
- [F/U mechanism diagnostic](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3_mechanism_diagnostic.json)

### 14.2 v0.2 M

- seed 42：
  - [alignment](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_s42/receipts/alignment.json)
  - [results](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_s42/receipts/results.json)
  - [calibration](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_s42/receipts/calibration.json)
  - [gate](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_s42/receipts/gate.json)
  - [COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_s42/COMPLETE.json)
- seed 43：
  - [alignment](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_s43/receipts/alignment.json)
  - [results](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_s43/receipts/results.json)
  - [calibration](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_s43/receipts/calibration.json)
  - [gate](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_s43/receipts/gate.json)
  - [COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_s43/COMPLETE.json)

### 14.3 P0-v1

- [frozen config](protocols/IRSTD-1K/p0_v1/p0_config.json)
- [r3 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_p0_v1_r3/COMPLETE.json)
- [r3 P0-A](runs/irstd1k_stage_a_seed42/cure_lite_p0_v1_r3/receipts/p0_a_geometry.json)
- [r3 P0-B](runs/irstd1k_stage_a_seed42/cure_lite_p0_v1_r3/receipts/p0_b_support.json)
- [r3 P0-C](runs/irstd1k_stage_a_seed42/cure_lite_p0_v1_r3/receipts/p0_c_separability.json)
- [r3 P0-D](runs/irstd1k_stage_a_seed42/cure_lite_p0_v1_r3/receipts/p0_d_exposure.json)
- [r3 decision](runs/irstd1k_stage_a_seed42/cure_lite_p0_v1_r3/receipts/decision.json)
- [r4 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_p0_v1_r4/COMPLETE.json)

### 14.4 geometry-safe P0-v2

- [frozen config](protocols/IRSTD-1K/geometry_safe_p0_v2/config.json)
- [r1 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_v2_r1/COMPLETE.json)
- [r1 geometry catalog](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_v2_r1/receipts/geometry_catalog.json)
- [r1 P0-A0](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_v2_r1/receipts/p0_a0_dataset_geometry_audit.json)
- [r1 P0-A1](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_v2_r1/receipts/p0_a1_population_eligibility.json)
- [r1 eligible view](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_v2_r1/receipts/eligible_view.json)
- [r1 decision](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_v2_r1/receipts/decision.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_v2_r2/COMPLETE.json)

### 14.5 geometry-safe P0-B/C follow-on

- [frozen config](protocols/IRSTD-1K/geometry_safe_p0_bc_v1/config.json)
- [r1 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r1/COMPLETE.json)
- [r1 population binding](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r1/receipts/population_binding.json)
- [r1 group accounting](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r1/receipts/group_accounting.json)
- [r1 P0-B](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r1/receipts/p0_b_support.json)
- [r1 P0-C](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r1/receipts/p0_c_screening.json)
- [r1 decision](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r1/receipts/decision.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r2/COMPLETE.json)

### 14.6 synthetic-state failure attribution

- [frozen config](protocols/IRSTD-1K/synthetic_state_failure_attribution_v1/config.json)
- [r1 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/COMPLETE.json)
- [r1 authority binding](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/authority_binding.json)
- [r1 population and factor inventory](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/population_factor_inventory.json)
- [r1 state contract audit](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/state_contract_audit.json)
- [r1 frozen feature evidence](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/frozen_feature_evidence.json)
- [r1 factor probe profile](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/factor_probe_profile.json)
- [r1 composition strata](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/composition_strata.json)
- [r1 coverage transition](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/coverage_transition_decomposition.json)
- [r1 factual signatures](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/factual_miss_signatures.json)
- [r1 diagnostic decision](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/diagnostic_decision.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r2/COMPLETE.json)

### 14.7 synthetic-state hypothesis review

- [H0 proposal receipt](protocols/IRSTD-1K/synthetic_state_hypothesis_review_v1/proposal_receipt.json)
- [完整机制假设审查](CURE_Lite_机制假设审查.md)

该阶段是绑定既有代码和正式 evidence receipts 的只读研究判断，不是新实验，因此不位于 `runs/`，也不生成 `COMPLETE.json`。

### 14.8 core learning-object redefinition

- [core-object proposal receipt](protocols/IRSTD-1K/core_learning_object_redefinition_v1/proposal_receipt.json)
- [完整核心学习对象重定义](CURE_Lite_核心学习对象重定义.md)

该阶段只定义 \(\Delta_gQ\) 的数学语义、退化反例、未来 controls、性能门槛和权限边界。在该历史时点尚未实现 pairwise objective，因此同样不位于 `runs/`，也不生成 `COMPLETE.json`；后续 additive 实现和运行证据单列如下。

### 14.9 paired-objective protocol

- [paired-objective proposal receipt](protocols/IRSTD-1K/paired_objective_v1/proposal_receipt.json)
- [完整 Paired Objective 协议](CURE_Lite_Paired_Objective_协议.md)

该阶段把 paired loss、双端梯度、零阶锚、response domain、null/control、
未来接口和停止条件固定为一个实现规范。receipt fingerprint 为
`5a2f357911fb5f1dc1a946b3dbad429d256c390677d238b2f395fe90ce91fac8`。
它不包含训练结果；协议时点“未授权实现或训练”的历史状态不反写。
后续 additive pair catalog、paired train step 和 schedule 已实现；本
proposal receipt 的历史权限状态不反写。

### 14.10 real \(D_R\) paired preflight

- [r1 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_preflight_v1_r1/COMPLETE.json)
- [r1 pair catalog manifest](runs/irstd1k_stage_a_seed42/cure_lite_paired_preflight_v1_r1/pair_preflight/pair_catalog_manifest.json)
- [r1 seed42 exposure](runs/irstd1k_stage_a_seed42/cure_lite_paired_preflight_v1_r1/receipts/exposure_seed42.json)
- [r1 seed43 exposure](runs/irstd1k_stage_a_seed42/cure_lite_paired_preflight_v1_r1/receipts/exposure_seed43.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_preflight_v1_r2/COMPLETE.json)

两次产物逐字节一致；`COMPLETE` fingerprint 为
`eac7f54eefd82d9194e0472c366bc5ebbd4e5b7d44b8a63fdd2e9d44a3d5bcb9`。

### 14.11 matched-control static preflight

- [r1 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_control_preflight_v1_r1/COMPLETE.json)
- [r1 DCT receipt](runs/irstd1k_stage_a_seed42/cure_lite_paired_control_preflight_v1_r1/receipts/dct_basis.json)
- [r1 permutation receipt](runs/irstd1k_stage_a_seed42/cure_lite_paired_control_preflight_v1_r1/receipts/target_permutation.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_control_preflight_v1_r2/COMPLETE.json)

两次产物逐字节一致；`COMPLETE` fingerprint 为
`da39a286c3d117b6d25d6ecb52af2eccef06609d966d06517037cd30c4707cfc`。

### 14.12 bounded \(D_R\) learnability

- [frozen config](protocols/IRSTD-1K/paired_bounded_learnability_v1/config.json)
- [r1 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_bounded_learnability_v1_r1/COMPLETE.json)
- [r1 result](runs/irstd1k_stage_a_seed42/cure_lite_paired_bounded_learnability_v1_r1/receipts/result.json)
- [r1 decision](runs/irstd1k_stage_a_seed42/cure_lite_paired_bounded_learnability_v1_r1/receipts/decision.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_bounded_learnability_v1_r2/COMPLETE.json)

两次 GPU0 运行逐字节一致；`COMPLETE` fingerprint 为
`c4ddf1d87d04ab57b574d9abc4f4602a75fa43ee9cfa3e94fb54698322d02215`。
该产物明确记录 `not_performance_evidence=true` 和
`authorizes_formal_800=false`。

### 14.13 paired spatial-tail companion

- [frozen config](protocols/IRSTD-1K/paired_spatial_tail_diagnostic_v1/config.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_spatial_tail_diagnostic_v1_r2/COMPLETE.json)
- [r2 result](runs/irstd1k_stage_a_seed42/cure_lite_paired_spatial_tail_diagnostic_v1_r2/receipts/result.json)
- [r3 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_spatial_tail_diagnostic_v1_r3/COMPLETE.json)

r2/r3 的全部文件逐字节一致。`COMPLETE` fingerprint 为
`6f44613020d3930d1b2379797395a2cd96e045c66637714de7c152e0662eb13d`，
两个 `COMPLETE.json` 的 SHA256 均为
`366f532719ea4d61d45225cc634b3ce7ec39211a67c39413e1f926706860e700`。

该 companion 从 fresh decoder 精确重放既有 400 updates，只描述
`clean_positive/component_null/identity_null` 的空间响应，不新增门槛。
最终 component-null 的 16 个 pair 中，13 个的最大绝对响应至少为 0.5；
全部 16 个绝对峰值位于被删组件 Chebyshev 半径 2 内，14 个峰值直接落在
投影变化 feature-cell 的输出支持内。阈值 0.5 上共有 236 个响应像素，
其中 220 个位于被删组件半径 2 内，201 个落在投影变化 cell 支持内。

这支持“强 component-null 尾部是目标局部的 coverage-deletion response”
这一描述，不支持自然漏检恢复、低误报或正式性能结论。它同时保留一个明确
风险：模型可能学习恢复被人为删除的已检目标，而未必迁移到真实漏检。
因此该产物记录 `authorizes_formal_800=false`，其风险只能由 matched
controls 与自然漏检正式结果继续区分。

历史 r1 封存了一次已定位的执行错误，不作为数值证据，也未被覆盖。

### 14.14 8-control bounded engineering execution

- [frozen config](protocols/IRSTD-1K/paired_control_bounded_execution_v1/config.json)
- [r1 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_control_bounded_execution_v1_r1/COMPLETE.json)
- [r1 result](runs/irstd1k_stage_a_seed42/cure_lite_paired_control_bounded_execution_v1_r1/receipts/result.json)
- [r1 decision](runs/irstd1k_stage_a_seed42/cure_lite_paired_control_bounded_execution_v1_r1/receipts/decision.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_control_bounded_execution_v1_r2/COMPLETE.json)

一次运行按固定顺序执行全部 8 个 controls：

```text
independent_endpoint
after_only
zero_feature
coordinate_basis
feature_only
target_permutation
plus_detach
minus_detach
```

每个 control 使用 fresh、逐字节相同的初始 decoder，完成 400 updates、
1,200 decoder forwards 和 4,800 state evaluations。合计完成 3,200
updates、9,600 forwards 和 38,400 states；全部 loss/gradient 有限，
每次更新的总梯度非零，且每个 control 的 decoder 参数状态均发生变化。target permutation
的 donor/recipient/target 运行时身份也与静态 preflight 完整闭合。

r1/r2 目录逐文件一致；`COMPLETE` fingerprint 为
`860f23f4a93df83e1939b6b810d8f0caa234526eac5d3b7ad67dca6fbb523a14`，
两个 `COMPLETE.json` 的 SHA256 均为
`52945a3f486937f60f87c75a76ca0ffbdd8c4eff0342038df1fbbc566810175d`，
两个 `result.json` 的 SHA256 均为
`9f8f5f2cd72e0eb620962e634edbdbc5be73df7745ddfa2bd399da36af7592be`。

正式决定为 `ENGINEERING_EXECUTION_PASS`。它仅证明 8 个对照不是纸面定义，
而能在与 proposed 一致的 `4+4+2`、`1:1:1` 和计算预算下实际训练。
配置预先固定 `require_positive_response_learning=false`，因此不得把本结果
写成某个 control 性能更好或更差，也不得据此授权 \(D_V/D_T\)、正式
800 epoch、Full CURE 或其他 backbone。

### 14.15 formal schedule preflight

- [frozen config](protocols/IRSTD-1K/paired_formal_preflight_v1/config.json)
- [r1 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_preflight_v1_r1/COMPLETE.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_preflight_v1_r2/COMPLETE.json)
- [r1 preflight receipt](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_preflight_v1_r1/preflight_receipt.json)

r1/r2 六个文件逐字节一致；`COMPLETE` fingerprint 为
`6cbee7c0ae65fb18054f9da1681b0a9150a639aebf829e6c33d9687a15fe41dd`。
seed 42/43 的 formal schedule fingerprint 分别为
`35ed3c818c1e126b4bdb2ae584b8c296795c60493862d558dae2edd224ba2309`
与
`f8485dfca0bc531f97cc7ad18b216b99f662cb71eb1569b6b8601ff3d5fe50c4`。
九种方法共享 binding
`18385315d26ee81ba1ae3f040c7850fa4fdd34ff0d8ee4f723fd292e3da61331`。

每个 seed 固定 800×40、32,000 updates、每步 4 factual-miss +
4 factual-no-miss + 2 clean pairs、12 states、3 forwards；206 个 pairs、
32 个 miss anchors 和 135 个 no-miss anchors 均具有非零暴露。
该产物仅封存 schedule，不训练、不推理、不校准，也不授权正式训练。

### 14.16 formal evidence closure 与 no-resume runner

- [common comparison protocol](protocols/IRSTD-1K/paired_formal_evaluation_v1/config.json)
- [no-resume runner config](protocols/IRSTD-1K/paired_formal_runner_v1/config.json)
- [formal runner](cure_lite/experiment/paired_formal_runner.py)
- [formal training engine](cure_lite/experiment/paired_formal_training.py)
- [formal control provider](cure_lite/experiment/paired_formal_controls.py)
- [formal evaluation contract](cure_lite/experiment/paired_formal_evaluation.py)
- [formal wave decision](cure_lite/experiment/paired_formal_decision.py)

共同 comparison protocol fingerprint 为
`cb2fb09c3ec7dbbb0f057d94f7f159e2b4a733296e6ea4a144d6302387014884`，
固定 IRSTD-1K \(D_V\) 的 120 张有序图像、170 个 targets、147 个已覆盖
targets、23 个固定漏检、51 个 residual thresholds、统一 FA/retention
预算和历史 fx_v3 双 seed 来源。审计中出现的“100 张”假设经权威产物核查
被纠正为 120 张，属于事实纠正。

runner config fingerprint 为
`760940a9f1e6c370ee869653205e6f60b0333501d33055eb80797f8a8ce1bd23`，
文件 SHA256 为
`1ed3f898bcaebb1f7190bef5bf37b5243049d3e0f2a18933fa0a1c65394bfaed`。
seed 42/43 的完整运行时训练人口 fingerprint 分别为
`3e8cf36b63693fcb27b3ea33c9ae55ebb6ac25a6ed1d56ca30b04ae4868028f5`
与
`31fa83894c60357ba89dcbd9f9355bf94a019a1b01879d463aac599019decc5f`。
它们动态覆盖 factual 与 actual scheduled pair tensors，并在训练前后重算。
control provider fingerprint 为
`5bc73c4c74873468c204d81f4bf2a1b081285b67304eb89112e71077f0b11376`，
且进入 control decoder artifact；`paired_difference` 必须保持 provider
为 null。

严格 \(D_R\)-only dry validation 已通过；9 methods×2 seeds 的配置、
共同 preflight、完整训练人口、provider、公共 comparison protocol 和实现
哈希闭合。专项合并回归为 `59 passed`，全仓回归为
`699 passed in 174.08s`。本阶段没有执行 optimizer update，没有创建正式
训练输出，也没有读取新的 \(D_V/D_T\)。因此它是正式实验的执行资格，
不是性能结果或创新成立结论。

### 14.17 formal Wave A 训练与一次性揭示

四个无 checkpoint/resume 的正式训练任务均完成 800 epoch、32,000 updates：

- [paired_difference seed 42](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_wave_a_paired_difference_seed42_r1)
- [paired_difference seed 43](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_wave_a_paired_difference_seed43_r1)
- [independent_endpoint seed 42](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_wave_a_independent_endpoint_seed42_r1)
- [independent_endpoint seed 43](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_wave_a_independent_endpoint_seed43_r1)

随后只按冻结配置执行一次 \(D_V\) reveal：

- [reveal config](protocols/IRSTD-1K/paired_formal_wave_a_reveal_v1/config.json)
- [published reveal](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_wave_a_reveal_v1_r1)
- [decision](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_wave_a_reveal_v1_r1/decision.json)
- [COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_wave_a_reveal_v1_r1/COMPLETE.json)

正式结果为：

| seed | 方法 | TP / 170 | Pd | recovered / 23 | retention | pixel FA | raw-bg FA | FP comp./MP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `paired_difference` | 147 | 0.864706 | 0 | 1.0 | \(2.1108\times10^{-5}\) | \(7.1716\times10^{-5}\) | 3.0518 |
| 42 | `independent_endpoint` | 154 | 0.905882 | 7 | 1.0 | \(2.8737\times10^{-5}\) | \(8.6339\times10^{-5}\) | 5.2134 |
| 43 | `paired_difference` | 152 | 0.894118 | 5 | 1.0 | \(3.3951\times10^{-5}\) | \(8.4813\times10^{-5}\) | 6.1035 |
| 43 | `independent_endpoint` | 152 | 0.894118 | 5 | 1.0 | \(2.6576\times10^{-5}\) | \(7.8583\times10^{-5}\) | 5.8492 |

历史冻结方法也进入同一逐 seed 比较。seed 42 的最佳比较值为 154 TP、
7 recovered；seed 43 为 152 TP、5 recovered。因而 proposed margins 分别
为 \(-7/-7\) 与 \(0/0\)，均未达到 `+2/+2`。所有方法均满足冻结
FA/retention 约束，所以正式结论是核心增益失败，不是约束失败。

```text
status               = PERFORMANCE_FAIL
all_seeds_pass       = false
next_action          = STOP_AND_PRESERVE_EVIDENCE
decision fingerprint = 481413dd8da3af95d4f6bcb8fc28ed001301ea83861cbf583cf27077709df28e
COMPLETE fingerprint = 4ee2c32ff6a00d437d18cad8ec14f8dd1ab790149c8cc80ef7a494c34caa66c8
```

该结果冻结的是当前 `paired_difference` 实例的失败，不是 CURE 总研究方向
失败。它不授权降低门槛、选择 seed、修改当前版本后重跑、Wave B/C、冻结
确认、\(D_T\)、Full CURE 或跨 backbone。

### 14.18 OC-APTO v3 模型代码与真实 \(D_R\) bounded 结果

OC-APTO v3 已完成独立的模型训练代码，而不是只停留在方案描述：

- [outcome value objects](cure_lite/paired_outcome_types.py)
- [unified outcome loss](cure_lite/paired_outcome_losses.py)
- [4/4/2 train step](cure_lite/train/paired_outcome_step.py)
- [real input materializer](cure_lite/experiment/paired_outcome_inputs.py)
- [222-pair schedule](cure_lite/experiment/paired_outcome_schedule.py)
- [bounded executor](cure_lite/experiment/paired_outcome_bounded.py)
- [create-only runner](tools/run_paired_outcome_bounded.py)
- [frozen bounded config](protocols/IRSTD-1K/outcome_complete_apto_v3/bounded_config.json)

v3 使用独立的 factual-anchor population/schedule，只包含实际消费的 16 个
factual-miss、16 个 factual-no-miss 和 16 个 identity-null 检查项，不再
携带旧版本未使用的 clean/component pair 字段。完整 outcome population 为
206 clean-positive + 16 component-null；400 updates 共 800 pair slots，
每个 pair 暴露 3 或 4 次。相关模型、调度、运行器联合测试为
`57 passed`。

一次性真实 \(D_R\) 运行已经完成并封存：

- [published run](runs/irstd1k_stage_a_seed42/cure_lite_oc_apto_v3_bounded_r1)
- [result](runs/irstd1k_stage_a_seed42/cure_lite_oc_apto_v3_bounded_r1/receipts/result.json)
- [decision](runs/irstd1k_stage_a_seed42/cure_lite_oc_apto_v3_bounded_r1/receipts/decision.json)
- [COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_oc_apto_v3_bounded_r1/COMPLETE.json)

结构执行全部通过：400 次更新、每步 3 次 decoder forward/12 states、
一次 backward/optimizer step、有限非零梯度、完整暴露账本和训练前后输入
核对均成立。模型门槛结果为：

| bounded 指标 | 观测值 | 门槛 | 结果 |
| --- | ---: | ---: | --- |
| factual-miss loss final/initial | 0.552662 | \(\le 0.75\) | pass |
| factual-no-miss loss final/initial | 0.001377 | \(\le 0.75\) | pass |
| plus-baseline loss final/initial | 0.120624 | \(\le 0.75\) | pass |
| clean transition loss final/initial | 0.566457 | \(\le 0.50\) | fail |
| clean mean delta on \(D\) | 0.388573 | \(\ge 0.50\) | fail |
| clean pairs with mean delta \(\ge0.25\) | 0.776699 | \(\ge0.75\) | pass |
| clean zero-strata macro mean \(|\Delta|\) | 0.083676 | \(\le0.05\) | fail |
| component footprint macro mean \(|\Delta|\) | 0.183828 | \(\le0.05\) | fail |
| component footprint global max \(|\Delta|\) | 0.907818 | \(\le0.25\) | fail |
| component context macro mean \(|\Delta|\) | \(2.2179\times10^{-5}\) | \(\le0.05\) | pass |
| identity-null global max \(|\Delta|\) | 0 | \(\le10^{-7}\) | pass |

```text
decision             = BOUNDED_MODEL_CODE_GATE_FAIL
structural pass      = true
model-code gate pass = false
COMPLETE fingerprint = 658f2ac7e26476003d9222342d6c73b82f5220301c0e137bb686f0f07cff4d17
```

因此 v3 的工程链路成立，但当前模型实例未设计成功。失败集中在“目标响应
不足”与“局部响应扩散”同时存在，尤其是 component-null footprint 上出现
0.907818 的最大响应；这不允许启动 800 epoch。该运行没有读取
\(D_V/D_T\)，没有校准、检测性能评估、Base/backbone 更新、resume 或自动
重试。下一步回到 decoder 结构代码，不能通过修改本次门槛或重复运行 v3
寻找正结果。

### 14.19 NLCC-v12 runner/evidence r2 与 Development

- [runner/evidence r2 amendment](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/dataset_free_runner_evidence_r2_amendment.json)
- [implementation closure](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/runner_implementation_closure_r1.json)
- [R0 verification receipt](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/runner_evidence_r2_r0_verification_receipt.json)
- [Development authorization](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_pre_run_authorization.json)
- [attempt](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/attempt.json)
- [training started](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/training_started.json)
- [raw result](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/result.json)
- [sealed decision](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/decision.json)
- [COMPLETE](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/COMPLETE.json)
- [正式负结果说明](CURE_Lite_NLCC_v12_Development正式负结果.md)

R0 targeted JUnit 为 43/43，repository JUnit 为 1105/1105。Development
完成 320/320 updates、960/960 training forwards 和 321/321 finite-state
audits；25/25 structural gates 通过，26/76 numeric gates 通过，八个 groups
均未整体通过。独立重算结果与 sealed decision 完全一致：

```text
decision = NLCC_V12_DEVELOPMENT_FAIL
```

Holdout pre-run authorization 与 `exposure_holdout_r1` 均不存在；后续阶段
按停止规则未运行。

## 15. 无虚构检查

- 文中所有性能数字来自仓库现有正式 receipt；
- 算术均值、差值与 ESS 由冻结数字直接计算；
- P0-v1 B/C 数值明确标注为 P0-A 失败后的调试记录；
- geometry-safe P0-B/C 数值来自逐字节一致的 r1/r2 正式 receipt；
- failure-attribution 数值来自两个同输入文件系统重放中逐字节一致的正式 receipt，不称为统计独立重复；
- AUC interval 明确限定为固定 OOF scores 的条件 group bootstrap；MMD q95 crossing 不称为显著性检验；
- 六块探针与 drop-one 结果只作低维描述，不写成因果贡献或 transformation 选择；
- hypothesis review 的 H0 只表示没有候选通过五项 transformation 门槛，不扩大为 CURE 总方向失败；
- core-object redefinition 只把 \(\Delta_gQ\) 记为概念上可继续的候选，不写成机制已成立、性能已提升或 ICLR 新颖性已确认；
- paired/control preflight 与 bounded learnability 的两次运行均只称为同输入、同环境的逐字节重放，不称为统计独立重复；
- bounded pass 只证明固定 \(D_R\) 微型人口上的计算可学习性；component-null 局部最大差分被保留为诊断，不以全图均值通过掩盖；
- P0-D、candidate S、S training、Wave B/C、冻结确认、\(D_T\)、Full CURE
  和跨 backbone 实验均明确标注为尚未发生；formal paired training 与本次
  \(D_V\) Wave A reveal 已明确标注为发生且失败；
- NLCC-v12 明确标注为 dataset-free Development 已发生且失败；其
  Holdout、真实 \(D_R\)、32,000-step exposure replay 和 formal800 明确
  标注为 `NOT_AUTHORIZED / NOT_RUN`；
- 未填写任何不存在的 NUAA-SIRST、NUDT-SIRST、\(D_T\) 或其他 detector 性能。
