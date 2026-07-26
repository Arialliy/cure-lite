# CURE-Lite 下一步方案

> 依据代码快照 `a9dcfb23f44e2cbd54849464d0d47c7e2f43e499`、geometry-safe P0、CURE-Lite v0.1/v0.2 与 synthetic-state failure-attribution 的正式产物修订。  
> 本文件以研究执行方案为主；已完成阶段的数值仅用于绑定当前决策，完整结果见 `CURE_Lite_全部结果与当前研究结论.md`。未在权威产物中出现的数值不得写成结果。

## 最新状态（2026-07-27）：NLCC-v12 停在 Development 门禁

CCFR-v11 负结果后建立的 NLCC-v12 已完成 runner/evidence r2 闭合，并执行
了唯一一次冻结的 320-update dataset-free Development：

```text
R0-C1--R0-C8                  = PASS
targeted tests                = 43 / 43
repository tests              = 1105 / 1105
updates                       = 320 / 320
training forwards             = 960 / 960
finite-state audits           = 321 / 321
structural gates              = 25 / 25 PASS
numeric gates                 = 26 / 76 PASS
final groups                  = 0 / 8 PASS
decision                      = NLCC_V12_DEVELOPMENT_FAIL
Holdout authorized            = false
real D_R authorized           = false
formal800 authorized          = false
Full CURE authorized          = false
```

这是有效的模型门禁负结果，不是执行异常，也不是真实检测性能结果。当前必须：

- 冻结 NLCC-v12 结果，不重跑、不放宽阈值；
- 不创建 Holdout authorization；
- 不读取新的真实 \(D_R/D_V/D_T\) 结果；
- 不运行 32,000-step exposure replay 或 seed 42/43 formal800；
- 不开始 Full CURE、其他 detector 或三数据集工作；
- 先对“方向正确但绝对端点、matched null、背景与 factual anchors 未同时
  成立”做固定失败归因；任何机制修改建立新的 CURE-Lite 模型版本。

完整结果见
[NLCC-v12 Development 正式负结果](CURE_Lite_NLCC_v12_Development正式负结果.md)。

总研究主线没有改变：

```text
新 CURE-Lite 核心候选
  -> dataset-free Development
  -> 独立 exposure Holdout
  -> IRSTD-1K D_R 受控验证
  -> 800×40 暴露重放
  -> seed 42/43、800 epoch 逐种子确认
  -> CURE-Lite 冻结
  -> Full CURE 设计与冻结
  -> NUAA-SIRST / NUDT-SIRST / IRSTD-1K
  -> 多个 IRSTD detector 分别重训验证
```

## 历史状态（2026-07-26）：CCFR-v11 停在 dataset-free 门禁

在历史 paired/Wave A 负结果之后建立的最新 CURE-Lite 候选是 CCFR-v11
（Coverage-Conditioned Feature Release）。其代码级 development regression
已经通过，但唯一一次 400-update dataset-free exposure holdout 已完成并得到：

```text
decision                         = CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_FAIL
execution_status                 = COMPLETED
training_contract                = PASS
updates                          = 400 / 400
forward_calls                    = 1200 / 1200
finite_nonzero_gradient_checks   = 2400 / 2400
final_groups                     = 0 / 8 passed
population_objective             = 1.6601788997650146; required < 0.1
factual_miss_target              = PASS
factual_miss_background          = FAIL
factual_no_miss                  = FAIL
real_D_R_authorized              = false
formal_800_authorized            = false
full_CURE_authorized             = false
cross_detector_authorized        = false
```

该结果是完整模型确认负结果，不是程序异常，也不是真实数据集检测性能结果。
当前 CCFR-v11 在冻结 holdout 上呈现一致的 paired 正向符号趋势，但没有同时
建立绝对目标端点、背景/no-miss 抑制和非目标区域稳定性。因此：

- CCFR-v11 不得重跑、放宽阈值或直接进入 IRSTD-1K `D_R`；
- 不得借用旧 factorized、paired 或 Stage-A runner 冒充 CCFR 真实验证；
- 不运行 32,000-step、seed 42/43 的 800 epoch、Full CURE、其他 detector
  或三数据集；
- 当前只冻结负结果并重新设计一个新的 CURE-Lite 核心候选；新候选必须重新
  通过同等级 dataset-free 门禁，才开始真实 `D_R` 验证。

正式结果、阈值和证据链见
[CCFR-v11 Exposure Holdout 正式负结果](CURE_Lite_CCFR_v11_Exposure_Holdout_正式负结果.md)。

总研究主线没有改变：

```text
新 CURE-Lite 核心候选
  -> dataset-free 冻结门禁
  -> IRSTD-1K D_R 受控验证
  -> 800×40 暴露重放
  -> seed 42/43、800 epoch 逐种子确认
  -> CURE-Lite 冻结
  -> Full CURE 设计与冻结
  -> NUAA-SIRST / NUDT-SIRST / IRSTD-1K
  -> 多个 IRSTD detector 分别重训验证
```

## 0. 历史权威状态（2026-07-24，归档）

下文第 1～7、9、11～12 节保留 Phase 1 正式运行前冻结的协议与决策规则，
属于归档记录；第 8、10 节记录从 Phase 1 负结果转入 failure attribution
的决策桥接；第 13 节记录已经完成的独立 hypothesis review；第 14 节记录
core-object redefinition、paired 实现和正式 Wave A 门禁。Phase 1、
\(D_R\)-only synthetic-state failure attribution、独立 hypothesis review、
core-object redefinition、四个 800-epoch 训练任务和一次性 Wave A 揭示均已
完成。

正式结果为：

```text
P0-A1 = pass
P0-B  = fail
  handcrafted coverage = 2/32
  decoder-joint coverage = 16/32
  required = 29/32 in each space
P0-C  = fail
P0-D  = not_evaluated
eligible_to_design_candidate_s = false
failure attribution execution = partial_inconclusive
strong full-population blocks  = P, F_background_global
computationally inconclusive   = 5 predeclared small-stratum probes
hypothesis review outcome      = H0
transformation authorized      = false
candidate S authorized         = false
core-object candidate           = same-source discrete coverage response
paired-objective protocol       = frozen
pairwise implementation         = core_implemented
paired static/toy tests          = passed
real D_R paired preflight        = passed; 206 clean + 16 component-null + 160 identity-null
matched-control static preflight = passed; permutation READY (206/206)
bounded D_R learnability         = passed; r1/r2 byte-identical
spatial-tail companion           = completed; successful r2/r3 byte-identical
8-control bounded execution      = ENGINEERING_EXECUTION_PASS; r1/r2 byte-identical
formal 800x40 joint schedule      = implemented and tested
formal schedule preflight         = passed; seed42/43 r1/r2 byte-identical
paired create-only artifact       = implemented and tested
formal paired training engine     = implemented and tested
formal matched-control provider   = implemented; real D_R static closure passed
formal D_V comparison protocol    = frozen; 120 images / 170 / 147 / 23
formal no-resume runner           = implemented; strict D_R-only dry validation passed
formal runner config fingerprint  = 760940a9f1e6c370ee869653205e6f60b0333501d33055eb80797f8a8ce1bd23
full software regression          = 723 passed
formal paired/control training    = completed; 4 x 800 epochs, no resume
Wave A reveal                     = completed and atomically published
Wave A decision                   = PERFORMANCE_FAIL
seed 42 paired_difference         = 147/170 TP; 0/23 recovered
seed 42 best comparator           = 154/170 TP; 7/23 recovered
seed 43 paired_difference         = 152/170 TP; 5/23 recovered
seed 43 best comparator           = 152/170 TP; 5/23 recovered
all FA/retention constraints      = passed
current paired version            = stopped and preserved
Full CURE / cross-backbone        = not authorized
next_route                        = failure attribution before any new version
```

两个阶段各自的 r1/r2 均是在两个独立输出目录中的同输入文件系统重放，全部文件逐字节一致；这证明确定性，不是统计独立重复。

当前不授权 transformation、S、P0-D、\(D_T\)、Full CURE 或其他
backbone。seed 42/43 的 `paired_difference` 与
`independent_endpoint` 已分别完成 800 epoch、32,000 updates；随后共同
comparison protocol 只揭示一次 \(D_V\)，正式决定为
`PERFORMANCE_FAIL`。两个 seed 的 FA、retention 和 budget 均通过，失败来自
核心增益门槛：seed 42 落后最佳比较方法 7 个真目标和 7 个固定漏检，seed 43
与最佳比较方法持平，均未达到预声明的 `+2/+2`。

因此，当前 paired 版本必须停止并保留，不能在同一版本内调 loss、结构、
pair、阈值或只报告均值。下一项工作不是 Wave B、确认实验、Full CURE 或跨
backbone，而是对冻结训练/评估证据作机制失败归因；只有归因能够导出一个
不依赖 \(D_V\) 反复试验、仍服务于同一 CURE 核心问题的单机制假设时，才可
另立并预冻结新版本。

早期 failure attribution 只筛查冻结低维摘要中的角色预测信号，不能确定
“主要原因”：

- `P` 是与角色定义耦合的 Base selection proxy，不是 decoder 输入；
- `F_background_global` 在全人口与 shared groups 中携带 strong role signal，但与 `F_local` 共享背景环信息；
- `O` 只在 selected dual-role-source 的 transductive sensitivity 中为 strong；
- `F_local` 没有形成稳定 strong 结论；
- MMD 只作 observed-summary 对 legal-reference q95 的描述性 crossing，不是显著性检验；
- drop-one 只有条件预测 log-loss 点差，不是因果消融或 transformation 排名。

独立 hypothesis review 已完成。代码证明当前 legal synthetic state 是“完整保留 observational frozen feature、只删除 occupancy 组件”的联合状态，但现有证据不能把这种结构事实识别为失败原因，也不能唯一选择 feature 或 occupancy 变换。五项门槛没有同时通过，因此正式结论为 H0。

后续 core-object redefinition 也已完成。唯一保留候选是同源离散覆盖响应：

\[
\Delta_gQ
=
Q(F,\Pi(O\setminus C_g))
-
Q(F,\Pi O).
\]

它不把 deleted endpoint 当作 factual miss，而把同一 source 的 before/after
pair 当作 completion operator 的离散响应观测。paired-objective protocol
现已冻结；additive 训练核心、真实产物封存、matched-control 静态预检和
真实 proposed/control bounded execution 均已完成；正式联合 schedule 与
paired artifact contract 也已实现；正式 800-epoch Wave A 性能证据现已产生，
并否定了当前 paired 版本的预声明机制支持。

这不修改 CURE 的总研究主线与创新目标：最终目标仍是
detector-independent correction；CURE-Lite 仍是验证核心机制的第一阶段。
已经停止的包括旧边缘重加权路线和本次未通过 Wave A 的 paired 版本；这两项
负结果都不能被扩大为 CURE 总体方向失败，也不能被删除或事后改写。

### 0.1 主线变更边界

**不可变的总研究主线**：先在 CURE-Lite 中建立一个与特定检测器解耦、
能够在冻结 FA/retention 约束下稳定找回 natural misses 的最小核心机制；
然后做冻结确认；确认通过后才设计 Full CURE；Full CURE 定义冻结后才在
多个 IRSTD detector 和 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 上分别重训验证。
通用 \((p_b,F_b)\) 边界、单次 Base + 单次修正推理以及不依赖某个
backbone 内部拓扑，仍是方法边界。

**已经冻结且不得回写的当前版本**：共享
coverage-conditioned residual completion operator
\(Q_\theta(F,\Pi O)\)、factual/no-miss 绝对锚、同源
\(\Delta_gQ=Q^- - Q^+\) paired objective，以及一次 decoder、occupancy
hard mask、冻结阈值和 hard union。Wave A 已证明这个具体版本没有获得稳定
机制支持。它现在是不可修改的负结果，不再是等待继续调参的活动实现。

“总研究主线不变”不等于“失败的技术节点永远不能被替换”。若失败归因支持
新的核心机制，必须新建版本、说明它与 CURE 问题定义的必然联系、保持单机制，
并在查看其性能前冻结协议。否则就会把坚持主线误解成维护一个已被否定的实现。

**允许的工程修正**：真实 cache loader/runner、shape/dtype/device 适配、
批处理和重复计算消除、确定性 schedule、fingerprint/receipt、目录封存、
错误修复与测试补齐；这些修改不得改变 \(Q\)、pair 语义、目标函数、decoder
拓扑或推理图。

**必须另行决策并先冻结协议的机制变更**：修改 decoder 拓扑或 feature-tap
语义、加入 attention/Transformer/多尺度/第二 decoder/feature editor、新增
训练分支或改变 loss、权重、stop-gradient、response domain、pair target，
恢复 factual-to-legal matching 或 S/reweighting、联合训练 Base、修改校准、
推理或 hard union、绑定某个特定 detector，以及开始 Full CURE。任何此类
变化都不能以“工程修正”名义进入当前 Lite 核心。

### 0.2 创新主张冻结边界

本次已检验但未获支持的创新假设不是“通用后处理”“冻结 Base”“轻量 decoder”
“掩码修复”“差分学习”或“降低 FA”中的任意单项。上述表述均已有直接或
相邻工作，不能作为 CURE 的核心新意。当前版本冻结并接受过 Wave A 检验的
机制组合是：

1. 在同一 source、同一冻结 feature 和同一 decoder 参数下，只对 detector
   coverage state 做具有稳定 target lineage 的受控干预；
2. 直接优化 hard mask 之前的耦合有限差分
   \(\Delta_gQ=Q(F,\Pi O^-)-Q(F,\Pi O^+)\)，而不是分别拟合两个 endpoint，
   也不是根据 pair 统计做边缘重采样；
3. 同时使用 factual-miss 与 factual-no-miss 零阶锚，约束绝对输出，避免仅靠
   任意整体平移满足差分；
4. 在冻结 Base、阈值、FA budget 与 detected-target retention 约束下，以
   natural-miss recovery 作为主要判定，而不是只报告分割均值或合成目标重建。

因此，attention、Transformer、多尺度或额外 decoder 即使可能增加容量，也
不得作为当前版本的创新补丁。本次正式证据要求：

- `paired_difference` 逐 seed 优于 independent-endpoint、after-only、
  target-permutation、两端 detach、zero-feature、coordinate-basis 和
  feature-only 等 matched controls；
- 提升同时表现为 Pd 与固定 factual misses 找回数增加，且不违反 FA 与
  retention 门槛；
- 冻结确认阶段能够排除一般容量、阈值移动和通用 mask refiner 的替代解释；
- 只有上述机制在 CURE-Lite 中成立，才进入 Full CURE 与跨 detector/
  三数据集验证。

实际 Wave A 未满足前两项，故当前组合不能再作为“已经成立的 CURE 创新点”
表述。可保留的只是研究问题、受控 coverage intervention 的候选洞见和完整
负证据；创新主张必须由后续新版本的机制与冻结确认重新建立。

这组条款冻结的是当前版本的科学对象和判定标准，不预先保证正结果或 ICLR
录用。正式门禁已经失败，必须保留负结果并停止当前版本；不得在同一协议内
改 loss、结构、pair 或阈值来寻找正结果。任何后续机制重设计都必须作为
新版本重新冻结，不能回写当前结果。

权威结果见：

- [r1 decision](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r1/receipts/decision.json)
- [r1 P0-B](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r1/receipts/p0_b_support.json)
- [r1 P0-C](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r1/receipts/p0_c_screening.json)
- [r1 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r1/COMPLETE.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_geometry_safe_p0_bc_v1_r2/COMPLETE.json)
- [failure-attribution r1 profile](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/factor_probe_profile.json)
- [failure-attribution r1 strata](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/composition_strata.json)
- [failure-attribution r1 23→16 decomposition](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/coverage_transition_decomposition.json)
- [failure-attribution r1 decision](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/receipts/diagnostic_decision.json)
- [failure-attribution r1 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r1/COMPLETE.json)
- [failure-attribution r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_synthetic_state_failure_attribution_v1_r2/COMPLETE.json)
- [hypothesis-review proposal receipt](protocols/IRSTD-1K/synthetic_state_hypothesis_review_v1/proposal_receipt.json)
- [hypothesis-review Markdown](CURE_Lite_机制假设审查.md)
- [core-object proposal receipt](protocols/IRSTD-1K/core_learning_object_redefinition_v1/proposal_receipt.json)
- [core-object redefinition Markdown](CURE_Lite_核心学习对象重定义.md)

## 1. 归档：Phase 1 正式运行前的状态与授权

> 第 1～7、9、11～12 节是 Phase 1 的预运行冻结记录。其中“当前”“唯一授权 Phase 1”“P0-B/C not_evaluated”等表述只描述当时状态；第 8、10 节是决策桥接，第 13 节是已完成的 hypothesis review，当前执行路线以第 14 节为准。

当前状态是：

```text
P0-A1 = pass
P0-B = not_evaluated
P0-C = not_evaluated
P0-D = not_evaluated
candidate S = not_constructed
S training = not_started
Full CURE = not_started
```

准确表述为：

> P0-A1 已通过；完整 P0 正等待 P0-B/C/D，不能解释为完整 P0 已通过或已失败。

当前唯一获得授权的工作是 Phase 1：

> 在冻结的 `D_R` 上，针对 geometry-safe 的 32 个 factual targets 与 206 个 legal targets，实现并运行独立的 P0-B/C follow-on protocol。

Phase 1 不授权：

- 构造或训练 S；
- 修改 decoder、loss、训练步数、校准或推理；
- 读取新的 `D_V` 指标；
- 读取 `D_T`；
- 实现候选 evidence intervention；
- 接入 DNANet、UIUNet、MSHNet、SCTransNet 或其他 IRSTD backbone；
- 开始 Full CURE。

### 1.1 冻结人口

| 人口 | candidate | eligible | source images | `manifest.group_id` |
|---|---:|---:|---:|---:|
| factual targets | 33 discovered，其中 1 个不在可达人口内 | 32 | 由冻结视图给出 | 24 |
| legal targets | 209 | 206 | 149 | 145 |

已知的 1 个不可达 factual target 为 `XDU526 / gt_component_id=2`。它不进入本轮 32-target factual population。

geometry-safe A1 已排除 3 个 legal targets：

| target | 原因 |
|---|---|
| `XDU486 / gt1 / pred1` | 多个 native ancestors |
| `XDU526 / gt1 / pred1` | 一个 native target 对应多个 evaluation descendants |
| `XDU965 / gt1 / pred1` | area ratio `0.470588`，低于冻结下限 |

以上排除属于已经完成的 A1。Phase 1 只能绑定并重建该人口，不能重新定义 A1。

## 2. 两种统计单位必须分开

### 2.1 Source image

`source image` 是产生 target 的具体图像。geometry-safe legal population 有 149 个 source images。

它用于：

- 标识 target 的直接来源；
- 报告每张图像贡献的 targets；
- 未来对 synthetic exposure 做 image-level 汇总。

### 2.2 `manifest.group_id`

`manifest.group_id` 是冻结 manifest 定义的相关样本组。多个 source images 可以属于同一个 group，因此：

```text
149 legal source images != 145 legal groups
```

P0-B/C 的统计独立单位必须是 `manifest.group_id`，而不是把每张 source image 自动视作独立组。

### 2.3 已确认的 group 关系

geometry-safe population 中：

| 项目 | 数量 |
|---|---:|
| factual/legal overlap groups | 14 |
| 位于 overlap groups 的 factual targets | 18 |
| 位于 overlap groups 的 legal targets | 25 |
| legal-exclusive groups | 131 |

这些关系直接约束 Phase 1：

- kNN：对每个 query 排除与其相同 `manifest.group_id` 的 legal neighbors；
- grouped OOF：同一 `manifest.group_id` 的全部 targets 必须进入同一 fold；
- grouped bootstrap：以 `manifest.group_id` 为重采样单位；
- MMD：按冻结规则从 legal reference 中移除 14 个 factual/legal overlap groups，只使用 131 个 legal-exclusive groups 构造 legal reference；
- receipt 必须同时报告 target、source image 和 group 三个层级的数量，不能混写。

原方案中的 `leave-one-source-out` 统一更正为：

> leave-one-manifest-group-out。

## 3. Phase 1 要回答的问题

### 3.1 P0-B：共同支持筛查

问题是：206 个 geometry-safe legal states 是否覆盖 32 个 factual-miss states？

必须分别在两个冻结表示空间中计算：

- handcrafted state space；
- decoder-joint state space。

每个空间的冻结门槛为：

```text
coverage >= 0.90
covered factual targets >= 29/32
```

P0-B 只说明 factual targets 是否落入当前 legal population 的经验支持范围，不证明两个总体分布相同。

### 3.2 P0-C：强可分性与状态偏移筛查

P0-C 不再称为“不可区分性验证”。AUC 接近 0.5 不能证明分布相同，MMD 未越过参考分位也不能证明两个分布相同。

P0-C 只能回答：

> 冻结的线性探针和 grouped MMD 是否观察到明显的 factual/legal 状态差异。

两个表示空间都要报告：

- 5-fold grouped OOF logistic AUC；
- 按 group bootstrap 的 AUC interval；
- grouped MMD observed；
- legal-vs-legal MMD reference distribution 及 q95。

### 3.3 旧 P0-v1 数值的边界

旧 209-target population 的调试数值为：

| 检查 | 旧观察值 | 旧冻结参考 |
|---|---:|---:|
| handcrafted kNN coverage | 2/32 = 0.0625 | 至少 29/32 |
| decoder-joint kNN coverage | 23/32 = 0.71875 | 至少 29/32 |
| handcrafted grouped OOF AUC | 0.999762 | 不高于 0.70 |
| decoder-joint grouped OOF AUC | 0.717361 | 不高于 0.70 |
| handcrafted MMD | 0.559901 | legal-vs-legal q95 = 0.017143 |
| decoder-joint MMD | 0.124867 | legal-vs-legal q95 = 0.022931 |

这些数值只能作为风险提示。它们来自目标身份不完全成立的旧人口，不能被复制为 geometry-safe 206-target population 的正式 P0-B/C 结果。

## 4. Phase 1 的冻结统计协议

### 4.1 权威输入绑定

新 follow-on protocol 必须绑定：

```text
A1 receipt fingerprint:
8ce8d3b00fbc6569a5aa5272a343c9293c0da4b06d592ae2b3c9d2066a93e351

eligible catalog fingerprint:
a7f3862e41272edb8cafa50398f104c5fab6a8ae55c78bf90c1f2ef34bf1bcd9
```

同时绑定：

- 冻结 manifest 及其文件摘要；
- Reference Base cache 及其状态 fingerprint；
- geometry-safe config 及其文件摘要；
- 代码提交；
- 完整 P0-B/C config 及其 fingerprint。

任一绑定不一致时应停止本次运行，不产生可解释为正式 P0-B/C 的结果。

### 4.2 必须原样冻结的 overlap 配置

Phase 1 应从 `protocols/IRSTD-1K/p0_v1/p0_config.json` 复制并冻结完整的 overlap 定义；除 geometry-safe population binding 外，不得在看到新结果后改变统计口径。

| 字段 | 冻结值 |
|---|---|
| factual population | `reachable-factual-misses` |
| legal population | `decoder-visible-legal-targets`，再施加 A1 eligibility |
| group key | `manifest.group_id` |
| exclude same-group neighbors | `true` |
| probability clip | `1e-6` |
| ring radii | inner `4`，outer `12` |
| decoder-joint feature components | `6` |
| joint feature residual | `legal-subspace-reconstruction-l2-per-sqrt-dimension-v1` |
| joint occupancy representation | `raw-local-patch-plus-global-fraction-v1` |
| joint occupancy patch radius | `2` |
| kNN k | `5` |
| legal reference quantile | `0.95` |
| coverage minimum | `0.90` |
| robust scale | `median-mad-maxdev-constant-floor-v1` |
| quantile rule | `sorted-higher-v1` |

handcrafted descriptor 的 12 个字段也必须保持不变：

```text
log1p_gt_area
log1p_supervision_area
supervision_fraction
log_gt_aspect_ratio
border_distance_normalized
clipped_logit_base_gt_mean
clipped_logit_base_ring_mean
log1p_feature_target_rms
feature_target_ring_l2_per_sqrt_channel
conditioning_gt_occupancy_fraction
conditioning_ring_occupancy_fraction
nearest_conditioning_component_centroid_distance_normalized
```

不能只冻结 `k=5` 与 q95，而遗漏 descriptor、scale、distance representation、same-group exclusion 或 quantile rule。

### 4.3 必须原样冻结的 separability 配置

| 字段 | 冻结值 |
|---|---|
| folds | `5` |
| classifier | `class-balanced-l2-logistic-irls-v1` |
| classifier L2 | `1.0` |
| maximum iterations | `100` |
| tolerance | `1e-10` |
| AUC reference threshold | `0.70` |
| legacy AUC point gate rule | `group-balanced-oof-point-estimate-v1` |
| bootstrap replicates | `2000` |
| bootstrap seed | `1729` |
| bootstrap interval | `[0.025, 0.975]` |
| bootstrap interpretation | `conditional-group-bootstrap-of-fixed-oof-scores-v1` |
| MMD | `group-u-multiscale-rbf-matched-legal-null-v1` |
| overlap policy | `remove-overlap-from-legal-reference-v1` |
| observed summary quantile | `0.5` |
| kernel scales | `[0.5, 1.0, 2.0]` |
| bandwidth rule | `legal-exclusive-source-disjoint-positive-distance-median-v1` |
| reference replicates | `1000` |
| reference seed | `2718` |
| reference quantile | `0.95` |
| require MMD within legal reference | `true` |

标准化器和 logistic classifier 只能在对应 training folds 内拟合。任何超参数选择也必须在 training folds 内完成。新 follow-on 不得根据 geometry-safe P0-B/C 数值重新选择这些配置。旧 point-estimate gate 作为兼容诊断字段保留；第 4.5 节的新三值筛查状态单独记录，不能覆盖或删除旧字段。

### 4.4 P0-B 三值状态

每个表示空间独立给出 `pass / fail / inconclusive`：

```text
pass:
  计算完整有效，且 coverage >= 29/32。

fail:
  计算完整有效，但 coverage < 29/32。

inconclusive:
  表示构造、有限距离邻居、分组参考或数值条件不足，
  无法按冻结定义完成计算。
```

P0-B 汇总规则：

```text
任一空间 fail                 -> P0-B = fail
两个空间均 pass               -> P0-B = pass
没有 fail 且至少一个不确定     -> P0-B = inconclusive
```

### 4.5 P0-C 三值状态

P0-C 使用“强可分性与状态偏移筛查”语义。

AUC 子项：

```text
pass:
  bootstrap interval 上界 <= 0.70。

fail:
  bootstrap interval 下界 > 0.70。

inconclusive:
  interval 跨越 0.70，或无法获得冻结定义下的有效 grouped OOF/interval。
```

无论状态如何，都同时记录旧协议定义的 AUC point estimate，不能用 interval 替换或隐藏 point estimate。

MMD 子项：

```text
pass:
  observed MMD <= legal-vs-legal q95。

fail:
  observed MMD > legal-vs-legal q95。

inconclusive:
  131 个 legal-exclusive groups 无法支持冻结的 matched reference，
  或数值计算无效。
```

每个表示空间必须同时通过 AUC 与 MMD 才是该空间 `pass`。P0-C 汇总采用与 P0-B 相同的规则：任一明确失败即 `fail`；全部通过才 `pass`；其余为 `inconclusive`。

`inconclusive` 必须阻止 candidate S，但不能被写成“synthetic state 已被证伪”。

### 4.6 Phase 1 总决策

| P0-B | P0-C | Phase 1 决策 |
|---|---|---|
| pass | pass | 允许进入下一阶段，单独设计并冻结 candidate S；仍不允许训练 |
| fail | 任意 | 停止边缘重加权路线，进入失败归因 |
| 任意 | fail | 停止边缘重加权路线，进入失败归因 |
| inconclusive | 无 fail | 补足统计条件；不构造 S，不宣称 synthetic state 失败 |

## 5. 代码实施方案：只实现 Phase 1

### 5.1 独立 follow-on protocol

建议新增：

```text
protocols/IRSTD-1K/geometry_safe_p0_bc_v1/config.json
cure_lite/experiment/geometry_safe_p0_bc_protocol.py
tools/run_geometry_safe_p0_bc.py
```

不得把 B/C 写回已有的 `tools/run_geometry_safe_p0.py`。原因是 A0/A1 已经形成冻结产物，新 follow-on 必须保持独立、只创建新目录且不得覆盖既有目录。

### 5.2 执行顺序

运行入口必须严格执行：

1. 读取冻结 manifest、Reference Base cache 和 geometry-safe config；
2. 验证 A1 receipt 与 eligible catalog fingerprint；
3. 从 native/evaluation lineage 与原始 catalog 重新构造 geometry-safe view；
4. 重新执行 A1 eligibility，但不覆盖历史 A0/A1；
5. 验证人口严格为 32 factual targets、24 factual groups、206 legal targets、149 legal sources、145 legal groups；
6. 验证 14 overlap groups、18 overlap factual targets、25 overlap legal targets、131 legal-exclusive groups；
7. 验证 descriptor、scale、group 和统计配置；
8. 运行 P0-B；
9. 运行 P0-C；
10. 生成三值 decision 与完整 receipts；
11. 以第二个全新输出目录独立重放，检查逐字节一致性。

禁止直接把历史 `eligible_view.json` 当作跳过重建步骤的 target 清单。

### 5.3 三值逻辑只用于新 follow-on

历史 A0/A1 receipt、decision 和 COMPLETE 文件必须保持逐字节不变。

新 follow-on 可定义：

```python
def tri_state_all(gates):
    if any(gate == "fail" for gate in gates):
        return "fail"
    if all(gate == "pass" for gate in gates):
        return "pass"
    return "inconclusive"
```

新 follow-on 的语义建议为：

```text
p0_a1_status = pass
p0_b_status = pass | fail | inconclusive
p0_c_status = pass | fail | inconclusive
p0_d_status = not_evaluated

candidate_s_design_authorized = true
  only if B=pass and C=pass

s_construction_authorized = false
s_training_authorized = false
full_cure_authorized = false
```

由于 P0-D 尚未运行，Phase 1 结束时 `all_p0_pass` 仍应为 `null/not_complete`，不能写为 `true`。

### 5.4 正式输出

建议创建：

```text
receipts/population_binding.json
receipts/group_accounting.json
receipts/p0_b_support.json
receipts/p0_c_screening.json
receipts/decision.json
COMPLETE.json
```

至少记录：

- protocol ID、代码提交、config 文件摘要与 fingerprint；
- A1 receipt、eligible catalog、manifest、cache 的绑定；
- factual/legal 的 target、source image、group 计数；
- 14 overlap groups 与 131 legal-exclusive groups 的确定性清单摘要；
- descriptor schema、scale 参数、距离与 kNN 配置；
- 每个 target 的距离、邻居 group 和 coverage 判定；
- grouped fold assignment 与其 fingerprint；
- 每个 fold 的 scaler 参数、classifier 参数与 OOF predictions；
- AUC point estimate、group bootstrap interval；
- MMD observed、reference distribution 和 q95；
- 每个子项及汇总项的 `pass/fail/inconclusive`；
- 所有输入与输出文件摘要。

### 5.5 必需测试

至少覆盖：

```text
test_bc_rejects_wrong_a1_fingerprint
test_bc_rejects_wrong_eligible_catalog_fingerprint
test_bc_reconstructs_geometry_safe_view
test_bc_never_reads_d_v
test_bc_never_reads_d_t
test_source_image_and_manifest_group_are_distinct
test_expected_overlap_group_accounting
test_knn_excludes_same_manifest_group
test_grouped_oof_keeps_group_in_one_fold
test_scaler_is_fit_inside_training_fold
test_classifier_is_fit_inside_training_fold
test_group_bootstrap_preserves_manifest_groups
test_mmd_uses_131_legal_exclusive_groups
test_bc_receipts_are_byte_deterministic
test_auc_interval_crossing_is_inconclusive
test_invalid_mmd_reference_is_inconclusive
test_failed_b_blocks_candidate_s_design
test_failed_c_blocks_candidate_s_design
test_inconclusive_bc_blocks_candidate_s_design
test_passed_bc_does_not_authorize_s_training
test_follow_on_does_not_modify_historical_a0_a1
```

优先复用现有实现：

```text
cure_lite/experiment/p0_geometry.py
cure_lite/experiment/p0_support.py
cure_lite/experiment/p0_protocol.py
cure_lite/experiment/geometry_catalog_protocol.py
cure_lite/experiment/geometry_safe_catalog.py
```

不要复制第二套 geometry、descriptor 或 fingerprint 算法。

## 6. Phase 1 完成后的决策

### 6.1 B/C 同时通过

得到的结论只能是：

> 当前 geometry-safe legal population 具备继续研究 factual-miss-informed marginal distribution correction 的必要条件。

它只授权下一阶段单独设计 candidate S，不授权立即构造、运行 P0-D 或训练。

### 6.2 B 或 C 明确失败

得到的结论是：

> 当前 legal synthetic state 与 factual miss state 的共同支持或状态一致性不足；仅在 206 个 legal targets 内重新分配采样概率不能解决该前提问题。

此时停止最近邻、核权重、密度比或其他只修改边缘采样分布的路线，先做失败归因。

### 6.3 B/C 不确定

只补足造成 `inconclusive` 的统计条件。不得把不确定状态改写为通过，也不得直接开始设计 synthetic-state intervention。

## 7. Candidate S 的未来设计边界（当前未授权）

本节只固定概念边界，避免 Phase 1 通过后再次混淆；它不是当前代码任务，也不是已经冻结的 S 协议。

### 7.1 真实 U 底座

历史 U 不是 target-uniform。它先在 149 个 legal source images 中均匀选择 source，再在该 source 的 targets 中均匀选择 target。

若 target `i` 属于 source `s(i)`，该 source 含 `n_s(i)` 个 eligible targets，则 geometry-safe U 的目标边缘概率为：

```text
q_U_geo206(i) = 1 / (149 * n_s(i))
```

未来 S 的骨架只能写为：

```text
q_S(i) = (1 - alpha) * q_U_geo206(i) + alpha * r_F(i)
```

其中 `r_F` 是 factual-miss-informed 的非负边缘质量并归一化为 1。不能再使用 `1/206` 作为 uniform floor，否则会同时改变 source distribution 和 factual-informed weighting，无法单独归因。

### 7.2 尚未冻结的内容

以下内容必须在 Phase 1 通过后，另开协议并在运行 P0-D 前冻结：

- 表示 `z` 使用 handcrafted、decoder-joint 还是预声明组合；
- factual groups 与 factual targets 的归一化规则；
- legal source、group 与 target 的归一化规则；
- kernel 或其他权重函数；
- `tau` 的唯一确定规则；
- `alpha` 的候选集合、可行约束和唯一决胜规则；
- seed-specific deterministic sampling；
- target、source image 和 group 三层 exposure gates；
- seed 42/43 间允许的差异；
- future scrambled control 如何同时保持 target/source/group 层统计。

这些规则不得依据新的 `D_V` 结果选择。

### 7.3 正确机制名称

在现有独立 factual/synthetic loss 结构下，S 只能称为：

> Factual-miss-informed support-preserving marginal distribution correction。

它不能称为 paired correction、conditional matching、instance transfer 或事实漏检到合法目标的配对学习，因为训练目标中没有 factual/legal pairwise loss，factual miss 也没有作为 synthetic decoder 的条件输入。

### 7.4 未来 P0-D

只有 S 的全部规则冻结后，才模拟每个 seed 的完整训练采样计划：

```text
800 epochs * 40 steps/epoch * 4 synthetic samples = 128000 draws
```

P0-D 必须同时报告 target、source image 和 `manifest.group_id` 三层的：

- 非零暴露数量和零暴露数量；
- ESS；
- maximum share；
- top-k concentration；
- seed 42/43 差异。

具体阈值必须在构造并查看候选 exposure 结果前写入独立配置；当前文件不替未来协议臆定 `z/tau/alpha` 或 P0-D 的最优设置。

### 7.5 未来训练必须有 matched U

若 S 进入训练，主对照必须包含重新训练的：

```text
U_geo206
```

`U_geo206` 与 S 必须共享：

- 相同 206-target geometry-safe population；
- 相同 149 source images；
- 相同 decoder、loss、optimizer、epoch 和 steps；
- 相同 factual batches、Base checkpoint、cache 和校准协议；
- 唯一差异为冻结的 synthetic marginal distribution。

历史 U 使用 209-target population，只能作为历史背景，不能代替 `U_geo206` 的严格对照。

### 7.6 未来成功门槛分两层

机制支持必须对 seed 42 和 43 分别满足：

```text
Pd(S_seed) > Pd(U_geo206_seed)
RecoveredMisses(S_seed) > RecoveredMisses(U_geo206_seed)
```

一个 seed 的提高不能抵消另一个 seed 的下降。

完整 Lite 成功还必须对每个 seed 满足：

```text
Pd(S_seed) > max(Base@B_seed, F_seed, F_cross_seed, U_geo206_seed)
```

并满足全部 pixel FA、raw-background FA、FP components/MP 和 covered-target retention 约束。nIoU 改善不能替代 Pd 与漏检找回门槛。

seed 42/43 即使都通过，也只获得进入冻结确认阶段的资格。应增加预先冻结的确认 seeds 或独立划分；确认完成前不授权 Full CURE，也不授权接入其他 IRSTD backbone。

## 8. P0-B/C 失败后的候选研究方向（当前未授权）

旧方案将 `Counterfactual Evidence Transport` 直接定名为 CURE-Lite v0.3，这一步过早。现在撤销该版本定名。

failure attribution 已完成。它只能筛查哪些冻结摘要携带角色预测信号，不能回答“差异主要由哪里造成”：

| 可能来源 | 必须先确认的问题 |
|---|---|
| target geometry | 全人口 AUC interval 跨越 0.70，MMD summary 高于 q95；mixed/inconclusive |
| Base probability | `P` 为 strong role signal，但它是 selection proxy 且非 decoder 输入 |
| frozen feature | `F_background_global` 为 strong；`F_local` 不稳定；两块共享背景环 |
| occupancy | 全人口 mixed/inconclusive；selected-source transductive sensitivity 为 strong |
| source/group composition | 分层结果发生变化，但同源中心化不等于消除 source effect |

这些结果没有明确指向单一因素。该桥接阶段因此要求先进行独立 hypothesis review；review 现已完成并得到 H0，详见第 13 节。任何候选仍不自动获得 `v0.3` 名称。

### 8.1 Derived synthetic feature 契约

未来若研究 feature intervention，必须建立 synthetic-only derived feature 契约：

```text
Base cache feature          immutable
factual branch feature      unchanged
factual-no-miss feature     unchanged
inference feature           unchanged
synthetic branch feature    derived from frozen Base feature
```

每个 derived synthetic feature 必须记录：

- 原始 Base feature 与 cache fingerprint；
- target lineage、source image 和 `manifest.group_id`；
- mask、ring、局部背景估计规则；
- intervention 参数及其冻结规则；
- transform implementation fingerprint；
- derived tensor fingerprint；
- 背景保持、几何保持与数值稳定检查。

不得覆盖 Base cache，也不得让 derived feature 进入 factual branch 或推理。

### 8.2 候选假设的限制

当前 decoder 消费 feature 与 occupancy。若失败主要由 P0 handcrafted descriptor 中的 Base probability 造成，只修改 feature 并不直接解决该差异；此时必须重新评估方法假设，不能机械实现 feature attenuation。

同样，不能在同一个 `D_R` 上反复选择 intervention strength，再把同一个 `D_R` 的 P0-B/C 当作独立验证。未来必须预先定义 group cross-fitting 或明确把结果标为开发诊断，并另设冻结确认。

因此，目前不实现：

- feature attenuation；
- 局部 feature replacement；
- 任何名为 CET 的正式模块；
- intervention 参数搜索；
- transformed-state 训练。

## 9. 归档：Phase 1 当时的禁止事项

Phase 1 期间禁止：

1. 训练 S 或创建 S checkpoint；
2. 实现 S sampler、`S-scrambled` 或 P0-D candidate search；
3. 实现 CET/evidence intervention；
4. 修改 residual decoder、loss、optimizer、epoch、校准或推理；
5. 使用 `D_V` 调整 P0 门槛或统计定义；
6. 读取 `D_T`；
7. 把旧 P0-v1 调试值当作 geometry-safe 正式结果；
8. 修改 256 标签或重写历史 geometry 产物；
9. 覆盖、迁移或重新生成权威 A0/A1 目录；
10. 接入其他 backbone 或开始 Full CURE；
11. 宣称 CURE-Lite 或 CURE 已经成功；
12. 把 M 或未来 S 描述为配对修复；
13. 将 `inconclusive` 解释为通过或明确失败；
14. 用 target-uniform `1/206` 替代真实 source-balanced U 底座。

## 10. 阶段决策树

```text
geometry-safe A1 passed
        |
        v
Phase 1: frozen D_R-only P0-B/C
        |
        +----------------------+----------------------+
        |                      |                      |
   B/C both pass        B or C fails         no fail but uncertain
        |                      |                      |
        v                      v                      v
authorize separate      stop marginal          resolve statistical
S design protocol       reweighting route       insufficiency only
        |                      |                      |
        |               perform failure                |
        |                 attribution                  |
        |                      |                       |
        |       hypothesis satisfies review gates?     |
        |                +-----+-----+                 |
        |                |           |                 |
        |               yes          no                |
        |                |           |                 |
        |                v           v                 |
        |         freeze a testable   retain current    |
        |         state hypothesis    stop decision     |
        |
        v
freeze z / tau / alpha / distribution rules
        |
        v
run P0-D exposure replay only
        |
        +---------------+
        |               |
      pass             fail
        |               |
        v               v
train matched         redesign S
U_geo206 and S
        |
        v
per-seed mechanism gate
        |
        v
frozen confirmation stage
        |
        v
only then discuss Full CURE and backbone validation
```

## 11. 归档：Phase 1 当时交给 Codex 的目标

下面的任务只包含 Phase 1，不包含 S、P0-D、训练或 evidence intervention：

```text
基于提交 a9dcfb23f44e2cbd54849464d0d47c7e2f43e499，
实现一个独立、只创建新产物、D_R-only 的
geometry-safe P0-B/C follow-on protocol。

硬约束：

1. 不修改、覆盖或重新生成现有 P0-v2 A0/A1 权威目录。
2. 绑定 A1 receipt fingerprint：
   8ce8d3b00fbc6569a5aa5272a343c9293c0da4b06d592ae2b3c9d2066a93e351。
3. 绑定 eligible catalog fingerprint：
   a7f3862e41272edb8cafa50398f104c5fab6a8ae55c78bf90c1f2ef34bf1bcd9。
4. 从冻结 D_R、native/evaluation lineage 和原始 catalog
   重新构造 geometry-safe view；不能用历史 JSON target 清单跳过重建。
5. 人口必须严格为：
   32 factual targets、24 factual groups、
   206 legal targets、149 legal source images、145 legal groups。
6. 验证并冻结：
   14 factual/legal overlap groups、
   overlap groups 中 18 factual targets 与 25 legal targets、
   131 legal-exclusive groups。
7. 明确区分 source image 与 manifest.group_id；
   P0-B/C 的分组统计单位是 manifest.group_id。
8. 完整复制并冻结 p0_v1 overlap 与 separability 配置，
   包括 12-field descriptor、scale、k=5、legal q95、
   5-fold class-balanced L2 logistic、2000 group bootstrap、
   multiscale grouped MMD 和 1000 次 legal reference。
9. kNN 排除 query 的同 manifest.group_id legal neighbors。
10. grouped OOF 中同一 manifest.group_id 的所有 targets 进入同一 fold；
    scaler 与 classifier 只能在 training fold 拟合。
11. MMD 从 legal reference 移除 14 overlap groups，
    使用 131 legal-exclusive groups 构造冻结 reference。
12. 禁止读取 D_V 和 D_T。
13. P0-B 在 handcrafted 与 decoder-joint 空间分别运行；
    coverage 门槛均为至少 29/32。
14. P0-C 命名为强可分性与状态偏移筛查；
    同时报告 AUC point estimate、group-bootstrap interval、
    observed MMD 与 legal-vs-legal q95。
15. 新 follow-on 使用 pass/fail/inconclusive 三值状态；
    AUC interval 跨 0.70、组数不足或数值无效均为 inconclusive。
16. 三值逻辑只用于新 follow-on，历史 A0/A1 decision 不得改写。
17. 生成 population_binding、group_accounting、p0_b_support、
    p0_c_screening、decision 和 COMPLETE receipts；
    绑定全部输入、配置、分组、统计与输出。
18. B/C 任一 fail：停止边缘重加权路线；
    任一 inconclusive：不授权 S 并保留不确定语义；
    B/C 同时 pass：只授权后续单独设计 S，不授权构造或训练。
19. 增加人口绑定、group accounting、same-group exclusion、
    fold-local fitting、三值状态、D_V/D_T 禁止访问、
    历史产物不变及逐字节确定性测试。
20. 运行完整测试，并在两个全新正式输出目录分别执行同输入文件系统重放；
    要求 receipts 与 COMPLETE 逐字节一致。
21. 不实现 S、S-scrambled、P0-D、CET 或任何 feature intervention。
22. 不修改 decoder、loss、Base、cache、标签、训练、校准、推理
    或任何历史结果。
23. 不接入其他 backbone，不开始 Full CURE。
```

## 12. 归档：Phase 1 完成标准

Phase 1 只有同时满足以下条件才算完成：

- 新协议与配置已经创建并冻结；
- geometry-safe population 和 group accounting 全部匹配；
- P0-B/C 的两个表示空间均产生完整 receipt，或产生有明确原因的 `inconclusive`；
- decision 使用新 follow-on 三值语义；
- 历史 A0/A1 逐字节不变；
- 两个独立输出目录中的同输入文件系统重放逐字节一致；
- 完整测试通过；
- 未读取新的 `D_V` 结果或 `D_T`；
- 未构造或训练 S；
- 未实现 evidence intervention；
- 未启动 Full CURE 或其他 backbone 实验。

本阶段的价值是获得一个可审计的研究分叉：要么证实边缘分布校正具备继续设计的必要条件，要么明确把问题前移到 synthetic state 定义；它本身不宣称 CURE-Lite 或 CURE 已经完成。

## 13. 已完成：独立 hypothesis review

本节记录已经完成的非实验性研究判断；第 1～7、9、11～12 节为 Phase 1 归档，第 8、10 节为决策桥接。当前执行路线见第 14 节。

### 13.1 审查所比较的候选解释

| 候选解释 | 当前支持 | 当前反证或限制 | 是否可直接实现 |
| --- | --- | --- | --- |
| Base probability shift | `P` 全人口 AUC 0.996058，strong | `P` 参与角色选择且不是 decoder 输入 | 否 |
| local target-feature evidence | `F_local` coverage 18/32，MMD summary 高于 q95 | AUC interval 跨 0.70；两个小分层固定拟合不确定 | 否 |
| context/global feature shift | `F_background_global` 全人口与 shared groups 为 strong | 与 `F_local` 共享背景环；selected-source 中不再 strong | 仅作为待审查假设 |
| conditioning occupancy shift | `O` 在 selected-source transductive sensitivity 为 strong | 全人口与 shared groups 的 interval 均跨 0.70 | 仅作为待审查假设 |
| source/group composition | 分层后多个 AUC 状态改变 | 现有中心化是 selected-overlap transductive sensitivity，不能消除 source effect | 需要新识别设计 |

当前没有证据支持“直接衰减目标局部 feature”。本轮重点审查了：

> 现有 synthetic state 只删除 occupancy，却保留 detected-target 所在图像的完整 frozen context feature；这种 feature–occupancy 联合状态是否形成了 factual miss 中少见的组合。

代码确认这种联合状态真实存在，但现有 probe 没有 interaction-specific estimand，不能区分 feature 边缘差异、occupancy 边缘差异、source/group composition 与真正的联合交互。因此它仍是结构性假设，不是已经确认的机制，也不是模块名称。

### 13.2 hypothesis review 的五个硬门槛

一个候选只有同时满足以下条件，才可以进入独立 transformation protocol：

1. **可作用性**：变量必须进入 decoder 的实际输入；`P` 只能用于诊断，不能作为干预对象。
2. **单机制**：只改变一个清晰的 synthetic-state 生成原则，不同时新增 attention、decoder、loss、sampler 和校准模块。
3. **状态保持**：Base cache、factual branch、factual-no-miss branch、inference、GT、target lineage 和非目标背景均保持不变并可逐项审计。
4. **不可自证**：不能在同一 \(D_R\) 上搜索干预形式/强度后，再把同一 \(D_R\) 的 P0-B/C 当作确认结果；必须预先冻结 group cross-fitting 或另设确认 groups/split。
5. **可证伪性**：必须提前写明 transformed-state 的 P0-A/B/C 失败条件；失败后停止该假设，不转为继续调参。

### 13.3 代码级问题的审查结果

1. decoder 只消费 detached frozen feature 与投影后的 binary occupancy；`P`、GT、target、valid mask、identity 和 descriptor 均不进入 decoder；
2. factual 与 synthetic branch 分别计算 loss 后只做标量相加，不存在 factual/legal pairwise loss 或条件输入；
3. 206/206 个 legal state 只改变 occupancy 与监督，synthetic feature 与 source feature 对象和值完全一致；
4. `F_background_global` 混合 ring、whole-grid statistics 与 source composition，当前结果没有预声明子量分解；
5. `F_local` 与 `F_background_global` 共享 ring，不能作为两个独立原因相加解释；
6. `O` 混合目标邻域、其他组件和全图 occupancy，selected-source strong signal 不能定位到某个可修改子量；
7. 当前没有由几何或代数规则唯一确定、同时保持非目标背景的 synthetic-only 变换；
8. 当前也没有独立 confirmation groups 或冻结 cross-fit rule，不能在同一 \(D_R\) 上选择变换后再自证。

### 13.4 review 的三种结局

```text
结局 H0：
  没有候选同时满足五个硬门槛
  -> 停止当前 occupancy-deletion synthetic paradigm
  -> 重新定义 CURE-Lite 的核心学习对象

结局 H1：
  只有 context/occupancy 联合一致性假设满足门槛
  -> 单独冻结 transformation protocol
  -> 先运行 transformed-state P0-A/B/C
  -> 不构造 S，不训练

结局 H2：
  多个候选都看似可行但无法预先唯一选择
  -> 不实现多个模块并行试验
  -> 设计独立确认数据或更强识别协议后再选择
```

即使 H1 的新状态通过 P0-B/C，也只获得“构造并冻结 S、随后运行 P0-D”的资格；仍不直接授权训练、Full CURE 或其他 backbone。

本轮实际结局为：

```text
review outcome = H0
reason = no candidate satisfies all five review gates
transformation protocol eligible = false
candidate S authorized = false
```

H0 只否定当前 transformation 授权，不否定 CURE 总方向。完整依据与机器可核验状态见：

- [CURE-Lite 机制假设审查](CURE_Lite_机制假设审查.md)
- [proposal receipt](protocols/IRSTD-1K/synthetic_state_hypothesis_review_v1/proposal_receipt.json)

### 13.5 当前禁止继续做的事情

- 不依据 `P` 的高 AUC 修改 decoder 或 synthetic feature；
- 不因 `F_background_global` strong 就直接实现 context replacement；
- 不因 selected-source 的 `O` strong 就直接修改 occupancy；
- 不把 drop-one 点差写成因素排名；
- 不调 IRLS、fold、AUC 边界、PCA 维数或 MMD 规则以消除 `inconclusive`；
- 不构造 S、运行 P0-D、训练 seed 42/43；
- 不读取新的 \(D_V\) 或 \(D_T\)；
- 不接入 DNANet、UIUNet、MSHNet、SCTransNet；
- 不开始 Full CURE。

本阶段已经以 H0 proposal receipt 完成；没有产出新模型、transformation 或正向性能结果。

## 14. 已完成：CURE-Lite 核心学习对象重定义

总研究主线没有改变：

```text
完成 CURE-Lite 最小核心机制
  -> 冻结确认机制成立
  -> 设计 Full CURE
  -> 跨 IRSTD backbone 与三数据集验证
```

改变的是 CURE-Lite 内部的当前技术节点。旧的：

```text
occupancy deletion
  + unchanged observational feature
  + residual decoder
```

不能构成已识别的核心机制。本轮已经完成只读重定义，没有先写 transformation 再寻找解释。

### 14.1 唯一保留候选

唯一保留候选不再是 factual/legal descriptor 或 compatibility score，而是：

\[
Q_\theta:
(F,\Pi O)\mapsto q,
\]

其中 \(q=\sigma(D_\theta(F,\Pi O))\) 是 occupancy hard mask 之前的概率分数，不作已校准概率声明。对同一 source、同一 frozen feature 和一个 geometry-safe legal component \(C_g\)：

\[
O^+=O,\qquad
O^-=O\setminus C_g,
\]

新的核心对象是：

\[
\boxed{
\Delta_g Q_\theta
=
Q_\theta(F,\Pi O^-)
-
Q_\theta(F,\Pi O^+)
}.
\]

它监督的是同一个 completion operator 对删除一个已验证 coverage component 的离散响应。

完整定义、反例和权限边界见：

- [CURE-Lite 核心学习对象重定义](CURE_Lite_核心学习对象重定义.md)
- [core-object proposal receipt](protocols/IRSTD-1K/core_learning_object_redefinition_v1/proposal_receipt.json)

receipt fingerprint：

```text
62461b5514b45d4082ea4001c4e8324b2f5ad0542a4ae11891b3db4fda980ef9
```

### 14.2 它为什么不是旧 synthetic route

legal deletion endpoint 不再被当作 factual-like iid sample，也不要求 factual/legal population 具有共同支持。

旧 P0-B/C 仍是旧 exchangeability/reweighting 路线的有效负结果，但不再是新 pairwise functional equation 的成立前提。P0-A1、lineage、legal deletion 和 projected occupancy visibility 仍然保留。

此外，matcher identity 不变并不足以保证标签差分只来自被删目标。未来
clean positive pair 必须额外满足：被删组件 \(C_g\) 在有效域内不与删除前
任何已有 unmatched GT 相交。实际标签增量先在有效域 \(V\) 内定义为
\(\mathcal R_{\mathcal G,V}(O^-)\setminus
\mathcal R_{\mathcal G,V}(O^+)\)；只有通过该非干扰审计时，
它才严格等于目标 \(g\) 的可写区域 \(A_g\)。

pair 必须是：

```text
同一 legal target 的 coverage-before
    versus
同一 legal target 的 coverage-after
```

而不是 factual miss 与 legal target 的配对。

未来 pair objective 必须直接消费 \(Q^- - Q^+\)，两端共同反传且都不 stop-gradient。如果分别对两个 endpoint 求 loss 后相加，仍然是旧的独立状态 ERM。

### 14.3 paired objective 已冻结；实现核心已完成

当前状态为：

```text
core-object candidate = conceptually admissible
paired-objective protocol = frozen
pairwise implementation = core_implemented
current-version mechanism support = failed at Wave A
real D_R paired preflight = passed
matched-control static preflight = passed
bounded D_R learnability = passed_with_exact_replay
spatial-tail companion = completed_with_exact_successful_replay
8-control bounded execution = ENGINEERING_EXECUTION_PASS
formal joint schedule = implemented
paired artifact contract = implemented
formal 800-epoch training = completed; four attempts sealed
Wave A = PERFORMANCE_FAIL
current paired version = stopped and preserved
Wave B/C and frozen confirmation = not authorized
```

paired-objective 的唯一固定形式、双端梯度、响应域、null control、
matched controls、未来接口和停止条件已经冻结：

- [CURE-Lite Paired Objective 协议](CURE_Lite_Paired_Objective_协议.md)
- [paired-objective proposal receipt](protocols/IRSTD-1K/paired_objective_v1/proposal_receipt.json)

receipt fingerprint：

```text
5a2f357911fb5f1dc1a946b3dbad429d256c390677d238b2f395fe90ce91fac8
```

该 receipt 是协议阶段的历史证据，仍准确记录“当时没有修改生产代码”；
其中历史状态 `pairwise implementation = false` 和
`training authorization = false` 必须保留，不能反写 proposal receipt。
根据后续用户指令，additive implementation 已新增 pair catalog、
`PairExample/PairBatch`、paired loss、paired train step 和确定性
800×40 schedule；旧 decoder、旧三分支训练入口和推理图保持不变。

真实 \(D_R\) paired preflight 已封存；seed 42/43 的完整
\(800\times40=32{,}000\) 暴露计划均通过。matched-control static
preflight 在 206 个 clean pairs 上得到 source-disjoint、无不动点的完整
target permutation。随后 GPU0 上 proposed bounded learnability、空间尾部
companion 与全部 8 个 matched controls 的 bounded engineering execution
均完成成功重放。它们只证明冻结目标在固定真实微型人口上可计算、可学习，
并证明所有正式对照可按相同预算执行；不是检测性能或自然漏检迁移证据。

正式 \(800\times40\) 联合 factual/pair schedule、完整 exposure ledger、
三次 forward/十二状态预算和独立 create-only paired artifact schema 已实现。
无恢复 runner、评估绑定与一次性 Wave A reveal 已闭合；四个正式任务均完成
800 epoch 和 32,000 updates。Wave A 决定为 `PERFORMANCE_FAIL`，所以当前
版本的执行链已经到达停止条件，不再进入后续 wave。

当前仍然禁止：

- 修改 decoder 拓扑或覆盖旧 loss/旧 branch engine；
- 构造 transformation 或 S；
- 修改或覆盖本版已封存的训练、校准、评估与 reveal 结果；
- 继续读取 \(D_V\) 进行本版调参，或读取 \(D_T\)；
- 运行 Wave B/C 或冻结确认；
- 开始 Full CURE；
- 接入 DNANet、UIUNet、MSHNet 或 SCTransNet。

### 14.4 到什么性能才能设计 Full CURE

实现核心已经通过结构与 toy 测试；真实 \(D_R\) 前置门槛通过后，正式训练的
seed 42 和 43 必须分别满足：

\[
\mathrm{TP}_{\Delta Q}
\ge
\max_c\mathrm{TP}_{c}+2,
\]

当前 170-target \(D_V\) 上等价于：

\[
\mathrm{Pd}_{\Delta Q}
\ge
\max_c\mathrm{Pd}_{c}+1.176\text{ 个百分点}.
\]

同时对固定 23 个 anchor misses：

\[
\mathrm{Recovered}_{\Delta Q}
\ge
\max_c\mathrm{Recovered}_{c}+2,
\]

即恢复率至少多 \(8.70\) 个百分点。

每个 seed 还必须满足：

```text
retention = 1.0
pixel FA <= 1e-4
raw-background FA <= 1e-4
FP components / MP <= 100
budget_violation = false
```

这里的 `retention=1.0` 是新的、更严格的阶段门槛；历史冻结 Stage-A
配置中的 `minimum_retention` 为 0.99。二者必须在 receipt 中分开记录，
不能把新门槛误写成历史预算。

其中 \(c\) 至少包括 Base@B、F、F×、geometry-matched independent-endpoint ERM 和旧 independent uniform legal。

实际 Wave A 结果为：

| seed | paired TP / recovered | 最佳比较方法 TP / recovered | margin | 判定 |
| ---: | ---: | ---: | ---: | --- |
| 42 | 147/170；0/23 | 154/170；7/23 | \(-7/-7\) | fail |
| 43 | 152/170；5/23 | 152/170；5/23 | \(0/0\) | fail |

两 seed 的 retention 均为 1.0，FA 与 budget 约束全部通过。因此不能把失败
归因于约束门槛；失败的是预声明的自然漏检增益。门槛公式保持原样，不能因
本次负结果而降低。

seed 42/43 若通过，本来也只允许冻结并进入确认。还需要预冻结额外 seeds、grouped
factual-recovery interval 排除零，以及一个未使用划分上的冻结确认，之后
才允许设计 Full CURE。单 seed、跨 seed 均值提升或只多找回 1 个目标均不
满足该门槛。本次未通过，因此这一确认分支没有被触发。

八个内部 controls 能检验独立 endpoint ERM、after-only 数据增加、
occupancy-only、feature-only、错误 pair identity 和单端梯度等主要替代解释。
即使最终全部通过，也只能支持“非因子化 pair relation、真实
feature/occupancy、正确 pair identity 与双端梯度共同构成的 coverage-response
机制”，不能把收益唯一归因于某一个代数交叉项。

此外，内部 controls 不能替代通用 mask-refinement prior。若 development
seed 42/43 通过，应在冻结确认阶段加入同一 frozen Base 输入、同一训练信息和
同一 FA/retention 约束下的强 generic-refiner 对照；该比较属于创新边界验证，
不允许反过来修改当前 proposed 主线。
