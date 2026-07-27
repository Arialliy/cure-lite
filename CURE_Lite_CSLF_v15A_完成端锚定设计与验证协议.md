# CURE-Lite CSLF-v15A：完成端锚定设计与验证协议

> 协议状态：**运行前冻结**  
> 适用范围：CURE-Lite scalar CSLF、仅 \(D_R\)、单次 bounded-400  
> 候选名称：`completion_rooted_response_joint`  
> 本文不包含 v15A 训练结果，也不授权读取 \(D_V/D_T\)、Formal-800、Full CURE、跨 backbone 或三数据集实验。

---

## 1. 父实验与证据边界

v15A 只允许从已封存的父实验
`cure_lite_cslf_v15_bounded_400_r1` 出发。父实验已经完整结束，其决定为：

```text
decision = BOUNDED_CSLF_GATE_FAIL
bounded_gate_passed = false
formal_800_authorized = false
D_V_accessed = false
D_T_accessed = false
```

父实验不是执行失败。三个 objective 均完成 400 次更新，全部训练、checkpoint、
零水平评估和 COMPLETE 产物均存在。父实验的 response-joint 关键结果为：

| 项目 | 父 v15 结果 |
|---|---:|
| clean-positive response sign | 1396/1396 |
| clean-positive added-target 负像素 | 6/149 |
| clean-positive compact exact | 0/16 |
| factual-miss gate | 15/16 |
| factual-no-miss gate | 16/16 |
| component-null gate | 17/17 |
| identity-null gate | 16/16 |

该结果只支持以下定位：

\[
\text{coverage response 的方向已经学到，}
\qquad
\text{completion endpoint 的绝对零水平仍未锚定。}
\]

它不授权增加训练步数、搜索阈值、加入 margin、改变表示、启用 PP-CSLF 或增加网络块。

### 1.1 父结果与源码闭合

以下绑定在 v15A 运行前冻结，任一文件不存在或 SHA256 不一致都必须停止：

| 对象 | 路径 | SHA256 / fingerprint |
|---|---|---|
| 父 COMPLETE | `runs/irstd1k_stage_a_seed42/cure_lite_cslf_v15_bounded_400_r1/COMPLETE.json` | `bf443854ec3137fd1507efb03cd7fda9dc2dfebef2ba22cb47c1818f20759437` |
| 父 complete fingerprint | 同上 | `faaa2395623f5edfa0e56ab849d20305b73df1e7b3446b22b834279a2637d14b` |
| 父源码归档 | `artifacts/source_closures/cure_lite_cslf_v15_bounded_400_faaa2395623f.tar` | `5904c04a8ed2b4af28e46d159df25cf26bce956c4b80c27a2c488ad8a31491c7` |
| 父源码清单 | `artifacts/source_closures/cure_lite_cslf_v15_bounded_400_faaa2395623f.json` | `fe1667e159cfc9ee0b0eb0a9f941e118a83725386215257211608c6146baaa3c` |

父 COMPLETE 内列出的 17 个实验文件还必须逐文件重新计算 SHA256，并与其中的
`artifact_files` 完全一致；不得只验证 COMPLETE 文件本身。

---

## 2. 唯一科学修改

对同一 frozen feature 和同一 source image 的 coverage deletion pair，定义：

\[
e_+
=
\phi_\theta(F_b,O_+)-\phi_+^*,
\qquad
e_-
=
\phi_\theta(F_b,O_-)-\phi_-^*,
\]

其中 \(O_-\subset O_+\)，minus endpoint 是删除 Base coverage 后必须产生 residual
completion 的状态。有限 coverage response error 为：

\[
\delta e
=
\left(\phi_- - \phi_+\right)
-
\left(\phi_-^*-\phi_+^*\right)
=e_- - e_+.
\]

### 2.1 父 v15 公式

父 response-joint 使用：

\[
\mathcal L_{\mathrm{pair}}^{v15}
=
\mathcal E_\mu
\left(
e_+,\,
e_- - e_+
\right),
\]

即把 covered plus endpoint 作为绝对根，minus endpoint 只通过有限响应项进入目标。

### 2.2 v15A 公式

v15A 唯一允许的修改是：

\[
\boxed{
\mathcal L_{\mathrm{pair}}^{v15A}
=
\mathcal E_\mu
\left(
e_-,\,
e_- - e_+
\right)
}
\]

即将绝对根从 plus endpoint 换到 completion-bearing minus endpoint。有限 response
项、二维 joint energy、integration measure 和全部空间项均不改变。

该坐标仍然可逆：

\[
e_-=a,\qquad
e_+=a-r,
\]

其中 \(a=e_-\)、\(r=e_- - e_+\)。因此：

\[
a=0,\ r=0
\iff
e_-=e_+=0.
\]

对于 clean-positive，\(\phi_-^*<0\) 被直接锚定；对于 component-null，
\(\phi_-^*=\phi_+^*\)，response target 仍为零，因此同一个目标同时约束绝对场和
null response。factual-no-miss 的 natural-state loss 完全不变。

---

## 3. 为什么这是单一机制而不是模块堆叠

v15A 不增加任何新的模型组成部分。修改前后均为一个 scalar coverage-state
residual completion field：

\[
(F_b,O)\longmapsto\phi_\theta(F_b,O),
\qquad
C_\theta=(\phi_\theta<0)\land\neg O.
\]

v15A 明确不增加：

- 新 decoder branch、head、gate 或辅助网络；
- 第三个 pair loss coordinate；
- endpoint loss 与 response loss 的额外加权和；
- hinge、margin、温度、阈值偏置或可调校准量；
- phase representation、proposal、component tree 或后处理；
- 新的训练样本、额外监督标签或推理输入。

pair coordinate 的维数仍为 2，模型参数量、前向次数和推理方程均不变。唯一变化是
二维 pair energy 中哪个 endpoint 承担绝对根。

特别地，本协议不采用 normalized Haar midpoint-response。当前 vector energy
先计算分量平方和；正交坐标变换会与 `identity_joint` 等价，不能继续检验有限
coverage response 的独立作用。也不采用
\([e_+,e_-,e_- - e_+]\)，因为它会形成冗余三坐标并改变相对权重。

---

## 4. 严格不变项

除第 2.2 节的一个 pair coordinate 外，以下内容全部冻结。

### 4.1 模型与推理

- 通用 Base 边界保持 \((F_b,O)\)，不得访问特定 backbone 内部结构；
- Base、state cache 和 frozen feature 不更新；
- scalar CSLF，width \(=32\)，feature stride \(=4\)；
- sample-wise global RMS feature normalization 不变；
- `input_projection -> SiLU -> depthwise spatial_mixing -> phase_projection -> PixelShuffle`
  不变；
- field amplitude \(=0.9\)，initial field value \(=0.9\)；
- 输出始终为非饱和 FP32 scalar field；
- residual threshold 固定为 \(0\)，不得搜索；
- completion 固定为 \((\phi<0)\land\neg O\)；
- final 固定为 \(O\lor completion\)。

### 4.2 Target、measure 与 Sobolev energy

- truncated signed chessboard-distance target 不变；
- truncation radius \(=4\)；
- equal-mass focus-support / exterior-band / far-background measure 不变；
- norm order \(=4\)，norm epsilon \(=10^{-3}\)；
- value 与 spatial energy 的组合不变；
- clean-positive、component-null 和 identity-null 的 target geometry 不变；
- 所有 target、mask、lineage 和 source-image identity 不变。

### 4.3 真实 \(D_R\) population

- 只使用 `IRSTD-1K/stage_a_seed42` 的冻结 \(D_R\)；
- 五个真实输入路径及其 SHA256 不变；
- representation-neutral raw catalog、observability 决定和 scalar cache 不变；
- bounded population、eligible/excluded identity 和选择顺序不变；
- 每次 update 固定使用 12 个逻辑状态；
- 每个 objective 使用同一批 16 factual-miss、16 factual-no-miss、
  16 clean-positive 和 16 component-null optimizer states；
- diagnostic-only component-null、identity-null 和 scalar-hidden 状态完整保留；
- 只允许解析冻结 manifest 的结构与绑定信息；不得实例化、读取或计算
  \(D_V/D_T\) 的样本 payload、cache tensor、预测或指标。

### 4.4 训练预算

- seed \(=42\)；
- 10 epochs；
- 每 epoch 40 updates；
- 每 objective 恰好 400 updates；
- optimizer 为 Adam；
- learning rate \(=10^{-3}\)；
- betas \(=(0.9,0.999)\)；
- epsilon \(=10^{-8}\)；
- weight decay \(=0\)；
- natural branches 继续按
  `factual_miss + factual_no_miss + pair` 相加；
- 三个 objective 使用同一初始模型、同一 schedule、同一 endpoint、同一 cache、
  同一 optimizer 配置和同一前向/反向/更新预算；
- 不允许 resume、automatic retry 或追加 updates。

### 4.5 执行

- 唯一设备为 `cuda:0`；
- `CUDA_VISIBLE_DEVICES=0`；
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
- 温度暂停阈值 \(82^\circ\mathrm C\)，恢复阈值 \(75^\circ\mathrm C\)；
- device-resident cache、显存预检查和完整 compute ledger 保留；
- 唯一输出目录为
  `runs/irstd1k_stage_a_seed42/cure_lite_cslf_v15a_completion_rooted_bounded_400_r1`；
- `--validate-create-only` 只验证静态契约且不得创建输出目录；随后只有
  `--run-once` 可以独占创建输出目录，不得覆盖已有目录。

---

## 5. 唯一改变项

本轮运行的唯一科学变量为：

```text
response pair absolute root:
    e_plus
        ->
    e_minus
```

对应 objective suite 固定为：

1. candidate：`completion_rooted_response_joint`
2. control：`identity_joint`
3. control：`separable_endpoint`

candidate 和 `identity_joint` 使用同一 joint measure；`separable_endpoint` 继续使用
原有 endpoint-specific absolute measures。不得因为 dataset-free 或 bounded
结果修改 objective 顺序、增加第四个 objective、调整 branch 权重或重新定义
control。

---

## 6. Dataset-free 21 项冻结门槛

bounded-400 之前必须完成 dataset-free gate。它不读取 \(D_R/D_V/D_T\)，由以下
固定 evidence 组成：

- \(16\) 个 case \(\times 2\) 个分辨率 \(\times 3\) 个 seed，共 96 个几何与表示
  case；
- \(2\) 个分辨率 \(\times 3\) 个 seed，共 6 个 completion-root counterexample
  probe；
- \(3\) 个 objective \(\times 3\) 个 seed，每项 3 updates，共 9 个小型计算结果。

以下 21 项必须全部为 true：

| # | 冻结检查名 | 必须证明的内容 |
|---:|---|---|
| 1 | `geometry_matrix_complete` | 96 个预声明 case 全部存在且无额外 case |
| 2 | `geometry_and_model_fields_finite` | target、model 和训练诊断场全部有限 |
| 3 | `target_zero_level_exact` | target field 的零水平符号契约精确成立 |
| 4 | `hard_union_exact` | completion exclusion 与 Base hard union 精确成立 |
| 5 | `identity_null_exact` | identity-null 的 field/output 精确一致 |
| 6 | `scalar_hidden_component_exact` | scalar-hidden 诊断按预声明保持精确一致 |
| 7 | `component_null_no_new_negative_island` | component-null 不产生新负场支持 |
| 8 | `empty_state_negative_component_count_zero` | empty state 的负像素和负组件均为 0 |
| 9 | `selected_scalar_representation_has_no_hidden_optimizer_pair` | optimizer pair 对 scalar representation 全部可观测 |
| 10 | `phase_roundtrip_exact` | pair occupancy phase roundtrip 精确 |
| 11 | `target_response_inside_selected_scalar_rf` | target response 不超出 scalar receptive support |
| 12 | `target_response_inside_phase_rf` | target response 不超出 phase receptive support |
| 13 | `completion_root_probe_complete` | 6 个 completion-root probe 完整并绑定 v15A policy |
| 14 | `completion_root_probe_response_correct_without_crossing` | 构造“response 已正确但尚未越零”的反例 |
| 15 | `completion_root_probe_direct_gradient` | 旧根对 minus absolute error 无直接梯度时，新根必须产生非零直接梯度 |
| 16 | `completion_root_probe_fixed_points` | exact clean 与 component-null fixed point 的新 loss 均为 0 |
| 17 | `three_objectives_computationally_learnable` | 三个 objective 均有限、参数均更新、natural role 正确 |
| 18 | `early_gradient_latency` | 所有参数按冻结 update 上限获得非零梯度 |
| 19 | `training_compute_ledger_exact` | 每个小型结果恰好 3 forward/backward/optimizer steps 和 36 个逻辑状态 |
| 20 | `matched_objective_fairness` | 每个 seed 的三目标初始模型和选择序列完全相同 |
| 21 | `no_dataset_split_access` | 未访问任何数据集 split |

### 6.1 两次独立重放

dataset-free 必须在冻结源码上独立运行两次。两次都必须满足：

1. 21/21 项全部通过；
2. case、probe、training evidence 数量完全一致；
3. 每一条 evidence 的排序和内容完全一致；
4. canonical receipt 逐字节一致；
5. receipt fingerprint 完全一致；
6. 两次运行之间不得修改源码、配置、环境变量或随机种子；
7. 任一不一致都停止，不得启动真实 \(D_R\) bounded-400。

这两次重放只证明固定 dataset-free 计算路径的确定性，不是多 seed 的性能稳定性
结论。

---

## 7. Candidate 门槛与 controls 的预声明语义

### 7.1 Candidate 必须保留父协议的全部门槛

`completion_rooted_response_joint` 是唯一 candidate。它必须同时通过以下七类门槛；
不得降低 95% 阈值、忽略任一 state 或使用均值代替逐项通过。

#### Factual-miss

对全部 16 个 bounded factual-miss：

- focus target 非空；
- completion 至少命中目标；
- 每个 record 的 target negative fraction \(\ge 0.95\)；
- invalid completion pixels \(=0\)。

因此 candidate 必须达到 factual-miss gate \(=16/16\)。

#### Factual-no-miss

对全部 16 个 factual-no-miss：

- negative pixels \(=0\)；
- negative components \(=0\)；
- invalid completion pixels \(=0\)。

因此 candidate 必须达到 factual-no-miss gate \(=16/16\)。

#### Clean-positive defined metrics 与 compact support

对全部 16 个 clean-positive pair：

- response support 非空；
- response sign 每个像素均正确；
- added-target 每个像素在 minus endpoint 均为负场；
- plus writable false-island components \(=0\)；
- target 外新增 completion pixels \(=0\)；
- new completion pixels 恰好等于 added-target pixels；
- new completion component 数与 added-target component 数一致；
- new completion 与 added-target 逐像素完全相等；
- plus/minus invalid completion pixels 均为 0。

即必须同时达到：

```text
clean response sign = 1396 / 1396
added-target negative pixels = 149 / 149
clean compact exact = 16 / 16
clean compact component match = 16 / 16
target-outside new completion pixels = 0
```

#### Component-null

- 16 个 optimizer component-null 均不得产生新负组件或 removed-footprint 负像素；
- 1 个 diagnostic-only scalar-hidden component-null 还必须保持 field 和 completion
  精确一致；
- plus/minus invalid completion pixels 均为 0。

因此 candidate 必须达到 component-null gate \(=17/17\)。

#### Identity-null

全部 16 个 identity-null 必须保持 field、completion 和 final 精确一致，且无 invalid
completion。因此 identity-null gate 必须为 \(16/16\)。

#### Scalar-hidden diagnostic

冻结的 scalar-hidden diagnostic 必须通过其全部定义指标，不得从结果中删除。

#### Candidate 总门槛

只有以下七项全部通过，candidate 才记为 bounded candidate pass：

```text
factual_miss
factual_no_miss
clean_defined_metrics
clean_compact_support
component_null
identity_null
scalar_hidden_diagnostic
```

### 7.2 Controls 只承担完整同预算报告

`identity_joint` 和 `separable_endpoint` 是机制对照，不是 candidate。二者必须：

- 完成与 candidate 相同的 400 updates；
- 使用相同初始化、schedule、cache、endpoint、optimizer 和计算预算；
- 生成完整 checkpoint、training receipt 和全部零水平诊断；
- 对同一七类门槛逐项报告实际 pass/fail；
- 不得因结果不利而省略、提前停止或更换 seed。

controls 的门槛结果不作为 candidate pass 的逻辑合取项。预声明的 bounded 决定为：

\[
\text{candidate\_qualified}
=
\text{candidate 的全部冻结门槛通过}
\land
\text{两个 controls 同预算完整结束并完整报告}
\land
\text{全部执行契约通过}.
\]

这不是降低 candidate 门槛：candidate 仍须通过父协议的全部角色和像素级要求。
该语义只是把“候选是否合格”与“对照实际表现如何”分开。control 的强结果也不能
弥补 candidate 失败；control 的弱结果也不能替 candidate 建立机制结论。

父 v15 中将三个 objective 的 `bounded_gate_passed` 直接取 `all(...)` 的顶层语义
不适用于本次候选—对照检验。v15A 在运行前明确冻结为：

```text
candidate_all_frozen_zero_level_gates
AND complete_matched_control_diagnostics
```

不得在看到 v15A 结果后再次修改。

---

## 8. 单次 bounded-400 与停止规则

### 8.1 启动前

只有以下条件全部满足，才允许 claim 唯一 r1 输出目录：

1. 父 v15 结果、17 个产物和 source closure 全部验证通过；
2. v15A 实现文件 SHA256 清单与 implementation fingerprint 已冻结；
3. dataset-free 两次独立重放均为 21/21，且逐字节一致；
4. 五个真实 \(D_R\) 输入及其 SHA256 未改变；
5. population、schedule、model config、device cache 和显存预检查全部通过；
6. create-only authorization 与运行命令完全绑定；
7. 输出目录尚不存在。

### 8.2 运行中

- 只允许运行一次 seed-42 bounded-400；
- 三个 objective 均必须完整训练，不因中间 loss 或 control 现象提前停止；
- 温度达到 \(82^\circ\mathrm C\) 时只暂停，降至 \(75^\circ\mathrm C\) 后继续；
- 暂停不等于 resume；不得从 checkpoint 重新启动；
- 任一异常必须写入 failure 产物并停止；
- 不允许自动重试、换 GPU、换 seed、减少 case 或追加 updates。

### 8.3 运行后

#### Candidate 失败

若 candidate 任一冻结门槛失败：

```text
decision = BOUNDED_COMPLETION_ROOTED_CSLF_GATE_FAIL
formal_800_authorized = false
automatic_retry = false
```

封存完整负结果并停止。不得在同一版本中增加训练步数、调整 loss、修改阈值或创建

#### Candidate 通过

若 candidate 全部门槛通过，且两个 controls 与全部执行契约完整：

```text
decision = BOUNDED_COMPLETION_ROOTED_CSLF_GATE_PASS
bounded_candidate_qualified = true
```

仍然必须在写入 COMPLETE、核对全部文件 SHA256 后停止。本协议中的 bounded pass
只表示 v15A 获得进入下一次冻结决策的资格，不自动启动或授权 Formal-800。

#### Control 结果

无论 controls 通过还是失败，都只按实际结果完整报告。不得用 control outcome
覆盖 candidate 决定，也不得据此在本次运行后继续训练。

---

## 9. 明确禁止的范围

在 v15A 单次 bounded-400 完整结束并形成新的独立决定之前，以及本协议所覆盖的
整个执行过程中，均禁止：

- 读取或计算任何新的 \(D_V\) 指标、固定漏检或 calibration 结果；
- 读取或计算任何新的 \(D_T\) 结果；
- 运行 Formal-800；
- 增加确认 seed 或创建 bounded r2；
- 开始 Full CURE；
- 接入 DNANet、UIUNet、MSHNet、SCTransNet 或其他 IRSTD backbone；
- 在 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 三个完整数据集上训练或比较；
- 形成跨 backbone、跨数据集、性能优越性或 ICLR 成功主张；
- 修改 Base、decoder、field threshold、loss measure、训练预算或推理流程。

本协议只回答一个问题：

> 在其余条件完全不变时，把 response-joint 的绝对根从 covered plus endpoint
> 移到 completion-bearing minus endpoint，能否同时保留 response、绝对零水平
> crossing 和 null-state behaviour？

除此之外的任何结论均超出本次证据范围。

---

## 10. 运行前冻结摘要

```text
parent_v15 = complete bounded negative
parent_complete_fingerprint =
    faaa2395623f5edfa0e56ab849d20305b73df1e7b3446b22b834279a2637d14b

candidate = completion_rooted_response_joint
only_scientific_change =
    E_mu(e_plus, e_minus - e_plus)
        ->
    E_mu(e_minus, e_minus - e_plus)

dataset_free_checks = 21
dataset_free_independent_replays = 2
dataset_free_all_pass_required = true
dataset_free_byte_identity_required = true

candidate_thresholds_relaxed = false
controls_same_budget_complete_reporting_required = true
control_outcomes_are_candidate_gates = false

bounded_seed = 42
bounded_updates_per_objective = 400
bounded_attempts = 1
resume_allowed = false
automatic_retry_allowed = false

D_V_accessed = false
D_T_accessed = false
formal_800_authorized = false
full_CURE_authorized = false
cross_backbone_authorized = false
three_dataset_training_authorized = false
```

该摘要与前述公式、21 项 dataset-free 门槛、candidate 逐项门槛和停止规则共同构成
v15A 运行前冻结协议。任何不一致都以更严格、停止执行的解释为准。

---

## 11. 运行前闭合记录

在实际 bounded-400 启动前，冻结实现已得到以下结果：

```text
tests_v15:
    161 passed, 1 skipped

dataset_free:
    status = DATASET_FREE_EXACT_REPLAY_PASS
    checks = 21 / 21
    geometry_cases = 96
    completion_root_probes = 6
    training_results = 9
    receipt_fingerprint =
        44f85b45adc42eaefc79278d4c519aac40a5ed17034b6eb51576452ff4db935d
    canonical_sha256 =
        dda87a98808a8ed4bf4deb9ee9249b74f41970c0cbfc5ebbd03d61f034425e52
    canonical_bytes = 82427
    two_replays_byte_identical = true
    D_R_accessed = false
    D_V_accessed = false
    D_T_accessed = false

create_only:
    static_contract_valid = true
    bounded_output_exists = false
    run_once_available = true
    config_fingerprint =
        1196a7902d12723d5274a4c888e755304e3cdd6385b2c44c4d6725f073056cfe
    implementation_fingerprint =
        7087f339973b5daf06108be5f4a25b3ee5466e79f581037488102d1981e6ec22
    receipt_fingerprint =
        07180d6f157395a9db373a78695f4c82977f9b6a3bc448b3a166c9e983d2b57f
```

其中 `1 skipped` 是测试套件预声明的条件性跳过，不是 v15A 新增失败。文档本身
不属于 implementation binding，因此记录这些结果不会改变冻结代码指纹。
