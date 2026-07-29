# CURE-Lite PACRE-VC v23 \(D_V\) FAIL：审计、候选机制假设与 v24 GCR-PACRE 修改方案

> 日期：2026-07-29  
> 仓库：`Arialliy/cure-lite`  
> 审查范围：PACRE-VC v23、Formal800、固定零水平集 \(D_V\) 评估、Base@A/Base@B 比较、PMOPE 训练目标和独立验签链  
> 当前正式结论：`PACRE_V23_FORMAL_D_V_GATE_FAIL`  
> 当前授权状态：\(D_T\) 未访问、未授权；Full CURE、跨 backbone 和三数据集均未授权  
> 推荐下一候选：**v24 GCR-PACRE（Gated Common-Residual PACRE）**  
> 单一科学修改：**保留 PACRE 的相位残差 odd interaction，但用一个无新增参数、flip-even、实数定义域为 \((0,2)\)、机器合同为 \([0,2]\) 的相位公共证据门控其幅度。**  
> 修订状态：`V24_GCR_PACRE_D_R_R1_EXECUTION_OBSERVABILITY_LOST_NO_DECISION`  
> 修订边界：设计、核心实现、generated-data verification 和正式 \(D_R\)
> preaccess 均已完成；r1 在不可逆 run-start marker 写入后失去执行可观测性，
> 恢复时 tool session、PID 和可认证的正式 \(D_R\) receipt 均不可得。精确退出
> 时间与 OS 原因未知，因而既不是 structural PASS，也不是科学 FAIL。
> r1 的 \(D_V/D_T\) payload authorization 均为 false，可得协议/会话记录中也
> 未观察到其访问；但缺少系统级 open-event ledger，不能把这一点升级为对失联
> 时段实际访问的独立审计证明。OOF、bounded、Formal、\(D_V\) 和 \(D_T\) 均未
> 获授权，也未观察到对应下游命令启动。

---

# 修订状态与阻断解除清单

| 原阻断 | 本版处理 | 状态 |
|---|---|---|
| 将 phase-common 缺口写成已证实根因 | 降级为待由 \(D_R\) OOF/消融检验的候选机制假设 | 已解除 |
| attribution 需要重新 materialize \(D_V\) | 删除 v24 选型对 \(D_V\) 逐目标 attribution 的依赖；v24 机制在不重开 \(D_V\) 的条件下冻结 | 已解除 |
| anchor 总数误写为 153 | 只引用已封存聚合事实：\(170=147+23\) | 已解除 |
| gate 严格开区间断言在 FP32 可失败 | 区分实数公式与机器合同；允许端点并强制饱和审计 | 已解除 |
| v24 继承 v22 `forward_reference` | 要求独立实现 GCR 方程 oracle，禁止继承或调用 fast path | 已解除 |
| policy/fields 不完整 | 冻结唯一 v24 policy identity 与完整中间量/shape ledger | 已解除 |
| OOF 缺 root lineage、cache 隔离、Base 包络与机制消融 | 增加 root-closure split、物理 cache 隔离、pooled evaluator、Base@A/Base@B、v23 和同权重 \(G\equiv1\) 对照 | 已解除 |
| bounded-400 只有原则、没有 paired 合同 | 冻结 `10×40=400`、配对初始化/批次、绝对执行安全 PASS 向量与非授权性 paired diagnostics | 已解除 |
| Formal/seed/\(D_V\)/\(D_T\) 顺序含混 | 冻结 `800×40=32,000`；仅 seed42 进入 \(D_V\)，seed43 仅作训练完整性；\(D_T\) 规则在 \(D_V\) 前预注册 | 已解除 |
| 将“参数不变”误写成“部署图不增加” | 增加 FLOPs、时延、峰值显存和产物大小审计 | 已解除 |
| 六位小数可能放宽正式门禁 | 正式 verifier 从冻结有效 Base ledger 全精度动态计算相对门禁 | 已解除 |

当前阶段状态：

```text
文档合同修订                COMPLETE
v24 core implementation     COMPLETE
prior core/runner/r2 matrix  147 PASS（历史隔离分组）
r2 supervision target        35 PASS（当前 generated/static/fault matrix）
OOF test matrix              25 PASS
full protocol matrix         75 PASS, 1 SKIP
current isolated release     279 PASS, 1 SKIP
dataset-free / efficiency   30/30 PASS（CUDA，正式 receipt 已验签）
D_R preaccess               COMPLETE（authorization 已验签）
D_R structural r1           OBSERVABILITY LOST AFTER MARKER; NO DECISION
D_R structural r2           NOT AUTHORIZED / NOT CREATED
D_R OOF/bounded/Formal      BLOCKED（无 structural PASS）
D_V / D_T                   BLOCKED
```

不能只用“代码完成”描述模型进度。按设计、实现、机械验证和 decision-bearing
真实运行四层拆分，当前状态是：

| 阶段 | 设计 | 代码 | generated/static/mechanical 验证 | 可判定真实运行 |
|---|---|---|---|---|
| GCR-PACRE 数学模型 | 完成 | 完成 | algebra、flip parity、gradient、FP64 oracle、fields/policy 已测 | 未获真实性能验证 |
| dataset-free / efficiency | 完成 | 完成 | CUDA 30/30 PASS，正式 receipt 已验签 | 只支持机制/效率，不支持性能主张 |
| \(D_R\) structural | 完成 | 完成 | generated、preaccess、source closure 与 runtime fault matrix 已测 | r1 无决定；r2 未创建、未授权 |
| source-disjoint OOF-4 | 完成 | 完成 | 25 PASS；160 samples/156 roots 的 metadata split 已预注册 | 未授权、未运行 |
| paired bounded-400 | 完成 | 完成 | synthetic cache 上 candidate/control 各 400 updates 已测 | 正式 full-\(D_R\) 未运行 |
| Formal800 | 完成 | 完成 | 两个 seed 的 32,000-update schedule/trace/artifact mechanics 已测；trainer 为测试替身 | 无真实 checkpoint/terminal/result |
| \(D_V\) | 相对门禁与一次性治理完成 | aggregate verifier/decision 已实现 | generated evidence 已测 | 未授权、未运行 |
| \(D_T\) | preregistration 完成 | validator/template 已实现 | schema 与顺序已测 | 未授权、未运行 |

因此当前是“工程准备度高、科学性能验证尚未越过 \(D_R\) structural”的状态，
不能把 supervisor scaffold、generated PASS 或 systemd exit 当成模型性能完成。

2026-07-30 的单进程全量诊断得到 `273 PASS, 1 SKIP, 6 FAIL`；六个失败均为
fail-closed source audit 看到同一 pytest 进程先前加载的 `cure_lite.toy` 测试模块，
而 source-closure-sensitive training-runner 文件在全新进程中为 `16 PASS`。其余
测试在另一个全新进程中为 `263 PASS, 1 SKIP`，所以当前有效的隔离 release
matrix 合计为 `279 PASS, 1 SKIP`。这不是模型数值 FAIL，而是单进程测试编排违反
“真实阶段必须使用隔离进程”的闭包合同后被正确拒绝。release/actual launch 必须
继续使用双进程矩阵；不能简单把 toy 模块加入正式 source closure，也不能在生产
scanner 中硬编码忽略它们。

正式 dataset-free/efficiency receipt：

```text
protocols/IRSTD-1K/gcr_pacre_v24/dataset_free_receipt_r2.json
receipt fingerprint =
efd55afcb8709ed20fb67aad8696c348bda166323a548018c7a7f282f1adad0a
file SHA-256 =
845132fa2930b16ad18f220c5496524cf65cb2d41874a85944c61fd3b0872883
```

该 receipt 是 generated-only 机制/效率证据：\(D_R/D_V/D_T\) 均未访问，
不包含真实性能主张；其唯一晋级含义是“可进入外部
\(D_R\)-structural authorization 检查”。

正式 \(D_R\) r1 的历史记录与权威解释性纠错回执：

```text
历史 v1（原字节永久保留）：
protocols/IRSTD-1K/gcr_pacre_v24/
    D_R_structural_attempt_r1_interruption_receipt.json
receipt fingerprint =
f5859ea86515eefaaadec5757b087f8e76bf2f379815d0809dbb33dafa30f0c5
file SHA-256 =
c97962bb351ee973f22af06defb80cd8d2151ff738f34a4eda6c7d89f0212eab

权威解释 v2：
protocols/IRSTD-1K/gcr_pacre_v24/
    D_R_structural_attempt_r1_interruption_receipt_v2.json
receipt fingerprint =
d599df00a201d65f8cbd4b9390a4ef784a2a504200923b9894e12cd26d249792
file SHA-256 =
5b0318c514366c897a5092f794f4aa4ead25c29d48d0d2d9b16a1f3dddebdc1c
```

v2 只取代 v1 的**证据解释**，不改写 v1 字节，不把 r1 转换为可判定运行，也不
替代仍不存在的正式 `D_R_structural_receipt.json`。v1/v2 应准确称为本地威胁
模型下的 create-once、read-only、self-fingerprinted 产物；0444、单硬链接、
文件 SHA 与内部 fingerprint 能支持本地变更检测，但不是 filesystem
append-only，不带外部签名，也不能防止恶意同用户替换。

第一次封存的 `dataset_free_receipt.json` 在任何 \(D_R\) 执行或下游
authorization 创建前，因训练时 validator 尚未逐项覆盖全部非布尔 ledger 的
finite 检查而失效。旧文件保留用于审计，不能授权任何下游阶段；失效原因、旧
hash、修复后 source hash 和 r2 binding 均追加记录在：

```text
protocols/IRSTD-1K/gcr_pacre_v24/
    dataset_free_receipt_r1_invalidation.json
```

---

# 0. 执行结论

PACRE-VC v23 的工程与证据链已经闭合：

```text
PACRE-VC 路线             已采用
FACRE 路线                本次正式路线已拒绝
D_R gate                  唯一真实运行，13/13 PASS
Formal800                 seed 42，从零训练
Formal updates            800 × 40 = 32,000
Formal terminal           独立验签 PASS
D_V Base@B threshold      0.14，来自冻结 51 点网格
D_T                       从未访问
```

正式 \(D_V\) 科学结果为：

| 指标 | Base@A | Base@B | CURE |
|---|---:|---:|---:|
| true targets | 147 | 150 | 149 |
| recovered misses | 0 | 3 | 2 |
| mIoU | 0.6095592799503414 | 0.6076294277929155 | 0.6021406727828746 |
| nIoU | 0.5653280526756584 | 0.5640138505062713 | 0.5608662365828017 |

因此，当前结果不是“接近 PASS”，而是同时失败于两类科学要求：

1. **检测数量不足**
   \[
   149<150,\qquad 2<3.
   \]

2. **分割质量回退**
   \[
   0.6021406727828746<0.6095592799503414,
   \]
   \[
   0.5608662365828017<0.5653280526756584.
   \]

当前严格规则没有任意 \(+2\) 门槛。整数指标只要求严格大于 best valid Base，所以新的候选在同一 \(D_V\) 上至少需要：

\[
\boxed{\text{true targets}\ge151}
\]

和：

\[
\boxed{\text{recovered misses}\ge4}.
\]

相对当前 CURE，这不是再增加 1，而是至少增加 2：

```text
149 -> 151
2   -> 4
```

相对当前 v23 CURE 的差值仅用于解释失败幅度：

\[
\Delta\mathrm{mIoU}
=
0.007418607167466784
\]

和：

\[
\Delta\mathrm{nIoU}
=
0.004461816092856785.
\]

这些差值不是 v24 的硬编码 uplift。正式 verifier 不保存六位小数门槛，也不
要求任意固定绝对提升量；它必须从绑定 fingerprint 的冻结有效 Base ledger
读取原始全精度数值，逐坐标动态执行“整数严格提升、连续指标不回退”。

Retention 和误警预算通过，说明当前失败不是“背景完全失控”，但也不能据此认为 completion 形状合格。误警预算是总体安全约束，不等价于目标区域像素精确、边界紧致或 IoU 非回退。

下一步应建立一个新版本，而不是：

- 重跑 v23；
- 选择其他 checkpoint；
- 改 field 阈值；
- 修改 Base@B 网格；
- 放宽 IoU；
- 访问 \(D_T\)；
- 将机械 verifier 纠错改写成科学 PASS。

推荐的 v24 仍保持 CURE-Lite 主线，只修改 PACRE 如何让 phase-common
compatibility 调节 residual interaction 的幅度。该解释是候选机制假设，不是
由现有 \(D_V\) 聚合数值证明的失败根因。

## 0.1 科学主线、候选身份与执行身份

“保持主线”不等于永久冻结所有实现细节。这里必须区分三种身份：

1. **科学主线**：在冻结 Base anchor 上，通过 common--residual compatibility
   学习 completion；保持单一零水平集和既定输出/匹配协议。晋级目标是主检测
   指标相对所有冻结有效 Base 严格提升，同时 mIoU、nIoU 不回退，并满足
   retention 与误警安全约束。该相对 Pareto 规则不包含任意固定绝对 uplift。
2. **科学候选身份**：GCR-PACRE-v24 的具体 gate、normalization、readout、
   objective、采样和优化配置一经封存，就不能在仍称为 v24 的运行中修改。
   非主线组件只要有明确的性能假设，允许修改；但每次实质修改必须建立新的
   candidate/version、重新预注册、重建 source closure，并提供能区分各修改贡献的
   消融证据。
3. **执行尝试身份**：fresh r2 只用于恢复 v24-r1 的不可判定执行，必须保持同一
   科学候选和数值语义。任何模型、loss、归一化、采样、优化器或日程修改都会使
   它不再是 r2，而是新科学候选的首次执行。

新候选选择不得读取 \(D_V/D_T\)，也不得在反复查看同一个 OOF 后继续把该 OOF
称为未见证据。多候选选择必须预注册候选集，或使用 nested/new internal-unseen
划分。r2 不是科学上的必经步骤：在 r2 authorization 创建前，可以停止评估 v24
并建立新候选；但新候选不能继承 r2 身份或 r1/r2 attempt lineage。

如果继续评估当前已经封存的 v24 候选，则唯一合法的阶段顺序为：

```text
v23 inheritance + v24 design freeze（不重开 D_V）
    -> implementation + unit/source closure
    -> dataset-free algebra/gradient/reference
    -> paired v23/v24 efficiency audit
    -> read-only D_R structural gate
    -> source-disjoint D_R OOF-4
    -> paired full-D_R bounded-400
    -> Formal seed42 primary, 800×40=32,000
    -> Formal seed43 training-integrity, 800×40=32,000
    -> freeze D_T preregistration while D_T remains unread
    -> one seed42 D_V adaptive evaluation
    -> only-if-D_V-PASS create one-shot seed42 D_T authorization
```

任一前置项 FAIL 即停止；后续 authorization 不得预建。

---

# 1. 当前门禁到底要求什么

v23 的比较不是要求 CURE 超过某一个单独 Base 行，而是分别取所有 valid Base 行的坐标最大值：

\[
T_{\mathrm{best}}
=
\max(T_{\mathrm{Base@A}},T_{\mathrm{Base@B}}),
\]

\[
R_{\mathrm{best}}
=
\max(R_{\mathrm{Base@A}},R_{\mathrm{Base@B}}),
\]

\[
I_{\mathrm{best}}
=
\max(I_{\mathrm{Base@A}},I_{\mathrm{Base@B}}),
\]

\[
N_{\mathrm{best}}
=
\max(N_{\mathrm{Base@A}},N_{\mathrm{Base@B}}).
\]

所以本次比较实际是：

```text
target count / recovered misses：对 Base@B
mIoU / nIoU：                 对 Base@A
```

即 CURE 必须同时实现：

- 比低阈值 Base 多检出目标；
- 又不能像低阈值 Base 那样牺牲分割质量；
- 还要满足 retention 与误警预算。

这相当于要求 CURE 超过两条 Base operating points 的坐标包络。它比“只超过 Base@B”更严格，但这是已经冻结的合理安全合同，不应在结果可见后修改。

---

# 2. 当前结果的定量失败签名

## 2.1 Hard union 说明少掉的不是 Base 已检目标

CURE 的正式输出为：

\[
O=\mathbf1[p_b\ge0.72],
\]

\[
C=\mathbf1[\phi<0]\land\neg O,
\]

\[
Y=O\lor C.
\]

因此 CURE 不会删除 Base@A 的正像素。

Retention 已经通过，意味着：

> CURE 的 149 个 true targets 是 Base@A 的 147 个目标，加上 completion 新恢复的 2 个目标；失败来自“没有再恢复足够目标”，而不是破坏了 Base@A 已有目标。

但是仅凭聚合数值不能断言 CURE 恢复的两个目标一定是 Base@B 三个目标的子集。可能存在：

- Base@B-only targets；
- CURE-only targets；
- 二者共同恢复；
- 二者均未恢复。

聚合数值不能回答这些集合关系。本 v24 路径为避免再次 materialize \(D_V\)，
不补做 target-ID 交集；相应地也不把任何一种集合关系写成已证实归因。

## 2.2 CURE 的形状效率明显低于简单降 Base 阈值

相对 Base@A：

\[
\Delta\mathrm{mIoU}_{\mathrm{Base@B}}
=
-0.001930
\]

并换来 3 个恢复目标。

相对 Base@A：

\[
\Delta\mathrm{mIoU}_{\mathrm{CURE}}
=
-0.007418
\]

只换来 2 个恢复目标。

若仅作诊断性比率：

\[
\frac{0.007418/2}{0.001930/3}
\approx5.77.
\]

nIoU 的对应比率约为：

\[
5.09.
\]

该比率不是新的正式指标，因为 mIoU/nIoU 并非对目标数量线性可分；但它明确提示：

> 当前 CURE 每恢复一个目标所付出的聚合分割代价，远高于 Base@B。

这通常对应以下一种或多种情况：

- completion 在 GT 外扩张；
- completion 只命中目标局部，形状不完整；
- completion 与 Base component 错误连接；
- 恢复目标的 mask precision 较差；
- 在未形成新 false-positive component 的情况下，增加了大量连接到已有 component 的错误像素；
- 新恢复的目标虽然满足 target-level match，但 mask 本身质量低。

Retention 和 component/pixel FA 预算并不能排除这些情况。

---

# 3. 候选机制假设 H1：common evidence 不能独立驱动 PACRE 输出

## 3.1 当前 PACRE compatibility

PACRE 计算：

\[
\bar A_F
=
\frac1P\sum_{p=1}^{P}A_F^p,
\]

然后：

\[
J^{\mathrm{common}}_p
=
A_U+\bar A_F,
\]

\[
J^{\mathrm{specific}}_p
=
A_U+A_F^p.
\]

compatibility hidden：

\[
H^{\mathrm{res}}_p
=
\operatorname{SiLU}
\left(
J^{\mathrm{specific}}_p
\right)
-
\operatorname{SiLU}
\left(
J^{\mathrm{common}}_p
\right).
\]

其读出为：

\[
R_p(O,F)
=
w^\top H^{\mathrm{res}}_p.
\]

PACRE 最终只使用 actual occupancy 与当前 phase binary flip 的 antisymmetric difference：

\[
D_p(O,F)
=
\frac12
\left[
R_p(O,F)
-
R_p(\widetilde O_p,F)
\right].
\]

\[
\phi
=
\operatorname{PixelShuffle}
\left(
0.9+D
\right).
\]

## 3.2 已证明的不变性与不能推出的结论

如果某个位置的 feature evidence 在所有 phase 上相同：

\[
A_F^p=\bar A_F,
\]

则：

\[
J^{\mathrm{specific}}_p
=
J^{\mathrm{common}}_p,
\]

因此：

\[
H^{\mathrm{res}}_p=0,
\]

\[
R_p=0,
\]

\[
D_p=0.
\]

在这个严格的 phase-identical 子空间内：

> phase-common feature 不能独立产生 PACRE interaction。

但不能把它外推成“所有 common evidence 均被消除”。令：

\[
A_F^p=M+\delta_p,\qquad M=\bar A_F,
\]

则 PACRE hidden 为：

\[
\operatorname{SiLU}(A_U+M+\delta_p)
-
\operatorname{SiLU}(A_U+M).
\]

当 \(\delta_p\ne0\) 时，\(M\) 仍会通过 SiLU 的非线性工作点和局部斜率影响
residual。准确表述只能是：

> common component 不能独立驱动输出；当 residual 存在时，它仍隐式调节
> residual 的工作点。GCR 显式引入的，是一个可审计的幅度门控路径。

## 3.3 与 Base@B 结果的关系

Base@B 将 Base probability threshold 从 0.72 降到 0.14，恢复了 3 个目标，并且仍在冻结误警预算内。

Base@B 直接利用的是：

> Base 输出中已经存在但低于 0.72 的绝对低置信度 evidence。

PACRE 的公开输入不包含 Base probability，只包含：

\[
(F_b,O_{0.72}).
\]

PACRE 以 phase-common 部分作为 reference，使其不能独立进入最终 odd
interaction；但它仍可影响非线性 residual 的工作点。

因此，当前结果与以下机制假设一致：

> 至少一部分 Base@B 能恢复的漏检，在冻结 feature 中表现为较强的绝对或 phase-common evidence，但其 phase residual 不足以让 PACRE field 穿过固定零水平集。

这不是由现有聚合结果证明的事实，Base probability 与新定义的 \(E_p\) 也未被
证明同号或可分。v24 把它作为**预先冻结的候选机制假设**，只允许在不读取
\(D_V/D_T\) 的 \(D_R\) OOF、\(G\equiv1\) 消融和 held-out \(E/G\) 分布中检验。

---

# 4. 候选机制假设 H2：显式有界 common gate 可能补足 residual 幅度

当前 field：

\[
\phi=0.9+D.
\]

要形成 completion，必须：

\[
D<-0.9.
\]

一个目标可能已经有方向正确的 residual interaction：

\[
D<0,
\]

但幅度只有：

\[
-0.9<D<0.
\]

此时：

- PACRE 知道正确变化方向；
- 但固定零水平集不会产生 completion；
- Base@B 可能已从绝对 probability evidence 中恢复该目标。

当前 PACRE 没有一个机制能够表达：

> “该 phase-relative evidence 虽然偏弱，但它所在位置具有强烈、可信的 phase-common target evidence，因此应增强其幅度。”

同理，PACRE 也没有利用 phase-common background evidence 去抑制某些错误 residual release。

这是 GCR-PACRE 要检验的可证伪命题，而不是既定事实。共用读出 \(w\) 未必
能在 target/background 上形成所需的 \(E_p\) 符号；若 held-out \(D_R\)
选择性证据不成立，v24 必须停止，不能借助 \(D_V\) 重新选机制。

---

# 5. 失败原因三：当前训练目标与最终目标级/IoU 门禁仍有距离

## 5.1 PMOPE 的目标

PMOPE 对 plus/minus 两个 endpoint 构造：

\[
q_\sigma
=
\left[
m_0
-
\operatorname{sign}(\phi_\sigma^\star)
\phi_\sigma
\right]_+,
\]

其中：

\[
m_0
=
\frac{0.9}{4}
=
0.225.
\]

然后将两个 violation fields 放入一个共同的 \(W^{1,4}\) energy。

该目标的优点是：

- sign 与固定零水平集一致；
- margin 不是实验搜索值；
- field 进入正确 orthant 后不继续受到过度惩罚；
- 同时约束 value 与 spatial variation。

## 5.2 仍然存在的差异

正式 \(D_V\) 检查的是：

- target-level connected matching；
- recovered miss count；
- mIoU；
- nIoU；
- false-positive components；
- pixel/raw-background FA。

PMOPE 优化的是：

- D_R 上的连续 field margin；
- 平衡 integration measure 下的平均 \(p=4\) energy；
- synthetic pair 与自然状态的 field geometry。

即使 PMOPE 很低，也不自动保证：

- 新 completion component 对应一个完整目标；
- 所有新增像素都位于 GT；
- 新 component 与已有 Base component 不发生错误连接；
- target-level count 超过低阈值 Base；
- 未见 D_V 的 mask IoU 不回退。

本次 D_V FAIL 说明这种训练—评估间距在真实泛化上没有被弥合。

## 5.3 为什么 v24 暂不同时修改 PMOPE

当前候选结构假设是：显式的 phase-common compatibility gain 可能弥补 residual
幅度不足；它尚未被证明，且 common component 在 \(\delta_p\ne0\) 时仍会影响
PACRE 的非线性工作点。

为了保持单变量机制检验，v24 应：

```text
修改 PACRE interaction
保持 PMOPE 不变
```

如果同时改模型与 objective，即使 v24 成功，也无法判断成功来自：

- 显式 common-gain path；
- 新 loss；
- 两者交互；
- 额外隐式容量。

因此 v24 首先测试 representation correction。

如果 v24 在内部未见 gate 中实现：

- target count 提升；
- IoU 不回退；

才进入 Formal。

如果 v24 仍表现为“count 增加但 IoU 回退”，后续 v25 才允许保持 v24 model、单独修改目标函数。

---

# 6. 失败原因四：D_R 13/13 PASS 不是性能泛化保证

v23 的 \(D_R\) gate 验证的是：

- PACRE algebra；
- phase residual；
- latent witness；
- collision；
- readout anchor；
- initialization gradient path；
- field direction；
- read-only execution。

它不训练、不评估 Pd/IoU，也没有证明：

- 每个未见目标都有 phase residual witness；
- witness 幅度足以穿过 \(\phi=0\)；
- common evidence 与 background 可分；
- D_R 学到的 interaction 在 D_V 上保留；
- completion geometry 优于 Base threshold relaxation。

因此：

```text
D_R 13/13 PASS
+
Formal800 正常
```

和：

```text
D_V FAIL
```

并不矛盾。

它们说明的是：

> PACRE 结构可执行、可训练、在 D_R 具备局部 witness，但该归纳偏置没有在 D_V 上胜过最强 valid Base operating point。

---

# 7. 失败原因五：Formal800 缺少真正的未见性能筛选

v23 Formal800 的直接 prerequisite 是：

```text
D_R 13/13 PASS
```

源码明确将 bounded-400 设为：

```text
bounded_400_required = false
```

所以模型在没有经历新的 source-disjoint performance gate 的情况下直接进行了：

```text
32,000 updates
    ->
D_V
```

这不是执行错误，因为它是预注册协议；但从下一版本设计看，它导致：

- 结构正确性已经验证；
- 优化完整性已经验证；
- 真实未见性能直到 D_V 才第一次暴露。

下一版本必须恢复一个**训练源分组隔离的 OOF performance gate**，避免再次用一次完整 Formal800 才发现同类泛化失败。

---

# 8. 独立 verifier 的机械错误不属于科学失败

原 independent verifier 的白名单遗漏：

```text
formal_result_fingerprint
```

导致对合法 Formal result 的假阳性报错。

该字段实际属于正式 model binding，并在绑定、重载和一致性检查中使用。

现有处理是正确的：

- 不重跑 D_V；
- 不改写原 result；
- 不改写 decision；
- 用 append-only 机械纠错完成验签；
- 科学决定仍是 FAIL。

v24 必须防止同类问题：

1. verifier 的字段集合由 schema/dataclass 自动生成；
2. required field 缺失必须失败；
3. schema 中合法字段必须接受；
4. schema 外未知字段必须失败；
5. `formal_result_fingerprint` 必须有专门回归测试；
6. mechanical correction receipt 与 scientific decision 分开。

---

# 9. \(D_V\) 结果可见后的数据治理

仓库预注册已经明确：

> v23 的 D_V 是 adaptive development evaluation，而不是 independent confirmation。

v24 的设计已经知晓 v23 的冻结聚合结果，但本版不重新 materialize \(D_V\)
的逐样本、逐目标、field 或 mask payload。必须承认：

- v24 是看过 v23 D_V 结果后设计的；
- v24 的 D_V 即使 PASS，也仍是 adaptive evidence；
- \(D_T\) 才是未来唯一未见最终确认；
- 不能不断在 D_V 上试多个版本后挑成功者，再把 D_T 当作普通测试集。

当前 v24 候选身份内的冻结纪律：

```text
v23 D_V FAIL
    -> 仅继承 sealed aggregate result / decision / fingerprints
    -> 在不重开 D_V 的条件下冻结 v24 机制
    -> dataset-free + D_R structural
    -> D_R source-group OOF gate
    -> paired bounded-400
    -> v24 Formal seed42 primary + seed43 training-integrity
    -> 预先冻结 D_T protocol
    -> 仅 seed42 进行一次 v24 D_V adaptive gate
    -> PASS 后才创建 D_T authorization
```

若 v24 D_V FAIL：

```text
冻结 v24
不访问 D_T
新版本必须重新建立内部未见 gate
```

---

# 10. v23 继承收据与“不重开 \(D_V\)”合同

v24 只允许建立 append-only inheritance receipt，读取范围限于已经封存的：

```text
v23 result / decision / COMPLETE fingerprints
valid Base row identifiers and exact aggregate metric scalars
v23 aggregate CURE metric scalars
D_V access counters already recorded in the sealed receipt
```

禁止读取或重建：

```text
D_V images / tensors / labels / logits / features
per-sample or per-target rows
target IDs and overlap sets
field / mask / component decomposition
counterfactual GCR outputs under v23 weights
```

现有封存聚合事实为：

```text
Base@A covered targets = 147
Base@A misses          = 23
anchor total           = 170
```

`170 = 147 + 23` 只用于修正文档算术，不能据此伪造不存在的逐目标 ledger。
H1/H2 的选择性、方向和 shape 证据全部迁移到 source-disjoint \(D_R\) OOF。
这样 v24 机制在第二次 \(D_V\) payload access 之前已经冻结；若未来确需
\(D_V\) attribution，必须作为新的 adaptive access 单独授权、增加 access
counter，并承认后续 \(D_V\) 不再提供一次性候选证据。本 v24 路径不采用该分支。

---

# 11. 推荐新模型：v24 GCR-PACRE

名称：

> **Gated Common-Residual PACRE**  
> **GCR-PACRE-v24**

核心思想：

> 保留 PACRE 的 phase-residual binary-flip interaction 作为唯一可改变 completion
> 符号的方向项；phase-common evidence 只能作为一个有界、flip-even 的幅度
> gate，不能独立生成 completion。

---

# 12. GCR-PACRE 数学定义

## 12.1 Residual compatibility 保持 PACRE

对 actual occupancy \(O\)：

\[
R_p(O,F)
=
w^\top
\left[
\operatorname{SiLU}(A_U(O)+A_F^p)
-
\operatorname{SiLU}(A_U(O)+\bar A_F)
\right].
\]

对当前 phase binary flip：

\[
R_p(\widetilde O_p,F).
\]

odd residual interaction：

\[
D_p(O,F)
=
\frac12
\left[
R_p(O,F)
-
R_p(\widetilde O_p,F)
\right].
\]

这与 v23 PACRE 完全相同。

## 12.2 新增无参数 common compatibility

定义：

\[
C_p(O,F)
=
w^\top
\left[
\operatorname{SiLU}(A_U(O)+\bar A_F)
-
\operatorname{SiLU}(A_U(O))
\right].
\]

它表示：

> phase-common feature evidence 相对于 occupancy-only state 的 compatibility。

使用同一个：

\[
w
\]

作为 scalar readout，因此不增加参数。

对 binary flip 同样计算：

\[
C_p(\widetilde O_p,F).
\]

## 12.3 Flip-even common score

\[
E_p(O,F)
=
\frac12
\left[
C_p(O,F)
+
C_p(\widetilde O_p,F)
\right].
\]

交换 actual 与 flipped occupancy 时：

\[
E_p
\]

保持不变。

## 12.4 固定有界 gate

\[
\boxed{
G_p(O,F)
=
2\sigma(E_p(O,F))
}
\]

在实数算术中：

\[
0<G_p<2.
\]

但 PyTorch FP32/FP16/BF16 的 sigmoid 可因舍入得到精确端点。正式机器合同
必须是：

\[
\boxed{
E_p\ \text{finite},
\qquad
G_p\ \text{finite},
\qquad
0\le G_p\le2
}
\]

不能因 `G == 0` 或 `G == 2` 抛出异常。每个 dataset-free、\(D_R\)、
bounded 和 Formal receipt 必须保存：

```text
gate_total
gate_eq_0_count / fraction
gate_eq_2_count / fraction
gate_interior_count / fraction
common_even_min / max / mean / q01 / q50 / q99
```

饱和率是强制报告的诊断量，不在结果可见后发明阈值。generated-state
dataset-free fixture 必须至少产生一个严格 interior gate，以排除实现恒定端点。

当 common evidence 为零时：

\[
E_p=0
\Rightarrow
G_p=1.
\]

没有 temperature、gain 参数或 threshold 搜索。

## 12.5 最终 interaction

\[
\boxed{
I_p(O,F)
=
G_p(O,F)D_p(O,F)
}
\]

最终：

\[
\boxed{
\phi_\theta(F,O)
=
\operatorname{PixelShuffle}
\left(
0.9+I
\right)
}
\]

推理仍为：

\[
Y_{\mathrm{completion}}
=
\mathbf1[\phi<0]\land\neg O,
\]

\[
Y_{\mathrm{final}}
=
O\lor Y_{\mathrm{completion}}.
\]

---

# 13. GCR-PACRE 的关键性质

## 13.1 Common evidence 不能独立造目标

如果 PACRE residual direction 为零：

\[
D_p=0,
\]

则无论 common evidence 多大：

\[
I_p=G_pD_p=0.
\]

因此：

> phase-common evidence 只能调节已有 residual interaction，不能凭空产生新的
> completion direction。

这比直接把 common term加到 field 更安全。

## 13.2 保留 binary-flip antisymmetry

交换 \(O\) 与 \(\widetilde O_p\)：

\[
D_p\mapsto-D_p,
\]

而：

\[
E_p\mapsto E_p,
\]

\[
G_p\mapsto G_p.
\]

所以：

\[
I_p\mapsto-I_p.
\]

原 PACRE binary flip 主线保持。

## 13.3 PACRE 是 GCR-PACRE 的中性特例

如果：

\[
E_p=0,
\]

则：

\[
G_p=1,
\]

\[
I_p=D_p.
\]

所以 GCR-PACRE 在无 common evidence 时退化为 PACRE。

## 13.4 对 target 与 background 具有相反的潜在效果

若 target pixel：

\[
D_p<0,
\qquad
G_p>1,
\]

则：

\[
0.9+G_pD_p
<
0.9+D_p.
\]

目标 field 更容易穿过零点。

若 background pixel 也存在错误负 residual：

\[
D_p<0,
\qquad
G_p<1,
\]

则：

\[
0.9+G_pD_p
>
0.9+D_p.
\]

错误 completion 被抑制。

这说明同一个机制**有可能**同时针对：

- target count 不足；
- IoU/nIoU 回退。

它不保证训练后 target 一定满足 \(G>1\)、background 一定满足 \(G<1\)。
这种选择性必须由 held-out \(D_R\) 的 \(E/G\) 分布和 \(G\equiv1\) 消融证明。

## 13.5 增益有上界

\[
0\le G_p\le2
\]

在机器算术中也不会像 exponential gate 一样产生无界放大。对于固定的 v23
权重，只有 \(D_p<-0.45\) 的位置才可能在最大 gain 下跨过零点；但 v24
重新训练会同时改变 \(D\)，因此该零样本诊断不能被用作架构授权或否决条件。

## 13.6 参数量不变

GCR-PACRE 复用：

```text
joint_state_weight
joint_hidden_bias
scalar_energy_weight
```

参数量仍为：

\[
64,064.
\]

参数张量数、输入和输出接口不增加；**计算图会增加** common hidden/readout、
sigmoid 和逐点乘法，因此不能宣称部署图或运行开销不变。第 18 节要求单独
封存 FLOPs、时延、峰值显存与产物大小审计。

---

# 14. 为什么不直接恢复被中心化的 common term

一个更直接的候选是：

\[
R_p+C_p.
\]

但这允许 common evidence独立改变 field，即使：

\[
D_p=0.
\]

这会重新引入：

- phase-common background 直接造目标；
- broad support；
- 更多 field spill；
- IoU 进一步回退。

GCR-PACRE 使用：

\[
G_pD_p
\]

而不是：

\[
D_p+C_p,
\]

是因为当前失败同时包含：

- 少检目标；
- shape 质量不足。

门控方案优先保持 PACRE 的相位定位，只补充 amplitude confidence。

---

# 15. 实现的规范性合同

新 package：

```text
cure_lite_v24/
```

模型文件：

```text
cure_lite_v24/gcr_pacre.py
```

下方代码只保留 algebra/dataflow 示意，**不是可直接复制的 production
implementation**。正式实现以本节以下规范为准；凡示意片段与规范冲突，以规范
为准。

## 15.1 唯一 policy identity

v24 必须只有一组 canonical identity：

```text
method_id          = cure_lite_gcr_pacre_v24
field_policy       = gcr_pacre_single_zero_level_set_field_v1
equation_policy    = flip_even_common_gate_times_flip_odd_residual_v1
interaction_policy= bounded_even_gate_times_binary_flip_odd_residual_v1
energy_policy      = shared_readout_residual_and_common_compatibility_v1
numerical_policy   = finite_closed_gate_interval_with_saturation_audit_v1
```

不再新增一个 `gated_equation_policy` 同时保留 PACRE 的旧
`equation_policy/field_policy/interaction_policy`。实现应直接基于 PAET 参数
拓扑与 helper，不能依赖 v22 模型的类型检查或旧 policy identity。构造、保存、
重载和 verifier 都必须逐字段拒绝 legacy PACRE identity。

## 15.2 完整 fields ledger

`CoverageStateGCRPACREFields` 至少保存并验证以下张量，不能用下划线丢弃：

```text
encoded_feature
phase_occupancy
occupancy_affine
coarse_feature_affine
upsampled_feature_affine
phase_feature_affine
phase_feature_mean
phase_feature_residual

actual_occupancy_only_joint_affine
actual_common_joint_affine
actual_specific_joint_affine
actual_common_silu
actual_residual_hidden / actual_residual_energy
actual_common_hidden / actual_common_energy

center_phase_weight
flip_delta
flipped_center_phase_value
flipped_occupancy_affine
flipped_occupancy_only_joint_affine
flipped_common_joint_affine
flipped_specific_joint_affine
flipped_common_silu
flipped_residual_hidden / flipped_residual_energy
flipped_common_hidden / flipped_common_energy

residual_odd_interaction
common_even_energy
common_gate
common_gate_zero_saturation
common_gate_two_saturation
gated_interaction
native_phase_field
field
output_size
```

其中 `flipped_center_phase_value` 是只表示每个 phase 中心 occupancy 反事实值的
布尔张量，shape 固定为 \([B,P,h,w]\)；它不是完整的
`flipped_phase_occupancy`，也绝不能扩成 \([B,P,P,h,w]\)。

验证分成两层，不能混为一谈：

1. 训练时轻量 validator：`_validate_gcr_fields` 必须逐项验证 exact shape、
   dtype、device、contiguity、finite、output size、policy identity、gate
   闭区间以及 center bit/flip delta、\(D/E/G/I\)、native field 和
   `PixelShuffle` 等本次前向已经形成的代数关系；一次前向只允许一次聚合
   device-to-host truth sync。
2. 外部完整 replay：`validate_gcr_pacre_fields` 必须从原始
   `feature/occupancy` 和当前参数重新形成 affine/hidden/energy/field，
   逐字段 exact replay；它只用于显式审计，不进入 32,000-update 训练热路径。

`common_joint` 的 SiLU 必须缓存并同时用于 residual/common hidden，不能重复
计算后再假定逐位相同。

## 15.3 独立 GCR oracle

v24 必须显式覆盖 `forward_reference`。该 oracle：

- 全程以 FP64 从原始 `feature/occupancy` 和参数开始重算；
- 直接逐 batch/phase/cell 重算 \(R(O),R(\widetilde O),C(O),C(\widetilde O)\)；
- 显式形成 \(D,E,G,I\) 和 `PixelShuffle(0.9 + I)`；
- 不调用 `forward_fields`、`_affine_states`、`_compatibility_components`、
  phase-centering fast helper 或任何 fast-path 中间结果；
- 不继承 v22 的 ungated `0.9 + D` oracle；
- 在 FP64 oracle 与生产 FP32 fast path 之间使用实现冻结前生成并由测试固定的
  双重 roundoff envelope：最终 field 最大绝对误差
  \(\le 2\times10^{-6}\)，且最大 ULP distance \(\le32\)；
- 单独验证数学 parity，不能要求两个不同 dtype/求和顺序逐位相等。

dataset-free tests 必须构造一个 `G != 1` 的 fixture，证明 v24
`forward_reference` 不是意外退回 v23 方程。

## 15.4 algebra/dataflow 示意

```python
# Pseudocode only. Config, complete Fields, validator, and independent
# forward_reference are the mandatory objects defined in 15.1--15.3.
class GCRPACREV24FastPath:
    def _compatibility_components(
        self,
        occupancy_affine: Tensor,
        phase_feature_affine: Tensor,
        phase_feature_mean: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        occupancy_only_joint = occupancy_affine.expand_as(
            phase_feature_affine
        )
        common_joint = (
            occupancy_affine + phase_feature_mean
        ).expand_as(phase_feature_affine)
        specific_joint = (
            occupancy_affine + phase_feature_affine
        )

        common_silu = F.silu(common_joint)
        residual_hidden = F.silu(specific_joint) - common_silu
        common_hidden = (
            common_silu - F.silu(occupancy_only_joint)
        )

        readout = self.scalar_energy_weight[
            None, None, :, None, None
        ]
        residual_energy = (
            residual_hidden * readout
        ).sum(dim=2)
        common_energy = (
            common_hidden * readout
        ).sum(dim=2)

        return (
            residual_hidden.contiguous(),
            residual_energy.contiguous(),
            common_hidden.contiguous(),
            common_energy.contiguous(),
        )

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CoverageStateGCRPACREFields:
        output_size = self._validate_inputs(feature, occupancy)

        encoded_feature = normalize_cslf_feature(
            feature,
            epsilon=self.config.normalization_epsilon,
        )
        phase_occupancy = pixel_unshuffle_bool_occupancy(
            occupancy,
            stride=self.config.feature_stride,
        )

        (
            occupancy_affine,
            coarse_feature_affine,
            upsampled_feature_affine,
            phase_feature_affine,
        ) = self._affine_states(
            encoded_feature,
            phase_occupancy,
        )

        phase_feature_mean = (
            phase_feature_affine.mean(dim=1, keepdim=True)
        )
        phase_feature_residual = (
            phase_feature_affine
            - phase_feature_mean
        )

        (
            actual_residual_hidden,
            actual_residual_energy,
            actual_common_hidden,
            actual_common_energy,
        ) = self._compatibility_components(
            occupancy_affine.unsqueeze(1),
            phase_feature_affine,
            phase_feature_mean,
        )

        center = self.config.coarse_radius
        center_phase_weight = self.occupancy_weight[
            :, :, center, center
        ].transpose(0, 1)

        flip_delta = (
            1.0
            - 2.0 * phase_occupancy.to(
                dtype=encoded_feature.dtype
            )
        ).unsqueeze(2) * center_phase_weight[
            None, :, :, None, None
        ]

        flipped_occupancy_affine = (
            occupancy_affine.unsqueeze(1)
            + flip_delta
        )

        (
            flipped_residual_hidden,
            flipped_residual_energy,
            flipped_common_hidden,
            flipped_common_energy,
        ) = self._compatibility_components(
            flipped_occupancy_affine,
            phase_feature_affine,
            phase_feature_mean,
        )

        residual_odd = 0.5 * (
            actual_residual_energy
            - flipped_residual_energy
        )
        common_even = 0.5 * (
            actual_common_energy
            + flipped_common_energy
        )
        common_gate = 2.0 * torch.sigmoid(common_even)
        gated_interaction = common_gate * residual_odd

        native_phase_field = (
            self.config.field_amplitude
            + gated_interaction
        )
        field = self.pixel_shuffle(native_phase_field)

        if (
            not bool(torch.isfinite(field).all())
            or not bool(torch.isfinite(common_gate).all())
            or bool(torch.any(common_gate < 0.0))
            or bool(torch.any(common_gate > 2.0))
        ):
            raise FloatingPointError(
                "GCR-PACRE field/gate contract failed"
            )

        fields = build_complete_fields_ledger(
            # Every tensor listed in 15.2 is supplied here.
            ...
        )
        self._validate_gcr_fields(
            fields,
            feature=feature,
            occupancy=occupancy,
        )
        return fields
```

正式实现应复用现有 phase-centering/transport helper，但必须按 15.1--15.3
建立独立 v24 config、完整 fields validator 和独立 oracle。上方片段没有列全
ledger 返回值和 reference path，不能直接提交为实现。

---

# 16. 当前 v24 候选内冻结的内容

为了保持当前候选的单机制归因，已经封存的 GCR-PACRE-v24 以及同候选 fresh r2
必须冻结：

```text
Base checkpoint
feature adapter
feature normalization
occupancy threshold 0.72
phase-preserving occupancy
PAET transport
binary center-phase flip
field amplitude 0.9
field threshold 0
PMOPE objective
Adam config
batch composition
800 × 40 schedule
hard union
connected-component convention
matching protocol
Base@B 51-point grid
false-alarm budgets
relative-performance gate
```

在 v24 候选身份内不允许：

- 新增 scalar gain 参数；
- 搜索 sigmoid temperature；
- 搜索 common/residual 混合系数；
- 新增 readout；
- 修改 PMOPE 权重；
- 使用 Base@B 的 0.14 作为 CURE 输入 threshold；
- 在 D_V 上校准 field threshold；
- morphology 或 component crop。

以上是**候选级冻结**，不是对 CURE-Lite 研究空间的永久禁令。若内部未见证据或
失败归因表明可提高性能，新的候选可以预注册修改 feature normalization、gate
parameterization、readout、objective/损失权重、采样、优化器、训练调度、宽度或
attention。此类候选必须使用新版本和独立 source closure；若同时修改多个组件，
必须增加相应消融，不能继续声称“唯一差异是 GCR field equation”。field threshold
零水平集、禁止从 \(D_V\) 选择超参数、冻结 Base/evaluator 与一次性 \(D_V/D_T\)
治理属于科学主线或评价合同，不因新候选而放宽。

---

# 17. 新 dataset-free 门禁

文件：

```text
cure_lite_v24/dataset_free.py
tools/audit_cure_lite_v24_gcr_pacre_dataset_free.py
```

至少验证：

## 17.1 主线合同

```text
parameter count = 64,064
parameter names/shapes与 v23 相同
input = (feature, occupancy)
one scalar field
additional head = 0
threshold = 0
hard union exact
canonical v24 policy identity exact
complete fields validator called
independent GCR forward_reference called and agrees within frozen envelope
```

## 17.2 Gate 数学

```text
common_even finite
common_gate finite and 0 <= common_gate <= 2
gate endpoint/interior counts reported
common_even(actual, flip) symmetric under the reference equation
residual_odd(actual, flip) antisymmetric under the reference equation
gated_interaction antisymmetric under the reference equation
common_even = 0 -> gate = 1
residual_odd = 0 -> gated_interaction = 0
zero feature -> field = +0.9
```

生产 dtype 的 parity 比较使用由 FP64 oracle 与 generated fixtures 在实现冻结前
生成并封存的 ULP/absolute envelope；不得沿用已知会受求和顺序影响的 naïve
bit-exact flip 断言。代数符号与零状态仍必须由 reference equation 精确验证。

## 17.3 选择性

构造三个 generated states：

1. target-like：
   ```text
   residual_odd < 0
   common_even > 0
   ```
   要求新 field 比 PACRE 更负；

2. background-like：
   ```text
   residual_odd < 0
   common_even < 0
   ```
   要求新 field 比 PACRE 更正；

3. common-only：
   ```text
   residual_odd = 0
   common_even != 0
   ```
   要求不产生 completion。

## 17.4 梯度

由于共享 `scalar_energy_weight` 从零初始化，step 0 有：

\[
D=E=0,\qquad G=1,
\]

且乘积导数中的 \(D\,\mathrm dG\) 项为零。因此不能声称初始化时 gate-path
gradient 已激活。必须分两阶段验证：

```text
stage A / step 0:
  v24 output == v23 output under the generated fixture
  total parameter gradients == v23 residual-path gradients
  all values and gradients finite

stage B / exactly one frozen warm-up update:
  fixture bytes, PMOPE call, optimizer hyperparameters and update count sealed
  residual-only Jacobian/gradient is finite and nonzero
  gate-only Jacobian/gradient is finite and nonzero
  total PMOPE gradient reaches the shared readout with no detach
```

“gate-only”必须通过显式 path isolation（例如固定/分离 \(D\) 后检查
\(D\,\mathrm dG\)）计算，不能用总参数 gradient 冒充。

## 17.5 计算与部署效率审计

在任何真实训练授权前，对 v23 与 v24 使用相同设备、dtype、batch、输入 shape、
warm-up 次数和测量重复次数，封存：

```text
parameter tensors / parameter count / checkpoint bytes
forward FLOPs or MACs（同一计数器）
forward latency median / p95
train-step latency median / p95
peak allocated and reserved device memory
field/output tensor bytes
additional op inventory
OOM / nonfinite = false
```

本版不事后发明一个“Lite”开销阈值；审计完成、条件一致、无 OOM/非有限值是
进入 \(D_R\) 的工程前提，所有开销变化必须如实报告。若未来需要声称某一部署
预算内的 Lite 性，预算必须在看到测量结果前另行冻结。

### 17.6 已封存的最小合法 generated profile

r2 receipt 在 RTX 3090 / FP32 上对两臂使用同一个
`[12,64,1,1] -> [12,1,4,4]` generated PMOPE batch。该 shape 的作用是
机械验证 formal topology、真实 forward/backward、同步计时和显存记录；它被
明确标记为 `minimum_legal_generated_shape_not_deployment_workload`，不能外推
真实部署吞吐。

| 观测 | v23 PACRE-VC | v24 GCR-PACRE | v24 相对变化 |
|---|---:|---:|---:|
| profiler forward FLOPs | 1,634,498 | 1,687,424 | +3.238% |
| forward median | 4.858041 ms | 7.076233 ms | +45.660% |
| train-step median | 31.440274 ms | 38.585184 ms | +22.725% |
| peak allocated | 1,522,688 B | 1,522,688 B | 0 |
| peak reserved | 2,097,152 B | 2,097,152 B | 0 |
| parameter bytes | 256,256 B | 256,256 B | 0 |
| checkpoint bytes | 258,401 B | 258,401 B | 0 |

时延仅有预注册的少量重复，且最小空间尺寸会放大 validator/launch 固定开销；
因此只报告原始 median/p95 与样本，不据此声称部署速度改善或回退。正式晋级
没有开销比例门槛，只要求同条件证据完整、无 OOM、无非有限值；科学性能仍由
后续 source-disjoint OOF 的相对门禁决定。

---

# 18. 新真实 \(D_R\) 结构门禁

文件：

```text
cure_lite_v24/dr_gate.py
```

继续保留 PACRE-VC 的 algebra、witness、collision、gradient 和 preservation
检查，并增加：

```text
14_common_compatibility_finite_and_non_degenerate
15_common_gate_even_and_residual_interaction_odd
16_common_evidence_cannot_create_completion_without_residual
17_target_groups_have_bound_residual_direction_and_gate_gradient_witness
18_no_exact_target_background_gated_latent_collision
19_model_population_cache_rng_and_grad_buffers_preserved
20_read_only_zero_update_D_R_scope
21_gate_saturation_distribution_recorded_without_post_hoc_threshold
22_fast_reference_and_complete_fields_ledger_agree
23_efficiency_receipt_bound
```

该 gate 的冻结授权与静态执行合同仍然是：

```text
optimizer steps authorized = 0
parameter updates authorized = 0
training authorized = false
D_V payload authorized = false
D_T payload authorized = false
```

这些字段约束合法执行范围；若正式 receipt 或系统级事件账本缺失，不能仅凭它们
把失联时段的实际运行事实写成已被独立审计。该 gate 只回答 v24 是否具有可训练的
candidate selective gate，不是性能门禁，也不能把 H1/H2 提升为已证实根因。

## 18.1 2026-07-29 r1 正式尝试：执行可观测性丢失，无门禁判定

正式命令于 `2026-07-29 05:52:06 +08:00` 启动；外部 authorization、access
audit 与固定设备预检通过后，于 `05:52:09.929794039 +08:00` 创建唯一的
只读 run-start marker。最后一个可验证健康观测为
`06:35:36.448 +08:00`：PID 2077593 为 `Rsl`，已运行 43 分 29 秒，累计 CPU
时间 1:40:20，GPU0 使用 1193 MiB、17%、57°C，正式 receipt 仍为 pending。

紧随其后的 30 秒轮询只有调用记录、没有返回记录；下一条托管会话事件直到
`16:03:48.860 +08:00` 才出现。恢复核查得到：tool session 35488 不存在、原
PID 不存在、没有重父化或同命令进程、固定
`D_R_structural_receipt.json` 在恢复时不存在，而 run-start marker 仍为单硬
链接、0444，文件哈希和内部 fingerprint 均未变化。

随后执行的有界搜索只覆盖
`protocols/IRSTD-1K/gcr_pacre_v24` 与
`runs/irstd1k_stage_a_seed42`、最大深度 4、时间条件为晚于
`2026-07-29 05:51:00 +08:00`。它只返回 access audit、authorization 和 marker，
在该固定范围内没有返回正式、临时或 partial receipt。该结果不能证明范围外或
历史瞬时文件的全局不存在；没有 filesystem event ledger 时，历史上是否曾短暂
创建后丢失不可独立审计。

取证后再次运行 metadata-only `preaccess-verify` 通过，source closure
fingerprint 仍为
`28d26759a68785e9c99917fcfa8b36430c7f6e5463282d66eeab5c711e425e9f`。这次
只读验签调用本身明确报告 \(D_R/D_V/D_T\) tensor payload 均未访问、未训练；
该报告只覆盖这次 verifier invocation，不覆盖此前已经失去可观测性的 r1。

因此权威分类是：

```text
attempt status          = EXECUTION_OBSERVABILITY_LOST_NO_DECISION
termination event       = NOT OBSERVED
exact process exit time = UNKNOWN_NOT_OBSERVABLE
exact OS cause          = UNKNOWN_NOT_OBSERVABLE
Python traceback        = NOT OBSERVED
OOM evidence            = NOT OBSERVED
D_R gate decision       = NO_AUTHENTICATED_DECISION_AVAILABLE
scientific PASS/FAIL    = NOT ESTABLISHED
```

不能把“没有 OOM 证据”写成“证明不是 OOM”，也不能把 tool session/PID 在恢复时
消失写成已经识别的终止原因。以下三层必须分开：

| 范围 | 授权或静态合同 | 可得协议/会话记录中的观测 | 失联时段实际事实的独立审计 |
|---|---|---|---|
| \(D_R\) | payload authorized；零更新 probe | 执行已启动，最后健康时 receipt pending | materialization/probe 是否完成未知 |
| \(D_V\) | payload authorization=false | 未观察到访问 | 无 system open-event ledger，未独立审计 |
| \(D_T\) | payload authorization=false | 未观察到访问 | 无 system open-event ledger，未独立审计 |
| training | authorization=false；optimizer/update=0 | 未观察到训练 | 无 system execution-event ledger，未独立审计 |

实现合同在 `O_EXCL` 成功后明确永不回滚 marker，正式 receipt 又只在完整 probe
返回并自验后才写；所以 r1 不允许 resume、自动 retry、删除 marker 后重跑，且
不得创建 OOF authorization。当前强制处置为：

```text
OOF / bounded-400 / Formal800       = NOT AUTHORIZED
D_V / D_T authorization             = false
D_V / D_T access in available logs  = NOT OBSERVED
D_V / D_T actual access after loss  = NOT INDEPENDENTLY AUDITED
r1 next action                       = STOP_AND_PRESERVE_R1_EVIDENCE
```

权威解释以只读、自指纹的
`D_R_structural_attempt_r1_interruption_receipt_v2.json` 为准；它绑定原 v1、
两条有界搜索 JSONL 记录及其逐行 SHA-256。r1 仍然是已消耗的第 1 次执行身份，
但不是 decision-bearing result。

## 18.2 仅在用户另行显式授权后才可建立 fresh r2

当前状态是：

```text
r2 runtime supervisor scaffold = IMPLEMENTED, DUMMY/STATIC VERIFICATION ONLY
r2 service unit                    = TEMPLATE ONLY; NOT INSTALLED/ENABLED/STARTED
r2 actual runtime spec             = NOT CREATED
r2 execution authorization        = NOT CREATED
r2 attempt/materialization marker  = NOT CREATED
r2 D_R access/run                  = NOT AUTHORIZED
```

上述“已实现”只表示无数据的持久监督基础设施已经准备好：
`tools/cure_lite_v24_runtime_supervisor.py` 与
`deploy/systemd/cure-lite-v24-gcr-pacre-dr-r2.service.template` 不属于 r1 冻结的
103 个执行文件，且不被 legacy r1 入口导入。它们不得据此推导出一次可执行的
r2；正式 runtime spec、其自指纹、unit 实例哈希、独立 source closure 和用户签发
authorization 缺一不可。

如果未来确需一次新的可判定执行，必须把它记为 `attempt_ordinal=2` 的 fresh
execution identity，而不是 r1 resume、retry、continuation 或统计独立重复。
新 authorization 与 marker 必须前瞻冻结并公开记录：

```text
prior_D_R_attempts                         = 1
prior_attempt_status                       = OBSERVABILITY_LOST_NO_DECISION
new_attempt_ordinal                        = 2
cumulative_D_R_attempts_after_r2_marker    = 2
```

在任何 r2 payload materialization 之前，至少同时满足：

- r1 的 interruption v1/v2、authorization、access audit、marker 及 marker 绑定的
  103 个源码文件逐字节复核通过；旧 marker 和所有旧路径不删除、不复用、不覆盖。
- 新建独立 schema、run/stage identity、access audit、authorization、marker、
  receipt 与 runtime root；r2 receipt 不得写入 r1 固定空缺路径。
- 用机器可验 diff allowlist 证明只改变 execution identity、r1 lineage、独立产物
  路径和不改变数值语义的持久监督层。
- candidate、模型参数初值、probe 数学、23 项 check 名称与实现、manifest/index/
  population/cache/source fingerprint、seed、设备、dtype、阈值、evaluator、Base
  ledger、安全预算和执行顺序与 r1 完全相同。
- r2 authorization 显式绑定本节 v2 纠错回执、原科学 preregistration、全新 source
  closure、runtime spec、supervisor 与 unit 文件哈希；\(D_V/D_T\) authorization
  继续为 false，training/optimizer/update 继续为 0。
- 由外部 `commit-and-start` 先以 `O_EXCL` 写入不可回滚的 attempt commit，再只
  发出一次 `systemctl --user start --no-block`；即使 unit 尚未进入
  `ExecStart`，这次 attempt identity 也已经消耗，不能靠 `reset-failed`、unit
  垃圾回收或人工再次 start 绕过。
- 以静态、detached user-systemd `Type=exec` 服务和独立 supervisor 运行；固定
  `Restart=no`、`NRestarts=0`、`KillMode=mixed`、`SendSIGKILL=yes`，
  `DropInPaths=[]`、`Transient=no`、`NeedDaemonReload=no`，不使用 PTY、timer、
  watchdog restart、`ExecReload` 或 shell child。实际 FragmentPath、unit 哈希和
  完整关键属性必须与 runtime spec 精确一致。
- `ExecCondition` 在 `ExecStart` 前以第二个 `O_EXCL` materialization claim 绑定
  systemd `InvocationID`、boot identity 与 supervisor 进程身份；`run-once` 只能
  验证该 claim，不能代写。worker stdout、stderr、15 秒编号 hash-chain
  heartbeat、child exit 与每个 systemd invocation 的 exit sidecar 均使用互不
  覆盖的 create-once 路径；第二次 invocation 本身就是协议失败证据。
- systemd 的 exit 0 只表示执行链完整；科学 PASS/FAIL 只能来自独立验签后的正式
  receipt。无合法 receipt 的任何 signal、非零退出或证据链缺口一律归为新的
  interruption/no-decision，且禁止自动开启第 3 次尝试。
- 运行后必须生成 post-run access/runtime audit；没有系统级事件证据时，继续使用
  “未观察到/未独立审计”的分层措辞，不能补写绝对未访问主张。
- 只有 r2 receipt 验签且 structural PASS 后，新版 OOF verifier/authorizer 才能
  显式绑定 r2 receipt 与 r1 interruption lineage；现有硬绑定 r1 的 OOF 路径
  不得复用。

r2 execution amendment 无权改变科学门禁：性能规则继续相对冻结有效 Base 动态
计算，`minimum_fixed_uplift_margin=null`，不设武断的固定绝对提升幅度；若最终
进入 Formal，仍严格执行 `800×40=32,000` updates。r2 无论 PASS、FAIL 还是再次
失联均必须接受，不得择优、改名或自动重试。

正式启动前还必须重验 user-systemd manager、目标 GPU 和磁盘状态。当前只读审计
发现 user manager 为 degraded，且存在两个无关的 `Restart=on-failure` 循环服务；
它们不得占用 r2 授权设备。停止或禁用这些无关服务属于独立状态变更，必须另行
获得相应权限，不能由本方案默示授权。

这是一项 launch hard reject，不是提示性告警。r1 冻结设备为 `cuda:0`；本次
2026-07-30 只读复核中，`confa-v41-mshnet-nudt-clean-formal-20260718-v1.service`
仍为 enabled、`Restart=on-failure` 且处于 `activating/auto-restart`，其 `ExecStart`
也明确请求 `--device cuda:0`。即使同一时刻 `nvidia-smi` 显示 GPU 0 无计算进程，
也不能证明后续 materialization 期间排他。未通过独立授权的停止/禁用与随后验签的
GPU exclusivity preflight 前，不得签发 actual r2 authorization，也不得调用
`commit-and-start`。

---

# 19. 恢复 source-group 未见性能门禁

下一版本不应再次从 D_R structural PASS 直接进入 Formal800。

推荐建立：

> **D_R source-group OOF-4 gate**

## 19.1 分组

split unit 不是派生 state row，而是冻结 D_R manifest 中的
`root_source_id`。在建 fold 前先生成 exact lineage closure：

```text
root_source_id
all derived sample/state IDs
factual / clean / null / target / context roles
occupancy-pair IDs
feature-source IDs
label-source IDs
```

当前 D_R population receipt 中的角色总量应逐字节核对为 32 个 target-role
states 与 96 个 context-role states；这两个 state 计数不能代替
`root_source_id` 数量。split receipt 必须另外列出 exact unique roots、每 fold
root IDs/role counts 和四 fold union/intersection 证明。

以 `root_source_id` 分 fold，再把全部派生角色按 closure 展开，保证同一 root 的
factual、clean、null、target、context 和 occupancy pair 全在同一 fold。任何
派生 row 对应多个 root、root 缺失或 closure 跨 fold，均 fail closed。

fold assignment 在训练前通过固定 hash 冻结，例如：

```python
fold = int(
    sha256(
        f"gcr-pacre-v24-oof4:{root_source_id}".encode()
    ).hexdigest(),
    16,
) % 4
```

如果该映射导致 fold 空或严重失衡，必须在读取任何训练结果前冻结一个
deterministic balanced assignment receipt，不能看结果后调整。

每 fold 的 candidate/control 固定使用：

```text
seed                     = 42
epochs                   = 10
steps_per_epoch          = 40
updates_per_model        = 400
optimizer / PMOPE        = 与冻结 Formal policy 相同
checkpoint selection     = none
terminal artifact        = final-only
```

### 19.1.1 物理 cache 隔离

split receipt 签署后，按以下独立目录从各自允许的 root 重建：

```text
oof4/fold_0/{train,holdout}/{candidate,v23_control,base_eval}/
...
oof4/fold_3/{train,holdout}/{candidate,v23_control,base_eval}/
```

硬约束：

- train runner 的 allowlist 只含该 fold 的 train roots，不能打开 holdout 路径；
- holdout cache 仅在训练 terminal 封存后由只读 evaluator 打开；
- 不允许跨 fold/candidate/control 的 symlink、hardlink、reflink、共享 tensor、
  mmap 或进程内 cache 复用；
- receipt 保存 `realpath`、device/inode、文件 digest、root IDs、创建阶段和
  reader allowlist；
- feature/occupancy/label cache 都遵循相同隔离；只隔离 sampler 而共享已展开
  tensor 不算 source-disjoint；
- 每个 held-out root 恰好出现一次，四 fold 合并后 exact 覆盖冻结 root
  universe。

## 19.2 对照

每个 fold 的完整比较集为：

```text
Base@A: fixed occupancy threshold 0.72, no training
Base@B-xfit: threshold 只在该 fold 的 train roots 上从冻结 51 点网格选择，
             再原样应用到 holdout；禁止使用 D_V 的 0.14
v23 PACRE-VC: separately trained paired control
v24 GCR-PACRE: trained candidate
v24 GCR-PACRE-G1: 同一 v24 terminal 权重的只读消融，强制 G=1，不重新训练
```

共享：

```text
train sources
holdout sources
initial parameter bytes
seed
Adam
schedule
updates
PMOPE
batch order
finite audit
evaluator
```

对 v23 control 与当前封存 v24 候选的主因果比较，唯一差异：

```text
field equation
```

`Base@B-xfit` 的 train-only threshold tie-break、validity、安全预算和 grid digest
必须在 split receipt 中冻结；holdout label 不得参与 threshold 选择。
`G1` 只用于识别 gain path 的贡献，不能替代 separately trained v23 control。
该“唯一差异”只约束当前 v24 的因果消融。新候选可以同时修改非主线组件，但
必须预注册候选集并补足逐组件或最小充分消融，不能沿用 v24 的单机制归因措辞。

每 fold 还必须保存 held-out target/background 的：

```text
E and G min/max/mean/q01/q10/q50/q90/q99
G == 0 / G == 2 / 0 < G < 2 counts
D sign and magnitude by role
G-D sign contingency
```

## 19.3 OOF 聚合门禁

四个 held-out prediction ledgers 先按冻结 root order 拼接，每个 root 恰好一次，
再调用与 Formal 完全相同的 target matching、connected-component、mIoU、nIoU、
retention 和 false-alarm evaluator。**正式 OOF 指标是 pooled prediction 的一次
重算，不是四个 fold 指标的算术平均。**

对 pooled Base rows 动态定义：

\[
\begin{aligned}
T_{\rm base}^{\max}&=\max_{\text{valid Base rows}}T,\\
R_{\rm base}^{\max}&=\max_{\text{valid Base rows}}R,\\
I_{\rm base}^{\max}&=\max_{\text{valid Base rows}}I,\\
N_{\rm base}^{\max}&=\max_{\text{valid Base rows}}N.
\end{aligned}
\]

候选的主性能向量必须逐项成立：

```text
candidate.true_targets     > T_base_max
candidate.recovered_misses > R_base_max
candidate.mIoU            >= I_base_max
candidate.nIoU            >= N_base_max

candidate.true_targets     > v23_control.true_targets
candidate.recovered_misses > v23_control.recovered_misses
candidate.mIoU            >= v23_control.mIoU
candidate.nIoU            >= v23_control.nIoU

candidate.true_targets     >= G1.true_targets
candidate.recovered_misses >= G1.recovered_misses
candidate.mIoU            >= G1.mIoU
candidate.nIoU            >= G1.nIoU
at least one of {true_targets, recovered_misses, mIoU, nIoU}
    is strictly greater than its G1 value

candidate.retention        = 1
candidate false-alarm budgets PASS
all Base rows used in envelope are valid
```

所有比较直接使用 receipt 中的原始整数/全精度浮点，不舍入、不设置任意固定
uplift。任一严格整数比较自然意味着至少 \(+1\)，不是人工幅度门槛。

\(G\equiv1\) 是机制消融，不是另一个必须被两项离散计数同时击败的 Base。
因此对 G1 采用预注册的 Pareto-style 主性能向量：四项均不回退、至少一项严格
改善。同时必须满足：

```text
candidate gated-field ledger digest != G1 field ledger digest
candidate pooled-prediction ledger digest != G1 prediction ledger digest
candidate and G1 root/order/evaluator bindings exact equal
```

这证明 gate path 在 field 与最终 prediction 上均非恒等；严格改善可以来自任一
主指标，不增加“必须多恢复固定几个目标”的任意离散门槛。Base 包络和 v23
control 的主门禁保持不变。

每 fold 指标、role 分布和 saturation 仍逐项报告，用于定位异质性；它们不再
附加没有先验依据的“至少 3/4 fold 不回退”规则。唯一逐 fold 安全要求是：

```text
retention = 1
false-alarm budgets PASS
finite / source closure / cache isolation PASS
```

这些是新的内部未见证据，不使用 D_V。

## 19.4 为什么使用 OOF 而不是一个小 holdout

D_R factual misses 数量有限。单一 20% holdout 的 target count 可能过小，一
个目标就会改变结论。

OOF-4：

- 所有 source 都被未见评估一次；
- 训练与评估仍 source-disjoint；
- 能聚合出较稳定的 target-level 结果；
- 同时检验 Base 包络、v23 control 和 \(G\equiv1\) 机制消融；
- 不读取或消耗 \(D_V/D_T\)。

---

# 20. bounded-400 必须恢复为 Formal 前置条件

v23 明确绕过 bounded-400。

v24 应改为：

```text
D_R structural PASS
+
D_R OOF-4 PASS
+
bounded-400 PASS
    ->
Formal800 authorization
```

bounded-400 需要：

```text
fresh seed 42
empty Adam
10 epochs × 40 steps = 400 updates
no resume
no retry
one terminal
post-step finite audits
D_V / D_T unread
```

candidate 与 v23 control 必须 paired 运行，不能只看 candidate 绝对 loss：

```text
initial parameter names / shapes / bytes exact equal
independent empty optimizer states
same frozen full-D_R population manifest
same ordered pair/role batches at every update
same PMOPE / Adam / dtype / device policy
different module instances and non-shared storage
final-only; no checkpoint selection
```

`target-role violation`、`background-role violation`、`zero-crossed target
states` 和 `false-completion states` 必须复用冻结 PMOPE/bounded evaluator 的
既有 role、integration measure、zero-level 和聚合定义，并在 authorization
中绑定 evaluator fingerprint。

### 20.1 绝对执行与安全 PASS 向量

runner 在启动前冻结以下授权向量；结果必须逐项全真：

```text
candidate_updates == control_updates == 400
candidate_terminal_count == control_terminal_count == 1
all losses / gradients / parameters finite at every post-step audit
source closure / cache / RNG / optimizer independence PASS
all frozen target/background roles exposed with exact scheduled counts
candidate retention / fixed safety budgets PASS
gate endpoint/interior counts and E/G role distributions present
D_V_accessed == D_T_accessed == false
```

### 20.2 非 paired 的优化 sanity

candidate 和 control 各自还必须满足：

```text
terminal parameter fingerprint != its own initialization
terminal PMOPE <= its own initial PMOPE
```

这只证明 optimizer 确实更新且各自没有沿训练目标整体恶化，不比较两种模型的
科学优劣。

### 20.3 强制 paired diagnostics，不作为独立授权阈值

下列同 batch、同 evaluator 差值必须完整保存，但不单独决定 PASS/FAIL：

```text
candidate - control terminal PMOPE
candidate - control target-role violation
candidate - control background-role violation
candidate - control zero-crossed target states
candidate - control false-completion states
candidate - same-weight G1 for the same five quantities
candidate/G1 field and role-prediction nonidentity witnesses
per-update paired loss and gradient-norm deltas
```

不能把尚未由 unseen evidence 证明合理的
`zero-crossed_target_states(candidate) > control/G1` 写成 bounded 唯一授权
条件，也不能在看到这些 diagnostics 后追加阈值。相对性能晋级已经由前一阶段
source-disjoint OOF 的 Base/v23/G1 合同决定；bounded-400 只要求配对执行真实、
优化有进展且绝对安全合同通过。

bounded-400 是 full-\(D_R\) 优化/执行 smoke evidence，不是 unseen
generalization evidence；晋级仍以前一阶段 source-disjoint OOF 为必要条件。

---

# 21. Formal800 计划

只有内部 gate 全部通过后，运行：

```text
v24 seed 42 primary：800 epochs × 40 steps = 32,000 updates，从零训练
v24 seed 43 integrity：800 epochs × 40 steps = 32,000 updates，从零训练
```

两 seed 都使用：

- 同一 full D_R；
- 同一冻结 schedule policy；
- 同一 PMOPE；
- 无 resume；
- 无 checkpoint selection；
- final-only artifact。

## 21.1 Seed 42

作为唯一冻结 primary model。只有其 final-only terminal artifact 可以绑定
\(D_V\) 和未来 \(D_T\)。

## 21.2 Seed 43

作为 training-integrity run，只证明第二个从零训练能完成同一 32,000-update
合同、source closure、finite audit、artifact sealing 和 D_R-only terminal
evaluation。它不进入 \(D_V\)，不与 seed42 做 checkpoint/model selection，
也不作为集成成员。

在读取 D_V 前应冻结：

```text
primary = seed 42
seed 43 = training-integrity-only
seed 43 D_V evaluation = forbidden
seed 43 D_T evaluation = forbidden
```

本版不允许 ensemble；它会改变已经冻结的推理合同。

两个 Formal runner 都必须在启动前将 exact schedule 写入 authorization：

```text
epochs = 800
steps_per_epoch = 40
expected_updates = 32_000
resume = false
retry = false
checkpoint_selection = none
terminal_artifacts = final_only
D_V_accessed = false
D_T_accessed = false
```

---

# 22. v24 的 D_V 门禁

v24 \(D_V\) 仍是 adaptive evidence，且只允许 seed42 primary 一次运行。

比较协议、Base validity、安全预算和 evaluator 保持 v23 完全不变。正式
verifier 先校验冻结 Base ledger 的 fingerprint 和 exact schema，然后直接从
所有 `valid=true` Base rows 读取原始全精度值：

\[
\begin{aligned}
T_{\rm best}&=\max T_{\rm valid\ Base},&
R_{\rm best}&=\max R_{\rm valid\ Base},\\
I_{\rm best}&=\max I_{\rm valid\ Base},&
N_{\rm best}&=\max N_{\rm valid\ Base}.
\end{aligned}
\]

唯一正式性能判断为：

```text
candidate.true_targets      > T_best
candidate.recovered_misses  > R_best
candidate.mIoU             >= I_best
candidate.nIoU             >= N_best
retention             = 1.0
pixel FA              <= 1e-4
raw-background FA     <= 1e-4
FP components / MP    <= 100
budget_violation      = false
D_T accessed          = false
```

禁止在 verifier 中复制六位小数字面量、先 round 再 compare、用当前 CURE
差值作为 uplift，或把一个 Base 行当成全部坐标的参照。以当前 ledger 仅作
人类可读推导，整数条件自然等价于：

```text
true targets >= 151
recovered misses >= 4
```

连续指标的当前参照快照为
`0.6095592799503414` 和 `0.5653280526756584`，但实现必须从冻结 ledger
动态读取，不能硬编码。这是相对 best valid Base 的严格改善/不回退，不是人为
\(+2\) 或固定绝对 uplift。

## 22.1 D_V 只允许一次正式候选运行

允许：

- 继承 v23 sealed aggregate result/decision/fingerprint，不重开 payload；
- 一次 v24 fixed candidate evaluation。

不允许：

- 根据 v24 D_V 调 gate；
- 修改 sigmoid；
- 换 seed；
- 选择 checkpoint；
- 调 field threshold；
- 重新定义 Base@B；
- 自动重跑。

---

# 23. \(D_T\) 最终确认

## 23.1 必须在 \(D_V\) 前冻结的 \(D_T\) preregistration

在创建 \(D_V\) authorization 之前，必须先封存
`D_T_preregistration.json`，且当时：

```text
D_T payload accessed = false
model binding = seed42 final-only fingerprint
candidate attempts = 1
automatic retry = false
Base@A threshold = 0.72
Base@B threshold = 0.14，绑定已封存 D_V selection ledger
D_T Base threshold search = false
Base envelope = 在 D_T 仅评估上述两个固定 operating points，
                再从其有效全精度指标逐坐标动态计算
integer metrics = strictly greater than every valid Base coordinate maximum
mIoU/nIoU = no regression against valid Base coordinate maxima
retention / FA / component budgets = frozen
evaluator / schema / tie rules / unknown-field policy = frozen
PASS/FAIL/STOP decision table = frozen
```

这一步只冻结规则，不创建运行 authorization，也不读取任何 \(D_T\) tensor。
`0.14` 是已经封存的 validation-selected Base@B operating point，不允许在
\(D_T\) 的 51 点网格上重新选择；否则 comparator 本身会获得 test-label oracle
优势。不得根据 v24 \(D_V\) 结果改变 \(D_T\) 门禁或 Base operating points。

## 23.2 \(D_T\) authorization

只有以下全部满足，才创建一次性 \(D_T\) authorization：

```text
v24 dataset-free PASS
v24 D_R structural PASS
v24 D_R OOF-4 PASS
v24 bounded-400 PASS
v24 Formal800 seed42 PASS
v24 Formal800 seed43 training-integrity complete
seed43 D_V/D_T accessed = false
D_T preregistration predates D_V authorization
v24 D_V adaptive PASS
all source/verifier/artifact checks PASS
```

\(D_T\) 只能对 seed42 运行一次，不能用于模型选择、调参或自动重试。

---

# 24. Verifier schema 修复

新 verifier 不再维护手写字段白名单。

建议：

```python
from dataclasses import fields


FORMAL_RESULT_FIELDS = frozenset(
    field.name
    for field in fields(GCRPACREFormalTrainingResult)
    if not field.name.startswith("_")
)
```

对于 canonical JSON，则由唯一 schema 常量声明 exact keys：

```python
_REQUIRED_KEYS = frozenset({
    "schema_version",
    "method",
    "run_id",
    "formal_result_fingerprint",
    "training_result_fingerprint",
    ...
})
```

验证顺序：

1. JSON 必须是 object；
2. `set(payload) == _REQUIRED_KEYS`；
3. required digest 全部合法；
4. 重新计算 fingerprint；
5. 重新计算 decision；
6. 比较 sealed artifact；
7. 未知字段失败；
8. 合法字段不能因 whitelist 漏项失败。

必须新增：

```text
test_formal_result_fingerprint_is_required_and_accepted
test_missing_formal_result_fingerprint_is_rejected
test_unknown_formal_field_is_rejected
test_append_only_mechanical_correction_does_not_change_scientific_decision
```

---

# 25. 文件级修改清单

## 25.1 永久冻结

```text
cure_lite_v23/**
v23 D_R receipt
v23 Formal800 result
v23 D_V result / decision / COMPLETE
v23 append-only mechanical correction
v23 source closure
```

不得覆盖。

## 25.2 新 package

```text
cure_lite_v24/
    __init__.py
    gcr_pacre.py
    factory.py
    inheritance.py
    dataset_free.py
    efficiency.py
    dr_gate.py
    oof_split.py
    oof_runner.py
    bounded_runner.py
    training.py
    formal_training.py
    formal_artifacts.py
    formal_evaluation.py
    verifier.py
    decision.py
    protocol.py
```

## 25.3 新 tools

```text
tools/audit_cure_lite_v24_gcr_pacre_dataset_free.py
tools/audit_cure_lite_v24_efficiency.py
tools/run_cure_lite_v24_gcr_pacre_dr_gate.py
tools/run_cure_lite_v24_gcr_pacre_oof4.py
tools/run_cure_lite_v24_gcr_pacre_bounded_400.py
tools/run_cure_lite_v24_gcr_pacre_formal_800.py
tools/run_cure_lite_v24_gcr_pacre_d_v.py
tools/verify_cure_lite_v24_artifact.py
```

## 25.4 新 tests

```text
tests/test_cure_lite_v24_gcr_pacre_core.py
tests/test_cure_lite_v24_gcr_pacre_flip_parity.py
tests/test_cure_lite_v24_gcr_pacre_gate_bounds.py
tests/test_cure_lite_v24_gcr_pacre_no_common_only_completion.py
tests/test_cure_lite_v24_gcr_pacre_gradients.py
tests/test_cure_lite_v24_gcr_pacre_forward_reference.py
tests/test_cure_lite_v24_gcr_pacre_fields.py
tests/test_cure_lite_v24_gcr_pacre_policy_identity.py
tests/test_cure_lite_v24_inheritance_no_d_v_reopen.py
tests/test_cure_lite_v24_dataset_free.py
tests/test_cure_lite_v24_efficiency.py
tests/test_cure_lite_v24_dr_gate.py
tests/test_cure_lite_v24_oof_split.py
tests/test_cure_lite_v24_oof_physical_cache_isolation.py
tests/test_cure_lite_v24_oof_pooled_metrics.py
tests/test_cure_lite_v24_oof_base_envelope_and_g1.py
tests/test_cure_lite_v24_oof_runner.py
tests/test_cure_lite_v24_bounded_runner.py
tests/test_cure_lite_v24_formal_artifacts.py
tests/test_cure_lite_v24_formal_evaluation.py
tests/test_cure_lite_v24_verifier_schema.py
tests/test_run_cure_lite_v24_cli.py
```

---

# 26. 新 protocol root

```text
protocols/IRSTD-1K/gcr_pacre_v24/
```

至少包含：

```text
v23_failure_inheritance_receipt.json
model_design_preregistration.md
dataset_free_receipt.json
efficiency_receipt.json
D_R_structural_authorization.json
D_R_structural_attempt_r1_interruption_receipt.json（历史 v1；原字节保留）
D_R_structural_attempt_r1_interruption_receipt_v2.json（r1 权威解释；不替代正式 receipt）
D_R_structural_receipt.json（r1 固定空缺路径；当前不存在且不得供 r2 复用）
D_R_structural_attempt_r2_*.json（仅在独立用户授权后创建）
D_R_OOF4_split_receipt.json
D_R_OOF4_cache_closure_receipt.json
D_R_OOF4_authorization.json
D_R_OOF4_result.json
bounded_400_authorization.json
bounded_400_result.json
implementation_closure.json
runner_verification_receipt.json
formal_seed42_authorization.json
formal_seed43_authorization.json
D_T_preregistration.json
D_V_pre_run_authorization.json
D_V_result.json
D_V_decision.json
```

每项 authorization 必须在对应 result directory 创建前冻结。
`D_T_preregistration.json` 必须早于 `D_V_pre_run_authorization.json`，而真正的
`D_T_authorization.json` 只能在 \(D_V\) PASS 后追加。

---

# 27. 测试与验收清单

创建 v24 \(D_R\) authorization 前：

```text
[ ] v23 result/decision/COMPLETE 未修改
[ ] v23 mechanical correction 独立保存
[ ] GCR-PACRE parameter count = 64,064
[ ] parameter names/shapes 与 v23 相同
[ ] canonical v24 policy identity 唯一，legacy PACRE equation identity 被拒绝
[ ] complete fields shape/dtype/device/finite/reconstruction validator PASS
[ ] independent GCR forward_reference PASS on G != 1 fixture
[ ] machine gate finite and bounded in [0,2]
[ ] gate endpoint/interior saturation audit present
[ ] gate flip-even
[ ] residual interaction flip-odd
[ ] gated interaction flip-odd
[ ] common-only evidence cannot create completion
[ ] zero feature field = +0.9
[ ] PMOPE bytes/function unchanged
[ ] step-0 v23-equivalent gradient contract PASS
[ ] one-warm-up residual-path and gate-path isolated gradients finite/nonzero
[ ] dataset-free target boost test PASS
[ ] dataset-free background suppression test PASS
[ ] v23/v24 FLOPs-latency-memory-artifact efficiency receipt complete
[ ] source closure unresolved imports = 0
[ ] verifier accepts formal_result_fingerprint
[ ] D_V/D_T unread by training gates
```

创建 Formal800 authorization 前：

```text
[ ] D_R structural PASS
[ ] D_R OOF-4 root closure and physical cache isolation PASS
[ ] D_R OOF-4 pooled candidate > valid Base envelope and v23 control
[ ] D_R OOF-4 candidate vs G1：主性能向量全项不回退、至少一项严格改善
[ ] D_R OOF-4 candidate/G1 field 与 pooled prediction 均非恒等
[ ] D_R OOF-4 mIoU/nIoU non-regression against all required comparators
[ ] bounded-400 absolute execution/safety PASS vector all true
[ ] bounded-400 candidate/control within-run optimization sanity PASS
[ ] bounded-400 paired diagnostic ledger complete（不作事后阈值）
[ ] fresh seed model fingerprint frozen
[ ] final-only artifact policy frozen
[ ] no checkpoint selection
[ ] Formal schedule exact 800 × 40 = 32,000 updates
[ ] seed42 only is bound to D_V
[ ] seed43 is training-integrity-only and D_V/D_T forbidden
[ ] D_T preregistration frozen before D_V authorization
```

---

# 28. 失败后的决策树

## 28.1 \(D_R\) OOF 显示 common evidence 不能区分 target/background

结论：

> H1/H2 的 held-out 选择性证据不成立，GCR 不能授权 Formal。

当前 v24 不得在看到该 OOF 结果后回调 sigmoid temperature。新候选可以优先考虑
frozen Base confidence 的 training-only hard-negative signal，也可以预注册有限的
gate/temperature/representation 搜索，但选择必须只使用 train-side 或 nested/new
internal-unseen 证据。

## 28.2 \(D_R\) OOF 的 target roles 没有 negative residual direction

若：

\[
D_p\ge0
\]

在 held-out target-role 全部像素成立，则当前 GCR 路径无法通过 gate 独立改变
方向。该结论只约束已冻结的 \(D_R\) mechanism gate，不把 v23 权重的
counterfactual 诊断误写成整个架构的数学不可能性。

此时：

```text
GCR-PACRE NOT AUTHORIZED
```

下一版本应优先检验 interaction representation，或其他保持科学主线且有明确失败
归因的修改；这是一条优先可证伪路线，不是排除所有非主线优化的永久命令。

## 28.3 OOF-4 target count 提升但 IoU 回退

结论：

> common gate改善幅度，但 completion geometry仍不紧致。

冻结 v24，不进入 Formal。

下一候选的首选可证伪分支：

```text
保持 GCR-PACRE model
只修改 objective
```

可优先研究 per-target / near-boundary role-complete orthant objective。这不是唯一
合法路线；任何保持主线、具有明确失败归因并能通过新内部未见证据的非主线修改，
都可以作为独立版本进入预注册候选池。

## 28.4 OOF-4 IoU改善但 target count不提升

结论：

> gate主要抑制 background，未恢复弱目标。

冻结 v24。下一模型需要允许 common evidence影响方向，但必须有新的防造目标
约束。

## 28.5 v24 D_V FAIL

不访问 \(D_T\)。

不得：

- 重跑；
- 调 gate；
- 换 seed；
- 选 checkpoint；
- 调 threshold。

## 28.6 v24 D_V PASS

仍不能宣称独立成功，因为 D_V 已经是 adaptive development data。

只有一次性 \(D_T\) PASS 后，才可建立最终外部成功结论。

---

# 29. 为什么当前 v24 不做事后快速修改

本节限制的是已经看过 v23 聚合结果后封存的 v24 候选，不是永久禁止新候选优化。
涉及数据泄漏或破坏科学主线的捷径继续禁止；容量、objective、normalization、gate
参数化、采样与优化器等非主线修改，可以在新版本、预注册选择和充分消融下进行。

## 29.1 不降低 PACRE field threshold

当前 field threshold 固定为 0 是模型定义的一部分。

降低阈值会：

- 直接增加 target count；
- 同时增加背景；
- 形成新的 calibration search；
- 破坏固定水平集主张；
- 无法解释模型是否真正学到 completion。

## 29.2 不把 Base@B 0.14 作为新 occupancy

这实际上将 CURE 变成 Base@B 后处理器，会丢失“从 Base@A 漏检状态补全”的
科学问题，而且 0.14 已由 D_V 选择，存在直接数据泄漏。

## 29.3 不添加 morphology

形态学可以提高某些 IoU，但会：

- 引入 kernel；
- 删除一像素目标；
- 改变推理合同；
- 掩盖 field 形状问题；
- 不解决 target count不足。

## 29.4 不增加宽度/attention

当前失败不是参数没有更新，也不是工程断路。

增加容量会使下一结果无法区分：

- common evidence修正；
- 纯容量收益；
- 过拟合 D_V。

因此当前 v24 不增加容量。若新候选有清楚的 representational bottleneck 假设，
宽度或 attention 可以修改，但必须版本化并与等训练预算的主线候选做消融，且不得
使用 \(D_V/D_T\) 选型。

## 29.5 不直接修改 PMOPE

当前模型不允许 common compatibility 独立驱动 interaction；GCR 是否能通过
显式有界 gate 改善 residual 幅度仍是候选假设。先只检验这一 representation
路径、保持 PMOPE 不变，才能维持单机制解释。

该约束只属于当前 v24。若 OOF 证据指向 objective mismatch，新候选可以修改
PMOPE 或其他训练目标，但必须重新预注册并用消融区分 representation 与 objective
收益。

---

# 30. 预期与诚实边界

GCR-PACRE 的设计目标不是某个硬编码 uplift，而是对冻结有效 Base 坐标包络：

| 指标 | v23 sealed snapshot | v24 正式比较 |
|---|---:|---|
| true targets | 149 | \(>\max_{\rm valid\ Base}T\) |
| recovered misses | 2 | \(>\max_{\rm valid\ Base}R\) |
| mIoU | 0.6021406727828746 | \(\ge\max_{\rm valid\ Base}I\) |
| nIoU | 0.5608662365828017 | \(\ge\max_{\rm valid\ Base}N\) |
| retention | PASS | exact PASS |
| false-alarm budgets | PASS | frozen safety budgets PASS |

这些是动态相对门禁，不是结果承诺。当前 Base ledger 会导出人类可读的
`151/4` 整数下界，但 verifier 始终读取全精度 ledger。

GCR-PACRE 的可验证优势是：

1. 为不能独立驱动输出的 common compatibility 增加显式、可审计的幅度路径；
2. common evidence只作为 gain，不能独立造 completion；
3. gain 有界且无可调参数；
4. binary flip antisymmetry保持；
5. 参数量和接口不变；
6. target 与 background 具有获得方向相反 gain 的潜力，必须由 OOF 证明；
7. 内部 OOF performance gate 可在再次消耗 D_V 前阻止无效候选。

---

# 31. 最终研究判断

v23 PACRE-VC 的失败不是 CURE-Lite 主线失败。

它更具体地否定了：

> **仅依赖 phase-centered residual compatibility、固定 PMOPE 和当前 Formal800
> 协议的 PACRE-VC，在该 \(D_V\) 上不能同时超过低阈值 Base 的目标恢复能力和
> 高阈值 Base 的分割质量。**

下一步最合理的问题是：

> **phase-common absolute feature evidence 能否在不成为独立目标生成器的前提
> 下，选择性地增强正确 residual completion、抑制背景 residual，从而同时提高
> target count 和 IoU？**

GCR-PACRE-v24 正是对这一问题的最小、单机制、无新增参数测试；其新颖性和
性能均未在本文档中预设为成立。

在 v24 完成内部未见验证、Formal800 和 adaptive D_V PASS 前：

```text
D_T                     BLOCKED
Full CURE               BLOCKED
cross-backbone          BLOCKED
three-dataset training  BLOCKED
success claim           NOT ESTABLISHED
```

---

# 32. 主要源码与协议

## PACRE-VC v23

- PACRE-VC wrapper  
  https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v23/pacre_vc.py
- PACRE/PACRE fields and equation  
  https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v22/pacre.py
- Formal800 training  
  https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v23/formal_training.py
- Formal \(D_V\) evaluation  
  https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v23/formal_evaluation.py
- PMOPE / Sobolev objective  
  https://github.com/Arialliy/cure-lite/blob/main/cure_lite/coverage_state_sobolev.py

## Frozen protocol

- v23 protocol directory  
  https://github.com/Arialliy/cure-lite/tree/main/protocols/IRSTD-1K/pacre_v23_verifier_corrected
- Relative-performance preregistration  
  https://github.com/Arialliy/cure-lite/blob/main/protocols/IRSTD-1K/pacre_v23_verifier_corrected/relative_performance_preregistration.md
