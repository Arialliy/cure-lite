# CURE-Lite Paired-Objective 协议

> **协议时点快照**：本文件第 1～9 节及
> `protocols/IRSTD-1K/paired_objective_v1/proposal_receipt.json`
> 冻结的是协议形成时的 `performed=false` / `authorization=false` 状态。
> 后续 additive 实现不反写该 receipt；当前工程状态与下一门槛以
> [CURE-Lite 下一步方案](CURE_Lite_下一步方案.md) 第 0、14.3 节为准。

## 1. 本阶段结论

本协议把已经冻结的同源离散覆盖响应
\(\Delta_gQ\) 收束为一个唯一、可反证的训练目标，但不授权实现或训练。

冻结状态为：

```text
core object = same-source discrete coverage response
paired objective = balanced pre-mask score-difference regression
zero-order anchor = frozen factual-miss + factual-no-miss losses
independent legal endpoint loss = control only
null pairs = control only
pairwise implementation = false
training authorization = false
novelty status = unsearched
```

机器可核验的非实验 proposal receipt 为：

- [paired-objective proposal receipt](protocols/IRSTD-1K/paired_objective_v1/proposal_receipt.json)
- fingerprint：`5a2f357911fb5f1dc1a946b3dbad429d256c390677d238b2f395fe90ce91fac8`

该 receipt 只冻结数学形式、未来接口、controls 和停止规则。它不位于
`runs/`，没有 `COMPLETE.json`，也不是性能结果。

## 2. 冻结依据与边界

本协议继承并且不修改：

- [核心学习对象重定义](CURE_Lite_核心学习对象重定义.md)；
- core-object receipt：
  `62461b5514b45d4082ea4001c4e8324b2f5ad0542a4ae11891b3db4fda980ef9`；
- H0、P0-B/C 和 v0.1/v0.2 的既有结论；
- \(Q=\sigma(\text{decoder logits})\) 的 pre-hard-mask score 语义；
- \(D_g=\mathcal R_{\mathcal G,V}(O^-)\setminus
  \mathcal R_{\mathcal G,V}(O^+)\)；
- clean positive pair 中 \(D_g=A_g\) 的资格不变量；
- 单次 \(Q\) inference、hard mask、固定 threshold 与 hard union。

本协议不把：

- deleted endpoint 当成 factual miss；
- factual miss 与 legal target 进行配对；
- \(D_g\) 改成简单的 \(G_g\cap C_g\)。

最后一点尤其重要。这里学习的是 matcher 定义下的实例补全状态转移，而
不是仅重建 occupancy hole。对删除前已匹配的目标 \(g\)，其删除前即使有
少量 writable 像素，也不属于 \(\mathcal R(O^+)\)；删除使 \(g\) 成为
unmatched 后，全部有效且可写的实例区域才进入 \(\mathcal R(O^-)\)。
因此：

\[
A_g=V\cap G_g\cap\neg O^-,
\]

可能严格大于 \(V\cap G_g\cap C_g\)。这正是“实例补全”与“复制删除孔洞”
的区别，不能在本协议中改回像素补集语义。

## 3. Pair catalog 契约

### 3.1 clean positive pair

一个 positive pair 的原子单位为：

\[
p=(s,g,F_s,O^+_s,O^-_{s,g},V_s,D_{s,g}).
\]

其中：

\[
O^-_{s,g}=O^+_s\setminus C_{s,g}.
\]

进入训练 pool 前必须逐对满足：

1. source、group、evaluation GT、native GT、prediction component identity
   完整；
2. \(F^+=F^-=F_s\)，内容指纹逐字节相同；
3. 只删除一个完整 prediction component；
4. \(O^+\setminus O^-=C_g\)，且 \(O^-\subset O^+\)；
5. selected GT 是唯一新增 unmatched GT；
6. 其他 target-component matching identity 不变；
7. native/evaluation target lineage 一对一且 geometry-safe；
8. \(\Pi O^-\ne\Pi O^+\)；
9. \(C_g\cap V\) 不与删除前已有 unmatched GT 相交；
10. 重新计算实际标签增量 \(D_g\)，并验证：

    \[
    D_g=A_g=V\cap G_g\cap\neg O^-;
    \]

11. \(D_g\ne\varnothing\)，且 \(V\setminus D_g\ne\varnothing\)；
12. 所有判断只使用 \(D_R\) 的冻结 cache、标签、matcher、lineage 和
    occupancy，不使用 \(D_V/D_T\) 或未来性能。

任一失败都必须写入原因码和排除清单，不能静默删除。

### 3.2 null 与 identity pairs

null pair 不进入 proposed objective 的训练，只用于检查特异性。

component-null 必须满足：

\[
O^-=O^+\setminus C,\qquad
\Pi O^-\ne\Pi O^+,\qquad
D=\varnothing.
\]

“unmatched prediction”或“false-positive component”的名称不能替代
\(D=\varnothing\) 的实际验证。

identity-null 固定为：

\[
O^-=O^+,\qquad D=\varnothing.
\]

它用于验证相同输入必须逐值产生 \(\Delta Q=0\)。

### 3.3 必须形成的 manifest 字段

每个 pair 至少记录：

- `pair_id`、`pair_kind`、`sample_id`、`group_id`；
- native/evaluation `gt_id`、`pred_id`；
- feature、\(O^+\)、\(O^-\)、\(C\)、\(V\)、\(D\)、\(A\) 的指纹；
- before/after match 指纹；
- \(\mathcal R^+\)、\(\mathcal R^-\) 的语义真值指纹；
- \(\Pi O^+\)、\(\Pi O^-\) 指纹；
- clean/noninterference/visibility/lineage 判定；
- \(A\cap C\)、\(A\setminus C\)、\(C\setminus G\) 的像素记账；
- eligibility、排除原因和 canonical order；
- source tree、配置和上游 receipt 绑定。

## 4. 唯一 paired objective

### 4.1 两端 score

对一个 clean positive pair：

\[
Q^+_\theta
=
\sigma\!\left(D_\theta(F,\Pi O^+)\right),
\]

\[
Q^-_\theta
=
\sigma\!\left(D_\theta(F,\Pi O^-)\right).
\]

两端必须：

- 使用同一个 decoder 和同一组参数；
- 在同一个 optimizer update 内计算；
- 都保留计算图；
- 都不 stop-gradient；
- 在 occupancy hard mask 之前取值。

定义：

\[
\Delta Q_\theta=Q^-_\theta-Q^+_\theta\in(-1,1).
\]

禁止把 logit difference、post-mask probability difference 或 thresholded
mask difference 替换为候选。

### 4.2 response 与 zero-response 域

positive response 域为：

\[
P_p=\{x:D_p(x)=1\}=A_p.
\]

zero-response 域固定为整个有效域中的补集：

\[
Z_p=V_p\setminus P_p.
\]

不能根据观察结果把 \(Z_p\) 缩成某个 dilation ring。当前 decoder 包含
GroupNorm 和空间卷积，局部 occupancy edit 可以通过归一化统计产生非局部
响应；所以架构上安全的响应审计域只能是完整 \(V_p\)。

为诊断 occupancy-hole shortcut，额外记账但不改变 loss 权重：

\[
P_{\mathrm{hole}}=A_p\cap C_p,
\qquad
P_{\mathrm{extend}}=A_p\setminus C_p,
\]

\[
Z_{\mathrm{spill}}=(C_p\cap V_p)\setminus G_p.
\]

### 4.3 class-balanced difference loss

唯一冻结的 per-pair loss 为：

\[
\ell_\Delta(p)
=
\frac{1}{2|P_p|}
\sum_{x\in P_p}
\left(
\frac{\Delta Q_\theta(x)-1}{2}
\right)^2
+
\frac{1}{2|Z_p|}
\sum_{x\in Z_p}
\left(\Delta Q_\theta(x)\right)^2.
\]

batch reduction 为：

\[
\mathcal L_\Delta
=
\frac1{B_p}
\sum_{p=1}^{B_p}\ell_\Delta(p).
\]

选择平方损失的原因是：

- \(\Delta Q\) 是有符号 score difference，不是 probability 或 logit；
- positive error 的原始范围为 \([-2,0]\)，除以 2 后与 zero-response
  error 的 \([-1,1]\) 范围一致，两个 stratum 均被限制在 \([0,1]\)；
- BCE 的输入语义不成立；
- Dice 对有符号差分没有固定概率解释；
- positive/zero 两域分别取均值，避免 tiny target 被全图背景数量淹没；
- 数学形式包含明确的跨端点项。

本协议不搜索 loss family、margin、temperature、dilation 或 focal 参数。

### 4.4 零阶 absolute anchor

有限差分满足：

\[
\Delta_g(Q+B(F))=\Delta_gQ,
\]

所以只训练 \(\mathcal L_\Delta\) 无法识别 \(Q\) 的绝对值。

零阶锚严格复用当前 factual 监督：

\[
\mathcal L_{\mathrm{abs}}
=
\mathcal L_{F+}
+
\mathcal L_{F0},
\]

其中 \(\mathcal L_{F+}\) 与 \(\mathcal L_{F0}\) 继续使用当前
`CURELiteLoss`、原子 factual target、原 valid mask 和各自 per-state
mean。不得把 legal endpoints 的独立 absolute losses 偷加到 proposed
objective。

最终唯一训练目标冻结为：

\[
\boxed{
\mathcal L_{\mathrm{CURE\text{-}Lite}}
=
\mathcal L_{F+}
+
\mathcal L_{F0}
+
\mathcal L_\Delta
}.
\]

三个系数固定为 \(1,1,1\)。这是预声明的 branch-level unit weighting，
不通过 \(D_R\) 或 \(D_V\) 搜索。

### 4.5 为什么它不可分解

对任一像素，令 \(d\in\{0,1\}\)，并令固定 range scale
\(\rho_0=1,\rho_1=\tfrac14\)，则损失中的像素项与下式成正比：

\[
\rho_d(Q^- - Q^+ - d)^2
=
\rho_d\left[
(Q^-)^2+(Q^+)^2-2Q^-Q^+
-2dQ^-+2dQ^++d^2
\right].
\]

其中在两个 stratum 中都存在非零的：

\[
-2\rho_dQ^-Q^+
\]

是跨端点项，并且：

\[
\frac{\partial^2\ell_\Delta}
{\partial Q^-\partial Q^+}
\ne0.
\]

在非最优、非饱和 toy case 中：

\[
\frac{\partial\ell_\Delta}{\partial Q^-}
=
-\frac{\partial\ell_\Delta}{\partial Q^+}
\ne0.
\]

因此改变 pair identity 会改变 loss 与 gradient。若未来实现的 mixed
derivative、双端 gradient 或 permutation toy 不满足这些性质，立即记为
`STRUCTURAL_FAIL`。

## 5. 训练预算与确定性 schedule

未来若另行授权实现，仍冻结：

```text
epochs = 800
steps_per_epoch = 40
optimizer updates = 32,000
factual-miss batch = 4 states
factual-no-miss batch = 4 states
paired batch = 2 pairs = 4 endpoint states
decoder endpoint states per update = 12
```

这样 paired route 与旧三分支训练都消费每步 12 个 decoder states：

```text
old: 4 factual-miss + 4 factual-no-miss + 4 synthetic
new: 4 factual-miss + 4 factual-no-miss + 2 × (before, after)
```

pair identities 使用 seed-specific stable-hash canonical cycle；在完整
32,000-step schedule 中，每个 eligible positive pair 的实际暴露次数相差
不得超过 1。每个 pair batch 优先取不同 source，避免同一步重复计算同一
\(Q^+\)。如果 pool 只含一个 source，协议直接失败。

同一步的四个 pair endpoints 必须拼成一个 batch forward，不重新计算 frozen
Base feature。decoder 参数每步变化，所以不能跨 optimizer updates 缓存
\(Q^+\)。

pair target/source exposure、零暴露率、最大份额和 source concentration
必须在训练前形成 receipt；本协议本身不运行该模拟，也不读取结果。

## 6. matched controls

以下 **pair-matched controls** 必须共享 decoder、初始化、optimizer、
32,000 updates、factual schedule、pair identities、before/after states、
endpoint forward 数和随机计划：

- independent-endpoint ERM；
- after-only；
- 两种 occupancy-only；
- feature-only；
- target permutation；
- plus/minus stop-gradient。

历史 factual-only `F` 与 factual-exposure-matched `F×` 是 baseline
comparators，不虚构为 pair-matched controls。`F` 保持每步 8 个 factual
states；`F×` 保持每步 12 个 states，其中额外 4 个是 factual replacements。
二者分别保存原生 exposure/forward ledger，并在结果表中显式报告与 paired
route 的计算差异。identity-null 与 component-null 是只读 evaluation
controls，不进入 optimizer。

### 6.1 geometry-matched independent-endpoint ERM

使用同样的两个 endpoints，但分别求 absolute state loss，不出现
\(Q^- - Q^+\)。对 \(e\in\{+,-\}\)，唯一冻结为：

\[
T^e=\mathcal R_{\mathcal G,V}(O^e),
\]

\[
B^e
=
V\cap\neg O^e\cap
\neg\left(\bigcup_jG_j\right),
\qquad
M^e=T^e\cup B^e.
\]

即 endpoint target 是该 occupancy 下完整的实例级 completion field；
valid domain 只含该 target 与可写真背景，其他 GT 和 occupancy 像素排除。
不得把标签草率写成 \(A_g\) 与 \(0\)，也不得复用旧 synthetic valid mask。

每个 pair 的 control loss 为：

\[
\ell_{\mathrm{ind}}(s,g)
=
\frac12\left[
\operatorname{CURELiteLoss}(z^+,T^+,M^+)
+
\operatorname{CURELiteLoss}(z^-,T^-,M^-)
\right],
\]

再对 pair 做算术平均。其总目标固定为：

\[
\mathcal L_{F+}+\mathcal L_{F0}+\mathcal L_{\mathrm{ind}},
\]

三个系数仍为 \(1,1,1\)。所以 endpoint 标签、valid mask、endpoint
reduction、pair reduction 和总权重均不再留给实现阶段选择。

它是判断“paired coupling 是否必要”的首要 matched control。

### 6.2 after-only control

仍计算相同 before/after forwards，但只有 after endpoint 接受旧
synthetic-state absolute loss。它隔离“额外 endpoint 数据”与“差分约束”。

### 6.3 occupancy-only control

保持输出网格、endpoint schedule 和 loss 不变，但禁止访问真实 \(F\)。

第一项是 nominal control：同一 decoder 接收 `zero_like(F)`。它匹配参数
文件、输出网格和 MAC 路径，但 feature-project 权重失活，所以其阴性结果
不能证明 feature 必需。

第二项是机制冻结前必须完成的 capacity-active control：以一个冻结、
source-independent、所有样本相同的二维 DCT 坐标基
\(B_{C,h,w}\) 替代 \(F\)，仍使用完全相同的 decoder。基函数按
\((k_y+k_x,k_y,k_x)\) 排序，排除 DC 项 \((0,0)\)，取前 \(C\) 项；每个通道
在 \(h\times w\) 网格上去均值并除以 RMS。若 \(C>hw-1\) 或任一通道无法
标准化，preflight 失败。basis 公式、shape、dtype 和 SHA 必须进入 receipt，
并在所有 source 与 seed 间保持相同。

该 control 不读取图像内容，但保持 feature-project 的全部输入通道活跃，
同时匹配 decoder 拓扑、参数量、输出网格和 MAC。完整方法必须优于 nominal
zero-feature 和 capacity-active coordinate-basis 两个 controls，才允许排除
occupancy-only 解释并冻结机制。否则只能说 nominal control 较弱，不能声明
feature necessity。

### 6.4 feature-only control

保持 decoder 接口和参数预算，但两个 endpoints 都接收同一个固定零
occupancy。它不能通过最终 hard mask 人为制造训练差分。

### 6.5 permutation 与 stop-gradient controls

- 对兼容 pair 构造 source-disjoint、无不动点的确定性 target permutation：
  recipient 与 donor 必须具有相同 tensor shape，donor \(A\) 必须位于
  recipient \(V\) 内，pair ID 与 source ID 均不得相同；在 stable-hash
  canonical order 上求字典序最小的二分图完美匹配。这样保持完整 endpoint
  集合和 target 边缘集合，只破坏 identity；
- 分别设置 `detach(Q+)` 与 `detach(Q-)`；
- identity-null 与 component-null 只作 control/evaluation，不进入 proposed
  objective 训练。

若不存在上述完美匹配，permutation control 记为
`COMPUTATIONALLY_INCONCLUSIVE`，不得静默丢弃 singleton source，也不得
用组内自映射代替；缺少该 control 时不能冻结 CURE-Lite 机制。

如果 permutation 不降低 paired/factual 结果，或 stop-gradient 与完整双端
目标等价，就不能声明 pair identity 或双端耦合是机制。

## 7. 未来最小实现接口

当前 `BranchBatch` 与 `multi_branch_train_step` 不能实现本协议：

- `BranchBatch` 只有一个 occupancy；
- `valid_mask` 被要求与 occupancy 不重叠；
- `step.py` 不保留两个 endpoint outputs；
- synthetic branch 只计算独立 absolute loss；
- `CURELiteLoss` 接收 logits，不能直接消费有符号 \(\Delta Q\)。

未来若获得单独授权，最小新接口应为：

```text
PairBatch
  feature: [B,C,h,w]
  occupancy_plus: [B,1,H,W] bool
  occupancy_minus: [B,1,H,W] bool
  label_increment: [B,1,H,W] float32 binary
  image_valid_mask: [B,1,H,W] bool
  pair_id/sample_id/group_id: metadata only
```

约束：

1. `feature` 只存一份；forward 时为两个 endpoints 扩展；
2. metadata、GT、ID、role 不得输入 decoder；
3. logits 一次 batch forward 后拆为 plus/minus；
4. 先 sigmoid，再在 raw pre-mask scores 上求差；
5. 一个联合 pair loss；
6. 一次 total backward 和 optimizer step；
7. optimizer 仍只含 decoder 参数；
8. inference graph、calibration 和 final union 完全不变。

本协议不指定生产类名或文件修改，也不实施上述接口。

## 8. 实现前 toy/static gates

以下门禁必须全部通过，才可以另行讨论实现授权：

1. receipt canonical fingerprint 与全部 source/evidence SHA 一致；
2. \(D_g\)、\(A_g\)、\(V\)、clean/noninterference 条件完整；
3. toy 证明 instance-level \(D_g\) 可严格大于 \(G_g\cap C_g\)，且这是冻结
   语义而非误标；
4. balanced loss 对 positive/zero 两域分别归一；
5. 非退化 toy 中 plus/minus gradients 都有限且非零；
6. mixed endpoint derivative 非零；
7. 固定 endpoint marginals、改变 pairing 后 loss/gradient 改变；
8. constant pre-mask score 的 raw difference 为零，而 post-mask difference
   可机械产生 \(cC\)；
9. identity pair 的 raw \(\Delta Q\) 严格为零；
10. null pair 只在实际 \(D=\varnothing\) 时成立；
11. 任何 hard mask、threshold、GT、ID 或 stop-gradient 进入 proposed
    objective 均失败；
12. stride-4 toy 记录 1–3 pixel target 的不可约表示误差；若当前 decoder
    无法表达而不产生超预算背景响应，记为结构容量失败，不修改 target 或
    dilation 规避。
13. independent-endpoint control 的 \(T^\pm,M^\pm\) 与完整
    \(\mathcal R(O^\pm)\) 逐像素一致；
14. permutation 必须产生 source-disjoint 完美匹配且无不动点；
15. DCT coordinate-basis control 的全部通道有效、source-independent 且
    feature-project 全部通道获得有限梯度。

## 9. 后续状态与停止规则

### 9.1 状态分类

`STRUCTURAL_FAIL`：

- target/valid 域错误；
- pair 跨目标干扰；
- projected occupancies 相同；
- post-mask 机械差分；
- loss 可分解；
- 一端无梯度；
- GT/ID/role 泄漏；
- 无 absolute anchor；
- 低分辨率不可表达。

`COMPUTATIONALLY_INCONCLUSIVE`：

- catalog 缺少 positive 或 component-null 支持；
- 数值非有限；
- 固定优化未收敛；
- group/fold 没有足够单位；
- source-disjoint target permutation 不存在完美匹配；
- capacity-active occupancy control 无法构造；
- 预声明区间跨零。

`PERFORMANCE_FAIL`：

- 结构和计算均有效，但 paired route 不优于 matched independent endpoints；
- nominal 或 capacity-active occupancy-only、feature-only 或 permutation
  可以解释结果；
- 只拟合 legal pair，不改善自然 factual misses；
- background、其他目标或 FA 约束退化；
- 任一开发 seed 未通过冻结性能门槛。

这些状态不得互相改名，也不得观察结果后修改 objective、权重、domain、
pair eligibility 或 schedule。

### 9.2 CURE-Lite 冻结性能门槛

本协议不产生新性能。若未来获得实现和训练授权，seed 42 与 43 必须逐种子：

\[
\Delta TP\ge2,\qquad
\Delta Pd\ge\frac2{170}=1.176\text{ 个百分点},
\]

\[
\Delta\mathrm{RecoveredAnchorMisses}\ge2,\qquad
\Delta RMR\ge\frac2{23}=8.70\text{ 个百分点},
\]

相对最佳 matched comparator 严格成立，并同时满足：

```text
retention = 1.0
pixel FA <= 1e-4
raw-background FA <= 1e-4
FP components / MP <= 100
budget_violation = false
```

通过只允许冻结并进入额外 seeds、grouped factual transfer 和未使用划分
确认，不直接授权 Full CURE。

## 10. 创新性和局限

本协议保留的单一方法贡献类型是：

> 新 objective / learning principle：用自然状态的零阶绝对值与同源 coverage
> query 的一阶离散响应共同识别一个 completion operator。

它不是 attention、decoder 堆叠、feature editor、matching sampler 或
后处理组合。其潜在科学价值来自：

- pair identity 实际进入梯度；
- 不要求 factual/legal population 可交换；
- 区分实例补全响应与 occupancy-hole reconstruction；
- 同一接口可以针对不同 frozen detector 重新训练。

仍未解决：

- detected-target response 是否迁移到自然 factual miss；
- occupancy-only 是否足以解释增益；
- stride-4 decoder 是否能表达极小目标；
- 当前文献中是否已有等价 derivative/equivariance supervision。

因此：

```text
method mechanism = specified but not implemented
protocol integrity = pass after matched-control closure
implementation status = in progress under subsequent user authorization
empirical support = not evaluated
ICLR novelty = unsearched
paper core = not established
```

## 11. 权限与主线

本协议 receipt 仍准确记录协议冻结时没有修改生产代码。后续用户已单独授权
additive pair catalog、paired loss 和 paired train-step 实现；该后续授权不
反写 receipt 的历史 `performed=false`。仍然禁止：

- 修改 production decoder 拓扑或覆盖旧 loss、旧 branch engine；
- 构造 transformation、S 或其他新模块；
- 在代码门槛通过前运行正式训练；
- 校准、推理或读取新的 \(D_V/D_T\)；
- 开始 Full CURE；
- 接入 DNANet、UIUNet、MSHNet 或 SCTransNet。

当前后续顺序是：

```text
additive implementation
  -> unit/toy/small-overfit validation
  -> D_R catalog and exposure preflight
  -> only if all pass, formal seed-42/43 training
```

总主线不变：

```text
完成 CURE-Lite 最小核心机制
  -> 冻结确认机制成立
  -> 设计 Full CURE
  -> 跨 IRSTD backbone 与三数据集验证
```
