# CURE-Lite NLCC-v12 模型与代码设计

## 1. 当前阶段

本版本是在 CCFR-v11 的独立 dataset-free holdout 正式失败后建立的新候选。
历史 v4--v11 源码、结果和封存产物保持只读，不回写、不重跑、不放宽门槛。

总研究主线保持：

```text
CURE-Lite 核心模型
  -> 独立冻结确认
  -> Full CURE 设计
  -> NUAA-SIRST / NUDT-SIRST / IRSTD-1K
  -> 多个冻结 IRSTD detector 分别验证
```

v12 只解决当前 CURE-Lite decoder 的状态表达问题。它不是 Full CURE，不接入
MSHNet、DNANet、UIUNet 或 SCTransNet，也不读取新的真实数据结果。

候选名称冻结为：

> **Null-Anchored Local Count Crossing（NLCC）**  
> 空状态锚定的局部计数交叉。

当前状态：

```text
method_id                         = nlcc_v12
design_status                     = PREFROZEN
implementation_status             = NOT_STARTED
dataset_free_development          = NOT_RUN
independent_dataset_free_holdout  = NOT_RUN
real_D_R_authorized               = false
formal800_authorized              = false
full_CURE_authorized              = false
cross_detector_authorized         = false
public_novelty_status             = BOUNDED_NOT_ESTABLISHED
```

## 2. v11 负结果要求 v12 改变什么

CCFR-v11 已完成 400/400 updates、1200/1200 decoder forwards 和全部有限非零
梯度检查，但最终 0/8 groups 通过。直接观测包括：

- factual-miss target 通过，但 factual-miss background 与 factual-no-miss 失败；
- 六个 clean groups 的方向计数均正确，但绝对端点均未通过；
- \(1\to0\) 的 clean \(D\) 明显高于 \(2\to1\) 和 \(3\to2\)；
- occupancy-conditioned tensor 在 depthwise GroupNorm 前进入，局部变化可改变
  全空间归一化统计；
- common-mode budget 一旦激活就必须通过 softmax 分配，缺少显式 null 输出。

因此下一候选不能只是：

- 更换一个 evidence activation；
- 继续使用 \(B(F)+E(F)g(C)\) 的末端幅值调制；
- 调整 reciprocal temperature、loss、训练步数或门槛；
- 将 v11 的 coverage multiplier 简单移动到另一个位置；
- 增加 attention、Transformer、多尺度、额外 head 或第二个 decoder。

v12 必须在同一个固定方程中同时满足：

1. occupancy 不进入任何空间统计归一化；
2. occupancy 能改变 phase active set，而不只是乘一个正标量；
3. \(1\to0\)、\(2\to1\)、\(3\to2\) 的一步 count release 不随原 count 衰减；
4. 固定 null 允许所有 phase 都不输出；
5. baseline 对 occupancy 保持不变；
6. paired difference 的支持严格局部；
7. 不增加模块、参数、训练分支或推理分支。

## 3. 冻结的输入与网络拓扑

公开 decoder 输入仍然只有：

\[
\left(\operatorname{sg}(F_b), O\right),
\]

其中 \(F_b\) 是任意冻结 IRSTD detector 提供的一个 feature tensor，\(O\) 是
由冻结 Base probability 和阈值得到的 hard occupancy。pair kind、GT、目标
ID、另一端标签和 source ID 均不进入 decoder。

这里冻结的是 **active v4 factorized topology**，不是早期 v0.1
`CURELiteDecoder`：

```text
freeze_v0_1_decoder                 = false
freeze_active_v4_factorized_topology = true
```

活动拓扑为：

```text
detached F_b
  -> 1x1 stem
  -> GroupNorm(affine=False)
  -> SiLU
  -> 3x3 depthwise
  -> GroupNorm(affine=False)
  -> SiLU
  -> 1x1 pointwise residual
  -> existing baseline/evidence phase heads
  -> PixelShuffle
  -> optional frozen bilinear fallback
  -> one residual logit map
```

Reference \(C_f=64,s=4\) 必须保持 4,385 个参数、6 个参数张量和 7 个
state-dict entries；dataset-free \(C_f=8,s=4\) 必须保持 2,593 个参数。

## 4. NLCC 的唯一状态方程

### 4.1 完全 feature-only 的共享 trunk

\[
H_0(F)
=
\operatorname{SiLU}
\left[
\operatorname{GN}_s
\left(
W_s\operatorname{sg}(F)
\right)
\right],
\]

\[
H(F)
=
H_0
+
\frac12 W_p
\operatorname{SiLU}
\left[
\operatorname{GN}_d
\left(
W_dH_0
\right)
\right].
\]

plus/minus endpoint 的 \(H(F)\) 完全相同。occupancy 不再进入 stem、
depthwise convolution 或 GroupNorm。

两个既有 head 产生：

\[
b=W_bH,\qquad r=W_eH,
\]

其中每个 feature cell 有 \(P=s^2\) 个 raw phase responses
\(r_1,\ldots,r_P\)。

### 4.2 固定 null 坐标与 phase-relative response

加入一个不占参数、不进入 state dict 的固定坐标：

\[
r_\varnothing=0.
\]

null-anchored reference 为：

\[
\mu_\varnothing(r)
=
\frac{r_\varnothing+\sum_{k=1}^{P}r_k}{P+1}
=
\frac{\sum_{k=1}^{P}r_k}{P+1}.
\]

phase-relative response 为：

\[
q_j(F)=r_j-\mu_\varnothing(r).
\]

该 reference 不强制分配总预算。任意 phase 都可以为 inactive，全部 phase
也可以同时为 inactive。

### 4.3 单位局部 count burden

occupancy 仍使用冻结投影：

\[
\widehat O=\Pi_{\max}(O).
\]

固定 \(3\times3\) 全一核计算：

\[
C(O)=\mathbf 1_{3\times3}*\widehat O,\qquad C\in\{0,\ldots,9\}.
\]

NLCC 不再使用 \(1/(1+C)\)、\(\log(1+C)\) 或 \(C/9\)，而直接定义一个
无参数的 unit-count boundary：

\[
m_j(F,O)=q_j(F)-C(O).
\]

因此任意一步删除都严格产生：

\[
m_j(C-1)-m_j(C)=1.
\]

这对 \(1\to0\)、\(2\to1\)、\(3\to2\) 完全相同。raw count 的单位由
exponential crossing 的自然 log-evidence 坐标确定，不提供可调 scale、
temperature 或 slope。

### 4.4 crossing forward 与恢复梯度

可观察 forward evidence 为：

\[
h(m)=
\begin{cases}
0,&m\le0,\\
\exp(m)-1,&m>0.
\end{cases}
\]

反向复用历史 CR primitive 的完整 recovery carrier：

\[
\widetilde h(m)
=
\operatorname{sg}(h(m))
+
\exp(m)
-
\operatorname{sg}(\exp(m)).
\]

所以：

\[
\frac{\partial\widetilde h}{\partial m}=\exp(m).
\]

最终：

\[
E(F,O)=\operatorname{PixelShuffle}
\left(\widetilde h(m(F,O))\right),
\]

\[
B(F)
=
-\operatorname{softplus}
\left[
\beta+\operatorname{PixelShuffle}(b(F))
\right],
\]

\[
\boxed{
z_\theta(F,O)=B(F)+E(F,O)
}.
\]

NLCC 的全部方法变化就是这个 joint state operator：

\[
\boxed{
E_j(F,O)
=
\left[
\exp
\left(
r_j
-
\frac{\sum_k r_k}{P+1}
-
C(O)
\right)
-1
\right]_+
}.
\]

不能把它拆写成多个新模块。null reference、local count 和 crossing 是一个
phase 是否释放及释放多少的联合状态定义。

## 5. 为什么不采用线性 crossing 或 \(C/9\)

若一次删除仅使 logit 增加 \(\delta\)，任意 baseline 下 sigmoid 差的最大值为：

\[
\max_x\left[\sigma(x+\delta)-\sigma(x)\right]
=
\tanh\left(\frac{\delta}{4}\right).
\]

线性 ReLU 与 \(C/9\) 给出 \(\delta=1/9\)，上界仅为 0.0277706；即使使用
raw \(C\)，线性上界也只有 0.2449187。二者均不可能达到冻结的：

\[
D_\Delta\ge0.8.
\]

端点约束：

\[
p^+<0.05,\qquad p^->0.95
\]

还要求：

\[
z^- - z^+>2\log19=5.88887796.
\]

使用 CR exponential 且两端都 active 时：

\[
\Delta z
=
\exp(m^+)\left(\exp(\delta)-1\right),
\]

差值没有有限上界。

若使用 \(C/9\)，最低可达解需要约 49--55 的 evidence 与约 \(-52\) 的
baseline 相互抵消；raw \(C\) 只需要约 2.43--8.32 的 evidence 与约
\(-5.37\) 的 baseline。v12 因此预先选择 raw \(C\)，不是在实验后搜索
count scale。

需要准确限定：强端点成功依赖 plus/minus 都处于 active exponential 区域，
不是只依赖一次从 inactive 到 active 的零点跨越。

## 6. 结构性质与可证伪边界

### 6.1 Null

\[
F=0
\Rightarrow
H=0,\ r=0,\ m=-C\le0,\ E=0.
\]

zero feature 不能凭 occupancy 单独生成正 evidence。

### 6.2 Locality

occupancy 在全部 convolution 和 GroupNorm 之后进入。固定 \(F\) 时，若两个
endpoint 的 count 只在集合 \(S\) 变化，则 native phase evidence 和
PixelShuffle 后的输出差也只在 \(S\) 对应的 subpixels 变化。标准整倍 stride
下不会产生 count-support 外的 paired difference。

### 6.3 Active-set nesting

\[
\mathcal A_C
=
\{j:q_j>C\},
\qquad
\mathcal A_C\subseteq\mathcal A_{C-1}.
\]

删除只能保持或扩大 active set，不能产生反方向 phase release。

### 6.4 Same-count limitation

\[
C(O_a)=C(O_b)
\Rightarrow
z(F,O_a)=z(F,O_b).
\]

NLCC 不区分 \(3\times3\) 内相同 count 的具体 occupancy 排列。若真实验证
表明修复必须依赖方向性 occupancy geometry，本候选应冻结失败，而不是临时
增加位置模块。

### 6.5 Phase capacity

phase contrast 直接进入 crossing；固定 null 只衰减 pure common shift
到 \(1/(P+1)\)，不会消除它。现有 \(P=s^2\) phase head 可表示 1、2、3
subpixel targets，也可表示全部 phase inactive。

### 6.6 数值边界

CR gradient 为 \(\exp(m)\)，不是全轴均匀梯度：

- 大负 margin 会使梯度变小；
- 大正 margin 可能产生非有限值；
- 不允许在失败后增加 clamp、温度或更换 surrogate。

实现必须记录 margin 与 recovery factor 范围；任意非有限 forward/loss/
gradient 立即失败。

## 7. v12 不修改的部分

| 层 | 冻结依赖 |
|---|---|
| Base | 一次 frozen Base extract，feature detach |
| topology | active v4 factorized topology |
| objective | PECO-v10 |
| batch | factual-miss 4 + factual-no-miss 4 + pair 2 |
| paired execution | 一个 \(2B\) decoder forward |
| update | 3 forwards、12 states、1 backward、1 optimizer step |
| wrapper | `CURELiteFactorizedModel` |
| inference | one Base + one decoder、pre-mask、frozen threshold、hard union |

不修改根 `cure_lite/__init__.py` 和 `train/__init__.py`。v12 只通过独立
子模块导入，不新增公共根 API。

## 8. dataset-free gate 的可达性修正

v11 holdout 的 222/222 个 far `completion_plus` 均没有同位 feature
witness；其 completion cell 到唯一 signal cell 的 Chebyshev 距离为 2 或
3，而 active-v4 的显式 spatial receptive radius 为 1。v11 可通过
GroupNorm 全局统计、边界和人口记忆产生输出，因此其正式 FAIL 仍然有效；
但该子门禁不能被解释为纯局部 coverage 能力。

v12 不复用该不对齐的 anchor 构造。新的 unique dataset-free holdout 必须在
任何 v12 训练结果被查看前冻结，并只作以下输入修正：

1. 在 `completion_plus` 对应 feature cell 的专用 channels 6/7 写入
   occupancy-invariant witness；
2. completion 继续位于 changed-count support 外；
3. 新增 matched anchor-null：移除 channels 6/7 witness，同时移除
   completion target；
4. positive completion 与 matched null 分别要求 \(>0.95\) 与 \(<0.05\)；
5. D/H/G、factual、component-null、400 updates、4/4/2 和原
   0.95/0.05/0.8 阈值均不放宽。

训练前 reachability receipt 必须满足：

```text
unwitnessed_completion_count                 = 0
completion_changed_support_overlap_count     = 0
opposite_label_identical_input_count         = 0
opposite_label_local_signature_conflict_count = 0
clean_D_without_feature_witness_count        = 0
clean_D_without_count_difference_count       = 0
```

## 9. 实现与验证顺序

```text
Stage 0  proposal receipt + exact source bindings
Stage 1  additive config/decoder + four focused test files
Stage 2  topology/state/init/locality/algebra/gradient/inference regression
Stage 3  dataset-free development regression
Stage 4  new unique exposure holdout, 400 updates
Stage 5  only if Stage 4 PASS: IRSTD-1K D_R bounded validation
Stage 6  only if D_R PASS: 800x40 exposure replay
Stage 7  only if all prior gates PASS: formal800 seed 42/43
Stage 8  only if both seeds pass: frozen confirmation
Stage 9  only after CURE-Lite freeze: Full CURE and three datasets
```

Stage 1 至少需要验证：

- v4/v12 module sequence、state key/shape/init 逐值一致；
- 4,385/2,593 参数、6 parameter tensors、7 state entries；
- 每个既有 module 只调用一次；
- unit-count equation、null reference 和 CR forward/backward 精确一致；
- 显式构造可达到 .05/.95 的双 active endpoint witness；
- \(1\to0\)、\(2\to1\)、\(3\to2\) 满足
  \((E^-+1)/(E^++1)=e\)；
- zero-feature、same-count、identity、active-set nesting；
- count-support 外 exact zero paired difference；
- 1/2/3 phase、stride-1、resize fallback；
- 2B paired forward、双端 VJP、6 tensors finite gradients；
- Base detach、one Base、one decoder、pre-mask 与 hard union 回归；
- 旧 v4--v11 和根 API 测试不回归。

## 10. 数据集使用时间

当前不读取新数据集结果。只有新的独立 dataset-free holdout 全部通过后，才
第一次使用真实 IRSTD-1K \(D_R\) 验证 NLCC-v12；只有 \(D_R\) 与
32,000-step 暴露门禁通过后，才运行 IRSTD-1K seed 42/43 的 800-epoch
正式验证。

CURE-Lite 获得冻结确认后，才使用完整三数据集：

- NUAA-SIRST；
- NUDT-SIRST；
- IRSTD-1K。

三数据集阶段用于 Full CURE 的跨数据与跨 detector 证据，不用于反复选择
当前 Lite 状态方程。

## 11. 创新边界

本地既有相关工作审计已经表明：“通用 mask refiner”“轻量 decoder”
“冻结 Base”“合成删除”“降低 FA”和“跨 backbone 插件”均不能单独作为
ICLR 创新点。

CURE 保留的研究主张仍是：

> 在同一冻结 detector feature 下，对 detector coverage state 做具有稳定
> lineage 的受控干预，并直接学习 hard mask 之前的 coupled target-response；
> 最终以 natural-miss recovery、固定 FA 和 detected-target retention 检验
> 该学习对象是否成立。

NLCC 是使这一学习对象可由轻量、detector-independent decoder 表达的候选
状态方程，不是靠模块堆叠形成的独立卖点。只有独立门禁、真实 \(D_R\)、
逐种子 formal800、冻结确认以及后续跨 detector/三数据集结果均成立后，
才能讨论完整 ICLR 贡献；当前不能预先宣称模型成功或录用可能。
