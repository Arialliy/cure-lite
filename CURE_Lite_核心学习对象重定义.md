# CURE-Lite 核心学习对象重定义

## 1. 本阶段结论

H0 之后，CURE-Lite 不再把 legal deletion state 当作 factual miss 的替代样本，也不再沿 factual-to-legal matching、边缘重采样或 synthetic feature 修改继续设计。

本轮只保留一个概念上可成立的核心候选：

> **同源离散覆盖响应**：学习一个 coverage-conditioned completion operator，并用同一 frozen feature 下、删除一个合法覆盖组件前后的输出差，监督该算子对 coverage 变化的响应。

对应状态是：

```text
core object candidate = same-source discrete coverage response
conceptually admissible = true
pairwise objective protocol = not yet drafted
pairwise objective implementation = not started
mechanism support = not established
novelty status = unsearched
training authorization = false
```

机器可核验的非实验 proposal receipt 为：

- [core-object proposal receipt](protocols/IRSTD-1K/core_learning_object_redefinition_v1/proposal_receipt.json)
- fingerprint：`62461b5514b45d4082ea4001c4e8324b2f5ad0542a4ae11891b3db4fda980ef9`

该 receipt 位于 `protocols/`，不位于 `runs/`，也没有 `COMPLETE.json`。它冻结的是研究对象与权限边界，不是一次实验结果。

## 2. 为什么必须重新定义学习对象

当前实现训练的是三个相互独立的状态风险：

\[
\mathcal L
=
\overline{\mathcal L}_{F+}
+\lambda_0\overline{\mathcal L}_{F0}
+\lambda_s\overline{\mathcal L}_{S}.
\]

即使 v0.2 建立了 factual-to-legal 映射，该映射进入训练后也只改变 legal target 的边缘出现频率。损失中不存在：

\[
\mathcal L(m_j,l_i),
\]

也不存在同一 source 的 occupancy-before/after 输出关系。

P0-B/C 又表明：按预冻结的 handcrafted 与 decoder-joint 低维操作门槛，
现有证据不支持把 geometry-safe legal population 当作 factual-miss
population 的经验替代；这是否定当前 operational exchangeability 假设，
不是对所有可能高维表示空间的数学证明：

- handcrafted support coverage：\(2/32\)；
- decoder-joint support coverage：\(16/32\)；
- 两者都低于冻结门槛 \(29/32\)；
- factual/legal role 在冻结低维空间中仍具有明显可分性。

因此，旧路线的根问题不是 sampler 还不够复杂，而是：

```text
把 legal deletion endpoint 当成 factual-like iid training example
```

这一学习单位本身不成立。

## 3. 被否定或降级为诊断量的候选

### 3.1 条件特征证据增量

可以定义：

\[
\Delta_F(u)=
\operatorname{logit}P(Y_u=1\mid F_b,p_b,O_b=0)
-
\operatorname{logit}P(Y_u=1\mid p_b,O_b=0).
\]

它能够回答 frozen feature 是否包含 Base 分数尚未利用的目标证据，但如果直接据此训练：

\[
P(Y\mid F_b,p_b,O_b),
\]

方法很容易退化为普通二阶段 residual predictor。它更适合作为后续解释量或 matched control，不作为 CURE 的首要核心对象。

### 3.2 factual/legal 兼容量

角色 AUC、MMD、kNN coverage、density ratio 或描述符距离只能说明样本关系。若它们不进入 state、loss 或 inference 的实际计算图，就仍然是诊断量。

若兼容量最终只产生：

\[
q_i=P(\text{sample legal target }i),
\]

它仍是边缘重采样，不是配对学习。

### 3.3 feature transformation

当前没有真实的“同一目标自然漏检时 feature”作为反事实监督。同一个兼容分数可以任意导出 attenuation、replacement、interpolation 或 residualization，无法唯一确定操作。

所以 feature modification、context replacement 和 occupancy 修补继续保持未授权。

### 3.4 目标级效用或事件强度

目标级 Pd–FA 效用和 factual-miss event intensity 都是可研究对象，但它们需要新的候选生成、目标级目标函数或校准解释，容易把 CURE 改造成另一套后处理系统。本阶段不并行引入。

## 4. 新核心对象：coverage-conditioned completion operator

### 4.1 基础状态

对一个训练 source \(s\)，冻结 Base 提供：

\[
p_s,\qquad
F_s\in\mathbb R^{C\times h\times w}.
\]

由固定阈值得到输出分辨率 occupancy：

\[
O_s=\mathbf 1[p_s\ge\tau_b].
\]

decoder 实际消费的是 occupancy 的固定投影：

\[
\Pi(O_s)\in\{0,1\}^{1\times h\times w}.
\]

定义共享补全算子：

\[
Q_\theta:
\left(F_s,\Pi(O_s)\right)
\longmapsto
q_s\in[0,1]^{H\times W}.
\]

这里固定：

\[
q_s=\sigma(D_\theta(F_s,\Pi O_s)).
\]

即 \(q_s\) 是 occupancy hard mask 之前的概率分数。它不作已校准概率声明，也不是最终 residual mask。后续协议不得再在 logit difference 与 probability difference 之间按结果选择。

### 4.2 实例级绝对补全语义

令训练标签的目标实例集合为：

\[
\mathcal G_s=\{G_{s,1},\ldots,G_{s,n_s}\}.
\]

固定 matcher 在 occupancy \(O\) 下得到未匹配目标集合：

\[
\mathcal U_s(O).
\]

定义该 coverage state 的实例级可写补全目标：

\[
\mathcal R_{\mathcal G_s,V_s}(O)
=
\bigcup_{j\in\mathcal U_s(O)}
\left(G_{s,j}\cap V_s\cap\neg O\right).
\]

这个定义与“所有 GT 中未被 occupancy 覆盖的像素”不同。一个已经被 matcher 判定为覆盖的目标，不会因为少量轮廓像素未覆盖就被重新定义为 factual miss。

自然 factual/no-miss states 提供 \(Q\) 的零阶绝对锚：

\[
Q_\theta(F_s,\Pi O_s)
\approx
\mathcal R_{\mathcal G_s,V_s}(O_s).
\]

现有 factual supervision 是这一绝对锚的原子化实现；它本身仍只是 factual-only residual ERM，不是新机制。

## 5. 同源离散覆盖响应

### 5.1 coverage-before/after pair

对一个 geometry-safe、decoder-visible 的合法目标 \(g\)，令 \(C_{s,g}\) 是与其一一对应的完整预测组件。

定义：

\[
O^+_s=O_s,
\qquad
O^-_{s,g}=O_s\setminus C_{s,g}.
\]

现有 legal-deletion 契约已经保证：

1. 只删除一个完整预测组件；
2. 目标 \(g\) 成为唯一新增的 unmatched GT；
3. 其他 target-component identity 不变；
4. target lineage 一对一；
5. occupancy 修改在 feature grid 上可见。

但这五项仍不足以推出“标签差分只来自 \(g\)”。被删组件
\(C_{s,g}\) 可能同时覆盖一个删除前已经 unmatched 的其他 GT 的少量像素；
删除后这些像素也会变为可写区域，而 matcher identity 可以完全不变。

因此，本候选的 clean positive pair 还必须增加一个显式的非干扰条件：

\[
\left(C_{s,g}\cap V_s\right)
\cap
\left(
\bigcup_{j\in\mathcal U_s(O^+_s)}G_{s,j}
\right)
=\varnothing .
\]

也就是：在有效域内，被删组件不得覆盖任何删除前已经 unmatched 的
GT 实例。该条件必须由未来 pair catalog 逐对审计；不能从“唯一新增
unmatched target”间接推断。

两个端点必须共享逐值完全相同的 frozen feature：

\[
F^+_s=F^-_{s,g}=F_s.
\]

### 5.2 标签侧的精确状态转移

由 matcher-stability 条件可得：

\[
\mathcal U_s(O^-_{s,g})
=
\mathcal U_s(O^+_s)\cup\{g\}.
\]

先定义始终具有明确语义的实际标签增量：

\[
D_{s,g}
=
\mathcal R_{\mathcal G_s,V_s}(O^-_{s,g})
\setminus
\mathcal R_{\mathcal G_s,V_s}(O^+_s).
\]

如果没有新增的非干扰条件，\(D_{s,g}\) 除了目标 \(g\) 之外，还可能包含
已有 factual miss 被 \(C_{s,g}\) 原先遮住的像素。只有 clean positive
pair 同时满足该条件时，才有精确恒等式：

\[
D_{s,g}
=
A_{s,g},
\]

其中：

\[
A_{s,g}
=
V_s\cap G_{s,g}\cap\neg O^-_{s,g}.
\]

\(A_{s,g}\) 是目标 \(g\) 在删除后真正可写的实例级补全区域。对通过
非干扰审计的 clean pair，其他 factual misses、其他目标和背景在标签
增量中均为零；未通过者必须排除或归入单独的干扰审计，不能作为正 pair
静默使用。

### 5.3 模型侧的离散响应

定义：

\[
\Delta_g Q_\theta
=
Q_\theta(F_s,\Pi O^-_{s,g})
-
Q_\theta(F_s,\Pi O^+_s).
\]

新的唯一核心学习对象不是普通 \(Q\)，而是：

\[
\boxed{\Delta_g Q_\theta}.
\]

目标函数方程为：

\[
\Delta_g Q_\theta
\approx
D_{s,g}=A_{s,g},
\]

其中等号只对通过非干扰条件的 clean positive pair 成立，并要求在预先
冻结的非目标响应域中，删除 \(C_{s,g}\) 不产生无关增量。

这可以概括为：

> CURE 从自然状态的零阶绝对监督和同源 coverage-unmasking pair 的一阶离散监督中，学习同一个补全算子。

## 6. 为什么它与 v0.1/v0.2 不同

### 6.1 legal endpoint 不再冒充 factual miss

新定义不要求：

\[
(F_s,O^-_{s,g})
\sim
\text{factual-miss state}.
\]

\(O^-_{s,g}\) 只是对 decoder 可操纵输入的一个同源 coverage query。其用途是观察同一个 \(Q\) 对 \(O^+\rightarrow O^-\) 的响应，而不是建立 factual/legal 样本交换性。

所以旧 P0-B/C 仍然是旧路线的有效负结果，但不再是 \(\Delta_gQ\) 函数方程的共同支持前提。

P0-A1、lineage、删除合法性和 projected occupancy visibility 仍然是必要前提。

### 6.2 pairing 必须进入同一个梯度

未来若定义 pair objective，它必须直接消费：

\[
Q^-_\theta-Q^+_\theta.
\]

两端共享参数、都保留梯度，并进入同一个不可分解的 objective：

\[
\nabla_\theta\mathcal L_{\mathrm{pair}}
\propto
\nabla_\theta Q^-_\theta
-
\nabla_\theta Q^+_\theta.
\]

如果只是分别计算：

\[
\ell\!\left(
Q^-_\theta,\mathcal R_{\mathcal G,V}(O^-)
\right)
+
\ell\!\left(
Q^+_\theta,\mathcal R_{\mathcal G,V}(O^+)
\right),
\]

或其逐实例原子化等价形式，再把两个标量相加，pair identity 仍未形成学习
对象，方法会退化为独立状态 ERM。只有在删除前不存在其他 factual miss
的特例下，两个绝对标签才可简写为 \(A_g\) 与 \(0\)。

### 6.3 不是 factual-to-legal pairwise learning

这里的 pair 是：

```text
同一 legal target 的 occupancy before
    versus
同一 legal target 的 occupancy after
```

不是：

```text
一个 factual miss
    versus
一个 legal target
```

因此不能重新使用 conditional matching、miss-to-target transfer 或 factual/legal paired correction 的叙事。

## 7. 三个必须排除的退化

### 7.1 hard-mask 机械差分

当前 inference 使用：

\[
\bar Q(F,O)
=
(1-O)\odot Q(F,O).
\]

取任意 \(c\in(0,1)\)，令 raw operator 为有限 logit 可精确实现的常数
\(Q_c(F,\Pi O)=c\)。若先施加 hard mask，则：

\[
\bar Q_c(F,O^-)-\bar Q_c(F,O^+)
=
c\,(O^+-O^-)
=
c\,C_g.
\]

这个差分完全不使用 feature，也不学习 pre-mask coverage response。因此如果先
hard mask 再做差，原端在被删除组件上恒为零，正响应可以由推理掩码机械产生。

因此：

```text
Delta 必须在 hard mask 前计算；
inference hard mask 只能保留在最终输出路径。
```

### 7.2 occupancy-only 捷径

pre-mask occupancy-only 对照必须使用与候选相同的输入、输出尺度和有限
sigmoid 参数化：

\[
Q_O(F,\Pi O)
=
\sigma\!\left(D_O(\Pi O)\right).
\]

其中 \(D_O\) 与完整 decoder 做容量、输出网格和计算量匹配，但禁止读取
\(F\)。由于 \(\Pi\) 非单射且有限 logit 的 sigmoid 不精确产生二值补集，
这里不声称解析差分恒等于 \(C_g\)；它是必须实测的同接口捷径对照。若该
对照已经解释完整 operator response 或自然 factual gain，候选失败。

所以未来协议必须包含：

- occupancy-only operator；
- feature-only / feature-plus-fixed-mask operator；
- unmatched 或 false-positive component deletion 的零响应控制；每个 null pair
  都必须由实际 \(D_g=\varnothing\) 验证，不能只凭组件名称推定零响应；
- group-preserving target-label permutation；
- \(A_g\setminus C_g\) 与 \(C_g\setminus G_g\) 的分区记账。

若 occupancy-only control 能解释完整响应或自然 factual gain，候选失败。

### 7.3 普通独立状态 ERM

必须使用完全相同的 before/after states、forward 次数、曝光和计算量，比较：

```text
independent endpoint absolute ERM
    versus
coupled finite-difference objective
```

若二者具有等价梯度或同等机制结果，就不能把 \(\Delta_gQ\) 声明为新学习原理。

## 8. 为什么有限差分不能单独训练

对任何不依赖 occupancy 的函数 \(B(F)\)：

\[
\widetilde Q(F,O)=Q(F,O)+B(F)
\]

都有：

\[
\Delta_g\widetilde Q=\Delta_gQ.
\]

所以 \(\Delta_gQ\) 只能识别 coverage response，不能单独确定绝对 completion field。

自然 factual/no-miss 监督必须作为零阶锚保留。这不是第二个模块，而是有限差分对象可识别所必需的边界条件。

确切的零阶/一阶 objective、归一化与固定权重尚未冻结，因此当前仍不授权修改 [step.py](cure_lite/train/step.py) 或 [losses.py](cure_lite/losses.py)。

## 9. FA、retention 与性能作用路径

CURE-Lite 的目标不是删除 Base false positive。hard union 决定它只能增加 residual：

\[
\hat Y_{\mathrm{final}}
=
O_b\lor\hat Y_{\mathrm{res}}.
\]

所以核心目标应准确写成：

> 在保留 Base 已检目标的前提下，在固定总误报约束内找回事实漏检。

\(\Delta_gQ\) 对误报的直接作用路径来自：

1. selected target 之外的响应目标为零；
2. unmatched/false-positive component deletion 应产生零 GT completion；
3. inference 继续在 occupancy 内 hard mask；
4. final mask 继续 hard union，因此已检目标 retention 不因 residual 被删除。

这只是设计路径，不是性能证据。当前不能声明它已经降低 FA 或提高 Pd。

## 10. detector-independent 的准确含义

本候选只允许声明：

> 同一算法、同一状态契约和同一训练原则可以针对不同 frozen detector 重新训练。

当前不允许声明：

> 一个训练好的 \(Q\) 可以 zero-shot 迁移到任意 detector。

接口必须满足：

- 输入来自通用 \((p_b,F_b)\) cache；
- \(C,h,w\) 可以变化；
- 不依赖固定 64 channels、固定 1/4 stride、具体层名或 channel 语义；
- occupancy 投影规则对每个 adapter 相同；
- 只允许输入通道适配，不为不同 detector 重新设计机制。

跨 detector 是否成立，仍需在 CURE-Lite 冻结确认和 Full CURE 定义完成后实验验证。

## 11. 它是不是模块堆叠

在下列边界内，它不是模块堆叠：

```text
一个共享 Q
+ 一个同源 Delta functional constraint
+ 冻结的 factual/no-miss 绝对锚
```

推理仍然只有：

```text
Frozen Base
    -> one Q forward
    -> occupancy hard mask
    -> fixed threshold
    -> hard union
```

不新增：

- attention；
- 第二 decoder；
- feature editor；
- factual/legal matcher；
- candidate S；
- 独立 calibration network；
- 多阶段推理。

但必须诚实地说，\(\Delta_gQ\) 属于定向 coverage-equivariance / finite-difference consistency 的学习原则。是否具有 ICLR 所需的文献新颖性尚未检索，不能因数学形式清晰就预先声称新颖。

## 12. 下一协议必须冻结的内容

在任何实现前，必须另立一个非授权的 paired-objective protocol，冻结：

1. \(Q\) 固定为 `sigmoid(decoder logits)` 的 pre-mask probability score，不允许改用结果更好的另一尺度；
2. factual/no-miss 零阶锚的样本单位与归一化；
3. 不可分解的 pair objective；
4. \(A_g\) 和零响应域；
5. actual label increment \(D_g\)、clean-pair 非干扰条件及逐对排除清单；
6. 两端共同反传且均不 stop-gradient；
7. 由 decoder 结构预先确定的响应评估域；不得观察结果后调 dilation；
8. legal positive pairs 与 unmatched/false-positive null pairs；null 身份必须
   由有效域内实际 \(D_g=\varnothing\) 逐对验证；
9. pair exposure、forward 数和总计算量；
10. factual-only、independent endpoint、feature-only、occupancy-only 与 permutation controls；
11. group-held-out factual transfer 指标；
12. failure/stop rule；
13. inference graph 保持单次 \(Q\) forward、hard mask、threshold 与 hard union。

当前只有“起草上述只读协议”获得授权。实现、训练和新指标计算均未授权。

## 13. 预声明反证条件

出现任一情况，候选不得进入实现：

1. pair objective 可代数分解为两个独立 state loss；
2. pair identity 在非退化 toy case 中不改变 loss 或梯度；
3. 差分在 occupancy hard mask 后计算；
4. 任一端 stop-gradient 或作为固定 teacher；
5. occupancy-only control 可以匹配完整 operator response；
6. independent endpoint ERM 在等状态、等曝光和等计算下与 coupled objective 等价；
7. 只改善 legal deletion endpoint，不改善 group-held-out factual misses；
8. background 或其他目标的响应违反预冻结误报/保留约束；
9. 公式依赖特定 detector 的层名、固定 stride 或 channel 语义；
10. objective、权重或阈值由 \(D_V\) 或观察同一 \(D_R\) 结果后选择。
11. 任一 clean positive pair 的 \(C_g\) 在有效域内覆盖删除前已有 unmatched GT，或该检查未被 receipt 逐对绑定。

后续若进入开发训练，seed 42 和 43 必须分别证明 paired objective 相对 factual-only 与 matched independent-endpoint control 的严格增益；一个 seed 的提升不能抵消另一个 seed 的下降。

## 14. 到什么性能才能设计下一阶段

这里必须区分三个门槛。

### 14.1 允许实现 \(\Delta_gQ\)

不看性能，只看结构契约。只有 pair objective、双端梯度、hard-mask 位置、null pairs、occupancy-only control 和 matched independent-endpoint control 全部冻结并通过 toy/静态测试，才允许实现。

### 14.2 允许冻结 CURE-Lite 并进入确认

当前 \(D_V\) 有 170 个目标，1 个目标对应约 \(0.588\) 个 Pd 百分点。为了避免把一个目标的离散波动当作机制成立，seed 42 和 43 必须分别满足：

\[
\mathrm{TP}_{\Delta Q}
\ge
\max_c \mathrm{TP}_{c}+2,
\]

即：

\[
\mathrm{Pd}_{\Delta Q}
\ge
\max_c\mathrm{Pd}_{c}+0.0117647.
\]

同时，在固定的 23 个 anchor misses 上：

\[
\mathrm{Recovered}_{\Delta Q}
\ge
\max_c\mathrm{Recovered}_{c}+2,
\]

相当于 anchor recovery rate 至少增加：

\[
\frac{2}{23}\approx 8.70\%.
\]

比较对象 \(c\) 至少包括：

- Base@B；
- factual-only；
- factual-exposure-matched；
- geometry-matched independent-endpoint ERM；
- 旧 independent uniform legal。

并且每个 seed 都必须满足：

```text
retention = 1.0
pixel FA <= 1e-4
raw-background FA <= 1e-4
FP components / MP <= 100
budget_violation = false
```

历史 Stage-A 配置的 `minimum_retention=0.99` 只作为来源记录；这里的
`retention=1.0` 是为进入 Full CURE 预声明的更严格开发门槛，不能反写成
历史预算原本就是 1.0。

均值通过、只有一个 seed 通过，或每个 seed 只多找回 1 个目标，都不足以冻结 CURE-Lite。

### 14.3 允许设计 Full CURE

seed 42/43 通过只允许冻结方法并进入确认，不能直接开始 Full CURE。还必须完成：

1. 预冻结的额外 decoder-seed 集合，且每个确认 seed 的 paired effect 都严格为正；
2. grouped factual-recovery interval 排除零；
3. CURE-Lite 完全冻结后，在一个未使用划分上进行一次确认；
4. 所有误报和 retention 约束继续成立；
5. occupancy-only 与 independent-endpoint controls 均不能解释增益。

上述全部通过后，才允许设计 Full CURE。此时设计的是 CURE 的完整扩展，不是在 CURE-Lite 后继续叠加一个临时模块。

## 15. 当前创新性判断

与旧 CURE-Lite 相比，新候选具有更清晰的概念增量：

```text
旧：
legal endpoint 作为额外正样本

新：
legal before/after pair 作为同一 completion operator 的离散响应监督
```

其潜在价值在于：

- 将 predictor-induced supervision scarcity 转化为函数值与离散响应的联合学习问题；
- pair identity 真正进入梯度；
- 不依赖 factual/legal exchangeability；
- 一个算子、一条函数方程，不依赖网络模块堆叠；
- 可形成 detector-interface-level 的统一训练原则。

但当前仍存在三个决定性未知：

1. coverage response 是否会退化为 occupancy-only shortcut；
2. detected-target 上学习到的响应是否能改善自然 factual miss；
3. prior art 中是否已有等价的 targeted equivariance 或 derivative supervision。

因此当前最准确结论是：

```text
development potential = meaningful
current method readiness = insufficient
ICLR novelty = needs literature search
ICLR evidence = not established
```

## 16. 主线与下一步

总主线保持不变：

```text
完成 CURE-Lite 最小核心机制
  -> 冻结确认机制成立
  -> 设计 Full CURE
  -> 跨 IRSTD backbone 与三数据集验证
```

当前节点更新为：

```text
H0 hypothesis review
  -> core learning-object redefinition
  -> retain Delta_g Q as the only candidate
  -> paired-objective protocol frozen
  -> additive pair catalog / loss / train-step implementation
  -> unit and small-scale model validation
  -> only if passed, formal seed-42/43 training
```

现在仍然禁止：

- 修改 decoder 拓扑或覆盖旧 loss、旧 branch engine；
- 构造 transformation 或 S；
- 在独立实现通过结构、目录和小规模过拟合验证前运行正式训练；
- 修改正式校准或推理；
- 读取新的 \(D_V\) 或 \(D_T\)；
- 开始 Full CURE；
- 接入 DNANet、UIUNet、MSHNet 或 SCTransNet。

后续 paired-objective 已由
[独立协议](CURE_Lite_Paired_Objective_协议.md) 固定；本文件自身仍只记录
core-object redefinition，没有改写 v0.1、v0.2、P0-B/C 和 H0 的既有结论。
