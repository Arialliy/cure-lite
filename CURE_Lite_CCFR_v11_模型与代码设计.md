# CURE-Lite CCFR-v11 模型与代码设计

## 1. 当前阶段与唯一目标

本版本只解决 PECO-v10 暴露确认所揭示的模型状态瓶颈，不修改论文主线：

```text
CURE-Lite
  -> 冻结确认
  -> Full CURE
  -> 跨 IRSTD detector / 三数据集验证
```

PECO-v10 的权威 dataset-free exposure-matched 结果为有效负结果：

- 8 个组中 0 个通过；
- population objective 为 0.742639，冻结门槛为小于 0.1；
- factual-miss target minimum 为 0.903703；
- factual-miss background maximum 为 0.492539；
- factual-no-miss maximum 为 0.166081。

该结果说明 PECO 的双端梯度存在，但原 CC-SEA 状态方程没有同时形成可靠的
目标恢复与背景保持。v11 因此只改变 decoder 内部的状态构造，PECO loss、
4/4/2 batch、训练 step、优化器、数据和推理均保持冻结。

v11 的唯一候选为：

> **Coverage-Conditioned Feature Release（CCFR）**：coverage 不再只在
> feature head 之后解析地缩放 evidence，而是在共享 trunk 内部调制
> detector feature 的可见状态，使 feature 与 coverage 在可学习的非线性
> 表示中发生交互。

当前证据状态：

```text
method_id = ccfr_v11
design_status = FROZEN
implementation_status = IMPLEMENTED_STATIC_VALIDATED
dataset_free_gate = NOT_RUN
D_R_access = NOT_AUTHORIZED
formal800 = NOT_AUTHORIZED
ICLR_novelty = NEEDS_LITERATURE_SEARCH
```

## 2. 为什么不实现原拟 TCF

原拟 TCF 可写为：

\[
z(F,O)=T(F)-S(F)b(O).
\]

它具有清晰的删除单调性，但仍属于：

\[
A(F)+B(F)g(O)
\]

这一可分离后端调制族。历史 v4、v7 和 v8 已经分别检验了 inverse-count、
log-burden crossing 和 coverage-conditioned budget。只更换后端解析函数，
即使参数解释更清楚，也不足以证明模型获得了新的联合状态能力。

因此本版本不增加第三个 head，也不继续更换 activation、surrogate、
burden、loss 权重或 phase allocator，而是改变 coverage 进入现有网络计算
的位置。

## 3. CCFR 的状态方程

### 3.1 输入契约

冻结 Base 对每张图像只提供：

\[
p_b,\qquad F_b\in\mathbb R^{C_f\times h\times w}.
\]

occupancy 仍由冻结阈值得到：

\[
O=\mathbf 1[p_b\ge\tau_b].
\]

decoder 的公开输入仍然只有：

```text
(detached frozen feature F_b, hard occupancy O)
```

pair kind、样本身份、GT、目标身份和另一端标签均不进入 decoder。

### 3.2 固定 coverage release field

先用既有 adaptive-max 投影得到 feature-grid occupancy：

\[
\Pi(O)\in\{0,1\}^{1\times h\times w}.
\]

再用冻结的 \(3\times3\) 全一卷积核计算局部 coverage count：

\[
C(O)=\mathbf 1_{3\times3} * \Pi(O).
\]

固定 feature release field 为：

\[
V(O)=\frac{1}{1+C(O)}.
\]

这里没有温度、阈值、可学习系数或额外超参数。

### 3.3 coverage 进入共享 trunk 内部

先对 detached feature 执行原有 stem 与 GroupNorm：

\[
h_0(F)=\operatorname{SiLU}
\left(
\operatorname{GN}
\left(
W_sF
\right)
\right).
\]

CCFR 的唯一新计算是：

\[
\widetilde h_0(F,O)=h_0(F)\odot V(O).
\]

随后原有 depthwise residual trunk 直接处理该联合状态：

\[
h(F,O)
=
\widetilde h_0
+
\frac12 W_p
\operatorname{SiLU}
\left(
\operatorname{GN}
\left(
W_d\widetilde h_0
\right)
\right).
\]

因此：

\[
h(F,O)\ne h_F(F)+h_O(O)
\]

通常也不能化成 head 输出后的 \(A(F)+B(F)g(O)\)。coverage 改变的是
learned spatial transform 的输入，而不是已经生成的 logit/evidence 的
末端倍率。

将 release 放在 stem normalization 之后是固定设计：若直接在 stem 前对
全部 feature 做统一缩放，GroupNorm 可能消除这一状态差异；放在
normalization 后既保留零 feature 不响应的性质，也使局部 coverage pattern
真实进入 depthwise spatial computation。

### 3.4 冻结的输出头

两个已有 head、PixelShuffle 和全局 baseline scalar 完全保留。为避免同时
更改状态交互与输出学习规律，v11 保留 CC-SEA-v8 的 common-mode crossing、
phase softmax 与质量守恒，只将输出端 occupancy burden 固定为零，因为
coverage 已经唯一地进入 trunk。

\[
B_\theta(F,O)
=
-\operatorname{softplus}
\left(
\beta+B_h(h(F,O))
\right),
\]

\[
r_{k,j}=E_h(h(F,O))_{k,j},
\qquad
\mu_k=\frac{1}{s^2}\sum_jr_{k,j},
\]

\[
M_k=f_{\mathrm{CR}}(\mu_k),
\qquad
\alpha_{k,j}
=
\operatorname{softmax}_j(r_{k,j}-\mu_k),
\]

\[
e_{k,j}=M_k\alpha_{k,j},
\qquad
\sum_j e_{k,j}=M_k,
\]

\[
z_\theta(F,O)
=
B_\theta(F,O)
+
\operatorname{PixelShuffle}(e).
\]

\(f_{\mathrm{CR}}\) 与 v8 使用同一个 crossing-recoverable primitive。
v11 不再在 \(M_k\) 中减去 occupancy burden，也不在输出端再次乘
\(V(O)\)。因此 v8 的 phase coordinate、allocation 和 conservation
完全保留，唯一状态差异是 coverage 从 output budget 移入 trunk。

最终 raw residual probability 为：

\[
q_\theta(F,O)=\sigma(z_\theta(F,O)).
\]

训练 pair 的两个 endpoint 仍在 hard mask 之前计算：

\[
z^+=z_\theta(F,O^+),\qquad
z^-=z_\theta(F,O^-).
\]

## 4. 网络结构与参数

Reference 配置 \(C_f=64,s=4\) 保持：

```text
detached F_b
  -> 1x1 stem
  -> GroupNorm + SiLU
  -> fixed coverage feature release
  -> 3x3 depthwise
  -> GroupNorm + SiLU
  -> 1x1 pointwise residual
  -> two existing phase heads
  -> PixelShuffle
  -> optional frozen bilinear fallback
  -> one residual logit map
```

不增加任何：

- convolution；
- attention；
- transformer；
- decoder；
- head；
- parameter；
- persistent buffer；
- inference branch。

必须保持与 v4 完全相同的 7 个 state-dict key、6 个参数 tensor 和参数量：

\[
32C_f+32\cdot3\cdot3+32^2+2\cdot32s^2+1.
\]

因此 \(C_f=64,s=4\) 时仍为 4,385 个可训练参数；dataset-free
\(C_f=8,s=4\) 时仍为 2,593 个。

## 5. 为什么它仍适用于任意 IRSTD detector

CCFR 不读取任何 detector 内部模块名称，也不要求 MSHNet 特有结构。
任意 IRSTD detector 只需通过冻结 adapter 提供：

```text
base probability p_b
one frozen feature F_b
feature_channels
feature_stride
base identity receipt
```

当前阶段只使用 Reference Base 建立 CURE-Lite，不接入 MSHNet、
DNANet、UIUNet 或 SCTransNet。跨 detector 是 CURE-Lite 冻结确认后的
独立验证阶段，不是当前网络的一部分。

## 6. 训练保持冻结

一次 update 仍为：

```text
4 factual-miss states
+ 4 factual-no-miss states
+ 2 same-source occupancy pairs
= 12 decoder states
```

三次 decoder call 的 batch size 均为 4；pair endpoint 使用一次 \(2B\)
batched forward。Base feature 全部 detach，只有 6 个 decoder parameter
tensor 被优化。

第一轮受控比较继续使用 PECO-v10 的 loss、Adam、学习率、400 updates、
800 pair slots 和全部门槛。这样与 v10 的唯一变化是：

```text
post-head coverage-conditioned budget
        ->
in-trunk joint feature-coverage state
```

旧 v10 的 222-role tensors、schedule 和结果均已被查看，所以再次运行只能
记为 `seen exposure-matched stress replay`，不能称为独立确认。新的 holdout
已在查看 CCFR 完整 development 结果前冻结为 5×5 feature-grid、222-role、
新 tensor 与新 schedule 的确认协议；其 receipt 位于：

- `protocols/IRSTD-1K/coverage_conditioned_feature_release_v11/`
  `exposure_holdout_design_receipt.json`

不得同时修改 loss、训练步数、role exposure 或门槛。

## 7. 推理流程完全不变

```text
image
  -> one frozen Base forward
  -> (p_b, F_b)
  -> O = p_b >= tau_b
  -> one CCFR decoder forward
  -> sigmoid residual probability
  -> residual is zeroed on O
  -> frozen residual threshold
  -> hard union with Base mask
```

训练时的 pair 只用于学习，不增加测试时的第二 endpoint 或额外 forward。

## 8. 实现前冻结的代码门槛

### 8.1 拓扑与接口

1. v4 与 v11 的 module 类型序列、state-dict key/shape 和初值逐值一致；
2. strict state-dict load 可双向执行，但 receipt 必须绑定不同 method ID；
3. 只有 \((F,O)\) 两个前向输入；
4. feature 必须 detach；
5. 不修改已有 model、loss、train step、root export 或 v4-v10 文件；
6. 现有 `CURELiteFactorizedModel` 直接接收 v11 decoder；
7. 推理仍为一次 Base 和一次 decoder。

### 8.2 状态构造

1. \(O=0\Rightarrow V=1\)；
2. \(F=0\) 时所有 occupancy 得到完全相同的 logits；
3. 相同 projected count 必须产生相同 release field；
4. count support 外的 release field 不变；
5. `released_stem = stem_feature * release_field` 逐项成立；
6. endpoint feature 共享且只由 occupancy 产生联合状态差异；
7. resize fallback 不改变输出 shape、dtype 和 device；
8. 全部字段有限。

### 8.3 paired 与梯度

1. 两端在一次 \(2B\) forward 中计算；
2. plus/minus logits 均连接同一 backward graph；
3. 6 个参数 tensor 均获得有限非零梯度；
4. frozen feature 不获得梯度；
5. PECO both-high、both-low、wrong-direction 反例继续保留纠正梯度；
6. pair kind 不进入 decoder 或 loss；
7. identical-input conflicting-outcome 继续作为不可满足负对照，不允许按
   role 分流。

### 8.4 小目标与表示

dataset-free case 至少覆盖：

- 1、2、3 像素 response；
- response 在删除组件内；
- response 在组件外但位于 \(3\times3\) count support 内；
- 同一 feature cell 的多个 subpixel phase；
- component-null；
- count 未改变的 null；
- 零 feature 的 occupancy-only control。

## 9. 串行验证顺序

```text
design freeze
  -> additive config / decoder
  -> config/operator/decoder/model unit tests
  -> paired 2B / gradient / null/control tests
  -> old v4-v10 regression
  -> frozen 6-case development regression
  -> optional old 222-role seen stress replay (not confirmation)
  -> pre-frozen new 5x5 / 222-role exposure holdout confirmation
  -> only if all dataset-free gates pass: D_R bounded
  -> only if real gate passes: 32,000-step exposure replay
  -> seed 42/43 formal800
  -> frozen confirmation
  -> Full CURE
  -> cross detector / NUAA-SIRST, NUDT-SIRST, IRSTD-1K
```

三大数据集不能用于当前候选选择。它们只用于冻结后的泛化验证，避免模型结构
根据三个测试环境反复调整。

## 10. 可证伪点

CCFR 不是预设成功。出现以下任一结果即停在相应阶段：

1. coverage modulation 被 normalization 或现有 head 实际消除；
2. paired endpoint 没有形成可测的 joint-state difference；
3. feature 为空时 occupancy 仍可单独产生目标响应；
4. response 只增加但 factual/no-miss 背景也同步升高；
5. component-null 或邻近区域泄漏超过冻结门槛；
6. 222-role population 仍有任一组失败；
7. 真实 \(D_R\) 门槛未全部通过。

失败时保留证据，不通过增加层数、调整 loss、放宽阈值或扩大数据集来掩盖。

## 11. 创新边界

当前可诚实提出的方法创新候选是：

> 把 frozen detector 的 coverage state 从 residual head 之后的解析调制变量，
> 转化为共享 completion representation 内部的 feature-release 条件，使
> 同源 occupancy intervention 直接改变可学习表示，同时保持单 decoder、
> 固定参数量和零额外推理分支。

这是一条统一机制，不是多个模块的组合。但代码可行不等于 ICLR 新颖性成立。
在正式论文主张前仍需完成：

- 最接近方法的公开文献检索；
- dataset-free 与真实 \(D_R\) 因果比较；
- seed 42/43 的逐种子正式结果；
- 冻结后的三数据集和跨 detector 泛化。
