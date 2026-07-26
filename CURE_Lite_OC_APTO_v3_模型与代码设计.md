# CURE-Lite OC-APTO v3：模型与代码设计

> 状态：设计冻结，代码尚未实现  
> 数据边界：只允许使用 \(D_R\)  
> 方法名：Outcome-Complete Anchored Paired Transition Objective  
> 简称：OC-APTO

## 1. 主线不变

CURE 的研究主线仍然是：

```text
CURE-Lite 模型设计与验证
    -> 冻结确认
    -> Full CURE
    -> 接入不同 IRSTD detector
    -> NUAA / NUDT-SIRST / IRSTD-1K 跨模型、跨数据集验证
```

本版本只修改 CURE-Lite 的训练目标和训练样本组织，不修改：

- 任意 IRSTD detector 提供的冻结 \((p_b,F_b)\) 接口；
- `CURELiteDecoder` 拓扑和参数量；
- Base 冻结边界；
- 单次 residual decoder 推理；
- occupancy hard mask；
- Base 与 residual 的 hard-union 输出；
- 校准、阈值和正式评估流程。

因此 OC-APTO 不是在 MSHNet 上修改网络，也不是给 CURE-Lite 继续叠加
attention、Transformer、多尺度分支或第二个 decoder。

## 2. v3 要修复的模型缺口

APTO v2 已能同时学习 pair-local baseline 和正向 coverage transition，但
\(D_R\) 受限训练出现了一个明确的模型缺口：

- clean-positive transition 可以被学习；
- component-null 的全图平均响应很小；
- component-null 的局部最大响应仍可接近 1。

原因是只训练 \(D\neq\varnothing\) 的删除事件，会把“删除 occupancy
component”与“应当产生补全响应”混淆；同时，在 \(256\times256\) 全图上
计算 zero-response mean 会把几像素的局部错误响应稀释。

v3 不增加一个独立的 null branch。它把同一种受控删除的完整真实结果
纳入一个 transition population：

\[
D_p\neq\varnothing:
\quad \text{删除后出现新增 completion},
\]

\[
D_p=\varnothing:
\quad \text{删除后不应出现 completion response}.
\]

## 3. 统一的同源转变模型

对 pair \(p\)，冻结特征只保存一次：

\[
F_p^+=F_p^-=F_p.
\]

两个 endpoint 只改变 occupancy：

\[
O_p^-\subset O_p^+.
\]

同一个 shared decoder 以一次 \(2B\) batched forward 计算：

\[
z_p^+=g_\theta(F_p,O_p^+),\qquad
z_p^-=g_\theta(F_p,O_p^-),
\]

\[
q_p^\pm=\sigma(z_p^\pm),\qquad
\Delta q_p=q_p^--q_p^+.
\]

completion truth 为：

\[
D_p=R_p^-\setminus R_p^+.
\]

所有进入 v3 optimizer 的 pair 必须满足：

\[
R_p^+\subseteq R_p^-.
\]

其中：

- `clean_positive` 当且仅当 \(D_p\neq\varnothing\)；
- `component_null` 当且仅当 \(D_p=\varnothing\) 且
  \(R_p^+=R_p^-\)；
- `identity_null` 不进入 optimizer，因为其两个 decoder 输入完全相同，
  \(\Delta q=0\) 已由确定性 forward 机械保证。

## 4. 直接投影条件足迹

原始删除组件为：

\[
C_p=O_p^+\setminus O_p^-.
\]

decoder 实际消费的是投影到 feature grid 的 occupancy。定义：

\[
J_p^{\mathrm{raw}}
=
C_p
\cup
\operatorname{Lift}_{\mathrm{nearest}}
\left[
\Pi(O_p^+)\oplus\Pi(O_p^-)
\right],
\]

其中 \(\Pi\) 与 `CURELiteDecoder` 使用完全相同的 adaptive-max occupancy
projection，`Lift` 只把发生变化的 feature cells 映射回输出网格。在有效
图像域内使用：

\[
J_p=J_p^{\mathrm{raw}}\cap V_p.
\]

\(J_p\) 不是额外 decoder 输入，也不是 decoder 的完整理论响应域。两层卷积
和 GroupNorm 仍可能使响应超出 \(J_p\)。它只是冻结的
`direct projected conditioning footprint`，用于定义与 occupancy 条件变化
直接对齐的局部分层；其余位置仍由全局上下文分层约束。该定义不包含可搜索
半径或可调尺度。

## 5. 唯一的 outcome-complete response risk

定义三个语义分层：

\[
S_D=D_p\cap V_p,
\]

\[
S_J=J_p\setminus D_p,
\]

\[
S_G=V_p\setminus(D_p\cup J_p).
\]

正响应项为：

\[
\ell_D(p)
=
\operatorname{mean}_{x\in S_D}
\left(
\frac{\Delta q_p(x)-1}{2}
\right)^2.
\]

局部和上下文零响应项为：

\[
\ell_J(p)
=
\operatorname{mean}_{x\in S_J}
\Delta q_p(x)^2,
\]

\[
\ell_G(p)
=
\operatorname{mean}_{x\in S_G}
\Delta q_p(x)^2.
\]

先对所有非空 zero-response strata 做等权平均：

\[
\ell_0(p)
=
\operatorname{ActiveMean}
\left(
\ell_J(p),\ell_G(p)
\right).
\]

再定义统一 transition risk：

\[
\ell_{\mathrm{trans}}(p)
=
\begin{cases}
\frac12\ell_D(p)+\frac12\ell_0(p),
&D_p\neq\varnothing,\\[4pt]
\ell_0(p),
&D_p=\varnothing.
\end{cases}
\]

这不是按 `pair_kind` 选择两个损失。实现只根据 tensor truth 中 \(D\) 是否
为空激活对应的语义分层。

该分层有两个重要性质：

1. clean pair 仍保持 positive-response 与 zero-response 的
   \(0.5:0.5\) 平衡；
2. component-null 自动得到 intervention-local 与 global-context 的
   \(0.5:0.5\) 平衡，不再被 65,536 个全图像素稀释。

## 6. pair-local baseline coordinate

plus endpoint 的绝对锚定保持 APTO v2 定义：

\[
B_p^+
=
V_p\cap\neg O_p^+\cap
\neg\left(\bigcup_jG_{p,j}\right),
\]

\[
M_p^+=R_p^+\cup B_p^+,
\]

\[
\ell_{\mathrm{base}}(p)
=
\operatorname{CURELiteLoss}
\left(
z_p^+,\mathbf 1_{R_p^+},M_p^+
\right).
\]

OC-APTO 的单一 pair risk 为：

\[
\ell_{\mathrm{OC}}(p)
=
\frac12\ell_{\mathrm{base}}(p)
+\frac12\ell_{\mathrm{trans}}(p).
\]

batch 内对所有 outcome pairs 做普通 pair mean。完整训练目标仍然只有：

\[
\mathcal L
=
\mathcal L_{\mathrm{factual\ miss}}
+\mathcal L_{\mathrm{factual\ no\ miss}}
+\mathcal L_{\mathrm{OC}}.
\]

固定权重：

```text
baseline : transition = 0.5 : 0.5
factual-miss : factual-no-miss : OC-APTO = 1 : 1 : 1
```

不存在 `null_weight`、`local_weight`、`positive_weight` 或第四个训练
branch。

## 7. 训练 population 和计算预算

使用已经冻结的 \(D_R\) pair catalog：

```text
clean_positive = 206
component_null = 16
union = 222
```

对 222 个 pair 做 pair-level uniform deterministic sampling，不做
clean/null 1:1，不做 null oversampling，不根据 loss 或结果改权重。

每个 update 保持：

```text
factual-miss states       = 4
factual-no-miss states    = 4
outcome pairs             = 2
outcome endpoints         = 4
decoder forward calls     = 3
decoder states            = 12
backward / optimizer step = 1 / 1
```

正式 32,000 updates 共有 64,000 pair slots：

\[
64000=222\times288+64.
\]

因此每个 pair 必须暴露 288 或 289 次，任意两个 pair 的暴露差不超过 1。
同一 update 的两个 pair 必须来自不同 source image。

受限训练先使用完整 222-pair population：

```text
updates = 400
pair slots = 800
```

\[
800=222\times3+134,
\]

因此每个 pair 必须暴露 3 或 4 次。

## 8. 新增代码边界

v1 和 APTO v2 文件保持不变。v3 使用独立文件：

```text
cure_lite/paired_outcome_types.py
cure_lite/paired_outcome_losses.py
cure_lite/train/paired_outcome_step.py
cure_lite/experiment/paired_outcome_inputs.py
cure_lite/experiment/paired_outcome_schedule.py
cure_lite/experiment/paired_outcome_bounded.py
tests/test_paired_outcome_types.py
tests/test_paired_outcome_losses.py
tests/test_paired_outcome_step.py
tests/test_paired_outcome_toy_overfit.py
tests/test_paired_outcome_inputs.py
tests/test_paired_outcome_schedule.py
tests/test_paired_outcome_bounded.py
```

核心类型必须保存并验证：

```text
PairBatch
completion_plus
completion_minus
gt_union
intervention_footprint
```

训练 preflight 必须发生在 `decoder.train()`、`zero_grad()` 和任何 forward
之前。失败时 decoder mode、梯度、参数和 optimizer state 均不得改变。

完整 population 指纹只在训练前验证一次、训练结束后验证一次；禁止在每个
update 重算 222 个 pair 的全部哈希。

## 9. 代码与 toy 门禁

至少验证：

1. clean-only 输入与 APTO v2 的 baseline/positive-zero 主结构一致；
2. empty-\(D\) 不除零、不产生 NaN；
3. component-null 的 transition risk 等于局部/上下文 active mean；
4. clean/component mixed batch 等于逐 pair risk 的算术平均；
5. loss 不读取 `pair_kind` dispatch；
6. 两个 endpoint 都获得有限梯度，mixed endpoint derivative 非零；
7. 一次 \(2B\) endpoint forward；
8. 每步严格 3 forward、12 states、1 backward、1 update；
9. 拒绝 identity-null、clean-empty-\(D\)、component-nonempty-\(D\)；
10. 拒绝 component-null 的 \(R^+\neq R^-\)；
11. 拒绝 \(R^+\not\subseteq R^-\)；
12. 干预足迹必须与 occupancy projection 精确一致；
13. GT、completion 和 footprint 不得进入 decoder；
14. 构造后 tensor 篡改必须在零副作用 preflight 被拒绝；
15. v1、control 和 APTO v2 回归全部通过。

toy 必须同时学会：

- clean \(D\) 上 \(\Delta q\rightarrow1\)；
- clean local/global zero strata 上 \(\Delta q\rightarrow0\)；
- component-null footprint 上 \(\Delta q\rightarrow0\)；
- component-null context 上 \(\Delta q\rightarrow0\)；
- plus baseline 的 target/background 绝对语义。

## 10. \(D_R\) bounded 计算门禁

这些门禁只判断模型代码能否学习冻结的 \(D_R\) 状态，不是检测性能：

```text
all 222 outcome pairs bound                         = true
all pairs exposed 3 or 4 times                      = true
all updates finite and non-zero-gradient            = true
decoder parameters changed                          = true
factual-miss anchor final / initial                  <= 0.75
factual-no-miss anchor final / initial               <= 0.75
plus baseline final / initial                       <= 0.75
clean transition final / initial                    <= 0.50
clean mean delta on D                               >= 0.50
clean pairs with mean delta on D >= 0.25            >= 75%
clean macro mean abs delta on zero strata           <= 0.05
component-null macro mean abs delta on footprint    <= 0.05
component-null global max abs delta on footprint    <= 0.25
component-null macro mean abs delta on context      <= 0.05
identity-null maximum abs delta                     <= 1e-7
```

其中 `macro mean` 表示先在每个 pair 的对应分层内求均值，再对 pair 求均值；
`global max` 表示对全部 component-null pairs 及其 footprint pixels 取唯一最大值，
不能用 pair 均值掩盖单个局部饱和响应。

只有上述门禁通过，才实现并冻结 seed 42/43 的 800-epoch 正式 runner。

## 11. 禁止事项

本版本禁止：

- 修改 decoder、Base、inference 或 hard-union；
- 增加 attention、Transformer、多尺度、第二 decoder；
- 单独增加 `null loss` 或第四个 optimizer branch；
- 搜索 clean/null 比例和任何 loss 权重；
- 读取 \(D_V\) 或 \(D_T\) 来设计 v3；
- 在受限训练失败后延长 update 数或放宽门槛；
- 在 CURE-Lite 冻结确认前开始 Full CURE；
- 在 CURE-Lite 冻结确认前接入其他 IRSTD detector。

## 12. 当前状态

```text
OC-APTO v3 proposal           = specified
v1 / APTO v2                  = frozen
v3 code                       = not implemented
v3 unit / toy                 = not run
v3 D_R 222-pair bounded       = not run
v3 formal seed 42/43          = not run
model success                 = not established
```

下一步只实现 v3 的 type、loss、train step 和对应测试。
