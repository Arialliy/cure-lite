# CURE-Lite APTO v2：模型与代码设计

> 状态：版本化模型方案，尚未实现、尚未训练、尚未产生 v2 性能结果  
> 目标：把已经冻结的 Wave-A 负结果转化为一个最小、单机制、可直接编码的
> CURE-Lite v2  
> 方法名：**Anchored Paired Transition Objective（APTO，同源锚定转变目标）**

## 1. 版本边界

CURE-Lite v1 的 `paired_difference` 已经完成 seed 42/43 的正式
800-epoch、32,000-update Wave A，正式决定为：

```text
status = PERFORMANCE_FAIL
next_action = STOP_AND_PRESERVE_EVIDENCE
```

冻结结果为：

| seed | paired_difference | best matched comparator | margin |
| --- | --- | --- | --- |
| 42 | 147/170 TP，0/23 recovered | 154/170 TP，7/23 recovered | -7 TP，-7 recovered |
| 43 | 152/170 TP，5/23 recovered | 152/170 TP，5/23 recovered | 0 TP，0 recovered |

v2 不回写 v1 的 loss、代码、协议、目录或结论。v1 是完整负结果，不是等待
继续调参的活动版本。APTO 必须使用新的类型、loss、step、runner 和 artifact
schema，以便 v1 与 v2 可以独立重放和比较。

本方案只定义下一版模型与代码路线，没有读取新的 \(D_V\) 或 \(D_T\)，没有
运行训练、校准或推理，也没有产生任何新数值。

## 2. v1 的模型级缺口

令同一 source、同一冻结特征下两个 coverage endpoints 的 pre-mask residual
scores 为：

\[
q^+=\sigma(z^+),\qquad q^-=\sigma(z^-).
\]

v1 只对：

\[
\Delta q=q^- - q^+
\]

施加 paired 监督。只要概率仍位于合法范围内，pair-specific 共同偏移：

\[
\widetilde q^+=q^+ + b(F,x),\qquad
\widetilde q^-=q^- + b(F,x)
\]

不会改变：

\[
\widetilde q^- - \widetilde q^+=q^- - q^+.
\]

v1 的 factual-miss 与 factual-no-miss absolute anchors 只约束它们各自的
source、state 和 valid domain，不能严格识别每个 legal pair 上的共同偏移。
因此，尤其在大面积 zero-response 域，v1 只要求两端相等，却没有在同一 pair
内规定这个共同值的绝对语义。

Wave A 中 geometry-matched independent-endpoint control 在 seed 42 高于
`paired_difference`、在 seed 43 与其持平。这是“pair-local absolute
identifiability 可能不足”的设计依据，但不是已经建立的因果结论。APTO
因此是一个待实现、待检验的新版本，不能被描述为已经成功。

## 3. APTO 的唯一核心机制

APTO 把一个同源 coverage intervention 表示为两个不可缺少的坐标：

1. **baseline coordinate**：在干预前 endpoint 上识别本 pair 的绝对
   completion state；
2. **transition coordinate**：显式学习删除一个 coverage component 后的
   score change。

它不是两个网络模块，也不是两个独立 decoder。它是同一 shared operator 的
“基线 + 效应”可识别分解。

### 3.1 pair truth

对 clean positive pair \(p\)，定义：

\[
O_p^- \subset O_p^+,
\]

\[
R_p^+=\mathcal R_{\mathcal G,V}(O_p^+),\qquad
R_p^-=\mathcal R_{\mathcal G,V}(O_p^-),
\]

\[
R_p^+\subseteq R_p^-,
\qquad
D_p=R_p^-\setminus R_p^+.
\]

其中 \(D_p\) 必须与 `label_increment` 逐像素一致。

干预前 endpoint 的可写真背景为：

\[
B_p^+
=
V_p\cap\neg O_p^+\cap
\neg\left(\bigcup_jG_{p,j}\right).
\]

baseline absolute valid domain 固定为：

\[
M_p^+=R_p^+\cup B_p^+.
\]

不把 \(D_p\) 偷加为独立的 plus-endpoint target；\(D_p\) 的监督只通过
transition coordinate 进入。这保持“覆盖前为 baseline、删除后为新增
completion”的 paired 语义。

### 3.2 baseline coordinate

\[
\ell_{\mathrm{base}}(p)
=
\operatorname{CURELiteLoss}
\left(
z_p^+,
\mathbf 1_{R_p^+},
M_p^+
\right).
\]

这里严格复用已有 `CURELiteLoss`。若 \(R_p^+\) 为空，该 loss 按既有契约退化
为 negative-only absolute anchor；不得临时更换 loss family 或权重。

### 3.3 transition coordinate

transition coordinate 复用 v1 已实现的 pre-mask paired difference 形式：

\[
\ell_\Delta(p)
=
\frac{1}{2|D_p|}
\sum_{x\in D_p}
\left(
\frac{q_p^-(x)-q_p^+(x)-1}{2}
\right)^2
+
\frac{1}{2|V_p\setminus D_p|}
\sum_{x\in V_p\setminus D_p}
\left(
q_p^-(x)-q_p^+(x)
\right)^2.
\]

两个 endpoints 均保持在计算图内；禁止 detach、teacher endpoint、hard mask
后差分或两次不同参数状态的 forward。

### 3.4 APTO per-pair 与总目标

每个 pair 的唯一目标固定为：

\[
\boxed{
\ell_{\mathrm{APTO}}(p)
=
\frac12\ell_{\mathrm{base}}(p)
+
\frac12\ell_\Delta(p)
}
\]

batch reduction 为：

\[
\mathcal L_{\mathrm{APTO}}
=
\frac1{B_p}
\sum_{p=1}^{B_p}\ell_{\mathrm{APTO}}(p).
\]

完整训练目标固定为：

\[
\boxed{
\mathcal L_{\mathrm{v2}}
=
\mathcal L_{\mathrm{factual\ miss}}
+
\mathcal L_{\mathrm{factual\ no\ miss}}
+
\mathcal L_{\mathrm{APTO}}
}
\]

冻结权重为：

```text
APTO internal coordinates = 0.5 : 0.5
total branches = 1 : 1 : 1
```

这些权重不是搜索空间，不允许通过 \(D_R\)、\(D_V\) 或运行后的性能调节。
内部取算术平均使 baseline/effect 两个必要坐标共同形成一个 unit-weight pair
branch；外部继续保持两个 factual branches 与一个 pair branch。

### 3.5 与 independent endpoint 的本质区别

independent endpoint ERM 分别拟合：

\[
\frac12\ell_{\mathrm{abs}}(z^+,R^+,M^+)
+
\frac12\ell_{\mathrm{abs}}(z^-,R^-,M^-),
\]

因此：

\[
\frac{\partial^2\ell_{\mathrm{ind}}}
{\partial z^+\partial z^-}=0.
\]

APTO 的 transition coordinate 保留跨端点项，所以在非退化、非饱和位置：

\[
\frac{\partial^2\ell_{\mathrm{APTO}}}
{\partial z^+\partial z^-}
=
\frac12
\frac{\partial^2\ell_\Delta}
{\partial z^+\partial z^-}
\ne0.
\]

APTO 不是把两个 endpoints 重新独立训练；它只锚定 before/baseline endpoint，
after endpoint 必须通过同源 transition 学到。

## 4. null pair 首版策略

APTO v2 首版的 optimizer 只接受：

```text
clean_positive
```

以下 pair 继续保持只读诊断：

```text
component_null
identity_null
```

不得在首版 optimizer 中加入 component-null 或 identity-null consistency
loss。原因是 null regularization 是第二个独立机制：它同时改变训练人口和
negative-intervention 约束。若与 APTO 一起加入，正结果无法归因于
pair-local anchor 还是 null suppression。

首版仍必须在 `torch.no_grad()` 下报告 null pairs 的：

```text
mean absolute delta
maximum absolute delta
RMS delta
```

若 APTO 通过其他代码和 \(D_R\) 阶段但 null response 异常，只能另建后续
版本并做 matched ablation；不能把 null loss 静默塞回 APTO v2。

## 5. 模型与推理边界

APTO 不修改：

- frozen Base；
- `CURELiteDecoder` 拓扑、宽度、GroupNorm 或 feature tap；
- occupancy 构造、投影和 threshold；
- 单次 Base + 单次 residual decoder 的推理图；
- residual hard mask；
- hard union；
- calibration grid；
- detector-independent \((p_b,F_b)\) 边界。

训练期新增的 \(R^+\)、\(R^-\) 与 GT union 只用于构造 APTO supervision，
不能成为 decoder 输入或推理期依赖。

因此 APTO 仍满足：

```text
detector -> (p_b, F_b) -> occupancy O
         -> shared CURE-Lite decoder Q(F_b, O)
         -> residual hard mask
         -> O union residual
```

它不绑定 MSHNet，也不依赖 DNANet、UIUNet、SCTransNet 等 detector 的内部
层名或拓扑。detector independence 指同一算法与接口在每个 detector 上分别
重训，不是 zero-shot 跨 detector 迁移。

## 6. 版本化代码设计

v1 文件保持不变。v2 首批新增文件规划为：

### 6.1 数据类型

`cure_lite/paired_transition_types.py`

新增不可变 `PairedTransitionBatch`，组合现有 `PairBatch` 并携带：

```text
completion_plus
completion_minus
gt_union
```

必须验证：

```text
completion_plus subset completion_minus
completion_minus minus completion_plus == label_increment
completion_plus is writable under occupancy_plus
completion_minus is writable under occupancy_minus
both completion fields are inside gt_union and image_valid_mask
all tensors share batch, shape, dtype and device contracts
only clean_positive may enter the optimizer batch
```

保留 `completion_minus` 虽然 loss 可由 \(R^+\) 与 \(D\) 重建，是为了让
catalog truth、monotonicity 和 label increment 能够独立审计。

### 6.2 loss

`cure_lite/paired_transition_losses.py`

新增：

```python
build_plus_baseline_supervision(...)
AnchoredPairedTransitionLoss
```

`AnchoredPairedTransitionLoss` 的公共构造函数不得暴露 baseline/effect
weights；实现内部固定 `0.5/0.5`，并分别返回 baseline、transition 和 total
诊断量。

### 6.3 train step

`cure_lite/train/paired_transition_step.py`

新增：

```python
anchored_paired_transition_train_step(...)
```

每次 update 保持：

```text
factual-miss = 4 states
factual-no-miss = 4 states
paired batch = 2 clean pairs = 4 endpoint states
decoder states = 12
paired endpoint forward = exactly one 2B call
optimizer updates = 1
Base feature = detached
```

### 6.4 bounded 与 formal 路径

后续版本化文件：

```text
cure_lite/experiment/paired_transition_bounded.py
cure_lite/experiment/paired_transition_formal_training.py
cure_lite/experiment/paired_transition_formal_runner.py
tools/run_paired_transition_bounded.py
tools/run_paired_transition_formal.py
```

不得把 APTO 方法名追加到 v1 的 artifact schema 或
`FORMAL_TRAINING_METHODS`。v2 必须使用新 schema、config、fingerprint 和
create-only output directory。

## 7. 分阶段实施与停止规则

### Stage 1：核心代码与单元测试

只实现 type、supervision、loss 和 step。必须通过：

1. \(R^+\subseteq R^-\) 与 \(R^-\setminus R^+=D\)；
2. \(T^+=R^+\)、\(B^+\)、\(M^+\) 精确像素恒等式；
3. 相同 delta、不同共同偏移时 v1 loss 相同而 APTO loss 不同；
4. APTO mixed endpoint derivative 非零，independent endpoint 为零；
5. plus/minus 两端均获得有限、非零梯度；
6. 单次 2B forward、每步 12 states、一次 optimizer step；
7. null pair 被 optimizer 拒绝；
8. v1 全部回归测试保持通过。

任一失败即停在代码阶段，不运行数据或训练。

### Stage 2：toy overfit

使用人工构造的 00/01/11 completion transition，检查：

\[
R^+:(q^+,q^-)\rightarrow(1,1),
\]

\[
D:(q^+,q^-)\rightarrow(0,1),
\]

\[
B^+:(q^+,q^-)\rightarrow(0,0).
\]

同时检查 pair identity 改变会改变 joint loss/gradient，且两个 endpoint
保持 attached。toy 失败不得通过增加模块、调权重或引入 null loss修补。

### Stage 3：仅 \(D_R\) bounded execution

在正式 800 epoch 前，只使用 \(D_R\)：

- 构造真实 `PairedTransitionBatch`；
- 检查 completion monotonicity 和 exact masks；
- 执行有限步 overfit/gradient/score trajectory；
- 对齐 v1 difference-only、plus-anchor-only 和 independent-endpoint
  bounded controls；
- 不读取 \(D_V\) 或 \(D_T\)；
- 不调 APTO 权重、decoder、threshold 或推理。

若真实 pair 无法稳定降低 baseline 与 transition 两个坐标，停止并保留产物。

### Stage 4：只读 null 诊断

在同一冻结 decoder 上对 component-null 与 identity-null 执行
`torch.no_grad()` 诊断。null 不进入 optimizer，也不参与选择 APTO 权重。

null 诊断异常只形成限制或下一版本问题，不授权在当前版本增加第二个 loss。

### Stage 5：正式协议冻结

只有 Stage 1～4 通过后，才能另建正式 config，冻结：

- method/schema 名称；
- source hashes；
- pair population 与 schedule；
- decoder initialization；
- optimizer；
- 800 epochs、40 steps/epoch、32,000 updates；
- calibration/evaluation；
- matched comparators；
- 逐 seed 成功门槛；
- create-only artifacts 和停止规则。

本 proposal receipt 本身不授权正式训练。

### Stage 6：seed 42/43

正式 config 冻结后，才允许运行 seed 42/43。两个 seed 必须独立通过，不能
用均值或一个 seed 的提升抵消另一个 seed 的失败。

沿用的成功门槛意图为：

```text
每个 seed：
APTO true targets >= best frozen matched comparator + 2
APTO recovered fixed misses >= best frozen matched comparator + 2
retention、pixel FA、raw-background FA、FP-components/MP 与 budget 全部通过
```

但 v2 的 exact comparator set、result binding 和正式阈值必须在查看 v2
正式性能前另行冻结。本文件不把历史门槛直接变成一份可运行授权。

两个开发 seed 通过只授权冻结确认，不直接授权 Full CURE。

## 8. 必要 matched controls

v2 至少保留：

1. 冻结的 v1 `paired_difference`；
2. geometry-matched `independent_endpoint`；
3. `plus_anchor_only`：相同 pairs/forwards，只保留 baseline coordinate；
4. factual-only 与 factual-exposure-matched baselines；
5. null pair 只读诊断。

`plus_anchor_only` 用于判断提升是否只是增加 pair-local absolute data；
`independent_endpoint` 用于判断显式 transition coupling 是否优于两个
endpoint 的普通 ERM。

## 9. 禁止项

APTO v2 首版禁止：

- 修改或覆盖 v1 负结果；
- 读取新的 \(D_V\) 或 \(D_T\)；
- 在正式协议冻结前运行 seed 42/43；
- 修改 Base、decoder 拓扑、feature tap、calibration 或推理；
- 加 attention、Transformer、多尺度、第二 decoder 或 feature editor；
- 联合训练 Base；
- 搜索 `0.5/0.5` 或 `1:1:1` 权重；
- 让 component-null 或 identity-null 进入 optimizer；
- 恢复 factual-to-legal matching、S 或 marginal reweighting；
- 接入其他 IRSTD detector；
- 开始 Full CURE 或三数据集扩展；
- 把代码通过、toy 通过或 bounded 通过写成性能成功；
- 虚构 APTO 的 Pd、FA、recovered misses、稳定性或创新性结果。

## 10. 当前正式状态

```text
v1 Wave A                         = frozen PERFORMANCE_FAIL
APTO v2 model proposal            = specified
APTO v2 code                      = not implemented
APTO v2 unit/toy                  = not run
APTO v2 D_R bounded               = not run
APTO v2 null diagnostics          = not run
APTO v2 formal protocol           = not frozen
APTO v2 seed 42/43                = not run
new D_V access                    = false
D_T access                        = false
Base/decoder/inference modified   = false
Full CURE                         = not started
cross-detector                    = not started
paper core                        = not established
novelty against closest work      = unsearched
```

下一步是按 Stage 1 实现版本化的 APTO type、loss、step 和单元测试。只有代码、
toy、\(D_R\) bounded 与只读 null 诊断依次通过，才另行冻结正式 seed 42/43
协议。
