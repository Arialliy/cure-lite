# CURE-Lite D-SVEF v5：模型与代码设计

> 状态：设计冻结，代码尚未实现  
> 数据边界：只允许使用 \(D_R\) 完成结构审计与 400-update bounded model-code gate  
> 方法名：Directed Subpixel Vacancy–Evidence Factorization  
> 简称：D-SVEF

## 1. 研究主线与阶段边界

```text
CURE-Lite 模型设计与验证
    -> 冻结确认
    -> Full CURE
    -> 接入不同 IRSTD detector
    -> NUAA / NUDT-SIRST / IRSTD-1K 跨模型、跨数据集验证
```

D-SVEF v5 仍是 CURE-Lite 的模型代码候选，不是 Full CURE。当前阶段禁止：

- 读取 \(D_V\) 或 \(D_T\)；
- 进行校准、推理或 Pd/FA 性能比较；
- 运行 formal 800 epoch；
- 修改 Base/backbone；
- 接入其他 IRSTD detector；
- 修改 OC-APTO v3 的 loss、pair population、训练 step 或 schedule；
- 修改 SVEF v4 的源码、协议和已封存结果；
- 在同一 v5 上自动重跑真实 bounded 实验。

v5 只新增文件。历史 `CURELiteDecoder`、SVEF v4 decoder、单次 Base
提取、occupancy hard mask 和 hard-union 推理图全部保持不变。

## 2. v4 已经证明与尚未证明的内容

SVEF v4 已完成唯一一次真实 \(D_R\) 400-update bounded 运行。其执行证据为：

- 918 项仓库测试全部通过；
- 222/222 outcome pairs 完整绑定并评估；
- 400/400 backward 与 optimizer step 完成；
- 4,385 个参数发生更新；
- 所有梯度有限且非零；
- 全部结构门禁通过；
- 12 项计算门禁中 5 项通过、7 项失败；
- clean 联合选择性仅为 \(1/206\)。

关键失败表现为：

\[
\overline{\Delta_D}=0.181157<0.5,
\]

\[
\operatorname{Frac}
\left[
\Delta_D\geq0.25
\land
\Delta_H\leq0.05
\right]
=
\frac1{206}.
\]

同时：

\[
\overline{|\Delta_H|}=0.069588>0.05,
\]

component-null footprint 的平均响应为 0.121929，最大响应为 0.928389。

因此 v4 证明：

1. 子像素输出、固定 vacancy path、单调删除响应和训练执行均可成立；
2. 模型确实学习了 factual anchors 与 plus baseline；
3. 当前证据参数化没有形成所需的目标内响应与局部零响应联合选择性。

这些结果不是检测性能结论，也没有否定 CURE 总方向。

## 3. v5 的唯一研究假设

v4 使用：

\[
E_{\mathrm{v4}}(r)
=
\operatorname{softplus}(r^2)-\log2.
\]

该映射满足非负和零锚定，但具有偶对称性：

\[
E_{\mathrm{v4}}(-r)=E_{\mathrm{v4}}(r).
\]

因此 feature head 的正、负方向都会产生正 evidence。若某一方向表示目标，
另一方向表示背景或局部排除证据，平方会删除这种方向信息；背景只能把
raw evidence 精确压到零才能关闭响应。

v5 的唯一修改是将 evidence 改为：

\[
E_{\mathrm{v5}}(r)
=
\operatorname{ReLU}
\left[
\operatorname{softplus}(r)-\log2
\right].
\]

于是：

\[
r\leq0
\Rightarrow
E_{\mathrm{v5}}(r)=0,
\]

\[
r>0
\Rightarrow
E_{\mathrm{v5}}(r)>0,
\]

\[
E_{\mathrm{v5}}(0)=0,\qquad E_{\mathrm{v5}}(r)\geq0.
\]

负半轴成为精确的“不响应”状态，正半轴表示可被 vacancy gate 释放的目标
证据。该 directed activation 还同时固定两项与同一函数不可分割的性质。

对弱正证据 \(r\rightarrow0^+\)：

\[
E_{\mathrm{v5}}(r)
=
\frac12r+O(r^2),
\]

而 v4 为：

\[
E_{\mathrm{v4}}(r)
=
\frac12r^2+O(r^4).
\]

因此 v5 在正半轴原点附近具有一阶梯度，避免弱目标 evidence 被平方压缩。

对强正证据 \(r\rightarrow+\infty\)：

\[
E_{\mathrm{v5}}(r)
=
r-\log2+o(1),
\]

而 v4 近似按 \(r^2\) 增长。v5 的线性尾部用于限制少数
component-null footprint 的极端响应。负半轴关闭、弱正证据一阶响应与强
正证据线性尾部共同定义同一个无参数 directed activation，不是三个附加
模块，也不是三个可独立调节的超参数。

这些性质只解释为什么 v5 值得验证；在 bounded gate 通过前，不能把它们
写成已经成立的因果结论。

## 4. 完整网络公式

v5 保持同一因子分解：

\[
z_\theta(F,O)
=
B_\theta(F)
+
E_\theta^+(F)\odot A(O).
\]

其中：

\[
B_\theta(F)
=
-\operatorname{softplus}
\left(
\beta_{\mathrm{raw}}+\widetilde b(F)
\right)
<0,
\]

\[
E_\theta^+(F)
=
\operatorname{ReLU}
\left[
\operatorname{softplus}
\left(
\widetilde e(F)
\right)
-\log2
\right],
\]

\[
A(O)
=
\operatorname{Nearest}_{H,W}
\left[
\frac{1}{
1+
\mathbf1_{3\times3}
*
\operatorname{ProjectMax}_{h,w}(O)}
\right].
\]

共享 trunk、两个 phase head 与 PixelShuffle 保持 v4 不变：

\[
T_0
=
\operatorname{SiLU}
\left[
\operatorname{GN}
\left(
W_s*F
\right)
\right],
\]

\[
T
=
T_0+
\frac12W_p*
\operatorname{SiLU}
\left[
\operatorname{GN}
\left(
W_d*T_0
\right)
\right],
\]

\[
\widetilde b
=
\operatorname{PixelShuffle}_s(W_B*T),
\qquad
\widetilde e
=
\operatorname{PixelShuffle}_s(W_E*T).
\]

当前 reference instance 继续固定：

```text
feature_channels = 64
feature_stride = 4
width = 32
groups = 8
trunk_residual_scale = 0.5
baseline_probability = 0.1
vacancy_kernel_size = 3
```

所有卷积继续 `bias=False`，GroupNorm 继续 `affine=False`，参数量公式保持：

\[
N_{\mathrm{param}}
=
32C+1313+64s^2.
\]

在 \(C=64,s=4\) 时：

\[
N_{\mathrm{param}}=4385.
\]

## 5. 初始化与梯度契约

初始化保持 v4 不变：

- trunk：Kaiming normal；
- baseline head：全零；
- evidence head：Xavier normal，gain=0.25；
- \(\beta_{\mathrm{raw}}\)：使零特征 baseline probability 为 0.1；
- vacancy kernel：固定 \(3\times3\) 全一 buffer。

对 \(r>0\)：

\[
\frac{\partial E_{\mathrm{v5}}}{\partial r}
=
\sigma(r)>0.
\]

对 \(r<0\)，evidence 与局部梯度均为零。evidence head 不能零初始化，否则
所有 phase 都落在 ReLU 边界而无法形成所需的初始方向分工。

必须用单元测试固定：

1. 负 raw evidence 输出精确零；
2. 零 raw evidence 输出精确零；
3. 正 raw evidence 严格为正；
4. 正 raw evidence 的梯度严格为正；
5. \(E(-r)\neq E(r)\)；
6. 所有 evidence 均有限且非负。

## 6. 保持不变的结构性质

若：

\[
O^-\subseteq O^+,
\]

则 v4 的固定 inverse-count vacancy 仍满足：

\[
A(O^-)\geq A(O^+).
\]

由于 \(E_{\mathrm{v5}}\geq0\)：

\[
z(F,O^-)-z(F,O^+)
=
E_{\mathrm{v5}}(F)
\odot
\left[A(O^-)-A(O^+)\right]
\geq0.
\]

因此以下性质必须继续精确成立：

- zero-feature occupancy invariance；
- identity endpoint exactness；
- deletion logit monotonicity；
- deletion probability monotonicity；
- vacancy gate 变化支持域之外 logit/probability delta 精确为零；
- 206/206 clean pairs 的全部 \(D\) 像素具有正 vacancy 路径；
- 16/16 factual anchors 的全部目标像素具有正 vacancy 路径；
- 原生 \(64\rightarrow256\) PixelShuffle 路径不触发 field resize；
- 单 shared decoder；
- 单次推理；
- hard occupancy mask 与 hard union 不变；
- 不能删除 Base false alarms。

## 7. 冻结训练对象

v5 继续使用：

- factual-miss anchors：16；
- factual-no-miss anchors：16；
- clean-positive pairs：206；
- component-null pairs：16；
- identity-null diagnostic pairs：16，optimizer exposure 为 0；
- `OutcomeCompleteTransitionLoss` v3；
- `outcome_complete_train_step` v3；
- seed 42；
- Adam，learning rate \(10^{-3}\)，weight decay 0；
- 每步 4 factual-miss、4 factual-no-miss、2 outcome pairs；
- 每步 3 次 decoder forward、12 个 decoder states；
- 10 epochs × 40 steps = 400 updates；
- 800 个 outcome pair slots；
- evaluation chunk size 32；
- 相同 deterministic factual/outcome schedules；
- 相同 \(D_R\) pair catalog、prepared catalog、anchor population 和
  materializer。

不允许根据 v4 结果修改 loss 权重、pair-kind 比例、学习率、训练步数或门槛。

## 8. v5 代码边界

只允许新增：

```text
cure_lite/directed_factorized_config.py
cure_lite/directed_factorized_decoder.py
cure_lite/experiment/directed_factorized_outcome_bounded.py
tools/run_directed_factorized_outcome_bounded.py
tests/test_directed_factorized_config.py
tests/test_directed_factorized_decoder.py
tests/test_directed_factorized_model.py
tests/test_directed_factorized_outcome_bounded.py
tests/test_directed_factorized_outcome_toy_overfit.py
tests/test_run_directed_factorized_outcome_bounded_cli.py
protocols/IRSTD-1K/directed_subpixel_vacancy_evidence_factorization_v5/*
```

v5 decoder 可以继承 v4 decoder 的参数拓扑、输入校验和 vacancy field，但必须
拥有独立 class identity、独立 state fingerprint domain 和独立配置类型。
现有 `CURELiteFactorizedModel` 接受该 decoder 子类，因此直接原样复用；
禁止增加一个没有计算差异的 v5 model wrapper。
禁止修改 v4 源文件来增加 factory 参数或 version branch。

## 9. 代码与 toy 门禁

至少验证：

1. 配置只接受冻结 topology 和
   `one_sided_zero_anchored_softplus_relu_v1`；
2. reference instance 参数量严格为 4,385；
3. v4 与 v5 在相同 raw evidence 下体现偶对称/单侧差异；
4. feature 在 decoder 内 detach；
5. 双 endpoint 梯度进入同一 v5 decoder；
6. 2B endpoint batched-forward 与分别 forward 数值等价；
7. zero-feature、identity、gate-support、单调性全部精确；
8. 16 个 phase channel 具有子像素自由度；
9. model 继续冻结 Base，单次提取并 hard-union；
10. clean toy 能形成高 \(D\)、低 \(H\)；
11. component-null toy footprint 保持低响应；
12. v1/v2/v3/v4 全部旧测试继续通过；
13. v4 实现与结果哈希保持不变。

## 10. 真实 \(D_R\) bounded 门禁

结构门禁与 v4 完全相同。计算门禁继续固定为：

\[
\frac{L_{\mathrm{factual\ miss}}^{\mathrm{final}}}
{L_{\mathrm{factual\ miss}}^{\mathrm{initial}}}
\leq0.75,
\]

\[
\frac{L_{\mathrm{factual\ no\ miss}}^{\mathrm{final}}}
{L_{\mathrm{factual\ no\ miss}}^{\mathrm{initial}}}
\leq0.75,
\]

\[
\frac{L_{\mathrm{plus}}^{\mathrm{final}}}
{L_{\mathrm{plus}}^{\mathrm{initial}}}
\leq0.75,
\]

\[
\frac{L_{\mathrm{clean\ transition}}^{\mathrm{final}}}
{L_{\mathrm{clean\ transition}}^{\mathrm{initial}}}
\leq0.5,
\]

\[
\overline{\Delta_D}\geq0.5,
\]

\[
\operatorname{Frac}(\Delta_D\geq0.25)\geq0.75,
\]

\[
\overline{|\Delta_H|}_{\mathrm{clean}}\leq0.05,
\]

\[
\overline{|\Delta_H|}_{\mathrm{component}}\leq0.05,
\]

\[
\max|\Delta_H|_{\mathrm{component}}\leq0.25,
\]

\[
\overline{|\Delta_G|}_{\mathrm{component}}\leq0.05,
\]

\[
\max|\Delta|_{\mathrm{identity}}\leq10^{-7},
\]

\[
\operatorname{Frac}
\left[
\Delta_D\geq0.25
\land
|\Delta_H|\leq0.05
\right]
\geq0.75.
\]

必须继续报告 1–3、4–7、8–15、16+ 像素四个 target strata，但不得据此调参。

## 11. 决策规则

若任何结构门禁失败：

```text
D-SVEF_STRUCTURAL_EXECUTION_FAIL
-> 零训练早停
-> 封存证据
```

若结构通过但任何计算门禁失败：

```text
D-SVEF_BOUNDED_MODEL_CODE_GATE_FAIL
-> 封存 v5 有效负结果
-> 禁止同版本重跑
-> 禁止 formal 800
```

若全部门禁通过：

```text
D-SVEF_BOUNDED_MODEL_CODE_GATE_PASS
-> 仅获得进入冻结确认评审的资格
-> 不直接授权 formal 800
```

无论 pass 或 fail，该实验都不是 Pd/FA 检测性能证据。

## 12. 当前状态

```text
v4_status = FROZEN_VALID_BOUNDED_NEGATIVE
v5_proposal = SPECIFIED
v5_code = NOT_IMPLEMENTED
v5_unit_toy = NOT_RUN
v5_D_R_bounded = NOT_RUN
formal_800 = NOT_AUTHORIZED
Full_CURE = NOT_STARTED
cross_backbone = NOT_STARTED
```

下一步是 additive 实现 v5 模型代码与测试。只有全部代码门禁和冻结绑定通过，
才允许创建唯一一次真实 v5 \(D_R\) bounded 运行。
