# CURE-Lite SVEF v4：模型与代码设计

> 状态：设计冻结，代码尚未实现  
> 数据边界：只允许使用 \(D_R\) 做结构可达性与受限模型代码验证  
> 方法名：Subpixel Vacancy–Evidence Factorization  
> 简称：SVEF

## 1. 研究主线不变

```text
CURE-Lite 模型设计与验证
    -> 冻结确认
    -> Full CURE
    -> 接入不同 IRSTD detector
    -> NUAA / NUDT-SIRST / IRSTD-1K 跨模型、跨数据集验证
```

SVEF v4 仍处于 CURE-Lite 模型设计阶段。它不是 Full CURE，不接入其他
detector，不读取 \(D_V/D_T\)，也不修改 Base/backbone。

v1、v2 和 v3 的代码、协议与结果全部冻结。v4 只新增 decoder/model 和独立
验证入口；不修改已有 `CURELiteDecoder`、单次 Base 提取、occupancy hard
mask、hard-union 推理、OC-APTO loss 或 paired train step。

## 2. v3 给出的结构性问题

OC-APTO v3 已完成真实 \(D_R\) 的 400-update 受限运行。训练和执行结构完整，
但模型门禁失败：

- clean \(D\) 平均响应为 0.388573，低于 0.5；
- clean 局部零响应为 0.083676，高于 0.05；
- component-null 局部平均响应为 0.183828；
- component-null 局部最大响应为 0.907818；
- 远背景平均响应约为 \(2.2\times10^{-5}\)。

旧 decoder 在 \(64\times64\) 网格上生成一个 logit，再双线性放大到
\(256\times256\)。一个 feature cell 的变化容易形成宽局部响应，无法在同一
\(4\times4\) footprint 内同时提高 1–3 像素的 \(D\) 并抑制 \(H\)。

因此 v4 不继续调 loss、采样权重或训练预算，而是改变 decoder 的响应
parameterization。

## 3. 单一核心机制

SVEF 将 residual logit 因子分解为：

\[
z_\theta(F,O)
=
B_\theta(F)
+
E_\theta(F)\odot A(O).
\]

其中：

- \(B_\theta(F)<0\) 是只依赖冻结特征的高分辨率抑制基线；
- \(E_\theta(F)\geq0\) 是高分辨率目标证据；
- \(A(O)\in(0,1]\) 是无学习参数的局部 vacancy gate；
- occupancy 不进入任何可学习卷积；
- \(B\) 与 \(E\) 由一个共享低分辨率 trunk 同时产生，不是两个 decoder。

该因子分解把“冻结特征中是否存在目标证据”和“该局部是否已被 Base
occupancy 解释”分开表示。occupancy 只能调制 feature evidence，不能独立
合成 residual response。

## 4. 高分辨率子像素证据场

输入契约为：

\[
F\in\mathbb R^{B\times C\times h\times w},\qquad
O\in\{0,1\}^{B\times1\times H\times W}.
\]

adapter/runtime receipt 显式绑定一个整数各向同性 `feature_stride=s`。当前
Reference cache 为：

```text
C = 64
h = w = 64
H = W = 256
s = 4
```

未来 detector-neutral 的含义是：不同 detector adapter 分别声明自己的
`feature_channels` 和整数 `feature_stride`，再实例化同一个 SVEF decoder
家族；不是让同一组 C64/s4 权重直接兼容所有 backbone。

共享 trunk 定义为：

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
T_0
+
\frac12 W_p*
\operatorname{SiLU}
\left[
\operatorname{GN}
\left(
W_d*T_0
\right)
\right].
\]

其中 \(W_s\) 为 \(1\times1\) projection，\(W_d\) 为 depthwise
\(3\times3\)，\(W_p\) 为 pointwise \(1\times1\)。所有卷积
`bias=False`，GroupNorm 固定 `affine=False`。

两个 \(1\times1\) head 各输出 \(s^2\) 个 phase channels：

\[
\tilde b
=
\operatorname{PixelShuffle}_s(W_B*T),
\]

\[
\tilde e
=
\operatorname{PixelShuffle}_s(W_E*T).
\]

若 \(H\times W\neq(sh)\times(sw)\)，必须先分别把 \(\tilde b\) 和
\(\tilde e\) 双线性变换到最终 evaluation grid，再构造 evidence 并乘
vacancy gate。禁止先相乘再 resize。当前真实 v4 bounded 运行必须满足
原生 \(64\rightarrow256\) 的严格 4 倍关系，因此不得触发 resize fallback。

## 5. 受约束基线与零锚定 evidence

定义：

\[
B_\theta(F)
=
-\operatorname{softplus}
\left(
\beta_{\mathrm{raw}}+\tilde b
\right)
<0.
\]

\(\beta_{\mathrm{raw}}\) 是一个可学习标量，其初值使：

\[
B_\theta(0)=\operatorname{logit}(0.1).
\]

baseline head 使用全零初始化。负基线保证没有 feature evidence 时 residual
不会以正 logit 绕过 paired 机制。

普通 `softplus(\tilde e)` 在 \(\tilde e=0\) 时仍等于 \(\log2\)，会让零
特征通过 vacancy gate 产生 occupancy-only response。因此冻结为：

\[
E_\theta(F)
=
\operatorname{softplus}
\left(
\tilde e^2
\right)
-\log2.
\]

由此：

\[
E_\theta(F)\geq0,\qquad E_\theta(0)=0.
\]

evidence head 使用非零 Xavier normal 初始化，固定 gain=0.25；不能零
初始化，否则平方映射在原点的梯度为零。共享 trunk 使用 Kaiming normal
初始化。

## 6. 固定 3×3 inverse-count vacancy gate

先用与历史 decoder 相同的 adaptive-max projection：

\[
P(O)
=
\operatorname{ProjectMax}_{h,w}(O)
\in\{0,1\}^{B\times1\times h\times w}.
\]

用固定、全为 1、无学习参数的 \(3\times3\) 核计算局部 occupancy count：

\[
N(O)
=
\mathbf1_{3\times3}*P(O).
\]

定义：

\[
A_{\mathrm{low}}(O)
=
\frac{1}{1+N(O)},
\]

\[
A(O)
=
\operatorname{Nearest}_{H,W}
\left[
A_{\mathrm{low}}(O)
\right].
\]

固定 3×3 是满足当前 \(D_R\) 全人口可达性的最小局部范围：

| 正核范围 | clean 完整覆盖 | clean 影响域平均占比 |
|---|---:|---:|
| 1×1 | 153/206 | 0.066605% |
| 3×3 | 206/206 | 0.330182% |
| 5×5 | 206/206 | 0.788361% |
| 7×7 | 206/206 | 1.438889% |

3×3 已使 2551/2551 个 clean \(D\) 像素和 16/16 个 factual anchors 具有
正的结构路径；5×5 和 7×7 不增加覆盖，只扩大影响范围。

inverse-count 而不是 binary dilation 的原因是：若邻域内仍有其他 occupancy，
binary dilation 会把删除作用完全遮蔽；inverse-count 在重叠情况下仍保持
严格正的变化。对 3×3 最密邻域删除一个 projected cell，仍有：

\[
\Delta A_{\min}
=
\frac19-\frac1{10}
=
\frac1{90}.
\]

该定义没有 temperature、radius search 或可学习 occupancy 参数。

## 7. 精确结构性质

若：

\[
O^-\subseteq O^+,
\]

则 adaptive-max projection、正核 count 和 reciprocal gate 给出：

\[
P(O^-)\leq P(O^+),
\]

\[
N(O^-)\leq N(O^+),
\]

\[
A(O^-)\geq A(O^+).
\]

由于 \(E(F)\geq0\)：

\[
z(F,O^-)-z(F,O^+)
=
E(F)\odot
\left[
A(O^-)-A(O^+)
\right]
\geq0.
\]

因此对应 probability delta 也逐像素非负。

还必须精确满足：

1. **零特征 occupancy 不变性**

   \[
   z(0,O_1)=z(0,O_2).
   \]

2. **identity 精确性**

   \[
   O^+=O^-\Rightarrow z^+=z^-.
   \]

3. **差分支持约束**

   在 \(A(O^-)=A(O^+)\) 的位置，paired logit/probability delta 必须精确
   为零。

4. **子像素自由度**

   每个 feature cell 产生 \(s^2\) 个独立 phase values。当前 s=4 时，同一
   feature cell 的 16 个输出像素不再被迫共享一个 bilinear 宽斑。

5. **单路径推理**

   模型仍只执行一次 shared decoder：

   \[
   M_r
   =
   \mathbf1[\sigma(z)\geq\tau_r]\land\neg O,
   \]

   \[
   M_{\mathrm{final}}
   =
   O\lor M_r.
   \]

   所以 CURE-Lite 仍不能删除 Base 已有的 false alarms；它解决的是漏检
   找回，同时约束新增 false alarms。

## 8. 与 OC-APTO 的边界

v4 只替换 decoder parameterization。以下对象保持逐字节冻结：

```text
cure_lite/paired_outcome_types.py
cure_lite/paired_outcome_losses.py
cure_lite/train/paired_outcome_step.py
cure_lite/experiment/paired_outcome_inputs.py
cure_lite/experiment/paired_outcome_schedule.py
cure_lite/experiment/paired_outcome_bounded.py
tools/run_paired_outcome_bounded.py
```

v4 继续使用：

- 同一 206 clean-positive + 16 component-null population；
- 同一 16 factual-miss + 16 factual-no-miss anchors；
- 同一 pair-uniform schedule；
- 同一 `CURELiteLoss`；
- 同一 `OutcomeCompleteTransitionLoss`；
- 同一 `outcome_complete_train_step`；
- 同一 400-update、3 forward/12 state/update 预算。

v3 的 \(J\) 在 v4 runner 中只称为“冻结的局部评估 envelope”。SVEF 的精确
gate-change support 可能扩展到 \(J\) 外，但这些非 \(D\) 像素仍属于旧 loss
的 \(H\) 或 \(G\)，因此继续受到 zero-response 约束。v4 runner 必须另外记录
精确 gate support，不能把旧 \(J\) 描述为 SVEF 的完整影响域。

## 9. 新增代码

只新增：

```text
cure_lite/factorized_config.py
cure_lite/factorized_decoder.py
cure_lite/factorized_model.py
cure_lite/experiment/factorized_outcome_bounded.py
tools/run_factorized_outcome_bounded.py
tests/test_factorized_config.py
tests/test_factorized_decoder.py
tests/test_factorized_model.py
tests/test_factorized_outcome_step.py
tests/test_factorized_outcome_toy_overfit.py
tests/test_factorized_outcome_bounded.py
tests/test_run_factorized_outcome_bounded_cli.py
```

bounded 阶段不修改根级 `__init__.py`、`train/__init__.py` 或
`experiment/__init__.py`。测试和 runner 从新模块直接导入。

## 10. 代码与 toy 门禁

代码实现后必须先通过：

1. config 全字段冻结和非法值拒绝；
2. \(F=0\) 时任意 occupancy 输出逐元素相同；
3. 随机 \(O^-\subseteq O^+\) 的 vacancy、logit 和 probability delta 全部
   非负；
4. gate-change support 外差分精确为零；
5. identity-null 最大差分不超过 \(10^{-7}\)；
6. 3×3 最密重叠反例的 vacancy delta 等于 \(1/90\)；
7. PixelShuffle 的 16 个 phase 与 4×4 子像素位置一一对应；
8. 当前 64→256 路径不触发 resize；
9. 非原生输出尺寸只允许“分别 resize 两个 field，再乘最终 gate”；
10. 2B endpoint forward 与两个独立 forward 等价；
11. feature 保持 detached，plus/minus 两端梯度有限且非零；
12. preflight 失败时 decoder、梯度和 optimizer 无副作用；
13. 1、2、3 像素 subcell clean toy 同时满足：

    \[
    \overline{\Delta q_D}\geq0.8,\qquad
    \max|\Delta q_H|\leq0.05;
    \]

14. component-null footprint 最大响应不超过 0.05；
15. factual-miss/factual-no-miss absolute anchors 同时可学习；
16. v1/v2/v3 文件哈希和聚焦回归保持不变。

## 11. 真实 \(D_R\) bounded 门禁

训练前静态门禁：

- 206/206 clean pairs 的全部 \(D\) 像素满足
  \(A(O^-)>A(O^+)\)；
- 16/16 factual-miss anchors 的 target 均具有正 vacancy；
- zero-feature occupancy delta = 0；
- gate support 外 delta = 0；
- deletion monotonicity violation = 0；
- 真实 C64/s4 路径未触发 field resize；
- pair、source、anchor exposure 与冻结 schedule 完全一致。

保留 v3 全部 computational gates，并增加逐 pair 联合门禁：

\[
\frac{1}{206}
\sum_i
\mathbf1
\left[
\overline{\Delta q}_{D_i}\geq0.25
\land
\overline{|\Delta q|}_{H_i}\leq0.05
\right]
\geq0.75.
\]

同时单独报告 1–3、4–7、8–15 和 \(\geq16\) 个 \(D\) 像素的联合通过率，
避免总体均值掩盖小目标失败。

只有代码/toy、静态可达性、结构执行和全部 bounded computational gates
同时通过，才授权进入 seed 42/43 的 800-epoch 正式训练。v4 同一冻结版本
只运行一次真实 bounded；失败后保留完整证据，不自动重试或调门槛。

## 12. 当前证据边界

本设计的创新点是一个带精确符号、支持和零特征不变性的 counterfactual
factorization，而不是 attention、Transformer、多尺度分支或 decoder
堆叠。

当前只能说明：

- v3 的低分辨率混合 decoder 已被有效负结果否定；
- SVEF v4 在结构上消除了 occupancy-only learned path；
- SVEF v4 为同一 feature cell 内的小目标提供独立子像素自由度；
- 固定 3×3 vacancy gate 对当前 \(D_R\) population 全部可达。

在代码、toy、真实 bounded、正式 800 epoch 和冻结确认完成前，不能宣称：

- CURE-Lite 已设计成功；
- Pd/FA 已改善；
- Full CURE 已建立；
- 已具备跨 detector 或跨数据集性能。
