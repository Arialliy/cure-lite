# CURE-Lite v22 PACRE：真实 \(D_R\) 代数门禁失败归因与 v23 FACRE 修改方案

> **2026-07-28 最终裁决：本方案不执行。**  
> 后续源码审计与 FP32 重放证明，v22 的唯一失败首先应按
> verifier false rejection 处理，而不是据此修改模型 forward。实施主线改为
> `PACRE-VC v23`：数值 forward、参数、PMOPE、输入与零水平集解码均与
> v22 保持一致，只修正验证器及证据合同。FACRE 仅保留为未来独立模型候选；
> 它不能用于“解释性修复”已封存的 v22 结果，也不会在本次运行中训练。
> 本文以下内容是被否决候选的历史分析，不是当前执行授权。

> 审查日期：2026-07-28  
> 仓库：`Arialliy/cure-lite`  
> 审查快照：`main@9f43815525a7e207e9a550dacf894f88ace80c5b`  
> 当前封存结论：`PACRE_V22_D_R_STRUCTURAL_FAIL`  
> 唯一失败项：`05_phase_residual_and_compatibility_algebra_valid`  
> 未运行：bounded-400、Formal-800、\(D_V\)、\(D_T\)  
> 推荐下一候选：**v23 FACRE（Factored Algebraic Compatibility Residual Energy）**  
> 修改边界：**保持 \((F_b,O)\)、PAET、binary flip、64,064 参数、单一 completion field、固定零水平集和 hard union；只修改 compatibility 的数值因式分解与代数验证契约。**

---

## 0. 证据边界

本报告直接审查了：

- `cure_lite_v22/pacre.py`
- `cure_lite_v22/dataset_free.py`
- `cure_lite_v22/dr_gate.py`
- `cure_lite_v22/factory.py`
- `cure_lite_v22/training.py`
- `cure_lite_v22/decision.py`
- `cure_lite_v22/bounded_runner.py`
- `tests/test_cure_lite_v22_pacre.py`
- `tests/test_cure_lite_v22_pacre_dr_gate.py`
- PAET/BFA 继承实现

本报告接受用户提供的封存事实：

```text
真实 D_R 门禁：FAIL
唯一失败 check：
05_phase_residual_and_compatibility_algebra_valid

bounded-400：未运行
Formal-800：未运行
D_V / D_T：未读取
自动重试：未发生
v22 后续修改：未发生
```

封存 receipt 只保存了：

```text
all_algebra_checks_passed = false
```

而没有保存六个代数子检查中具体哪一个失败、最大误差、失败元素坐标及尺度。因此，不能从封存 JSON **逐字节证明**具体子式。

但是，根据代码调用链、模型自身的 fail-closed 校验和 FP32 压力复现，可以把高概率根因收敛到：

> `actual_specific_joint_affine - actual_common_joint_affine ≈ phase_feature_residual`  
> 以及对应 flipped 版本的固定容差检查。

报告会明确区分：

- 已由代码直接证明的事实；
- 由排除法得到的高置信归因；
- v23 必须新增的确认性诊断。

---

# 1. 当前失败是什么性质

## 1.1 不是训练负结果

真实 \(D_R\) gate 的源码明确说明，它是：

```text
read-only
structural / identifiability gate
zero optimizer step
no performance estimate
```

它只检查：

- PACRE 表示是否完整；
- phase residual 是否满足定义；
- compatibility algebra 是否成立；
- target/background 是否存在 latent witness；
- PMOPE 初始化梯度路径是否非退化；
- 整个过程是否保持模型、cache、population 和 RNG 不变。

所以当前失败不能解释为：

- 模型训练不收敛；
- target completion 不紧致；
- Pd、FA、IoU 不达标；
- PACRE 在 bounded-400 中效果差；
- PACRE 已被真实性能实验否定。

当前准确状态是：

> **PACRE-v22 的真实 \(D_R\) 结构授权失败；失败发生在数值代数闭合层，而不是训练性能层。**

## 1.2 v22 仍然必须冻结

即使后续证明 check 05 是数值 false negative，也不能：

- 改写 v22 receipt；
- 修改 v22 gate 后继续称为同一个 attempt；
- 直接授权 v22 bounded-400；
- 将 v22 的失败改成 PASS。

正确做法是：

```text
保留 v22 唯一终态
    ↓
建立新版本 v23
    ↓
重新执行 dataset-free
    ↓
重新执行只读 D_R gate
    ↓
只有全部通过后才允许 bounded-400
```

---

# 2. PACRE-v22 的当前方程

## 2.1 输入与主线

公开输入仍为：

\[
\left(\operatorname{sg}(F_b), O\right).
\]

其中：

- \(F_b\)：冻结 Base detector feature；
- \(O\)：Base occupancy；
- 不输入 GT、pair kind、target ID、source ID；
- 输出只有一个非饱和 scalar completion field。

推理：

\[
Y_{\mathrm{completion}}
=
\mathbf 1[\phi<0]\land \neg O,
\]

\[
Y_{\mathrm{final}}
=
O\lor Y_{\mathrm{completion}}.
\]

## 2.2 phase-aligned feature

PAET 将 coarse feature affine 双线性传输到输出网格，再按 row-major phase
打包：

\[
A_F
\longrightarrow
A_F^p,
\qquad p=1,\ldots,P,
\]

其中：

\[
P=s^2.
\]

正式配置：

\[
s=4,\qquad P=16.
\]

## 2.3 phase centering

当前实现：

\[
\mu
=
\frac1P\sum_{p=1}^{P}A_F^p,
\]

\[
r_p
=
A_F^p-\mu.
\]

代码对应：

```python
phase_mean = phase_feature_affine.mean(dim=1, keepdim=True)
residual = phase_feature_affine - phase_mean
```

## 2.4 compatibility energy

当前代码独立构造：

\[
J^{\mathrm{common}}_p(O)
=
A_U(O)+\mu,
\]

\[
J^{\mathrm{specific}}_p(O)
=
A_U(O)+A_F^p.
\]

然后：

\[
H_p(O,F)
=
\operatorname{SiLU}
\left(
J^{\mathrm{specific}}_p
\right)
-
\operatorname{SiLU}
\left(
J^{\mathrm{common}}_p
\right),
\]

\[
C_p(O,F)
=
w^\top H_p(O,F).
\]

binary flip interaction：

\[
\Delta_p
=
\frac12
\left[
C_p(O,F)-C_p(\operatorname{flip}_p(O),F)
\right].
\]

最终：

\[
\phi
=
\operatorname{PixelShuffle}
\left(
a+\Delta
\right),
\qquad a=0.9.
\]

## 2.5 参数契约

正式配置：

\[
C=64,\qquad s=4,\qquad w=32.
\]

参数：

```text
joint_state_weight   [32, 80, 5, 5]
joint_hidden_bias    [32]
scalar_energy_weight [32]
```

总参数：

\[
32\cdot80\cdot5\cdot5+32+32
=
64,064.
\]

v23 将保持完全相同的参数名、参数形状、初始化和数量。

---

# 3. 门禁 05 实际包含什么

`_algebra_checks(fields)` 一共执行六个子检查。

## 3.1 Phase reconstruction

\[
\mu+r
\approx
A_F.
\]

代码：

```python
torch.allclose(
    phase_feature_mean + residual,
    phase_feature_affine,
    rtol=0.0,
    atol=scale_aware_tolerance,
)
```

## 3.2 Residual centering

\[
\left|
\sum_p r_p
\right|
\le
\tau_{\mathrm{center}}.
\]

## 3.3 Actual joint residual identity

\[
J^{\mathrm{specific}}(O)
-
J^{\mathrm{common}}(O)
\approx r.
\]

当前固定：

```python
rtol = 2.0e-6
atol = 2.0e-7
```

## 3.4 Flipped joint residual identity

\[
J^{\mathrm{specific}}(\operatorname{flip}O)
-
J^{\mathrm{common}}(\operatorname{flip}O)
\approx r.
\]

使用同样固定容差。

## 3.5 Actual compatibility hidden recomputation

\[
H(O,F)
\approx
\operatorname{SiLU}(J^{\mathrm{specific}})
-
\operatorname{SiLU}(J^{\mathrm{common}}).
\]

## 3.6 Flipped compatibility hidden recomputation

对 flipped endpoint 执行同样检查。

只要任意 tensor 中任意一个元素违反 `torch.allclose`，该 state 的
`_algebra_checks` 就是 false。

`_representation_probe` 又在：

```text
32 target states
96 context states
总计 128 states
```

上执行：

```python
all_algebra &= _algebra_checks(fields)
```

因此：

> 128 个 state 中任意一个 state 的任意一个元素失败，门禁 05 整体失败。

最终 receipt 只保存：

```text
all_algebra_checks_passed = false
```

没有保存六个子检查的分解。

---

# 4. 为什么高概率不是 phase centering 本身失败

## 4.1 Residual-sum 检查已被模型内部先行验证

`forward_fields()` 返回前会调用 `_validate_pacre_fields()`。

其中已经使用与 D_R gate 相同形式的尺度自适应 tolerance 检查：

\[
\sum_p r_p\approx0.
\]

若该检查失败，forward 会直接抛出：

```text
AssertionError("phase residual is not centered")
```

而不是正常形成一个 `PACRE_V22_D_R_STRUCTURAL_FAIL` receipt。

因此，在正式 gate 能遍历完整 population 并形成唯一终态的前提下，
subcheck 2 高概率已通过。

## 4.2 Reconstruction tolerance 是尺度自适应的

phase reconstruction 使用：

\[
\tau
=
4P\epsilon_{32}
\left(
1+\|A_F\|_\infty
\right).
\]

它随 phase-affine 幅值增长，且只涉及：

```text
mean + residual
```

一层重构。

该检查仍应在 v23 中拆开记录，但它不是最可疑项。

## 4.3 Hidden 重算使用同一组已存 tensor

actual/flipped hidden 在模型中定义为：

```python
F.silu(specific_joint) - F.silu(common_joint)
```

gate 又对同一 stored tensors 重算相同表达式。

在同一进程、同一设备、同一 deterministic scope 下，该检查通常稳定。

## 4.4 剩余最可疑项

排除后最可疑的是：

```python
specific_joint - common_joint
```

与：

```python
phase_feature_affine - phase_feature_mean
```

的比较。

两者数学等价，但 FP32 运算图不同。

---

# 5. 核心数值问题：共同项消去

## 5.1 两条不同的 FP32 路径

目标 residual：

\[
\widehat r
=
\operatorname{fl}
\left(
A_F^p-\mu
\right).
\]

gate 中重建 residual：

\[
\widehat d
=
\operatorname{fl}
\left[
\operatorname{fl}(A_U+A_F^p)
-
\operatorname{fl}(A_U+\mu)
\right].
\]

无限精度下：

\[
d=r.
\]

FP32 下：

\[
\widehat d
\neq
\widehat r
\]

是正常现象。

误差主要由共同项：

\[
A_U
\]

以及 joint affine 的尺度决定，而当前 `allclose` tolerance 只使用：

\[
2\times10^{-7}
+
2\times10^{-6}|\widehat r|.
\]

它没有包含：

\[
|J^{\mathrm{specific}}|,
\qquad
|J^{\mathrm{common}}|,
\qquad
|A_U|.
\]

## 5.2 条件数问题

当：

\[
|r|
\ll
|A_U+\mu|,
\]

需要从两个接近的大数中恢复一个小 residual。

这是经典 cancellation：

\[
(x+\delta)-x.
\]

即使：

- 所有输入有限；
- 数学方程正确；
- forward 确定性正常；

恢复出的 \(\delta\) 仍可能偏离独立计算的 residual 数个 ulp。

## 5.3 一个确定的 FP32 反例

取：

\[
A_U+\mu=100,
\qquad
r=0.1.
\]

FP32 中：

```text
fl((100 + 0.1) - 100)
= 0.09999847412109375
```

与 0.1 的绝对误差约：

\[
1.53\times10^{-6}.
\]

当前允许误差：

\[
2\times10^{-7}
+
2\times10^{-6}\cdot0.1
=
4\times10^{-7}.
\]

因此数学恒等式正确，但当前 gate 返回 false。

更温和的例子：

```text
common = 10
residual = 0.01
recovered = 0.010000228881835938
absolute error ≈ 2.291e-7
allowed error ≈ 2.2e-7
```

同样失败。

## 5.4 本报告的 formal-shape 无数据压力复现

本报告按正式形状构造了一个非 \(D_R\) FP32 probe：

```text
feature channels  = 64
stride             = 4
phases             = 16
hidden width       = 32
feature grid       = 64 × 64
kernel             = 5 × 5
weight init        = PyTorch Kaiming normal
feature            = global-RMS normalized random tensor
occupancy          = sparse random binary tensor
```

一个 state 的 joint-residual tensor 包含：

\[
16\cdot32\cdot64\cdot64
=
2,097,152
\]

个元素。

结果：

```text
phase reconstruction：PASS
phase residual sum：  PASS
joint subtraction allclose failures：62
failure rate：约 2.96e-5
maximum absolute mismatch：4.768e-7
```

这不是实际 \(D_R\) 结果，但证明：

> 当前 gate 在完全正常的正式尺度 FP32 tensor 上，能够稳定产生代数 false negative。

真实 D_R probe 要检查 128 个 states，因此“所有元素必须通过”的策略会显著
放大极少数 ulp mismatch。

---

# 6. dataset-free 为什么没有发现

## 6.1 当前 toy 规模很小

dataset-free 使用：

```text
feature channels = 2
stride           = 2
phases           = 4
width            = 4
feature grid     = 3 × 4
```

而正式配置是：

```text
feature channels = 64
stride           = 4
phases           = 16
width            = 32
真实 feature grid 远大于 3 × 4
```

toy test 中需要同时满足的元素数量小很多。

## 6.2 当前 dataset-free 没有复用完整 D_R algebra check

dataset-free 主要检查：

- residual sum；
- phase-common compatibility 为零；
- zero-feature anchor；
- fast/reference；
- binary flip oddness；
- phase response 非退化；
- 参数、接口和梯度。

它没有在 formal-shape stress tensor 上逐元素执行：

```python
(specific_joint - common_joint) ≈ residual
```

的完整 cancellation audit。

## 6.3 小测试通过不代表大 tensor 的 “all elements” 通过

PyTorch `torch.allclose` 的条件是：

\[
|x_i-y_i|
\le
\text{atol}+\text{rtol}|y_i|
\]

并且要求**所有元素**都满足。

小规模单测只说明：

> 当前 seed 的少量元素没有落入临界舍入区。

它不能证明数百万元素、128 states 下仍不会出现一个临界点。

---

# 7. 失败的准确研究归因

## 7.1 当前可以确定

- v22 的 gate 05 设计成六项 aggregate；
- 真实失败没有子项 ledger；
- current model 独立计算 common 和 specific joint；
- gate 再用减法恢复 residual；
- fixed `rtol/atol` 不考虑共同项尺度；
- dataset-free 没有 formal-shape cancellation stress；
- bounded/performance 从未运行。

## 7.2 高置信推断

最可能失败的是 subcheck 3 和/或 4：

```text
actual specific - common ≈ residual
flipped specific - common ≈ residual
```

其根因是：

> **用数值不稳定的逆向消去来验证一个本应由前向构造保证的代数关系。**

## 7.3 当前仍不能断言

由于封存 probe 没有子项：

- 不能说 100% 确定只有 subcheck 3/4；
- 不能声称实际 D_R 最大误差是多少；
- 不能声称模型真实效果会成功；
- 不能直接跳过新版本的 D_R gate。

---

# 8. 推荐下一候选：v23 FACRE

名称：

> **Factored Algebraic Compatibility Residual Energy**  
> **FACRE-v23**

建议类名：

```python
CURELitePhaseAlignedFactoredCenteredResidualCompatibilityEnergyLevelSet
```

## 8.1 主线保持

FACRE 继续使用：

- frozen Base；
- \((F_b,O)\)；
- phase-preserving occupancy；
- PAET phase transport；
- phase centering；
- binary current-center-phase flip；
- shared SiLU energy；
- one scalar completion field；
- fixed zero threshold；
- hard union；
- 64,064 parameters。

## 8.2 唯一方程修改：显式因式化

定义：

\[
\mu
=
\frac1P
\sum_p A_F^p,
\]

\[
r_p
=
A_F^p-\mu.
\]

不再独立计算：

\[
A_U+A_F^p.
\]

改为：

\[
B_p(O)
=
A_U(O)+\mu,
\]

\[
S_p(O,F)
=
B_p(O)+r_p.
\]

compatibility：

\[
C_p(O,F)
=
w^\top
\left[
\operatorname{SiLU}(S_p)
-
\operatorname{SiLU}(B_p)
\right].
\]

binary flip 与 field 不变：

\[
\Delta_p
=
\frac12
\left[
C_p(O,F)
-
C_p(\operatorname{flip}_p(O),F)
\right],
\]

\[
\phi
=
\operatorname{PixelShuffle}(a+\Delta).
\]

无限精度下，FACRE 与 PACRE-v22 相同：

\[
B_p+r_p
=
A_U+\mu+(A_F^p-\mu)
=
A_U+A_F^p.
\]

但程序图发生了关键改变：

```text
v22：
common   = occupancy + mean
specific = occupancy + phase

v23：
common   = occupancy + mean
specific = common + residual
```

因此 residual 是 compatibility forward 的显式输入，而不是 gate 事后通过两个
joint tensors 的减法恢复出来。

## 8.3 为什么这不是改变研究主线

FACRE 不增加任何：

- 参数；
- head；
- branch；
- detector module；
- target metadata；
- threshold；
- training budget。

变化只在：

> mathematically equivalent expression 的 FP32 evaluation order。

因此它属于：

> 数值代数闭合版本，而不是新的容量模型。

---

# 9. v23 的 gate 不能继续使用旧检查

即使：

```python
specific = common + residual
```

gate 再计算：

```python
specific - common
```

仍可能因为加法舍入得到：

\[
\operatorname{fl}
[
\operatorname{fl}(common+residual)-common
]
\neq residual.
\]

所以不能只改 forward，不改 verifier。

新的 verifier 必须检查：

> 程序实际承诺的前向构造关系。

而不是检查一个数值病态的逆运算。

---

# 10. 新的代数验证契约

## 10.1 分解 receipt

不再只保存：

```text
all_algebra_checks_passed
```

必须保存每一个子项：

```text
phase_reconstruction_backward_error_valid
phase_zero_sum_backward_error_valid
actual_common_constructed_exactly
actual_specific_constructed_from_common_plus_residual_exactly
flipped_common_constructed_exactly
flipped_specific_constructed_from_common_plus_residual_exactly
actual_hidden_recomputed_valid
flipped_hidden_recomputed_valid
actual_energy_recomputed_valid
flipped_energy_recomputed_valid
fast_literal_reference_valid
```

每项同时记录：

```text
maximum error
error bound
argmax state ID
argmax tensor coordinate
local operand magnitudes
failed element count
```

## 10.2 Phase reconstruction 使用归一化后向误差

定义：

\[
e_{\mathrm{recon}}
=
\max
\frac{
|(\mu+r)-A_F|
}{
1+|\mu|+|r|+|A_F|
}.
\]

不使用固定绝对 tolerance。

## 10.3 Zero-sum 使用 reduction-aware error

定义：

\[
e_{\mathrm{center}}
=
\max
\frac{
|\sum_p r_p|
}{
1+\sum_p|A_F^p|
}.
\]

令：

\[
\epsilon
=
\operatorname{finfo}(\mathrm{float32}).\epsilon,
\]

\[
\gamma_n
=
\frac{n\epsilon}{1-n\epsilon}.
\]

冻结：

\[
e_{\mathrm{center}}
\le
4\gamma_P.
\]

当：

\[
P=16
\]

时，该 bound 由 FP32 phase reduction 的操作数数量解析得到，不从 \(D_R\)
观测值搜索。

## 10.4 Joint construction 直接检查前向式

actual：

```python
expected_common = occupancy_affine + phase_mean
expected_specific = actual_common_joint_affine + residual
```

检查 stored tensors 与 expected tensors。

flipped 同理。

这里检查的是：

\[
S=B+r,
\]

而不是：

\[
S-B=r.
\]

前者是前向构造；后者是病态逆运算。

## 10.5 Compatibility hidden

检查：

\[
H
=
\operatorname{SiLU}(S)-\operatorname{SiLU}(B).
\]

使用：

- 同设备 deterministic recomputation；
- scale-normalized backward error；
- 可选 FP64 oracle 诊断。

## 10.6 Compatibility energy

检查：

\[
C=w^\top H.
\]

同样输出：

- normalized error；
- absolute error；
-最大坐标；
- error bound。

## 10.7 旧 subtractive identity 只作诊断

仍可记录：

\[
e_{\mathrm{legacy}}
=
\left|
(S-B)-r
\right|.
\]

但它必须标记为：

```text
diagnostic_only = true
gate_eligible = false
reason = cancellation_conditioned_inverse_identity
```

不能继续用它决定模型代数是否正确。

---

# 11. 推荐代码实现

## 11.1 新包

```text
cure_lite_v23/
    __init__.py
    algebra.py
    facre.py
    factory.py
    dataset_free.py
    dr_gate.py
    training.py
    decision.py
    bounded_runner.py
```

v22 文件保持原样。

---

## 11.2 `algebra.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


FACRE_FP32_EPS = torch.finfo(torch.float32).eps
FACRE_ALGEBRA_POLICY = (
    "forward_factored_identity_and_scale_normalized_"
    "backward_error_v1"
)


def fp32_gamma(operation_count: int) -> float:
    if (
        isinstance(operation_count, bool)
        or not isinstance(operation_count, int)
        or operation_count < 1
    ):
        raise ValueError("operation_count must be a positive integer")

    value = float(operation_count) * FACRE_FP32_EPS
    if value >= 1.0:
        raise ValueError("gamma bound is undefined")
    return value / (1.0 - value)


def normalized_backward_error(
    actual: Tensor,
    expected: Tensor,
) -> Tensor:
    if (
        actual.dtype != torch.float32
        or expected.dtype != torch.float32
        or actual.shape != expected.shape
        or actual.device != expected.device
    ):
        raise ValueError("backward-error tensors must align")

    numerator = (actual - expected).abs()
    denominator = (
        1.0 + actual.abs() + expected.abs()
    )
    return numerator / denominator


def phase_reconstruction_error(
    phase_affine: Tensor,
    phase_mean: Tensor,
    phase_residual: Tensor,
) -> Tensor:
    expected = phase_mean + phase_residual
    return normalized_backward_error(expected, phase_affine)


def phase_centering_error(
    phase_affine: Tensor,
    phase_residual: Tensor,
) -> Tensor:
    residual_sum = phase_residual.sum(dim=1)
    scale = 1.0 + phase_affine.abs().sum(dim=1)
    return residual_sum.abs() / scale


@dataclass(frozen=True)
class FACREAlgebraObservation:
    check_name: str
    passed: bool
    maximum_error: float
    allowed_error: float
    failed_element_count: int
    maximum_coordinate: tuple[int, ...]
```

正式实现还应：

- 拒绝 NaN/Inf；
- 拒绝 dtype/device drift；
- 用 deterministic lexicographic argmax；
- 将浮点值保存为 hex；
- 生成 stable fingerprint。

---

## 11.3 `facre.py`

核心修改：

```python
def _factored_compatibility_energy(
    self,
    occupancy_affine: Tensor,
    phase_feature_mean: Tensor,
    phase_feature_residual: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    common_joint = (
        occupancy_affine + phase_feature_mean
    ).expand_as(phase_feature_residual)

    specific_joint = (
        common_joint + phase_feature_residual
    )

    compatibility_hidden = (
        F.silu(specific_joint)
        - F.silu(common_joint)
    )

    compatibility_energy = (
        compatibility_hidden
        * self.scalar_energy_weight[
            None, None, :, None, None
        ]
    ).sum(dim=2)

    return (
        common_joint.contiguous(),
        specific_joint.contiguous(),
        compatibility_hidden.contiguous(),
        compatibility_energy.contiguous(),
    )
```

actual：

```python
actual = self._factored_compatibility_energy(
    occupancy_affine.unsqueeze(1),
    phase_feature_mean,
    phase_feature_residual,
)
```

flipped：

```python
flipped = self._factored_compatibility_energy(
    flipped_occupancy_affine,
    phase_feature_mean,
    phase_feature_residual,
)
```

其余：

```text
flip_delta
odd projection
native_phase_interaction
field amplitude
PixelShuffle
```

全部保持。

---

## 11.4 reference oracle

旧 reference 使用：

```python
SiLU(occupancy + phase_feature)
-
SiLU(occupancy + phase_mean)
```

v23 literal oracle 必须使用新冻结式：

```python
common = occupancy_affine + phase_mean
specific = common + phase_residual
hidden = F.silu(specific) - F.silu(common)
```

这样 fast/reference 比较验证的是同一个 v23 公式。

---

# 12. 新 dataset-free gate

## 12.1 保留全部 v22 checks

保留：

- residual centered；
- phase-common compatibility zero；
- zero feature anchor；
- fast/reference；
- binary flip odd；
- phase response nondegenerate；
- parameter topology；
- single field；
- first/second gradient；
- initialization path；
- factory；
- generated-only；
- state preserved。

## 12.2 新增 formal-shape arithmetic stress

至少加入：

```text
feature channels = 64
stride           = 4
phases           = 16
width            = 32
feature grid     = 64 × 64
```

可以 chunked 执行，避免不必要峰值内存。

## 12.3 Adversarial cancellation grid

固定生成：

```text
common scales:
0, 1, 10, 100, 1000

residual scales:
1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5
```

要求：

```text
old subtractive allclose 至少有一个已知反例
new forward-construction checks 全部通过
new backward-error checks 全部通过
all fields finite
all gradients finite
```

旧检查失败不是 v23 failure，而是证明新 gate 覆盖了已知病态场景。

## 12.4 Formal random stress

使用与正式模型相同：

- Kaiming initialization；
- global-RMS normalized feature；
- phase occupancy；
- 5×5 convolution；
- bilinear phase transport。

至少 seeds：

```text
42
43
44
```

要求：

```text
all FACRE gate-eligible algebra checks pass
legacy cancellation mismatch may be nonzero
```

## 12.5 CPU / selected-device consistency

若正式 gate 在 CUDA 运行，应：

- CPU 运行 generated stress；
- selected device 再运行；
- 使用相同 scale-normalized error formula；
- 不要求 CPU/GPU bitwise equal；
- 分别要求满足同一解析 error bound。

PyTorch 官方文档明确指出 mathematically identical floating operations
不保证跨平台 bitwise 相同，因此 gate 应验证误差界，而不是平台字节一致。

---

# 13. 新真实 \(D_R\) gate

## 13.1 新 check 名称

建议：

```text
01_dataset_free_prerequisite_exact_and_passed
02_real_D_R_seed42_population_bound
03_exact_facre_model_config_factory_and_parameter_contract
04_complete_state_forward_ledger_and_exact_fields_type
05_factored_phase_residual_compatibility_backward_error_valid
06_each_target_group_has_one_bound_residual_flip_latent_witness
07_no_exact_target_positive_latent_collision
08_zero_readout_anchor_and_fixed_readout_witness
09_real_pmope_initialization_gradient_path
10_field_loss_direction_correct_for_all_roles
11_model_population_cache_rng_and_grad_buffers_preserved
12_read_only_zero_update_D_R_scope
```

除 check 05 外，其余语义不变。

## 13.2 Check 05 的内部 ledger

每个 state 保存：

```json
{
  "state_id": "...",
  "phase_reconstruction": {
    "passed": true,
    "maximum_error_hex": "...",
    "bound_hex": "...",
    "failed_element_count": 0,
    "argmax_coordinate": [0, 0, 0, 0, 0]
  },
  "phase_centering": {},
  "actual_common_construction": {},
  "actual_specific_construction": {},
  "flipped_common_construction": {},
  "flipped_specific_construction": {},
  "actual_hidden": {},
  "flipped_hidden": {},
  "actual_energy": {},
  "flipped_energy": {},
  "legacy_subtractive_residual": {
    "gate_eligible": false,
    "maximum_absolute_error_hex": "...",
    "failed_under_v22_allclose_count": 62
  }
}
```

全局 receipt 保存：

```text
state count
subcheck count
all gate-eligible subchecks passed
global maximum error per subcheck
global argmax state/coordinate
legacy diagnostic only
```

## 13.3 不得读取结果后改 bound

所有 bounds 必须在：

```text
dataset-free receipt
implementation closure
D_R authorization
```

中提前冻结。

不能：

- 看到 \(D_R\) 最大误差后再扩大；
- 使用 percentile；
- 忽略少量失败元素；
- 对 target/background 使用不同数值 tolerance；
- 只在 CPU 通过后换设备。

## 13.4 D_R gate 仍然零训练

保持：

```text
optimizer_constructed = false
optimizer_steps = 0
parameter_updates = 0
training_performed = false
D_R_accessed = true
D_V_accessed = false
D_T_accessed = false
```

---

# 14. 为什么不能只“把 atol 调大”

不推荐：

```python
atol = 1e-5
```

或：

```python
rtol = 1e-4
```

原因：

1. 它仍然没有显式考虑共同项尺度；
2. 相同 residual 在不同 common scale 下的误差不同；
3. 固定大 tolerance 可能掩盖真正的实现错误；
4. 它看起来像根据 \(D_R\) 失败事后放宽门禁；
5. 无法给出跨 CPU/GPU 的一致数值语义。

FACRE 的方案是：

```text
改前向构造
+
检查前向构造
+
使用解析后向误差
+
保留详细 ledger
```

而不是简单扩大容差。

---

# 15. 为什么不使用 float64 训练

不建议把整个 model forward 改为 float64，因为：

- 改变正式计算契约；
- 显著增加训练成本和显存；
- 改变与历史 PAET/BFA 的公平性；
- 不能证明 FP32 部署图本身闭合；
- 当前问题只发生在验证恒等式的方式。

float64 可以用于：

```text
read-only oracle diagnostic
```

但正式 field 与训练仍保持 FP32。

---

# 16. 文件级修改计划

## 16.1 保持冻结

```text
cure_lite_v22/**
v22 dataset-free receipt
v22 D_R receipt
v22 decision / COMPLETE
v22 implementation closure
```

不得覆盖。

## 16.2 新增

```text
cure_lite_v23/__init__.py
cure_lite_v23/algebra.py
cure_lite_v23/facre.py
cure_lite_v23/factory.py
cure_lite_v23/dataset_free.py
cure_lite_v23/dr_gate.py
cure_lite_v23/training.py
cure_lite_v23/decision.py
cure_lite_v23/bounded_runner.py
```

## 16.3 新 tools

```text
tools/audit_cure_lite_v23_facre_dataset_free.py
tools/run_cure_lite_v23_facre_dr_gate.py
tools/verify_cure_lite_v23_facre_dr_receipt.py
tools/run_cure_lite_v23_facre_bounded_400.py
```

bounded tool 可以先存在，但必须在 D_R PASS 前拒绝执行。

## 16.4 新 tests

```text
tests/test_cure_lite_v23_facre_core.py
tests/test_cure_lite_v23_facre_algebra.py
tests/test_cure_lite_v23_facre_fp32_stress.py
tests/test_cure_lite_v23_facre_dataset_free.py
tests/test_cure_lite_v23_facre_dr_gate.py
tests/test_cure_lite_v23_facre_factory.py
tests/test_cure_lite_v23_facre_training.py
tests/test_cure_lite_v23_facre_decision.py
tests/test_cure_lite_v23_facre_bounded_runner.py
tests/test_run_cure_lite_v23_facre_dr_gate_cli.py
tests/test_run_cure_lite_v23_facre_bounded_400_cli.py
```

## 16.5 新 protocol root

```text
protocols/IRSTD-1K/
phase_aligned_factored_centered_residual_compatibility_v23/
```

至少包含：

```text
v22_failure_inheritance_receipt.json
numerical_algebra_design.md
fp32_stress_receipt.json
dataset_free_receipt.json
implementation_closure.json
dr_gate_pre_run_authorization.json
```

D_R authorization 必须最后创建。

---

# 17. 必须新增的测试

## 17.1 旧反例必须可复现

```text
common = 10
residual = 0.01
```

要求：

```text
v22 subtractive allclose = false
FACRE forward construction = pass
FACRE backward error = pass
```

以及：

```text
common = 100
residual = 0.1
```

## 17.2 Formal-shape all-element stress

至少：

```text
[B,P,W,h,w] = [1,16,32,64,64]
```

要求：

- reconstruction pass；
- centering pass；
- common construction pass；
- specific construction pass；
- hidden pass；
- energy pass；
- finite；
- no exception；
- legacy mismatch 被记录但不决定 gate。

## 17.3 Scale equivariance of validation

将 common joint 同时增加：

```text
0
1
10
100
1000
```

保持 residual 不变。

新的 normalized validation 结论必须保持一致。

## 17.4 真错误必须被拒绝

人工篡改：

```python
specific_joint[..., index] += 1.0e-3
```

必须失败。

篡改：

```python
phase_residual[..., index] += 1.0e-3
```

必须导致：

- construction 或 centering失败；
- 详细坐标正确；
- receipt fail closed。

## 17.5 Algebra ledger 完整性

- 每个 subcheck 都有 count、max error、bound、argmax；
- 不能只保存 aggregate；
- JSON 拒绝 NaN/Infinity；
- recomputation 与 receipt 完全一致；
- 修改任一 observation 后 fingerprint 失效。

## 17.6 主线不变

v22 与 v23：

```text
parameter names identical
parameter shapes identical
parameter count identical = 64,064
public input identical = (feature, occupancy)
one field
threshold 0
hard union
additional heads = 0
additional branches = 0
```

## 17.7 Gradient path

继续验证：

- initial scalar readout gradient 非零；
- upstream initial gradient 按冻结设计允许 dormant；
- readout-to-upstream cross-gradient finite/nonzero；
- 开始更新后所有参数可参与；
- 不因 factored ordering 断梯度。

---

# 18. 执行顺序

```text
R23-0
保持 v22 结果永久封存
        │
        ▼
R23-1
实现 FACRE core 与 algebra ledger
        │
        ▼
R23-2
运行 scalar counterexample tests
运行 formal-shape FP32 stress
        │
        ▼
R23-3
运行完整 dataset-free gate
        │
        ├── FAIL：停止，不读取 D_R
        │
        ▼ PASS
R23-4
生成 implementation closure
独立复核全部 error bounds
        │
        ▼
R23-5
创建唯一 D_R read-only authorization
        │
        ▼
执行 v23 D_R structural gate
        │
        ├── FAIL：封存 v23，不运行 bounded-400
        │
        ▼ PASS
R23-6
创建 bounded-400 authorization
fresh seed-42 / empty optimizer
        │
        ▼
只有 bounded-400 全门禁通过
才允许 Formal-800
```

---

# 19. v23 D_R PASS 条件

所有旧结构门禁继续保留。

Check 05 必须满足：

```text
all states inspected
all phase reconstruction bounds pass
all phase centering bounds pass
all common constructions pass
all specific constructions pass
all hidden recomputations pass
all energy recomputations pass
all fields finite
no missing subcheck ledger
no percentile / skipped element
no observed-value tolerance
legacy cancellation mismatch is diagnostic only
```

同时：

```text
all target groups have residual/flip/latent witness
exact target-positive latent collision = 0
zero-readout anchor = true
fixed-readout witness = true
PMOPE gradient path = true
field direction = true
model/cache/population/RNG preserved
optimizer steps = 0
D_V / D_T unread
```

---

# 20. v23 失败后的决策树

## 20.1 Reconstruction 或 centering 仍失败

结论：

> phase mean/residual projection本身在正式状态上不满足冻结数值界。

下一版本才考虑：

- compensated phase reduction；
- fixed contrast basis；
- pairwise/Haar phase projection。

不能继续扩大 bound。

## 20.2 Forward construction 失败

若：

```text
specific != common + residual
```

则是实现错误，直接修复代码并建立新版本；不得授权训练。

## 20.3 Hidden/energy oracle 失败

检查：

- device/backend；
- deterministic mode；
- dtype drift；
- hidden/energy stored tensor是否来自其他表达式；
- TF32/compile 是否介入。

仍不得训练。

## 20.4 Algebra PASS，但 witness check 06 失败

结论：

> 数值代数已经闭合，但真实 \(D_R\) 中 PACRE residual/flip latent 不具备
> target/background 可分 witness。

这才是 PACRE 表示层的真实结构负结果。

下一版本应修改 representation，而不是继续改数值 gate。

## 20.5 D_R 全部 PASS，bounded-400 失败

此时才能讨论：

- completion quality；
- compactness；
- factual recovery；
- FA；
- pair objective；
- PACRE 的真实训练效果。

---

# 21. 明确禁止

## 21.1 不修改 v22

不覆盖：

- v22 code；
- v22 receipt；
- v22 decision；
- v22 source closure。

## 21.2 不把 v22 重判为 PASS

v23 PASS 也只说明：

```text
FACRE-v23 PASS
```

不改变：

```text
PACRE-v22 FAIL
```

## 21.3 不直接运行 bounded-400

v23 必须先：

```text
dataset-free PASS
D_R structural PASS
```

## 21.4 不使用结果驱动 tolerance

禁止：

- 读取 D_R 最大误差后乘 2；
- 使用 99.9 percentile；
- 允许少量失败元素；
- 为 target/background 设置不同 tolerance；
- CPU 失败后切 CUDA 挑结果。

## 21.5 不同时改模型效果机制

v23 不修改：

- PMOPE；
- training schedule；
- feature adapter；
- field threshold；
- binary flip；
- phase transport；
- width/kernel；
- optimizer；
- hard union。

否则无法判断 check 05 的修复是否来自数值闭合。

---

# 22. 对“使其通过门槛”的诚实判断

FACRE-v23 能够有针对性地消除当前最可能的失败原因：

1. residual 显式进入 compatibility forward；
2. 不再用共同项消去作为核心合法性判据；
3. error bound 随实际运算尺度变化；
4. dataset-free 覆盖 formal tensor 尺寸；
5. receipt 能定位每一个子式和坐标；
6. 参数、主线和推理不变。

因此，它显著提高 check 05 正确通过的概率。

但是在新的真实 \(D_R\) gate 运行前，不能承诺：

- 所有其他门禁仍必然通过；
- PACRE/FACRE 会通过 bounded-400；
- 模型真实检测性能成功。

当前最准确的预期是：

> **v23 应先把“数值验证 false negative”与“真实表示不成立”严格分离。若 v23 的新代数 gate 通过而其他结构门禁继续通过，才有资格启动 bounded-400。**

---

# 23. 最终建议

下一步建立：

> **CURE-Lite v23 FACRE**

核心修改只有两部分：

### A. Forward algebra

```text
specific = occupancy + phase
```

改为：

```text
common   = occupancy + phase_mean
specific = common + phase_residual
```

### B. Verification algebra

删除 gate-eligible 的：

```text
specific - common ≈ residual
with fixed rtol/atol
```

改为：

```text
specific == forward_construct(common + residual)
phase centering uses reduction-aware backward error
hidden/energy use scale-normalized recomputation
legacy subtraction is diagnostic only
```

保持：

```text
(F_b, O)
PAET
binary flip
64,064 parameters
single scalar field
phi < 0
hard union
zero training in D_R gate
```

在 v23 D_R gate 正式通过之前：

```text
bounded-400  BLOCKED
Formal-800   BLOCKED
D_V / D_T    UNREAD
Full CURE    BLOCKED
cross-backbone BLOCKED
```

---

# 24. 主要证据与源码

## 仓库快照

- https://github.com/Arialliy/cure-lite/commit/9f43815525a7e207e9a550dacf894f88ace80c5b

## PACRE-v22

- https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v22/pacre.py
- https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v22/dataset_free.py
- https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v22/dr_gate.py
- https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v22/factory.py
- https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v22/training.py
- https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v22/decision.py
- https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v22/bounded_runner.py

## 继承主线

- https://github.com/Arialliy/cure-lite/blob/main/cure_lite/coverage_state_phase_aligned_evidence_transport.py
- https://github.com/Arialliy/cure-lite/blob/main/cure_lite/coverage_state_binary_flip_antisymmetric.py
- https://github.com/Arialliy/cure-lite/blob/main/cure_lite/coverage_state_phase_preserving.py
- https://github.com/Arialliy/cure-lite/blob/main/cure_lite/coverage_state_level_set.py

## 测试

- https://github.com/Arialliy/cure-lite/blob/main/tests/test_cure_lite_v22_pacre.py
- https://github.com/Arialliy/cure-lite/blob/main/tests/test_cure_lite_v22_pacre_dr_gate.py
- https://github.com/Arialliy/cure-lite/blob/main/tests/test_cure_lite_v22_pacre_dataset_free.py

## PyTorch 数值语义

- `torch.allclose`  
  https://docs.pytorch.org/docs/stable/generated/torch.allclose.html
- Numerical accuracy  
  https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- `torch.finfo`  
  https://docs.pytorch.org/docs/stable/type_info.html
