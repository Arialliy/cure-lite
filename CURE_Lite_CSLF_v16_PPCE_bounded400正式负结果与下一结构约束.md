# CURE-Lite CSLF v16 PPCE bounded-400 正式负结果与下一结构约束

## 1. 正式结论

v16 的唯一正式运行已经完整结束：

```text
run_id:
cure_lite_cslf_v16_ppce_support_oriented_bounded_400_r1

decision:
BOUNDED_PPCE_SUPPORT_ORIENTED_CSLF_GATE_FAIL

failed_check:
candidate_original_zero_level_gates
```

这是一项完整且有效的结构负结果，不是训练中断或实现异常。

| 项目 | 正式状态 |
| --- | --- |
| PPCE 核心模型 | 已实现 |
| PPCE dataset-free gate | 通过 |
| 3 seeds × 3 objectives 短训练 | 通过 |
| bounded-400 三目标公平训练 | 完成 |
| bounded candidate gate | 失败 |
| `COMPLETE.json` | 已生成 |
| 17 项运行产物 | 完整 |
| `.incomplete` | 已移除 |
| `FAILURE.json` | 不存在 |
| \(D_V/D_T\) | 未读取 |
| Formal-800 | 未授权 |
| Full CURE | 未授权 |
| 其他 backbone / 三数据集 | 未授权 |

当前数值属于 \(D_R\) 结构机制门禁，不是 Pd、Fa、IoU 等正式检测
性能。

## 2. v16 的唯一结构改动

v16 没有增加 head、branch、decoder stage、后处理、loss 项、阈值或
训练步数。它仅将有损 occupancy 输入

\[
\bar O=\operatorname{MaxPool}_s(O)
\in\{0,1\}^{1\times h\times w}
\]

替换为无损相位表示：

\[
U_s(O)=\operatorname{PixelUnshuffle}_s(O)
\in\{0,1\}^{s^2\times h\times w}.
\]

模型仍为单一路径：

\[
H_0=
\operatorname{SiLU}
\left(
W_{3\times3}*[N(F_b),U_s(O)]
\right),
\]

\[
H=
H_0+
\operatorname{SiLU}(DW_{3\times3}*H_0),
\]

\[
\phi=
\operatorname{PixelShuffle}_s(W_{1\times1}H).
\]

正式配置为 \(C=64,s=4,w=32\)，参数量由 19,536 增加到 23,856，
增加 4,320 个相位输入连接。

## 3. 代码与协议验证

运行前完成：

- PPCE phase roundtrip 与 16/16 phase-index 对齐；
- 输入 phase \(k\) 到输出 phase \(k\) 的对角连接验证；
- 相同 scalar projection、不同亚像素 occupancy 的状态区分；
- 单路径结构和精确参数量公式；
- target、SDF、integration measure 不变；
- SORR selector、null、fixed point、逆变换和有限梯度；
- seeds 42/43/44 × 三目标的 9 条短训练记录；
- 两次 dataset-free canonical bytes 完全一致；
- PPCE zero evaluator 使用完整 phase 输入标识，不再错误复用
  scalar-collision 状态；
- 全量测试 `244 passed, 1 skipped`。

冻结的 PPCE dataset-free receipt 标识为：

```text
3d0bf1c771966f04f96319bb3605d1aff90827843153ca8e89ba8965c5a79d2b
```

旧 SORR dataset-free receipt 保持：

```text
56f56912359c5b12e10110323f01aeced279c1934f04080e5fe473f82c4d7c35
```

因此，v16 结果不能归因为 phase 顺序错误、错误输出复用、训练目标
变化或历史路径回归。

### 3.1 phase-visible diagnostic-null 规范修正

正式 r1 使用的是冻结的 zero-level evaluator v2，原始产物和哈希保持
不变。后续代码审计发现，v2 对历史上属于 `scalar_hidden` 的
diagnostic pair 无条件要求：

```text
field_exact_equal
and completion_exact_equal
```

这个要求只在 scalar 表示下成立，因为 plus/minus 的实际模型输入完全
相同，属于 exact replay。PPCE 下两端的 phase 输入已经可区分，连续
field 可以不同；正确的 null 语义是：

```text
actual inputs distinct
completion_exact_equal
new_negative_components == 0
removed_footprint_negative_pixels == 0
invalid_completion_pixels == 0
```

因此当前 evaluator 已将 phase 协议显式升级为 v3：

```text
cure-lite-pp-cslf-zero-level-evaluation-v3
cure-lite-pp-cslf-zero-level-evaluation-config-v3
```

v3 receipt 会记录 `actual_inputs_equal` 和 `input_relation`，并使用
`diagnostic_null`，不再把当前 phase-visible 输入误写成
`scalar_hidden_diagnostic`。scalar v2 的 canonical payload 和既有
config fingerprint 保持逐字节兼容。

冻结 r1 中该 pair 的原始事实为：

```text
actual input fingerprints: distinct
field_exact_equal:         false
completion_exact_equal:    true
new_negative_components:   0
removed-footprint negative:0
invalid completion pixels: 0
```

所以在修正后的 v3 语义下，这一项应为 `1/1`，而不是 v2 记录的
`0/1`。但以下独立失败完全不受该修正影响：

```text
factual gate:       15/16
clean compact exact: 0/16
clean full gate:     0/16
outside completion:  88
```

因此 v16 的总体决定仍是
`BOUNDED_PPCE_SUPPORT_ORIENTED_CSLF_GATE_FAIL`。本次修正没有重写
r1、没有重训练、没有读取 \(D_V/D_T\)，也不能把 v16 改判为成功。
修正代码的定向测试为 `10 passed`，phase/runner 相邻回归为
`30 passed`；版本化修正后的完整扩展测试面为
`245 passed, 1 skipped`。

## 4. v16 candidate 的正式结果

candidate 为 PPCE 架构下的
`support_oriented_response_joint`。

| 门禁量 | v16 PPCE |
| --- | ---: |
| factual gate | 15/16 |
| factual recovered | 16/16 |
| factual target negative pixels | 315/335 |
| factual no-miss | 16/16 |
| clean response sign | 1395/1396 |
| clean added target negative | 149/149 |
| clean compact exact | 0/16 |
| clean component match | 16/16 |
| clean full gate | 0/16 |
| outside added-target completion | 88 |
| plus false islands | 2 |
| trainable component-null | 16/16 |
| identity-null | 16/16 |
| phase-exposed diagnostic null | r1-v2: 0/1；v3 语义复核: 1/1 |

唯一未通过的 factual 样本仍为 `XDU680`：

```text
57 / 75 = 76.00%
```

它仍被部分找回且命中 1/1 component，但没有达到冻结的 95% target
negative fraction。

16 个 clean-positive 状态全部将 added target 压到负场，但全部产生
target 外扩张，因此没有一个满足 exact compact support。

## 5. 与 v15、v15A、v15B 的关键比较

| 指标 | v15 | v15A | v15B SORR | v16 PPCE+SORR |
| --- | ---: | ---: | ---: | ---: |
| factual gate | 15/16 | 13/16 | 15/16 | 15/16 |
| factual recovered | 16/16 | 16/16 | 16/16 | 16/16 |
| factual target negative | 321/335 | 289/335 | 324/335 | 315/335 |
| clean response sign | 1396/1396 | 1396/1396 | 1395/1396 | 1395/1396 |
| clean added target negative | 6/149 | 125/149 | 145/149 | 149/149 |
| clean compact exact | 0/16 | 1/16 | 9/16 | 0/16 |
| clean component match | 3/16 | 15/16 | 15/16 | 16/16 |
| clean full gate | 0/16 | 1/16 | 8/16 | 0/16 |
| outside completion | 8 | 43 | 12 | 88 |
| plus false islands | 4 | 3 | 4 | 2 |

PPCE 对三个 objective 都产生了相同方向的变化：

| Objective | 结构 | added target negative | compact exact | outside |
| --- | --- | ---: | ---: | ---: |
| SORR | scalar | 145/149 | 9/16 | 12 |
| SORR | PPCE | 149/149 | 0/16 | 88 |
| identity | scalar | 4/149 | 0/16 | 0 |
| identity | PPCE | 50/149 | 0/16 | 6 |
| separable | scalar | 137/149 | 9/16 | 2 |
| separable | PPCE | 145/149 | 8/16 | 8 |

这说明 PPCE 确实增强了模型对 occupancy 变化的响应，并非“结构没有
生效”。问题在于增强的主要是 synthetic deletion 信号，而不是
factual-miss-like completion。

## 6. 机制解释

### 6.1 被否定的假设

v16 否定了以下假设：

> 当前失败主要来自 scalar max projection 丢失亚像素 occupancy
> phase；恢复 phase 后即可同时改善 target recovery 和 compactness。

结果恰好相反：

\[
\text{added-target coverage}\uparrow
\quad\text{但}\quad
\text{compactness}\downarrow,\ 
\text{outside completion}\uparrow,\ 
\text{factual coverage}\downarrow.
\]

因此，输入 phase 丢失不是当前主导瓶颈。

### 6.2 当前最强的结构性解释

synthetic deletion 仍保留原本已检目标的强 frozen feature，只改变
occupancy；factual miss 则通常对应弱、混淆或低置信 feature。

PPCE 将 occupancy 的精确删除形状暴露给模型后，synthetic 分支变得
更容易学习。最后一个 epoch 中，PPCE 相比 v15B：

```text
pair loss:          0.20853 -> 0.15421
factual-miss loss:  0.08599 -> 0.10388
factual-no-miss:    0.03516 -> 0.04144
gradient L2:        1.16723 -> 2.26309
```

即模型更好地拟合了 synthetic pair，却同时降低了 factual
completion 的质量。该结果与“synthetic 强特征捷径被进一步强化”
一致。

此外，hidden field 仍在 \(1/4\) 网格进行 \(3\times3\) 空间混合。
PPCE 只恢复相位，不改变 coarse receptive field 的扩散方式，因此
能够精确选择输出 phase，但不能保证零水平集只停留在目标支持内。

### 6.3 没有被否定的内容

本结果没有否定：

- 通用 frozen \((F_b,O)\) 接口；
- coverage-aware residual completion 问题；
- 单一 completion field 的研究方向；
- CURE-Lite 或 Full CURE 的全部可能设计。

它只否定了：

```text
absolute zero-level scalar CSLF
+ 当前 synthetic state
+ 仅恢复 occupancy phase
```

这一具体组合。

## 7. 冻结决定

根据预先写入 v16 协议的停止规则，当前必须：

- 冻结 v16 为有效结构负结果；
- 不运行 Formal-800；
- 不增加训练步数；
- 不调整 threshold；
- 不继续修改 root coordinate；
- 不在同一开发链中用 binary target 补救；
- 不接入其他 IRSTD backbone；
- 不启动三大数据集完整实验。

下一结构不能只是 PPCE 后再加一个局部模块。它必须改变当前模型利用
synthetic evidence 的方式，并同时限制 coarse spatial diffusion，
否则仍属于对已失败路径的局部修补。

## 8. 下一结构的不可妥协约束

下一候选必须同时满足：

1. 仍只消费冻结的 \((F_b,O)\)，不访问特定 backbone 内部层；
2. 仍输出一个 residual completion field，不新增多头投票或后处理；
3. synthetic 与 factual 必须共享同一种 evidence coordinate，不能
   让“已检目标残留强特征”成为容易的 synthetic-only 信号；
4. output-grid locality 必须由状态方程直接保证，不能只靠 loss
   期待 coarse \(3\times3\) 卷积学会紧致边界；
5. 不新增 loss 权重、阈值搜索、训练步数或校准自由度；
6. dataset-free 必须先构造“强 synthetic feature 与弱 factual-like
   feature”反例，并证明模型没有仅依赖强幅值；
7. \(D_R\) 小规模验证必须同时改善 factual 与 clean compactness，
   不能再接受一边改善、另一边明显恶化。

下一结构的具体状态方程应在完成本次负结果归因后单独冻结，不能边
训练边改变。

## 9. 正式产物

运行目录：

```text
runs/irstd1k_stage_a_seed42/
cure_lite_cslf_v16_ppce_support_oriented_bounded_400_r1
```

核心标识：

```text
COMPLETE fingerprint:
7eba70fc32f70411f915a1d63261c32ac814232613bc68868cd6bb441b5bf599

COMPLETE file SHA256:
c73bcd79848f1b57fbfdcf92ded572cde8dae08f81334b0b1eaded62952ba649

bounded result fingerprint:
e6c343bfee6fe22c08423d2f4e0045df231a73c0a7d24bdfdb6451dfe917f78d
```

源码封存：

```text
archive:
artifacts/source_closures/
cure_lite_cslf_v16_ppce_support_oriented_bounded_400_7eba70fc32f7.tar

archive SHA256:
ef0d38a487ee14ed48dc79817c7e61042d61ee7aaca30a7b95cb2fd165409c4c

source files:
181

manifest SHA256:
d36cf9450d5974eafa45926c45f35103015e61676576ce346bb2e7d23cec501f
```

## 10. 当前模型设计进度

| 阶段 | 状态 |
| --- | --- |
| 通用 frozen \((F_b,O)\) 接口 | 完成 |
| representation-neutral \(D_R\) population/cache | 完成 |
| 12-state fused training 与公平对照 | 完成 |
| scalar/phase observability | 完成 |
| scalar CSLF 系列验证 | 完成，负结果 |
| PPCE-CSLF 结构验证 | 完成，负结果 |
| 当前 absolute zero-level CSLF 路线 | 正式停止 |
| 下一单结构候选 | 待冻结 |
| CURE-Lite 最终模型 | 尚未完成 |
| Formal-800 / 三数据集 | 尚未授权 |
