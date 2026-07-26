# CURE-Lite PB-NAES v9：Dataset-Free Toy 负结果

## 正式结论

PB-NAES v9 已完成冻结的 dataset-free 代码门禁：

```text
decision=PB_NAES_V9_TOY_GATE_FAIL
passed_cases=0/6
passed_families=0/2
real_D_R_authorized=false
formal_800_authorized=false
```

这是模型学习门禁负结果，不是实现异常，也不是检测性能结果。权威产物：

- [toy result](protocols/IRSTD-1K/phase_balanced_null_anchored_evidence_surplus_v9/toy_gate_result.json)
- [implementation closure](protocols/IRSTD-1K/phase_balanced_null_anchored_evidence_surplus_v9/toy_implementation_closure_receipt.json)
- [negative closure](protocols/IRSTD-1K/phase_balanced_null_anchored_evidence_surplus_v9/toy_negative_closure_receipt.json)

## 实现门禁

以下项目全部通过：

- phase-balanced implicit-null 方程逐元素一致；
- uniform-zero、uniform-negative、uniform-positive、stride-1 和
  wrong-winner 反例；
- inactive phase 恢复梯度；
- occupancy 删除单调与局部上界；
- Base feature detach；
- 两个 endpoint 同时具有梯度；
- paired endpoint 使用一次 \(2B\) forward；
- 6 个参数 tensor、toy 配置 2,593 参数；
- 320/320 updates 中所有参数梯度有限且非零；
- v4 拓扑、初始化、PixelShuffle、resize 和 hard-union 不变；
- `tests_v9`: 43 passed。

因此 0/6 不能归因于代码未连接、梯度中断或训练没有执行。

## 六个固定 case

| family | case | clean \(D\) | clean \(H_{\max}\) | component \(H_{\max}\) | 判定 |
|---|---|---:|---:|---:|:---:|
| component contains response | 1 pixel | 0 | 0 | 0 | fail |
| component contains response | 2 pixels | 0 | 0 | 0 | fail |
| component contains response | 3 pixels | \(7.95\times10^{-8}\) | 0 | 0 | fail |
| response outside component | 1 pixel | 0 | 0.837675 | 0 | fail |
| response outside component | 2 pixels | 0 | 0.069947 | 0.100120 | fail |
| response outside component | 3 pixels | 0 | 0.063184 | 0.554480 | fail |

冻结门槛要求每个 case：

\[
\operatorname{mean}_{D}(\Delta p)\ge0.8.
\]

六个 case 均未满足，均值或其他已通过检查不得覆盖这一失败。

## 结构解释

PB-NAES 的 phase active set 只由 feature 决定，occupancy 只通过：

\[
\frac{1}{1+C(O)}
\]

缩放已经选中的证据。因此 plus/minus 两个 endpoint 使用相同 active set。
在当前 absolute factual branch 和 probability-difference response risk 下，
模型可以把两个 endpoint 一起推向高置信，再通过相近的饱和值获得很小的
概率差。PB-NAES 提供了有效的前向 null 状态，但没有改变这一 paired
optimization 退化路径。

所以本结果否定的是：

> 在现有 paired objective 不变时，仅以 phase-balanced null surplus
> 替换 v8 evidence allocation，足以得到目标 endpoint transition。

不允许为 v9 增加步数、修改学习率、降低阈值或再次运行。

## 下一项模型代码

下一候选不再排列 decoder 激活函数。保持 v8 decoder 前向方程、4,385
参数、单次推理和 hard-union 不变，只重新定义 paired response risk：

\[
\mathcal L_D^{\mathrm{cross}}
=
\frac12
\left[
\operatorname{softplus}(z^+)
+
\operatorname{softplus}(-z^-)
\right].
\]

它明确监督：

```text
covered plus endpoint   -> residual low
uncovered minus endpoint -> residual high
```

而不是只监督两端概率差。\(H/G\) invariance、plus anchor、factual branches、
batch、optimizer 和训练预算保持不变。这是下一项待冻结的单机制候选，
目前尚未写入正式生产代码，也未获得真实 \(D_R\) 运行授权。
