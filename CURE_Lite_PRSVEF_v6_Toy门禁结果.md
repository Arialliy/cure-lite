# CURE-Lite PR-SVEF v6：Toy 模型代码门禁结果

## 1. 结论

PR-SVEF v6 在冻结的 1/2/3 像素 toy 上逐 case **3/3 通过**。这证明当前
单一前向/反向 evidence operator 具备进入真实 \(D_R\) bounded 代码实现
阶段的资格。

它还不能证明真实检测性能、稳定性或 ICLR 级方法结论，也不授权直接运行
formal 800 epoch。

| 项目 | 状态 |
|---|---|
| 设计、proposal、toy config | FROZEN |
| config/operator/decoder/model tests | PASS |
| frozen toy | PASS，3/3 |
| bounded executor/CLI 编写 | AUTHORIZED |
| 真实 \(D_R\) bounded 运行 | 尚未授权，需先完成 bounded 代码门禁 |
| \(D_V/D_T\) | NOT_ACCESSED |
| Pd/FA 性能 | NOT_EVALUATED |
| formal 800 | NOT_AUTHORIZED |

## 2. 冻结算子

前向：

\[
\phi(r)
=
\operatorname{softplus}
\left(
\operatorname{ReLU}(r)^2
\right)-\log2,
\qquad
r\leq0\Rightarrow\phi(r)=0.
\]

反向：

\[
\frac{\widetilde{\partial E}}{\partial r}
=
\begin{cases}
2r\sigma(r^2), & r>0,\\
\sigma(r), & r\leq0.
\end{cases}
\]

零点梯度固定为 0.5。算子没有新增模块、参数、head、loss、train step 或
推理分支；参考 decoder 仍为 4,385 个参数。

## 3. 固定运行契约

```text
seed = 7817
optimizer = Adam
updates = 320
learning_rate = 0.004
feature_channels = 8
feature_stride = 4
cases = one_pixel / two_pixels / three_pixels
```

toy config、阈值、case 和停止规则均在运行前冻结。三次独立进程重放生成
逐字节相同的 canonical JSON：

`protocols/IRSTD-1K/polarity_recoverable_subpixel_vacancy_evidence_factorization_v6/toy_gate_result.json`

- SHA256：
  `675856650ebee518c9290d82562bca662d521e049da4ebc87cc3b7f186d9cead`
- result fingerprint：
  `c8239340ad19b98d9cbc0aab01e1421971938bea065e0b4a575047146677f5d3`

## 4. 逐 case 结果

| case | total loss | clean \(D\) | plus target | factual target | component \(H\) max | 结果 |
|---|---:|---:|---:|---:|---:|---|
| one pixel | 0.001506 | 0.900137 | 0.999728 | 0.999399 | 0.004967 | PASS |
| two pixels | 0.001573 | 0.889203 | 0.999742 | 0.999132 | 0.003875 | PASS |
| three pixels | 0.001722 | 0.882475 | 0.999739 | 0.998900 | 0.001748 | PASS |

所有 case 还同时满足：

- plus background \(<0.001\)；
- factual miss background \(<0.001\)；
- factual no-miss \(<0.002\)；
- clean \(H=0\)；
- clean \(G<0.0003\)；
- component \(G<0.00003\)；
- plus/minus endpoint 初始梯度均有限且非零。

## 5. 相对 v5 的直接结果

v5 三个 case 的目标概率停留在约 0.5，原因是负初始化目标 phase 在负半轴
同时失去前向响应和梯度。v6 保持相同负半轴前向零语义，但通过冻结的恢复
梯度让这些 phase 能够跨过原点。相同 seed、预算、数据、loss 和阈值下，
三个 case 均通过。

这支持“v6 修复了 frozen toy 中的训练可达性问题”，但 toy 已参与候选
开发，不能将其写成独立科学验证或真实检测提升。

## 6. 下一步

现在只授权：

1. 新增 v6 独立 bounded config；
2. 新增 v6 bounded executor、CLI 与相应 tests；
3. 严格绑定 v4 的同一 \(D_R\) population、400 updates、loss、schedule 和
   计算门槛；
4. bounded 代码与输入门禁全部通过后，只运行一次真实 \(D_R\)。

在真实 bounded 结果产生前，不运行 \(D_V/D_T\)、Pd/FA、formal 800、
Full CURE 或其他 detector。
