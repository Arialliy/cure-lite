# CURE-Lite PB-NAES v9：模型与代码设计

> 方法名：Phase-Balanced Null-Anchored Evidence Surplus  
> 简称：PB-NAES v9  
> 当前状态：单方程候选的代码与 dataset-free 门禁阶段  
> 边界：仅设计 CURE-Lite；不读取新的 \(D_R/D_V/D_T\)，不启动
> formal800、Full CURE 或其他 detector

## 1. 为什么需要 v9

CC-SEA v8 已在唯一一次真实 \(D_R\) bounded400 中完整执行：

```text
structural_execution_pass=true
computational_model_code_gate_pass=false
passed_computational_gates=6/12
```

v8 将每个 feature cell 的证据总量固定为一份预算，再用稠密 softmax
分配给 \(P=s^2\) 个 subpixel phase。它降低了 clean-zero 和局部扰动，
但也产生了两个结构瓶颈：

1. 当预算尚未激活或很小时，phase 对比梯度被预算幅值一同压小；
2. 所有有限 softmax 权重都严格为正，并且整份预算必须被花完，模型没有
   “此处无需修复”的前向状态。

真实结果与这一结构判断一致：clean \(D\) 平均增量仅为 0.107294，
达到 \(D\ge0.25\) 的 clean pair 仅为 4.85%；component-null footprint
mean/max 虽低于 v7，仍分别为 0.083611/0.395187。

PB-NAES 不增加 head、attention、loss 或训练步数。它只替换证据 head
之后的一个原子状态方程，使“目标 phase 选择”和“无需修复”成为同一
相对证据判定的两个结果。

## 2. 唯一核心方程

对于一个 feature cell，原 evidence head 仍输出
\(P=s^2\) 个 raw phase 值：

\[
r_{k,1},\ldots,r_{k,P}.
\]

将每个值解释为正证据强度：

\[
x_{k,j}=\exp(r_{k,j}).
\]

PB-NAES 不新增可学习的 null logit，而是引入与 \(P\) 个 phase 总质量
平衡的固定 null 参考。固定 null 的单个强度为
\(\exp(0)=1\)，总参考质量为 \(P\)。phase 与 null 的局部平衡参考为：

\[
m_k
=
\frac{P+\sum_{j=1}^{P}x_{k,j}}{2P}
=
\frac{1+\overline{x}_k}{2}.
\]

occupancy 路径保持无参数且局部：

\[
C_k(O)
=
\mathbf 1_{3\times3}*\operatorname{ProjectMax}(O),
\qquad
v_k(O)=\frac{1}{1+C_k(O)}.
\]

每个 phase 的 signed surplus 为：

\[
z_{k,j}(F,O)
=
v_k(O)\left(x_{k,j}-m_k\right).
\]

最终证据采用“前向稀疏、反向可恢复”的同一个算子：

\[
e_{k,j}
=
\operatorname{CR}_{+}(z_{k,j}),
\]

其中：

\[
\operatorname{forward}\!\left(\operatorname{CR}_{+}(z)\right)
=
\max(z,0),
\qquad
\frac{\partial\operatorname{CR}_{+}(z)}{\partial z}=1.
\]

代码必须直接使用完整 signed carrier：

```python
forward = torch.clamp_min(signed_surplus, 0.0)
evidence = (
    forward.detach()
    + signed_surplus
    - signed_surplus.detach()
)
```

不能将现有 crossing primitive 乘在外部因子上并宣称等价；两者仅前向
可能等价，在 inactive phase 上的 Jacobian 并不相同。

最终网络仍为：

\[
z(F,O)
=
B(F)+\operatorname{PixelShuffle}(e(F,O)).
\]

## 3. 该方程解决什么

### 3.1 稀疏而非稠密广播

只有满足

\[
x_{k,j}>m_k
\]

的 phase 才有非零前向证据。其余 phase 逐元素精确为零，而不是获得一个
很小但必为正的 softmax 份额。

### 3.2 内生的“无需修复”状态

当全部 raw phase 为零时：

\[
x_{k,j}=1,\qquad m_k=1,\qquad e_{k,j}=0.
\]

当全部 phase 的强度低于固定 null 参考形成的平衡阈值时，整个 cell
同样可以在一个有限开区间内输出零证据。无需新增 null head，也不需要
从稀少的 component-null pair 中额外学习一个独立 null 参数。

### 3.3 不再把单个小目标稀释到 \(P\) 个位置

若一个 phase 的证据远强于其余 phase，则其正 surplus 接近自身强度，
而不是像 v8 那样先形成一份 common budget、再被 softmax 分配。固定
null 与 phase 总质量按 \(1:1\) 平衡，避免“有 \(P\) 个 correction
候选但只有一个 null 候选”造成的先验数量偏置。

### 3.4 保留多 phase 和 stride-1 表示能力

PB-NAES 不是 hard argmax。多个高于平衡参考的 phase 可以同时非零；
当全部 phase 都具有足够强的共同证据时，全部 phase 也可以同时输出。
因此它不强制每个 cell 只能修复一个像素，并且在 \(P=1\) 时仍非退化。

### 3.5 零前向仍可恢复

在 \(r=0\) 处，signed carrier 对 raw phase 的 Jacobian 为：

\[
J
=
v(O)
\left(
I-\frac{1}{2P}\mathbf 1\mathbf 1^\top
\right).
\]

其 contrast 特征值为 \(v(O)\)，common-mode 特征值为
\(v(O)/2\)，因此满秩。inactive phase、错误赢家以及全 null 初始化
都仍然具有有限恢复梯度。

## 4. 由方程直接得到的不变量

在有限输入下，必须逐项成立：

1. \(e_{k,j}\ge0\)；
2. raw phase 与 occupancy 不变时，输出严格恒等；
3. 删除 occupancy 只会增大 \(v_k(O)\)，因此每个 phase 的证据不减；
4. occupancy count 不变的 cell 上，两个 endpoint 的证据差精确为零；
5. phase active set 只由 frozen feature 路径决定，不随 paired endpoint
   的 occupancy 改变；
6. inactive phase 前向精确为零，但其 raw coordinate 仍有恢复梯度；
7. 总输出证据满足局部上界
   \[
   \sum_j e_{k,j}
   \le
   v_k(O)\sum_j x_{k,j}
   <
   v_k(O)\left(P+\sum_jx_{k,j}\right);
   \]
8. 不新增训练参数、参数 tensor、decoder forward 或推理分支。

这些是实现门禁，不是用实验均值替代的经验主张。

## 5. 保持不变的网络和训练

```text
frozen detector feature F_b
    -> 1x1 stem + GN + SiLU
    -> depthwise 3x3 + GN + SiLU
    -> pointwise 1x1 + residual
    -> baseline head ----------------------------> baseline B
    -> evidence head -> P raw phases
                         |
                         +-> exp intensity
                         +-> balanced implicit-null reference
                         +-> recoverable positive surplus
                         +-> PixelShuffle
occupancy O -> ProjectMax -> 3x3 count -> reciprocal vacancy
                                                  |
                                      residual logits B + evidence
```

Reference Base 的 \(C_f=64,s=4\) 仍为：

- 4,385 个可训练参数；
- 6 个可训练 tensor；
- 原 shared trunk、baseline head、evidence head 和初始化；
- Base 冻结，feature detach；
- 原 PairExample/PairBatch、clean/component-null catalog；
- 原 OC-APTO paired loss、4/4/2 update 和 optimizer；
- 一次 Base、一次 CURE-Lite decoder、pre-mask 和 hard union；
- 不读取 pair kind、GT、\(D/H/G\)、另一 endpoint 或 detector 名称。

因此，本候选仍通过统一 \((p_b,F_b)\) adapter 面向任意 IRSTD detector；
当前阶段不接入任何具体 detector。

## 6. 预先冻结的反例门禁

在任何 toy overfit 前，算子必须通过：

1. `uniform_zero_null`：\(r=0\) 时全部 phase 精确为零；
2. `uniform_negative_null`：共同负移存在有限全-null 区；
3. `uniform_positive_full_phase`：共同正移可使全部 phase 非零；
4. `stride_one_non_degenerate`：\(P=1\) 可产生零和正证据；
5. `wrong_winner_recovery`：目标 phase 初始 inactive 时，目标输出对该
   raw coordinate 的梯度有限且非零；
6. `multi_phase_capacity`：同 cell 的 1/2/3 个目标 phase 均可同时激活；
7. `peaked_component_null`：强错误 phase 不被结构上宣称为自动消除，
   必须进入随后的 component-null toy 作为显式可证伪项；
8. `endpoint_selection_invariance`：同一 feature 的 plus/minus active set
   完全一致；
9. `deletion_monotonicity_and_locality`：删除单调且 count-support 外严格
   零差分；
10. `topology_identity`：与 v4 初始 state、参数量和 module 拓扑相同。

其中第 7 项明确限制当前主张：PB-NAES 提供了可表示的 null 状态，但不在
训练前声称它已经学会区分 clean target 与 component-null。

## 7. 串行验证路线

```text
v9 config / operator / decoder
    -> equation and topology unit tests
    -> fixed 1/2/3-pixel toy selectivity
    -> exposure-matched 206:16 / 800-slot toy
    -> 8-step deterministic dry-run
    -> freeze implementation receipt
    -> exactly one real D_R bounded400
    -> 12/12 bounded gate 才能授权 formal800 seed42/43
```

toy 必须继续使用原 loss、step 和预算，不允许同时修改 loss 权重来帮助
新方程。真实 \(D_R\) bounded 仍使用 v8 已冻结的 12 项阈值，均值不能
覆盖单项失败。

当前只授权 dataset-free 代码与测试。相关工作新颖性尚未完成独立检索，
所以 PB-NAES 目前是结构明确、可证伪的创新候选，不是已经成立的 ICLR
论文结论。
