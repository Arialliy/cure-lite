# CURE-Lite CC-SEA v8：模型与代码设计

> 方法名：Coverage-Conserving Subpixel Evidence Allocation  
> 简称：CC-SEA v8  
> 当前状态：新模型代码实现与版本隔离测试阶段  
> 边界：只建立 CURE-Lite；不启动 formal800、Full CURE 或其他 detector

## 1. 当前模型进度

CR-LVEC v7 已完成真实 \(D_R\) bounded400。其 31 项结构检查和 400 次
更新均正常，说明代码、梯度和训练路径成立；但模型门禁只通过 5/12：

- clean \(D\) 响应由 v6 的 0.133460 提高到 0.185189；
- component-null footprint mean 同时由 0.075819 增至 0.137834；
- clean \(D/H\) 联合通过率由 0.029126 降为 0。

因此 v7 是有效负结果：连续恢复改善了响应能力，但逐像素独立 crossing
把目标响应和邻域泄漏一起放大。v7 已冻结，不在同一版本内重试。

## 2. v8 的唯一机制

CC-SEA 不增加 attention、第二个 decoder、辅助 head 或新 loss。它只替换
v7 最后的 evidence composition 原子算子。

对一个 feature cell，evidence head 仍输出 \(P=s^2\) 个 subpixel phase：

\[
r_{k,1},\ldots,r_{k,P}.
\]

首先把同一个 head 的输出解析为两个正交坐标：

\[
\mu_k
=
\frac{1}{P}\sum_{j=1}^{P}r_{k,j},
\qquad
\widetilde r_{k,j}=r_{k,j}-\mu_k.
\]

\(\mu_k\) 是控制总预算的 common mode；零均值的
\(\widetilde r_{k,j}\) 只控制预算落点。共同增加所有 phase 只改变预算，
保持均值不变地拉开 phase 则只改变分配，避免为了选中一个子像素而同时把
两个 occupancy 端点都推到饱和。

occupancy 路径保持原来的无参数定义：

\[
C_k(O)
=
\mathbf 1_{3\times3}*\operatorname{ProjectMax}(O),
\qquad
b_k(O)=\log(1+C_k(O)).
\]

每个 cell 只产生一份 coverage-conditioned evidence budget：

\[
u_k=\mu_k-b_k(O),
\qquad
M_k=f_{\mathrm{CR}}(u_k).
\]

这里的 (f_{\mathrm{CR}}) 继承 v7 已经通过数值与梯度检查的 crossing
primitive；v8 不改写该 primitive，也不继承 v7 的逐像素合成。v8 的唯一
变化是把它作用于 common mode 产生一份 cell budget，再按零均值 phase
contrast 守恒分配。后续实现收据必须绑定该冻结 primitive 的源码哈希。

同一 cell 内的高分辨率位置必须竞争这份预算：

\[
\alpha_{k,j}
=
\frac{e^{\widetilde r_{k,j}}}
{\sum_{\ell=1}^{P}e^{\widetilde r_{k,\ell}}},
\qquad
e_{k,j}=M_k\alpha_{k,j}.
\]

由此得到精确的核心约束：

\[
e_{k,j}\ge0,
\qquad
\sum_{j=1}^{P}e_{k,j}=M_k.
\]

v7 对每个 phase 分别产生 evidence，删除一个 coarse occupancy cell 时最多
可把相同增益广播到 \(9P\) 个高分辨率位置。CC-SEA 中，每个受影响 cell
只有一份预算；给背景 \(H\) 的 evidence 会直接减少可分给目标 \(D\) 的
份额。competition 不是额外模块，而是守恒约束的数学结果。

最终仍为：

\[
z(F,O)=B(F)+\operatorname{PixelShuffle}(e(F,O)).
\]

## 3. 保持不变的网络

```text
frozen feature F_b
    -> 1x1 stem + GN + SiLU
    -> depthwise 3x3 + GN + SiLU
    -> pointwise 1x1 + residual
    -> baseline head -----------------------> baseline B
    -> shared evidence head -> P phases
                              |
occupancy O -> ProjectMax -> 3x3 count -> log burden
                              |
                     one conserved budget
                              |
                 phase competition/allocation
                              |
                         PixelShuffle
                              |
                         residual logits
```

Reference Base 的 \(C_f=64,s=4\) 情况仍为：

- 4,385 个可训练参数；
- 6 个可训练 tensor；
- 一个 shared trunk、一个 baseline head、一个 evidence head；
- Base 冻结；
- 原 OC-APTO loss、batch 和 optimizer budget；
- 推理时一次 Base、一次 CURE decoder、occupancy pre-mask 和 hard union。

不同 IRSTD detector 仍只通过 \((p_b,F_b)\) adapter 接入。stride 为 1
时 \(P=1\)，算子自然退化为单位置预算；stride 较大时，守恒约束处理
subpixel ambiguity。模型不读取 detector 名称或内部拓扑。

## 4. 创新点边界

本版本的创新候选不是“PixelShuffle + softmax”，也不是多个常见模块的
组合，而是：

> 将冻结 detector 后的漏检修正定义为 coverage-conditioned conserved
> evidence allocation：coverage release 决定局部总证据，subpixel
> positions 在固定总量内竞争，从同一状态方程同时约束 recovery 与局部
> leakage。

必须由代码和实验建立：

1. 非负性、局部守恒、删除单调、identity 恒等、count-support 外零差分；
2. 相同参数、loss 和预算下，v8 同时提高 \(D\) 并降低 \(H\)，而不是整体
   压低响应；
3. bounded 通过后，formal800 的 seed 42/43 分别提高 natural-miss
   recovery，并满足固定 FA/retention。

当前尚未做与最新相关工作的系统检索，因此只能称为有明确科学动机的创新
候选，不能提前宣称 ICLR 新颖性已经成立。

## 5. 分阶段代码路线

```text
v8 config + decoder
    -> operator / topology / conservation unit tests
    -> toy selectivity overfit
    -> 8-step dry run and complete gradient checks
    -> frozen v8 implementation receipt
    -> exactly one real D_R bounded400
    -> bounded pass 才能授权 formal800 seed42/43
```

v8 只新增独立文件，不修改 v4/v6/v7 文件和冻结结果。v8 测试位于
`tests_v8/`，避免改变已经封存的 v7 `tests/` inventory。

截至本文件生成时，只完成了第一版 config、decoder 和单元测试代码；尚未
产生 toy、bounded、Pd、FA 或 formal800 结果。
