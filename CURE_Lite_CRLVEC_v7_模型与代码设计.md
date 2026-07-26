# CURE-Lite CR-LVEC v7：模型与代码设计

> 方法名：Continuously-Recoverable Log-Vacancy Evidence Crossing  
> 简称：CR-LVEC v7  
> 当前状态：单机制模型候选，先完成代码与 toy 门禁  
> 证据边界：只使用冻结的 \(D_R\) 设计证据；不得读取 \(D_V/D_T\)

## 1. 阶段位置与不变主线

研究主线保持为：

```text
CURE-Lite 模型设计与验证
    -> 冻结后的确认实验
    -> Full CURE
    -> 接入不同 IRSTD detector
    -> NUAA / NUDT-SIRST / IRSTD-1K 验证
```

CR-LVEC v7 仍属于 CURE-Lite 模型本体，不是 Full CURE，也不是附加到
MSHNet、SCTransNet 或其他特定 detector 上的模块。

本版本只允许新增独立的 v7 config、decoder、toy evaluator、测试和
指纹化产物。以下内容保持冻结：

- 任意 IRSTD Base adapter 的 \((p_b,F_b)\) 输出契约；
- Base 权重与 Base forward；
- shared trunk、baseline head、evidence head 和 PixelShuffle 拓扑；
- clean-positive、component-null、factual-miss、factual-no-miss 人口；
- OC-APTO loss、每次训练的 batch 组成和 optimizer step；
- 单次 decoder 推理、pre-mask、residual mask 和 hard-union；
- 真实 \(D_R\) bounded 的 400-update 预算和既有数值门槛；
- \(D_V/D_T\)、校准、Pd/FA、formal 800 和其他 detector 均未授权。

## 2. v6 负结果给出的结构问题

PR-SVEF v6 使用：

\[
z(F,O)=B(F)+E_{\mathrm{PR}}(r(F))A(O),
\]

\[
A(O)=
\operatorname{Nearest}
\left[
\frac{1}{1+C(O)}
\right],
\qquad
C(O)=
\mathbf 1_{3\times3}*
\operatorname{ProjectMax}(O).
\]

同一 pair 的两个端点共享完全相同的冻结特征，因此：

\[
\Delta z
=
z(F,O^-)-z(F,O^+)
=
E_{\mathrm{PR}}(r(F))
\left[A(O^-)-A(O^+)\right].
\]

这是一种可分离的幅值调制。提高 \(E_{\mathrm{PR}}\) 会同时增强 clean
response 和具有相似局部特征的 component-null response。冻结的真实
\(D_R\) 400-update 结果与此一致：

| 指标 | v6 最终值 | 冻结门槛 |
|---|---:|---:|
| clean \(D\) macro mean | 0.133460 | \(\geq0.5\) |
| clean pair 中 \(D\geq0.25\) 的比例 | 0.097087 | \(\geq0.75\) |
| component-null \(H\) macro mean | 0.075819 | \(\leq0.05\) |
| component-null \(H\) global max | 0.790334 | \(\leq0.25\) |
| clean \(D/H\) 联合通过率 | 0.029126 | \(\geq0.75\) |

v6 已完成 400/400 次更新，所有参数梯度均有限且非零，factual 分支也有
下降。因此 v7 不增加训练步数、不放宽门槛，也不再次只修复“有没有梯度”。
v7 必须改变 occupancy 与 feature evidence 的耦合方式。

## 3. 为什么不能缩成逐像素 occupancy 差分

一个更简单的候选是：

\[
p(F,O)=p_{\mathrm{occ}}(F)+(1-O)Q(F).
\]

它具有：

\[
p(F,O^-)-p(F,O^+)
=(O^+-O^-)Q(F),
\]

但其可达区域只等于被删除的预测组件。冻结 \(D_R\) clean population 的
只读统计为：

| 可达区域 | 完整覆盖的 pair | 覆盖的 \(D\) 像素 |
|---|---:|---:|
| 被删除组件 | 67/206 | 2101/2551 = 82.36% |
| 直接投影变化区域 | 153/206 | 94.98% |
| 当前 \(3\times3\) count 变化区域 | 206/206 | 100% |

clean \(D\) 表示删除一个已匹配预测组件后变成未覆盖的完整 GT 区域，并不
要求 \(D\) 等于该预测组件。逐像素公式会让 139/206 个 pair 至少有部分
监督在结构上不可达。即使它可能通过某个聚合门槛，也属于为了更容易的
当前测试缩小学习对象，不能作为本项目的 v7 主线。

因此 v7 保留无参数的 \(3\times3\) occupancy-count 可达域，但删除
`evidence × reciprocal vacancy` 这一可分离合成。

## 4. 唯一新机制：连续可恢复的占用负担越界

### 4.1 解析占用负担

沿用冻结的 occupancy projection 和 \(3\times3\) local count：

\[
C(O)=
\mathbf 1_{3\times3}*
\operatorname{ProjectMax}(O).
\]

把原来的 reciprocal vacancy 改写为同一个量的对数负担：

\[
b(O)
=
\operatorname{Nearest}_{H,W}
\left[\log(1+C(O))\right].
\]

\(b(O)\) 没有可训练参数、温度、阈值或搜索系数。删除 occupancy 时：

\[
O^-\subseteq O^+
\Rightarrow
C(O^-)\leq C(O^+)
\Rightarrow
b(O^-)\leq b(O^+).
\]

### 4.2 feature evidence 与负担的非可分离耦合

evidence head 仍产生 raw evidence \(r(F)\)，但不再先独立激活后乘以
vacancy。定义 crossing margin：

\[
u(F,O)=r(F)-b(O).
\]

定义 evidence-to-occupancy ratio 的零中心形式：

\[
q(u)=\operatorname{expm1}(u)=e^u-1.
\]

推理前向证据为：

\[
f(u)=
\begin{cases}
\operatorname{expm1}(u),&u>0,\\
0,&u\leq0.
\end{cases}
\]

由于 \(u=r-\log(1+C)\)，该式等价于：

\[
f(u)
=
\max
\left(
\frac{e^{r(F)}}{1+C(O)}-1,
0
\right).
\]

因此 evidence 只有在 feature evidence scale 超过局部 occupancy capacity
时才开启；这不是新增的可调阈值，而是 ratio 的解析单位边界。

最终 logit 为：

\[
\boxed{
z(F,O)=B(F)+f_{\mathrm{CR}}(r(F)-b(O))
}.
\]

这把 v6 的“证据幅值调制”改成了“状态越界”：

- clean response：删除正确组件后负担下降，margin 可以从关闭侧跨到开启侧；
- component-null：如果其 raw evidence 不足以越过删除后的负担，两端都严格关闭；
- identity pair：两个端点的 margin 完全相同，输出逐位一致；
- count 未变化的位置：两个端点的负担完全相同，输出逐位一致；
- occupancy 删除只会降低负担，前向输出保持单调不减。

该机制仍由同一个 feature head 学习“哪些冻结特征值得恢复”，但 occupancy
不再只是对所有正 evidence 做同向缩放，而是参与决定证据状态是否开启。

### 4.3 连续恢复的反向规则

仅使用“负侧恢复、正侧真实梯度”的分段 surrogate 会在 crossing 零点形成
反向不连续。CR-LVEC 使用：

\[
s(u)=e^u
\]

作为全轴 recovery carrier：

\[
f_{\mathrm{CR}}(u)
=
\operatorname{sg}[f(u)]
+s(u)
-\operatorname{sg}[s(u)].
\]

前向严格等于 \(f(u)\)，反向在整个实数轴为：

\[
\frac{\widetilde{\partial f_{\mathrm{CR}}}}{\partial u}
=e^u.
\]

它始终为正，并在 \(0^-\)、\(0\) 与 \(0^+\) 连续取值 1。删除端满足
\(u^-\geq u^+\)，因此：

\[
e^{u^-}\geq e^{u^+},
\]

clean endpoint difference 在零点附近不会被左右梯度不连续推回错误方向。

使用显式 `exp(u)` carrier 而不是依赖 `expm1(u)` 的自动反向非常关键：
float32 中较负的 `expm1` 会先舍入为 -1，部分实现随后可能返回错误的零
梯度；显式 `exp` 保留冻结公式要求的负侧恢复梯度。

该反向规则是同一个 ratio-crossing 原子算子的训练定义，不是新增 loss、
模块或第二条网络路径。测试必须分别验证前向值和声明的 surrogate
gradient；普通有限差分不能验证负半轴 surrogate。

`expm1` 对小正 margin 避免 \(e^u-1\) 的相消，但大正 margin 仍可能溢出，
极负 margin 的 `exp` 也可能下溢为零。实现不得通过事后 clamp 改写公式；
必须在算子边界执行一次合并检查，非有限 continuation 或零 recovery 直接
令该次代码门禁失败，并在 toy/bounded 产物中记录观察到的 margin 范围。

## 5. 完整网络结构

共享 trunk 与两个 subpixel heads 保持不变：

\[
T_0=\operatorname{SiLU}[\operatorname{GN}(W_s*F)],
\]

\[
T=T_0+\frac12W_p*
\operatorname{SiLU}[\operatorname{GN}(W_d*T_0)],
\]

\[
\widetilde b=\operatorname{PixelShuffle}_s(W_B*T),
\qquad
r=\operatorname{PixelShuffle}_s(W_E*T).
\]

baseline 保持：

\[
B(F)=
-\operatorname{softplus}
\left(\beta_{\mathrm{raw}}+\widetilde b(F)\right)<0.
\]

v7 只替换最后的无参数合成：

```text
F -> shared trunk -> baseline head -> PixelShuffle -> B ----------------┐
F -> shared trunk -> evidence head -> PixelShuffle -> raw r ----┐       │
O -> ProjectMax -> 3x3 count -> log1p -> nearest -> burden b ---(-)      │
                                                                │       │
                                              CR crossing f(r-b) -------(+)
                                                                        │
                                                                        v
                                                                      logits
```

参考配置保持：

```text
feature_channels = 64
feature_stride = 4
width = 32
groups = 8
trunk_residual_scale = 0.5
baseline_probability = 0.1
vacancy_kernel_size = 3
trainable_parameter_count = 4,385
```

v7 新增：

- trainable parameters：0；
- convolution/normalization/head modules：0；
- loss terms：0；
- inference branches：0；
- tunable thresholds：0。

## 6. 训练目标和检测流程不变

训练仍近似为：

\[
\mathcal L=
\mathcal L_{\mathrm{factual\ miss}}
+\mathcal L_{\mathrm{factual\ no\ miss}}
+\mathcal L_{\mathrm{OC\mbox{-}APTO}}.
\]

每次更新仍使用：

- factual-miss states：4；
- factual-no-miss states：4；
- paired endpoints：2 pairs \(\times\) 2 endpoints；
- decoder forward：3 次；
- decoder state evaluations：12；
- backward：1 次；
- optimizer step：1 次。

单图推理仍是：

```text
image
  -> 任意冻结 IRSTD Base adapter
  -> Base probability p_b 与 feature F
  -> O = 1[p_b >= fixed pre-mask threshold]
  -> CR-LVEC decoder(F, O)，只执行一次
  -> residual probability
  -> residual mask 屏蔽 O
  -> final mask = O OR residual mask
```

ground truth、pair kind、completion field 和 label increment 均不进入
decoder。它们只在训练 loss 中使用。

## 7. 代码门禁

在运行任何真实 \(D_R\) 训练前，独立 v7 实现必须全部证明：

1. config 固定拓扑、算子语义和 4385 参数参考值；
2. forward crossing 在 \(u\leq0\) 精确为零，在 \(u>0\) 等于
   `expm1(u)`；
3. surrogate gradient 在全轴等于 \(e^u\)，零点左右连续且取值 1；
4. occupancy burden 精确等于 `nearest(log1p(3x3_count))`；
5. occupancy 删除时 logits/probability 逐位不减；
6. identity endpoints 逐位一致；
7. local count 未变化位置的 endpoint difference 逐位为零；
8. feature 在 decoder 内 detach，Base 不接收梯度；
9. 4385 个参数均获得有限梯度；
10. 2B batched endpoints 与两个单独 forward 一致；
11. single-inference、pre-mask 和 hard-union 与冻结模型图一致；
12. `expm1` 非有限值必须 fail-fast，禁止隐藏 clamp；
13. v7 不得先计算 reciprocal vacancy 再丢弃，burden 路径只做一次
    projection、count、`log1p` 和 nearest lift；
14. v4/v5/v6 与旧路径回归保持通过。

## 8. 联合 toy 门禁

第一组沿用冻结的 1/2/3 像素 toy、seed、optimizer、320 updates 和全部
既有阈值。每个 case 必须同时满足：

\[
\mathrm{clean}\ D\geq0.8,
\quad
\max|\mathrm{clean}\ H|\leq0.05,
\]

\[
\max|\mathrm{component\ null}\ H|\leq0.05,
\]

以及 factual-miss、factual-no-miss、plus-anchor、background 和双端梯度
门槛。

第二组必须增加“\(D\) 同时超出被删除组件和 direct projected-change
区域、但位于 \(3\times3\) count-change halo 内”的 1/2/3 像素 case。
该组用于同时阻止模型缩小到 full-resolution component support，或缩小到
只能覆盖 153/206 个真实 pair 的 direct projected support。只有这一组仍
可学习，toy 才实际消费了选择 \(3\times3\) count 的完整可达域。

在正式 proposal/toy freeze 之前进行过同预算内部公式筛选。该筛选只使用
合成 toy，不访问真实 \(D_R\) 训练结果，也不读取 \(D_V/D_T\)。旧的
squared-softplus crossing 在六个 case 上均未达到冻结的 clean
\(D\geq0.8\)；逐像素 occupancy-difference 只通过组件内 case，而在第二组
case 上的 clean \(D\) 严格为零。最终冻结的 exponential ratio crossing
必须在正式 evaluator 中从头独立重放，内部筛选输出不能替代正式证据。

必须执行至少两次独立进程重放并比较规范化结果。任一 case 失败即冻结 v7
toy 负结果，不允许增加更新次数、改变 seed、放宽阈值或自动重试。

## 9. 真实 \(D_R\) 的后续边界

只有代码门禁和两组 toy 全部通过后，才允许：

1. 新增独立 v7 bounded executor 和 CLI；
2. 绑定 v6 negative closure、冻结 pair catalog、loss、schedule 和门槛；
3. 先完成 focused tests 与全量回归；
4. 生成 create-only、exact-run-count=1 的运行凭据；
5. 仅在 GPU0 温控下运行一次真实 \(D_R\) 400-update。

若 bounded 任一冻结门槛失败：

- 冻结为 v7 负结果；
- 不重跑 v7；
- 不启动 formal 800；
- 不读取 \(D_V/D_T\)；
- 不接入其他 IRSTD detector。

若 bounded 全部门槛通过，也只获得进入冻结审查的资格。随后才决定是否运行
seed 42/43 的 800-epoch 性能训练。

## 10. 创新性与主张边界

CR-LVEC 的模型贡献候选是：

> 将冻结检测器后的 residual correction 从可分离的
> feature-evidence/vacancy 幅值调制，重参数化为由解析 occupancy burden
> 控制、在 log 域比较 feature evidence scale 与 occupancy capacity 的
> 可恢复 ratio crossing。

它不是模块堆叠，但当前仍只是待验证的模型假设。toy 或 bounded 通过只证明
代码中的学习对象具有相应计算行为；在完成最新相关工作检索、正式性能实验、
多种子和跨 detector 验证前，不声明 ICLR 新颖性、优越性或稳定性。
