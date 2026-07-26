# CURE-Lite PECO v10：模型与代码设计

> 方法名：Paired Endpoint Crossing Objective  
> 简称：PECO v10  
> 当前状态：新 objective 的冻结设计阶段  
> 边界：保持 CC-SEA v8 decoder、Base、推理图和训练预算不变；不读取新的
> \(D_R/D_V/D_T\)，不启动 formal800、Full CURE 或其他 detector

## 1. 从 v8/v9 得到的模型问题

CC-SEA v8 在真实 \(D_R\) bounded400 中完成 400/400 updates，但只通过
6/12 门禁。它改善了 clean-zero 和局部扰动，却没有形成足够的 clean
endpoint transition：

\[
\overline{\Delta p_D}=0.107294,
\qquad
P(\overline{\Delta p_D}\ge0.25)=0.04854.
\]

PB-NAES v9 随后保持相同拓扑，只尝试通过稀疏 phase/null 前向改善这一
问题。它的方程、恢复梯度、双端反传和全部实现门禁通过，但固定 toy 为
0/6，clean \(D\) 全部接近零。因此 v9 已在 dataset-free 阶段停止。

两个结果共同指出：继续排列 evidence activation 不是当前最有依据的
修改。现有 response risk 只监督：

\[
\Delta p
=
\sigma(z^-)-\sigma(z^+)
\longrightarrow 1.
\]

它没有分别规定两个 endpoint 的状态。特别是，当 absolute factual branch
推动 \(z^-\) 变大时，\(z^+\) 也可能一起进入高置信区；sigmoid 饱和后，
两端都高但差值很小，response 梯度也随之衰减。

PECO 不再修改 decoder。它只将 response stratum \(D\) 的一个风险项替换
为显式的 paired endpoint crossing。

## 2. 唯一机制

保持同一个 completion function：

\[
Q_\theta(F,O)=z.
\]

每个 clean pair 仍由相同 feature 和两个 occupancy endpoint 构成：

\[
z^+=Q_\theta(F,O^+),
\qquad
z^-=Q_\theta(F,O^-).
\]

对 label increment 定义的 response stratum \(D\)，PECO 使用：

\[
\mathcal L_D^{\mathrm{PECO}}
=
\frac12
\left[
\operatorname{softplus}(z^+)
+
\operatorname{softplus}(-z^-)
\right].
\]

这等价于在同一个 response pixel 上同时施加：

```text
covered plus endpoint:     target 0
uncovered minus endpoint:  target 1
```

其梯度为：

\[
\frac{\partial\mathcal L_D}{\partial z^+}
=
\frac12\sigma(z^+)>0,
\qquad
\frac{\partial\mathcal L_D}{\partial z^-}
=
-\frac12\sigma(-z^-)<0.
\]

因此最小化时 plus logit 被降低、minus logit 被提高。两端同时高时，
\(z^+\) 仍有接近 \(1/2\) 的下降梯度；两端同时低时，\(z^-\) 仍有接近
\(-1/2\) 的上升梯度。旧 probability-difference risk 的共同饱和退化
不再是低风险解。

## 3. 其余 objective 完全不变

局部和全局零响应仍使用原 probability difference：

\[
\mathcal L_H
=
\operatorname{mean}_{H}(\Delta p^2),
\qquad
\mathcal L_G
=
\operatorname{mean}_{G}(\Delta p^2).
\]

两者仍按 active-stratum mean 组成：

\[
\mathcal L_0
=
\operatorname{ActiveMean}(\mathcal L_H,\mathcal L_G).
\]

response 与 zero risk 仍按每 pair 等权 active mean：

\[
\mathcal L_{\mathrm{transition}}
=
\operatorname{ActiveMean}(\mathcal L_D^{\mathrm{PECO}},\mathcal L_0).
\]

原 plus anchor 仍为：

\[
\mathcal L_{\mathrm{pair}}
=
\frac12\mathcal L_{\mathrm{plus\ anchor}}
+
\frac12\mathcal L_{\mathrm{transition}}.
\]

factual-miss、factual-no-miss 两个 absolute branch、4/4/2 batch、三次
decoder forward、12 个 state、Adam、学习率、schedule 和 updates 均不变。
PECO 不引入可调权重，也不读取 pair kind；当 \(D\) 为空时，response
group 自然 inactive，component-null pair 仍只使用相同的 zero risk。

## 4. 模型与推理结构

PECO v10 的训练模型为：

```text
frozen detector -> (p_b, F_b)
                       |
                  hard occupancy O
                       |
        frozen CC-SEA v8 decoder topology
        one shared trunk + two existing heads
                       |
         paired endpoints z+ and z- during training
                       |
             PECO response objective
```

推理仍为：

```text
one frozen Base forward
    -> one CURE-Lite decoder forward
    -> sigmoid residual
    -> occupancy pre-mask
    -> frozen threshold
    -> hard union
```

Reference Base 的 \(C_f=64,s=4\) 仍为：

- 4,385 个可训练参数；
- 6 个参数 tensor；
- Base 冻结且 feature detach；
- 不增加 head、attention、decoder、token 或推理分支；
- 不读取 GT、pair identity、pair kind 或另一 endpoint；
- 任意 detector 仍只需提供 \((p_b,F_b)\) adapter。

因此，v10 的唯一变化是训练时 \(D\) 的状态语义，不是模块堆叠。

## 5. 开发筛查与证据边界

在正式代码冻结前，使用原 6 个 dataset-free case 对精确公式做了一次
内存筛查。所有其他代码、seed、320 updates、学习率和阈值保持不变：

| family | 1 pixel | 2 pixels | 3 pixels |
|---|---:|---:|---:|
| component contains response | 0.993196 | 0.992387 | 0.992893 |
| response outside component | 0.992305 | 0.991908 | 0.991626 |

表中数值是 clean \(D\) probability delta；六个 case 的 \(H/G\) 与
component-null 检查也均通过。

这些数值只能证明 PECO 值得进入代码实现。因为同一 6-case toy 已参与
候选选择，后续正式重放只能作为实现回归，不能当作独立确认，更不能作为
检测性能。

## 6. 预先冻结的门禁

### 6.1 objective 单元门禁

1. response risk 与两端 BCE-with-logits 的平均逐元素一致；
2. plus gradient 严格为正、minus gradient 严格为负；
3. both-high、both-low 和 opposite-wrong 三种饱和反例均保留纠正梯度；
4. \(D=\varnothing\) 时不计算空 reduction，component-null 仍只走 zero risk；
5. \(H/G\)、plus anchor、hierarchical equal weighting 与旧实现逐位一致；
6. loss 不读取 pair kind、sample ID、detector 名称或推理信息；
7. 原 outcome train step 无修改即可消费 PECO；
8. 两个 endpoint 与全部 6 个 decoder parameter tensor 均有有限梯度。

### 6.2 已使用 toy 的实现回归

原 6 个 case 必须 6/6 逐 case 通过，但只记录为
`development_regression_pass`。

### 6.3 新的选择后确认

在 dry-run 或真实 \(D_R\) 前，必须冻结一个 exposure-matched dataset-free
population：

- 206 个 clean-positive 与 16 个 component-null role；
- 完整 800 pair slots；
- 每个 pair 仅有冻结的 3/4 次暴露；
- 至少 340/400 updates 不含 component-null，与真实 bounded schedule
  的稀疏程度一致；
- 固定 1/2/3-pixel 与 response-outside-component 几何混合；
- 结果必须逐 role、逐几何组报告，均值不得覆盖组失败；
- 增加 identical-input conflicting-outcome negative control，证明 loss
  没有通过 pair kind 或 batch 顺序获得不可用信息。

该确认的生成规则、seed、阈值和 exposure schedule 必须在运行前写入独立
配置。若确认失败，PECO 停止，不运行真实 \(D_R\)。

## 7. 串行代码路线

```text
PECO loss implementation
    -> exact formula / gradient / empty-stratum unit tests
    -> old 6-case development regression
    -> freeze exposure-matched synthetic confirmation
    -> confirmation pass
    -> 8-step deterministic dry-run
    -> exactly one real D_R bounded400
    -> 12/12 bounded gate 才能授权 formal800 seed42/43
```

真实 bounded 继续使用 v8 的同一 population、400 updates 和 12 项阈值。
不允许同时修改 decoder、component exposure、loss 权重或评估门槛。

PECO 的数学原子本身不是论文新颖性的充分条件。论文级创新候选是：

> 将冻结 IRSTD detector 的 residual completion 学习定义为 coverage-state
> paired endpoint crossing，并以 legal occupancy intervention 提供
> covered/uncovered 两个可执行状态，同时保持推理零额外分支。

这一主张仍需真实门禁、formal seed、独立确认和相关工作检索共同支持。
