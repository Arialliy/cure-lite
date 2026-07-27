# CURE-Lite NLCC-v12 正式负结果后的下一步修改方案

> 日期：2026-07-27  
> 证据基线：`NLCC_V12_DEVELOPMENT_FAIL`  
> 当前阶段：有效模型负结果；Holdout、真实 \(D_R\)、32,000-step replay、formal800 均未授权  
> CURE 主线：**detector-independent feature–coverage relevance learning 与受约束 evidence release**  
> 下一诊断候选：**CURE-Lite GCDE-v13（Gate-Covered Dual-Endpoint）**  
> 推荐原则：**保留 NLCC-v12 decoder、字节级相同输入和推理图，只检验极值敏感双端点训练代理是否优于 PECO-v10。GCDE 是中间诊断，不是 CURE 的核心模型创新。**
> 当前终态：**R13-0 已完成；R13-1 输入识别检查已运行并 FAIL。GCDE loss、Development authorization 与训练均未获授权。**

> 2026-07-27 后续状态：R13-1A 已完成完整 \(2\times2\times2\) 冲突归因；旧
> NLCC/GCDE 路线继续冻结。独立的 CURE-Lite PFCR input contract v2、关系控制
> evidence-release decoder 与 320-update learned Development 已实现，seed 42/43
> 均通过。新主结果见 `CURE_Lite_PFCR_v2_模型设计与Development正式结果.md`。
> 这不是对 v13 的静默续接，也不改变本文记录的 v13 终态。

---

## 0. 执行结论

NLCC-v12 不应继续训练、延长 update、修改阈值、调整 loss 权重或更换 seed 后重跑。正式结果已经证明：

```text
执行链有效
结构门禁 25/25 通过
数值门禁 26/76 通过
8 个 group 全部失败
D 区变化方向 6/6 正确
远端 G_norm_tail 8/8 稳定
绝对端点、matched-null、背景和 factual anchors 未同时成立
```

下一步不建议立即扩大 decoder、加入 attention、引入连续 Base 概率、替换 hard union，或者同时修改 normalization、状态方程和损失。

推荐先建立一个新的、独立命名的 CURE-Lite 诊断版本：

```text
诊断版本：GCDE-v13
decoder：原 NLCC-v12 decoder，逐字节保持
公共输入：(detach(F_b), O)
参数量：2,593
参数张量：6
推理：一次 Base + 一次 decoder + hard union
候选变量：训练风险变为 gate-covered statewise dual-endpoint surrogate
对照：同输入、同初始化、同 schedule 的 PECO-v10
```

GCDE-v13 要回答的单一问题是：

> 在字节级相同输入、初始化、schedule 和 decoder 下，极值敏感双端点训练代理是否比 PECO-v10 更接近冻结判据？

GCDE-v13 单独失败不能证明表示不足，也不能自动授权 LACR-v14；失败还可能来自固定系数冲突、极值归约、共享参数梯度抵消或有限更新预算。只有在输入识别、free-logit 可行性、参数空间梯度和成对对照共同排除这些解释后，才可进入 decoder/state 重构。

本文件不改变此前已经冻结的 CURE 主线：

> 面向任意 IRSTD detector，仅从通用接口 \((F_b,O)\) 学习 feature–coverage relevance；当覆盖确实抑制目标证据时释放 completion residual，在无关覆盖和背景上保持静默。

GCDE 只服务于判断当前 NLCC 失败中有多少来自训练代理，不替代上述模型机制。

---

# 1. 当前结果证明了什么

## 1.1 已经排除的解释

本次结果不能再解释为：

- runner 未闭合；
- result/decision 被错误信任；
- 训练被重复启动；
- optimizer state 非法继承；
- 更新次数不足于冻结计划；
- 非有限值导致训练提前损坏；
- 源码没有完整绑定；
- 结果目录没有形成唯一终态。

正式运行完成：

| 项目 | 结果 |
|---|---:|
| optimizer updates | 320 / 320 |
| decoder training forwards | 960 / 960 |
| finite-state audits | 321 / 321 |
| structural gates | 25 / 25 |
| nonfinite elements | 0 |
| targeted tests | 43 passed |
| repository tests | 1105 passed |
| source closure | 43 nodes / 226 edges |
| terminal artifact | 唯一 `result.json` |
| decision | `NLCC_V12_DEVELOPMENT_FAIL` |

因此这是一个**可解释的模型/目标联合系统失败**。

## 1.2 仍然成立的部分

NLCC-v12 学到了部分预期机制：

1. 六个 clean-D group 的变化方向全部正确；
2. 所有 D wrong-direction pixel count 都为 0；
3. \(G_{\mathrm{norm\_tail}}\) 在八个 group 中全部通过；
4. 320 updates 内总 loss 从约 3.1586 降到约 1.2846；
5. operator margin、recovery factor、参数、buffer 与 Adam state 全部有限。

这说明模型不是完全没有学习，也不是梯度路径断开。

## 1.3 真正失败的部分

四个全局门禁全部失败：

| 门禁 | 结果 | 要求 |
|---|---:|---:|
| population total loss | 1.2869663 | `< 0.1` |
| factual-miss target minimum | 0.2876031 | `> 0.95` |
| factual-miss background maximum | 0.9996876 | `< 0.05` |
| factual-no-miss maximum | 0.7426276 | `< 0.05` |

按门禁类型：

| 门禁类型 | 通过数 |
|---|---:|
| positive anchor | 6 / 8 |
| matched anchor-null | 0 / 8 |
| plus background | 0 / 8 |
| zero-H | 3 / 8 |
| zero-G-near | 2 / 8 |
| zero-G normalized tail | 8 / 8 |
| clean-D mean delta | 1 / 6 |
| clean-D plus endpoint | 0 / 6 |
| clean-D minus endpoint | 0 / 6 |
| D wrong direction | 6 / 6 |

D 区平均响应随 count/geometry 明显衰减：

| group | clean-D mean delta |
|---|---:|
| same-cell 1px | 0.8601 |
| adjacent-cell 1px | 0.7538 |
| adjacent-cell 3px | 0.5263 |
| same-cell 3px | 0.4851 |
| multicount \(2\to1\) | 0.3021 |
| multicount \(3\to2\) | 0.1664 |

该模式不能单独证明某一个根因，但足以说明：

> “正确变化方向”没有转化为“满足绝对端点、最差像素和多计数响应幅度的可用完成算子”。

---

# 2. 代码层面的失败归因

以下分为“代码直接证明”“结果强烈支持”和“尚待新版本验证”三类，避免把相关性写成因果结论。

## 2.1 代码直接证明：训练风险和正式门禁不一致

当前 `CURELiteLoss` 的负样本项为：

\[
\operatorname{mean}_{i\in N}\operatorname{softplus}(z_i),
\]

正样本项也是正像素上的均值，再加 positive-only Dice。

但正式门禁使用：

- positive **minimum**；
- background **maximum**；
- no-miss **maximum**；
- matched-null **maximum**；
- D plus **maximum**；
- D minus **minimum**；
- H/G 的 **maximum absolute delta**。

因此当前优化器可以取得很低的平均负样本 loss，同时保留少量很高的错误像素。正式结果正好出现了这种情况：

```text
factual-no-miss loss ≈ 0.0400
factual-no-miss maximum ≈ 0.7426
```

平均 loss 很低，但最差像素远未通过门禁。

这不是阈值太严格，而是当前 risk functional 没有直接覆盖正式判据。

## 2.2 代码直接证明：PECO 只对 plus endpoint 做绝对 anchor

当前 PECO：

1. 对 plus endpoint 运行 `CURELiteLoss`；
2. 在 D 上优化：
   \[
   \frac12\left[
   \operatorname{softplus}(z^+)
   +
   \operatorname{softplus}(-z^-)
   \right];
   \]
3. 在 H/G 上只优化：
   \[
   \left(\sigma(z^-)-\sigma(z^+)\right)^2.
   \]

问题在于：

- minus endpoint 没有完整绝对 completion/background anchor；
- H/G 的两个 endpoint 即使同时很高，只要相等，delta loss 仍可接近 0；
- component-null 中，被删除区域在 plus endpoint 上可能因 occupancy 为 1 而不属于 writable background；删除后它在 minus endpoint 上变成可写背景，但当前 loss 没有直接要求该 minus 区域低于 0.05；
- 当前 D loss主要推动跨过 logit 0，而正式门禁要求：
  \[
  p^+<0.05,\qquad p^->0.95.
  \]

对应 logit 门槛为：

\[
\kappa_{\mathrm{eval}}=\operatorname{logit}(0.95)=\ln 19\approx2.944439,
\]

即：

\[
z^+<-\kappa_{\mathrm{eval}},\qquad z^->\kappa_{\mathrm{eval}}.
\]

两个 endpoint 至少需要约：

\[
2\kappa_{\mathrm{eval}}\approx5.888878
\]

的 logit separation。当前 PECO 没有显式使用这一冻结 margin。

## 2.3 代码直接证明：单位 count 释放不等于单位输出释放

NLCC 定义：

\[
m=q-C.
\]

删除一个 count 时：

\[
m^--m^+=1.
\]

这只能证明 margin 位移恒为 1，不能证明 probability delta 恒定。

若两个 endpoint 都在 active 区：

\[
h(m)=e^m-1,
\]

则：

\[
h(m+1)-h(m)=e^m(e-1).
\]

它依赖绝对 margin \(m\)。count 越高，plus margin 越低，实际 evidence delta 越容易衰减。

正式结果中：

\[
0.8601
\rightarrow
0.3021
\rightarrow
0.1664
\]

的趋势与这一风险一致，但尚不能只凭结果宣称 count-only 是唯一原因。

## 2.4 代码直接证明：GroupNorm 会引入空间共同模式

当前 feature trunk 的两个 normalization 都是：

```python
nn.GroupNorm(groups=8, num_channels=32, affine=False)
```

PyTorch GroupNorm 的统计量覆盖每个 group 内的通道和空间位置。对于稀疏 feature：

- 1×1 stem 后，未激活位置原本为 0；
- GroupNorm 会减去整个 group 的空间均值；
- 因而远端原本为 0 的位置通常变成非零值；
- 这些值再进入 SiLU、depthwise 和 phase heads。

所以局部 feature witness 可以产生 feature-only 的全图共同模式。

这能解释一个重要组合：

```text
G_norm_tail 的 paired delta 全部稳定
但 factual/plus 的绝对背景仍可能很高
```

因为 pair 两端共享相同 feature，共同模式会在 delta 中抵消；绝对输出却不会自动消失。

该现象是明确的结构风险，但本报告不把它直接写成 v12 唯一失败原因。

## 2.5 代码直接证明：固定 null 不能完全消除 phase common mode

若所有 phase raw response 都等于 \(a\)：

\[
r_j=a,
\]

则：

\[
\mu_\varnothing=\frac{Pa}{P+1},
\]

\[
q_j=a-\frac{Pa}{P+1}=\frac{a}{P+1}.
\]

因此正的共同模式仍会保留一个 \(\frac1{P+1}\) 分量。背景处通常 \(C=0\)，该分量仍可能激活 exponential crossing。

## 2.6 需要新增审计：当前 reachability 是“精确张量可达”，不是“角色等价可识别”

当前输入生成器把：

- group ID；
- dyad index；
- match ID；
- channel；
- spatial coordinate；

都写入哈希，再生成非零 feature 幅度。

当前 local reachability signature 又直接包含原始 feature patch fingerprint。因此，只要两个 group 的随机幅度不同，就不会被记为 local signature collision。

但从机制角度，必须额外检查：

> 去掉 group ID、幅度和符号这一类样本标识后，相同 feature-role / occupancy-role 是否要求相反输出？

一个需要重点审计的角色等价例子是：

```text
adjacent plus：
response cell 中有 target feature；
center occupancy = 0；
一个邻居 occupancy = 1；
要求 residual 低。

multicount 2→1 minus：
response cell 中有 target feature；
center occupancy = 0；
一个固定邻居 occupancy = 1；
要求 residual 高。
```

在只观察“response-cell feature + 当前局部 occupancy”的情况下，这两个状态可能落入同一角色等价类。当前 raw hash 会因随机 feature 数值不同而把它们视为不同输入。

这不证明当前输入一定非法，但必须在新版本前增加：

```text
role-quotient identifiability audit
```

否则模型可能通过记忆 group-specific feature 幅度，而不是学习可泛化机制。

---

# 3. 为什么先做成对目标函数诊断，而不是改写 CURE 主线

推荐的研究顺序是：

```text
v12：表示、输入、当前目标和有限预算的联合实例失败
        │
        ▼
v13：保持表示和字节级输入不变，成对比较 PECO 与极值敏感双端点代理
        │
        ├── PECO FAIL / GCDE PASS：支持 objective alignment 有贡献
        ├── PECO PASS / GCDE PASS：原 v12 结果可能含运行/population 敏感性
        ├── PECO PASS / GCDE FAIL：GCDE 引入优化冲突
        └── PECO FAIL / GCDE FAIL：结论仍不充分，继续做可行性归因
```

这样做有四个优点。

## 3.1 成对实验只改变训练代理

v13 不改变：

- NLCC 方程；
- null reference；
- count boundary；
- exponential forward；
- straight-through recovery carrier；
- GroupNorm；
- width；
- phase heads；
- 参数量；
- 输入；
- optimizer；
- update budget；
- inference graph。

在同一成对实验的两个 arm 之间，唯一允许改变的是：

> PECO-v10 与 gate-covered statewise dual-endpoint surrogate。

新增 8 项 minus-background 判据属于扩展评估语义，而不是 objective 本身。必须将旧 76 项和扩展 84 项分开报告，不能把整个 v13 描述为“只改变 objective”。

## 3.2 它直接覆盖所有主要失败类型

| v12 失败 | v13 对应风险 |
|---|---|
| factual target minimum 失败 | positive worst-pixel logit margin |
| factual background maximum 失败 | negative worst-pixel logit margin |
| factual no-miss maximum 失败 | negative worst-pixel logit margin |
| positive anchor minimum 失败 | plus positive worst-pixel margin |
| matched-null maximum 失败 | plus negative worst-pixel margin |
| plus background maximum 失败 | plus negative worst-pixel margin |
| D plus endpoint 失败 | D plus logit margin |
| D minus endpoint 失败 | D minus logit margin |
| H/G maximum delta 失败 | max-delta margin |
| component-null deleted footprint | minus absolute background anchor |

## 3.3 v13 只能收窄归因，不能单独证明表示失败

如果一个原始独立约束：

- 有对应的直接风险；
- 违反时有非零、方向正确的梯度；
- 在共享 decoder 参数空间中仍有有限、非零且不被其他项抵消的梯度；
- free-logit oracle 证明该组监督本身可同时满足；
- PECO/GCDE 使用字节级相同输入、初始化和 schedule；
- GCDE 仍然不能通过；

那么“旧 loss 没有直接覆盖该问题”的解释才会减弱。即使如此，320-update 失败仍不能单独区分表示不足与优化预算不足；进入 state representation 或 normalization 之前仍需冻结归因收据。

## 3.4 它不会用额外容量制造假阳性结论

参数仍为：

```text
2,593 parameters
6 parameter tensors
```

若结果改善，不能归因于增加 head、宽度或模块容量。

---

# 4. 推荐新版本：GCDE-v13

建议正式名称：

```text
CURE-Lite Gate-Covered Dual-Endpoint v13
缩写：GCDE-v13
method_id：gate_covered_dual_endpoint_v13
decoder_id：nlcc_v12_frozen_decoder
```

不要命名为“修复后的 NLCC-v12”，因为 v12 已经冻结失败。

---

# 5. GCDE-v13 的数学定义

## 5.1 评估门槛与冻结训练安全余量

\[
p_{\mathrm{high}}=0.95,\qquad p_{\mathrm{low}}=0.05,
\]

\[
\kappa_{\mathrm{eval}}
=
\log\frac{0.95}{0.05}
=
\ln 19
\approx2.944438979.
\]

由于正式判据使用严格的 \(p>0.95\) 和 \(p<0.05\)，训练代理不能在
\(\pm\kappa_{\mathrm{eval}}\) 处已经为零。冻结一个仅用于数值边界的、不可搜索的概率余量：

\[
\delta_p=0.001,
\qquad
p_{\mathrm{train,high}}=0.951,
\qquad
p_{\mathrm{train,low}}=0.049,
\]

\[
\kappa_{\mathrm{train}}
=
\log\frac{0.951}{0.049}
\approx2.965693764
>
\kappa_{\mathrm{eval}}.
\]

\(\delta_p\) 不通过 Development 或 Holdout 调整；实现、decision 和测试共同冻结
`float32`、严格比较规则及该常数的十六进制表示。这样训练 violation 为零时，
正式 0.95/0.05 严格门槛仍保留正余量。

定义：

\[
[x]_+=\max(x,0).
\]

## 5.2 Gate-covered statewise absolute surrogate

对一个状态的正像素集合 \(P\) 和负像素集合 \(N\)，定义最差像素 margin violation：

\[
V_{+,b}(z;P_b)
=
\begin{cases}
\max\limits_{i\in P_b}[\kappa_{\mathrm{train}}-z_{b,i}]_+^2,&P_b\neq\varnothing,\\
0,&P_b=\varnothing,
\end{cases}
\]

\[
V_{-,b}(z;N_b)
=
\begin{cases}
\max\limits_{i\in N_b}[z_{b,i}+\kappa_{\mathrm{train}}]_+^2,&N_b\neq\varnothing,\\
0,&N_b=\varnothing.
\end{cases}
\]

这里的 \(b\) 是一个 factual state 或一个 endpoint state。每个 state 先独立取
worst-pixel violation，再对 minibatch 中等权 state 求均值；禁止对整个 minibatch
直接取一个全局 maximum。该训练代理覆盖每个被采样 state 的极值违反，但不宣称与
完整 population/group maximum 完全相同。

保留当前 dense supervision：

\[
L_{\mathrm{old\ abs}}
=
L_{\mathrm{balanced\ logistic}}
+
L_{\mathrm{positive\ Dice}}.
\]

新绝对风险：

\[
\boxed{
L_{\mathrm{GCAbs},b}
=
L_{\mathrm{old\ abs},b}
+
V_{+,b}
+
V_{-,b}
}.
\]

这样：

- 平均 BCE/Dice 提供稠密梯度；
- \(V_+\) 直接优化最小正像素；
- \(V_-\) 直接优化最大背景像素；
- 一旦全部像素越过冻结 margin，tail violation 变为 0；
- 不搜索新权重；但所有固定系数仍属于设计选择，必须在运行前展开记录。

## 5.3 两个 endpoint 都要有绝对 anchor

冻结：

\[
V=\mathrm{image\_valid\_mask},
\qquad
P^\pm=Y^\pm_{\mathrm{completion}},
\]

\[
N^\pm
=
V\cap\neg O^\pm\cap\neg GT,
\qquad
A^\pm=P^\pm\cup N^\pm.
\]

因此新 absolute criterion 的数学接口统一写为
\(L_{\mathrm{GCAbs},b}(z_b;P_b,N_b)\)；内部用 \(P_b\cup N_b\) 作为
valid mask。factual 分支沿用 \(P=V\cap Y\)、\(N=V\cap\neg Y\)。

plus endpoint：

\[
B^+
=
V
\cap
\neg O^+
\cap
\neg GT,
\]

\[
A^+
=
Y^+_{\mathrm{completion}}
\cup
B^+.
\]

minus endpoint：

\[
B^-
=
V
\cap
\neg O^-
\cap
\neg GT,
\]

\[
A^-
=
Y^-_{\mathrm{completion}}
\cup
B^-.
\]

然后：

\[
L_{\mathrm{end},b}
=
\frac12
\left[
L_{\mathrm{GCAbs},b}(z^+;P^+,N^+)
+
L_{\mathrm{GCAbs},b}(z^-;P^-,N^-)
\right].
\]

关键变化是：

> `completion_minus` 和 `occupancy_minus` 必须显式传入 criterion。

不能只从 plus endpoint 和 delta 间接推断。

## 5.4 D 区显式 endpoint margin 与空集语义

定义：

\[
V_{D+,b}
=
\begin{cases}
\max\limits_{i\in D_b}[z_{b,i}^++\kappa_{\mathrm{train}}]_+^2,&D_b\neq\varnothing,\\
0,&D_b=\varnothing,
\end{cases}
\]

\[
V_{D-,b}
=
\begin{cases}
\max\limits_{i\in D_b}[\kappa_{\mathrm{train}}-z_{b,i}^-]_+^2,&D_b\neq\varnothing,\\
0,&D_b=\varnothing.
\end{cases}
\]

\[
L_{D,b}
=
\frac12(V_{D+,b}+V_{D-,b}).
\]

若 \(D_b=\varnothing\)，则：

\[
L_{D,b}=0,\qquad D_{\mathrm{active},b}=\mathrm{false}.
\]

这直接要求：

\[
z^+\le-\kappa_{\mathrm{train}}<-\kappa_{\mathrm{eval}},
\qquad
z^-\ge\kappa_{\mathrm{train}}>\kappa_{\mathrm{eval}}.
\]

当这两个条件满足时，自动得到：

\[
p^- - p^+ \ge 0.902,
\]

因此会同时支持：

- clean-D mean delta \(\ge0.8\)；
- wrong-direction pixel count \(=0\)。

仍应保留这两项正式评估，不因数学蕴含而删除。

## 5.5 H、\(G_{\mathrm{near}}\)、\(G_{\mathrm{norm\_tail}}\) 分别使用最大变化

定义：

\[
\Delta p=\sigma(z^-)-\sigma(z^+).
\]

复用 v12 runner 已冻结的三个 zero-response stratum，不重新定义像素集合：

\[
S\in\{H,G_{\mathrm{near}},G_{\mathrm{norm\_tail}}\}.
\]

对每个 pair \(b\) 和任一非空 stratum：

\[
V_{S,b}
=
\left[
\max_{i\in S_b}|\Delta p_{b,i}|
-
0.05
\right]_+^2.
\]

若 stratum 为空，记为 inactive，不参与该 pair 的 active mean。

\[
L_{\mathrm{zero},b}
=
\operatorname{activeMean}
\left(
V_{H,b},
V_{G_{\mathrm{near}},b},
V_{G_{\mathrm{norm\_tail}},b}
\right).
\]

不能将两个 \(G\) 子集先合并后只取一个 maximum；虽然零 violation 的可行集合
等价，但同时违反时只会给较大者梯度，不能覆盖两个独立正式判据。

## 5.6 Pair risk 保持现有层级，不引入搜索权重

\[
L_{\mathrm{transition},b}
=
\operatorname{activeMean}(L_{D,b},L_{\mathrm{zero},b}),
\]

\[
\boxed{
L_{\mathrm{pair}}
=
\operatorname{mean}_b
\left[
\frac12L_{\mathrm{end},b}
+
\frac12L_{\mathrm{transition},b}
\right]
}.
\]

总目标：

\[
\boxed{
L_{\mathrm{total}}
=
L_{\mathrm{GCAbs}}^{\mathrm{factual\ miss}}
+
L_{\mathrm{GCAbs}}^{\mathrm{factual\ no\ miss}}
+
L_{\mathrm{pair}}
}.
\]

固定系数展开如下：

| 条件 | plus absolute | minus absolute | \(D\) risk | zero-response risk |
|---|---:|---:|---:|---:|
| \(D\) 与 zero 均 active | 0.25 | 0.25 | 0.25 | 0.25 |
| \(D\) inactive、zero active | 0.25 | 0.25 | 0 | 0.50 |

factual-miss 和 factual-no-miss 两个分支各以系数 1 进入总目标。D-minus 同时出现在
minus dense completion supervision 和显式 D margin 中；这是冻结的重叠监督，
必须在 coefficient/mask receipt 中显式报告，并在参数梯度检查中确认没有导致
端点梯度失衡。

仍然是：

```text
3 decoder forwards
12 decoder states
1 backward
1 optimizer step
```

---

# 6. 必须新增的约束—梯度覆盖检查

正式 Development 前，必须区分：

1. **原始独立约束**：需要直接风险和梯度检查；
2. **派生或包含判据**：需要数学蕴含证明，不强行构造不可能的“仅违反该项”样例。

检查不能停留在独立 logits 上；还必须通过同一个 2,593 参数 decoder 计算
parameter-space VJP，检查多项同时 active 时是否出现完全抵消、非有限梯度或端点
严重失衡。

新增：

```python
audit_gate_gradient_coverage(...)
```

至少覆盖：

| 门禁 | 必须验证的梯度方向 |
|---|---|
| factual target min | 最差正像素 logit 上升 |
| factual background max | 最差背景 logit 下降 |
| factual no-miss max | 最差像素 logit 下降 |
| positive anchor min | 最差 anchor logit 上升 |
| matched-null max | 最差 null logit 下降 |
| plus background max | 最差 plus background logit 下降 |
| minus background max | 最差 minus background logit 下降 |
| D plus max | 最差 plus D logit 下降 |
| D minus min | 最差 minus D logit 上升 |
| H max delta | 最大变化绝对值下降 |
| \(G_{\mathrm{near}}\) max delta | 最大变化绝对值下降 |
| \(G_{\mathrm{norm\_tail}}\) max delta | 最大变化绝对值下降 |

要求：

```text
所有可独立构造的原始约束案例：
loss finite
logit gradient finite / nonzero / direction correct
shared-parameter VJP finite / nonzero
联合 active 时记录 cosine、norm ratio 和 cancellation ratio
覆盖等号边界、并列 extrema、sigmoid 饱和、空集合和重叠 mask
不读取 group_id / pair_kind / anchor_role
```

matched-null 与 plus-background 等包含关系，以及 D mean/wrong-direction 与严格
D endpoints 的关系，使用单独的 implication receipt；不要求伪造“只违反派生
判据”的样例。这项检查不训练模型，不消耗正式 attempt。

---

# 7. Development 门禁调整

## 7.1 原 76 项门禁全部保留

不能因为 v12 失败而删除或放宽：

- 4 个 global gates；
- 8 group × 6 个通用 gates；
- 6 clean group × 4 个 D gates。

## 7.2 增加 8 项 minus-background 门禁

对每个 group 增加：

\[
\max_{B^-}\sigma(z^-)<0.05.
\]

因此 v13 数值门槛固定为：

\[
76+8=84.
\]

冻结布尔关系：

```text
legacy_76_pass = all(original_76)
minus_background_8_pass = all(new_minus_background_8)
extended_84_pass = legacy_76_pass AND minus_background_8_pass
extended_group_pass = 每个 group 的旧判据与新增 minus-background 判据均通过
```

`extended_84_pass` 包含而不是独立于 `legacy_76_pass`。

这项门禁直接防止：

- component-null 删除后产生 residual；
- clean pair 的 minus endpoint 在非目标可写背景产生高响应；
- H/G 只保持“两个 endpoint 同样高”。

## 7.3 不改变阈值

保留：

```text
positive > 0.95
negative < 0.05
D mean delta >= 0.8
H/G max abs delta <= 0.05
population total loss < 0.1
wrong-direction count = 0
```

## 7.4 `population total loss` 冻结为唯一双轨定义

- runner 同时记录 `optimizer_objective` 和 `legacy_population_loss`；
- 正式 `<0.1` 门槛只作用于使用 v12 原始 absolute loss 与 PECO pair loss 完整
  population 重算得到的 `legacy_population_loss`；
- `optimizer_objective` 记录新 GCDE 目标，不套用 `<0.1`；
- gate-tail violation 单独逐项报告；
- 不能把新增 tail 项导致的量纲变化误报成表示失败。

本版本不再保留“把新 objective 直接作为 population total”的备选项。任何改变
都必须建立新版本，不能运行后再解释。

---

# 8. Dataset-free 输入在 v13 前必须补一项识别审计

## 8.1 不直接改写 v12 输入

以下实际文件和收据保持只读：

```text
cure_lite/nlcc_dataset_free_inputs.py
cure_lite/nlcc_development_inputs.py
cure_lite/nlcc_holdout_inputs.py
protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/
    dataset_free_input_freeze_and_reachability_receipt.json
    development_regression_r1/result.json
    development_regression_r1/decision.json
    development_regression_r1/COMPLETE.json
```

## 8.2 新增两层输入识别检查

第一层是硬不可识别检查：

1. 键只由 decoder 实际接收或确定性计算的状态构成：
   - 完整 `feature` tensor；
   - 由 `occupancy` 确定、且 NLCC 方程实际消费的完整 local count field；
   - 输出像素相对坐标和 PixelShuffle phase；
2. 不放入 group ID、sample ID、match ID、anchor role、pair kind 或显式
   plus/minus 角色标签；
3. 检查字节级相同 decoder-accessible state 是否要求相反输出约束。

第二层是 role-shortcut quotient：

1. 删除所有 ID；
2. 对非零 feature 幅度分别执行“保留符号归一化”和“去符号归一化”；
3. 以被监督输出位置为原点表示完整 feature support 与 local count field；
4. endpoint 身份只能由实际 occupancy/count 差异体现，不能作为额外键；
5. 检查同一 quotient state 是否要求相反 absolute/transition label。

旋转/翻转只作为独立诊断：必须联合变换 feature、count field、输出坐标、
channel/phase 语义；由于当前卷积—PixelShuffle 图没有预先证明 D4 等变，
rotation quotient 不作为 Development 硬门槛。

输出至少包括：

```text
exact_tensor_collision_count
role_quotient_collision_count
rotation_quotient_collision_count
amplitude_identity_shortcut_count
input_fingerprint
algorithm_version
hard_gate_pass
role_gate_pass
conflict_examples
```

## 8.3 审计决策

```text
exact tensor collisions = 0
and role-quotient collisions = 0
    -> 仅解除 R13-1 阻断，进入 R13-2；
       完成 R13-2～R13-4 和第16节全部检查后，
       才能创建 paired Development authorization。

exact tensor collisions > 0
or role-quotient collisions > 0
    -> 不运行 GCDE-v13；
       冻结输入识别负结果；
       若要修改输入，建立独立 input-contract 版本和因子对照。
```

## 8.4 输入 contract v2 的边界

若确有 adjacent-plus / multicount-minus 角色冲突，禁止向输入注入 group、endpoint、
匹配角色、GT 谱系或只在训练时存在的标记。任何新信号必须能在实际推理时从任意
IRSTD detector 的通用接口 \((F_b,O)\) 确定性得到。

允许研究的方向是 detector-accessible state representation，例如显式局部 occupancy
basis 或 feature-conditioned coverage relevance；不允许用标签谱系伪造推理不可得
的 witness。新输入必须：

- 建立独立版本和 input-contract receipt；
- 在同一新输入上形成 `PECO × GCDE` 对照；
- 原 v12 population 保留为冻结 regression diagnostic；
- 不再把该实验称为 objective-only v13。

## 8.5 R13-1 正式运行结果

2026-07-27 使用冻结的 v12 Development builder 运行：

```text
states                         96
supervised records          75,192
positive records              112
negative records           75,080

exact effective-input conflicts       0
signed role-quotient conflicts        8
unsigned role-quotient conflicts      7
records in unsigned conflicts        64

hard_gate_pass              true
role_gate_pass             false
development_authorized     false
D4 diagnostic      not_evaluated
```

输入 fingerprint：

```text
4f387e3e513a93a1cee58ee68d9d67eb5b2746688da42ec40572ae6fc1df55a7
```

receipt fingerprint：

```text
58eb73948da441891849353b41aa7493b9485500f73b120fbb128be33a454728
```

receipt 文件 SHA256：

```text
c81fdcd77c9f7157b2ae965359e228579ee7b51c9e9826ef94b97979a9791571
```

定向测试：

```text
7 passed in 15.56s
第二次完整 receipt 重建与已封存文件逐字节一致
隔离测试矩阵：1,491 passed，0 failed/error/skipped
```

正式 receipt：

```text
protocols/IRSTD-1K/gate_covered_dual_endpoint_v13/
    input_identifiability_receipt.json
```

结论：冻结输入没有字节级有效状态硬冲突，但部分相反监督只有依赖随机
feature 幅度/符号或绝对空间身份才能区分。按第8.3节，R13-1 FAIL，当前流程
停在输入状态设计层；不实现或训练 GCDE-v13。

---

# 9. 文件级实施方案

以下为推荐新增文件。原则是 additive，不覆盖 v12。

## 9.1 Loss

### `cure_lite/gate_covered_absolute_losses.py`

包含：

```python
GateCoveredAbsoluteLoss
worst_positive_margin_violation(...)
worst_negative_margin_violation(...)
```

要求：

- 复用或明确嵌入当前 balanced logistic + Dice；
- tail risk 按 state 独立计算；
- 空 positive/negative 集合安全；
- 不产生 NaN；
- 返回 raw worst logit、margin violation 和 active flag。

### `cure_lite/gate_covered_dual_endpoint_losses.py`

包含：

```python
GateCoveredDualEndpointLoss
```

显式输入：

```python
logits_plus
logits_minus
completion_plus
completion_minus
occupancy_plus
occupancy_minus
gt_union
image_valid_mask
intervention_footprint
```

输出：

```text
plus_absolute
minus_absolute
D_plus_margin
D_minus_margin
H_max_delta_violation
G_near_max_delta_violation
G_norm_tail_max_delta_violation
endpoint_risk
transition_risk
total
```

## 9.2 Training step

### `cure_lite/train/gate_covered_outcome_step.py`

保持：

```text
factual miss batch = 4
factual no-miss batch = 4
pair batch = 2
paired endpoints = one 2B forward
decoder forwards = 3
states = 12
backward = 1
optimizer step = 1
```

不得：

- 增加第二 optimizer；
- 分开 backward；
- stop-gradient 某个 endpoint；
- 使用 group ID 分支；
- 使用 anchor role 分支；
- clipping、retry、resume。

## 9.3 Inputs

若 role-quotient audit 通过，GCDE 与 PECO control 必须共同调用冻结的
`cure_lite/nlcc_development_inputs.py` builder。训练前分别物化两个 arm 的全部
tensor，并验证完整 tensor-byte fingerprint、profile ID、design seed 和 input
fingerprint 完全相同；这些字段会改变 hash feature，因此不得更换。只允许新的
method ID、arm ID、输出目录、authorization 和 implementation closure。

若 audit 失败，本版本停止。任何 input-contract v2 都建立独立版本，并执行
`old input/new input × PECO/GCDE` 因子对照，不静默续接 v13。

## 9.4 Runner

建议新增：

```text
cure_lite/gcde_dataset_free_runner_config.py
cure_lite/gcde_dataset_free_runner.py
tools/evaluate_gcde_development.py
tools/evaluate_gcde_exposure_holdout.py
tools/verify_gcde_result.py
```

可复用 r2 runner 的完整性设计，但不得修改 v12 已绑定源码后声称 v12 closure 不变。

GCDE runner 继续要求：

- exact-once authority；
- durable `training_started.json`；
- 每 step 后 parameter/buffer/optimizer finite audit；
- strict JSON；
- raw metric recomputation；
- single terminal；
- implementation closure；
- independent decision verifier。

## 9.5 Protocol root

新建：

```text
protocols/IRSTD-1K/gate_covered_dual_endpoint_v13/
```

至少包含：

```text
model_and_objective_design.md
r13_0_v12_freeze_baseline_receipt.json
gate_gradient_coverage_receipt.json
input_identifiability_receipt.json
dataset_free_input_freeze_receipt.json
dataset_free_evaluation_preregistration.json
runner_implementation_closure.json
runner_verification_receipt.json
development_pre_run_authorization.json
```

Development authorization 必须最后创建。

---

# 10. 测试计划

## 10.1 Loss 单元测试

1. positive 最差像素低于 \(\kappa_{\mathrm{train}}\) 时，梯度必须使其上升；
2. negative 最差像素高于 \(-\kappa_{\mathrm{train}}\) 时，梯度必须使其下降；
3. 已满足 margin 时 tail violation 为 0；
4. 在 \(\pm\kappa_{\mathrm{eval}}\) 等号处仍有正 violation 和正确梯度；
5. 一个高背景像素不能被大量低背景像素平均掉；
6. 并列 extrema 的 loss/gradient 有限且至少一个违反像素获得梯度；
7. D both-high 被处罚；
8. D both-low 被处罚；
9. D correct-sign but insufficient-margin 被处罚；
10. H/G equal-high：
   - delta term可为 0；
   - absolute minus/plus background仍必须处罚；
11. \(G_{\mathrm{near}}\) 与 \(G_{\mathrm{norm\_tail}}\) 同时违反时两项均 active；
12. component-null deleted footprint 高响应必须被 minus absolute loss处罚；
13. 空 D 的 component-null 返回 inactive，不产生空张量 reduction；
14. sigmoid 饱和、重叠 mask 下所有 loss 和 gradient 有限；
15. criterion 不接收 pair kind、group ID、anchor role。

## 10.2 Gate-gradient coverage 测试

对每个原始独立约束构造 violation case；对派生或包含判据提供数学蕴含测试。检查：

```text
violation detected
loss increases
target logit gradient direction correct
shared decoder parameter VJP finite/nonzero
联合 objective 的 gradient cancellation ratio 在冻结范围内
unrelated tensor gradient 为0或符合预期
```

## 10.3 输入识别测试

1. exact collision = 0；
2. role-quotient collision = 0；
3. amplitude permutation 后结论不变；
4. 去除 group ID 后结论不变；
5. adjacent/multicount 角色冲突被显式检测或消除；
6. matched twin 仍只在 anchor witness/target 上不同；
7. Development 两个 objective arm 的输入、初始化、schedule fingerprint 完全相同；
8. Holdout 在读取 Development 输出前冻结；
9. D4 诊断不影响 hard/role gate decision。

## 10.4 Runner 异常路径测试

沿用 v12 r2 的全部异常路径测试，并新增：

1. 删除 minus-background metric 时 result invalid；
2. 伪造 84/84 embedded pass 但 raw minus background 失败时必须拒绝；
3. 只计算 plus absolute、漏算 minus 时 structural gate fail；
4. 训练 criterion 与 decision threshold fingerprint 不一致时拒绝授权；
5. gate-gradient receipt 与 closure fingerprint 不一致时拒绝授权。

## 10.5 全仓回归

要求：

```text
所有 v12 测试继续通过
v12 正式结果文件 SHA 不变
新 GCDE 测试全部通过
完整仓库测试 0 failed/error/skipped
```

---

# 11. 成对 Development / Holdout 计划

## 11.1 字节级同输入的成对 Development

建立两个 arm，每个 arm 均保持与 v12 相同的资源预算：

```text
fresh decoder seed = 42
fresh empty Adam
rows = 32
updates = 320
forward pattern = (4,4,4)
one attempt per arm
no checkpoint
no resume
```

两个 arm 必须相同：

```text
input fingerprint
schedule fingerprint
decoder initialization fingerprint
optimizer initialization contract
batch order
device/dtype
```

两个 arm 仅允许不同：

```text
criterion/method_id
arm_id
artifact directory
authorization fingerprint
implementation closure
```

建议目录：

```text
protocols/IRSTD-1K/gate_covered_dual_endpoint_v13/
    paired_development_r1/
        peco_control/
        gcde_candidate/
```

## 11.2 Development PASS

必须同时满足：

```text
both arms execution validity PASS
both arms input/init/schedule fingerprints identical
GCDE legacy_76_pass = true
GCDE minus_background_8_pass = true
GCDE extended_84_pass = true
GCDE extended_group_pass = 8/8
independent recomputation identical
each arm has one sealed terminal
```

此外必须报告 PECO 与 GCDE 的逐项差值。只有 GCDE 的 `extended_84_pass=true`
（其定义已包含 legacy-76）才允许使用已经预先冻结的 Holdout。该条件授权后续
诊断，但不自动证明 GCDE 优于 PECO。

成对 Development 的比较结论按四象限冻结：

| PECO legacy-76 | GCDE extended-84 | 允许结论 |
|---|---|---|
| FAIL | PASS | GCDE objective alignment 对当前输入有贡献 |
| PASS | PASS | 两者均可满足；不能声明 GCDE 更优 |
| PASS | FAIL | GCDE 引入目标或优化冲突 |
| FAIL | FAIL | 仍不能区分输入、表示、目标冲突与有限预算 |

## 11.3 Development FAIL

GCDE 任一 legacy 或 extended 数值门槛失败：

```text
GCDE_V13_DEVELOPMENT_FAIL
```

然后：

- 不重跑；
- 不延长 updates；
- 不调整 tail loss；
- 不换 seed；
- 不运行 Holdout；
- 不直接归因于表示失败；
- 先执行 free-logit feasibility、参数梯度冲突、exact-max 活跃像素和
  optimization-curve 归因；
- 不自动授权 LACR-v14。

## 11.4 Holdout 必须在 Development 输出前冻结

可以使用 v12 已冻结但从未读取输出的 Holdout，前提是先形成 independence review：

- v13 设计没有读取其具体 tensor；
- 没有依据其 group 数值调整 loss；
- 它与 Development 源、seed 和 exposure 独立；
- 在任何 Development arm 输出产生前封存 input/schedule fingerprint。

若上述条件不能证明，则必须在运行 Development 前重新生成并冻结专属 Holdout，
而不是 Development PASS 后再生成。

```text
rows = 222
updates = 400
fresh-fit replication:
    fresh seed-42 decoder
    fresh Adam
frozen-decoder unseen-state evaluation:
    freeze Development GCDE decoder
    no Holdout optimizer update
```

两类结果必须分开命名：

- `fresh_fit_replication`：另一 population 上能否重新学会；
- `frozen_decoder_unseen_state`：Development 模型能否直接泛化。

冻结终态：

```text
fresh_fit_replication_pass =
    execution_valid
    AND extended_84_pass
    AND extended_group_pass = 8/8
    AND independent_recomputation_identical

frozen_decoder_unseen_state_pass =
    development_decoder_fingerprint_unchanged
    AND optimizer_update_count = 0
    AND extended_84_pass
    AND extended_group_pass = 8/8
    AND independent_recomputation_identical

holdout_pass =
    fresh_fit_replication_pass
    AND frozen_decoder_unseen_state_pass
```

84 项在两种模式下使用相同 raw metric 定义；区别只在是否允许 Holdout 更新。
任一模式失败均冻结为 Holdout FAIL，不回到 Development 调整。不得把 fresh-fit
replication 写成模型参数的未见状态泛化；只有 `holdout_pass=true` 才允许真实
\(D_R\)。

---

# 12. 结果诊断决策树

## 12.1 GCDE-v13 全部通过

```text
paired Development:
  PECO control valid
  GCDE legacy-76 PASS
  GCDE extended-84 PASS
  -> pre-frozen Holdout
  -> fresh-fit replication and frozen-decoder evaluation
  -> real IRSTD-1K D_R bounded validation
```

此时只能说明：

> 在该冻结 dataset-free population 上，GCDE 能满足 extended-84；是否优于 PECO
> 必须按第11.2节四象限解释。只有预冻结 Holdout 的两个终态满足下述条件后，当前
> NLCC 实例才具有进入真实 \(D_R\) bounded validation 的资格。

仍不能宣称真实红外检测成功，也不能把 GCDE 本身写成 CURE 的核心模型创新。

## 12.2 绝对门禁明显改善，但 multicount D 仍失败

典型模式：

```text
factual anchors PASS
matched-null PASS
background PASS
D direction PASS
same/adjacent D PASS
2→1、3→2 D delta/endpoint FAIL
```

结论只能是：

> GCDE 没有使 multicount 判据通过；scalar count crossing / endpoint placement、
> 参数梯度冲突和有限预算仍是竞争解释。

先运行预注册的 free-logit、参数 VJP 和活跃 extrema 归因。只有这些检查共同支持
表示限制时，才建立新的 state-equation 候选。

## 12.3 加入 worst-pixel risk 后背景仍无法压低

典型模式：

```text
tail losses持续非零
最差背景仍接近1
target与background形成不可解冲突
G_norm_tail delta稳定但absolute background高
```

待归因假设：

> feature trunk 的空间共同模式、phase baseline/evidence 解耦、极值像素梯度稀疏
> 或固定系数冲突均可能导致该现象。

只有参数 VJP 与高容量/free-logit oracle 排除目标冲突后，才可建立“只修改
normalization”的独立候选：

```text
GroupNorm
  ->
per-cell channel RMS normalization
```

保持 state equation 和冻结 GCDE 代理，建立独立版本；该版本不能与 LACR 的
occupancy/affinity 修改合并。

## 12.4 D endpoints 通过但 H/G 或 component-null 失败

结论：

> 当前状态方程在 D 上可工作，但在 zero-response 约束上失败；这与“不能区分相关
> coverage component 和无关邻居/空组件”一致，但尚不是唯一解释。

完成输入 quotient、mask coverage 和参数梯度归因后，才决定是否建立独立的
relevant-coverage affinity 候选。

## 12.5 Development PASS、Holdout FAIL

结论：

> v13 在小型 Development 中可记忆/拟合，但没有 exposure 泛化。

首先检查：

- role-quotient shortcut；
- group-specific feature amplitude；
- count/geometry alias；
- affinity/phase role是否依赖 Development seed。

不允许回到 Development 调到 Holdout 通过。必须建立新的模型版本和新的未见 Holdout。

---

# 13. CURE 主线的结构假设：LACR（当前不授权）

本节只记录与原 CURE 主线一致的结构假设，不由一次 GCDE 失败自动触发，也不是
当前授权实现。其价值来自 feature–coverage relevance，而不是“GCDE 后再堆模块”。

名称：

> **Local Affinity Coverage Release（LACR）**

核心思想：

> 不再用一个无方向的局部 count 代表 coverage，而是由 feature 预测“哪一个局部 occupancy cell 与当前 target evidence 相关”，只删除相关 coverage 时才释放 evidence。

## 13.1 零保持局部 trunk

替换空间 GroupNorm 为每个 spatial cell 独立的 channel RMS normalization：

\[
\operatorname{CellRMS}(x)_{b,c,y,x}
=
\frac{x_{b,c,y,x}}
{\sqrt{
\frac1C\sum_{c'}x_{b,c',y,x}^2+\epsilon
}}.
\]

性质：

- 不跨空间位置聚合；
- 零 cell 保持精确零；
- 无 affine 参数；
- 不会把一个局部 witness 通过均值减法写到全图。

## 13.2 固定 \(3\times3\) occupancy basis

\[
U(F,O)
=
\operatorname{Unfold}_{3\times3}
\left(
\Pi_{\max}(O)
\right)
\in\{0,1\}^{9\times h\times w}.
\]

不是只保留：

\[
C=\sum_{\delta=1}^{9}U_\delta.
\]

## 13.3 Feature-conditioned relevant coverage

由 feature trunk 预测 9 个局部 affinity：

\[
\pi_\delta(F)
=
\operatorname{softmax}_\delta
\left(
W_aH(F)
\right).
\]

相关 coverage：

\[
c_{\mathrm{rel}}(F,O)
=
\sum_{\delta=1}^{9}
\pi_\delta(F)U_\delta(O).
\]

释放：

\[
v_{\mathrm{rel}}=1-c_{\mathrm{rel}}.
\]

## 13.4 稳定 evidence

保留 feature-only phase support：

\[
s_j(F)
=
\operatorname{softplus}(a_j(F)^2)-\log 2.
\]

它满足：

\[
F=0\Rightarrow s_j=0.
\]

最终：

\[
z_j(F,O)
=
-\operatorname{softplus}(\beta+b_j(F))
+
s_j(F)\,v_{\mathrm{rel}}(F,O).
\]

不再使用：

- `exp(m)`；
- straight-through exponential carrier；
- scalar count boundary；
- fixed null mean。

## 13.5 预估参数量

在：

\[
C=8,\quad s=4,\quad width=32
\]

下：

```text
原 trunk/heads/scalar：2,593
新增 32→9 affinity head：288
总参数：2,881
参数张量：7
```

在 \(C=64,s=4\) 下约为：

```text
4,673 parameters
```

仍属于轻量 decoder。

## 13.6 LACR 必须解决的输入问题

feature 必须包含可识别的 component-to-response 关系。若 dataset-free
输入仍只在 response cell 写入 group-specific 随机 feature，affinity head
可能无法区分：

```text
“邻居 component 正在覆盖目标”
与
“中心 component 已删除但无关邻居仍存在”
```

LACR 不能以当前整组改动直接授权，因为本节同时包含 normalization、occupancy
basis、affinity 和状态方程变化。若后续归因支持该方向，必须拆成逐项版本或
明确的因子实验，并保证所有输入在任意 detector 的 \((F_b,O)\) 接口下推理可得。

---

# 14. 明确禁止的修改

## 14.1 不延长 v12 或 v13 的训练

禁止：

- 320 改 640；
- 反复运行挑 checkpoint；
- 自动 retry；
- 换 seed 找正结果；
- 从失败结果继续训练。

## 14.2 不改变正式阈值来制造通过

禁止：

- 0.95 降到 0.90；
- 0.05 放到 0.10；
- D delta 0.8 降低；
- 只报告 mean、不报告 max/min；
- 删除失败 group。

## 14.3 不把 clamp 当作 v12 修复

本次：

```text
margin finite
recovery factor finite
321/321 finite audits pass
```

所以没有证据支持把主要失败归因于 overflow。给 `exp` 加 clamp 既会改变
方程，也不会直接解决绝对端点、minus background 和 max-risk mismatch。

## 14.4 不立即加宽或堆模块

禁止先做：

- width 32→64；
- attention；
- Transformer；
- 多尺度 decoder；
- 第二 head/第二 branch；
- 多轮 refinement；
- CRF/形态学后处理。

否则无法区分机制修正和容量收益。

## 14.5 不把方向正确写成模型成功

D wrong-direction 6/6 通过只表示符号正确，不等于：

- endpoint calibrated；
- natural misses 可恢复；
- FA 受控；
- 真实 IRSTD 性能提升。

---

# 15. 推荐实施顺序

```text
R13-0 [COMPLETE]
冻结 v12 所有产物，生成 v12 freeze-baseline receipt
        │
        ▼
R13-1 [FAIL, CURRENT TERMINAL]
实现 role-quotient identifiability audit
        │
        ├── FAIL：冻结输入识别负结果；v13 停止
        │          input contract v2 另立版本与因子对照
        │
        ▼ PASS
R13-2
实现 GateCoveredAbsoluteLoss
实现 GateCoveredDualEndpointLoss
        │
        ▼
R13-3
完成 primitive-constraint VJP、loss 和 runner 异常路径测试
        │
        ▼
R13-4
生成 GCDE-v13 implementation closure
独立复核 legacy-76 / extended-84 decision recomputation
        │
        ▼
R13-5
冻结 Holdout 输入但不读取输出
创建 paired Development authorization
同输入、同初始化、同 schedule：
PECO control 与 GCDE candidate 各一次 320-update
        │
        ├── FAIL：冻结 v13；执行归因检查，不自动选择 v14
        │
        ▼ PASS
R13-6
创建预冻结 Holdout authorization
fresh-fit replication 与 frozen-decoder evaluation 分开报告
        │
        ├── FAIL：冻结 v13
        │
        ▼ PASS
R13-7
真实 IRSTD-1K D_R bounded validation
        │
        ▼ PASS
32,000-step replay
formal800 seed 42/43
CURE-Lite frozen confirmation
        │
        ▼
Full CURE
```

---

# 16. 验收清单

创建 GCDE-v13 Development authorization 前，必须全部满足：

```text
[ ] v12 正式目录、result、decision、COMPLETE 未修改
[ ] v12 source baseline 与 closure 保持可复核
[ ] 新方法 ID、schema、protocol root 已建立
[ ] decoder 与 v12 逐参数结构一致
[ ] 参数量仍为 2,593，参数张量仍为 6
[ ] PECO-v10 未被原地覆盖
[ ] GC absolute loss 已新增而非修改旧类
[ ] plus/minus completion 与 occupancy 均显式进入新 criterion
[ ] worst-positive / worst-negative margin 已实现
[ ] D endpoint logit margin 已实现
[ ] H、G_near、G_norm_tail max-delta violation 已分别实现
[ ] minus writable background metric 与 8 项门禁已实现
[ ] 84 项 raw metrics 可独立重算
[ ] primitive-constraint logit gradient 与 shared-parameter VJP 全部通过
[ ] 派生/包含判据的 implication receipt 已完成
[ ] role-quotient identifiability audit 通过
[ ] PECO/GCDE Development 输入、初始化和 schedule fingerprint 相同
[ ] Holdout 在 Development 输出前已冻结
[ ] legacy_population_loss 与 optimizer_objective 定义唯一且分开
[ ] exact-once、strict JSON、finite-state、single-terminal 测试通过
[ ] implementation closure unresolved local imports = 0
[ ] targeted tests 与完整仓库测试全部通过
[ ] 正式结果目录尚不存在
[ ] authorization 尚未提前创建
```

---

# 17. 对当前研究主线的判断

## 17.1 CURE 总问题仍可继续

v12 失败没有证明：

- 冻结 detector 无法补漏；
- occupancy-conditioned correction 不成立；
- paired intervention 没有价值；
- 红外小目标漏检无法由中间特征恢复。

它只否定了：

> 当前 NLCC-v12 decoder、PECO-v10、均值型 absolute risk、plus-only endpoint anchor、冻结输入和 320-update Development 的联合实例。

## 17.2 CURE 主线保持 feature–coverage relevance 与受约束 evidence release

核心研究问题仍是：

> 对任意冻结 IRSTD detector，如何仅从 \((F_b,O)\) 判断哪一部分 coverage 与被
> 抑制的目标证据相关，并只在该处释放 residual，同时在无关覆盖和背景上保持静默？

GCDE-v13 只回答训练代理问题：

> worst-pixel、双 endpoint 约束是否比 PECO 更适合训练当前 NLCC 实例。

即使 GCDE 通过，它也只是 CURE 的配套训练约束；即使 GCDE 失败，也不能单独证明
scalar count 不充分。真正可能形成模型创新的是可推理、detector-independent 的
feature–coverage relevance operator，而不是逐项把 Development 判据写进 loss。

这比无目的增加网络容量更接近可发表的机制贡献。

## 17.3 当前不能作出的声明

仍不能声明：

- CURE 模型成功；
- NLCC 有效；
- 已提高 Pd、IoU 或自然漏检找回；
- 已满足创新；
- 可以设计 Full CURE；
- 可以接入其他 detector 或三数据集。

---

# 18. 最终建议

R13-1 已经运行并失败，因此 **GCDE-v13 成对诊断当前停止**。不能继续实现：

```text
GateCoveredAbsoluteLoss
GateCoveredDualEndpointLoss
paired PECO/GCDE Development
GCDE Holdout
```

当前结果说明，冻结 v12 population 在去除随机幅度/符号和绝对位置身份后存在
7 个相反监督 quotient 冲突；仅替换 objective 不能修复输入角色可识别性。

下一步应建立独立的 CURE 状态版本，聚焦：

> 从通用 \((F_b,O)\) 接口学习 feature-conditioned relevant coverage，并以受约束
> evidence release 取代无方向的 scalar count crossing。

该方向延续而不是修改此前 CURE 主线。它必须先解决当前 7 个 quotient 冲突，
且所有新增状态都能在任意 detector 的推理接口获得；需要拆成可归因版本，不能把
normalization、occupancy basis、affinity 和状态方程一次性同时修改。只有新输入
再次通过 R13-1 等价检查，才重新考虑 PECO/GCDE 成对训练代理实验。

---

# 19. 代码与证据索引

## 正式结果

- [NLCC-v12 Development 正式负结果](https://github.com/Arialliy/cure-lite/blob/main/CURE_Lite_NLCC_v12_Development%E6%AD%A3%E5%BC%8F%E8%B4%9F%E7%BB%93%E6%9E%9C.md)
- [原始 result.json](https://github.com/Arialliy/cure-lite/blob/main/protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/result.json)
- [正式 decision.json](https://github.com/Arialliy/cure-lite/blob/main/protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/decision.json)
- [全部结果与当前研究结论](https://github.com/Arialliy/cure-lite/blob/main/CURE_Lite_%E5%85%A8%E9%83%A8%E7%BB%93%E6%9E%9C%E4%B8%8E%E5%BD%93%E5%89%8D%E7%A0%94%E7%A9%B6%E7%BB%93%E8%AE%BA.md)
- [仓库当前下一步方案](https://github.com/Arialliy/cure-lite/blob/main/CURE_Lite_%E4%B8%8B%E4%B8%80%E6%AD%A5%E6%96%B9%E6%A1%88.md)

## 核心代码

- [NLCC-v12 decoder](https://github.com/Arialliy/cure-lite/blob/main/cure_lite/null_anchored_local_count_crossing_decoder.py)
- [active-v4 factorized topology](https://github.com/Arialliy/cure-lite/blob/main/cure_lite/factorized_decoder.py)
- [PECO-v10 loss](https://github.com/Arialliy/cure-lite/blob/main/cure_lite/paired_endpoint_crossing_losses.py)
- [当前 absolute loss](https://github.com/Arialliy/cure-lite/blob/main/cure_lite/losses.py)
- [当前 dataset-free inputs](https://github.com/Arialliy/cure-lite/blob/main/cure_lite/nlcc_dataset_free_inputs.py)
- [当前 4/4/2 train step](https://github.com/Arialliy/cure-lite/blob/main/cure_lite/train/paired_outcome_step.py)

## 框架语义

- [PyTorch GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)
- [PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
