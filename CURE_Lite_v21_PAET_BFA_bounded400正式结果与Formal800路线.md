# CURE-Lite v21 PAET-BFA：bounded-400 正式结果与 Formal800 路线

## 1. 当前结论

PAET-BFA v21 已完成并通过唯一一次固定 `seed=42` 的真实
`D_R -> bounded-400` 链路：

```text
decision = PAET_BFA_V21_BOUNDED_400_GATE_PASS
bounded_gate_passed = true
formal800_eligible = true
formal_800_authorized = false
formal_800_executed = false
```

这表示：

1. v21 相比冻结的 v20 同时改善了目标内响应和目标外扩散；
2. PAET-BFA 已获得设计独立 Formal800 协议的资格；
3. 当前还不能声称 CURE-Lite 已完成真实性能验证；
4. 当前不能进入 Full CURE、其他 backbone 或三数据集正式对比。

正式运行目录：

```text
runs/irstd1k_stage_a_seed42/
  cure_lite_paet_bfa_v21_pmope_bounded_400_r1
```

完整运行指纹：

```text
ffc2e3c1cedb63931657f98323f16eee09d34735664da53b52ff40343b5290ef
```

44 个运行源码文件已经独立封存：

```text
artifacts/source_closures/
  cure_lite_paet_bfa_v21_pmope_bounded_400_ffc2e3c1cedb.tar
```

归档 SHA256：

```text
2acbf01c363467c2b7a94c0ee3ec1123c0a1dc3e380a7c8cc19b04ec6edb8776
```

归档成员数量、逐文件 SHA256、正式产物 SHA256 和 COMPLETE 指纹均已重新
核对。

## 2. v21 的单一结构修改

v20 已经将 coverage-conditioned completion 写成 Boolean occupancy
边上的 binary-flip 反对称场，但一个粗网格单元中的 16 个输出相位仍共享
同一个粗特征仿射。因此，occupancy 保留了相位，特征证据却没有和输出相位
对齐。

v21 只修改这一处：

\[
A_F=\operatorname{Conv}(\operatorname{Norm}(F_b);W_F),
\]

\[
A_F^{1:s^2}
=
\operatorname{PhasePack}
\left(
\operatorname{Bilinear}_{\mathrm{align\_corners=False}}(A_F)
\right),
\]

\[
G_p(U,F_b)
=
w^\top
\left[
\operatorname{SiLU}(A_U(U)+A_F^p)
-
\operatorname{SiLU}(A_U(U))
\right],
\]

\[
\Delta_p
=
\frac{1}{2}
\left[
G_p(U,F_b)-G_p(\operatorname{flip}_p U,F_b)
\right],
\]

\[
\phi
=
\operatorname{PixelShuffle}
\left(
0.9+\Delta_{1:s^2}
\right).
\]

其结构边界保持不变：

- 输入仍只有通用的 \((F_b,O)\)；
- 输出仍只有一个 completion field；
- 没有额外 decoder、head、并行分支或曲率模块；
- 参数仍是 3 个张量、64,064 个参数；
- PMOPE、优化器、采样、阈值 0 和推理规则均未改变；
- 变化仅是共享反对称能量内部的 phase-aligned evidence transport。

因此，当前结果支持的是一个单一表示—作用对齐机制，而不是模块堆叠。

## 3. 代码门槛与真实 D_R 门槛

代码级状态：

```text
PAET/BFA 相关联合回归：134 passed
GPU 温控包装器：16 passed
create-only 实际重放：两次逐字一致
create-only 后正式输出目录：不存在
```

真实 `D_R` 门槛结果：

```text
decision = PAET_D_R_IDENTIFIABILITY_PASS
checks = 16/16
target-pass forwards = 32
positive-pass forwards = 96
total forwards = 128
necessary exact collision count = 0
```

同一 \((p,q)\) 绑定检查覆盖 32 个 target group。共有 798 对合法的同粗格
target/background 相位对，798 对的三项分离均严格高于预先固定的
\(128\epsilon_{\mathrm{float32}}\)。

聚合结果为：

| 分离量 | 最小值 | 中位数 | 最大值 |
| --- | ---: | ---: | ---: |
| phase feature \(p\) vs \(q\) | 0.075613 | 0.280579 | 0.570686 |
| transported odd hidden \(p\) vs \(q\) | 0.052707 | 0.101539 | 0.159074 |
| PAET target odd hidden vs BFA-common | 0.005363 | 0.029864 | 0.124467 |

484 个目标像素中有 299 个具有同粗格 Chebyshev-1 合法背景相位。门槛采用
“每个 target group 至少存在一对”的组级条件，因此不会因多像素目标的内部
像素没有直接背景邻相位而错误失败。

`exact collision = 0` 只是一项必要检查，不构成线性读取可行性或正间隔证明，
也不单独授权训练。

## 4. bounded-400 正式结果

### 4.1 v20 到 v21 的直接比较

两次运行使用同一真实 `D_R` population、同一 `seed=42`、同一
`10 x 40 = 400` updates、同一 PMOPE、同一优化器、同一阈值 0。

| 指标 | v20 BFA-CMIF | v21 PAET-BFA | 变化 | v21 门槛 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| factual recovered | 16/16 | 16/16 | 0 | 16/16 | 通过 |
| factual strict | 14/16 | 15/16 | +1 | \(\ge14/16\) | 通过 |
| factual target negative | 310/335 | 324/335 | +14 | \(\ge310/335\) | 通过 |
| clean target negative | 115/149 | 145/149 | +30 | \(\ge124/149\) | 通过 |
| clean outside completion | 54 | 17 | -37 | \(\le46\) | 通过 |
| clean compact support | 1/16 | 5/16 | +4 | \(\ge1/16\) | 通过 |
| component-null | 16/16 | 16/16 | 0 | 16/16 | 通过 |
| factual-no-miss | 16/16 | 16/16 | 0 | 16/16 | 通过 |
| identity-null | 16/16 | 16/16 | 0 | 16/16 | 通过 |
| diagnostic-null | 1/1 | 1/1 | 0 | 1/1 | 通过 |
| invalid completion | 0 | 0 | 0 | 0 | 通过 |

最关键的结果不是某一个计数单独增加，而是：

\[
\text{clean target negative}:115\rightarrow145
\]

与

\[
\text{clean outside completion}:54\rightarrow17
\]

同时发生。

这正面命中了 v20 的失败归因：v20 保留了响应量，但没有把响应可靠地分配到
目标内部；v21 的 phase-aligned feature evidence 显著改善了这一空间分配。

### 4.2 训练状态

```text
completed updates = 400
forward/backward/optimizer steps = 400/400/400
logical state evaluations = 4,800
initial model fingerprint =
  a4086bcffba4035984a8c334b3fa194910bcb7376a573f7f96ef8d36e097240d
final model fingerprint =
  45bbc32f85285bfd1b8bf0e2963d970c3ded432f1f10340795a9f85ffdcd6050
```

平均总损失从第 1 个 epoch 的约 `1.0888` 降至第 10 个 epoch 的约
`0.1967`。三个参数张量均获得非零梯度：

```text
scalar_energy_weight: update 0
joint_state_weight:   update 1
joint_hidden_bias:    update 1
```

### 4.3 开销

| 项目 | 实测值 |
| --- | ---: |
| 参数量 | 64,064 |
| 训练 updates | 400 |
| 训练区间耗时 | 40.839 s |
| 平均每 update | 102.10 ms |
| 峰值 allocated memory | 1,804,080,128 bytes |
| 峰值 reserved memory | 1,918,894,080 bytes |
| OOM | false |

v20 的冻结产物没有同口径的实测峰值显存和 step time，因此当前不构造虚假的
跨版本比率：

```text
NOT_EVALUATED_NO_MATCHED_V20_MEASUREMENT
```

这不影响本轮结构门槛，但正式实验仍需持续报告 v21 的实际时间与显存。

## 5. 为什么现在还不是最终模型成功

本轮通过的是预先声明的“结构晋级门槛”，其问题是：

> PAET 是否同时改善 v20 的目标内响应与目标外扩散，并保留 factual/null
> 能力？

答案是肯定的。

但通用 zero-level 完整门槛仍为 false，主要因为它要求更接近全人口零误差：

- factual strict 目前是 15/16，而不是 16/16；
- clean compact support 目前是 5/16，而不是 16/16；
- 32 对证书仍有 29 个 raw sign error pixels。

因此两个状态不矛盾：

```text
structural advancement gate = PASS
full zero-level population gate = NOT YET PASS
real Pd/Fa/IoU/nIoU performance = NOT YET EVALUATED
```

正确结论是：

> PAET-BFA v21 是迄今第一个通过 bounded-400 结构晋级门槛的 CURE-Lite
> 候选；它证明当前结构值得进入 Formal800，但尚未证明完整 CURE-Lite
> 性能成功。

## 6. Formal800 冻结路线

Formal800 只能在独立代码与协议完成核对后运行，不能直接延长本次 400-step
checkpoint。

必须固定：

```text
model = PAET-BFA v21
objective = PMOPE
seed = 42
epochs = 800
steps_per_epoch = 40
total updates = 32,000
initialization = 与 bounded-400 相同的 seed42 初始状态
resume_from_bounded = false
threshold search on D_R = false
```

执行边界：

1. 在 `D_R` 从相同初始状态独立训练 32,000 updates；
2. 不修改公式、loss、优化器、采样或模型宽度；
3. 不把 bounded checkpoint 当作第 401 步继续训练；
4. 模型选择和结构选择不读取 `D_V/D_T`；
5. `D_V` 只允许按冻结规则进行一次校准；
6. `D_T` 只允许在模型和校准规则固定后进行最终性能评估；
7. 保留 Base-only 与 Base+CURE-Lite 的同前端、同数据、同指标比较；
8. 完整报告 Pd、Fa、IoU、nIoU、参数、显存和推理时间。

Formal800 至少要区分三层结论：

| 层次 | 必需证据 | 当前状态 |
| --- | --- | --- |
| 结构延续 | 32,000-step 后不退化，并保持 bounded 的 target/outside/null 改善 | TBD |
| IRSTD-1K 性能 | Base+CURE-Lite 相对同一 Base 的 Pd/IoU/nIoU 提升且 Fa 受控 | TBD |
| CURE-Lite 正式成功 | 冻结判定全部通过，结果可重放，开销合理 | TBD |

Formal800 的具体数值门槛、校准网格和 `D_T` 读取时点必须先写入代码与凭据，
再启动训练；不得根据正式结果反向修改。

### 6.1 完整 \(D_R\) 的 Formal800 预检结果

本轮在不训练、不读取 `D_V/D_T` 的前提下，已经对完整 \(D_R\) 构造固定
seed42 的 `800 × 40` 抽样计划：

```text
D_R images = 160
natural states = 167
pair states = 383
clean-positive optimization pairs = 206
component-null optimization pairs = 16
formal updates = 32,000

full D_R cache fingerprint =
  569b0fb97d819cf1281ca1d148227bc1c5e229b8301065cb536656b5e578e645
formal schedule fingerprint =
  abc1625c93dc9521b1e824ed4b2e685e867d755d8be5e7b1af3a4a5638240431
formal exposure-gate fingerprint =
  c942578b53fd1ba9524cfcb28d504e9ea205f34af758bda6e9d3b466e5ce2c63
formal exposure gate = PASS
```

| 分支 | record support | record ESS | record max share | source support | source ESS | source max share | zero exposure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| factual miss | 32 | 32.000 | 0.03125 | 24 | 19.692 | 0.09375 | 0 |
| factual no-miss | 135 | 135.000 | 0.007414 | 135 | 135.000 | 0.007414 | 0 |
| clean positive | 206 | 205.998 | 0.004875 | 149 | 126.298 | 0.019438 | 0 |
| component null | 16 | 15.999 | 0.063125 | 14 | 12.774 | 0.125625 | 0 |

这证明完整 \(D_R\) 的正式训练计划可以执行，而且 Formal800 不会误用
bounded-400 的每角色 16 个样本子集。

### 6.2 Formal800 之后的冻结判定

Formal800 结束后先在同一冻结 bounded population 上检查结构保持，最低要求沿用
运行前声明的门槛，不根据 bounded-400 已观察到的更好数值事后收紧：

| 指标 | 最低要求 |
| --- | ---: |
| factual recovered | 16/16 |
| factual strict | ≥14/16 |
| factual target negative | ≥310/335 |
| clean target negative | ≥124/149 |
| clean outside completion | ≤46 |
| compact support | ≥1/16 |
| component / no-miss / identity / diagnostic null | 全部通过 |
| invalid completion | 0 |
| completed updates | 恰好 32,000 |
| threshold search | false |
| 非有限 loss、gradient 或 field | 0 |

同时单独报告通用 population zero-level 结果。它当前是 `FAIL`，不能与
PAET 自定义结构晋级门槛混写：

```text
paet_structural_advancement_gate_passed = true
generic_zero_level_population_gate_passed = false
performance_gate_passed = NOT_EVALUATED
```

只有 Formal800 完整结束且结构保持门槛通过，才允许一次性读取 `D_V`。
PAET 的输出规则固定为：

\[
C=(\phi<0)\land\neg O,\qquad Y_{\mathrm{CURE}}=O\lor C.
\]

`D_V` 不搜索 PAET 的 field threshold；特别地，`phi == 0` 不属于
completion。`D_V` 只允许为 Base-only 对照选择既有 `Base@B` 阈值。

`D_V` 上的开发晋级条件预先固定为全部满足：

\[
TP(C)\geq\max\{TP(A),TP(B)\}+2,
\]

\[
\mathrm{Recovered}(C)\geq
\max\{\mathrm{Recovered}(A),\mathrm{Recovered}(B)\}+2,
\]

\[
\mathrm{Retention}(C)=1,
\]

\[
\mathrm{IoU}(C)\geq\max\{\mathrm{IoU}(A),\mathrm{IoU}(B)\},
\]

\[
\mathrm{nIoU}(C)\geq\max\{\mathrm{nIoU}(A),\mathrm{nIoU}(B)\},
\]

并且：

```text
pixel Fa <= 1e-4
raw-background Fa <= 1e-4
FP components / MP <= 100
```

其中 `A` 是固定 anchor threshold `0.72` 的 Base，`B` 是同一 Base 的
`Base@B` 阈值对照，`C` 是固定 `0.72` occupancy 和固定 `phi < 0`
输出的 Base+CURE-Lite。`D_V` 通过只表示允许准备一次独立测试，不表示
CURE-Lite 已经成功。

## 7. 三数据集与 Full CURE 的授权顺序

只有 Formal800 通过后，才进入：

```text
NUAA-SIRST
NUDT-SIRST
IRSTD-1K
```

三数据集阶段比较：

```text
同一冻结 Base
vs
同一冻结 Base + CURE-Lite
```

它验证的是 CURE-Lite 的跨数据集有效性，不是把 CURE-Lite 写进某一个
特定 backbone。

只有三数据集结果支持通用性后，才讨论：

```text
Full CURE
-> DNANet / UIUNet / MSHNet / SCTransNet 等不同前端
```

当前仍不授权 Full CURE 或跨 backbone 接入。

## 8. 证据—主张对应

| 当前主张 | 已有证据 | 能否成立 |
| --- | --- | --- |
| PAET-BFA 不是额外模块堆叠 | 同一 \((F_b,O)\)、单场、3 参数张量、64,064 参数 | 是 |
| 相位对齐修复了 v20 的空间错配 | clean target `+30` 且 outside `-37`，其他关键约束保持 | bounded 范围内成立 |
| PAET-BFA 已完成真实性能验证 | 尚无 Formal800 的 Pd/Fa/IoU/nIoU | 否 |
| CURE-Lite 已可接入其他 backbone | 尚无三数据集和通用前端验证 | 否 |
| 当前方向具备继续设计价值 | 首次通过全部 bounded 晋级门槛 | 是 |

## 9. 下一项唯一主线

```text
冻结 v21 bounded 正结果
-> 实现并测试独立 Formal800 协议
-> 固定 seed42 从头训练 800 epochs
-> 一次 D_V 校准
-> 一次 D_T 性能评估
-> 依据 Pd/Fa/IoU/nIoU 与开销决定 CURE-Lite 是否正式成功
```

在 Formal800 结果产生前，不修改 PAET 公式，不增加下一模块，不接入其他
backbone。
