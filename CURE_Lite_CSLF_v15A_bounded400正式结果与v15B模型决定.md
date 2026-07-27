# CURE-Lite CSLF-v15A：bounded-400 正式结果与 v15B 模型决定

> 结果状态：**完整结束，候选门禁未通过**  
> 数据范围：仅冻结 \(D_R\)  
> 运行：seed 42，三目标各 400 updates  
> \(D_V/D_T\)：未读取样本、预测或指标  
> Formal-800、Full CURE、其他 backbone、三数据集：均未授权

---

## 1. 正式结论

v15A 不是运行中断，也不是代码执行失败。它完整训练、评估并封存了 candidate
和两个 matched controls，得到的科学判定为：

```text
decision = BOUNDED_COMPLETION_ROOTED_CSLF_GATE_FAIL
failed_checks = [candidate_zero_level_gates]
formal_800_authorized = false
D_V_accessed = false
D_T_accessed = false
```

唯一失败的顶层检查是 `candidate_zero_level_gates`。其余 27 项 bounded 执行检查
全部通过。

完成端锚定产生了明确且很大的预期作用：

\[
\text{added-target negative pixels}:
\quad
\frac{6}{149}
\longrightarrow
\frac{125}{149}.
\]

但它同时形成过宽的 synthetic completion，并使部分 factual-miss 状态回退。因此
当前不能进入 Formal-800，也不能把 v15A 作为成功模型。

最准确的阶段判断是：

> v15A 证明了“绝对根放在哪个 endpoint”确实控制零水平穿越；但全域
> minus-root 把“越零不足”变成了“越零显著、支撑外扩与 factual 竞争”。

这否定的是“全域 completion-rooted”这一具体坐标，不是否定单一 coverage-state
field、有限 coverage response 或整个 CURE 方向。

---

## 2. 产物与执行完整性

权威目录：

```text
runs/irstd1k_stage_a_seed42/
  cure_lite_cslf_v15a_completion_rooted_bounded_400_r1/
```

完整性结果：

| 项目 | 结果 |
|---|---:|
| `COMPLETE.json` | 存在 |
| `.incomplete` | 不存在 |
| `FAILURE.json` | 不存在 |
| 声明产物 | 17 |
| 磁盘实际产物 | 17 |
| 产物 SHA256 | 17/17 匹配 |
| receipt 连接 | 9/9 匹配 |
| checkpoint 连接 | 3/3 匹配 |
| 跨产物关系核对 | 51/51 通过 |
| threshold search | 未执行 |
| calibration | 未执行 |
| detection performance evaluation | 未执行 |

关键指纹：

```text
COMPLETE file SHA256 =
    a08089470c107e1ac8b3c0a799c4abc2ad865e3cb43616b378c89db168140a37

complete fingerprint =
    f925ece389a96cd6e8ef5487d91428d7981764b12601133cc3eaf9d11b782d35

bounded-result receipt fingerprint =
    7972b5aa14b2c6fd0055e4d60dd40982ea731f3af1c688d2b1c978902ee68e4e

decision receipt fingerprint =
    7d116ad93ae9cc5ba37687680b0c7b918c3ae350fba43028255fe726f39d565f

runtime implementation fingerprint =
    7087f339973b5daf06108be5f4a25b3ee5466e79f581037488102d1981e6ec22
```

三个 checkpoint 权重 SHA256：

```text
completion_rooted_response_joint =
    7202467004a573f2bc0d9b00e80b66ae8b4f5baea7786b71cbe04fed1ae90ab2

identity_joint =
    a7a3735f9565bf129520b31e7af2f6653aea9824f4e5c8978b915c945369ca42

separable_endpoint =
    147ec39931f5c775d33a0917c237c4d3ca1c621f659dfc28594e3d2ac9f89ad3
```

---

## 3. 本轮唯一变量

父 v15 response-joint：

\[
\mathcal L_{\mathrm{pair}}^{v15}
=
\mathcal E_\mu
\left(
e_+,\,
e_- - e_+
\right).
\]

v15A candidate：

\[
\mathcal L_{\mathrm{pair}}^{v15A}
=
\mathcal E_\mu
\left(
e_-,\,
e_- - e_+
\right).
\]

其中：

\[
e_\pm=\phi_\pm-\phi_\pm^*.
\]

除绝对根从 plus endpoint 移到 minus endpoint 外，以下内容均未改变：

- 模型结构、参数量、\((F_b,O)\) 输入；
- 单一 scalar completion field；
- finite coverage response；
- target field、measure 和空间 Sobolev energy；
- natural factual-miss / no-miss branches；
- population、cache、schedule、初始化和 optimizer；
- 每目标 400 updates；
- 推理零水平 \(\phi<0\) 与 hard union。

两个 controls 的训练日志、最终模型指纹以及 130 个状态的 field/completion/final
指纹均与父 v15 逐项一致。因此 candidate 的变化可以归因于 root coordinate，
而不是数据、初始化、schedule 或 control 漂移。

---

## 4. 同预算训练结果

三个 objective 都满足：

```text
seed = 42
epochs = 10
steps_per_epoch = 40
updates = 400
forward = 400
backward = 400
optimizer steps = 400
logical states = 4800
finite-state checks = 401
device = cuda:0
```

最后一个 epoch 的均值：

| Objective | pair loss | factual-miss loss | no-miss loss | total loss | grad L2 |
|---|---:|---:|---:|---:|---:|
| v15A completion-rooted | 0.201619 | 0.093546 | 0.036156 | 0.331321 | 1.779030 |
| identity-joint | 0.172346 | 0.094316 | 0.033241 | 0.299903 | 1.261232 |
| separable-endpoint | 0.085010 | 0.097875 | 0.039316 | 0.222200 | 2.185821 |
| 父 v15 response-joint | 0.209696 | 0.088521 | 0.033752 | 0.331969 | 1.358607 |

v15A 相对父 response 的 total loss 只下降约 \(0.000649\)，但零水平几何发生了
大变化。这再次说明：训练 loss 大小不能替代固定零水平门槛。

---

## 5. Zero-level 总结果

| 指标 | 父 v15 response | v15A candidate | identity control | separable control |
|---|---:|---:|---:|---:|
| response sign | 1396/1396 | **1396/1396** | 1395/1396 | 1396/1396 |
| added-target negative | 6/149 | **125/149** | 4/149 | 137/149 |
| added target 全负 pair | 0/16 | **8/16** | 0/16 | 10/16 |
| clean compact exact | 0/16 | **1/16** | 0/16 | 9/16 |
| clean component match | 3/16 | **15/16** | 2/16 | 15/16 |
| factual-miss gate | 15/16 | **13/16** | 13/16 | 15/16 |
| factual target negative | 321/335 | **289/335** | 294/335 | 308/335 |
| factual recovered | 16/16 | **16/16** | 16/16 | 16/16 |
| factual-no-miss | 16/16 | **16/16** | 16/16 | 15/16 |
| component-null | 17/17 | **17/17** | 17/17 | 15/17 |
| identity-null | 16/16 | **16/16** | 16/16 | 16/16 |
| scalar-hidden | pass | **pass** | pass | pass |

Controls 只用于解释机制，不参与 candidate 的逻辑合取。v15A candidate 自身失败的
三个门槛为：

```text
clean_defined_metrics = false
clean_compact_support = false
factual_miss = false
```

通过的四个门槛为：

```text
factual_no_miss = true
component_null = true
identity_null = true
scalar_hidden_diagnostic = true
```

---

## 6. v15A 证明了什么

### 6.1 完成端绝对锚定有效

added-target 负像素从 6 增至 125，增量为 119 个像素；越零比例从约 4.0% 提高到
约 83.9%。全目标为负的 pair 从 0/16 提高到 8/16，component match 从 3/16
提高到 15/16。

因此可以正式保留：

> 在相同模型、数据、初始化和计算预算下，将绝对根从 covered plus endpoint
> 移到 completion-bearing minus endpoint，会显著增强目标区域的零水平穿越。

### 6.2 Finite response 与 null 行为未被破坏

v15A 保持：

- response sign 1396/1396；
- no-miss 16/16；
- component-null 17/17；
- identity-null 16/16；
- scalar-hidden 通过；
- invalid completion pixels 为 0。

因此本轮不是以破坏所有 null states 为代价换取越零。

---

## 7. v15A 为什么仍然失败

### 7.1 主要失败：支撑外扩

v15A clean states 中：

```text
added-target pixels = 149
target 内 negative pixels = 125
target 内未越零 pixels = 24
new completion pixels = 168
target 外 new completion pixels = 43
plus false-island components = 3
compact exact pairs = 1 / 16
```

16 条 clean pair 中：

- 8 条目标全部越零，但其中 7 条仍存在 spill；
- 12/16 存在目标外新增 completion；
- 唯一完整通过的是 `XDU853`；
- `XDU678` 仍为 0/7，并且 component 不匹配。

所以 v15A 的新失败签名是：

\[
\text{completion crossing 显著增强}
\quad\land\quad
\text{completion support 过宽}.
\]

这与现有 response target 的宽空间支撑一致：1396 个 response pixels 远大于
149 个 added-target pixels。全域使用 \(e_-\) 作为绝对根，会把 completion 端的
压力施加到整个 pair measure，而不是只施加到实际新增目标支持。

### 7.2 次要失败：factual-miss 回退

v15A factual-miss 失败记录：

| Sample | v15A | 父 v15 | 95% 门槛 |
|---|---:|---:|---:|
| XDU680 | 46/75 = 61.33% | 63/75 = 84.00% | fail / fail |
| XDU325 | 72/81 = 88.89% | 80/81 = 98.77% | fail / pass |
| XDU774 | 95/103 = 92.23% | 102/103 = 99.03% | fail / pass |

总覆盖由 321/335 降至 289/335。13/16 factual-miss 的二值 completion 与父结果
一致，但 XDU325、XDU774 从通过变为失败。这说明 synthetic pair 的全域
minus-root 梯度与 natural factual branch 在共享参数上产生了更强竞争。

---

## 8. 当前模型状态

当前可以正式固定的状态是：

```text
v15 = frozen negative:
    response correct, absolute crossing largely absent

v15A = frozen negative with positive mechanism evidence:
    absolute crossing strongly improved
    response/null preserved
    compact support and factual stability failed

CURE-Lite complete model = not yet successful
Formal-800 = not authorized
Full CURE = not started
cross-backbone = not started
three-dataset training = not started
```

这不是返回 P0 或重新证明数据前提。模型代码和真实 \(D_R\) 训练路径已经工作；当前
问题已经收敛到一个明确的模型目标坐标：如何在 added-target 内保留 minus-root，
同时避免在其余空间使用全域 minus-root。

---

## 9. v15B 唯一候选：Support-Oriented Response Root

下一版暂定：

```text
CSLF-v15B
support_oriented_response_joint
Support-Oriented Response Root, SORR
```

定义 added-target 支持：

\[
A(x)
=
\mathbf 1
\left[
Y_-(x)=1,\;
Y_+(x)=0
\right].
\]

由于现有 target field 在 target 内严格为负、valid non-target 内严格为正，也可由
冻结 target fields 精确计算：

\[
A
=
\left(\phi_-^*<0\right)
\land
\left(\phi_+^*>0\right).
\]

定义空间定向的绝对根：

\[
e_{\mathrm{root}}(x)
=
A(x)e_-(x)
+
\left[1-A(x)\right]e_+(x).
\]

新 pair objective：

\[
\boxed{
\mathcal L_{\mathrm{pair}}^{v15B}
=
\mathcal E_\mu
\left(
e_{\mathrm{root}},\,
e_- - e_+
\right)
}
\]

它表达一个统一原则：

- 在真正新增 target 的支持内，使用 v15A 已证明有效的 minus-root；
- 在其余位置，绝对值坐标恢复 plus-root；
- finite response coordinate 完全保留；
- null pair 中 \(A=\varnothing\)，整个目标精确退化为父 v15 response-joint；
- natural factual/no-miss branches 完全不变。

必须注意：非空 \(A\) 时，只能声称 \(A\) 外的**绝对值坐标**等于 plus-root。
Sobolev spatial difference 会跨越 \(A\) 的边界，因此不能声称 \(A\) 外的完整能量
与父 v15 完全等价。

### 9.1 可识别性

逐像素地：

- \(A=1\)：\(e_{\mathrm{root}}=e_-=0\)，再由
  \(e_- - e_+=0\) 得 \(e_+=0\)；
- \(A=0\)：\(e_{\mathrm{root}}=e_+=0\)，再由
  \(e_- - e_+=0\) 得 \(e_-=0\)。

因此两个 endpoint 仍被唯一确定。

### 9.2 为什么不是模块堆叠

v15B 不增加：

- 网络层、head、branch 或 decoder；
- 第三个 loss coordinate；
- loss 权重、margin、温度或阈值；
- 新数据、新标签或推理输入；
- 训练步数或后处理。

模型仍然只有一个：

\[
(F_b,O)\longmapsto\phi_\theta(F_b,O).
\]

pair coordinate 仍然是二维 joint energy。\(A\) 只由训练时已有的 target difference
确定，推理时不存在。因此这是同一 coverage-state response 机制的空间定向根，
不是多个模块拼接。

---

## 10. v15B 代码与验证顺序

下一步严格按以下顺序进行：

1. 保留 v15 与 v15A 的函数、枚举、CLI 和产物，不改写历史语义。
2. 新增 `support_oriented_response_joint` policy 和独立 loss 入口。
3. 从冻结 target fields 精确构造 \(A\)，不新增 cache 字段。
4. 仅替换绝对根；response、measure、`_pair_energy`、模型和 natural branches 不变。
5. 新增 dataset-free 判别：
   - \(A\) 内 anchor value 精确等于 \(e_-\)；
   - \(A\) 外 anchor value 精确等于 \(e_+\)；
   - added-target 内 minus endpoint 有直接梯度；
   - null pair 整体 loss 精确退化为父 v15；
   - exact endpoint fixed point 为零；
   - \(A\) 边界梯度有限；
   - 两次 canonical receipt 逐字节一致。
6. 记录 clean-pair 与 factual-miss 的梯度 cosine 和分层范数，只作诊断，不加入
   loss，也不作为新门槛。
7. 全量测试与 dataset-free 两次重放全部通过后，才能另行冻结一次 v15B
   bounded-400 协议。

v15B 的完整门槛不放宽。相对 v15A 的中间预期只用于解释：

- added-target crossing 不应大幅退回；
- spill pixels 必须下降；
- plus false islands 必须下降；
- factual 13/16 必须恢复。

最终成功仍要求原完整门槛全部通过：

```text
factual-miss = 16/16
factual-no-miss = 16/16
clean response = 1396/1396
added-target negative = 149/149
compact exact = 16/16
target-outside new completion = 0
component-null = 17/17
identity-null = 16/16
scalar-hidden = pass
```

若 v15B 失败，必须冻结结果并重新判断；不得在同版本追加步数、调阈值或创建 r2。

---

## 11. v15A 源码闭合

在任何 v15B 代码修改前，v15A 的 37 个实际运行源码已经建立独立确定性归档：

```text
archive =
  artifacts/source_closures/
  cure_lite_cslf_v15a_completion_rooted_bounded_400_f925ece389a9.tar

archive SHA256 =
  c6c6cf6ae53d7a8a1dcae0c7d531eec8abaa7cead55c0e73ef37442c9d98ff79

archive bytes = 1013760
source files = 37

manifest =
  artifacts/source_closures/
  cure_lite_cslf_v15a_completion_rooted_bounded_400_f925ece389a9.json

manifest SHA256 =
  232239773c5b327828352907a5e7281b96a47ae63685a0f5d8ed33317477ee32
```

归档内 37 个文件均已逐文件重新计算 SHA256，并与本次
`receipts/config.json:implementation.files` 完全一致。

---

## 12. 最终阶段决定

当前必须同时保留两句话：

1. **v15A 没有通过模型门槛，不能进入 Formal-800。**
2. **v15A 提供了强机制证据：completion endpoint 的绝对根显著控制目标越零。**

所以正确下一步不是扩大数据集、接 backbone、增加训练量或退回纯诊断，而是实现
一个更精确、仍为单机制的空间定向 root：

\[
\boxed{
\left[
A e_-+(1-A)e_+,\;
e_- - e_+
\right].
}
\]

只有 v15B 在同一冻结 \(D_R\) 门槛上完整通过后，才讨论 Formal-800。CURE-Lite
获得稳定成功证据后，才进入 Full CURE、其他 IRSTD backbone 和三数据集验证。
