# CURE-Lite v20 BFA-CMIF bounded-400 正式结果与下一步结构判断

> 文档状态：正式结果记录  
> 方法：BFA-CMIF（Binary-Flip Antisymmetrized CMIF）  
> 有效正式运行：`cure_lite_bfa_cmif_v20_pmope_bounded_400_r2`  
> 数据范围：仅 `D_R`  
> 固定随机种子：42  
> 固定训练预算：10 epochs × 40 steps = 400 updates  
> 正式决定：`BFA_CMIF_V20_BOUNDED_400_GATE_FAIL`

---

## 1. 最终结论

v20 BFA-CMIF 已完成：

1. 数据无关结构检查；
2. 更正后的真实 `D_R` 前置门槛；
3. 唯一 seed-42 bounded-400 训练；
4. 训练后 32-pair 证书；
5. 固定零阈值检测评估；
6. 16 个正式产物的字节完整性封存。

有效运行 r2 的正式结论为：

```text
BFA_CMIF_V20_BOUNDED_400_GATE_FAIL
```

这不是程序失败，也不是训练未发生。r2 的真实 `D_R` 前置门槛
15/15 通过，400/400 次更新完整执行。最终失败来自两个预声明的
零阈值推进条件：

```text
clean_target_negative_gt_123_of_149 = false
clean_outside_completion_lt_47 = false
```

因此：

- v20 BFA-CMIF 是一个完整、有效的 bounded-400 模型负结果；
- v20 不进入 Formal800；
- 不进行 `D_V`、`D_T` 或完整三数据集训练；
- 不进入 Full CURE；
- 不接入 DNANet、UIUNet、MSHNet 或 SCTransNet；
- 当前结果不支持正式性能提升或跨骨干泛化结论。

但是，v20 不是“完全无效”。它相对 v18 同时改善了事实漏检修复、
compact support 和 null 保持，只是在 clean-positive 的目标对齐与
外部完成控制之间形成了不合格的混合权衡。

---

## 2. r1 为什么不是方法结果

第一次运行：

```text
cure_lite_bfa_cmif_v20_pmope_bounded_400_r1
```

终止于：

```text
BFA_CMIF_V20_DR_GATE_FAIL
```

r1 报告了 10 条“同一 BFA 奇分量表示需要互斥输出”的冲突。后续逐条审计
证明，这 10 条全部来自前置门槛的角色掩码错误：

- 5 个事实漏检目标坐标；
- 每个坐标在 component-null 的 plus/minus 两端各被错误计为一次背景；
- 合计 `5 × 2 = 10` 条冲突；
- 没有发现其他冲突类型。

实际 PMOPE 在这些 endpoint target 像素要求负场：

\[
\phi \le -0.225
\quad\Longleftrightarrow\quad
\Delta \le -1.125.
\]

旧门槛却把同一像素误标为背景，要求：

\[
\phi \ge 0.225
\quad\Longleftrightarrow\quad
\Delta \ge -0.675.
\]

两者严格互斥，但互斥来自门槛标注错误，而不是 BFA 表达能力。

修正仅为：

```text
component-null background
= valid
  ∩ writable
  \ removed_component
  \ endpoint_target
```

以下内容均未改变：

- BFA 模型方程；
- `(F_b, O)` 输入；
- PMOPE 损失；
- 数据人口；
- seed 42；
- 400-step 预算；
- optimizer；
- 数值门槛；
- 零阈值推理。

r1 被完整保留，但其科学状态是：

```text
gate_annotation_invalid
not_a_method_result
```

r1 完成指纹：

```text
32ec204dff8563aa7aa301e773d0110da72534f7e8492a5a94f9254a7942297c
```

r1 源码封存：

```text
artifacts/source_closures/
  cure_lite_bfa_cmif_v20_pmope_dr_gate_32ec204dff85.tar
```

---

## 3. r2 的真实 `D_R` 前置门槛

更正后的 r2 结果为：

```text
BFA_D_R_IDENTIFIABILITY_PASS
```

15 项检查全部通过：

| 检查 | 结果 |
| --- | --- |
| dataset-free 结构门槛 | pass |
| 固定 BFA 模型契约 | pass |
| 固定 seed-42 人口 | pass |
| 真实输入仅来自 `D_R` | pass |
| 96 个唯一状态各前向一次 | pass |
| 32 个目标组均有有限非零奇分量 | pass |
| 奇分量与曲率非退化 | pass |
| target/background/component 损失下降方向 | pass |
| 零 readout 的非零梯度路径 | pass |
| 非零 readout 的 joint weight/bias 梯度路径 | pass |
| 模型、cache 与 RNG 不变 | pass |
| optimizer/parameter step 为 0 | pass |
| 完整分布收据 | pass |
| 实际角色坐标非空 | pass |
| 精确互斥表示冲突 | `0`，pass |

关键计数：

| 项目 | 数值 |
| --- | ---: |
| 唯一 context states | 96 |
| 实际 forward | 96 |
| factual target groups | 16 |
| clean target groups | 16 |
| target coordinates | 484 |
| component coordinates | 218 |
| background coordinates | 6,289,272 |
| 精确冲突 | 0 |

`D_R` evidence fingerprint：

```text
fed3667f08dab8ae678abd194877fbdb60237121c2dc84c51e879db92ecac623
```

这说明 BFA 在固定真实输入上具备有限、非退化、可训练的结构路径。它不等价于
训练后检测门槛必然通过。

---

## 4. bounded-400 训练已真实完成

正式训练完成：

```text
400 forward
400 backward
400 optimizer steps
4,800 logical state evaluations
```

三组参数的首个非零梯度更新：

| 参数 | 首个非零更新 |
| --- | ---: |
| `scalar_energy_weight` | 0 |
| `joint_state_weight` | 1 |
| `joint_hidden_bias` | 1 |

训练日志：

| 指标 | 前 40 steps | 后 40 steps | 变化 |
| --- | ---: | ---: | ---: |
| total loss | 1.089172 | 0.247448 | -77.28% |
| factual-miss loss | 0.739664 | 0.148500 | -79.92% |
| pair loss | 0.349128 | 0.061616 | -82.35% |
| factual-no-miss loss | 0.000379 | 0.037333 | 上升 |

模型从初始指纹：

```text
a4086bcffba4035984a8c334b3fa194910bcb7376a573f7f96ef8d36e097240d
```

更新到：

```text
e67186d5c86127c15b18076cc4518d44adae94ee03d8b91389f6ea891f64bd6a
```

因此，v20 不是“梯度为零”或“训练没有作用”。损失显著下降，同时
factual-no-miss loss 从接近零上升，已经显示出修复强度与背景保持之间的
实际权衡。

---

## 5. 零阈值正式结果

### 5.1 事实状态

| 指标 | v20 | 门槛 | 状态 |
| --- | ---: | ---: | --- |
| factual strict | 14 / 16 | `> 11/16` | pass |
| factual recovered | 16 / 16 | `16/16` | pass |
| factual target negative pixels | 310 / 335 | 诊断量 | — |
| factual-no-miss | 16 / 16 | `16/16` | pass |
| invalid completion pixels | 0 | `0` | pass |

两个未达到 strict 的样本仍然成功找回目标，但目标区域没有全部形成负场：

- `XDU680`：62/75 target pixels 为负；
- `XDU774`：94/103 target pixels 为负。

### 5.2 clean-positive

| 指标 | v20 | 门槛 | 状态 |
| --- | ---: | ---: | --- |
| added-target pixels | 149 | 固定人口 | — |
| target-negative pixels | 115 | `> 123`（至少 124） | fail |
| new completion pixels | 169 | 诊断量 | — |
| completion outside target | 54 | `< 47`（至多 46） | fail |
| compact-support pairs | 1 / 16 | `> 0/16` | pass |
| 完整 clean pair gate | 1 / 16 | 诊断量 | — |

失败不是“没有生成完成区域”。相反，模型生成了 169 个新 completion
pixels，但其中 54 个落在目标之外，同时目标内部只有 115/149 个像素变负。
因此问题是空间归属不够准确：

```text
completion response exists
but target alignment and exterior control are insufficient
```

### 5.3 null 状态

| 指标 | v20 | 门槛 | 状态 |
| --- | ---: | ---: | --- |
| component-null | 16 / 16 | `>= 15/16` | pass |
| diagnostic null | 1 / 1 | pass | pass |
| identity-null | 16 / 16 | `16/16` | pass |

这说明 BFA 没有通过破坏 null 保持来换取事实漏检修复。

---

## 6. 与 v18、v19 的固定人口比较

| 指标 | v18 PMOPE | v19 USCOPE | v20 BFA |
| --- | ---: | ---: | ---: |
| factual strict | 11/16 | 6/16 | **14/16** |
| factual recovered | 15/16 | 14/16 | **16/16** |
| factual target-negative | 296/335 | 236/335 | **310/335** |
| clean target-negative | **123/149** | 49/149 | 115/149 |
| clean new completion | 170 | 77 | 169 |
| clean outside completion | 47 | **28** | 54 |
| clean compact support | 0/16 | 0/16 | **1/16** |
| component-null | 15/16 | 15/16 | **16/16** |
| factual-no-miss | 16/16 | 16/16 | 16/16 |
| identity-null | 16/16 | 16/16 | 16/16 |

v20 相对 v18 的变化为：

- factual strict：`+3`；
- factual recovered：`+1`；
- factual target-negative：`+14 pixels`；
- compact-support：`+1 pair`；
- component-null：`+1 pair`；
- clean target-negative：`-8 pixels`；
- clean outside completion：`+7 pixels`。

因此，v20 证明了二值翻转反对称场能够改善事实漏检方向和 null 保持；但它没有
同时保持 v18 的 clean target coverage 与外部控制。预声明门槛要求所有方向
同时改善，不能用前四项的提升抵消后两项下降。

---

## 7. 训练后 32-pair 证书

证书完整性通过，但证书本身按预声明只作为训练后诊断，不替代零阈值推进门槛。

| pair 类型 | 数量 | certificate pass | raw sign errors |
| --- | ---: | ---: | ---: |
| clean-positive | 16 | 1 | 88 |
| component-null | 16 | 14 | 4 |
| 合计 | 32 | 15 | 92 |

这与零阈值结果方向一致：主要剩余问题集中在 clean-positive，而不是
component-null。

---

## 8. 结构判断

v20 的主要正结果不是“性能已经成功”，而是：

> binary-flip antisymmetry 将事实漏检修复从 v18 的 11/16 提高到
> 14/16，并同时把 recovered、compact support 和 component-null 推进到
> 更好的方向。

主要失败不是奇分量退化、梯度缺失或 null 崩溃，而是：

> 当前逐相位局部翻转场对目标内部与邻近可写背景的空间归属仍不够精确。

同一个中心相位的局部交互能够产生足够强的负响应，但其响应边界与目标几何
没有被场方程本身充分约束，结果表现为：

1. 事实漏检方向增强；
2. clean-positive 仍有 34/149 target pixels 未变负；
3. 同时产生 54 个目标外 completion pixels；
4. factual-no-miss loss 在训练中由接近零升至 0.037333。

所以，下一步不能回到：

- 调阈值；
- 增加 seed 寻找正结果；
- 延长到 800 epochs；
- 更换 loss 权重；
- 叠加边界模块、注意力模块或后处理；
- 接入其他 backbone。

这些做法不会直接修复当前已定位的场方程空间选择性问题。

---

## 9. 正式产物与源码封存

有效运行目录：

```text
runs/irstd1k_stage_a_seed42/
  cure_lite_bfa_cmif_v20_pmope_bounded_400_r2/
```

完整性状态：

```text
16 / 16 artifact SHA256 exact
no FAILURE.json
no .incomplete
400 / 400 optimizer steps
D_V_accessed = false
D_T_accessed = false
```

这里的“完整性”严格指文件清单、SHA256、receipt fingerprint、模型 checkpoint
和数值链。独立审计发现一项必须披露的归档身份字段缺陷：

```text
receipts/bounded_result.json: result.run_id
= cure_lite_bfa_cmif_v20_pmope_bounded_400_r1
```

该字段来自 `coverage_state_bfa_bounded_runner.py` 中遗留的硬编码字符串。其余
运行身份均为 r2，包括：

- 运行目录；
- `attempt.json`；
- `config.json`；
- `decision.json`；
- `COMPLETE.json`；
- protocol correction binding；
- checkpoint、训练、证书和零阈值结果的 SHA256 链。

r1 没有发生训练，也没有 checkpoint；r2 是唯一完成 400 次更新和训练后评估
的运行。因此，这个陈旧字符串不改变 r2 的数值、权重或
`BFA_CMIF_V20_BOUNDED_400_GATE_FAIL` 科学判定，但它意味着不能声称
“所有运行身份字段完全一致”。

正式状态为：

```text
scientific_negative_conclusion_valid = true
artifact_hash_integrity = true
artifact_semantic_identity_fully_consistent = false
stale_nested_run_id_disclosed = true
formal800_eligible = false
```

现有 r2 不做静默改写，也不为修复一个归档标签重新训练。后续 runner 必须从
外层运行配置传入 `run_id`，不得再硬编码运行编号。

主要指纹：

| 对象 | 指纹 |
| --- | --- |
| COMPLETE | `8908a8c1896951e46fd737aa6f7fef2c9935e6524632b3576b8069faa026e2eb` |
| config | `b8996b25ffafc4db4a943a6ebf5902a91cfa5375390e29d9037b1ad7bef82d23` |
| bounded result | `77e0e9382be807aa991ce6ee7b189e8fd10b0bfcbd819359802e9500cf1154b8` |
| decision | `a1d9e78f307da683cda59fd27059154de87bffe3326fb678d8f0bcd2343ec945` |
| `D_R` evidence | `fed3667f08dab8ae678abd194877fbdb60237121c2dc84c51e879db92ecac623` |
| checkpoint | `040d2ca4ffa012c813e2c3e5dfa2c6f4877a91c8ff0b901bf8dc83df62026c42` |

源码封存：

```text
artifacts/source_closures/
  cure_lite_bfa_cmif_v20_pmope_bounded_400_8908a8c18969.tar
```

归档 SHA256：

```text
f4e004a5cb2f74036008e4ea0567b6cdf9132104f5c14669b086c155ac6f35b5
```

归档包含配置绑定的 44 个源码文件，逐文件 SHA256 与正式配置一致。

---

## 10. 下一步

当前应冻结 v20，不再重跑。

下一阶段只允许提出一个新的单机制场方程，其目标必须同时是：

\[
\text{保留 v20 的 factual/null 改善}
\]

以及：

\[
\text{提高 clean target alignment}
\quad\text{并}\quad
\text{减少 outside completion}.
\]

下一候选仍需保持：

- 通用 `(F_b, O)` 输入；
- 单一 completion field；
- 不访问特定 backbone 内部；
- 不引入 decoder 堆叠；
- 不修改 PMOPE、seed、数据人口和 400-step 预算；
- 先经过数据无关检查与真实 `D_R` 前置门槛；
- 只有 bounded-400 同时超过全部 v18 门槛，才进入 Formal800。

v20 结果没有改变 CURE 的总体研究主线。它将下一步问题从泛化的
“能否学习修复场”收缩为更具体的：

```text
如何让同一个反对称完成场在保持事实漏检响应的同时，
获得目标内外一致的空间选择性。
```
