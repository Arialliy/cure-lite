# CURE-Lite CC-SEA v8 r2 真实 \(D_R\) Bounded 负结果

## 1. 正式结论

CC-SEA v8 的修正版 r2 已在冻结的真实 \(D_R\) bounded400 协议下完整执行并封存。

```text
decision=CC_SEA_V8_R2_BOUNDED_MODEL_CODE_GATE_FAIL
structural_execution_pass=true
computational_model_code_gate_pass=false
optimizer_updates_completed=400
passed_computational_gates=6/12
failed_computational_gates=6/12
formal_800_authorized=false
```

这是一个正式的模型代码门禁负结果，不是 r1 的发布核验异常，也不是 Pd、FA 或最终检测性能结果。

权威产物：

- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_cc_sea_v8_bounded_r2/COMPLETE.json)
- [r2 core result](runs/irstd1k_stage_a_seed42/cure_lite_cc_sea_v8_bounded_r2/receipts/result.json)
- [r2 decision](runs/irstd1k_stage_a_seed42/cure_lite_cc_sea_v8_bounded_r2/receipts/decision.json)
- [r2 negative closure](protocols/IRSTD-1K/coverage_conserving_subpixel_evidence_allocation_v8/verifier_correction_r2/bounded_negative_closure_receipt.json)

## 2. r1 与 r2 的状态必须分开

| 运行 | 正式状态 | 能否评价模型门禁 |
|---|---|---:|
| r1 | `EXECUTOR_RESULT_TO_PUBLICATION_CONTRACT_ERROR` | 否 |
| r2 | `BOUNDED_MODEL_CODE_GATE_FAIL` | 是 |

r1 的 verifier 错把结构化 gate record 当作布尔值，因此没有发布原始计算结果。r2 只修正 verifier；模型、core executor、loss、输入、采样计划、400-step 预算和全部阈值均未改变。

r1 保持原样且一次性授权已经消耗；r2 是预先声明的版本化修正执行，不是新增 seed，也不是观察结果后的重复试验。

## 3. 执行完整性

| 项目 | r2 结果 |
|---|---:|
| 预训练结构检查 | 56/56 通过 |
| 完整结构检查 | 48/48 通过 |
| optimizer updates | 400/400 |
| backward calls | 400/400 |
| optimizer steps | 400/400 |
| outcome pairs | 222 |
| outcome pair slots | 800 |
| factual-miss slots | 1600 |
| factual-no-miss slots | 1600 |
| trainable parameters | 4,385 |
| trainable tensors | 6 |
| 最小 update 梯度 L2 | 0.108243 |
| 最大 update 梯度 L2 | 19.170517 |
| 非有限梯度 update | 0 |
| 零梯度 update | 0 |
| decoder forward calls（训练） | 1200 |
| decoder state evaluations（训练） | 4800 |
| 输入携带梯度违规 | 0 |

参数确实发生更新：

```text
initial_decoder_fingerprint=
039cf44ba107bee9c4c11277363ee8808ce26eb3b7fcfbaa53d83096d3d44cb7

final_decoder_fingerprint=
93146751c26c9f3d6d01983f83f3dafc4810a1306470d926e03e2a0912b416b4
```

因此，本次 non-pass 不能归因于训练未执行、梯度中断、参数未更新、采样未覆盖或数值异常。

## 4. 十二项冻结计算门禁

| 门禁 | 方向 | 阈值 | v7 | v8 r2 | r2 |
|---|:---:|---:|---:|---:|:---:|
| factual miss loss：final/initial | ↓ | ≤0.75 | 0.459371 | 0.456103 | PASS |
| factual no-miss loss：final/initial | ↓ | ≤0.75 | 0.039690 | 0.071953 | PASS |
| plus-baseline loss：final/initial | ↓ | ≤0.75 | 0.368250 | 0.407299 | PASS |
| clean transition loss：final/initial | ↓ | ≤0.50 | 0.824516 | 0.844719 | FAIL |
| clean \(D\) mean delta | ↑ | ≥0.50 | 0.185189 | 0.107294 | FAIL |
| clean pairs with \(D\ge0.25\) | ↑ | ≥0.75 | 0.271845 | 0.048544 | FAIL |
| clean joint \(D\ge0.25,H\le0.05\) | ↑ | ≥0.75 | 0.000000 | 0.009709 | FAIL |
| clean-zero mean absolute delta | ↓ | ≤0.05 | 0.061484 | 0.036802 | PASS |
| component-null footprint mean delta | ↓ | ≤0.05 | 0.137834 | 0.083611 | FAIL |
| component-null footprint maximum delta | ↓ | ≤0.25 | 0.899251 | 0.395187 | FAIL |
| component-null context mean delta | ↓ | ≤0.05 | 0.000070 | 0.000037 | PASS |
| identity-null maximum delta | ↓ | ≤\(10^{-7}\) | 0 | 0 | PASS |

## 5. 机制判断

CC-SEA v8 得到了两类真实正信号：

1. factual miss、factual no-miss 和 plus-baseline 三类训练目标均明显下降；
2. identity-null、clean-zero 和 component context 的抑制约束通过；
3. 相比 v7，clean-zero 从失败变为通过，component footprint 的平均值和最大值也明显下降。

这说明共享证据预算与 phase allocation 确实增强了非目标区域的约束，并非完全无效。

但是，核心修复目标没有成立：

1. clean \(D\) 平均增量只有 0.1073，距离 0.50 很远；
2. 只有约 4.85% 的 clean pairs 达到 \(D\ge0.25\)，门槛为 75%；
3. 同时满足修复强度与保持约束的 pair 只有约 0.97%；
4. clean transition loss 仅降到初始值的 84.47%，未达到 50%；
5. component footprint 扰动虽显著小于 v7，仍超过两个冻结上限。

联合解释是：

> CC-SEA 的单一守恒分配方程改善了约束性，却把可用于 clean deletion correction 的有效证据压得过弱；它形成了“更保守，但修复覆盖与强度不足”的结构性权衡。

因此，当前结果不是简单增加训练步数或调阈值即可解决的问题。按照预先冻结的协议，不允许为 v8 调学习率、预算、阈值或再次运行来寻找正结果。

## 6. 当前研究状态

```text
CC_SEA_v8_model_candidate=REJECTED_BY_BOUNDED_MODEL_CODE_GATE
r1_status=PUBLICATION_CONTRACT_ERROR
r2_status=FROZEN_BOUNDED_MODEL_CODE_NONPASS
formal_800_status=NOT_AUTHORIZED
Pd_FA_status=NOT_EVALUATED
D_V_status=NOT_ACCESSED
D_T_status=NOT_ACCESSED
Full_CURE_status=NOT_STARTED
cross_detector_status=NOT_STARTED
```

本结果否定的是这一个冻结的 CC-SEA v8 候选方程，不是否定整个 CURE 研究方向。

主线保持不变：

```text
CURE-Lite 模型设计
    -> bounded 模型代码门禁
    -> 冻结确认
    -> 800-epoch seed 42/43
    -> Full CURE
    -> 跨 IRSTD detector / 三数据集
```

当前停在第一项。下一步必须回到 CURE-Lite 的模型方程设计，解决“修复强度/覆盖”和“footprint 保持”之间的矛盾；在新候选重新通过 toy、dry-run 和真实 \(D_R\) bounded 门禁前，不启动 formal-800、Full CURE 或其他 detector 实验。
