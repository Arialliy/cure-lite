# CURE-Lite CCFR-v11 Exposure Holdout 正式负结果

## 1. 正式状态

| 项目 | 状态 |
| --- | --- |
| 方法 | CURE-Lite CCFR-v11（Coverage-Conditioned Feature Release） |
| 阶段 | dataset-free exposure holdout confirmation |
| 权威运行 | r2，唯一科学运行次数 |
| 执行状态 | `COMPLETED` |
| 冻结判定 | `CCFR_V11_EXPOSURE_HOLDOUT_CONFIRMATION_FAIL` |
| 训练契约 | PASS |
| 最终科学门禁 | FAIL |
| 真实 `D_R` 验证 | 未授权、未运行 |
| 32,000-step 暴露重放 | 未授权、未运行 |
| seed 42/43、800 epoch | 未授权、未运行 |
| Full CURE | 未授权、未设计 |
| 跨 detector / 三数据集 | 未授权、未运行 |

本结果是**完成训练后的冻结科学门禁负结果**，不是程序异常，也不是 IRSTD-1K 检测性能实验失败。运行没有访问 `D_R`、`D_V`、`D_T` 或任何真实数据集。

## 2. 运行与封存完整性

授权前测试结果：

| 范围 | 结果 | JUnit SHA256 |
| --- | ---: | --- |
| r2 定向测试 | 18 passed，1 个 receipt-dependent 自检按协议排除 | `f4a6efaf1a4c15ed3b245c574516b1c17f2cfd2f2b5ab9982354ae07b71a2d10` |
| v8–v11 广泛回归 | 261 passed，1 个 receipt-dependent 自检按协议排除 | `afa79fed37ba3b215efe2a7a2205e7987d4da1a88d8e7d369dd8958b6599be94` |
| receipt-dependent 独立自检 | 1 passed | 不作为授权前测试自证据 |

权威产物：

| 产物 | SHA256 | 稳定指纹 |
| --- | --- | --- |
| [attempt](protocols/IRSTD-1K/coverage_conditioned_feature_release_v11/exposure_holdout_attempt_r2.json) | `4a9f9cb7c9a88c7cf54d199b630633bd54890c52febd68fdc5744275eb62286a` | `0c12b7b4346fd8b4727128b029caadbd2a0ac200f7cd70caec2312690e17a0e5` |
| [result](protocols/IRSTD-1K/coverage_conditioned_feature_release_v11/exposure_holdout_result_r2.json) | `8f1ef523212f8eb1b8acd804597f53726e39b64d82b0ae51a8d0d127fb6e0615` | `dc0d81fcaa108c09cc67288e26c90e83598c32c2cdbdfc2c72e5197f6657984a` |
| [COMPLETE](protocols/IRSTD-1K/coverage_conditioned_feature_release_v11/exposure_holdout_result_r2.COMPLETE.sha256) | `e4b1180fd756122984061951c59c297c49a3aaac8c673c123f7e8b3f86838b3b` | 直接绑定 result、attempt、failure、closure 与 pre-run |
| [closure-v2](protocols/IRSTD-1K/coverage_conditioned_feature_release_v11/exposure_holdout_implementation_closure_r2_v2.json) | `e2fff79daf30a0edaa3544e4c6a37f55cf49e13d67b1817a357a77fb87e70740` | `84cee287eb2124f2a484c9289b5c0db5dd0aa7cf749f8cc18f222e535e49ed20` |
| [pre-run-v2](protocols/IRSTD-1K/coverage_conditioned_feature_release_v11/exposure_holdout_r2_pre_run_verification_receipt_v2.json) | `8a2bb3a0fa654d0b4234e1c39a60bd59e9b96a245ef8b010a5bdcdd515f09685` | `444361e7a9942d6be696cbad912a44e50463596998c26a6d342cdf502aff7e91` |

独立审计结果：

- COMPLETE 的固定 10 行哈希链全部匹配；
- closure 中 49 个源码绑定为 0 缺失、0 漂移；
- attempt、result 与所有 receipt 的稳定指纹均独立重算一致；
- 再次启动会在训练前被 create-only authority guard 拒绝，因此本结果不可重跑或覆盖；
- r1 预 attempt 失败与已失效的 r2-v1 证据仍原样保留，没有被当作科学运行次数。

## 3. 训练执行是否成功

训练执行本身完整通过：

| 检查 | 观测 | 状态 |
| --- | ---: | --- |
| 更新次数 | 400 / 400 | PASS |
| 每步 factual miss / factual no-miss / paired | 4 / 4 / 2 | PASS |
| decoder forward | 1200 / 1200 | PASS |
| 每步 decoder state 数 | 12 | PASS |
| 梯度观测 | 2400 次，全部有限且非零 | PASS |
| 梯度 L2 范围 | `5.7682e-5`–`5.2456` | PASS |
| 可训练参数 | 2593，6 个参数张量 | PASS |
| 首步 total loss | `3.544645` | 已记录 |
| 末步 total loss | `1.662279` | 已记录 |

因此不能将最终 FAIL 解释为“没有训练”“梯度断开”“批次错误”或“前向没有执行”。

## 4. 冻结最终门禁

### 4.1 总体与 factual 状态

| 门槛 | 冻结要求 | 观测 | 状态 |
| --- | ---: | ---: | --- |
| population objective | `< 0.1` | `1.660179` | FAIL |
| factual miss target minimum | `> 0.95` | `0.995523` | PASS |
| factual miss background maximum | `< 0.05` | `0.999054` | FAIL |
| factual no-miss maximum | `< 0.05` | `0.503835` | FAIL |

模型能够在 factual miss 目标处产生高响应，但不能同时抑制 factual miss 背景，也不能在 no-miss 状态保持近零输出。

### 4.2 八个预冻结 group

准确的 group population 已被重现，但八组中通过数为 `0/8`。

| 检查 | 冻结要求 | 八组观测范围 | 通过组数 |
| --- | ---: | ---: | ---: |
| plus completion minimum | `> 0.95` | `0.299182`–`0.415316` | 0 / 8 |
| plus background maximum | `< 0.05` | `0.502222`–`0.566760` | 0 / 8 |
| `H` 最大绝对变化 | `≤ 0.05` | `0.038638`–`0.879148` | 2 / 8 |
| `G_near` 最大绝对变化 | `≤ 0.05` | `0.040302`–`0.887233` | 2 / 8 |
| `G_norm_tail` 最大绝对变化 | `≤ 0.05` | `0.037901`–`0.097309` | 2 / 8 |

六个 clean-positive group 的差分检查：

| 检查 | 冻结要求 | 六组观测范围 | 通过组数 |
| --- | ---: | ---: | ---: |
| `D_delta_mean` | `≥ 0.8` | `0.035565`–`0.714521` | 0 / 6 |
| `D_plus_max` | `< 0.05` | `0.166591`–`0.561438` | 0 / 6 |
| `D_minus_min` | `> 0.95` | `0.044722`–`0.997373` | 2 / 6 |
| wrong-direction pixel count | `≤ 0` | 全部为 `0` | 6 / 6 |

## 5. 机制层面的最小合法解释

当前证据支持以下判断：

1. **冻结 holdout 上存在一致的正向符号趋势。** 六个 clean-positive group 的 wrong-direction pixel count 全部为零；这支持当前运行中的方向一致性，但不能单独证明方向机制已经稳定学成。
2. **绝对端点没有成立。** `D_delta`、`D_plus`、completion 和 background 的绝对门槛均未通过；方向正确没有转化为可用的前景/背景端点。
3. **背景抑制是一个显著失败面。** factual miss background 最大值达到 `0.999054`，各组 plus background 最大值约为 `0.50`–`0.57`，说明高目标响应同时伴随显著的非目标峰值；这些 maximum 指标本身不证明响应在空间上广泛分布，也不排除端点幅值和 `H/G` 局部性是并列失败面。
4. **状态局部性没有稳定成立。** `H`、`G_near`、`G_norm_tail` 大部分 group 未保持在冻结的 `0.05` 范围内。
5. **这不是实际 Fa 数值。** 本轮没有真实图像或连通域级检测评估；当前只是不通过预冻结的背景代理门槛，不能据此推断或报告数据集 Fa。

matched v8 comparator 同样为 FAIL。CCFR 的 population objective（`1.660179`）略低于 v8（`1.678487`），population pair loss（`0.782449`）也略低于 v8（`0.818695`），且 factual miss target 从 v8 的 `0.922162` 提升到 `0.995523`。这些只是冻结 holdout 内的诊断性差异，不能抵消 CCFR 自身绝对门槛失败，也不能构成方法成功或检测性能提升。

## 6. 阶段决策

本轮结论仅否定 **CCFR-v11 当前冻结候选**，不把它扩大为整个 CURE 方向失败。但是，当前候选不能进入真实数据验证：

```text
CCFR-v11 dataset-free holdout = FAIL
    -> freeze negative result
    -> no rerun / no threshold relaxation
    -> real IRSTD-1K D_R = NOT AUTHORIZED
    -> 32,000-step replay = NOT AUTHORIZED
    -> seed 42/43 formal800 = NOT AUTHORIZED
    -> Full CURE / other detectors / three datasets = NOT AUTHORIZED
```

下一项研究工作必须先形成一个新的、预声明的核心候选，用同等级的 dataset-free 门禁重新证明：

- paired 方向正确；
- absolute foreground endpoint 成立；
- background/no-miss endpoint 同时成立；
- `H/G` 非目标区域变化受控；
- 不通过增加独立模块、放宽阈值或重复运行寻找正结果。

只有新的 CURE-Lite 候选通过这些代码级门禁后，才使用 IRSTD-1K 的 `D_R` 开始真实数据受控验证；之后才是 32,000-step 暴露重放和 seed 42/43 的 800-epoch 正式训练。三大数据集仍位于 CURE-Lite 冻结确认、Full CURE 定型之后。

## 7. 无虚构声明

本文档只使用已封存的 r2 真实产物及冻结阈值。没有生成、推断或补写任何 IRSTD-1K、NUAA-SIRST、NUDT-SIRST 的 Pd、Fa、IoU、nIoU 或训练结果。
