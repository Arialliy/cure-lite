# CURE-Lite NLCC-v12 Development 正式负结果

> 结果日期：2026-07-27  
> 方法版本：`nlcc_v12`  
> runner/evidence revision：`r2`  
> 源码基线：`538660aa7e200bd7acad8964af25121ea56142cf`  
> 证据角色：dataset-free Development learnability gate，不是真实检测性能  
> 正式决定：`NLCC_V12_DEVELOPMENT_FAIL`

## 1. 结论

NLCC-v12 的唯一一次 320-update Development 已完整执行并形成单一终态：

```text
execution validity          = PASS
structural gates            = 25 / 25 PASS
numeric gates               = 26 / 76 PASS
development decision        = NLCC_V12_DEVELOPMENT_FAIL
automatic retry             = forbidden
Holdout authorization       = false
real D_R authorization      = false
formal800 authorization     = false
Full CURE authorization     = false
```

这是一项有效的模型门禁负结果，不是执行异常。NLCC-v12 在固定
decoder、PECO-v10 loss、输入、阈值、训练日程和推理图下，没有同时建立
factual anchors、背景/no-miss 抑制与 paired endpoint separation。按预先冻结
的停止规则，本候选停在 Development，不运行 Holdout，也不允许修改同一
v12 后重试。

## 2. R0 runner/evidence r2 闭合

Development 启动前，R0-C1 至 R0-C8 已全部闭合：

| 项目 | 结果 |
|---|---|
| raw metrics 独立重算 | PASS；76 项数值门禁全部由原始字段重算 |
| structural gates 独立重算 | PASS；不接受嵌入的汇总结论作为输入 |
| 一次性启动 | PASS；`attempt.json` 与 `training_started.json` 持久化 |
| 逐步有限状态检查 | PASS；首次更新前一次，此后每次 step 后一次 |
| 本地源码闭包 | 43 nodes / 226 edges；unresolved local imports = 0 |
| 单终态发布 | PASS；result/failure 只允许其一 |
| 严格 JSON | PASS；拒绝重复键和非有限常量 |
| targeted tests | 43 passed，0 failed/error/skipped |
| repository tests | 1105 passed，0 failed/error/skipped |

关键绑定：

```text
runner implementation closure fingerprint
  = 143ea8240e538b7c714d1691c3ef4deb4e63044b8ead6aaa11307fbff0264959

R0 verification receipt fingerprint
  = 4805059fc1536301a40db17e6b957cc2d7167bb8a722fe681ccda83140501123

Development authorization fingerprint
  = ada7d20e3e7d406919b29e19d94fdfac534311c0f1d9c9a1149ff31b20bef68c
```

因此本次 Development 的科学结论不依赖旧版 runner 中已经修复的派生结论
接受路径。

## 3. 单终态与执行完整性

正式目录恰有五个文件：

```text
attempt.json
training_started.json
result.json
decision.json
COMPLETE.json
```

其中：

```text
result.json  = 唯一 terminal
failure.json = 不存在
.incomplete  = 不存在
```

执行记录：

| 项目 | 结果 |
|---|---:|
| optimizer updates | 320 / 320 |
| decoder training forwards | 960 / 960 |
| 每步 forward pattern | `(4,4,4)` × 320 |
| decoder states | 12 / update |
| backward calls | 1 / update |
| optimizer steps | 1 / update |
| step-contract failures | 0 |
| gradient failures | 0 |
| 六个参数梯度每步有限且非零 | true |
| checkpoint loaded | false |
| optimizer state loaded | false |
| decoder initialization | from-scratch seed 42 |
| Adam state before first update | empty |

有限状态检查：

```text
finite-state audits          = 321 / 321
before first update          = 1
after optimizer step         = 320
nonfinite elements           = 0
```

最终检查覆盖 6 个参数张量、1 个 buffer 张量和 18 个 optimizer-state
张量，全部有限。

## 4. 独立门禁重算

从重新读取的 `result.json` 原始字段调用独立
`recompute_result_decision()`，得到：

```text
numeric gates     = 76
numeric passed    = 26
numeric failed    = 50

structural gates  = 25
structural passed = 25
structural failed = 0

status            = GATE_FAIL
decision          = NLCC_V12_DEVELOPMENT_FAIL
```

该重算结果与 `decision.json` 中的完整 gate ledger 逐值一致。

四个全局门禁全部失败：

| 门禁 | 实际值 | 冻结要求 | 结果 |
|---|---:|---:|---|
| population total loss | 1.2869663 | `< 0.1` | fail |
| factual-miss target minimum | 0.2876031 | `> 0.95` | fail |
| factual-miss background maximum | 0.9996876 | `< 0.05` | fail |
| factual-no-miss maximum | 0.7426276 | `< 0.05` | fail |

八个冻结 groups 均未整体通过：

```text
clean_adjacent_cell_1px   fail
clean_adjacent_cell_3px   fail
clean_multicount_2to1     fail
clean_multicount_3to2     fail
clean_same_cell_1px       fail
clean_same_cell_3px       fail
component_null_block      fail
component_null_sparse     fail
```

按门禁类型汇总：

| 门禁类型 | 通过 |
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
| D wrong-direction count | 6 / 6 |

由此可以作出的最窄解释是：

- v12 在适用的 clean-D strata 上保持了正确变化方向；
- 远端 normalized tail 保持稳定；
- 但响应幅度、两个绝对端点、matched null 和背景抑制没有同时成立；
- 当前固定的 NLCC-v12 + PECO-v10 联合系统没有通过 Development。

不能仅凭本结果把失败唯一归因于 count-only 表达、指数 crossing、surrogate
gradient 或某一个 loss 项；这些归因需要新版本前的固定对照。尤其不能在
查看本结果后调整阈值、训练步数、loss 权重或状态方程，并继续称为 v12。

## 5. 数值状态与证据边界

最终 operator 记录为：

```text
crossing margin range  = [-6.1761494, 3.8279479]
recovery factor range  = [0.0020784, 45.9681091]
field tensors          = 45
field elements         = 997248
nonfinite elements     = 0
```

因此本次结果不能解释为非有限数值导致的执行失败。

运行边界明确记录：

```text
dataset accessed                 = false
D_R / D_V / D_T accessed        = false / false / false
detection performance evaluated = false
real performance claim          = not authorized
```

所以本结果不提供 Pd、FA、mIoU、nIoU 或真实自然漏检恢复结论。

## 6. 文件绑定

| 文件 | SHA256 |
|---|---|
| `attempt.json` | `f01395ebf0f809190e31b208f50f1c13febd45e2c9ae999ed84936f5a4c09c6e` |
| `training_started.json` | `60d326dca65cc0a85713ac96269e1997b7f95ad148dda19d3e97a89889d3f1b3` |
| `result.json` | `665325f774844ebd2026a9448903d3142653bac1f66729748b9cb5699ab6ae9e` |
| `decision.json` | `c539b5d5d4b5a5cf81f5a4e3865039e252d002aa8afe3005859adb10f73144c2` |
| `COMPLETE.json` | `dfc0219b85366774be97bf2eb5501f4a1cebe1949d7762f8643a5fa971c4807c` |

内部 fingerprints：

```text
attempt          = d798e6b351fd04f188e3c8e1ee0f3dbda90c80d099ea5fdccca52156c85812b9
training-start   = ef9e715eec40a0977705ed0bb1b367f60faf27a5807bafa680e0b31530d3394d
result           = a10127a9b406d2fac464a8037a0edcffa60623d6260052607ff99418bde516bc
decision         = b2572c9caefbabeb0bd60f68c001c92669b090e70966daf27b917039307edfcb
complete         = 75c00f3f71aeefaf6ef8fe7c098265a6224610179fef0ad2350e47658015ad12
```

`COMPLETE.json` 中的四文件 SHA map 与实文件逐项一致。

## 7. 阶段停止决定

```text
R0 runner/evidence r2        PASS
Development execution       COMPLETE
Development scientific gate FAIL
Exposure Holdout            NOT_AUTHORIZED / NOT_RUN
real IRSTD-1K D_R            NOT_AUTHORIZED / NOT_RUN
32,000-step exposure replay NOT_AUTHORIZED / NOT_RUN
formal800 seed 42/43        NOT_AUTHORIZED / NOT_RUN
CURE-Lite freeze            NOT_ACHIEVED
Full CURE                   NOT_AUTHORIZED / NOT_DESIGNED
cross-detector datasets     NOT_AUTHORIZED / NOT_RUN
```

本结果只冻结 NLCC-v12 这一具体候选，不扩大为 CURE 总研究问题已被否定。
若继续研究，下一项工作应先分析 v12 的联合约束失败，并以新的模型版本、冻结
方程和新的 Development/Holdout 路径提出单一机制修订；不能把同一 v12
重跑，也不能跳过 CURE-Lite 直接设计 Full CURE。

## 8. 权威产物

- [R0 verification receipt](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/runner_evidence_r2_r0_verification_receipt.json)
- [Development authorization](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_pre_run_authorization.json)
- [attempt](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/attempt.json)
- [training started](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/training_started.json)
- [raw result](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/result.json)
- [sealed decision](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/decision.json)
- [COMPLETE](protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/development_regression_r1/COMPLETE.json)
