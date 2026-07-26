# CURE-Lite Paired 持久目标完成审计

> 审计日期：2026-07-24  
> 审计模式：claim-audit + numeric-audit  
> 范围：paired 模型代码、冻结边界、分阶段验证、正式 Wave A、失败停止规则  
> 非范围：证明当前机制有效、设计 Full CURE、跨 detector/三数据集验证

## 1. 总结论

本轮持久目标的准确完成状态是：

```text
paired code implementation         = complete
required software/toy validation   = complete
real D_R catalog and schedule gate = complete
formal seed 42/43 training         = complete
frozen Wave A evaluation           = complete
scientific gate                    = PERFORMANCE_FAIL
required failure preservation      = complete
required stop at failed stage      = complete
scoped execution objective         = INCOMPLETE_ONE_MISSING_STATIC_GATE
current mechanism/model success    = false
Full CURE / cross-detector work     = not authorized and not performed
```

正式 `decision.json` 中的 `PERFORMANCE_FAIL` 保持不变。完成审计发现：
代码、正式训练、逐种子判定、失败封存和阶段停止均已发生，但冻结提案要求的
1–3 pixel、stride-4 小目标可表达性尚无独立二值门禁 receipt。因此本轮持久
目标暂时不能结项。

## 2. Claim–evidence matrix

| 目标要求 | 权威证据 | 审计状态 |
| --- | --- | --- |
| 保持 CURE-Lite → 冻结确认 → Full CURE → 跨 detector/三数据集主线 | [README](README.md)、[STAGE](STAGE.md)、[下一步方案](CURE_Lite_下一步方案.md)；正式决定 `authorizes_full_cure=false`、`authorizes_cross_backbone=false` | supported |
| 冻结 Base | [model.py](cure_lite/model.py) 第 47～92 行只接受 `FrozenBaseAdapter`，强制 `requires_grad_(False)`，optimizer 只取 decoder；正式训练只读取冻结 \(D_R\) cache | supported |
| 冻结 `CURELiteDecoder` 拓扑 | [decoder.py](cure_lite/decoder.py) 第 51～94 行；SHA256 `228ec6ac02c637051d87aa12c9c0d6beb79dfb8b87c18fb492ddd00ad5d738d4` 与 proposal 和 formal runner 绑定一致 | supported |
| 冻结单次 Base、单次 decoder、hard union 推理 | [model.py](cure_lite/model.py) 第 94～150 行：一次 `base.extract`、一次 `self.decoder`、`final_mask = occupancy \| residual_mask`；文件 SHA256 与冻结绑定一致 | supported |
| additive `PairExample` / `PairBatch` | [paired_types.py](cure_lite/paired_types.py) 第 86～282、539～721 行；不替换旧类型或旧 step | supported |
| clean/null pair catalog | [paired_catalog.py](cure_lite/experiment/paired_catalog.py)、[paired_types.py](cure_lite/paired_types.py)；权威人口为 206 clean-positive、16 component-null、160 identity-null，只有 clean-positive 可训练 | supported |
| paired difference loss | [paired_losses.py](cure_lite/paired_losses.py) 第 14～152 行直接消费 pre-mask `sigmoid(logits_minus)-sigmoid(logits_plus)`，双端均保留梯度 | supported |
| paired train step | [paired_step.py](cure_lite/train/paired_step.py) 第 197～292 行固定两个 factual anchor 加一个 coupled pair loss，一次 backward 和一次 optimizer step | supported |
| 2B endpoint batched forward | [paired_step.py](cure_lite/train/paired_step.py) 第 130～179 行将 plus/minus 合为一次 `2B` decoder call；[test_paired_step.py](tests/test_paired_step.py) 第 100 行起验证与两次独立 forward 等价 | supported |
| 确定性 pools/schedule | [paired_pools.py](cure_lite/train/paired_pools.py) 固定 800×40、32,000 updates、64,000 pair exposures、source-disjoint batching 和序列指纹；formal schedule 同时绑定 factual/pair 序列 | supported |
| 运行入口与指纹化产物 | [run_paired_formal_training.py](tools/run_paired_formal_training.py)、[paired_formal_runner.py](cure_lite/experiment/paired_formal_runner.py)、[paired_artifacts.py](cure_lite/experiment/paired_artifacts.py)；create-only、COMPLETE-last、严格 loader | supported |
| unit/toy | 全量 `723 passed`；paired 专项覆盖 catalog、loss、step、schedule、artifact、formal runner/evaluation/reveal | supported |
| 双端梯度 | [test_paired_losses.py](tests/test_paired_losses.py) 第 91、112 行起验证双端有限非零梯度和非零 mixed partial | supported |
| pre-mask 语义 | [paired_losses.py](cure_lite/paired_losses.py) 第 17～29 行；[test_paired_losses.py](tests/test_paired_losses.py) 第 143 行起验证相同 raw logits 的差分为零，不由 hard mask 人为产生差分 | supported |
| 小目标表示 | occupancy 投影可见性已有实现、单测和真实 206-pair gate；但 [paired protocol](CURE_Lite_Paired_Objective_协议.md) 第 534～536 行要求的 1–3 pixel stride-4 decoder 可表达性及预冻结背景响应联合门禁没有独立二值 receipt | partial / missing required gate |
| null/control | null 只进入只读诊断、不能进入 optimizer；8 个 matched controls 均完成固定 400-update 工程执行；control provider 与 proposed 隔离 | supported |
| 旧路径回归 | `decoder.py`、`model.py`、`losses.py`、`train/step.py` 哈希保持冻结；paired 代码为新文件/新入口；全量回归通过 | supported |
| 小规模联合过拟合 | [test_paired_toy_overfit.py](tests/test_paired_toy_overfit.py) 第 74 行起验证固定 4/4/2、1:1:1 objective 可联合过拟合；真实 bounded \(D_R\) 运行得到 `COMPUTATIONAL_LEARNABILITY_PASS` | supported |
| 仅使用 \(D_R\) 构造真实 pair catalog | [pair preflight COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_paired_preflight_v1_r1/COMPLETE.json) 和 nested manifest/receipt 均绑定 `split=D_R`、`D_V_accessed=false`、`D_T_accessed=false` | supported |
| seed 42/43 重放 800×40 暴露计划 | [seed 42 exposure](runs/irstd1k_stage_a_seed42/cure_lite_paired_preflight_v1_r1/receipts/exposure_seed42.json)、[seed 43 exposure](runs/irstd1k_stage_a_seed42/cure_lite_paired_preflight_v1_r1/receipts/exposure_seed43.json)：各 32,000 updates、64,000 pair exposures、206 targets 零遗漏 | supported |
| 正式训练前静态代码/数据条件通过 | formal runner 严格绑定 [formal preflight](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_preflight_v1_r1/COMPLETE.json) 和 control preflight；45 个实现文件当前哈希 45/45 与 runner config 一致；但未绑定独立 tiny-target representability receipt | partial，见下方门禁缺口 |
| seed 42/43 的 800-epoch 正式训练 | proposed/control 共四个 attempt 均 strict-load；每个为 800 epochs、32,000 steps、96,000 forwards、384,000 states，参数改变且梯度有限 | supported |
| 无 checkpoint/resume/recovery | 四个 attempt 均 `checkpoint_written=false`、`resume_used=false`；runner 不提供 horizon/checkpoint/resume override | supported |
| 按冻结 Pd/找回数/FA/retention 逐种子判定 | [Wave A decision](runs/irstd1k_stage_a_seed42/cure_lite_paired_formal_wave_a_reveal_v1_r1/decision.json) 由 12 条同协议 evidence 重新计算，禁止一个 seed 抵消另一个 seed | supported |
| 未通过时保留失败证据并停止 | 正式状态 `PERFORMANCE_FAIL`，`next_action=STOP_AND_PRESERVE_EVIDENCE`；published reveal 通过 strict loader 和 COMPLETE inventory 校验 | supported |
| CURE-Lite 确认前不设计 Full CURE、不接其他 detector、不做三数据集扩展 | 未发现 Wave B/C、confirmation、Full CURE 或 CURE 跨 detector 运行目录；\(D_T\) 未读取；正式决定不授权这些分支 | supported |
| 不把额外模块堆入核心机制 | decoder/model 哈希未变；attention、第二 decoder、feature editor 等不存在于 proposed；control 代码是隔离对照，不进入 `paired_difference` provider | supported |

## 3. 冻结实现一致性

[formal runner config](protocols/IRSTD-1K/paired_formal_runner_v1/config.json)
绑定 45 个实现文件。本次审计逐文件重新计算 SHA256：

```text
bound files = 45
hash mismatches = 0
```

三个最关键的冻结文件为：

```text
cure_lite/decoder.py
  228ec6ac02c637051d87aa12c9c0d6beb79dfb8b87c18fb492ddd00ad5d738d4
cure_lite/model.py
  0a1f58895b17f643d46488e5d3e2a082372d54b91ae56e6b21eaaa8f0258fc53
cure_lite/train/step.py
  db89bb7c4f88a4b7e2b066901fa9109347ffba999c0e9b101934a4b29bdf9ffe
```

这支持“paired 实现是 additive 且旧模型/推理/旧 step 没有被覆盖”，不支持
“当前 paired 机制有效”。

### 3.1 未闭合的预声明门禁

[paired proposal receipt](protocols/IRSTD-1K/paired_objective_v1/proposal_receipt.json)
把以下项目列为训练前 required gate：

```text
stride_four_tiny_target_representability_must_be_audited_before_training
```

[paired protocol](CURE_Lite_Paired_Objective_协议.md) 进一步规定：stride-4 toy
必须记录 1–3 pixel target 的不可约表示误差；如果 decoder 不能在预冻结背景
响应约束下表达，状态应为 `STRUCTURAL_FAIL`。

当前已有的：

- adaptive-max occupancy projection；
- 单像素投影不消失测试；
- 206 个 trainable pair 全部 `projection_visible=true`；
- 事后 spatial-tail companion。

这些证据证明“conditioning edit 对 decoder 可见”，但没有单独证明“固定
decoder 能把 1–3 pixel target 表达到冻结前景/背景门槛”。因此不能用它们
替代缺失门禁。

## 4. 真实 \(D_R\) 与 schedule 数字一致性

### 4.1 Pair population

```text
clean-positive  = 206
component-null  = 16
identity-null   = 160
included total  = 382
trainable       = 206
control         = 176
```

算术闭合：

\[
206+16+160=382,\qquad 16+160=176.
\]

权威 pair catalog fingerprint：

```text
4886e52d2cfb3392d0f4fdda376159d6e7f694fd449dc809cf8874793febde76
```

### 4.2 Seed-specific schedule

每个 seed：

\[
800\times40=32{,}000\ \text{updates},
\]

\[
32{,}000\times2=64{,}000\ \text{pair exposures}.
\]

两 seed 的 target exposure 均为 310 或 311，target ESS 为
`205.99953529401705`，zero exposure 为 0。每个 update 的两个 pair 来自
不同 source。

formal joint schedule 进一步固定：

```text
4 factual-miss + 4 factual-no-miss + 2 clean pairs
= 12 decoder states/update
= 3 decoder forwards/update

32,000 × 12 = 384,000 state evaluations
32,000 × 3  = 96,000 decoder forwards
```

formal schedule fingerprints：

```text
seed 42 = 35ed3c818c1e126b4bdb2ae584b8c296795c60493862d558dae2edd224ba2309
seed 43 = f8485dfca0bc531f97cc7ad18b216b99f662cb71eb1569b6b8601ff3d5fe50c4
```

## 5. 正式训练产物

| 方法 | seed | epochs | updates | COMPLETE fingerprint |
| --- | ---: | ---: | ---: | --- |
| `paired_difference` | 42 | 800 | 32,000 | `95e22f2f2641cbe141902aa19f38a1ca476671c5d6bf2b59aaa1575ee318012d` |
| `paired_difference` | 43 | 800 | 32,000 | `68caf02aab2c7eff2923d46a9c85064640580191d4dfa4196550434188bd183f` |
| `independent_endpoint` | 42 | 800 | 32,000 | `762f07ef3f1750cd722bc17b9b583c8f1b7ceaf3ff88ba12681be0fbb2eb8a3c` |
| `independent_endpoint` | 43 | 800 | 32,000 | `5274a1a302beb7f720b71e045a15d0a7e3edf3bf9b55295e2d0f7094be133d16` |

四个 attempt 均满足：

```text
complete_800_by_40 = true
checkpoint_written = false
resume_used = false
D_V_accessed during training = false
D_T_accessed = false
parameters_changed = true
all_gradients_finite = true
```

### 5.1 前置门槛与启动授权的语义边界

可以证明：

- static pair/control/formal preflight 均先完成；
- formal runner 严格绑定其 SHA256 和 fingerprint；
- 当前 45 个冻结实现文件全部通过测试并与运行时 binding 一致；
- 正式训练随后才执行。

不能写成：

> bounded/control diagnostic receipt 自己授权了 formal 800。

这些早期 receipt 明确记录 `authorizes_formal_800=false`。正式启动来自后续
独立冻结执行决策。正确措辞是“全部静态前置条件先满足并绑定后，才由后续
独立决策启动正式训练”。

## 6. Wave A 数字核验

共同 comparison protocol：

```text
fingerprint = cb2fb09c3ec7dbbb0f057d94f7f159e2b4a733296e6ea4a144d6302387014884
D_V images = 120
targets = 170
covered anchors = 147
fixed misses = 23
```

| seed | 方法 | TP / 170 | Pd | recovered / 23 | retention | pixel FA | raw-bg FA | FP comp./MP |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | paired | 147 | 0.8647058824 | 0 | 1.0 | \(2.1107992\times10^{-5}\) | \(7.1716309\times10^{-5}\) | 3.0518 |
| 42 | independent endpoint | 154 | 0.9058823529 | 7 | 1.0 | \(2.8737386\times10^{-5}\) | \(8.6339315\times10^{-5}\) | 5.2134 |
| 43 | paired | 152 | 0.8941176471 | 5 | 1.0 | \(3.3950806\times10^{-5}\) | \(8.4813436\times10^{-5}\) | 6.1035 |
| 43 | independent endpoint | 152 | 0.8941176471 | 5 | 1.0 | \(2.6575724\times10^{-5}\) | \(7.8582764\times10^{-5}\) | 5.8492 |

历史冻结方法也进入同一 cumulative Wave A 比较。最终逐 seed margins 为：

```text
seed 42: -7 TP, -7 recovered
seed 43:  0 TP,  0 recovered
required: +2 TP, +2 recovered for each seed
```

全部方法满足冻结 FA/retention 约束。故失败项是 natural-miss/TP 增益，不是
约束或执行完整性。

正式决定：

```text
status = PERFORMANCE_FAIL
all_seeds_pass = false
next_action = STOP_AND_PRESERVE_EVIDENCE
decision fingerprint =
481413dd8da3af95d4f6bcb8fc28ed001301ea83861cbf583cf27077709df28e
COMPLETE fingerprint =
4ee2c32ff6a00d437d18cad8ec14f8dd1ab790149c8cc80ef7a494c34caa66c8
```

## 7. 软件验证

在当前工作树使用项目运行环境重新执行：

```text
/home/md0/ly/MSHNet/.venv/bin/python -m pytest -q
723 passed in 183.17s
```

另完成：

```text
git diff --check
strict load: pair preflight, formal preflight, four formal attempts,
             reveal config and published Wave A reveal
implementation binding: 45 / 45 hashes match
```

测试证明实现符合已编码契约；它不替代 Wave A 性能结论。

## 8. 未越级边界

本次审计未发现：

- Wave B 或 Wave C 正式 attempt/result；
- frozen-confirmation 运行；
- \(D_T\) 读取；
- Full CURE 实现或运行；
- CURE 接入 DNANet、UIUNet、SCTransNet 的正式验证；
- CURE 在 NUAA-SIRST、NUDT-SIRST 上的扩展实验。

仓库存在历史 MSHNet baseline/adapter，因此准确表述是“没有开展 CURE
跨 detector 验证”，不能扩大成“仓库从未运行 MSHNet”。

## 9. 不支持的声明

以下声明均不受当前证据支持：

- 当前 `paired_difference` 是成功模型；
- 当前 paired 机制有效或必要；
- CURE-Lite 已完成冻结确认；
- 当前结果已经建立 ICLR 所需创新；
- Full CURE 已设计；
- CURE 可以即插即用到任意 IRSTD detector；
- 跨 backbone 或三数据集性能成立；
- \(D_T\) 泛化成立。

## 10. 严重度与后续所有权

```text
engineering integrity severity = one missing required static gate
scientific mechanism severity  = critical negative result
next CCFA owner                = experiment designer for a read-only
                                 tiny-target representability audit
```

当前 paired 版本不得继续调 loss、decoder、pair、阈值或选择性重跑。允许的
唯一补齐项是：不训练、不读取 \(D_V/D_T\)、不修改冻结文件，使用冻结 decoder
与预声明约束生成一个明确标注为 late compliance audit 的小目标表示 receipt。
它不能改写“该门禁没有在正式训练前封存”的时间事实。

## 11. No-invention status

- 所有性能数字均来自已封存的正式结果；
- 所有 fingerprint 均来自当前严格 loader 可验证的配置或产物；
- 没有把测试通过写成性能成功；
- 没有把阶段执行完成写成机制成立；
- 没有虚构 Wave B/C、\(D_T\)、Full CURE 或跨 detector 结果；
- 没有因 Wave A 失败修改冻结门槛。
- 没有把 occupancy projection visibility 冒充完整的小目标可表达性门禁。
