# CURE-Lite v19 USCOPE bounded-400 正式负结果与 v20 结构决定

> 文档状态：正式结果记录  
> 记录对象：CURE-Lite CMIF v19 USCOPE  
> 正式运行：`cure_lite_cmif_v19_uscope_bounded_400_r1`  
> 数据范围：仅 `D_R`  
> 固定随机种子：42  
> 固定训练预算：10 epochs × 40 steps = 400 updates  
> 正式决定：`USCOPE_V19_BOUNDED_400_GATE_FAIL`

---

## 1. 结论

v19 USCOPE 已完成预声明的 dataset-free、真实 `D_R`、唯一 seed-42
bounded-400 训练、训练后统一证书和零层检测门禁。

正式结果是：

```text
USCOPE_V19_BOUNDED_400_GATE_FAIL
```

失败的两项总门禁为：

```text
every_pair_uniform_certificate
zero_level_detection_gates
```

因此：

- v19 是完整、可审计的 bounded-400 正式负结果；
- v19 不具备进入 Formal800 的资格；
- 未进行新的 `D_V` 或 `D_T` 读取；
- 未进行完整 NUAA、NUDT、IRSTD-1K 三数据集训练与性能验证；
- 未授权 Full CURE 或其他 IRSTD backbone 接入；
- 当前结果不支持性能提升、跨数据集泛化或投稿级有效性主张。

这次结果只否定固定实现与固定协议下的 **v19 USCOPE**。它不等价于
CMIF、CURE-Lite 或 CURE 总方向已经被否定。

下一版本只能重新提出一个新的单机制结构假设。本文档不冻结 v20 的名称、
数学公式、损失形式或实现细节。

---

## 2. 正式证据对象

### 2.1 运行目录

```text
runs/irstd1k_stage_a_seed42/
  cure_lite_cmif_v19_uscope_bounded_400_r1/
```

运行目录包含：

- `COMPLETE.json`；
- `attempt.json`；
- 1 个 safetensors checkpoint；
- 1 个 checkpoint receipt；
- 13 个阶段 receipt。

`COMPLETE.json` 登记的正式产物数为 16。只读审计结果为：

```text
16 / 16 artifact SHA256 exact
all JSON receipt fingerprints exact
COMPLETE fingerprint exact
no FAILURE
no .incomplete
no second v19 bounded-400 run
```

### 2.2 完成标记与主要指纹

| 对象 | 指纹或 SHA256 |
| --- | --- |
| COMPLETE fingerprint | `8ba9d7baea2980d4e24fc66d810d25b7208e518112332dc7f91e231b6ecd7130` |
| COMPLETE file SHA256 | `5abb8ac39614210cef51a6ee0823339a6540bcaefbadc6b5cf7a8e9d4fd1024f` |
| decision receipt fingerprint | `d12ad93582043d95ac80372388f4e4a8decef26d8497aa455f94565bc69dbcc0` |
| bounded result fingerprint | `f830aeb43e4f4b2b65d1304cdf4c779ac97348d4793344dbf4c9fde75022b09d` |
| training receipt fingerprint | `635d6ae9cecd274564e5bd4632fd7307adbf8f3fcb6cf67aca7e7bf81b0ffc34` |
| checkpoint receipt fingerprint | `a10dd532b2c1a9c44fb5d8bb3d62fb83320205a923d638fd87963c059f35bcab` |
| checkpoint file SHA256 | `023976c0ca4705cc84d16fd20978e3aa4838955f51866386156d5ab95cd62378` |
| real-`D_R` evidence fingerprint | `4e629ca3f432b830b680dcc571185b34718353f3477f8243fd69adabaa77816a` |
| zero-level decision fingerprint | `55d8e6a8c4fee0ffe09f22ac1369f83b2d01c06b890a0c1b944c713f6f14605f` |

### 2.3 单次执行与数据边界

正式 receipts 一致记录：

| 项目 | 正式值 |
| --- | --- |
| candidate 数量 | 1 |
| candidate | `uscope_joint` |
| seed | 42 |
| epochs | 10 |
| steps per epoch | 40 |
| completed updates | 400 |
| optimizer steps | 400 |
| training invocations | 1 |
| post-training certificate invocations | 1 |
| zero-level evaluation invocations | 1 |
| resume | false |
| automatic retry | false |
| runtime split | `D_R` |
| `D_V_accessed` | false |
| `D_T_accessed` | false |
| threshold | 0 |
| threshold search | false |

这证明的是固定运行的产物完整性和协议一致性，不是多随机种子稳定性，也不是
三数据集性能复现。

---

## 3. v19 与 v18 的固定比较条件

v19 的正式 authorization 将 v18 保持为只读历史负结果。两次运行共享以下
训练坐标：

| 固定项 | v18 PMOPE | v19 USCOPE |
| --- | --- | --- |
| 模型 | CMIF | CMIF |
| 参数量 | 64,064 | 64,064 |
| 输入表示 | phase-preserving `(F_b, O)` | phase-preserving `(F_b, O)` |
| 初始模型指纹 | `a4086bcffba4035984a8c334b3fa194910bcb7376a573f7f96ef8d36e097240d` | 相同 |
| cache 指纹 | `c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca00754c206092aaf66d8` | 相同 |
| device-cache 指纹 | `76ed2f94b4187154bad62896b93d637f131865f2e4e8dad38becb4bebc71119f` | 相同 |
| schedule 指纹 | `641699803ee6d0472e447c3d4150b7fbe02bc20160d2707658b0d90a175c3c9c` | 相同 |
| optimizer 指纹 | `2d058b1cad606e3c1b723aab05925efb2e873c2b3bf021aeaf0f7df40e0690f0` | 相同 |
| seed | 42 | 42 |
| 预算 | 400 updates | 400 updates |
| 数据范围 | `D_R` | `D_R` |
| 零层阈值 | 0，无搜索 | 0，无搜索 |
| 唯一允许变化 | PMOPE pair objective | USCOPE pair objective |

因此，这次比较可用于判断在上述固定坐标下，更换 pair objective 后观察到的
bounded 行为。由于两个 objective 的数值尺度不同，loss 数值不能直接作为
二者优劣或检测性能的比较指标。

---

## 4. v19 训练已真实发生

v19 完成了 400/400 次更新：

| 训练量 | 首个 40-step 日志 | 最后 40-step 日志 | 变化 |
| --- | ---: | ---: | ---: |
| total | 1.340958786 | 0.443511548 | -66.93% |
| pair loss | 0.599899314 | 0.187858135 | 下降 |
| factual-miss loss | 0.740940762 | 0.201144594 | 下降 |

梯度路径记录为：

- `scalar_energy_weight` 在 update 0 出现首个非零梯度；
- `joint_hidden_bias` 在 update 1 出现首个非零梯度；
- `joint_state_weight` 在 update 1 出现首个非零梯度。

这些事实说明 USCOPE 生成了有限训练信号，CMIF 参数也确实被更新。它们不能
证明最终结构门禁成立；最终判断必须服从训练后 certificate 和 zero-level
结果。

---

## 5. 训练后统一证书

证书使用固定 margin：

```text
m0 = 0.225
```

固定的 32 个 optimizer pair 结果为：

| pair 类型 | 总数 | 证书通过 | raw-sign error |
| --- | ---: | ---: | ---: |
| clean-positive | 16 | 0 | 136 pixels |
| component-null | 16 | 14 | 26 pixels |
| 合计 | 32 | 14 | 162 pixels |

clean-positive 的 `gamma` 统计为：

| 统计量 | gamma |
| --- | ---: |
| min | 0.225528449 |
| mean | 0.308383707 |
| median | 0.288012534 |
| max | 0.460161358 |

clean-positive 的最小值仍高于固定 margin，因此 16/16 clean pair 均未获得
uniform certificate。其 136 个 raw-sign error 中：

- minus endpoint：132；
- plus endpoint：4。

component-null 的两项证书失败均来自样本 `XDU906`，合计 26 个 raw-sign
error。

所以：

```text
every_pair_uniform_certificate = false
```

这不是训练中断或数值异常，而是训练完成后的正式结构门禁失败。

---

## 6. 零层检测门禁

### 6.1 自然状态

| 门禁 | v19 结果 | 正式状态 |
| --- | ---: | --- |
| factual-miss strict | 6 / 16 | fail |
| factual-miss recovered | 14 / 16 | 诊断量 |
| factual target-negative pixels | 236 / 335 | 诊断量 |
| factual-no-miss | 16 / 16 | pass |

`recovered=14/16` 不能替代 strict factual-miss gate。固定协议要求的是完整
strict gate，因此：

```text
factual_miss = false
```

### 6.2 clean-positive

| 指标 | v19 结果 |
| --- | ---: |
| fixed pairs | 16 |
| added-target pixels | 149 |
| minus added-target negative pixels | 49 |
| new completion pixels | 77 |
| completion pixels outside added target | 28 |
| compact-support pass | 0 / 16 |

虽然 v19 产生了 77 个 completion pixels，但没有一个 clean pair 满足完整
compact-support gate：

```text
clean_compact_support = false
```

### 6.3 null checks

| 门禁 | v19 结果 | 状态 |
| --- | ---: | --- |
| fixed component-null | 15 / 16 | fail |
| diagnostic-only null | 1 / 1 | pass |
| identity-null | 16 / 16 | pass |
| factual-no-miss | 16 / 16 | pass |

fixed component-null 中唯一的 zero-level failure 也来自 `XDU906`，产生 2 个
不应出现的 completion pixels。

### 6.4 response 只作为诊断

same-sign response 的正式记录为：

```text
8 / 16 clean pairs all-correct
1318 / 1396 response pixels correct
```

它被预声明为 diagnostic-only，不参与二值完成门禁。因此 response 的改善
不能覆盖 clean compact、component-null 或 factual-miss 的失败。

最终：

```text
zero_level_detection_gates = false
```

---

## 7. 与 v18 PMOPE 的固定条件比较

下表只比较两份正式 bounded-400 receipts 中使用同一统计定义的量：

| 观察量 | v18 PMOPE | v19 USCOPE | 观察 |
| --- | ---: | ---: | --- |
| factual-miss strict | 11 / 16 | 6 / 16 | 减少 5 |
| factual-miss recovered | 15 / 16 | 14 / 16 | 减少 1 |
| factual target-negative | 296 / 335 | 236 / 335 | 减少 60 pixels |
| factual-no-miss | 16 / 16 | 16 / 16 | 不变 |
| clean added-target negative | 123 / 149 | 49 / 149 | 减少 74 pixels |
| clean compact-support | 0 / 16 | 0 / 16 | 均失败 |
| clean new completion | 170 | 77 | 减少 93 pixels |
| completion outside target | 47 | 28 | 减少 19 pixels |
| fixed component-null | 15 / 16 | 15 / 16 | 不变 |
| component-null false completion | 6 | 2 | 减少 4 pixels |
| identity-null | 16 / 16 | 16 / 16 | 不变 |
| response all-correct pairs | 6 / 16 | 8 / 16 | 增加 2，仅诊断 |

该比较支持以下有限结论：

1. v19 的 completion 与 negative-field 输出数量减少；
2. outside-target completion 和 component-null false completion 也减少；
3. 但 added-target completion、factual strict pass 和 factual target-negative
   同时明显下降；
4. clean compact-support 仍为 0/16；
5. v19 没有把这种变化转换为通过证书和零层检测门禁。

因此，v19 不能被描述为性能提升，也不能被描述为只差更多训练步数。当前
证据只支持：

> 在固定 CMIF、输入、初始化、数据、schedule、optimizer 和 400-update
> 预算下，USCOPE 没有形成满足预声明门禁的 target completion。

这仍然只是对固定 v19 的判断，不是对 CMIF 或 CURE 总方向的全局否定。

---

## 8. 源码闭包

### 8.1 v19

源码归档：

```text
artifacts/source_closures/
  cure_lite_cmif_v19_uscope_bounded_400_8ba9d7baea29.tar
```

```text
archive SHA256:
480a80455f19cba4bb6aa1ed0e61a84d0c91ccd968cb9fedde077075172d65d3
```

源码清单：

```text
artifacts/source_closures/
  cure_lite_cmif_v19_uscope_bounded_400_8ba9d7baea29.json
```

```text
manifest SHA256:
d17e16beb7b5ffac1fff306e471229c71e2edcf271970483eb7e264dfcd76226
```

闭包元数据：

| 项目 | 值 |
| --- | --- |
| source files | 44 |
| implementation fingerprint | `8ddfc2af79a33c83e479277937efa8dee7f4116242bbcddd4501443cb0a27fe9` |
| source manifest | `receipts/config.json:implementation.files` |
| archive bytes | 1,464,320 |

### 8.2 v18 历史比较对象

v18 源码归档：

```text
artifacts/source_closures/
  cure_lite_cmif_v18_pmope_bounded_400_bd791fd17a6e.tar
```

```text
archive SHA256:
e32d32cc3ea0e01a5f823a69bc3250e0091dcfdda3ec7d45f2abce05c0bed1a0
```

v18 源码清单：

```text
artifacts/source_closures/
  cure_lite_cmif_v18_pmope_bounded_400_bd791fd17a6e.json
```

```text
manifest SHA256:
c7320a0ea7c2393f4ddfc647ec32be9c2ae15ee01e8167b1f8550f8e48cdba33
```

v18 仅作为只读历史比较对象；v19 没有重新训练或重新评估 v18。

---

## 9. 当前授权边界

正式状态必须写为：

```text
v19_dataset_free_gate = PASS
v19_real_D_R_gate = PASS
v19_bounded_training = COMPLETED
v19_post_training_certificate = FAIL
v19_zero_level_detection_gate = FAIL
v19_bounded_400 = FAIL

formal800_eligible = false
formal_800_authorized = false
performance_evaluation_performed = false
performance_claim_supported = false

D_V_accessed = false
D_T_accessed = false
three_dataset_training = NOT_STARTED
full_CURE_authorized = false
cross_backbone_authorized = false
```

“IRSTD-1K `D_R` bounded-400 已完成”不等于“IRSTD-1K 完整数据集性能训练已
完成”。NUAA、NUDT 和 IRSTD-1K 的完整对比实验均未开始。

---

## 10. v20 结构决定

### 10.1 已经确定的决定

v19 失败后，不允许通过以下操作继续沿用同一版本：

- 增加训练步数；
- 改 seed 寻找正结果；
- 调 margin、threshold 或校准网格；
- 给 USCOPE 增加额外 role、权重或第二个 loss；
- 保留 USCOPE 再叠加补丁模块；
- 直接进入 Formal800；
- 直接进行三数据集训练；
- 直接接入其他 IRSTD backbone。

下一项模型工作必须是：

> 提出一个新的、可独立说明的单机制结构版本。

### 10.2 新结构必须保持的研究边界

下一版本在正式冻结前，至少应保持：

- backbone-independent 的 `(F_b, O)` 输入边界；
- CMIF 主体与单 completion field 主线；
- 固定参数量级与可比较训练预算；
- clean-positive、component-null、identity-null、factual-miss 和
  factual-no-miss 的统一评价；
- same-sign response 继续只作诊断，除非下一版本在训练前明确重新定义问题；
- dataset-free → real `D_R` → seed-42 bounded-400 的串行门禁。

新机制必须同时面对本次已经观察到的三个结果：

1. added-target completion 不足；
2. completion 不能扩散到目标外或 null state；
3. factual-miss strict gate 不能因抑制输出而继续下降。

这三个要求是下一结构的反例集合，不是要求叠加三个独立模块或三个独立
loss。

### 10.3 本文档没有冻结的内容

本文档不决定：

- v20 的方法名称；
- v20 的数学公式；
- v20 使用能量、约束、投影还是其他参数化；
- 新机制的具体层、算子或损失；
- 是否保留 USCOPE 的任何有限风险形式；
- 新增参数的数量或具体超参数。

这些内容必须在独立结构设计中完成，并在查看新的训练结果前形成：

1. 单一结构假设；
2. dataset-free 反例；
3. real-`D_R` 前置门禁；
4. 固定训练和失败停止规则；
5. 新的源码闭包。

只有新版本通过自身 bounded-400 门禁后，才能另行决定是否授权
Formal800。Formal800 通过后，才讨论 NUAA、NUDT、IRSTD-1K 的完整训练与
性能评估。

---

## 11. 不允许形成的结论

当前证据不能支持：

- USCOPE 提高了 IRSTD 性能；
- CURE-Lite 已经设计成功；
- CURE 已经可以作为通用插件接入任意 backbone；
- v19 的 400-step loss 下降能够预测 Formal800 结果；
- 减少 outside completion 等于整体检测更好；
- 单个 seed-42 bounded 结果具有统计稳定性；
- 当前机制已经具备 ICLR 新颖性或投稿充分性。

当前唯一正式结论是：

> v19 USCOPE 在固定 seed-42、真实 `D_R`、400-update CMIF 协议中完成了
> 训练，但统一证书和零层检测门禁同时失败。v19 被保留为正式负结果；
> Formal800 和三数据集实验未获授权。下一步必须设计一个新的单机制结构
> 版本，具体 v20 数学形式尚未冻结。
