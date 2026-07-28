# CURE-Lite v23 PACRE-VC 正式结果

日期：2026-07-28  
数据集：IRSTD-1K  
正式 seed：42  
结论状态：`PACRE_V23_FORMAL_D_V_ADAPTIVE_FAIL`

## 1. 最终结论

v23 应采用 PACRE-VC，而不是 FACRE。PACRE-VC 的代数、dataset-free
前置验证和唯一真实 D_R 均通过，随后唯一 Formal800 从零完成了
`800 epoch × 40 step = 32,000` 次更新。

但是，唯一自适应 D_V 的真实性能结果为 **FAIL**：

- CURE 相对固定 Base@A 的 true targets 和 recovered misses 都增加 2；
- 但冻结协议要求比较 endpoint-wise best valid Base，而 51 点自适应
  Base@B 在阈值 `0.14` 达到 150 个 true targets、恢复 3 个 anchor misses；
- CURE 只有 149 个 true targets、恢复 2 个 anchor misses，因此两项相对
  best Base 的 margin 都是 `-1`；
- CURE 的 mIoU、nIoU 也低于相应 best valid Base；
- retention 和三项误警安全约束全部通过。

因此，这次 FAIL **不是固定 `+2` 幅度门槛导致的**。当前协议没有任意固定
提升幅度；相对 best valid Base 严格多 1 个 true target、严格多 1 个
recovered miss 即足够。实际结果没有达到严格正提升。

D_T 未访问、未授权，不能进入 D_T 确认。

## 2. 方案裁决

### 2.1 接受 PACRE-VC

采用：

- `CURE_Lite_v22_PACRE_D_R失败修订分析与v23_PACRE_VC实施方案.md`
- residual：`r = p - p_base`
- 补全场：`field = r - stopgrad(r_target)`
- 固定解码：`completion = (field < 0) & ~occupancy`
- `field == 0` 不计为补全

该设计解决了 v22 PACRE 的零读出锚点与训练/推理语义错位问题，并保留
Base occupancy 的硬并集。

### 2.2 拒绝 FACRE 作为本次正式路线

`CURE_Lite_v22_PACRE_D_R代数门禁失败归因与v23_FACRE修改方案.md`
保留为历史分析，但 FACRE 不用于本次冻结执行。原因是 PACRE-VC 已给出
更直接、可验证的 residual/value-centering 修正，并通过 13/13 D_R。

## 3. 门禁含义

### 3.1 没有任意固定性能提升门槛

性能门禁为：

- `true_targets(CURE) > best_valid_Base`
- `recovered_anchor_misses(CURE) > best_valid_Base`
- `mIoU(CURE) >= best_valid_Base`
- `nIoU(CURE) >= best_valid_Base`

其中前两项严格多 1 即可，`minimum_fixed_uplift_margin = null`。

### 3.2 仍然存在必要的技术与安全阈值

下列数值不是“必须提升 +2”一类性能幅度门槛：

- Base@A occupancy threshold：`0.72`
- PACRE-VC field threshold：`0.0`
- Base@B：只允许在冻结的 51 点网格 `0.00, 0.02, ..., 1.00` 中选择
- retention：`1.0`
- pixel Fa：不高于 `1e-4`
- raw background Fa：不高于 `1e-4`
- false-positive components/MP：不高于 `100`

前两项定义固定输出语义；后四项是防止以误警或丢失既有目标换取表面提升的
安全约束。

## 4. 正式执行结果

### 4.1 D_R

唯一真实 D_R：`13/13 PASS`。

关键收据：

- D_R gate receipt fingerprint：
  `0edb3e99259e55b4591b38c7adee1261acf261cd6a7e847cbe261cdef6250d82`
- D_R complete fingerprint：
  `ee9d4b30a225ff73d56224874057475fac45785884a3148047942ef8641e35d3`

D_R 覆盖了数据无关前置、真实 seed42 population、模型/工厂/参数身份、
完整 forward ledger、target algebra/phase、residual flip witness、零读出
锚点、PMOPE 初始梯度、field loss 方向、cache/RNG/gradient 不变性、只读
零更新及 context algebra/phase。

### 4.2 Formal800

| 项目 | 结果 |
|---|---:|
| seed | 42 |
| 从零训练 | 是 |
| epoch | 800 |
| 每 epoch step | 40 |
| 总更新 | 32,000 |
| 训练调用次数 | 1 |
| resume/retry | 否 |
| 中间 checkpoint | 未保存 |
| D_V/D_T 训练期访问 | 均未访问 |

关键指纹：

- Formal artifact：
  `da84597611966cdcd9cb048931a3bcc57394e18252af49294d8f0652377ac3f0`
- model file SHA256：
  `73c497eb79c812ab88959972ffb1dc97e26c55fa0ad67217d70d440e1122df6c`
- final model：
  `7958a9e9d2d3ad072773fec4b6e2963b1fa787fb5488d8edeac217adaf966090`
- Formal complete：
  `4f88f712a0f6e740a0db1a36be537b0202e16dcdb9e4e082e5e8a11a0d60680f`
- source closure：
  `d08a1d84348d8caf8ecee3b0fef3d5efcd56e05e50f46e25b1cf17bd71dfe48c`

独立 Formal verifier 确认：

- `epochs = 800`
- `updates = 32000`
- `steps_per_epoch = 40`
- `from_scratch = true`
- `training_invocations = 1`
- `D_V_preregistration_eligible = true`
- `performance_claim_supported = false`

最后一项是正确状态：Formal800 训练完成本身不构成性能声明，必须等 D_V。

## 5. 唯一 D_V 结果

D_V 共 120 张图。Base@B 在冻结 51 点网格上独立选择阈值 `0.14`。
PACRE-VC 不搜索 field threshold，固定为 `0.0`。

| 指标 | Base@A | Base@B（0.14） | CURE（Base@A+CURE） |
|---|---:|---:|---:|
| true targets | 147 | **150** | 149 |
| recovered anchor misses | 0 | **3** | 2 |
| Pd | 0.8647058824 | **0.8823529412** | 0.8764705882 |
| mIoU | **0.6095592800** | 0.6076294278 | 0.6021406728 |
| nIoU | **0.5653280527** | 0.5640138505 | 0.5608662366 |
| retention | 1.0 | 1.0 | 1.0 |
| pixel Fa | 0.0000211080 | 0.0000242869 | 0.0000255585 |
| raw background Fa | 0.0000717163 | 0.0000820160 | 0.0000778198 |
| false-positive components/MP | 3.0517578125 | 3.4332275391 | 4.3233235677 |
| budget violation | false | false | false |

endpoint-wise best valid Base 为：

- true targets：150（Base@B）
- recovered anchor misses：3（Base@B）
- mIoU：0.6095592799503414（Base@A）
- nIoU：0.5653280526756584（Base@A）

CURE 相对 endpoint-wise best valid Base：

| 门禁项 | margin | 结果 |
|---|---:|---|
| true targets | `149 - 150 = -1` | FAIL |
| recovered anchor misses | `2 - 3 = -1` | FAIL |
| mIoU | `-0.007418607167466784` | FAIL |
| nIoU | `-0.004461816092856785` | FAIL |
| retention = 1 | 满足 | PASS |
| pixel Fa ≤ 1e-4 | 满足 | PASS |
| raw background Fa ≤ 1e-4 | 满足 | PASS |
| FP components/MP ≤ 100 | 满足 | PASS |

这说明 PACRE-VC 确实比 Base@A 多恢复 2 个漏检，但冻结的自适应 Base@B
无需 PACRE-VC 就恢复了 3 个，且分割质量更高。当前模型没有证明相对最强
有效 Base 的增量价值。

## 6. D_V 独立 verifier 纠错

### 6.1 原始失败

唯一 D_V runner 先原子发布：

- `claim.json`
- `receipt.json`
- `decision.json`
- `COMPLETE.json`

随后调用独立 verifier。原 verifier 报：

```text
RuntimeError: persisted model/D_V binding changed
```

只读审计确认这是假阳性：生产端 canonical
`model_binding.formal_artifact` 含合法字段
`formal_result_fingerprint`，原 verifier 的 exact-key 白名单遗漏了该字段。
该字段实际值与 Formal terminal 完全一致。

### 6.2 处置

没有：

- 重跑 D_V；
- 删除合法字段；
- 改写四个发布文件；
- 就地修改冻结 verifier；
- 改性能门禁；
- 访问 D_T。

新增 append-only 纠错登记与更正验签器，仅：

1. 接受并核对 `formal_result_fingerprint`；
2. 将其余 payload 原样委托给 SHA256 已冻结的原 verifier；
3. 继续运行原 terminal graph、51 点独立重选和 gate 重算；
4. 运行前后逐字节复核原 D_V 四文件不变。

更正验签结果：

- `terminal_verified = true`
- `status = PACRE_V23_FORMAL_D_V_ADAPTIVE_FAIL`
- `gate_passed = false`
- `D_V_payload_reopened_by_verifier = false`
- `D_T_payload_accessed = false`
- corrigendum receipt fingerprint：
  `926cc6642a60924ef73a8ebad6206df1eb3561ee551c5fc510aa6c53eabfe182`

纠错材料：

- [纠错登记](audits/pacre_v23_dv_verifier_corrigendum_v1/preregistration.md)
- [更正验签器](audits/pacre_v23_dv_verifier_corrigendum_v1/verify_corrigendum.py)
- [更正验签收据](audits/pacre_v23_dv_verifier_corrigendum_v1/verification.json)
- [纠错审计完成清单](audits/pacre_v23_dv_verifier_corrigendum_v1/COMPLETE.json)

纠错审计完成指纹：
`ca0e95b77d354a2f766338df604bd40f78b39a71e03f3d4b30d936406775f16d`。

原 D_V 文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `claim.json` | `49782cfa6d4b4933733c476d89c3d827356ae573dbf940905a5fe2d19cb50cbf` |
| `receipt.json` | `89527eb76eeded00a9fcf8a8cc96fc69fd8970f30ea53b13d7f11f43101fb1ad` |
| `decision.json` | `8bb002b2b73c974e9bce0b03931625cc49e284d49afda9a8d75feb4aa9aa4b81` |
| `COMPLETE.json` | `2a9b648933dd1d9635943f2a8debd255ac91635722efe17f75b85438ee43a86b` |

## 7. 证据位置

- D_R：
  `runs/irstd1k_stage_a_seed42/cure_lite_pacre_v23_vc_dr_seed42_r1/`
- Formal800：
  `runs/irstd1k_stage_a_seed42/cure_lite_pacre_v23_vc_pmope_formal_800_seed42_r1/`
- D_V：
  `runs/irstd1k_stage_a_seed42/cure_lite_pacre_v23_vc_formal_d_v_seed42_r1/`
- D_V verifier 纠错审计：
  `audits/pacre_v23_dv_verifier_corrigendum_v1/`

## 8. 决策

1. v23 PACRE-VC 的代数修正成立，D_R 和 Formal800 执行有效。
2. v23 PACRE-VC 的 D_V 性能主张不成立。
3. 不因“曾相对 Base@A 增加 2 个目标”而忽略更强的自适应 Base@B。
4. 不引入任意固定 `+2` 性能门槛；严格正提升仍是正确标准。
5. 当前结果不授权 D_T。
6. 如继续研究，应在新的、独立冻结方案中解释：
   - 为什么 Base@B 可恢复 3 个 anchor misses，而 PACRE-VC 只恢复 2 个；
   - 为什么 PACRE-VC 的 mIoU/nIoU 同时回退；
   - 如何在不牺牲 retention/误警预算的前提下获得相对 best Base 的真实增量。
