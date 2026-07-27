# CURE-Lite CSLF-v15：真实 \(D_R\) 可观测性正式结果

> 日期：2026-07-27  
> 数据范围：IRSTD-1K 冻结 \(D_R\)  
> 协议：`irstd1k-dr-cslf-observability-v1`  
> 代码快照基线：`70178d461386052d4bf7b0b66c0258b3a187b9c7` 加本地 CSLF-v15 修改  
> 当前决定：`AUTHORIZE_SCALAR_CSLF`  
> 训练状态：未授权、未执行

## 1. 本轮回答的问题

本轮不比较检测性能，也不训练 decoder。它只回答：

1. representation-neutral 的真实 \(D_R\) coverage states 是否完整且可追踪；
2. 同一实际模型输入是否对应唯一的绝对 completion target field；
3. 标量 occupancy projection 是否足以实现该 target contract；
4. 是否必须切换到 phase-preserving occupancy basis。

本轮没有读取新的 \(D_V\) 或 \(D_T\)，没有运行 Formal-800，没有接入其他
IRSTD backbone，也没有修改 Base、decoder 深度、loss 阶数或推理流程。

## 2. 先发现并修正的 target contract 问题

初次重放把同一多漏检图像中的每个 focus target 分别作为绝对 target field。
因此，同一 \((F_b,O)\) 会对应多个不同目标场，产生 7 个 scalar/phase 共同的
duplicate-input target conflict。

这不是 occupancy representation 的失败，而是监督对象定义不满足函数单值性。
修正后采用：

\[
Y_{\mathrm{scene}}(F_b,O)
=
\bigcup_{g\in\mathcal M_{\mathrm{eligible}}}
\left(GT_g\land\neg O\right).
\]

即同一 actual input 只有一个 scene-complete absolute completion field。每个
目标仍保留独立 focus record，但它只改变 loss integration measure，不改变
绝对场：

\[
\phi_g^\star=\phi^\star(Y_{\mathrm{scene}}),\qquad
\mu_g\ \text{可以因 focus target 而不同}.
\]

这项修改保持 CSLF 的单一 coverage-state completion field 主线，不增加新网络
模块，也不改变模型输入。

## 3. 正式人口

| 项目 | 数量 |
|---|---:|
| natural records | 167 |
| clean-positive pairs | 206 |
| component-null pairs | 17 |
| identity-null pairs | 160 |
| pair records 合计 | 383 |
| explicit exclusions | 216 |
| unique normalized features | 160 |
| unique occupancy states | 289 |
| unique target fields | 231 |

旧 scalar-specific population 曾排除
`XDU792 / component_null / pred_id=2`。representation-neutral raw catalog
恢复了该状态，因此 component-null 从 16 增至 17；是否对 scalar/phase 可见
由后续 observability 审计决定，而不再改变原始人口。

## 4. 正式可观测性结果

| 指标 | 结果 |
|---|---:|
| full-grid changed pairs | 223 |
| phase changed pairs | 223 |
| scalar projected changed pairs | 222 |
| scalar-hidden pairs | 1 |
| scalar-hidden clean-positive pairs | 0 |
| scalar-hidden component-null pairs | 1 |
| target-response pixels | 19,722 |
| response outside scalar RF | 0 |
| response outside phase RF | 0 |
| response hidden only by scalar | 0 |
| scalar duplicate-input target conflicts | 0 |
| phase duplicate-input target conflicts | 0 |
| identity-null nonidentical pairs | 0 |

唯一 scalar-hidden pair 是
`XDU792 / component_null / pred_id=2`，且其 target response 为零。它说明
phase representation 保留了更多 occupancy 相位信息，但没有揭示 scalar
CSLF 无法实现的监督响应。

## 5. 正式决定

三项决定性条件均满足：

\[
\text{scalar target conflicts}=0,
\]

\[
\text{target response outside scalar RF}=0,
\]

\[
\text{identity-null nonidentical}=0.
\]

因此：

```text
decision = AUTHORIZE_SCALAR_CSLF
selected_representation = scalar_max
PP-CSLF-v16 = NOT_AUTHORIZED / NOT_NEEDED
training_authorized = false
next_route = build_scalar_cslf_cache_and_fused_step
```

这里“授权 scalar”只表示现有实际输入能够实现冻结的 target contract，不表示
CSLF 已经训练成功或性能有效。

## 6. 确定性重放

正式目录：

- `runs/irstd1k_stage_a_seed42/cure_lite_coverage_state_observability_v1_r1`
- `runs/irstd1k_stage_a_seed42/cure_lite_coverage_state_observability_v1_r2`

r1/r2 的 `config.json`、`raw_catalog.json`、`observability.json`、
`decision.json` 与 `COMPLETE.json` 全部逐字节一致。

关键语义指纹：

```text
config_fingerprint
= ddae14af0bf10d4be9d5cb4549f04b3df84ec35475c970ea6b95ac1296c8bf92

raw_catalog_fingerprint
= 15eaf6d5482d908f2c0f9899e4495eaff329ccb4d5c2aae648c4bba98ef79a24

observability_receipt_fingerprint
= 0c603611582534ca686fb177ac5786db4fee3f0c03f434b994062312e97ec214

complete_fingerprint
= 9baaf7b08959ac47d25ed8d725fdf2ffd58289685cd4cd453d9833eef7fbe7b3
```

逐文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `receipts/config.json` | `2f69349473cf871a887b0893a29d1afbb2ceadc8c419d6d327c2e29a0a59547c` |
| `receipts/raw_catalog.json` | `3d5730fd446d94d7c3c2b631a69c3181affb8bd09de2cd438bbf65bfa01f064d` |
| `receipts/observability.json` | `849db7061c0d45759be8652e097aa495ce83be549034dfce9dc6da461be70d69` |
| `receipts/decision.json` | `afd3f8108348d0024af5029727be070ad69dfb8ae4936c9c2b83dc6f6c7e6d07` |
| `COMPLETE.json` | `0cb23d750c08393f5a320bf4646c9a430cb301cd089953106724577615359b14` |

## 7. 测试

当前定向矩阵：

```text
tests_v15/test_coverage_state_raw_catalog.py
tests_v15/test_coverage_state_observability.py
tests_v15/test_coverage_state_level_set.py
```

当时结果：

```text
42 passed in 4.78s
```

随后完成 cache、batch、fused step、schedule 与 matched runner 后，当前合并
定向矩阵为：

```text
76 passed in 26.51s
```

## 8. 能证明与不能证明的内容

本轮已经证明：

- raw population 不再由 scalar visibility 预先筛选；
- scene-complete target contract 对实际输入是单值的；
- 目标响应均位于当前 scalar CSLF 的结构感受域内；
- 当前真实 \(D_R\) 不需要 PP-CSLF；
- 固定输入下正式证据包可精确重放。

本轮尚未证明：

- 12-state fused optimization 正确；
- response-joint 优于 identity-joint 或 separable-endpoint；
- bounded training 能收敛且无额外负岛；
- CSLF 提升 \(P_d\)、漏检找回或误报约束；
- seed 42/43 稳定；
- CURE-Lite 已设计成功；
- Full CURE 或跨 backbone、跨数据集有效。

## 9. 下一步

原定的前五项代码现已实现并通过真实 \(D_R\) GPU 烟测。当前唯一主线是保持
scalar CSLF 核心方程不变，继续：

```text
create-only cache/schedule receipt
    -> expanded dataset-free structural gate
    -> D_R-only bounded gate
    -> 通过后才授权 Formal-800
```

在这些阶段完成前，不启用 PP-CSLF，不进入 Full CURE，不接入其他 IRSTD
backbone，也不开始三大数据集的完整性能实验。

## 10. 后续代码闭合补充

真实 \(D_R\) cache 已成功生成：

```text
cache_fingerprint
= 569b0fb97d819cf1281ca1d148227bc1c5e229b8301065cb536656b5e578e645

natural = 167
pairs = 383
clean-positive trainable = 206
component-null total/trainable/diagnostic-only = 17/16/1
identity-null diagnostic = 160
```

GPU0 三目标各 2 updates 的代码烟测满足：

- 同一初始模型指纹；
- 同一 schedule；
- 每 update 一次 forward、一次 backward、一次 optimizer step；
- 每 update 恰好 12 logical states；
- `phase_projection` 在 update 0 获得非零梯度；
- trunk 在 update 1 获得非零梯度；
- identity-null 与 diagnostic-only component-null 优化暴露均为 0；
- 未读取 \(D_V/D_T\)，未执行检测性能评估。

```text
common_initial_model_fingerprint
= 25b467cae0c23a4ce55a5d1153a85a4345999acb9cbd4ab498a970978423aaf1

two_update_schedule_fingerprint
= 8c64722d5440d9e86cde93d6faaa93b6d19be80642c3689ee316c512874b5f6a

matched_smoke_result_fingerprint
= c0646c6142121c4b1f7f32c80ab4104ea51d76c3a54e76a8d71a4819a950f351
```

这些是模型代码闭合证据，不是性能结果；正式 bounded 与 Formal-800 仍未授权。
