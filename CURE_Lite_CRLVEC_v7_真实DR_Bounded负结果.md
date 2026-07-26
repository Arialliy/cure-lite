# CURE-Lite CR-LVEC v7：真实 \(D_R\) Bounded400 负结果

## 1. 正式结论

CR-LVEC v7 已完成唯一一次、预先冻结的 IRSTD-1K \(D_R\) bounded400 运行。

最终状态为：

```text
decision=BOUNDED_MODEL_CODE_GATE_FAIL
structural_execution_pass=true
bounded_model_code_gate_pass=false
optimizer_updates_completed=400
formal_800_authorized=false
same_version_retry_authorized=false
```

这不是程序异常，也不是数据加载失败。模型完成了全部 400 次反向传播和参数更新，31 项结构执行检查全部通过；但 12 项学习结果检查中仅 5 项通过、7 项失败。因此，当前 CR-LVEC v7 不能进入 formal800 seed 42/43。

本结果只评价 CURE-Lite 模型代码在冻结 \(D_R\) bounded 任务上的可学习性，不是 Pd、FA 或最终检测性能结果。

## 2. 冻结模型与运行条件

- 方法：Continuously Recoverable Log-Vacancy Evidence Crossing，CR-LVEC v7。
- decoder：`CURELiteCrossingFactorizedDecoder`。
- 参数量：4,385，共 6 个参数张量。
- 输入：冻结 Base 特征和 occupancy。
- 优化器：Adam。
- 学习率：\(10^{-3}\)。
- 更新量：\(10\times40=400\)。
- 每次更新：3 次 decoder forward、12 个 decoder states。
- 设备：GPU0，NVIDIA GeForce RTX 3090。
- 温控：82°C 暂停、75°C 恢复；本次未触发暂停。
- 数据范围：仅 \(D_R\)；未访问 \(D_V\) 或 \(D_T\)。
- 未执行校准、检测推理或 Pd/FA 评估。
- 未更新 Base 或 backbone。
- 禁止恢复、自动重跑和同版本第二次运行。

实际执行完全符合预算：

| 项目 | 预期 | 实际 |
|---|---:|---:|
| optimizer updates | 400 | 400 |
| backward calls | 400 | 400 |
| optimizer steps | 400 | 400 |
| training forward calls | 1,200 | 1,200 |
| training state evaluations | 4,800 | 4,800 |
| total forward calls，不含结构检查 | 1,220 | 1,220 |
| margin observation calls | 1,251 | 1,251 |
| margin 额外 forward | 0 | 0 |

全部梯度有限且非零：

\[
\min_t\lVert g_t\rVert_2=0.13233,\qquad
\max_t\lVert g_t\rVert_2=4.44169.
\]

## 3. 十二项冻结结果

### 3.1 通过的 5 项

| 检查 | 实际值 | 阈值 |
|---|---:|---:|
| factual miss final/initial | 0.45937 | \(\leq0.75\) |
| factual no-miss final/initial | 0.03969 | \(\leq0.75\) |
| plus baseline final/initial | 0.36825 | \(\leq0.75\) |
| component-null context mean \(|\Delta|\) | 0.000070 | \(\leq0.05\) |
| identity-null max \(|\Delta|\) | 0 | \(\leq10^{-7}\) |

这说明：

1. v7 的可恢复梯度路径确实能够训练；
2. factual 两个分支均能显著降低损失；
3. 完全相同的 identity endpoints 保持严格一致；
4. 远离干预区域的 component-null context 基本不受影响。

### 3.2 失败的 7 项

| 检查 | 实际值 | 阈值 |
|---|---:|---:|
| clean transition final/initial | 0.82452 | \(\leq0.50\) |
| clean mean \(\Delta\) on \(D\) | 0.18519 | \(\geq0.50\) |
| clean pairs with \(\Delta_D\geq0.25\) | 0.27184 | \(\geq0.75\) |
| clean zero-region mean \(|\Delta|\) | 0.06148 | \(\leq0.05\) |
| component-null footprint mean \(|\Delta|\) | 0.13783 | \(\leq0.05\) |
| component-null footprint global max \(|\Delta|\) | 0.89925 | \(\leq0.25\) |
| joint \(D\geq0.25,\ H\leq0.05\) fraction | 0 | \(\geq0.75\) |

最关键的失败不是“完全学不动”，而是：

\[
\text{目标删除区域响应增强}
\quad\text{但}\quad
\text{局部非目标区域与 component-null footprint 同时增强}.
\]

因此 v7 获得了更强响应，却没有获得所需的选择性。

## 4. 相对 v6 的变化

| 指标 | v6 | v7 | 方向 |
|---|---:|---:|---|
| clean mean \(\Delta_D\) | 0.13346 | 0.18519 | 改善 |
| clean pair \(\Delta_D\geq0.25\) fraction | 0.09709 | 0.27184 | 改善 |
| clean transition final/initial | 0.86621 | 0.82452 | 改善 |
| clean zero-region mean \(|\Delta|\) | 0.03800 | 0.06148 | 变差 |
| component-null footprint mean \(|\Delta|\) | 0.07582 | 0.13783 | 变差 |
| component-null footprint max \(|\Delta|\) | 0.79033 | 0.89925 | 变差 |
| joint selectivity fraction | 0.02913 | 0 | 变差 |

四个目标尺度层均呈现同一趋势：\(D\) 上的响应高于 v6，但 \(H\) 上的非目标变化也同步升高，最终所有尺度层的 joint pass fraction 均为 0。

所以 v7 的正式结论是：

> 连续可恢复 crossing 解决了 v6 的弱响应问题的一部分，但把主要瓶颈转化为更严重的局部选择性问题。

## 5. 不能得出的结论

本次结果不能说明：

- CURE 整体方向失败；
- CURE-Lite 已经成功；
- v7 提升或降低了最终 Pd、FA；
- 应当放宽阈值以让 v7 通过；
- 应当在同一 v7 上增加训练步数或重跑随机种子；
- 已经可以设计 Full CURE；
- 已经可以接入 DNANet、UIUNet、MSHNet 或 SCTransNet。

## 6. 冻结决定

当前执行决定为：

1. 冻结 CR-LVEC v7 的代码、配置、唯一真实 \(D_R\) 运行及完整负结果；
2. 不运行 formal800 seed 42/43；
3. 不读取 \(D_V\) 或 \(D_T\)；
4. 不放宽 12 项阈值；
5. 不恢复、不自动重跑、不创建 v7-r2；
6. 下一版本必须直接解决“\(D\) 响应与 footprint 泄漏耦合”，不能仅继续放大 crossing evidence；
7. 新版本仍需保持单一机制、相同 decoder 拓扑、相同 loss、相同训练预算与相同推理图，避免模块堆叠。

## 7. 证据绑定

- 运行目录：`runs/irstd1k_stage_a_seed42/cure_lite_cr_lvec_v7_bounded_r1`
- COMPLETE SHA256：`c51d5733085bb500245589eb62cc9c1b7d4b58e2828b425d96a533e781adb32a`
- COMPLETE fingerprint：`9e7c87c831b569a019fed4bb76d667f53e47fdd1b954ec86b04dedf43e3c404b`
- result SHA256：`085b7f03825d4a7f86ce817e08ca730a8f2fb2cd5fdd3dbc0507b9547866e74c`
- result fingerprint：`030028253f9f768cadb010c6c648c449296d3e42bc286046e07951f9849d5b03`
- result receipt fingerprint：`19c59aececf99222ca3f82ca17b6ebd30cd5817d0d9097787e0251584c44abac`
- decision SHA256：`b70cc4b886f528b57c17e007b7b2db789491b79c7c244d14edc115126d787012`
- decision fingerprint：`d1f06366d3f675d071f419338e8d8bed2b0a0cf4a63ea4efa9fa1c550af5548f`

