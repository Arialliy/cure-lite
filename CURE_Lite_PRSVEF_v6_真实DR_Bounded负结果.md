# CURE-Lite PR-SVEF v6：真实 \(D_R\) bounded 负结果

## 1. 正式结论

PR-SVEF v6 已完成唯一一次、GPU0 温控的真实 \(D_R\) 400-update bounded 运行。

最终判定为：

```text
structural_execution_pass = true
bounded_model_code_gate_pass = false
decision = PR_SVEF_BOUNDED_MODEL_CODE_GATE_FAIL
```

因此，PR-SVEF v6 不能进入 formal800 seed 42/43，不能读取新的 \(D_V/D_T\) 结果，也不能进入 Full CURE 或跨 detector 验证。

这不是 Pd/FA 性能实验。它只说明当前模型代码在冻结的真实 \(D_R\) outcome population 上未达到预声明的 bounded 学习门槛。

## 2. 运行边界

| 项目 | 结果 |
|---|---:|
| 数据 | IRSTD-1K \(D_R\) |
| GPU | GPU0 |
| 温控 | 82°C 暂停，75°C 继续 |
| 实际最高已记录温度 | 42°C |
| optimizer updates | 400/400 |
| backward calls | 400/400 |
| optimizer steps | 400/400 |
| 自动重跑 | 0 |
| Base/backbone 更新 | 否 |
| \(D_V/D_T\) 访问 | 否 |
| calibration/inference | 否 |
| formal800 | 未执行 |

运行目录：

```text
runs/irstd1k_stage_a_seed42/cure_lite_pr_svef_v6_bounded_r1
```

## 3. 已通过的结构检查

全部结构检查通过，包括：

- 222 个 outcome pairs 完整绑定并在初始、最终状态全部评估；
- 400 次更新、反向和 optimizer step 数量准确；
- 梯度全部有限，且每次更新总梯度范数均为正；
- decoder 参数确实发生变化；
- pair、source image、factual anchor exposure ledger 全部准确；
- identity-null 未进入 optimizer；
- 训练与评估 forward budget 完全符合冻结值；
- PR-SVEF v6 六项前向/反向算子检查全部通过；
- 4,385 个可训练参数与冻结结构一致。

梯度范围：

```text
minimum update L2 norm = 0.1110012837
maximum update L2 norm = 2.5037824243
nonfinite updates = 0
zero-norm updates = 0
```

所以此次失败不是代码没有训练、梯度消失、预算不足执行或 exposure 错误。

## 4. 计算门禁结果

### 4.1 通过的 6 项

| 门禁 | 观测值 | 阈值 |
|---|---:|---:|
| factual miss final/initial | 0.442935 | \(\le 0.75\) |
| factual no-miss final/initial | 0.041164 | \(\le 0.75\) |
| plus-baseline final/initial | 0.405922 | \(\le 0.75\) |
| clean-zero macro mean \(|\Delta|\) | 0.037995 | \(\le 0.05\) |
| component-null context macro mean \(|\Delta|\) | 0.0000296 | \(\le 0.05\) |
| identity-null max \(|\Delta|\) | 0.0 | \(\le 10^{-7}\) |

### 4.2 失败的 6 项

| 门禁 | 观测值 | 阈值 |
|---|---:|---:|
| clean transition final/initial | 0.866215 | \(\le 0.50\) |
| clean mean delta on \(D\) | 0.133460 | \(\ge 0.50\) |
| clean pairs with delta \(\ge 0.25\) | 0.097087 | \(\ge 0.75\) |
| component-null footprint macro mean \(|\Delta|\) | 0.075819 | \(\le 0.05\) |
| component-null footprint global max \(|\Delta|\) | 0.790334 | \(\le 0.25\) |
| joint \(D\ge0.25,H\le0.05\) fraction | 0.029126 | \(\ge 0.75\) |

tiny-target 分层也没有形成稳定正结果：

| \(D\) 像素数 | pair 数 | joint pass fraction |
|---|---:|---:|
| 1–3 | 33 | 0.060606 |
| 4–7 | 59 | 0.016949 |
| 8–15 | 67 | 0.044776 |
| 16+ | 47 | 0.0 |

## 5. 机制解释

v6 的 polarity-recoverable backward 解决了“负半轴无法恢复梯度”的代码可学习性问题。真实运行也证明 factual anchor loss 可以显著下降。

但它没有解决两个更关键的问题：

1. clean outcome 上的目标差分仍然过弱。平均 \(D\) 增量只有 0.1335，只有约 9.71% 的 clean pairs 达到 0.25 增量。
2. component-null footprint 上出现过强残差。宏平均为 0.0758，单点最大值达到 0.7903。

因此，当前模型表现为：

```text
能够优化 factual states
但不能把优化稳定转换为 clean outcome 的强目标差分
且在部分 component-null footprint 上发生明显泄漏
```

toy 1/2/3 像素 3/3 通过只证明算子在受控样本上可优化，不能替代真实 \(D_R\) population 的 outcome separation。

## 6. 发布自检问题

训练及门禁计算已经完成并写入 `result.json`、`decision.json` 和 `COMPLETE.json`，但 CLI 最后的 strict-loader round-trip 因实现绑定包装层级不一致而返回 exit code 1：

- 运行前 authorization 校验使用未加 receipt fingerprint 的 implementation binding；
- 发布后 loader 将已加 receipt fingerprint 的 implementation receipt 再传入同一 authorization 校验；
- 因多出该字段，implementation fingerprint 比较失败。

只读诊断中移除发布包装增加的 `receipt_fingerprint` 后，完整 strict loader 验证通过，且模型判定仍为 `BOUNDED_MODEL_CODE_GATE_FAIL`。

该发布问题不改变模型负结果。为保存唯一一次运行的原始状态：

- 未重新训练；
- 未删除 `.incomplete`；
- 未改写任何运行 receipt；
- 未把该目录伪装成正常完成发布。

## 7. 冻结决定

当前正式决定：

```text
PR-SVEF v6 = frozen bounded negative result
same-version retry = forbidden
threshold relaxation = forbidden
formal800 = not authorized
Full CURE = not authorized
cross-backbone validation = not authorized
next = new-version model-code redesign
```

下一版本必须直接解决“clean \(D\) 差分不足 + component-null footprint 泄漏”的联合结构问题，而不能仅继续更换 surrogate gradient、增加训练步数或降低门槛。

