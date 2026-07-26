# CURE-Lite CR-LVEC v7 Toy 门禁结果

## 1. 正式状态

- 方法：CR-LVEC v7（Continuously-Recoverable Log-Vacancy Evidence Crossing）
- 阶段：真实 \(D_R\) bounded 运行前的模型代码门禁
- 正式决定：`CR_LVEC_V7_TOY_GATE_PASS`
- 六个冻结案例：`6/6 passed`
- 失败检查：无
- 独立进程重放：3 份结果逐字节一致
- 正式结果 SHA256：
  `01a582dbd645e47be64601ae3b42e62b889f5c838e926ed5b09c76af3f5c24b6`
- 正式结果 fingerprint：
  `ff5e76894066eb953773c444060ce6783dd12cdc2d99aa9d46700086db75ffaa`

本结论只证明冻结 v7 算子和既有 paired 训练路径在 toy
构造上满足代码与可学习性门禁，不是检测性能结论。

## 2. 冻结算子

局部 occupancy burden 为：

\[
b(O)=\operatorname{Nearest}\left(
\log\left(1+
\operatorname{Conv}_{3\times3}
(\operatorname{ProjectMax}(O))
\right)\right).
\]

令：

\[
u=r(F)-b(O).
\]

可观测前向 evidence 为：

\[
e(u)=
\begin{cases}
0,&u\leq0,\\
\operatorname{expm1}(u),&u>0.
\end{cases}
\]

反向采用显式的 \(\exp(u)\) recovery carrier：

\[
e_{\mathrm{ST}}(u)
=
\operatorname{stopgrad}(e(u))
+
\left[
\exp(u)-\operatorname{stopgrad}(\exp(u))
\right].
\]

该表达式不改变前向值，不增加参数、模块、损失项或推理分支。
标准 64-channel、stride-4 配置保持 4,385 个可训练参数。

## 3. 冻结案例结果

| 案例 | 总损失 | clean \(D\) mean | component \(H\) max | component \(G\) max | 全程最大 \(|u|\) |
|---|---:|---:|---:|---:|---:|
| legacy one-pixel | 0.001381 | 0.895021 | 0.003325 | 0 | 10.305702 |
| legacy two-pixel | 0.001378 | 0.894434 | 0.004241 | \(2.38\times10^{-7}\) | 10.546761 |
| legacy three-pixel | 0.001423 | 0.891277 | 0.004152 | \(1.19\times10^{-7}\) | 10.906691 |
| support one-pixel | 0.001430 | 0.891612 | 0 | 0 | 9.142893 |
| support two-pixel | 0.001191 | 0.911594 | \(9.30\times10^{-7}\) | 0 | 10.079000 |
| support three-pixel | 0.001083 | 0.915357 | 0 | \(1.19\times10^{-7}\) | 10.333025 |

所有案例同时满足：

- plus endpoint 目标完成概率大于 0.95；
- plus endpoint 背景概率小于 0.05；
- factual miss 目标概率大于 0.95；
- factual 背景概率小于 0.05；
- clean \(D\) mean 不低于 0.80；
- 双 endpoint 梯度有限且非零；
- 六个参数张量在首个和最后一个 update 的梯度均有限且非零；
- identity 区域逐位不变；
- ratio identity、前向 crossing 和几何条件通过。

每个案例在已有 `forward_fields` 调用中观测 970 次 crossing
margin；没有为统计最大值额外重复网络计算。

## 4. 局部性与数值边界

独立的 \(4\times5\) feature-grid 局部性检验得到：

- count-change support：144 个输出像素；
- unchanged support：176 个输出像素；
- unchanged logits 与 probability 均逐位一致；
- 删除 occupancy 后 logits 单调不减；
- 所有字段有限。

float32 数值契约：

- \(u=-80\)：前向为 0，恢复梯度
  \(1.8048513\times10^{-35}>0\)；
- \(u=-104\)：恢复梯度下溢，按协议立即失败；
- \(u=88\)：前向和梯度均有限；
- \(u=89\)：发生非有限值，按协议立即失败；
- 不使用静默 clamp。

## 5. 确定性与测试证据

两个临时独立进程与一个 create-only 正式进程生成的结果：

- 文件长度均为 36,334 字节；
- SHA256 均为
  `01a582dbd645e47be64601ae3b42e62b889f5c838e926ed5b09c76af3f5c24b6`；
- 三份文件逐字节一致。

聚焦测试：

```text
40 passed in 34.30s
```

测试覆盖 config、decoder、通用冻结 Base 模型图、hard-union、
双端梯度、参数梯度、数值边界、非空局部性和六案例 toy overfit。

## 6. 开发期实现缺陷记录

第一次开发期检查曾有四个案例仅失败
`forward_crossing_exact`。原因是直通表达式被写成左结合：

```python
forward.detach() + recovery - recovery.detach()
```

float32 的先加后减产生了最多约 \(1.53\times10^{-5}\) 的前向回差。
修复仅恢复必要括号：

```python
forward.detach() + (recovery - recovery.detach())
```

公式、随机种子、320 updates、学习率、损失和全部阈值均未改变。
修复前没有发布正式 result 或 closure。已增加 float32 多尺度
bit-exact 单元测试，防止该实现缺陷回归。

## 7. 边界与下一步

本阶段没有访问 \(D_R\)、\(D_V\) 或 \(D_T\)，没有执行校准、
Pd/FA 评估、formal800、Full CURE 或其他 detector 接入。

Toy closure 完成后只授权：

> 创建并验证 v7 的真实 \(D_R\) bounded executor。

它不直接授权真实数据运行。下一阶段必须先解决或量化生产算子中
有限性检查造成的 GPU 同步开销，并完成 bounded executor 的静态、
单元和 dry-run 验证；随后才能单次运行冻结的真实 \(D_R\) bounded
实验。
