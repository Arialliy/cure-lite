# CURE-Lite 机制假设审查

## 1. 审查结论

本轮结论为：

```text
review outcome = H0
```

这里的 H0 只表示：

> 现有证据没有给出一个同时满足可作用性、单机制、状态保持、不可自证和可证伪性五项门槛的 synthetic-state transformation 假设。

它不否定 CURE 总方向，也不表示事实漏检不可学习。它表示当前不能把“只删除 occupancy、完整保留 frozen feature”这一构造继续修补为某个 feature attenuation、context replacement 或 occupancy 变换。

当前状态必须保持为：

| 项目 | 状态 |
| --- | --- |
| v0.2 | 冻结负结果 |
| P0-B | fail |
| P0-C | fail |
| P0-D | not evaluated；由冻结停止规则阻断 |
| failure attribution | partial inconclusive |
| synthetic transformation | 未构造、未授权 |
| candidate S | 未构造、未授权 |
| 训练、校准、推理 | 未运行、未授权 |
| 新 \(D_V\)、\(D_T\) | 未读取 |
| Full CURE、其他 backbone | 未开始 |
| CURE-Lite 核心机制 | 尚未建立 |

机器可核验的非实验 proposal receipt 为：

- [proposal receipt](protocols/IRSTD-1K/synthetic_state_hypothesis_review_v1/proposal_receipt.json)
- fingerprint：`4addd47f9e9bae221b0e100be105ad03fbaf865bfa86e6dd2b209d9c014a2c35`

该文件位于 `protocols/` 而不是 `runs/`，也没有 `COMPLETE.json`。它记录的是研究判断，不是一次新实验。

## 2. 代码能够证明的结构事实

### 2.1 decoder 的真实输入

[decoder.py](cure_lite/decoder.py) 第 96～133 行表明，decoder 只计算：

\[
r_\theta=D_\theta(F_b,O),
\]

其中：

- \(F_b\) 是 detached frozen feature；
- \(O\) 是投影到 feature grid 的 binary occupancy；
- 两者拼接后生成低分辨率 residual logits，再双线性放大到输出分辨率。

以下量不直接进入 decoder：

- Base probability \(p_b\)；
- GT、target、valid mask；
- factual/legal identity；
- descriptor、距离和 v0.2 的 factual-to-legal 映射。

其中 \(p_b\) 只通过固定阈值间接产生 \(O_b\)。

### 2.2 factual 与 legal synthetic state

事实漏检状态使用同一图像自然产生的：

\[
x_{\mathrm{fact}}
=
\left(F_b^{\mathrm{miss}},O_b^{\mathrm{miss}}\right).
\]

当前 legal synthetic state 使用：

\[
x_{\mathrm{syn}}
=
\left(F_b^{\mathrm{det}},
O_b^{\mathrm{det}}\setminus C_{\mathrm{pred}}\right).
\]

[training_pipeline.py](cure_lite/experiment/training_pipeline.py) 第 1184～1236 行证明 factual 和 synthetic example 都直接引用各自 source 的 frozen feature；synthetic 分支只替换为删除预测组件后的 occupancy 和相应监督。

[intervention.py](cure_lite/intervention.py) 第 147～202 行证明删除对象是一个完整预测连通域，并要求其他匹配关系不变。

正式 state receipt 进一步确认 206/206 个 legal target 的 source feature 与 synthetic feature 对象和值完全一致。

因此，当前代码确实构造了：

```text
observational frozen feature + edited occupancy
```

但这个结构事实不等价于：

```text
该联合状态已经被证明是训练失败的原因
```

### 2.3 v0.2 仍然不是 pairwise learning

[step.py](cure_lite/train/step.py) 第 202～227 行对 factual、no-miss 和 synthetic 三个 branch 分别执行 decoder 与 criterion；第 116～146 行只对三个标量分支均值加权求和。

因此训练目标是：

\[
\mathcal L
=
\overline{\mathcal L}_{\mathrm{factual}}
+\lambda_0\overline{\mathcal L}_{\mathrm{no\text{-}miss}}
+\lambda_s\overline{\mathcal L}_{\mathrm{synthetic}},
\]

而不是：

\[
\mathcal L(m_j,l_i).
\]

v0.2 的 factual-to-legal 映射只改变 synthetic examples 的边缘出现频率，不把 factual miss 作为 synthetic decoder 的条件，也不产生配对损失。

## 3. 现有证据不能推出什么

### 3.1 `P` 不能成为机制

`P` 在全人口具有 strong role signal，但它：

1. 参与 factual/legal 角色形成；
2. 不直接进入 decoder；
3. 更适合作为 Base role-selection proxy。

因此不能依据 `P` 修改 decoder，也不能把 `P` 的可分性写成 CURE 的执行机制。

### 3.2 不能直接削弱目标局部 feature

`F_local` 只说明低维局部 feature 摘要存在部分分布差异：

- coverage 为 18/32；
- 全人口 AUC interval 跨 0.70；
- 两个小人口分层的固定 IRLS 未给出可靠结果；
- 它与 `F_background_global` 共享 ring 信息。

这不足以确定：

- 应修改哪些 channel；
- 应修改 target、ring 还是两者；
- 应使用缩放、替换、残差化还是其他运算；
- 变换幅度如何由数据外规则唯一决定。

所以 `feature attenuation`、局部替换或任意 feature 修补均未获授权。

### 3.3 `F_background_global` 不能解释成背景因果作用

`F_background_global` 在全人口和 shared manifest groups 中为 strong，但它混合：

- ring channel mean/std；
- whole-grid channel mean/std；
- source image 与 group composition；
- 与 `F_local` 重复的 ring 信息。

selected dual-role sources 中，该块经过 transductive source centering 后不再 strong。这个变化不能证明 source effect 已被消除，也不能把剩余或消失的信号解释成因果贡献。

因此不能实施 context replacement；它既没有唯一目标，也会与“非目标背景保持不变”的契约冲突。

### 3.4 `O` 的信号不能直接授权 occupancy 修改

`O` 只在 selected-source transductive sensitivity 中为 strong；全人口和 shared groups 均为 inconclusive。当前 `O` 同时包含：

- 目标区域 occupancy fraction；
- ring occupancy fraction；
- 最近组件距离；
- 局部 projected occupancy patch；
- 全图 occupancy fraction。

现有结果没有分离目标邻域、其他预测组件和全图组件数量。legal state 又必然比原状态少一个预测组件，因此 `O` 的信号可以由该确定性边缘差异产生，不必假设存在特殊的 feature–occupancy interaction。

### 3.5 联合状态探针尚未识别 interaction

`F_local + F_background_global + O` 的固定低维 union probe：

- AUC 为 0.807789；
- interval 为 [0.688933, 0.904504]；
- 冻结判定为 inconclusive。

该探针只是把三个相关摘要拼接后做角色预测。它没有显式估计：

\[
\text{feature}\times\text{occupancy interaction},
\]

也没有区分：

\[
\text{feature marginal shift},
\quad
\text{occupancy marginal shift},
\quad
\text{source/group shift}.
\]

因此，“feature–occupancy joint mismatch”只能保留为结构性假设，不能升级为已确认机制。

## 4. 五项硬门槛复核

| 门槛 | 状态 | 判断 |
| --- | --- | --- |
| 可作用性 | partial | umbrella hypothesis 涉及 decoder 输入，但最强 `P` 信号不可作用 |
| 单机制 | fail | 没有证据唯一选择 feature 或 occupancy 的一个运算 |
| 状态保持 | unresolved | 局部修改可能保持多数状态，但当前 strong 信号偏向 context/global，无法同时声称精确背景保持 |
| 不可自证 | unresolved | 尚无独立确认 groups 或冻结的 cross-fit transformation rule |
| 可证伪性 | partial | 未来固定状态可由 P0-A/B/C 证伪，但当前还不存在唯一 transformed state |

由于五项没有同时通过，按预先写入 [下一步方案](CURE_Lite_下一步方案.md) 的决策树，本轮必须选择 H0。

## 5. 唯一可以保留的研究假设

可以保留、但不能实施的更窄假设是：

> 在 occupancy 缺席的条件下，detected target 与 factual miss 的局部 frozen-feature residual geometry 可能不同。

这里的重点不是“检测目标 feature 更强，所以把它缩小”，而是：

\[
\mathcal G
\left(
F_T-F_R,\,
O_T=0
\right)
\]

是否存在一个可预先定义、跨 group 稳定、超出 feature-only 与 occupancy-only 边缘差异的兼容量。

当前尚未定义这样的 \(\mathcal G\)，也没有真实 counterfactual feature 可作为监督。因此它只是重新定义 CURE-Lite 核心学习对象时的候选问题，不是新模块、正式方法或 v0.3。

## 6. 对 ICLR 方案创新性的影响

当前实现还不足以形成 ICLR 方法核心，原因不是网络不够复杂，而是核心学习对象没有被识别：

```text
occupancy deletion
    + unchanged feature
    + residual decoder
```

目前只能被解释为一个 synthetic-state construction 加一个 residual learner。继续加入 attention、额外 decoder、matching、sampler 或 calibration，不会自动形成统一的新原理。

若未来能建立一个明确的：

```text
counterfactual state compatibility principle
```

并同时做到：

1. 数学量有明确含义；
2. 只由 \((p_b,F_b)\)、occupancy 与训练标签构造；
3. 不依赖具体 backbone 内部结构；
4. synthetic state 的生成规则唯一且可复现；
5. 先改善 factual/legal state support，再稳定提升漏检找回；

那么它才可能成为 CURE 的单一核心机制，而不是模块组合。

当前没有进行 prior-art 检索，因此不能声明该潜在方向具有文献新颖性。

## 7. 下一步

下一步不再是 transformation proposal，而是：

```text
redefine CURE-Lite core learning object
    -> predefine an identifiable compatibility quantity
    -> independent D_R training-group falsification
    -> only if supported, draft one transformation protocol
```

若尝试保留联合状态方向，至少先完成：

1. 预声明 ring、non-ring global mean 和 non-ring global scale 的正交分解；
2. 预声明 target-neighborhood、other-components 和 global occupancy 的分解；
3. 使用 group-held-out 或严格 cross-fit，不能把 selected-overlap transductive centering 当作确认；
4. 明确定义 interaction-specific estimand，并证明它提供超出 feature-only、occupancy-only 与 source/group 变量的信息；
5. 在查看结果前冻结通过/停止门槛；
6. 仍然不读取 \(D_V\)，不构造 transformation，不训练。

若这些条件不能产生一个唯一、可作用且可保持状态的核心量，则应正式停止 occupancy-deletion synthetic paradigm，并重新定义 CURE-Lite 的训练对象，而不是继续修补现有 synthetic state。

## 8. 后续核心对象重定义结果

本文件冻结的是 transformation hypothesis review 的 H0，结论保持不变。后续重定义阶段没有修改 H0，而是改变了 legal deletion 的学习语义：

```text
旧：
deleted endpoint 被当作 factual-like independent sample

新候选：
同一 source 的 occupancy before/after
被用来定义 completion operator 的离散 coverage response
```

新候选为：

\[
\Delta_gQ_\theta
=
Q_\theta(F,\Pi(O\setminus C_g))
-
Q_\theta(F,\Pi O).
\]

它不再要求 factual/legal exchangeability，也不再使用 compatibility score 生成采样分布或 feature transformation。pair 关系只有在未来的同一个不可分解 objective 中直接消费两端输出差时才成立。

该候选当前仅为 `conceptually admissible`，没有被实现或获得性能支持。完整边界见：

- [CURE-Lite 核心学习对象重定义](CURE_Lite_核心学习对象重定义.md)
- [core-object proposal receipt](protocols/IRSTD-1K/core_learning_object_redefinition_v1/proposal_receipt.json)

当前授权仍然只到“起草独立的只读 paired-objective protocol”；不授权修改 loss、训练、读取新的 \(D_V/D_T\)、设计 Full CURE 或接入其他 backbone。
