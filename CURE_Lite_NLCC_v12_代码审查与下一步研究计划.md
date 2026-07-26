# CURE-Lite NLCC-v12 代码审查、模型判断与下一步研究计划

> 原始审查日期：2026-07-26；执行状态更新：2026-07-27  
> 仓库：`Arialliy/cure-lite`  
> 审查快照：`main@538660aa7e200bd7acad8964af25121ea56142cf`  
> 当前决策：**FREEZE_NLCC_V12_DEVELOPMENT_FAIL**  
> 当前执行状态：**R0 PASS；DEVELOPMENT COMPLETE/GATE_FAIL；HOLDOUT NOT_AUTHORIZED**  
> 适用范围：NLCC-v12 核心方程、decoder 实现、dataset-free 输入、正式运行器、验证顺序与创新边界。

---

## 2026-07-27 正式执行更新

本文第 0～15 节主体是 Development 启动前的代码审查快照。其关于
“R0 尚未闭合”“Development 未授权/未运行”的表述现作为历史记录保留，并由
本节及正式产物取代，不再代表当前状态。

当前权威状态：

| 阶段 | 状态 | 证据 |
|---|---|---|
| R0-C1～R0-C8 | PASS | 43/43 targeted、1105/1105 repository tests；R0 receipt PASS |
| Development authorization | CREATED / VALID | 绑定 runner/evidence r2、源码闭包与固定输入 |
| Development execution | COMPLETE | 320/320 updates、960/960 forwards、321/321 finite-state audits |
| structural gates | 25/25 PASS | 无 step/gradient/finite-state failure |
| numeric gates | 26/76 PASS | 50 项失败 |
| final groups | 0/8 PASS | 全局四门禁全部失败 |
| sealed decision | `NLCC_V12_DEVELOPMENT_FAIL` | 独立 raw-field 重算与 gate ledger 一致 |
| Exposure Holdout | NOT_AUTHORIZED / NOT_RUN | Development stop rule |
| 真实 \(D_R\) 及后续 | NOT_AUTHORIZED / NOT_RUN | 未跨越 Development 门禁 |

正式解释：

- runner/evidence 闭合与执行结构已经成立；
- NLCC-v12 的固定学习系统没有通过 Development 数值门禁；
- 这是模型候选的有效负结果，不是执行异常；
- 不重跑 v12，不调整同一版本的阈值、loss、训练步数或状态方程；
- 不创建 Holdout authorization，不进入真实 \(D_R\)、exposure replay、
  formal800、Full CURE 或跨 detector 验证。

完整结果见
[CURE-Lite NLCC-v12 Development 正式负结果](CURE_Lite_NLCC_v12_Development正式负结果.md)。

---

## 0. 证据边界

本报告基于以下材料：

1. GitHub 仓库在提交 `538660aa7e200bd7acad8964af25121ea56142cf` 下的源码、设计文档、协议文件和测试源码。
2. 用户提供的冻结状态：
   - NLCC-v12 核心网络与 65 项核心测试通过；
   - dataset-free 输入 9/9 通过；
   - 运行器首轮结构测试 21/21 通过；
   - Development 为 32 rows / 320 updates；
   - Holdout 为 222 rows / 400 updates。
3. 仓库内的 NLCC-v12 设计说明与 2026-07-24 文献边界审计。

本报告**没有直接读取用户机器上的** `/tmp/nlcc_v12_runner_targeted_initial.junit.xml`，因此 65、9 和 21 项通过数作为用户提供的冻结证据接受，而不是本次审查重新执行所得。代码审查发现，当前 21 项运行器测试中有一项实际上固化了直接接受 `all_pass` 的行为；因此“21/21 通过”说明当前实现符合当前测试，而不等价于正式运行器已经闭合。

截至本报告快照，NLCC-v12 的 Development、Exposure Holdout、真实 \(D_R\) 与 formal800 路径均未获得执行授权，也未发生相应的正式训练。本报告讨论的是代码与研究协议，不是新的实验结果。

---

# 1. 结论先行

## 1.1 总体判断

| 问题 | 结论 | 解释 |
|---|---|---|
| 整体模型设计如何 | **自洽候选，但 Development 已否决当前实例** | 设计保持容量基本不变、机制隔离明确且可证伪；正式数值门禁表明当前固定实例没有学成。 |
| NLCC-v12 方程是否自洽 | **基本自洽** | 零锚定参考、单位局部计数边界、active-set crossing、occupancy-invariant baseline 和 hard union 的目标相互一致。 |
| 当前研究顺序是否正确 | **主线正确且停止规则已触发** | R0 后执行 Development；因 Development FAIL，不进入 Holdout、真实 \(D_R\) 或 formal800。 |
| 当前正式运行器是否正确 | **runner/evidence r2 已闭合** | R0-C1～R0-C8 全部通过；sealed decision 由重新读取的原始字段独立重算。 |
| 当前能否判断模型成功 | **可以判断当前 v12 未成功** | 结构执行 25/25 通过，但数值门禁仅 26/76、groups 0/8；不形成真实检测性能结论。 |
| 当前是否已经满足创新 | **尚未建立** | 符合“窄范围创新候选”的结构条件，但通用 refiner、冻结 Base、轻量 decoder、差分学习、合成删除、低 FA 均已有强先例。 |
| 现在是否应该改模型 | **不能修改或重跑 v12** | 先冻结本次结果并做失败归因；任何方程、loss、输入、阈值或日程修改都必须成为新模型版本。 |

## 1.2 最重要的研究决策

**冻结 `NLCC_V12_DEVELOPMENT_FAIL`；不重跑 v12，不创建 Holdout
authorization，不进入后续阶段。**

以下八项 R0-C 条件现已全部完成，并构成本次有效负结果的执行前提：

1. **R0-C1**：从原始指标和原始结构字段重新计算全部门禁。
2. **R0-C2**：同一执行凭据只能启动一次训练，并以持久化启动收据跨进程约束。
3. **R0-C3**：每次 `optimizer.step()` 后检查参数、buffer 和 optimizer state 有限。
4. **R0-C4**：由正式入口构建本地源码传递闭包，并只对执行范围内的源码、配置、协议和输入漂移设置硬门禁。
5. **R0-C5**：先独立复核并冻结 implementation closure，再由 authorization 绑定其 fingerprint。
6. **R0-C6**：结果发布只允许一个 terminal，并把执行、校验、发布和封存分开。
7. **R0-C7**：严格解析 JSON；退出码和派生结论只能来自重算后的 sealed decision。
8. **R0-C8**：runner/evidence 与 protocol 采用显式修订号并由独立 verifier 复核；仅修复运行路径时不得冒充新的模型版本。

---

# 2. 模型设计复原

## 2.1 输入与固定拓扑

decoder 的公开输入仍为：

\[
\left(\operatorname{sg}(F_b), O\right),
\]

其中：

- \(F_b\) 是冻结 Base detector 的 feature；
- `sg` 表示 detach；
- \(O\) 是由冻结 Base 输出和冻结阈值得到的 hard occupancy；
- GT、pair kind、target ID、source ID 和另一端标签不进入 decoder。

活动网络沿用 active-v4 factorized topology：

```text
detached feature
  -> 1x1 stem
  -> GroupNorm(affine=False)
  -> SiLU
  -> 3x3 depthwise convolution
  -> GroupNorm(affine=False)
  -> SiLU
  -> 1x1 pointwise residual
  -> existing baseline/evidence phase heads
  -> PixelShuffle
  -> optional frozen bilinear resize
  -> one residual logit map
```

在 `C=8, stride=4` 的 dataset-free 配置下：

- 参数量：2,593；
- 参数张量：6；
- 无新 head；
- 无第二 decoder；
- 无训练分支或推理分支增加。

这是一个重要优点：v12 与旧候选之间的差异主要来自**状态方程**，而不是参数容量。

## 2.2 NLCC 状态方程

设每个 feature cell 有 \(P=s^2\) 个 phase response：

\[
r=(r_1,\ldots,r_P).
\]

固定 null coordinate：

\[
r_\varnothing=0.
\]

零锚定参考：

\[
\mu_\varnothing(r)=\frac{\sum_{k=1}^{P}r_k}{P+1}.
\]

phase-relative response：

\[
q_j=r_j-\mu_\varnothing(r).
\]

occupancy 先经过 max projection，再以固定 \(3\times3\) 全一核得到局部计数：

\[
C(O)=\mathbf 1_{3\times3} * \widehat O,\qquad C\in\{0,\ldots,9\}.
\]

crossing margin：

\[
m_j(F,O)=q_j(F)-C(O).
\]

可观察前向 evidence：

\[
h(m)=
\begin{cases}
0,&m\le 0,\\
\exp(m)-1,&m>0.
\end{cases}
\]

实现使用直通恢复载体：

\[
\widetilde h(m)
=
\operatorname{sg}(h(m))
+
\exp(m)-\operatorname{sg}(\exp(m)),
\]

因此前向仍为 \(h(m)\)，反向为：

\[
\frac{\partial \widetilde h}{\partial m}=\exp(m).
\]

最终：

\[
B(F)=-\operatorname{softplus}\left(\beta+\operatorname{PixelShuffle}(b(F))\right),
\]

\[
E(F,O)=\operatorname{PixelShuffle}(\widetilde h(m(F,O))),
\]

\[
z_\theta(F,O)=B(F)+E(F,O).
\]

推理保持一次 Base、一次 decoder、固定阈值和 hard union。

---

# 3. 模型设计的优点

## 3.1 occupancy 不再污染空间归一化统计

v11 的一个明确问题是 occupancy-conditioned tensor 在 GroupNorm 前进入，局部 occupancy 改动能够改变整张 feature map 的归一化统计。v12 把 occupancy 放到全部 convolution 和 GroupNorm 之后，因此在固定 \(F\) 时：

- plus/minus endpoint 共享完全相同的 feature trunk；
- occupancy 只改变局部 count boundary；
- paired response 更接近设计所声称的局部状态干预。

这对机制识别非常重要。

## 3.2 固定 null 允许“全部 phase 不激活”

softmax 或固定总预算通常迫使 evidence 在 phase 间分配。固定零 null coordinate 不要求 phase 总量守恒，因此：

- 任意 phase 可 inactive；
- 全部 phase 可同时 inactive；
- zero-feature 时不能仅凭 occupancy 产生正 evidence。

这与“只恢复有 feature witness 的漏检，不允许 occupancy 自行造目标”的目标一致。

## 3.3 单位 count release 消除了旧式 count 衰减

由

\[
m_j=q_j-C
\]

可得：

\[
m_j(C-1)-m_j(C)=1.
\]

因此 \(1\to0\)、\(2\to1\)、\(3\to2\) 在 **margin 坐标**上拥有相同的一步释放量。相比 \(1/(1+C)\)、\(\log(1+C)\) 或 \(C/9\)，这一设计直接针对 v11 中多计数转换响应衰减的问题。

必须准确表述：它保证的是**相同的 margin 位移**，不是自动保证相同的 evidence 差或 probability 差。后者仍取决于两个端点处于哪个 crossing 区域及其 \(q_j\) 分布。

## 3.4 baseline 与 occupancy 解耦

baseline 只由 feature 产生，occupancy 只进入 evidence crossing。这样：

- background anchor 不会因为 occupancy 改动而整体漂移；
- paired difference 的解释更清楚；
- 绝对 factual anchors 与 paired response 可以分别约束 baseline 和 evidence。

## 3.5 hard union 与冻结 Base 的部署契约清楚

hard union 保证 Base 已有正像素不被 decoder 删除，适合将方法定位为：

> 冻结检测器上的 residual miss-completion operator。

它不适合定位为“纠正或删除 Base false positives 的通用后处理器”。正确的科学主张应是：

> 在保持 Base 检出和限制新增 FA 的条件下恢复 natural misses。

## 3.6 研究设计比单纯结构创新更强

Development / Holdout 输入剖面分离、matched exposure strata、D/H/G strata、固定 exposure schedule、输入 fingerprint、一次尝试和逐阶段授权，形成了相对严谨的机制验证链。这里的 matched metadata 用于构造和分层输入，不代表 `match_id` 被 decoder 或 loss 消费。该验证链的价值不只是工程可复现性，还在于限制：

- 事后改阈值；
- 对 Holdout 调参；
- 重跑挑 seed；
- 将通用容量收益误当成 NLCC 机制收益。

---

# 4. 模型的核心风险与可证伪点

## 4.1 直通梯度不是前向算子的真实梯度

当 \(m\le0\) 时，前向函数恒为 0，其经典导数应为 0；当前实现却使用：

\[
\frac{\partial \widetilde h}{\partial m}=\exp(m)>0.
\]

这是一个有意设计的 surrogate gradient，而不是数学恒等式。其意义是让 inactive phase 仍可获得恢复梯度，但也带来两个风险：

1. 模型可在“前向完全不响应”的区域积累参数更新；
2. surrogate 所优化的局部方向不一定对应真实前向目标的下降方向。

因此论文中不应把它描述为“NLCC 的精确梯度”，而应描述为：

> full-axis exponential recovery surrogate / straight-through recovery carrier。

必须预先报告：

- active/inactive phase 比例；
- inactive 区域的梯度质量；
- margin 分布；
- recovery factor 分布；
- full surrogate 与 true-gradient control 的后续消融。

## 4.2 数值风险是结构性的，不是普通实现细节

\[
\exp(m)
\]

在大正 margin 下会溢出，在大负 margin 下会快速趋近 0。即使前向和梯度在 `optimizer.step()` 前有限，Adam 更新后的：

- 参数；
- `exp_avg`；
- `exp_avg_sq`；
- buffer；

仍可能出现非有限值。

当前设计禁止失败后临时加 clamp、temperature 或更换 surrogate，因此逐 step 的 post-update finite audit 是科学协议的一部分，而不是可选调试功能。

## 4.3 “单位释放”不等于“单位响应”

当两个端点都 active 时：

\[
h(m+1)-h(m)=e^m(e-1),
\]

其大小依赖 \(m\)。当两个端点都 inactive 时，差为 0；当跨越零点时，差又是另一种形式。

因此，raw count 解决的是**count transition 在 margin 上的尺度一致性**，但网络是否能把不同 group 的 margin 放到合适区域，仍必须由 76 项门禁检验。这个边界应在论文和结果解释中明确。

## 4.4 same-count blindness

若两个 occupancy 在局部窗口内具有相同计数：

\[
C(O_a)=C(O_b),
\]

则在固定 feature 下：

\[
z(F,O_a)=z(F,O_b).
\]

NLCC 看不到：

- occupancy 的方向；
- 具体排列；
- 连通关系；
- 中心与边缘位置差异；
- 同 count 下不同目标几何。

如果真实 \(D_R\) 表明恢复依赖方向性 occupancy geometry，v12 应冻结失败。不能在看过 Holdout 或 \(D_R\) 后向 v12 临时加入位置模块。

## 4.5 max projection 丢失 sub-cell occupancy 几何

pixel-level occupancy 先被 max-project 到 feature grid。一个 cell 内只要有任一 positive pixel，该 cell 就是 occupied。因此：

- sub-cell 位置被抹去；
- 不同小目标形状可能投影为同一 occupancy；
- phase 输出位置主要依赖 feature head，而不是 occupancy 几何。

dataset-free 中专用 witness channels 6/7 证明“存在可学解”，但不能证明真实冻结 detector feature 中存在同样可分离的 witness。

## 4.6 locality 对 resize fallback 是有条件的

在原生 `feature_stride` 整倍对应下，count support 经 PixelShuffle 后可保持严格局部。若启用 bilinear resize fallback，单个 native phase evidence 可能影响相邻输出像素，严格的 exact-zero support 会被插值扩大。

因此跨 detector 阶段必须分别记录：

- 是否发生 resize；
- 原生输出尺寸与最终尺寸；
- count-support 外最大绝对差；
- locality 结论是 exact 还是 approximate。

## 4.7 hard union 无法删除 Base false positives

hard union 的结构决定 decoder 只能增加正预测，不能删除 Base 正预测。因此：

- “降低 Base 原有 FA”不是可支持主张；
- “保持或限制新增 FA”才是正确目标；
- added evidence 还可能连接原有 components，改变 target/component matching。

即使像素级 Base positives 全部保留，component-level detected-target retention 仍应实际计算，不能直接按 100% 写入。

## 4.8 hard occupancy 对 Base 阈值和校准敏感

Base probability 的微小变化可能在阈值处造成离散 occupancy 改变，继而使 \(3\times3\) count 和 crossing 发生阶跃。跨 detector 泛化不仅依赖 feature adapter，也依赖：

- Base threshold contract；
- occupancy density；
- calibration；
- feature/occupancy 空间对齐。

这些都需要在 Full CURE 前固定。

## 4.9 “干预”不自动等于因果识别

固定 feature、改变 occupancy 是对 decoder 输入状态的受控操作，能够识别 NLCC operator 的响应；但它不是自动成立的现实世界因果效应，因为真实 detector 中 \(F\) 与 \(O\) 通常共同由同一图像和 Base 产生。

限定表述是：

> controlled state intervention / operator intervention。

若要使用更强的 causal claim，需要明确：

- consistency；
- reachability/positivity；
- target lineage；
- pair 唯一差异；
- intervention 不携带标签捷径；
- synthetic state 与真实 detector state 的对应假设。

当前 reachability receipt 是正确方向，但真实 \(D_R\) 仍是不可替代的外部验证。

---

# 5. R0-C：正式运行器闭合审计

## 5.1 R0-C1：`decision.json` 直接接受 `all_pass`

### 当前行为

`_seal_directory()` 当前使用：

```python
all_pass = (
    terminal_name == "result.json"
    and terminal.get("all_pass") is True
)
```

然后直接生成 PASS/FAIL。

这只验证 `result_fingerprint` 与文件内容一致，却没有从：

- `global_metrics`；
- 8 个 group 的原始 metrics；
- 76 项阈值；
- 原始 structural training fields；

重新计算结论。

`verify_development_authorization_artifacts()` 又继续信任：

```python
result["all_pass"]
decision["all_pass"]
decision["decision"]
```

因此，一个字段不完整但 fingerprint 自洽的 PASS JSON 仍可能授权 Holdout。

### 当前测试还固化了该缺口

`tests_v12/test_nlcc_dataset_free_runner.py` 中 `_passing_result()` 只构造：

- schema；
- method/profile；
- `decision=...PASS`；
- `all_pass=True`；
- `initial_decoder_fingerprint`；

没有任何 76 项 raw metrics，也没有真实训练结构记录。随后测试将该 skeletal result 发布为 PASS，并用它授权 Holdout。

所以这不是“测试尚未覆盖”的普通缺陷，而是：

> 当前测试明确把缺少原始门禁字段的结果当作可授权结果。

### 必须修改

新增唯一权威纯函数：

```python
recompute_result_decision(
    result: Mapping[str, object],
    config: NLCCDatasetFreeRunnerConfig,
) -> RecomputedDecision
```

该函数必须：

1. 严格验证 schema、profile、config fingerprint 和字段集合；
2. 忽略所有嵌入的：
   - `all_pass`；
   - `decision`；
   - `global_checks`；
   - group `checks`；
   - group `all_pass`；
   - structural `all_pass`；
3. 仅从 raw metrics 和 raw structural fields 重算；
4. 重建 76 项 numeric gate ledger：
   - 4 项 global；
   - \(8\times6=48\) 项各 group 通用 gate；
   - \(6\times4=24\) 项 clean-D gate；
   - 合计 76；
5. 从 raw structural fields 重算结构结论；
6. 检查 group 数量、ID、pair kind、D applicability 和 exact gate count；
7. 拒绝缺失值、额外 gate、错误类型、`NaN`、`Infinity` 和非有限实数；
8. 比较嵌入派生字段与重算结果：
   - 不一致时属于 artifact invalid；
   - 不能把它当作普通 scientific FAIL；
9. 生成包含每一项 gate 输入、运算符、阈值和布尔结果的 decision ledger。

### 建议的结果分类

```text
PASS
GATE_FAIL
INVALID_RESULT
EXECUTION_FAILURE
```

- `GATE_FAIL`：raw metrics 合法，但至少一个科学或结构门禁失败；
- `INVALID_RESULT`：schema、fingerprint、字段、派生结论或数值合法性不一致；
- `EXECUTION_FAILURE`：训练或评估抛出异常。

这三类失败不能混为一个 `FAIL`，否则无法区分模型失败与证据链失败。

## 5.2 R0-C2：同一执行凭据可被重复使用

### 当前行为

当前用进程内集合：

```python
_AUTHORITY_NONCES: set[object] = set()
```

`claim_execution()` 创建 nonce 并加入集合；`_require_authority()` 只检查 nonce 是否仍在集合中，却从未消费或转换状态。

结果是：

- 同一个 `ExecutionAuthority` 对象可多次调用 `build_training_components()`；
- 同一个 authority 可多次调用 `execute_authorized_profile()`；
- 当前只防止再次创建同一 artifact directory；
- 没有防止同一个已签发凭据在同一进程内再次启动训练。

### 必须修改

建立双层 exact-once 机制。

#### 进程内状态机

```text
CLAIMED -> TRAINING_STARTED -> SEALED
```

使用 `threading.Lock` 保护状态转换。只有 `CLAIMED` 可以被消费一次。

#### 持久化训练启动收据

在构造 optimizer 之前，以 create-only 方式写入：

```text
training_started.json
```

建议字段：

```json
{
  "schema_version": "...training-started.v1",
  "attempt_fingerprint": "...",
  "authorization_fingerprint": "...",
  "closure_fingerprint": "...",
  "config_fingerprint": "...",
  "profile_id": "...",
  "process_id": 123,
  "runtime_environment_fingerprint": "...",
  "automatic_retry_allowed": false,
  "training_started_fingerprint": "..."
}
```

如果该文件已存在：

- 当前进程重放必须失败；
- 新进程重启重放也必须失败；
- 不得恢复 optimizer；
- 不得自动重试。

#### API 收口

正式路径应改为：

```python
execute_authorized_profile(authority, cache)
    -> consume authority
    -> durably write training_started.json
    -> construct decoder/optimizer exactly once
    -> train
```

`build_training_components()` 不应继续作为可用 authority 重复调用的公开 API。单元测试需要构造组件时，可以使用私有纯 helper，但正式执行入口必须独占消费 authority。

## 5.3 R0-C3：缺少 `optimizer.step()` 后的有限性检查

### 当前行为

`outcome_complete_train_step()` 在：

1. `total.backward()` 后检查 gradient finite；
2. 调用 `optimizer.step()`；
3. 直接返回 logs。

runner 随后再次检查 gradient norm，但没有检查 step 后的：

- model parameters；
- buffers；
- Adam state。

这不能满足“每次 optimizer step 后模型状态有限”。

### 必须修改

新增：

```python
audit_finite_training_state(
    decoder,
    optimizer,
    *,
    update_index: int,
    phase: Literal["before_first_update", "after_optimizer_step"],
) -> dict[str, object]
```

至少检查：

1. 所有 named parameters；
2. 所有 floating/complex buffers；
3. optimizer state 中所有 tensor：
   - `exp_avg`；
   - `exp_avg_sq`；
   - tensor-form `step`；
   - 未来 PyTorch 版本新增 tensor state；
4. 参数、buffer 和 state 的 min/max/norm 是否可序列化为有限实数。

执行时机：

- optimizer 创建后、第一次 update 前一次；
- 每个 `optimizer.step()` 后立即一次；
- 在下一次 forward 之前完成。

发现非有限值时：

```python
raise FloatingPointError(...)
```

禁止：

- 自动恢复；
- 跳过 update；
- gradient clipping；
- clamp；
- 重置 Adam state；
- retry。

这些都将改变冻结方法。

## 5.4 R0-C4：正式源码绑定不是传递闭包

### 当前行为

`REQUIRED_AUTH_SOURCE_PATHS` 是手工枚举表。它包含 runner、loss、paired types 和协议文件，但未完整覆盖实际继承和 import 链。

可直接确认的遗漏包括：

- `cure_lite/decoder.py`
- `cure_lite/factorized_config.py`
- `cure_lite/factorized_decoder.py`

因为：

- NLCC config 继承/转换自 `FactorizedDecoderConfig`；
- NLCC decoder 继承 `CURELiteFactorizedDecoder`；
- factorized decoder 又依赖 `decoder.py` 和 `factorized_config.py`。

此外，Python 导入子模块时会执行 package initializer，因此还应考虑：

- `cure_lite/__init__.py`
- `cure_lite/cache/__init__.py`
- `cure_lite/train/__init__.py`

即使设计规定“不修改它们”，也不意味着正式运行不受它们影响。

### 必须修改

由正式入口自动生成 AST-based local import closure。

#### closure roots

至少包括：

```text
tools/evaluate_nlcc_development_regression.py
tools/evaluate_nlcc_exposure_holdout.py
```

#### closure 节点

递归解析所有：

- `import cure_lite...`
- `from cure_lite... import ...`
- 相对 import；
- package `__init__.py`；
- profile-conditioned local import；
- 明确登记的动态 import。

#### closure 输出

生成：

```text
runner_implementation_closure.json
```

建议 schema：

```json
{
  "schema_version": "...runner-implementation-closure.v1",
  "repository_commit": "538660...",
  "in_scope_drift": {"count": 0, "paths": []},
  "out_of_scope_worktree_changes_recorded": true,
  "roots": ["..."],
  "nodes": [
    {
      "path": "cure_lite/factorized_decoder.py",
      "sha256": "...",
      "role": "executable_source"
    }
  ],
  "edges": [
    {
      "from": "cure_lite/null_anchored_local_count_crossing_decoder.py",
      "to": "cure_lite/factorized_decoder.py",
      "kind": "relative_import",
      "line": 18
    }
  ],
  "dynamic_imports": ["..."],
  "external_modules": ["torch", "..."],
  "unresolved_local_imports": [],
  "symlink_escape_count": 0,
  "closure_fingerprint": "..."
}
```

#### 两种闭包都应保留

1. **静态 union closure**：绑定所有可能执行的本地源码；
2. **profile runtime import trace**：证明具体 Holdout 执行没有 import Development input module 或真实数据 pipeline。

手工 `REQUIRED_AUTH_SOURCE_PATHS` 可以保留为 root expectation，但不能继续作为权威闭包。

硬门禁只检查本次执行范围内的源码、配置、协议、输入与工具是否偏离 authorization 所绑定的 hash。仓库全局 dirty 状态可以记录为诊断信息，但不能因为无关文件存在用户改动就阻断当前执行；任何范围内漂移则必须阻断授权，并列出具体路径与期望/实际 hash。

## 5.5 R0-C5：implementation closure 必须先于 authorization

正确顺序应为：

```text
源码修改完成
  -> 全部测试通过
  -> 生成 implementation closure
  -> 独立重算 closure
  -> 冻结 closure receipt
  -> 创建 Development authorization
  -> claim execution
  -> training_started
  -> train
```

authorization 必须绑定：

- closure file SHA256；
- closure fingerprint；
- repository commit；
- in-scope drift manifest；
- config fingerprint；
- input/profile fingerprints；
- runtime environment receipt；
- protocol amendment fingerprint。

不能由 authorization 再临时计算一份不同 closure。

---

# 6. R0-C：发布、重算与版本边界

## 6.1 R0-C6：发布事务可能产生双 terminal 冲突

`run_canonical_profile()` 当前在同一个 `try/except BaseException` 中：

1. 执行训练；
2. 写 `result.json`；
3. seal；
4. 任意异常进入 `publish_failure()`。

若 `result.json` 已成功写入，但 `_seal_directory()` 随后失败，catch 会尝试再写 `failure.json`。这可能导致：

- 同一目录同时存在 result 和 failure；
- exact inventory 永远无法闭合；
- failure seal 再次失败；
- 原始错误被二次错误覆盖。

### 修改方案

将事务分为：

```text
A. execution phase
B. in-memory result validation
C. result publication
D. reload + decision recomputation
E. COMPLETE sealing
```

只有 A/B 在 `result.json` 尚未写入时失败，才允许发布 `failure.json`。

若 C/D/E 在 result 已写入后失败：

- 保留 `.incomplete`；
- 不得写第二 terminal；
- 不得自动 retry；
- 由独立的无效结果处理流程记录和复核。

不要用一个宽泛的 `except BaseException` 包围训练与发布全流程。

## 6.2 R0-C7：退出码、严格解析与派生诊断必须来自重算结果

当前：

```python
return sealed, 0 if result["all_pass"] is True else 2
```

修复后，退出码必须只依赖 sealed、reloaded、recomputed decision：

```text
0 = PASS
2 = GATE_FAIL
3 = EXECUTION_FAILURE / INVALID_RESULT
```

### Python 默认 JSON 解析允许非标准 `NaN/Infinity`

当前 `_load_json_object()` 直接调用 `json.loads()`。Python 标准库默认可接受 `NaN`、`Infinity` 和 `-Infinity`，这与严格 JSON 和“所有指标有限”的协议不一致。

应使用拒绝非标准常量的 parser，并对所有 metric 做递归 finite validation。

### operator diagnostics 中存在派生布尔常量

当前 final diagnostics 直接写入：

```python
"all_finite": True
"same_three_final_forward_fields_calls": True
```

这些值不是独立重算所得。更合适的做法是只保存原始观测：

- call count；
- batch sizes；
- field min/max；
- nonfinite count；

由 decision verifier 重算结论。

## 6.3 R0-C8：runner、protocol 与 model version 必须分开

三类修订不能混用：

- **runner/evidence revision**：只改变执行、重算、发布、收据或 verifier；不改变模型图、损失、输入、阈值、训练日程和科学门禁，因此不增加 NLCC 模型版本号；
- **protocol revision**：改变指标定义、门禁、exposure、attempt 语义或执行顺序；必须显式 amendment、重新授权，并按是否已经读取相应结果决定是否需要新的未见输入；
- **model version**：只有模型方程、网络图、损失、输入状态、阈值或训练日程发生变化时才增加。

独立 verifier 必须从冻结的 raw metrics、raw structural fields、config 和 protocol 重算决策。全局工作树状态仅作诊断记录；真正的硬门禁是 authorization 范围内路径与冻结 hash 是否发生漂移。

---

# 7. 建议的代码修改清单

## 7.1 文件级计划

以下按职责列出最小改动面，不要求为了每一项职责新建独立文件；能在现有 runner/test 中清晰实现的内容应就地完成，避免把证据闭包扩展成与模型无关的大型子系统。

| 文件 | 修改 |
|---|---|
| `cure_lite/nlcc_dataset_free_runner.py` | authority 状态机、durable training-start receipt、finite audit 调用、发布事务重构、退出码改为 sealed decision。 |
| `cure_lite/nlcc_dataset_free_runner_config.py` | bump schema；加入 closure、runtime receipt 和新 inventory 的冻结字段。 |
| `cure_lite/nlcc_dataset_free_decision.py`（新增，runner-only） | 严格 schema、76 项 gate registry、raw metric/structural recomputation、decision ledger。 |
| `cure_lite/nlcc_runner_source_closure.py`（按需新增，runner-only） | local-import closure、已登记动态 import 和路径边界检查；只覆盖正式入口可达的本地执行范围。 |
| `tools/verify_nlcc_dataset_free_result.py`（新增） | 不启动训练的独立结果验证器；重算 decision。 |
| `tests_v12/test_nlcc_dataset_free_runner.py` | 删除 skeletal PASS 被接受的逻辑；覆盖 exact-once、finite state、单 terminal 和退出码。 |
| `tests_v12/test_nlcc_dataset_free_decision.py`（按需新增） | schema、NaN/Inf、derived mismatch 和边界运算符测试。 |
| `tests_v12/test_nlcc_runner_source_closure.py`（按需新增） | 传递依赖、initializer、范围内漂移、未解析 import 和路径边界测试。 |
| `protocols/.../runner_implementation_closure.json` | 机器生成闭包。 |
| `protocols/.../runner_closure_amendment.json` | 说明为何 v1 runner schema 需要闭合修订；旧冻结收据不覆盖、不回写。 |
| `protocols/.../runner_implementation_closure_receipt.json` | 审核通过后冻结，authorization 绑定。 |

新增的是 runner/evidence 模块，不是模型 head、训练分支或推理分支，不改变 2,593 参数和部署图。

## 7.2 schema 必须 bump

当前 `*.v1` 的语义允许信任 `all_pass`，而修复后会引入：

- decision ledger；
- training start receipt；
- 新 COMPLETE inventory；
- source closure binding；
- 新结果有效性分类。

因此不应静默复用旧 schema。建议至少 bump：

```text
pre-run-authorization.v2
attempt.v2
training-started.v1
result.v2
decision.v2
failure.v2
complete.v2
```

这些是 runner/evidence schema 修订，不是 NLCC 模型版本升级。旧协议文件保持只读，以 amendment 说明修复是运行器证据链闭合。若模型方程、网络图、损失、输入、推理阈值或训练日程改变，则建立新的 model version；若指标、门禁、exposure 或 attempt 语义改变，则建立新的 protocol revision。两者均不得由 runner revision 代替。

---

# 8. 必须新增的负向测试

## 8.1 Decision recomputation

必须证明以下输入不能获得 PASS：

1. top-level `all_pass=True`，但一个 raw metric 失败；
2. group `all_pass=True`，但其 raw metric 失败；
3. structural `all_pass=True`，但 `gradient_failure_count>0`；
4. 缺少任意一个 raw metric；
5. 多出未知 group 或 gate；
6. group/pair kind/D applicability 不匹配；
7. `numeric_gate_count` 不是 76；
8. 结果中出现 `NaN`、`Infinity`、`-Infinity`；
9. exclusive threshold 在等号处被错误接受；
10. inclusive threshold 在等号处被错误拒绝；
11. 修改 embedded checks 而不改 raw metrics；
12. skeletal `_passing_result()`；
13. result 与 config/profile fingerprint 不一致；
14. result 中 embedded decision 与重算 decision 不一致。

对第 14 项，建议判为 `INVALID_RESULT`，而不是偷偷覆盖后继续 PASS。

## 8.2 Execution authority

1. 同一 authority 调用正式执行两次，第二次必须在 optimizer 构造前失败；
2. 同一 authority 调用 component builder 后再执行，必须失败；
3. 两线程同时消费同一 authority，只能一个成功；
4. 模拟进程重启后存在 `training_started.json`，再次启动必须失败；
5. authority 与 profile/config 不匹配必须失败；
6. 第二次 claim 同一目录继续失败；
7. start receipt 写入失败时 optimizer 不得构造。

## 8.3 Finite state

通过 monkeypatch 或 test optimizer 注入：

1. step 后一个 parameter 变为 NaN；
2. step 后一个 parameter 变为 Inf；
3. buffer 变为 NaN；
4. Adam `exp_avg` 变为 NaN；
5. Adam `exp_avg_sq` 变为 Inf；
6. audit 调用次数必须等于 `1 + updates`；
7. 任意失败后不得执行下一 update；
8. failure artifact 明确记录 update index 和 tensor name；
9. 不得自动 retry。

## 8.4 Source closure

1. 修改 `factorized_decoder.py` 后 closure fingerprint 必须变化；
2. 修改 `factorized_config.py` 后必须变化；
3. 修改 `decoder.py` 后必须变化；
4. 修改 package `__init__.py` 后必须变化；
5. 新增一个本地 import 后闭包自动扩展；
6. unresolved local import 必须阻止授权；
7. `importlib` 动态 import 未登记必须阻止授权；
8. symlink 指向 repo 外必须阻止授权；
9. Holdout runtime trace 不得载入 Development input module；
10. runtime trace 不得载入真实数据 pipeline。

## 8.5 Publication transaction

1. result 写入前异常：可发布唯一 failure terminal；
2. result 写入后 seal 异常：不得再写 failure；
3. COMPLETE inventory 必须包含新的 `training_started.json`；
4. exit code 只能来自 sealed decision；
5. `.incomplete` 只能在完整 inventory 独立复核后删除。

---

# 9. Runner closure 的验收门槛

创建 Development authorization 前，必须同时满足：

```text
[ ] 模型、输入、阈值、训练日程未修改
[ ] 旧冻结协议未覆盖写入
[ ] 新 amendment 已生成并绑定
[ ] 全部现有核心测试通过
[ ] 全部新负向完整性测试通过
[ ] skeletal PASS 结果被拒绝
[ ] 76 项 gate 可由 raw metrics 独立重算
[ ] structural gate 可由 raw fields 独立重算
[ ] strict JSON 拒绝 NaN/Infinity
[ ] authority same-object replay 被拒绝
[ ] training_started.json 跨进程阻止重跑
[ ] 每次 step 后 parameter/buffer/optimizer state finite audit
[ ] static local source closure unresolved count = 0
[ ] closure 包含 decoder/factorized config/factorized decoder/package initializers
[ ] authorization 范围内 drift count = 0；范围外工作树变化只记录、不作硬门禁
[ ] profile runtime import trace 满足 Development/Holdout 隔离
[ ] publication 只允许一个 terminal
[ ] sealed decision 与独立 verifier 输出逐字节一致
[ ] 正式结果目录仍不存在
[ ] Development authorization 仍不存在，直到 closure receipt 冻结
```

不建议用“新增测试达到某个数量”作为验收标准。关键是上述关键失败路径均有负向测试，并且所有测试通过。

---

# 10. 正式执行顺序

## Phase R0：只闭合运行器，不训练

1. 以 `538660a` 作为本次审查基线。
2. 完成 R0-C1 至 R0-C8；不启动 Development/Holdout/真实数据/formal800 正式训练。
3. bump runner/evidence schemas；NLCC 模型仍为 v12，不创建 v13。
4. 创建 runner/protocol amendment，明确科学门禁、模型方程、输入、阈值和训练日程均未改变；旧收据只读。
5. 运行全部核心、输入、runner 和新负向测试。
6. 生成 static closure、profile runtime traces 与 in-scope drift manifest。
7. 独立复核 closure。
8. 生成 `runner_implementation_closure_receipt.json`。
9. 只有全部通过后，创建 Development pre-run authorization。

## Phase R1：Development，320 updates，一次尝试

必须：

- 全新 seed-42 decoder；
- 空 Adam state；
- durable `training_started.json` 后才构造 optimizer；
- 每个 step 后 finite audit；
- 固定 3 forward / 12 states / 1 backward / 1 step；
- 最终一次 \(2N\) pair forward；
- 两次 factual forward；
- 从 reloaded raw metrics 重算 decision；
- 用独立 verifier 再算一次。

### Development PASS

才允许创建 Holdout authorization。

### Development GATE_FAIL

- 冻结 v12 Development 失败；
- 不运行 Holdout；
- 不重试；
- 不修改同一 v12 后重跑；
- 诊断结果只能用于新版本设计。

### Development INVALID/EXECUTION_FAILURE

这是运行证据失败，不是模型科学失败；但只要 `training_started.json` 已建立，本次 attempt 就已消费。后续处理必须按改动类型区分：

- 只修复执行、重算、发布或收据：建立新的 runner/evidence revision、amendment 和 attempt ID，模型仍是 NLCC-v12；
- 改变指标、门禁、exposure 或 attempt 语义：建立新的 protocol revision，并重新判断未见 Development/Holdout 是否仍成立；
- 改变模型方程、网络图、损失、输入、阈值或训练日程：建立新的 model version，并从新的 Development/Holdout 路径开始。

任何一种情况都不得把修复后的运行记为原 attempt 的延续。

## Phase R2：Exposure Holdout，400 updates，一次尝试

Holdout 必须：

- 使用相同冻结代码和 closure；
- 全新 seed-42 decoder；
- 全新空 Adam；
- 不加载 Development checkpoint、optimizer state 或 training trace；
- Development artifacts 只用于授权；
- 从 raw metrics 独立重算；
- 不允许依据 Holdout 结果调 v12。

### Holdout PASS

才进入真实 \(D_R\)。

### Holdout GATE_FAIL

- 将 v12 Holdout 结果冻结为负结果；
- 不读取 \(D_R\)；
- 不运行 formal800；
- 若修改模型或科学协议，新候选必须使用新的 Development 与新的未见 Holdout。

### Holdout INVALID/EXECUTION_FAILURE

不形成模型科学结论，也不允许自动重跑。只修 runner 时使用新的 runner/evidence revision 和 attempt；是否还能使用同一 Holdout，必须由结果读取前已冻结的 attempt/未见输入规则决定，不能依据故障后观察临时决定。若模型发生变化，则进入新的 model version；若科学协议发生变化，则进入新的 protocol revision。任一种变化都必须重新建立未见 Holdout。

## Phase R3：真实 IRSTD-1K \(D_R\) 有界验证

R3 当前**尚未授权**。在启动或查看任何 R3 结果前，必须先冻结下列对象及一张可机器读取的量化门禁表：

- Base detector 与 checkpoint；
- feature adapter；
- Base threshold；
- component matching；
- natural-miss 定义；
- recovered-miss 定义；
- added-FA 定义；
- detected-target retention；
- IoU/nIoU/Pd/FA 实现；
- stopping rules；
- failure rules。

量化门禁表必须为每项主要结论写明：原始统计量、统计单位、比较对象、方向与运算符、数值阈值、聚合规则、缺失值处理和总体 PASS 逻辑。本报告不凭空填写这些数值；它们只能来自尚未读取 R3 结果时冻结的既有协议，或新的 R3 预注册。若任一必要阈值尚未量化，R3 保持未授权。

建议主要结果至少包含：

| 类别 | 指标 |
|---|---|
| 恢复 | natural misses recovered、target-level Pd gain |
| 误报约束 | added false-positive components、added false-positive pixels、FA/image |
| 保留 | Base detected targets retained、component merge/split 影响 |
| 分割 | IoU、nIoU |
| 机制 | D/H/G response、active fraction、margin/recovery ranges |
| 效率 | 参数、FLOPs、延迟、峰值内存 |

## Phase R4：暴露重放与 formal800

R4 当前同样**尚未授权**。在读取 R3 结果前，必须预先决定并 fingerprint 化：

- 800×40 exposure replay / 32,000-step 门禁是否沿用；
- exposure 的目标级与 source-image 级统计量及其数值阈值；
- formal seed 42/43 各自比较的冻结基线；
- 每个 seed 在 Pd、固定漏检找回、added FA、retention 及其他必要指标上的运算符与数值门槛；
- “frozen confirmation”的数据、重复次数和逐次通过规则。

本报告不新增这些数值。若既有冻结协议不能直接提供，就必须在 R3 结果读取前建立新的预注册；不得在观察 R3 后再决定“协议是否仍有效”。量化定义齐备后，顺序才是：

1. \(D_R\) PASS；
2. 已预先冻结的 800×40 exposure replay / 32,000-step 门禁；
3. formal 800-epoch seed 42；
4. formal 800-epoch seed 43；
5. 两个 seed 分别满足预先量化的冻结条件；
6. frozen confirmation。

两 seed 只能作为逐 seed 复现，不应把两个 seed 平均后掩盖单 seed 失败。

## Phase R5：Full CURE、跨 detector、三数据集

只有 Lite 机制冻结确认后，才授权 **Full CURE 设计阶段**；这不等于立即开始跨 detector 训练。首先必须定义并冻结 Full CURE 的 detector 接口、输入/输出契约、训练目标、推理组合方式、adapter 边界和成功门禁。Full CURE 代码与协议完成独立检查后，才进入：

- NUAA-SIRST；
- NUDT-SIRST；
- IRSTD-1K；
- DNANet / UIU-Net / MSHNet / SCTransNet 或预注册子集。

每个 Base 单独冻结、单独训练 Full CURE、单独报告；三数据集结果也分别报告，不把多个 detector 或数据集合并成一个平均数字。该顺序保持：CURE-Lite 冻结确认 → Full CURE 设计与冻结 → 跨 detector/三数据集验证。

---

# 11. 后续最小机制控制：分阶段预注册、暂不执行

这些控制不应插入当前 Development/Holdout 主实验，也不能用于选择 v12 参数。只有 v12 通过初步真实验证后，才按下列优先级建立独立 protocol；不一次性实现所有可想到的消融。

## 11.1 第一优先级：决定性对照

1. **Independent-endpoint control**  
   去掉同一 outcome row 内 occupancy-plus/minus 端点的 coupled term，只训练端点独立损失，检验实际进入 loss 的端点耦合是否必要。

2. **v4/v11 equal-topology control**  
   保持参数量、训练输入和 exposure 相同，隔离 NLCC 状态方程收益。

不保留原 `Shuffled-twin control`：当前 `match_id` 和 positive/null twin 对应关系没有进入 decoder 或 loss，单纯打乱元数据不会改变优化对象，因而不能提供机制证据。也不直接执行“固定 occupancy 的 feature-only”或未定义的 count permutation；它们会改变当前监督契约，必须先形成合法的新状态构造协议。

## 11.2 第二优先级：仅在第一优先级结果支持后执行

3. **No-null control**  
   删除固定 null reference 或改为普通 phase mean，检验 null 是否贡献全 inactive 能力。

4. **True-gradient control**  
   使用前向算子的真实 inactive-zero gradient，与 full-axis recovery surrogate 比较。

5. **Same-count diagnostic**  
   在不训练新模型的前提下分析不同 geometry、相同 count 的状态，确认 v12 的已知不变性；只有真实失败案例指向该限制时，才升级为训练对照。

## 11.3 dataset-free witness 检查

dataset-free 专用 channels 6/7 是 reachability witness。最小检查只保留两项：

- witness 与 label 解耦后，Development 信号是否消失；
- 真实 detector feature 中是否存在与候选状态量对应的可读信号。

这些检查只限定 dataset-free 证据能说明什么，不用于调 v12，也不替代真实 \(D_R\) 验证。

---

# 12. 失败后的下一模型设计分支

以下不是对 v12 的当前修改，而是 v12 正式冻结失败后才允许建立的新候选。

## 12.1 若失败原因是指数数值不稳定

只有在日志直接显示 nonfinite margin/recovery、step 后参数或 optimizer state 非有限，并能定位到 crossing 计算时，才允许采用这一归因。普通 GATE_FAIL 或恢复不足不能自动归因于指数形式。

可研究一个新的 **Threshold-Calibrated Bounded-Gradient Count Crossing** 候选。

思路：

- active-set 仍由 \(m=q-C\) 决定；
- 将前向 logit release 改为固定斜率、数值稳定的 crossing；
- 斜率由冻结端点门槛解析推导，而非实验搜索；
- 梯度有上界，避免 `exp(m)` 溢出。

例如未来候选可从：

\[
h_a(m)=a\,[m]_+
\]

或稳定 softplus crossing 出发，其中 \(a\) 由 `.05/.95` 端点需要的 logit separation 预先推导。该方案会改变 v12 核心方程，必须命名为新版本并使用新 Holdout。

## 12.2 若真实失败来自 same-count geometry

只有在固定 feature 与局部 count、改变几何布局的合法对照中观察到系统性差异，或真实失败案例集中违反 same-count 表达前提时，才允许采用这一归因。

可研究 **geometry-aware fixed occupancy basis**：

- 不增加 trainable head；
- 用固定方向/距离基函数替代单一标量 count；
- 为 phase 提供中心、方向或距离相关的固定 burden；
- 保留 feature-only trunk 与 occupancy-late-entry。

这会放弃 v12 的 same-count invariance，也必须成为新版本。应先用真实失败案例证明 geometry 确实是必要因素，不能因直觉提前加模块。

## 12.3 若 dataset-free PASS、真实 \(D_R\) 无恢复

此结果只说明 dataset-free 条件下的可达性没有转化为真实恢复。下列内容是并列候选解释，当前证据不能预先排序：

- runner/adapter 在真实路径上的实现或契约差异；
- synthetic witness 与真实 feature witness 不同；
- 选取的 Base feature 分辨率/语义层不适合 miss completion；
- hard occupancy count 对真实 detector state 不够充分；
- paired synthetic intervention 不对应自然漏检机制。

下一步应先用冻结日志和最小干预区分 runner/adapter、feature witness、state sufficiency 与 intervention correspondence；在归因成立前，不简单扩大 decoder。

## 12.4 若恢复成功但新增 FA 过高

下列同样是待区分假设，不是由“FA 过高”单独推出的结论：

- common-mode phase shift；
- baseline/evidence 抵消失败；
- inactive surrogate 在 background 上积累；
- occupancy 稀疏区域释放过强；
- component connection 导致的 FA 计数变化。

当前 v12 不能在看到结果后直接调 threshold。若后续证据表明推理阈值或校准本身是独立研究变量，阈值变化应进入新的 model version，指标与门禁变化应进入新的 protocol revision，并预先定义校准规则、使用新的未见输入；否则应优先从状态约束或训练对象解释。

---

# 13. 创新性判断

## 13.1 已被现有工作覆盖的宽泛主张

以下内容不能单独作为主要创新：

- model-agnostic mask refinement；
- universal coarse-mask refinement；
- frozen Base + lightweight decoder；
- synthetic mask corruption/deletion；
- before/after difference learning；
- IRSTD 的 miss-vs-FA 平衡；
- target-level sensitivity；
- plug-and-play IRSTD module/loss；
- cross-backbone improvement；
- local iterative mask repair。

相关强先例包括：

- SegRefiner；
- SAMRefiner；
- RNCA；
- Miss Detection vs. False Alarm；
- Self-Supervised Difference Detection；
- IRSTD-Diff；
- MSHNet/SLS；
- NS-FPN。

因此，“轻量”“冻结”“通用”“差分”“减少 FA”都只能作为工程属性或背景，不是足够的论文主张。

## 13.2 当前仍有可能成立的窄创新边界

更有防御力的候选贡献是以下**完整组合**：

1. 在同一冻结 detector feature 下；
2. 对 detector coverage state 做具有稳定 target lineage 的同源受控干预；
3. 将同一 outcome row 内实际进入 loss 的 occupancy-plus/minus 端点作为联合训练对象，而不是把未被消费的 positive/null `match_id` 当作配对机制；
4. 直接优化 hard mask 之前的 coupled response；
5. 同时使用 factual-miss / factual-no-miss 零阶 anchors；
6. 以固定 Base、固定 FA 和 detected-target retention 检验 natural-miss recovery；
7. 用相对 active-v4 不增加 trainable parameter count 的 null-anchored local count crossing 表达该状态响应；这不等于没有额外计算或状态表达变化。

在本次有界检索和仓库已有检索中，没有发现完全相同的组合。但“未检索到”不等于“世界首创”，尤其不能排除：

- 不同术语；
- 未发表工作；
- 相邻任务中的同构方法；
- 近期 2026 工作。

## 13.3 NLCC 方程本身是否足够创新

单独看构件：

- mean/null anchoring；
- local count convolution；
- thresholded exponential；
- straight-through estimator；
- negative baseline；
- PixelShuffle；

都很难单独支撑顶会创新。

NLCC 的可能价值来自：

- 它是否是上述 intervention-coupled learning object 的必要、有效且可泛化的状态方程；
- 它是否在相同容量下优于合理 controls；
- 它是否能在真实 natural misses 上实现可重复恢复；
- 它是否跨冻结 detectors 成立。

因此，NLCC 更适合作为**机制实现贡献**，而不是单独依靠算子形式主张创新。

## 13.4 当前创新状态

最准确的状态仍是：

```text
BOUNDED_NOT_ESTABLISHED
```

当前可以说：

> NLCC-v12 是一个针对冻结 detector coverage intervention 的、容量受控的候选状态方程。

当前不能说：

- 模型已经成功；
- 方法已经创新；
- 首个通用 IRSTD refiner；
- 首个冻结 detector completion 方法；
- 已优于现有工作；
- 已具备 ICLR 录用级证据。

## 13.5 建立创新所需的最小证据链

1. runner closure 完整；
2. Development PASS；
3. 未参与 Development 的冻结 exposure Holdout PASS（这是独立输入剖面，不是独立统计重复）；
4. 真实 \(D_R\) natural-miss recovery PASS；
5. 固定 FA 与 retention PASS；
6. formal seed 42/43 分别 PASS；
7. outcome-row endpoint coupling、null、count crossing、surrogate 的分阶段必要性 controls；
8. generic refiners 和相同容量 comparator；
9. 至少多个冻结 detector；
10. 三数据集或足够有说服力的跨域证据；
11. 更新至投稿前的系统文献检索。

---

# 14. 推荐的论文定位

## 不推荐

> A novel lightweight universal mask refiner for infrared small target detection.

该表述会直接进入 SegRefiner、SAMRefiner、RNCA、OSCAR、NS-FPN 等已有工作密集区。

## 推荐的候选定位

> A frozen-detector residual completion framework that learns pre-threshold target response from controlled same-source coverage interventions, with zero-order anchors and false-alarm/retention-constrained natural-miss evaluation.

NLCC 可作为其中的状态方程贡献：

> A null-anchored local count-crossing operator that converts local coverage-count changes into phase active-set release without changing decoder capacity.

在结果成立前，使用 “candidate”“we study”“we test” 等措辞；只有机制、真实性能和必要性证据齐备后再使用更强结论。

---

# 15. 最终回答

## 整体模型设计如何？

**整体设计是内部自洽、可证伪的候选，但当前 v12 实例已经没有通过
Development。**  
它不是通过增加 attention、Transformer、多尺度 head 或第二 decoder 获得容量，而是在固定 active-v4 拓扑下针对 v11 的具体失败模式修改状态表达。occupancy late-entry、固定 null、unit count boundary、occupancy-invariant baseline、同一 outcome row 的端点干预和 hard-union 部署契约共同形成了一个清晰、可证伪的研究对象。

正式结果证明执行结构成立，但 factual anchors、背景/no-miss 抑制和 paired
绝对端点没有同时建立。它仍不能被直接归因于 surrogate、指数形式或
same-count limitation；这些属于新版本前的失败归因问题。

## 当前方案是否正确？

分层回答：

- **模型与方程层面：条件正确。**
- **输入冻结与验证顺序：正确。**
- **runner/evidence r2：已闭合。**
- **Development 执行：完整。**
- **Development 数值门禁：失败。**
- **现在创建 Holdout authorization：不正确。**
- **修改同一 v12 或重跑：不正确。**

当前状态必须保持为：NLCC-v12 Development 负结果冻结；Holdout、真实
\(D_R\)、exposure replay、formal800 和 R5 均未获授权。

## 是否满足创新？

**当前尚未满足“创新已建立”的证据标准。**

它曾满足的是：

> 一个窄范围、机制导向、可能有创新空间的候选设计。

当前结果说明 NLCC-v12 尚不能把这一候选提升为创新贡献。更高层的“同源
coverage intervention + coupled pre-hard-mask response + zero-order anchors
+ natural-miss recovery under fixed FA/retention”研究主线没有被一次
dataset-free 负结果整体否定；但任何后续实现都必须是新的、预先冻结的
CURE-Lite 模型版本，并重新从 Development 开始。

---

# 16. 主要源码与文献

## 仓库

- Repository commit:  
  https://github.com/Arialliy/cure-lite/commit/538660aa7e200bd7acad8964af25121ea56142cf
- NLCC-v12 design:  
  https://github.com/Arialliy/cure-lite/blob/538660aa7e200bd7acad8964af25121ea56142cf/CURE_Lite_NLCC_v12_%E6%A8%A1%E5%9E%8B%E4%B8%8E%E4%BB%A3%E7%A0%81%E8%AE%BE%E8%AE%A1.md
- Dataset-free runner:  
  https://github.com/Arialliy/cure-lite/blob/538660aa7e200bd7acad8964af25121ea56142cf/cure_lite/nlcc_dataset_free_runner.py
- Runner config:  
  https://github.com/Arialliy/cure-lite/blob/538660aa7e200bd7acad8964af25121ea56142cf/cure_lite/nlcc_dataset_free_runner_config.py
- NLCC decoder:  
  https://github.com/Arialliy/cure-lite/blob/538660aa7e200bd7acad8964af25121ea56142cf/cure_lite/null_anchored_local_count_crossing_decoder.py
- Factorized decoder:  
  https://github.com/Arialliy/cure-lite/blob/538660aa7e200bd7acad8964af25121ea56142cf/cure_lite/factorized_decoder.py
- Runner tests:  
  https://github.com/Arialliy/cure-lite/blob/538660aa7e200bd7acad8964af25121ea56142cf/tests_v12/test_nlcc_dataset_free_runner.py
- Repository literature audit:  
  https://github.com/Arialliy/cure-lite/blob/538660aa7e200bd7acad8964af25121ea56142cf/literature-search-20260724-cure-novelty/papers.md

## 相关主要工作

- ISNet: Shape Matters for Infrared Small Target Detection, CVPR 2022  
  https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_ISNet_Shape_Matters_for_Infrared_Small_Target_Detection_CVPR_2022_paper.html
- Infrared Small Target Detection with Scale and Location Sensitivity / MSHNet, CVPR 2024  
  https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html
- SegRefiner, NeurIPS 2023  
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc0cc55dca3d791c4a0bb2d8ddeefe4f-Abstract-Conference.html
- SAMRefiner, ICLR 2025  
  https://proceedings.iclr.cc/paper_files/paper/2025/file/90800f46f84381b7891e1378ee850013-Paper-Conference.pdf
- RNCA: Self-Repairing Segmentation Masks, MIDL/PMLR 2026  
  https://proceedings.mlr.press/v315/silbernagel26a.html
- Miss Detection vs. False Alarm, ICCV 2019  
  https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.html
- Self-Supervised Difference Detection, ICCV 2019  
  https://openaccess.thecvf.com/content_ICCV_2019/papers/Shimoda_Self-Supervised_Difference_Detection_for_Weakly-Supervised_Semantic_Segmentation_ICCV_2019_paper.pdf
- IRSTD-Diff / target-level insensitivity, 2024  
  https://arxiv.org/abs/2403.08380
- Seeing Through the Noise / NS-FPN, CVPR 2026  
  https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html
