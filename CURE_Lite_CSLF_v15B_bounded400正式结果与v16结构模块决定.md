# CURE-Lite CSLF v15B bounded-400 正式结果与 v16 结构模块决定

## 1. 正式状态

本轮候选为：

```text
support_oriented_response_joint
```

权威运行目录：

```text
runs/irstd1k_stage_a_seed42/
cure_lite_cslf_v15b_support_oriented_bounded_400_r1
```

正式决策：

```text
status = complete
decision = BOUNDED_SUPPORT_ORIENTED_CSLF_GATE_FAIL
failed_checks = [candidate_original_zero_level_gates]
bounded_gate_passed = false
```

这是一项完整结束、结果未过门槛的模型实验，不是执行失败。运行目录中：

- `COMPLETE.json` 存在；
- `.incomplete` 已移除；
- 不存在 `FAILURE.json`；
- 17 项正式产物全部存在，文件集合和 SHA256 与
  `COMPLETE.json` 逐项一致；
- 三个 checkpoint 均为 tensor-only safetensors，完成
  weights-only roundtrip；
- 训练严格使用 `D_R`；
- `D_V_accessed=false`、`D_T_accessed=false`；
- 未校准、未执行正式性能评估；
- 未授权 Formal-800、Full CURE 或其他 backbone。

关键封存值：

```text
COMPLETE file SHA256
= 58460fde25d08123231e2ab1ae5767f46ae3e40896b605b9e77c144413f6a896

complete fingerprint
= 13cc94f4f5140031fc050ac8d1726e13f9e5e1bbfa8a433bda28783088121f95

bounded result fingerprint
= f07beb7734d447dfd0954128cc0c8e05cbb2c171031ece2670055f110b930d84

candidate checkpoint SHA256
= 88082976404f4b25b01980e7bf06ad512cd5357ff5de4b04b56509f81904e754
```

源码闭包：

```text
archive
= artifacts/source_closures/
  cure_lite_cslf_v15b_support_oriented_bounded_400_13cc94f4f514.tar

archive SHA256
= e6ced21bef5926cb4fd6b9c79181980614eef3bf0fd7c14ac1cead63815cc069

manifest SHA256
= d5d5df197eab3bf4423777a4192f7d1bc0781518a54d9e56d37a8dbb48d9da8f

bound source files
= 178
```

## 2. 公平性与运行完整性

v15B 的 candidate、identity control 和 separable control 共享：

- seed 42；
- 同一个初始模型状态；
- 同一个 scalar cache 和 device cache；
- 同一个确定性选择序列；
- 同一个 optimizer；
- 每项目标 10 epochs × 40 steps；
- 每项目标 400 次 forward、backward 和 optimizer step；
- 每项目标 4,800 次 logical-state evaluation；
- 固定阈值 0，不进行阈值搜索。

identity 和 separable checkpoint 与 v15/v15A 逐字节一致：

```text
identity_joint
= a7a3735f9565bf129520b31e7af2f6653aea9824f4e5c8978b915c945369ca42

separable_endpoint
= 147ec39931f5c775d33a0917c237c4d3ca1c621f659dfc28594e3d2ac9f89ad3
```

因此三轮之间的 control、数据选择和训练预算没有漂移。

## 3. 三代候选的正式对比

| 指标 | v15 response-root | v15A completion-root | v15B support-oriented root |
|---|---:|---:|---:|
| factual-miss 逐样本通过 | 15/16 | 13/16 | **15/16** |
| factual miss 找回 | 16/16 | 16/16 | **16/16** |
| factual 目标内负像素 | 321/335 | 289/335 | **324/335** |
| factual-no-miss 通过 | 16/16 | 16/16 | **16/16** |
| clean response sign | 1396/1396 | 1396/1396 | 1395/1396 |
| clean added-target 负像素 | 6/149 | 125/149 | **145/149** |
| clean compact exact | 0/16 | 1/16 | **9/16** |
| clean component match | 3/16 | 15/16 | **15/16** |
| clean 完整门槛通过 | 0/16 | 1/16 | **8/16** |
| clean 新 completion | 14 | 168 | 157 |
| clean 目标外新增像素 | 8 | 43 | **12** |
| component-null | 17/17 | 17/17 | **17/17** |
| identity-null | 16/16 | 16/16 | **16/16** |
| invalid completion | 0 | 0 | **0** |

v15B 的核心改善是同时恢复 factual 覆盖并显著改善 clean
compactness：

```text
v15A -> v15B

factual gate:        13/16 -> 15/16
factual target:      289/335 -> 324/335
clean target:        125/149 -> 145/149
compact exact:       1/16 -> 9/16
outside completion:  43 -> 12
```

所以 support-oriented root 并非无效。它证实：

1. 在真实 added-target support 内锚定 completion endpoint，能够把
   finite response 转化为实际的零水平集 crossing；
2. support 外保留 plus-root，能够明显抑制 v15A 的全域外扩；
3. 这种改善没有破坏 factual-no-miss、component-null、
   identity-null 或 scalar-hidden 约束。

但它仍未达到完整模型门槛，不能授权 Formal-800。

## 4. 具体失败样本

### 4.1 factual miss

唯一失败样本：

```text
XDU680
target-negative = 64/75 = 85.33%
required        >= 95%
target recovered = true
connected support = 1/1
```

它不是完全漏检，而是目标区域覆盖不足。

### 4.2 clean pair

8 个通过，8 个失败。失败类型为：

| 样本 | 目标内负像素 | response | 目标外新增 | 主要问题 |
|---|---:|---:|---:|---|
| XDU153 | 21/23 | 119/119 | 0 | 少 2 个目标像素 |
| XDU318 | 13/13 | 103/103 | 2 | 轻微外扩 |
| XDU411 | 10/10 | 87/88 | 0 | 单个 response-sign 错误 |
| XDU678 | 7/7 | 85/85 | 1 | 轻微外扩 |
| XDU754 | 11/11 | 101/101 | 2 | 轻微外扩 |
| XDU812 | 22/23 | 146/146 | 6 | 少 1 个目标像素并外扩 |
| XDU849 | 0/1 | 49/49 | 0 | 单像素目标未 crossing，另有 3 个 false islands |
| XDU970 | 4/4 | 70/70 | 1 | 外扩并有 1 个 false island |

剩余问题已经从 v15A 的大范围外扩，收束为：

- 4 个目标像素未 crossing；
- 12 个目标外新增像素；
- 4 个 plus-state false-island components；
- 1 个 response-sign 像素错误；
- factual XDU680 的局部覆盖不足。

## 5. 研究决定：停止继续更换 loss root

v15、v15A、v15B 已经覆盖：

```text
plus-root
minus-root
support-dependent plus/minus root
```

三者均使用同一个 scalar field、同一个有限响应、同一个 measure 和
同一个固定零水平输出。v15B 已把 root-placement 的主要矛盾显著收束。
继续扩大 support mask、增加 halo、换另一组可逆坐标、增加 loss 权重、
延长训练或调整阈值，都属于围绕同一坐标族继续调节权衡，不再是合理的
下一模型模块。

因此正式冻结：

```text
v15B = valid complete negative result
root-coordinate search = stopped
Formal-800 = not authorized
```

## 6. v16：Phase-Preserving Coverage Encoding

### 6.1 当前结构缺口

当前模型先把原尺寸 occupancy 压缩为一个 coarse bool channel：

\[
\bar O=\operatorname{MaxPool}_{s}(O)
\in\{0,1\}^{1\times h\times w}.
\]

但输出端却预测 \(s^2\) 个 PixelShuffle phases：

\[
\phi=
\operatorname{PixelShuffle}_{s}
\left(W_\phi H\right).
\]

这形成了明确的信息分辨率不对称：

```text
input occupancy:  one value per stride cell
output field:     s² phase values per stride cell
```

同一 stride cell 内目标覆盖的具体相位和形状在输入端被抹去，输出端却
被要求精确恢复 1–3 像素目标、边界和紧致 support。v15B 的单像素
XDU849、少量 spill 和 false islands 与这一结构缺口一致。

### 6.2 单一结构修改

v16 暂定名：

```text
PPCE-CSLF
Phase-Preserving Coverage Encoding CSLF
```

定义无损 occupancy phase encoding：

\[
U_s(O)_{(r,c),i,j}
=
O_{si+r,sj+c},
\qquad
r,c\in\{0,\ldots,s-1\}.
\]

即：

\[
U_s(O)=\operatorname{PixelUnshuffle}_{s}(O)
\in\{0,1\}^{s^2\times h\times w}.
\]

模型仍是一条 scalar-field 路径：

\[
H_0=
\operatorname{SiLU}
\left(
W_{3\times3}
*
[N(F_b),U_s(O)]
\right),
\]

\[
H=
H_0+
\operatorname{SiLU}
\left(
DW_{3\times3}*H_0
\right),
\]

\[
\phi=
\operatorname{PixelShuffle}_{s}
\left(W_{1\times1}H\right).
\]

它只替换 occupancy 的有损表示，不增加：

- 新 head；
- 新 branch；
- 新 decoder stage；
- attention；
- 后处理；
- loss 项；
- 新的损失权重、可调权衡系数、margin、温度或阈值；
- 推理输入；
- 训练步数。

这不是在 v15B 后堆叠一个模块，而是修正同一个
\((F_b,O)\mapsto\phi\) 状态方程的输入表示，使 coverage phase 与
输出 phase 一一对齐。

正式配置 \(C=64,s=4,w=32\) 时，参数量由 19,536 增加到
23,856，即增加 4,320 个输入投影参数。这是
\((s^2-1)\times w\times3\times3\) 个相位输入连接，不是新增
decoder、head 或损失项。

### 6.3 为什么暂不采用 compact binary target

另一种可能是把 signed-distance target 改成目标内 \(-0.9\)、目标外
\(+0.9\) 的二值相位场。它可能增大零点裕量，但会把当前 signed
level-set supervision 收缩为普通 mask regression，并继续改变
target/loss，而不是修正模型结构。

本项目当前重心是完成模型代码。因而 v16 优先测试 PPCE；本轮不得把
PPCE 与 binary target 同时加入。

## 7. v16 冻结验证协议

保持不变：

- frozen Base 的通用 \((F_b,O)\) 接口；
- 单一 scalar completion field；
- v15B SORR objective；
- signed-distance target 与 integration measure；
- width；
- optimizer；
- seed 42；
- 400 updates；
- 12-state fused training；
- identity/separable matched controls；
- threshold \(=0\)；
- 全部原 zero-level gates；
- 只使用 `D_R`。

dataset-free 必须先证明：

1. PixelUnshuffle 后可逐字节重建原 occupancy；
2. \(s^2\) phase 顺序与 PixelShuffle 输出严格互逆；
3. 无目标、单像素、跨 cell、多组件 occupancy 均无信息丢失；
4. 模型仍为单一路径并满足固定参数量公式；
5. SORR selector、null reduction、fixed point 和有限梯度不变；
6. 两次 dataset-free 运行 canonical bytes 一致。

正式 bounded-400 仍只允许一个 `r1`。成功必须满足全部原 candidate
门槛，而不是“均值改善”：

```text
factual-miss              = 16/16
each factual target       >= 95% negative
factual-no-miss           = 16/16, zero residual

clean response            = 1396/1396
clean added target        = 149/149
clean compact exact       = 16/16
clean component match     = 16/16
outside completion        = 0
plus false islands        = 0

component-null            = 17/17
identity-null             = 16/16
scalar-hidden             = pass
invalid completion        = 0
```

若 PPCE 任一原门槛失败：

```text
stop current absolute scalar zero-level CSLF route
no more root changes
no binary-target rescue in the same development chain
no extra steps
no threshold adjustment
```

若全部通过，也只获得冻结确认资格，不自动授权 Formal-800、Full CURE
或其他 IRSTD backbone。

## 8. 当前模块进度

按完整模型设计链划分：

1. 通用 frozen Base \((F_b,O)\) 接口：完成；
2. coverage-state 数据、cache、12-state 训练契约：完成；
3. 单一 scalar level-set decoder：完成；
4. finite-response/SORR 学习坐标：完成并得到正式负结果；
5. phase-preserving coverage encoding：核心模型与 matched-training
   构造已实现，dataset-free、评估指纹和 bounded 协议正在闭合；
6. Formal-800 和冻结确认：尚未授权；
7. Full CURE、三数据集和跨 backbone：尚未开始。

当前不是最终模型成功。当前已经进入第 5 个模型设计模块。
