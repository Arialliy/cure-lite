# CURE-Lite CSLF-v15：完整模型设计与阶段结果

> 更新时间：2026-07-27  
> 当前阶段：真实 \(D_R\) bounded-400 已完成，进入单场绝对零水平锚定修订  
> 当前结论：训练与比较代码闭合；response-joint 学到相对响应，但未通过固定零水平门禁  
> 当前完成度：约 65%（代码与协议进度；不代表模型已经设计成功）

> **后续状态说明：** 本文保留 v15 的冻结设计与结果。其后 v15A
> completion-rooted bounded-400 也已完整结束：added-target 越零由 6/149
> 提高至 125/149，但 compact exact 仅 1/16、目标外新增 43 像素，factual
> gate 降至 13/16，因此仍未通过。当前结果与 v15B support-oriented root
> 决定见
> [`CURE_Lite_CSLF_v15A_bounded400正式结果与v15B模型决定.md`](CURE_Lite_CSLF_v15A_bounded400正式结果与v15B模型决定.md)。

---

## 1. 主线是否改变

没有改变。

CURE-Lite 当前只研究一个核心学习对象：

\[
(F_b,O)
\xrightarrow{\mathrm{CSLF}_\theta}
\phi_\theta,
\]

其中：

- \(F_b\) 是任意冻结 IRSTD detector 可导出的空间特征；
- \(O\) 是该 detector 的基础检测占用状态；
- \(\phi_\theta\) 是一个全分辨率、连续、带符号的 residual-completion
  field；
- \(\phi_\theta<0\) 表示 Base 尚未覆盖、但应补充为目标的区域；
- \(O\) 是不可改写集合，最终输出固定为

\[
\widehat Y
=
O\cup
\bigl(
\{\phi_\theta<0\}\cap \neg O
\bigr).
\]

因此 CURE-Lite 不是在 MSHNet、SCTransNet 或其他特定 backbone 内部修改
encoder、decoder、跳跃连接或注意力。当前 detector-independent 边界始终是
\((F_b,O)\)。

---

## 2. 它解决什么问题

现有 IRSTD detector 常同时出现两类错误：

1. 小而弱的真实目标没有进入 Base 占用区域；
2. 直接放宽检测阈值虽然可能找回目标，却也会产生额外背景响应。

CURE-Lite 不重新执行前端目标检测，而是在冻结 Base 之后学习：

> 给定相同的冻结目标证据，当 Base 的 coverage state 改变时，未覆盖区域中的
> residual completion field 应如何响应。

模型只允许“补充”，不能删除或覆盖 Base 已检测区域。这使目标找回与 Base
保留成为同一输出约束，而不是两个后处理分支。

---

## 3. 网络结构

当前 scalar CSLF 只包含一条场预测路径：

```text
Frozen detector feature F_b
        │ sample-wise global RMS normalization
        ▼
normalized feature
        │
Base occupancy O ── max projection to feature grid
        │
        └──────── concatenate ────────┐
                                      ▼
                               3×3 input projection
                                      ▼
                               depthwise 3×3 mixing
                                      ▼
                         phase projection to stride² channels
                                      ▼
                                PixelShuffle
                                      ▼
                    one full-resolution signed field φ
                                      ▼
                    fixed zero level set + hard Base union
```

结构由三组可训练权重组成：

1. `input_projection`：联合读取冻结特征和 coverage state；
2. `spatial_mixing`：在同一隐藏场内传播局部空间证据；
3. `phase_projection`：一次性恢复到像素相位坐标并输出唯一 signed field。

当前正式配置固定：

- feature stride：4；
- hidden width：32；
- floating point：FP32；
- AMP：关闭；
- threshold：固定为 0，不校准；
- 输出对象：一个 field，不含 proposal head、额外 null head、额外分支或后处理
  搜索。

这是一种单机制模型，不是“注意力 + 多尺度 + 边缘头 + 拓扑头”的模块堆叠。

---

## 4. 学习目标

监督目标是固定幅值、截断的 signed distance field：

\[
\phi^\star(x)<0,\quad x\in T,
\qquad
\phi^\star(x)>0,\quad x\notin T.
\]

训练使用离散 Sobolev 风险，同时约束场值和空间变化：

\[
\mathcal L_{\mathrm{abs}}
=
\|\phi-\phi^\star\|_{L^p(\mu)}
+
\|\nabla\phi-\nabla\phi^\star\|_{L^p(\mu)}.
\]

一次更新固定包含：

- 4 个 factual-miss states；
- 4 个 factual-no-miss states；
- 1 个 clean-positive pair 的两个 endpoint；
- 1 个 component-null pair 的两个 endpoint。

所以每次更新是 12 个 logical states，但只执行一次模型 forward、一次 backward
和一次 optimizer step。

---

## 5. 三个 matched objective 的作用

三个模型使用完全相同的：

- 初始化；
- 网络结构；
- 训练 population；
- update schedule；
- optimizer；
- batch endpoints；
- 训练步数；
- 数值精度。

唯一允许变化的是 pair coordinate：

1. `response_joint`：监督 coverage 干预对应的场响应；
2. `identity_joint`：同联合测度下的恒等响应对照；
3. `separable_endpoint`：分别学习两个 endpoint 的绝对目标。

它们不是三个模型模块，而是对同一个模型机制的三个等预算学习坐标。只有
`response_joint` 在后续固定验证中稳定优于两个对照，coverage-state response
才具有独立方法意义。

---

## 6. 已完成的真实 \(D_R\) 输入结果

严格 create-only 重建已经完成，未训练，也未读取 \(D_V/D_T\)：

| 项目 | 结果 |
|---|---:|
| \(D_R\) 图像 | 160 |
| natural states | 167 |
| factual miss | 32 |
| factual no-miss | 135 |
| pair states | 383 |
| clean-positive eligible | 206 |
| component-null eligible | 16 |
| component-null diagnostic | 1 |
| identity diagnostic | 160 |
| exclusions | 216 |
| observability decision | `AUTHORIZE_SCALAR_CSLF` |
| feature stride / field radius | 4 / 4 |

关键指纹：

```text
geometry:
16a0a587341d5403614fcc9de90581727f107f6566b195c1b10a857f7d185fa8

raw catalog:
15eaf6d5482d908f2c0f9899e4495eaff329ccb4d5c2aae648c4bba98ef79a24

observability:
0c603611582534ca686fb177ac5786db4fee3f0c03f434b994062312e97ec214

scalar cache:
569b0fb97d819cf1281ca1d148227bc1c5e229b8301065cb536656b5e578e645

complete real-input build:
ee717a7e13461fb86cacc65d33efd331abcf9b27611f254f981082d45eb7bfb4
```

真实构建耗时 185.38 秒，峰值内存约 2.47 GiB。新入口得到的 scalar-cache
指纹与此前正式结果一致，说明复用入口没有改变人口或监督语义。

---

## 7. 训练前门禁结果

### 7.1 Observability

两次正式运行逐字节一致，结果为：

- scalar duplicate input-target conflicts：0；
- phase duplicate input-target conflicts：0；
- scalar receptive field 外 target-response pixels：0；
- phase receptive field 外 target-response pixels：0；
- scalar-hidden pair：1，但其 target response 为 0；
- scalar CSLF：授权；
- PP-CSLF：当前不需要，也未授权。

### 7.2 Dataset-free

expanded dataset-free gate 包含：

- 16 类状态；
- 2 个分辨率；
- 3 个 seeds；
- 96 个几何/表示结果；
- 3 objectives × 3 seeds 的短训练结果；
- 17 项重算门禁。

结果为 `DATASET_FREE_GATE_PASS`，receipt 指纹：

```text
c8353f89c8594879400d5dee687165405cdab6eeeccdc8b817dc4a7765a33131
```

门禁布尔值现在必须从原始 case/training evidence 重新计算，不能通过修改汇总
字段制造通过结果。

### 7.3 Formal schedule exposure

seed 42/43 的完整 \(800\times40=32{,}000\) schedule 均已通过训练前暴露门禁：

- 每个 factual-miss record 和 focus target 的暴露次数均为 4000；
- zero exposure：0；
- factual-miss target ESS：32；
- identity/diagnostic optimizer exposure：0；
- 每步固定 4/4/1/1 和 12 个 logical states。

这只证明日程分散性，不代表 Formal-800 已经获得运行许可。

---

## 8. 已完成的性能工程

训练数据现在先打包为设备常驻 cache：

- natural states 全部打包；
- 只打包 eligible clean/component pairs；
- identity 和 diagnostic pairs 不进入 optimizer device store；
- 三个 matched objectives 共用一次打包结果；
- update loop 内只执行设备本地 row gather；
- update loop 内不再构造目标几何；
- update loop 内不再执行 CPU→GPU payload 传输；
- packed tensor 内容、device、dtype、source cache 和 ID index 均有绑定。

对完整真实 \(D_R\) population，设备常驻 payload 约 1.2205 GiB；bounded
16-per-role population 会明显小于该数值。相较旧路径，32,000 步估计可避免约
757.8 GiB 的重复 payload 传输。该改动只优化执行，不改变模型、loss、schedule
或研究主张。

---

## 9. 当前代码验证

当前完整 `tests_v15`：

```text
138 passed
1 skipped
```

唯一默认 skipped 项是耗时较长的真实 \(D_R\) create-only integration test；同一
真实构建入口已单独执行成功，并得到第 6 节结果。

与设备缓存、训练、bounded authorization 直接相关的定向验证均已通过。随后
真实 \(D_R\) bounded-400 已完整执行并封存；结果见：

[CURE-Lite CSLF-v15：bounded-400 正式结果](./CURE_Lite_CSLF_v15_bounded400正式结果.md)

正式决定为 `BOUNDED_CSLF_GATE_FAIL`，唯一顶层失败项是
`zero_level_gates`。这不是程序失败：三个 objective 均完成 400 次更新，产物
完整，未读取新的 \(D_V/D_T\)。

### 9.1 bounded-400 核心数值

| objective | factual miss | factual no-miss | clean compact | component null | identity null |
|---|---:|---:|---:|---:|---:|
| response-joint | 15/16 | 16/16 | 0/16 | 17/17 | 16/16 |
| identity-joint | 13/16 | 16/16 | 0/16 | 17/17 | 16/16 |
| separable-endpoint | 15/16 | 15/16 | 9/16 | 15/17 | 16/16 |

response-joint 的 clean-positive 响应方向为 1396/1396，但 added-target 中只有
6/149 个像素越过零水平。它保住了全部 no-miss/null 状态，却没有形成完整紧致
支持。separable-endpoint 有 137/149 个像素越过零水平，但破坏了部分 no-miss
和 component-null。当前问题因此被定位为：相对响应、绝对零水平锚定和
null-state 保持尚未在同一个场中统一。

---

## 10. 历史负结果与当前模型的关系

已经冻结的 v0.2、PFCR、NLCC、GCDE、CSLF 前置候选等负结果不能删除，也不能
改写成成功。它们分别否定了旧版本中的：

- 只改变 legal target 边缘频率的硬匹配；
- 不具有充分状态支撑的局部关系量；
- 不能形成稳定 residual support 的旧候选表示。

CSLF-v15 是根据这些失败重新定义学习对象后的新模型。此前负结果没有证明
CSLF 成功，但它们解释了为什么当前必须直接预测一个受 coverage state 约束的
完整 residual field。

---

## 11. 当前创新点的准确边界

不能单独作为创新的部分包括：

- level-set/SDF；
- PixelShuffle；
- Sobolev loss；
- frozen feature 后接轻量 head；
- 一般 paired supervision。

当前真正需要验证的组合性创新是：

> 在冻结 detector evidence 不变时，只改变 detector coverage state；模型以同一
> signed field 同时表达空 residual、多目标 residual、边界和空间支持，并通过
> completion-only hard union 保证只在 Base 未覆盖区域增加目标。

它形成的是“状态干预—场响应—受限完成”一条机制链，而不是多个独立增强模块。
但只有 matched objectives、错误 pairing、输入消融和同容量 residual-refiner
对照都支持该机制后，才具有 ICLR 级方法主张的基础。

---

## 12. 什么结果才算 bounded 模型成立

真实 \(D_R\) bounded-400 对每个 objective 固定执行：

\[
10\times40=400
\]

次更新，并在 threshold \(=0\) 下检查：

1. factual miss：产生位于固定目标支持内的负场；
2. factual no-miss：不产生额外负岛；
3. clean positive：新增 completion 与 added target 精确对应且无外溢；
4. component null：删除已覆盖组件不能制造新 completion；
5. identity null：相同实际输入必须得到完全相同的场；
6. 三个 objective：计算账本、初始化、日程、optimizer、device cache 完全匹配。

若任一 objective 未通过，当前 bounded gate 就是失败；不会用平均值掩盖，也不会
自动重跑。失败后只根据失败角色修改模型，不进入 Formal-800。

---

## 13. 后续顺序

```text
[DONE] CSLF 单场核心
  -> [DONE] 真实 D_R population / scalar authorization
  -> [DONE] scalar target cache / fused step / matched runner
  -> [DONE] dataset-free / schedule exposure / zero-level evaluator
  -> [DONE] device-resident training path
  -> [DONE/FAIL] 单次真实 D_R bounded-400
  -> [CURRENT] 最小绝对零水平锚定修订
  -> 重新通过 dataset-free 与 D_R bounded-400
  -> Formal-800
  -> 冻结后一次 D_V 揭示
  -> seed 42/43 逐 seed 门槛
  -> 独立确认
  -> CURE-Lite 成功
  -> 才开始 Full CURE 和跨 backbone/三数据集验证
```

因此，现在还不是使用 NUAA、NUDT-SIRST、IRSTD-1K 三个完整数据集训练的时候。
先用当前 IRSTD-1K 的冻结开发协议判断 CURE-Lite 本体是否能够学习正确的
coverage-state field。只有 Lite 本体通过稳定验证，才把 CURE 作为统一后端接到
DNANet、UIUNet、MSHNet、SCTransNet 等 detector 的 \((F_b,O)\) 接口，并在三大
数据集比较同一 backbone 的 Base 与 Base+CURE。

---

## 14. 当前最终判断

- 主线：未修改；
- 模型结构：已实现；
- 真实输入入口：已实现并运行成功；
- 训练循环：已实现并消除主要重复计算；
- dataset-free：通过；
- bounded 运行代码与授权：已实现并完成单次运行；
- bounded 结果：完整阴性结果，`BOUNDED_CSLF_GATE_FAIL`；
- 主要失败签名：response 方向正确，但 clean-positive 绝对零水平穿越不足；
- Formal-800：未授权；
- \(D_V/D_T\)：未读取新结果；
- CURE-Lite 是否成功：尚不能判断；
- Full CURE：尚未开始；
- ICLR 核心主张：尚未由性能和 matched controls 建立。

当前正确动作不是再增加模块或扩大数据集，而是在同一 CSLF 单场主线内完成
最小绝对零水平锚定修订，并重新运行预先固定的 \(D_R\)-only bounded-400。
