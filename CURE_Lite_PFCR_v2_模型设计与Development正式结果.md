# CURE-Lite PFCR-v2：模型设计、代码与 Development 正式结果

> 日期：2026-07-27  
> 当前范围：**只设计并验证 CURE-Lite**  
> 当前主线：**detector-independent feature–coverage relevance learning 与受约束 evidence release**  
> 当前模型：**Phase-resolved Feature–Coverage Relation（PFCR）**  
> 当前状态：**核心关系状态、完整 decoder、训练目标与 Development runner 已实现；固定 population 上两个 seed 均通过；真实数据集训练尚未开始**

---

## 0. 结论

当前不能再把项目状态写成“仍在分析 NLCC 为什么失败”。本轮已经完成一条新的
CURE-Lite 模型代码链：

```text
R13-1 角色冲突
    ↓ 分因素归因
旧输入依赖样本专属 float 数值
    ↓
独立 relation-state input contract v2
    ↓
phase-resolved feature–coverage relation
    ↓
relation-controlled evidence release decoder
    ↓
最差正/负端点训练目标
    ↓
seed 42 / 43 的 320-update Development 均通过
```

可以正式记录：

```text
old NLCC-v12 training                    NOT_AUTHORIZED
GCDE-v13 objective-only training         NOT_AUTHORIZED
PFCR input contract v2                   PASS
PFCR analytic representational check     PASS
PFCR learned Development seed 42         PASS
PFCR learned Development seed 43         PASS
real dataset training implementation     AUTHORIZED_NEXT
real dataset model success               NOT_ESTABLISHED
full CURE                                OUT_OF_CURRENT_SCOPE
```

因此，当前模型已经从“结构设想”进入“完整核心方程可训练、代码可运行”的阶段；
但只有固定 dataset-free population 的机制支持，尚不能宣称真实红外小目标检测模型
已经成功。

---

# 1. CURE-Lite 要解决的问题

冻结 Base IRSTD detector 给出：

\[
F_b\in\mathbb R^{B\times C\times h\times w},
\qquad
O\in\{0,1\}^{B\times1\times H\times W}.
\]

其中 \(F_b\) 是 Base 特征，\(O\) 是 Base 已检测区域。

CURE-Lite 的目标不是重新做一次完整检测，而是预测 Base 没有释放出来的目标残差：

\[
\Delta O=\operatorname{CURELite}(F_b,O),
\qquad
O_{\mathrm{final}}=O\lor\Delta O.
\]

关键困难不是“局部是否存在 occupancy”，而是：

> 局部 occupancy 是否与当前候选目标证据属于同一个目标。

无方向局部 count：

\[
C(x)=\sum_{\delta}U_\delta(x)
\]

不能区分：

- 同一目标的已检测分量，应抑制重复释放；
- 邻近的另一个目标或无关分量，不应抑制当前目标找回。

同时，1～3 像素小目标在同一个 coarse feature cell 内具有不同输出 phase。若输入
只保留一个 cell-level scalar，模型只能依赖样本专属数值猜测应释放哪些 subpixel。

PFCR 同时处理这两个瓶颈，但只使用一个共享关系算子，不增加多尺度分支、额外
Transformer 或人工角色标识。

---

# 2. 旧输入失败的正式归因

新增的完整 \(2\times2\times2\) quotient lattice 分别控制：

- 是否去除 feature magnitude；
- 是否去除 feature sign；
- 是否去除绝对空间 origin。

正式结果：

| feature 表示 | 坐标 | key 数 | 冲突 key | 冲突记录 | 相反监督记录对 |
|---|---|---:|---:|---:|---:|
| exact float | absolute | 75,192 | 0 | 0 | 0 |
| exact float | relative | 75,192 | 0 | 0 | 0 |
| absolute value | absolute | 75,192 | 0 | 0 | 0 |
| absolute value | relative | 75,192 | 0 | 0 | 0 |
| signed support | absolute | 69,704 | 8 | 16 | 8 |
| signed support | relative | 47,897 | 8 | 46 | 74 |
| unsigned support | absolute | 56,378 | 22 | 56 | 40 |
| unsigned support | relative | 36,524 | 7 | 64 | 300 |

最准确的解释是：

1. 单独去除绝对位置不产生冲突；
2. 单独去除符号不产生冲突；
3. 首先去除样本专属精确幅值后立即产生冲突；
4. 在幅值被去除后，再去除位置或符号会合并更多相反监督记录；
5. 旧 population 的可识别性主要依赖按 state/group 生成的 float 数值，而不是
   可迁移的 feature–coverage 关系。

冲突包括两类：

```text
1px negative phase ↔ 3px positive phase
adjacent-plus negative ↔ multicount-minus positive
```

第一类说明旧特征没有结构化 subpixel evidence；第二类说明 scalar count 无法区分
相关覆盖与邻近无关覆盖。

正式产物：

```text
protocols/IRSTD-1K/gate_covered_dual_endpoint_v13/
    r13_1a_conflict_attribution_receipt.json
```

receipt fingerprint：

```text
6b5f45f520b8f8f637ec047b5fc2e7795aca650fd46299d1786be34b5ef5c20e
```

文件 SHA256：

```text
2f7924d15a000dcf576021556c789900cd32e0e87d81ffc4d54fa1a9a3bb36a0
```

两次重建逐字节一致。

---

# 3. PFCR-v2 的核心机制

## 3.1 唯一外部接口

模型 forward 只接收：

```text
feature:   float [B,C,h,w]
occupancy: bool  [B,1,H,W]
```

不读取：

```text
GT
group
pair
endpoint
geometry family
prototype ID
match ID
absolute coordinate embedding
```

不同 IRSTD 前端只需提供 \(F_b\)、\(O\)、feature channel 数和 stride。

## 3.2 Phase query 与 coverage key

一个无 bias 的共享 \(1\times1\) 投影产生 \(P=s^2\) 个 phase query 和一个
coverage key：

\[
[Q_1,\ldots,Q_P,K]=\Phi(F_b),
\]

\[
Q_j,K\in\mathbb R^{B\times d\times h\times w}.
\]

采用逐 cell、零保持归一化：

\[
\widehat q=
\frac{q}{\max(\lVert q\rVert_2,\epsilon)},
\qquad
\widehat k=
\frac{k}{\max(\lVert k\rVert_2,\epsilon)}.
\]

因此：

\[
F_b=0\Rightarrow Q=K=0.
\]

## 3.3 Directional occupancy basis

先用 adaptive max 将输出 occupancy 投影到 feature grid：

\[
\bar O=\Pi_{\max}(O).
\]

然后保留完整 \(3\times3\) 方向 basis：

\[
U_\delta(x)=\bar O(x+\delta),
\qquad
\delta\in\{-1,0,1\}^2.
\]

不再提前求和为 scalar count。

## 3.4 平滑、零保持的 feature–coverage relation

令：

\[
\rho_{j,\delta}(x)
=
\widehat Q_j(x)^\top
\widehat K(x+\delta).
\]

实际 learned decoder 使用固定 \(\tau=4\) 的平滑关系：

\[
A_{j,\delta}(x)=
\frac{\operatorname{softplus}(\tau\rho_{j,\delta}(x))}
{\operatorname{softplus}(\tau)}
\cdot
\mathbf 1[\lVert Q_j(x)\rVert>0]
\cdot
\mathbf 1[\lVert K(x+\delta)\rVert>0].
\]

这样满足：

- 相同方向映射为 1；
- 不同方向连续可分；
- 负 cosine 区域仍有梯度；
- query 或 key 为零时 affinity 精确为零；
- 不引入额外参数。

相关覆盖：

\[
R_{j,\delta}(x)=U_\delta(x)A_{j,\delta}(x).
\]

有界 burden：

\[
C^{rel}_j(x)
=
1-\prod_{\delta}
\left(1-R_{j,\delta}(x)\right).
\]

完整 \(R\in\mathbb R^{B\times P\times9\times h\times w}\) 被保留用于审计；
\(C^{rel}\) 只是状态方程中的确定性归约。

## 3.5 同一个 query 提供 phase evidence

不再增加独立 evidence branch。query norm 本身给出证据强度：

\[
E_j(F_b)
=
\operatorname{softplus}
\left(\lVert Q_j\rVert_2^2\right)
-\log2.
\]

因此：

\[
Q_j=0\Rightarrow E_j=0.
\]

## 3.6 Relation-controlled evidence release

释放 gate：

\[
V_j=1-C^{rel}_j.
\]

证据：

\[
\widetilde E_j=E_jV_j.
\]

负 baseline：

\[
B=-\operatorname{softplus}(\beta)<0.
\]

最终 native phase logit：

\[
Z_j=B+\widetilde E_j.
\]

通过 PixelShuffle 得到原分辨率 residual logit：

\[
Z=\operatorname{PixelShuffle}(Z_1,\ldots,Z_P).
\]

模型明确禁止对 tiny-target logits 做双线性插值：

\[
H=sh,\qquad W=sw.
\]

校准后：

\[
\Delta O=\mathbf 1[\sigma(Z)>\tau_{\mathrm{cal}}],
\qquad
O_{\mathrm{final}}=O\lor\Delta O.
\]

---

# 4. 为什么这不是模块堆叠

PFCR-v2 的核心不是：

```text
NLCC + attention + multi-scale + extra decoder
```

而是一次状态替换：

\[
\text{scalar local count}
\quad\Longrightarrow\quad
\text{phase-resolved feature–coverage relation}.
\]

query 同时提供：

- phase-specific target evidence；
- 与局部 coverage key 的关系坐标。

occupancy 只通过一个固定 directional basis 进入；relation 只通过一个共享投影学习；
evidence release 是一条封闭方程。

当前最强的诚实创新类型是：

1. **新状态表示与推理机制**：把 vacancy/count 改写为 feature-conditioned relevant
   coverage；
2. **细粒度证据释放原则**：只在目标证据与 Base coverage 不相关时释放 residual。

GCDE、最差端点 loss 和 dataset-free population 都是训练/验证工具，不是模型核心
创新。

---

# 5. 参数量与复杂度

参数公式：

\[
\#\theta=C(P+1)d+1,
\qquad P=s^2.
\]

固定 population 预检使用：

```text
C=32
s=4
P=16
d=2
parameters=1,089
```

若真实 Base feature 为：

```text
C=64
s=4
d=8
```

则：

\[
64\times17\times8+1=8,705
\]

个参数。

PFCR 的主要中间量为：

\[
B\times P\times9\times h\times w.
\]

它不增加第二次 Base forward，也不使用全局 \(HW\times HW\) attention。

---

# 6. 独立 input contract v2 结果

v2 population 先生成潜在目标场，再统一编码 feature，最后才选择 endpoint
occupancy 并生成 completion truth。

```text
pair specs                         16
endpoint states                    32
supervised records             25,088
positive records                   64
relation-role keys                 27
opposite-supervision conflicts      0
same-geometry relation groups       8
same-geometry groups passed       8/8
analytic mismatch pixels            0
```

门禁：

```text
input_contract_v2_pass=true
relation_state_implementation_authorized=true
full_decoder_training_authorized=false  # receipt 生成时 decoder 尚未闭合
```

正式产物：

```text
protocols/IRSTD-1K/phase_resolved_feature_coverage_relation_v2/
    input_contract_preflight_receipt.json
```

receipt fingerprint：

```text
51cdda0004d4be4eed511154239e3145da58ff1bd56354b95f5e427a0d9563e8
```

文件 SHA256：

```text
93abf554f67b432995bfd94a849b0b8d098777bb42a2310c20c4b9d0955fdc8a
```

两次重建逐字节一致。

---

# 7. Learned Development 正式结果

## 7.1 冻结训练设置

```text
updates                   320
optimizer                 Adam
learning rate             0.01
betas                     (0.9, 0.999)
weight decay              0
train positive target     0.951
train negative target     0.049
evaluation positive       > 0.95
evaluation negative       < 0.05
threshold mismatch        0 pixels at threshold 0.5
```

训练风险不是像素均值，而是每个 state 的：

\[
\mathcal L_i^+
=
\operatorname{softplus}
\left(\kappa-\min_{p\in T_i}z_p\right),
\]

\[
\mathcal L_i^-
=
\operatorname{softplus}
\left(\kappa+\max_{p\in N_i}z_p\right),
\]

\[
\kappa=\log\frac{0.951}{0.049}.
\]

有正、负两类监督的 state 对两项取平均；targetless state 只消费负端点。

## 7.2 两个 seed

| seed | mismatch pixels | positive min | negative max | baseline prob | final loss | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 0 | 0.994816601 | 0.009262450 | 0.007628834 | 0.100820862 | PASS |
| 43 | 0 | 0.998534203 | 0.007652447 | 0.007386379 | 0.094545759 | PASS |

seed 42：

```text
result fingerprint
f75c30a34866af9ade002efea08ca6f4437dfebb7fa72e319add97b971e9b8bf

file SHA256
997a966c335ca136db3107f9f74609f40ba03633730f3a7088da6d1290007e1b
```

seed 43：

```text
result fingerprint
7552948328be41192c3cbb72331e28a4f9e433927306cb83fb2a47113aaad4a4

file SHA256
8891924ea7974ef9194a16cf9e41dba3a050f2b31bd1b6b0ce3f63eae5783997
```

两个正式结果均完成内存重建并与文件逐字节一致。

## 7.3 优化过程中的有效负结果

本轮没有隐藏两次失败尝试：

1. 像素均值负损失：
   - positive min 已高于 0.998；
   - 仍有 20/22 个错误负像素；
   - 原因是少量错误被大量背景均值稀释。
2. hard ReLU cosine + 最差端点：
   - 负 cosine 区域梯度为零；
   - 两个 seed 分别剩 16/14 个错误。
3. smooth zero-preserving cosine：
   - seed 43 先通过；
   - seed 42 仍受 query/key 独立随机起点影响。
4. query/key 耦合初始化：
   - 不增加参数；
   - seed 42/43 均通过。

这条修改链说明最终结果不是通过延长训练或更换 seed 获得，而是分别修复：

- risk 与最差像素门禁不一致；
- relation 在负 cosine 区域不可优化；
- query/key 初始关系不稳定。

---

# 8. 当前代码

核心模型：

```text
cure_lite/phase_resolved_feature_coverage_relation.py
cure_lite/phase_resolved_relation_decoder.py
```

输入与预检：

```text
cure_lite/phase_resolved_relation_population.py
cure_lite/phase_resolved_relation_preflight.py
```

训练：

```text
cure_lite/phase_resolved_relation_training.py
```

命令入口：

```text
tools/audit_phase_resolved_relation_preflight.py
tools/train_phase_resolved_relation_development.py
```

归因：

```text
cure_lite/nlcc_role_conflict_attribution.py
tools/audit_nlcc_role_conflict_attribution.py
```

测试：

```text
tests_v14/
```

当前：

```text
tests_v14                  27 passed
tests_v13 + tests_v14      39 passed
```

---

# 9. 当前证据边界

当前可以说：

> 在固定、无样本专属 float 标识的 phase-resolved scene population 上，
> feature-conditioned relevant coverage 能区分相同 occupancy 几何下的相关覆盖与
> 邻近其他目标；完整 CURE-Lite relation-controlled release decoder 能从随机初始化
> 在 seed 42/43 的 320 updates 内满足全部冻结端点。

当前不能说：

- 已在 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 上提高 Pd、IoU 或降低 Fa；
- 已证明对真实 frozen detector feature 有效；
- 已证明跨 detector 通用；
- 已完成 CURE-Lite；
- 已满足 ICLR 投稿证据；
- 已授权 Full CURE。

---

# 10. 下一步：进入真实 CURE-Lite，而不是再做 toy 模块

下一阶段只做 CURE-Lite：

```text
Stage R1  generic real-cache adapter
    -> 读取现有 Stage-A (F_b, O, Y, valid_mask)
    -> 不修改数据集名称
    -> 不依赖 MSHNet 模型代码

Stage R2  real-state materialization
    -> factual miss / factual no-miss
    -> lineage-safe synthetic endpoint
    -> relation fields 与 phase-resolution 审计

Stage R3  real mini-batch training
    -> 复用 PFCR decoder
    -> 复用最差端点绝对目标
    -> checkpoint / resume / finite-state / receipt

Stage R4  calibration and evaluation
    -> D_R 训练
    -> D_V 校准
    -> D_T 一次冻结评估
    -> Pd、Fa、IoU、nIoU、fixed-miss recovery

Stage R5  三数据集正式验证
    -> NUAA-SIRST
    -> NUDT-SIRST
    -> IRSTD-1K
    -> 每个数据集固定 800 epochs
```

数据目录可以从：

```text
/home/md0/ly/MSHNet/datasets
```

只读使用现有 NUAA、NUDT-SIRST、IRSTD-1K 数据，不修改数据集命名，也不把
CURE-Lite 接到 MSHNet 网络结构中。

当前最近的代码任务是：

> 为 PFCR decoder 补齐 detector-independent 的真实 cache adapter、mini-batch
> trainer、calibration 和 evaluation 入口，然后先在一个冻结开发划分上运行短程
> preflight；通过后再启动 800-epoch 三数据集训练。
