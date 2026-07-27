# CURE-Lite v17 CMIF 正式负结果与 v18 PMOPE 修改方案

## 1. 阶段结论

CURE-Lite v17 使用固定 `seed=42` 完成了唯一一次 bounded-400 正式运行。运行、训练、评估和产物封存均正常，但候选目标函数未通过预声明门禁：

```text
status = CMIF_V17_BOUNDED_400_GATE_FAIL
execution_failure = false
formal_800_authorized = false
full_CURE_authorized = false
cross_backbone_authorized = false
```

这是一项完整负结果，不是程序错误，也不是多 seed 波动。v17 不追加训练、不更换 seed、不重跑。

## 2. 正式结果

| Pair objective | Factual 严格通过 | Factual 找回 | Clean defined | Clean compact | Component-null | 其他 null | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| SORR candidate | 7/16 | 15/16 | 0/16 | 0/16 | 16/16 | 全部通过 | Fail |
| Identity joint | 9/16 | 15/16 | 0/16 | 0/16 | 16/16 | 全部通过 | Fail |
| Separable endpoint | 14/16 | 16/16 | 1/16 | 1/16 | 15/16 | 全部通过 | Fail |

更细的像素结果为：

```text
SORR factual target coverage       = 232 / 335 = 69.25%
Identity factual target coverage   = 272 / 335 = 81.19%
Separable factual target coverage  = 320 / 335 = 95.52%

SORR clean target crossing         = 18 / 149 = 12.08%
Identity clean target crossing     = 15 / 149 = 10.07%
Separable clean target crossing    = 122 / 149 = 81.88%
```

SORR 的 clean response 方向有 `1327/1396` 个像素正确，但只有 `18/149` 个新增目标像素真正跨过零点；12/16 个 clean pair 完全没有目标像素跨零。Separable 可以产生更强响应，却产生 29 个目标外 completion 像素，并有一个 component-null 失败。

因此 v17 的核心失败是：

> 在固定 400-step 预算下，现有 SORR 连续场回归没有把大体正确的响应方向转化为足够、紧致且可控的零点穿越。

## 3. 已排除的错误解释

当前证据不支持以下解释：

1. **实现或符号错误**：CMIF 方程、SORR selector、训练分支、参数更新和零层评估均通过契约检查；响应方向也大体正确。
2. **目标在积分测度中被全图淹没**：16 个 clean pair 中，added target 占总积分质量平均 27.17%，占 focus stratum 平均 81.50%。
3. **已证明的强分支梯度冲突**：一个冻结 batch 上的 miss-pair 与 no-miss-pair 梯度余弦分别为 0.0080 和 -0.0274，仅能说明近似正交，不能支持强对冲。
4. **CMIF 完全没有表达能力**：相同 CMIF、初始化、缓存和训练预算下，separable control 达到 320/335 factual 像素覆盖和 122/149 clean 像素跨零。
5. **只需事后增加训练步数**：v17 在冻结预算下已失败；即使末期损失仍下降，也不能追加步数为同一版本寻找正结果。

## 4. 被否定的 v18 候选：OMCO

候选坐标

\[
c=\frac{e_++e_-}{2},\qquad
d=\frac{e_--e_+}{2}
\]

满足

\[
T^\top T=\frac12I.
\]

当前 pair energy 对两个分量采用各向同性联合范数，因此 OMCO 仅改变径向尺度和极近零区域的权重，不提供新的误差方向、监督信息或机制。它与已经失败的 identity-joint control 实质等价，禁止进入实现和训练。

## 5. v18 唯一机制：PMOPE

v18 固定为：

**Paired Minimum-Margin Orthant Projection Energy，简称 PMOPE。**

CMIF 网络结构、参数量、输入边界与推理规则全部不变。只修改 pair objective。

对每个端点 \(\sigma\in\{+,-\}\)，令：

- \(\phi^\sigma\)：CMIF 输出的连续场；
- \(\psi^\sigma\)：现有严格非零的 signed-distance target field；
- \(V\)：有效区域；
- \(s^\sigma=\operatorname{sign}(\psi^\sigma)\in\{-1,+1\}\)；
- \(a=0.9\)，\(r=4\)；
- \(m_0=a/r=0.225\)：现有离散 signed-distance field 的最小非零量化单位。

定义 minimum-margin orthant violation：

\[
v^\sigma(x)
=
\mathbf 1_V(x)
\operatorname{ReLU}
\left[
m_0-s^\sigma(x)\phi^\sigma(x)
\right].
\]

pair loss 只计算一次现有联合 \(W^{1,4}\)/Sobolev energy：

\[
\mathcal L_{\mathrm{PMOPE}}
=
\mathcal E_{\mathrm{pair}}
\left(v^+,v^-\right).
\]

不增加 loss weight、margin 搜索、temperature、top-k、辅助头或新网络参数。

## 6. PMOPE 为什么针对本次失败

### 6.1 直接约束实际零层符号

推理使用：

\[
C^\sigma=(\phi^\sigma<0)\land\neg O^\sigma.
\]

PMOPE 在完整有效域约束两个端点的场符号。当 PMOPE 为零时：

\[
s^\sigma\phi^\sigma\ge m_0>0.
\]

因此在完整有效域上：

\[
(\phi^\sigma<0)
\Longleftrightarrow
(\psi^\sigma<0),
\]

从而直接保证场的零层符号集合与目标一致，并进一步保证 writable completion 与目标一致。

不能只在 \(\neg O^\sigma\) 上约束。否则 plus 端在已覆盖目标处可以保持负场，删除 occupancy 后仅通过解除遮挡显露已有负场，而无需学习 plus-to-minus 的真实状态响应。完整有效域约束要求 clean pair 在新增目标处满足：

\[
\phi^+\ge m_0,\qquad \phi^-\le-m_0,
\]

从而排除这种 shortcut。

### 6.2 Clean deletion 的梯度方向

在新增可写区域 \(A=O^+\setminus O^-\)：

- plus 端目标场为正，违反项保持或推动 \(\phi^+>0\)；
- minus 端目标场为负，违反项推动 \(\phi^-<0\)；
- 两个端点必须产生真实、带固定间隔的符号转换。

### 6.3 Component-null 的梯度方向

若删除区域不是目标，则 \(\psi^->0\)。任何负场或正间隔不足都会产生反向梯度，提高 \(\phi^-\)，阻止错误 completion。

同一个可行性机制同时处理“需要补全”“必须保持为空”和“必须产生真实端点响应”，不是多个模块。

### 6.4 与旧目标严格不等价

Identity、SORR 和 separable 都以一个精确 signed-distance field 为回归代表，其零损失集合基本是一个点。PMOPE 的零损失集合是：

\[
\mathcal K
=
\left\{
\phi:
s\phi\ge m_0
\text{ on all valid pixels}
\right\},
\]

即保持正确零层拓扑的可行锥。

正确越过间隔后，PMOPE 不再惩罚更大的同号场值，而旧回归目标仍有非零误差。因此 PMOPE 不是旧坐标旋转、常数缩放或模块叠加。

## 7. 保持不变的主线

v18 不修改：

- 通用冻结输入边界 \((F_b,O)\)；
- CMIF 场方程；
- 64 个 Base 特征通道；
- 16 个精确 occupancy phase；
- radius 2、width 32；
- 64,064 个训练参数；
- 单一 scalar completion field；
- fixed threshold 0；
- hard-union inference；
- factual/no-miss natural losses；
- optimizer、schedule 和 400-step 预算；
- 固定 `seed=42`；
- \(D_V/D_T\)、Full CURE 和其他 backbone 均保持未授权。

## 8. 分阶段实现

### Stage 1：核心 objective

新增：

1. PMOPE policy 与 violation 计算；
2. PMOPE pair-loss fields；
3. fused objective dispatch；
4. CMIF + PMOPE 固定训练入口。

核心测试必须证明：

- clean plus 梯度保持正场，minus 梯度严格推动负向跨零；
- component-null/background 梯度严格推动正向恢复；
- PMOPE 为零严格推出完整有效域零层符号及 writable completion 精确一致；
- 与 Identity、SORR、Separable 存在明确非等价反例；
- 没有新增模型参数、阈值或可调超参数；
- 所有三类 CMIF 参数获得有限梯度。

### Stage 2：dataset-free 与 \(D_R\) 最小门禁

只检查：

- 固定公式与 \(m_0=0.225\)；
- clean/component-null/identity-null 的梯度符号；
- raw zero-level sign set 在完整有效域内与目标符号严格等价；
- 应用 \(\neg O\) 后，实际 completion 与可写目标严格等价；
- 固定 seed 42 的确定性；
- 不读取 \(D_V/D_T\)，不训练。

门禁失败则停止，不运行 bounded-400。

### Stage 3：唯一 seed-42 bounded-400

只训练一个 PMOPE candidate；v17 的 SORR、Identity、Separable 结果作为已封存对照，不重复计算。

v18 authorization 必须显式绑定 v17 的 `COMPLETE.json`、`decision.json`、三份 checkpoint、source manifest 和 source archive 的路径、SHA256 与指纹。不能只读取一个历史 pass/fail 布尔值。

PMOPE 必须逐项通过：

- factual miss；
- factual no-miss；
- clean defined；
- clean compact；
- component-null；
- identity-null；
- diagnostic-null。

任一失败即冻结 v18，不追加步数、不修改 margin、不重跑。

### Stage 4：800 epoch

只有 PMOPE bounded-400 全部门禁通过后，才设计并执行一次固定 `seed=42` 的 800-epoch 真实性能验证。届时以 Pd、Fa、IoU、nIoU 和开销判断模型是否成功。

## 9. v17 冻结证据

```text
run:
runs/irstd1k_stage_a_seed42/
  cure_lite_cmif_v17_support_oriented_bounded_400_r1

COMPLETE fingerprint:
50a9963ae620dc7140deebf604a4344f78af5560af2a1737d58efb256070aeb0

source closure:
artifacts/source_closures/
  cure_lite_cmif_v17_support_oriented_bounded_400_50a9963ae620.tar

source archive SHA256:
4f333efe993151c36dcab80d83f266387d4a8e0ade0059b3a2538522a20f532c

source manifest:
artifacts/source_closures/
  cure_lite_cmif_v17_support_oriented_bounded_400_50a9963ae620.json

source manifest SHA256:
9d04e13ba781163e7114f33607d4e67f633eac9765ea8313676200fe1f906d98

archive content check:
40 / 40 paths and file SHA256 values exactly match
receipts/config.json:implementation.files
```

v17 的 40 个正式实现文件、运行收据、三份 checkpoint 和终态哈希已经闭合。共享代码可以进入 v18 演进，但不得改写或重签 v17 结果。
