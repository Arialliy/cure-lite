# CURE-Lite v18 正式负结果与 v19 USCOPE 方案

> 文档状态：v19 研究与实现规范  
> 当前证据：v18 PMOPE bounded-400 正式失败  
> v19 名称：**USCOPE — Uniform Sobolev–Chebyshev Orthant Projection Energy**  
> 中文名称：**统一 Sobolev–Chebyshev 正交域投影能量**  
> 重要决定：**此前提出的 RCMOPE 作废，不得实现或训练。**  
> 当前操作：USCOPE 核心、训练注册与 dataset-free 门禁已实现；真实
> \(D_R\) 门禁和 bounded-400 尚未运行。

---

## 1. 阶段结论

v18 PMOPE 已完成唯一一次固定 `seed=42` 的 bounded-400 正式运行。
训练和封存流程正常，但科学门禁失败：

| 项目 | v18 正式结果 |
|---|---:|
| factual miss 严格通过 | 11/16 |
| factual miss 至少找回 | 15/16 |
| factual no-miss | 通过 |
| clean full gate | 0/16 |
| clean target negative | 123/149 |
| clean target 漏像素 | 26 |
| clean 新增 completion | 170 |
| clean target 外 completion | 47 |
| component-null | 15/16 |
| identity-null | 通过 |

因此：

```text
v18 PMOPE bounded-400 = FAIL
Formal-800             = NOT AUTHORIZED
三数据集训练           = NOT AUTHORIZED
跨 backbone            = NOT AUTHORIZED
Full CURE              = NOT AUTHORIZED
```

该结果说明 PMOPE 能产生明显的 completion 响应，但在固定预算下无法保证
每个像素都进入正确零层一侧。v19 只检验一个更窄的结构性假设：

> PMOPE 的平均型风险会稀释少量关键像素；在保留同一全域 orthant
> projection residual 的前提下，同时加入其全局 Chebyshev 功率，能否使
> 有限训练预算直接控制最差像素。

这不是结果承诺。USCOPE 是否有效必须由预声明门禁和真实训练决定。

---

## 2. RCMOPE 正式作废

旧版 RCMOPE 曾计划引入：

- writable-only endpoint roles；
- persistent/added/released/near/far 等十类角色；
- response role；
- role 内 dense/max；
- role 间 mean/max。

该方案不再采用，原因如下。

### 2.1 它不是足够纯粹的单机制

虽然 RCMOPE 可以写成一个总 loss，但十类手工角色和两层风险聚合会使其
在科学上表现为多个规则的组合，而不是一个统一几何对象。

### 2.2 它过度贴合现有 evaluator

角色定义几乎逐项复制 clean、component-null 和 response-sign 门禁。
这会使训练目标与诊断项形成循环，也难以说明机制为何能推广到新的数据集
和新的 IRSTD 前端。

### 2.3 writable-only 约束会重新打开隐藏负场捷径

若 clean pair 的 added target 在 plus endpoint 被 occupancy 覆盖，
writable-only loss 不约束其 plus raw field。模型可以使：

\[
\phi^+<0,\qquad \phi^-<0,
\]

再利用 occupancy 删除显露已有负场，而不是学习真实的正负零层转换。
这与 CURE-Lite 希望从 \((F_b,O)\) 学习状态相关 completion 的主线不符。

### 2.4 同号 response 不应成为训练目标

当两个 endpoint 的目标符号相同且都位于正侧时，
\(\phi^--\phi^+\) 的连续数值次序不影响最终二值 completion。
强制拟合 signed-distance response 的次序既不是检测输出的必要条件，也可能
把背景场推近零点。

因此，v19 不继承 RCMOPE 的任何 role、weight、minimax 或 response loss。

---

## 3. CURE-Lite 主线保持不变

### 3.1 通用接口

\[
(F_b,O)\longmapsto\phi_\theta(F_b,O).
\]

其中：

- \(F_b\) 是任意 IRSTD 前端提供的冻结 Base feature；
- \(O\) 是该前端的 Base occupancy；
- CURE-Lite 只输出一个 scalar completion field；
- 模型输入不包含 GT、pair kind、target ID 或 sample ID。

### 3.2 CMIF 结构

CMIF 继续使用 centered mixed finite interaction：

\[
\phi_\theta(F,O)
=
a+
E_\theta(B,U)
-
E_\theta(0,U)
-
E_\theta(B,U_{\mathrm{mid},p})
+
E_\theta(0,U_{\mathrm{mid},p}),
\]

其中：

- \(B=N(F_b)\)；
- \(U\) 是 phase-preserving occupancy；
- \(U_{\mathrm{mid},p}\) 只将当前 phase coordinate 置为 \(0.5\)；
- \(a=0.9\)；
- 模型仍只有一个 scalar field。

固定结构继续为：

```text
feature channels    64
feature stride      4
phase channels      16
coarse radius       2
width               32
trainable params    64,064
```

### 3.3 推理

\[
Y_{\mathrm{completion}}
=
\mathbf1[\phi<0]\land\neg O,
\]

\[
Y_{\mathrm{final}}
=
O\lor Y_{\mathrm{completion}}.
\]

固定：

```text
one Base forward
one CMIF forward
residual threshold = 0
threshold search   = false
hard union
```

v19 不修改 CMIF、输入表示、参数量、自然分支、训练预算、阈值或推理图。

---

## 4. v18 PMOPE 的准确缺口

### 4.1 全域 orthant residual 本身是正确的

设两个 endpoint 为 \(\sigma\in\{+,-\}\)，定义：

- \(\phi_b^\sigma(x)\)：CMIF 输出；
- \(\psi_b^\sigma(x)\)：冻结 signed-distance target field；
- \(s_b^\sigma(x)=\operatorname{sign}\psi_b^\sigma(x)\in\{-1,+1\}\)；
- \(V_b\)：完整有效域；
- \(a=0.9\)、\(r=4\)；
- \(m_0=a/r=0.225\)。

v18 violation 为：

\[
q_b^\sigma(x)
=
\mathbf1_{V_b}(x)
\left[
m_0-s_b^\sigma(x)\phi_b^\sigma(x)
\right]_+.
\]

该定义必须原样保留：

1. 两个 endpoint 都参与；
2. 使用完整 \(V_b\)，不是 writable-only domain；
3. occupied pixels 仍受 raw-field 约束；
4. margin 固定为 \(0.225\)；
5. 不读取 pair kind。

完整有效域约束禁止模型在 occupied plus target 下隐藏负场，因此不能删除或
放宽。

### 4.2 缺口在有限预算下的风险几何

PMOPE 把 \((q^+,q^-)\) 放入一个加权 \(W^{1,4}\) 风险。该风险对稠密误差
和空间变化有意义，但一个大 stratum 中少量像素的测度可能很小。

若只有一个像素存在固定 violation \(c>0\)，其平均型 value power 会随该
像素测度下降。固定 400 步可以显著降低总体损失，却仍留下：

- 少量 target 像素未跨过零点；
- 少量 background 像素越过零点；
- 某个 component-null footprint 产生负场。

v19 不重新定义 target，不增加角色，只改变同一 projection residual 的
统一 gauge。

---

## 5. v19 USCOPE

## 5.1 名称

**USCOPE：Uniform Sobolev–Chebyshev Orthant Projection Energy**

- **Uniform**：所有 endpoint 和所有 valid pixels 使用同一个 residual，
  不划手工角色；
- **Sobolev**：保留稠密 value 与 spatial \(p=4\) 功率；
- **Chebyshev**：增加同一 residual 在 endpoint–pixel 乘积域上的全局
  supremum；
- **Orthant Projection**：目标仍是完整 margin orthant。

冻结 policy ID：

```text
uniform_sobolev_chebyshev_orthant_projection_energy_v1
```

## 5.2 同一个全域 violation

USCOPE 完全复用：

\[
q_b^\sigma(x)
=
\mathbf1_{V_b}(x)
\left[
m_0-s_b^\sigma(x)\phi_b^\sigma(x)
\right]_+.
\]

不增加第二种误差场。

## 5.3 Sobolev power

令现有 balanced measure 为 \(\mu_b(x)\)，并满足：

\[
\mu_b(x)>0,\quad x\in V_b,
\qquad
\sum_{x\in V_b}\mu_b(x)=1.
\]

value \(p=4\) power 保持为：

\[
P_{\mathrm{value},b}
=
\sum_{x\in V_b}
\mu_b(x)
\left[
\frac{
\left(q_b^+(x)\right)^2+
\left(q_b^-(x)\right)^2
}{2}
\right]^2.
\]

对 valid horizontal/vertical edge \(e\)，令
\(\Delta_e q^\sigma\) 为相邻像素差，\(\bar\mu_b(e)\) 为现有 edge measure。
spatial \(p=4\) power 保持为：

\[
P_{\mathrm{spatial},b}
=
\frac{
\sum_{e}
\bar\mu_b(e)
\left[
\frac{
\left(\Delta_e q_b^+\right)^2+
\left(\Delta_e q_b^-\right)^2
}{2}
\right]^2
}{
\sum_e\bar\mu_b(e)
}.
\]

定义：

\[
A_b
=
\frac12
\left(
P_{\mathrm{value},b}
+
P_{\mathrm{spatial},b}
\right).
\]

这部分与 v18 的稠密/空间风险语义一致。

## 5.4 Chebyshev power

将 endpoint 和 valid pixel 合成一个乘积域：

\[
\Omega_b
=
\{+,-\}\times V_b.
\]

定义全局最差 violation：

\[
\gamma_b
=
\max_{\sigma\in\{+,-\}}
\max_{x\in V_b}
q_b^\sigma(x).
\]

实现语义必须是对完整 \(q\) tensor 的全局 `amax`，不得：

- 先按 target/background 分组；
- 按 endpoint 分配权重；
- 使用 top-k；
- 使用 temperature；
- 使用 observed-value threshold。

定义 Chebyshev power：

\[
C_b=\gamma_b^4.
\]

## 5.5 最终统一能量

固定：

\[
\ell_b
=
\left[
\frac12
\left(
A_b+C_b
\right)
+
\varepsilon^4
\right]^{1/4}
-
\varepsilon,
\]

\[
\mathcal L_{\mathrm{USCOPE}}
=
\frac1B\sum_{b=1}^{B}\ell_b.
\]

其中 \(\varepsilon\) 完全复用现有 `norm_epsilon`。

最终公式中只有两个固定的对称平均：

1. value/spatial power 的 \(1/2\)；
2. Sobolev/Chebyshev power 的 \(1/2\)。

二者不搜索、不按数据调整，也不随 pair kind 改变。

---

## 6. 可证语义

## 6.1 USCOPE 是一个统一 projection gauge

定义完整 margin orthant：

\[
\mathcal K_{m_0}
=
\left\{
(\phi^+,\phi^-):
s^\sigma(x)\phi^\sigma(x)\ge m_0,\;
\forall(\sigma,x)\in\Omega_b
\right\}.
\]

\(q\) 是 field pair 对 \(\mathcal K_{m_0}\) 的逐坐标投影残差。
\(A_b\) 和 \(C_b\) 都只测量这一个残差：

- \(A_b\) 测量其稠密值和空间变化；
- \(C_b\) 测量其全局最差坐标。

因此 USCOPE 不是多个语义 loss 的拼接，而是同一 projection residual 的
统一 Sobolev–Chebyshev gauge。

## 6.2 零损失语义

由于：

\[
A_b\ge0,\qquad C_b\ge0,
\]

有：

\[
\ell_b=0
\iff
A_b=0\land C_b=0
\iff
\gamma_b=0
\iff
q_b^\sigma(x)=0,\quad\forall(\sigma,x)\in\Omega_b.
\]

因此：

\[
\ell_b=0
\Rightarrow
s_b^\sigma(x)\phi_b^\sigma(x)\ge m_0.
\]

## 6.3 有限预算证书

USCOPE 的核心不只在零损失。对任意 valid pixel：

\[
q_b^\sigma(x)\le\gamma_b,
\]

所以：

\[
s_b^\sigma(x)\phi_b^\sigma(x)
\ge
m_0-\gamma_b.
\]

若：

\[
\gamma_b<m_0,
\]

则：

\[
s_b^\sigma(x)\phi_b^\sigma(x)>0,
\quad
\forall(\sigma,x)\in\Omega_b.
\]

于是完整 \(V_b\) 上：

\[
\phi_b^\sigma(x)<0
\iff
\psi_b^\sigma(x)<0.
\]

这给出一个不要求 loss 精确为零的有限预算零层证书。

还可以从总 loss 得到保守上界：

\[
\gamma_b
\le
\left\{
2\left[
\left(\ell_b+\varepsilon\right)^4-\varepsilon^4
\right]
\right\}^{1/4},
\]

但正式运行必须直接记录 \(\gamma_b\)，不能只使用该间接上界。

## 6.4 单像素不会被稀释

若任意一个 endpoint 的任意一个 valid pixel 有 violation \(c>0\)，则：

\[
\gamma_b\ge c,\qquad C_b\ge c^4.
\]

从而：

\[
\ell_b
\ge
\left(
\frac12c^4+\varepsilon^4
\right)^{1/4}
-
\varepsilon.
\]

该下界不依赖：

- 图像面积；
- target 面积；
- background 面积；
- removed component 面积；
- 该像素在 balanced measure 中的质量。

## 6.5 hidden-negative 捷径被继续禁止

假设 plus endpoint 的某个 occupied pixel 目标符号为正，但：

\[
\phi^+(x)<0.
\]

则：

\[
q^+(x)
=
m_0-\phi^+(x)
>
m_0.
\]

因此：

\[
\gamma_b>m_0.
\]

即使该像素在 plus completion 中被 occupancy 遮挡，USCOPE 仍会检测并惩罚
隐藏负场。v19 不允许通过“删除 occupancy 后显露旧负场”完成 clean pair。

---

## 7. \(V\) 外的前提与边界

所有上述证明只在 \(V_b\) 内成立。因为：

\[
q^\sigma(x)=0,\quad x\notin V_b,
\]

USCOPE 不对 \(V_b\) 外的 field 符号提供保证。

所以每个 protocol 必须同时冻结并验证：

1. \(V_b\) 非空；
2. target 和 occupancy 不越出 \(V_b\)；
3. evaluator 的有效输出域与 \(V_b\) 一致，或推理显式限制在 \(V_b\)；
4. `invalid_completion_pixels = 0`；
5. \(V_b\) 外像素不得被用于支持 clean/component 成功结论。

不能用 \(V_b\) 内的 \(\gamma_b<m_0\) 推断整张未受约束图像都正确。

---

## 8. response-sign 的新状态

## 8.1 异号 endpoint

若：

\[
s^+(x)=+1,\qquad s^-(x)=-1,
\]

且 \(\gamma_b<m_0\)，则：

\[
\phi^+(x)>0,\qquad \phi^-(x)<0,
\]

所以：

\[
\phi^-(x)-\phi^+(x)<0.
\]

异号区域的 response direction 已由双端零层证书自动保证。

## 8.2 同号 endpoint

若两个 target signs 相同，USCOPE 只要求两端处于正确 orthant，不要求：

\[
\operatorname{sign}
\left(\phi^--\phi^+\right)
\]

复制 signed-distance magnitude 的次序。

原因是同号情况下该次序不改变最终 binary completion。

因此 v19 冻结为：

```text
sign-changing support response  由 endpoint certificate 蕴含
same-sign SDF response           仅诊断，不进入 loss
same-sign SDF response           不作为 bounded pass/fail 条件
```

该变化必须在 v19 训练前写入 authorization，且不得回头改变 v18 的正式
失败结论。

---

## 9. 为什么 USCOPE 不是模块堆叠

USCOPE 的算法改动只有：

```text
same q as v18
same V
same endpoint signs
same margin
same balanced measure
same p=4 value/spatial powers

新增：
同一个 q 的 global amax power
```

它没有：

- 新网络层；
- 新 head 或 branch；
- 新输入；
- target/background role；
- near/far role；
- released-component role；
- response loss；
- pair-kind 分支；
- Dice/BCE/focal loss；
- top-k、temperature 或可调权重；
- 阈值搜索或后处理。

USCOPE 与 PMOPE 具有相同的零集合，但有限子水平集不同。

设 \(N\) 个像素中只有一个像素 violation 为 \(c\)，其余为零。平均型风险
可以随该像素的 measure 缩小；USCOPE 的：

\[
C=c^4
\]

保持不变。这是 v19 唯一需要验证的非等价性。

---

## 10. 保持冻结的实验因素

v19 必须保持：

```text
input interface            (F_b, O)
model                     CMIF
parameter count           64,064
field count               1
feature channels          64
feature stride            4
coarse radius             2
width                     32
field amplitude           0.9
margin                    0.225
residual threshold        0
natural factual loss      unchanged
natural no-miss loss      unchanged
optimizer                 same as v18
population/cache          same frozen population
schedule                  same frozen schedule
bounded seed              42
bounded updates           400
retry/resume              false
D_V/D_T during gates      false
```

禁止在同一版本中：

- 修改 CMIF；
- 修改 natural loss；
- 修改 margin；
- 修改步数；
- 修改阈值；
- 增加 loss weight；
- 增加第二个 USCOPE candidate；
- 更换 seed 寻找通过结果；
- 从 v18 checkpoint warm-start；
- 读取新的 \(D_V\) 或 \(D_T\)；
- 接入其他 backbone。

---

## 11. 最小代码边界

当前实现保持以下最小代码边界。

建议新增：

```text
cure_lite/coverage_state_supremal_projection.py
cure_lite/experiment/coverage_state_uscope_dataset_free.py
cure_lite/experiment/coverage_state_uscope_dr_gate.py
cure_lite/experiment/coverage_state_uscope_decision.py
cure_lite/experiment/coverage_state_uscope_bounded_runner.py
tools/audit_coverage_state_uscope_v19.py
tools/run_coverage_state_cmif_uscope_bounded_400.py
```

建议核心 API：

```python
coverage_state_uscope_pair_loss_from_targets(
    field_plus,
    field_minus,
    targets,
    *,
    config,
    validate=True,
)
```

loss 不需要：

- occupancy；
- pair kind；
- evaluator output；
- v18 失败像素列表；
- role metadata。

建议返回并封存：

```text
loss
per_state_loss
value_power
spatial_power
sobolev_power A
chebyshev_violation gamma
chebyshev_power C
violation_plus
violation_minus
margin
integration_measure
valid_mask
```

v18 的代码、checkpoint、结果、decision、COMPLETE 和 source closure 均保持
历史只读。v19 使用新的 implementation closure。

---

## 12. 分阶段门禁

## 12.1 Stage 0：冻结 v18

必须验证：

```text
v18 decision        unchanged
v18 COMPLETE        unchanged
v18 checkpoints     unchanged
v18 source archive  unchanged
v18 result remains  FAIL
```

RCMOPE 文本被本方案取代，不保留任何训练授权。

## 12.2 Stage 1：dataset-free

不读取任何 runtime split，不训练。

必须证明：

1. `q_plus/q_minus` 与 v18 full-valid PMOPE 定义逐元素一致；
2. \(m_0=0.225\)；
3. value/spatial power 与冻结定义一致；
4. \(\gamma\) 是 endpoint–pixel 乘积域的联合 supremum；实现可等价地
   对两个 endpoint 分别 `amax` 后取 `maximum`；
5. \(C=\gamma^4\)；
6. 最终 loss 严格等于冻结公式；
7. loss 为零当且仅当完整 margin orthant 满足；
8. \(\gamma<m_0\) 推出 raw zero-level sign 完全正确；
9. 一个错误 target pixel 不被 65,535 个正确像素稀释；
10. 一个错误 background pixel 不被 65,535 个正确像素稀释；
11. target violation 的下降方向使 field 变负；
12. background violation 的下降方向使 field 变正；
13. occupied hidden-negative 必须产生 \(\gamma>m_0\)；
14. \(V\) 外不作保证并被显式审计；
15. 无 role、pair-kind、top-k、temperature 或 threshold search；
16. 所有张量和梯度有限；
17. 不访问 \(D_R/D_V/D_T\)；
18. 不构造 optimizer，不执行 step。

dataset-free 失败则停止，不进入 \(D_R\)。

## 12.3 Stage 2：真实 \(D_R\) preflight

只读取现有冻结 \(D_R\) population/cache，不训练。

必须检查：

```text
target fields strictly nonzero on V
V nonempty and evaluator-aligned
target/occupancy inside V
q covers both endpoints and all V pixels
balanced measure mass = 1
global amax semantics exact
clean target violation gradient direction correct
clean outside/background gradient direction correct
component-null footprint gradient direction correct
hidden-negative construction rejected
model state unchanged
optimizer steps = 0
D_V/D_T accessed = false
```

sealed v18 的 26 个漏像素和 47 个越界像素可以作为只读 attribution 输出，
但不得成为 USCOPE 的输入、role 或数据依赖，也不得用于调参。

\(D_R\) gate 失败则停止，不创建 bounded authorization。

## 12.4 Stage 3：唯一 bounded-400

只运行：

```text
candidate  USCOPE
seed       42
updates    400
epochs     10
steps      40 / epoch
```

必须与 v18 绑定：

```text
same CMIF class/config/parameters
same update-0 model fingerprint
same optimizer
same cache/population
same schedule
same natural branches
same evaluator except predeclared same-sign response status
```

每个 pair 必须记录：

```text
gamma
gamma_plus
gamma_minus
worst endpoint
worst pixel coordinate
worst target sign
gamma < m0 certificate
raw sign-error pixel count
```

正式 pass 条件如下。

### Natural gates

```text
factual miss strict          16 / 16
factual target negative      each >= 95%
factual recovered            16 / 16
factual no-miss              16 / 16
invalid completion           0
```

### Clean gates

```text
USCOPE clean compact gate    16 / 16
added target negative        149 / 149
new completion pixels        149
outside added target         0
compact exact                16 / 16
component match              16 / 16
plus false islands           0
invalid completion           0
pair gamma < m0              every clean pair
```

`same-sign response-sign` 只报告，不影响 pass/fail。

### Null gates

```text
component-null               16 / 16
identity-null                16 / 16
diagnostic-null              PASS
removed footprint negative   0
new negative components      0
invalid completion           0
pair gamma < m0              every component-null pair
```

### Engineering gates

```text
optimizer updates            400 / 400
backward calls               400
optimizer steps              400
finite state audits          401
final checkpoint != init     true
3 parameter tensors active   true
single terminal              true
retry/resume                 false
D_V accessed                 false
D_T accessed                 false
Formal-800 executed          false
```

任一必需项失败即：

```text
V19_USCOPE_BOUNDED_400_FAIL
```

随后冻结，不增加 step、不改 \(1/2\)、不换 seed、不调 margin。

只有全部通过才允许：

```text
V19_USCOPE_BOUNDED_400_PASS
```

## 12.5 Stage 4：Formal-800

只有 bounded-400 全部门禁通过，才允许单独设计和签发 Formal-800
authorization。

Formal-800 仍固定：

```text
seed = 42
epochs = 800
no threshold search
no v19 hyperparameter change
```

Formal-800 才开始判断 Pd、Fa、IoU、nIoU 和开销。bounded-400 只证明
结构候选具有进入真实性能验证的资格，不证明性能成功。

三大数据集和跨 backbone 验证必须使用后续独立冻结协议，且性能实验均固定
`seed=42`。因此不能声称多 seed 统计稳定性。

---

## 13. 正确执行顺序

```text
freeze v18 negative result
        │
        ▼
implement one USCOPE core
        │
        ▼
dataset-free algebra / gradient / V-boundary gates
        │
        ├── FAIL → freeze v19 implementation
        ▼
real D_R preflight, no training
        │
        ├── FAIL → no training authorization
        ▼
single seed-42 bounded-400
        │
        ├── FAIL → freeze v19; no extra step/seed/tuning
        ▼
separate Formal-800 authorization
        │
        ▼
single seed-42 Formal-800
        │
        ▼
only after CURE-Lite success:
three datasets (fixed seed-42) → other IRSTD frontends
```

在 CURE-Lite 本体成功前，不设计 Full CURE。

---

## 14. 失败后的决策

## 14.1 clean/outside 仍失败

若 bounded 训练后仍有 target 漏像素或 outside 像素，并且这些像素就是
USCOPE 的最大 violation，则说明：

> 全局最差像素已经被目标直接看到，但 CMIF、自然分支和固定优化预算仍无法
> 同时满足这些约束。

此时冻结 USCOPE。下一版本才允许审查 CMIF 表示或训练目标兼容性，不能给
USCOPE 增加 role、top-k 或额外权重。

## 14.2 clean 通过但 factual 不足

若：

```text
clean exact       16 / 16
component-null    16 / 16
factual strict    < 16 / 16
```

则说明 pair-risk dilution 已被解决，但 factual 问题属于：

- natural-state objective；
- factual feature support；
- CMIF 对自然漏检状态的表示能力。

下一版本不得继续修改 pair objective。

## 14.3 \(\gamma\) 下降但未低于 \(m_0\)

这只说明最差 violation 被降低，不能生成零层正确证书。不得把
“接近 \(m_0\)”解释为结构成功，也不得事后延长 400 步。

## 14.4 bounded 全部通过

这只说明 USCOPE 具有进入 Formal-800 的资格，不证明：

- 三数据集性能；
- 多 seed 稳定性；
- 跨 backbone 泛化；
- ICLR 创新性；
- Full CURE 已成立。

---

## 15. ICLR 研究叙事

## 15.1 核心问题

红外小目标 completion 由严格零层决定，但常用平均风险可以在有限训练预算下
隐藏少数关键像素。对一像素目标，一个错误像素就是整个目标失败；对背景，
一个错误像素可能成为新的虚假 component。

## 15.2 核心洞见

> 对同一个 full-domain orthant projection residual，同时保留 Sobolev
> 稠密/空间功率与 Chebyshev 最差坐标功率，可以把平均场优化与逐像素零层
> 证书放入一个统一能量，而不引入角色规则或后处理。

## 15.3 最诚实的创新类型

当前可讨论的是：

- 面向隐式 completion field 的新风险几何；
- 有限预算下从全局 violation 到零层正确性的可检查证书；
- 与 CMIF 混合差分场的结构性组合。

不能把以下内容写成已证明贡献：

- SOTA 性能；
- 多数据集泛化；
- backbone-independent 实证；
- 稳定提升；
- 新颖性已经确认。

## 15.4 非堆叠叙事

论文叙事必须围绕一个对象。对逐坐标 margin-orthant projection，有：

\[
\Pi_{\mathcal K_{m_0}}(\phi)-\phi=s\,q,
\]

因此 \(q\) 与真实投影修正只相差逐坐标符号，是等范数的同一 projection
residual。

以及一个统一 gauge：

\[
\left[
\frac12
\left(
\|q\|_{W^{1,4}_\mu}^4+
\|q\|_\infty^4
\right)
+
\varepsilon^4
\right]^{1/4}
-
\varepsilon.
\]

不能重新描述成：

- Sobolev 模块；
- max 模块；
- target 模块；
- background 模块；
- response 模块。

它们不是网络组件，而是同一 projection residual 的统一测度。

---

## 16. 最小证据包

### 16.1 机制证据

1. 单像素 violation 不随图像面积衰减；
2. \(\gamma<m_0\) 严格推出零层符号正确；
3. full-domain 约束排除 hidden-negative；
4. PMOPE 与 USCOPE 具有相同零集合、不同有限子水平集；
5. same-sign response 不是 binary completion 的必要条件。

### 16.2 工程证据

1. 与 v18 相同模型、初始化、缓存、schedule；
2. 唯一变化是 pair gauge；
3. 完整 source closure；
4. 确定性重放；
5. \(D_V/D_T\) 隔离；
6. 单 terminal、无 retry/resume。

### 16.3 后续性能证据

只有通过 bounded 后才要求：

1. Formal-800 的 Pd、Fa、IoU、nIoU；
2. 三个公开数据集；
3. 三个数据集均固定 `seed=42`，并明确不作多 seed 稳定性主张；
4. 与无 CURE-Lite 的相同前端公平比较；
5. 参数、显存、训练和推理开销；
6. 小目标尺寸分层和失败案例；
7. 最终才讨论跨 IRSTD 前端。

---

## 17. 风险与证据边界

### 17.1 优化风险

global `amax` 非光滑，梯度可能集中在当前最差像素并发生 worst-pixel
hopping。Sobolev power 提供稠密和空间梯度，但不能保证固定 400 步一定
收敛。

### 17.2 固定 \(1/2\) 的风险

Sobolev/Chebyshev 的对称平均是预声明结构选择，不是理论唯一选择。
v19 禁止在 \(D_R\) 上搜索该系数。若失败，不得调系数救回同一版本。

### 17.3 factual 风险

USCOPE 只修改 pair objective。factual natural loss 不变，因此 clean
compactness 改善不保证 factual strict 从 11/16 提升到 16/16。

### 17.4 数据与协议风险

当前设计来自一个固定 \(D_R\) population 和一次 seed-42 负结果。
bounded 通过仍可能是开发集适配，必须经过后续三数据集固定
`seed=42` 验证；该协议不能建立多 seed 稳定性。

### 17.5 新颖性风险

Chebyshev/max risk、Sobolev risk 和 margin projection 均有广泛相关思想。
USCOPE 的 ICLR 新颖性不能仅凭名称或当前文档确认。

正式状态必须写为：

```text
novelty_status = NEEDS_SEARCH
closest_work_status = NOT YET VERIFIED
ICLR_readiness = NOT ESTABLISHED
```

在提出论文级创新主张前，需要进行公开、安全的相关工作检索，确认
“CMIF mixed interaction + full-domain orthant projection +
finite-budget Sobolev–Chebyshev certificate”的组合及理论增量是否已有直接
先例。

---

## 18. 最终决定

当前唯一允许的 v19 方向是：

> **CURE-Lite CMIF + USCOPE**

其科学修改严格限定为：

\[
\text{PMOPE 的同一 full-valid orthant violation}
\quad+\quad
\text{统一 global Chebyshev power}.
\]

RCMOPE 已作废。不得实现其角色划分、writable-only loss、response loss 或
role minimax。

在 USCOPE 的 dataset-free、真实 \(D_R\) 和唯一 seed-42 bounded-400 全部
通过前：

```text
Formal-800        NOT AUTHORIZED
三大数据集        NOT AUTHORIZED
跨 backbone       NOT AUTHORIZED
Full CURE         NOT AUTHORIZED
性能结论          NOT SUPPORTED
创新性结论        NEEDS SEARCH
```

本文件完成的是方案冻结，不是模型成功证明，也不包含任何未运行结果。
