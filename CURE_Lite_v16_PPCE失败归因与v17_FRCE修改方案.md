# CURE-Lite v16 PPCE 失败归因与 v17 CMIF 修改方案

> 文档状态：已纠正原 FRCE 提案  
> v16 状态：冻结的 bounded-400 正式负结果  
> FRCE 状态：`FRCE_V17_P0_FAIL / NOT_IMPLEMENTABLE / NOT_AUTHORIZED`  
> v17 候选：**CMIF-CSLF-v17（Centered Mixed Interaction Field）**  
> 当前授权：只允许修改方案、实现核心网络并运行 dataset-free 与冻结 \(D_R\) 的训练前门禁  
> 当前禁止：bounded-400 训练、Formal800、\(D_V/D_T\)、Full CURE、其他 IRSTD backbone

---

## 0. 结论

v16 PPCE 的负结果不能通过继续调训练步数、阈值或损失权重修复。它表明：

1. exact phase occupancy 确实被模型使用；
2. 但 PPCE 允许 occupancy 独立写入场；
3. synthetic deletion 因而成为比自然漏检更容易利用的状态线索；
4. 模型能够恢复被删除目标，却不能同时保证 factual、compact 与 outside gates。

原文提出的 FRCE：

\[
\phi=a+\eta(O)\rho(F)
\]

不能作为 v17。它虽然限制了 occupancy-only 路径，却把每个像素的两个 occupancy 端点压缩为同一个 \(\rho(F)\) 的比例缩放。对于目标端点 \(t^+,t^-\)，必须满足：

\[
\eta^-(t^+-a)=\eta^+(t^--a).
\]

这个必要条件在冻结 \(D_R\) 上大规模不成立。因此 FRCE 的失败发生在训练之前，是状态方程不可表达，不是优化不充分。

v17 改为 CMIF：

\[
\boxed{
\phi
=
a+
\Delta_{\text{feature presence}}
\Delta_{\text{center phase relative to }1/2}
\mathcal E_\theta(B,U)
}
\]

它不是新增多个模块，而是重新定义唯一 completion field 的状态方程：

- feature-only 项严格消去；
- occupancy-only 项严格消去；
- 只有 feature 与 coverage phase 的混合交互能写入场；
- 两个 occupancy 端点不再受 FRCE 的一维比例约束；
- 外部输入仍只有通用的冻结 \((F_b,O)\)；
- 输出仍只有一个标量场和固定零水平集。

v17 只改变场方程。原 SORR、identity、separable、训练预算、阈值、hard union 与 evaluator 全部保持不变。CZB 不进入 v17。

---

## 1. 冻结的 v16 事实

权威运行：

```text
runs/irstd1k_stage_a_seed42/
  cure_lite_cslf_v16_ppce_support_oriented_bounded_400_r1
```

正式决定：

```text
BOUNDED_PPCE_SUPPORT_ORIENTED_CSLF_GATE_FAIL
```

关键结果：

| 指标 | v16 |
|---|---:|
| factual gate | 15 / 16 |
| factual recovered | 16 / 16 |
| factual target negative | 315 / 335 |
| XDU680 | 57 / 75 |
| factual no-miss | 16 / 16 |
| clean response | 1395 / 1396 |
| clean added-target negative | 149 / 149 |
| compact exact | 0 / 16 |
| component match | 16 / 16 |
| clean full | 0 / 16 |
| outside completion pixels | 88 |
| false islands | 2 |
| trainable component-null | 16 / 16 |
| identity-null | 16 / 16 |

phase evaluator v3 修正后的 diagnostic-null 语义为通过；它不改变 v16 在 factual、compact、outside 三个关键门禁上的失败。

必须保留以下解释边界：

- v16 证明 PPCE 存在强 occupancy-role signal；
- 它没有证明 occupancy 是唯一失败原因；
- 它没有证明 frozen feature 的 factual/synthetic 强弱差距已经解决；
- 它没有授权修改 \(D_V/D_T\) 或开始 Full CURE。

---

## 2. 为什么原 FRCE 必须撤销

### 2.1 双端表达不成立

FRCE 对同一个像素定义：

\[
\phi^+=a+\eta^+\rho,\qquad
\phi^-=a+\eta^-\rho.
\]

两个目标端点要求同一个 \(\rho\) 同时满足：

\[
\rho=\frac{t^+-a}{\eta^+}
=\frac{t^--a}{\eta^-}.
\]

冻结全 \(D_R\) 的只读审计得到：

- clean pairs：206；
- response pixels：19,722；
- \(\Delta\eta=0\) 的 response pixels：111；
- 违反双端必要条件：17,030 / 19,722；
- plus 端点为 \(0.9\)、但 minus 端点要求非零响应的主要冲突：17,025；
- added-target pixels：2,551；
- added-target 双端不相容：450。

因此扩大 exposure radius、增加训练步数或修改初始化都不能修复 FRCE。

### 2.2 zero-feature 静默证明不成立

原 skeleton 使用带可训练 bias 的 release projection。即使初始化 bias 为零，训练后仍可能出现：

\[
\rho(0)=b,\qquad
\phi=a+\eta(O)b.
\]

这重新引入 occupancy-dependent zero-feature response。设置 `bias=False` 只能修复这一处，不能解决双端不可达。

### 2.3 与已停止的可分离家族重复

FRCE 仍属于：

\[
A(F)+B(F)g(O)
\]

的可分离状态方程。此前 SVEF/CCFR 路线已对同类结构给出正式负结果与停止决定。仅把 vacancy gate 换成 distance exposure 不构成新的核心机制。

### 2.4 原代码草案不能直接运行

原草案还存在两项确定性实现错误：

1. 子 config 覆盖 `field_policy`，但父类 `__post_init__` 固定旧 policy；
2. 子模型绕过父构造后继续调用父 `_validate_inputs`，会访问不存在的 `input_projection`。

### 2.5 FRCE 与 CZB 同时修改破坏归因

同时改变网络方程和训练目标，无法判断结果来自状态结构还是 barrier。v17 必须先只检验新状态方程；CZB 若以后仍有必要，必须作为单独版本与独立因子实验。

---

## 3. CURE 主线保持不变

CMIF 仍满足 CURE-Lite 的固定边界：

```text
任意冻结 IRSTD detector
          │
          ├── frozen feature F_b
          └── frozen Base occupancy O
                         │
                         ▼
                  CMIF-CSLF-v17
                         │
                         ▼
             one scalar completion field φ
                         │
                         ▼
              C = (φ < 0) ∧ ¬O
                         │
                         ▼
                    Y = O ∨ C
```

不增加：

- 新 detector head；
- 多分支 prediction；
- attention stack；
- learned threshold；
- 后处理搜索；
- GT-dependent inference；
- backbone 内部专用接口。

创新点不是“再加一个模块”，而是一个受约束的状态交互原理：

> completion 只能由 frozen feature evidence 与当前 coverage state 的混合交互产生，任一变量单独存在都不能写入 completion field。

---

## 4. v17 状态坐标

### 4.1 feature 坐标保持冻结

v17 不同时修改 feature normalization。继续使用现有 sample-global RMS：

\[
B(F_b)=
\frac{\operatorname{stopgrad}(F_b)}
{\max(\operatorname{RMS}(F_b),10^{-6})}.
\]

这样 v16 到 v17 的唯一主要变化是场方程，而不是“新方程 + 新特征变换”。

这不代表 synthetic strong-feature 与 factual weak-feature 差距已解决。该问题必须由后续 factual/clean gates 判定，不能从结构上直接宣称解决。

### 4.2 exact phase occupancy

\[
U=\operatorname{PixelUnshuffle}_s(O)
\in\{0,1\}^{B\times s^2\times h\times w}.
\]

phase 顺序继续使用 PixelShuffle 的 row-major 约定，并保持精确可逆。

### 4.3 固定局部范围

coarse radius 固定：

\[
r_c=2,\qquad k=2r_c+1=5.
\]

每个输出 coarse cell 只读取 \(5\times5\) 的 joint \((B,U)\) patch。radius 不是搜索参数。

---

## 5. 唯一局部标量能量

对 coarse cell \(u\)：

\[
q_u(B,U)
=
W*\mathcal P_u^{5\times5}[B,U]+b.
\]

运行时只把同一个参数张量 \(W\) 视作 feature/occupancy 两个输入切片：

\[
W=[W_F,W_U],
\qquad
q_u=W_F*B+W_U*U+b.
\]

\[
\mathcal E_{\theta,u}(B,U)
=
v^\top\operatorname{SiLU}(q_u(B,U)).
\]

这里只有一个 joint affine state 和一个共享 scalar energy：

- \(W_F\) 与 \(W_U\) 不是两条分支，而是同一个
  `joint_state_weight` 的两个 view；
- 只有一个 hidden state；
- 只有一个 energy vector \(v\)；
- 没有 energy-output bias；
- 没有 feature head、occupancy head、gate head 或 phase head。

---

## 6. 中心 phase 的固定中性状态

对每个 coarse 位置 \(u\) 与输出 phase \(p\)，定义：

\[
U^{(u,p\leftarrow1/2)}.
\]

它只执行一项操作：

\[
U_{u,p}\leftarrow\frac12.
\]

其余所有 phase、所有其他 coarse cell 完全保持实际 occupancy。

必须明确：

- \(1/2\) 是固定解析中点；
- 它不是 learned 参数；
- 它不依赖 \(D_R\)、配对关系或总体统计；
- 它不改变外部缓存与推理输入；
- 不能错误地把整个 phase-\(p\) 平面同时设为 \(1/2\)。

---

## 7. CMIF 单场方程

固定：

\[
a=0.9.
\]

定义混合交互：

\[
\chi_{u,p}
=
\mathcal E_u(B,U)
-\mathcal E_u(0,U)
-\mathcal E_u(B,U^{(u,p\leftarrow1/2)})
+\mathcal E_u(0,U^{(u,p\leftarrow1/2)}).
\]

最终 native phase field：

\[
\boxed{\phi_{u,p}=a+\chi_{u,p}}.
\]

经 PixelShuffle 得到全分辨率单场：

\[
\phi=\operatorname{PixelShuffle}_s(\phi_{\text{native}}).
\]

推理保持：

\[
C=(\phi<0)\land\neg O,
\qquad
Y=O\lor C.
\]

---

## 8. 高效等价实现

不允许为每个像素运行一次完整反事实网络。

先计算：

\[
q_u^0=W_U*U+b,
\qquad
f_u=W_F*B,
\qquad
q_u=q_u^0+f_u.
\]

令 \(w_p\) 为 occupancy kernel 的中心 phase 列：

\[
w_p=W_U[:,p,r_c,r_c].
\]

中性替换只造成：

\[
\delta_{u,p}
=
\left(\frac12-U_{u,p}\right)w_p.
\]

采用以下分组，保证 \(B=0\) 时精确消去：

\[
d_u
=
\operatorname{SiLU}(q_u)
-\operatorname{SiLU}(q_u^0),
\]

\[
d^{1/2}_{u,p}
=
\operatorname{SiLU}(q_u+\delta_{u,p})
-\operatorname{SiLU}(q_u^0+\delta_{u,p}),
\]

\[
\boxed{
\chi_{u,p}
=
v^\top(d_u-d^{1/2}_{u,p})
}.
\]

这一实现需要：

- 一次 \(5\times5\) feature projection；
- 一次 \(5\times5\) occupancy projection；
- 一次按 phase 广播的中心列更新；
- 一个共享 energy vector。

代码 receipt 必须明确内部 energy evaluation 数与实际 FLOPs/显存，不能只报告“一个 model forward”而隐藏内部交互计算。

---

## 9. 可证明的结构性质

### 9.1 zero-feature 精确静默

若 \(B=0\)，则 \(q=q^0\)：

\[
d=0,\qquad d^{1/2}=0,
\]

\[
\boxed{\phi\equiv0.9}.
\]

该性质与 occupancy、hidden bias、energy weights 和 checkpoint 状态无关。

### 9.2 additive pure-feature 路径消去

任意只依赖 \(B\) 的可加项 \(g(B)\) 满足：

\[
g(B)-g(0)-g(B)+g(0)=0.
\]

### 9.3 additive pure-occupancy 路径消去

任意只依赖 \(U\) 的可加项 \(h(U)\) 满足：

\[
h(U)-h(U)-h(U^{1/2})+h(U^{1/2})=0.
\]

因此允许的正式主张是：

```text
pure_feature_additive_path_absent = true
pure_occupancy_additive_path_absent = true
```

不能在训练前宣称：

```text
synthetic_feature_gap_solved = true
```

### 9.4 常数与 energy-output bias 消去

常数项在四角差分中不可辨识，梯度恒为零，所以实现中禁止 energy-output bias。

### 9.5 只有 joint interaction 能写场

若 \(W_F=0\) 或 \(W_U=0\)，均有：

\[
\chi=0,\qquad\phi=0.9.
\]

### 9.6 双端不受反对称或比例约束

单 hidden unit 的解析 witness：

\[
f=1,\quad w=1,\quad b=0,\quad v=1
\]

产生：

\[
\chi(0)\approx-0.1840734701,
\qquad
\chi(1)\approx+0.1154035286.
\]

两者之和非零，因此不是强制 flip 反对称。

第二个 hidden unit：

\[
f=1,\quad w=2,\quad b=-0.5
\]

与第一个 unit 的两个端点向量组成的矩阵行列式约为：

\[
0.0168249123\neq0.
\]

因此两个 hidden units 即可形成 rank-2 endpoint basis；CMIF 不存在 FRCE 的 rank-1 比例限制。

### 9.7 固定局部性

\[
\frac{\partial\phi_u}{\partial B_v}
=
\frac{\partial\phi_u}{\partial U_v}
=0,
\qquad
\|u-v\|_\infty>2.
\]

full-grid 最大 phase-to-phase Chebyshev 作用距离为：

\[
r_{\text{full}}=2s+(s-1)=3s-1.
\]

当 \(s=4\)：

\[
r_{\text{full}}=11.
\]

这是 joint energy 的真实 Jacobian support，不是额外构造的 exposure mask。

严格结论只能是：PPCE 的多级传播被替换为一个 radius-2 有界
stencil，radius 2 外影响精确为零。radius 2 内仍可能形成宽响应；
compactness 是否改善必须由后续门禁证明，不能在训练前宣称“扩散已经
解决”。

---

## 10. 固定结构与参数量

正式 v17 配置：

```text
feature_channels = 64
feature_stride   = 4
phase_channels   = 16
coarse_radius    = 2
kernel_size      = 5
width            = 32
field_amplitude  = 0.9
neutral_phase    = 0.5
```

参数量：

\[
N
=
w(C+s^2)k^2+2w.
\]

代入 \(C=64,s=4,k=5,w=32\)：

\[
\boxed{N_{\rm CMIF}=64,064}.
\]

仅有三个可训练张量：

```text
joint_state_weight
joint_hidden_bias
scalar_energy_weight
```

参数量增加来自冻结的 radius-2 joint patch，不来自叠加 prediction heads。64,064 仍远小于常规 IRSTD backbone。

初始化：

- joint state weight 使用冻结初始化规则；
- hidden bias 初始化为 0；
- scalar energy weight 初始化为 0；
- update 0 的 field 精确为 \(+0.9\)；
- update 0 的 completion 为空；
- scalar energy weight 在首个有效 update 获得梯度；
- joint state 参数必须在后续 update 获得有限非零梯度。

---

## 11. v17 不修改的内容

以下内容必须保持 v16 冻结版本：

- raw catalog；
- geometry-safe lineage；
- frozen \(D_R\) cache；
- scene-complete continuous SDF；
- SORR candidate；
- identity-joint control；
- separable-endpoint control；
- factual miss/no-miss branches；
- 400-update bounded schedule；
- optimizer 与 seed；
- threshold \(0\)；
- hard union；
- zero-level evaluator；
- \(D_V/D_T\) 访问规则。

CZB、binary target、morphology、connected-component crop、额外 background weight、learned threshold 均不进入 v17。

---

## 12. dataset-free 门禁

任何一项失败，均不得读取真实 \(D_R\) tensor。

### G1：reference 与高效实现等价

逐 cell、逐 phase 显式构造 \(U^{(u,p\leftarrow1/2)}\)，与中心 kernel-column 高效式逐元素比较。

### G2：exact phase contract

\[
\operatorname{PixelShuffle}
(\operatorname{PixelUnshuffle}(O))=O.
\]

逐一验证 16 个 phase 的 row-major 位置。

### G3：center-only neutralization

只允许当前 cell 的当前 phase 变为 \(0.5\)；其他 phase 和邻居 cell 不得改变。

### G4：zero-feature exact silence

在 empty、single-phase、dense、multi-component 与 random occupancy 上：

\[
F=0\Rightarrow\phi\equiv0.9.
\]

要求 exact equality，不只要求 completion 为空。

### G5：pure-path annihilation

分别设置 \(W_F=0\) 与 \(W_U=0\)，同时保持 \(v\neq0\)，两种情况下均要求 \(\phi\equiv0.9\)。

### G6：energy gauge invariance

feature-only、occupancy-only 与常数 additive gauge 不得改变 CMIF field。

### G7：endpoint asymmetry 与 rank

冻结单-unit asymmetry witness 与双-unit rank-2 witness；不能只检查“两个值不同”。

### G8：radius-2 exact locality

分别翻转 feature coarse cell 与 occupancy phase bit：

- radius 2 外变化精确为 0；
- radius 1 与 radius 2 均存在可构造的非零 witness；
- 邻居 occupancy phase 能影响当前输出。

### G9：mixed interaction nondegeneracy

\[
\Delta_B\Delta_U\mathcal E\neq0
\]

必须可构造。该 probe 人工设置 \(v\neq0\)，要求三个参数张量均可获得
有限非零梯度；禁止实现退化为线性可加模型。它不等同于真实零初始化的
update-0 梯度门禁。

### G10：null exactness

当 \((F^+,O^+)=(F^-,O^-)\) 时：

\[
\phi^+=\phi^-,
\qquad
\Delta\phi=0.
\]

### G11：初始化与数值

- field 全 \(+0.9\)；
- completion 为空；
- forward/backward 有限；
- 真实初始化的 update 0 只要求 scalar energy weight 获得非零梯度；
- 第一次 optimizer step 后，joint weight 与 hidden bias 必须在
  update 1 或 2 之前获得非零梯度；
- 不得把 update 0 的上游零梯度误判为永久 dead parameter；
- 两次固定输入重放逐字节一致。

### G12：结构 receipt

必须固化 config、参数形状、参数量、policy、方程版本、reference/effective compute ledger、无辅助输出和无 learned threshold 等事实。

---

## 13. 冻结 \(D_R\) 的训练前 P0

dataset-free 全部通过后，才允许只读构造冻结 \(D_R\) receipt。

### P0-A：全人口 response reachability

对 206 个 clean pairs 的每个非零 SORR response pixel，计算其 coarse output cell 到 removed occupancy 支撑的最小 Chebyshev 距离。

门槛：

\[
\text{reachable}=19,722/19,722,
\qquad
\text{unreachable}=0.
\]

历史只读计算只能作为线索；v17 必须由新代码重新生成指纹化 receipt。

### P0-B：endpoint key consistency

固定 local key：

\[
(B_{5\times5},U_{5\times5},p).
\]

同一 key 不得对应冲突的有效 target endpoint。

审计域必须与实际 objective 对齐：

- natural factual-miss / factual-no-miss 使用各自的
  `loss_valid_mask`；
- clean-positive 与正式 component-null 使用
  `joint_targets.valid_mask`，并分别审计 plus/minus 两端；
- `identity_null` 的 optimizer exposure 为 0，只绑定其身份清单，
  不把它加入 endpoint 训练约束；
- diagnostic-only component 不得混入正式 component-null 人口。

每个 endpoint 的有效域必须被以下三层互斥且完备地划分：

```text
target     = binary target
ring       = valid & ~target & (target_field != +0.9)
background = valid & (target_field == +0.9)
```

必须分别报告全人口与 bounded 16-role population 中
`role × endpoint × stratum` 的 observation 数、冲突数和会计恒等式。
由于 background 的 target value 恒为 \(+0.9\)，允许采用严格的两遍
流式实现：先建立 target/ring exact keys，再完整扫描 background，
只对命中 active occupancy-key 候选的背景构造完整 feature key。必须
同时记录 `background_total`、`background_scanned`、候选数和 exact
active-key 命中数，且 `background_total == background_scanned`。

exact key 的数学分组必须直接使用完整 tuple，不得把二次 SHA 摘要
当作数学 key。SHA256 只能作为索引；摘要重复时必须回到原始
dtype、shape 和 bytes 做精确确认。样本 ID、source ID 和空间坐标
均不得进入 key。

### P0-C：transition key consistency

固定 transition key：

\[
(B_{5\times5},U^+_{5\times5},U^-_{5\times5},p).
\]

同一 key 不得同时要求：

- clean nonzero response；
- clean zero response；
- component-null zero response。

P0-C 只遍历 \(U^+_{5\times5}\neq U^-_{5\times5}\) 的 radius-2
可影响区域；该区域外 CMIF 的 response 解析为零，而非零 target
response 是否越界由 P0-A 单独阻断。

### P0-D：feature witness 与表示下界

必须报告：

- endpoint target/ring 位置的 \(B\) patch 是否 exact zero，并按
  factual、clean±、component± 分层；
- transition response-core 与 response-ring 位置的 \(B\) patch
  是否 exact zero；
- local lookup 在冲突合并后的最小 endpoint residual 下界；
- local transition lookup 的最小 response residual 下界；
- 不能只报告 support coverage。

这里的 \(B\) 是保留全部通道和空间结构的冻结特征 patch，仅使用
样本级 RMS 做归一化；它不是一维 RMS descriptor。background 或
zero-response 位置的 zero-\(B\) 不构成不可达证据。

### P0-E：确定性

同一输入独立运行两次：

- canonical receipt 逐字节一致；
- SHA256 一致；
- 不读取 \(D_V/D_T\)；
- 不发生任何 \(D_R\) 数据训练。

dataset-free 门禁中固定存在一次 generated-only gradient probe，
receipt 必须将
`dataset_training_performed=false` 与
`synthetic_gradient_probe_optimizer_steps=1` 分开记录，不能把二者
混写为“完全没有 optimizer step”。

单次 P0 receipt 即使 P0-A～P0-D 全部通过，也只能写：

```text
eligible_for_replay = true
training_authorized = false
```

只有外层 replay authorization 同时绑定两个不同运行对象、两份
canonical bytes、两个文件 SHA256、相同正式 `IRSTD-1K/D_R`
source binding、相同 bounded population 和相同 implementation
closure，并确认逐字节一致后，才允许写：

```text
CMIF_V17_BOUNDED_400_AUTHORIZED
training_authorized = true
```

任何一项失败：

```text
CMIF_V17_P0_FAIL
training_authorized = false
```

不得通过增加 width、修改 radius、加入 loss 或搜索中点继续运行。

---

## 14. bounded-400 授权条件

只有 dataset-free G1–G12 与 \(D_R\) P0-A–P0-E 全部通过，才能生成：

```text
CMIF_V17_BOUNDED_400_AUTHORIZED
```

训练固定为：

- dataset：IRSTD-1K 的冻结 \(D_R\)；
- seed：42；
- epochs：10；
- steps per epoch：40；
- updates：400；
- objectives：SORR / identity / separable；
- 同一 CMIF class/config/初始 state；
- 相同 schedule、optimizer、cache 与计算预算；
- threshold：0；
- 不访问 \(D_V/D_T\)；
- 单次运行，不自动重试。

v17 不加入 CZB。若 CMIF + 原 SORR 仍只因 compact gate 失败，则冻结 v17，再以独立 v18 讨论目标函数，不允许在 v17 结果后补 barrier。

---

## 15. bounded-400 判定

candidate 必须同时满足现有全部 absolute gates：

- factual miss；
- factual recovered；
- factual target-negative；
- XDU680；
- factual no-miss；
- clean response；
- clean added-target-negative；
- compact exact；
- component match；
- clean full；
- outside completion；
- false islands；
- component-null；
- identity-null。

并保留 matched-control 判断：

- candidate 与 identity/separable 使用完全相同 CMIF；
- 不能用 candidate 的均值优势掩盖 absolute gate 失败；
- 任一必要门槛失败即：

```text
CMIF_V17_BOUNDED_400_GATE_FAIL
formal_800_authorized = false
```

只有 bounded-400 全部门槛通过，才讨论冻结后的 Formal800；Formal800 仍不自动授权 Full CURE 或其他 backbone。

---

## 16. 代码修改顺序

### R17-0：冻结证据

- 保留 v16 result、decision、checkpoint 与 source closure；
- 将 FRCE 标记为 P0 数学失败；
- 不删除历史负结果。

### R17-1：实现 CMIF core

新增：

```text
cure_lite/coverage_state_centered_mixed_interaction.py
```

包含：

- frozen config；
- exact phase 输入；
- 显式 reference equation；
- 高效中心列实现；
- auditable fields；
- single-field forward/predict_completion/predict_union。

### R17-2：实现 dataset-free receipt

新增：

```text
cure_lite/experiment/coverage_state_cmif_dataset_free.py
tests_v15/test_coverage_state_centered_mixed_interaction.py
tests_v15/test_coverage_state_cmif_dataset_free.py
```

### R17-3：实现冻结 \(D_R\) P0

新增：

```text
cure_lite/experiment/coverage_state_cmif_p0.py
tools/audit_coverage_state_cmif_v17.py
tests_v15/test_coverage_state_cmif_p0.py
```

P0 工具只允许 create-only，不得包含训练入口。

### R17-4：运行测试与两次 P0 重放

顺序：

```text
targeted tests
    -> tests_v15 full suite
    -> dataset-free r1/r2
    -> frozen D_R P0 r1/r2
```

### R17-5：条件式实现 bounded runner

仅当 R17-4 全部通过后，才新增：

```text
cure_lite/experiment/coverage_state_cmif_bounded_runner.py
tools/run_coverage_state_cmif_support_oriented_bounded_400.py
```

### R17-6：单次 bounded-400

只有 create-only authorization 为真时运行一次。否则按 P0 停止规则结束。

---

## 17. ICLR 创新性边界

CMIF 当前具备研究价值的原因不是参数更多，而是提出了一个可检验的 completion-state 原理：

\[
\text{completion}
\quad\Longleftrightarrow\quad
\text{feature evidence}
\times
\text{coverage-state interaction}.
\]

与直接 concat decoder 不同，CMIF 对两类捷径施加精确代数约束；与 FRCE/SVEF 可分离 gate 不同，CMIF 使用非可分离 mixed finite difference，并保留 rank-2 endpoint 表达。

但 ICLR 资格不能由结构描述直接得到。至少还需要后续证据：

1. CURE-Lite bounded 与 Formal800 稳定通过；
2. 完整三数据集训练与独立 seeds；
3. Full CURE 插入多个 IRSTD detector；
4. 一致降低漏检且不增加 FA；
5. 严格机制消融证明 mixed interaction，而非参数量或训练预算，带来收益。

当前阶段只解决第一个问题的模型设计与训练前可行性。

---

## 18. 最终状态表

| 项目 | 当前状态 |
|---|---|
| v16 PPCE | frozen negative |
| FRCE 方程 | P0 fail |
| FRCE 代码/训练 | not authorized |
| CZB | excluded from v17 |
| CMIF 方程 | frozen candidate |
| CMIF core | to implement |
| dataset-free gates | to run |
| frozen \(D_R\) P0 | to run after dataset-free |
| bounded-400 | conditional, not yet authorized |
| Formal800 | not authorized |
| \(D_V/D_T\) | not accessed |
| Full CURE | not started |
| cross-backbone | not started |

最终主线：

```text
freeze v16 negative
    -> reject infeasible FRCE
    -> implement one-field nonseparable CMIF
    -> dataset-free structural gates
    -> frozen D_R representability gates
    -> only-if-pass bounded-400
```

这次修改没有降低 CURE 的目标，也没有转向某个特定 backbone。它把 v17 从“看似合理但数学不可达的 gate”修正为“可被代码、结构门禁和后续训练明确证伪或支持的单场机制”。
