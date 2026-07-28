# CURE-Lite v19 USCOPE 失败归因与 v20 BFA-CMIF 修改方案

> 文档状态：v20 运行前冻结方案  
> v19 状态：完整 bounded-400 正式负结果  
> v20：**BFA-CMIF — Binary-Flip Antisymmetrized CMIF**  
> 中文：**二值翻转反对称混合交互场**  
> 固定性能种子：`seed=42`  
> 当前范围：模型核心、dataset-free、只读 `D_R` 与 bounded-400  
> 当前不进入：Formal800、`D_V/D_T`、三数据集、Full CURE、其他 backbone

---

## 1. 冻结结论

v19 USCOPE 已完成唯一一次固定 `seed=42`、400-update 正式运行：

```text
USCOPE_V19_BOUNDED_400_GATE_FAIL
```

它不是执行失败。16/16 个正式产物及其哈希、receipt 指纹和
`COMPLETE.json` 均已核对。运行只读取 `D_R`，未进行 Formal800 或真实性能
评估。

v19 相对 v18 的固定输出比较为：

| 指标 | v18 PMOPE | v19 USCOPE |
| --- | ---: | ---: |
| factual strict | 11/16 | 6/16 |
| factual recovered | 15/16 | 14/16 |
| factual target negative | 296/335 | 236/335 |
| clean target negative | 123/149 | 49/149 |
| clean completion | 170 | 77 |
| clean target 外 completion | 47 | 28 |
| component-null | 15/16 | 15/16 |

USCOPE 降低了新增响应，但目标完成下降得更严重。其 hard maximum
没有把最差坐标证书转化为可用的零水平检测结果。因此：

1. v19 保持只读负结果；
2. 不增加 USCOPE 步数、不调 margin、不换 seed；
3. v20 不设计第四种 loss；
4. v20 回到 v18 的冻结 PMOPE，只修改模型场方程。

这样，v20 相对 v18 只有一个研究变量：

```text
CMIF midpoint mixed field
    -> BFA binary-flip antisymmetrized mixed field
```

---

## 2. 为什么下一步必须是真正的模型修改

现有 CMIF 对每个 coarse cell 和输出 phase 使用实际 occupancy 与固定中点
`0.5` 的混合能量差。改变 PMOPE 为 USCOPE 后，target crossing 与背景抑制
发生明显移动，但两版都未同时通过 clean、factual 和 component 门槛。

下一候选必须满足：

- 真正改变 `O=0` 可补区域的函数映射；
- 不是输出后乘 mask；
- 不只是改变训练采样、loss 或阈值；
- 不增加多个 head、角色分支或后处理；
- 保持任意 IRSTD 前端可提供的通用 `(F_b,O)` 接口；
- 保持一个 completion field。

以下方案因此不作为 v20：

- **AP-CMIF**：固定参数下的 completion/union 与旧 CMIF 完全相同，只改
  pair 训练路径；
- **CPOE**：仍是另一种 loss，不符合本阶段模型代码优先级；
- **phase transport 扩容**：会新增较大的 phase-specific 参数对象；在尚未
  判断 midpoint 公共曲率是否为瓶颈前，不先增加容量。

---

## 3. v20 的唯一学习对象

### 3.1 输入

保持：

\[
B=N(F_b),
\qquad
U=\operatorname{PixelUnshuffle}_4(O).
\]

其中：

- \(B\) 是冻结 Base feature 的现有归一化表示；
- \(U\in\{0,1\}^{16\times h\times w}\) 精确保留 4×4 输出 phase；
- 外部输入仍只有 `(F_b,O)`。

### 3.2 共享 feature-presence energy

继续使用一个共享 joint affine 和一个 scalar energy：

\[
q_\theta(B,U)=W_F*B+W_U*U+b,
\]

\[
H_\theta(B,U)
=
\operatorname{SiLU}(q_\theta(B,U))
-
\operatorname{SiLU}(q_\theta(0,U)),
\]

\[
G_\theta(B,U)=v^\top H_\theta(B,U).
\]

这里的 \(W_F,W_U\) 仍只是同一个 `joint_state_weight` 的两个 view。

### 3.3 精确二值翻转

对 coarse cell \(i\) 和 phase \(p\)，只翻转当前中心 phase：

\[
\tau_{i,p}(U)_{i,p}=1-U_{i,p},
\]

\[
\tau_{i,p}(U)_{j,q}=U_{j,q},
\quad (j,q)\ne(i,p).
\]

该操作是 involution：

\[
\tau_{i,p}^2(U)=U.
\]

### 3.4 BFA 场方程

定义二值翻转的反对称奇部：

\[
\Delta^{\mathrm{BFA}}_{i,p}
=
\frac12
\left[
G_\theta(B,U)
-
G_\theta(B,\tau_{i,p}(U))
\right].
\]

唯一输出场为：

\[
\boxed{
\phi_{i,p}=a+\Delta^{\mathrm{BFA}}_{i,p}
},
\qquad a=0.9.
\]

经现有 PixelShuffle 得到全分辨率 field。推理保持：

\[
C=(\phi<0)\land\neg O,
\]

\[
Y_{\mathrm{final}}=O\lor C.
\]

没有第二个输出、阈值搜索或新增后处理。

---

## 4. 它相对 CMIF 删除了什么

固定其余 occupancy，令：

\[
H_0=H(B,U^{p\leftarrow0}),
\quad
H_1=H(B,U^{p\leftarrow1}),
\quad
H_m=H(B,U^{p\leftarrow1/2}).
\]

定义：

\[
o=\frac12(H_0-H_1),
\]

\[
e=\frac12(H_0+H_1)-H_m.
\]

其中 \(o\) 是 binary-flip odd component，\(e\) 是 SiLU 中点产生的 even
curvature。

旧 CMIF 为：

\[
\Delta_{\mathrm{old}}(U_p=0)=v^\top(o+e),
\]

\[
\Delta_{\mathrm{old}}(U_p=1)=v^\top(-o+e).
\]

新 BFA-CMIF 为：

\[
\Delta_{\mathrm{BFA}}(U_p=0)=v^\top o,
\]

\[
\Delta_{\mathrm{BFA}}(U_p=1)=-v^\top o.
\]

所以 v20 的唯一结构操作是：

> 从同一个 feature-presence energy 中保留二值 coverage flip 的奇部，精确
> 删除同时平移两个 endpoint 的中点曲率项。

它满足：

\[
\Delta(\tau U)=-\Delta(U),
\]

\[
\phi(U)+\phi(\tau U)=2a.
\]

若 energy 关于当前 phase 恰好是仿射的，则 \(e=0\)，BFA 与旧 CMIF
完全相同。因而真实 `D_R` 必须先证明 \(e\) 不是数值上可忽略的量。

---

## 5. 为什么这不是模块堆叠

BFA-CMIF 仍只有：

- 一个 joint state kernel；
- 一个 hidden energy；
- 一个 scalar energy readout；
- 一个 scalar completion field；
- 一个固定零水平。

两个 energy 状态是同一个离散差分算子的两个端点，参数完全共享。v20
不增加：

- encoder、decoder stage、attention 或 Transformer；
- feature head、occupancy head、target head 或 background head；
- pair-kind 输入；
- role-specific loss；
- learned threshold；
- morphology 或 connected-component 后处理；
- 新模型参数。

参数仍为：

\[
32\times(64+16)\times5\times5+32+32
=
64,064.
\]

初始化仍为：

```text
joint_state_weight    Kaiming normal
joint_hidden_bias     0
scalar_energy_weight  0
initial field         +0.9
initial completion    empty
```

---

## 6. 三类状态使用同一个标量区间

令：

\[
d=v^\top o,
\qquad
m_0=0.225.
\]

在固定局部邻域的两个二值端点：

\[
\phi_{U=0}=a+d,
\qquad
\phi_{U=1}=a-d.
\]

### 6.1 factual miss 与 clean-minus target

可补目标要求：

\[
a+d\le-m_0,
\]

即：

\[
d\le-(a+m_0)=-1.125.
\]

同一条件同时用于自然漏检和合法删除后的 target，不需要 factual/synthetic
两套 head。

### 6.2 writable background

背景要求：

\[
a+d\ge m_0,
\]

即：

\[
d\ge m_0-a=-0.675.
\]

target 与 background 因此在同一个 scalar contrast 上具有固定间隔。

### 6.3 component-null

若同一局部背景的两个 occupancy endpoint 都应保持正场，则：

\[
a+d\ge m_0,
\qquad
a-d\ge m_0,
\]

等价于：

\[
|d|\le a-m_0=0.675.
\]

clean target、普通背景与 component-null 都由同一个 BFA contrast 判断，
不是多个规则的合取。

---

## 7. 与 v18 失败的对应关系

v18 的主要结果为：

```text
clean target negative = 123 / 149
clean outside         = 47
clean compact         = 0 / 16
factual strict        = 11 / 16
component-null        = 15 / 16
```

旧 midpoint curvature \(e\) 可以把两个 binary endpoint 同向移动：

- 为使 minus target 变负时，也可能把 plus 或附近背景向负侧移动；
- 为压制 outside 时，又可能同时削弱 target crossing。

BFA 删除该 common-mode，但保留真正的 binary response \(o\)。因此它直接
检验一个结构假设：

> v18 的 crossing–outside 权衡是否主要来自 midpoint even curvature，而
> binary odd response 本身仍包含可用的 target/background 区分信息。

这是可证伪假设，不是性能承诺。

---

## 8. Dataset-free 门禁

不读取任何真实数据，不训练。必须全部通过：

1. flip 是 exact Boolean involution；
2. 显式逐 cell/phase 两角 energy 与高效实现逐元素一致；
3. \(\Delta(\tau U)=-\Delta(U)\)；
4. \(\phi(U)+\phi(\tau U)=2a\)；
5. \(B=0\Rightarrow\phi\equiv a\)；
6. pure-feature 或 pure-occupancy additive energy 不能单独写 field；
7. 仿射 energy witness 中 BFA 与 midpoint CMIF 相同；
8. 非线性 SiLU witness 中 curvature 非零、新旧 writable field 不同，并存在
   一个零水平输出不同的构造；
9. target、background、component 的区间条件同时可行；
10. target/background violation 的下降方向正确且有限；
11. phase unshuffle/shuffle roundtrip 精确；
12. 参数 key、shape、count、初始化与 v18 一致；
13. staged gradient path 有限；
14. forward 不读取 role、pair kind、sample ID 或 GT metadata；
15. 不构造 optimizer，不读取 `D_R/D_V/D_T`。

任一失败即停止，不运行真实 `D_R`。

---

## 9. 真实 `D_R` 训练前门禁

只读取冻结 `D_R`，不更新参数。

### 9.1 odd–curvature 非退化

对真实 target、background 和 component 坐标计算：

\[
\rho_i
=
\frac{\|e_i\|_2}
{\|o_i\|_2+\epsilon}.
\]

必须报告完整分布。若所有真实可补坐标的 \(e\) 都为零，或总体
\(\|e\|_2/\left(\|o\|_2+\epsilon\right)\) 不高于
`128 × float32 epsilon`，则：

```text
BFA_D_R_IDENTIFIABILITY_FAIL
```

不得训练。

### 9.2 odd representation 可用

必须满足：

- clean target 与 factual target 的 odd basis 均有限；
- 每个 target group 的 odd basis 至少一个坐标非零；
- component/background 也被同一个 basis 计算；
- 不存在逐字节完全相同的 odd representation，却被要求同时处于互斥区间的
  确定性冲突。

### 9.3 梯度与实现

必须满足：

- target field 的下降方向使 field 变负；
- writable background/component 的下降方向使 field 变正；
- scalar readout 在第一步具有非零有限梯度；
- 固定非零 readout witness 下 joint weight/bias 梯度非零；
- model、cache 和 RNG 前后不变；
- optimizer 未构造、step 为 0；
- `D_V/D_T` 未读取；
- v18/v19 失败像素不进入模型输入、权重或 loss。

该门禁只判断 BFA 是否在真实输入上形成了非退化的新结构坐标，不根据
`D_V` 指标调任何值。

---

## 10. 唯一 bounded-400

只有前两级门禁通过，才运行：

```text
model             BFA-CMIF
pair objective    frozen PMOPE
seed              42
epochs            10
steps per epoch   40
updates           400
optimizer         与 v18 相同
threshold         0
retry/resume      false
D_V/D_T           false
```

### 10.1 必须保持的底线

```text
factual no-miss       16/16
identity-null         16/16
diagnostic-null       pass
invalid completion    0
```

### 10.2 结构推进门槛

为避免继续要求一个小型诊断 population 在 400 步内达到不必要的全零误差，
v20 的 bounded 门槛预先固定为同时严格改善 v18 的 target–outside 权衡：

```text
factual strict             > 11/16
factual recovered          = 16/16
clean target negative      > 123/149
clean outside completion   < 47
clean compact support      > 0/16
component-null             >= 15/16
```

六项必须同时成立。一个指标的强提升不能抵消另一个指标下降。

同时完整报告但不单独替代上述门槛：

- 32 pair raw sign errors；
- 每对 \(\gamma\) 与 `gamma < m0`；
- target/background/component 的 odd 与 curvature；
- same-sign response；
- 参数量、显存、forward 次数和训练时长。

若结构推进门槛失败：

```text
BFA_CMIF_V20_BOUNDED_400_GATE_FAIL
```

随后冻结，不追加步数、不换 seed、不叠加 phase transport 或新 loss。

若全部通过：

```text
BFA_CMIF_V20_BOUNDED_400_GATE_PASS
```

这只表示可以签发独立 Formal800 计划，不代表性能已经成功。

---

## 11. 后续性能顺序

```text
freeze v19
    -> BFA core
    -> dataset-free
    -> real D_R identifiability
    -> seed42 bounded-400
        -> FAIL: freeze v20
        -> PASS: fixed-seed42 Formal800
            -> IRSTD-1K performance
            -> NUAA-SIRST / NUDT-SIRST / IRSTD-1K
            -> only then consider Full CURE and other frontends
```

Formal800 才使用 Pd、Fa、IoU、nIoU 与开销判断真实性能。三数据集实验必须
比较同一 Base 前端“无 CURE-Lite / 有 CURE-Lite”，且保持 Base 冻结。

---

## 12. 风险与创新边界

主要风险：

1. 单像素 binary flip 可能偏离自然 component intervention；
2. 多像素删除时，plus/minus 的其他邻域坐标也会变化；
3. 反对称约束可能删除有用的 even component；
4. target 的 \(d\le-1.125\) 与 component 的 \(|d|\le0.675\) 可能仍不可分；
5. frozen feature 可能缺少 factual miss 所需信息；
6. 若真实 curvature 很小，BFA 近似旧 CMIF；
7. PMOPE 本身仍可能是剩余瓶颈。

当前最诚实的潜在创新是：

> 将 coverage-conditioned completion 写成共享 feature-presence energy 在
> Boolean occupancy 边上的反对称离散导数，并用同一个 scalar field 同时
> 表达 factual completion、clean coverage response 与 component null。

但 binary finite difference、energy antisymmetrization 和 level-set risk 均可能
存在相关工作。当前必须保持：

```text
novelty_status      = NEEDS_SEARCH
closest_work_status = NOT_VERIFIED
ICLR_readiness      = NOT_ESTABLISHED
```

本文冻结的是 v20 代码与实验方向，不包含任何未运行的 v20 结果。
