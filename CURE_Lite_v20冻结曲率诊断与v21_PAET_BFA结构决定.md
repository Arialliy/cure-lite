# CURE-Lite：v20 冻结曲率诊断与 v21 PAET-BFA 结构决定

## 1. 阶段结论

冻结 v20 BFA-CMIF 权重的一次性曲率诊断已经完整结束，正式结论为：

```text
V21_FORMULA_REJECTED
```

这不是运行失败。诊断完整重建了固定 \(D_R\) 人口并正常结束，同时满足：

- 训练次数为 0；
- backward 次数为 0；
- optimizer 构造和更新次数均为 0；
- 未写入新 checkpoint；
- 未读取 \(D_V\) 或 \(D_T\)；
- 阈值仍为 0，没有搜索阈值；
- 没有搜索曲率尺度、符号或非线性；
- v20 checkpoint、population 和 cache 均保持不变。

因此，该结果只否决下面这一条固定场方程：

\[
e=\frac{G(U)+G(\tau U)}2-G(U_{1/2}),
\]

\[
\boxed{
\Delta'=\Delta_{\mathrm{BFA}}
\cdot
\left[1-\tanh\left(\frac{e}{0.9}\right)\right]
}.
\]

代码中的唯一实现是：

```text
delta_prime = delta_bfa * (1 - tanh(e / 0.9))
```

该负结果不否定 BFA、CMIF 或完整 CURE-Lite 方向。

## 2. 冻结输入与结果绑定

| 项目 | 固定值 |
|---|---|
| v20-r2 COMPLETE fingerprint | `8908a8c1896951e46fd737aa6f7fef2c9935e6524632b3576b8069faa026e2eb` |
| v20 checkpoint SHA256 | `040d2ca4ffa012c813e2c3e5dfa2c6f4877a91c8ff0b901bf8dc83df62026c42` |
| v20 module fingerprint | `0393532f8ea62e790c120ca0c0b86bf04c67b88c863e333f3f7c640d865ab5c0` |
| v20 zero-level receipt fingerprint | `4301c8e9f3393c2bc64c28b20e3b6e16bdc98b974281b7d9c67d239a86c76219` |
| 固定 cache fingerprint | `c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca00754c206092aaf66d8` |
| 本次 COMPLETE fingerprint | `e4d9621bbd6ac91165cd3dba35fcf5bf977e0f87fc987adb282ccd80d177c650` |
| 本次 receipt fingerprint | `b0108ac4f8e90b908421d1fa0c164d8222ec23f6f827fa4639d522743d66ada8` |
| 本次 audit fingerprint | `ce1463acdae350e8ee90cd374dda3fd5c9ad2194f6538c53db35e1ba39572b9c` |
| 结果文件 SHA256 | `afade289de472286af41e0bb47ef2574a95d74b32d4b22bc3cd593925f4413ed` |

结果目录：

```text
runs/irstd1k_stage_a_seed42/
  cure_lite_bfa_cmif_v20_curvature_audit_r1/
```

完整性复核结果：

- COMPLETE 自指纹一致；
- receipt 自指纹一致；
- audit payload 指纹一致；
- COMPLETE 中记录的结果文件 SHA256 与实际文件一致；
- 不存在 `.incomplete`。

## 3. 曲率假设为什么被否决

预声明假设要求：

- clean target 的曲率 \(e<0\)，从而放大目标负场；
- v20 outside completion 的曲率 \(e>0\)，从而衰减目标外负场；
- factual target 的曲率 \(e<0\)，从而保持自然漏检修复。

真实结果并不满足这个前提。

### 3.1 全局曲率中位数

| 固定坐标组 | 数量 | 曲率中位数符号 |
|---|---:|---:|
| clean added target | 149 | 正 |
| v20 outside completion | 54 | 正 |
| factual target | 335 | 正 |
| clean true background | 1,048,349 | 正 |
| component-null | 109 | 正 |

三个关键十六进制中位数为：

```text
clean target = 0x1.4520000000000p-8
v20 spill    = 0x1.eb8c000000000p-8
factual      = 0x1.27a4000000000p-8
```

曲率不是“目标为负、溢出为正”的角色信号。它在目标和溢出位置均主要为正。

### 3.2 按样本或 pair 的方向一致性

| 判据 | 正确/有效组 | 结论 |
|---|---:|---|
| clean target \(e<0\) | 1/16 | fail |
| factual target \(e<0\) | 0/16 | fail |
| spill \(e>0\) | 14/14 | pass |
| 同 pair 中 \(e_T<e_S\) | 9/14 | fail |

因此，固定调制会稳定衰减 spill，但也会同时衰减绝大多数目标响应。

## 4. 冻结权重代理结果

| 指标 | v20 BFA | 固定曲率代理 | 变化 |
|---|---:|---:|---:|
| clean target negative | 115/149 | 110/149 | -5 |
| clean outside completion | 54 | 47 | -7 |
| factual target negative | 310/335 | 302/335 | -8 |
| factual strict | 14/16 | 12/16 | -2 |
| factual recovered | 16/16 | 16/16 | 0 |
| clean compact | 1/16 | 1/16 | 0 |
| component-null | 16/16 | 16/16 | 0 |
| factual no-miss | 16/16 | 16/16 | 0 |
| identity-null | 16/16 | 16/16 | 0 |
| invalid completion | 0 | 0 | 0 |

该公式确实减少了目标外响应，但代价是进一步削弱 clean target 和 factual target。它没有解决 v20 的空间分配问题，只是进行了整体衰减。

按照一次性规则：

- 不修改 \(0.9\)；
- 不反转 tanh 符号；
- 不增加 clamp；
- 不试相邻尺度；
- 不训练这个候选。

## 5. 下一结构：PAET-BFA

下一版使用唯一备用机制：

```text
PAET-BFA
Phase-Aligned Evidence Transport inside
Binary-Flip Antisymmetric Field
```

### 5.1 要解决的结构缺口

v20 已经把 occupancy \(O\) 保留为 \(4^2=16\) 个亚像素 phase，但同一粗网格内的 16 个输出位置仍共享同一个 feature affine：

\[
A_F=W_F*\operatorname{Norm}(F_b).
\]

因此，模型具有 phase 身份，却没有与 phase 对齐的特征证据。这与 v20 的真实失败形式一致：

- completion 总量几乎没有下降；
- 目标内响应不足；
- 相邻目标外位置响应过多。

### 5.2 单一机制

对 stride \(s=4\) 和 phase \(p=(r,c)\)，固定偏移为：

\[
\delta_p=
\left(
\frac{r+0.5}{s}-0.5,\,
\frac{c+0.5}{s}-0.5
\right).
\]

将粗特征证据在进入共享非线性能量之前运输到相应 phase：

\[
A_F^p(i,j)=
\operatorname{Bilinear}
\left(A_F,i+\delta_p^y,j+\delta_p^x\right).
\]

再计算：

\[
G_p(F_b,U)=
w^\top
\left[
\operatorname{SiLU}(A_U+A_F^p)
-\operatorname{SiLU}(A_U)
\right],
\]

\[
\Delta_p=
\frac{G_p(F_b,U)-G_p(F_b,\tau_pU)}2,
\]

\[
\phi=
\operatorname{PixelShuffle}(0.9+\Delta_{1:s^2}).
\]

高效实现只需要一次固定双线性插值和一次 phase packing，不需要 16 个独立 head。

### 5.3 主线保持不变

PAET-BFA 仍满足：

- 输入只有 \((F_b,O)\)；
- 输出只有一个 completion field；
- 不增加 decoder、head 或并行分支；
- 不加入 curvature 模块；
- 参数仍是原来的 3 个 tensor、64,064 个参数；
- binary-flip antisymmetry 保持；
- \(F_b=0\) 时场仍为 0.9；
- loss、PMOPE、数据、seed=42、阈值 0 和推理规则均不改。

它只修改“粗特征证据如何进入 16 个亚像素 phase 的同一个共享能量”，属于单一表示—作用对齐机制。

## 6. 后续顺序

```text
PAET-BFA core
  -> dataset-free
  -> real-D_R identifiability
  -> fixed seed-42 bounded-400
  -> Formal800（仅 bounded-400 通过后）
  -> 三大数据集（仅 CURE-Lite 正式通过后）
```

bounded-400 的两个空间主门槛保持为：

\[
\text{clean target negative}\ge124/149,
\]

\[
\text{clean outside completion}\le46.
\]

同时必须保持 v20 已获得的 factual、component-null、null 和 invalid-completion 结果。两个主门槛必须同时通过，不能用一个方向的提升抵消另一个方向的下降。

PAET-BFA 的新颖性目前仍标记为 `NEEDS_SEARCH`。后续需要专门核查它与普通 bilinear upsampling、PixelShuffle 和已有 subpixel decoder 的差异；真正的核心主张只能是“在共享反对称能量内部进行 phase-aligned evidence transport”，不能把固定插值本身包装成创新。
