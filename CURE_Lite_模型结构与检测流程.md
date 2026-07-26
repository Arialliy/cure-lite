# CURE-Lite 模型结构与目标检测流程

## 1. 模型定位

CURE-Lite 当前不是一个替代所有 IRSTD 网络的图像编码器，也不是把
MSHNet、SCTransNet 或 DNANet 改成另一套 backbone。它是一个位于冻结
Base detector 之后的轻量目标补全模型：

\[
\text{image}
\xrightarrow{\text{Frozen Base}}
(p_b,F_b)
\xrightarrow{\text{CURE-Lite}}
\hat Y_{\mathrm{final}}.
\]

其中：

- \(p_b\in[0,1]^{B\times1\times H\times W}\)：Base 的初始目标概率图；
- \(F_b\in\mathbb R^{B\times C_f\times h\times w}\)：Base 暴露的一层冻结特征；
- CURE-Lite 只训练自己的 residual completion decoder；
- Base 始终冻结，不与 CURE-Lite 联合更新；
- 推理时只增加 Base 没有占用的像素，不删除 Base 已有检测。

因此它解决的具体问题是：

> 在保留 Base 已检目标的前提下，利用 Base 尚未完全使用的特征证据，
> 在受限误报预算内补回遗漏的小目标。

它当前不承担删除 Base false positive 的任务。

## 2. 整体网络

```text
image [B,Cin,H,W]
       |
       v
Frozen Base detector
       |-------------------------------|
       |                               |
       v                               v
base probability p_b              frozen feature F_b
[B,1,H,W]                         [B,Cf,h,w]
       |                               |
threshold at tau_b                     |
       |                               |
       v                               |
occupancy O [B,1,H,W]                  |
       |                               |
adaptive-max projection                |
       |                               |
       v                               v
Pi(O) [B,1,h,w]            1x1 Conv(Cf->32)+GN+SiLU
       |                               |
       |--------------- concat --------|
                       |
                       v
                 [B,33,h,w]
                       |
              3x3 Conv(33->32)
                + GroupNorm(8)
                + SiLU
                       |
              3x3 Conv(32->32)
                + GroupNorm(8)
                + SiLU
                       |
                1x1 Conv(32->1)
                       |
                bilinear upsample
                       |
                       v
        residual logits z_r [B,1,H,W]
                       |
                    sigmoid
                       |
                       v
        pre-mask residual score Q [B,1,H,W]
                       |
             zero scores inside O
                       |
              threshold at tau_r
                       |
                       v
             residual mask Y_r
                       |
                  hard union with O
                       |
                       v
           final mask Y_final = O OR Y_r
```

## 3. Base 前端

### 3.1 通用接口

所有 Base detector 通过同一个接口只提供：

\[
(p_b,F_b)=\operatorname{Base}(x).
\]

CURE-Lite 不读取 detector 名称、内部跳跃连接、具体通道含义或 GT。
不同 detector 可以具有不同 \(C_f,h,w\)，但必须输出：

```text
probability: [B,1,H,W], float32, in [0,1]
feature:     [B,Cf,h,w], float32
```

这里的“通用”是算法与接口通用，不是同一组权重通用。每个 detector 都要用
自己的冻结特征重新训练一个 CURE decoder。adapter 只能选择该 detector
本来就存在的特征，不得为某个 detector 私自改写 CURE 机制。

此外，任意 \(h,w\) 只表示代码不写死固定 stride，并不表示任意低分辨率特征
都有效。所选 \(F_b\) 必须通过 tiny-target representability preflight：
目标在该 feature grid 与 occupancy projection 下仍可见。若某 detector 只暴露
\(1/8\) 或 \(1/16\) 特征，1--3 像素目标的信息可能已经消失；CURE-Lite 应更换
为该 detector 的高分辨率 decoder feature，不能依靠上采样虚构细节。

### 3.2 当前正式开发前端

当前 IRSTD-1K Stage-A 使用项目内的 Reference Base，而不是 MSHNet。
Reference Base 是一个标准残差 U-Net，只负责提供开发阶段的固定前端：

```text
input grayscale:    [B,1,256,256]
stem:               1 -> 24
down 1/2:           24 -> 40
down 1/4:           40 -> 64        <- 暴露给 CURE-Lite 的 F_b
down 1/8:           64 -> 96
down 1/16:          96 -> 128
up 1/8:             128+96 -> 96
up 1/4:             96+64 -> 64
up 1/2:             64+40 -> 40
up full:            40+24 -> 24
head:               24 -> 1
```

其 \(F_b\) 形状为：

\[
[B,64,64,64].
\]

Reference Base 约有 128.2 万参数，但它不是 CURE 方法贡献，只是当前固定
输入提供者。

仓库另有 MSHNet adapter。它从冻结 MSHNet 的 `decoder_0` 读取
\([B,16,H,W]\) 特征；该 adapter 证明接口可以容纳另一种 detector，但当前
paired CURE-Lite 的正式开发实验仍不以 MSHNet 为主线。

## 4. Occupancy 条件

Base 概率图使用固定阈值构成：

\[
O=\mathbf1[p_b\ge \tau_b].
\]

当前 Stage-A 的 \(\tau_b=0.5\)。这里不使用 GT，也不在推理时进行目标删除。

当 \(F_b\) 的空间尺寸低于输出尺寸时，使用 adaptive max pooling 得到：

\[
\Pi O\in\{0,1\}^{B\times1\times h\times w}.
\]

使用 max projection 的原因是红外小目标可能只有 1--3 个像素；普通最近邻
下采样可能把一个非空小目标变成全零。max projection 保证一个 feature cell
只要覆盖过占用像素，就仍然保留占用状态。

## 5. CURE-Lite completion decoder

当前 decoder 是一个共享算子：

\[
Q_\theta(F_b,\Pi O)
=
\sigma\!\left(D_\theta(F_b,\Pi O)\right).
\]

它的逐层结构为：

| 层 | 输入 | 输出 | 操作 |
| --- | --- | --- | --- |
| feature projection | \(C_f\) | 32 | \(1\times1\) Conv + GN(8) + SiLU |
| condition fusion | 32 + 1 | 33 | 与 \(\Pi O\) 通道拼接 |
| local decoding 1 | 33 | 32 | \(3\times3\) Conv + GN(8) + SiLU |
| local decoding 2 | 32 | 32 | \(3\times3\) Conv + GN(8) + SiLU |
| response head | 32 | 1 | \(1\times1\) Conv |
| grid recovery | \(h\times w\) | \(H\times W\) | 双线性上采样 |

在当前 \(C_f=64\) 时，CURE-Lite decoder 有 21,089 个可训练参数；
接 MSHNet 的 \(C_f=16\) 时有 19,553 个参数。

decoder 中没有 attention、Transformer、额外 encoder、第二 decoder 或
多尺度模块。当前研究贡献不依赖堆叠这些结构，而是依赖同一个 completion
operator 的学习原则。

当前 Reference Base 的 \([1,64,64,64]\) feature shape 下，已有静态产物记录
decoder 为 85,196,800 Conv2d MAC（170,393,600 FLOPs，未计 GN、激活、投影
和上采样）；RTX 3090、batch 1 的历史 shape-probe 中位延迟为 0.792 ms。
该延迟只包含 decoder，不包含 Base、数据传输和指标计算。

参数少不等于任意 feature grid 上计算都少。若直接把 MSHNet 的 full-resolution
\([B,16,256,256]\) feature 输入相同 decoder，卷积计算会显著增加。因此未来
跨 detector 实验必须同时冻结 feature-tap 规则和增量计算预算；现有 MSHNet
adapter 证明接口可连接，不等于已经形成最终的高效 MSHNet 配置。

## 6. 一张图如何得到检测结果

### 6.1 Base 初检

Base 对图像产生 \(p_b\) 和 \(F_b\)，由 \(p_b\) 得到 occupancy \(O\)。
此时 \(O\) 表示 Base 已经声明为目标的像素。

### 6.2 CURE-Lite 补全响应

decoder 同时读取冻结特征和 occupancy：

\[
z_r=D_\theta(F_b,\Pi O),\qquad Q=\sigma(z_r).
\]

\(F_b\) 告诉模型“图像中还存在什么目标证据”，\(\Pi O\) 告诉模型
“哪些位置已经被 Base 覆盖”。模型需要在二者联合条件下判断还应补充哪里。

### 6.3 去除重复添加

在 Base 已占用区域中把 residual score 置零：

\[
Q_{\mathrm{write}}=Q\odot(1-O).
\]

这一步只用于推理融合。新的 paired loss 必须在这一步之前计算，否则删除
occupancy 自身会机械地产生差分，模型并没有真正学习响应。

### 6.4 残差阈值与单调融合

\[
\hat Y_r
=
\mathbf1[Q_{\mathrm{write}}\ge\tau_r],
\]

\[
\boxed{
\hat Y_{\mathrm{final}}=O\lor\hat Y_r
}.
\]

\(\tau_r\) 由冻结的校准协议选择，使 residual 带来的新增响应满足像素 FA、
背景 FA 和组件数量约束。

最终二值 mask 经过固定连通域和目标匹配规则计算：

- object-level Pd；
- pixel-level FA；
- FP components per megapixel；
- 已检目标 retention；
- 固定漏检目标找回数。

## 7. 新的 paired CURE-Lite 如何训练

旧 v0.1/v0.2 把 factual、no-miss 和 synthetic state 分别求 loss。synthetic
样本之间没有真正的 pair 关系，因此不能作为新模型主线。

新的训练保持网络拓扑不变，但给同一个 decoder 构造两个相关输入。

### 7.1 同源 before/after pair

对一个被 Base 正确覆盖且几何身份稳定的目标 \(g\)，取与它对应的完整
occupancy component \(C_g\)：

\[
O^+=O,\qquad O^-=O\setminus C_g.
\]

两个 endpoint 满足：

```text
同一 source image
同一 frozen feature F_b
同一共享 decoder
只改变一个经过验证的 occupancy component
```

分别计算：

\[
Q^+=Q_\theta(F_b,\Pi O^+),\qquad
Q^-=Q_\theta(F_b,\Pi O^-).
\]

真正进入新目标的是：

\[
\Delta_gQ_\theta=Q^--Q^+.
\]

### 7.2 标签

删除组件后，目标 \(g\) 从“已覆盖”变成“应由 completion operator 补回”。
实际新增的有效标签区域为：

\[
D_g
=
\mathcal R(O^-)\setminus\mathcal R(O^+).
\]

只有满足全部一对一谱系、匹配稳定和非干扰条件的 clean pair 才进入训练，
并验证：

\[
D_g=A_g=V\cap G_g\cap\neg O^-.
\]

这里的监督单位是完整目标实例在当前状态下的有效可写区域，不简单等同于
\(G_g\cap C_g\)。

### 7.3 paired loss

令：

\[
P=A_g,\qquad Z=V\setminus A_g.
\]

单个 clean pair 的目标为：

\[
\mathcal L_{\mathrm{pair}}
=
\frac12
\operatorname{mean}_{P}
\left(\frac{\Delta_gQ_\theta-1}{2}\right)^2
+
\frac12
\operatorname{mean}_{Z}
\left(\Delta_gQ_\theta\right)^2.
\]

含义是：

- 删除一个真正负责覆盖目标的组件后，目标区域的 completion score 应增加；
- 目标以外的有效区域不应产生额外变化；
- \(Q^+\) 和 \(Q^-\) 两端共同反向传播；
- pair identity 直接进入 loss，而不是把两个 endpoint 当作独立样本。

### 7.4 绝对锚

只学习差分会留下共同偏移的不确定性，因此仍保留已有 factual-miss 与
factual-no-miss 绝对监督：

\[
\mathcal L
=
\mathcal L_{F+}
+
\mathcal L_{F0}
+
\mathcal L_{\mathrm{pair}}.
\]

三个固定系数均为 1。旧 synthetic/legal endpoint 独立 ERM 不再进入主目标。
null pair 只用于验证模型是否对无标签增量的删除产生错误响应，不进入训练。

## 8. 训练和推理为何不同

paired before/after 只存在于训练阶段，用来约束同一个 \(Q_\theta\)。
正式推理仍然只有一次 decoder forward：

```text
Frozen Base once
  -> CURE-Lite decoder once
  -> occupancy hard mask
  -> residual threshold
  -> hard union
```

因此新训练原则不会把推理变成双分支网络，也不会要求在线删除组件。

## 9. 分步骤实现与验证

### Step 1：独立实现模型训练路径

新增而不覆盖旧代码：

- `PairExample`、`PairBatch`；
- clean/null pair catalog；
- paired difference loss；
- paired train step；
- 旧 `CURELiteDecoder` 和正式推理保持不变。

### Step 2：代码级验证

必须验证：

- \(O^+\)、\(O^-\)、\(D_g\)、\(A_g\) 的关系正确；
- pre-mask score 路径正确；
- 两个 endpoint 都有梯度；
- 2B batched forward 与两次 separate forward 一致；
- pair 顺序改变会改变 paired loss；
- identity/null pair 的实际标签增量为空；
- 1--3 像素目标经过 occupancy projection 后仍可见；
- 旧训练入口和旧测试不发生回归。

### Step 3：\(D_R\) 目录核验

只构造并统计：

- clean pair 数量与 source 数量；
- 每种排除原因；
- null pair 数量；
- projected occupancy 是否发生变化；
- target/source exposure；
- 32,000-step 计划是否过度集中。

这一步不读取新的 \(D_V\) 或 \(D_T\) 结果。

### Step 4：小规模可学习性验证

先在 toy 与极小 \(D_R\) 子集上验证：

- loss 是否下降；
- clean pair response 是否学到；
- null/background 是否保持接近零；
- 模型是否只是读取 occupancy；
- paired objective 是否不同于独立 endpoint ERM。

### Step 5：正式 seed 42/43

前四步通过后，按固定：

```text
epochs = 800
steps/epoch = 40
optimizer updates = 32,000
```

分别训练 seed 42 和 43。两个 seed 必须逐个通过，不用均值掩盖单个失败。

### Step 6：冻结确认

seed 42/43 通过只表示 CURE-Lite 候选可以冻结；仍需额外 seed 和未使用划分
确认，之后才设计 Full CURE。

### Step 7：Full CURE 与跨 detector 验证

只有 CURE-Lite 的最小机制确认成立后，才设计 Full CURE。Full CURE 冻结后，
再通过相同 adapter contract 接到 DNANet、UIUNet、MSHNet、SCTransNet，
并在 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 上分别重新训练和验证。这里验证的是
同一结构、同一目标与同一训练原则的 detector-level transfer，不是一个
checkpoint 的 zero-shot transfer。

## 10. 当前准确状态

```text
Base-to-CURE interface       已实现
CURE-Lite decoder            已实现
单次推理与 hard union        已实现
旧 v0.1/v0.2 训练            已实现且得到负结果
paired objective protocol    已冻结
paired catalog/loss/step     已实现
800x40 paired schedule       已实现
小规模联合过拟合             已通过
真实 D_R paired preflight    已通过并完成两次逐字节重放
matched-control preflight    已通过；206/206 target permutation READY
bounded D_R learnability     已通过并完成两次逐字节重放
正式训练与揭示入口           已实现并冻结
paired 800-epoch training    seed 42/43 已完成
Wave A 逐种子判定            PERFORMANCE_FAIL
CURE-Lite mechanism success  当前版本未建立
Full CURE                    尚未开始
cross-backbone validation    尚未开始
```

Wave A 的冻结结果为：

| seed | paired difference | 当 seed 最佳比较项 | 逐种子结论 |
| --- | --- | --- | --- |
| 42 | \(147/170\) true targets，\(0/23\) fixed misses recovered | \(154/170\)，\(7/23\) | 明确低于最佳比较项 |
| 43 | \(152/170\) true targets，\(5/23\) fixed misses recovered | \(152/170\)，\(5/23\) | 仅与最佳比较项持平 |

两组 seed 的像素 FA、背景 FA、FP components 和 retention 约束均通过，但
冻结门槛要求 proposed 在每个 seed 上都至少超过最佳比较项 2 个 true targets
和 2 个 fixed-miss recoveries。seed 42 下降，seed 43 没有严格提升，因此不能
用约束通过或跨 seed 均值把结果解释为机制成功。

正式状态是：

```text
Wave A status                 PERFORMANCE_FAIL
current version action        STOP_AND_PRESERVE_EVIDENCE
current combination effective 未证明
paired mechanism necessary    未证明
frozen confirmation           未授权
Full CURE                     未授权
cross-backbone validation     未授权
```

这否定的是当前 decoder、paired objective、绝对锚与训练配置构成的具体
CURE-Lite 版本，而不是自动否定 CURE 的总体研究问题。研究主线仍保持：

```text
重新形成可检验的 CURE-Lite 候选
  -> 重新通过逐种子机制门禁
  -> 冻结确认
  -> Full CURE
  -> 跨 IRSTD detector 与三数据集验证
```

在新的 CURE-Lite 候选重新通过门禁前，不开始 Full CURE，也不把当前结果
扩展到 DNANet、UIUNet、MSHNet 或 SCTransNet。
