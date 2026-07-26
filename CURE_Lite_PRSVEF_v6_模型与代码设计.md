# CURE-Lite PR-SVEF v6：模型与代码设计

> 方法名：Polarity-Recoverable Subpixel Vacancy–Evidence Factorization  
> 简称：PR-SVEF v6  
> 设计层状态：公式、边界与分阶段门禁冻结  
> 执行状态：以同目录 proposal、toy config、result 和 closure receipts 为准

## 1. 阶段位置

研究主线保持为：

```text
CURE-Lite 模型设计与验证
    -> CURE-Lite 冻结确认
    -> Full CURE
    -> 接入不同 IRSTD detector
    -> NUAA / NUDT-SIRST / IRSTD-1K 验证
```

PR-SVEF v6 是 CURE-Lite 的新模型代码候选，不是 Full CURE，也不是为某个
特定 backbone 增加的附加模块。当前只允许：

1. 新增 v6 config、decoder、测试和 toy evaluator；
2. 使用冻结的 1/2/3 像素 toy 完成模型可学习性门禁；
3. toy 3/3 通过后，新增独立 v6 bounded executor/CLI；
4. 之后只允许一次真实 \(D_R\) 400-update bounded 运行。

当前不允许读取 \(D_V/D_T\)，不运行校准、Pd/FA 评估或 formal 800 epoch，
不修改 Base、v4/v5、paired population、loss、train step、schedule 或推理图。

## 2. 前序结果与唯一待修复问题

SVEF v4 的唯一真实 \(D_R\) bounded 结果表明，模型能够执行完整 paired
训练，但当前证据函数的联合选择性不足。D-SVEF v5 进一步使用：

\[
E_{\mathrm{v5}}(r)
=
\operatorname{ReLU}
\left[
\operatorname{softplus}(r)-\log 2
\right],
\]

试图关闭负 evidence 并保留弱正证据的一阶响应。冻结 toy 的 3 个 case
全部失败。共同现象是目标概率停留在约 0.5。

失败定位是明确的：需要学习为正 evidence 的部分 subpixel phase 初始化在
负半轴，而 v5 在负半轴同时具有零前向响应和零梯度。这些 phase 无法跨过
原点。v6 只修复这一项“负相位训练不可达性”，不同时修改其他机制。

## 3. 唯一新机制：前向语义与训练可达性分离

### 3.1 前向证据函数

定义：

\[
\phi(r)
=
\operatorname{softplus}
\left(
\operatorname{ReLU}(r)^2
\right)
-\log 2.
\]

实现中对 \(r\leq0\) 使用显式零分支，保证：

\[
r\leq0
\Rightarrow
\phi(r)=0.
\]

对 \(r>0\)，前向函数与 v4 正半轴完全相同：

\[
\phi(r)
=
\operatorname{softplus}(r^2)-\log2.
\]

因此 v6 保留 v4 已有的正 evidence 表达能力，同时保持 v5 所要求的负半轴
精确关闭。由于 float32 中极小正数可能受到相减消去影响，本方法只声明
数学实数域中的严格正性；数值实现以测试中冻结的 dtype 和容差为准。

### 3.2 反向恢复规则

定义用于训练的 surrogate：

\[
\psi(r)
=
\begin{cases}
\phi(r), & r>0,\\
\operatorname{softplus}(r)-\log2, & r\leq0.
\end{cases}
\]

最终算子为：

\[
E_{\mathrm{PR}}(r)
=
\operatorname{sg}[\phi(r)]
+\psi(r)
-\operatorname{sg}[\psi(r)],
\]

其中 \(\operatorname{sg}\) 表示 stop-gradient。于是前向严格等于：

\[
E_{\mathrm{PR}}^{\mathrm{forward}}(r)=\phi(r),
\]

训练时使用：

\[
\frac{\widetilde{\partial E_{\mathrm{PR}}}}{\partial r}
=
\begin{cases}
2r\sigma(r^2), & r>0,\\
\sigma(r), & r\leq0.
\end{cases}
\]

零点固定使用恢复分支：

\[
\left.
\frac{\widetilde{\partial E_{\mathrm{PR}}}}{\partial r}
\right|_{r=0}
=0.5.
\]

这不是前向函数在负半轴的真实导数，而是预先声明的优化规则。其目的仅是
使负初始化的目标 phase 有机会恢复到正 evidence 区域；它不保证任意损失
下梯度方向必然为正，也不改变推理时的前向函数。

### 3.3 为什么这是单机制

v6 新增：

- 模块：0；
- 参数：0；
- head：0；
- loss：0；
- train step：0；
- 推理分支：0；
- 可调超参数：0。

唯一变化是 evidence field 的一个“前向函数 + 反向规则”算子。前向负责
证据语义，反向负责训练可达性，二者共同定义同一操作，不能拆成两个模块。

## 4. 完整网络结构

v6 继续使用：

\[
z_\theta(F,O)
=
B_\theta(F)
+
E_{\mathrm{PR}}
\left(
\widetilde e_\theta(F)
\right)
\odot A(O).
\]

其中：

\[
B_\theta(F)
=
-\operatorname{softplus}
\left(
\beta_{\mathrm{raw}}+\widetilde b_\theta(F)
\right)
<0,
\]

\[
A(O)
=
\operatorname{Nearest}_{H,W}
\left[
\frac{1}{
1+
\mathbf 1_{3\times3}
*
\operatorname{ProjectMax}_{h,w}(O)}
\right].
\]

共享 trunk 与 subpixel phase heads 保持：

\[
T_0
=
\operatorname{SiLU}
\left[
\operatorname{GN}(W_s*F)
\right],
\]

\[
T
=
T_0
+
\frac12 W_p*
\operatorname{SiLU}
\left[
\operatorname{GN}(W_d*T_0)
\right],
\]

\[
\widetilde b
=
\operatorname{PixelShuffle}_s(W_B*T),
\qquad
\widetilde e
=
\operatorname{PixelShuffle}_s(W_E*T).
\]

参考实例保持：

```text
feature_channels = 64
feature_stride = 4
width = 32
groups = 8
trunk_residual_scale = 0.5
baseline_probability = 0.1
vacancy_kernel_size = 3
parameter_count = 4,385
```

模型输入仍只有冻结 Base feature \(F\) 与 hard occupancy \(O\)。ground
truth、pair kind、Base probability 均不进入 decoder。

## 5. 检测流程保持不变

单张图像推理仍是：

```text
image
  -> 任意 IRSTD Base adapter
  -> frozen Base probability p_b、feature F
  -> hard occupancy O = 1[p_b >= fixed pre-mask threshold]
  -> PR-SVEF decoder(F, O)
  -> residual probability
  -> fixed hard-union composition
  -> final mask
```

v6 不重新运行 Base，不生成第二条 detector 路径，不改变 pre-mask、hard
occupancy 或 hard-union。`CURELiteFactorizedModel` 可直接接收 v6 decoder，
无需新 model wrapper。

## 6. 必须保持的结构不变量

正式实现必须逐项验证：

1. feature 在 decoder 内 detach，Base 不接收梯度；
2. state dict、子模块、参数形状和参数数量与 v4 一致；
3. baseline、vacancy、occupancy projection、resize policy 不变；
4. \(r<0\) 与 \(r=0\) 的前向 evidence 精确为零；
5. \(r>0\) 前向 evidence 等于 v4；
6. \(r<0\) surrogate gradient 等于 \(\sigma(r)\)；
7. \(r=0\) surrogate gradient 等于 0.5；
8. \(r>0\) gradient 等于 \(2r\sigma(r^2)\)；
9. 负初始化 phase 在受控优化测试中能够跨过零点；
10. zero-feature occupancy invariance；
11. identity endpoint exactness；
12. 删除 logit/probability 单调性；
13. delta 在 vacancy gate change support 外严格为零；
14. subpixel phase freedom；
15. 2B batched endpoints 与两个独立 forward 一致；
16. 单次 Base/decoder 推理、pre-mask 和 hard-union 图不变。

普通有限差分 `gradcheck` 不适用于负半轴，因为 surrogate gradient 被明确
设计为不同于前向函数的真实导数。测试必须分别核验前向值与声明的反向值。

## 7. 冻结 toy 门禁

正式 toy 使用与 v4/v5 相同的：

```text
seed = 7817
optimizer = Adam
updates = 320
learning_rate = 0.004
feature_channels = 8
feature_stride = 4
cases = 1 / 2 / 3 subpixel targets
```

每个 case 必须同时满足：

\[
\mathrm{total\ loss}<0.10,
\]

\[
\min p_{\mathrm{plus,target}}>0.95,
\qquad
\max p_{\mathrm{plus,bg}}<0.05,
\]

\[
\min p_{\mathrm{factual\ miss,target}}>0.95,
\qquad
\max p_{\mathrm{factual\ miss,bg}}<0.05,
\]

\[
\max p_{\mathrm{factual\ no\ miss}}<0.05,
\]

\[
\overline{\Delta_D}\geq0.80,
\]

\[
\max|\Delta_H|,
\max|\Delta_G|,
\max|\Delta_{H,\mathrm{component}}|,
\max|\Delta_{G,\mathrm{component}}|
\leq0.05.
\]

两个 endpoint 的初始梯度还必须同时有限且非零。必须逐 case 3/3 通过，
不能以均值替代，不能修改预算或阈值。

## 8. 分阶段停止规则

```text
设计/proposal/toy config 冻结
  -> config/decoder/结构测试
  -> frozen toy 3/3
       ├─ fail: 冻结 v6 负结果，停止 v6
       └─ pass: 才新增 v6 bounded executor/CLI
                    -> 唯一一次真实 D_R 400-update
                         ├─ fail: 冻结 v6 负结果
                         └─ pass: 才讨论 formal 800 seed 42/43
```

任何失败都不允许在同一版本自动重试。toy 已参与候选开发，因此只能作为
模型代码门禁；真实 \(D_R\) bounded 也仍是内部模型门禁，不是 Pd/FA 性能
结论。

## 9. 创新主张边界

现阶段可提出的核心思想是：

> 在带有离散 occupancy 约束的子像素 residual correction 中，将证据的
> 前向语义约束与训练可达性共同编码为一个显式算子，使“负证据在推理时
> 精确关闭”与“负初始化目标 phase 在训练时可恢复”同时成立。

这比附加 attention、额外 decoder 或多项 loss 更集中，也直接对应 v5 的
可复现失败。但在真实 \(D_R\)、formal seeds 和最终检测性能完成前，不能
声称该机制已经提高 Pd、降低 FA、优于其他方法或达到 ICLR 证据要求。
