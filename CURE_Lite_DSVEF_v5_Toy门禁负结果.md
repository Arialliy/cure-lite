# CURE-Lite D-SVEF v5：Toy 门禁负结果

## 1. 正式状态

| 项目 | 状态 |
|---|---|
| 方法 | D-SVEF v5 |
| 实现与结构契约 | PASS |
| focused tests | 27 passed |
| frozen toy gate | FAIL，0/3 cases 通过 |
| 真实 \(D_R\) 400-update bounded | NOT_RUN_BY_TOY_STOP_RULE |
| \(D_V/D_T\) | NOT_ACCESSED |
| Pd/FA 检测性能 | NOT_EVALUATED |
| formal 800 epoch | NOT_AUTHORIZED |
| Full CURE / 其他 detector | NOT_AUTHORIZED |

D-SVEF v5 已形成可执行模型代码，但没有通过真实数据运行之前的固定
toy/code gate。因此 v5 在 toy 阶段停止，不创建真实 \(D_R\) 运行入口，
不运行真实 bounded，不进行自动重试，也不修改阈值、训练预算、loss 或
decoder 其他结构。

这个结论是模型代码门禁负结果，不是 Pd、FA 或分割性能结论。

## 2. 冻结机制

v5 相对 SVEF v4 只改变 evidence operator：

\[
E_{\mathrm{v5}}(r)
=
\operatorname{ReLU}
\left[
\operatorname{softplus}(r)-\log 2
\right].
\]

其余共享 trunk、baseline head、evidence head、PixelShuffle、固定 vacancy
gate、paired loss、train step、数据、schedule、优化预算与推理组合均保持
不变。参考实例的可训练参数仍为 4,385。

## 3. 固定 toy 契约

三个 case 共用：

- seed：7817；
- optimizer：Adam；
- updates：320；
- learning rate：0.004；
- feature channels：8；
- feature stride：4；
- 与 v4 完全相同的 loss、toy population 和判定阈值。

结果文件：

`protocols/IRSTD-1K/directed_subpixel_vacancy_evidence_factorization_v5/toy_gate_result.json`

- SHA256：
  `664f62cd955b0ac8ec79d271336295ac5ca70ddce50291582f04880ad3acfad2`
- result fingerprint：
  `9f6f17f3465fe471111b679fd323436160252ade4e1a0967bd7748afee6d2bf5`

三次独立进程重放得到逐字节相同的 canonical JSON。命令以退出码 2
表示“计算完整且门禁为负”，不是运行异常。

## 4. 三个 case 的结果

| case | total loss | clean \(D\) | plus target | factual target | 失败项 |
|---|---:|---:|---:|---:|---|
| one pixel | 0.489673 | 0.840815 | 0.499781 | 0.499781 | total loss、plus completion、factual miss target |
| two pixels | 0.381572 | 0.798499 | 0.499634 | 0.499634 | total loss、clean \(D\)、plus completion、factual miss target |
| three pixels | 0.517995 | 0.023706 | 0.499848 | 0.499491 | total loss、clean \(D\)、plus completion、factual miss target |

三个 case 的背景、局部零响应、全局零响应和双端梯度契约大多能够成立；
共同失败点是正目标得分停留在约 0.5，无法达到固定的 0.95 门槛。

## 5. 失败定位

v5 将负半轴设置为精确零响应，同时也使用普通 ReLU 的零梯度。部分需要
学习为正 evidence 的 subpixel phase 在初始化时位于负半轴，例如已经观察
到的 raw phase 值 \(-0.006720\) 和 \(-0.132050\)。这些 phase 的前向响应
为零，局部梯度也为零，因而无法跨过原点进入正 evidence 区域。

因此，v5 的失败链条是：

\[
\text{目标 phase 负初始化}
\rightarrow
\text{负半轴前向为零}
\rightarrow
\text{负半轴梯度为零}
\rightarrow
\text{目标 phase 无法恢复}
\rightarrow
\text{目标概率停留在约 }0.5.
\]

这说明当前 v5 operator 的训练可达性不足；不说明 vacancy factorization、
subpixel output 或 CURE 总方向失败。

## 6. 决定与下一步

v5 冻结为有效 toy/code-gate 负结果。同一版本不再修改、不进入真实
\(D_R\)，也不以调学习率、增加 updates 或降低门槛寻找正结果。

下一候选必须使用全新的 additive 版本，并且只修复已定位的训练可达性：

1. 保持负半轴前向响应精确为零；
2. 保持 v4 正半轴的 evidence 表达能力；
3. 为负初始化 phase 提供明确、冻结的恢复梯度；
4. 不增加模块、参数、head、loss 或推理分支；
5. 先通过同一 frozen toy 3/3，之后才允许一次真实 \(D_R\) bounded。

CURE-Lite → 冻结确认 → Full CURE → 跨 detector/三数据集验证的主线不变。
