# CURE-Lite PFCR：有界证据修正、真实预检与 Formal-800 正式结果

> 日期：2026-07-27  
> 范围：只记录 CURE-Lite PFCR；不进入 Full CURE，不接入其他 IRSTD 前端  
> 证据来源：已封存的 Development、真实 \(D_R\) 预检、seed 42/43 Formal-800、一次冻结的正式 \(D_V\) 揭示，以及揭示后的严格 \(D_R\)-only 失败归因  
> 当前结论：**PFCR 满足保留率及全部误报预算，但两个 seed 均未超过各自最强固定比较器；后续 \(D_R\) 归因进一步证明局部 occupancy relation 在绝大多数正目标处没有输入支持，当前实例正式冻结为负结果**

---

## 0. 当前正式状态

| 项目 | 正式状态 | 证据边界 |
| --- | --- | --- |
| PFCR input contract v2 | PASS | 支持通用 \((F_b,O)\) 输入边界 |
| relation policy v2 | 已冻结 | `phase_query_coverage_key_zero_preserving_softplus_cosine_v2` |
| bounded evidence policy v3 | 已冻结 | `dimension_normalized_bounded_query_energy_with_fixed_ceiling_v3` |
| Development bounded-v3 seed 42/43 | PASS / PASS | 只支持固定 population 上的可学习性 |
| 真实 \(D_R\) 10-update 预检 seed 42/43 | PASS / PASS | 只支持真实缓存训练链可执行且数值有限 |
| 真实 Formal-800 seed 42/43 | COMPLETE / COMPLETE | 每个 seed 均完成 \(800\times40=32{,}000\) 次更新 |
| 最终 decoder artifact | 两个 seed 均已封存 | COMPLETE 清单、权重、日志和 receipt 完整 |
| checkpoint / resume | 禁止且未发生 | 不保存中间模型或优化器状态；不完整 attempt 不可续跑或评估 |
| 正式 \(D_V\) 揭示 | COMPLETE | 只执行一次 create-only 揭示；seed 42/43 顺序评估；\(D_T\) 未读取 |
| seed 42 | 151/170；找回 4/23 | 预算与 147/147 保留通过；相对最强比较器为 \(-3/-3\) |
| seed 43 | 150/170；找回 3/23 | 预算与 147/147 保留通过；相对最强比较器为 \(-2/-2\) |
| CURE-Lite PFCR 性能门禁 | **FAIL** | `PFCR_D_V_GATE_FAIL`；不是训练、完整性或误报预算失败 |
| 严格 \(D_R\)-only 失败归因 | COMPLETE | r2 严格加载；未读取新的 \(D_V\) 或 \(D_T\) |
| 局部 occupancy 支持 | factual 2/32；synthetic 8/206 | 当前 coverage relation 在大多数正目标处无法参与 |
| 冻结确认实验 | **NOT_AUTHORIZED** | 两个 seed 均未达到逐种子 \(+2\) TP / \(+2\) 找回门槛 |
| Full CURE | **NOT_AUTHORIZED** | 本轮没有设计或训练 Full CURE |

这一步完成的是：

```text
真实缓存适配
    -> 谱系安全状态构造
    -> 两个 seed 的数值预检
    -> 两个 seed 的完整 Formal-800
    -> 最终 decoder 与证据包封存
```

随后已经完成：

```text
严格加载最终 decoder
    -> 在冻结 D_V 上生成 completion scores（seed 42: cuda:0；seed 43: cuda:2）
    -> 在预冻结的 51 个阈值与显式 null 中独立校准
    -> 与 12 行冻结比较器证据进行逐 seed 判定
    -> 两个 seed 均未通过性能门禁
```

当前尚未完成、且未获授权的是：

```text
冻结确认种子
    -> D_T
    -> Full CURE
    -> 三数据集或跨 IRSTD 前端验证
```

---

## 1. 旧 Development v1 文档的状态

[旧 PFCR-v2 Development 文档](CURE_Lite_PFCR_v2_模型设计与Development正式结果.md)
应保留为设计演化记录，但不再作为当前实现的数值权威。其中至少有四处已经过期：

1. 旧证据式为

   \[
   E_j=\operatorname{softplus}(\lVert Q_j\rVert_2^2)-\log 2.
   \]

   当前真实训练采用有界、维度归一化的 evidence-v3，旧式不再代表当前 decoder。

2. 旧 Development 表对应较早的训练产物；当前 Formal-800 绑定的是
   `development_bounded_v3_seed42/43_r1.json`。

3. 旧文档的固定预检配置使用 \(d=2\)、1,089 个参数；当前 bounded-v3
   Development 使用 \(d=8\)、4,353 个参数，真实 \(C=64\) decoder 使用
   8,705 个参数。

4. 旧文档把 `checkpoint / resume` 列为真实训练任务；当前正式协议明确规定：
   `continuation_supported=false`、`checkpoint_saved=false`、
   `optimizer_state_saved=false`。

因此，版本语义应写为：

```text
PFCR-v2 = feature–coverage relation 与 release 主线
evidence-v3 = 为真实特征尺度加入的有界证据参数化
training algorithm v2 = 当前 Development 与真实训练所绑定的训练实现
```

这里不是把 PFCR 主线换成第三个模型；而是在保持同一个 feature–coverage
relation 和 evidence-release 方程的条件下，修正真实冻结特征上的尺度问题。

---

## 2. 当前模型主线没有改变

CURE-Lite 的外部推理接口仍只有：

\[
F_b\in\mathbb R^{B\times C\times h\times w},
\qquad
O\in\{0,1\}^{B\times1\times H\times W}.
\]

其中 \(F_b\) 来自任意冻结 IRSTD detector，\(O\) 是该 detector 的 Base
occupancy。PFCR 不读取 detector 名称、GT、目标 ID、角色标签或配对 ID。

当前主线仍是：

\[
(F_b,O)
\rightarrow
\text{phase-resolved feature--coverage relation}
\rightarrow
\text{relevant coverage burden}
\rightarrow
\text{constrained evidence release}
\rightarrow
\Delta O,
\]

\[
O_{\mathrm{final}}=O\lor\Delta O.
\]

单个共享、无 bias 的 \(1\times1\) 投影同时产生 phase query 和 coverage key：

\[
[Q_1,\ldots,Q_{s^2},K]=\Phi(\bar F_b).
\]

关系仍由 phase query 与 \(3\times3\) 方向 coverage key 的平滑 cosine
相关性定义；相关覆盖仍通过 noisy-OR 归约为 \(C_j^{rel}\)；释放门仍为：

\[
V_j=1-C_j^{rel}.
\]

最终仍采用原生 PixelShuffle，不对 tiny-target residual logits 做双线性插值，并以
hard union 保留 Base 检测。

这仍然是一条封闭的状态方程，而不是增加 attention、多尺度、Transformer 或多个
独立修复分支。

---

## 3. 有界 evidence-v3 修正

### 3.1 逐 cell、零保持特征归一化

在共享投影之前，对冻结特征按空间位置归一化：

\[
\bar F_b(x)=
\frac{F_b(x)}
{\max(\lVert F_b(x)\rVert_2,\epsilon)},
\qquad
\epsilon=10^{-6}.
\]

因此零特征仍映射为零，并消除任意正的全局特征缩放对 PFCR 输出的影响。

### 3.2 维度归一化的有界 query energy

当前证据为：

\[
E_j(x)
=
E_{\max}
\frac{\lVert Q_j(x)\rVert_2^2}
{d+\lVert Q_j(x)\rVert_2^2},
\qquad
E_{\max}=10,\quad d=8.
\]

于是：

\[
0\le E_j(x)<10.
\]

释放证据仍为：

\[
\widetilde E_j=E_j(1-C_j^{rel}),
\]

\[
Z_j=-\operatorname{softplus}(\beta)+\widetilde E_j.
\]

这次修正只解决真实冻结特征的 evidence scale；没有增加第二条 decoder
分支，也没有改变 feature–coverage relation、coverage burden、PixelShuffle 或
hard-union 主线。

---

## 4. bounded-v3 Development 权威结果

权威文件：

- [seed 42 r1](protocols/IRSTD-1K/phase_resolved_feature_coverage_relation_v2/development_bounded_v3_seed42_r1.json)
- [seed 42 r2](protocols/IRSTD-1K/phase_resolved_feature_coverage_relation_v2/development_bounded_v3_seed42_r2.json)
- [seed 43 r1](protocols/IRSTD-1K/phase_resolved_feature_coverage_relation_v2/development_bounded_v3_seed43_r1.json)
- [seed 43 r2](protocols/IRSTD-1K/phase_resolved_feature_coverage_relation_v2/development_bounded_v3_seed43_r2.json)

r1/r2 对每个 seed 逐字节一致：

```text
seed 42 SHA256
73e9f49107a048796c4eb35a7c8275f01f31f3826b23bfc3c09aacb220635e94

seed 43 SHA256
d98ba640bbeadb26c1cba590a9e422346baeccddd8ea7bdf439e333befa12ee1
```

冻结设置为 320 updates、Adam、learning rate 0.01、weight decay 0、
\(C=32\)、stride 4、\(d=8\)、4,353 个参数。

| seed | positive probability min ↑ | negative probability max ↓ | threshold mismatch pixels | final logged loss | 结果 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 0.967563510 | 0.019381003 | 0 | 0.342326224 | PASS |
| 43 | 0.969093621 | 0.017244458 | 0 | 0.334254354 | PASS |

这证明 bounded-evidence PFCR 能在固定的 dataset-free population 上被两个初始化
学习；它不证明真实红外数据上的 Pd 或 FA。

---

## 5. 真实 \(D_R\) 输入与 population

两个真实预检和两个 Formal-800 attempt 绑定同一组输入：

```text
cache contract fingerprint
6b4745ebce098c9bed0dc0b3f0aeaf1bc4f6eb6b528c77f117add5b27010bf51

state catalog fingerprint
f6c2bf22ae323897faa2e9918d1eefd54646855a80cee25e4cc69d183946c192

lineage allowlist fingerprint
f815c0aeb30eb7a0d69ca05b493ed3e02f1b826436a1bd37513da8e467257f05
```

真实 \(D_R\) population：

| 项目 | 数量 |
| --- | ---: |
| \(D_R\) samples | 160 |
| factual targets | 32 |
| factual source images | 24 |
| factual-no-miss source images | 135 |
| lineage-safe legal targets | 206 |
| lineage-safe legal source images | 149 |

被谱系门禁排除的三个 legal identity：

```text
("XDU486", 1, 1)
("XDU526", 1, 1)
("XDU965", 1, 1)
```

---

## 6. 两个 seed 的真实 10-update 预检

权威产物：

- [seed 42 result](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_preflight_v1_s42_r1/result.json)
- [seed 42 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_preflight_v1_s42_r1/COMPLETE.json)
- [seed 43 result](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_preflight_v1_s43_r1/result.json)
- [seed 43 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_preflight_v1_s43_r1/COMPLETE.json)

每个 update 同时消费 factual miss、factual no-miss 和 synthetic 三个分支，
batch size 固定为 \(4/4/4\)，12 个 state 只执行一次 decoder forward。

| seed | GPU | updates | final positive min | final negative max | final evidence max | 全部字段有限 | cache 未变 | 预检 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 42 | RTX 3090，CUDA:0 | 10 | 0.011007693 | 0.027412355 | 1.016232014 | true | true | PASS |
| 43 | RTX 3090，CUDA:2 | 10 | 0.011037718 | 0.020714119 | 0.729210496 | true | true | PASS |

预检仅用于确认：

- 真实 cache/state 合同成立；
- 三分支联合更新可执行；
- 每次更新只做一次 decoder forward；
- evidence 保持有界；
- 梯度、参数与输出有限；
- 冻结 cache 不发生变化。

约 0.011 的 positive probability 不是性能结论，也不是预检失败；该阶段没有读取
\(D_V\)，且 receipt 明确记录 `performance_evaluation=false`。

预检 result fingerprint：

```text
seed 42
01442b9413b06b97a928bc38f5967af903bb8d8588b0183fc70b17ed0bd98c80

seed 43
b4612e00babdb735295bbd85ece0cf210e6716f2f19819aa2fb5473f76cf6457
```

---

## 7. Formal-800 冻结协议

两个 seed 的共同训练设置：

| 项目 | 冻结值 |
| --- | --- |
| epochs | 800 |
| steps per epoch | 40 |
| optimizer updates | 32,000 |
| optimizer | Adam |
| learning rate | 0.001 |
| weight decay | 0 |
| numerical precision | FP32 |
| branch batch sizes | factual miss 4 / factual no-miss 4 / synthetic 4 |
| states per update | 12 |
| decoder forwards per update | 1 |
| feature channels | 64 |
| feature stride | 4 |
| relation dimension | 8 |
| trainable parameters | 8,705 |
| runtime split | \(D_R\) only |

训练协议明确禁止恢复：

```text
create_only_output=true
fresh_initialization=true
continuation_supported=false
checkpoint_written=false
intermediate_optimizer_state_written=false
incomplete_attempt_may_be_reused=false
incomplete_attempt_may_be_evaluated=false
complete_written_last=true
```

因此，不存在“从中间 epoch 接着训练”的正式语义。只有完整跑完并最后生成
`COMPLETE.json` 的 attempt 才能进入下一阶段。

---

## 8. Seed 42/43 Formal-800 完成结果

权威目录：

- [seed 42 Formal-800](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_formal_v1_seed42_r1)
- [seed 43 Formal-800](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_formal_v1_seed43_r1)

### 8.1 完整执行量

| 项目 | seed 42 | seed 43 |
| --- | ---: | ---: |
| completed epochs | 800 | 800 |
| optimizer updates | 32,000 | 32,000 |
| backward calls | 32,000 | 32,000 |
| decoder forward calls | 32,000 | 32,000 |
| decoder state evaluations | 384,000 | 384,000 |
| factual-miss state evaluations | 128,000 | 128,000 |
| factual-no-miss state evaluations | 128,000 | 128,000 |
| synthetic state evaluations | 128,000 | 128,000 |
| parameters changed | true | true |
| all trace values finite | true | true |
| all optimizer moments finite | true | true |
| cache unchanged | true | true |
| minimum gradient \(L_2\) norm | 0.439207137 | 0.502962291 |
| maximum gradient \(L_2\) norm | 2.029462814 | 1.722838283 |

两次 attempt 共完成：

\[
64{,}000\ \text{optimizer updates},
\qquad
768{,}000\ \text{decoder state evaluations}.
\]

### 8.2 训练日志首尾

这些是优化日志，不是检测指标。

| seed | epoch | mean factual-miss loss | mean factual-no-miss loss | mean synthetic loss | mean total loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0 | 3.834288198 | 0.373180805 | 3.873504555 | 8.080973554 |
| 42 | 799 | 2.883539510 | 0.614233480 | 3.304292113 | 6.802065098 |
| 43 | 0 | 3.818924338 | 0.312429365 | 3.859341365 | 7.990695083 |
| 43 | 799 | 2.958684796 | 0.602702005 | 3.313535869 | 6.874922669 |

两个 seed 的最后 epoch mean total loss 都低于第一个 epoch，同时完整训练期间没有
非有限 trace 或 optimizer moment。该现象只说明冻结优化过程完整且数值稳定；在
\(D_V\) 评估前，不能解释为 Pd 改善或 FA 降低。

---

## 9. 最终 decoder artifact 与完整性证据

### 9.1 Seed 42

```text
artifact fingerprint
43caebac7454f719c504ef2b648f46944cbcf9cf6b8aea6defa586cb01c1b99a

final model fingerprint
fcf2441ecd1451cec7ff3573902ce72070fa90e4ab9f56fd4fab8b7da1d07181

decoder.safetensors SHA256
59e7a8be0f0cc1cc376fdf2ff8593b0b799d1bfbb7d597782584eb9a8a37263a

train_log.json SHA256
82624eb6e8da97af0ee9f4cbd64068d416344c3f2a123fd7a74aa79b2c05bf33

COMPLETE fingerprint
ed3b9bb448767eb6047203f50de12203e6f2b906375e62e053f2fc1e818aa60f
```

### 9.2 Seed 43

```text
artifact fingerprint
2ff168bf7ee6e6ca84fb37fb585377451e24757d07b90caa3755d560827244a2

final model fingerprint
01e87540cc18695d5ae7b8fcff360dcd31fa94d3048492de933c7581125c23e0

decoder.safetensors SHA256
d7a0d7f72556becf4371ab42f688e2f395945e77f931e9254a8fbb845b497d3c

train_log.json SHA256
0aefb16882188da6999b70eb9bd43154608a76755479d3042b4b943dafe33ecd

COMPLETE fingerprint
4cf683e89077d2ce4df1c78029907c87c953feec0f127c8627d3d6c55df6f299
```

本次整理重新核对了两个 COMPLETE 中列出的全部 inventory SHA256，均与磁盘文件
一致；两个 `train_log.json` 均恰好含 800 条 epoch 记录，最后一条均记录
`optimizer_updates_completed=32000`。

每个 COMPLETE 均记录：

```text
formal_800_by_40_training_complete=true
artifact_strictly_loadable=true
D_V_evaluation_authorized=true
performance_success_claimed=false
full_CURE_authorized=false
```

---

## 10. 一次性正式 \(D_V\) 揭示

权威目录：

- [PFCR Formal \(D_V\) reveal](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_formal_d_v_reveal_v1_r1)
- [正式 decision](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_formal_d_v_reveal_v1_r1/decision.json)
- [seed 42 result](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_formal_d_v_reveal_v1_r1/results/PFCR_seed42.json)
- [seed 43 result](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_formal_d_v_reveal_v1_r1/results/PFCR_seed43.json)
- [COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_formal_d_v_reveal_v1_r1/COMPLETE.json)

正式绑定：

```text
config fingerprint
257dda08bb6bb286a3b95f1949b9a45296646d036b24493bb88ec3394d372dd7

comparison protocol fingerprint
cb2fb09c3ec7dbbb0f057d94f7f159e2b4a733296e6ea4a144d6302387014884

decision fingerprint
5cd371fc25a9b8890ea7db37a5fa8e3b91dd33df204af7e46cd461a09c9cf0e2

COMPLETE fingerprint
95636a5f813037c61a2831a99aac090e37ea549f7e2ca4704d27e9b0f7e69b9c
```

严格加载器重新核验了完整目录清单、全部文件 SHA256、两个正式训练 attempt、
12 行冻结比较器证据、14 行最终决策证据和 COMPLETE-last 语义。揭示只物化一次
\(D_V\)，没有读取 \(D_T\)，没有 resume、overwrite 或目录复用。

### 10.1 逐 seed 结果

| seed | device | threshold | true targets | Pd | recovered / 23 | retention | pixel FA | raw-background FA | FP components / MP | mIoU | nIoU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | cuda:0 | 0.08 | 151/170 | 0.888235 | 4 | 1.0 | \(4.6031\times10^{-5}\) | \(9.7148\times10^{-5}\) | 16.5304 | 0.575102 | 0.511133 |
| 43 | cuda:2 | 0.08 | 150/170 | 0.882353 | 3 | 1.0 | \(4.4250\times10^{-5}\) | \(9.5367\times10^{-5}\) | 15.8946 | 0.577171 | 0.516782 |

两个 seed 都满足：

```text
retention = 1
pixel FA <= 1e-4
raw-background FA <= 1e-4
FP components / MP <= 100
budget_violation = false
```

因此，本轮不是 Base 保留失败，也不是误报预算越界。

### 10.2 与冻结比较器的完整目标级比较

| seed | Base@B | F | F× | U | paired difference | independent endpoint | PFCR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 true / recovered | 150 / 3 | **154 / 7** | 149 / 2 | 151 / 4 | 147 / 0 | **154 / 7** | 151 / 4 |
| 43 true / recovered | 150 / 3 | **152 / 5** | 147 / 0 | **152 / 5** | **152 / 5** | **152 / 5** | 150 / 3 |

预冻结成功门槛要求每个 seed 同时满足：

\[
TP_{\mathrm{PFCR}}\ge TP_{\mathrm{best}}+2,
\qquad
R_{\mathrm{PFCR}}\ge R_{\mathrm{best}}+2.
\]

实际逐 seed ledger：

| seed | 最强比较器 | PFCR 要求 | PFCR 实际 | 实际相对最强比较器 | 判定 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 154 / 7 | 至少 156 / 9 | 151 / 4 | \(-3/-3\) | FAIL |
| 43 | 152 / 5 | 至少 154 / 7 | 150 / 3 | \(-2/-2\) | FAIL |

所以正式总决定为：

```text
status = PFCR_D_V_GATE_FAIL
all_seeds_pass = false
next_action = STOP_AND_PRESERVE_EVIDENCE
authorizes_frozen_confirmation = false
authorizes_full_cure = false
authorizes_cross_backbone = false
D_T_accessed = false
```

### 10.3 结果的准确含义

可以正式确定：

1. 通用 \((F_b,O)\) 接口、PFCR decoder、真实缓存训练、Formal-800、校准、
   评估和完整性封存均已真实运行；
2. PFCR 的 hard union 在两个 seed 中都保留了 147/147 个 Base-covered targets；
3. PFCR 可以在预算内释放部分 residual evidence，但没有形成优于固定比较器的
   稳定目标排序；
4. seed 42 的 PFCR 与 U 都是 151/4，但 PFCR 的 pixel FA、raw-background FA
   和 component FA 都更高；
5. seed 43 的 PFCR 只达到 Base@B 的 150/3，同时 component FA 高于全部主要
   比较器；
6. 当前 PFCR relation/evidence-release 实例没有通过 CURE-Lite 性能门禁。

当前产物只保存了冻结校准后选择点，不包含足以完成因果归因的全量 target/background
score 分布和逐目标空间响应。因此，现阶段只能把“残差排序不稳定、空间响应更碎片化”
列为与结果一致的诊断方向，不能把它写成已经证明的唯一失败原因。

本结果否定的是当前：

```text
PFCR-v2 relation
+ bounded evidence-v3
+ 当前三分支训练目标
+ 当前冻结训练协议
```

的组合候选。它不等于否定通用 CURE 研究目标，但也不能把“主线仍有意义”替代当前
候选的正式性能失败。

### 10.4 严格 \(D_R\)-only 失败归因

正式揭示后没有再次读取 \(D_V\)，而是严格加载原有 \(D_R\) cache 与两个冻结
decoder，对 32 个 factual miss、135 个 factual-no-miss state 和 206 个
lineage-safe synthetic target 进行前向归因。

权威产物：

- [r2 result](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_d_r_failure_attribution_v1_r2/result.json)
- [r2 COMPLETE](runs/irstd1k_stage_a_seed42/cure_lite_pfcr_real_d_r_failure_attribution_v1_r2/COMPLETE.json)

```text
result fingerprint
e23cf378d28468d884e8a221e9537e4e23e293943fd5087e1f3eb2ebd3af45f8

COMPLETE fingerprint
dcf6dac9eed683c90c67c3b1df1ff79212a87b62ee1970dd8cf2a2e7cbcb3eb0

result.json SHA256
1ac3d9fa2df8a8475dd8c937cfeb2986bcc821a340e23bfe4c899aed55efaaf8

COMPLETE.json SHA256
557bd5fedde86f51e78438bd25a810de13242d06e2cf0a9ce44ff9c326c7d494
```

r1 保留为工程历史，但其加载器允许不完整、自签名对象通过，因此不再作为权威
证据。r2 使用全新 create-only 目录，不覆盖、不续接 r1，并新增：

- 严格 \(D_R\) cache 与 state catalog 身份绑定；
- 两个正式 decoder 的磁盘重新加载和 seed/receipt/artifact 交叉验证；
- 确定性后端、GPU 0/2 与实现文件哈希绑定；
- 精确顶层及嵌套字段验证；
- target 数量、逐行 identity、branch summary 和 seed consistency 重算；
- COMPLETE-last、文件 SHA256、内存变更和目录 inventory 检查。

#### 局部 occupancy 支持

此处只测量 target 所在 feature cell 的方向 occupancy basis 是否非零，准确名称是
`local_occupancy_support`；它不是 learned affinity 或因果参与声明。

| population | 具有任一局部支持 | 零局部支持 | active cells / positive cells | target-macro active fraction |
| --- | ---: | ---: | ---: | ---: |
| factual miss | 2/32 | 30/32 | 21/113 = 0.185841 | 0.040572 |
| synthetic | 8/206 | 198/206 | 15/576 = 0.026042 | 0.024272 |

因此在 93.75% 的 factual miss 和 96.12% 的 synthetic target 上，当前局部
coverage basis 为零；在这些位置，已实现的 coverage burden 必然为零，logit
退化为 baseline 与 feature-derived evidence 的组合。这个结果不表示目标无法由
feature evidence 检出，但证明当前声称的 feature–coverage relation 在大多数正
目标排序中没有实际输入条件。

#### 冻结 decoder 的 target/background 排序

| population | seed | target max \(>\) background q99.9 | target min \(>\) background max |
| --- | ---: | ---: | ---: |
| factual miss | 42 | 32/32 | 24/32 |
| factual miss | 43 | 32/32 | 25/32 |
| synthetic | 42 | 206/206 | 56/206 |
| synthetic | 43 | 206/206 | 54/206 |

两个 seed 在 factual `target max - background q99.9` margin 上的 Spearman
相关为 0.990103，符号一致 32/32；在 `target min - background max` margin 上
的 Spearman 相关为 0.997067，符号一致 31/32。两个 seed 共同未通过完整
target-vs-background 极值分离的七个 identity 为：

```text
("XDU309", 2)
("XDU325", 1)
("XDU325", 2)
("XDU680", 1)
("XDU774", 1)
("XDU865", 2)
("XDU980", 1)
```

这说明两个独立初始化稳定学到了相近但不足的规则：目标处通常存在强 residual
峰值，但峰值没有稳定扩展为紧致、完整且能压过同图极端背景的目标支持。它与正式
\(D_V\) 上“只找回少数目标且 component FA 偏高”一致，但该归因仍是诊断证据，
不能单独宣称唯一因果原因。

归因产物明确记录：

```text
diagnostic_only = true
causal_failure_attribution_established = false
replacement_equation_authorized = false
new_formal_training_authorized = false
D_V_read = false
D_T_read = false
full_CURE_authorized = false
cross_backbone_authorized = false
```

---

## 11. 当前可以和不能作出的结论

### 11.1 可以正式作出的结论

1. 当前 PFCR 已建立完整、可严格加载、可真实评估的 detector-independent
   CURE-Lite 工程链。
2. 两个 seed 均从全新初始化完成 800×40 训练，未使用 checkpoint 或 resume。
3. 当前 PFCR 满足 Base 全保留及全部误报预算。
4. 当前 PFCR 在两个 seed 上都没有超过最强固定比较器，且未达到预声明的
   \(+2/+2\) 逐 seed 门槛。
5. CURE-Lite 当前模型尚未设计成功；Full CURE、\(D_T\)、三数据集和跨前端验证
   均未获授权。

### 11.2 当前不能作出的结论

1. 不能宣称 PFCR 提高了 Pd、IoU 或 nIoU。
2. 不能宣称 PFCR 降低了 FA；两个 seed 的 component FA 反而高于主要比较器。
3. 不能根据训练 loss 下降或 Development PASS 宣称真实检测机制成立。
4. 不能把失败唯一归因于阈值、feature–coverage relation、evidence 参数化、
   synthetic state 或某一 loss 项。
5. 不能在已经读取的 \(D_V\) 上调 PFCR 阈值、结构或超参数后继续把同一结果称为
   冻结验证。

---

## 12. 下一阶段：仍然只做 CURE-Lite

当前正式结果与 r2 归因必须冻结，不重跑 seed 42/43，也不在 \(D_V\) 上调参。
PFCR failure attribution 已经完成，下一项代码工作转为一个统一的新 CURE-Lite
结构定义：

```text
冻结 PFCR 负结果
    -> 保留“冻结前端证据 + Base hard union”研究问题
    -> 停止当前局部二值 occupancy 乘性 relation
    -> 冻结一个直接生成结构化、紧致 correction support 的单一学习对象
    -> 先实现表示能力、梯度、空状态和 hard-union 代码门禁
    -> 再执行真实 D_R 训练门禁
```

新结构必须满足：

- 仍由任意冻结 IRSTD 的通用输出驱动，不读取 detector 名称或内部专有层；
- 仍解决“在不破坏 Base covered targets 的条件下释放缺失目标证据”；
- 只有一个结构化预测对象，不能叠加独立 attention、Transformer、多尺度或多个
  修复分支；
- 必须利用归因中已经存在的 target-ranking signal，同时直接约束连通支持和空图
  输出，而不是再次产生无约束像素峰；
- 在新的 dataset-free 与 \(D_R\)-only 门禁通过前，不运行新的 \(D_V\)、
  \(D_T\)、Full CURE 或其他 backbone。

当前最准确的项目状态是：

> CURE-Lite PFCR 的完整模型代码、真实训练和正式评估链已经实现，但当前方程的真实
> 性能未通过逐 seed 门禁；严格 \(D_R\) 归因也确认局部二值 occupancy relation
> 在绝大多数正目标处没有输入支持。项目继续停留在 CURE-Lite 模型设计与代码实现
> 阶段，下一步是一个直接学习结构化 correction support 的统一模型，而不是开始
> Full CURE 或跨模型验证。
