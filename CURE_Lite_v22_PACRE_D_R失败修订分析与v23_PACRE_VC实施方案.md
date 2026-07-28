# CURE-Lite v22 PACRE \(D_R\) 失败修订与 v23 PACRE-VC 最终方案

> 性质：执行前方案与审计合同  
> 候选：PACRE-VC-v23  
> 阶段：看到 v22 聚合 \(D_R\) 失败后的自适应验证器修订  
> 正式训练：seed 42，800 epochs × 40 steps = 32,000 updates  
> 本文不声明尚未生成的 receipt、门禁结果、训练结果或性能结果

---

## 1. 最终判断

本轮选择 **PACRE-VC（PACRE Verifier-Corrected）**，不选择 FACRE。

PACRE-VC：

- 保持 v22 PACRE 的全部数值 forward、参数、初始化与 PMOPE/Adam 语义；
- 只修正 v22 \(D_R\) 中不适合直接作为 FP32 门禁的代数验证方式；
- 使用独立 v23 config、model FQCN 与 exact-type factory，隔离新旧 receipt。

FACRE 可能是后续新模型方向，但它会改变实际模型或 forward，不能回答本轮
最窄的归因问题：

> v22 的代数门禁失败是 PACRE 机制失败，还是验证器错误地把合法的 FP32
> 消去/重算差异判成模型失败？

只有保持模型不变的 PACRE-VC 能直接回答这个问题。

---

## 2. v22 失败的归因边界

### 2.1 封存事实不可改写

v22 的 `attempt.json`、receipts、`COMPLETE.json` 与 FAIL decision 必须保持
原样。v23 只能继承并重新核验这些产物，不能把新解释回填成“v22 已通过”。

v23 是看到 v22 失败后提出的，因此是：

- adaptive verifier correction；
- 非独立确认；
- 可用于失败归因；
- 不可伪装为事前注册的独立复现。

### 2.2 为什么旧代数门禁可能是假失败

旧检查试图从已舍入 FP32 下游量反推出微小上游 residual，或重复执行并假设
raw-bit 恒等。浮点运算不满足实数代数的可逆性：

\[
\operatorname{fl}_{32}\!\left(
\operatorname{fl}_{32}(a+b)-a
\right)\neq b
\]

并不自动说明产生 \(a+b\) 的 forward 错误。v23 因而改为验证 forward 实际
保存、实际消费的表达式，不再把病态逆减法当核心判据。

### 2.3 \(D_R\) 结论上限

即使 v23 \(D_R\) 13/13 PASS，也只支持：

1. v22 的该项失败可归因于旧验证器的 FP32 代数假设；
2. 修复该失败不需要改变 PACRE forward；
3. 候选具备进入 Formal-800 的结构资格。

它不等于训练完成，更不等于 \(D_V/D_T\) 性能成功。

---

## 3. v23 模型与 parity 合同

### 3.1 数值 forward、参数与 PMOPE 等价

PACRE-VC-v23 必须保持：

- 相同输入 \((F_b,O)\)、phase、centered residual 与 flip construction；
- 相同 common/specific compatibility、hidden、energy、interaction 和 field；
- 相同 hard union 与固定零水平集解码；
- 相同参数初始化、数量、dtype 与 requires-grad；
- 参数名仍为 `joint_state_weight`、`joint_hidden_bias`、
  `scalar_energy_weight`；
- `forward_fields` 精确返回
  `cure_lite_v22.pacre.CoverageStatePACREFields`；
- 相同 PMOPE target preparation、pair loss、Adam 超参数与 step 顺序；
- verifier 诊断量不参与反向传播。

v23 只增加：

- `CoverageStatePACREVerifierCorrectedConfig`；
- `CURELitePACREVerifierCorrectedLevelSet`；
- exact-type factory；
- verifier policy 标识。

### 3.2 Generated-only parity

读取新的真实 \(D_R\) tensor 前，必须在 seeds 42/43/44、CPU 与逻辑
`cuda:0` 上验证：

- raw model state parity；
- raw `forward_fields` parity；
- raw gradient parity；
- 3-step Adam/PMOPE parity；
- RNG 与 deterministic 状态保持。

generated input 只能由独立 CPU `torch.Generator` 生成、指纹化后再搬到设备。
任何 parity 失败都说明 wrapper 改变了语义，必须停止。

---

## 4. 修正后的代数验证器

### 4.1 决策原则

验证器只接受：

1. 同设备、同 dtype、同运算图可直接重放的 raw-exact 表达式；
2. 重复执行 transcendental kernel 时事前冻结的局部解析误差界。

旧逆向减法、FP64/FP32 误差幅度和 signal-swallow 数量仅用于归因，不能决定
模型 PASS/FAIL。

### 4.2 15 个 forward-expression 子检查

总数固定为 **15 = 13 raw exact + 2 SiLU analytic-bound**。

| 类别 | 名称 | 判据 |
|---|---|---|
| exact | `phase_mean_forward_exact` | same-device raw equality |
| exact | `phase_residual_forward_exact` | same-device raw equality |
| exact | `flip_delta_forward_exact` | same-device raw equality |
| exact | `flipped_occupancy_affine_forward_exact` | same-device raw equality |
| exact | `actual_common_forward_exact` | same-device raw equality |
| exact | `actual_specific_forward_exact` | same-device raw equality |
| exact | `flipped_common_forward_exact` | same-device raw equality |
| exact | `flipped_specific_forward_exact` | same-device raw equality |
| exact | `actual_energy_forward_exact` | same-device raw equality |
| exact | `flipped_energy_forward_exact` | same-device raw equality |
| exact | `native_interaction_forward_exact` | same-device raw equality |
| exact | `native_field_forward_exact` | same-device raw equality |
| exact | `output_field_forward_exact` | same-device raw equality |
| bounded | `actual_hidden_forward_bounded` | frozen SiLU bound |
| bounded | `flipped_hidden_forward_bounded` | frozen SiLU bound |

13 个 exact 项的 `bound = 0`。两个 hidden 项使用：

\[
B_H=4\epsilon_{32}
\left(1+|\operatorname{SiLU}(J_s)|
+|\operatorname{SiLU}(J_c)|\right)
+\mathrm{TINY}_{32}.
\]

该界只覆盖同值 tensor 因 layout/vectorized kernel 不同而重复执行 SiLU 的
合法 FP32 差异，不是宽泛 `allclose`。

### 4.3 Phase semantics 与负向测试

phase reconstruction、phase centering 使用
`cure_lite_v23/algebra_verifier.py` 中冻结的 FTZ-safe 解析界：

- error/bound 在 canonical CPU FP64 中审计；
- bound 必须有限、非负；
- reconstruction 与 centering 分开记录；
- 不以经验 tolerance 替代解析界。

分别向 stored residual、hidden 或 energy 注入 `1e-3` 时，对应 replay/bound
必须失败；否则验证器不得进入真实 \(D_R\)。

### 4.4 Diagnostic-only

- v22 六个旧 algebra subcheck：原公式与 tolerance 完整重放，decision weight
  固定为 0；
- FP64 oracle：构造完整性可 gate，误差幅度不可 gate；
- signal-swallow：保存 count、幅度和坐标，但不要求 count 为 0；
- fixed readout：CPU FP32 `linspace(0.5, 1.5, width)` 后复制到设备，仅用于
  非零功能诊断，不改变参数。

---

## 5. Dataset-free gate：总计 22 项

dataset-free gate 固定为 **22 checks**，不是 25：

- checks 01--13：继承并重新核验 v22 dataset-free 合同；
- checks 14--22：v23 新证据。

| 编号 | v23 新检查 |
|---:|---|
| 14 | v22/v23 forward、field、gradient、optimizer parity |
| 15 | scalar cancellation counterexamples reproduced |
| 16 | formal-shape CPU algebra stress |
| 17 | formal-shape selected-device algebra stress |
| 18 | phase reconstruction bound valid |
| 19 | phase centering bound valid |
| 20 | legacy subtraction is diagnostic-only |
| 21 | FP64 oracle and swallow ledger complete |
| 22 | runtime environment frozen |

formal-shape 固定为 batch 1、channels 64、stride 4、width 32、feature
\(64\times64\)、occupancy/output \(256\times256\)、FP32、seeds 42/43/44、
CPU 与逻辑 `cuda:0`。

此阶段不得读取 \(D_R/D_V/D_T\) tensor、不得构造真实 cache、不得训练、
不得使用 checkpoint。只有实际 receipt 的 22/22 PASS 才能授权 \(D_R\)；
本文不预声明它已通过。

---

## 6. 单个三层 source closure 与 runtime lock

### 6.1 一个 closure，三个逻辑层

只生成一个 `implementation_closure.json`，内部含：

1. `root`：`cure_lite/**/*.py` 与根 `pyproject.toml`；
2. `v22`：`cure_lite_v22/**/*.py` 与其 `pyproject.toml`；
3. `v23_and_runners`：`cure_lite_v23/**/*.py`、其 `pyproject.toml`、
   官方 v23 tools、`tools/__init__.py`、温控 wrapper 和 preregistration。

closure 保存全局及分层文件清单、SHA-256、file count、binding fingerprint 与
closure fingerprint。任一纳入文件改变后，旧 generated receipt 不得继续授权；
必须重建 closure 和所有受影响证据。本文不编造尚未生成的 fingerprint。

### 6.2 统一 runtime

所有官方命令固定前缀：

```text
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
```

解释器固定为：

```text
/home/md0/ly/MSHNet/.venv/bin/python
```

runtime receipt 至少绑定 Python/PyTorch/CUDA/cuDNN、GPU identity、
deterministic switches、TF32、matmul precision、threads、上述环境变量与
subnormal/FTZ probe。独立 verifier 也必须使用相同前缀，因为它会重建 live
runtime/source binding。

---

## 7. 唯一真实 \(D_R\) 结构门禁

### 7.1 固定输入与 run ID

唯一 run ID：

```text
pacre_v23_verifier_corrected_D_R_structural_r1
```

输出：

```text
runs/irstd1k_stage_a_seed42/pacre_v23_verifier_corrected_D_R_structural_r1
```

规模固定为：

- 32 target states；
- 96 context states；
- 合计 128 条 scope-qualified forward ledger；
- 原 source-state ID 仍保留用于封存回放。

总门禁为 13 项，PASS 条件严格为 **13/13**：

| # | 检查 |
|---:|---|
| 01 | dataset-free prerequisite exact and passed |
| 02 | real \(D_R\) seed-42 population bound |
| 03 | exact v23 config/factory/parameter contract |
| 04 | complete ledger and exact v22 fields type |
| 05 | 32 target-state forward algebra and phase semantics |
| 06 | each target group has bound-residual/flip latent witness |
| 07 | no exact target-positive latent collision |
| 08 | zero-readout anchor and fixed-readout witness |
| 09 | real PMOPE initialization gradient path |
| 10 | field-loss direction correct for all roles |
| 11 | model/population/cache/RNG/grad buffers preserved |
| 12 | read-only zero-update \(D_R\) scope |
| 13 | 96 context-state forward algebra and phase semantics |

### 7.2 只读与数据边界

\(D_R\) 不构造 optimizer，optimizer steps = parameter updates = 0，不训练，
不读取 \(D_V/D_T\) tensor payload。

构造 \(D_R\) 绑定会读取并校验项目 split manifest metadata，因此 receipt
必须区分：

- `split_manifest_metadata_read = true`；
- `D_V_tensor_payload_accessed = false`；
- `D_T_tensor_payload_accessed = false`。

manifest 中存在 \(D_V/D_T\) 条目不等于访问其 tensor payload。

v23 还必须重算 v22 artifact graph、初始模型 fingerprint、有序 32 target
state-ID fingerprint 与 live target ledger；不得信任存储的 `matches: true`。

### 7.3 决策

- 13/13 PASS：仅允许创建独立 Formal-800 authorization；
- 任一 FAIL：封存 negative result 并停止；
- 不改阈值、不 retry、不 resume、不覆盖同名目录；
- 无论结果如何，本阶段都不执行 bounded-400。

本文只冻结 13/13 判据，不声明该 run 已执行或已通过。

---

## 8. Append-only artifact graph

### 8.1 通用发布规则

1. `mkdir(..., exist_ok=False)` 抢占唯一目录；
2. 立即 create-only 写 `.incomplete` 与 `attempt.json`；
3. receipts 均以 create-only、flush、fsync 写入；
4. 正常终止（包括可审计的科学 negative decision）时最后写
   `COMPLETE.json`，再移除 `.incomplete`；
5. 执行异常或产物图未完成时保留 `.incomplete`，create-only 写
   `FAILURE.json`；
6. 禁止 overwrite、retry、resume、symlink、special file 和额外文件。

正常终态只有 `attempt.json`、白名单 receipts/产物与 `COMPLETE.json`；
其中 decision 可以是 PASS，也可以是封存的科学 FAIL。执行异常终态才保留
`.incomplete`、已写产物与 `FAILURE.json`。

\(D_R\) 成功白名单核心为：

```text
attempt.json
receipts/inputs.json
receipts/preflight.json
receipts/dr_gate.json
receipts/decision.json
COMPLETE.json
```

独立 verifier 必须严格校验文件白名单、canonical JSON、自指纹、artifact
hash map、receipt 关联和 post-run authorization；不得重新打开 \(D_R\) tensor。

### 8.2 Formal-800 预期入口

```text
tools/run_cure_lite_v23_pacre_vc_formal_800.py
tools/verify_cure_lite_v23_pacre_vc_formal_800_receipt.py
```

正式运行前必须冻结其 exact whitelist、schema 与 source closure。本文不为
尚不存在的 Formal receipt 编造 hash、fingerprint 或运行结果。

---

## 9. Formal-800 训练合同

run ID：

```text
cure_lite_pacre_v23_vc_pmope_formal_800_seed42_r1
```

预算：

\[
800\ \text{epochs}\times40\ \text{steps/epoch}
=32{,}000\ \text{optimizer updates}.
\]

必须同时满足：

- seed 42，from scratch；
- 不复用 v21/v22/v23 checkpoint；
- 完整冻结 \(D_R\) training scalar cache；
- 冻结 Formal-800 exposure schedule；
- exact v23 config/factory；
- 继承 Adam/PMOPE；
- 只保存 final model，不做中间 checkpoint selection；
- 不 retry、resume、overwrite 或 threshold search；
- 不访问 \(D_V/D_T\) tensor payload。

只有独立 verifier 确认 \(D_R\) terminal 13/13 PASS 后，才能创建
Formal authorization。authorization 必须绑定：

- terminal \(D_R\) 与 dataset-free receipts；
- runtime lock 与三层 source closure；
- full real-input cache 与 Formal schedule；
- seed-42 initial model、exact config/factory；
- 唯一输出路径与 process-local claim/consume。

receipt 中一个 `eligible: true` 布尔值不能替代上述重建。

### 9.1 不运行 bounded-400

\(D_R\) PASS 后直接进入 Formal-800。bounded-400：

- 不是前置门禁；
- 不是最终证据；
- 本次不执行；
- 不得以旧固定门槛阻止已通过 \(D_R\) 的预注册 Formal-800。

### 9.2 训练完成不等于性能

Formal `COMPLETE.json` 最多证明 32,000 次更新、final checkpoint 与 artifact
graph 按合同完成。它不证明 true targets、recovered misses、mIoU/nIoU、
retention 或 FA 改善；性能必须由后续单独授权 evaluation 产生。

---

## 10. 相对性能合同

### 10.1 取消任意 \(+2\) uplift margin

不再要求“至少比 Base 多 \(+2\) 个 target”。该数值并非任务定义、功效分析、
最小工程效应量或安全边界，会把真实可复现的 \(+1\) 误判为失败。

新原则是：

> 相对事前冻结的 best Base 严格正提升即可；\(+1\) 可以通过，但不得以
> segmentation 回退或背景 false alarm 为代价。

### 10.2 后续 \(D_V\) 判据

在单独冻结 comparator、evaluator、decode rule 与 authorization 后，候选需
同时满足：

1. `true_targets` 严格大于冻结 best Base；
2. `recovered_anchor_misses` 严格大于冻结 best Base；
3. mIoU 不低于其冻结 best Base；
4. nIoU 不低于其冻结 best Base；
5. retention = 1.0；
6. 所有冻结 FA operating-safety budget 满足。

若沿用当前冻结安全合同，包括：

- pixel false-alarm rate \(\le 10^{-4}\)；
- raw-background false-alarm rate \(\le 10^{-4}\)；
- false components per megapixel \(\le 100\)。

固定零水平集是模型解码定义；exact/analytic bound 是验证器正确性合同；
retention/FA 是运行安全约束。它们都不是任意的性能 \(+2\) 门槛，不应取消，
也不得在看到 v23 结果后放宽。

“best Base”必须在候选 evaluation 前冻结 comparator 集合、Base checkpoint、
metric 方向、split 与 pre/post-processing；不得在看到候选结果后换 Base 或
为不同指标临时选择更有利的比较对象。

### 10.3 \(D_V\) 自适应，\(D_T\) 最终确认

项目级 \(D_V\) 已被 v21 揭示，因此任何 v23 \(D_V\) 都是 adaptive evidence，
可用于工程选择与失败分析，但不能作为独立确认。

\(D_T\) 保留作后续一次性确认：先冻结 final candidate、evaluator 和相对
性能合同，再授权一次访问；不得在 \(D_T\) 失败后回到 \(D_T\) 调参。

---

## 11. 执行顺序与官方命令

先进入工作目录：

```bash
cd /home/md0/ly/cure_lite
```

### 11.1 测试

测试清单以 live repository 为准；不得在方案中虚构不存在的文件：

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python -m pytest -q \
tests/test_cure_lite_v23*.py \
tests/test_run_cure_lite_v23*.py \
tests/test_verify_cure_lite_v23*.py
```

### 11.2 生成并验证 generated-only 证据

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/run_cure_lite_v23_pacre_vc_preflight.py --run-generated
```

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/run_cure_lite_v23_pacre_vc_preflight.py --validate-generated
```

### 11.3 最后创建 \(D_R\) authorization

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/run_cure_lite_v23_pacre_vc_preflight.py --authorize-dr
```

### 11.4 唯一 \(D_R\) 调用及独立验证

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/run_cure_lite_v23_pacre_vc_dr_gate.py --run-once
```

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/verify_cure_lite_v23_pacre_vc_dr_receipt.py --verify
```

只有 verifier 确认 13/13 PASS 才继续。

### 11.5 唯一 Formal-800 调用

Formal runner、独立 verifier、D_V runner/verifier 全部完成实现、测试与
source freeze 后：

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/run_with_gpu_temperature_control.py \
--gpu 0 --pause-temp 82 --resume-temp 75 -- \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/run_cure_lite_v23_pacre_vc_formal_800.py --run-once
```

温控 pause/continue 不改变 epoch、step、update、optimizer 或 schedule，
也不提供 retry/resume。

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/verify_cure_lite_v23_pacre_vc_formal_800_receipt.py --verify
```

Formal verifier 通过后只能声明训练/artifact 完成，不能声明性能通过。
它还必须为该次独立验收实际加载的 exact artifact 签发进程内终态 identity
seal；后续 \(D_V\) model binding 只接受这个 sealed terminal，不接受裸
artifact、通用 training ledger 或调用方自行拼装的 receipt。

### 11.6 固定 adaptive \(D_V\) 相对性能评估

先做不读取 \(D_V\) tensor 的 create-only 验证：

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/run_cure_lite_v23_pacre_vc_formal_d_v.py --validate-create-only
```

随后只允许一次固定运行：

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/run_cure_lite_v23_pacre_vc_formal_d_v.py --run-once
```

最后独立验收；verifier 不重开 \(D_V\) tensor，而是从持久化的全部 51 个
Base@B aggregate 条目独立重选对照并重算相对门禁：

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
/home/md0/ly/MSHNet/.venv/bin/python \
tools/verify_cure_lite_v23_pacre_vc_formal_d_v_receipt.py --verify
```

该结果必须标为 adaptive；无论 PASS/FAIL，本流程都不访问或授权 \(D_T\)。

---

## 12. Stop rules 与状态

| 阶段 | PASS 后 | FAIL/异常后 |
|---|---|---|
| source/runtime/parity/22-check | 最后创建 \(D_R\) authorization | 不读真实 \(D_R\)，修订后重建 closure |
| \(D_R\) 13 checks | 独立验证；创建 Formal authorization | 封存 negative；不训练、不 retry |
| Formal-800 | 独立验证；训练与性能状态分离 | 保留 `.incomplete`+`FAILURE.json`；不 resume |
| adaptive \(D_V\) | 按相对 best Base 报告，保留 \(D_T\) | 不改 comparator/阈值/安全预算 |
| one-shot \(D_T\) | 最终确认 | 如实报告，不回到 \(D_T\) 调参 |

当前合同/结果必须区分：

| 项目 | 已冻结合同 | 本文声明的结果 |
|---|---|---|
| 方法选择 | PACRE-VC，不采用 FACRE | 不涉及运行 |
| forward/参数/PMOPE | 与 v22 数值等价 | 以 parity receipt 为准 |
| verifier | 15 = 13 exact + 2 bounded | 以实际 receipt 为准 |
| dataset-free | 总计 22 checks | 不预声明 22/22 |
| \(D_R\) | 32 target + 96 context，13/13 才 PASS | 不预声明已运行/PASS |
| bounded-400 | 从本次流程移除 | 不执行 |
| Formal-800 | seed 42、from scratch、800×40 | 不预声明完成 |
| \(D_V\) | 相对 best Base 严格正提升，\(+1\) 可过 | 尚无 v23 性能声明 |
| \(D_T\) | 最终一次性确认 | 保留未访问 |

---

## 13. 最终执行摘要

1. 冻结 v22 FAIL，不改写历史；
2. 用 PACRE-VC 修验证器，不用 FACRE 改 forward；
3. 证明 v22/v23 forward、参数、gradient 和 3-step Adam/PMOPE 等价；
4. 使用 15 项 replay：13 raw exact + 2 bounded SiLU；
5. 完成 22 项 generated-only gate；
6. 唯一 \(D_R\)：32 target + 96 context，严格 13/13；
7. \(D_R\) 通过后不跑 bounded-400，直接授权 Formal-800；
8. seed 42 from scratch，800 epochs × 40 = 32,000 updates；
9. Formal COMPLETE 不等于性能；
10. 取消任意 \(+2\)，相对 best Base 严格正提升，\(+1\) 可通过；
11. 同时要求 mIoU/nIoU 不回退、retention 与 FA 安全预算满足；
12. v23 \(D_V\) 标为 adaptive，\(D_T\) 才是最终确认。

这把三个问题严格分开：

- **验证器是否正确**：generated-only 与真实 \(D_R\)；
- **训练是否按合同完成**：Formal-800；
- **性能是否提高**：单独授权的 adaptive \(D_V\)，最终由 \(D_T\) 确认。
