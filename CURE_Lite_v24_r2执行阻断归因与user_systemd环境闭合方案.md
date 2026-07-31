# CURE-Lite v24 GCR-PACRE：r2 执行阻断归因、user-systemd 闭合与后续科学验证方案

> 日期：2026-07-30  
> 仓库：`Arialliy/cure-lite`  
> 当前候选：`GCR-PACRE-v24`  
> 当前状态：数学模型、核心实现、dataset-free、训练器、协议与下游 verifier 已完成；真实科学执行尚停在 \(D_R\)-structural 之前  
> 当前实际阻断：目标 user-systemd unit 尚未实现，user manager 为 degraded，存在两个 `Restart=on-failure` 循环服务，其中一个持续请求 `cuda:0`  
> 审定状态：**方案方向成立，但原稿含执行越权、收据循环依赖、systemd 动静态状态混用和 GPU TOCTOU 等 P0；以下文本已按本机只读审计结果修订，修订闭合前禁止 actual r2。**  
> 推荐决策：**若继续 v24，则保持同候选 r2，只修执行层；同时保留“终止 v24、另立 v25”的合法并行选择。r2 是检验 v24 的路径，不是科学上必须完成的路径。**

---

# 0. 结论先行

当前不能把状态写成：

> “v24 模型失败。”

也不能写成：

> “v24 已经通过 \(D_R\)-structural，只差启动训练。”

准确结论是：

> **v24 的候选设计和静态/生成式工程链已完成，但真实 r2 尚未形成可授权的运行环境，因此没有新的 \(D_R\)-structural 科学决定。**

当前阻断发生在三层身份中的**执行身份**：

| 身份层 | 当前状态 |
|---|---|
| 科学主线 | 已确定：Base anchor、单一零水平集 completion、解决 common–residual compatibility，并以相对有效 Base 的严格提升/非回退和安全约束判断价值；不设武断的固定绝对提升幅度 |
| 候选身份 | 已冻结：GCR-PACRE-v24 的 flip-even common gate × flip-odd residual 具体实现，以及 normalization、readout、PMOPE、采样、优化器和模型配置 |
| 执行身份 | r1 无可认证决定；fresh r2 尚未创建 spec、authorization、commit、claim 或实际 unit |

若选择继续同候选 r2，则不得修改：

- `gcr_pacre.py`；
- gate 参数化；
- normalization；
- PMOPE；
- loss 权重；
- sampling；
- optimizer；
- schedule；
- width；
- attention；
- \(D_R\)/OOF/bounded/Formal 决策规则。

这些改动都必须成为新的候选版本，不能夹带进 r2。

在继续 r2 的分支中，当前允许完成的是：

1. 审计、清理并冻结真实 user-systemd 环境；
2. 用无数据 dummy unit 完成一次真实 user-systemd 集成测试；
3. 实现而非“假设存在”实际 r2 unit；
4. 将 manager endpoint/identity、冲突 unit、GPU consumer 和 GPU lease 纳入 runtime spec 与 authorization；
5. 在写入 create-once `attempt_commit` **之前**重新验证这些环境条件；
6. 重新生成 supervisor/source/unit closure；
7. 最后才创建 actual r2 spec 和 authorization；
8. 发出唯一一次 `systemctl --user start --no-block`，并进行有界 dispatch/acknowledgement 观察。

这些修改能使当前**执行门槛**可合法通过，但不能保证 \(D_R\)-structural 的科学结果一定 PASS。科学结果仍必须由真实 r2 唯一运行产生。

用户也可以在 actual r2 authorization 之前终止 v24，建立 v25 并修改非主线组件；该分支必须使用新版本、新 preregistration、新 closure 和新 \(D_R\)-structural。只要科学主线、相对有效 Base 的性能目标和安全约束保持明确，“非主线可提升性能”就是合法的候选设计理由，但不能把修改夹带进同身份 r2。

## 0.1 只读现状快照（易变事实）

以下仅是 `2026-07-30T01:05:47+08:00` 的只读观测，不替代之后 create-only inventory receipt：

| 项目 | 观测 |
|---|---|
| user manager | `degraded`；systemd `255.4-1ubuntu8.16` |
| failed units | 3 个：两个 SCTransNet recovery unit 与一个 snap firmware notifier |
| actual r2 target | `LoadState=not-found`, `ActiveState=inactive`, `SubState=dead` |
| GPU0 冲突 unit | `confa-v41-mshnet-nudt-clean-formal-20260718-v1.service`，`enabled + activating/auto-restart`，`Restart=on-failure`, `RestartUSec=30s`, `NRestarts=18425`，请求 `cuda:0` |
| 另一循环 unit | `confa-v41-dnanet-clean-formal-seed42-v2.service`，同样 `enabled + activating/auto-restart`，请求 `cuda:2` |
| selected GPU0 | UUID `GPU-12cdabd0-7910-8f4a-e4d7-e3c7867d1296`，PCI `0000:02:00.0`，minor `0`，compute mode `Default`；该瞬时 compute-app 列表为空 |
| manager endpoint | `/run/user/1008` owner UID 1008、mode 0700；bus 为 UID 1008 socket；actual manager control group 为 `/user.slice/user-1008.slice/user@1008.service` |

`NRestarts` 与 GPU process 列表会继续变化；任何执行判断必须重新采样。当前快照已足以证明：**现在不能直接启动 actual r2。**

---

# 1. 当前 GCR-PACRE v24 候选实例

当前主方程为：

\[
D_p
=
\frac12
\left[
R_p(O)-R_p(\widetilde O_p)
\right],
\]

\[
E_p
=
\frac12
\left[
C_p(O)+C_p(\widetilde O_p)
\right],
\]

\[
G_p=2\sigma(E_p),
\]

\[
I_p=G_pD_p,
\]

\[
\phi_p=0.9+I_p.
\]

其中：

- \(D_p\) 是 binary flip 下的 odd residual compatibility；
- \(E_p\) 是 binary flip 下的 even common evidence；
- \(G_p\) 是有限 common gate；
- \(I_p\) 仍为 flip-odd；
- 最终只有一个 scalar completion field；
- completion 固定为：
  \[
  \mathbf1[\phi<0]\land\neg O;
  \]
- 最终结果与 Base hard union。

当前正式配置固定：

```text
feature channels  = 64
feature stride    = 4
width             = 32
parameters        = 64,064
Formal epochs     = 800
steps / epoch     = 40
updates / seed    = 32,000
```

这里的 exact gate/readout/normalization 是 **v24 候选实例**，不是不可更改的永久科学主线。本方案在“继续同候选 r2”分支中不改变这些内容；若另立 v25，可以有证据地修改非主线组件。Formal 训练预算仍固定为 **800 epochs × 40 steps/epoch**。

---

# 2. 当前不是科学失败，而是执行前置条件失败

## 2.1 r1 的历史问题

首个授权的 \(D_R\)-structural r1 在 create-once marker 之后丢失了执行可观测性：

- 没有可独立认证的完整运行事实；
- 没有可信 gate decision；
- 不能将“未观察到结果”解释为科学 FAIL 或 PASS；
- 不能自动恢复或续跑。

r2 supervisor 的设计正是为了补齐：

- attempt commit；
- systemd invocation binding；
- materialization claim；
- heartbeat hash chain；
- child process group；
- cgroup descendants；
- terminal sidecar；
- supervisor、child entrypoint 和 unit fragment 的结束时复核。

这一设计方向正确。

## 2.2 r2 当前尚未真正存在

仓库中的 systemd 文件明确是：

```text
TEMPLATE ONLY
not installed
not enabled
not authorized
```

而且没有 `[Install]` section，目标本来就是一个静态、不可 enable 的一次性 unit。

因此当前：

```text
systemctl --user show <target>
```

得到 not-found，并不是模型代码错误，而是实际 unit 尚未实现。

由于 actual runtime spec 必须绑定：

- `FragmentPath`；
- unit fragment SHA；
- 完整 systemd shadow；
- exact `ExecCondition`/`ExecStartPre`/`ExecStart`/`ExecStopPost`；

所以 unit 不存在时，不应创建 actual spec，更不能创建 authorization。

## 2.3 user manager degraded

当前 user manager 为 `degraded`，只说明至少有一个 failed user unit；它**不等价于 manager 不可用，也不自动等价于 r2 的科学或执行失败**。

硬门禁应作用于本项目的可归因范围：

1. user bus 可连接，manager PID/starttime、boot ID、`XDG_RUNTIME_DIR` 与 bus endpoint 身份稳定；
2. target unit、其依赖以及已识别 GPU 冲突 unit 不处于 failed/重启循环；
3. 与项目无关的 failed units 必须形成带时间、命令和状态的 exact allowlist，不得用全局 `reset-failed` 擦除；
4. 若项目仍选择要求全局 `SystemState=running`/failed count 0，必须明确这是更保守的**项目执行政策**，不是科学主线或 systemd 必要条件；每个无关 unit 的修复与 exact reset 均需单独授权。

因此，当前真正的 blocker 是：两个已识别冲突训练服务仍在 `Restart=on-failure` 循环，actual target 仍 `not-found`，且尚无 scoped manager/GPU 环境闭包。单独的 `degraded` 字段只能作为已记录的环境事实。

## 2.4 自动重启服务与 `cuda:0`

当前至少有一个无关 user service 以：

```text
Restart=on-failure
```

持续请求 `cuda:0`。

瞬时：

```text
nvidia-smi 显示 GPU idle
```

不能证明独占，因为该 service 可能在下一次 restart timer 到期后重新启动。

这会造成：

- CUDA context 竞争；
- 显存竞争或 OOM；
- kernel scheduling 和时序漂移；
- 结构 probe 中的非确定性资源故障；
- r2 attempt 在非模型原因下被消耗。

正确证据应是：

> 冲突 unit 已在单独授权下 runtime-mask 并停止，其必要 trigger unit 也已处理；manager endpoint/identity 稳定；无关 failed units 已如实保留在 exact allowlist（或逐项授权修复）；观测窗口内 `NRestarts` 不再增长；选定 GPU 无未知 consumer；r2 持有唯一 cooperative lease。

## 2.5 现有科学 runner 仍冻结在 r1 执行身份

只读代码审计又发现一个独立 P0：现有

```text
cure_lite_v24/dr_gate.py
tools/run_cure_lite_v24_gcr_pacre_dr_gate.py
```

仍把 `run_id`、preaccess schema/status、access-audit path、authorization path 和结果 receipt path 固定为 r1。若 actual unit 直接调用旧入口，即使 unit 名称写成 r2，也会复用历史 r1 的身份与 create-once 路径；这既不是 fresh r2，也可能与历史 marker/receipt 冲突。

修复必须同时满足：

1. 103 个冻结的数值/科学实现文件逐字节保持；
2. r2 使用全新的 run ID、schema 与 create-once artifact paths；
3. 身份转换只能发生在 fresh isolated process、旧 CLI import 和任何 token 签发之前；
4. 外层适配器必须进入 runtime/source authorization closure，但不得进入 103 文件科学 closure；
5. 适配器本身不授予 \(D_R\) payload 权限，不创建 preaccess 文件，也不授权 \(D_V/D_T\)、训练、retry 或 resume。

因此新增独立外层入口：

```text
tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2.py
```

它只完成经精确校验的 r1→r2 执行身份转换，再延迟导入冻结的旧 CLI。`--r2-execution-identity-summary` 是无数据自检；它不得创建任何 r2 protocol artifact。真正的 `preaccess-create` 仍须等环境、dummy integration、unit realization 和 release gates 全部通过后，且由 actual r2 authorization 链单独控制。

运行时不能只检查“路径数是 103”或让当前代码自签。适配器必须在任何身份转换前重新计算 103 个 `(repo-relative path, SHA256)` 的 canonical binding，并严格等于 r1 已冻结的 source-closure fingerprint：

```text
28d26759a68785e9c99917fcfa8b36430c7f6e5463282d66eeab5c711e425e9f
```

任一文件或路径漂移都必须拒绝 `preaccess-create`、`preaccess-verify` 和 `real`。适配器也不得接受 `generated`：generated audit 有自己的冻结身份，不能形成“旧 generated schema + r2 run ID”的混合收据。

r2 的 access/preaccess/result 和 run-start marker 还必须使用与 r1 不同、由当前 UID 拥有且 mode `0700` 的私有父目录：

```text
protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/
runs/irstd1k_stage_a_seed42/gcr_pacre_v24_D_R_structural_attempt_r2/
```

adapter 在 `preaccess-create` 成功后把新 access/auth 文件封存为 `0444` 并用同一 fd 复核 inode、nlink、owner、canonical JSON 和 fingerprint；`preaccess-verify`/`real` 只接受该封存状态。真实结果 receipt 同样在返回成功前封存。目录本身的建立属于执行准备，不授予 payload 权限。

---

# 3. 当前 supervisor 已经解决的部分

现有 `tools/cure_lite_v24_runtime_supervisor.py` 已经严格约束：

```text
shell                       false
launch_limit                1
automatic_retry_allowed     false
resume_allowed              false
Restart                     no
NRestarts                   0
Type                        exec
KillMode                    mixed
SendSIGKILL                 yes
```

它还完成：

- authorization 缺失时在任何 attempt artifact 前拒绝；
- child argv 和 environment 的 exact binding；
- Python injection 环境拒绝；
- attempt root 的 exact-empty 检查；
- supervisor 和 child entrypoint SHA；
- unit fragment SHA；
- exact systemd shadow；
- one-shot start；
- runtime materialization claim；
- invocation ID 与 cgroup 验证；
- child heartbeat；
- process-group/cgroup descendant 清理；
- terminal 与 systemd sidecar；
- finalizer 中当前 supervisor、child entrypoint 和 unit closure 复核。

特别是 finalizer 已能在文件漂移时产生：

```text
audit_valid = false
```

这一部分应保留。

---

# 4. 当前 supervisor 尚未闭合的部分

## 4.1 Runtime spec 不包含外部环境合同

当前 spec 主要绑定：

- authorization；
- child；
- artifacts；
- runtime mechanics；
- target systemd unit；
- source bindings。

它没有绑定：

- user manager state；
- failed unit inventory；
- 冲突 unit 清单；
- 冲突 unit 的 fragment SHA；
- `Restart` 与 `NRestarts` 的稳定性；
- trigger timers/sockets；
- GPU UUID；
- GPU process inventory；
- GPU lease；
- 环境清理 receipt；
- 真实 user-systemd integration receipt。

## 4.2 `commit_and_start()` 在 attempt commit 前没有环境 recheck

当前顺序为：

```text
verify authorization
validate filesystem
validate empty artifacts
query target unit shadow
write attempt_commit
systemctl --user start --no-block
```

如果 target unit 已存在，但此时：

- manager 变成 degraded；
- 冲突 service 重新启动；
- GPU consumer 出现；
- unit trigger 状态改变；

当前代码仍可先写入 `attempt_commit`。

一旦 commit 创建，该 attempt 已被消费。之后即使 `systemctl start` 或 child 因环境失败，也不能把它当成“还没有运行”。

所以必须采用：

```text
静态/授权闭包
→ 初态审计
→ O_EXCL active GPU lease
→ lease 后最终 manager/unit/GPU 双快照
→ attempt commit
```

并在 systemd invocation 内再次复核以检测 commit 后漂移。只在 lease 前审计仍存在 TOCTOU；只看 `systemctl start --no-block` 返回 0 也只证明请求已排队，不证明 InvocationID、claim 或 terminal 已出现。

## 4.3 Target unit 的 immutable shadow 与 phase state 尚未拆分

现有 exact shadow 适合冻结 fragment 身份和静态执行合同，但不能把所有运行态字段加入同一个跨阶段 exact map。unit 从 precommit 的 `inactive/dead` 进入 `activating/start-pre` 后，`ActiveState`、`SubState`、`Result`、`ExecMain*` 必然变化；若所有阶段仍比对同一 shadow，会造成自锁。

必须拆为：

```text
immutable fragment shadow
  LoadState=loaded
  UnitFileState=static
  FragmentPath / fragment dev+inode+SHA
  DropInPaths
  ExecCondition / ExecStartPre / ExecStart / ExecStopPost
  Restart / Type / KillMode / I/O / sandbox 等静态字段

phase state ledger
  precommit: inactive/dead, NRestarts=0, NeedDaemonReload=no
  preclaim: activating/active 的集成测试冻结允许集合
  child: InvocationID、ControlGroup、MainPID
  final: Result、ExecMainCode、ExecMainStatus、NRestarts
```

`LoadState`/`UnitFileState` 可以进入 immutable shadow；`ActiveState`/`SubState`/`Result`/`ExecMain*` 只能按 phase 记录和断言。

## 4.4 35 项 supervisor 测试不是实际 user-systemd 集成

当前定向测试覆盖静态、生成式和 fault-injection 行为。任何“35 项”“279 PASS”之类计数，都必须附测试命令、时间、代码/测试树 SHA 与原始日志 fingerprint 才能成为证据，不能把正文中的历史数字当作当前事实。

但它们没有证明当前机器上的实际 user manager 能完成：

```text
unit file realization
daemon-reload
ExecCondition
ExecStartPre
ExecStart
ExecStopPost
INVOCATION_ID
ControlGroup
cgroup cleanup
journal I/O
terminal sidecar
unit removal
```

所以必须增加一个无数据、无科学 payload 的真实集成门禁，并在 supervisor schema 中定义独立的：

```text
execution_kind = systemd_integration_dummy
```

现有 `generated_dummy` 不验证真实 `INVOCATION_ID`、`ControlGroup`、静态 fragment 和 user-manager 生命周期，不能充当决定性 systemd integration。

---

# 5. 身份边界：哪些修改仍可属于 r2

## 5.1 仍可属于同候选 r2 的修改

在 actual r2 spec、authorization、attempt commit 都尚不存在的前提下，以下是执行工程修改：

- 新增 environment auditor；
- 新增 cleanup executor；
- 新增 true user-systemd dummy integration；
- 新增 unit realizer；
- 扩展 runtime spec 的环境字段；
- 增加 manager/GPU live preflight；
- 增加 GPU lease；
- 增加 fresh-process r2 执行身份适配器；
- 增加 target unit load-state 验证；
- 扩展 terminal 环境审计字段；
- 重新生成 source/runtime/unit closure；
- 重新运行 supervisor 与 release tests。

这些修改不改变科学候选。

## 5.2 不能属于 r2 的修改

以下任一项都要求新候选版本，例如 v25：

- normalization；
- gate 公式或 gate 参数；
- readout；
- PMOPE/objective；
- loss 权重；
- sampling；
- optimizer；
- schedule；
- width；
- attention；
- feature adapter；
- occupancy representation；
- field threshold；
- matching/evaluation convention。

fresh r2 只能恢复执行，不能提升模型性能。

特别地，改变 evaluator/matching convention 属于**协议版本变化**，不能只视为候选模型版本变化。

## 5.3 r2 与 v25 是授权前的并行合法选择

在 actual r2 authorization、attempt commit 均不存在时，可以二选一：

1. 保留 v24 身份，先闭合执行环境，再运行同候选 r2；
2. 终止 v24，不消费 r2，建立 v25 并修改有性能依据的非主线组件。

第二条不否定“先确定主线、再做模型设计修改”的原则；它只要求版本与证据链诚实分离。不得一边保留 v24-r2 身份，一边改变候选公式、训练或评价协议。

---

# 6. 推荐新增：环境政策与收据

新增：

```text
tools/cure_lite_v24_runtime_environment.py
```

它必须是：

- standard-library-only；
- 不导入 torch；
- 不导入 dataset；
- 不读取 \(D_R/D_V/D_T\)；
- 只读取 systemd、procfs、cgroupfs 和 `nvidia-smi` 元数据。

建议 schema：

```text
cure-lite-v24-runtime-environment-policy-v1
cure-lite-v24-runtime-environment-inventory-v1
cure-lite-v24-runtime-cleanup-plan-v1
cure-lite-v24-runtime-cleanup-receipt-v1
cure-lite-v24-runtime-stability-receipt-v1
cure-lite-v24-user-systemd-integration-authorization-v1
cure-lite-v24-user-systemd-integration-receipt-v1
cure-lite-v24-unit-realization-authorization-v1
cure-lite-v24-unit-realization-receipt-v1
cure-lite-v24-runtime-preflight-receipt-v1
cure-lite-v24-gpu-lease-v1
cure-lite-v24-gpu-lease-tombstone-v1
cure-lite-v24-gpu-lease-release-complete-v1
cure-lite-v24-runtime-phase-receipt-v1
```

所有收据均须 create-only、canonical JSON、owner/mode/nlink 检查、文件与父目录 `fsync`、拒绝 symlink/hardlink，并绑定生成命令、时间、源码/可执行文件 SHA 与输入 fingerprint。若无需清理，必须生成显式 `cleanup_not_required_receipt`，不能用缺失字段或空文件表达。

---

# 7. Environment policy

建议冻结：

```json
{
  "schema_version": "cure-lite-v24-runtime-environment-policy-v1",
  "candidate": "GCR-PACRE-v24",
  "stage_id": "D_R_structural",
  "attempt_id": "r2",
  "manager_scope_policy": "endpoint-identity-and-project-units",
  "unrelated_failed_unit_policy": "exact-recorded-allowlist",
  "global_system_state_running_required": false,
  "target_unit_name": "cure-lite-v24-gcr-pacre-dr-r2.service",
  "target_unit_file_state": "static",
  "target_unit_precommit_active_state": "inactive",
  "target_unit_precommit_sub_state": "dead",
  "selected_gpu_uuid": "...",
  "selected_gpu_pci_bus_id": "...",
  "selected_gpu_minor_number": 0,
  "required_compute_mode": "Default",
  "required_mig_mode": "Disabled",
  "required_mps_state": "inactive",
  "gpu_conflict_unit_ids": ["..."],
  "required_conflict_unit_state": "masked-runtime",
  "unknown_gpu_consumer_allowed": false,
  "stability_window_basis": "derived-from-max-restart-and-trigger-period",
  "lease_path": "/run/user/<uid>/cure-lite-v24-<gpu-uuid>.lease",
  "payload_authority": "none"
}
```

这里的 unit ID、GPU UUID/PCI/minor/MIG/MPS/compute mode 必须来自环境 inventory，不允许手写猜测。`payload_authority="none"` 只描述环境工具无 payload 权限；actual \(D_R\) 权限只能由之后唯一的 r2 authorization 赋予，不能由 environment policy 与其竞争。\(D_V/D_T\) 始终未授权。

---

# 8. Manager 与 unit inventory

环境工具应执行固定绝对路径命令，例如：

```text
/usr/bin/systemctl --user is-system-running
/usr/bin/systemctl --user --failed --all --no-legend --plain
/usr/bin/systemctl --user list-units --type=service --all --no-legend --plain
/usr/bin/systemctl --user show <unit> ...
```

每个相关 unit 至少记录：

```text
Id
Description
LoadState
ActiveState
SubState
UnitFileState
FragmentPath
DropInPaths
Restart
RestartUSec
NRestarts
Result
ExecMainCode
ExecMainStatus
MainPID
ControlGroup
Environment
ExecStart
TriggeredBy
Triggers
WantedBy
RequiredBy
PartOf
```

同时保存：

- FragmentPath SHA；
- DropIn 文件 SHA；
- boot ID；
- user manager PID 和 starttime；
- manager control group、`/run/user/<uid>` dev+inode、bus socket type/owner；
- 固定的 `XDG_RUNTIME_DIR`、`DBUS_SESSION_BUS_ADDRESS`、`LC_ALL=C` 和确定性 `PATH`；
- query command/exit status；
- inventory fingerprint。

所有 `systemctl --user` 调用必须使用上述经验证的固定 endpoint 环境，不能继承任意调用者环境。inventory 中的 `Result`/`ExecMain*` 是带时间的运行态观测，不进入静态 unit fingerprint。

---

# 9. GPU inventory

固定绝对路径并绑定 SHA：

```text
/usr/bin/nvidia-smi
```

首先冻结：

```text
UUID <-> PCI bus ID <-> minor number
compute mode / MIG mode / MPS state
```

然后查询：

```text
pid
gpu_uuid
process_name
used_gpu_memory
```

每个 GPU PID 再映射：

```text
/proc/<pid>/cmdline
/proc/<pid>/cgroup
/proc/<pid>/stat starttime
UID
systemd unit（若可归属）
```

每一轮必须采用：

```text
nvidia-smi snapshot A
→ 对 A 中每个 PID 读取 /proc identity、UID、starttime、cgroup
→ 用 unit 的 ControlGroup 做双向归属核验
→ nvidia-smi snapshot B
```

A/B PID 集变化、PID 消失/复用、`/proc` 不可读、system/root/MPS/MIG consumer 或 cgroup 归属不唯一时，整轮 FAIL/重采。cmdline 只作诊断，不能作为 PID→unit 的身份依据。

以下均为 blocker：

- selected GPU 上存在未知 compute PID；
- PID 不属于 allowlist；
- PID 对应 auto-restarting conflict unit；
- PID 在稳定观测窗口中反复出现；
- GPU UUID 与 policy 不一致；
- `nvidia-smi` 输出无法完整解析；
- GPU consumer 无法映射到进程或 cgroup。

actual child 必须冻结：

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<selected GPU UUID>
child logical device = cuda:0
```

并拒绝任何未绑定的继承 CUDA 环境。cooperative lease 不能排除 root 或不合作进程；若需要强独占，仍须调度器或管理员提供独占证明。

---

# 10. 单独的环境清理授权

停止或 mask 无关服务不是科学 runner 应自动做的事情。

必须先生成 audit-only：

```text
environment_cleanup_plan.json
```

其中逐 unit 保存：

```text
unit_name
current fragment SHA
current Restart
current NRestarts
current ActiveState/SubState
current trigger units
current GPU evidence
authorized actions
reversal actions
```

最小临时清理顺序：

```text
systemctl --user mask --runtime <exact service>
systemctl --user mask --runtime <exact necessary timer/socket/path triggers>
systemctl --user stop <exact service>
验证 exact service = masked-runtime + inactive
```

不默认执行 persistent `disable`。禁止无参数或全局 `systemctl --user reset-failed`：它会擦除无关失败历史并重置 start-rate counter。只有在保全 before-ledger、单独授权且明确列出 exact unit 时，才允许在 attempt commit 前 exact reset；commit 后不得 reset actual target。

`--runtime` mask 的寿命绑定 `/run/user/<uid>`、user-manager session/linger 与 boot，而不只是“当前 boot”。receipt 必须绑定 boot ID、manager PID/starttime、`XDG_RUNTIME_DIR` dev+inode，并在 manager/session 身份变化后重新审计。`daemon-reload` 本身不应使 mask 消失，但必须由 dummy integration 实测。

执行工具：

```text
tools/cure_lite_v24_environment_cleanup.py
```

必须满足：

- 默认 audit-only；
- 没有 cleanup authorization 时拒绝执行；
- authorization exact 列出 unit/action；
- 执行前重新验证 fragment SHA 和 current state；
- 禁止通配符；
- 禁止 shell；
- 禁止未知 action；
- 每条命令和返回值进入 receipt；
- 清理后 manager endpoint/identity 必须可验证且稳定；
- target/conflict/dependency failed set 必须为空；
- 无关 failed units 保留 exact allowlist，不为“变绿”而清零；
- conflict units 必须 inactive 且 runtime-masked；
- trigger units 必须不能重新激活冲突服务；
- 不自动恢复服务。

恢复无关服务必须使用独立 restoration authorization。若尚未写 attempt commit 而项目取消，也可以经该授权恢复；不必机械地等到不存在的 r2 terminal。

---

# 11. 稳定观测门禁

单个时刻的状态不够。

清理后进行多轮采样。轮数与间隔是执行 profile，观测窗口至少覆盖已识别 unit 的最大 `RestartUSec` 与 trigger 周期，不能把“3 次 × 30 秒”宣称为科学不变量。一个可审定 profile 可以是：

```text
sample 0
sample 1
sample 2
```

每次记录：

- manager state；
- failed unit set；
- conflict unit states；
- `NRestarts`；
- trigger state；
- GPU processes；
- target unit state；
- boot ID。

要求：

```text
manager endpoint/identity 始终稳定
target/conflict/dependency failed set 始终空
unrelated failed exact allowlist 不发生未解释漂移
conflict units 始终 masked-runtime + inactive
NRestarts 不增长
selected GPU 始终无未知 compute process
boot ID 不变
```

若任一次失败，则不创建 actual r2 authorization。无数据 dummy integration 可以在 manager `degraded` 但 scoped manager/target 合同满足时单独授权执行，以便先证明 systemd 路径；不得为了运行 dummy 而先清除无关 failed 证据。

---

# 12. 真实 user-systemd dummy integration

新增：

```text
tools/cure_lite_v24_user_systemd_integration.py
deploy/systemd/cure-lite-v24-supervisor-integration.service.template
tools/cure_lite_v24_dummy_child.py
```

dummy child 只能：

- 打印固定内容；
- 创建固定 generated artifact；
- 等待一个短固定周期；
- 正常退出；
- 不导入 torch；
- 不读取任何 split；
- 不访问 GPU。

该集成必须使用独立的 `execution_kind=systemd_integration_dummy`、独立 integration authorization 和 create-only **static fragment**。`systemd-run` 的 transient unit 在 `Transient`、`UnitFileState`、`FragmentPath` 上与 actual 不同，最多用于非决定性 manager smoke test，不能替代本门禁。

真实集成必须验证：

```text
unit 被实际 user manager load
UnitFileState = static
ExecCondition 执行
ExecStartPre 执行
ExecStart 执行
ExecStopPost 执行
INVOCATION_ID 一致
self cgroup 与 unit ControlGroup 一致
materialization claim 有效
heartbeat chain 有效
child PID 被正确 reap
terminal sidecar 有效
Restart=no
NRestarts=0
source drift fault injection 使 audit_valid=false
unit 可干净移除
daemon-reload 后 LoadState=not-found
```

还必须覆盖以下 postcommit 故障路径：

```text
第一个 ExecCondition=false → 第二个 condition 未执行 → ExecStopPost 写 consumed sidecar
claim condition 失败 → ExecStopPost 写 consumed sidecar
ExecStartPre 失败 → ExecStopPost 写 consumed sidecar
start --no-block dispatch 成功但 acknowledgement 超时
start request 非零 → committer 同步写 START_REQUEST_FAILED_AFTER_COMMIT
daemon-reload / manager GC 后静态 fragment 身份仍可验证
lease acquire/release/tombstone 完整
```

本机 systemd 255 的 `systemd.service(5)` 明确：`ExecCondition` 非零或异常退出仍会运行 `ExecStopPost`；1..254 通常跳过余下命令且 unit 不标 failed，255/异常会标 failed。该手册事实仍须由真实 integration 验证机器行为与 sidecar 分类，不能只靠 mock。

它使用独立：

```text
unit name
spec
artifact root
lease path
run ID
```

不得创建 actual r2 的任何文件。

`StartLimitBurst=1` 与 create-once claim 意味着同一 dummy unit 不得复用。成功路径和每个 fault-injection 场景都必须使用唯一 unit/spec/artifact root/lease path，禁止通过 `reset-failed` 回收同一个 unit。

集成通过后保存：

```text
user_systemd_integration_receipt.json
```

并将其 fingerprint 绑定到 actual r2 authorization。

每个 dummy 清除顺序必须是：

```text
terminal/sidecar 封存
→ 有界确认 inactive/dead
→ 验证 exact FragmentPath、dev+inode+nlink=1、SHA
→ unlink fragment + fsync(parent)
→ daemon-reload
→ 有界轮询 LoadState=not-found 且 FragmentPath 为空
```

任一步失败均不得签发 PASS integration receipt。

---

# 13. 实现 actual r2 unit

新增：

```text
tools/cure_lite_v24_realize_systemd_unit.py
```

actual unit realization 会写 user systemd 搜索路径并执行 `daemon-reload`，属于外部状态变更；必须先有独立、create-only 的：

```text
r2_unit_realization_authorization.json
```

dummy integration authorization 也必须逐项授权 fragment install/run/remove 与 `daemon-reload`。cleanup authorization、integration authorization、unit-realization authorization 和 actual r2 scientific authorization 互不替代。

执行顺序：

1. 通过 `/usr/bin/systemd-path --suffix=systemd/user user-runtime` 取得 boot/session-bound user runtime unit directory；若政策明确选择持久配置目录，则使用 `/usr/bin/systemd-path --suffix=systemd/user user-configuration`。裸 `systemd-path user-configuration` 只返回配置根，不能直接当 unit 目录；
2. 对照 `/usr/bin/systemd-analyze --user unit-paths` 验证该目录确在搜索路径中，并冻结目录 canonical、owner、mode、dev+inode 及其优先级；
3. 从模板渲染 exact unit；
4. 以 `O_CREAT|O_EXCL|O_NOFOLLOW` create-only 写入，拒绝 symlink/hardlink，固定 owner/mode；
5. `fsync(file)` 和 `fsync(parent directory)`；
6. 用 fixed manager endpoint 执行一次 `systemctl --user daemon-reload`；
7. 查询 unit shadow；
8. 验证：
   ```text
   LoadState=loaded
   UnitFileState=static
   ActiveState=inactive
   SubState=dead
   Restart=no
   NRestarts=0
   NeedDaemonReload=no
   DropInPaths=""
   ```
9. 保存 unit fragment SHA 和完整 shadow；
10. 生成：
   ```text
   r2_unit_realization_receipt.json
   ```

目标 unit 仍然：

- 无 `[Install]`；
- 不 enable；
- 不自动启动；
- `Restart=no`；
- `StartLimitBurst=1`。

可以在 `[Unit]` 增加冻结的 `ConditionPathExists=<exact attempt_commit path>`，仅用于阻止 commit 前的同 UID 误启动。Unit-level condition-false 通常不进入 service `ExecStopPost`，因此它不能替代 postcommit sidecar；其适用语义恰好是“尚未 commit，无需 terminal”。

---

# 14. 修改 systemd shadow 合同

将合同拆成 immutable fragment shadow 与 phase ledger。`_SYSTEMD_SHADOW_KEYS` 只增加安全静态字段：

```python
"LoadState",
"UnitFileState",
```

并继续冻结 `FragmentPath`/DropIns/所有 exec lines/Restart/Type/KillMode/I/O 等现有静态项。以下字段**不得**加入跨阶段 exact shadow：

```python
"ActiveState",
"SubState",
"Result",
"ExecMainCode",
"ExecMainStatus",
```

precommit phase ledger 单独要求：

```python
shadow["LoadState"] == "loaded"
shadow["ActiveState"] == "inactive"
shadow["SubState"] == "dead"
shadow["UnitFileState"] == "static"
shadow["Restart"] == "no"
shadow["NRestarts"] == "0"
shadow["NeedDaemonReload"] == "no"
```

`ActiveState`/`SubState`/`Result`/`ExecMain*`/`NRestarts`/`NeedDaemonReload` 按 phase 查询并带时间进入 ledger；它们不参与跨阶段静态身份比对。各 phase 允许集合必须由 dummy integration 冻结。

---

# 15. Runtime spec v2

由于 actual r2 spec 尚未存在，可以安全升级 supervisor schema：

```text
cure-lite-v24-dr-runtime-supervisor-spec-v2
```

新增顶层字段：

```python
"environment"
```

建议 exact keys：

```python
_ENVIRONMENT_KEYS = {
    "policy_path",
    "policy_file_sha256",
    "inventory_path",
    "inventory_file_sha256",
    "cleanup_plan_path",
    "cleanup_plan_file_sha256",
    "cleanup_authorization_path",
    "cleanup_authorization_file_sha256",
    "cleanup_receipt_path",
    "cleanup_receipt_file_sha256",
    "stability_receipt_path",
    "stability_receipt_file_sha256",
    "integration_authorization_path",
    "integration_authorization_file_sha256",
    "integration_receipt_path",
    "integration_receipt_file_sha256",
    "unit_realization_authorization_path",
    "unit_realization_authorization_file_sha256",
    "unit_realization_receipt_path",
    "unit_realization_receipt_file_sha256",
    "selected_gpu_uuid",
    "selected_gpu_pci_bus_id",
    "selected_gpu_minor_number",
    "gpu_lease_path",
    "gpu_lease_tombstone_path",
}
```

policy/inventory/plan/authorization/receipt 等**已存在证据文件**必须：

- absolute；
- canonical；
- regular file；
- non-symlink；
- read-only；
- exact SHA。

`gpu_lease_path` 与 `gpu_lease_tombstone_path` 是未来 pathname，不是预存 receipt：preauthorization 时要求 canonical parent 合法、active/tombstone target 按协议应不存在；acquire 后 active lease 才是 mode 0600 的 transient regular file。不得对它们套用“预先存在、只读、exact SHA”的合同。

禁止形成 `spec ↔ preflight receipt` 循环：

```text
immutable policy/inventory/cleanup/stability/integration/unit receipts
→ metadata-only scientific r2 access audit + preaccess authorization
→ runtime spec v2
→ runtime launch authorization 绑定 spec、scientific preaccess 与完整环境链
→ runtime launch authorization 后做 fresh live precommit observation
→ 该 observation 直接嵌入并绑定 attempt_commit
```

其中 `environment.inventory_path` 明确指向清理后、稳定门禁通过后的
`audit-only` PASS receipt；清理前那个带唯一 blocker 的 FAIL receipt 由
policy/cleanup plan 绑定，不能冒充 runtime inventory。

unit realization 也必须保持有向无环：realization authorization 绑定
**未来且当时不存在**的 exact runtime-spec absolute pathname，以及
template/rendered fragment/supervisor/Python/manager generation，但不绑定
尚不可能存在的 spec file SHA；安装与 `daemon-reload` 完成后产生
unit-realization receipt。随后 runtime spec 绑定 realization
authorization/receipt 的 path、SHA、fingerprint，最后 runtime-launch
authorization 再绑定 spec 的 SHA/fingerprint。禁止用占位 spec 打破这个
依赖顺序。

因此 fresh precommit receipt 不作为 spec 输入；postcommit 的 preclaim/prespawn/finalize 证据是 create-once phase receipt，由 terminal/sidecar 引用。

---

# 16. Scientific preaccess 与 runtime launch authorization 必须分离

冻结的 `dr_gate.py` 对 scientific preaccess authorization 使用严格 closed schema；它不能同时容纳 supervisor 的环境、unit 和 spec 字段。反过来，runtime supervisor 的 launch authorization 也不能冒充 gate 的 preaccess 文件。因此 fresh r2 必须有两个互不替代的授权对象：

```text
D_R_structural_attempt_r2_authorization.json
    scientific preaccess authorization
    由冻结 gate 的 metadata-only builder 生成
    绑定 dataset-free receipt、D_R metadata/source closure 与 D_R-only 权限

D_R_structural_attempt_r2_runtime_launch_authorization.json
    runtime launch authorization
    由 supervisor 在 spec 冻结后验证
    绑定 scientific preaccess/access audit、环境链、unit、spec 和 supervisor
```

不得把额外字段塞入 scientific 文件，也不得让 runtime 文件省略 scientific 文件的 path、SHA 和 fingerprint。runtime launch authorization 必须绑定：

```text
scientific_preaccess_authorization_path
scientific_preaccess_authorization_file_sha256
scientific_preaccess_authorization_fingerprint
scientific_access_audit_path
scientific_access_audit_file_sha256
scientific_access_audit_receipt_fingerprint
r2_execution_adapter_path
r2_execution_adapter_file_sha256
frozen_scientific_source_closure_fingerprint
environment_policy_fingerprint
environment_inventory_fingerprint
cleanup_plan_fingerprint
cleanup_authorization_fingerprint
cleanup_receipt_fingerprint
environment_stability_receipt_fingerprint
user_systemd_integration_authorization_fingerprint
user_systemd_integration_receipt_fingerprint
unit_realization_authorization_fingerprint
unit_realization_receipt_fingerprint
selected_gpu_uuid
selected_gpu_pci_bus_id
manager_endpoint_identity_fingerprint
gpu_lease_policy_fingerprint
runtime_spec_v2_fingerprint
runtime_spec_v2_file_sha256
supervisor_v2_source_closure_fingerprint
unit_fragment_sha256
```

并继续固定：

```text
candidate = GCR-PACRE-v24
attempt_ordinal = 2
prior_attempt_count = 1
authorization_kind = runtime_launch
fresh_attempt_authorized = true
scientific_D_R_preaccess_bound = true
D_R_payload_authorized = true
D_V_payload_authorized = false
D_T_payload_authorized = false
training_authorized = false
resume_allowed = false
automatic_retry_allowed = false
```

两份 authorization 都必须约束 observed preauthorization payload access 为 none。只有 scientific preaccess **与** runtime launch authorization 同时有效，且 attempt commit、materialization claim、active GPU lease 和当前 systemd InvocationID/cgroup 被 create-once runtime attestation 联合绑定时，r2 adapter 的 `real` 子命令才可继续。adapter 的 `preaccess-create`/`preaccess-verify` 仍是 metadata-only；`real` 必须要求：

```text
--runtime-launch-authorization <absolute path>
CURE_LITE_V24_RUNTIME_ATTESTATION_PATH=<supervisor-only injected path>
```

并在导入 D_R loader 前复核全部绑定。这样，scientific preaccess 文件存在本身仍不足以绕过 supervisor 手工启动 payload。

---

# 17. Precommit 环境验证

新增：

```python
def verify_live_environment(
    spec: Mapping[str, object],
    *,
    phase: Literal[
        "prelease",
        "precommit",
        "systemd_preclaim",
        "child_prespawn",
        "finalize",
    ],
) -> dict[str, object]:
    ...
```

## 17.1 `prelease` 与 lease acquire

先验证 immutable authorization/spec/fragment/source/receipt closure、attempt root exact-empty、manager endpoint、target unit 和 GPU 初态；然后以 O_EXCL 获取 active GPU lease。该初态审计不能替代 lease 后的最终审计。

## 17.2 `precommit`

持有 lease 后、写 attempt commit 前立即做最终 live audit：

```text
manager bus/endpoint/identity 与 policy 一致
target/conflict/dependency failed set empty
unrelated failed exact allowlist 无漂移
conflict units runtime-masked and inactive
NRestarts stable
target unit loaded/static/inactive/dead
selected GPU 完成 nvidia-smi(A) → /proc+cgroup → nvidia-smi(B) 双快照且无未知 consumer
GPU UUID/PCI/minor/compute/MIG/MPS 与 policy 一致
boot ID、manager identity、XDG_RUNTIME_DIR inode 与证据链一致
unit fragment/current source 未漂移
active GPU lease 的 dev+inode+nlink/payload 属于当前 attempt
```

该 live observation 直接进入 `attempt_commit`，不回写 runtime spec，避免循环。失败时若 commit path 可证明从未创建，才可按 owner/inode/payload 规则回收未消费 lease。

## 17.3 `systemd_preclaim`

unit 已进入 activating/active，此时允许 target unit 自身存在；仍要求：

```text
manager endpoint/identity稳定
target/conflict/dependency无新 failure
conflict units仍被 mask
GPU 双快照仍无外部 consumer
active lease属于当前 attempt
INVOCATION_ID / ControlGroup 可建立
```

## 17.4 `child_prespawn`

该检查必须在 `run_once` 进程内部、紧邻唯一一次 `Popen`，不能只放在单独 `ExecStartPre`，否则存在 TOCTOU：

```text
active lease dev+inode/payload有效
GPU 最终双快照无外部 consumer
当前 cgroup/InvocationID有效
CUDA_VISIBLE_DEVICES=<selected UUID> 且 child logical cuda:0
```

## 17.5 `finalize`

记录环境末态：

- manager state；
- failed units；
- GPU processes；
- conflict units；
- lease；
- source closure。

外部环境漂移不改变 child 的科学 decision，但必须使：

```text
runtime_environment_audit_valid=false
```

并阻止下游 authorization。

每个 postcommit phase 都必须写 create-once phase receipt。condition skip、claim failure、prespawn rejection、start dispatch failure/ack timeout也都是 consumed attempt，必须有可独立验证的 sidecar，不得只抛异常。

---

# 18. GPU lease

采用 create-once cooperative active lease；它是本项目 runner 的排他证据，不冒充系统级 GPU 独占。

## 18.1 Create-once lease file

固定路径：

```text
/run/user/<uid>/cure-lite-v24-<selected-gpu-uuid>.lease
```

创建要求：

- parent canonical；
- owner 当前 UID；
- mode 0700；
- `O_CREAT | O_EXCL | O_NOFOLLOW`；
- file mode 0600；
- payload 绑定：
  ```text
  boot ID
  GPU UUID
  runtime spec fingerprint
  target unit
  attempt ID
  authorization fingerprint
  attempt-commit preimage fingerprint
  committer PID/starttime
  ```

只要文件存在，其他合作的 CURE runner 必须拒绝启动。

lease payload 不包含自身完整文件 fingerprint，避免自指循环；fingerprint 由 canonical payload 写完并 `fsync` 后从外部计算。stale/unowned lease 不得自动删除，必须用单独 forensic cleanup authorization，并证明 commit path 不存在或已按 consumed 处理。

## 18.2 In-process inode/liveness guard

`run_once` 重新打开 active lease，核对 fd/path 的 dev+inode、nlink、owner、mode、payload，并可持有 nonblocking exclusive `flock` 作为 inode/liveness guard，直到 child 被 reap、descendants quiesce、terminal 写入。`flock` 不能原子地从 committer 向 systemd unit 传递，也不能阻止同 UID unlink，因此不是主要排他机制。

## 18.3 Release

`ExecStopPost` 在 terminal sidecar 写入且 target cgroup/child GPU consumer 均 quiescent 后：

1. 验证 lease fd/path dev+inode+nlink 与 payload属于当前 attempt；
2. 验证没有 r2 descendant；
3. 将 active lease原子 rename 到该 attempt 的 immutable tombstone，并 `fsync(parent)`；
4. 再 create-only 写 `gpu_lease_release_complete_receipt.json` 并 `fsync`；
5. 任一步失败不得伪报 release complete，进入独立 forensic cleanup。

若 active lease 已移出而 completion receipt 写失败，下游因缺 receipt fail-closed；不会出现“receipt 声称已释放但 active lease 仍在”的矛盾。

该 lease 不能约束不合作的外部进程，因此仍必须配合：

- runtime-masked conflict services；
- live GPU process audit；
- 若存在 system/root GPU workload，则要求管理员或调度器提供独占授权。

---

# 19. 修改 `commit_and_start()` 的顺序

推荐改为：

```python
def commit_and_start(spec_path):
    spec = load_runtime_spec(spec_path)

    authorization = verify_authorization(spec)
    validate_runtime_filesystem(spec)
    validate_precommit_artifacts(spec)
    observed_unit = query_systemd_shadow(
        spec["runtime"]["systemd"]["unit_name"]
    )
    validate_systemd_shadow(spec, observed_unit)
    verify_live_environment(spec, phase="prelease")

    lease = acquire_gpu_lease(
        spec,
        authorization=authorization,
    )

    # Final TOCTOU-closing observation while the active lease is held.
    environment = verify_live_environment(spec, phase="precommit")

    try:
        commit = attempt_commit_payload(
            spec,
            authorization,
            observed_unit,
            environment,
            lease,
        )
        write_new_json(attempt_commit_path, commit)
    except BaseException:
        # Release only when commit non-creation is positively proven.
        # If the path exists or fsync/create outcome is uncertain, preserve
        # the lease and record a consumed forensic state.
        handle_commit_write_failure_fail_closed(
            attempt_commit_path,
            lease,
        )
        raise

    dispatch = subprocess.run(
        [
            "/usr/bin/systemctl",
            "--user",
            "start",
            "--no-block",
            unit_name,
        ],
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        env=verified_fixed_manager_environment(spec),
    )

    write_new_json(
        start_dispatch_receipt_path,
        start_dispatch_payload(dispatch),
    )

    if dispatch.returncode != 0:
        write_new_json(
            consumed_sidecar_path,
            consumed_payload("START_REQUEST_FAILED_AFTER_COMMIT"),
        )
        return POSTCOMMIT_CONSUMED_FAILURE

    # --no-block only proves that the request was queued.
    acknowledgement = bounded_watch_for_invocation_claim_or_terminal(spec)
    write_new_json(start_ack_receipt_path, acknowledgement)
    if not acknowledgement["accepted"]:
        write_new_json(
            consumed_sidecar_path,
            consumed_payload("START_ACK_TIMEOUT_AFTER_COMMIT"),
        )
        return POSTCOMMIT_CONSUMED_FAILURE

    return 0
```

关键性质：

- manager/GPU/unit 环境失败发生在 commit 前；
- lease acquisition 发生在 commit 前；
- lease 后最终双快照关闭主要 precommit TOCTOU；
- commit 后任何失败都被诚实视为已消耗 attempt；
- `start --no-block` 的“排队成功”与 InvocationID/claim/terminal acknowledgement 分开；
- 没有 retry loop；
- 没有 resume。

---

# 20. 修改 unit lifecycle

建议 realized unit 增加两个环境复核入口：

```ini
[Unit]
Description=CURE-Lite v24 GCR-PACRE D_R structural fresh attempt r2
StartLimitIntervalSec=infinity
StartLimitBurst=1
ConditionPathExists=<exact attempt_commit path>

[Service]
Type=exec
ExitType=main

ExecCondition=... runtime_environment.py verify-live --phase systemd_preclaim --spec ...
ExecCondition=... runtime_supervisor.py claim-materialization --spec ...

ExecStartPre=... runtime_supervisor.py verify-runtime-spec --spec ...

ExecStart=... runtime_supervisor.py run-once --spec ...
ExecStopPost=... runtime_supervisor.py record-systemd-exit --spec ...

Restart=no
...
```

`run-once` 必须在同一进程内、唯一 `Popen` 紧前执行 `child_prespawn`；独立 `ExecStartPre` 最多作为冗余早期检查，不能充当最终检查。

禁止向 actual unit 添加 `Conflicts=<unit-a> ...`：它会让 start transaction 隐式停止其他 unit，绕过独立 cleanup authorization。`After=` 也不提供 inactive/隔离保证，默认不加入；冲突闭包由已授权 runtime mask + stop + live audit 证明。

---

# 21. 新 terminal 字段

建议 runtime terminal 与 systemd sidecar增加：

```text
environment_policy_fingerprint
environment_inventory_fingerprint
cleanup_authorization_fingerprint
cleanup_receipt_fingerprint
stability_receipt_fingerprint
integration_authorization_fingerprint
integration_receipt_fingerprint
unit_realization_authorization_fingerprint
unit_realization_receipt_fingerprint
precommit_environment_fingerprint
systemd_preclaim_environment_fingerprint
child_prespawn_environment_fingerprint
start_dispatch_receipt_fingerprint
start_ack_receipt_fingerprint
selected_gpu_uuid
selected_gpu_pci_bus_id
gpu_lease_fingerprint
gpu_lease_valid
gpu_lease_tombstone_fingerprint
gpu_lease_release_complete_fingerprint
manager_state_at_finalize
failed_units_at_finalize
unknown_gpu_consumers_at_finalize
runtime_environment_audit_valid
```

最终下游 prerequisite 要求：

```text
runtime_environment_audit_valid = true
audit_valid = true
systemd_success = true
scientific gate decision present
```

---

# 22. 测试计划

## 22.1 Environment auditor

必须测试：

1. fixed manager endpoint/identity稳定 PASS；
2. manager `degraded` 且无关 failed exact allowlist稳定可按 scoped policy PASS；
3. target/conflict/dependency failed 非空 FAIL；
4. 无关 failed allowlist 漂移 FAIL；
5. conflict unit active FAIL；
6. conflict unit `Restart=on-failure`/`NRestarts` 增长 FAIL；
7. trigger timer/socket 未 mask FAIL；
8. GPU UUID/PCI/minor/compute/MIG/MPS 漂移 FAIL；
9. 未知 GPU PID FAIL；
10. GPU A/B 快照变化、PID消失/复用 FAIL/重采；
11. cgroup 与 unit `ControlGroup` 非双向吻合 FAIL；
12. boot/manager/runtime-dir identity变化 FAIL；
13. receipt SHA/owner/mode/nlink 漂移 FAIL；
14. symlink/hardlink receipt FAIL；
15. non-finite/非规范 JSON FAIL；
16. inherited CUDA env 与 UUID→logical cuda:0 绑定不一致 FAIL。

## 22.2 Cleanup executor

1. 无 authorization 拒绝；
2. fragment SHA 漂移拒绝；
3. unit state 漂移拒绝；
4. wildcard unit拒绝；
5. 未列 action拒绝；
6. runtime-mask exact service/necessary trigger → stop → verify 的 exact order；
7. trigger cleanup；
8. 裸/global reset-failed 永久拒绝，exact reset需单独授权和 before-ledger；
9. post scoped manager/target/conflict contract；
10. restoration plan可验证但不自动执行。

## 22.3 Unit realizer

1. existing target path拒绝；
2. symlink拒绝；
3. template SHA漂移拒绝；
4. create-only；
5. 无 unit-realization authorization 拒绝；
6. 正确使用 `systemd-path --suffix=systemd/user ...` 和 fixed manager env；
7. fsync(file+parent) 后 daemon-reload；
8. static/no Install；
9. loaded/inactive/dead；
10. immutable shadow 与 phase ledger分离；
11. removal仅限 authorized dummy integration unit；
12. actual unit不得在 authorization 后被修改。

## 22.4 GPU lease

1. O_EXCL；
2. symlink attack拒绝；
3. wrong owner拒绝；
4. stale boot lease拒绝；
5. wrong GPU UUID拒绝；
6. wrong attempt拒绝；
7. concurrent acquire只有一个成功；
8. fd/path dev+inode/nlink/payload不一致拒绝；
9. commit write outcome不确定时不得释放；
10. child运行期间 inode/liveness guard；
11. terminal+quiescence后 active→tombstone→release-complete；
12. tombstone完成但receipt失败时下游 fail-closed。

## 22.5 Supervisor

保留现有测试并新增；任何 PASS 数字必须由命令、时间、代码/测试 SHA 与日志 fingerprint 证明：

1. precommit manager endpoint/target scoped contract失败不写 attempt commit；
2. precommit GPU conflict 不写 attempt commit；
3. target unit not-found 不写 attempt commit；
4. target unit active 不写 attempt commit；
5. environment receipt漂移不写 attempt commit；
6. lease占用不写 attempt commit；
7. lease后最终 audit失败且 commit确未创建时安全回收；
8. commit写入结果不确定时保留lease并标记consumed/forensic；
9. commit后 systemctl start失败同步写 consumed sidecar；
10. `--no-block` acknowledgement超时写 consumed sidecar；
11. preclaim环境漂移产生 terminal；
12. run-once内 child prespawn GPU漂移不 Popen；
13. finalizer环境漂移使 environment audit false；
14. source/unit/environment三类 closure分别复核；
15. immutable shadow在动态phase不自锁；
16. no retry/no resume保持。

## 22.6 真实 integration

实际 user manager integration 必须是单独测试组，不能用 mock 代替。它至少覆盖两级 ExecCondition/StopPost、ExecStartPre失败、dispatch/ack timeout、唯一unit-per-scenario、daemon-reload/GC、lease释放以及 dummy removal fail-closed；`systemd-run` transient 测试不得计为决定性 PASS。

---

# 23. 文件级修改清单

## 保持不变

```text
cure_lite_v24/gcr_pacre.py
cure_lite_v24/dataset_free.py
cure_lite_v24/dr_gate.py
cure_lite_v24/oof*.py
cure_lite_v24/bounded*.py
cure_lite_v24/formal*.py
cure_lite_v24/D_V*.py
cure_lite_v24/D_T*.py
preregistration scientific fields
candidate model closure
```

## 修改

```text
tools/cure_lite_v24_runtime_supervisor.py
deploy/systemd/cure-lite-v24-gcr-pacre-dr-r2.service.template
protocols/IRSTD-1K/gcr_pacre_v24/README.md
runtime supervisor schemas/tests
```

## 新增

```text
tools/cure_lite_v24_runtime_environment.py
tools/cure_lite_v24_environment_cleanup.py
tools/cure_lite_v24_user_systemd_integration.py
tools/cure_lite_v24_dummy_child.py
tools/cure_lite_v24_realize_systemd_unit.py
tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2.py

deploy/systemd/cure-lite-v24-supervisor-integration.service.template

tests_v24/test_cure_lite_v24_runtime_environment.py
tests_v24/test_cure_lite_v24_environment_cleanup.py
tests_v24/test_cure_lite_v24_user_systemd_integration.py
tests_v24/test_gcr_pacre_v24_dr_r2_execution_identity.py
tests_v24/test_cure_lite_v24_unit_realizer.py
tests_v24/test_cure_lite_v24_gpu_lease.py
tests_v24/test_cure_lite_v24_runtime_supervisor_environment.py
```

## 新增协议证据

```text
runtime_environment_policy.json
runtime_environment_inventory.json
environment_cleanup_plan.json
environment_cleanup_authorization.json
environment_cleanup_receipt.json
environment_stability_receipt.json
user_systemd_integration_authorization.json
user_systemd_integration_receipt.json
r2_unit_realization_authorization.json
r2_unit_realization_receipt.json
```

这些都不授权 split payload，直到 actual r2 authorization 单独创建。

---

# 24. 推荐执行顺序

```text
E0
生成 read-only manager/unit/GPU inventory
        │
        ├── inventory不完整：停止
        ▼
E1
生成环境清理计划（或 cleanup-not-required plan）
获取独立 exact cleanup authorization
        │
        ▼
runtime-mask exact冲突 unit及必要triggers
stop exact冲突 unit
禁止global reset；仅可对单独授权unit exact reset
        │
        ▼
E2
按最大RestartUSec/trigger周期完成稳定观测
manager endpoint/identity稳定
target/conflict/dependency failed set为空
unrelated failed exact allowlist无漂移
GPU无未知consumer
        │
        ├── FAIL：不创建 actual r2 authorization
        ▼
E3
获取独立 integration authorization
以每场景唯一static unit运行无数据真实 user-systemd integration
验证condition/StopPost/dispatch/ack/lease/removal
        │
        ├── FAIL：修复执行层，不创建 actual unit
        ▼
E4
获取独立 unit-realization authorization
create-only实现 actual r2 static unit
daemon-reload
冻结 immutable fragment shadow + precommit phase receipt
        │
        ▼
E5
验证 fresh-process r2 执行身份适配器
确认103文件科学closure逐字节未变
确认run ID/schema/access/auth/receipt paths均为fresh r2
确认identity summary不创建protocol artifact
创建metadata-only scientific r2 access audit/preaccess authorization
生成 actual r2 runtime spec v2
生成新 runtime/source/unit/environment closure
运行完整 release matrix
        │
        ├── FAIL：不创建 authorization
        ▼
E6
创建唯一 runtime launch authorization
        │
        ▼
E7
commit_and_start：
验证静态/授权闭包
acquire GPU lease
lease后最终manager/unit/GPU双快照
write attempt commit
single systemctl start --no-block
bounded dispatch/acknowledgement
        │
        ▼
E8
systemd invocation：
preclaim audit
materialization claim
runtime verify
run-once内部紧邻Popen的child prespawn audit
run once
heartbeat
terminal
lease active→tombstone→release-complete
finalizer
        │
        ▼
E9
独立验证 r2 artifact
```

---

# 25. r2 结果后的分支

## 25.1 r2 scientific PASS

进入真实 OOF-4。

当前 OOF arms 已包括：

```text
BaseA
BaseB_train_fold_selected
PACRE_VC_v23_control
GCR_PACRE_v24
GCR_PACRE_v24_forced_G1
```

这能回答：

- v24 是否优于 v23；
- common gate 是否优于强制 \(G=1\)；
- target recovery 是否提高；
- mIoU/nIoU 是否回退；
- FA 和 retention 是否安全。

只有 OOF PASS 才运行正式 full-\(D_R\) bounded-400。

## 25.2 r2 scientific FAIL

冻结 v24。

不得通过修改 normalization、gate、objective 或 optimizer 后继续称为 r2。

只要产生了可认证的 decision-bearing structural `PASS` 或 `FAIL`，该科学决定均已消费；其中 `FAIL` 不能通过 forensic receipt 获得同候选 r3。

新候选必须：

- 新版本；
- 新 preregistration；
- 新 closure；
- 新 D_R structural；
- 新 OOF。

## 25.3 r2 execution failure

一旦 `attempt_commit` 已写入，r2 被消费。

不得自动重试。

只有 **没有产生可认证科学决定** 的执行/环境中断，才可由新的 forensic receipt 讨论是否允许同候选 r3：

- 环境故障；
- systemd故障；
- child故障；
- source drift；
- GPU lease故障；
- commit/dispatch/ack/sidecar 可观测性故障。

即便属于 no-authenticated-decision，也不自动获得 r3；仍需证明候选数值语义未执行到产生决定、封存全部证据，并取得新的 attempt authorization。科学 gate 的有效 FAIL 不属于 execution failure。

## 25.4 OOF 显示 gate 无收益

若：

```text
GCR-PACRE-v24 ≈ forced-G1
```

则 common gate 没有建立价值，冻结 v24。

## 25.5 OOF 增加 target 但 IoU 回退

建立新候选，例如 v25：

- 保持 GCR 主线；
- 单独修改 objective、role weighting 或 shape regularization。

不能把这些改动放进 v24-r2。

## 25.6 OOF 没有 target 增益

下一候选才允许研究：

- normalization；
- gate 参数化；
- readout；
- width；
- attention；
- sampling；
- optimizer。

仍需新版本与新闭包。

---

# 26. “顺利通过门槛”的诚实边界

本方案能够有针对性地解决当前已经明确的执行阻断：

| 当前阻断 | 修改 |
|---|---|
| target unit not-found | create-only unit realizer + shadow receipt |
| user manager degraded | scoped manager identity + unrelated-failure allowlist + stability gate |
| restart-loop services | exact runtime-mask + stop + trigger closure |
| cuda:0 自动竞争 | GPU consumer audit + conflict mask + lease |
| 静态测试不等于真实 systemd | dummy true integration |
| 环境变化可能消耗 attempt | precommit live audit |
| supervisor只验证自身 unit | environment policy/spec/terminal |
| 瞬时 GPU idle不充分 | stable samples + service state + lease |

它不能保证：

- \(D_R\)-structural 13/13 一定 PASS；
- OOF 一定改善；
- bounded-400 一定 PASS；
- Formal800 一定成功；
- \(D_V/D_T\) 一定超过 Base。

科研上不能通过修改 gate 或结果解释来“保证成功”。

这里的目标是：

> **让 v24 在一个可认证、无已知资源竞争的执行环境中接受真实检验，并显著降低当前已知环境风险。**

cooperative lease 和 live audit 不能保证系统外不存在所有竞态，因此不得承诺“不会浪费 attempt”。

---

# 27. Authorization checklist

创建 actual r2 spec 前：

```text
[ ] v24 scientific files未修改
[ ] r1 interruption evidence未修改
[ ] manager/unit/GPU inventory完整
[ ] cleanup plan已冻结
[ ] cleanup authorization有效
[ ] conflict units及triggers已runtime-mask
[ ] manager endpoint/identity稳定
[ ] target/conflict/dependency failed set为空
[ ] unrelated failed exact allowlist无漂移
[ ] NRestarts稳定
[ ] selected GPU UUID/PCI/minor/compute/MIG/MPS固定
[ ] unknown GPU consumer count = 0
[ ] 覆盖最大RestartUSec/trigger周期的稳定观测PASS
[ ] integration authorization有效
[ ] true user-systemd dummy integration PASS
[ ] dummy unit已移除
```

创建 actual r2 authorization 前：

```text
[ ] unit-realization authorization有效
[ ] actual unit LoadState=loaded
[ ] actual unit UnitFileState=static
[ ] actual unit inactive/dead
[ ] actual unit Restart=no / NRestarts=0
[ ] NeedDaemonReload=no
[ ] unit fragment SHA冻结
[ ] r2 execution adapter path/SHA已进入runtime/source closure
[ ] 103文件科学closure逐字节未变
[ ] r2 run ID/schema/access/auth/receipt paths与r1完全分离
[ ] identity summary未创建任何r2 protocol artifact
[ ] metadata-only scientific r2 access audit/preaccess authorization有效
[ ] runtime spec精确绑定scientific access/preaccess path+SHA+fingerprint
[ ] runtime supervisor v2 tests PASS
[ ] isolated release matrix PASS
[ ] environment policy/receipts已绑定
[ ] runtime/source/unit/environment closure完整
[ ] actual result root不存在
[ ] actual attempt commit不存在
[ ] actual materialization claim不存在
[ ] GPU lease不存在
[ ] 项目receipt未观察到preauthorization payload访问；若无系统open-event ledger则明确“未独立审计”
[ ] D_V/D_T仍未授权
```

启动前：

```text
[ ] scientific preaccess与runtime launch两份authorization均只读且验签
[ ] runtime launch authorization精确绑定scientific preaccess/access audit
[ ] live manager endpoint/identity稳定
[ ] target/conflict/dependency failed set为空
[ ] unrelated failed allowlist无漂移
[ ] conflict units仍masked/inactive
[ ] GPU仍无未知consumer
[ ] target unit仍loaded/static/inactive
[ ] source closure未漂移
[ ] unit fragment未漂移
[ ] r2 adapter只允许real/preaccess modes且real要求runtime attestation
[ ] runtime spec未漂移
[ ] GPU lease成功create-once
[ ] lease后最终manager/unit/GPU双快照PASS
```

---

# 28. 最终研究判断

科学主线已经确定为 Base anchor、单一零水平集 completion、解决 common–residual compatibility，以及相对有效 Base 的性能提升/非回退与安全约束。**具体 gate 公式不是永久主线，而是 v24 候选。**

当前执行选择是：

1. 推荐分支：保留 v24 候选身份，完成 user-systemd/环境闭合并运行 fresh r2；structural PASS 后进入 OOF-4；
2. 合法并行分支：在 actual r2 authorization 前终止 v24，建立 v25，有证据地修改 objective/normalization/gate/readout 等非主线组件；
3. 无论哪一分支，性能门禁不采用武断的固定绝对提升幅度，而采用相对冻结有效 Base 的严格提升或预注册非回退，以及 FA/retention 等安全约束；
4. 若进入 Formal，预算固定为 **800 epochs × 40 steps/epoch**。

这条路线最能避免：

- 把环境故障误报为模型失败；
- 把模型修改夹带进恢复 attempt；
- 把“主线不变”误写成“任何非主线永不允许修改”；
- 再次因运行可观测性丢失而浪费唯一 attempt。

---

# 29. 主要源码与协议索引

## GCR-PACRE

- 核心模型  
  https://github.com/Arialliy/cure-lite/blob/main/cure_lite_v24/gcr_pacre.py

## Runtime supervisor

- supervisor  
  https://github.com/Arialliy/cure-lite/blob/main/tools/cure_lite_v24_runtime_supervisor.py
- systemd template  
  https://github.com/Arialliy/cure-lite/blob/main/deploy/systemd/cure-lite-v24-gcr-pacre-dr-r2.service.template

## Protocol

- v24 protocol README  
  https://github.com/Arialliy/cure-lite/blob/main/protocols/IRSTD-1K/gcr_pacre_v24/README.md
- protocol utilities  
  https://github.com/Arialliy/cure-lite/blob/main/tools/gcr_pacre_v24_protocol.py
