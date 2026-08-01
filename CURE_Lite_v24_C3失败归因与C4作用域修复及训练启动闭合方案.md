# CURE-Lite v24：C3/C4 失败归因、C5 追加式修复与训练启动闭合方案

> 状态快照：2026-08-01 12:31（Asia/Shanghai）  
> 仓库：`Arialliy/cure-lite`  
> 当前科学候选：`GCR-PACRE-v24`  
> 当前科学执行（attempt 2）：\(D_R=0\)、OOF-4 \(=0\)、bounded-400 \(=0\)、Formal seed42 \(=0/800\)、seed43 \(=0/800\)；\(D_V/D_T\) 未授权、未访问  
> 当前运行状态：无训练进程、无相关后台进程、无 GPU compute process  
> 当前决定：**C5 已在新授权下执行至 B5 seal；R5/E5 均 PASS，但 B5 seal 因 9 字段与 19 字段 archival-root 契约不一致非零，已按失败即停终止；不得重试 C5，仍不得启动科学训练**  
> C3 终态：**已唯一封存并独立验签 PASS**（raw SHA256 `527eb5c12c92e19dac8f797868de2bc8462e53b8113c24f6e701e0e54a26180a`）  
> C4 终态：**R4 与 E4 均 PASS，B4 receipt seal 失败已由 create-once terminal 唯一封存；C4 receipt/r14/runtime/scientific 均未创建。**  
> C5 当前状态：**B5 authorization、R5 authorization/receipt、C5 static unit、E5 scope/policy/attempt/stability/postcleanup 已 create-once；B5 receipt、r14、L5、runtime、run 全部不存在。**  
> 推荐修复：**保持 v24 候选和科学 attempt r2 完全不变；不修改、不重跑 C4/C5；封存 C5 seal 失败后建立 append-only C6，B6 只通过 E6 权威 loader 消费扩展后 archival roots。**

---

# -3. 2026-08-01 C5 生产执行结果与 C6 必要性

## -3.1 已成功且不可重放的 C5 阶段

在用户重新明确授权“修改后继续”后，以下命令各执行一次并以
退出码 0 完成：

```text
B5 authorize-c5
R5 authorize
R5 apply
E5 create-policy
E5 stability-gate（2 samples，30 秒间隔）
E5 postcleanup
```

C5 unit 当前为：

```text
loaded / inactive / dead / static
Restart=no
NRestarts=0
NeedDaemonReload=no
InvocationID=""
```

E5 权威 loader 对完整 scope/policy/stability/postcleanup 闭合的只读复验
PASS。全部 C5 JSON 均为 `0444`/uid1008/nlink1，且
`D_R_payload_accessed=false`、`D_V_payload_accessed=false`、
`D_T_payload_accessed=false`。

## -3.2 唯一失败点

`B5 seal-receipt` 唯一调用以退出码 1 结束：

```text
PermissionError: r5 archival evidence root changed
```

失败发生在 B5 receipt writer 之前。按失败即停规则，r14、L5、S5、
runtime spec、launch authorization、runtime artifacts、GPU lease 与科学 run
均未创建。

## -3.3 精确根因：R5 producer root 与 E5 archival root 层次混用

R5 权威 producer `validate_archival_realization_chain()` 返回的两个身份
root 是 9 字段 supplied identity：

```text
path, resolved_path, path_is_symlink, file_sha256,
device, inode, owner_uid, mode, nlink
```

E5 的 `_production_archival_validator()` 随后调用
`_bind_r5_archival_root()`，将其扩展为 19 字段 archival root，新增：

```text
owner_gid, size, mtime_ns, ctime_ns,
parent_path, parent_device, parent_inode,
parent_owner_uid, parent_owner_gid, parent_mode
```

E5 scope handoff 已正确封存这两个 19 字段 root，且 E5 权威
`load_c5_environment_closure()` 可完整验证。但 B5 `_collect_full_closure()`
没有使用该 loader，而是：

1. 直接调用 R5 producer，得到 9 字段 roots；
2. 自行构造 `archival` mapping；
3. 将这个 9 字段 mapping 传给 E5
   `validate_c5_environment_closure()`；
4. E5 `_validate_r5_archival_root()` 精确要求 19 字段，因此拒绝。

实物只读对比证明：

```text
B5/R5 direct authorization_identity fields = 9
B5/R5 direct receipt_identity fields       = 9
E5 scope-handoff authorization_identity     = 19
E5 scope-handoff receipt_identity           = 19
E5 authoritative closure loader             = PASS
```

因此这不是文件、inode、hash、systemd 或 GPU 漂移，而是 B5
跨 producer 时跳过 E5 root-extension 权威边界。

## -3.4 C6 修复边界

C5 已产生 authorization、R5/E5 PASS 证据与 static unit，因此不得修改
B5/E5 后重试 C5。后续必须 append-only 建立 C6：

1. create-once 封存 C5 B5-seal failure；
2. B6 不得自行把 R6 producer 的 9 字段 roots 直接传入 E6 validator；
3. B6 必须只通过 E6 权威 `load_c6_environment_closure()` 消费完整
   environment closure；
4. 如需交叉比对 R6，必须将 R6 supplied identity 与 E6 返回的扩展
   root 按明确 projection 比对，不得混用两种 schema；
5. 增加不 mock R6/E6 producer 边界的实际文件闭合测试；
6. C6/S6/R6/E6/L6/adapter6/template6 必须全部使用新命名空间和新
   source pins。

科学候选仍为 v24，attempt 仍为 r2/ordinal 2，未消耗任何科学
step。Formal 计划仍为每 seed 800 epochs，不引入固定性能提升门槛。

---

# -2. 2026-08-01 C5 写前授权拒绝与离线修复

## -2.1 已发生的唯一生产调用

`B5 authorize-c5` 在任何 writer、systemd、GPU 或科学入口之前以退出码
1 拒绝：

```text
PermissionError: c3 environment-failure transition changed
```

已按“任一生产步骤非零即停”立即终止执行。R5、E5、B5 seal、
r14、L5、S5 均未调用，训练未启动。

## -2.2 精确归因

冻结的 C3 terminal 正确表达历史 `C3 -> C4` 过渡：

```json
{"c4_required":true}
```

B5 却在验证这份历史 C3 terminal 时错误要求：

```json
{"c5_required":true}
```

`c5_required:true` 只属于直接前驱 C4 receipt-seal failure terminal。C3
terminalizer SHA、terminal SHA/fingerprint、mode/uid/nlink 与全部内嵌 root
均精确匹配，因此这不是环境或证据漂移。

## -2.3 已完成的修复与重冻结

B5 已改为对 C3 检查 `c4_required:true`，对 C4 仍严格检查
`c5_required:true`。新增了真实 C3/C4 terminal 正向闭合、3 组 C3 负例、
2 组 C4 负例，并使 authorization 临时目录测试不再 mock 这两个
terminal validator。

修正后冻结 SHA256：

```text
B5  388843b9b840db41610d57543f4982666cdf442ba81fa5acb208033de062319f
S5  12c93e469b03e5b4b6f626e875a0934f603061c840b6614221748ac2cdd3dda2
R5  dbe35cd096554c4fd4c64b34213b0f7ac3ccb79e396f6d1d8e620c2c4c1d1be5
E5  69c6f3f77acd68de94cf839dfece53ccc9c81858b3867d17edfda894792b13fb
L5  92563a61403f3a626699b94d3d60abce5f5d9a31ae5017a103b5cfdc484ac1cf
A5  bc31c82378291ef19d747c1594e0c5a9bf92b9e6fc410a82224603e2a00e8f6f
T5  f2a3da0862addb90e61301c97e0d5c1d109e8cbf59ad86c2e5130235f8387216
```

离线验证：

```text
B5                                      42 passed
S5 + adapter5                          108 passed
R5 + E5 + L5                           187 passed
C4 terminal + 全部 C5（同一解释器）   355 passed
shared/base 扩展回归                    529 passed, 1 timing-sensitive failure
```

唯一扩展回归失败是未修改的 base supervisor 的 90 ms toy child
心跳数断言：当首个 create-once heartbeat 的 fsync 耗时超过 toy child 寿命时，
只产生 `child_started` 与 `child_reaped` 两份完整哈希链事件，而测试固定要求
至少 3 份。该文件不在本次 diff 中，不改冻结 base 实现。

当前 C5 生产路径全部 absent，unit 仍为
`not-found/inactive/dead/NRestarts=0/NeedDaemonReload=no`，3 张 GPU 均为 0%
utilization。下一步必须先获得新的显式执行授权，不能直接重放旧命令。

---

# -1. 2026-07-31 C4 生产执行结果

## -1.1 已成功且不可改写的部分

C4 生产窗口只执行了一次。以下阶段均以退出码 0 完成：

```text
B4 authorize-c4
R4 authorize
R4 apply
E4 create-policy
E4 stability-gate（精确 2 samples、1 × 30 秒等待）
E4 postcleanup
```

当前 C4 unit 为：

```text
loaded / inactive / dead / static
Restart=no
NRestarts=0
NeedDaemonReload=no
InvocationID=""
```

R4 terminal 与 E4 stability terminal 均不存在。C4 runtime spec、launch
authorization、runtime artifact、GPU lease、run alias、科学 receipt、r14 与
Formal 训练均不存在。

冻结源码为：

```text
B4  ad660b7afe7ca87f690bc9565bd6674684c2b62824394751a39114a6efcf178a
S4  faffe980cba4cad668a7d0f525bed8f2005950503d46f2b7c6888d79813c64ce
R4  8708f8a13d74623f510992e23c6c23e1c4bfe70db09092c04fe56d44d29c5b65
E4  f4335efdb3865efe68dbbb6aac5f7977fd2157452b557f83428e4dd4a5d8932b
L4  4538cb198edc03ac18ac22400324fb572dfb4c3af62bc3550831b65c85a486f9
A4  fa5fe28eacf3980720616a9b54dc1dc878f1e1280883f643a49ec4c4800e92ed
```

生产前最终组合测试为 `264 passed`。

## -1.2 唯一失败点

`B4 seal-receipt` 唯一调用以退出码 1 结束，首个确定性拒绝点为：

```text
PermissionError: sealed fingerprint changed:
.../r2_preaccess_compat_c4_unit_realization_authorization.json
```

证据没有被改写。根因是 producer/consumer canonical profile 不一致：

```text
R4 producer: ensure_ascii=False，字面 UTF-8
B4 consumer: ensure_ascii=True，非 ASCII 转义为 \uXXXX
```

R4 authorization 中含：

```json
{"authorization_basis":"user instruction: 修改后运行"}
```

因此同一 JSON 对象产生：

```text
R4 recorded / UTF-8 recomputation:
543f794fd27e6277471eb2e52ab290a228415091c3071070cf3f0920c3d28c10

B4 ASCII-escaped recomputation:
11b4f19ae10d7b032af4eb7611e8b36155be6cf577149450128d6b439b14cb44
```

即使只特判 R4 authorization，E4 scope handoff 内嵌相同中文，仍会在下一步
发生同类失败。因此不能修改一个哈希或采用“双 profile 尝试”来绕过。

## -1.3 C5 决策

C4 fragment 固定调用 S4，而 S4 固定要求缺失的 B4 receipt；R4/E4 又已经封存
该 fragment。故不得修改 B4、S4、C4 fragment，也不得向旧 C4 receipt 路径
伪写恢复结果。

C5 必须：

1. 先 create-once 封存 C4 B4-seal failure；
2. 新建 B5、S5、R5、E5、L5、adapter5 与独立 C5 static unit；
3. B5 自有 JSON 统一使用 `ensure_ascii=False`；
4. foreign evidence 必须按固定路径、schema、producer source 调用其权威验证器，
   禁止自动猜测 canonical profile；
5. 重新执行一次完整 E5 30 秒环境门禁；
6. 科学 candidate、attempt ID/ordinal、输出身份与后续 800 × 40 Formal 计划保持不变。

---

# 0. 执行结论

当前没有新的模型性能失败，也没有消耗科学训练 attempt。

C3 的唯一稳定性门禁在第一个 live sample 之前失败，根因不是：

- GPU 波动；
- user-systemd 状态漂移；
- GCR-PACRE 数学问题；
- \(D_R\) 数据问题；
- OOF、bounded 或 Formal 性能不足。

根因是一个确定的**作用域重绑定错误**：

1. C3 首先用旧 precleanup 的原始目标作用域，合法重放历史合同；
2. 随后只把：
   ```text
   target_unit_id
   require_target_ready
   ```
   两个字段替换为 C3；
3. C3 policy 也按这个新合同成功构造；
4. 但正式稳定性函数又调用旧的高层：
   ```python
   run_environment_stability_gate(...)
   ```
5. 该高层入口会重新执行：
   ```python
   prepare_environment_stability_contract(...)
   ```
   并把旧 precleanup 当作 C3 目标的原生根；
6. 旧 precleanup 中的 target scope 与 C3 target scope 不同，因此在 sample loop 之前失败；
7. `audit_environment_once()` 从未被调用，故不是一次实际环境稳定性负结果。

C4 修订前草稿仍然存在同一错误：

```python
if gate is None:
    gate = _frozen.run_environment_stability_gate
```

所以即使现在补齐 C4 authorization 和 unit，仍会在第一个 sample 之前重复失败。

此外，修订前 B4 草稿仍然只继承 C1/C2 失败链：

- 顶部文档仍写“after consumed c2 failure”；
- source labels 没有 C3 failure finalizer；
- evidence labels 没有 C3 failure terminal；
- `authorize_c4()` 只检查 old/C1/C2；
- C3 authorization、R3 receipt、E3 policy 与 C3 预采样失败尚未成为 B4 的合法前驱；
- B4 引用的 R4、S4、L4、adapter4 文件当前尚不存在。

因此下一步必须按以下顺序：

```text
C3 失败事实封存
    ↓
完成 C4 所有可执行源码和测试
    ↓
重写 B4，使其从 C3 terminal 迁移
    ↓
重写 E4，禁止调用旧高层稳定性入口
    ↓
建立 R4 static unit
    ↓
生成 E4 policy
    ↓
以显式 C4 contract 执行两次 live sample
    ↓
E4 postcleanup
    ↓
B4 最终 compatibility receipt
    ↓
r14 无数据真实 user-systemd integration
    ↓
验签已提前落盘并冻结的 S4/L4/adapter4 source closure
创建 runtime spec/actual r2 authorization
    ↓
唯一一次真实 \(D_R\)-structural
```

该方案能够使**当前 C4 元数据门槛**具有正确、可执行的通过路径，但不能承诺后续真实 \(D_R\)-structural、OOF、bounded 或 Formal 的科学结果一定通过。

---

# 1. 当前身份边界

## 1.1 科学主线

当前主线保持：

\[
D=\frac12\left(R(O)-R(\widetilde O)\right),
\]

\[
E=\frac12\left(C(O)+C(\widetilde O)\right),
\]

\[
G=2\sigma(E),
\]

\[
I=GD.
\]

最终：

\[
\phi=0.9+I,
\]

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

当前 C4 修复不能改变：

- GCR-PACRE 数学；
- normalization；
- gate 参数化；
- readout；
- objective；
- loss 权重；
- scientific training/data sampling（不含本方案修复的 environment audit sampling）；
- optimizer；
- schedule；
- width；
- attention；
- feature adapter；
- output threshold；
- matching/evaluation 规则。

任何上述科学模型/训练修改都必须创建新的模型候选版本，不能称为 v24-r2；
metadata、systemd 与 environment-audit 实现修复不改变候选身份。

## 1.2 候选身份

GCR-PACRE-v24 继续作为冻结候选。

当前仅允许修改：

- compatibility metadata lane；
- failure terminal；
- source/evidence closure；
- environment wrapper；
- unit realizer；
- runtime supervisor/release/adapter；
- user-systemd integration；
- authorization 和 receipt schema。

## 1.3 执行身份

C3：

```text
B3 authorization             已生成、已过期
R3 unit authorization        已生成
R3 unit receipt              PASS
E3 policy                    已生成
C3 stability receipt         不存在
C3 postcleanup receipt       不存在
C3 compatibility receipt     不存在
C3 scientific runtime spec   不存在
C3 unit start                未发生
C3 payload access            未发生
```

因此：

- C3 已经消耗其 compatibility authorization；
- C3 不得重试；
- 科学 attempt 2 尚未消耗；
- C4 是新的 metadata compatibility generation，不是新的科学候选。

---

# 2. C3 已经成立的事实

## 2.1 R3 unit 状态正确

封存的 R3 receipt 记录：

```text
LoadState      loaded
ActiveState    inactive
SubState       dead
UnitFileState  static
Restart        no
NRestarts      0
started        false
enabled        false
runtime spec   absent
```

这证明：

- C3 unit 实际被 user manager 加载；
- unit 没有启动；
- 没有 restart；
- 没有 runtime spec；
- 没有 child；
- 没有科学 payload。

## 2.2 C3 policy 的新 scope 已经形成

C3 wrapper 明确：

1. 用 `OLD_TARGET_UNIT` 和 `require_target_ready=False` 重放旧 scope；
2. 通过 `dataclasses.replace()` 只改变：
   ```text
   target_unit_id = C3_TARGET_UNIT
   require_target_ready = True
   ```
3. 检查其他字段逐项未改变。

这一部分思路正确。

## 2.3 失败发生在 sample 之前

冻结高层 `run_environment_stability_gate()` 的顺序是：

```text
validate request
prepare_environment_stability_contract(...)
load/validate policy
compare policy with rebuilt contract
initialize samples=[]
for sample:
    audit_environment_once(...)
```

重新准备 contract 在 sample list 和循环之前。

C3 传给该入口的是：

```text
old precleanup path
old cleanup path
C3 target unit
require_target_ready=True
```

所以它在调用 `audit_environment_once()` 之前，就用旧根重建 C3 scope，并触发冲突。

---

# 3. 精确失败调用链

当前调用链可写成：

```text
run_c3_stability_in_memory()
    │
    ├── replay_old_scope_and_handoff()
    │       │
    │       ├── prepare(old precleanup,
    │       │           target=OLD_TARGET,
    │       │           ready=False)
    │       │       -> old_contract       PASS
    │       │
    │       └── replace(old_contract,
    │                   target=C3_TARGET,
    │                   ready=True)
    │               -> c3_contract        PASS
    │
    ├── validate E3 policy against c3_contract
    │       -> PASS
    │
    └── run_environment_stability_gate(
            old precleanup,
            old cleanup,
            target=C3_TARGET,
            ready=True
        )
            │
            └── prepare_environment_stability_contract(
                    old precleanup,
                    target=C3_TARGET,
                    ready=True
                )
                    -> scope mismatch
                    -> exception
                    -> no sample 0
```

这里的逻辑矛盾是：

> C3 wrapper 已经知道旧 precleanup 只能按旧 target 解释，却又把同一个旧 precleanup
> 传给一个会按 C3 target 重新解释根证据的高层函数。

---

# 4. 为什么这不是环境稳定性失败

环境稳定性失败至少要求：

```text
audit_environment_once() 被调用
sample receipt 被生成
inventory 被观察
state projection 被比较
NRestarts/GPU/manager 等产生 blocker
```

C3 当前没有任何这些事实。

因此不能写成：

- “GPU 在 sample 0 波动”；
- “manager 在 30 秒窗口内不稳定”；
- “C3 target unit 状态不合格”；
- “conflict service 重启”；
- “环境门禁采样失败”。

准确结果应为：

> **C3 compatibility generation 在进入 stability sampling 前，因 root-scope
> re-preparation 冲突而终止。**

---

# 5. 为什么 C3 不得重试

C3 的短期 authorization 已过期，并且唯一一次稳定性门禁已经被调用。

即使没有 scientific payload，也不能：

- 重新签发同一个 C3 authorization；
- 修改 E3 policy 后再跑；
- 在 C3 名字下更换函数；
- 补写 C3 stability PASS；
- 将 C3 的失败删除后继续。

否则会破坏：

- 一次性 compatibility generation；
- create-once chronology；
- authorization scope；
- source closure；
- 失败证据。

正确做法是：

```text
C3 -> append-only terminal
C4 -> fresh compatibility generation
```

---

# 6. C3 失败终结器（已完成）

已使用下列实际文件名完成 17 项定向测试、生产前只读预检、
唯一一次 create-once 封存与两条独立归档验签：

```text
tools/
cure_lite_v24_preaccess_compat_c3_environment_stability_failure_terminal.py
```

已封存 terminal：

```text
protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/
r2_preaccess_schema_compat_c3_environment_stability_failure_terminal.json
```

冻结身份：

```text
terminalizer SHA256  b55a916dade97b9d49f1cd80758aeaac316d55725eb7ee4e1148c7c206aa9d9f
terminal raw SHA256  527eb5c12c92e19dac8f797868de2bc8462e53b8113c24f6e701e0e54a26180a
terminal fingerprint c31159e7033450ecc2a8dea071fd125ab756e43afbc8d8c433c425a045713670
mode / nlink / size   0444 / 1 / 14781 bytes
sealed_at_utc         2026-07-31T13:58:52.550423Z
```

## 6.1 Terminal schema

建议：

```text
cure-lite-v24-r2-preaccess-schema-compat-c3-
environment-stability-failure-terminal-v1
```

## 6.2 必须绑定的证据

```text
B3 authorization
R3 unit authorization
R3 unit receipt
E3 policy
sealed cleanup receipt
C3 bridge source
C3 environment wrapper source
frozen runtime-environment source
C3 unit fragment
C3 unit current shadow
C3 authorization expiry
C3 stability path absence
C3 postcleanup path absence
C3 compatibility receipt absence
C3 runtime spec absence
C3 runtime artifact root absence
scientific output absence
D_R/D_V/D_T access false
```

## 6.3 必须保存的失败身份

```json
{
  "failure_phase": "before_first_stability_sample",
  "sample_count_observed": 0,
  "inventory_collector_called": false,
  "failure_class": "root_scope_reprepare_conflict",
  "failed_generation": "c3",
  "same_generation_retry_allowed": false,
  "next_compatible_generation": "c4",
  "scientific_attempt_consumed": false,
  "training_attempt_consumed": false
}
```

## 6.4 不得伪造运行异常

若 C3 当时没有生成 durable stderr/exit receipt，terminal 不能凭记忆写入一个
“实际 exception message”。

推荐使用一个只读、无 payload 的**纯 sealed scope projection 证明**。不得再次
调用 `run_c3_stability_in_memory()`、E3 gate、inventory collector、sleeper、
monotonic clock 或任何 writer：

```python
historical_scope = project_scope(
    sealed_precleanup_receipt["inventory"]
)
requested_scope = project_scope(sealed_c3_policy)

assert historical_scope["target_unit_id"] == OLD_TARGET_UNIT
assert historical_scope["require_target_ready"] is False
assert requested_scope["target_unit_id"] == C3_TARGET_UNIT
assert requested_scope["require_target_ready"] is True
assert scope_equal_except(
    historical_scope,
    requested_scope,
    {"target_unit_id", "require_target_ready"},
)
assert frozen_scope_validator_category(
    historical_scope,
    requested_scope,
) == "root_scope_reprepare_conflict"
```

该证明只读取已封存 JSON 和已固定源码根，不重入 C3 运行路径，不读取 split、
不查询 GPU、不写 C3 stability。若没有 durable 原始 stderr/exit artifact，terminal
必须标记为 `post_hoc_deterministic_read_only_reproduction`，不得声称原始 argv、
失败时间或 traceback。

## 6.5 Terminal 成功条件

```text
C3 authorization已过期
C3 unit仍 static/inactive/dead/NRestarts=0
C3 stability不存在
C3 postcleanup不存在
C3 compatibility receipt不存在
C3 runtime spec不存在
deterministic pre-sample failure replay成立
collector call count = 0
所有历史 evidence SHA未变化
terminal fd 已打开期间再次确认固定 roots、absence 与 C3 inert shadow 未变化
```

---

# 7. 当前 B4 草稿为什么不能使用

当前 B4 文件顶部仍声明：

> c4 在 consumed c2 failure 后闭合。

这与当前真实 lineage 不一致。

## 7.1 Source closure 缺 C3

当前 `_SOURCE_LABELS` 包含：

```text
c1_failure_terminalizer
c2_mode_contract_failure_terminalizer
c2_prewrite_failure_terminalizer
```

但没有：

```text
c3_environment_stability_failure_terminalizer
c3_environment_wrapper
c3_unit_realizer
c3_unit_template
```

## 7.2 Evidence closure 缺 C3

当前 `_EVIDENCE_LABELS` 包含：

```text
c1_failure_terminal
c2_mode_contract_failure_terminal
c2_prewrite_failure_terminal
r10_authorization
r10_receipt
...
```

但没有：

```text
c3_failure_terminal
B3 authorization
R3 authorization
R3 receipt
E3 policy
```

## 7.3 `authorize_c4()` 仍只检查 old/C1/C2

当前授权代码读取：

```text
old_state
c1_state
c2_state
```

并验证 C1/C2 predecessor terminal。

它没有：

- 读取 C3 terminal；
- 检查 C3 authorization 过期；
- 检查 C3 unit；
- 检查 C3 stability/postcleanup/receipt absence；
- 检查 C3 未启动科学运行。

所以当前 B4 无法合法证明：

> C4 是在 C3 已经不可重试之后建立的下一代 compatibility lane。

## 7.4 B4 引用了不存在的 C4 源码

当前草稿预期：

```text
cure_lite_v24_actual_runtime_release_preaccess_compat_c4.py
cure_lite_v24_runtime_supervisor_preaccess_compat_c4.py
run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c4.py
cure_lite_v24_actual_unit_realization_preaccess_compat_c4.py
```

但当前仓库中这些文件尚不存在。

因此：

- source closure 无法完成；
- SHA 不能冻结；
- C4 unit template 中的 supervisor path 尚无合法目标；
- B4 authorization 不得创建。

## 7.5 B4/E4 还存在十六进制哈希盲替换损坏

当前草稿不能通过“继续补文件”修好，因为机械 `c3 -> c4` 替换误改了 SHA-256
字符串本身。

B4 当前错误固定：

```text
C2 mode-contract terminal
草稿：e478e0cc4516...
真实：e478e0cc3516...

C2 prewrite terminal
草稿：6984dc9d...b7bc4b...
真实：6984dc9d...b7bc3b...
```

E4 当前错误固定：

```text
frozen environment core
草稿：a40465786ce453...a1dc402...
真实：a40465786ce353...a1dc302...

R4 bridge
草稿：cdbbe...fe43de1
真实状态：R4 尚不存在，不能拥有有效 SHA
```

因此必须：

1. 从可信前驱重新恢复所有历史固定哈希；
2. 对尚不存在或尚未冻结的 C4 源使用显式 fail-closed sentinel；
3. 文件冻结后用 `sha256sum` 重新计算并由测试独立核对；
4. 禁止对源码全文做无边界的代际字符串替换；
5. 在签发 B4 authorization 前运行“源码常量 SHA = 实际文件 SHA”的全闭包检查。

---

# 8. B4 的正确迁移合同

## 8.1 新前驱

B4 的唯一最新前驱应为：

```text
C3 stability-presample failure terminal
```

C1/C2 仍作为历史祖先，但不再作为直接 continuation gate。

## 8.2 新 source labels

至少加入：

```text
c3_failure_terminalizer
c3_environment_wrapper
c3_unit_realizer
c3_unit_template
compat_bridge_c4
environment_wrapper_e4
unit_realizer_r4
runtime_supervisor_s4
runtime_release_l4
runtime_adapter4
unit_template_c4
```

## 8.3 新 evidence labels

至少加入：

```text
c3_failure_terminal
c3_bridge_authorization
c3_unit_realization_authorization
c3_unit_realization_receipt
c3_environment_policy
c4_scope_handoff_receipt
c4_environment_policy
c4_environment_stability
c4_environment_postcleanup
c4_unit_realization_authorization
c4_unit_realization_receipt
```

## 8.4 B4 authorization 前置条件

```text
C3 failure terminal PASS
C3 authorization expired
C3 same-generation retry forbidden
C3 unit static/inactive/dead/NRestarts=0
C3 stability/postcleanup/compatibility outputs absent
C3 runtime spec/claim/start/artifacts absent
scientific outputs absent
D_R/D_V/D_T payload unaccessed
C4 output namespace empty
R4/S4/L4/adapter4 sources already exist
所有 C4 sources/tests已经冻结
```

## 8.5 B4 authorization 时长

建议继续使用 metadata-only 的短授权，但在签发前完成全部源码和测试。

可保持：

```text
validity <= 300 seconds
```

关键不是延长，而是：

> authorization 必须最后签发，签发后只执行已经测试完的确定性 metadata lane。

## 8.6 B4 不授权科学执行

固定：

```text
D_R_payload_authorized        false
D_V_payload_authorized        false
D_T_payload_authorized        false
training_authorized           false
runtime_start_authorized      false
scientific_attempt_consumed   false
```

---

# 9. E4 的核心修复原则

E4 不得再调用：

```python
_frozen.run_environment_stability_gate
```

因为该函数将旧根与请求的新 target scope 重新绑定。

E4 必须：

1. 用旧 scope 重放 old contract；
2. 只替换两个 handoff 字段形成 C4 contract；
3. 验证 R4 unit realization；
4. 验证 E4 policy 与 C4 contract；
5. 直接调用：
   ```python
   audit_environment_once(c4_contract)
   ```
6. 收集两次 sample；
7. 调用：
   ```python
   evaluate_environment_stability(
       c4_contract,
       root_evidence,
       samples,
       ...
   )
   ```
8. 验证 receipt；
9. 全程不再用 precleanup 重建 C4 contract。

---

# 10. E4 应减少可调用 API

当前 C3/C4 wrapper 的 frozen read facade 暴露：

```text
prepare_environment_stability_contract
run_environment_stability_gate
audit_environment_once
evaluate_environment_stability
```

E4 应从 `_FROZEN_READ_API` 中移除：

```text
run_environment_stability_gate
```

仅从 `_FROZEN_READ_API` 删除还不够：当前 wrapper 还保存
`_FROZEN_ALL_CALLABLES`。E4 必须同时保证该高层 gate 不存在于任何可达 facade、
callable registry、alias 或默认参数中，并由 AST 与运行时 identity 测试共同证明。

保留：

```text
prepare_environment_stability_contract
```

仅用于：

```text
OLD_TARGET_UNIT
require_target_ready=False
```

这样 production E4 代码无法再次误调用高层入口。

建议增加静态断言：

```python
if "run_environment_stability_gate" in _FROZEN_READ_API:
    raise PermissionError(
        "E4 must not expose scope-repreparing high-level gate"
    )

if "run_environment_stability_gate" in _FROZEN_ALL_CALLABLES:
    raise PermissionError(
        "E4 must not retain the scope-repreparing gate"
    )
```

---

# 11. 新增显式 scope-handoff receipt

建议新增：

```text
runtime_environment_scope_handoff_preaccess_compat_c4.json
```

schema：

```text
cure-lite-v24-runtime-environment-scope-handoff-c4-v1
```

内容：

```json
{
  "old_contract": {},
  "new_contract": {},
  "changed_fields": [
    "target_unit_id",
    "require_target_ready"
  ],
  "old_target_unit_id": "cure-lite-v24-gcr-pacre-dr-r2.service",
  "new_target_unit_id": "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service",
  "old_require_target_ready": false,
  "new_require_target_ready": true,
  "precleanup_root": {},
  "cleanup_root": {},
  "c3_failure_terminal_root": {},
  "r4_unit_receipt_root": {},
  "D_R_payload_accessed": false,
  "D_V_payload_accessed": false,
  "D_T_payload_accessed": false
}
```

要求：

- exactly two changed fields；
- 所有其他 contract fields exact equal；
- old roots 仍按 old scope 解释；
- new contract 不声称来自 precleanup 原生 scope；
- R4 unit receipt证明 new target 已 ready；
- receipt create-once。

该 receipt 不修改 frozen environment policy schema；它由 E4 wrapper 和 B4 closure
单独绑定。

---

# 12. E4 正确实现骨架

建议将当前 `run_c4_stability_in_memory()` 改成显式合同版本。

```python
def run_c4_stability_from_contract(
    policy,
    *,
    realization_validator,
    inventory_collector=None,
    sleeper=None,
    monotonic_clock=None,
    auditor=None,
    evaluator=None,
    live_roots=None,
):
    # 1. Frozen source and namespace.
    _require_frozen_environment_source()
    _require_c4_namespace()

    if auditor is None:
        auditor = _frozen.audit_environment_once
    if evaluator is None:
        evaluator = _frozen.evaluate_environment_stability
    if sleeper is None:
        sleeper = time.sleep
    if monotonic_clock is None:
        monotonic_clock = time.monotonic

    # 2. Replay roots only under their historical scope.
    old_contract, c4_contract, roots_before = (
        replay_old_scope_and_handoff()
    )

    _assert_exact_two_field_handoff(
        old_contract,
        c4_contract,
        changed_fields={
            "target_unit_id",
            "require_target_ready",
        },
    )

    # 3. Verify C4 unit archival state.
    archival_before = _resolve_archival(
        realization_validator,
        contract=c4_contract,
    )
    _verify_c3_terminal_and_inert_generation()

    # 4. Load/verify the sealed policy and its live root.
    policy_value, policy_root = _read_live_sealed(
        C4_POLICY_PATH,
        "policy_fingerprint",
    )
    policy_value = _frozen.validate_environment_policy(
        policy_value
    )
    if not _frozen._deep_exact_equal(
        policy_value,
        policy,
    ):
        raise PermissionError("C4 policy payload changed")

    toolchain_before = (
        _frozen.current_runtime_toolchain_binding()
    )

    expected_policy = _frozen.build_environment_policy(
        c4_contract,
        precleanup_root_binding=(
            roots_before["precleanup_inventory_receipt"]
        ),
        cleanup_root_binding=roots_before["cleanup_receipt"],
        toolchain_binding=toolchain_before,
        minimum_sample_count=SAMPLE_COUNT,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
    )
    _require_policy_equal_except_timestamp(
        policy,
        expected_policy,
    )

    root_evidence = {
        **roots_before,
        "policy": policy_root,
    }

    # Durable one-shot boundary.  A kill or power loss after this point
    # consumes C4 stability and may only end in PASS receipt or terminal.
    attempt_commit, attempt_commit_root = (
        create_c4_stability_attempt_commit(
            c3_terminal_root=sealed_c3_terminal_root,
            handoff_root=sealed_handoff_root,
            r4_root=sealed_r4_root,
            policy_root=policy_root,
            source_roots=sealed_source_roots,
        )
    )

    # 5. Sample directly from the explicit C4 contract.
    samples = []
    monotonic_rows = []

    for index in range(SAMPLE_COUNT):
        _require_frozen_environment_source()
        _verify_c4_live_roots(root_evidence)
        _verify_scope_handoff_receipt(c4_contract)
        _verify_c3_terminal_and_inert_generation()
        _resolve_archival(
            realization_validator,
            contract=c4_contract,
        )

        audit_kwargs = {}
        if inventory_collector is not None:
            audit_kwargs["inventory_collector"] = (
                inventory_collector
            )
        sample = auditor(c4_contract, **audit_kwargs)
        samples.append(sample)
        monotonic_rows.append(float(monotonic_clock()))

        if (
            index + 1 < SAMPLE_COUNT
            and SAMPLE_INTERVAL_SECONDS > 0
        ):
            sleeper(SAMPLE_INTERVAL_SECONDS)

    # 6. Rebuild historical roots after sampling and compare.
    old_after, c4_after, roots_after = (
        replay_old_scope_and_handoff()
    )

    if not _frozen._deep_exact_equal(
        asdict(old_after),
        asdict(old_contract),
    ):
        raise PermissionError(
            "historical environment contract changed"
        )

    if not _frozen._deep_exact_equal(
        asdict(c4_after),
        asdict(c4_contract),
    ):
        raise PermissionError("C4 handoff contract changed")

    if not _frozen._deep_exact_equal(
        roots_after,
        roots_before,
    ):
        raise PermissionError("C4 root evidence changed")

    if not _frozen._deep_exact_equal(
        _frozen.current_runtime_toolchain_binding(),
        toolchain_before,
    ):
        raise PermissionError(
            "runtime toolchain changed during C4 sampling"
        )

    archival_after = _resolve_archival(
        realization_validator,
        contract=c4_contract,
    )
    if not _frozen._deep_exact_equal(
        archival_after,
        archival_before,
    ):
        raise PermissionError(
            "C4 realization archival changed"
        )

    # Close every TOCTOU window after sample 1 and before evaluation.
    _verify_c3_terminal_and_inert_generation()
    _verify_scope_handoff_receipt(c4_contract)
    _verify_live_sealed(
        C4_POLICY_PATH,
        policy_root,
        "policy_fingerprint",
    )
    _verify_exact_c4_source_roots(sealed_source_roots)
    _verify_attempt_commit(attempt_commit, attempt_commit_root)

    # 7. Evaluate the samples without rebuilding the contract.
    result = evaluator(
        c4_contract,
        root_evidence,
        samples,
        sample_interval_seconds=(
            SAMPLE_INTERVAL_SECONDS
        ),
        sample_monotonic_seconds=monotonic_rows,
    )

    result = _frozen.validate_environment_stability_receipt(
        result
    )

    validate_c4_environment_closure(
        policy,
        result,
        None,
        archival=archival_after,
        c4_contract=c4_contract,
        live_roots={
            "policy": policy_root,
        },
    )

    # The create-once stability writer must repeat the same complete
    # verification while its output fd is open and again after close.

    return result
```

正式实现应复用 wrapper 已有：

- live sealed root 读取；
- source generation guard；
- archival validator；
- no-payload recursion；
- exact JSON；
- create-once writer。

---

# 13. E4 绝对禁止的调用

测试与静态检查必须证明 production source 中不存在：

```python
_frozen.run_environment_stability_gate(...)
```

以及：

```python
prepare_environment_stability_contract(
    old_precleanup,
    target_unit_id=C4_TARGET_UNIT,
    require_target_ready=True,
)
```

允许的 `prepare` 调用只能是：

```python
target_unit_id=OLD_TARGET_UNIT
require_target_ready=False
```

---

# 14. E4 exception 行为

C4 也只能有一次 compatibility stability attempt。

因此必须在第一次 `audit_environment_once()` 前先 create-once 写入：

```text
runtime_environment_stability_attempt_preaccess_compat_c4.json
```

attempt commit 必须绑定 C3 terminal、B4 authorization、R4 receipt、handoff、
policy、C4 source roots、sample_count=2、interval=30.0、no-retry/no-payload。
一旦该 commit 存在，C4 stability attempt 即被消费；进程被 kill、掉电或 terminal
writer 失败都不得再次执行 C4。恢复动作只能封存中断终态，不能继续采样。

E4 发生异常时不能留下无解释的空洞。

建议新增独立 terminal：

```text
runtime_environment_stability_terminal_preaccess_compat_c4.json
```

字段：

```text
failure phase
samples completed
last completed sample fingerprint
exception class
canonical failure category
policy root
handoff root
R4 root
source roots
attempt commit root
same-generation retry allowed = false
scientific attempt consumed = false
```

成功时：

```text
stability receipt exists
terminal absent
```

失败时：

```text
stability PASS receipt absent
failure terminal exists
postcleanup absent
B4 final receipt absent
```

不得自动重试 C4。

成功 receipt 与失败 terminal 必须互斥，并共同绑定同一 attempt commit。任何缺少
二者的中断状态也由该 commit 证明“已经尝试”，后续只能 append-only 建立 C5。

---

# 15. R4 unit realizer

当前 C4 template 已存在，但 R4 realizer 尚不存在。

必须新增：

```text
tools/
cure_lite_v24_actual_unit_realization_preaccess_compat_c4.py
```

R4 应从 R3 机械迁移，但绑定：

```text
C4 unit name
C4 template
S4 supervisor source
C4 future runtime spec path
B4 authorization
```

## 15.1 R4 前置

```text
B4 authorization fresh
C4 template frozen
S4 source frozen
C4 fragment path absent
C4 runtime spec absent
C4 unit not found
C3 unit inert
no payload access
```

## 15.2 R4 输出

```text
R4 unit authorization
R4 unit receipt
```

## 15.3 R4 receipt 必须证明

```text
LoadState      loaded
ActiveState    inactive
SubState       dead
UnitFileState  static
Restart        no
NRestarts      0
NeedDaemonReload no
started        false
enabled        false
runtime spec   absent
```

R4 不能启动 unit。

---

# 16. S4、L4、adapter4 必须先落盘再签 B4

当前 B4 草稿已经引用这些路径，但文件不存在。

建议新增：

```text
tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c4.py
tools/cure_lite_v24_actual_runtime_release_preaccess_compat_c4.py
tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c4.py
```

## 16.1 S4

从 S3/r2 supervisor 机械迁移：

- C4 namespace；
- C4 runtime spec path；
- C4 unit；
- C4 source closure；
- 同一 no-retry/no-resume；
- same candidate；
- no scientific modification。

## 16.2 L4

负责：

- runtime spec create-once；
- launch authorization；
- source/unit/environment closure；
- result/terminal verification；
- C4 unit and adapter binding。

## 16.3 adapter4

只负责调用现有 v24 \(D_R\)-structural runner。

不得：

- 修改 model config；
- 修改 gate；
- 修改 split；
- 修改 threshold；
- 修改 scientific decision；
- 添加 retry。

## 16.4 B4 source closure

在以下文件全部存在并测试前：

```text
B4 authorization不得创建
```

禁止在 authorization 中使用：

```text
__TO_BE_FROZEN__
```

或手写未验证 SHA。

---

# 17. r14 无数据真实 user-systemd integration

建议新增一代真实 integration：

```text
r14
```

它必须在：

```text
B4 final compatibility receipt PASS
```

之后运行。

r14 使用全新、互不重叠的 scenario/evidence root，但优先复用并重新验签现有通用
integration harness，而不是复制一套未必要的新源码：

- 独立 dummy child；
- 独立 runtime spec；
- 独立 artifact root；
- hash-pinned `cure_lite_v24_user_systemd_integration.py`；
- hash-pinned `cure_lite_v24_dummy_child.py`；
- hash-pinned `cure_lite_v24_realize_systemd_unit.py`；
- hash-pinned supervisor-integration template 与最终 S4；
- 不导入 torch；
- 不访问 GPU；
- 不访问任何 split。

必须验证：

```text
C4 supervisor命令行
C4 unit fragment
ExecCondition
ExecStartPre
ExecStart
ExecStopPost
InvocationID
ControlGroup
heartbeat
process group cleanup
terminal sidecar
source drift detection
unit no restart
dummy integration unit removal
```

r14 不得创建 scientific r2 commit/claim。

这里的 removal/cleanup **只能**作用于 r14 独立 dummy integration unit。R4 创建的
生产 C4 static unit 及其 fragment 全程为 protected generation，不得被 r14 启动、
停止、重写或删除；r14 结束后必须再次验签其 inode、SHA、完整 systemd shadow 和
`NRestarts=0`。

---

# 18. B4 最终 compatibility receipt

建议 dependency：

```text
C3 terminal
    ↓
B4 authorization
    ↓
R4 receipt
    ↓
C4 handoff receipt
    ↓
E4 policy
    ↓
E4 stability PASS
    ↓
E4 postcleanup
    ↓
B4 final receipt
```

B4 receipt 必须同时绑定：

- C3 terminal；
- B4 authorization；
- R4 unit state；
- E4 three receipts；
- C4 source closure；
- all prior generations protected；
- no payload；
- no scientific attempt consumption。

---

# 19. C4 完整代码依赖图

```text
C3 failure terminalizer
        │
        ▼
C3 terminal
        │
        ▼
B4 bridge authorization
        │
        ├───────────────┐
        ▼               ▼
R4 realizer         E4 environment wrapper
        │               │
        ▼               │
R4 receipt              │
        │               │
        └──────┬────────┘
               ▼
        C4 handoff receipt
               │
               ▼
          E4 policy
               │
               ▼
        E4 direct sampling
               │
               ▼
        E4 stability PASS
               │
               ▼
        E4 postcleanup
               │
               ▼
        B4 final receipt
               │
               ▼
        r14 integration
               │
               ▼
S4 + L4 + adapter4 actual release
```

该图描述的是 evidence/authorization 时序，不是源码创建时序。S4、L4、adapter4、
r14 harness/dummy/template 源码必须在 B4 authorization **之前**完成测试并冻结；
r14 PASS evidence 只能在 B4 metadata receipt 之后产生，并由 L4 scientific release
closure 绑定，不能反向成为 B4 authorization 的运行证据前置。

---

# 20. 测试计划：C3 finalizer

至少新增：

1. C3 auth 未过期时拒绝 terminal；
2. C3 stability 已存在时拒绝；
3. C3 postcleanup 已存在时拒绝；
4. C3 final compatibility receipt 已存在时拒绝；
5. C3 runtime spec 已存在时拒绝；
6. C3 unit active 时拒绝；
7. C3 NRestarts 非零时拒绝；
8. deterministic reproduction 在 sample 前失败；
9. terminalizer 不导入/调用 E3 gate、collector、sleep、clock 或 writer；
10. exception category exact；
11. C3 source SHA 漂移拒绝；
12. terminal create-once；
13. terminal 无 payload/training；
14. terminal 不授权 C4；
15. terminal fingerprint 可独立重算。

---

# 21. 测试计划：B4

1. 缺 C3 terminal 拒绝 authorization；
2. C3 terminal fingerprint 错误拒绝；
3. C3 auth 仍 fresh 拒绝；
4. C3 same-generation retry 被标为 true 时拒绝；
5. C3 unit非 inert 拒绝；
6. C3 stability/path 出现拒绝；
7. C3 runtime spec出现拒绝；
8. C3 payload flag true拒绝；
9. C4 output namespace非空拒绝；
10. R4/S4/L4/adapter4 任一缺失拒绝；
11. source SHA 任一占位符拒绝；
12. source labels 必须含 C3 terminalizer；
13. evidence labels 必须含 C3 terminal；
14. authorization chronology晚于 C3 terminal；
15. authorization有效期固定；
16. old/C1/C2/C3 scientific paths全部保护；
17. authorization不允许 runtime start；
18. receipt必须绑定 E4/R4完整链；
19. receipt只有环境 PASS 后可写；
20. mechanical migration不得改变 candidate identity。

---

# 22. 测试计划：E4 核心修复

## 22.1 防止回归

1. monkeypatch `_frozen.run_environment_stability_gate` 为立即抛错；
2. E4 正常路径仍然 PASS；
3. 静态 AST 检查 production source 无该调用；
4. `_FROZEN_READ_API` 不暴露该名字。

## 22.2 Scope

5. historical prepare 只收到 OLD target；
6. historical prepare 的 readiness 必须 false；
7. C4 contract 只改变两个字段；
8. 第三个字段变化必须拒绝；
9. handoff receipt exact；
10. C4 policy exact匹配 handoff contract。

## 22.3 Sampling

11. auditor 确实被调用两次；
12. 第一次 sample index 为 0；
13. interval 至少 30 秒；
14. monotonic timestamp严格非递减；
15. target unit在每个 sample 中 static/inactive/dead/NRestarts0；
16. manager generation不变；
17. conflict state不变；
18. GPU inventory无 blocker；
19. sample contract等于 C4 contract；
20. `evaluate_environment_stability`收到 C4 contract，而非 old contract。

## 22.4 Root closure

21. precleanup root中途漂移拒绝；
22. cleanup root漂移拒绝；
23. E4 policy漂移拒绝；
24. handoff receipt漂移拒绝；
25. R4 receipt漂移拒绝；
26. frozen toolchain漂移拒绝；
27. activation guard漂移拒绝；
28. wrapper source generation漂移拒绝；
29. archival unit shadow漂移拒绝；
30. sample之后重放 old/c4 contract 必须一致。

## 22.5 Failure

31. sample 0 前失败生成 C4 terminal；
32. sample 1 前失败保存 completed sample count；
33. C4失败不写 PASS receipt；
34. C4失败不写 postcleanup；
35. C4失败不自动重试；
36. 所有 failure receipt 不读取 payload。
37. 首次 audit 前 attempt commit 已持久化；kill/power-loss 后拒绝再次执行；
38. sample 1 后再次验签 C3 terminal、handoff、policy、R4 与全部 source roots；
39. success receipt 与 failure terminal 互斥且绑定同一 attempt commit；
40. C3 unit 在 sample 前、两次 sample 间和 sample 后均保持受保护 inert 状态。

---

# 23. 测试计划：R4/S4/L4/adapter4/r14

## R4

```text
create-only fragment
no symlink
template SHA exact
supervisor path存在且冻结
daemon-reload
static/inactive/dead
Restart=no
NRestarts=0
runtime spec absent
no start
```

## S4

```text
same candidate
one launch
no retry
no resume
no model changes
source drift -> audit false
unit drift -> audit false
child drift -> audit false
```

## L4

```text
requires B4 receipt
requires r14 PASS
requires E4 PASS
requires exact unit/source closure
runtime spec create-once
authorization last
no D_V/D_T
```

## adapter4

```text
argv exact
environment exact
no scientific overrides
same v24 runner
payload access only after scientific authorization
```

## r14

```text
real user-systemd
dummy child
no torch
no CUDA
no split
InvocationID/cgroup exact
terminal complete
unit cleanup
```

---

# 24. C4 production 执行顺序

```text
P0
完成 C3 failure finalizer 的测试
        │
        ▼
P1
生成 C3 terminal
        │
        ▼
P2
完成 B4/E4/R4/S4/L4/adapter4/r14 全部源码
完成 C4 stability attempt-commit/failure-terminal 源码
        │
        ▼
P3
语法、静态、unit、fault-injection 测试全部 PASS
        │
        ▼
P4
冻结 source closure
确认 C4 namespace empty
        │
        ▼
P5
创建唯一 B4 authorization
        │
        ▼
P6
创建 R4 authorization
realize C4 static unit
生成 R4 receipt
        │
        ▼
P7
生成 C4 scope-handoff receipt
生成 E4 policy
创建前再次确认 stability attempt commit 尚不存在
        │
        ▼
P8
E4 direct-contract stability：
create-once attempt commit
sample 0
wait 30 s
sample 1
        │
        ├── FAIL：C4 terminal，停止
        ▼ PASS
P9
生成 E4 postcleanup
        │
        ▼
P10
生成 B4 final receipt
        │
        ▼
P11
运行 r14 dummy integration
        │
        ├── FAIL：停止，不创建 scientific runtime spec
        ▼ PASS
P12
验签 P2 已冻结的 S4/L4/adapter4 source closure
绑定 r14 PASS evidence roots，生成 scientific release closure
        │
        ▼
P13
创建 scientific r2 runtime spec
创建 launch authorization
        │
        ▼
P14
唯一一次真实 r2
        │
        ├── FAIL/中断：封存 r2 terminal，停止，不创建 r3
        ▼ PASS
P15
独立验签 r2 scientific receipt/decision
        │
        ▼
P16
创建并执行唯一 OOF-4 authorization
验签 fold/cache/pooled/gate-path decision
        │
        ├── FAIL：封存 terminal，停止
        ▼ PASS
P17
创建并执行唯一 paired bounded-400 authorization
candidate/control 各 10 × 40，fresh seed42
        │
        ├── FAIL：封存 terminal，停止
        ▼ PASS
P18
Formal seed42：create-once authorization，800 × 40
        │
        ├── FAIL：封存 terminal，停止；不得 resume/retry
        ▼ PASS
P19
Formal seed43：create-once authorization，800 × 40
        │
        ├── FAIL：封存 terminal，停止；不得 resume/retry
        ▼ PASS
P20
seed42/43 pair verification
生成 final-stage receipt
D_V/D_T authorization 仍不存在、payload 仍未访问
```

---

# 25. C4 通过门槛

C4 最终兼容通过必须满足：

```text
C3 failure terminal VALID/SEALED
C3 no-retry
B4 authorization有效
R4 unit PASS
C4 handoff exact-two-fields
E4 policy PASS
E4 sample_count = 2
E4 observed window >= 30 s
E4 blockers = []
E4 target ready in every sample
E4 postcleanup PASS
B4 receipt PASS
r14 integration PASS
D_R/D_V/D_T access = false
scientific attempt consumed = false
```

---

# 26. C4 之后的科学路线

C4 完成只意味着：

> 真实 r2 具备进入 scientific release、runtime-spec、precommit、GPU lease 和
> launch-authorization 构造的资格；C4 本身不授权或直接启动 r2。

它不意味着 GCR-PACRE 性能成功。

后续顺序保持：

```text
real D_R structural
    ↓ PASS
OOF-4
    ↓ PASS
full-D_R bounded-400
    ↓ PASS
Formal seed 42: 800 × 40
Formal seed 43: 800 × 40
    ↓ PASS
seed42/43 pair verification
    ↓
本阶段结束；D_V/D_T 仍未授权、未访问
```

放行仍采用：

- Base A 使用冻结阈值 `0.72`；
- Base B 仅在每个 fold 的 train roots 上从冻结 51 点网格选择，禁止看 held-out；
- candidate 的 `true_targets`、`recovered_misses` 分别严格超过 valid-Base 动态包络
  与 v23 control；
- mIoU/nIoU 分别不低于对应 valid Base 与 control；
- 相对 forced-G1 四项均不回退且至少一项严格提高，field/prediction ledger 非恒等；
- 使用原始整数和全精度浮点，不舍入后判门；
- **不采用固定绝对 \(+2\)、固定百分比等 uplift 幅度**；
- 安全可行域仍固定保留：`retention=1`、pixel FA `<=1e-4`、raw-background
  FA `<=1e-4`、FP components `<=100/Mpix`、`budget_violation=false`。

最后一组是安全上限，不是任意性能提升门槛，不能因“性能有提升”而取消。

---

# 27. 若 C4 或 r2 再失败

## 27.1 C4 在 sample 前失败

封存 C4 terminal。

不重试 C4；如需继续，建立 C5。

## 27.2 C4 在实际 sample 中出现 blocker

这才属于真实环境稳定性失败。

保存：

- sample index；
- inventory；
- blocker；
- target/systemd/GPU state；
- no-payload proof。

不得把它解释为模型失败。

## 27.3 r2 执行失败

一旦 scientific attempt commit 已创建，r2 即被消费。

本阶段必须封存 terminal 并停止，不得创建同候选 r3，不得 retry/resume。未来只有
用户建立全新的持久目标后，才能独立做 forensic 审查；它不能被解释为本目标续跑。

## 27.4 \(D_R\)-structural 科学 FAIL

冻结 v24。

不得修改模型后仍称为 r2/r3。

新模型必须新版本、重做 dataset-free、closure、\(D_R\)、OOF。

## 27.5 OOF 表明 gate 无增益

封存 OOF terminal/decision，冻结 v24 当前路线，禁止 bounded/Formal authorization，
进入新候选设计。

## 27.6 OOF target 提升但 IoU 回退

下一版本可优先检查 objective/shape regularization，但不武断限制只能修改这两项。
应根据 ledger 与消融证据修改真正导致回退的非主线或主线组件，并使用新候选身份
重新闭合 dataset-free、\(D_R\)、OOF。

## 27.7 bounded-400 FAIL/中断

封存 bounded terminal，禁止 Formal authorization。candidate/control 任一不满足
`10×40=400`、非有限、参数未改变、配对失真或 terminal PMOPE 高于自身初值均为
FAIL；不得 resume/retry，也不得事后新增性能 uplift 阈值。

## 27.8 Formal seed42/seed43 FAIL/中断

每个 seed 都是独立 create-once run。任一失败即封存该 seed terminal，并禁止当前
或后继 seed 的 retry/resume/partial checkpoint continuation；没有两份正式 receipt
及 pair verification，阶段不得宣称完成。

---

# 28. 明确禁止

当前不得：

- 重试 C3；
- 生成伪 C3 stability receipt；
- 改写 B3/R3/E3；
- 直接用当前 E4 草稿；
- 创建 B4 authorization；
- realize C4 unit 后绕过 E4；
- 直接创建 scientific runtime spec；
- 直接启动 r2；
- 修改 GCR-PACRE model；
- 修改 PMOPE；
- 修改 optimizer/schedule；
- 在 scientific r2 authorization 前访问 \(D_R/D_V/D_T\) payload；
- 在 r2/OOF/bounded/Formal 阶段访问除授权 \(D_R\) 外的任何 payload；
- 在本阶段任何时刻创建 \(D_V/D_T\) authorization 或访问其 payload；
- 将 C3 failure 解释为环境波动；
- 将 C3 failure 解释为性能失败；
- 在缺 R4/S4/L4/adapter4 时冻结 B4 source closure；
- 自动 retry C4。

---

# 29. 文件级修改清单

## 29.1 已完成测试、冻结并封存 production terminal

```text
tools/
cure_lite_v24_preaccess_compat_c3_environment_stability_failure_terminal.py

tests_v24/
test_cure_lite_v24_preaccess_compat_c3_environment_stability_failure_terminal.py
```

当前状态：17/17 定向测试 PASS；production terminal 已唯一封存、
CLI `validate-terminal` 与独立 `validate_archival` 均 PASS。C3 不得重试。

## 29.2 重写

```text
tools/
cure_lite_v24_preaccess_schema_compatibility_c4.py

tools/
cure_lite_v24_runtime_environment_preaccess_compat_c4.py
```

## 29.3 新增

```text
tools/
cure_lite_v24_actual_unit_realization_preaccess_compat_c4.py
cure_lite_v24_runtime_supervisor_preaccess_compat_c4.py
cure_lite_v24_actual_runtime_release_preaccess_compat_c4.py
run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c4.py
```

r14 不要求复制上述通用 harness；必须新增的是 disjoint r14 scenario/evidence 根、
authorization/receipt/terminal/removal-state 以及对应测试。只有现有通用工具无法表达
C4 绑定时，才允许新增窄 wrapper，并将理由与新增 source root 写入 B4 authorization。

## 29.4 保留模板并冻结

```text
deploy/systemd/
cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service.template
```

## 29.5 新 tests

```text
tests_v24/
test_cure_lite_v24_b4_bridge.py
test_cure_lite_v24_e4_scope_handoff.py
test_cure_lite_v24_e4_direct_stability.py
test_cure_lite_v24_e4_failure_terminal.py
test_cure_lite_v24_r4_unit_realizer.py
test_cure_lite_v24_s4_supervisor.py
test_cure_lite_v24_l4_release.py
test_cure_lite_v24_adapter4.py
test_cure_lite_v24_r14_integration.py
test_cure_lite_v24_c4_full_chain.py
```

还必须新增 OOF4、bounded400、Formal seed42、Formal seed43、pair verification 与
final-stage receipt 的 authorization/terminal/verifier 测试；所有阶段均需覆盖 kill、
失败、重复调用、resume/retry 和前驱 fingerprint 漂移。

---

# 30. 创建 B4 authorization 前的验收清单

```text
[ ] C3 terminal已生成并验签
[ ] C3 terminal状态为 VALID/SEALED，不使用“failure PASS”语义
[ ] C3 auth已过期
[ ] C3 unit仍 static/inactive/dead/NRestarts0
[ ] C3 stability不存在
[ ] C3 postcleanup不存在
[ ] C3 compatibility receipt不存在
[ ] C3 runtime spec不存在
[ ] scientific output不存在
[ ] C4 B4 docstring已改为C3 continuation
[ ] B4 source labels含C3 terminalizer
[ ] B4 evidence labels含C3 terminal
[ ] B4 authorize/verify/receipt均检查C3
[ ] E4不暴露高层 stability gate
[ ] E4不以C4 target重跑 prepare
[ ] E4 low-level sampling tests PASS
[ ] E4 high-level gate在facade/all-callables/AST中均不可达
[ ] E4 attempt commit在首次audit前create-once
[ ] E4 sample 1后及writer打开期间重验全部根
[ ] E4每次sample边界均验证C3 unit/terminal仍受保护
[ ] R4文件存在
[ ] S4文件存在
[ ] L4文件存在
[ ] adapter4文件存在
[ ] r14文件存在
[ ] C4 template SHA冻结
[ ] 全部 source SHA非占位符
[ ] 历史固定SHA逐文件重算，确认不存在十六进制盲替换
[ ] 全部 tests PASS
[ ] C4 evidence namespace empty
[ ] D_R/D_V/D_T payload未访问
```

---

# 31. 创建 scientific r2 authorization 前的验收清单

```text
[ ] B4 final receipt PASS
[ ] E4 stability PASS
[ ] E4 postcleanup PASS
[ ] R4 unit PASS
[ ] r14 integration PASS
[ ] C4 unit current shadow exact
[ ] S4 source closure exact
[ ] L4 source closure exact
[ ] adapter4 source closure exact
[ ] runtime spec create-once
[ ] scientific result root absent
[ ] attempt commit absent
[ ] materialization claim absent
[ ] GPU lease absent
[ ] no related process
[ ] no GPU compute process
[ ] authorization last-created
[ ] authorization D_R-only
[ ] no retry/no resume
[ ] D_R receipt/decision具有独立 verifier
[ ] OOF4 authorization、terminal、pooled decision schema已冻结
[ ] bounded400 authorization、terminal、paired verifier已冻结
[ ] Formal seed42/43各自一次性授权与terminal schema已冻结
[ ] pair verification及final-stage receipt schema已冻结
[ ] D_V/D_T authorization路径不存在且payload未访问
```

---

# 32. 预期完成状态

完成本方案后，合理状态应是：

```text
C3：
    sealed pre-sample compatibility failure
    no retry

C4：
    B4 PASS
    R4 PASS
    E4 stability PASS
    E4 postcleanup PASS
    r14 PASS

v24 scientific:
    model unchanged
    D_R structural verified PASS
    OOF-4 verified PASS
    bounded-400 verified PASS
    Formal seed42 = 800/800, 32,000 updates
    Formal seed43 = 800/800, 32,000 updates
    pair/final-stage receipt PASS
    D_V/D_T authorization absent, payload unread
```

其中 C4 完成只允许进入 scientific release 构造；只有完整 D_R→OOF→bounded
前驱链验签后才能创建 Formal authorization。

---

# 33. 诚实边界

该修改方案能确定解决的，是当前已知的逻辑错误：

> 旧 precleanup 被高层入口按新 target scope 再解释。

通过显式 C4 contract 直接采样，可以避免这一必然的 pre-sample failure。

不能提前保证：

- live C4 samples 无环境 blocker；
- \(D_R\)-structural 科学 gate PASS；
- OOF-4 PASS；
- bounded-400 PASS；
- Formal seed42/43 及 pair verification PASS。

\(D_V/D_T\) 不属于本阶段的待验证结果；它们在整个当前持久目标内保持禁止、
未授权和未访问。

---

# 34. 最终建议

当前最小且正确的修复是：

> **封存 C3；将 C4 从“高层入口重建合同”改为“旧根重放一次、显式 scope handoff、
> attempt commit 先行、对新合同直接采样、采样后完整重验全部根”。**

同时：

> **B4 必须从 C3 terminal 迁移，而不是继续从 C1/C2 迁移。**

在 C3 terminal、B4、E4、R4、S4、L4、adapter4 和 r14 全部完成并通过测试前，
继续保持：

```text
TRAINING START      BLOCKED
D_R                 UNREAD
OOF-4               NOT RUN
BOUNDED-400         NOT RUN
FORMAL seed42       0 / 800
FORMAL seed43       0 / 800
D_V                 UNAUTHORIZED / UNREAD
D_T                 UNAUTHORIZED / UNREAD
SCIENTIFIC ATTEMPT  UNCONSUMED
```

---

# 35. 主要源码索引

## 当前 C3/E3

- C3 environment wrapper  
  https://github.com/Arialliy/cure-lite/blob/main/tools/cure_lite_v24_runtime_environment_preaccess_compat_c3.py
- C3 schema bridge  
  https://github.com/Arialliy/cure-lite/blob/main/tools/cure_lite_v24_preaccess_schema_compatibility_c3.py
- C3 unit realization receipt  
  https://github.com/Arialliy/cure-lite/blob/main/protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/r2_preaccess_compat_c3_unit_realization_receipt.json
- C3 authorization  
  https://github.com/Arialliy/cure-lite/blob/main/protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/r2_preaccess_schema_compat_c3_authorization.json
- E3 policy  
  https://github.com/Arialliy/cure-lite/blob/main/protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/runtime_environment_policy_preaccess_compat_c3.json

## 当前 C4 草稿

- B4 draft  
  https://github.com/Arialliy/cure-lite/blob/main/tools/cure_lite_v24_preaccess_schema_compatibility_c4.py
- E4 draft  
  https://github.com/Arialliy/cure-lite/blob/main/tools/cure_lite_v24_runtime_environment_preaccess_compat_c4.py
- C4 template  
  https://github.com/Arialliy/cure-lite/blob/main/deploy/systemd/cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service.template

## Frozen environment core

- Runtime environment core  
  https://github.com/Arialliy/cure-lite/blob/main/tools/cure_lite_v24_runtime_environment.py
- `prepare_environment_stability_contract`  
  https://github.com/Arialliy/cure-lite/blob/main/tools/cure_lite_v24_runtime_environment.py#L3738
- `audit_environment_once`  
  https://github.com/Arialliy/cure-lite/blob/main/tools/cure_lite_v24_runtime_environment.py#L3074
- `evaluate_environment_stability`  
  https://github.com/Arialliy/cure-lite/blob/main/tools/cure_lite_v24_runtime_environment.py#L3877
- `run_environment_stability_gate`  
  https://github.com/Arialliy/cure-lite/blob/main/tools/cure_lite_v24_runtime_environment.py#L4325

这些 `main` 链接仅用于阅读，不构成 source closure。正式 authorization 必须记录
本地绝对路径、文件 SHA-256、inode/device/mode/owner/size 以及冻结 commit/工作树状态。
