# CURE-Lite GCDE-v13 R13-1 输入识别正式结果

> 日期：2026-07-27  
> 输入：冻结的 NLCC-v12 Development population  
> 当前决定：`R13_1_ROLE_QUOTIENT_FAIL`  
> GCDE 实现授权：`false`  
> GCDE 训练授权：`false`

## 1. 本轮执行范围

本轮先修订
`CURE_Lite_NLCC_v12负结果后的GCDE_v13修改方案.md`，随后仅执行修订方案允许的
R13-1 输入识别检查。

没有：

- 修改 NLCC-v12 decoder、PECO-v10 或冻结输入 builder；
- 实现 GCDE loss；
- 创建 Development authorization；
- 运行 GCDE、PECO 或其他模型训练；
- 读取新的 Holdout、真实 \(D_R\) 或三数据集结果。

## 2. 修订后的研究关系

主线保持为：

```text
任意冻结 IRSTD detector
    -> 通用输入 (F_b, O)
    -> feature–coverage relevance
    -> 受约束 evidence release
    -> 补回应恢复目标并保持背景静默
```

GCDE 只用于检查“当前 NLCC 实例是否被均值型、plus-only 训练目标限制”，不是
CURE 的核心模型创新。

## 3. 新增实现

```text
cure_lite/nlcc_role_quotient_audit.py
tools/audit_nlcc_role_quotient.py
tests_v13/test_nlcc_role_quotient_audit.py
protocols/IRSTD-1K/gate_covered_dual_endpoint_v13/
    r13_0_v12_freeze_baseline_receipt.json
    input_identifiability_receipt.json
```

检查键不使用 group ID、sample ID、match ID、anchor role 或 endpoint role。

- hard key：完整 feature、NLCC 实际消费的完整 \(3\times3\) local-count field、
  输出坐标；
- role key：去除随机 feature 幅度和符号身份，保留通道支持、相对完整 count
  field 与 PixelShuffle phase；
- D4：只作联合变换诊断，不影响正式决定，本轮未执行。

## 4. 正式结果

| 项目 | 结果 |
|---|---:|
| states | 96 |
| supervised records | 75,192 |
| positive records | 112 |
| negative records | 75,080 |
| exact effective-input conflict keys | 0 |
| signed quotient conflict keys | 8 |
| unsigned role-quotient conflict keys | 7 |
| records in unsigned conflicts | 64 |
| hard gate | PASS |
| role gate | FAIL |
| Development authorized | false |

输入 fingerprint：

```text
4f387e3e513a93a1cee58ee68d9d67eb5b2746688da42ec40572ae6fc1df55a7
```

receipt fingerprint：

```text
58eb73948da441891849353b41aa7493b9485500f73b120fbb128be33a454728
```

receipt 文件 SHA256：

```text
c81fdcd77c9f7157b2ae965359e228579ee7b51c9e9826ef94b97979a9791571
```

第二次完整构建与封存 receipt 逐字节一致。

## 5. 结果解释

`exact conflicts = 0` 表示当前实际 feature 数值可以在字节层面区分监督状态，
不存在已经证明的硬不可识别矛盾。

`role quotient conflicts = 7` 表示去除随机幅度、符号和绝对位置身份后，仍有
7 个等价输入角色同时要求相反输出。也就是说，当前 population 若被模型拟合，
可能依赖人为 hash 身份，而不是学习能够迁移到任意 detector 的
feature–coverage 机制。

因此：

```text
GCDE 在该输入上即使通过，也不足以支持 CURE 机制成立；
只修改 objective 不能修复输入角色语义；
R13-1 必须判为 FAIL；
GCDE Development 不得启动。
```

这不是 GCDE 训练负结果，因为训练从未发生；也不是 CURE 总方向失败。它否定的是：

> 在冻结 v12 dataset-free population 上，直接开展 objective-only GCDE
> Development 能产生可信机制结论。

## 6. 测试状态

定向 v13：

```text
7 passed in 15.56s
```

隔离测试矩阵：

| 测试范围 | 结果 |
|---|---:|
| `tests` | 1,105 passed |
| `tests_v8` | 90 passed |
| `tests_v9` | 43 passed |
| `tests_v10` | 36 passed |
| `tests_v11` | 93 passed |
| `tests_v12`，每个文件独立 Python 进程 | 117 passed |
| `tests_v13` | 7 passed |
| **合计** | **1,491 passed，0 failed/error/skipped** |

`tests_v12` 整目录放在同一 pytest 进程时，收集 integration 文件会先导入
`cure_lite.experiment`，随后 runner 的进程隔离检查按设计拒绝继续。其 10 个测试
文件分别在全新进程中运行时全部通过，因此正式测试口径使用逐文件隔离矩阵，不
放宽 runner 检查。

完整测试 receipt：

```text
protocols/IRSTD-1K/gate_covered_dual_endpoint_v13/
    r13_1_isolated_test_matrix_receipt.json
```

## 7. v12 冻结产物核对

| 文件 | SHA256 |
|---|---|
| `result.json` | `665325f774844ebd2026a9448903d3142653bac1f66729748b9cb5699ab6ae9e` |
| `decision.json` | `c539b5d5d4b5a5cf81f5a4e3865039e252d002aa8afe3005859adb10f73144c2` |
| `COMPLETE.json` | `dfc0219b85366774be97bf2eb5501f4a1cebe1949d7762f8643a5fa971c4807c` |

上述值与本轮修改前一致。

## 8. 下一步

下一版本不能给输入增加只在训练时可用的 group、endpoint、配对或 GT 谱系标记。
应继续此前 CURE 主线，在通用 \((F_b,O)\) 接口内重新构造可识别状态：

```text
冻结 R13-1 负结果
    -> 对 7 个 quotient conflict 做来源归因
    -> 设计 detector-accessible feature–coverage relevance state
    -> 新 input contract 独立版本
    -> 重新运行输入识别检查
    -> 只有通过后才讨论训练代理
```

不能同时修改 normalization、occupancy basis、affinity 和状态方程后将结果归因
于单一机制。
