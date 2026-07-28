# PACRE-VC v23 D_V 独立验签器 schema 纠错登记

登记日期：2026-07-28  
性质：D_V 已完成并原子发布后的机械验签器纠错；不是新实验、不是 D_V 重跑、不是门槛修改。

## 1. 触发事实

唯一 D_V runner 在四个终态成员原子发布后调用冻结的独立 verifier，并以如下异常退出：

```text
RuntimeError: persisted model/D_V binding changed
```

runner 退出前已经完成并发布唯一 D_V 结果。禁止删除、覆盖、续跑或重跑该 D_V；禁止访问 D_T。

## 2. 已冻结输入

| 输入 | SHA256 |
|---|---|
| 原始独立 verifier | `bb85c2746589ce60f5b9d59c834cba00a4ac39580084bfee7f6813b87d442ce6` |
| `claim.json` | `49782cfa6d4b4933733c476d89c3d827356ae573dbf940905a5fe2d19cb50cbf` |
| `receipt.json` | `89527eb76eeded00a9fcf8a8cc96fc69fd8970f30ea53b13d7f11f43101fb1ad` |
| `decision.json` | `8bb002b2b73c974e9bce0b03931625cc49e284d49afda9a8d75feb4aa9aa4b81` |
| `COMPLETE.json` | `2a9b648933dd1d9635943f2a8debd255ac91635722efe17f75b85438ee43a86b` |

原始源码闭包指纹保持为：

```text
d08a1d84348d8caf8ecee3b0fef3d5efcd56e05e50f46e25b1cf17bd71dfe48c
```

## 3. 唯一允许的纠错

生产端 canonical `model_binding.formal_artifact` 固定包含七个字段：

1. `artifact_fingerprint`
2. `formal_result_fingerprint`
3. `training_result_fingerprint`
4. `authorization_fingerprint`
5. `source_closure_fingerprint`
6. `model_state_fingerprint`
7. `model_config_fingerprint`

原始 verifier 的 exact-key 白名单只列出其中六个字段，遗漏
`formal_result_fingerprint`，所以所有真实 canonical binding 都会确定性地被拒绝。

纠错器只允许做以下两件事：

1. 要求该第七字段存在且等于 Formal terminal 的
   `formal_result_fingerprint`；
2. 临时移除该字段，将其余未改动 payload 交给冻结原始函数完成原有全部检查，
   然后对完整七字段 payload 重算 binding fingerprint。

除此之外，不修改原 verifier 的任何检查、常量、性能门禁、51 点 Base@B
重选逻辑或输出关联规则。

## 4. 验证规则

- 更正验签器运行前后必须逐字节复核上述五个输入文件；
- 必须复核原始 verifier SHA256 与源码闭包；
- 不重新运行模型推理；
- 不打开 D_V tensor payload；
- 不访问 D_T；
- 不训练、不更新模型、不选择 checkpoint；
- 完整调用原 verifier 的 terminal graph、独立 51 点重选和 gate 重算；
- 结果无论 PASS 或 FAIL 都必须如实保留；
- 更正验签收据只能 create-once，禁止覆盖。

## 5. 解释边界

该纠错不会把科学 FAIL 改为 PASS，也不会改变“严格正提升即可、无任意固定
`+2` 门槛”的已冻结判据。它只修复独立 verifier 对生产端 canonical schema
的一个字段白名单遗漏。原始 verifier 文件和原始 D_V 产物均保持不变。
