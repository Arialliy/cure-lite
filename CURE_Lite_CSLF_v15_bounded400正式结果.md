# CURE-Lite CSLF-v15：\(D_R\)-only bounded-400 正式结果

> 运行日期：2026-07-27  
> 数据范围：仅 \(D_R\)  
> 设备：GPU0（温控阈值 82°C 暂停、75°C 恢复；本次未触发暂停）  
> 运行状态：完整结束并封存  
> 正式决定：`BOUNDED_CSLF_GATE_FAIL`  
> Formal-800：未授权

## 1. 结果边界

本次运行是 seed 42、每个 objective 400 次更新的有界模型检验。它回答的是
CSLF 是否已经具备进入 Formal-800 的最低学习能力，不是完整数据集性能实验。

运行期间：

- 没有读取新的 \(D_V\) 或 \(D_T\)；
- 没有搜索阈值，输出零水平固定为 \(\phi=0\)；
- 没有校准；
- 没有自动重跑；
- 没有修改 Base、decoder 接口、训练步数或门槛。

正式产物：

```text
/home/md0/ly/cure_lite/runs/irstd1k_stage_a_seed42/
    cure_lite_cslf_v15_bounded_400_r1
```

完成指纹：

```text
faaa2395623f5edfa0e56ab849d20305b73df1e7b3446b22b834279a2637d14b
```

`COMPLETE.json` 存在，`.incomplete` 和 `FAILURE.json` 均不存在。完成清单中的
17 个文件与目录内 17 个文件一致，逐文件 SHA256 全部匹配。因此这是一次完整
的有界阴性结果，不是执行中断。

## 2. 公平训练条件

三个 objective 均使用：

- 相同初始模型；
- 相同模型结构与参数量；
- 相同 optimizer；
- 相同确定性 schedule；
- 相同 natural branches 与 pair endpoints；
- 相同设备常驻 cache；
- 相同更新数和 logical-state 数量。

| objective | updates | forward/backward/step | logical states | total loss：首轮→末轮 | pair loss：首轮→末轮 |
|---|---:|---:|---:|---:|---:|
| response-joint | 400 | 400/400/400 | 4,800 | 0.9320→0.3320 | 0.3144→0.2097 |
| identity-joint | 400 | 400/400/400 | 4,800 | 0.9120→0.2999 | 0.3029→0.1723 |
| separable-endpoint | 400 | 400/400/400 | 4,800 | 0.7819→0.2222 | 0.2060→0.0850 |

三个 objective 的目标函数不同，末轮 loss 不能直接用于方法排序；这里的下降只
证明三条训练路径都实际执行且数值正常。

## 3. 固定零水平门禁

| objective | 总门禁 | clean compact | clean defined | factual miss | factual no-miss | component null | identity null |
|---|---|---|---|---|---|---|---|
| response-joint | FAIL | FAIL | FAIL | FAIL | PASS | PASS | PASS |
| identity-joint | FAIL | FAIL | FAIL | FAIL | PASS | PASS | PASS |
| separable-endpoint | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | PASS |

因此三个 objective 均未达到预先固定的完整门槛，Formal-800 不获授权。

## 4. Natural-state 结果

| objective | factual-miss 门禁 | 至少找回一点 | 目标负像素 | 连通组件命中 | factual-no-miss |
|---|---:|---:|---:|---:|---:|
| response-joint | 15/16 | 16/16 | 321/335 | 19/20 | 16/16 |
| identity-joint | 13/16 | 16/16 | 294/335 | 19/20 | 16/16 |
| separable-endpoint | 15/16 | 16/16 | 308/335 | 19/20 | 15/16 |

“至少找回一点”不是正式通过标准。冻结门槛还要求每个 factual-miss focus target
至少 95% 的像素位于负场中且没有无效区域输出。

response-joint 唯一未通过的 factual-miss 样本是 `XDU680`：

```text
focus target negative pixels = 63 / 75 = 84%
```

## 5. Pair-state 结果

### 5.1 Clean-positive

| objective | 响应方向正确 | added-target 负像素 | compact component match | compact exact | 新 completion | target 外 completion |
|---|---:|---:|---:|---:|---:|---:|
| response-joint | 1396/1396 | 6/149 | 3/16 | 0/16 | 14 | 8 |
| identity-joint | 1395/1396 | 4/149 | 2/16 | 0/16 | 4 | 0 |
| separable-endpoint | 1396/1396 | 137/149 | 15/16 | 9/16 | 138 | 2 |

### 5.2 Null states

| objective | component-null | identity-null |
|---|---:|---:|
| response-joint | 17/17 | 16/16 |
| identity-joint | 17/17 | 16/16 |
| separable-endpoint | 15/17 | 16/16 |

## 6. 结构性判断

response-joint 并非完全没有学习：

1. clean-positive 的场响应方向达到 1396/1396；
2. factual-miss 比 identity-joint 从 13/16 提升到 15/16；
3. factual target 负像素从 294/335 提升到 321/335；
4. factual-no-miss、component-null 和 identity-null 全部保持。

但相对响应没有稳定转化为绝对零水平穿越：

\[
\Delta\phi_{\text{direction}}\ \text{正确}
\quad\not\Rightarrow\quad
\phi_{\text{minus}}<0.
\]

clean-positive 中只有 6/149 个 added-target 像素进入负场，且 16 对中没有一对
得到精确紧致支持。与此同时，separable-endpoint 可以把 137/149 个像素推过
零水平，却在部分 factual-no-miss 和 component-null 上产生了不应出现的
completion。

因此当前真正未闭合的是同一个单场中的三者统一：

\[
\text{coverage response}
\;+\;
\text{absolute zero-level anchoring}
\;+\;
\text{null-state preservation}.
\]

这是模型对象内部的结构性权衡，不是数据加载、GPU、训练未发生或结果缺失。

## 7. 正式决定与下一步

当前可固化为：

```text
bounded_execution_complete = true
bounded_gate_passed = false
formal_800_authorized = false
D_V_accessed = false
D_T_accessed = false
automatic_retry = false
mainline_changed = false
```

下一步应冻结本次结果，针对“相对响应正确、绝对零水平锚定不足”形成一个最小
模型修订。不得：

- 放宽 95% 或 compact-support 门槛；
- 只增加训练步数重跑同一版本；
- 用 separable-endpoint 的较强 clean-positive 表现掩盖其 null-state 失败；
- 直接进入 Formal-800；
- 开始 Full CURE、跨 backbone 或三数据集验证。

修订仍须保持一个 coverage-state residual field 主线，并重新经过 dataset-free
和单次 \(D_R\)-only bounded-400。只有新版本通过全部冻结角色门禁后，才允许
进入 Formal-800。
