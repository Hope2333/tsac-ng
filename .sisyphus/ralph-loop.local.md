---
active: true
iteration: 1
max_iterations: 500
completion_promise: "VERIFIED"
initial_completion_promise: "DONE"
verification_attempt_id: "22ff914b-f2d1-44a8-b145-f94485d2a3ce"
verification_session_id: "ses_19eda7d17ffeh0gBrfBjVdCCl7"
started_at: "2026-05-25T21:43:30.242Z"
session_id: "ses_1c6adebcaffeWBD8wd0UyZgibZ"
ultrawork: true
verification_pending: true
strategy: "continue"
message_count_at_start: 1697
---
=== 大致任务指导表单 ===
- 请自主做好实际任务规划，遵循与更新 .ai 状态，委派 Agents 可提及并在 .ai 中创建md文档/json或必要时创建其他标记文档、其他例式代码文件等。
- 委派 Agents 以任务驱动为主导并按实际情况选择是否接入 .ai。
- 我们还需要更深度的挖掘信息，分多轮的研究环节规划并启动心跳模式（必要时）。
- 我们将深度推理后多代理深度 Deep Agent。
- 委派Agent时请务必派发极为详细的提示词。
- 执行命令时建议设定永不超时。

=== 质量与审查 ===
- 需要使用 fuck-u-code 分析与持续优化代码质量。
- 需要使用 time-complexity 做算法分析和优化。
- 需要使用 CodeWrench 做性能分析和优化。
- 当代码和轮次进行到一个可以冻结并专注质量阶段时，启用专门的任务轮次来专注于优化代码。

=== 约合任务 ===
- 继续深度逆向、反汇编、GDB捕获、-v详细日志分析原版tsac（fast与非fast皆需要）。

=== 附加条件 ===
- 需要时可以把4.8(wed)MOGRA × #DSPM presents ぷらぷらうんじ.opus（长djs音频，实际内容09:28开始3:15:20结束，其他时段为无声内容，切片可自由发挥）切成不同段、不同时间长度的wav丢给原版tsac以获得更多日志数据样本和GDB信息捕获样本。
- 继续多轮推进，自拓展和规划Round，自行启动3个Round并做完它们
