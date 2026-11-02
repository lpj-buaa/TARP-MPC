# Intervention Information Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task. User authorized inline implementation and first acquisition comparisons; no further stage approvals.

**Goal:** 比较信息采集，而不训练风险预警器。
**Architecture:** 独立tarp_info包处理有限缓存Oracle、机制描述、预算采集和统计；现有tarp37物理模块用于V5小验证，保持源码冻结。
**Tech Stack:** Python, NumPy, PyTorch CPU, SciPy, pandapower, matplotlib, PowerShell。
**Spec:** docs/superpowers/specs/2026-11-02-intervention-information-design.md。

## Global Constraints
- 0.9kA，0.93–1.07pu，原生机组与充电站能力不改；旧银行及RL不变。
- 初始经验和新增查询单列；未来数据只由Oracle.query揭示；seed42/43/44，四臂同预算。
- 保存状态初始化的真实V5演化与V4缓存比较分开呈现。

## Review Focus
- 选择器接触尚未查询的未来数据；通过隐藏标签测试约束。
- 早终止被补成完整资源画像；缺失必须返回unknown。
- profile混配或自己作为最近邻；完整情境分组及自身排除。
- 归一化随已知测试画像变化；只用物理标尺。
- 存在缓存时静默改配置；RunId绑定内容hash。

## Task 1 — Descriptor and access contract
- [x] tests/test_info_acquisition.py先验证能量方向、未知、上下文隔离、重复动作、Oracle访问。
- [x] src/tarp_info/profiles.py：describe(record)->array|None，distance(a,b)->float。
- [x] src/tarp_info/cache.py：prepare source corpus，PublicCandidate不含未来；Oracle.query(id)记录访问。
- [x] 验证RED→GREEN并提交。

## Task 2 — Acquisition and matched replay
- [x] tests/test_info_acquisition.py增加预算、均衡轮转、奖励重算和未揭示标签不影响选择的测试。
- [x] src/tarp_info/acquire.py：run_arm(public,initial,oracle,method,budget,seed)，共享TD3式双critic实现；不训练风险模型。
- [x] scripts/prepare_info_utility.py使用冻结V4模型只前向计算Oracle私有误差效用，并封存checkpoint/scaler/schema/实现hash。
- [x] src/tarp_info/cli.py：prepare/replay/stats，RunId/Resume与输入hash校验。
- [x] 先8查询smoke，再128查询×4臂×3seed；记录所有访问及覆盖曲线。

## Task 3 — V5 physical sanity
- [x] 新调压扩展类多初始化测试，原regulation.py不改。
- [x] src/tarp_info/native.py（python -m tarp_info.native）从两个train快照执行8分支，记录未来view与正常Q、处置及硬件hash。
- [x] 数值与反送未知不改正负标签，前后物理平衡与资源边界核验。

## Task 4 — Synthesis
- [x] scripts/analyze_information.py与中文Markdown/HTML：预算覆盖曲线、能力—响应图、发现类型统计；缓存与真实V5分列。
- [x] 完整单测、原物理源码hash、source级统计、命令行与独立review。
- [x] 保留分支与RunId，报告真实增益及下一轮是否值得扩大。

## 完成核验与审阅修正

最终采集运行information_pilot_v2，四臂×3seed×128查询全部完成，Resume逐臂复用；v1保留为Sobol接口修正前现场。独立审阅发现并修正Sobol坐标/情境序列与真实AcquisitionTD3依赖hash两项问题，新增回归测试。原tarp37源码与V5一致，原V4/V5工作树无修改。

V5真实接续8分支48小时完成；另复用其view执行24个只读MPC方向支持区间。该补充衡量业务目标预算内的后续调度区间，不纳入本轮奖励，也不改物理执行。独立审阅要求所有必要端点均核验成功才返回known，已增加数值失败回归测试并重算24状态。最终12项新测试和5项既有测试共17通过。

最终图表与中文结果见docs/information_acquisition_2026-11-02/index.html，数据记录两类目标的取舍，不宣称机制采集已胜过random/Sobol。预警器、专家、旧RL训练调用均为0。
