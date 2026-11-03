# TARP-MPC

干预下持续受控的车网耦合系统：学习当前动作如何改变未来可用资源、调度响应与恢复边界，并在有限查询预算下采集有信息价值的行动后果。

本仓库包括完整研究源码、配置、脚本、测试、规格与执行计划。大文件及完成阶段结果位于 [Releases](https://github.com/lpj-buaa/TARP-MPC/releases/tag/research-snapshot-2026-11-03)。

- [新电脑恢复、环境安装与完整命令行](docs/GITHUB_REPRODUCTION_CN.md)
- [序贯采集V4执行文档](docs/SEQUENCE_V4_EXECUTION.md)
- [V4模型和MDP设计规格](docs/superpowers/specs/2026-11-02-sequential-acquisition-v4.md)
- [已有信息价值排序与后果学习结果](docs/acquisition_evidence_2026-11-02/report.md)
- [代码和运行版本记录](publication/snapshot.json)
- [原项目阶段说明](docs/ORIGINAL_PROJECT_README.md)

## 当前实验

正式实验 sequence_formal_s42_v1：28个source、112情境、3584分支，162条历史分支复用，其余3422条为新原生六小时闭环。MPC、容量、阈值、旧RL及风险预警器冻结。采集结束自动进行三种子学习和独立confirm评价。本次上传时仍在采集，正式效果尚未给出。

稳定后果模型学习未来资源支持区间、恢复工作量、恢复缺口、后续调度和偏好变化；集合策略以已查询的行动与定量后果作为经验状态。比较序贯PPO、即时PPO、组相对序列策略、后果模型即时贪心、局部插值、括区＋Random、参数Sobol和Random，查询预算8/16/24。

开发缓存pilot已跑通并完成Resume核验；其学习策略优于Random，尚未超过强插值/括区对照。开发结果不替代独立confirm。

## 目录

```text
src/tarp37/           冻结物理环境、有限信息MPC及风险边界
src/tarp_info/        可见状态、资源后果画像与原生分支
src/tarp_boundary/    后续调度支持区间与恢复LP
src/tarp_acq_evidence/空间状态、定量经验及稳定后果学习
src/tarp_sequence/    知识状态MDP、后果模型、序贯策略与等预算评价
scripts/             完整流程、核验、诊断、报告和Release数据恢复
tests/               接口、监督、预算、泄漏与物理机制测试
configs/             既有实验配置
docs/                科学问题、规格、计划、结果及执行说明
requirements/        物理与学习环境依赖
publication/         代码/输入/运行快照与归档校验
```

复现数据按SHA256校验，源码和轻量文档进入Git，状态快照与checkpoint进入Release。两个电脑编辑时先拉取再创建工作分支，避免修改另一台正在运行且被hash冻结的版本。
