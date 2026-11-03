# V5 信息对齐实施计划

Spec: `docs/superpowers/specs/2026-11-03-value-alignment-v5-design.md`。用户已授权直接实施与复用验证，逐任务不再确认。

## Task 1: 可验证的教师和观测修正

新增 `src/tarp_alignment_v5/core.py`，测试 `tests/test_alignment_v5.py`。先运行缺失接口测试得到RED，再实现：固定距离残差修正、可微分残差、精确边际教师、加权成对损失。测试含未知mask、零观测、目标排除、未查真值不变性及参数不增。Expected: 新测试及原项目套件通过。

## Task 2: 同容量训练与消融执行器

新增 `training.py/policy.py/pipeline.py` 与 `scripts/run_alignment_v5.ps1`。先测试隔离数据角色/原PPO零系数更新等价/特征不含教师，再实现3个条件化后果模型与5×3策略。绑定源哈希和代码，逐阶段封存、Resume验证；修正与静态策略使用同样种子、情境/预算抽样、120批×8episode×4更新。Expected: smoke执行及Resume无新增fit，原暂停采集不变。

## Task 3: 复用评价与归因报告

新增 `analysis.py/report.py`。复用原PPO及基线评价，评价5新方法预算8/16/24；公共历史诊断只统计未查动作。输出source/seed原始行与主2×2效应、固定规则对照，中文表及多面板SVG/PNG/HTML。Expected: 3个模型、15个策略全部封存；源角色相同、原正式confirm使用0、旧训练0、仿真0。

## Task 4: 核验、独立审查与交付

运行全套测试与新run完整Resume，核对原源哈希与暂停状态；新上下文审查整个增量一次，处理重要问题并回归。提交源与报告。依已有授权，在独立发布镜像先fetch，保留用户新提交后同步新代码和轻量报告，不修改原实验工作树。Expected: 中文结果明确增益来自排序监督、条件化学习还是非参数修正。
