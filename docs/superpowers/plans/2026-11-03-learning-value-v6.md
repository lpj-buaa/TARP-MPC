# Learning Value V6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task; one fresh whole-change review before freezing the formal reuse run.

**Goal:** 在固定数据、固定网络容量下检验实际预测学习收益驱动的采集及PPO增量。
**Architecture:** 新隔离包tarp_learning_value_v6。公开分割、同容量初始模型、现有输出层快速更新、实际误差奖励与虚拟学习增益、受限PPO、合并选样后的全参数模型、冻结独立评价。
**Tech Stack:** Python/NumPy/PyTorch/SciPy/Matplotlib，复用现有运行环境。
**Spec:** docs/superpowers/specs/2026-11-03-learning-value-v6-design.md

## Global Constraints

- 原数据、状态scaler、MPC/物理/容量/阈值/旧模型冻结；0新仿真。
- 神经网络136639/121794参数；训练查询池24、共同initial3、奖励/审计5，预算8/16/24。
- 10训练、4调参、3独立测试、2次要开发source；测试只在冻结后读取标签作评价。
- 负奖励保留；奖励真值不进入采集器；现有标签能力边界准确表述。

## Review Focus

1. 初始模型是否已看过查询答案；后果模型训练ID是否混入奖励/测试。
2. 函数样本在候选与参考上是否联合；预计价值是否读取未查答案。
3. 期望增益、PPO实际执行分布与更新时logp是否完全同定义。
4. 不同方法是否共享快速/全参数学习器；预算24同集合是否只训练一次。
5. 混合是否调参触发并在测试前冻结，Resume是否复用正确版本。

### Task 1: 数据曝光与任务误差

Files: src/tarp_learning_value_v6/data.py, core.py; tests/test_learning_value_v6.py。
- [ ] RED公开划分标签不变性、初始/查询/审计严格互斥、任务误差及参数测试。
- [ ] GREEN实现固定hash分割、角色、尺度与精确任务误差。
- [ ] 验证新测试，记录ledger。

### Task 2: 后果学习与可观测信息价值

Files: learner.py, acquisition.py。
- [ ] RED正则输出更新对照线性求解、顺序不变、未查答案不影响评分、负奖励望远镜。
- [ ] GREEN实现同容量先验、公开函数抽样、虚拟更新及规则适配。
- [ ] 验证零残差贪心与预算；完整接口测试。

### Task 3: PPO、选样与全参数样本价值

Files: policy.py, training.py, pipeline.py; scripts/run_learning_value_v6.ps1。
- [ ] RED行为/更新logp一致、训练ID无审计/测试、24预算指纹、混合触发。
- [ ] GREEN完成相同预算策略与学生训练、阶段seal/Resume/独立测试门。
- [ ] 运行smoke及全套测试；新上下文整体review并完成重要修复。

### Task 4: 冻结运行与科研交付

Files: analysis.py, report.py, docs/learning_value_2026-11-03/。
- [ ] 新RunId lv_reuse_s42_v1完整三种子串行GPU，先全部选择和训练冻结后独立评价。
- [ ] 中文source表、覆盖/误差组合图、RL/规则增量与触发结论，HTML/命令行。
- [ ] 全套测试、Resume0训练、原现场hash/暂停核验，提交源与报告并按已有授权同步发布镜像。
