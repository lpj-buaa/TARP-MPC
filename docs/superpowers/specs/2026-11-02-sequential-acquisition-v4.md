# 后果模型辅助的序贯采集 V4

用户已批准架构并要求直接完成编码、测试、扩展仿真、训练、评价全流程。基准8feb466；沿用已隔离linked worktree，新包src/tarp_sequence、新artifact根artifacts/sequence_v4。tarp37/tarp_info/tarp_boundary/tarp_acq_evidence及旧RL、MPC、预警器只读。

## 数据与物理
28 source×pool32/40×profile0/1=112情境、56物理时间点状态。每情境32分支：共同初始参考及throttle/release端点3条，原两族j/13的24条剂量动作，以及5条曲线外控制块/空间重分配动作。原162条V5分支及486测量按hash复用；其余执行一次干预＋6小时V5常规有界Q/MPC接续，保持所有容量、阈值和保护。每情境复用一个环境实例、每分支恢复原完整快照。测量沿用原+1/+3/+5h MPC支持区间与恢复LP，不以能量proxy替代新LP真值。未知完整保留，不改判据或制造标签。

role沿原source：train/meta_train共12用于训练；tune/meta_tune共4用于模型选择；cal4用于预先检查部署支撑，不再调模型；confirm8仅冻结后评价。统计按source聚合，不把多个时间/偏好作为独立source。

## 模型
独立训练稳定60维后果预测器（128×128）与三时刻恢复分类头，训练数据来自train角色。输入为原49状态＋774空间量＋51动作；scaler仅训练source。连续目标沿前轮；补成对后果差分与恢复标签监督，记录组件误差。episode内冻结预测器。

集合记忆采集策略以完整已查(action,真实response)集合、剩余动作及预算为状态，候选动作token读取预测后果、当前context及对经验集合的attention。输出候选分布，value预测剩余序列收益；actor/critic为小型共享集合模型。查询一步=揭示一条6h分支，并更新集合/预算，非法重复动作masked。

三方法同架构：PPO序列gamma1、即时PPO gamma0、GRPO式整段组相对策略（同情境/相同D0/预算，不训练value）。主实现PPO，组相对为替代对照。比较Random、参数Sobol、局部物理插值（曲线外近邻）、括区＋Random、冻结后果模型的即时贪心读出。种子42/43/44，部署预算8/16/24，初始3条免费且共同；候选29条。

## 回报/信息隔离
训练episode使用模拟器内部专用真值评价U(D)：固定候选池描述距离的降低＋已定位相邻恢复边界比例；各部分固定归一化至初始可改善范围，等权，费用按分支数。r=U(D_next)-U(D)，gamma1终局和与U(D_B)-U(D0)严格相等。完整真值不进入策略输入；非训练role的回报只在episode结束后评价，不参与更新。未知查询消耗预算，不冒充风险。训练银行完整预计算的物理成本与策略查询预算分别报告。

## 训练规格
预测器最多100epoch，patience12，Adam3e-4，batch128；损失连续组MSE＋0.25成对差分MSE＋0.2已核验恢复BCE。训练目标按有效mask，稀少恢复边界采用训练正负权重（上限20），确认集不重采样。
策略每seed/方法120批（smoke3批），每批8episode、随机预算8/16/24，PPO4轮、lr3e-4、clip0.2、entropy0.01、GAE0.95。即时版gamma0、序列gamma1，使用相同查询轨迹数量/架构/初始化。GRPO每组4条相同情境episode，用终局U组相对优势、PPO式clip，训练episode总数匹配。每20批仅tune角色评价预算16选择checkpoint。训练共享单GPU串行，物理最多4CPU worker。第一次small smoke必须真实验证后启动正式采集，不启动重复任务。

## 验收
TDD：上下文/自标签泄漏、重复query、预算、reward望远镜和、真实跨步return、集合顺序不变、下一动作随反馈可改变、gamma0/1差别、source角色/scaler隔离、未知mask、物理缓存hash、Resume0拟合。独立whole-branch review。交付独立confirm的source表/CI、预算曲线、方法组件结果、中文HTML、可复现命令及冻结审计。多步收益是待检验命题，不能由实现替代证据。
