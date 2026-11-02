# 序贯信息采集V4：执行与验证

用户已授权完整执行，沿用隔离worktree `D:\博士资料包\科研\科研\AAAI_RL\tarp_37_information_acquisition`，分支codex/intervention-information。新包src/tarp_sequence；旧MPC、物理源码、RL和预警器不修改。

## 完整流程

正式银行：28 source×2物理时间点×2偏好=112情境，每情境32条分支，共3584条。原162条V5闭环数据及MPC测量复用；其余3422条执行真实V2SIM＋AC＋MPC接续，每条6小时，再读取+1/+3/+5小时的支持区间及恢复LP。原生小规模另4条、运行资源等价性核查另1条属于工程验证，不混入正式采集预算。线程试验未支持替换当前运行方式，正式保持4个CPU worker及原数值配置。

物理阶段RunId：sequence_formal_s42_v1。正式训练/评价随后自动串行执行，GPU只在学习阶段使用。角色保持原有train6、meta_train6、tune2、meta_tune2、cal4、confirm8个source；cal/confirm不训练或选模型。最终效果只按confirm，缺乏完整confirm支持时明确输出unsupported，绝不以tune/cal替代。

## 已完成的工程验证

sequence_smoke_s42_v1：4条真实分支、24/24小时AC核验、12/12组MPC测量有效；1个后果模型＋3个策略及8方法比较、报告和Resume跑通。该工程小样本会查询完全部可选项，不作为效能证据。

sequence_cache_pilot_s42_v1：旧162条、6个历史source的开发复用。3个后果模型及9个新策略拟合完成；4 source训练、1调参、1仅作开发检查。输入继承前轮公开初始状态归一化，不能当成新独立confirm。所有策略预算8/16/24；最后一个预算穷尽候选池。该pilot验证多步更新机制和比较接口，不用于决定正式模型参数。

实际多步回报、集合顺序不变、未查询真值隔离、绝对剩余预算与已核验边界标记均有测试。即时PPO gamma0与序贯PPO gamma1共用架构、初始权重和episode预算日程；组相对策略在同情境/初始经验/预算下比较4条完整轨迹。

## 科学比较与基线

八种方法：序贯PPO、即时PPO、组相对序列、固定后果模型即时贪心、局部插值、括区＋Random、参数Sobol、Random。贪心/局部插值都以自身预测的后果计算与RL相同的即时终局效用增量，不采用旧information_reward，也不加随机选择。局部方法插值资源量和LP超出必要能量边界的残差，避免把二元端点标签线性插值出伪50%物理门槛。

训练奖励为固定池画像覆盖改善与相邻恢复边界定位的增量；gamma1的累计增量严格等于终局改善。完整银行真值只用于训练奖励或事后评价，不进入actor特征。held-out执行只终局评分。固定后果预测器episode内不更新。cal检查正负支持、分项MAE、恢复分类与Brier/ECE，不做新校准拟合。边界无支持的情境不进入边界率分母；所有指标先聚合同source的时间/偏好/种子，再source bootstrap。

## 当前运行与恢复命令

当前采集已启动，无需重复启动。完成后续查由runner自动接续。运行文件：

- `artifacts/sequence_v4/sequence_formal_s42_v1/physical_protocol.json`：输入/物理代码hash。
- `inventory.json`及`contexts/*/progress.json`：逐情境进度与封存分支。
- `collector_runtime.json`：当前原生采集进程身份。
- `workflow_runtime.json`、`workflow_full.out.log/err.log`：后续串行runner。
- `historical_reuse_verification.json`：旧snapshot/代码/原始与测量hash兼容性核查。
- `workflow_status.json`：完整流程成功或明确失败。

已有物理采集运行时，跟随而不另起采集：

```powershell
Set-Location -LiteralPath 'D:\博士资料包\科研\科研\AAAI_RL\tarp_37_information_acquisition'
powershell -ExecutionPolicy Bypass `
  -File .\scripts\run_sequence_v4.ps1 `
  -Mode formal -RunId sequence_formal_s42_v1 `
  -Stage finish -Device cuda -Resume
```

仅在原采集进程和串行runner都已经退出、需要恢复时：

```powershell
powershell -ExecutionPolicy Bypass `
  -File .\scripts\run_sequence_v4.ps1 `
  -Mode formal -RunId sequence_formal_s42_v1 `
  -Stage all -Device cuda -Workers 4 -Resume
```

脚本会保留已封存分支、模型和旧现场；只为未完情境归档旧simulation/log并从原快照恢复。独占锁阻止重复runner。原容量/阈值/标签没有可调参数被改变。

只运行已完成银行的学习/核验：

```powershell
powershell -ExecutionPolicy Bypass `
  -File .\scripts\run_sequence_v4.ps1 `
  -Mode formal -RunId sequence_formal_s42_v1 `
  -Stage learn -Device cuda -Resume
```

该命令的Resume会逐项校验数据、预测和策略checkpoint；已完成时新增拟合为0。

## 产物与审阅

每个正式run输出dataset.json、response_s*/、policies/*/、evaluation.json、response_diagnostics.json、analysis.json及report/index.html和组合图PNG/SVG。缓存开发报告在artifacts/sequence_v4/sequence_cache_pilot_s42_v1/report/index.html。

独立review的四项Important已针对修复：即时对照同目标；历史物理复用绑定snapshot与代码；缺confirm不能替换角色；补分项后果与cal支持检查。相关RED→GREEN与Resume记录在run日志、测试及版本修订文件。原cache模型数值训练代码未改，基线/终局评价与依赖核验修复后只重算评价。正式物理collector代码运行中保持不动。
