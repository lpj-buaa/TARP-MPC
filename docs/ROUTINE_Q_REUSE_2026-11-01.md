# 常规有界调压V5：实施与复用核查

## 已完成的隔离改造

项目为`D:\博士资料包\科研\科研\AAAI_RL\tarp_37_voltage_v5`，分支`codex/voltage-routine-v5`。V4项目保持只读，原全流程继续进行。本轮新增的是常规有界Q控制、其后的Voltage标签、额外处置证据，以及冻结状态的离线复用审计；没有新生产采集、RL训练、专家训练或GPU任务。

正常运行先尝试在原生机组Q范围内恢复AC合规，固定本地有功、逐源逐节点RE利用、备用、基础负荷比例和EV交付。完整正常点有效时可直接执行；否则再进入已有有限资源AC处置。Voltage使用正常调压后的电压，数值失败保留未知。0.93–1.07 pu、0.9 kA、原生P/Q、线路拓扑阻抗和充电站配置全部不变，12组硬件/容差字段与V4 manifest逐项一致。

原点同时违反禁止反送且电压越限时，不用增加无功环流和线路损耗来消耗有功盈余。该状态进入额外有功处置分类，未核验的常规Voltage保留未知。没有放开原生根节点Pmin来使Q问题看起来可行。

额外处置按P/Q变化、逐源弃发、备用、基础负荷失供和成本分列。正常点完整有效时才计算相对成本差，差值保留正负；没有有效正常基准时仍记录实际处置总量与实际成本，不编造差值。

## 首个完整开发source

`2024-03-04_s101`的32分支、192小时已封存。这里复用V4实际运行状态计算新常规Q基准，不是重新生成了V5闭环轨迹。

| 六小时风险头 | V4正例 | V5审查正例 | V5审查负例 | 未知 |
|---|---:|---:|---:|---:|
| Voltage | 32 | 0 | 30 | 2 |
| NetworkCapacity | 0 | 0 | 32 | 0 |
| ChargingService | 4 | 4 | 28 | 0 |
| RenewableCurtailment | 18 | 18 | 14 | 0 |

全部三类非Voltage事件及其掩码与原记录逐项相同。新口径下有10条全正常分支、2条完整复合分支；2条Voltage未知没有补成安全标签。192小时中，150小时经Q纠正后完整合规，28小时原本已完整合规，12小时电压本身合规但仍有其他边界问题，2小时Q求解数值未知。

这个source支持“原Voltage高频正例大量来自调压基准”的判断，不支持将全部日期的Voltage风险已经归零。全量28 source结果完成前不启动重训，也不把首个日期外推为总体发生率。

## 额外处置与成本的第一项发现

178个小时具有完整有效的常规Q基准，可以比较原V4执行点与该基准的成本。差值全部为负，中位数约−3899.16，范围约−17757.62至−2046.04，单位沿用原生模型成本单位，非测量人民币或CAISO美元。

它说明原V4 AC执行器还承担了经济性重调度，而非仅在越限时付出更高代价保护系统。不能将所有有功变化写成“额外应急损失”，也不能把这组负差值写成新V5闭环的节约效果：比较对象是同一冻结小时状态下的两个运行点。新代码分别保存正常Q、实际经济/保护处置和有符号成本差，避免再次混用。

## 全量复用与判断顺序

运行号`routine_q_reuse_v5_s42_20261101`正在复用全部28 source、896分支。首个source按封存hash复用，后续使用4个CPU worker，不重复交通rollout，也不接触V4的GPU流程。源文件、派生输出与控制代码均纳入hash；恢复时先检查原source输出，再检查派生缓存。

待全部source封存后，先按source与角色整理四头正/负/未知、全正常和复合支持度，以及正常Q工作量、额外处置和成本。若Voltage仍无正例，不能直接训练并宣称其识别能力；若未知集中在特定数值工况，先定位未知原因。是否需要新的闭环采集，要看常规Q纳入实际执行后是否改变后续资源与行动，以及新任务是否已有足够的有效正负支持。

离线派生数据标记`training_eligible=false`，packing和在线特征接口均拒绝混用旧契约。后续如需V5训练，必须采用与V5接续机制一致的轨迹及独立验证来源。当前审查不是V5独立confirm性能结果。

## 核验与运行位置

完整测试72 passed；独立只读审查发现的3项Important全部修复并有回归测试。硬件比较为12组字段零差异。审计新增采集、RL、专家训练计数均为0。

- 状态与源分布：`artifacts/regulation_reuse/routine_q_reuse_v5_s42_20261101/summary.json`
- 后台进程信息：同目录`runner.json`
- 全量stdout/stderr：同目录`full_reuse.out.log`、`full_reuse.err.log`
- 每source行证据与分支标签：同目录`sources/<source>/rows.json`、`branches.json`、`complete.json`
- 硬件逐项对照：同目录`hardware_comparison.json`

下列恢复命令仅在当前审计进程已退出后使用；现有进程运行时无需再次执行。

```powershell
Set-Location -LiteralPath 'D:\博士资料包\科研\科研\AAAI_RL\tarp_37_voltage_v5'
Get-Content -LiteralPath '.\artifacts\regulation_reuse\routine_q_reuse_v5_s42_20261101\summary.json' -Raw
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_regulation_reuse.ps1 `
  -RunId routine_q_reuse_v5_s42_20261101 `
  -Workers 4 `
  -Resume
```

V4对照的状态仍位于兄弟项目`artifacts/workflows/full_v4_s42_20261101/status.json`。现有heartbeat已同时跟进两项运行，V5只做本轮授权的标签/处置审计，不自行启动新一轮训练。
