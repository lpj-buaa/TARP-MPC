# 37nodes主案例统一闭环与重训执行方案

## 1 研究和版本边界

本轮先建立一个可解释的物理与控制基准：一次站点或储能配额调整，改变实际充放电、车辆SOC、可用储能与等待状态，继而改变后五小时MPC的调度响应。研究对象是这个闭环后果，而非单纯比较MPC和TD3控制优劣。新RL学习哪些干预值得采样，MPC执行日常滚动调度，预警器学习执行前的多维后果，专家融合利用独立经验组合风险证据。

项目位于 `D:\博士资料包\科研\科研\AAAI_RL\tarp_37_unified`，独立Git分支 `codex/unified37`。原始V2SIM、dirty R3、旧TD3 checkpoint和旧专家库未作原地修改。输入、阶段输出、模型配置和冻结资产均记录SHA256；改变物理代码、数据源或学习配置须使用新RunId。

最终物理版本 `native37_pq_ac_0p9ka_economics_v3`、风险版本 `four_physical_risks_0p93_1p07_v3`。v1–v3小运行用于发现和修正执行、费用尺度问题，不能拼入当前训练。最终小闭环数据为 `unified37_smoke_s42_v4`。它只有一个train日期，用于物理与接口核验，不能代替独立训练/验证/确认实验。

## 2 保留和统一的物理现实

“37nodes”是道路案例名，电网实际上有66个bus、64条线、两个独立33节点馈线。两个根节点各30 MW，另8台本地机组各30 MW；两类机组都参与MPC和实际有功/无功调度，保留原始P/Q范围。总计300 MW是机组铭牌上限相加，不等于任意位置都能取得300 MW。线路阻抗、连接、道路、充电站数量和插头数原样保留；线路电流统一0.9 kA，非根节点电压0.93–1.07 pu，根节点1.0 pu。没有另加旧TARP的25 MW总进口限额，没有将R/X乘0.8，也没有将本地无功截为±5 Mvar。

原生10个FCS各10位、37个SCS各20位，共840插头。原XML未给有限每桩功率；实际由车型充放电上限、连接状态和站点配额限制。MPC规划包络由车型最大功率乘位数得到，不能解释为厂家额定值。车辆文件复用已生成5000辆原生出行。V2G的50%参与和协议电价处理是显式场景设定，未改车型物理功率或容量。四台2 MWh、±1 MW ESS以及两个1 MW/6 MWh备用资源是新增、逐项记录的场景资产；备用能量用掉即减少，不能自动无限补满。

配电网采用一小时准稳态控制与结算。交通内部1秒推进、10秒交换控制，输出15分钟遥测；每个电网小时聚合四个交通区间。1秒处理修复了旧10秒reaction-time误用造成的道路容量失真。这不是秒级暂态或继电保护仿真。

EIA风光荷实际值与已有发布时刻约束的预测列被共同缩放，不使用未来实测安排当前动作。基准非EV平均负荷10.386 MW，独立正常EV参考57.663 MWh/日，规划总均值12.789 MW。按参考期可用新能源电量70%配置：PV约28.247 MW、wind约7.430 MW，并按负荷节点/本地发电节点分布。70%是参考期可用电量目标，不等于每个测试日实际供能占比。原始CSV/Parquet与装机来源留在数据导出manifest中。

## 3 MPC与保护怎样闭合

MPC只使用当前可观测车辆需求、接入状态、资源存量、两小时前序、已发布六小时预测和固定开发期到达先验。经济、服务、资源和调整平滑四项偏好随可见状态平滑变化。偏好不包含预警风险分数。MPC原生机组成本按500P²+1000P+10原曲线分段近似；线路、电压和机组能力来自同一个PhysicalContract。线性化电压与内接16边形电流约束是规划近似，实际执行通过AC方程独立复核。

执行前仅用可见预测检验配额包络，必要时下调配额；车辆在该配额下不可逆推进一次。随后取得当小时实际风光荷和实际交付功率，以有功/无功调度、弃发、有限备用及有限非EV失供阶梯进行AC补救。已经交付的EV能量不追溯撤销，不回看整小时交通后重新挑选配额。AC无法验证时保留真实交通和已知正例，未知后缀不充当安全负例。

本轮纠正了两个实质费用问题。首先原机组边际成本约几千native cost units/MWh，不能沿用几十的费用归一化，否则MPC会普遍拒绝快充。当前共同费用尺度为2278.895，备用155000/ MWh、非EV失供620000/ MWh、弃发455.779/ MWh，均为场景相对代价，不声称实测人民币。其次仅奖励消纳会使多损耗变得有利，因此固定交付后的AC优化增加2278.895/MWh损耗正则，并单列输出；不把这一供给偏置加到无损耗MPC充电项，也不将正则混入货币费用。

## 4 四类事件与后果标签

电气风险是“真实交付后、主动AC补救前，按当前常规有功计划和有限无功调节暴露的应力”。原始配额请求的预测压力、实际交付的补救前压力、补救后执行点三者分别存储。

|风险头|六小时窗口事件|辅助物理量与解释|
|---|---|---|
|Voltage|任一常规响应下bus电压超出[0.93,1.07]，超过1e-6 pu容差|超限幅度、时段、节点；补救后电压另作安全检验|
|NetworkCapacity|任一线路I/0.9kA>1+1e-6，或PCC实际所需有功超过原始[0,30] MW边界|逐线电流、根节点供电/反送要求；反送禁止与热容量分别解释|
|ChargingService|发生离站能量不足/因能量不足而延迟，或实际FCS等待超过30分钟|去重vehicle-trip缺口、实际排队车·分钟、SOC及等待；30分钟是服务场景标准|
|RenewableCurtailment|wind或solar任一在完整六小时窗口内弃发/可用电量>10%|按风光分别计算、保留可用和弃发MWh；窗口运行标准，不称年度消纳考核结论|

同时预测19项物理后果、六小时6维资源状态、51维实际交付动作、4维MPC偏好。成本和线路损耗是后果指标，不再混入一个“效率风险”替代物理事件。部分窗口中已经出现的事件可作已知正例；没有完整观测不能判为负例；弃发比例需完整窗口。

## 5 为什么需要重新训练，以及哪些不用重做

可以复用原始道路、电网、站点、车辆行程、EIA原始数据及图/Transformer/GRU代码，避免重新下载和造交通。旧25 MW边界、旧无功限制、旧风险类别与旧接续TD3生成的rollout不能作为新物理版本标签。必须重新生成当前MPC接续配对轨迹、拟合train scaler、训练新预警器和专家、独立训练融合并重新校准阈值。

MPC不是神经网络，不需要“重训”，需要核验模型、费用尺度、预测可用性和偏好规则。原TD3控制策略不重训、旧专家不覆盖。新增信息采样器从scratch学习，其checkpoint与旧控制RL分开。旧模型保留作历史对照；在新标签上评价方法增量，应优先使用同数据重训的state-only、单专家、均值、校准并集和证据融合，不能直接把旧标签的报警当作新四头真值。

当前模型沿用现有ConsequenceStudent图编码、历史Transformer和GRU演化解码；将“首小时请求配额”改为输入条件，首小时“实际交付”也是学习目标。两小时前序包含2个电网状态token和8个真实交通quarter token。未来预测、资源、上一步响应、候选动作和偏好参与六步演化。训练联合绝对后果、同状态参考相对delta、资源、实际响应、偏好及四类窗口事件；未来真值只进入loss。

首轮专家包括full/local/wide_graph/service/resource/boundary六个，另训state-only。当前融合基线以浅树学习风险条件证据，meta_train独立于基座训练；已有高级通信系统保存在旧项目，未冒称已完成新风险适配。先确认新专家是否真正学到动作差异，再决定是否移植通信。

## 6 信息采样RL的实际定义

新采样器是一次干预、随后MPC接续的terminal acquisition问题，gamma=0；使用双critic和延迟actor更新的TD3式实现，不是声称运行控制器具有近似Pareto前沿保证。状态只含公开决策信息，动作是51维可执行站点/ESS请求，真实查询在保存的train状态上执行六小时原生交通与MPC。

奖励由冻结初始student的配对后果预测误差、相对于固定初始训练库的新风险组合以及未知覆盖惩罚组成。它鼓励有学习价值的行动，而不只追求损害最大。历史种子动作必须能在当前策略动作域中精确反向映射，超范围动作不能裁剪后盗用旧奖励。所有meta/cal/confirm状态禁止成为RL采样源。TD3、Sobol、random按相同查询数和相同初始库比较，新增标签只补入train。

## 7 数据角色、指标与结果门槛

pilot用6个独立日期，一种角色一个source，仅作流程和支持度检查。formal用28个独立日期：train6、tune2、meta_train6、meta_tune2、cal4、confirm8，每source两个分支时刻、8候选、2未来偏好，共896条六小时分支。source表示不同EIA运行日期与独立分支实验；车辆总体仍复用同一5000辆行程库，不声称28个独立城市。

主要比较在cal设定同一source-macro OR FPR预算5%，冻结后报告confirm实际FPR和完整复合召回。完整复合召回要求真实存在的至少两类风险全部识别，额外误报另计。单专家在meta_tune选择；融合只在meta_train训练；按source配对bootstrap，不能把相邻行动当独立样本。候选并集覆盖是证据上限，不是预算受控部署效果。

训练前先检查每个角色、每个风险、每个时段的正负样本支持。缺少的风险不能靠收紧线路制造；可在不改设备条件下增加季节、EV压力、SOC和已知可行行动组合。正式阶段首要报告同状态delta误差/方向准确率、严重端点、首次事件时段、完整复合召回/FPR、资源末端变化与审查耗时。RL有效性需看等预算增加的真实经验能否改善独立confirm预测，不能只看采到更高损害。

## 8 完整命令行

全部命令在PowerShell中执行。`-Resume`可以从首次运行就提供：新RunId正常创建，已完成阶段只在配置和文件校验通过后复用；中断的原生查询保留旧attempt并新建恢复尝试。物理采集若某source仅部分完成，不覆盖旧现场，需新DataRunId。

```powershell
Set-Location -LiteralPath 'D:\博士资料包\科研\科研\AAAI_RL\tarp_37_unified'

# 0. 核验当前最终小规模物理结果，不重复采集。
powershell -ExecutionPolicy Bypass -File .\scripts\run_unified37.ps1 `
  -Mode smoke -RunId unified37_smoke_s42_v4 -Workers 1 -Resume
if ($LASTEXITCODE -ne 0) { throw 'Smoke verification failed' }

# 1. Pilot物理数据：6角色日期、48分支；保留原5000EV/硬件。
powershell -ExecutionPolicy Bypass -File .\scripts\run_unified37.ps1 `
  -Mode pilot -RunId unified37_pilot_s42_v1 -Workers 4 -Resume
if ($LASTEXITCODE -ne 0) { throw 'Pilot collection failed' }

# 2. Pilot完整学习闭环：初始预警器→12次新采样→专家→融合→校准→评价。
powershell -ExecutionPolicy Bypass -File .\scripts\run_retraining.ps1 `
  -Mode pilot -DataRunId unified37_pilot_s42_v1 -RunId unified37_train_pilot_s42_v1 `
  -Task all -Device cuda -Policy td3 -Queries 12 -Seed 42 -Resume
if ($LASTEXITCODE -ne 0) { throw 'Pilot training failed' }

# 3. Formal物理数据：28source、896分支。根据内存可将Workers从4降至2。
powershell -ExecutionPolicy Bypass -File .\scripts\run_unified37.ps1 `
  -Mode formal -RunId unified37_formal_s42_v1 -Workers 4 -Resume
if ($LASTEXITCODE -ne 0) { throw 'Formal collection failed' }

# 4. Formal全流程：新信息采样RL128次，每次6小时MPC接续。
powershell -ExecutionPolicy Bypass -File .\scripts\run_retraining.ps1 `
  -Mode formal -DataRunId unified37_formal_s42_v1 -RunId unified37_train_formal_s42_v1 `
  -Task all -Device cuda -Policy td3 -Queries 128 -Seed 42 -Resume
if ($LASTEXITCODE -ne 0) { throw 'Formal training failed' }

# 5. 不增采样的完整对照，单独RunId；Full初始模型可在此Run内部复用。
powershell -ExecutionPolicy Bypass -File .\scripts\run_retraining.ps1 `
  -Mode formal -DataRunId unified37_formal_s42_v1 -RunId unified37_no_acq_formal_s42_v1 `
  -Task all -Device cuda -SkipRL -Seed 42 -Resume
if ($LASTEXITCODE -ne 0) { throw 'No-acquisition baseline failed' }

# 6. 等预算采样对照。每个Run使用相同初始数据与随机种子，模型独立重训。
foreach ($sampling in @('sobol','random')) {
  powershell -ExecutionPolicy Bypass -File .\scripts\run_retraining.ps1 `
    -Mode formal -DataRunId unified37_formal_s42_v1 `
    -RunId "unified37_${sampling}_formal_s42_v1" -Task all -Device cuda `
    -Policy $sampling -Queries 128 -Seed 42 -Resume
  if ($LASTEXITCODE -ne 0) { throw "Sampling baseline failed: $sampling" }
}
```

需要按阶段检查时，依次将`-Task`设为：`pack_initial`、`fit_initial`、`explore`、`pack_augmented`、`fit_experts`、`predict`、`fit_fusion`、`calibrate`、`evaluate`，保持DataRunId、RunId、Mode、Policy、Queries、Seed完全一致。不要在完成无RL实验的RunId中再补RL，应使用上面的独立名称。

## 9 输出和运行环境

物理数据：`artifacts\runs\<DataRunId>\sources\<source>\records.json`；每个pool保留原生snapshot；`audit`记录物理/经济合同、EIA容量、资产hash；`analysis`含summary、配对资源/调度差异及HTML。

学习数据与模型：`artifacts\retraining\<RunId>\pack_initial|pack_augmented`；`fit_initial`；`explore\proposals|queries|queries_summary.json|acquisition.pt`；`fit_experts\<expert>`；`predict`；`fit_fusion`；`calibrate\thresholds.json`；`evaluate\metrics.json|coverage.json|comparison_ci.json`。根级protocol和每阶段complete标记禁止不同数据或配置混用。

脚本复用现有环境，不安装新依赖：物理仿真用`tarp_risk_discovery\.venv`，GPU训练用`tarp_x_expert_evaluation\.venv`，首次Parquet导出用`D:\anaconda\python.exe`。原始数据export完成后按hash复用。单机不要同时启动多个GPU全流程。

南京的十万辆交通已在修复reaction-time后完成，并有24小时EIA AC回放。另有有符号V2G脉冲核验：19小时点18个验证通过、1个未知；放电后回充义务确实存在。该项目用于规模与机制验证，不把它冒称已完成本项目完整MPC/新四头迁移。后续应以相同合同接口接入165bus案例，仍保留其原始35kV、线路与站点能力，再做同任务规模实验。
