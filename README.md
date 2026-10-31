# 37nodes统一闭环与全流程重训

主项目：`D:\博士资料包\科研\科研\AAAI_RL\tarp_37_unified`。37nodes用于方法主实验，南京用于规模与交通/V2G验证。此项目独立于原R3、专家库和RL，所有旧资产只读。

先阅读 [完整重训与执行方案](docs/RETRAINING_EXECUTION_CN.md)。当前物理版本为 `native37_pq_ac_0p9ka_economics_v3`。有效最终小规模核验使用 `unified37_smoke_s42_v4`；较早v1–v3保留为诊断，不能混入此版本训练数据。

## 最短执行路径

```powershell
Set-Location -LiteralPath 'D:\博士资料包\科研\科研\AAAI_RL\tarp_37_unified'

# 小规模闭环；已有结果只校验复用。
powershell -ExecutionPolicy Bypass -File .\scripts\run_unified37.ps1 `
  -Mode smoke -RunId unified37_smoke_s42_v4 -Workers 1 -Resume

# 六个独立日期分别承担六种数据角色，检查学习全流程。
powershell -ExecutionPolicy Bypass -File .\scripts\run_unified37.ps1 `
  -Mode pilot -RunId unified37_pilot_s42_v1 -Workers 4 -Resume
if ($LASTEXITCODE -ne 0) { throw 'Pilot collection failed' }

powershell -ExecutionPolicy Bypass -File .\scripts\run_retraining.ps1 `
  -Mode pilot -DataRunId unified37_pilot_s42_v1 `
  -RunId unified37_train_pilot_s42_v1 -Task all -Device cuda -Queries 12 -Resume
if ($LASTEXITCODE -ne 0) { throw 'Pilot training failed' }
```

物理报告：`artifacts\runs\<DataRunId>\analysis\index.html`。学习结果：`artifacts\retraining\<RunId>\evaluate\metrics.json`、`coverage.json`、`comparison_ci.json`。

## 组件

`contract/network/mpc/traffic/environment`统一物理能力与实际交付；`risks/collection`生成版本化配对数据；`packing/model/learning`训练图与时序编码、GRU演化及专家；`acquisition/query`训练新的一次干预采样器并调用真实六小时MPC分支；`report`只读取已发布记录。

没有自动启动正式重训。旧运行策略不继续训练；新RL用于信息采样，日常接续由MPC完成。当前专家融合是有独立meta数据的浅树证据融合基线，旧通信模型保留在原项目，尚未移植到新标签上。
