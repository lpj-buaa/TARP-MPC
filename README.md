# 干预后的资源与调度空间：信息采集隔离实验

本项目位于 `D:\博士资料包\科研\科研\AAAI_RL\tarp_37_information_acquisition`，分支 `codex/intervention-information`，从 V5 `54f4408` 建立独立工作树。原有物理模块、容量、阈值、RL与专家不改。本轮只运行信息采集学习；不训练风险预警器或专家融合。

先读 [中文结果与图表](docs/information_acquisition_2026-11-02/index.html)，[设计规格](docs/superpowers/specs/2026-11-02-intervention-information-design.md) 与 [实施核验计划](docs/superpowers/plans/2026-11-02-intervention-information.md)。

## 已完成结果复用

```powershell
Set-Location -LiteralPath 'D:\博士资料包\科研\科研\AAAI_RL\tarp_37_information_acquisition'
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_information.ps1 `
  -Mode pilot -RunId information_pilot_v2 -Resume

# 只重绘结果，不训练、不采集
& 'D:\anaconda\python.exe' -B .\scripts\analyze_information.py `
  --run-id information_pilot_v2
```

v2为修正Sobol公共坐标与独立情境序列后的四臂完整对照。共同初始192条、历史候选384条；每臂每seed揭示128条答案，seed42/43/44。该实验属于有限历史缓存上的采集比较。两种学习臂使用相同TD3式双critic，actor不用于本轮动作提出。保留v1作核查，不混选运行结果。

## 独立小验证的复现实验命令

以下会运行新的8条物理分支；已有 `v5_information_sanity_v1` 无需再次执行。请为确需的新实验使用新RunId。

```powershell
$infoProject = 'D:\博士资料包\科研\科研\AAAI_RL\tarp_37_information_acquisition'
$infoSource = 'D:\博士资料包\科研\科研\AAAI_RL\tarp_37_risk_audit'
$infoPhysicalPython = 'D:\博士资料包\科研\科研\AAAI_RL\tarp_risk_discovery\.venv\Scripts\python.exe'
Set-Location -LiteralPath $infoProject
$env:PYTHONPATH = Join-Path $infoProject 'src'
& $infoPhysicalPython -B -u -m tarp_info.native `
  --source $infoSource --run-id v5_information_sanity_repeat_v1 --workers 4

# 只读重算已存在分支的后续MPC支持区间，不重复交通推演
& $infoPhysicalPython -B -u .\scripts\opportunity_probe.py `
  --native-run v5_information_sanity_v1

# 本轮12项新接口/机制测试及5项既有风险、MPC测试
& $infoPhysicalPython -B -m pytest `
  .\tests\test_info_acquisition.py .\tests\test_risks.py .\tests\test_mpc.py -q
```

`src/tarp_info`负责画像、Oracle、采集、版本绑定与真实接续扩展；`scripts/opportunity_probe.py`只读复用原MPC构造器测量方向支持区间。`src/tarp37`继承V5且保持不变。结果分别位于 `artifacts/information/information_pilot_v2` 和 `artifacts/native_information/v5_information_sanity_v1`，二者物理版本及证据用途分别记录。


## 第二轮：调度范围、转折与恢复边界

代码在独立的src/tarp_boundary，结果在artifacts/boundary_v2/boundary_formal_s42_v2；前一轮tarp_info及原tarp37保持不变。新库为六个同状态情境、两类动作的12个内点剂量，每个情境另有3条公共初始经验。先真实推演、再用同一Oracle池比较四个主方法和两个组件消融，预警器仍不训练。

```powershell
Set-Location -LiteralPath 'D:\博士资料包\科研\科研\AAAI_RL\tarp_37_information_acquisition'

# 已完成结果的核验复用；不重新采集、不重新拟合
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_boundary_acquisition.ps1 `
  -RunId boundary_formal_s42_v2 -Stage fit

# 新完整实验使用新RunId：会调用真实仿真并训练新采集器
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_boundary_acquisition.ps1 `
  -RunId boundary_repeat_s42_v1 -Stage all -Workers 4

# 只读最终验收
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& '..\tarp_risk_discovery\.venv\Scripts\python.exe' -B `
  .\scripts\verify_boundary.py `
  --root .\artifacts\boundary_v2\boundary_formal_s42_v2 --final
```

本轮设计与操作定义：docs/superpowers/specs/2026-11-02-dispatch-boundary-design.md；执行计划：docs/superpowers/plans/2026-11-02-dispatch-boundary.md。原生采集与域读出运行时不要修改其协议绑定源码。Windows不得用stdin脚本启动ProcessPool；已提供模块入口和PowerShell主流程。
