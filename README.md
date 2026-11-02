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
