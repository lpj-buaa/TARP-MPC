# TARP-MPC：代码、冻结输入与运行复现

本仓库对应当前的37节点主案例与序贯信息采集研究。代码从隔离分支 `codex/intervention-information` 的 `7779f58` 导出，保留既有历史。上传过程不修改正在运行的原项目、MPC、网络容量、风险阈值、旧RL或预警器。

## 内容与数据入口

Git仓库包含全部研究源码、配置、脚本、单元测试、设计规格、执行计划和已完成阶段的图文文档。大文件在本仓库的 [Releases](https://github.com/lpj-buaa/TARP-MPC/releases) 中提供，版本为 `research-snapshot-2026-11-03`。

`frozen-inputs-and-completed-evidence-2026-11-03.zip` 包含冻结原生输入、28个source的原始记录和32/40小时快照、原生V2SIM源代码及许可证、交通适配器、5000辆EV案例与需求先验、校准风光荷、旧模型保留件、162条已完成配对分支及MPC测量、历史开发学习结果和checkpoint。该727MB归档分为22段传输。`input-manifest.json`逐文件记录位置、字节数与SHA256，`parts-manifest.json`记录分段校验，`release-summary.json`记录整个归档的SHA256。下载脚本自动核验分段并恢复原ZIP。

当前正式序贯实验 `sequence_formal_s42_v1` 仍在原机采集。本次归档不把正在写入的分支或中间模型当成已完成结果。112情境、3584条分支和后续三种子评价完成后，再单独封存正式结果。现有开发pilot只有一个检查source，不能用作正式独立confirm结论。

## 新电脑恢复

当前冻结协议保留绝对路径，原生V2SIM pickle亦保留原模块身份。为了按原hash直接复用，推荐使用相同Windows目录结构；下面命令展示该结构。改变盘符或目录需要另建迁移版本及协议，不能直接对旧RunId使用Resume。

```powershell
$workspace = 'D:\博士资料包\科研\科研\AAAI_RL'
New-Item -ItemType Directory -Path $workspace -Force | Out-Null
git clone https://github.com/lpj-buaa/TARP-MPC.git (Join-Path $workspace 'tarp_37_information_acquisition')
Set-Location -LiteralPath (Join-Path $workspace 'tarp_37_information_acquisition')

# 自动下载、逐段核验并合并归档；已通过校验的分段会复用。
py -3.11 .\scripts\download_release_bundle.py --output "$env:USERPROFILE\Downloads\TARP-MPC"
Get-FileHash -Algorithm SHA256 "$env:USERPROFILE\Downloads\TARP-MPC\frozen-inputs-and-completed-evidence-2026-11-03.zip"
# 已有同hash输入会跳过，不覆盖不同内容。
py -3.11 .\scripts\restore_release_bundle.py `
  --archive "$env:USERPROFILE\Downloads\TARP-MPC\frozen-inputs-and-completed-evidence-2026-11-03.zip" `
  --workspace $workspace

# 两个环境对应原有物理/学习运行时；不复制虚拟环境本身。
py -3.11 -m venv ..\tarp_risk_discovery\.venv
& ..\tarp_risk_discovery\.venv\Scripts\python.exe -m pip install -r .\requirements\physics.txt
py -3.11 -m venv ..\tarp_x_expert_evaluation\.venv
& ..\tarp_x_expert_evaluation\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& ..\tarp_x_expert_evaluation\.venv\Scripts\python.exe -m pip install -r .\requirements\learning.txt

$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& ..\tarp_x_expert_evaluation\.venv\Scripts\python.exe -m pytest .\tests\test_sequence_v4.py .\tests\test_acquisition_evidence.py -q
```

物理环境和学习环境记录见 `publication/environments.json`。GPU机器需匹配PyTorch CUDA运行条件；也可使用CPU进行小规模检查。原生交通后端为V2SIM自带UXsim模块，主案例不依赖外置SUMO进程。

## 运行顺序

在新电脑先运行独立smoke；原机当前正式RunId继续由既有runner管理。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_sequence_v4.ps1 `
  -Mode smoke -RunId github_smoke_s42_v1 -Stage all -Workers 4 -Device cuda

# 完整实验：新RunId；112情境，每情境32条分支。
powershell -ExecutionPolicy Bypass -File .\scripts\run_sequence_v4.ps1 `
  -Mode formal -RunId github_formal_s42_v1 -Stage all -Workers 4 -Device cuda

# 复用归档中的已完成缓存开发实验，不新仿真、不重新拟合。
powershell -ExecutionPolicy Bypass -File .\scripts\run_sequence_v4.ps1 `
  -Mode formal -RunId sequence_cache_pilot_s42_v1 -Stage learn -Device cuda -Resume
```

最后一条命令复用的旧learning_protocol绑定原始代码路径；只在文档指定的原目录中使用。学习文件或依赖已改变时，脚本会拒绝Resume，此时保留旧现场，使用新RunId，不跳过hash检查。

## 当前科学比较

稳定后果模型学习行动对未来调度支持区间、储能恢复工作量、恢复缺口、后续调度和偏好的影响。采集策略的MDP状态包含当前运行情境、已查询动作及其定量后果、未查询动作集合与剩余预算。

正式比较序贯PPO（gamma=1）、即时PPO（gamma=0）、组相对序列策略、后果模型即时贪心、局部插值、括区＋Random、参数Sobol和Random，预算8/16/24。指标为固定候选池后果画像覆盖、相邻恢复边界检出、未知查询消耗与实际仿真成本；按独立source聚合及配对bootstrap。

这项实验检验有限候选池中的信息采集效率，不主张已经穷尽连续动作空间。开发pilot支持学习策略优于Random，但尚未显示多步策略超过即时策略及强插值对照；正式结论等待独立confirm。

## 两台电脑协同

开始编辑前 `git pull --ff-only`，每项工作新建分支，提交后推送并通过Pull Request合并。数据与大checkpoint通过Release同步，Git仅管理源码和轻量证据。不要同时修改另一台电脑正在运行且被hash冻结的源码；正在运行的版本保留，修改另建分支与RunId。

```powershell
git pull --ff-only
git switch -c experiment/your-task
# 编辑、测试
git add src scripts tests configs docs
git commit -m "Describe the research change"
git push -u origin experiment/your-task
```
