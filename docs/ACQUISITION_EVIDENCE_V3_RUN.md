# 定量经验与稳定后果监督：实施记录

主报告：[中文HTML](acquisition_evidence_2026-11-02/index.html)，保留完整[Markdown](acquisition_evidence_2026-11-02/report.md)及两张组合图的PNG/SVG。

## 运行与隔离

工作目录`D:\博士资料包\科研\科研\AAAI_RL\tarp_37_information_acquisition`，分支`codex/intervention-information`。新增`src/tarp_acq_evidence`、对应scripts和tests；未修改tarp37/tarp_info/tarp_boundary。

正式运行`artifacts/acquisition_evidence_v3/evidence_formal_s42_v1`：复用162条分支、486组缓存MPC测量。3种子×2协议×5方法=30任务，其中24个新采集评分器拟合，6个无训练插值任务。两次小规模流程另有16个评分器拟合，全部独立RunId留存。全程无新增物理仿真、LP测量或旧RL/专家/预警器训练，不占GPU。

四种配置使用相同823维物理通道、361维经验通道、51维动作、相同双128层架构，共365178参数；关闭通道补零。空间和定量经验作为一个输入因素，固定60维后果作为一个监督因素。初始参考未来属于共同已知经验，未查候选真值仅在Oracle查询后训练、或事后评价中读取。60维固定后果可重建全部162条分支原22维画像（1e-10内）。

## 完整命令

```powershell
Set-Location -LiteralPath 'D:\博士资料包\科研\科研\AAAI_RL\tarp_37_information_acquisition'

# 已完成正式实验：核验并复用，新增拟合0
powershell -ExecutionPolicy Bypass `
  -File .\scripts\run_acquisition_evidence_v3.ps1 `
  -Mode formal -RunId evidence_formal_s42_v1 -Resume

# 从头重复同一配方：新RunId；仍不启动物理仿真
powershell -ExecutionPolicy Bypass `
  -File .\scripts\run_acquisition_evidence_v3.ps1 `
  -Mode formal -RunId evidence_formal_repeat_s42_v1

# 相关测试
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& '..\tarp_x_expert_evaluation\.venv\Scripts\python.exe' -m pytest `
  .\tests\test_acquisition_evidence.py .\tests\test_boundary_checkpoint.py -q
```

脚本自动生成analysis.json、source_records.json、diagnostics.json和每个RunId自己的report/index.html；制图解释器默认D:\anaconda\python.exe，可用-ReportPython替换。旧RunId不能更改配置；中断后使用-Resume。

## 核心结果

64次共同查询后，六个source的排序Spearman（三种子平均后聚合）：旧摘要0.2495，空间＋定量经验0.3123，稳定后果监督0.2951，两者结合0.3033，局部物理插值0.7358。结合相对旧摘要+0.0538，source配对95%CI[-0.0841,0.2116]；Top3真实效用56.3%对59.5%，没有改善。

固定预测的二级读出诊断：稳定后果预测＋物理投影＋显式原奖励计算，Spearman0.5141，较该模型直接Q值0.2951提高0.2190，CI[0.0222,0.4226]。不增加训练，但尚未将这一读出接入新的主动采集臂。该投影利用本缓存能量必要边界与LP缺口一致的已核验事实，不能延伸成通用网络可恢复性定理。

主动查询64预算：旧摘要/稳定监督/括区＋Random边界定位100%，结合93.3%，局部插值86.7%，Random46.7%。固定144候选池的画像距离：插值0.003311，Random0.003607，旧摘要0.004192，结合0.004602。该指标对全部候选平均，已查项距离为零；不是剩余80条的条件均值。

结论：输入补齐尚未兑现稳定的价值排序/采集收益；连续后果有可学习规律，而且显式后果投影比直接Q读出更能利用这些规律。下一步指向解耦后果学习与经验库相关的价值计算，不是扩大网络。所有比较仅对应本六个历史source的复用缓存。

## 验证与修复

13项相关测试通过。465项冻结输入/旧代码/旧checkpoint哈希前后一致。正式30任务均封存，128次查询逐任务无重复；共同历史与剩余候选逐快照一致。正式Resume跳过全部30任务、拟合0次。见运行目录final_verification.json。

一次独立whole-branch review发现Windows分析路径分隔错误：`str(relative_path)`改为`.as_posix()`。测试先复现失败再修复通过；训练结束后才修改分析器，未改训练代码或产物。原分析代码、原protocol和新旧哈希保存在analysis_training_original.py、protocol_training_original.json、analysis_code_revision.json中。无须重训；修复后再核验Resume。未发现未来泄漏或匹配历史比较错误。代码保留隔离分支，不合并、不推送。
