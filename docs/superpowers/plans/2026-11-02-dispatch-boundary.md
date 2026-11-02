# Dispatch Boundary Acquisition Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. User explicitly authorized implementation and execution without further stage approvals.

**Goal:** 获得恢复边界两侧成对动作及等预算信息采集对照。
**Architecture:** 新包tarp_boundary负责真实V5共库、MPC支持与恢复读出、公共/私有Oracle、匹配采集和统计；原tarp_info与tarp37不改。
**Tech Stack:** 原Python环境、NumPy、SciPy、pandapower、PyTorch CPU、matplotlib。
**Spec:** docs/superpowers/specs/2026-11-02-dispatch-boundary-design.md。

## Global Constraints
- 0.9kA，0.93–1.07pu与所有原生硬件冻结；旧RL/专家/预警器训练0。
- 全部未来信息须在一次付费查询之后揭示；未知不补零。
- 6source、18初始/144候选；主预算64、前缀8/16/32/64/128；3seed。

## Review Focus
- 首步ESS恢复表达式的符号和效率；通过一维能量约束已知可恢复/不可恢复反例测试。
- mode/端点数值失败不能伪装已知域；注入失败测试。
- 查询顺序不得使用全集边界；隐藏未来替换/预算/邻接测试。
- 恢复参考按source及future hour严格配对，禁止跨情境。
- 中断复用不重复native和训练；文件hash验证。

## Task 1 — Domain reader
- [x] tests/test_boundary_acquisition.py先写恢复缺口/边界邻接/未知测试，确认RED。
- [x] src/tarp_boundary/domain.py私有clone原MPC，支持区间与恢复缺口同一次LP捕获，所有状态保存；GREEN。

## Task 2 — Native corpus
- [x] src/tarp_boundary/corpus.py生成6情境/两族12内点，复用已有6条真实记录并按上下文核验；首18初始再144候选，4CPU workers。
- [x] 保存协议、代码/初态hash、future views；每分支独立完成，不因未知重标标签。
- [x] 对未来三时点并行读出与冻结教师前向；生成公共候选与私有答案。

## Task 3 — Matched learning
- [x] src/tarp_boundary/learning.py实现已查询集合上的新颖性/成对边界/转折奖励，复用旧双critic但不改旧代码。
- [x] tests验证同情境、两端收费、隐藏答案、无自邻居；四臂3seed，完整预算前缀。

## Task 4 — Evidence and verification
- [x] src/tarp_boundary/cli.py及scripts/run_boundary_acquisition.ps1，支持全流程/Resume与独立阶段。
- [x] 组合图HTML/source配对统计、原始物理处置核查、全部新测试与17既有测试。
- [x] 独立review、冻结文件hash与最终结果中文说明；提交不推送。

## 实施记录

用户已直接授权实施和跑结果。已有隔离worktree codex/intervention-information，无需再建worktree。新增包/RunId隔离，原tarp_info/tarp37没有修改。

Builder审阅后固定相近剂量1/13、转折容差为名义功率标尺1%、新颖性距离0.1截断。增加奖励-only和括区+random消融；三个学习臂匹配18维证据知识摘要，均使用原128宽双critic。

Windows首个stdin驱动无法spawn，未执行新native，旧现场保留。有效run为boundary_formal_s42_v2，corpus和measurements分别通过模块入口在4CPU worker执行；各自代码与依赖已冻结，不可直接改正在运行的domain/corpus/io/measurements/tarp_info/tarp37。

基础11项新测试已RED→GREEN，原17项测试也通过（新增知识状态测试后会合并最终28项）。旧模型只读前向接口用一个真实V5记录验证，18个同名物理量对齐，排除旧raw/新regulated Voltage语义差异。仅恢复读出测试已发现0/0/0.410526MWh（ref/throttle/release），不用于正式方法选型或阈值调整。


## 运行中的交接状态

有效运行 artifacts/boundary_v2/boundary_formal_s42_v2。native执行session72850（4CPU）、MPC测量session80239（4CPU），finish串行后处理session95358正在等待collection_complete.json及measurement_complete.json，然后自动执行冻结教师、bundle、18臂拟合与图表。不要再次启动采集或测量。首次stdin失败session78917已退出，未新执行native，v1现场保留。

核查reader无剩余Critical/Important。修复reference左端括区统计，已RED→GREEN回归；最终29项pytest通过（12新+17既有）。三个学习臂额外通过真实PyTorch维度/检查点/首次查询未来隔离的合成接口测试，这些数据不用于科研图表。外部verify_boundary.py负责Resume前核对旧seal，避免重写seal掩盖变化；不要改正在运行的原生/测量代码。

当前实现与预先规格将先提交，所有正式方法结果尚未用于调参。最终仍需读取analysis.json、核验全部source与副作用、执行Stage fit零训练复用、运行verify_boundary.py --final并更新本计划及中文报告结论。


## 采集价值诊断补充（拟合前冻结）

保存budget64时的真实critic权重，事后在当时尚未查询的80个候选上评价Q与真实效用的source内Spearman；不得使用最终128权重冒充64模型，不回流训练。新增checkpoint回归先FAIL（无b64文件）再补保存位置，ML runtime测试单列。当前finish PowerShell进程在脚本更新前启动，因此结束后须手工运行diagnose_boundary_value.py并重建analysis报告，后续Stage fit命令已自动包含诊断。


## 初态等价审计

直接JSON逐位比较报告view/routine_generation差异，经151条记录逐项数值核查，最大仅8.899e-16，来自集合求和顺序导致的浮点舍入。没有车辆、资源、外部预测或物理初态改变。外部审计使用绝对容差1e-10递归比较并保存最大差值，不改仿真输入、物理限值或风险阈值；新增舍入可接受/0.001实质差异必须拒绝的RED→GREEN测试。物理runtime最终30tests通过，ML runtime的64预算检查点测试另1通过。


## 最终结果与核验

已完成162原生分支（新增156、复用6）、972小时实际AC有效、486个MPC future读出；18初始/144候选。主六配置×3seed共18臂，加原生family-dose Sobol×3共21臂。所有128查询预算匹配，主64。

补充原生Sobol原因：两条一维动作线的51维MW-box映射偏向中部，旧映射保留作补充；新增2D family-dose Sobol在共同公共参数域公平采样，不重新仿真或拟合。测试确认双族、剂量两端与唯一访问。

64预算新版5/5=100%，Random46.7%、参数Sobol26.7%，Error及BracketRandom也100%。新版剩余画像距离vsError降34.2%，source配对差CI[0.0001499,0.0053675]；vsRandom无显著优势。没有宣称RL独立贡献已确立。

物理规律：9个正恢复缺口均在+1小时，+3/+5全部可恢复；全部486状态rho与ESS能量/充电速率必要下界一致。8/9恢复受限时ESS支持区间更宽；19/78释放动作跨4source先扩后缩。四类标签另列：Voltage0/118/44、Capacity0/162/0、Service14/148/0、RE162/0/0（正/负/未知）。不把该库当全风险预警验证数据。

value rank：Error自身rho0.727、对机制目标-0.135；新目标only0.019、混合0.087。下一版优先让critic读取已查询的连续调度/恢复证据，并设计更耦合恢复任务；不堆网络或立即重训预警器。

Stage fit已零训练复用18臂，参数Sobol复用3臂。最终31项物理runtime pytest及1项ML检查点pytest通过。原tarp37/tarp_info、旧V4/V5项目、小时数据hash均保持一致。完整中文报告和3组可编辑SVG/PNG见docs/boundary_acquisition_2026-11-02/index.html；数据与seal见artifacts/boundary_v2/boundary_formal_s42_v2。


## 最终同行复核

新参数Sobol三个seed查询顺序/计费/评价均复算一致，末轮review无剩余重要问题。报告突出两个可复现方向机制：8/9受限恢复分支却ESS范围更宽，19/78释放动作跨4source先扩后缩；不混同“区间更宽”和“任务更可恢复”。报告表格、CI、标签支持、critic排序均与封存数据一致。

完整执行与命令已在README与HTML给出，保持独立分支/项目，按用户要求不合并到旧项目、不重训预警器。全部采集、测量、后处理进程已结束。
