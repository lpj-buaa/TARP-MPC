# Ledger — plan: docs/superpowers/plans/2026-10-31-unified-37.md

2026-10-31：独立项目创建，不修改dirty R3。Task1合同/数据导出、Task2 MPC/AC、Task3原生交通/结算主体完成；Task4新风险/采集/未知mask主体完成。原模型和资产hash冻结。

TDD：每模块先写测试并观察缺失/实际错误，再实现。已观察红→绿：原始P/Q和电流边界、发布时间、保留无功、有限真实削负荷、MPC约束、原生功率/快照、电量、V2G效率、风险部分窗口、RunId和实际消耗车辆hash。

独立代码审查：发现并修复逐站弃发后节点注入错误、车辆多次缺口覆盖、未冻结实际消耗case、可选MPC计划改变标签、回看整小时交通选择保护、全未知分析空集合。完整初轮23项通过；新增全未知回归单独通过。最终全套待完成。

Ruling: 预防配额只用可见预测；原生车辆不可逆只推进一次；已交付EV固定，发电/Q/弃发/非EV失供作小时准稳态补救。失败保留交通和已知标签，不回放。未声称秒级电网保护。

Ruling: 四头中的电气风险明确为预防配额下、真实交付后、有功AC补救前的应力；原始容量请求另作预测基准诊断，未消耗配额不冒称真实负荷。合同及行记录均写明scope。

unified37_smoke_s42_v1在审查时停止，保留为未完成诊断，不作最终结果。修正版unified37_smoke_s42_v2启动；26h MPC前序，3行动×2偏好×6h，5000EV原始出行文件。日志在artifacts/logs。运行期间冻结物理import closure；新增学习/文档模块可以并行编写，但不改冻结物理代码。

以上为早期实施记录；下列是当前交付状态。

2026-10-31最终：unified37_smoke_s42_v4完成26h暖启动+36h分支，62/62 AC有效。原生费用归一化修正为2278.895，AC损耗正则单列；不用旧v2/v3风险数据训练。真实query接口重现6h配对分支，功率误差2.22e-16 MW，电压1.40e-14 pu。采集Resume通过全部冻结输入校验，旧RL与专家无训练。

重训代码已具备：train-only scaler和支持度；图/Transformer/GRU后果学习；gamma0一次干预双critic采样器；真实原生查询；新专家与matched state-only；独立meta浅树证据融合；source-macro OR FPR校准；完整复合召回与source bootstrap。六角色synthetic全阶段与Resume已运行验证，真实6行也已packing；正式经验训练留给用户命令执行。

审查闭合：超动作域的seed不再裁剪偷换奖励；查询重试用独立attempt；数据RunId/config/code/source根协议冻结；每个训练/预测阶段绑定packed路径与seal，避免增采样后复用不同scaler旧专家。对应回归均已验证。当前文档README、RETRAINING_EXECUTION_CN及SMALL_LOOP_RESULTS_CN为交付入口。

南京：已完成原始100k交通修复、EIA24小时回放和signed V2G空间注入重核验。19点中18个AC有效、1个未知；原始Pmin和容量不绕过。新主项目的南京MPC/四风险迁移仍是后续规模实验，未标记为已完成。
