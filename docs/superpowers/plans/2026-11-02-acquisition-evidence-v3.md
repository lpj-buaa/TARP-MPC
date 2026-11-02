# 零新仿真实施计划

规格：docs/superpowers/specs/2026-11-02-acquisition-evidence-v3.md。用户已授权连续执行。原linked worktree及新包/运行目录隔离，原代码只读。

## Task 1 数据与泄漏边界
先写失败测试，再实现data.py/features.py。核对空间通道、固定60维目标、同状态/去自身经验、已查定量敏感性及输入hash。缓存预处理不导入物理环境或LP。接口输出public/initial/private answers，状态823、动作51、经验维数固定。

## Task 2 等结构学习与查询
先写失败测试，再实现model.py/experiment.py。双128层critic，稳定头仅依赖状态动作；四种配置参数数目一致；匹配历史和主动查询、局部插值均严格只读公开/已查信息。warm100＋每查询10，三种子。保存64快照、完整预测/真值行键、查询路径、模型与训练计数。

## Task 3 运行恢复与正式复用
先写失败测试，再实现pipeline.py/CLI及脚本，输入/代码/输出hash，独立RunId、Resume跳过已完成单元。先小预算集成，再24个采集模型串行CPU正式复用，保留所有种子。旧模型训练和仿真计数为零。

## Task 4 source统计与呈现
匹配64排序为主要比较；主动边界/画像覆盖为另一个层面。按source配对计算四格增量，补充稳定后果误差与推导值诊断。生成2×2组合图、中文报告及完整命令。独立whole-branch review，重要发现RED→GREEN修复，验证Resume和冻结hash后提交。

## Review focus
未来泄漏(尤其初始参考未来与候选未来)、自己标签进入经验、剂量插值是否隐藏调用LP、同历史/同候选比较、source统计分母、相同模型参数、64checkpoint时点、Resume输入输出哈希、仿真与旧训练严格为零。
