# Interim Sample Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task.

**Goal:** 暂停原生采集，复用完整样本跑通可追溯的中期后果与采集诊断，并记录正式完成后的容量对照契约。

**Architecture:** 新src/tarp_interim读取冻结tarp_sequence接口，不编辑旧模块。新数据集保存原角色并固定两个训练来源作开发留出。监测训练器保持既有优化配方，完整执行三种子策略及八方法评价；新报告只报告开发角色。

**Tech Stack:** Python3.11/PyTorch/NumPy/SciPy/Matplotlib；原两个venv与报告Python；PowerShell原生进程挂起/恢复。

**Spec:** docs/superpowers/specs/2026-11-03-interim-capacity-diagnostic-design.md

## Global Constraints

- 原7进程及heartbeat保持暂停，恢复只允许用户明确要求。
- 原src/tarp37、tarp_info、tarp_boundary、tarp_acq_evidence、tarp_sequence均只读。
- 只用76完整情境；两个开发留出source固定s102/s112，原正式confirm不参与。
- seed42/43/44、后果100epoch/patience12、策略120×8episode、查询8/16/24。

## Review Focus

- 进程PID复用不得恢复无关进程；暂停/恢复不能重复改变计数。
- 半写入分支即使存在record，也不能进入数据集。
- 诊断角色不得写回原inventory或冒称原confirm。
- masked缺失标签不得被计为零，train/tune监测必须同口径。
- 额外无梯度监测不得改变梯度预算、随机序列或选择依据。

## Task 1 — 完整情境快照与角色契约

- [x] tests/test_interim.py固定半池排除、source隔离和源文件不修改的失败测试。
- [x] src/tarp_interim/data.py实现snapshot(source_root,out)、保持原文件路径及hash；调用旧pack生成目标与scaler。
- [x] 验证全部record/measurement/complete seal，并统计五组mask和边界支持。

## Task 2 — 同配方训练/验证监测

- [x] 先测试缺失标签排除、成对参考差分和best checkpoint只依tune。
- [x] src/tarp_interim/training.py实现fit_response；复用旧ResponseModel及损失，逐epoch无梯度监测，保存best/last与fit.json。
- [x] 新旧优化配置/梯度步数记录；旧fit_policy、evaluate接口直接复用。

## Task 3 — 排序、闭环与报告

- [x] 固定共享Random历史，测试真实边际排序标签与候选可用mask。
- [x] src/tarp_interim/pipeline.py完成prepare/fit/evaluate/diagnose/report，hash seal与Resume。
- [x] src/tarp_interim/report.py生成训练验证、五组后果、行动增量、信息排序、预算曲线和开发source比较组合图及中文HTML/MD。
- [x] scripts/run_sequence_interim.ps1及scripts/control_sequence_pause.ps1提供隔离运行与精确进程恢复入口。

## Task 4 — 完整执行和核验

- [x] 测试GREEN后冻结诊断源码，串行GPU运行3个后果模型和9个策略。
- [x] 完成八方法同预算比较、无训练Resume、原物理hash及暂停CPU复核。
- [x] 一次独立review；只修真实接口/统计错误，保留旧现场与协议。
- [x] 交付中文结果、训练/验证误差和排序诊断；原流程继续暂停，不自动恢复。
