# Sequential Acquisition V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task. User approved direct execution without further permission gates.

**Goal:** Complete expanded physical bank and real multi-step acquisition comparison.
**Architecture:** Frozen V5 physical executor; standalone data/response model; set-memory acquisition MDP; matched PPO/myopic/group policies and strong baselines.
**Tech Stack:** Existing Python runtimes, numpy/scipy, torch, pandapower/V2SIM; no new dependencies.
**Spec:** docs/superpowers/specs/2026-11-02-sequential-acquisition-v4.md

## Global Constraints
- Old source packages/physical capacities/thresholds/controllers/models read-only.
- 112 contexts,32branches/context; reuse162; source roles retained.
- Six-hour branch labels from actual native continuation and existing MPC LP measurement.
- Only train/meta_train data update parameters; source-level statistics.
- CPU4 physical workers, one GPU training stage at a time; hash-bound Resume.

## Review Focus
- A changed preference must be applied before computing context/continuation, without modifying first intervention.
- Cache reuse must bind physical action, initial state/time/preference and raw/measured hashes.
- Unknown responses must not create zero-gap or false safe labels.
- The actor must see queried history only; training reward oracle never enters evaluation inputs.
- PPO must actually use future returns; group trajectory costs counted; confirm not used to choose checkpoints.

## Task 1 Physical bank
- [ ] Write tests for32 unique/finite actions, context identity and source roles; observe failure.
- [ ] Implement corpus.py/context workers with162 reuse and branch sealing. Test real small context.
- [ ] Run full physical stage inbackground4CPU while coding ML; oldhashes sealed. No duplicate launch.

## Task 2 MDP and stable targets
- [ ] Tests forreward telescoping, budget/repeat masks, unknown nofalse labels, train-onlyscaler; RED→GREEN.
- [ ] Implement data.py/environment.py using original60target definitions and new32pool.

## Task 3 Models and optimizers
- [ ] Tests set invariance, feedback-dependentfeatures, multistepGAE, identical architectures; RED→GREEN.
- [ ] Implement response model, set attention actor/value, PPO/myopic/trajectorygroup updates and baselines. Run cache-only smoke.

## Task 4 Pipeline and evaluation
- [ ] Implement stageCLI/PowerShell orchestration withaudit, collect, pack, fit-response, fit-policy, evaluate, report, Resume manifests.
- [ ] Formal three seeds, all eight methods and three budgets on heldoutsource.
- [ ] source-pairedCI and figures; independentreview; fiximportantfindings withRED→GREEN; frozenhash/Resumeverification; commit anddeliver actualresults.
