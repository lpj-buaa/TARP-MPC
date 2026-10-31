# Physical risk and corrective execution audit

> Execute inline under systematic-debugging, test-driven-development and verification-before-completion. The user's request authorizes the audit and necessary corrections without additional stage approvals.

**Goal:** Make the four risk labels describe explicit constraint/commitment events and connect the complete routine plan, corrective controls and verified realized consequences.

**Spec:** Current user decision: voltage event uses the explicit 0.93–1.07 pu boundary; voltage quality and spatial differences remain explanatory quantities. Preserve original topology, impedance, generation capability, 0.9 kA line rating, existing RL and expert bank.

**Isolation:** `tarp_37_risk_audit`, branch `codex/physical-risk-audit`, base `62ec7ae`. Native worktree creation rejected the nested repository root; use a sibling Git worktree so existing read-only asset paths remain valid. Original outputs stay unchanged. Copy only the existing hourly data cache into the new ignored artifacts directory.

## Task 1 — Align the same operating point

- [ ] Add failing tests showing that routine reactive dispatch, source/bus renewable output, backup and planned base-load shedding survive the pre-correction AC calculation.
- [ ] Publish all those planned controls from MPC. Solve diagnostic AC with fixed local P/Q, bounded renewable output and slack-root balancing. Root limit excess is measured, not executed.
- [ ] Separate line loading, PCC import and export-permission excess. Do not interpret Pmin=0 as a thermal rating.
- [ ] Verify reconstruction of an executed dispatch as a diagnostic operating point, plus root violations and normal voltage deviations.

## Task 2 — Connect controls and event semantics

- [ ] Add failing tests for complete-plan propagation, explicit guard-limiting attribution, correction logs, consistent source-wise RE ratios and the shared risk thresholds.
- [ ] Retain causal allocation protection before native delivery. Publish request/routine-plan diagnostic, realized-delivery/routine-plan diagnostic, corrections and actual AC result separately. Never change delivered EV energy retrospectively.
- [ ] Log redispatch, Q adjustment, extra curtailment/backup/base shedding relative to the routine plan. Record numerical solver fallback separately; do not claim it proves unavoidable physical shedding.
- [ ] Version changed physics/labels; refuse to mix old labels into new training. Service remains observed recovery-accessibility deficit or explicit 30-minute start-of-charge SLA, not an unimplemented universal departure promise. RE 10% remains a six-hour operating budget, not a statutory six-hour failure limit.

## Task 3 — Verify and document

- [ ] Run the whole test suite in the existing GPU runtime with read-only physical dependencies.
- [ ] Run an isolated small native closed loop, inspect stage consistency, physical bounds and label support.
- [ ] Write Chinese audit results and the four-risk boundary/action/consequence table with primary sources. Supply repeatable commands; do not retrain RL/experts in this audit.

## Review focus

Actual renewable availability below the planned dispatch; zero/small renewable denominators; unverified pre-correction power flow with known service observations; numerical OPF failure before a feasible shedding trial; identical boundary values and numerical tolerances in dispatch, logs and labels.

## Progress

Initial baseline attempt: 32 passed / 8 failed before copying the ignored data cache, and using the physical-only Python without sklearn. Correct runtime and data cache are now supplied; rerunning the unchanged baseline before implementation.
