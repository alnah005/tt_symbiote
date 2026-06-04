---
name: deep-work
description: Iterative plan-then-execute loop. Always begins with a full deep-plan flow, then spawns executor subagent(s) to carry it out. On critical execution failure, generates a new deep-plan that explicitly references the previous plan and the failure report, then re-executes. Continues until execution completes as expected. All plans and execution reports accumulate in ./deep-work/ as deep-plan_0, deep-plan_1, ... Use when the user wants a feature or solution carried through from problem statement to working result with autonomous re-planning on failure.
---

# deep-work — plan → execute → re-plan loop

You are orchestrating the deep-work flow. Every iteration: produce a deep plan, execute it with subagents, and on critical failure generate a new plan that explicitly relates to the previous one. Continue until execution completes as expected.

Use `TaskCreate` to track iterations so the user can see progress.

---

## Phase 0 — Setup

Create the deep-work directory in the current working directory (do **not** delete it if it already exists — accumulate across runs):

```bash
mkdir -p ./deep-work
ls ./deep-work
```

If the directory already contains `deep-plan_*` files from a prior `deep-work` run, ask the user: "I see existing `deep-work/` artifacts. Continue from the highest existing index, or start fresh at `deep-plan_0` (existing files preserved but ignored)?" Wait for their answer.

Initialize `iteration = 0` (or the chosen starting index). Set `max_iterations = 10` (safety cap).

Capture the user's original request to `deep-work` verbatim. Save it as `ORIGINAL_REQUEST` for use across iterations.

---

## Phase 1 — Deep plan (iteration N)

Run the **full deep-plan flow** defined at `.claude/skills/deep-plan/SKILL.md`. Follow every phase of that skill — interview, ask for N planners, planner/evaluator convergence loop, synthesis/verification convergence loop. **Do not skip the interview phase.**

The seed context passed to deep-plan's Phase 1 interviewer changes with iteration:

### Iteration 0 — fresh planning

Seed context = the user's original request to `deep-work`, verbatim. Standard deep-plan flow.

### Iteration > 0 — failure-driven re-planning

Seed context = the following composite block (fill in the placeholders):

> **Original user request:**
> ```
> <ORIGINAL_REQUEST>
> ```
>
> **Context:** Previous iteration(s) of this work have failed. Before asking the user any questions, read:
> - Previous plan: `./deep-work/deep-plan_<N-1>.md`
> - Previous execution report: `./deep-work/deep-plan_<N-1>_execution.md`
> - (Optional) All earlier plans and reports in `./deep-work/` for full history.
>
> **Your job is to plan around what went wrong while preserving what worked.** The interviewer must NOT re-interview the user on the basic framework — that's already established in prior iterations. Ask only **targeted questions** about:
> - What specifically failed in the last execution
> - Why it likely failed (root cause hypotheses)
> - What assumptions in the previous plan turned out to be wrong
> - What should be done differently this iteration
> - What should be explicitly preserved from prior plans
> - Whether the user wants to adjust scope or success criteria given the failure
>
> The new plan must **explicitly reference** the prior plan and failure report — call out which prior steps are kept, which are replaced, and which are added.

### Planner count (N)

deep-plan's Phase 2 asks the user for N (number of planner agents). For iteration 0, ask normally. For iteration > 0, ask again — the user may want a different N given the failure's scope. Do not silently reuse the previous N.

### Copy final plan

When deep-plan completes (Phase 7), it copies its final synthesis to `./deep-plan-final.md` in CWD. **Immediately move/rename** that file to `./deep-work/deep-plan_<N>.md`:

```bash
mv ./deep-plan-final.md ./deep-work/deep-plan_<N>.md
```

The deep-plan scratch directory under `/tmp/deep-plan-*` remains as an audit trail — record its path in the execution report below.

---

## Phase 2 — Execute

Spawn executor subagent(s) of type `general-purpose` to carry out the plan. Number of executors:

- If the plan is naturally sequential, spawn **one** executor.
- If the plan explicitly identifies parallelizable workstreams, spawn one executor per workstream (in a single parallel message).

If unsure, spawn one. Do not over-parallelize.

### Executor prompt (template, fill placeholders)

> You are executor `<i>` for plan iteration `<N>` of a `deep-work` flow. Read `./deep-work/deep-plan_<N>.md` carefully and execute it.
>
> Your scope: `<scope description — "the entire plan" for single executor, or specific workstream for parallel executors>`.
>
> Execution rules:
> - Make **real changes** — edit files, run commands, install dependencies, run tests, etc. — exactly as the plan specifies. Do not simulate or describe.
> - Track progress step-by-step. If you hit an obstacle, attempt reasonable workarounds **within the spirit of the plan**. Do not silently skip steps.
> - If a step is genuinely blocked, document the blocker rather than working around it dishonestly.
> - Do **not** modify files outside the plan's scope.
> - Do **not** invoke other skills or spawn further subagents.
>
> When done, write a detailed execution report to `./deep-work/deep-plan_<N>_execution.md` (append if it already exists from a parallel executor — coordinate via section headers `## Executor <i>`). The report must cover:
> - Each plan step: status (`done` / `partial` / `failed` / `skipped`) and what you actually did (commands run, files changed)
> - Any deviations from the plan and why
> - Any errors encountered and how (or whether) you resolved them
> - The current state of the work (what is built, what is broken)
> - The original deep-plan scratch dir path (if known): `/tmp/deep-plan-*`
>
> **End your report (or your section, if multiple executors) with exactly one of these lines:**
> - `EXECUTION_STATUS: success`
> - `EXECUTION_STATUS: critical_failure`
>
> Choose `success` when every essential step completed and the plan's stated success criteria are met, even if some minor steps required reasonable adaptation.
>
> Choose `critical_failure` when a core step cannot be completed, a key assumption in the plan was wrong, or the final state diverges significantly from the plan's success criteria.

### After executors return

1. Read `./deep-work/deep-plan_<N>_execution.md`.
2. If multiple executors, the overall status is `success` only if **every** executor reported `success`. Any `critical_failure` makes the iteration a critical failure.
3. If an executor's status line is missing or malformed, retry that executor **once** asking for a clear status line. If it fails twice, treat it as `critical_failure`.

---

## Phase 3 — Branch on execution status

- **All executors succeeded** → proceed to Phase 4 (done).
- **Any executor reported `critical_failure`** →
  - Increment `iteration` (so `N` becomes `N+1`).
  - If `iteration < max_iterations`, return to **Phase 1** with the new iteration index.
  - If `iteration >= max_iterations`, stop. Report the cap to the user. Show what's been tried (`./deep-work/`). Ask: "Continue past the cap, abort, or intervene manually?"

---

## Phase 4 — Done

Report to the user **in one concise block**:

- Final plan: `./deep-work/deep-plan_<final_N>.md`
- Final execution report: `./deep-work/deep-plan_<final_N>_execution.md`
- Total iterations: `<final_N + 1>`
- Per-iteration summary: for each failed iteration, one sentence on what went wrong and what the next plan changed.
- Current state of the work (what was built/modified in CWD).
- Audit trail: all `./deep-work/` files plus the `/tmp/deep-plan-*` scratch dirs (one per iteration).

Then ask: **"What would you like to do next?"**

---

## Rules (apply throughout)

- **Always start with a full deep-plan.** Even when the user's request seems simple or the failure seems trivial. The interview + multi-agent convergence is the entire reason this skill exists.
- **Iteration > 0 plans must reference prior plans explicitly.** A re-plan that doesn't acknowledge what came before is a bug — re-do it.
- **File naming is strict and zero-indexed:** `deep-plan_0.md`, `deep-plan_0_execution.md`, `deep-plan_1.md`, `deep-plan_1_execution.md`, ... Underscore, not dash.
- **Do not edit executor reports** as the orchestrator. If a status line is ambiguous, re-invoke the executor for clarification.
- **Do not delete or overwrite** prior iteration files. Each iteration adds; nothing subtracts.
- **`/tmp/deep-plan-*` scratch dirs are kept** as audit trails. Record their paths in the per-iteration execution report.
- **The user's `ORIGINAL_REQUEST` is captured once at Phase 0** and re-used verbatim across all iterations. Do not paraphrase or "improve" it between iterations.
