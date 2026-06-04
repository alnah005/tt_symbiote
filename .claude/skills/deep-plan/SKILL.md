---
name: deep-plan
description: Multi-agent iterative deep planning flow. An interviewer subagent exhaustively probes the user until the problem framework is fully understood, then N planner/evaluator pairs converge on a plan via repeated amendment rounds, then synthesizer/verifier agents iterate on a master comprehensive plan. All intermediate artifacts dump to a unique /tmp directory; the final plan is copied to CWD. Use when the user wants a thoroughly-vetted plan produced by an ensemble of agents rather than a single pass.
---

# deep-plan — multi-agent iterative deep planning

You are orchestrating a multi-phase iterative planning flow. Follow every phase **in order**. Do not skip phases. Do not summarize prematurely. Do not collapse loops.

The user's request that triggered this skill is the **only seed context** for the problem. Treat it as the topic to be planned around — not as a finished spec.

Use `TaskCreate` to track the phases below so the user can see progress.

---

## Phase 0 — Set up scratch directory

Create a unique working directory and remember its path for the entire flow:

```bash
WORKDIR=/tmp/deep-plan-$(date +%Y%m%d-%H%M%S)
mkdir -p "$WORKDIR"/{interview,plans,evaluations,synthesis,verification}
echo "$WORKDIR"
```

Store `$WORKDIR` (the actual absolute path that was created) — every artifact in this flow lives under it. Reference it as an absolute path in all subsequent tool calls.

---

## Phase 1 — Interview (fresh-context interviewer, exhaustive Q&A)

Spawn **one** `general-purpose` subagent as the interviewer. Its sole job is to ask the user questions until it fully understands the problem framework. No question is silly — encourage breadth.

**Interviewer spawn prompt** (pass verbatim, filling in `<USER_REQUEST>`):

> You are an interviewer with **no prior context** about the user's situation, codebase, or goals. Your only job is to fully understand the user's problem before any planning begins. The user's initial request is:
>
> ```
> <USER_REQUEST>
> ```
>
> Ask as many questions as needed — broad, narrow, weird, obvious, edge-case. **No question is silly.** Cover at minimum: goals, motivation, success criteria, constraints (time, budget, tech, people), stakeholders, deliverable format, edge cases, dependencies, prior attempts, risks, what "done" looks like, and anything domain-specific you'd want to know.
>
> Output **one batch of 5–10 questions at a time** as a numbered list. After each batch you will receive the user's answers from the orchestrator. Decide whether you have a complete picture. If not, ask another batch — keep going until you genuinely cannot think of another useful question.
>
> When (and only when) you have a complete understanding, output exactly:
>
> ```
> INTERVIEW COMPLETE
> ```
>
> followed by a **structured summary** of the problem framework covering every dimension you asked about. This summary is the handoff to the planning phase — it must stand alone.

**Orchestration loop:**

1. Spawn the interviewer agent with the prompt above.
2. The interviewer returns a batch of questions. Display them to the user in the main conversation (numbered exactly as the interviewer wrote them) and ask the user to answer. Do **not** filter, condense, or reorder questions.
3. Collect the user's answers as a single block of text.
4. Continue the interviewer subagent with the answers (via `SendMessage` to the same agent if available; otherwise spawn a fresh `general-purpose` agent passing the **full Q&A history so far** plus the original interviewer prompt).
5. Repeat steps 2–4 until the interviewer emits `INTERVIEW COMPLETE` followed by its structured summary.
6. Write the full transcript (every question, every answer, plus the final summary) to `$WORKDIR/interview/transcript.md`.
7. Write the structured summary alone to `$WORKDIR/interview/framework.md` — this is the canonical handoff document for all subsequent phases.

**Do not move on until the interviewer has explicitly emitted `INTERVIEW COMPLETE`.** If it tries to stop early, send back: "Are you sure you have no more questions? Probe deeper."

---

## Phase 2 — Ask the user for N

Ask the user directly in the main conversation:

> How many planner agents (N) should I spawn?

Wait for their answer. Parse the integer. Save as `N`. Sanity-check: if N < 1 or N > 20, ask again.

---

## Phase 3 — Initial plans (N planners, in parallel)

Spawn N `general-purpose` subagents **in a single message** (parallel Agent tool calls). Each planner gets index `i ∈ {1..N}` and the framework document.

**Planner spawn prompt** (for each, with `<i>` filled in):

> You are planner #<i> of N. Read the problem framework at `$WORKDIR/interview/framework.md`. Produce a **complete, thorough plan** to solve this problem. Cover: objectives, step-by-step actions, dependencies, risks and mitigations, success criteria, deliverables, timeline (if relevant), open questions, and assumptions you're making. Be exhaustive — no missing details. Take your own perspective; do not try to be average.
>
> Write your plan to `$WORKDIR/plans/round-1/planner-<i>.md`. When done, return a one-line confirmation with the file path.

After all N return, verify all N files exist before proceeding.

---

## Phase 4 — Planner/evaluator convergence loop

Initialize: `round = 1`, `max_rounds = 10` (safety cap).

**Loop:**

1. **Evaluators (N in parallel).** Spawn N `general-purpose` subagents in a single message. Evaluator `i` reads planner `i`'s current plan.

   Prompt for each:

   > You are evaluator #<i>. Read the plan at `$WORKDIR/plans/round-<round>/planner-<i>.md` and the problem framework at `$WORKDIR/interview/framework.md`. **Critically evaluate** the plan: identify gaps, errors, vague steps, missing dependencies, unaddressed risks, internal contradictions, unrealistic assumptions, and any way the plan fails to fully solve the problem. Be rigorous and specific — cite line numbers or sections.
   >
   > Write your evaluation to `$WORKDIR/evaluations/round-<round>/evaluator-<i>.md`.
   >
   > **End your evaluation with exactly one of these two lines** (and nothing after it):
   > - `AMENDMENTS_NEEDED: yes` — if the plan needs any changes at all
   > - `AMENDMENTS_NEEDED: no` — only if the plan is genuinely complete, sound, and needs no further changes
   >
   > Set the bar high. Choose `no` only when you'd stake your reputation on the plan.

2. **Check convergence.** Read the last line of each evaluation file. If **all N** say `AMENDMENTS_NEEDED: no` → exit loop and proceed to Phase 5.

3. **Planners amend (N in parallel).** Increment `round`. Spawn N `general-purpose` planner subagents in a single message. Planner `i` receives its previous plan AND its evaluator's feedback.

   Prompt for each:

   > You are planner #<i>, round <round>. Read your previous plan at `$WORKDIR/plans/round-<round-1>/planner-<i>.md` and your evaluator's feedback at `$WORKDIR/evaluations/round-<round-1>/evaluator-<i>.md`. Also re-read the framework at `$WORKDIR/interview/framework.md`.
   >
   > Produce an **amended plan** that addresses every concern the evaluator raised. Do not merely respond to the feedback — rewrite the plan as a standalone document that incorporates the improvements. Write to `$WORKDIR/plans/round-<round>/planner-<i>.md`.

4. **Safety cap.** If `round >= max_rounds`, exit loop and warn the user that convergence wasn't reached within `max_rounds` rounds — but proceed to Phase 5 anyway.

5. Go to step 1.

---

## Phase 5 — Synthesis (first pass)

Spawn **one** `general-purpose` synthesizer subagent.

Prompt:

> You are the synthesizer. Read **every** plan under `$WORKDIR/plans/` (all rounds, all planners) and **every** evaluation under `$WORKDIR/evaluations/`. Also read the framework at `$WORKDIR/interview/framework.md`.
>
> Produce a **single master comprehensive plan** that:
> - Incorporates the strongest elements from every planner's output across all rounds
> - Addresses every valid concern surfaced in any evaluation
> - Resolves contradictions between competing plans with explicit reasoning
> - Has **no missing details** — anyone reading it should be able to execute without further questions
>
> Structure it for clarity: objectives, prerequisites, step-by-step plan, dependencies, risks/mitigations, success criteria, deliverables, open items.
>
> Write to `$WORKDIR/synthesis/iter-1.md`.

---

## Phase 6 — Synthesis/verification convergence loop

Initialize: `iter = 1`, `max_iters = 10`.

**Loop:**

1. **Verifier.** Spawn one `general-purpose` verifier subagent.

   Prompt:

   > You are the verifier. Read the current synthesized plan at `$WORKDIR/synthesis/iter-<iter>.md`. Cross-reference against every plan under `$WORKDIR/plans/`, every evaluation under `$WORKDIR/evaluations/`, and the framework at `$WORKDIR/interview/framework.md`.
   >
   > Identify: missing details, contradictions, unaddressed concerns, weaknesses, gaps relative to the source material, anything the synthesis dropped that shouldn't have been dropped, anything the synthesis kept that shouldn't have been kept.
   >
   > Write your verification report to `$WORKDIR/verification/iter-<iter>.md`.
   >
   > **End with exactly one of:**
   > - `AMENDMENTS_NEEDED: yes`
   > - `AMENDMENTS_NEEDED: no`
   >
   > Choose `no` only if the synthesized plan is genuinely complete, internally consistent, and faithful to the source material.

2. **Check convergence.** Read the last line of the verification file. If `AMENDMENTS_NEEDED: no` → exit loop and proceed to Phase 7.

3. **Synthesizer amends.** Increment `iter`. Spawn one synthesizer subagent.

   Prompt:

   > You are the synthesizer, iteration <iter>. Read the previous synthesis at `$WORKDIR/synthesis/iter-<iter-1>.md` and the verifier's feedback at `$WORKDIR/verification/iter-<iter-1>.md`. Also re-read source plans and evaluations under `$WORKDIR/plans/` and `$WORKDIR/evaluations/` as needed.
   >
   > Produce an **amended master plan** that addresses every concern the verifier raised. Rewrite as a complete standalone document. Write to `$WORKDIR/synthesis/iter-<iter>.md`.

4. **Safety cap.** If `iter >= max_iters`, exit and warn the user — proceed to Phase 7 anyway.

5. Go to step 1.

---

## Phase 7 — Final output and handoff

1. Identify the final synthesis file: `$WORKDIR/synthesis/iter-<final-iter>.md`.
2. Copy it to the current working directory:
   ```bash
   cp "$WORKDIR/synthesis/iter-<final-iter>.md" "$(pwd)/deep-plan-final.md"
   ```
3. Report to the user **in one concise block**:
   - Final plan path (in CWD): `./deep-plan-final.md`
   - Scratch dir with full audit trail: `$WORKDIR`
   - Stats: N planners, planner/evaluator rounds taken, synthesizer/verifier iterations taken, total subagents spawned
   - Whether either convergence loop hit its safety cap
4. Ask the user: **"What would you like to do next?"**

---

## Orchestration rules (apply throughout)

- **Parallelism is mandatory** where specified. Planner and evaluator rounds spawn all N subagents in a **single message** with N concurrent `Agent` tool calls. Sequential spawning is a bug.
- **Verify file existence** after every phase that produces files, before reading or proceeding.
- **Do not summarize subagent outputs into your own context** unless a downstream phase needs the content directly. Read files on demand from `$WORKDIR`.
- **Convergence signals are literal.** Match the last line of evaluator/verifier outputs exactly against `AMENDMENTS_NEEDED: no`. Anything else means continue.
- **Retry once on subagent failure.** If a subagent errors or returns no output, retry with the same prompt. If it fails twice, surface the error to the user and ask how to proceed.
- **Never edit subagent output files** as the orchestrator. Subagents own their files.
- **Do not skip the interview** even if the user's initial request seems detailed. The interviewer often surfaces gaps the user didn't realize were gaps.
