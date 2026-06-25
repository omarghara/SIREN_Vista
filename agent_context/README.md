# agent_context — start here

This folder is the onboarding pack for a **new agent** (or a returning one) working
on this repo. Read these files in order before doing anything else. They tell you
what the project is, where it stands, what we have already learned (including dead
ends), the current goal, and how to run things.

## Reading order
1. `01_project_and_pipeline.md` — what the project is, the bilevel pipeline, repo map.
2. `02_current_status.md` — latest results, current checkpoints, what is local-only.
3. `03_key_learnings_and_math.md` — what works, what does NOT, and why (the math).
4. `04_current_goal_and_next_steps.md` — the immediate objective and how we judge success.
5. `05_how_to_run.md` — environment, scripts, commands for train/makeset/classifier/PGD/eval.

## The one-line current goal (2026-06)
**Improve robustness of the CIFAR-10 parameter-space classifier to PGD attacks.**

## The one rule that overrides everything
**Do NOT claim robustness from PGD alone.** A result only counts if it *also* holds
up under stronger / adaptive attacks (e.g. more PGD restarts+steps, AutoAttack,
transfer attacks, attacks that reason about the inner fitting loop). PGD is a
*screening* tool here, not proof. See `04_current_goal_and_next_steps.md`.

## Deeper background (older but still useful)
The longer-form notes live in `../context/`:
- `cursor_context.md` — full thesis framing, papers, hypotheses, evaluation rules.
- `siren_modulation_robustness_findings.md` — detailed empirical/math findings.
- `attack_currenct_results.md` — historical PGD result tables.
- `cifar10_spatial_functa_status.md` — CIFAR status detail.
- `hard_svd_projection_plan.md` — the hard SVD projection design + math.

`agent_context/` is the curated summary; `context/` is the full archive.
