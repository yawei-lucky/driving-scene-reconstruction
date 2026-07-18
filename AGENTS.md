# Repository Agent Guidance

This file applies to the entire `driving-scene-reconstruction` repository.
It defines how agents should work here; it does not replace the project
description, current-state record, or experiment evidence.

## Authority And Document Priority

Use the following order when instructions or project descriptions differ:

1. Current system, developer, and explicit user instructions.
2. This `AGENTS.md` for repository-wide working rules.
3. The root `README.md` for current product scope and `PROJECT_STATE.md` for
   the implemented state, known limitations, and next action.
4. Current stage or component documentation that matches the code being
   changed, such as `docs/stage_h2_reconstruction_renderer.md`.
5. Planning references and historical records.

`docs/codex_next_task_stage_h0.md` is an archived H0 task specification.
The files under `experiments/` are historical experiment records.
`docs/research_and_implementation_plan.md` preserves the initial roadmap.
Their commands, prohibitions, environment observations, and future-tense
wording must not be treated as current instructions when they differ from
`README.md` or `PROJECT_STATE.md`.

Preserve historical records as evidence of what happened. Do not silently
rewrite old results to make them look current; add a clear status note or a new
current document instead.

## Working Principles

- Prefer the smallest reliable solution that satisfies the real current need.
- Avoid speculative abstractions, unnecessary configuration, duplicate entry
  points, and defensive complexity that is not justified by actual risk.
- Expose only inputs that genuinely vary. Keep fixed, well-understood project
  facts in one authoritative place.
- Match validation effort to the risk of the change. State meaningful
  speed-versus-completeness tradeoffs and recommend the lighter adequate path.
- If a task is drifting into brittle automation, excessive polish, or work
  outside the requested outcome, pause and make the tradeoff explicit.

## Environment And Execution Scope

This checkout is normally operated directly on the Linux project machine at:

```text
/home/yawei/driving-scene-reconstruction
```

Before environment-sensitive, GPU-heavy, data-heavy, service, or Git
operations, confirm the relevant context with `pwd`, `hostname`,
`git branch --show-current`, `git status`, and `git remote -v` as appropriate.

When the active shell is already attached to the intended Linux project
machine, run project commands directly. Do not add a Windows, WSL, or SSH relay
because another project's runbook uses one. If remote access is genuinely
required, use non-interactive, fail-fast SSH with timeouts and never leave a
command waiting for a password prompt.

Verify the active Python/Conda environment, GPU visibility, CUDA compatibility,
available disk space, checkpoint path, and output path before expensive work.
Do not infer machine capability from a sandbox-only failure.

## Current Project Scope

Always reread `README.md` and `PROJECT_STATE.md` before choosing the next
research or implementation step.

The current product is a human-drivable simulator built from reconstructed real
driving logs. The current system is geometry-grounded and human-controlled.
Autonomous-driving-agent integration and a general world-model renderer are
later extensions, not current priorities.

At the current stage, prioritize:

- measurable nearby-pose rendering quality;
- geometry, temporal, and driving-relevant consistency;
- explicit warm-up and rendering/display latency measurement;
- logged-trajectory time progression;
- investigation of dynamic-object reconstruction limitations.

Do not expand the scope merely because an archived plan mentions another
dataset, model, or stage. Update `PROJECT_STATE.md` when the implemented state
or agreed next action materially changes.

## Preserve User Work

- Inspect `git status` and the relevant diffs before editing or staging.
- Treat existing modified or untracked files as user-owned unless their origin
  and task scope are clear.
- Do not overwrite, revert, reformat, stage, or commit unrelated changes.
- Stage explicit paths in a mixed worktree; do not default to `git add -A`.
- Never discard work with destructive Git or filesystem commands unless the
  user explicitly authorizes the exact action.
- Keep changes focused. Do not combine unrelated cleanup with the requested
  implementation.

## Research Evidence And Claims

- Do not present research conclusions without a traceable source or experiment.
- For literature, dataset, license, benchmark, API, and tool-behavior claims,
  cite the primary source when available and record the relevant version or
  access date when it can affect reproducibility.
- Clearly distinguish sourced facts, direct observations, measurements,
  inferences, hypotheses, and proposed next steps.
- Never invent citations, metrics, completed runs, visual findings, or
  validation results.
- A failed or partial experiment is still evidence. Record the failure
  condition and preserve it rather than reporting only the intended outcome.
- When reporting a metric, identify the data split, evaluator, configuration,
  checkpoint or commit, and important runtime conditions.

## Experiments, Data, And Artifacts

- Keep raw driving logs, large datasets, checkpoints, rendered images/videos,
  TensorBoard events, caches, and other heavy outputs outside Git.
- Use Git for source, lightweight tests, small fixtures, metadata templates,
  reproducibility instructions, and concise experiment records.
- Never commit private customer data, credentials, tokens, proxy details, or
  machine secrets.
- Treat dataset terms separately from code licenses. WayveScenes101 data is
  restricted to non-commercial research use; do not imply broader rights.
- Existing Stage H1 artifacts are stored outside the repository under
  `/home/yawei/stage1_external`. Confirm space and provenance before creating,
  moving, or replacing large artifacts.
- Prefer predictable output directories and record important artifact paths,
  configs, commands, environment versions, and checksums or sizes when useful.

## Implementation And Validation

- Preserve the existing lightweight simulator interfaces and dependency-light
  tests unless the current task justifies a broader change.
- Use the narrowest relevant validation first, then run broader tests in
  proportion to impact. The baseline lightweight suite is:

```bash
python3 -m unittest discover -s tests -v
```

- For documentation-only changes, inspect the rendered structure where useful
  and run `git diff --check`.
- Do not claim GPU, browser, visual, latency, or end-to-end validation unless
  that exact path was exercised. State what was not tested.
- Startup and long-running scripts should offer one clear command, validate
  prerequisites early, expose important environment and role information, and
  make normal operation and failures visible in logs.

## Commit And Push Default

After completing an authorized change and its appropriate validation, commit
and push the task's changes to the current project branch by default unless the
user explicitly says not to commit or not to push.

Before committing:

- verify the diff and validation result;
- stage only files belonging to the task;
- use a concise commit message describing the outcome;
- confirm the remote and branch when there is any ambiguity.

After committing, push the corresponding branch directly. If authentication,
permissions, remote divergence, branch protection, or an unclear target makes
the push unsafe, stop and report the exact blocker without rewriting history or
discarding work. Do not create a pull request unless the user requests one or
the repository workflow explicitly requires it.

The final report should include the changed files, validation performed,
commit, branch, push result, and whether unrelated or uncommitted work remains.
