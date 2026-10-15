# Repository working conventions

## Commit completed work

- The user explicitly requests local commits for project work. After completing a
  task that changes repository content, commit the relevant changes without
  asking for permission again. Read-only tasks do not need empty commits.
- Use coherent commits with specific, relevant subjects that describe the actual
  change, such as `feat: add MATLAB run export` or `fix: reject stale RPM windows`.
- In the commit body, summarize the change, verification performed, and anything
  that still needs testing. Never claim an unexecuted check passed.
- Stage the files belonging to the task. Preserve unrelated work and do not
  discard or silently include another contributor's changes.
- Include source, documentation, configuration, tests, and dependency lockfiles.
  Keep secrets, local environments, caches, generated datasets, and large runtime
  artifacts out of Git according to `.gitignore`.
- Do not push, rewrite history, or amend existing commits unless requested.
- After committing, report the commit hash, a concise change summary, test results,
  and any pending verification. State any remaining working-tree changes.

## Flag verification gaps

- Run checks appropriate to the changed behavior. Reuse valid existing evidence
  for documentation-only changes; avoid unnecessary repeated test runs.
- Fix actionable failures before committing completed work. If a failure cannot
  be resolved, explicitly record the failure and its effect rather than hiding it.
- Keep outstanding project verification in `docs/TESTING.md`, with the affected
  component, required check, and reason it remains pending.
- Distinguish implemented behavior, scaffolded modules, and future capabilities.
  Synthetic performance never establishes real-engine diagnostic validity.
- MATLAB execution and specialized toolbox licenses remain unverified until
  checked in a working MATLAB session. Optional Python dependencies being locked
  does not mean they are installed or their imports have been tested.
