# AGENTS.md

Project rules for AGY / Jetski agents working in this repository.

## Purpose of this fork

`tematef/harness2` is a fork of `google-research/harness2` (Harness²), adapted to drive a
**local AGY (Jetski) CLI** as the agent executor. The adaptation adds:

- `harness2/substrate/agy.py` — the `agy` substrate (`--substrate agy`): headless
  `-p … --output-format stream-json` runs, stream decoder, throwaway-`HOME` staging for both
  arm kinds (`skill`: `with` / `without`; `suite`: `baseline` / `suite-<name>`), side-effect
  shims, model guard, preflight.
- `harness2/benchmarks/skillbench_suite.py` + `harness2/benchmarks/skillbench/suites/cc-thingz.toml`
  — `SuiteConfig` (`suite.toml` loader, `suite_hash`, `redaction_names`, `skill_asks_user`,
  cross-family `[matrix.judge_for]` validation, `active_suite()`).
- `harness2/benchmarks/skillbench.py` + `harness2/cli/skillbench_tasks.py` (`skillbench-tasks`)
  — generic skill and plugin-suite benchmark (`kind = "skill" | "suite"`), external corpus
  resolution (`--tasks` → `suite.tasks_root` → `SKILLBENCH_TASKS_ROOT` → bundled corpus),
  task scaffolder (`skillbench-tasks new`) and validator (`skillbench-tasks check`).
- `harness2/cli/skillbench_matrix.py` (`skillbench-matrix`) — model × arm matrix runner with
  cross-family judge pairing, pre-run budget estimation (`estimate_budget`), `--max-rollouts`
  and TTY confirmation (`--yes`), in-process sweep + judge, cell-level resume, and `matrix.json`.
- `harness2/cli/skillbench_report.py` (`skillbench-report`), `harness2/cli/skillbench_summary.py`,
  and `harness2/cli/skillbench_html.py` — schema-v2 `report.json` (`schema_version: 2`,
  `suite: {name, hash}`, `models[]`, `summary[]`), per-executor task hardness (`informative`,
  `ceiling`, `unclassified`, `--hardness-only`), 4-level recommendation rule
  (`needs harder tasks` > `consider removing` > `fix` > `keep`), one-page `summary.md` + HTML
  Summary tab (`--summary-only`), and `history.jsonl` trend sparklines (`history append|show`).
- `harness2/cli/skillbench_improve.py` (`skillbench-improve`) — targeted Propose–Probe–Judge
  loop for weak skills: contrastive `WITH` vs `WITHOUT` evidence (no rubric leak unless
  `--include-verdicts`), candidate validation (`banned_tools`, `--max-growth`), same-model
  probing enforcement, multi-model gate (`--models`), multi-round refinement (`--rounds`),
  3-way report (`improve_report.{json,md,html}`), and `best.patch` (`--apply` onto a branch).
- `harness2/cli/skillbench_cycle.py` (`skillbench-cycle`) — end-to-end orchestrator
  (`matrix → report → history append → improve`) with `--from`, `--skip`, and `cycle.log`.
- `scripts/run_skillbench.sh` (single-model helper chain), `scripts/agy_spike.sh` (records real streams).
- Docs: `docs/BENCHMARKS.md#skillbench`, `docs/SETUP.md#agy-executor`, `docs/skillbench-spike.md`.

Upstream benchmarks (JobBench, LAB, WorkBuddy) and the OpenCode/Codex substrates are unchanged;
`--substrate agy` is rejected for them. skillbench runs bases only — Propose–Probe–Compose is
untouched.

## Local AGY = Jetski CLI

- The executable is `/google/bin/releases/jetski-devs/tools/cli`. `jet` in the user's shell is a
  bash **function** wrapping it, so subprocesses cannot call `jet` — always set
  `HARNESS2_AGY_BIN` to the full path.
- Model IDs embed the effort level: `opus-5.5-high` (executor), `fable-5.1-high` (judge). List
  them with `<cli> models`. Plain `opus` / `fable` do not exist. `SKILLBENCH_EFFORT` is optional.
- Internal Jetski authenticates without credential files; `SKILLBENCH_AUTH_FILES` is optional.
- Plugins are discovered only from `$HOME/.gemini/config/plugins/` of the (throwaway) `HOME`.

## Verified stream-json facts (from `tests/fixtures/agy/real/`)

- First event: `{"event":"init","init":{"model":…,"cwd":…,"tools":[…]}}` — the only place the
  model name appears; the model guard reads `init.model`.
- Skill reads: `step_type:"tool"`, `tool_name:"read_file"`, `tool_info.parameters.Path` ending in
  `skills/<skill>/SKILL.md`. Writes use `write_file` + `Path`.
- Final answer: `result.response` (the last `DONE` text delta is often just `"\n"`).
- `--json-schema` → `result.structured_output` as a JSON object.
- In print mode the models do not call `ask_question`; interactive skills get answers from the
  simulated-user preamble (`user_answers` in `task.toml`), identical in both arms.
- Rollout lines get a `t` timestamp from `_pump`; spike files do not.

## Rules

- **No paid runs without explicit user approval.** Unit/e2e tests use the stdlib stub
  `tests/fixtures/agy_stub.py` (`HARNESS2_AGY_BIN=<stub>`) and fixtures in
  `tests/fixtures/agy/`. Re-running the spike, the pilot (`PILOT=1 K=3`) or the full corpus
  costs model quota — ask first.
- **Suite configuration (`suite.toml`) and no suite hard-coding.** Anything suite-specific
  (`name`, `plugins_dir`, `plugins`, `tasks_root`, `banned_tools`, `[families]`,
  `[matrix].executors`, `[matrix.judge_for]`) belongs in `suite.toml` (`--suite` /
  `SKILLBENCH_SUITE`) or in the task corpus — never hard-code suite or skill names (such as
  `cc-thingz`, `make-plan`, `root-cause`, `brainstorm`) in Python runtime logic under
  `harness2/`. Judge redaction names and "asks the user" (`ask_question` in frontmatter
  `allowed-tools`) are derived dynamically from the staged plugins.
- **Arm kinds and `report.json` schema v2.** Support both arm kinds: `skill` (`with` vs
  `without`, leave-one-out of `target_skill`) and `suite` (`suite-<name>` vs `baseline` = zero
  plugins, plus optional `--suites` competitors on `family` tasks). `report.json` uses
  `schema_version: 2` (`suite: {name, hash}`, `models[]`, `summary[]`) while mirroring the
  first `models[]` entry at top level so v1 consumers keep working.
- Keep synthetic fixtures in the same shape as the real streams; when Jetski output changes,
  re-run the spike, update `tests/fixtures/agy/real/`, the decoder, and
  `docs/skillbench-spike.md` together.
- After editing any task under the task corpus (`tasks_root`), validate with
  `skillbench-tasks check` and re-freeze the manifest
  (`python -m harness2.benchmarks.skillbench freeze --force`); runs refuse to start on drift.
- Never stage the user's real `~/.gemini/config`. Staging only reads the checkout; only
  `--apply` writes, on a branch, never on a dirty tree (`skillbench-improve --apply` resolves
  `plugins_dir` symlinks, verifies a clean git status, commits onto
  `skillbench/improve-<skill>-<date>`, and never pushes).
- Keep `README.md` (fork section) and `docs/` in sync with skillbench / agy changes.

## Testing

```bash
.venv/bin/python -m unittest discover -s tests     # all suites (stub-based, no model calls)
.venv/bin/ruff check .
bash scripts/agy_spike.sh --dry-run                # prints spike probes, calls nothing
```

## Git remotes

- `origin` → `google-research/harness2` (upstream, read-only for us).
- `fork` → `tematef/harness2` (push target). Work branch: `skillbench`.
