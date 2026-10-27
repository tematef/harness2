# AGENTS.md

Project rules for AGY / Jetski agents working in this repository.

## Purpose of this fork

`tematef/harness2` is a fork of `google-research/harness2` (Harness²), adapted to drive a
**local AGY (Jetski) CLI** as the agent executor. The adaptation adds:

- `harness2/substrate/agy.py` — the `agy` substrate (`--substrate agy`): headless
  `-p … --output-format stream-json` runs, stream decoder, throwaway-`HOME` staging,
  side-effect shims, model guard, preflight.
- `harness2/benchmarks/skillbench.py` + `harness2/benchmarks/skillbench/tasks/` — skillbench,
  a leave-one-out A/B benchmark of the `cc-thingz` skills (arm `with` = full allowlisted plugin
  suite, arm `without` = same suite minus `skills/<target>/`).
- `harness2/cli/skillbench_report.py` (`skillbench-report`) — paired lift, two-level bootstrap
  CI, verdicts, time-to-skill, `verify` pass rate, diff/token medians;
  `harness2/cli/skillbench_html.py` renders `report.html`.
- `scripts/run_skillbench.sh` (full chain), `scripts/agy_spike.sh` (records real streams).
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
- Keep synthetic fixtures in the same shape as the real streams; when Jetski output changes,
  re-run the spike, update `tests/fixtures/agy/real/`, the decoder, and
  `docs/skillbench-spike.md` together.
- After editing any task under `skillbench/tasks/`, re-freeze the manifest
  (`python -m harness2.benchmarks.skillbench freeze --force`); runs refuse to start on drift.
- Never stage the user's real `~/.gemini/config` or touch the `cc-thingz` checkout; the plugin
  source is read via `SKILLBENCH_PLUGINS_DIR`.
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
