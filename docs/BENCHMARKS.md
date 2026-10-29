# Benchmark guides

Complete the [installation and credentials](SETUP.md) first. Each
benchmark installs separately; install the ones you intend to run. Every command
below takes `--substrate cx` to use the
[Codex executor](SETUP.md#codex-executor) instead of OpenCode. Choose the
recursion mode with `--mode parallel` or `--mode sequential` and the task-level
recursion budget with `--k`. Keep both identical across the run, judge, and
report commands.

## LAB

Upstream: [harveyai/harvey-labs](https://github.com/harveyai/harvey-labs).
Legal work tasks with source documents and rubric-based grading. The adapter
groups tasks by practice area and removes criteria from the material passed to
the improver. Needs Pandoc and Poppler.

```bash
harness2-setup lab
harness2-check --bench lab
harness2 --bench lab --k 1 --mode parallel
harness2-judge --bench lab --k 1 --mode parallel
harness2-report --bench lab --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`.

Tasks arrive with the pinned checkout at
`dependencies/harvey-labs/tasks/<practice-area>/<task>/task.json`, and the adapter
discovers all installed tasks; no separate task list is required. Setup installs
the LAB dependency environment, the runner overlays, and the included Vertex
judge patch.

The judge reads outputs only after selection. Reports use Partial (fraction of
criteria passed) and All-pass (all criteria passed). A failed judge remains
unscored and writes its output to `judge.log` in the run directory; a Vertex
`429 RESOURCE_EXHAUSTED` there means quota, so rerun `harness2-judge` later.
Changing the judge model changes the measurement. Outputs are under
`runs/lab/`.

## JobBench

Upstream: [Job-Bench/job-bench-eval](https://github.com/Job-Bench/job-bench-eval).
Data: [JobBench/job-bench](https://huggingface.co/datasets/JobBench/job-bench).
Professional tasks, with the library grouped by the dataset's occupation
directories.

```bash
harness2-setup jb
harness2-check --bench jb
harness2 --bench jb --k 1 --mode parallel
harness2-judge --bench jb --k 1 --mode parallel
harness2-report --bench jb --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`.

Setup downloads only the main dataset from Hugging Face — public and ungated, so
no account, token, or terms acceptance is required — records its revision, and
places tasks at `dependencies/job-bench-eval/dataset/main/`. It also installs the
document extraction dependencies and a judge patch that handles reasoning effort
and empty completions. Rubrics stay outside the improver's evidence.

With a Gemini `HARNESS2_JUDGE_MODEL`, the judge uses the project's Vertex
OpenAI-compatible endpoint and Application Default Credentials. For a separate
judge provider, set `HARNESS2_JUDGE_BASE_URL`, `HARNESS2_JUDGE_API_KEY`, and the
model ID that endpoint accepts. The reported score uses the upstream rubric
weights; unscored tasks are reported separately. Results are under `runs/jb/`.

Note: JobBench data, including `RUBRICS.json`, is public on Hugging Face, so an
executor with web access may retrieve it. Block those hosts during runs as
described under [Web access](EXPERIMENTS.md#web-access).

## WorkBuddy-Bench

Upstream: [Tencent/workbuddy-bench](https://github.com/Tencent/workbuddy-bench).
Data: [tencent/workbuddy-bench](https://huggingface.co/datasets/tencent/workbuddy-bench).
Office, web, and code subsets. Needs Docker.

The upstream custom Tencent license includes a territorial limitation; read
[the license](https://github.com/Tencent/workbuddy-bench/blob/main/LICENSE)
before downloading or using the benchmark. Derived runner files and patches
retain that license.

```bash
docker info
harness2-setup wb --subset office
harness2-check --bench wb --subset office
harness2 --bench wb --subset office --k 1 --mode parallel
harness2-judge --bench wb --subset office --k 1 --mode parallel
harness2-report --bench wb --subset office --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`.

`harness2-setup wb` without `--subset` installs all three subsets in one run;
pass `--subset` to install only the one you intend to run. Every other command
takes one subset at a time.

Setup installs the pinned Harbor-based runtime, applies the included adapter,
mount, and credential/transport patches, copies the harness runners, creates
solver and judge routes from `.env`, downloads the requested subset with the
upstream fetch script, and builds the OpenCode harness image. The public Vertex
setup uses a `google-vertex/gemini...` executor and a bare `gemini...` judge ID;
no model YAML or account-specific proxy configuration needs to be copied from
another checkout. Setup refuses to run unless `HARNESS2_SOLVER_MODEL` is a
`google-vertex/gemini...` ID and `HARNESS2_JUDGE_MODEL` a bare `gemini...` one,
with `GOOGLE_CLOUD_PROJECT` set; `--dry-run` reports the requirement and still
prints the rest of the plan. The benchmark's job-local proxy refreshes ADC tokens
and preserves Gemini tool-call thought signatures.

Setup patches the upstream fetch script to download with `curl` (then `wget`,
then `hf`) and to require the dataset repository's `SHA256SUMS`: an archive that
cannot be verified is not unpacked. A verified download prints
`wb-bench-<subset>-v1.0.tar.gz: OK`.

Tasks are discovered from `task.toml` and `instruction.md` under
`dependencies/workbuddy-bench/datasets/wb-bench-<subset>-v1.0/tasks/`. Libraries use
`metadata.role` on code and `metadata.category` on office and web. Each rollout
uses a fresh benchmark container. The adapter reads the verifier's `score.json`;
the improver never receives verifier output. Missing verifier results remain
unscored. Runs live under `runs/wb/<subset>/`.

For Codex, start the bridge with `harness2-bridge --host 0.0.0.0` so that the containers can reach it.

## skillbench

Generic skill and plugin-suite benchmark for AGY skill sets on the [AGY executor](SETUP.md#agy-executor) (`--substrate agy`, the default for `--bench skillbench`). A suite config (`suite.toml`, passed via `--suite <path>` or `SKILLBENCH_SUITE`) describes the plugin suite under test (for example, `harness2/benchmarks/skillbench/suites/cc-thingz.toml`). Skill tasks are grouped by `target_skill` (`--only-domains <skill>`), suite tasks by `family` (`--only-domains <family>`), and runs execute reference ("base") rollouts only (`--bases-only` is implied).

Run a multi-executor matrix, report, history update, and optional improvement pass with `skillbench-cycle` or the individual console scripts:

```bash
harness2-check --bench skillbench --substrate agy --suite harness2/benchmarks/skillbench/suites/cc-thingz.toml
skillbench-tasks check --suite harness2/benchmarks/skillbench/suites/cc-thingz.toml
skillbench-matrix --suite harness2/benchmarks/skillbench/suites/cc-thingz.toml --arms both --k 3
skillbench-report --suite harness2/benchmarks/skillbench/suites/cc-thingz.toml --results-root runs/skillbench/results --k 3
skillbench-report history append --suite harness2/benchmarks/skillbench/suites/cc-thingz.toml --report runs/skillbench/report.json
skillbench-improve --suite harness2/benchmarks/skillbench/suites/cc-thingz.toml --auto --models opus-5.5-high,gemini-3.8-flash-high --k 3
```

Or run the entire `matrix → report → history append → improve` pipeline in one command:

```bash
skillbench-cycle --suite harness2/benchmarks/skillbench/suites/cc-thingz.toml --arms both --k 3 --improve --yes
```

Single-executor sweeps also work directly via `harness2` / `harness2-judge` or `scripts/run_skillbench.sh`:

```bash
harness2 --bench skillbench --arm with --k 3
harness2 --bench skillbench --arm without --k 3
harness2-judge --bench skillbench --arm with --k 3 --bases
harness2-judge --bench skillbench --arm without --k 3 --bases
skillbench-report --k 3
PILOT=1 K=3 bash scripts/run_skillbench.sh   # 3-skill paid pilot (sets SKILLBENCH_JUDGE_REPEAT=2, --pilot)
K=3 bash scripts/run_skillbench.sh           # full corpus
```

### Suite configuration (`suite.toml`) reference

All suite-specific settings live in `suite.toml` (selected by `--suite <path>` or `SKILLBENCH_SUITE`, with a deprecated fallback to `SKILLBENCH_PLUGINS_DIR` / `SKILLBENCH_PLUGINS` when no suite is configured):

| Key | Type | Meaning |
|---|---|---|
| `name` | string (**required**) | Suite identifier (`[a-z0-9][a-z0-9_-]*`, e.g. `"cc-thingz"`); determines the suite-mode arm label `suite-<name>` and is included in judge redaction |
| `plugins_dir` | path (**required**) | Git checkout directory containing plugin subdirectories (relative to `suite.toml` or `~`-expanded); symlinks are resolved when staging or applying patches |
| `plugins` | list of strings or `"*"` | Allowlisted plugin directory names under `plugins_dir`, or `"*"` to discover every subdirectory containing a `plugin.json` |
| `tasks_root` | path (optional) | External task corpus directory; corpus resolution order is `--tasks` → `suite.tasks_root` → `SKILLBENCH_TASKS_ROOT` → bundled `harness2/benchmarks/skillbench/tasks` |
| `banned_tools` | list of strings (optional) | Non-AGY tool names rejected by `skillbench-improve` candidate validation (defaults to `["AskUserQuestion", "EnterPlanMode", "TodoWrite", "Bash", "Read", "Write", "Edit", "Glob", "Grep", "Agent", "Task"]`) |
| `[families]` | table of `family = ["skill", ...]` | Groups skills into functional families (e.g. `planning`, `analysis`) for `kind = "suite"` tasks |
| `[matrix].executors` | list of strings | Default executor model IDs swept by `skillbench-matrix` and `skillbench-cycle` (overridden by `--executors`) |
| `[matrix.judge_for]` | table of `"<prefix>" = "<judge>"` | Maps each executor model-ID prefix (e.g. `"opus-"`, `"gemini-"`, `"fable-"`) to a judge model from a different family; same-prefix pairs are refused |

`load_suite` computes `suite_hash` (SHA-256 over all regular files in the allowlisted plugins) and records `{suite, suite_hash}` in `run.json` and the resume identity.

### Console scripts

| Script | Purpose |
|---|---|
| `skillbench-tasks` | `skillbench-tasks new --skill <s> --id <id>` (or `--kind suite --family <f>`) scaffolds `task.toml`, 4-criterion `rubric.json`, and `fixture/README.md` with `forbidden_terms` and `asks_user` pre-populated from the skill's `SKILL.md`; `skillbench-tasks check` validates every task in the corpus and prints a status table |
| `skillbench-matrix` | Expands `(executor, arm, task, rep)` jobs for `--arms skill\|suite\|both\|baseline` (and optional `--suites path1.toml,path2.toml`), prints the pre-run rollout/wall-time/token budget table, checks `--max-rollouts` and TTY confirmation (`--yes`), runs `harness2` + `harness2-judge --bases` in-process per cell with automatic resume, and writes `matrix.json` |
| `skillbench-report` | Builds schema-v2 `report.json`, `report.md`, `report.html`, and `summary.md` from `--results-root` (or `--with-dir` / `--without-dir`); supports `--hardness-only`, `--summary-only`, `--suites`, and `skillbench-report history append\|show` |
| `skillbench-improve` | Propose–Probe–Judge loop for weak skills (`--auto`, `--all-weak`, or `--skill`), enforcing same-model probing, multi-model acceptance gates (`--models`), multi-round refinement (`--rounds`), candidate bloat cap (`--max-growth`), and git branch commits (`--apply`) |
| `skillbench-cycle` | Runs `matrix → report → history append → improve` in sequence (`--from <stage>`, `--skip <stage,...>`, `--improve`), writing per-stage timing and token totals to `<runs-root>/cycle.log` |

### Task format and manifest freeze

Tasks live under the resolved corpus root (`<tasks_root>/<skill-or-_suite>/<task-id>/`, defaulting to `harness2/benchmarks/skillbench/tasks/` when no suite `tasks_root` or `SKILLBENCH_TASKS_ROOT` is set):

- `task.toml`: `kind = "skill" | "suite"` (default `"skill"`), `target_skill` (required for `kind = "skill"`), `family` (required for `kind = "suite"`, must match a key in `[families]`), `prompt`, `timeout` (seconds), optional `asks_user = true|false` (overrides automatic detection from `ask_question` in `SKILL.md` frontmatter `allowed-tools`), optional `user_answers = [...]` (required when the task asks the user, e.g. `brainstorm`, `exec`, `make-plan`, `dialectic`, `backlog`, `clarify`, `learn`, `wrong`), optional `forbidden_terms = [...]`, `pilot = true|false`, `min_baseline_fail = 1` (minimum criteria a control rep must miss to count as `informative`), and an optional deterministic check: `verify = "<command>"` (split like a shell would, never run through one) plus `verify_restore = ["<fixture-relative file>", ...]`.
- `rubric.json`: 4–8 binary, outcome-based criteria (`[{"id": "...", "criterion": "..."}]`) that grade the deliverable rather than skill rituals or step headers.
- `fixture/`: small self-contained repository copied into a fresh `git init` working directory (`<run>/work`) for each rollout.

Prompts are checked on load (`check_prompt`) and via `skillbench-tasks check`: explicit invocations (`<name> skill`, `/<name>`, `skills/<name>`, `SKILL.md`, or any `forbidden_terms`) are rejected, while ordinary English uses of a skill word ("add this to the backlog") are allowed. Every arm appends the identical simulated-user preamble supplying `user_answers` in order and instructing the agent never to open an editor or interactive review.

The corpus is frozen into `<tasks_root>/manifest.json` with per-task SHA-256 hashes. Runs and `harness2-check` refuse to start on manifest drift:

```bash
skillbench-tasks check --suite harness2/benchmarks/skillbench/suites/cc-thingz.toml
python -m harness2.benchmarks.skillbench check
python -m harness2.benchmarks.skillbench freeze --force
```

### Arm kinds (`skill` vs `suite`), plugin allowlist, and side-effect shims

`--arm` is required for `--bench skillbench` and accepts four arm labels across two arm kinds:

- **Skill arms (`with` / `without`)**: run `kind = "skill"` tasks (`--arm with` and `--arm without`) with leave-one-out staging. Leave-one-out staging puts the suite's allowlisted plugins into `<run>/home/.gemini/config/plugins/` for arm `with`, whereas arm `without` omits only `skills/<target_skill>/` while keeping every other skill and all lifecycle hooks identical.
- **Suite arms (`baseline` / `suite-<name>`)**: run `kind = "suite"` family tasks (`--arm baseline` and `--arm suite-<name>`). Arm `baseline` stages zero plugins (measuring AGY's built-in behavior), while `suite-<name>` stages the full named suite. Pass `--suites other1.toml,other2.toml` to compare additional suites (`suite-<other>`) against the same `baseline` cell on the same family tasks.
- **Allowlist**: plugins are staged from `suite.plugins_dir` filtered by `suite.plugins` (or from the deprecated `SKILLBENCH_PLUGINS_DIR` / `SKILLBENCH_PLUGINS` fallback, e.g. `brainstorm,planning,review,skill-eval,thinking-tools,workflow`). Unlisted plugins in the user's real config are never staged.
- **Isolation and side-effect shims**: each rollout runs with a throwaway `HOME` (`<run>/home`) containing copied — never symlinked — credentials from the optional `SKILLBENCH_AUTH_FILES` (none for internal Jetski; scrubbed from `<run>/home` as soon as the rollout finishes), `EDITOR=true`, `VISUAL=true`, and `PATH` shims for `xclip`, `xsel`, `wl-copy`, `pbcopy`, and `revdiff` that write standard input to `<run>/side_effects/<tool>.txt` instead of touching the host clipboard or launching an interactive UI.

### Metrics, judging, schema v2, and Summary recommendations

`harness2-judge --bench skillbench --arm <with|without|baseline|suite-<name>> --bases` runs an isolated `agy` judge (`--model $SKILLBENCH_JUDGE_MODEL --json-schema`) in an empty `HOME` with no plugins. Its prompt contains only the task's `rubric.json` and the scrubbed deliverable (`final.md`, `diff.patch`, `new_files/`, `side_effects/` with the suite name, every staged `skills/<name>/` identifier, `SKILL.md` paths, and plugin paths replaced by `[redacted]`) — never the trajectory or arm label. Judgments are cached in `<run>/judgments.json`.

Before judging, tasks with a `verify` command get a deterministic check: the agent's work tree is copied to a scratch dir, the `verify_restore` files are put back from `fixture/` (so editing the tests cannot make them pass), and the command runs there with a 300 s timeout. The result (`passed`, `exit_code`, `timed_out`, output tail) is cached in `<run>/verify.json`; the judge never sees it.

`skillbench-report` reads single-model (`--with-dir` / `--without-dir`) or matrix (`--results-root runs/<id>/results`) namespaces and writes `report.json`, `report.md`, `report.html`, and `summary.md` (or only `summary.md` + `summary.html` with `--summary-only`):

- **Outcome lift**: for each task $t$, $d_t = \text{mean}_k(\text{frac}_{\text{treatment}}) - \text{mean}_k(\text{frac}_{\text{control}})$ over valid graded reps, and $\text{lift} = \text{mean}_t(d_t)$. The 95% confidence interval uses a two-level bootstrap (10,000 draws, fixed seed: resample tasks with replacement, then reps within each task). Verdicts are `helps` ($\text{CI}_{\text{low}} > 0$), `hurts` ($\text{CI}_{\text{high}} < 0$), `inconclusive`, or `insufficient data` when fewer than 3 tasks (2 in `--pilot` mode) have at least 2 valid graded reps in both arms. Invalid rollouts (timeout, non-zero exit, missing result event, or model mismatch) and ungraded rollouts are excluded rather than zeroed.
- **Time-to-skill**: invocation rate (`invoked / valid treatment reps`) and p50 / p90 of `t_skill_s` (seconds from process spawn to the first `read_file` or `view_file` of `skills/<target>/SKILL.md` in skill mode, or of any staged `SKILL.md` in suite mode, recorded with `skills_read` in `skill_timing.json`).
- **Deterministic checks and cost**: per arm, the `verify` pass rate (`tests pass WITH` / `WITHOUT`), and medians of lines changed in `diff.patch`, total tokens, and wall time over valid reps, plus `tokens_delta_pct` and `wall_delta_pct`.
- **Schema v2 (`report.json`)**: carries `schema_version: 2`, `suite: {name, hash}`, `models: [{executor, judge, skills, by_skill, suites, by_family, ...}]`, `model_summary: [...]`, and `summary: [{skill | family, kind, model, verdict, recommendation, reason, lift, ci, tokens_delta_pct, wall_delta_pct, invocation_rate, t_skill_p50, t_skill_p90, ceiling_tasks, total_tasks, best_model}]`. The first `models[]` entry is mirrored at top level for v1 compatibility.
- **Hardness flag (per executor, never an exclusion)**: from an executor's control arm (`without` / `baseline`) alone, a task is `informative` when that arm misses at least `min_baseline_fail` criteria (`task.toml`, default 1) in at least one valid graded rep, `ceiling` when every valid control rep passes the whole rubric (the task cannot show a lift for this executor), and `unclassified` with fewer than 2 valid control reps. Ceiling tasks stay in the lift, CI, and verdict (dropping them would select tasks on the control arm's luck and bias the lift upward); each per-skill row reports `ceiling_tasks / total_tasks` and sets `needs_harder_tasks` when more than 50% of the skill's tasks are at the ceiling. `skillbench-report --hardness-only` prints the task × executor hardness table and exits without writing a report; it needs only control-arm namespaces (e.g. after `skillbench-matrix --arms baseline`), which is the corpus-authoring loop.
- **One-page Summary, recommendation rule, and `history.jsonl`**: `summary.md` and the first tab (`Summary`) of `report.html` present per-skill recommendations, per-family suite vs built-in baseline verdicts, and per-model efficiency on one screen. Each row's `recommendation` follows a 4-level precedence rule (first match wins; `reason` cites the matched rule):
  1. `needs harder tasks` — `> 50%` of the skill's tasks are at the `ceiling` for this executor;
  2. `consider removing` — verdict `hurts`, or `inconclusive` with tokens p50 `>= +15%` and wall p50 `>= +15%` and no `verify` gain;
  3. `fix` — `inconclusive` with invocation `< 80%` or `t_skill` p90 `> 45s`, or the improver has an accepted candidate;
  4. `keep` — verdict `helps`, or `inconclusive` with `>= 15%` tokens or wall-time saving, or none of the above.
  `skillbench-report history append` idempotently records one line per `(date, suite_hash, executor, skill | family)` in `history.jsonl` next to `suite.toml` (`skillbench-report history show` renders it as Markdown or JSON), and the Summary page draws dependency-free inline `<svg>` sparklines with vertical ticks where `suite_hash` changed.
- **HTML dashboard**: `report.html` is one self-contained file (the AGY-allowlisted Tailwind script, `--app-*` theme colours with fallbacks, vanilla JS tabs): Summary tab, KPI cards, per-skill table with CI bars, per-family suite table, per-task deltas and criterion pass rates per arm, side-by-side answers with judge reasons, and diagnostics. Open it in a browser or embed it in AGY chat with `<agent-embed src="file:///…/report.html" height="800px"></agent-embed>`.
- **Model guard**: the stream's leading `init` event names the model that ran (`init.model`, e.g. `opus-5.5-high`); a model that disagrees with the requested one invalidates the rollout. A stream without any model field (`actual_model = null`) keeps the rollout valid and the report header notes `model unverified (agy reports no model)`.
- **Statistical power**: with 4 tasks per skill, the task-level bootstrap has only $\binom{4+4-1}{4} = 35$ distinct resamples, so only large lifts (roughly $\ge 0.2$ of the rubric) will achieve a `helps` or `hurts` verdict; most skills are expected to land `inconclusive` in the first full run. The report prints this note alongside the table.

`skillbench-report` is added by this fork. The upstream `harness2-report` also reads skillbench runs, one arm at a time: it reports the arm's mean rubric score over its judged reference executions (`base_judged.json`; `--bases` is implied for skillbench). Use it as a plain per-arm cross-check; it does not pair the arms or compute lift:

```bash
harness2-report --bench skillbench --arm with --k 3
harness2-report --bench skillbench --arm without --k 3
```

### Environment variables

| Variable | Meaning |
|---|---|
| `HARNESS2_AGY_BIN` | Executable name or path for AGY (default `agy`); for Google-internal Jetski use `/google/bin/releases/jetski-devs/tools/cli` (a shell alias such as `jet` cannot be used) |
| `SKILLBENCH_SUITE` | Path to `suite.toml` describing the plugin suite under test (`name`, `plugins_dir`, `plugins`, `tasks_root`, `banned_tools`, `[families]`, `[matrix]`); alternative to `--suite` |
| `SKILLBENCH_TASKS_ROOT` | Task corpus root directory when `suite.toml` sets no `tasks_root` and `--tasks` is not passed (optional, defaults to the bundled corpus under `harness2/benchmarks/skillbench/tasks`) |
| `SKILLBENCH_PLUGINS_DIR` | Deprecated env fallback when no `suite.toml` is configured: path to a plugin suite's `plugins/` directory |
| `SKILLBENCH_PLUGINS` | Deprecated env fallback: comma-separated plugin allowlist (e.g. `brainstorm,planning,review,skill-eval,thinking-tools,workflow`, or unset to discover every subdirectory with `plugin.json`) |
| `SKILLBENCH_AUTH_FILES` | Comma-separated credential file paths relative to `$HOME` copied into each rollout's throwaway `HOME` (optional, default none: internal Jetski needs no files; a listed file that is missing fails preflight) |
| `SKILLBENCH_EXEC_MODEL` | Executor model id passed to `agy --model` in single-model runs (default `opus-5.5-high`; jet ids embed the effort: `-low`, `-medium`, `-high`, `-max`) |
| `SKILLBENCH_JUDGE_MODEL` | Judge model id passed to `agy --model` in single-model runs (default `fable-5.1-high`) |
| `SKILLBENCH_IMPROVER_MODEL` | Improver model id passed to `agy --model` by `skillbench-improve` when proposing candidate `SKILL.md` files (optional, defaults to `SKILLBENCH_EXEC_MODEL`) |
| `SKILLBENCH_EFFORT` | Extra `agy --effort` value (`low`, `medium`, `high`, `xhigh` or `max`) for both arms and the judge (optional, default empty: no `--effort`, the model id pins it); recorded in the run identity |
| `SKILLBENCH_JUDGE_REPEAT` | Repeated judge calls per rollout to measure criterion-level judge noise (default `1`, pilot sets `2`) |
| `SKILLBENCH_PILOT` | Set to `1` to restrict runs and reports to `pilot = true` tasks |

### Pre-pilot spike and cost gates

Run skillbench in three gated stages to control API spend:

1. **Pre-pilot spike** (10 short calls, done once with Jetski on 2026-10-26): `HARNESS2_AGY_BIN=/google/bin/releases/jetski-devs/tools/cli SKILLBENCH_PLUGINS_DIR=~/projects/cc-thingz/plugins bash scripts/agy_spike.sh` captures real `stream-json` traces under `tests/fixtures/agy/real/`; [`docs/skillbench-spike.md`](skillbench-spike.md) records what they showed (model ids, `init.model`, `read_file` skill reads, `ask_question` under `-p`, no auth files needed). Re-run it after a Jetski upgrade and keep `tests/test_agy_real_streams.py` green.
2. **Paid pilot** (3 pilot skills: `make-plan`, `root-cause-investigator`, `code-review-refactor`; 2 tasks each): run `PILOT=1 K=3 bash scripts/run_skillbench.sh` (36 executor rollouts + 72 judge calls with `SKILLBENCH_JUDGE_REPEAT=2`). If the report's criterion-level judge disagreement exceeds ~10%, tighten the rubrics, re-freeze with `--force`, and re-run the pilot.
3. **Cost gate and full run**: extrapolate per-rollout time and token usage from the pilot to the full 18-skill corpus (72 tasks × 2 arms × `K=3` = 432 executor rollouts + judging) and obtain explicit approval before running `K=3 bash scripts/run_skillbench.sh`.

