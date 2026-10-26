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

Leave-one-out skill lift benchmark for the 18 `cc-thingz` skills (`<cc-thingz>/plugins/*/skills/*/SKILL.md`) on the [AGY executor](SETUP.md#agy-executor) (`--substrate agy`, the default for `--bench skillbench`). Each skill is its own domain (`--only-domains <skill>`), and runs execute reference ("base") rollouts only (`--bases-only` is implied).

```bash
harness2-check --bench skillbench --substrate agy
harness2 --bench skillbench --arm with --k 3
harness2 --bench skillbench --arm without --k 3
harness2-judge --bench skillbench --arm with --k 3 --bases
harness2-judge --bench skillbench --arm without --k 3 --bases
skillbench-report --k 3
```

Or run the full chain with the helper script:

```bash
PILOT=1 K=3 bash scripts/run_skillbench.sh   # 3-skill paid pilot (sets SKILLBENCH_JUDGE_REPEAT=2, --pilot)
K=3 bash scripts/run_skillbench.sh           # full corpus
```

### Task format and manifest freeze

Tasks live under `harness2/benchmarks/skillbench/tasks/<skill>/<task-id>/`:

- `task.toml`: `target_skill`, `prompt`, `timeout` (seconds), optional `user_answers = [...]` (required for the 8 interactive `ask_question` skills: `brainstorm`, `exec`, `make-plan`, `dialectic`, `backlog`, `clarify`, `learn`, `wrong`), optional `forbidden_terms = [...]`, and `pilot = true|false`.
- `rubric.json`: 4–8 binary, outcome-based criteria (`[{"id": "...", "criterion": "..."}]`) that grade the deliverable rather than skill rituals or step headers.
- `fixture/`: small self-contained repository copied into a fresh `git init` working directory (`<run>/work`) for each rollout.

Prompts are checked on load (`check_prompt`): explicit invocations (`<name> skill`, `/<name>`, `skills/<name>`, `SKILL.md`, or any `forbidden_terms`) are rejected, while ordinary English uses of a skill word ("add this to the backlog") are allowed. Both arms append the identical simulated-user preamble supplying `user_answers` in order and instructing the agent never to open an editor or interactive review.

The corpus is frozen into `harness2/benchmarks/skillbench/tasks/manifest.json` with per-task SHA-256 hashes. Runs and `harness2-check` refuse to start on manifest drift:

```bash
python -m harness2.benchmarks.skillbench check
python -m harness2.benchmarks.skillbench freeze --force
```

### Arms, plugin allowlist, and side-effect shims

`--arm with` and `--arm without` are required for `--bench skillbench` and map to the run `tag`, giving each arm its own namespace (`skillbench-agy-agg<k>-with` / `skillbench-agy-agg<k>-without`), identity check, and results directory under `runs/skillbench/`.

- **Allowlist**: plugins are staged from `SKILLBENCH_PLUGINS_DIR` filtered by `SKILLBENCH_PLUGINS` (default `brainstorm,planning,review,skill-eval,thinking-tools,workflow`). Unlisted plugins in the user's real config (such as `revmux` or telemetry extensions) are never staged.
- **Leave-one-out**: for a task targeting `<skill>`, leave-one-out staging puts the full allowlisted suite into `<run>/home/.gemini/config/plugins/` for arm `with`, whereas arm `without` omits only `skills/<skill>/` while keeping every other skill and all lifecycle hooks (including `skill-eval`) identical.
- **Isolation and side-effect shims**: each rollout runs with a throwaway `HOME` (`<run>/home`) containing copied — never symlinked — credentials from the optional `SKILLBENCH_AUTH_FILES` (none for internal Jetski; scrubbed from `<run>/home` as soon as the rollout finishes), `EDITOR=true`, `VISUAL=true`, and `PATH` shims for `xclip`, `xsel`, `wl-copy`, `pbcopy`, and `revdiff` that write standard input to `<run>/side_effects/<tool>.txt` instead of touching the host clipboard or launching an interactive UI.

### Metrics, judging, and statistical power

`harness2-judge --bench skillbench --arm <with|without> --bases` runs an isolated `agy` judge (`--model $SKILLBENCH_JUDGE_MODEL --json-schema`) in an empty `HOME` with no plugins. Its prompt contains only the task's `rubric.json` and the scrubbed deliverable (`final.md`, `diff.patch`, `new_files/`, `side_effects/` with skill names, `SKILL.md` paths, and plugin paths replaced by `[redacted]`) — never the trajectory or arm label. Judgments are cached in `<run>/judgments.json`.

`skillbench-report` reads both namespaces and writes `runs/skillbench/report.json` and `runs/skillbench/report.md`:

- **Outcome lift**: for each task $t$, $d_t = \text{mean}_k(\text{frac}_{\text{WITH}}) - \text{mean}_k(\text{frac}_{\text{WITHOUT}})$ over valid graded reps, and $\text{lift} = \text{mean}_t(d_t)$. The 95% confidence interval uses a two-level bootstrap (10,000 draws, fixed seed: resample tasks with replacement, then reps within each task). Verdicts are `helps` ($\text{CI}_{\text{low}} > 0$), `hurts` ($\text{CI}_{\text{high}} < 0$), `inconclusive`, or `insufficient data` when fewer than 3 tasks (2 in `--pilot` mode) have at least 2 valid graded reps in both arms. Invalid rollouts (timeout, non-zero exit, missing result event, or model mismatch) and ungraded rollouts are excluded rather than zeroed.
- **Time-to-skill**: invocation rate (`invoked / valid WITH reps`) and p50 / p90 of `t_skill_s` (seconds from process spawn to the first `read_file` (or older `view_file`) of `skills/<target>/SKILL.md`) across invoked valid WITH reps.
- **Model guard**: the stream's leading `init` event names the model that ran (`init.model`, e.g. `opus-5.5-high`); a model that disagrees with the requested one invalidates the rollout. A stream without any model field (`actual_model = null`) keeps the rollout valid and the report header notes `model unverified (agy reports no model)`.
- **Statistical power**: with 4 tasks per skill, the task-level bootstrap has only $\binom{4+4-1}{4} = 35$ distinct resamples, so only large lifts (roughly $\ge 0.2$ of the rubric) will achieve a `helps` or `hurts` verdict; most skills are expected to land `inconclusive` in the first full run. The report prints this note alongside the table.

### Environment variables

| Variable | Meaning |
|---|---|
| `HARNESS2_AGY_BIN` | Executable name or path for AGY (default `agy`); for Google-internal Jetski use `/google/bin/releases/jetski-devs/tools/cli` (a shell alias such as `jet` cannot be used) |
| `SKILLBENCH_PLUGINS_DIR` | Path to a `cc-thingz` `plugins/` directory (**required**) |
| `SKILLBENCH_PLUGINS` | Comma-separated plugin allowlist (default `brainstorm,planning,review,skill-eval,thinking-tools,workflow`) |
| `SKILLBENCH_AUTH_FILES` | Comma-separated credential file paths relative to `$HOME` copied into each rollout's throwaway `HOME` (optional, default none: internal Jetski needs no files; a listed file that is missing fails preflight) |
| `SKILLBENCH_EXEC_MODEL` | Executor model id passed to `agy --model` (default `opus-5.5-high`; jet ids embed the effort: `-low`, `-medium`, `-high`, `-max`) |
| `SKILLBENCH_JUDGE_MODEL` | Judge model id passed to `agy --model` (default `fable-5.1-high`) |
| `SKILLBENCH_EFFORT` | Extra `agy --effort` value (`low`, `medium`, `high`, `xhigh` or `max`) for both arms and the judge (optional, default empty: no `--effort`, the model id pins it); recorded in the run identity |
| `SKILLBENCH_JUDGE_REPEAT` | Repeated judge calls per rollout to measure criterion-level judge noise (default `1`, pilot sets `2`) |
| `SKILLBENCH_PILOT` | Set to `1` to restrict runs and reports to `pilot = true` tasks |

### Pre-pilot spike and cost gates

Run skillbench in three gated stages to control API spend:

1. **Pre-pilot spike** (10 short calls, done once with Jetski on 2026-10-26): `HARNESS2_AGY_BIN=/google/bin/releases/jetski-devs/tools/cli SKILLBENCH_PLUGINS_DIR=~/projects/cc-thingz/plugins bash scripts/agy_spike.sh` captures real `stream-json` traces under `tests/fixtures/agy/real/`; [`docs/skillbench-spike.md`](skillbench-spike.md) records what they showed (model ids, `init.model`, `read_file` skill reads, `ask_question` under `-p`, no auth files needed). Re-run it after a Jetski upgrade and keep `tests/test_agy_real_streams.py` green.
2. **Paid pilot** (3 pilot skills: `make-plan`, `root-cause-investigator`, `code-review-refactor`; 2 tasks each): run `PILOT=1 K=3 bash scripts/run_skillbench.sh` (36 executor rollouts + 72 judge calls with `SKILLBENCH_JUDGE_REPEAT=2`). If the report's criterion-level judge disagreement exceeds ~10%, tighten the rubrics, re-freeze with `--force`, and re-run the pilot.
3. **Cost gate and full run**: extrapolate per-rollout time and token usage from the pilot to the full 18-skill corpus (72 tasks × 2 arms × `K=3` = 432 executor rollouts + judging) and obtain explicit approval before running `K=3 bash scripts/run_skillbench.sh`.

