<h3 align="center">
  <img src="assets/Harness2-wordmark.png" alt="Harness²" width="360"><br>
  Recursive Agent Harnessing for an Open World
</h3>

<p align="center">
  <a href="https://arxiv.org/pdf/2610.14224"><img src="https://img.shields.io/badge/Paper-arXiv-EA4335?logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://harness2.github.io"><img src="https://img.shields.io/badge/Website-harness2.github.io-4285F4?logo=googlechrome&logoColor=white" alt="Website"></a>
  <a href="docs/BENCHMARKS.md"><img src="https://img.shields.io/badge/Docs-Benchmark%20guides-FBBC04" alt="Benchmark guides"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-34A853" alt="License"></a>
</p>

<p align="center">
  <a href="docs/BENCHMARKS.md#jobbench">
    <img src="assets/harness2-setup.gif" alt="Install, set up, and run JobBench: uv pip install -e '.[render]', harness2-setup jb, then harness2 --bench jb --k 1 --mode parallel. Click for prerequisites and commands." width="640">
  </a>
</p>

A **harness** is the system around a model that enables it to act as an agent:
prompts, tools, memory, skills, orchestration, and more. **Harnessing** is the
process of optimizing that system.

Harness² is a plug-and-play framework for **recursive agent harnessing** in an
open world, where agents encounter tasks unseen during harness design. It
connects two levels of recursion while keeping model weights fixed:

* **Task-level recursion** refines the execution harness for the current task
  through a contrastive **Propose–Probe–Compose** step: propose edits, probe
  their effects on the same task, and compose a refined harness from behavioral
  contrasts between executions, without ground-truth feedback.
* **Harnessing-level recursion** updates a persistent *improvement harness*
  with adopted harnesses and evidence-supported lessons from each task to guide
  future refinement.

<p align="center">
  <a href="assets/harness2_pipeline.png">
    <img src="assets/harness2_pipeline.png" alt="Harness² pipeline. Task-level recursion: an agent executor runs the initial harness, a harness editor proposes invariant and domain edits, candidate probes execute them, and a harness composer combines them into the final harness. Harnessing-level recursion: an experience reflector reflects on the step and updates the improvement harness used for the next task." width="100%">
  </a>
</p>

Across three professional benchmarks and four model–harness configurations, a
single Harness² step improves GLM-5 and Gemini 3.5 Flash over their base
harnesses, with gains of up to 11.5 points on WorkBuddy-Bench, 4.5 points on
JobBench, and 11.8 percentage points on LAB's partial pass rate. Increasing the
task-level recursion budget brings further gains.

The release supports **parallel and sequential recursion**, **OpenCode and
Codex executors**, and **JobBench, LAB, and WorkBuddy-Bench**.

> [!NOTE]
> **About this fork.** [`tematef/harness2`](https://github.com/tematef/harness2)
> is a fork of [`google-research/harness2`](https://github.com/google-research/harness2)
> that adds a third executor, **AGY / Jetski** (`--substrate agy`), and a
> generic skill and plugin-suite benchmark, **skillbench**, with a model-matrix
> runner, one-page Summary recommendations, and a model-aware skill improver
> ([`cc-thingz`](https://github.com/tematef/cc-thingz) is included as a reference
> suite). See
> [AGY / Jetski executor and skillbench](#agy--jetski-executor-and-skillbench).

## Video overview

https://github.com/user-attachments/assets/76b10548-b14b-4c8f-9aa5-a93662df4d56

## Contents

- [Video overview](#video-overview)
- [Harness components](#harness-components)
- [Install](#install)
- [Run a benchmark](#run-a-benchmark)
- [AGY / Jetski executor and skillbench](#agy--jetski-executor-and-skillbench)
- [Experiments](#experiments)
- [Web access](#web-access)
- [Results and resume](#results-and-resume)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Disclaimer](#disclaimer)

## Harness components

Harness² exposes eight editable components around the model: **system prompt,
guardrails, memory, tool guidance, subagents, skills, scripts, and plugins**.
They control context construction, the tool interface, delegation, and behavior
inside the tool loop.

<p align="center">
  <a href="assets/components.png">
    <img src="assets/components.png" alt="Eight harness components grouped by runtime role: system prompt, guardrails, memory, and skills construct context; tool guidance and scripts define the tool interface; subagents orchestrate delegated work; plugins control middleware." width="100%">
  </a>
</p>

## Install

Use **Linux, Python 3.12, and [uv](https://docs.astral.sh/uv/getting-started/installation/)**.
From the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[render]'
harness2-demo --output runs/demo --k 2
```

The local demo runs both recursion modes with deterministic editors and real
Python executions. It requires no model credentials.

For model-backed benchmarks, install **Node.js 20+, the Google Cloud CLI**, and
the [benchmark dependencies](docs/SETUP.md#requirements). LAB needs Pandoc and
Poppler; WorkBuddy-Bench needs Docker. Then configure your models:

```bash
cp .env.example .env
# Edit .env with your Vertex AI project and model IDs.
gcloud auth application-default login
```

See the [setup guide](docs/SETUP.md) for model settings, authentication
alternatives, and the [Codex executor](docs/SETUP.md#codex-executor).

## Run a benchmark

The following example installs LAB and runs one task, then evaluates and reports
its result. Model-backed runs incur API costs; start with a small task selection.

```bash
harness2-setup lab
harness2-check --bench lab
harness2 --bench lab --k 1 --mode parallel \
  --only-domains antitrust-competition --max-tasks 1
harness2-judge --bench lab --k 1 --mode parallel
harness2-report --bench lab --k 1 --mode parallel
```

Sequential recursion runs the same commands with `--mode sequential`. `--k` is
the task-level recursion budget; see [Experiments](#experiments). Judging and
reporting must use the same benchmark, executor, subset, `--k`, `--mode`, and
seed as the refinement run. Add `--dry-run` to `harness2` to preview task
selection without model calls.

| Benchmark | Tasks | Install | Guide |
|---|---|---|---|
| JobBench | Professional work | `harness2-setup jb` | [JobBench](docs/BENCHMARKS.md#jobbench) |
| LAB | Legal work | `harness2-setup lab` | [LAB](docs/BENCHMARKS.md#lab) |
| WorkBuddy-Bench | Office, web, and code | `harness2-setup wb` | [WorkBuddy](docs/BENCHMARKS.md#workbuddy-bench) |
| skillbench | Generic skill and plugin-suite lift on AGY (e.g. `cc-thingz`) | `harness2-check --bench skillbench` | [skillbench](docs/BENCHMARKS.md#skillbench) |

Choose one benchmark per command. WorkBuddy also takes `--subset office`,
`--subset web`, or `--subset code` on every command after setup. The
[benchmark guides](docs/BENCHMARKS.md) cover each one's data, domains, and
grading.

## AGY / Jetski executor and skillbench

This fork adapts Harness² to run a **local AGY (Jetski) CLI** as the agent
executor and provides **skillbench**, a generic benchmark and improvement loop
for any AGY skill or plugin suite (with `cc-thingz` shipped as an example suite
at `harness2/benchmarks/skillbench/suites/cc-thingz.toml`).

**How skillbench works**

* **Two arm kinds (`skill` and `suite`).**
  * **Skill arms (`with` vs `without`)**: leave-one-out per skill. Arm `with`
    stages the suite's allowlisted plugins; arm `without` is identical except
    that `skills/<target>/` is removed, keeping every other skill and lifecycle
    hook unchanged.
  * **Suite arms (`suite-<name>` vs `baseline`, plus optional `--suites`)**:
    compares the whole plugin suite against AGY's built-in behavior with zero
    plugins (`baseline`) — and optionally against competing suites passed via
    `--suites path1.toml,path2.toml` — on multi-step family tasks (`kind = "suite"`).
* **Model matrix with cross-family judging.** `skillbench-matrix` sweeps one or
  more executor models (`[matrix].executors`) and pairs each executor with a
  judge from a different model family (`[matrix.judge_for]`, e.g.
  `opus- → gemini`, `gemini- → fable`, `fable- → gemini`). The judge runs in an
  empty `HOME` with no plugins and grades scrubbed deliverables against a binary
  rubric.
* **Outcome lift, hardness, and time-to-skill.** `skillbench-report` computes
  paired rubric lift with a 95% two-level bootstrap confidence interval, per-task
  hardness (`informative`, `ceiling`, `unclassified`) from each executor's
  control arm (`without` / `baseline`), skill discovery rate and p50/p90
  `t_skill_s`, deterministic `verify` test pass rates, diff lines, token and
  wall-time deltas, and a one-page **Summary** (`summary.md` + the first tab of
  `report.html`).
* **Isolation.** Each rollout runs in a fresh git repository with a throwaway
  `HOME` containing only the staged plugins. Clipboard and review tools
  (`xclip`, `xsel`, `wl-copy`, `pbcopy`, `revdiff`) are replaced with shims
  writing to `<run>/side_effects/`. Interactive skills that ask the user
  questions (detected automatically from `ask_question` in `SKILL.md`
  frontmatter `allowed-tools` or `asks_user` in `task.toml`) receive scripted
  `user_answers` from an identical simulated-user preamble in every arm.
* **Bases only.** skillbench runs reference executions only (`--bases-only` is
  implied); the upstream Propose–Probe–Compose recursion is unchanged.

**skillbench CLI tools**

| Command | Purpose |
|---|---|
| `skillbench-tasks` | Scaffold (`new --skill <s\|--kind suite --family <f>> --id <id> [--prompt ... --timeout ... --force --tasks ... --suite ...]`) and validate (`check [--tasks ... --suite ...]`) task corpora against `suite.toml` |
| `skillbench-matrix` | Expand `(executor, arm, task, rep)` jobs (`--suite`, `--suites`, `--executors`, `--k`, `--arms`, `--parallel`, `--seed`, `--split`, `--split-file`, `--only-domains`, `--max-tasks`, `--pilot`, `--runs-root`, `--history`, `--dry-run`), print the rollout/time/token budget table, enforce `--max-rollouts` / `--yes`, and run sweep + judge in-process with resume |
| `skillbench-report` | Build schema-v2 `report.json`, `report.md`, `report.html`, and `summary.md` (`--suite`, `--suites`, `--results-root`, `--runs-root`, `--with-dir`, `--without-dir`, `--out-dir`, `--k`, `--seed`, `--bootstrap-seed`, `--draws`, `--pilot`, `--history`, `--hardness-only`, `--summary-only`, and `history append\|show` with `--report`, `--date`, `--skill`, `--executor`, `--json`) |
| `skillbench-improve` | Targeted **Propose → Probe → Verify/Judge** loop for weak skills (`--suite`, `--runs-dir`, `--skill`, `--auto`, `--all-weak`, `--k`, `--pilot`, `--include-verdicts`, `--improver-model`, `--dry-run`) with same-model enforcement (`--allow-model-mismatch`), multi-model gating (`--models`), multi-round refinement (`--rounds`), size cap (`--max-growth`), and branch commit (`--apply`) |
| `skillbench-cycle` | End-to-end orchestrator (`matrix → report → history append → improve`) with `--from` / `--from-stage`, `--skip`, `--improve`, `--runs-root` / `--runs-dir`, `--cycle-log`, and `cycle.log` accounting |

### Benchmark any skill set

1. **Write `suite.toml`.** Everything suite-specific lives in a `suite.toml` file
   (selected with `--suite <path>` or `SKILLBENCH_SUITE`):

   ```toml
   name = "my-suite"
   plugins_dir = "~/projects/my-suite/plugins"
   plugins = "*"  # or explicit list ["core", "workflows"]
   tasks_root = "~/projects/my-suite/benchmarks/skillbench/tasks"
   banned_tools = ["AskUserQuestion", "EnterPlanMode", "TodoWrite"]

   [families]
   planning = ["design", "make-plan", "exec"]

   [matrix]
   executors = ["opus-5.5-high", "gemini-3.8-flash-high", "fable-5.1-high"]

   [matrix.judge_for]  # keyed by executor prefix; same-family pairs are refused
   "opus-" = "gemini-3.8-flash-high"
   "gemini-" = "fable-5.1-high"
   "fable-" = "gemini-3.8-flash-high"
   ```

2. **Scaffold tasks and freeze the manifest.** Create 4+ outcome-graded tasks per
   skill (and family tasks for suite mode), validate prompt hygiene and rubrics,
   then freeze `manifest.json`:

   ```bash
   skillbench-tasks new --suite suite.toml --skill make-plan --id Dag-migration
   skillbench-tasks new --suite suite.toml --kind suite --family planning --id auth-rollout
   skillbench-tasks check --suite suite.toml
   SKILLBENCH_SUITE=suite.toml python -m harness2.benchmarks.skillbench freeze --force
   ```

3. **Run a baseline-only hardness check.** Before spending quota on treatment
   arms, run the control arms (`without` / `baseline`) on a fast executor and
   inspect task hardness (`informative` when the control arm misses `>= 1`
   criterion in `>= 1` valid rep, `ceiling` when every valid control rep scores
   `1.00`, `unclassified` with `< 2` valid control reps):

   ```bash
   export HARNESS2_AGY_BIN=/google/bin/releases/jetski-devs/tools/cli
   skillbench-matrix --suite suite.toml --arms baseline \
     --executors gemini-3.8-flash-high --k 2 --max-rollouts 60
   skillbench-report --suite suite.toml --results-root runs/skillbench/results --hardness-only
   ```

   Ceiling tasks are never dropped from lift or CIs (dropping them would select
   on control-arm luck), so harden any `ceiling` tasks before the full sweep.

4. **Run the matrix (or full cycle).** `skillbench-matrix` prints the estimated
   rollouts, wall minutes, and tokens per executor (using `history.jsonl` or
   `skill_timing.json` medians, falling back to `3.0 min/rollout`) and asks for
   confirmation (`--yes` to skip in non-interactive runs). Re-running an
   interrupted command resumes and schedules `0` new rollouts for completed cells:

   ```bash
   skillbench-matrix --suite suite.toml --arms both --k 3 --max-rollouts 520
   skillbench-report --suite suite.toml --results-root runs/skillbench/results --k 3
   skillbench-report history append --suite suite.toml --report runs/skillbench/report.json
   ```

   Or run all stages (`matrix → report → history → improve`) in one command:

   ```bash
   skillbench-cycle --suite suite.toml --arms both --k 3 --improve --yes
   ```

5. **Read the Summary and `history.jsonl`.** `runs/skillbench/summary.md` and the
   **Summary** tab of `runs/skillbench/report.html` answer three questions on one
   screen:
   * **Per-skill recommendation** (`keep`, `fix`, `consider removing`, or
     `needs harder tasks`) with inline SVG sparklines from `history.jsonl`
     (vertical tick markers show where `suite_hash` changed);
   * **Per-family suite vs built-in baseline verdict** (`suite-<name>` vs
     `baseline` and any `--suites` competitors);
   * **Per-model efficiency** (mean `WITH` / `WITHOUT` scores, lift, median
     tokens, median wall time, and invalid rollout count per executor).

   **Recommendation precedence rule** (evaluated in order; first match wins and
   its rule number is cited in `reason`):
   1. `needs harder tasks` — `> 50%` of the skill's tasks are at the `ceiling`
      for this executor;
   2. `consider removing` — verdict is `hurts`, **or** verdict is `inconclusive`
      with median tokens `>= +15%` **and** median wall time `>= +15%` **and** no
      deterministic `verify` gain;
   3. `fix` — verdict is `inconclusive` with invocation rate `< 80%` or
      `t_skill` p90 `> 45s`, **or** `skillbench-improve` has an accepted
      candidate;
   4. `keep` — verdict is `helps`, **or** verdict is `inconclusive` with
      `>= 15%` token or wall-time saving, **or** none of the above.

### Improve a weak skill (`skillbench-improve`)

When a skill is flagged `fix` or `consider removing` in `report.json`,
`skillbench-improve` runs a model-aware **Propose → Probe → Verify/Judge** loop:

```bash
skillbench-improve --suite suite.toml --auto --models opus-5.5-high,gemini-3.8-flash-high \
  --rounds 2 --max-growth 50 --k 3
```

1. **Contrastive evidence (no rubric leak).** Builds an evidence packet from the
   `WITH` and `WITHOUT` trajectories, deliverables, and `verify.json` (excluding
   `rubric.json` and judge verdicts unless `--include-verdicts` is passed). In
   multi-round mode (`--rounds N`), rounds `2..N` append the previous round's
   candidate diff, 3-way metrics, and failed judge reasons.
2. **Candidate validation and bloat cap.** Prompts an isolated `agy` improver
   session (`--improver-model` / `$SKILLBENCH_IMPROVER_MODEL`) for a revised
   `SKILL.md` and validates YAML frontmatter, headless fallback, the suite's
   `banned_tools`, and `--max-growth` (default `50%` maximum character growth
   over the original `SKILL.md`).
3. **Same-model probing and multi-model gate.** Probes the candidate with the
   **exact executor model** recorded in the baseline cell (refuses a model
   mismatch unless `--allow-model-mismatch` is set). With `--models m1,m2,...`,
   reuses each model's matrix cells for `WITHOUT` and `Current WITH`, probes the
   candidate on every listed model, and accepts the candidate only if the
   acceptance gate (no `verify`, invocation, or rubric regression, plus a rubric
   gain or `>= 15%` token/wall-time saving) passes on **every** model.
4. **3-way report and `--apply` onto a branch.** Writes
   `runs/skillbench/improve/improve_report.{json,md,html}` and `best.patch`.
   By default, `skillbench-improve` prints the git commands as a dry run; pass
   `--apply` to resolve `plugins_dir` symlinks (`realpath`), verify a clean git
   working tree, create branch `skillbench/improve-<skill>-<date>`, apply
   `best.patch`, and commit with the 3-way numbers (never pushes).

**Verified against real Jetski output.** `scripts/agy_spike.sh` records real
`stream-json` traces into `tests/fixtures/agy/real/`. The decoder, the
synthetic test fixtures, and the model guard follow those traces; the findings
are in [docs/skillbench-spike.md](docs/skillbench-spike.md).

See the [skillbench guide](docs/BENCHMARKS.md#skillbench) for the `suite.toml`
reference, task format, schema v2, and cost gates, and the
[AGY executor setup](docs/SETUP.md#agy-executor) for all environment variables.

## Experiments

Scale task-level recursion in two ways: **parallel recursion** explores multiple
candidate harnesses and composes refinements from their execution evidence;
**sequential recursion** carries the refined harness and accumulated evidence
through successive Propose–Probe–Compose steps.

<p align="center">
  <a href="assets/harness2_recursion_modes.png">
    <img src="assets/harness2_recursion_modes.png" alt="Two ways to scale task-level recursion. Parallel: probe k general and k domain-specific candidate harnesses, then produce k compositions from the shared evidence. Sequential: carry the refined harness and accumulated evidence through k Propose–Probe–Compose steps." width="100%">
  </a>
</p>
<p align="center"><em>Parallel recursion explores wider; sequential recursion refines deeper.</em></p>

| Option | Behavior |
|---|---|
| `--k 1` | One Propose–Probe–Compose step |
| `--k K --mode parallel` | Probe `K` general and `K` domain-specific candidates, then produce `K` compositions |
| `--k K --mode sequential` | Refine the harness through `K` successive steps |
| `--substrate cx` | Use Codex instead of the default OpenCode executor |
| `--substrate agy` | Use the AGY / Jetski CLI (skillbench only; the default for `--bench skillbench`) |

After each task, harnessing-level recursion retains the adopted harness and
evidence-supported lessons to guide future refinement. See the
[experiment guide](docs/EXPERIMENTS.md) for task selection, concurrency,
reproducibility, and baseline comparisons.

## Web access

Executor network access is unrestricted, and Codex runs with its sandbox
disabled. Public benchmark rubrics may be accessible online. Before evaluation,
configure [network restrictions](docs/EXPERIMENTS.md#web-access) to prevent
access to benchmark answers and grading criteria.

## Results and resume

Results are saved under `runs/`, organized by benchmark and subset where
applicable. They include task summaries, deliverables, adopted harnesses,
reflection notes, and evaluation results.

Repeat the same command to resume. Use a new `HARNESS2_RUNS_DIR` when changing
models or task selection. See [results and resume](docs/EXPERIMENTS.md#results-and-resume)
for details; each command also supports `--help`.

## Citation

```bibtex
@article{xu2026harness2,
  title   = {Harness$^2$: Recursive Agent Harnessing for an Open World},
  author  = {Xu, Ruiyao and Chen, Yanfei and CuiZhu, Zhongying and Dalvi Mishra, Bhavana and Ming, Yifei and Yu, Han and Han, Rujun and Lee, Chen-Yu and Pfister, Tomas},
  journal = {arXiv preprint arXiv:2610.14224},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.14224}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Harness² is licensed under [Apache 2.0](LICENSE). Third-party code under
[`third_party/`](third_party/README.md), benchmarks, and datasets retain their
upstream licenses; see
[third-party notices](dependencies/README.md).

## Disclaimer

This is not an officially supported Google product. This project is not eligible
for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
