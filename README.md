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
> that adds a third executor, **AGY / Jetski** (`--substrate agy`), and a new
> benchmark, **skillbench**. skillbench measures whether the
> [`cc-thingz`](https://github.com/tematef/cc-thingz) agent skills actually
> help a local AGY (Jetski) agent. See
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
| skillbench | Leave-one-out skill lift (`cc-thingz` on AGY) | `harness2-check --bench skillbench` | [skillbench](docs/BENCHMARKS.md#skillbench) |

Choose one benchmark per command. WorkBuddy also takes `--subset office`,
`--subset web`, or `--subset code` on every command after setup. The
[benchmark guides](docs/BENCHMARKS.md) cover each one's data, domains, and
grading.

## AGY / Jetski executor and skillbench

This fork adapts Harness² to run a **local AGY (Jetski) CLI** as the agent
executor. Its first use is skillbench, an A/B benchmark that measures how much
each of the 18 `cc-thingz` skills improves the agent's results.

**How skillbench works**

* **Two arms per task.** Arm `with` stages the full allowlisted `cc-thingz`
  plugin suite. Arm `without` is identical except that the target skill's
  directory is removed (leave-one-out). Every other skill and every lifecycle
  hook stays the same, so the only difference is the skill under test.
* **Outcome lift.** A separate AGY judge, running in an empty `HOME` with no
  plugins, grades each deliverable against a binary rubric. The report gives
  the paired `with` − `without` rubric score per skill, with a 95% two-level
  bootstrap confidence interval and a verdict (`helps`, `hurts`,
  `inconclusive`, or `insufficient data`).
* **Time-to-skill.** For the `with` arm, the report gives the invocation rate
  and the p50/p90 seconds from process start to the agent's first read of the
  target `SKILL.md`.
* **Isolation.** Each rollout runs in a fresh git repository with a throwaway
  `HOME` that contains only the staged plugins. Clipboard and review tools
  (`xclip`, `xsel`, `wl-copy`, `pbcopy`, `revdiff`) are replaced with shims
  that write to `<run>/side_effects/`. Skills that would ask the user a
  question get answers from an identical simulated-user preamble in both arms.
* **Bases only.** skillbench runs reference executions only (`--bases-only` is
  implied); the Propose–Probe–Compose recursion is unchanged.

**Run it with the local Jetski CLI**

The Google-internal Jetski CLI is used as `agy`. Point `HARNESS2_AGY_BIN` at
the binary itself; a shell function or alias such as `jet` cannot be launched
by the harness.

```bash
export HARNESS2_AGY_BIN=/google/bin/releases/jetski-devs/tools/cli
export SKILLBENCH_PLUGINS_DIR=~/projects/cc-thingz/plugins
export SKILLBENCH_EXEC_MODEL=opus-5.5-high     # executor
export SKILLBENCH_JUDGE_MODEL=fable-5.1-high   # judge

harness2-check --bench skillbench --substrate agy
PILOT=1 K=3 bash scripts/run_skillbench.sh     # 3-skill paid pilot
K=3 bash scripts/run_skillbench.sh             # full corpus (ask before spending)
```

Model IDs come from `jet models`; they include the effort level, so
`SKILLBENCH_EFFORT` is optional. Internal Jetski authenticates without
credential files, so `SKILLBENCH_AUTH_FILES` is optional too.

**Verified against real Jetski output.** `scripts/agy_spike.sh` records real
`stream-json` traces into `tests/fixtures/agy/real/`. The decoder, the
synthetic test fixtures, and the model guard follow those traces; the findings
are in [docs/skillbench-spike.md](docs/skillbench-spike.md).

**Reports.** `skillbench-report` (added by this fork) writes
`runs/skillbench/report.md`, `report.json` and a self-contained `report.html`
dashboard with the per-skill lift, confidence interval, verdict, invocation rate,
time-to-skill, deterministic test pass rate (`verify`), diff size, tokens, and
side-by-side WITH / WITHOUT answers. The upstream
`harness2-report --bench skillbench --arm <with|without>` prints one arm's plain
mean rubric score as a cross-check.

See the [skillbench guide](docs/BENCHMARKS.md#skillbench) for the task format,
metrics, and cost gates, and the [AGY executor setup](docs/SETUP.md#agy-executor)
for all environment variables.

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
