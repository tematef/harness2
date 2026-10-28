# Skillbench v2: generic skill & suite benchmark, model matrix, one-page report, model-aware improver

## Overview

- Turn the pilot `skillbench` into a **generic, reusable pair of tools** for any AGY skill set:
  1. **Benchmark** (`harness2-sweep` + `skillbench-matrix` + `skillbench-report`): measures the
     effectiveness of a single skill (leave-one-out) or a whole skill family / suite (suite vs built-in
     behaviour vs optional further suites) in **quality, tokens, time and discovery**, across **several
     executor models**, and renders a **one-page summary** with a per-skill recommendation
     (`keep` / `fix` / `consider removing` / `needs harder tasks`) plus a trend across dated runs.
  2. **Improver** (`skillbench-improve`): proposes a better `SKILL.md` for weak skills, probes it with the
     **same executor model(s)** as the baseline, over `N` rounds, and gates it on quality, tokens and time.
- Problems it solves (from the pilot):
  - every result was `inconclusive` because tasks were at the ceiling (both arms 1.00) and only 2 tasks per
    skill existed → per-executor **hardness flag** (reported, never silently excluded), task scaffolder,
    4 harder tasks per skill;
  - the tool hard-codes `cc-thingz` (plugin allowlist, `KNOWN_SKILLS` redaction set, `ASK_QUESTION_SKILLS`,
    improver prompt text, bundled corpus path) → everything suite-specific moves into a **suite config**;
  - one executor model per sweep, no cross-model table → **model matrix** runner + report;
  - full dashboard but no "answer in one screen" → **Summary page + history**;
  - the live improver probe ran on a different model than the baseline → same-model probing enforced,
    multi-round, multi-model gate.
- Success criteria: a synthetic 2-plugin suite that is not cc-thingz runs through check → sweep → report →
  improve with zero code changes; the Summary page answers "keep / fix / remove", "family vs built-in" and
  "which model" on one screen; re-running an interrupted matrix schedules 0 new rollouts.
- Integrates with the existing Harness² fork (`tematef/harness2`, branch `skillbench`): same substrate,
  rollout/judge/resume machinery and report pipeline; new behaviour is additive and old flags keep working.

## Context (from discovery)

- Files/components involved:
  - `harness2/substrate/agy.py` — `DEFAULT_PLUGINS`, `plugin_allowlist()`, `plugins_dir()`,
    `ARMS = ("with","without")`, `stage_plugins(dest, arm, target_skill, skill_override)`, `exec_model()`
    (`SKILLBENCH_EXEC_MODEL`), `AgySubstrate.materialise`.
  - `harness2/benchmarks/skillbench.py` — `ASK_QUESTION_SKILLS` (hard-coded names), `KNOWN_SKILLS` (l. 473,
    feeds `_discovered_skills()` → `scrub_deliverable` redaction before judging), `SKILLBENCH_PROMPTS`,
    `TASKS_ROOT` (bundled corpus), `load_task`, `check_prompt`, `SkillbenchAdapter(arm=...)`, `score()`,
    `judge_model()` (`SKILLBENCH_JUDGE_MODEL`), `freeze`/`check_manifest`, `main` (`freeze|check` only —
    there is **no** `skillbench` console script; the benchmark entry point is `harness2-sweep`).
  - `harness2/benchmarks/tasks.py` — `discover("skillbench")` reads `skillbench.TASKS_ROOT` directly
    (must honour the resolved corpus root).
  - `harness2/cli/run_sweep.py` — `harness2-sweep --bench skillbench --arm …`; `_check_identity` builds
    the resume identity from `exec_model()`, `judge_model()`, `manifest_hash`.
  - `harness2/cli/judge.py` (`make_adapter(..., arm=args.arm)`, `arm_tag`) and `harness2/cli/_common.py`
    (arm validation, namespaces).
  - `harness2/cli/skillbench_report.py` + `skillbench_html.py` — `build_report(with_dir, without_dir)`,
    `bootstrap_ci`, `decide_verdict`, `render_markdown`, dependency-free HTML string templates.
  - `harness2/cli/skillbench_improve.py` — `BANNED_CLAUDE_TOOLS`, `build_improver_prompt` (mentions
    "`cc-thingz` plugin suite"), `select_weak_skills`, `build_evidence_packet`, `propose_skill`,
    `validate_candidate_skill`, `probe_candidate`, `evaluate_acceptance`, 3-way renderers; parser help
    text mentions cc-thingz (l. 1141).
  - `harness2/config.py` (env helpers, help text), `harness2/benchmarks/skillbench/tasks/…` (bundled corpus).
  - Docs under test: `docs/BENCHMARKS.md`, `docs/SETUP.md`, `.env.example`, `README.md`,
    `.agents/AGENTS.md` — `tests/test_skillbench_docs.py` asserts every `SKILLBENCH_*` key and arm flag
    is documented.
  - Tests: `tests/test_skillbench_{adapter,cli,corpus,docs,e2e,improve,judge,preflight,report,tasks,verify}.py`
    (306 tests, synthetic fixtures in `tmp_path`, stdlib stub for agy).
  - Presentation docs: `/usr/local/google/home/balandin/projects/presentations/skill-evaluations/`.
- Hard-coding inventory (every hit of `cc-thingz|make-plan|root-cause|brainstorm` in `harness2/` that is
  logic, not docstring): `agy.DEFAULT_PLUGINS`, `skillbench.ASK_QUESTION_SKILLS`, `skillbench.KNOWN_SKILLS`,
  `skillbench.SKILLBENCH_PROMPTS` (root-cause wording), `skillbench.TASKS_ROOT` via `tasks.discover`,
  `skillbench_improve.BANNED_CLAUDE_TOOLS`, `skillbench_improve.build_improver_prompt`, parser help text in
  `skillbench_improve.py` and `config.py`. Task 13 confirms this list is empty of logic hits.
- Conventions: env-driven config via `harness2.config.env*`; results `runs/.../results/<arm>/<task>/base<i>`;
  resume by fingerprint + identity; judge in isolated `HOME`; every CLI has `_build_parser()` + `main(argv)`;
  AGENTS.md: **no paid runs without explicit user approval**; docs must stay in sync (docs test).
- Dependencies: Jetski CLI, model IDs `opus-5.5-high`, `gemini-3.8-flash-high`, `fable-5.1-high`;
  the cc-thingz **checkout** at `~/projects/cc-thingz/plugins` (the `~/.gemini/config/plugins` entries are
  symlinks into it) is one concrete suite, never a default.

## Development Approach

- **testing approach**: Regular — code first, then unit tests with synthetic fixtures (existing harness2 style)
- complete each task fully before moving to the next; small, focused changes
- **CRITICAL: every task MUST include new/updated tests** for code changes in that task — success and
  error cases, as separate checklist items. Exception: Task 12 is a live data run; its check is schema
  validation of the produced `report.json` with the existing report loader.
- **CRITICAL: all tests must pass before starting next task** - no exceptions
  (`.venv/bin/python -m unittest discover -s tests` and `.venv/bin/ruff check .`)
- **CRITICAL: update this plan file when scope changes during implementation**
- **CRITICAL: no paid run without explicit user approval** — Tasks 11 and 12 start with a budget printout
  and wait for the user's OK; `--max-rollouts` is set from that approval
- backward compatibility: existing `report.json` consumers (v1 parity test), corpus layout and env vars keep
  working; new keys are additive (`schema_version: 2`)
- no suite / skill-name hard-coding anywhere in `harness2/`: suite-specific facts live in `suite.toml` or
  in the task corpus

## Testing Strategy

- **unit tests**: required for every task (synthetic run dirs / suites / reports under `tmp_path`, mocked
  runner where a live call would be needed); docs test extended for every new env var, flag and CLI
- **live validation**: Tasks 11–12 run the real pipeline once end to end, each behind an approval gate
- no e2e UI tests (CLI project); HTML checked by string assertions + `html.parser`

## Progress Tracking

- mark completed items with `[x]` immediately when done
- add newly discovered tasks with ➕ prefix
- document issues/blockers with ⚠️ prefix
- update plan if implementation deviates from original scope

## Solution Overview

Approaches considered:

- **A (chosen): extend `skillbench` in the harness2 fork.** Suite config, a second arm kind, a matrix
  runner and a summary layer on top of the proven rollout/judge/bootstrap code. Pros: reuses 306 tests,
  resume, judge isolation, statistics; smallest delta. Cons: harness2 carries more skill-specific CLI.
- **B: extract a standalone `skillbench` package.** Cleaner boundary, but a packaging project before any new
  measurement; revisit once more than ~5 suites exist.
- **C: in-session subagent benchmark** (the PDF approach). Rejected earlier: leaks host plugins into the
  baseline, no discovery/time-to-skill, no CIs.

Key design decisions:

1. **Suite config (`suite.toml`)** is the single place for anything suite-specific: `name`, `plugins_dir`
   (a checkout path), `plugins` (list or `"*"` = every dir with `plugin.json`), `tasks_root`,
   `banned_tools`, `families`, `matrix.executors`, `matrix.judge_for`. Selected by `--suite path` or
   `SKILLBENCH_SUITE`. The redaction set for the judge scrubber and the "asks the user" detection are
   derived from the staged plugins, not from constants. The bundled cc-thingz corpus becomes one example.
2. **Two arm kinds**: `skill` (existing leave-one-out `with` / `without` for `target_skill`) and `suite`
   (`suite-<name>` vs `baseline` = no plugins = built-in behaviour, plus any further suites given as
   `--suites path,...` — slot for a ported superpowers / spec-kit, not ported here). Invocation for suite
   arms = first read of any `SKILL.md` under the staged plugins.
3. **Hardness flag, per executor, never an exclusion**: a task is `informative` for executor `M` if the
   baseline arm (`without` / `baseline`) of **that executor** misses ≥ 1 criterion in ≥ 1 valid rep, with
   ≥ 2 valid baseline reps required to classify (else `unclassified`). Hardness comes from the same matrix
   cells that feed the lift — no extra rollouts — and ceiling tasks **stay in** the lift / CI / verdict
   (excluding them would condition on the control arm's luck and bias lift upward). The report shows
   `ceiling_tasks / total_tasks` per skill per executor and sets the recommendation
   `needs harder tasks` when > 50 % are at the ceiling. Authoring loop: `skillbench-matrix --arms baseline`
   on the cheapest executor plus `skillbench-report --hardness-only` tells the task author what to harden.
4. **Model matrix**: `executors = [...]`; each executor is paired with a judge whose model-id prefix
   (`<family>-`) differs from the executor's; an executor without a `judge_for` match is an error.
   Pairing `opus→gemini`, `gemini→fable`, `fable→gemini` (two executors share a judge; stated in the
   legend). Results namespaced `results/<executor>/<arm-label>/…`; resume identity gains `suite_hash`
   and executor; one rollout budget estimate printed and confirmed (`--yes`) before anything runs.
5. **Recommendation rule with precedence** (first match wins; the reason string names the rule):
   1. `needs harder tasks` — > 50 % of the skill's tasks at the ceiling for this executor;
   2. `consider removing` — verdict `hurts`, **or** `inconclusive` with tokens p50 ≥ +15 % **and** wall p50
      ≥ +15 % **and** no verify gain;
   3. `fix` — `inconclusive` with invocation < 80 % or `t_skill` p90 > 45 s, or the improver has an
      accepted candidate;
   4. `keep` — `helps`, or `inconclusive` with ≥ 15 % tokens or time saving, or none of the above.
6. **Improver v2**: baseline and candidate always probed with the same executor; `--models` probes on
   several executors and the gate must pass on **all** of them; `--rounds N` feeds the previous round's
   3-way numbers back into the evidence packet; `--apply` writes `best.patch` onto a git branch of the
   suite checkout (resolved symlinks, clean work tree required); candidate growth capped by `--max-growth`.
7. **No fifth console script for calibration**: hardness is a report function (`task_hardness`) plus a
   matrix `--arms baseline` option; `skillbench-cycle` stages are `matrix → report → history → improve`.

## Technical Details

- `suite.toml` (example at `harness2/benchmarks/skillbench/suites/cc-thingz.toml`):
  ```toml
  name = "cc-thingz"
  plugins_dir = "~/projects/cc-thingz/plugins"   # a git checkout; ~/.gemini/config/plugins symlinks here
  plugins = ["brainstorm", "planning", "review", "skill-eval", "thinking-tools", "workflow"]  # or "*"
  tasks_root = "~/projects/cc-thingz/benchmarks/skillbench"   # external corpus, lives with the skills
  banned_tools = ["AskUserQuestion", "EnterPlanMode", "TodoWrite", "Bash", "Read", "Write", "Edit"]
  [families]
  planning = ["brainstorm", "make-plan", "exec"]
  analysis = ["root-cause-investigator", "dialectic", "clarify", "wrong"]
  [matrix]
  executors = ["opus-5.5-high", "gemini-3.8-flash-high", "fable-5.1-high"]
  [matrix.judge_for]            # keyed by executor model-id prefix; same-prefix pairs are refused
  "opus-" = "gemini-3.8-flash-high"
  "gemini-" = "fable-5.1-high"
  "fable-" = "gemini-3.8-flash-high"
  ```
- `task.toml` additions (optional, backward compatible): `kind = "skill" | "suite"` (default `skill`),
  `family = "planning"` (suite tasks), `asks_user = true | false` (overrides the frontmatter detection),
  `min_baseline_fail = 1`.
- "Asks the user" detection: frontmatter `allowed-tools` of the skill's `SKILL.md` contains `ask_question`
  (verified to separate asking from non-asking skills in the current suite); `asks_user` in `task.toml`
  overrides; a missing `user_answers` on an asking task is a `skillbench-tasks check` error, not a
  `load_task` error.
- Judge redaction set: every `skills/<name>/` directory name found under the staged plugins of the arm (plus
  the suite name); replaces `KNOWN_SKILLS`.
- Corpus root resolution: `--tasks` → `suite.tasks_root` → `SKILLBENCH_TASKS_ROOT` → bundled corpus;
  `tasks.discover("skillbench")` and `freeze` / `check_manifest` all use the resolved root.
- Results layout: `runs/<run-id>/results/<executor>/<arm-label>/<task>/base<i>/`, `arm-label` ∈
  `with | without | suite-<name> | baseline`; `run.json` stores suite name, `suite_hash`, manifest hash,
  executor, judge. `harness2-sweep --arm` accepts the new labels; `_check_identity` includes `suite_hash`.
- `report.json` schema v2: `{schema_version: 2, suite: {name, hash}, models: [{executor, judge,
  skills: [...], suites: [...]}], summary: [{skill | family, model, verdict, recommendation, reason, lift,
  ci, tokens_delta_pct, wall_delta_pct, invocation_rate, t_skill_p50, t_skill_p90, ceiling_tasks,
  total_tasks, best_model}]}`; `summary.md` is the markdown twin of the Summary page. Single-model input
  yields the same per-skill numbers as v1 (parity test against a stored v1 fixture).
- `history.jsonl` lives next to `suite.toml` (per suite); one line per `(date, suite_hash, executor,
  skill | family)`; the Summary page draws an inline `<svg>` polyline per row (no JS dependency).
- Cost estimate: `rollouts = Σ_tasks arms × k × executors`; `est_minutes = rollouts × median wall of the
  last run for that executor (fallback 3 min) / parallelism`; printed as a table before the run.
- `skillbench-matrix` drives sweep and judge **in-process** (imports `run_sweep.main` / `judge.main` per
  executor with the executor's env set via a context manager), so resume, identity and namespaces are
  exactly those of the single-model CLIs.
- `--apply`: `plugins_dir` is `realpath`-resolved; must be inside a git work tree with a clean status;
  branch `skillbench/improve-<skill>-<date>`; commit message carries the 3-way numbers; never pushes;
  dry-run (print the commands) is the default.

## What Goes Where

- **Implementation Steps** (`[ ]` checkboxes): code, tests and docs in `harness2`; task corpus and suite
  file in the `cc-thingz` checkout; presentation docs update.
- **Post-Completion** (no checkboxes): porting other skill sets, scheduling the periodic run, quota.

## Implementation Steps

### Task 1: Suite config replaces every cc-thingz constant

**Files:**
- Create: `harness2/benchmarks/skillbench_suite.py`
- Create: `harness2/benchmarks/skillbench/suites/cc-thingz.toml`
- Modify: `harness2/substrate/agy.py`
- Modify: `harness2/benchmarks/skillbench.py`
- Modify: `harness2/cli/skillbench_improve.py`
- Modify: `harness2/cli/_common.py`
- Create: `tests/test_skillbench_suite.py`

- [x] add `SuiteConfig` dataclass + `load_suite(path | None)` (resolves `~`, `"*"` discovery via
      `plugin.json`, validates names, `families`, `banned_tools`, `matrix`, computes `suite_hash` over the
      staged plugin files) and `active_suite()` (`--suite` → `SKILLBENCH_SUITE` → env-fallback suite built
      from `SKILLBENCH_PLUGINS_DIR` / `SKILLBENCH_PLUGINS`, with a deprecation note)
- [x] `agy.plugin_allowlist()` / `plugins_dir()` read the active suite; delete `DEFAULT_PLUGINS` as a default
- [x] replace `KNOWN_SKILLS` / `_discovered_skills()` with `redaction_names(staged_plugins_dir, suite)`
      (every `skills/*/` dir + suite name); `scrub_deliverable` uses it
- [x] replace `ASK_QUESTION_SKILLS` with `skill_asks_user(skill_md_text)` (frontmatter `allowed-tools`
      only) + `task.toml` `asks_user` override; `load_task` no longer errors on missing `user_answers`
- [x] move suite-specific wording out of `SKILLBENCH_PROMPTS` and `build_improver_prompt` (use
      `suite.name`); `BANNED_CLAUDE_TOOLS` becomes the default of `suite.banned_tools`; fix parser help text
- [x] add `--suite` to `harness2-sweep`, `harness2-judge`, `skillbench-report`, `skillbench-improve` via
      `_common.py`; record suite name + `suite_hash` in `run.json`
- [x] write tests: suite load (explicit list, `"*"`, `"*"` on a dir with no `plugin.json` → error, bad
      name → error, missing plugin → error), env fallback, `redaction_names`, `skill_asks_user` true/false
      + override, `--suite` plumbing, no `cc-thingz` string in prompts
- [x] run tests - must pass before task 2

### Task 2: External task corpus and task scaffolder

**Files:**
- Modify: `harness2/benchmarks/skillbench.py`
- Modify: `harness2/benchmarks/tasks.py`
- Create: `harness2/cli/skillbench_tasks.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_skillbench_tasks.py`
- Create: `tests/test_skillbench_tasks_cli.py`

- [x] `tasks_root()` resolution (`--tasks` → `suite.tasks_root` → `SKILLBENCH_TASKS_ROOT` → bundled);
      `tasks.discover("skillbench")`, `freeze`, `check_manifest`, `SkillbenchAdapter` all use it
- [x] `skillbench-tasks new --skill X --id Y [--kind suite --family F]`: writes `task.toml`
      (`forbidden_terms` auto-filled from the skill name, script names found in its `SKILL.md`, `SKILL.md`;
      `asks_user` pre-set from frontmatter), `rubric.json` with 4 criterion stubs, `fixture/README.md`,
      `verify` stub
- [x] `skillbench-tasks check`: `check_prompt`, rubric size 4–8, `user_answers` present for asking tasks,
      fixture hash, `kind`/`family` validity; table output; non-zero exit on failure
- [x] register `skillbench-tasks` console script
- [x] write tests: resolution order (4 sources), `discover` honours it, scaffold output, `check` success
      and each failure path (in `test_skillbench_tasks_cli.py`); update `test_skillbench_tasks.py` for the
      moved `user_answers` check
- [x] run tests - must pass before task 3

### Task 3: Multi-namespace `build_report` and schema v2 skeleton

**Files:**
- Modify: `harness2/cli/skillbench_report.py`
- Modify: `harness2/cli/skillbench_html.py`
- Create: `tests/fixtures/skillbench/report_v1_fixture/` (small synthetic WITH/WITHOUT namespaces + expected v1 numbers)
- Create: `tests/test_skillbench_report_v2.py`

- [x] `build_report(cells: list[ReportCell])` where `ReportCell = (executor, judge, arm_label, ns_dir)`;
      the old `(with_dir, without_dir)` call becomes one cell pair with the executor from `run.json`
- [x] emit `schema_version: 2` with `suite`, `models[]` (one entry per executor, each holding the existing
      per-skill block) — no new metrics yet; `render_markdown` / HTML iterate over `models[]`
- [x] `--with-dir/--without-dir` keep working; new `--results-root runs/<id>/results` auto-discovers
      `<executor>/<arm-label>` namespaces
- [x] write tests: v1 parity (same per-skill lift, CI, verdict, medians from the stored fixture), two
      executors produce two `models[]` entries, mixed-judge cells for one executor → error, auto-discovery
- [x] run tests - must pass before task 4

### Task 4: Per-executor hardness flag

**Files:**
- Modify: `harness2/cli/skillbench_report.py`
- Modify: `harness2/cli/skillbench_html.py`
- Create: `tests/test_skillbench_hardness.py`

- [x] `task_hardness(baseline_records, min_baseline_fail=1, min_reps=2)` → `informative | ceiling |
      unclassified`, computed per executor from that executor's `without` / `baseline` cell
- [x] per skill per executor: `ceiling_tasks`, `total_tasks`, `unclassified_tasks`; ceiling tasks stay in
      lift / CI / verdict; a `needs harder tasks` flag when > 50 % are at the ceiling (consumed by Task 8's
      recommendation rule)
- [x] `skillbench-report --hardness-only`: prints the task × executor hardness table and exits (the
      corpus-authoring loop); task tab in HTML shows the hardness badge
- [x] legend text: what ceiling means and why such tasks are flagged, not dropped
- [x] write tests: classification incl. `min_reps`, per-executor independence (ceiling for one executor,
      informative for another), flag threshold, `--hardness-only` output, lift unchanged by ceiling tasks
- [x] run tests - must pass before task 5

### Task 5: Suite-mode arms (suite vs built-in vs further suites)

**Files:**
- Modify: `harness2/substrate/agy.py`
- Modify: `harness2/benchmarks/skillbench.py`
- Modify: `harness2/cli/_common.py`
- Modify: `harness2/cli/run_sweep.py`
- Modify: `harness2/cli/judge.py`
- Modify: `harness2/cli/skillbench_report.py`
- Create: `tests/test_skillbench_suite_arms.py`

- [x] `ArmSpec(kind="skill"|"suite", label, suite, target_skill)`; `parse_arm_label("with" | "without" |
      "baseline" | "suite-<name>")`; `_common.py` arm validation and `arm_tag` accept the new labels
- [x] `stage_plugins(dest, arm_spec, ...)`: nothing for `baseline`, the full named suite for `suite-<name>`,
      unchanged leave-one-out for `with` / `without`; `materialise` passes the spec
- [x] `SkillbenchAdapter` takes an `ArmSpec`; results root `results/<executor>/<label>/` (executor segment
      introduced here so Task 6 only adds the loop); `kind = "suite"` tasks are scheduled only for suite
      arms and `kind = "skill"` tasks only for skill arms; `harness2-sweep --arm` / `harness2-judge --arm`
      pass the label through
- [x] invocation + `t_skill` for suite arms: first `read_file` / `view_file` of any `SKILL.md` under the staged
      plugins dir; record the set of skills read
- [x] `build_report` `suites` section per family and per task: `suite-<name>` vs `baseline` lift, CI,
      verdict, tokens/time medians, skills-read histogram; `--suites a.toml,b.toml` renders one column per
      extra suite
- [x] write tests: staging per arm kind, label parsing errors, scheduling by kind, suite invocation parsing,
      `suites` section with 2 and 3 arms, sweep/judge accept new labels
- [x] run tests - must pass before task 6

### Task 6: Model matrix runner with cost estimate

**Files:**
- Create: `harness2/cli/skillbench_matrix.py`
- Modify: `harness2/cli/run_sweep.py`
- Modify: `harness2/substrate/agy.py`
- Modify: `pyproject.toml`
- Create: `tests/test_skillbench_matrix.py`

- [x] `skillbench-matrix --suite S [--executors a,b,c] [--k 3] [--arms skill|suite|both|baseline]
      [--parallel 3] [--max-rollouts N] [--yes]`: expands `(executor, arm, task, rep)` jobs; per executor
      sets exec/judge model through a context manager and calls `run_sweep.main` then `judge.main`
      in-process; `judge_for` lookup by prefix, same-prefix pair or missing mapping → error
- [x] `_check_identity` in `run_sweep.py` includes `suite_hash` and executor; namespaces per executor so
      resume skips finished cells across runs
- [x] `estimate_budget(jobs, history)`: rollouts, est. wall, est. tokens per executor and total; aborts
      without `--yes` / TTY confirmation; `--max-rollouts` hard cap; writes `matrix.json` (executors,
      judges, per-cell status, invalid counts by reason)
- [x] register `skillbench-matrix` console script
- [x] write tests: job expansion for each `--arms` value, judge same-family refused, missing mapping error,
      budget numbers from synthetic history, resume skips finished cells, `--max-rollouts` abort, identity
      mismatch on changed `suite_hash` (runner mocked)
- [x] run tests - must pass before task 7

### Task 7: Cross-model report tables

**Files:**
- Modify: `harness2/cli/skillbench_report.py`
- Modify: `harness2/cli/skillbench_html.py`
- Modify: `tests/test_skillbench_report_v2.py`

- [x] matrix table per skill: rows = executors; columns = lift [CI], verdict, tokens p50 W/WO (Δ %), wall
      p50 W/WO (Δ %), invocation, `t_skill` p50/p90, ceiling/total; `best_model` marker (highest lift,
      ties → fewest tokens)
- [x] model-level summary per executor: mean lift across skills, total tokens, total wall, invalid rate
- [x] markdown renderer mirrors both tables; HTML gets a "Models" tab; legend states the judge pairing
- [x] write tests: matrix values, best-model tie-break, model-level sums, markdown/HTML contain the cells
- [x] run tests - must pass before task 8

### Task 8: One-page Summary, recommendations and history trend

**Files:**
- Create: `harness2/cli/skillbench_summary.py`
- Modify: `harness2/cli/skillbench_report.py`
- Modify: `harness2/cli/skillbench_html.py`
- Create: `tests/test_skillbench_summary.py`

- [x] `recommend(row) -> (label, reason)` implementing the precedence in Solution Overview §5 (first match
      wins)
- [x] Summary = first page of `report.html` and `summary.md`: one row per skill (and per family in suite
      mode): recommendation badge, quality Δ with CI, tokens Δ %, time Δ %, discovery (`inv %`, `t_skill`
      p50), best model, reason; legend with the rule; dashboard tabs follow
- [x] `skillbench-report history append` / `history show` on `history.jsonl` next to `suite.toml`;
      idempotent per `(date, suite_hash, executor, skill)`; inline `<svg>` sparkline per row (lift, tokens Δ)
      with a tick where `suite_hash` changed
- [x] `--summary-only` writes just `summary.md` + minimal `summary.html`
- [x] write tests: each recommendation branch, precedence when several rules match, reason text, summary
      rows from a v2 report, history idempotence, sparkline series, `--summary-only` output set
- [x] run tests - must pass before task 9

### Task 9: Improver v2 — same-model probing, rounds, multi-model gate

**Files:**
- Modify: `harness2/cli/skillbench_improve.py`
- Modify: `tests/test_skillbench_improve.py`

- [ ] probe executor defaults to the executor recorded in the baseline cell; a different one is an error
      unless `--allow-model-mismatch`; the 3-way report prints the executor of every arm
- [ ] `--models a,b`: probe on each executor (reusing matrix cells for `WITHOUT` / `Current WITH`);
      `evaluate_acceptance` must pass on every executor; per-model gate row in the report
- [ ] `--rounds N`: round `r+1` packet includes round `r`'s candidate, 3-way numbers and gate reasons; early
      stop on acceptance; best round by (verify, mean, −tokens, −wall)
- [ ] skip skills flagged `needs harder tasks` for the chosen executor and say why; `--max-growth 50`
      (percent, flag with rationale in help) rejects candidates that grow beyond the threshold;
      `banned_tools` from the suite
- [ ] write tests: mismatch error and override, multi-model gate pass/fail mix, rounds feedback in packet,
      early stop, ceiling skip, growth guard pass/fail
- [ ] run tests - must pass before task 10

### Task 10: Improver weak-skill selection from the Summary and `--apply`

**Files:**
- Modify: `harness2/cli/skillbench_improve.py`
- Modify: `tests/test_skillbench_improve.py`

- [ ] `select_weak_skills` reads v2 `summary[]`: default `fix` + `consider removing`; `--all-weak` adds
      `inconclusive` with cost overhead; `--skill` forces one
- [ ] `--apply`: `realpath(plugins_dir)` must be inside a git work tree with clean status; branch
      `skillbench/improve-<skill>-<date>`; apply `best.patch`; commit with the 3-way numbers; never push;
      dry-run default prints the commands
- [ ] `improve_report` top "Decision" line per skill: `ACCEPTED on [models]` / `REJECTED: reason` + diff stat
- [ ] write tests: selection from v2 summary, dry-run output, symlinked `plugins_dir` resolves to the
      checkout, dirty tree refused, not-a-git-dir refused, commit message content (git repo in `tmp_path`)
- [ ] run tests - must pass before task 11

### Task 11: `skillbench-cycle` runner

**Files:**
- Create: `harness2/cli/skillbench_cycle.py`
- Modify: `pyproject.toml`
- Create: `tests/test_skillbench_cycle.py`

- [ ] `skillbench-cycle --suite S [--executors ...] [--k 3] [--arms both] [--improve] [--yes]`: stages
      `matrix → report → history append → improve (optional)`; prints the Summary table and the paths of
      `report.html`, `summary.md`, `improve_report.html`
- [ ] every stage resumable and skippable (`--from report`); `cycle.log` with per-stage wall time and
      token totals
- [ ] register `skillbench-cycle` console script
- [ ] write tests: stage ordering with mocked stages, `--from`, a failing stage stops the cycle and reports,
      `--improve` off by default
- [ ] run tests - must pass before task 12

### Task 12: Author the external corpus for the planning / analysis family (cc-thingz suite)

**Files:**
- Create: `/usr/local/google/home/balandin/projects/cc-thingz/benchmarks/skillbench/suite.toml`
- Create: `/usr/local/google/home/balandin/projects/cc-thingz/benchmarks/skillbench/tasks/<skill>/<task>/…`
  (4 tasks each for `brainstorm`, `make-plan`, `exec`, `root-cause-investigator`, `dialectic`,
  `code-review-refactor`; the 6 pilot tasks move here)
- Create: `/usr/local/google/home/balandin/projects/cc-thingz/benchmarks/skillbench/tasks/_suite/<task>/…`
  (4 `kind = "suite"` tasks, family `planning`: plan → implement → review end to end)
- Modify: `/usr/local/google/home/balandin/projects/cc-thingz/README.md`
- Modify: `tests/test_skillbench_corpus.py`

- [ ] **approval gate**: print the `estimate_budget` output for `skillbench-matrix --arms baseline
      --executors gemini-3.8-flash-high --k 2` on the drafted corpus and wait for explicit user approval
      before any rollout; set `--max-rollouts` from it
- [ ] write `suite.toml` for cc-thingz (checkout `plugins_dir`, families, executors, judge map, banned tools)
- [ ] scaffold tasks with `skillbench-tasks new`; each fixture has a real failure mode the baseline is
      likely to miss (hidden coupling, misleading symptom, multi-file plan dependency), 5–8 outcome-only
      criteria, `verify` + `verify_restore` where code is produced, scripted `user_answers` for asking tasks
- [ ] `skillbench-tasks check` passes for the whole corpus; `freeze` writes `manifest.json`
- [ ] run the approved baseline-only matrix + `skillbench-report --hardness-only`; harden any `ceiling`
      task until ≥ 3 informative tasks per skill and ≥ 3 informative suite tasks (re-approve if the
      re-run exceeds the approved budget)
- [ ] write tests: corpus loads under `load_tasks(root)` (prompt hygiene, rubric sizes, kinds, families,
      `asks_user` ↔ `user_answers`) — data-only, no live calls
- [ ] run tests - must pass before task 13

### Task 13: Live validation run and presentation update

**Files:**
- Modify: `/usr/local/google/home/balandin/projects/presentations/skill-evaluations/01-method.md`
- Modify: `/usr/local/google/home/balandin/projects/presentations/skill-evaluations/02-numbers.md`
- Modify: `/usr/local/google/home/balandin/projects/presentations/skill-evaluations/03-findings-and-fixes.md`
- Modify: `/usr/local/google/home/balandin/projects/presentations/skill-evaluations/build_deck.py`
- Create: `/usr/local/google/home/balandin/projects/presentations/skill-evaluations/report_v2.html`,
  `summary.md`, `improve_report_v2.html`, `data/report_v2.json`, `data/history.jsonl`

- [ ] **approval gate**: print the `estimate_budget` output for the full matrix (3 executors, `--arms both`,
      `--k 3`; ≈ 6 skills × 4 tasks × 2 arms × 3 reps × 3 + 4 suite tasks × 2 × 3 × 3 ≈ 504 rollouts) and
      a `--k 2` alternative; wait for explicit user approval of one of them and set `--max-rollouts`
- [ ] run `skillbench-cycle --suite cc-thingz … --improve --yes` with resume across quota windows; record
      the actual budget, wall time and invalid counts
- [ ] copy `report.html`, `summary.md`, `improve_report.html`, `report.json`, `history.jsonl` into the
      presentation folder; update numbers docs with the matrix table, the Summary recommendations and the
      per-model efficiency table; regenerate `index.html`
- [ ] record in `03-findings-and-fixes.md` every task that had to be hardened, every skill recommendation
      and every improver decision with its reason
- [ ] check: the produced `report.json` loads through the v2 report tests' loader (schema validation; no
      new unit tests — live data run)
- [ ] run tests - must pass before task 14

### Task 14: Verify acceptance criteria

- [ ] a synthetic 2-plugin suite under `tests/fixtures/skillbench/other-suite/` (not cc-thingz, different
      skill names, one asking skill) runs through `skillbench-tasks check`, staging for all four arm labels,
      report (mocked cells) and improve (mocked improver) with no code change — the no-hard-coding proof
- [ ] `grep -rn "cc-thingz\|make-plan\|root-cause\|brainstorm" harness2/` returns only docstrings, the
      example `suites/cc-thingz.toml` and the bundled corpus — no logic
- [ ] Summary page answers the three questions on one screen: per-skill recommendation, per-family
      suite-vs-built-in verdict, per-model efficiency; `summary.md` renders in GitHub markdown
- [ ] matrix resume: re-running the same command after an interruption schedules 0 new rollouts
- [ ] improver refuses a model mismatch, passes a multi-model gate in the mocked test, `--apply` dry-run
      prints a correct branch/commit
- [ ] run full test suite: `.venv/bin/python -m unittest discover -s tests` and `.venv/bin/ruff check .`
- [ ] every public function in the new modules has at least one test

### Task 15: [Final] Update documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/BENCHMARKS.md`
- Modify: `docs/SETUP.md`
- Modify: `.env.example`
- Modify: `.agents/AGENTS.md`
- Modify: `tests/test_skillbench_docs.py`

- [ ] README: "Benchmark any skill set" (write `suite.toml`, scaffold tasks, baseline-only hardness run,
      run the matrix, read the Summary) and "Improve a weak skill" sections; CLI table; recommendation rule
- [ ] `docs/BENCHMARKS.md` / `docs/SETUP.md` / `.env.example`: `SKILLBENCH_SUITE`, `SKILLBENCH_TASKS_ROOT`,
      new arm labels, `suite.toml` reference, schema v2, new console scripts
- [ ] `.agents/AGENTS.md`: suite-config rule, arm kinds, schema v2, "no suite hard-coding"; replace the
      "never touch the cc-thingz checkout" rule with "staging only reads the checkout; only `--apply`
      writes, on a branch, never on a dirty tree"; paid-run approval rule unchanged
- [ ] extend `tests/test_skillbench_docs.py` for every new env var, flag and CLI
- [ ] run tests; commit on branch `skillbench`, push to `fork`
- [ ] move this plan to the `completed/` directory next to it

## Post-Completion

*Items requiring manual intervention or external systems - no checkboxes, informational only*

**Manual verification:**
- read the Summary page with a colleague who did not build it: can they say in 1 minute which skills to
  keep, fix or remove, and which model to use?
- sanity-read 3 random judge verdicts per executor against the deliverables (judge noise is measured,
  judge bias is not)

**External follow-ups:**
- port one external skill set (superpowers or spec-kit) to AGY plugin format and run it through
  `--suites` for a three-way family comparison
- schedule the periodic run (e.g. monthly `skillbench-cycle … --yes` via `/schedule` or a sidecar) and
  arrange executor quota: the opus daily quota was exhausted once during the pilot; the matrix needs
  several windows or `--k 2`
- when more than ~5 suites exist, revisit approach B (extract `skillbench` as its own package)
