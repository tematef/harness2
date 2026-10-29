# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render the skillbench leave-one-out report (`report.json`, `report.md` and `report.html`).

Reads judged arm namespaces produced by `harness2 --bench skillbench` and
`harness2-judge --bench skillbench --bases`, one `with` / `without` pair per executor
(`--with-dir` / `--without-dir`, the conventional `adapt/` namespaces of `--k` / `--seed`, or
`--results-root runs/<id>/results` holding `<executor>/<arm-label>/` namespaces), and
computes per executor:

- Outcome lift: per-task paired delta `d_t = mean_k(frac WITH) - mean_k(frac WITHOUT)` over valid
  graded reps, `lift = mean_t(d_t)`, 95% CI from a two-level bootstrap (resample tasks, then reps
  within each task; 10 000 draws; fixed seed), and verdict (`helps`, `hurts`, `inconclusive`, or
  `insufficient data` when fewer than 3 tasks — 2 in pilot mode — have >= 2 valid graded reps in
  both arms).
- Time-to-skill: invocation rate (`invoked / valid WITH reps`) and p50 / p90 of `t_skill_s` over
  invoked valid WITH reps.
- Deterministic checks and cost (reported next to the rubric lift, never mixed into it): pass
  rate of the task's `verify` command per arm (`verify.json`), median lines changed in
  `diff.patch`, median total tokens and wall time per valid rep.
- Diagnostics: invalid rollout counts per reason, judge-noise rate (criterion-level disagreement
  across repeated judgments when `SKILLBENCH_JUDGE_REPEAT >= 2`), model-verification status, and
  the statistical-power note.
- Hardness, per executor and never an exclusion: from that executor's control arm alone a task is
  `informative` (the control arm misses >= `min_baseline_fail` criteria in >= 1 valid rep),
  `ceiling` (every valid control rep passes the whole rubric) or `unclassified` (< 2 valid
  control reps). Ceiling tasks stay in the lift / CI / verdict; the per-skill row carries
  `ceiling_tasks / total_tasks` and `needs_harder_tasks` (> 50 % at the ceiling), the flag
  Task 8's recommendation rule consumes. `--hardness-only` prints the task x executor hardness
  table from control-arm cells alone (the corpus-authoring loop) and writes nothing.

`report.json` is schema v2: `{schema_version: 2, suite: {name, hash}, models: [{executor, judge,
skills, by_skill, ...}], ...}`; the first executor's block is also mirrored at the top level, so
a single-executor report reads exactly like schema v1. `report.html` shows the same data as a
self-contained dashboard (KPI cards, per-skill and per-task tables, criteria pass rates,
side-by-side WITH / WITHOUT answers, diagnostics), one section per executor.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import config
from ..benchmarks.skillbench import (
    DEFAULT_JUDGE_MODEL,
    JUDGMENTS_FILE,
    SKILL_TIMING_FILE,
    VERIFY_FILE,
    ManifestError,
    TaskError,
    _default_runs_root,
    _env_pilot,
    check_manifest,
    diff_stats,
    judge_model,
    load_tasks,
)
from ..benchmarks.skillbench_suite import SuiteConfig, active_suite, load_suite
from ..core import schedule as SCHED
from ..substrate.agy import (
    ARM_LABEL_HELP,
    BASELINE_ARM,
    DEFAULT_EXEC_MODEL,
    SUITE_ARM_PREFIX,
    arm_kind,
    exec_model,
    read_events,
    skills_read,
    suite_arm_label,
)
from . import _common as C
from .run_sweep import RUN_JSON
from .skillbench_html import render_html
from .skillbench_summary import (
    REC_CONSIDER_REMOVING as REC_CONSIDER_REMOVING,
)
from .skillbench_summary import (
    REC_FIX as REC_FIX,
)
from .skillbench_summary import (
    REC_KEEP as REC_KEEP,
)
from .skillbench_summary import (
    REC_NEEDS_HARDER_TASKS as REC_NEEDS_HARDER_TASKS,
)
from .skillbench_summary import (
    RECOMMENDATION_RULE_NOTE,
    _delta_pct,
    _str_or_none,
    append_history,
    build_summary,
    default_history_path,
    read_history,
    recommend,
    render_history_markdown,
    render_summary_markdown,
    write_summary,
)

SCHEMA_VERSION = 2
DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
MIN_TASKS_FULL = 3
MIN_TASKS_PILOT = 2
MIN_VALID_REPS_PER_ARM = 2

# Hardness flag: computed per executor from its control arm; a label, never an exclusion.
HARDNESS_INFORMATIVE = "informative"
HARDNESS_CEILING = "ceiling"
HARDNESS_UNCLASSIFIED = "unclassified"
HARDNESS_LABELS = (HARDNESS_INFORMATIVE, HARDNESS_CEILING, HARDNESS_UNCLASSIFIED)
DEFAULT_MIN_BASELINE_FAIL = 1
MIN_BASELINE_REPS = 2
# A skill whose ceiling share (ceiling_tasks / total_tasks) exceeds this is flagged.
CEILING_FLAG_SHARE = 0.5
NEEDS_HARDER_TASKS = "needs harder tasks"
HARDNESS_NOTE = (
    "Hardness is judged per executor from its control arm (WITHOUT / baseline) alone: a task is "
    "`informative` when that arm misses at least `min_baseline_fail` criteria (default 1) in at "
    "least one valid graded rep, `ceiling` when every valid control rep passes the whole rubric "
    "(the task cannot show a lift for this executor), and `unclassified` with fewer than "
    f"{MIN_BASELINE_REPS} valid control reps. Ceiling tasks are flagged, not dropped: they stay "
    "in the lift, CI and verdict, because excluding them would select tasks on the control arm's "
    "luck and bias the lift upward. A skill with more than "
    f"{CEILING_FLAG_SHARE * 100:.0f} % of its tasks at the ceiling is flagged "
    f"`{NEEDS_HARDER_TASKS}` for that executor."
)

MODEL_UNVERIFIED = "model unverified (agy reports no model)"
POWER_NOTE = (
    "With 4 tasks per skill, the task-level bootstrap has only 35 distinct resamples, so only "
    "large lifts (roughly >= 0.2 of the rubric) will get a `helps`/`hurts` verdict; most skills "
    "are expected to land `inconclusive` in the first full run."
)

# Identity keys that the two arms of one executor must agree on (where both are known).
_IDENTITY_KEYS = ("executor", "judge", "effort", "manifest_hash", "suite_hash")
# Identity keys every cell of the whole report must agree on (where known).
_GLOBAL_IDENTITY_KEYS = ("manifest_hash", "suite_hash")
FINAL_EXCERPT_CHARS = 6000

# Arm pairs the per-skill block is computed from: treatment label -> control label.
ARM_PAIRS: dict[str, str] = {"with": "without"}
CONTROL_LABELS = frozenset({"without", BASELINE_ARM})
ARM_LABELS = frozenset(ARM_PAIRS) | CONTROL_LABELS

# Per-executor keys mirrored at the top level of `report.json` for schema-v1 consumers.
_LEGACY_MODEL_KEYS = (
    "requested_exec_model",
    "actual_exec_model",
    "exec_model_verified",
    "requested_judge_model",
    "actual_judge_model",
    "judge_model_verified",
    "model_verified",
    "judge_noise_rate",
    "judge_noise_disagreed",
    "judge_noise_total",
    "invalid_total",
    "invalid_by_reason",
    "skills",
    "by_skill",
    "suites",
    "by_family",
    "suite_arms",
)


@dataclass(frozen=True)
class ReportCell:
    """One judged arm namespace of the matrix: `arm_label` run on `executor` and graded by
    `judge`, living at `ns_dir`. `executor` / `judge` may be `None`: they are then read from
    the namespace's `run.json` / `config.json`, else from the rollouts, else from the env."""

    executor: str | None
    judge: str | None
    arm_label: str
    ns_dir: Path


def cells_from_dirs(with_ns_dir: Path, without_ns_dir: Path) -> list[ReportCell]:
    """The schema-v1 call: one `with` / `without` pair; each cell's executor and judge come
    from its own `run.json` / `config.json`, else from the other namespace's."""
    with_meta = _load_cell_meta(Path(with_ns_dir))
    without_meta = _load_cell_meta(Path(without_ns_dir))

    def pick(own: Mapping[str, Any], other: Mapping[str, Any], key: str) -> str | None:
        return _str_or_none(own.get(key)) or _str_or_none(other.get(key))

    return [
        ReportCell(
            pick(with_meta, without_meta, "executor"),
            pick(with_meta, without_meta, "judge"),
            "with",
            Path(with_ns_dir),
        ),
        ReportCell(
            pick(without_meta, with_meta, "executor"),
            pick(without_meta, with_meta, "judge"),
            "without",
            Path(without_ns_dir),
        ),
    ]


def discover_cells(results_root: Path) -> list[ReportCell]:
    """Auto-discover `<results_root>/<executor>/<arm-label>/` namespaces that have been judged
    (`base_judged.json` present). The executor is the directory name; `run.json` may refine
    the judge. Raises `FileNotFoundError` when nothing judged is found."""
    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"results root {root} does not exist")
    cells: list[ReportCell] = []
    for exec_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for arm_dir in sorted(p for p in exec_dir.iterdir() if p.is_dir()):
            if not (arm_dir / "base_judged.json").is_file():
                continue
            meta = _load_cell_meta(arm_dir)
            label = _str_or_none(meta.get("arm")) or arm_dir.name
            cells.append(
                ReportCell(exec_dir.name, _str_or_none(meta.get("judge")), label, arm_dir)
            )
    if not cells:
        raise FileNotFoundError(
            f"no judged namespaces under {root}; expected <executor>/<arm-label>/base_judged.json"
        )
    return cells


@dataclass(frozen=True)
class _RepRecord:
    task_id: str
    skill: str
    label: str
    run_id: str
    valid: bool
    invalid_reason: str | None
    invoked: bool
    t_skill_s: float | None
    requested_model: str | None
    actual_model: str | None
    frac: float | None
    judgments: tuple[dict[str, Any], ...] = ()
    judge_model: str | None = None
    judge_actual_model: str | None = None
    has_judgments_file: bool = False
    verify_passed: bool | None = None
    diff: dict[str, int] | None = None
    total_tokens: int | None = None
    t_total_s: float | None = None
    final_excerpt: str = ""
    # Criteria the judge marked failed (`n_criteria - n_passed`); None when ungraded / invalid.
    n_missed: int | None = None
    # Sorted unique skill names whose `SKILL.md` was read in the rollout.
    skills_read: tuple[str, ...] = ()

    @property
    def diff_lines(self) -> int | None:
        return None if self.diff is None else self.diff["added"] + self.diff["removed"]

    def detail(self) -> dict[str, Any]:
        """Per-rep entry of `report.json` (`tasks.<id>.reps.<arm>[]`)."""
        criteria = self.judgments[0].get("criteria") if self.judgments else None
        return {
            "label": self.label,
            "run_id": self.run_id,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "invoked": self.invoked,
            "t_skill_s": self.t_skill_s,
            "skills_read": list(self.skills_read),
            "frac": self.frac,
            "n_missed": self.n_missed,
            "verify_passed": self.verify_passed,
            "diff": self.diff,
            "total_tokens": self.total_tokens,
            "t_total_s": self.t_total_s,
            "criteria": criteria if isinstance(criteria, list) else [],
            "final_excerpt": self.final_excerpt,
        }


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile `q` in [0, 1] over non-empty ascending `sorted_vals`."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("cannot compute quantile of empty sequence")
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def bootstrap_ci(
    paired_tasks: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Two-level bootstrap 95% CI `(ci_low, ci_high)` over paired task `(with_fracs, without_fracs)`.

    In each draw, resamples `len(paired_tasks)` tasks with replacement, then within each sampled
    task resamples `len(with_fracs)` WITH reps and `len(without_fracs)` WITHOUT reps with
    replacement. Reproducible for a fixed `seed`.
    """
    if draws < 1:
        raise ValueError(f"draws must be >= 1, got {draws}")
    tasks = [(tuple(float(x) for x in w), tuple(float(x) for x in u)) for w, u in paired_tasks]
    if not tasks or any(not w or not u for w, u in tasks):
        raise ValueError("paired_tasks must be non-empty with >= 1 valid rep in each arm")
    rng = random.Random(seed)
    n_tasks = len(tasks)
    boot_means: list[float] = []
    for _ in range(draws):
        total = 0.0
        for _ in range(n_tasks):
            w, u = tasks[rng.randrange(n_tasks)]
            nw, nu = len(w), len(u)
            w_mean = sum(w[rng.randrange(nw)] for _ in range(nw)) / nw
            u_mean = sum(u[rng.randrange(nu)] for _ in range(nu)) / nu
            total += w_mean - u_mean
        boot_means.append(total / n_tasks)
    boot_means.sort()
    return (_quantile(boot_means, 0.025), _quantile(boot_means, 0.975))


def decide_verdict(
    ci_low: float | None,
    ci_high: float | None,
    *,
    n_qualifying_tasks: int,
    pilot: bool = False,
) -> str:
    """Verdict from the 95% CI and the number of tasks with >= 2 valid graded reps in both arms."""
    min_tasks = MIN_TASKS_PILOT if pilot else MIN_TASKS_FULL
    if n_qualifying_tasks < min_tasks or ci_low is None or ci_high is None:
        return "insufficient data"
    if ci_low > 0.0:
        return "helps"
    if ci_high < 0.0:
        return "hurts"
    return "inconclusive"


def task_hardness(
    baseline_records: Iterable[Any],
    min_baseline_fail: int = DEFAULT_MIN_BASELINE_FAIL,
    min_reps: int = MIN_BASELINE_REPS,
) -> str:
    """Hardness of one task for one executor from that executor's control-arm reps
    (`without` / `baseline`), each with `.valid` and `.n_missed` (criteria failed):

    - `unclassified` — fewer than `min_reps` valid graded control reps;
    - `informative` — some valid control rep misses >= `min_baseline_fail` criteria;
    - `ceiling` — every valid control rep passes the whole rubric (for this executor the task
      cannot show a lift).

    A label only: ceiling tasks are never removed from the lift, CI or verdict.
    """
    if isinstance(min_baseline_fail, bool) or min_baseline_fail < 1:
        raise ValueError(f"min_baseline_fail must be >= 1, got {min_baseline_fail!r}")
    if isinstance(min_reps, bool) or min_reps < 1:
        raise ValueError(f"min_reps must be >= 1, got {min_reps!r}")
    missed = baseline_missed(baseline_records)
    if len(missed) < min_reps:
        return HARDNESS_UNCLASSIFIED
    if any(m >= min_baseline_fail for m in missed):
        return HARDNESS_INFORMATIVE
    return HARDNESS_CEILING


def baseline_missed(records: Iterable[Any]) -> list[int]:
    """Criteria missed per valid graded rep, in record order."""
    return [
        int(r.n_missed)
        for r in records
        if getattr(r, "valid", False) and getattr(r, "n_missed", None) is not None
    ]


def skill_hardness(task_labels: Iterable[str]) -> dict[str, Any]:
    """Per-skill (per-executor) hardness counts from its tasks' labels: `ceiling_tasks`,
    `informative_tasks`, `unclassified_tasks`, `total_tasks`, `ceiling_share`
    (`ceiling / total`, None without tasks) and `needs_harder_tasks` (share > 50 %)."""
    labels = list(task_labels)
    total = len(labels)
    ceiling = sum(1 for lbl in labels if lbl == HARDNESS_CEILING)
    share = (ceiling / total) if total else None
    return {
        "ceiling_tasks": ceiling,
        "informative_tasks": sum(1 for lbl in labels if lbl == HARDNESS_INFORMATIVE),
        "unclassified_tasks": sum(1 for lbl in labels if lbl == HARDNESS_UNCLASSIFIED),
        "total_tasks": total,
        "ceiling_share": share,
        "needs_harder_tasks": share is not None and share > CEILING_FLAG_SHARE,
    }


Thresholds = tuple[int, dict[str, int]]


def _thresholds(spec: int | Mapping[str, int] | None) -> Thresholds:
    """`(default, per_task)` `min_baseline_fail` thresholds: an int applies to every task, a
    mapping is per task id (others default to 1), None reads the task corpus."""
    if spec is None:
        return (DEFAULT_MIN_BASELINE_FAIL, corpus_thresholds())
    if isinstance(spec, Mapping):
        per_task = {str(tid): int(v) for tid, v in spec.items()}
        bad = {tid: v for tid, v in per_task.items() if v < 1}
        if bad:
            raise ValueError(f"min_baseline_fail must be >= 1 per task, got {bad}")
        return (DEFAULT_MIN_BASELINE_FAIL, per_task)
    if isinstance(spec, bool) or int(spec) < 1:
        raise ValueError(f"min_baseline_fail must be >= 1, got {spec!r}")
    return (int(spec), {})


def corpus_thresholds() -> dict[str, int]:
    """`task_id -> min_baseline_fail` from the resolved task corpus; empty when the corpus
    cannot be loaded (the report then falls back to the default of 1 for every task)."""
    try:
        tasks = load_tasks()
    except (TaskError, config.ConfigError, OSError, ValueError):
        return {}
    return {tid: int(t.min_baseline_fail) for tid, t in tasks.items()}


def corpus_domains() -> dict[str, str]:
    """`task_id -> domain` (skill name for `kind="skill"`, family for `kind="suite"`) from the
    resolved task corpus; empty when the corpus cannot be loaded."""
    try:
        tasks = load_tasks()
    except (TaskError, config.ConfigError, OSError, ValueError):
        return {}
    return {tid: str(t.domain) for tid, t in tasks.items()}


def hardness_rows(
    control_recs: Sequence[_RepRecord], thresholds: Thresholds
) -> dict[str, dict[str, Any]]:
    """`task_id -> {skill, hardness, baseline_missed, valid_reps, min_baseline_fail}` from one
    executor's control-arm records (sorted by task id)."""
    default, per_task = thresholds
    by_task: dict[str, list[_RepRecord]] = {}
    for r in control_recs:
        by_task.setdefault(r.task_id, []).append(r)
    rows: dict[str, dict[str, Any]] = {}
    for tid in sorted(by_task):
        recs = by_task[tid]
        thr = per_task.get(tid, default)
        missed = baseline_missed(recs)
        rows[tid] = {
            "skill": recs[0].skill,
            "hardness": task_hardness(recs, min_baseline_fail=thr),
            "baseline_missed": missed,
            "valid_reps": len(missed),
            "min_baseline_fail": thr,
        }
    return rows


def judge_noise(
    judgments_lists: Iterable[Sequence[Mapping[str, Any]]],
) -> tuple[float | None, int, int]:
    """Criterion-level disagreement `(rate, disagreed, total)` across repeated judgments."""
    disagreed = 0
    total = 0
    for judgments in judgments_lists:
        if not isinstance(judgments, Sequence) or len(judgments) < 2:
            continue
        maps: list[dict[str, bool]] = []
        ordered_ids: list[str] = []
        for idx, j in enumerate(judgments):
            if not isinstance(j, Mapping):
                continue
            raw_criteria = j.get("criteria")
            if not isinstance(raw_criteria, list):
                continue
            cmap: dict[str, bool] = {}
            for item in raw_criteria:
                if not isinstance(item, Mapping):
                    continue
                cid = item.get("id")
                passed = item.get("pass")
                if isinstance(cid, str) and isinstance(passed, bool):
                    cmap[cid] = passed
                    if idx == 0:
                        ordered_ids.append(cid)
            if cmap:
                maps.append(cmap)
        if len(maps) < 2:
            continue
        for cid in ordered_ids:
            outcomes = [m[cid] for m in maps if cid in m]
            if len(outcomes) >= 2:
                total += 1
                if len(set(outcomes)) > 1:
                    disagreed += 1
    if total == 0:
        return (None, 0, 0)
    return (disagreed / total, disagreed, total)


def _score_to_frac(score: Any) -> float | None:
    counts = _score_counts(score)
    return None if counts is None else counts[0] / counts[1]


def _score_to_missed(score: Any) -> int | None:
    counts = _score_counts(score)
    return None if counts is None else counts[1] - counts[0]


def _score_counts(score: Any) -> tuple[int, int] | None:
    """`(n_passed, n_criteria)` of a judge score, or None when it is not a graded score."""
    if not isinstance(score, Mapping):
        return None
    n_passed = score.get("n_passed")
    n_criteria = score.get("n_criteria")
    if (
        isinstance(n_passed, bool)
        or isinstance(n_criteria, bool)
        or not isinstance(n_passed, int)
        or not isinstance(n_criteria, int)
        or n_criteria <= 0
    ):
        return None
    return (n_passed, n_criteria)


def _read_json(path: Path, *, strict: bool = False) -> Any:
    """Load a JSON file; when `strict=False` (default), missing/unreadable/non-dict files
    return `{}`."""
    if strict:
        return json.loads(path.read_text(encoding="utf-8"))
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


_META_FILES = ("config.json", RUN_JSON)


def _load_cell_meta(ns_dir: Path) -> dict[str, Any]:
    """What a namespace was run with: `config.json` (resume identity: executor, judge, effort,
    manifest_hash, seed) overlaid with `run.json` (adds suite, suite_hash, arm). Unreadable or
    missing files contribute nothing; `None` values never override a known one."""
    meta: dict[str, Any] = {}
    for name in _META_FILES:
        loaded = _read_json(Path(ns_dir) / name)
        meta.update({k: v for k, v in loaded.items() if v is not None})
    return meta


def _load_arm_records(ns_dir: Path) -> tuple[dict[str, Any], list[_RepRecord]]:
    ns_dir = Path(ns_dir)
    if not ns_dir.is_dir():
        raise FileNotFoundError(
            f"namespace directory {ns_dir} does not exist; run `harness2 --bench skillbench` first"
        )
    judged_path = ns_dir / "base_judged.json"
    if not judged_path.is_file():
        raise FileNotFoundError(
            f"{judged_path} does not exist; run `harness2-judge --bench skillbench --bases` first"
        )
    try:
        raw_judged = _read_json(judged_path, strict=True)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{judged_path}: unreadable JSON ({e})") from None
    if not isinstance(raw_judged, list):
        raise ValueError(f"{judged_path}: expected a JSON list of judged base rows")

    scores_by_key: dict[tuple[str, str], Any] = {}
    scores_by_run_id: dict[str, Any] = {}
    for row in raw_judged:
        if not isinstance(row, Mapping):
            continue
        tid = str(row.get("task") or row.get("task_id") or "")
        lbl = str(row.get("label") or "")
        rid = str(row.get("run_id") or "")
        sc = row.get("score")
        if tid and lbl:
            scores_by_key[(tid, lbl)] = sc
        if rid:
            scores_by_run_id[rid] = sc

    cfg = _load_cell_meta(ns_dir)
    known_domains = corpus_domains()

    records: list[_RepRecord] = []
    for summary_path in sorted(ns_dir.glob("*/summary.json")):
        try:
            summary = _read_json(summary_path, strict=True)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"{summary_path}: unreadable JSON ({e})") from None
        if not isinstance(summary, Mapping):
            continue
        task_id = str(summary.get("task_id") or "")
        if not task_id:
            continue
        raw_dom = summary.get("family") or summary.get("domain")
        if isinstance(raw_dom, str) and raw_dom and raw_dom != "_suite":
            skill = raw_dom
        elif task_id in known_domains:
            skill = known_domains[task_id]
        else:
            skill = task_id.split("/", 1)[0] if "/" in task_id else task_id
        for entry in summary.get("base") or []:
            if not isinstance(entry, Mapping):
                continue
            label = str(entry.get("label") or "")
            run_id = str(entry.get("run_id") or "")
            raw_run_dir = entry.get("run_dir")
            run_dir = Path(raw_run_dir) if isinstance(raw_run_dir, str) and raw_run_dir else None
            if run_dir is not None and not run_dir.is_absolute():
                run_dir = ns_dir / run_dir  # stored fixtures keep run dirs inside the namespace
            records.append(
                _load_rep(
                    task_id=task_id,
                    skill=skill,
                    label=label,
                    run_id=run_id,
                    run_dir=run_dir,
                    scores_by_key=scores_by_key,
                    scores_by_run_id=scores_by_run_id,
                )
            )
    return cfg, records


def _load_rep(
    *,
    task_id: str,
    skill: str,
    label: str,
    run_id: str,
    run_dir: Path | None,
    scores_by_key: Mapping[tuple[str, str], Any],
    scores_by_run_id: Mapping[str, Any],
) -> _RepRecord:
    valid = False
    invalid_reason: str | None = "missing run_dir"
    invoked = False
    t_skill_s: float | None = None
    req_model: str | None = None
    act_model: str | None = None
    judgments: tuple[dict[str, Any], ...] = ()
    j_model: str | None = None
    j_act_model: str | None = None
    has_judgments_file = False
    verify_passed: bool | None = None
    diff: dict[str, int] | None = None
    total_tokens: int | None = None
    t_total_s: float | None = None
    final_excerpt = ""
    s_read: tuple[str, ...] = ()

    if run_dir is not None:
        timing_path = run_dir / SKILL_TIMING_FILE
        if not timing_path.is_file():
            valid = False
            invalid_reason = "missing skill_timing.json"
        else:
            try:
                timing = _read_json(timing_path, strict=True)
            except (OSError, json.JSONDecodeError):
                timing = None
            if not isinstance(timing, Mapping):
                valid = False
                invalid_reason = "unreadable skill_timing.json"
            else:
                valid = timing.get("valid") is True
                raw_reason = timing.get("invalid_reason")
                invalid_reason = (
                    None
                    if valid
                    else (
                        str(raw_reason)
                        if isinstance(raw_reason, str) and raw_reason.strip()
                        else "invalid rollout"
                    )
                )
                invoked = timing.get("invoked") is True
                raw_t = timing.get("t_skill_s")
                if isinstance(raw_t, (int, float)) and not isinstance(raw_t, bool):
                    t_skill_s = float(raw_t)
                raw_req = timing.get("requested_model")
                if isinstance(raw_req, str) and raw_req:
                    req_model = raw_req
                raw_act = timing.get("actual_model")
                if isinstance(raw_act, str) and raw_act:
                    act_model = raw_act
                raw_tok = timing.get("total_tokens")
                if isinstance(raw_tok, int) and not isinstance(raw_tok, bool):
                    total_tokens = raw_tok
                raw_tt = timing.get("t_total_s")
                if isinstance(raw_tt, (int, float)) and not isinstance(raw_tt, bool):
                    t_total_s = float(raw_tt)
                raw_sr = timing.get("skills_read")
                if isinstance(raw_sr, list):
                    s_read = tuple(sorted({str(s) for s in raw_sr if isinstance(s, str) and s}))
                elif (run_dir / "events.jsonl").is_file():
                    s_read = tuple(
                        skills_read(
                            read_events(run_dir / "events.jsonl"),
                            plugins_dir=run_dir / "home" / ".gemini" / "config" / "plugins"
                            if (run_dir / "home" / ".gemini" / "config" / "plugins").exists()
                            else None,
                        )
                    )

        v_data = _read_json(run_dir / VERIFY_FILE)
        if isinstance(v_data.get("passed"), bool):
            verify_passed = v_data["passed"]
        try:
            diff = diff_stats(run_dir)
        except OSError:
            diff = None
        final_path = run_dir / "final.md"
        if final_path.is_file():
            try:
                text = final_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            final_excerpt = text[:FINAL_EXCERPT_CHARS]
            if len(text) > FINAL_EXCERPT_CHARS:
                final_excerpt += f"\n… ({len(text) - FINAL_EXCERPT_CHARS} more characters)"

        j_path = run_dir / JUDGMENTS_FILE
        if j_path.is_file():
            try:
                j_data = _read_json(j_path, strict=True)
            except (OSError, json.JSONDecodeError):
                j_data = None
            if isinstance(j_data, Mapping):
                has_judgments_file = True
                raw_jm = j_data.get("judge_model")
                if isinstance(raw_jm, str) and raw_jm:
                    j_model = raw_jm
                raw_jam = j_data.get("actual_model")
                if isinstance(raw_jam, str) and raw_jam:
                    j_act_model = raw_jam
                raw_jlist = j_data.get("judgments")
                if isinstance(raw_jlist, list):
                    judgments = tuple(j for j in raw_jlist if isinstance(j, dict))
                    if j_act_model is None and judgments:
                        first_act = judgments[0].get("actual_model")
                        if isinstance(first_act, str) and first_act:
                            j_act_model = first_act

    if (task_id, label) in scores_by_key:
        raw_score = scores_by_key[(task_id, label)]
    elif run_id and run_id in scores_by_run_id:
        raw_score = scores_by_run_id[run_id]
    else:
        raw_score = None

    frac = _score_to_frac(raw_score) if valid else None
    n_missed = _score_to_missed(raw_score) if valid else None
    return _RepRecord(
        task_id=task_id,
        skill=skill,
        label=label,
        run_id=run_id,
        valid=valid,
        invalid_reason=invalid_reason,
        invoked=invoked,
        t_skill_s=t_skill_s,
        requested_model=req_model,
        actual_model=act_model,
        frac=frac,
        judgments=judgments,
        judge_model=j_model,
        judge_actual_model=j_act_model,
        has_judgments_file=has_judgments_file,
        verify_passed=verify_passed if valid else None,
        diff=diff,
        total_tokens=total_tokens,
        t_total_s=t_total_s,
        final_excerpt=final_excerpt,
        n_missed=n_missed,
        skills_read=s_read,
    )


def _median(vals: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return float(st.median(clean)) if clean else None


def pick_best_model(rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Best executor for a skill across models: highest `lift`; ties broken by fewest tokens
    (`tokens_with_median`, then `tokens_without_median`, then `total_tokens`, then row order).
    Returns `None` when no row has a non-null `lift`."""
    candidates: list[tuple[float, float, float, int, str]] = []
    for idx, row in enumerate(rows):
        lift = row.get("lift")
        executor = row.get("executor") or row.get("requested_exec_model")
        if lift is None or isinstance(lift, bool) or not executor:
            continue
        tok_w = row.get("tokens_with_median")
        if tok_w is None:
            tok_w = row.get("tokens_suite_median")
        if tok_w is None:
            tok_w = row.get("total_tokens")
        tok_wo = row.get("tokens_without_median")
        if tok_wo is None:
            tok_wo = row.get("tokens_baseline_median")
        key_lift = -round(float(lift), 9)
        key_tok_w = (
            float(tok_w)
            if tok_w is not None and not isinstance(tok_w, bool)
            else float("inf")
        )
        key_tok_wo = (
            float(tok_wo)
            if tok_wo is not None and not isinstance(tok_wo, bool)
            else float("inf")
        )
        candidates.append((key_lift, key_tok_w, key_tok_wo, idx, str(executor)))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][4]


def _verify_counts(recs: Sequence[_RepRecord]) -> tuple[int, int, float | None]:
    checked = [r.verify_passed for r in recs if r.valid and r.verify_passed is not None]
    passed = sum(1 for v in checked if v)
    return (passed, len(checked), (passed / len(checked)) if checked else None)


def _criteria_rates(recs: Sequence[_RepRecord]) -> dict[str, float]:
    """Per-criterion pass rate over the first judgment of each valid, judged rep."""
    seen: dict[str, list[bool]] = {}
    for r in recs:
        if not r.valid or not r.judgments:
            continue
        for item in r.judgments[0].get("criteria") or []:
            if isinstance(item, Mapping) and isinstance(item.get("pass"), bool):
                seen.setdefault(str(item.get("id")), []).append(item["pass"])
    return {cid: sum(v) / len(v) for cid, v in seen.items()}


def _check_arm_configs(
    with_cfg: Mapping[str, Any], without_cfg: Mapping[str, Any], *, executor: str = ""
) -> None:
    if not with_cfg or not without_cfg:
        return
    who = f" of executor {executor!r}" if executor else ""
    for key in _IDENTITY_KEYS:
        if (
            key == "suite_hash"
            and with_cfg.get("suite") is not None
            and without_cfg.get("suite") is not None
            and with_cfg.get("suite") != without_cfg.get("suite")
        ):
            continue
        v_with = with_cfg.get(key)
        v_without = without_cfg.get(key)
        if v_with is not None and v_without is not None and v_with != v_without:
            raise ValueError(
                f"WITH and WITHOUT namespaces{who} disagree on {key}: {v_with!r} vs {v_without!r}"
            )


@dataclass(frozen=True)
class _LoadedCell:
    cell: ReportCell
    executor: str
    judge: str
    meta: dict[str, Any]
    recs: list[_RepRecord]


def _load_cell(cell: ReportCell) -> _LoadedCell:
    try:
        arm_kind(cell.arm_label)
    except ValueError as exc:
        raise ValueError(
            f"unknown arm label {cell.arm_label!r} for {cell.ns_dir}; "
            f"expected one of {ARM_LABEL_HELP}"
        ) from exc
    meta, recs = _load_arm_records(cell.ns_dir)
    meta_exec = _str_or_none(meta.get("executor"))
    meta_judge = _str_or_none(meta.get("judge"))
    if cell.executor and meta_exec and cell.executor != meta_exec:
        raise ValueError(
            f"{cell.ns_dir}: cell executor {cell.executor!r} but the namespace was run with "
            f"{meta_exec!r}"
        )
    if cell.judge and meta_judge and cell.judge != meta_judge:
        raise ValueError(
            f"{cell.ns_dir}: cell judge {cell.judge!r} but the namespace was judged by "
            f"{meta_judge!r}"
        )
    executor = (
        cell.executor
        or meta_exec
        or next((r.requested_model for r in recs if r.requested_model), None)
        or exec_model()
    )
    judge = (
        cell.judge
        or meta_judge
        or next((r.judge_model for r in recs if r.judge_model), None)
        or judge_model()
    )
    return _LoadedCell(cell, executor, judge, meta, recs)


def _group_cells(
    loaded: Sequence[_LoadedCell], *, require_pairs: bool = True
) -> dict[str, dict[str, _LoadedCell]]:
    """`executor -> arm label -> cell`, in first-seen executor order; every executor must hold
    at least one complete treatment / control pair (`with` + `without`, or `baseline` + one or
    more `suite-<name>` cells) graded by one judge (with `require_pairs` off — the
    hardness-only path — a control cell alone is enough)."""
    groups: dict[str, dict[str, _LoadedCell]] = {}
    for lc in loaded:
        arms = groups.setdefault(lc.executor, {})
        if lc.cell.arm_label in arms:
            raise ValueError(
                f"executor {lc.executor!r} has two {lc.cell.arm_label!r} cells: "
                f"{arms[lc.cell.arm_label].cell.ns_dir} and {lc.cell.ns_dir}"
            )
        arms[lc.cell.arm_label] = lc
    for executor, arms in groups.items():
        judges = sorted({lc.judge for lc in arms.values()})
        if len(judges) > 1:
            raise ValueError(
                f"cells of executor {executor!r} were graded by different judges: "
                f"{', '.join(judges)}; one executor needs one judge"
            )
        if not require_pairs:
            if not any(lbl in arms for lbl in CONTROL_LABELS):
                raise ValueError(
                    f"executor {executor!r} has no control arm cell "
                    f"({' / '.join(sorted(CONTROL_LABELS))}) to judge hardness from "
                    f"(labels: {sorted(arms)})"
                )
            continue
        for treatment, control in ARM_PAIRS.items():
            have = [lbl for lbl in (treatment, control) if lbl in arms]
            if len(have) == 1:
                missing = control if have[0] == treatment else treatment
                raise ValueError(
                    f"executor {executor!r} has a {have[0]!r} cell but no {missing!r} cell"
                )
        suite_arms = [lbl for lbl in arms if lbl.startswith(SUITE_ARM_PREFIX)]
        has_baseline = BASELINE_ARM in arms
        if suite_arms and not has_baseline:
            raise ValueError(
                f"executor {executor!r} has a {suite_arms[0]!r} cell but no {BASELINE_ARM!r} cell"
            )
        if has_baseline and not suite_arms:
            raise ValueError(
                f"executor {executor!r} has a {BASELINE_ARM!r} cell but no "
                f"'{SUITE_ARM_PREFIX}<name>' cell"
            )
        if not any(t in arms for t in ARM_PAIRS) and not suite_arms:
            raise ValueError(
                f"executor {executor!r} has no complete arm pair (labels: {sorted(arms)})"
            )
    return groups


def _check_global_identity(loaded: Sequence[_LoadedCell]) -> None:
    primary_suite = next(
        (
            lc.meta.get("suite")
            for lc in loaded
            if not lc.cell.arm_label.startswith(SUITE_ARM_PREFIX)
            and lc.meta.get("suite") is not None
        ),
        _first_meta(loaded, "suite"),
    )
    for key in _GLOBAL_IDENTITY_KEYS:
        seen: dict[Any, Path] = {}
        for lc in loaded:
            if (
                key == "suite_hash"
                and lc.cell.arm_label.startswith(SUITE_ARM_PREFIX)
                and lc.meta.get("suite") is not None
                and primary_suite is not None
                and lc.meta.get("suite") != primary_suite
            ):
                continue
            value = lc.meta.get(key)
            if value is None:
                continue
            if seen and value not in seen:
                other_value, other_dir = next(iter(seen.items()))
                raise ValueError(
                    f"namespaces disagree on {key}: {other_value!r} ({other_dir}) vs "
                    f"{value!r} ({lc.cell.ns_dir}); report one corpus and one suite at a time"
                )
            seen.setdefault(value, lc.cell.ns_dir)


def _first_meta(loaded: Sequence[_LoadedCell], key: str) -> Any:
    return next((lc.meta.get(key) for lc in loaded if lc.meta.get(key) is not None), None)


def _suite_info(loaded: Sequence[_LoadedCell]) -> dict[str, Any]:
    """`{name, hash}` from `run.json`; the active suite fills in only when no namespace
    recorded a suite at all (older runs)."""
    non_suite_arm = [
        lc for lc in loaded if not lc.cell.arm_label.startswith(SUITE_ARM_PREFIX)
    ] or list(loaded)
    name = _str_or_none(_first_meta(non_suite_arm, "suite")) or _str_or_none(
        _first_meta(loaded, "suite")
    )
    suite_hash = _str_or_none(_first_meta(non_suite_arm, "suite_hash")) or _str_or_none(
        _first_meta(loaded, "suite_hash")
    )
    if name is None and suite_hash is None:
        try:
            suite = active_suite(quiet=True)
            name, suite_hash = suite.name, suite.suite_hash
        except (config.ConfigError, OSError):
            pass
    return {"name": name, "hash": suite_hash}


def _load_extra_suites(
    suites: Iterable[SuiteConfig | Path | str] | str | Path | None,
) -> list[SuiteConfig]:
    if suites is None:
        return []
    if isinstance(suites, SuiteConfig):
        return [suites]
    if isinstance(suites, (str, Path)):
        raw_items: list[SuiteConfig | Path | str] = list(C.parse_suites(suites))
    else:
        raw_items = []
        for item in suites:
            if isinstance(item, SuiteConfig):
                raw_items.append(item)
            else:
                raw_items.extend(C.parse_suites(item))
    out: list[SuiteConfig] = []
    seen: set[str] = set()
    for item in raw_items:
        sc = item if isinstance(item, SuiteConfig) else load_suite(item)
        if sc.name not in seen:
            seen.add(sc.name)
            out.append(sc)
    return out


def _order_suite_cells(
    arms: Mapping[str, _LoadedCell],
    *,
    primary_suite_name: str | None,
    extra_suites: Sequence[SuiteConfig],
    executor: str,
) -> dict[str, _LoadedCell]:
    raw_suite_cells = {
        lbl: lc for lbl, lc in arms.items() if lbl.startswith(SUITE_ARM_PREFIX)
    }
    if not raw_suite_cells:
        return {}
    ordered: dict[str, _LoadedCell] = {}
    if primary_suite_name:
        try:
            primary_lbl = suite_arm_label(primary_suite_name)
        except ValueError:
            primary_lbl = ""
        if primary_lbl in raw_suite_cells:
            ordered[primary_lbl] = raw_suite_cells[primary_lbl]
    for sc in extra_suites:
        lbl = suite_arm_label(sc.name)
        if lbl not in raw_suite_cells:
            raise ValueError(
                f"executor {executor!r} has no {lbl!r} cell for suite {sc.name!r} "
                f"(available: {sorted(raw_suite_cells)})"
            )
        ordered[lbl] = raw_suite_cells[lbl]
    for lbl, lc in raw_suite_cells.items():
        ordered.setdefault(lbl, lc)
    return ordered


def _resolve_cell_list(
    cells: Sequence[ReportCell] | Path | str, without_ns_dir: Path | str | None
) -> list[ReportCell]:
    """The cells behind the three accepted call shapes (see `build_report`)."""
    if isinstance(cells, (str, Path)):
        cell_list = (
            discover_cells(Path(cells))
            if without_ns_dir is None
            else cells_from_dirs(Path(cells), Path(without_ns_dir))
        )
    else:
        if without_ns_dir is not None:
            raise ValueError("pass either a list of ReportCell or (with_dir, without_dir)")
        cell_list = list(cells)
    if not cell_list:
        raise ValueError("the report needs at least one ReportCell")
    return cell_list


def build_report(
    cells: Sequence[ReportCell] | Path | str,
    without_ns_dir: Path | str | None = None,
    *,
    k: int | None = None,
    seed: int | None = None,
    bootstrap_seed: int | None = None,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    pilot: bool | None = None,
    min_baseline_fail: int | Mapping[str, int] | None = None,
    suites: Iterable[SuiteConfig | Path | str] | str | Path | None = None,
    history: Sequence[Mapping[str, Any]] | Path | str | None = None,
) -> dict[str, Any]:
    """Compute the schema-v2 skillbench report dictionary from a list of `ReportCell`s.

    Accepted inputs:
    - `build_report(cells)` — any number of `(executor, judge, arm_label, ns_dir)` cells; the
      cells are grouped per executor and each executor's `with` / `without` pair (and/or
      `baseline` + `suite-<name>` cells) yields one `models[]` entry;
    - `build_report(with_ns_dir, without_ns_dir)` — the schema-v1 call: one pair, executor
      and judge taken from the namespaces' `run.json` / `config.json`;
    - `build_report(results_root)` — one path: auto-discovers `<executor>/<arm-label>`
      namespaces under it (see `discover_cells`).

    `min_baseline_fail` sets the hardness threshold: an int for every task, a `task_id -> int`
    mapping, or None to read each task's `min_baseline_fail` from the resolved corpus
    (default 1 when the corpus or the task is not found).

    Shape: `{schema_version: 2, suite: {name, hash}, k, seed, bootstrap_*, pilot,
    manifest_hash, power_note, hardness_note, models: [{executor, judge, arms, k,
    requested_* / actual_* model fields, judge_noise_*, invalid_*, skills, by_skill,
    suites, by_family, suite_arms, task_hardness}], ...}` plus the per-executor keys of
    `models[0]` mirrored at the top level, so schema-v1 consumers keep reading the same
    `skills` / `by_skill` numbers from a single-model report.
    """
    cell_list = _resolve_cell_list(cells, without_ns_dir)
    thresholds = _thresholds(min_baseline_fail)
    extra_suites = _load_extra_suites(suites)

    loaded = [_load_cell(c) for c in cell_list]
    _check_global_identity(loaded)
    groups = _group_cells(loaded)
    suite_meta = _suite_info(loaded)

    is_pilot = (
        bool(pilot)
        if pilot is not None
        else _env_pilot()
    )
    eff_seed = seed if seed is not None else _first_meta(loaded, "seed")
    eff_boot_seed = (
        int(bootstrap_seed)
        if bootstrap_seed is not None
        else (int(eff_seed) if eff_seed is not None else DEFAULT_BOOTSTRAP_SEED)
    )

    manifest_hash = _first_meta(loaded, "manifest_hash")
    if not manifest_hash:
        try:
            manifest_hash = check_manifest()
        except ManifestError:
            manifest_hash = "unverified"

    models: list[dict[str, Any]] = []
    for executor, arms in groups.items():
        with_cell = arms.get("with")
        without_cell = arms.get("without")
        if with_cell is not None and without_cell is not None:
            _check_arm_configs(with_cell.meta, without_cell.meta, executor=executor)
        baseline_cell = arms.get(BASELINE_ARM)
        suite_cells = _order_suite_cells(
            arms,
            primary_suite_name=_str_or_none(suite_meta.get("name")),
            extra_suites=extra_suites,
            executor=executor,
        )
        if baseline_cell is not None:
            for sc in suite_cells.values():
                _check_arm_configs(sc.meta, baseline_cell.meta, executor=executor)
        any_cell = with_cell or baseline_cell or next(iter(arms.values()))
        models.append(
            _model_block(
                executor=executor,
                judge=any_cell.judge,
                with_cell=with_cell,
                without_cell=without_cell,
                baseline_cell=baseline_cell,
                suite_cells=suite_cells,
                k=k,
                draws=draws,
                boot_seed=eff_boot_seed,
                pilot=is_pilot,
                thresholds=thresholds,
            )
        )

    _annotate_best_models(models)
    skill_matrix = build_skill_matrix(models)
    model_summary = build_model_summary(models)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "suite": suite_meta,
        "k": int(k) if k is not None else max(m["k"] for m in models),
        "seed": eff_seed,
        "bootstrap_seed": eff_boot_seed,
        "bootstrap_draws": draws,
        "pilot": is_pilot,
        "manifest_hash": manifest_hash,
        "power_note": POWER_NOTE,
        "hardness_note": HARDNESS_NOTE,
        "recommendation_note": RECOMMENDATION_RULE_NOTE,
        "judge_pairing_note": judge_pairing_note(models),
        "models": models,
        "model_summary": model_summary,
        "skill_matrix": skill_matrix,
    }
    report["summary"] = build_summary(report, history=history)
    if extra_suites:
        report["extra_suites"] = [
            {"name": sc.name, "hash": sc.suite_hash} for sc in extra_suites
        ]
    report.update({key: models[0][key] for key in _LEGACY_MODEL_KEYS})
    return report


def build_hardness(
    cells: Sequence[ReportCell] | Path | str,
    without_ns_dir: Path | str | None = None,
    *,
    min_baseline_fail: int | Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """The task x executor hardness view from control-arm cells alone (treatment cells are
    accepted and ignored), for the corpus-authoring loop (`skillbench-matrix --arms baseline`
    then `skillbench-report --hardness-only`). Same inputs as `build_report`.

    Shape: `{schema_version: 2, suite, manifest_hash, min_reps, hardness_note,
    models: [{executor, judge, control_arm, tasks: {task_id: {skill, hardness,
    baseline_missed, valid_reps, min_baseline_fail}}, skills: {skill: {ceiling_tasks,
    informative_tasks, unclassified_tasks, total_tasks, ceiling_share, needs_harder_tasks}}}]}`.
    """
    cell_list = _resolve_cell_list(cells, without_ns_dir)
    thresholds = _thresholds(min_baseline_fail)
    loaded = [_load_cell(c) for c in cell_list]
    _check_global_identity(loaded)
    groups = _group_cells(loaded, require_pairs=False)

    models: list[dict[str, Any]] = []
    for executor, arms in groups.items():
        control_labels = [lbl for lbl in sorted(CONTROL_LABELS) if lbl in arms]
        control_recs = [r for lbl in control_labels for r in arms[lbl].recs]
        first_control = arms[control_labels[0]]
        tasks = hardness_rows(control_recs, thresholds)
        skills: dict[str, dict[str, Any]] = {}
        for skill in sorted({row["skill"] for row in tasks.values()}):
            skills[skill] = skill_hardness(
                row["hardness"] for row in tasks.values() if row["skill"] == skill
            )
        models.append(
            {
                "executor": executor,
                "judge": first_control.judge,
                "control_arm": "/".join(control_labels),
                "tasks": tasks,
                "skills": skills,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": _suite_info(loaded),
        "manifest_hash": _first_meta(loaded, "manifest_hash"),
        "min_reps": MIN_BASELINE_REPS,
        "hardness_note": HARDNESS_NOTE,
        "models": models,
    }


def hardness_of_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """The `build_hardness` view of a full `build_report` dictionary (schema v2 with the
    hardness fields), so one rendering serves both."""
    models: list[dict[str, Any]] = []
    for model in models_of(report):
        tasks: dict[str, dict[str, Any]] = {}
        skills: dict[str, dict[str, Any]] = {}
        rows = list(model.get("skills") or []) + list(model.get("suites") or [])
        for row in rows:
            skill = str(row.get("skill") or row.get("family"))
            for tid, t in sorted((row.get("tasks") or {}).items()):
                tasks[str(tid)] = {
                    "skill": skill,
                    "hardness": t.get("hardness", HARDNESS_UNCLASSIFIED),
                    "baseline_missed": list(t.get("baseline_missed") or []),
                    "valid_reps": len(t.get("baseline_missed") or []),
                    "min_baseline_fail": t.get("min_baseline_fail", DEFAULT_MIN_BASELINE_FAIL),
                }
            skills[skill] = skill_hardness(
                t.get("hardness", HARDNESS_UNCLASSIFIED) for t in (row.get("tasks") or {}).values()
            )
        arms_map = model.get("arms") if isinstance(model.get("arms"), Mapping) else {}
        control_arm = (
            BASELINE_ARM
            if BASELINE_ARM in arms_map and "without" not in arms_map
            else "without"
        )
        models.append(
            {
                "executor": model.get("executor") or model.get("requested_exec_model"),
                "judge": model.get("judge") or model.get("requested_judge_model"),
                "control_arm": control_arm,
                "tasks": tasks,
                "skills": skills,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": report.get("suite"),
        "manifest_hash": report.get("manifest_hash"),
        "min_reps": MIN_BASELINE_REPS,
        "hardness_note": report.get("hardness_note") or HARDNESS_NOTE,
        "models": models,
    }


def _skills_histogram(recs: Iterable[_RepRecord]) -> dict[str, int]:
    """Count of valid reps in `recs` that read each skill (`skill_name -> rep_count`)."""
    counts: dict[str, int] = {}
    for r in recs:
        if not r.valid:
            continue
        for s in set(r.skills_read):
            if s:
                counts[s] = counts.get(s, 0) + 1
    return dict(sorted(counts.items()))


def _build_suites_section(
    *,
    baseline_cell: _LoadedCell,
    suite_cells: Mapping[str, _LoadedCell],
    draws: int,
    boot_seed: int,
    pilot: bool,
    thresholds: Thresholds,
    overall_invalid_by_reason: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], int]:
    """Compute `(suites_rows, by_family, invalid_count)` for one executor's suite-mode cells."""
    base_recs = baseline_cell.recs
    suite_recs_all = [r for sc in suite_cells.values() for r in sc.recs]
    hardness = hardness_rows(base_recs, thresholds)
    families = sorted({r.skill for r in base_recs + suite_recs_all})
    suite_arm_labels = list(suite_cells.keys())

    suite_rows: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, Any]] = {}
    suite_invalid_total = 0

    for family in families:
        f_base = [r for r in base_recs if r.skill == family]
        task_ids = sorted(
            {r.task_id for r in f_base}
            | {
                r.task_id
                for sc in suite_cells.values()
                for r in sc.recs
                if r.skill == family
            }
        )
        valid_f_base = [r for r in f_base if r.valid]
        inv_f_base = [r for r in f_base if not r.valid]
        ungraded_f_base = sum(1 for r in f_base if r.valid and r.frac is None)
        vb_pass, vb_total, vb_rate = _verify_counts(f_base)
        tok_base_med = _median(r.total_tokens for r in valid_f_base)
        wall_base_med = _median(r.t_total_s for r in valid_f_base)
        diff_base_med = _median(r.diff_lines for r in valid_f_base)

        family_reasons: dict[str, int] = {}
        for r in inv_f_base:
            reason = r.invalid_reason or "invalid rollout"
            family_reasons[reason] = family_reasons.get(reason, 0) + 1
            overall_invalid_by_reason[reason] = overall_invalid_by_reason.get(reason, 0) + 1
        suite_invalid_total += len(inv_f_base)

        per_arm_family: dict[str, dict[str, Any]] = {}
        per_task_by_arm: dict[str, dict[str, dict[str, Any]]] = {tid: {} for tid in task_ids}
        per_task_reps: dict[str, dict[str, list[dict[str, Any]]]] = {
            tid: {BASELINE_ARM: [r.detail() for r in f_base if r.task_id == tid]}
            for tid in task_ids
        }

        for arm_label, s_cell in suite_cells.items():
            suite_name = arm_label[len(SUITE_ARM_PREFIX) :]
            f_suite = [r for r in s_cell.recs if r.skill == family]
            task_deltas: dict[str, float] = {}
            paired_for_boot: list[tuple[list[float], list[float]]] = []
            suite_task_means: list[float] = []
            base_task_means: list[float] = []
            n_qualifying = 0

            for tid in task_ids:
                t_suite = [r for r in f_suite if r.task_id == tid]
                t_base = [r for r in f_base if r.task_id == tid]
                s_fracs = [r.frac for r in t_suite if r.valid and r.frac is not None]
                b_fracs = [r.frac for r in t_base if r.valid and r.frac is not None]
                s_mean = st.mean(s_fracs) if s_fracs else None
                b_mean = st.mean(b_fracs) if b_fracs else None
                delta = (
                    (s_mean - b_mean)
                    if (s_mean is not None and b_mean is not None)
                    else None
                )
                if delta is not None and s_mean is not None and b_mean is not None:
                    task_deltas[tid] = delta
                    paired_for_boot.append((s_fracs, b_fracs))
                    suite_task_means.append(s_mean)
                    base_task_means.append(b_mean)
                    t_ci_low, t_ci_high = bootstrap_ci(
                        [(s_fracs, b_fracs)], draws=draws, seed=boot_seed
                    )
                else:
                    t_ci_low, t_ci_high = None, None

                t_qualifies = (
                    len(s_fracs) >= MIN_VALID_REPS_PER_ARM
                    and len(b_fracs) >= MIN_VALID_REPS_PER_ARM
                )
                if t_qualifies:
                    n_qualifying += 1
                if not t_qualifies or t_ci_low is None or t_ci_high is None:
                    t_verdict = "insufficient data"
                elif t_ci_low > 0.0:
                    t_verdict = "helps"
                elif t_ci_high < 0.0:
                    t_verdict = "hurts"
                else:
                    t_verdict = "inconclusive"

                valid_t_suite = [r for r in t_suite if r.valid]
                inv_t_suite = [r for r in valid_t_suite if r.invoked]
                n_valid_ts = len(valid_t_suite)
                n_inv_ts = len(inv_t_suite)
                t_inv_rate = (n_inv_ts / n_valid_ts) if n_valid_ts > 0 else None
                t_skills = sorted(r.t_skill_s for r in inv_t_suite if r.t_skill_s is not None)
                t_p50 = float(st.median(t_skills)) if t_skills else None
                t_p90 = _quantile(t_skills, 0.90) if t_skills else None
                ts_pass, ts_total, _ = _verify_counts(t_suite)
                t_hist = _skills_histogram(valid_t_suite)

                per_task_by_arm[tid][arm_label] = {
                    "suite_arm": arm_label,
                    "suite_name": suite_name,
                    "suite_fracs": s_fracs,
                    "with_fracs": s_fracs,
                    "baseline_fracs": b_fracs,
                    "without_fracs": b_fracs,
                    "suite_mean": s_mean,
                    "with_mean": s_mean,
                    "baseline_mean": b_mean,
                    "without_mean": b_mean,
                    "delta": delta,
                    "lift": delta,
                    "ci_low": t_ci_low,
                    "ci_high": t_ci_high,
                    "ci": (
                        [t_ci_low, t_ci_high]
                        if (t_ci_low is not None and t_ci_high is not None)
                        else None
                    ),
                    "verdict": t_verdict,
                    "invoked": n_inv_ts,
                    "valid_suite_reps": n_valid_ts,
                    "valid_with_reps": n_valid_ts,
                    "invocation_rate": t_inv_rate,
                    "t_skill_s": t_skills,
                    "t_skill_p50": t_p50,
                    "t_skill_p90": t_p90,
                    "skills_read_histogram": t_hist,
                    "verify_suite": [ts_pass, ts_total],
                    "verify_with": [ts_pass, ts_total],
                    "diff_lines_suite_median": _median(r.diff_lines for r in valid_t_suite),
                    "diff_lines_with_median": _median(r.diff_lines for r in valid_t_suite),
                    "tokens_suite_median": _median(r.total_tokens for r in valid_t_suite),
                    "tokens_with_median": _median(r.total_tokens for r in valid_t_suite),
                    "t_total_suite_median": _median(r.t_total_s for r in valid_t_suite),
                    "t_total_with_median": _median(r.t_total_s for r in valid_t_suite),
                    "criteria_suite": _criteria_rates(t_suite),
                    "criteria_with": _criteria_rates(t_suite),
                }
                per_task_reps[tid][arm_label] = [r.detail() for r in t_suite]

            if paired_for_boot:
                fam_suite_mean: float | None = st.mean(suite_task_means)
                fam_base_mean: float | None = st.mean(base_task_means)
                lift: float | None = st.mean(task_deltas.values())
                ci_low, ci_high = bootstrap_ci(
                    paired_for_boot, draws=draws, seed=boot_seed
                )
            else:
                fam_suite_mean = None
                fam_base_mean = None
                lift = None
                ci_low, ci_high = None, None

            verdict = decide_verdict(
                ci_low, ci_high, n_qualifying_tasks=n_qualifying, pilot=pilot
            )

            valid_f_suite = [r for r in f_suite if r.valid]
            invoked_f_suite = [r for r in valid_f_suite if r.invoked]
            n_valid_suite = len(valid_f_suite)
            n_invoked = len(invoked_f_suite)
            inv_rate = (n_invoked / n_valid_suite) if n_valid_suite > 0 else None
            t_skills_fam = sorted(
                r.t_skill_s for r in invoked_f_suite if r.t_skill_s is not None
            )
            t_p50_fam = float(st.median(t_skills_fam)) if t_skills_fam else None
            t_p90_fam = _quantile(t_skills_fam, 0.90) if t_skills_fam else None
            fam_hist = _skills_histogram(valid_f_suite)

            inv_f_suite = [r for r in f_suite if not r.valid]
            arm_reasons: dict[str, int] = dict(family_reasons)
            for r in inv_f_suite:
                reason = r.invalid_reason or "invalid rollout"
                arm_reasons[reason] = arm_reasons.get(reason, 0) + 1
                overall_invalid_by_reason[reason] = overall_invalid_by_reason.get(reason, 0) + 1
            suite_invalid_total += len(inv_f_suite)

            ungraded_f_suite = sum(1 for r in f_suite if r.valid and r.frac is None)
            vs_pass, vs_total, vs_rate = _verify_counts(f_suite)
            tok_suite_med = _median(r.total_tokens for r in valid_f_suite)
            wall_suite_med = _median(r.t_total_s for r in valid_f_suite)
            diff_suite_med = _median(r.diff_lines for r in valid_f_suite)

            per_arm_family[arm_label] = {
                "suite_arm": arm_label,
                "suite_name": suite_name,
                "n_tasks": len(task_ids),
                "n_paired_tasks": len(paired_for_boot),
                "n_qualifying_tasks": n_qualifying,
                "suite_mean": fam_suite_mean,
                "with_mean": fam_suite_mean,
                "baseline_mean": fam_base_mean,
                "without_mean": fam_base_mean,
                "lift": lift,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci": (
                    [ci_low, ci_high]
                    if (ci_low is not None and ci_high is not None)
                    else None
                ),
                "verdict": verdict,
                "invoked": n_invoked,
                "valid_suite_reps": n_valid_suite,
                "valid_with_reps": n_valid_suite,
                "invocation_rate": inv_rate,
                "t_skill_s": t_skills_fam,
                "t_skill_p50": t_p50_fam,
                "t_skill_p90": t_p90_fam,
                "skills_read_histogram": fam_hist,
                "invalid_suite": len(inv_f_suite),
                "invalid_with": len(inv_f_suite),
                "invalid_baseline": len(inv_f_base),
                "invalid_without": len(inv_f_base),
                "invalid_total": len(inv_f_suite) + len(inv_f_base),
                "invalid_by_reason": dict(sorted(arm_reasons.items())),
                "ungraded_suite": ungraded_f_suite,
                "ungraded_with": ungraded_f_suite,
                "ungraded_baseline": ungraded_f_base,
                "ungraded_without": ungraded_f_base,
                "ungraded_total": ungraded_f_suite + ungraded_f_base,
                "verify_suite_passed": vs_pass,
                "verify_suite_total": vs_total,
                "verify_suite_rate": vs_rate,
                "verify_with_passed": vs_pass,
                "verify_with_total": vs_total,
                "verify_with_rate": vs_rate,
                "verify_baseline_passed": vb_pass,
                "verify_baseline_total": vb_total,
                "verify_baseline_rate": vb_rate,
                "verify_without_passed": vb_pass,
                "verify_without_total": vb_total,
                "verify_without_rate": vb_rate,
                "diff_lines_suite_median": diff_suite_med,
                "diff_lines_with_median": diff_suite_med,
                "diff_lines_baseline_median": diff_base_med,
                "diff_lines_without_median": diff_base_med,
                "tokens_suite_median": tok_suite_med,
                "tokens_with_median": tok_suite_med,
                "tokens_baseline_median": tok_base_med,
                "tokens_without_median": tok_base_med,
                "tokens_delta_pct": _delta_pct(tok_suite_med, tok_base_med),
                "t_total_suite_median": wall_suite_med,
                "t_total_with_median": wall_suite_med,
                "t_total_baseline_median": wall_base_med,
                "t_total_without_median": wall_base_med,
                "wall_delta_pct": _delta_pct(wall_suite_med, wall_base_med),
                "task_deltas": task_deltas,
            }

        primary_arm = suite_arm_labels[0]
        tasks_detail: dict[str, dict[str, Any]] = {}
        for tid in task_ids:
            t_base = [r for r in f_base if r.task_id == tid]
            valid_t_base = [r for r in t_base if r.valid]
            b_fracs = [r.frac for r in valid_t_base if r.frac is not None]
            b_mean = st.mean(b_fracs) if b_fracs else None
            tb_pass, tb_total, _ = _verify_counts(t_base)
            hard = hardness.get(tid) or {
                "hardness": HARDNESS_UNCLASSIFIED,
                "baseline_missed": [],
                "min_baseline_fail": thresholds[1].get(tid, thresholds[0]),
            }
            primary_task = per_task_by_arm[tid][primary_arm]
            tasks_detail[tid] = {
                "task_id": tid,
                "family": family,
                **primary_task,
                "baseline_fracs": b_fracs,
                "without_fracs": b_fracs,
                "baseline_mean": b_mean,
                "without_mean": b_mean,
                "hardness": hard["hardness"],
                "baseline_missed": hard["baseline_missed"],
                "min_baseline_fail": hard["min_baseline_fail"],
                "verify_baseline": [tb_pass, tb_total],
                "verify_without": [tb_pass, tb_total],
                "diff_lines_baseline_median": _median(r.diff_lines for r in valid_t_base),
                "diff_lines_without_median": _median(r.diff_lines for r in valid_t_base),
                "tokens_baseline_median": _median(r.total_tokens for r in valid_t_base),
                "tokens_without_median": _median(r.total_tokens for r in valid_t_base),
                "t_total_baseline_median": _median(r.t_total_s for r in valid_t_base),
                "t_total_without_median": _median(r.t_total_s for r in valid_t_base),
                "criteria_baseline": _criteria_rates(t_base),
                "criteria_without": _criteria_rates(t_base),
                "by_arm": per_task_by_arm[tid],
                "arms": per_task_by_arm[tid],
                "reps": {
                    **per_task_reps[tid],
                    "with": per_task_reps[tid][primary_arm],
                    "without": per_task_reps[tid][BASELINE_ARM],
                },
            }

        primary_fam = per_arm_family[primary_arm]
        row = {
            "family": family,
            "skill": family,
            **primary_fam,
            "suite_arms": suite_arm_labels,
            "by_arm": per_arm_family,
            "arms": per_arm_family,
            **skill_hardness(t["hardness"] for t in tasks_detail.values()),
            "tasks": tasks_detail,
        }
        rec_lbl, rec_rsn = recommend(row)
        row["recommendation"] = rec_lbl
        row["reason"] = rec_rsn
        suite_rows.append(row)
        by_family[family] = row

    return suite_rows, by_family, suite_invalid_total


def _model_block(
    *,
    executor: str,
    judge: str,
    with_cell: _LoadedCell | None = None,
    without_cell: _LoadedCell | None = None,
    baseline_cell: _LoadedCell | None = None,
    suite_cells: Mapping[str, _LoadedCell] | None = None,
    k: int | None,
    draws: int,
    boot_seed: int,
    pilot: bool,
    thresholds: Thresholds = (DEFAULT_MIN_BASELINE_FAIL, {}),
) -> dict[str, Any]:
    """One `models[]` entry: the per-skill leave-one-out block and/or per-family suite block
    of a single executor."""
    suite_cells_map = dict(suite_cells or {})
    with_recs = with_cell.recs if with_cell is not None else []
    without_recs = without_cell.recs if without_cell is not None else []
    base_recs = baseline_cell.recs if baseline_cell is not None else []
    suite_recs = [r for sc in suite_cells_map.values() for r in sc.recs]
    is_pilot = pilot
    eff_boot_seed = boot_seed
    all_recs = with_recs + without_recs + base_recs + suite_recs
    hardness = hardness_rows(without_recs, thresholds) if without_recs else {}
    if k is None:
        counts_per_task: dict[str, int] = {}
        treatment_lists = (
            [with_recs]
            if with_recs
            else [sc.recs for sc in suite_cells_map.values()]
        )
        for rlist in treatment_lists:
            local_counts: dict[str, int] = {}
            for r in rlist:
                local_counts[r.task_id] = local_counts.get(r.task_id, 0) + 1
            for tid, count in local_counts.items():
                counts_per_task[tid] = max(counts_per_task.get(tid, 0), count)
        eff_k = max(counts_per_task.values(), default=1)
    else:
        eff_k = int(k)

    req_exec = executor
    req_judge = judge

    valid_recs = [r for r in all_recs if r.valid]
    exec_model_verified = bool(valid_recs) and all(r.actual_model is not None for r in valid_recs)
    observed_exec = sorted({r.actual_model for r in valid_recs if r.actual_model is not None})
    actual_exec_model = ", ".join(observed_exec) if exec_model_verified else None

    recs_with_jfile = [r for r in valid_recs if r.has_judgments_file]
    if recs_with_jfile:
        judge_model_verified = all(r.judge_actual_model is not None for r in recs_with_jfile)
        observed_judge = sorted(
            {r.judge_actual_model for r in recs_with_jfile if r.judge_actual_model is not None}
        )
        actual_judge_model = ", ".join(observed_judge) if judge_model_verified else None
        model_verified = exec_model_verified and judge_model_verified
    else:
        judge_model_verified = False
        actual_judge_model = None
        model_verified = exec_model_verified

    noise_rate, noise_disagreed, noise_total = judge_noise(r.judgments for r in valid_recs)

    skills = sorted({r.skill for r in with_recs + without_recs})
    skill_rows: list[dict[str, Any]] = []
    by_skill: dict[str, dict[str, Any]] = {}
    overall_invalid_by_reason: dict[str, int] = {}
    overall_invalid_total = 0

    for skill in skills:
        s_with = [r for r in with_recs if r.skill == skill]
        s_without = [r for r in without_recs if r.skill == skill]
        task_ids = sorted({r.task_id for r in s_with} | {r.task_id for r in s_without})

        tasks_detail: dict[str, dict[str, Any]] = {}
        task_deltas: dict[str, float] = {}
        paired_for_boot: list[tuple[list[float], list[float]]] = []
        with_task_means: list[float] = []
        without_task_means: list[float] = []
        n_qualifying = 0

        for tid in task_ids:
            t_with = [r for r in s_with if r.task_id == tid]
            t_without = [r for r in s_without if r.task_id == tid]
            w_fracs = [r.frac for r in t_with if r.valid and r.frac is not None]
            u_fracs = [r.frac for r in t_without if r.valid and r.frac is not None]
            w_mean = st.mean(w_fracs) if w_fracs else None
            u_mean = st.mean(u_fracs) if u_fracs else None
            delta = (w_mean - u_mean) if (w_mean is not None and u_mean is not None) else None
            if delta is not None and w_mean is not None and u_mean is not None:
                task_deltas[tid] = delta
                paired_for_boot.append((w_fracs, u_fracs))
                with_task_means.append(w_mean)
                without_task_means.append(u_mean)
            if len(w_fracs) >= MIN_VALID_REPS_PER_ARM and len(u_fracs) >= MIN_VALID_REPS_PER_ARM:
                n_qualifying += 1
            tw_pass, tw_total, _ = _verify_counts(t_with)
            tu_pass, tu_total, _ = _verify_counts(t_without)
            hard = hardness.get(tid) or {  # task swept in WITH only: no control reps at all
                "hardness": HARDNESS_UNCLASSIFIED,
                "baseline_missed": [],
                "min_baseline_fail": thresholds[1].get(tid, thresholds[0]),
            }
            tasks_detail[tid] = {
                "with_fracs": w_fracs,
                "without_fracs": u_fracs,
                "with_mean": w_mean,
                "without_mean": u_mean,
                "delta": delta,
                "hardness": hard["hardness"],
                "baseline_missed": hard["baseline_missed"],
                "min_baseline_fail": hard["min_baseline_fail"],
                "verify_with": [tw_pass, tw_total],
                "verify_without": [tu_pass, tu_total],
                "diff_lines_with_median": _median(r.diff_lines for r in t_with if r.valid),
                "diff_lines_without_median": _median(
                    r.diff_lines for r in t_without if r.valid
                ),
                "criteria_with": _criteria_rates(t_with),
                "criteria_without": _criteria_rates(t_without),
                "reps": {
                    "with": [r.detail() for r in t_with],
                    "without": [r.detail() for r in t_without],
                },
            }

        if paired_for_boot:
            skill_with_mean: float | None = st.mean(with_task_means)
            skill_without_mean: float | None = st.mean(without_task_means)
            lift: float | None = st.mean(task_deltas.values())
            ci_low, ci_high = bootstrap_ci(paired_for_boot, draws=draws, seed=eff_boot_seed)
        else:
            skill_with_mean = None
            skill_without_mean = None
            lift = None
            ci_low, ci_high = None, None

        verdict = decide_verdict(
            ci_low, ci_high, n_qualifying_tasks=n_qualifying, pilot=is_pilot
        )

        # Time-to-skill and invocation rate over valid WITH reps only
        valid_with = [r for r in s_with if r.valid]
        invoked_with = [r for r in valid_with if r.invoked]
        n_valid_with = len(valid_with)
        n_invoked = len(invoked_with)
        inv_rate = (n_invoked / n_valid_with) if n_valid_with > 0 else None
        t_skills = sorted(r.t_skill_s for r in invoked_with if r.t_skill_s is not None)
        if t_skills:
            t_p50: float | None = float(st.median(t_skills))
            t_p90: float | None = _quantile(t_skills, 0.90)
        else:
            t_p50 = None
            t_p90 = None

        # Invalid and ungraded counts
        inv_with = [r for r in s_with if not r.valid]
        inv_without = [r for r in s_without if not r.valid]
        reasons: dict[str, int] = {}
        for r in inv_with + inv_without:
            reason = r.invalid_reason or "invalid rollout"
            reasons[reason] = reasons.get(reason, 0) + 1
            overall_invalid_by_reason[reason] = overall_invalid_by_reason.get(reason, 0) + 1
        inv_total = len(inv_with) + len(inv_without)
        overall_invalid_total += inv_total

        ungraded_with = sum(1 for r in s_with if r.valid and r.frac is None)
        ungraded_without = sum(1 for r in s_without if r.valid and r.frac is None)

        vw_pass, vw_total, vw_rate = _verify_counts(s_with)
        vu_pass, vu_total, vu_rate = _verify_counts(s_without)
        valid_without = [r for r in s_without if r.valid]
        tok_with_med = _median(r.total_tokens for r in valid_with)
        tok_without_med = _median(r.total_tokens for r in valid_without)
        wall_with_med = _median(r.t_total_s for r in valid_with)
        wall_without_med = _median(r.t_total_s for r in valid_without)

        row = {
            "skill": skill,
            "n_tasks": len(task_ids),
            "n_paired_tasks": len(paired_for_boot),
            "n_qualifying_tasks": n_qualifying,
            "with_mean": skill_with_mean,
            "without_mean": skill_without_mean,
            "lift": lift,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci": [ci_low, ci_high] if (ci_low is not None and ci_high is not None) else None,
            "verdict": verdict,
            "invoked": n_invoked,
            "valid_with_reps": n_valid_with,
            "invocation_rate": inv_rate,
            "t_skill_s": t_skills,
            "t_skill_p50": t_p50,
            "t_skill_p90": t_p90,
            "invalid_with": len(inv_with),
            "invalid_without": len(inv_without),
            "invalid_total": inv_total,
            "invalid_by_reason": dict(sorted(reasons.items())),
            "ungraded_with": ungraded_with,
            "ungraded_without": ungraded_without,
            "ungraded_total": ungraded_with + ungraded_without,
            "verify_with_passed": vw_pass,
            "verify_with_total": vw_total,
            "verify_with_rate": vw_rate,
            "verify_without_passed": vu_pass,
            "verify_without_total": vu_total,
            "verify_without_rate": vu_rate,
            "diff_lines_with_median": _median(r.diff_lines for r in valid_with),
            "diff_lines_without_median": _median(r.diff_lines for r in valid_without),
            "tokens_with_median": tok_with_med,
            "tokens_without_median": tok_without_med,
            "tokens_delta_pct": _delta_pct(tok_with_med, tok_without_med),
            "t_total_with_median": wall_with_med,
            "t_total_without_median": wall_without_med,
            "wall_delta_pct": _delta_pct(wall_with_med, wall_without_med),
            **skill_hardness(t["hardness"] for t in tasks_detail.values()),
            "task_deltas": task_deltas,
            "tasks": tasks_detail,
        }
        rec_lbl, rec_rsn = recommend(row)
        row["recommendation"] = rec_lbl
        row["reason"] = rec_rsn
        skill_rows.append(row)
        by_skill[skill] = row

    if baseline_cell is not None and suite_cells_map:
        suite_rows, by_family, suite_invalid = _build_suites_section(
            baseline_cell=baseline_cell,
            suite_cells=suite_cells_map,
            draws=draws,
            boot_seed=eff_boot_seed,
            pilot=is_pilot,
            thresholds=thresholds,
            overall_invalid_by_reason=overall_invalid_by_reason,
        )
        overall_invalid_total += suite_invalid
    else:
        suite_rows = []
        by_family = {}

    arms_dict: dict[str, str] = {}
    if with_cell is not None:
        arms_dict["with"] = str(with_cell.cell.ns_dir)
    if without_cell is not None:
        arms_dict["without"] = str(without_cell.cell.ns_dir)
    if baseline_cell is not None:
        arms_dict[BASELINE_ARM] = str(baseline_cell.cell.ns_dir)
    for arm_label, sc in suite_cells_map.items():
        arms_dict[arm_label] = str(sc.cell.ns_dir)

    task_hardness_map = {
        tid: t["hardness"]
        for row in (skill_rows + suite_rows)
        for tid, t in row["tasks"].items()
    }

    lifts = [float(r["lift"]) for r in skill_rows if r.get("lift") is not None]
    if not lifts and suite_rows:
        lifts = [float(r["lift"]) for r in suite_rows if r.get("lift") is not None]
    mean_lift: float | None = float(st.mean(lifts)) if lifts else None
    total_tokens = sum(int(r.total_tokens) for r in valid_recs if r.total_tokens is not None)
    total_wall_s = float(sum(float(r.t_total_s) for r in valid_recs if r.t_total_s is not None))
    total_rollouts = len(all_recs)
    valid_total = len(valid_recs)
    invalid_rate: float | None = (
        (overall_invalid_total / total_rollouts) if total_rollouts > 0 else None
    )

    return {
        "executor": executor,
        "judge": judge,
        "arms": arms_dict,
        "k": eff_k,
        "requested_exec_model": req_exec,
        "actual_exec_model": actual_exec_model,
        "exec_model_verified": exec_model_verified,
        "requested_judge_model": req_judge,
        "actual_judge_model": actual_judge_model,
        "judge_model_verified": judge_model_verified,
        "model_verified": model_verified,
        "judge_noise_rate": noise_rate,
        "judge_noise_disagreed": noise_disagreed,
        "judge_noise_total": noise_total,
        "mean_lift": mean_lift,
        "total_tokens": total_tokens,
        "total_wall_s": total_wall_s,
        "total_wall": total_wall_s,
        "valid_total": valid_total,
        "total_rollouts": total_rollouts,
        "invalid_total": overall_invalid_total,
        "invalid_rate": invalid_rate,
        "invalid_by_reason": dict(sorted(overall_invalid_by_reason.items())),
        "skills": skill_rows,
        "by_skill": by_skill,
        "suites": suite_rows,
        "by_family": by_family,
        "suite_arms": list(suite_cells_map.keys()),
        "task_hardness": task_hardness_map,
    }


def models_of(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The per-executor blocks of a report; a schema-v1 dictionary is its own single block."""
    models = report.get("models")
    if isinstance(models, list) and models:
        return [m for m in models if isinstance(m, Mapping)]
    return [report]


def _model_exec_name(model: Mapping[str, Any]) -> str:
    return str(model.get("executor") or model.get("requested_exec_model") or "executor")


def _model_judge_name(model: Mapping[str, Any]) -> str:
    return str(model.get("judge") or model.get("requested_judge_model") or "judge")


def _skill_rows_of(model: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    by_skill = model.get("by_skill")
    if isinstance(by_skill, Mapping) and by_skill:
        return {
            str(k): dict(v)
            for k, v in by_skill.items()
            if isinstance(v, Mapping)
        }
    out: dict[str, dict[str, Any]] = {}
    for row in model.get("skills") or []:
        if isinstance(row, Mapping) and row.get("skill"):
            out[str(row["skill"])] = dict(row)
    return out


def _annotate_best_models(models: Sequence[dict[str, Any]]) -> None:
    """Populate `best_model` and `is_best_model` on every per-skill and per-family row across
    `models` in place."""
    skills = sorted(
        {
            str(row.get("skill"))
            for m in models
            for row in (m.get("skills") or [])
            if isinstance(row, Mapping) and row.get("skill")
        }
    )
    for skill in skills:
        candidates: list[dict[str, Any]] = []
        target_rows: list[tuple[str, dict[str, Any]]] = []
        for m in models:
            ex = _model_exec_name(m)
            for row in m.get("skills") or []:
                if isinstance(row, dict) and str(row.get("skill")) == skill:
                    candidates.append({"executor": ex, **row})
                    target_rows.append((ex, row))
        best = pick_best_model(candidates)
        for ex, row in target_rows:
            row["best_model"] = best
            row["is_best_model"] = best is not None and ex == best

    families = sorted(
        {
            str(row.get("family") or row.get("skill"))
            for m in models
            for row in (m.get("suites") or [])
            if isinstance(row, Mapping) and (row.get("family") or row.get("skill"))
        }
    )
    for fam in families:
        fam_candidates: list[dict[str, Any]] = []
        fam_targets: list[tuple[str, dict[str, Any]]] = []
        for m in models:
            ex = _model_exec_name(m)
            for row in m.get("suites") or []:
                if isinstance(row, dict) and str(row.get("family") or row.get("skill")) == fam:
                    fam_candidates.append({"executor": ex, **row})
                    fam_targets.append((ex, row))
        best_fam = pick_best_model(fam_candidates)
        for ex, row in fam_targets:
            row["best_model"] = best_fam
            row["is_best_model"] = best_fam is not None and ex == best_fam


def build_skill_matrix(
    report_or_models: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Cross-model matrix per skill: `skill -> {skill, best_model, rows, by_executor}`.

    Each entry in `rows` corresponds to one executor (in `models[]` order) and carries:
    `executor`, `judge`, `lift`, `ci_low`, `ci_high`, `ci`, `verdict`, `tokens_with_median`,
    `tokens_without_median`, `tokens_delta_pct`, `t_total_with_median`,
    `t_total_without_median`, `wall_delta_pct`, `invoked`, `valid_with_reps`,
    `invocation_rate`, `t_skill_p50`, `t_skill_p90`, `ceiling_tasks`, `total_tasks`,
    `unclassified_tasks`, `needs_harder_tasks`, `best_model`, `is_best_model`.
    """
    models = (
        models_of(report_or_models)
        if isinstance(report_or_models, Mapping)
        else [m for m in report_or_models if isinstance(m, Mapping)]
    )
    per_model = [(_model_exec_name(m), _model_judge_name(m), _skill_rows_of(m)) for m in models]
    skills = sorted({s for _, _, srows in per_model for s in srows})

    matrix: dict[str, dict[str, Any]] = {}
    for skill in skills:
        raw_rows: list[dict[str, Any]] = []
        for ex, jg, srows in per_model:
            if skill not in srows:
                continue
            srow = srows[skill]
            tok_w = srow.get("tokens_with_median")
            tok_wo = srow.get("tokens_without_median")
            tok_dp = srow.get("tokens_delta_pct")
            if tok_dp is None:
                tok_dp = _delta_pct(tok_w, tok_wo)
            wall_w = srow.get("t_total_with_median")
            wall_wo = srow.get("t_total_without_median")
            wall_dp = srow.get("wall_delta_pct")
            if wall_dp is None:
                wall_dp = _delta_pct(wall_w, wall_wo)
            ci_low = srow.get("ci_low")
            ci_high = srow.get("ci_high")
            ci = srow.get("ci")
            if ci is None and ci_low is not None and ci_high is not None:
                ci = [ci_low, ci_high]
            rec_lbl = srow.get("recommendation")
            rec_rsn = srow.get("reason")
            if not rec_lbl or not rec_rsn:
                rec_lbl, rec_rsn = recommend(srow)
            raw_rows.append(
                {
                    "skill": skill,
                    "executor": ex,
                    "judge": jg,
                    "n_tasks": int(srow.get("n_tasks", 0)),
                    "with_mean": srow.get("with_mean"),
                    "without_mean": srow.get("without_mean"),
                    "lift": srow.get("lift"),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci": ci,
                    "verdict": str(srow.get("verdict", "insufficient data")),
                    "recommendation": rec_lbl,
                    "reason": rec_rsn,
                    "tokens_with_median": tok_w,
                    "tokens_without_median": tok_wo,
                    "tokens_delta_pct": tok_dp,
                    "t_total_with_median": wall_w,
                    "t_total_without_median": wall_wo,
                    "wall_delta_pct": wall_dp,
                    "invoked": int(srow.get("invoked", 0)),
                    "valid_with_reps": int(srow.get("valid_with_reps", 0)),
                    "invocation_rate": srow.get("invocation_rate"),
                    "t_skill_p50": srow.get("t_skill_p50"),
                    "t_skill_p90": srow.get("t_skill_p90"),
                    "ceiling_tasks": srow.get("ceiling_tasks"),
                    "total_tasks": srow.get("total_tasks"),
                    "unclassified_tasks": int(srow.get("unclassified_tasks") or 0),
                    "needs_harder_tasks": bool(srow.get("needs_harder_tasks")),
                }
            )
        best = pick_best_model(raw_rows)
        for r in raw_rows:
            r["best_model"] = best
            r["is_best_model"] = best is not None and r["executor"] == best
        matrix[skill] = {
            "skill": skill,
            "best_model": best,
            "rows": raw_rows,
            "by_executor": {r["executor"]: r for r in raw_rows},
        }
    return matrix


def _iter_unique_reps(model: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """All per-rep detail dicts stored under `model['skills']` and `model['suites']`."""
    reps: list[Mapping[str, Any]] = []
    for srow in model.get("skills") or []:
        if not isinstance(srow, Mapping):
            continue
        for t in (srow.get("tasks") or {}).values():
            if not isinstance(t, Mapping):
                continue
            rmap = t.get("reps") if isinstance(t.get("reps"), Mapping) else {}
            for arm in ("with", "without"):
                for rep in rmap.get(arm) or []:
                    if isinstance(rep, Mapping):
                        reps.append(rep)
    suite_arms = [str(a) for a in (model.get("suite_arms") or [])]
    for frow in model.get("suites") or []:
        if not isinstance(frow, Mapping):
            continue
        arms_to_read = [BASELINE_ARM, *(suite_arms or list(frow.get("suite_arms") or []))]
        for t in (frow.get("tasks") or {}).values():
            if not isinstance(t, Mapping):
                continue
            rmap = t.get("reps") if isinstance(t.get("reps"), Mapping) else {}
            for arm in arms_to_read:
                for rep in rmap.get(arm) or []:
                    if isinstance(rep, Mapping):
                        reps.append(rep)
    return reps


def build_model_summary(
    report_or_models: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Model-level summary per executor: `executor`, `judge`, `n_skills`, `mean_lift` across
    skills, `total_tokens`, `total_wall_s` (`total_wall`), `valid_total`, `invalid_total`,
    `total_rollouts`, and `invalid_rate`."""
    models = (
        models_of(report_or_models)
        if isinstance(report_or_models, Mapping)
        else [m for m in report_or_models if isinstance(m, Mapping)]
    )
    out: list[dict[str, Any]] = []
    for m in models:
        ex = _model_exec_name(m)
        jg = _model_judge_name(m)
        skills = [r for r in (m.get("skills") or []) if isinstance(r, Mapping)]
        suites = [r for r in (m.get("suites") or []) if isinstance(r, Mapping)]
        if "mean_lift" in m:
            mean_lift = m.get("mean_lift")
        else:
            lifts = [
                float(r["lift"])
                for r in (skills or suites)
                if r.get("lift") is not None and not isinstance(r.get("lift"), bool)
            ]
            mean_lift = float(st.mean(lifts)) if lifts else None

        reps = _iter_unique_reps(m)
        if isinstance(m.get("total_tokens"), int) and not isinstance(m.get("total_tokens"), bool):
            total_tokens = int(m["total_tokens"])
        else:
            total_tokens = sum(
                int(r["total_tokens"])
                for r in reps
                if r.get("valid")
                and isinstance(r.get("total_tokens"), int)
                and not isinstance(r.get("total_tokens"), bool)
            )

        raw_wall = m.get("total_wall_s", m.get("total_wall"))
        if isinstance(raw_wall, (int, float)) and not isinstance(raw_wall, bool):
            total_wall_s = float(raw_wall)
        else:
            total_wall_s = float(
                sum(
                    float(r["t_total_s"])
                    for r in reps
                    if r.get("valid")
                    and isinstance(r.get("t_total_s"), (int, float))
                    and not isinstance(r.get("t_total_s"), bool)
                )
            )

        if isinstance(m.get("total_rollouts"), int) and not isinstance(
            m.get("total_rollouts"), bool
        ):
            total_rollouts = int(m["total_rollouts"])
            valid_total = int(m.get("valid_total", total_rollouts - int(m.get("invalid_total", 0))))
            invalid_total = int(m.get("invalid_total", total_rollouts - valid_total))
        elif reps:
            total_rollouts = len(reps)
            valid_total = sum(1 for r in reps if r.get("valid"))
            invalid_total = total_rollouts - valid_total
        else:
            invalid_total = int(
                m.get("invalid_total")
                if m.get("invalid_total") is not None
                else sum(int(r.get("invalid_total", 0)) for r in (skills + suites))
            )
            valid_with = sum(int(r.get("valid_with_reps", 0)) for r in (skills + suites))
            valid_total = valid_with * 2
            total_rollouts = valid_total + invalid_total

        if "invalid_rate" in m:
            invalid_rate = m.get("invalid_rate")
        else:
            invalid_rate = (invalid_total / total_rollouts) if total_rollouts > 0 else None

        out.append(
            {
                "executor": ex,
                "judge": jg,
                "n_skills": len(skills),
                "mean_lift": mean_lift,
                "total_tokens": total_tokens,
                "total_wall_s": total_wall_s,
                "total_wall": total_wall_s,
                "valid_total": valid_total,
                "invalid_total": invalid_total,
                "total_rollouts": total_rollouts,
                "invalid_rate": invalid_rate,
            }
        )
    return out


def judge_pairing_note(
    report_or_models: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> str:
    """Legend note stating the `executor -> judge` pairing, whether any executors share a
    judge under cross-family `judge_for` mapping, and the `best_model` selection rule."""
    models = (
        models_of(report_or_models)
        if isinstance(report_or_models, Mapping)
        else [m for m in report_or_models if isinstance(m, Mapping)]
    )
    pairs: list[str] = []
    by_judge: dict[str, list[str]] = {}
    for m in models:
        ex = _model_exec_name(m)
        jg = _model_judge_name(m)
        pairs.append(f"`{ex}` → `{jg}`")
        by_judge.setdefault(jg, []).append(ex)
    shared = [
        f"`{jg}` grades {', '.join(f'`{e}`' for e in exs)}"
        for jg, exs in by_judge.items()
        if len(exs) > 1
    ]
    pair_str = ", ".join(pairs) if pairs else "none"
    if shared:
        shared_str = (
            f" ({'; '.join(shared)} — two executors share a judge under cross-family "
            "`judge_for` pairing so no executor is graded by its own model family)"
        )
    else:
        shared_str = (
            " (each executor is paired with a cross-family judge via `judge_for` so no "
            "executor is graded by its own model family; when 3 families are paired two "
            "executors share a judge)"
        )
    return (
        f"Judge pairing: {pair_str}{shared_str}. "
        "`best_model` ★ marks the executor with the highest rubric lift for that skill "
        "(ties → fewest WITH-arm median tokens)."
    )


def _fmt_frac(val: float | None) -> str:
    return "n/a" if val is None else f"{val:.2f}"


def _fmt_lift_ci(lift: float | None, lo: float | None, hi: float | None) -> str:
    if lift is None or lo is None or hi is None:
        return "n/a"
    return f"{lift:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def _fmt_invoked(invoked: int, valid_with: int, rate: float | None) -> str:
    if valid_with <= 0 or rate is None:
        return f"{invoked}/{valid_with} (n/a)"
    return f"{invoked}/{valid_with} ({rate * 100:.0f}%)"


def _fmt_timing(p50: float | None, p90: float | None) -> str:
    if p50 is None or p90 is None:
        return "n/a"
    return f"{p50:.1f}s / {p90:.1f}s"


def _fmt_invalid(total: int, by_reason: Mapping[str, int]) -> str:
    if total <= 0:
        return "0"
    if not by_reason:
        return str(total)
    details = ", ".join(f"{reason}: {count}" for reason, count in sorted(by_reason.items()))
    return f"{total} ({details})"


def _model_str(model: Mapping[str, Any]) -> str:
    req_exec = model.get("requested_exec_model") or model.get("executor") or DEFAULT_EXEC_MODEL
    act_exec = model.get("actual_exec_model")
    return (
        f"requested `{req_exec}`, actual `{act_exec}`"
        if act_exec
        else f"requested `{req_exec}`, actual: {MODEL_UNVERIFIED}"
    )


def _judge_str(model: Mapping[str, Any]) -> str:
    req_judge = model.get("requested_judge_model") or model.get("judge") or DEFAULT_JUDGE_MODEL
    act_judge = model.get("actual_judge_model")
    return (
        f"requested `{req_judge}`, actual `{act_judge}`"
        if act_judge
        else f"requested `{req_judge}`, actual: {MODEL_UNVERIFIED}"
    )


def _noise_str(model: Mapping[str, Any]) -> str:
    noise_rate = model.get("judge_noise_rate")
    if noise_rate is None:
        return "n/a (single judgment per rollout)"
    dis = model.get("judge_noise_disagreed", 0)
    tot = model.get("judge_noise_total", 0)
    return f"{noise_rate * 100:.1f}% ({dis}/{tot} criteria disagreed)"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Format `report` as `report.md`: header metadata and the power note, then one block per
    executor in `models[]` (its models, judge noise and the two per-skill tables)."""
    models = models_of(report)
    multi = len(models) > 1

    k = report.get("k", 1)
    seed = report.get("seed")
    seed_str = "none" if seed is None else str(seed)
    boot_seed = report.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)
    draws = report.get("bootstrap_draws", DEFAULT_BOOTSTRAP_DRAWS)
    pilot = bool(report.get("pilot"))
    min_tasks = MIN_TASKS_PILOT if pilot else MIN_TASKS_FULL
    mode_label = "pilot" if pilot else "full"
    manifest_hash = report.get("manifest_hash") or "unverified"
    power_note = report.get("power_note") or POWER_NOTE
    suite = report.get("suite") if isinstance(report.get("suite"), Mapping) else {}

    lines = ["# skillbench report", ""]
    if suite.get("name") or suite.get("hash"):
        lines.append(
            f"- **Suite**: `{suite.get('name') or 'unnamed'}` "
            f"(hash `{str(suite.get('hash') or 'unknown')[:12]}`)"
        )
    if multi:
        lines.append(
            "- **Executors**: "
            + ", ".join(
                f"`{m.get('executor') or m.get('requested_exec_model')}` "
                f"(judge `{m.get('judge') or m.get('requested_judge_model')}`)"
                for m in models
            )
        )
    else:
        lines += [
            f"- **Executor model**: {_model_str(models[0])}",
            f"- **Judge model**: {_judge_str(models[0])}",
        ]
    lines += [
        f"- **Manifest hash**: `{manifest_hash}`",
        f"- **Run scale / seed**: `k={k}`, `seed={seed_str}` "
        f"(bootstrap `seed={boot_seed}`, `{draws}` draws, mode `{mode_label}`, "
        f"min `{min_tasks}` tasks with >= `{MIN_VALID_REPS_PER_ARM}` valid reps/arm)",
    ]
    if not multi:
        lines.append(f"- **Judge-noise rate**: {_noise_str(models[0])}")
    lines.append(f"- **Statistical power**: {power_note}")
    lines.append(f"- **Hardness**: {report.get('hardness_note') or HARDNESS_NOTE}")

    if multi:
        lines += _render_cross_model_tables(report)

    for model in models:
        if multi:
            lines += [
                "",
                f"## executor `{model.get('executor') or model.get('requested_exec_model')}`",
                "",
                f"- **Executor model**: {_model_str(model)}",
                f"- **Judge model**: {_judge_str(model)}",
                f"- **Judge-noise rate**: {_noise_str(model)}",
            ]
        lines += _render_model_tables(model)

    if not multi and isinstance(report.get("models"), list) and report.get("models"):
        lines += _render_cross_model_tables(report)

    return "\n".join(lines) + "\n"


def _fmt_signed_frac(val: float | None) -> str:
    if val is None or isinstance(val, bool):
        return "n/a"
    return f"{float(val):+.2f}"


def _fmt_wall_sum(val: float | None) -> str:
    if val is None or isinstance(val, bool):
        return "n/a"
    f = float(val)
    return f"{f:.0f}s" if f.is_integer() else f"{f:.1f}s"


def _fmt_invalid_rate(invalid_total: int, total_rollouts: int, rate: float | None) -> str:
    if total_rollouts <= 0 or rate is None:
        return f"{invalid_total}/{total_rollouts} (n/a)"
    return f"{invalid_total}/{total_rollouts} ({rate * 100:.0f}%)"


def _fmt_pair_delta(
    a: float | None,
    b: float | None,
    delta_pct: float | None = None,
    *,
    suffix: str = "",
) -> str:
    pair = _fmt_pair(a, b, suffix=suffix)
    if pair == "n/a":
        return "n/a"
    if delta_pct is None:
        delta_pct = _delta_pct(a, b)
    if delta_pct is None:
        return f"{pair} (n/a)"
    return f"{pair} ({delta_pct:+.0f}%)"


def _render_cross_model_tables(report: Mapping[str, Any]) -> list[str]:
    """Render the model-level summary table, the cross-model per-skill matrix tables, and
    the judge-pairing legend."""
    raw_ms = report.get("model_summary")
    model_rows = (
        [r for r in raw_ms if isinstance(r, Mapping)]
        if isinstance(raw_ms, list) and raw_ms
        else build_model_summary(report)
    )
    raw_sm = report.get("skill_matrix")
    skill_matrix = (
        {str(k): dict(v) for k, v in raw_sm.items() if isinstance(v, Mapping)}
        if isinstance(raw_sm, Mapping) and raw_sm
        else build_skill_matrix(report)
    )
    pairing = str(report.get("judge_pairing_note") or judge_pairing_note(report))

    lines: list[str] = [
        "",
        "## Model summary",
        "",
        f"- **Legend**: {pairing}",
        "",
        "| executor | judge | skills | mean lift | total tokens | total wall | invalid rate |",
        "| :--- | :--- | ---: | :---: | ---: | ---: | :---: |",
    ]
    for mrow in model_rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(mrow.get("executor", "executor")),
                    str(mrow.get("judge", "judge")),
                    str(mrow.get("n_skills", 0)),
                    _fmt_signed_frac(mrow.get("mean_lift")),
                    str(int(mrow["total_tokens"]))
                    if isinstance(mrow.get("total_tokens"), (int, float))
                    and not isinstance(mrow.get("total_tokens"), bool)
                    else "n/a",
                    _fmt_wall_sum(mrow.get("total_wall_s", mrow.get("total_wall"))),
                    _fmt_invalid_rate(
                        int(mrow.get("invalid_total", 0)),
                        int(mrow.get("total_rollouts", 0)),
                        mrow.get("invalid_rate"),
                    ),
                )
            )
            + " |"
        )

    if skill_matrix:
        lines += ["", "## Skill × executor matrix"]
        for skill in sorted(skill_matrix):
            entry = skill_matrix[skill]
            best = entry.get("best_model")
            best_label = f"`{best}`" if best else "n/a"
            lines += [
                "",
                f"### `{skill}` (best model: {best_label})",
                "",
                "| executor | lift [95% CI] | verdict | tokens p50 W/WO (Δ %) "
                "| wall p50 W/WO (Δ %) | invocation | t_skill p50/p90 | ceiling/total |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ]
            for r in entry.get("rows") or []:
                if not isinstance(r, Mapping):
                    continue
                ex_cell = (
                    f"{r.get('executor')} ★ (best)"
                    if r.get("is_best_model")
                    else str(r.get("executor"))
                )
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            ex_cell,
                            _fmt_lift_ci(r.get("lift"), r.get("ci_low"), r.get("ci_high")),
                            str(r.get("verdict", "insufficient data")),
                            _fmt_pair_delta(
                                r.get("tokens_with_median"),
                                r.get("tokens_without_median"),
                                r.get("tokens_delta_pct"),
                            ),
                            _fmt_pair_delta(
                                r.get("t_total_with_median"),
                                r.get("t_total_without_median"),
                                r.get("wall_delta_pct"),
                                suffix="s",
                            ),
                            _fmt_invoked(
                                int(r.get("invoked", 0)),
                                int(r.get("valid_with_reps", 0)),
                                r.get("invocation_rate"),
                            ),
                            _fmt_timing(r.get("t_skill_p50"), r.get("t_skill_p90")),
                            _fmt_ceiling(r),
                        )
                    )
                    + " |"
                )
    return lines


def _fmt_skills_histogram(hist: Mapping[str, int] | None) -> str:
    if not hist:
        return "none"
    return ", ".join(f"{skill}: {count}" for skill, count in sorted(hist.items()))


def _fmt_extra_suite_cell(arm_row: Mapping[str, Any], *, is_task: bool = False) -> str:
    mean_str = _fmt_frac(arm_row.get("suite_mean"))
    if is_task:
        delta = arm_row.get("delta")
        lift_str = "n/a" if delta is None else f"{delta:+.2f}"
    else:
        lift_str = _fmt_lift_ci(
            arm_row.get("lift"), arm_row.get("ci_low"), arm_row.get("ci_high")
        )
    verdict = str(arm_row.get("verdict", "insufficient data"))
    tok = arm_row.get("tokens_suite_median")
    wall = arm_row.get("t_total_suite_median")
    tok_str = "n/a" if tok is None else f"{tok:.0f}"
    wall_str = "n/a" if wall is None else f"{wall:.0f}s"
    hist_str = _fmt_skills_histogram(arm_row.get("skills_read_histogram"))
    return (
        f"{mean_str} ({lift_str}, {verdict}, tok {tok_str}, wall {wall_str}, "
        f"skills: {hist_str})"
    )


def _render_suite_tables(report: Mapping[str, Any]) -> list[str]:
    """The per-family and per-task suite outcome tables of one model block."""
    suite_rows = list(report.get("suites") or [])
    if not suite_rows:
        return []
    suite_arms = list(report.get("suite_arms") or [])
    if not suite_arms:
        suite_arms = list(
            suite_rows[0].get("suite_arms")
            or [suite_rows[0].get("suite_arm") or "suite"]
        )
    primary_arm = str(suite_arms[0])
    extra_arms = [str(a) for a in suite_arms[1:]]

    fam_headers = [
        "family",
        "tasks",
        primary_arm,
        BASELINE_ARM,
        "lift [95% CI]",
        "verdict",
        "invoked",
        "t_skill p50/p90",
        "tokens p50 S/B",
        "wall time p50 S/B",
        "skills read",
        "ceiling/total",
        *extra_arms,
    ]
    fam_align = [
        ":---",
        "---:",
        "---:",
        "---:",
        ":---:",
        ":---:",
        ":---:",
        ":---:",
        ":---:",
        ":---:",
        ":---",
        ":---",
        *(":---" for _ in extra_arms),
    ]
    lines = [
        "",
        "Suite mode (per family vs built-in baseline):",
        "",
        "| " + " | ".join(fam_headers) + " |",
        "| " + " | ".join(fam_align) + " |",
    ]
    for row in suite_rows:
        by_arm = row.get("by_arm") if isinstance(row.get("by_arm"), Mapping) else {}
        primary = by_arm.get(primary_arm, row)
        cells = [
            str(row.get("family") or row.get("skill")),
            str(row.get("n_tasks", 0)),
            _fmt_frac(primary.get("suite_mean", row.get("suite_mean"))),
            _fmt_frac(primary.get("baseline_mean", row.get("baseline_mean"))),
            _fmt_lift_ci(
                primary.get("lift", row.get("lift")),
                primary.get("ci_low", row.get("ci_low")),
                primary.get("ci_high", row.get("ci_high")),
            ),
            str(primary.get("verdict", row.get("verdict", "insufficient data"))),
            _fmt_invoked(
                int(primary.get("invoked", row.get("invoked", 0))),
                int(primary.get("valid_suite_reps", row.get("valid_suite_reps", 0))),
                primary.get("invocation_rate", row.get("invocation_rate")),
            ),
            _fmt_timing(
                primary.get("t_skill_p50", row.get("t_skill_p50")),
                primary.get("t_skill_p90", row.get("t_skill_p90")),
            ),
            _fmt_pair(
                primary.get("tokens_suite_median", row.get("tokens_suite_median")),
                primary.get("tokens_baseline_median", row.get("tokens_baseline_median")),
            ),
            _fmt_pair(
                primary.get("t_total_suite_median", row.get("t_total_suite_median")),
                primary.get("t_total_baseline_median", row.get("t_total_baseline_median")),
                suffix="s",
            ),
            _fmt_skills_histogram(
                primary.get("skills_read_histogram", row.get("skills_read_histogram"))
            ),
            _fmt_ceiling(row),
        ]
        for arm in extra_arms:
            cells.append(_fmt_extra_suite_cell(by_arm.get(arm) or {}, is_task=False))
        lines.append("| " + " | ".join(cells) + " |")

    task_headers = [
        "family",
        "task",
        primary_arm,
        BASELINE_ARM,
        "lift [95% CI]",
        "verdict",
        "tokens p50 S/B",
        "wall time p50 S/B",
        "skills read",
        "hardness",
        *extra_arms,
    ]
    task_align = [
        ":---",
        ":---",
        "---:",
        "---:",
        ":---:",
        ":---:",
        ":---:",
        ":---:",
        ":---",
        ":---",
        *(":---" for _ in extra_arms),
    ]
    lines += [
        "",
        "Suite tasks (per task vs built-in baseline):",
        "",
        "| " + " | ".join(task_headers) + " |",
        "| " + " | ".join(task_align) + " |",
    ]
    for row in suite_rows:
        family_name = str(row.get("family") or row.get("skill"))
        tasks_map = row.get("tasks") if isinstance(row.get("tasks"), Mapping) else {}
        for tid, task in sorted(tasks_map.items()):
            t_by_arm = task.get("by_arm") if isinstance(task.get("by_arm"), Mapping) else {}
            t_primary = t_by_arm.get(primary_arm, task)
            t_cells = [
                family_name,
                str(tid),
                _fmt_frac(t_primary.get("suite_mean", task.get("suite_mean"))),
                _fmt_frac(task.get("baseline_mean")),
                _fmt_lift_ci(
                    t_primary.get("lift", task.get("lift")),
                    t_primary.get("ci_low", task.get("ci_low")),
                    t_primary.get("ci_high", task.get("ci_high")),
                ),
                str(t_primary.get("verdict", task.get("verdict", "insufficient data"))),
                _fmt_pair(
                    t_primary.get("tokens_suite_median", task.get("tokens_suite_median")),
                    task.get("tokens_baseline_median"),
                ),
                _fmt_pair(
                    t_primary.get("t_total_suite_median", task.get("t_total_suite_median")),
                    task.get("t_total_baseline_median"),
                    suffix="s",
                ),
                _fmt_skills_histogram(
                    t_primary.get(
                        "skills_read_histogram", task.get("skills_read_histogram")
                    )
                ),
                str(task.get("hardness", HARDNESS_UNCLASSIFIED)),
            ]
            for arm in extra_arms:
                t_cells.append(
                    _fmt_extra_suite_cell(t_by_arm.get(arm) or {}, is_task=True)
                )
            lines.append("| " + " | ".join(t_cells) + " |")
    return lines


def _render_model_tables(report: Mapping[str, Any]) -> list[str]:
    """The per-skill outcome table, deterministic-checks table, and optional suite tables
    of one model block."""
    lines: list[str] = []
    has_skills = bool(report.get("skills"))
    has_suites = bool(report.get("suites"))

    if has_skills or not has_suites:
        lines += [
            "",
            "| skill | tasks | WITH | WITHOUT | lift [95% CI] | verdict | invoked | t_skill p50/p90 "
            "| invalid | ceiling/total |",
            "| :--- | ---: | ---: | ---: | :---: | :---: | :---: | :---: | :--- | :--- |",
        ]

        for row in report.get("skills") or []:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(row["skill"]),
                        str(row["n_tasks"]),
                        _fmt_frac(row.get("with_mean")),
                        _fmt_frac(row.get("without_mean")),
                        _fmt_lift_ci(row.get("lift"), row.get("ci_low"), row.get("ci_high")),
                        str(row.get("verdict", "insufficient data")),
                        _fmt_invoked(
                            int(row.get("invoked", 0)),
                            int(row.get("valid_with_reps", 0)),
                            row.get("invocation_rate"),
                        ),
                        _fmt_timing(row.get("t_skill_p50"), row.get("t_skill_p90")),
                        _fmt_invalid(
                            int(row.get("invalid_total", 0)),
                            row.get("invalid_by_reason") or {},
                        ),
                        _fmt_ceiling(row),
                    )
                )
                + " |"
            )

        lines += [
            "",
            "Deterministic checks and cost (valid reps; not part of the rubric lift):",
            "",
            "| skill | tests pass WITH | tests pass WITHOUT | diff lines p50 W/WO "
            "| tokens p50 W/WO | wall time p50 W/WO |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for row in report.get("skills") or []:
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(row["skill"]),
                        _fmt_verify(row.get("verify_with_passed"), row.get("verify_with_total")),
                        _fmt_verify(
                            row.get("verify_without_passed"), row.get("verify_without_total")
                        ),
                        _fmt_pair(
                            row.get("diff_lines_with_median"),
                            row.get("diff_lines_without_median"),
                        ),
                        _fmt_pair(
                            row.get("tokens_with_median"), row.get("tokens_without_median")
                        ),
                        _fmt_pair(
                            row.get("t_total_with_median"),
                            row.get("t_total_without_median"),
                            suffix="s",
                        ),
                    )
                )
                + " |"
            )
    if has_suites:
        lines += _render_suite_tables(report)
    return lines


def _fmt_verify(passed: Any, total: Any) -> str:
    if not isinstance(total, int) or total <= 0 or not isinstance(passed, int):
        return "n/a"
    return f"{passed}/{total} ({passed / total * 100:.0f}%)"


def _fmt_pair(a: float | None, b: float | None, *, suffix: str = "") -> str:
    def one(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.0f}{suffix}"

    if a is None and b is None:
        return "n/a"
    return f"{one(a)} / {one(b)}"


def _fmt_ceiling(row: Mapping[str, Any]) -> str:
    """`ceiling/total` of a per-skill row (v1 rows without the fields render `n/a`), with the
    `needs harder tasks` flag and the unclassified count when present."""
    ceiling = row.get("ceiling_tasks")
    total = row.get("total_tasks")
    if not isinstance(ceiling, int) or not isinstance(total, int):
        return "n/a"
    out = f"{ceiling}/{total}"
    unclassified = row.get("unclassified_tasks") or 0
    if unclassified:
        out += f" ({unclassified} unclassified)"
    if row.get("needs_harder_tasks"):
        out += f" — {NEEDS_HARDER_TASKS}"
    return out


def _fmt_hardness_cell(task: Mapping[str, Any] | None) -> str:
    if task is None:
        return "—"
    label = str(task.get("hardness") or HARDNESS_UNCLASSIFIED)
    missed = list(task.get("baseline_missed") or [])
    if label == HARDNESS_UNCLASSIFIED:
        n = len(missed)
        return f"{label} ({n} valid rep{'' if n == 1 else 's'})"
    return f"{label} (missed {', '.join(str(m) for m in missed)})"


def render_hardness_markdown(hardness: Mapping[str, Any]) -> str:
    """The task x executor hardness table (`build_hardness` / `hardness_of_report` output):
    one row per task with every executor's label and the criteria its control arm missed per
    valid rep, then `ceiling/total` per skill and executor with the `needs harder tasks` flag."""
    models = [m for m in hardness.get("models") or [] if isinstance(m, Mapping)]
    suite = hardness.get("suite") if isinstance(hardness.get("suite"), Mapping) else {}
    lines = ["# skillbench task hardness", ""]
    if suite.get("name") or suite.get("hash"):
        lines.append(
            f"- **Suite**: `{suite.get('name') or 'unnamed'}` "
            f"(hash `{str(suite.get('hash') or 'unknown')[:12]}`)"
        )
    if hardness.get("manifest_hash"):
        lines.append(f"- **Manifest hash**: `{hardness['manifest_hash']}`")
    lines.append(
        "- **Executors**: "
        + (
            ", ".join(
                f"`{m.get('executor')}` (control arm `{m.get('control_arm')}`, "
                f"judge `{m.get('judge')}`)"
                for m in models
            )
            or "none"
        )
    )
    lines.append(f"- **Rule**: {hardness.get('hardness_note') or HARDNESS_NOTE}")

    executors = [str(m.get("executor")) for m in models]
    task_ids = sorted({tid for m in models for tid in (m.get("tasks") or {})})
    lines += [
        "",
        "| skill | task | min fail | " + " | ".join(executors) + " |",
        "| :--- | :--- | ---: | " + " | ".join(":---" for _ in executors) + " |",
    ]
    for tid in task_ids:
        cells = [(m.get("tasks") or {}).get(tid) for m in models]
        first = next((c for c in cells if c), {})
        thr = first.get("min_baseline_fail", DEFAULT_MIN_BASELINE_FAIL)
        lines.append(
            f"| {first.get('skill') or tid.split('/', 1)[0]} | {tid.split('/', 1)[-1]} | {thr} | "
            + " | ".join(_fmt_hardness_cell(c) for c in cells)
            + " |"
        )

    skills = sorted({s for m in models for s in (m.get("skills") or {})})
    lines += [
        "",
        "Per skill: ceiling tasks / total tasks (flagged when more than "
        f"{CEILING_FLAG_SHARE * 100:.0f} % are at the ceiling; ceiling tasks stay in the lift):",
        "",
        "| skill | " + " | ".join(executors) + " |",
        "| :--- | " + " | ".join(":---" for _ in executors) + " |",
    ]
    for skill in skills:
        lines.append(
            f"| {skill} | "
            + " | ".join(_fmt_ceiling((m.get("skills") or {}).get(skill) or {}) for m in models)
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_report(
    report: Mapping[str, Any],
    out_dir: Path,
    *,
    history: Sequence[Mapping[str, Any]] | Path | str | None = None,
) -> tuple[Path, Path, Path]:
    """Write `report.json`, `report.md`, `report.html` and `summary.md` inside `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    html_path = out_dir / "report.html"
    summary_md_path = out_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    html_path.write_text(render_html(report, history=history), encoding="utf-8")
    summary_md_path.write_text(
        render_summary_markdown(report, history=history), encoding="utf-8"
    )
    return (json_path, md_path, html_path)


def _resolve_cells(args: argparse.Namespace) -> tuple[list[ReportCell], Path]:
    """The cells to report on and the default output directory, from `--results-root`,
    `--with-dir` / `--without-dir`, or the conventional `adapt/` namespaces of `--k` / `--seed`."""
    explicit_pair = args.with_dir is not None or args.without_dir is not None
    if args.results_root is not None:
        if explicit_pair:
            raise ValueError("--results-root cannot be combined with --with-dir / --without-dir")
        root = Path(args.results_root)
        return (discover_cells(root), root.parent)
    if explicit_pair:
        if args.with_dir is None or args.without_dir is None:
            raise ValueError("pass both --with-dir and --without-dir, or neither")
        with_dir, without_dir = Path(args.with_dir), Path(args.without_dir)
    else:
        with_ns = SCHED.ns("skillbench", "agy", "agg", args.k, seed=args.seed, tag="with")
        without_ns = SCHED.ns("skillbench", "agy", "agg", args.k, seed=args.seed, tag="without")
        base = Path(args.runs_root) if args.runs_root is not None else _default_runs_root()
        adapt = base / "adapt"
        with_dir, without_dir = adapt / with_ns, adapt / without_ns
    out_dir = with_dir.parent.parent if with_dir.parent.name == "adapt" else with_dir.parent
    return (cells_from_dirs(with_dir, without_dir), out_dir)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skillbench-report",
        description=__doc__,
        formatter_class=C.HelpFormatter,
    )
    p.add_argument("--k", type=int, default=1, help="base rollouts per task in the sweep")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="stream-order seed of the sweep (also default bootstrap seed)",
    )
    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="PRNG seed for the two-level bootstrap (defaults to --seed if set, else 0)",
    )
    p.add_argument(
        "--draws",
        type=int,
        default=DEFAULT_BOOTSTRAP_DRAWS,
        help="number of two-level bootstrap resamples",
    )
    p.add_argument(
        "--pilot",
        action="store_true",
        help="use the pilot verdict threshold (>= 2 tasks with >= 2 valid reps/arm)",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="skillbench runs root (default: $HARNESS2_RUNS_DIR/skillbench)",
    )
    p.add_argument("--with-dir", type=Path, default=None, help="explicit WITH namespace dir")
    p.add_argument("--without-dir", type=Path, default=None, help="explicit WITHOUT namespace dir")
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="matrix layout: runs/<id>/results holding <executor>/<arm-label>/ namespaces; "
        "every judged pair becomes one models[] entry (default out-dir: its parent)",
    )
    C.add_suite_arg(p)
    C.add_suites_arg(p)
    p.add_argument(
        "--history",
        type=Path,
        default=None,
        help="path to history.jsonl for sparkline trends (default: history.jsonl next to suite.toml)",
    )
    p.add_argument(
        "--hardness-only",
        action="store_true",
        help="print the task x executor hardness table (informative / ceiling / unclassified "
        "per executor, from the control arm alone) and exit without writing a report; works "
        "on control-arm namespaces alone, e.g. after `skillbench-matrix --arms baseline`",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="write only summary.md and minimal summary.html inside --out-dir",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory where report.json, report.md and report.html (or summary.md + summary.html) are written",
    )
    return p


def _build_history_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skillbench-report history",
        description="Append to or inspect history.jsonl next to suite.toml.",
        formatter_class=C.HelpFormatter,
    )
    sub = p.add_subparsers(dest="action", required=True)

    ap = sub.add_parser(
        "append",
        help="idempotently append per-(date, suite_hash, executor, skill) rows to history.jsonl",
        formatter_class=C.HelpFormatter,
    )
    ap.add_argument(
        "target",
        nargs="?",
        default=None,
        help="optional path to report.json or a results directory",
    )
    ap.add_argument("--report", type=Path, default=None, help="path to report.json")
    ap.add_argument("--history", type=Path, default=None, help="explicit history.jsonl path")
    ap.add_argument(
        "--date",
        default=None,
        help="override YYYY-MM-DD date stamp for appended history rows",
    )
    ap.add_argument("--k", type=int, default=1, help="base rollouts per task in the sweep")
    ap.add_argument("--seed", type=int, default=None, help="stream-order seed of the sweep")
    ap.add_argument("--bootstrap-seed", type=int, default=None, help="bootstrap seed")
    ap.add_argument("--draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS, help="bootstrap draws")
    ap.add_argument("--pilot", action="store_true", help="use pilot threshold")
    ap.add_argument("--runs-root", type=Path, default=None, help="skillbench runs root")
    ap.add_argument("--with-dir", type=Path, default=None, help="explicit WITH namespace dir")
    ap.add_argument("--without-dir", type=Path, default=None, help="explicit WITHOUT namespace dir")
    ap.add_argument("--results-root", type=Path, default=None, help="matrix results directory")
    C.add_suite_arg(ap)
    C.add_suites_arg(ap)

    sp = sub.add_parser(
        "show",
        help="render history.jsonl rows (optionally filtered by --skill / --executor)",
        formatter_class=C.HelpFormatter,
    )
    sp.add_argument(
        "target",
        nargs="?",
        default=None,
        help="optional path to history.jsonl or suite.toml",
    )
    sp.add_argument("--history", type=Path, default=None, help="explicit history.jsonl path")
    sp.add_argument("--skill", default=None, help="filter to a single skill or family")
    sp.add_argument("--executor", default=None, help="filter to a single executor")
    sp.add_argument("--json", action="store_true", help="emit JSON array instead of Markdown")
    C.add_suite_arg(sp)
    return p


def _load_or_build_report_for_history(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path | None]:
    report_candidate = args.report or (Path(args.target) if args.target else None)
    if report_candidate is not None:
        cand = Path(report_candidate).expanduser()
        if cand.is_file():
            data = json.loads(cand.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{cand}: expected a JSON object")
            return data, cand.parent
        if cand.is_dir():
            rjson = cand / "report.json"
            if rjson.is_file():
                data = json.loads(rjson.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError(f"{rjson}: expected a JSON object")
                return data, cand
            cells = discover_cells(cand)
            rep = build_report(
                cells,
                k=args.k,
                seed=args.seed,
                bootstrap_seed=args.bootstrap_seed,
                draws=args.draws,
                pilot=True if args.pilot else None,
                suites=args.suites,
            )
            return rep, cand.parent
        raise FileNotFoundError(f"report path does not exist: {cand}")

    cells, default_out = _resolve_cells(args)
    rep = build_report(
        cells,
        k=args.k,
        seed=args.seed,
        bootstrap_seed=args.bootstrap_seed,
        draws=args.draws,
        pilot=True if args.pilot else None,
        suites=args.suites,
    )
    return rep, default_out


def _run_history_cli(argv: Sequence[str]) -> int:
    args = _build_history_parser().parse_args(list(argv))
    C.apply_suite(args.suite)
    if args.action == "append":
        report, fallback_dir = _load_or_build_report_for_history(args)
        hpath = (
            Path(args.history).expanduser().resolve()
            if args.history is not None
            else default_history_path(args.suite, fallback_dir=fallback_dir)
        )
        merged = append_history(hpath, report, date=args.date)
        print(f"updated {hpath} ({len(merged)} records)")
        return 0
    if args.action == "show":
        if args.history is not None:
            hpath = Path(args.history).expanduser().resolve()
        elif args.target is not None:
            hpath = default_history_path(args.target)
        else:
            hpath = default_history_path(args.suite)
        records = read_history(hpath)
        if args.json:
            filtered = [
                r
                for r in records
                if (not args.skill or (r.get("skill") or r.get("family")) == args.skill)
                and (not args.executor or r.get("executor") == args.executor)
            ]
            print(json.dumps(filtered, indent=2))
        else:
            print(
                render_history_markdown(
                    records, skill=args.skill, executor=args.executor
                ),
                end="",
            )
        return 0
    raise ValueError(f"unknown history action: {args.action!r}")


def main(argv: list[str] | None = None) -> int:
    eff_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if eff_argv and eff_argv[0] == "history":
            return _run_history_cli(eff_argv[1:])
        args = _build_parser().parse_args(eff_argv)
        if args.k < 1:
            raise ValueError("--k must be positive")
        C.apply_suite(args.suite)
        cells, default_out = _resolve_cells(args)
        if args.hardness_only:
            control_cells = [c for c in cells if c.arm_label in CONTROL_LABELS]
            print(render_hardness_markdown(build_hardness(control_cells)), end="")
            return 0
        report = build_report(
            cells,
            k=args.k,
            seed=args.seed,
            bootstrap_seed=args.bootstrap_seed,
            draws=args.draws,
            pilot=True if args.pilot else None,
            suites=args.suites,
            history=args.history,
        )
        out_dir = args.out_dir if args.out_dir is not None else default_out
        if args.summary_only:
            summary_md, summary_html = write_summary(
                report, out_dir, history=args.history
            )
            print(summary_md.read_text(encoding="utf-8"), end="")
            print(f"wrote {summary_md} and {summary_html}")
            return 0
        json_path, md_path, html_path = write_report(
            report, out_dir, history=args.history
        )
        print(md_path.read_text(encoding="utf-8"), end="")
        print(f"wrote {md_path}, {json_path} and {html_path}")
        return 0
    except (FileNotFoundError, ValueError, config.ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
