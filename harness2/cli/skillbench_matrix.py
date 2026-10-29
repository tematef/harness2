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

"""Run a skillbench model x arm evaluation matrix with pre-run budget estimation.

Expands `(executor, arm, task, rep)` jobs for a suite, validates every executor's cross-family
judge mapping from `[matrix.judge_for]`, prints a rollout / wall-time / token budget table
estimated from prior `skill_timing.json` or `history.jsonl` records (falling back to 3.0 min
per rollout when an executor has no history), enforces `--max-rollouts` and interactive TTY
confirmation (unless `--yes`), drives `run_sweep.main` and `judge.main --bases` in-process per
`(executor, arm)` cell under `executor_env`, and writes `matrix.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import config
from ..benchmarks.skillbench import (
    SKILL_TIMING_FILE,
    SkillbenchAdapter,
    _default_runs_root,
    _path_segment,
    check_manifest,
)
from ..benchmarks.skillbench import (
    tasks_root as resolve_tasks_root,
)
from ..benchmarks.skillbench_suite import (
    SuiteConfig,
    SuiteError,
    _family_prefix,
    _judge_prefix,
    active_suite,
    load_suite,
)
from ..core import schedule
from ..substrate.agy import (
    ARMS as SKILL_ARMS,
)
from ..substrate.agy import (
    BASELINE_ARM,
    arm_kind,
    executor_env,
    suite_arm_label,
)
from . import _common as C
from . import judge, run_sweep

ARMS_MODES = ("skill", "suite", "both", "baseline")
DEFAULT_K = 3
DEFAULT_PARALLEL = 3
DEFAULT_ROLLOUT_MINUTES = 3.0
DEFAULT_ROLLOUT_WALL_S = DEFAULT_ROLLOUT_MINUTES * 60.0
MATRIX_JSON = "matrix.json"
CELL_DONE_JSON = ".cell_done.json"
HISTORY_JSONL = "history.jsonl"


class BudgetDict(dict):
    """A `dict` that also supports attribute access (`budget.rollouts` == `budget["rollouts"]`)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class MatrixJob(BudgetDict):
    """One `(executor, arm, task_id, rep)` rollout job in a matrix.

    Subclasses `BudgetDict` (`dict`) so both attribute access (`job.executor`, `job.arm`,
    `job.task_id`, `job.rep`, `job.judge`, `job.done`) and dict indexing (`job["executor"]`)
    work seamlessly.
    """

    def __init__(
        self,
        executor: str,
        judge: str,
        arm: str,
        task_id: str,
        rep: int,
        *,
        arm_kind: str = "skill",
        done: bool = False,
    ) -> None:
        super().__init__(
            executor=str(executor),
            judge=str(judge),
            arm=str(arm),
            task_id=str(task_id),
            task=str(task_id),
            rep=int(rep),
            arm_kind=str(arm_kind),
            done=bool(done),
        )


_csv = C.csv_list


def resolve_judge(
    executor: str,
    suite_or_judge_for: SuiteConfig | Mapping[str, str],
) -> str:
    """Resolve and validate the judge model for `executor` by longest prefix in `judge_for`.

    Refuses missing mappings and same-prefix / same-family pairs (`SuiteError`).
    """
    if isinstance(suite_or_judge_for, SuiteConfig):
        return suite_or_judge_for.judge_for_executor(executor)
    if not isinstance(suite_or_judge_for, Mapping):
        raise TypeError(f"expected SuiteConfig or judge_for mapping, got {type(suite_or_judge_for)!r}")
    prefix = _judge_prefix(suite_or_judge_for, executor)
    if prefix is None:
        raise SuiteError(
            f"no [matrix.judge_for] prefix matches executor {executor!r}; add e.g. "
            f"\"{_family_prefix(executor)}\" = \"<judge model id>\""
        )
    judge_id = str(suite_or_judge_for[prefix])
    fam = _family_prefix(executor)
    if judge_id.startswith(prefix) or judge_id.startswith(fam) or judge_id == executor:
        shared = prefix if judge_id.startswith(prefix) else fam
        raise SuiteError(
            f"judge {judge_id!r} shares the prefix {shared!r} with executor {executor!r}; "
            f"a model must not grade its own family"
        )
    return judge_id


def arms_for_mode(
    arms_mode: str,
    suite: SuiteConfig,
    extra_suites: Sequence[SuiteConfig] = (),
) -> tuple[str, ...]:
    """Arm labels expanded for `--arms skill|suite|both|baseline`."""
    if arms_mode not in ARMS_MODES:
        raise ValueError(f"--arms must be one of {ARMS_MODES}, got {arms_mode!r}")
    suite_arms = (
        BASELINE_ARM,
        suite_arm_label(suite.name),
        *(suite_arm_label(s.name) for s in extra_suites),
    )
    if arms_mode == "skill":
        return tuple(SKILL_ARMS)
    if arms_mode == "suite":
        return tuple(dict.fromkeys(suite_arms))
    if arms_mode == "both":
        return tuple(dict.fromkeys((*SKILL_ARMS, *suite_arms)))
    # "baseline": control arms only (`without` for skill tasks, `baseline` for suite tasks)
    return ("without", BASELINE_ARM)


def _resolve_suite_and_extras(
    suite: SuiteConfig | Path | str | None,
    suites: Iterable[SuiteConfig | Path | str] = (),
) -> tuple[SuiteConfig, tuple[SuiteConfig, ...]]:
    if isinstance(suite, SuiteConfig):
        main_suite = suite
    else:
        if suite is not None and str(suite):
            C.apply_suite(suite)
        main_suite = active_suite(suite)
    extras = tuple(
        s if isinstance(s, SuiteConfig) else load_suite(s) for s in C.parse_suites(suites)
    )
    return main_suite, extras


def _resolve_executors(
    executors: Sequence[str] | str | None,
    suite: SuiteConfig,
) -> tuple[str, ...]:
    parsed = _csv(executors)
    if parsed:
        return parsed
    if suite.executors:
        return tuple(suite.executors)
    raise SuiteError(
        f"suite {suite.name!r} defines no [matrix].executors and --executors was not given; "
        f"pass --executors <model,...> or list executors under [matrix] in suite.toml"
    )


def _cell_dirs(
    base_runs_root: Path,
    executor: str,
    arm: str,
    k: int,
    seed: int | None,
) -> tuple[Path, Path]:
    """`(results_cell_dir, adapt_ns_dir)` for one `(executor, arm)` cell."""
    seg = _path_segment(executor)
    results_cell_dir = base_runs_root / "results" / seg / arm
    ns = schedule.ns("skillbench", "agy", "agg", k, seed=seed, tag=arm)
    adapt_ns_dir = base_runs_root / "adapt" / seg / ns
    return results_cell_dir, adapt_ns_dir


def _completed_reps_in_dir(ns_dir: Path, k: int) -> set[tuple[str, int]]:
    """Completed `(task_id, rep_idx)` pairs recorded in `ns_dir`."""
    done: set[tuple[str, int]] = set()
    if not ns_dir.is_dir():
        return done

    marker = ns_dir / CELL_DONE_JSON
    if marker.is_file():
        try:
            raw = json.loads(marker.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for item in raw.get("completed") or []:
                    if isinstance(item, list) and len(item) == 2:
                        done.add((str(item[0]), int(item[1])))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    judged_path = ns_dir / "base_judged.json"
    if not judged_path.is_file():
        return done
    try:
        rows = json.loads(judged_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return done
    if not isinstance(rows, list):
        return done

    graded_labels: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        tid = str(row.get("task") or row.get("task_id") or "")
        label = str(row.get("label") or "")
        if tid and label and row.get("score") is not None:
            graded_labels.setdefault(tid, set()).add(label)

    failed_labels: dict[str, set[str]] = {}
    for sp in ns_dir.glob("*/summary.json"):
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary, Mapping):
            continue
        tid = str(summary.get("task_id") or "")
        if not tid:
            continue
        if summary.get("status") not in ("bases-only", "ok", "skipped"):
            for rep_idx in range(1, k + 1):
                failed_labels.setdefault(tid, set()).add(f"base{rep_idx}")
            continue
        for entry in summary.get("base") or []:
            if isinstance(entry, Mapping) and entry.get("ok") is False:
                lbl = str(entry.get("label") or "")
                if lbl:
                    failed_labels.setdefault(tid, set()).add(lbl)

    for tid, labels in graded_labels.items():
        bad = failed_labels.get(tid, set())
        for rep_idx in range(1, k + 1):
            lbl = f"base{rep_idx}"
            if lbl in labels and lbl not in bad:
                done.add((tid, rep_idx))
    return done


def _completed_reps(
    results_cell_dir: Path,
    adapt_ns_dir: Path,
    *,
    k: int,
) -> set[tuple[str, int]]:
    return _completed_reps_in_dir(results_cell_dir, k) | _completed_reps_in_dir(adapt_ns_dir, k)


def expand_jobs(
    suite: SuiteConfig | Path | str | None = None,
    *,
    executors: Sequence[str] | str | None = None,
    k: int = DEFAULT_K,
    arms: str = "both",
    suites: Iterable[SuiteConfig | Path | str] = (),
    split: str = "all",
    split_file: Path | None = None,
    seed: int | None = None,
    only_domains: Sequence[str] | str = (),
    max_tasks: int | None = None,
    pilot: bool = False,
    runs_root: Path | None = None,
    tasks_root: Path | None = None,
    judge_for: Mapping[str, str] | None = None,
) -> list[MatrixJob]:
    """Expand `(executor, arm, task_id, rep)` jobs for the requested `--arms` mode.

    Validates every executor's judge mapping before expanding tasks. Marks `job.done = True`
    when that `(executor, arm, task_id, rep)` is already completed under `runs_root`.
    """
    if isinstance(k, bool) or int(k) < 1:
        raise ValueError(f"--k must be >= 1, got {k!r}")
    k = int(k)
    main_suite, extra_suites = _resolve_suite_and_extras(suite, suites)
    exec_list = _resolve_executors(executors, main_suite)
    judge_lookup: SuiteConfig | Mapping[str, str] = (
        judge_for if judge_for is not None else main_suite
    )
    judges = {ex: resolve_judge(ex, judge_lookup) for ex in exec_list}
    arm_labels = arms_for_mode(arms, main_suite, extra_suites)
    only_dom = _csv(only_domains)
    base_runs = Path(runs_root).expanduser().resolve() if runs_root is not None else _default_runs_root()

    stream_by_arm: dict[str, list[str]] = {}
    for arm in arm_labels:
        adapter = SkillbenchAdapter(
            arm,
            split_file=split_file,
            pilot=True if pilot else None,
            root=tasks_root,
            runs_root=base_runs,
            suite=main_suite,
            suites=extra_suites,
            executor=exec_list[0],
        )
        by_dom = schedule.task_stream(
            adapter,
            split=split,
            seed=seed,
            only_domains=only_dom,
            max_tasks=max_tasks,
        )
        stream_by_arm[arm] = [tid for dom_tasks in by_dom.values() for tid in dom_tasks]

    jobs: list[MatrixJob] = []
    for ex in exec_list:
        judge_id = judges[ex]
        for arm in arm_labels:
            tids = stream_by_arm.get(arm) or []
            if not tids:
                continue
            results_cell_dir, adapt_ns_dir = _cell_dirs(base_runs, ex, arm, k, seed)
            done_reps = _completed_reps(results_cell_dir, adapt_ns_dir, k=k)
            kind = arm_kind(arm)
            for tid in tids:
                for rep in range(1, k + 1):
                    jobs.append(
                        MatrixJob(
                            executor=ex,
                            judge=judge_id,
                            arm=arm,
                            task_id=tid,
                            rep=rep,
                            arm_kind=kind,
                            done=(tid, rep) in done_reps,
                        )
                    )
    return jobs


# --------------------------------------------------------------------------------------------
# Budget estimation from history
# --------------------------------------------------------------------------------------------


def _num(val: Any) -> float | None:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def _extract_samples_from_record(
    rec: Mapping[str, Any],
    wall_by_exec: dict[str, list[float]],
    tok_by_exec: dict[str, list[float]],
    *,
    default_executor: str | None = None,
) -> None:
    if rec.get("valid") is False:
        return
    ex = (
        rec.get("executor")
        or rec.get("requested_model")
        or rec.get("model")
        or default_executor
    )
    if not isinstance(ex, str) or not ex.strip():
        return
    ex = ex.strip()

    wall_s: float | None = None
    for sec_key in ("t_total_s", "wall_s", "median_wall_s", "duration_s", "est_wall_s"):
        v = _num(rec.get(sec_key))
        if v is not None and v > 0:
            wall_s = v
            break
    if wall_s is None:
        for min_key in ("wall_min", "median_wall_min", "est_minutes", "minutes"):
            v = _num(rec.get(min_key))
            if v is not None and v > 0:
                wall_s = v * 60.0
                break
    if wall_s is not None:
        wall_by_exec.setdefault(ex, []).append(wall_s)

    for tok_key in ("total_tokens", "tokens", "median_tokens", "est_tokens"):
        v = _num(rec.get(tok_key))
        if v is not None and v >= 0:
            tok_by_exec.setdefault(ex, []).append(v)
            break


def _read_records_from_source(source: Any) -> list[dict[str, Any]]:
    """Read raw history / timing record dicts from a path, mapping, or iterable."""
    out: list[dict[str, Any]] = []
    if source is None:
        return out
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.exists():
            return out
        if path.is_dir():
            for tf in sorted(path.rglob(SKILL_TIMING_FILE)):
                try:
                    rec = json.loads(tf.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(rec, Mapping):
                    out.append(dict(rec))
            hist_file = path / HISTORY_JSONL
            if hist_file.is_file():
                out.extend(_read_records_from_source(hist_file))
            return out
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return out
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, Mapping):
                    out.append(dict(obj))
            return out
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return out
        return _read_records_from_source(loaded)

    if isinstance(source, Mapping):
        if any(
            k in source
            for k in ("executor", "requested_model", "model", "t_total_s", "wall_s", "total_tokens", "tokens")
        ):
            return [dict(source)]
        for ex, val in source.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out.append({"executor": str(ex), "t_total_s": float(val)})
            elif isinstance(val, Mapping):
                out.append({"executor": str(ex), **val})
            elif isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
                for item in val:
                    if isinstance(item, Mapping):
                        out.append({"executor": str(ex), **item})
                    elif isinstance(item, (int, float)) and not isinstance(item, bool):
                        out.append({"executor": str(ex), "t_total_s": float(item)})
        return out

    if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        for item in source:
            if isinstance(item, Mapping):
                out.append(dict(item))
            elif isinstance(item, (str, Path)):
                out.extend(_read_records_from_source(item))
    return out


def _parse_history_records(
    history: Any,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Turn `history` into `(wall_seconds_by_executor, tokens_by_executor)`."""
    wall_by_exec: dict[str, list[float]] = {}
    tok_by_exec: dict[str, list[float]] = {}
    for rec in _read_records_from_source(history):
        _extract_samples_from_record(rec, wall_by_exec, tok_by_exec)
    return wall_by_exec, tok_by_exec


def load_history(
    source: Any = None,
    *,
    suite: SuiteConfig | None = None,
    runs_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect historical rollout timing / `history.jsonl` records for budget estimation."""
    if source is not None:
        return _read_records_from_source(source)
    records: list[dict[str, Any]] = []
    if suite is not None and suite.path is not None:
        cand = suite.path.parent / HISTORY_JSONL
        if cand.is_file():
            records.extend(_read_records_from_source(cand))
    if runs_root is not None:
        res_dir = Path(runs_root) / "results"
        if res_dir.is_dir():
            records.extend(_read_records_from_source(res_dir))
    return records


def estimate_budget(
    jobs: Iterable[MatrixJob | Mapping[str, Any]],
    history: Any = None,
    *,
    parallel: int = DEFAULT_PARALLEL,
) -> BudgetDict:
    """Estimate rollout count, wall time and token usage per executor and in total.

    For each executor:
    - `rollouts`: pending jobs for that executor (`done` is False).
    - `median_wall_s`: median wall time in seconds from `history` for that executor, or
      `180.0` (3.0 minutes) when `history` has no wall-time samples for that executor.
    - `est_wall_s`: `rollouts * median_wall_s / parallel`; `est_minutes = est_wall_s / 60.0`.
    - `median_tokens`: median tokens per rollout from `history` (or `None` if absent);
      `est_tokens = round(rollouts * median_tokens)` (0 when `median_tokens` is `None`).
    """
    if isinstance(parallel, bool) or int(parallel) < 1:
        raise ValueError(f"parallel must be >= 1, got {parallel!r}")
    parallel = int(parallel)
    job_list = list(jobs)
    wall_by_exec, tok_by_exec = _parse_history_records(history)

    exec_order: list[str] = []
    counts: dict[str, int] = {}
    for j in job_list:
        ex = str(j["executor"] if isinstance(j, Mapping) else getattr(j, "executor"))
        done = bool(j.get("done", False) if isinstance(j, Mapping) else getattr(j, "done", False))
        if ex not in counts:
            exec_order.append(ex)
            counts[ex] = 0
        if not done:
            counts[ex] += 1

    per_exec_map: dict[str, BudgetDict] = {}
    by_exec_list: list[BudgetDict] = []
    for ex in exec_order:
        n_rollouts = counts[ex]
        w_samples = wall_by_exec.get(ex) or []
        t_samples = tok_by_exec.get(ex) or []
        from_history = bool(w_samples or t_samples)
        median_wall_s = (
            float(statistics.median(w_samples)) if w_samples else DEFAULT_ROLLOUT_WALL_S
        )
        median_wall_min = median_wall_s / 60.0
        est_wall_s = (n_rollouts * median_wall_s) / parallel
        est_minutes = est_wall_s / 60.0
        median_tokens = float(statistics.median(t_samples)) if t_samples else None
        est_tokens = int(round(n_rollouts * median_tokens)) if median_tokens is not None else 0
        entry = BudgetDict(
            executor=ex,
            rollouts=n_rollouts,
            est_wall_s=est_wall_s,
            est_minutes=est_minutes,
            est_tokens=est_tokens,
            median_wall_s=median_wall_s,
            median_wall_min=median_wall_min,
            median_tokens=median_tokens,
            from_history=from_history,
        )
        per_exec_map[ex] = entry
        by_exec_list.append(entry)

    total_rollouts = sum(e["rollouts"] for e in by_exec_list)
    total_wall_s = sum(e["est_wall_s"] for e in by_exec_list)
    total_minutes = sum(e["est_minutes"] for e in by_exec_list)
    total_tokens = sum(e["est_tokens"] for e in by_exec_list)
    total_entry = BudgetDict(
        rollouts=total_rollouts,
        est_wall_s=total_wall_s,
        est_minutes=total_minutes,
        est_tokens=total_tokens,
    )
    return BudgetDict(
        rollouts=total_rollouts,
        est_wall_s=total_wall_s,
        est_minutes=total_minutes,
        est_tokens=total_tokens,
        parallel=parallel,
        executors=BudgetDict(per_exec_map),
        by_executor=by_exec_list,
        total=total_entry,
    )


def format_budget(budget: Mapping[str, Any]) -> str:
    """Format `estimate_budget(...)` as a pre-run summary table."""
    parallel = int(budget.get("parallel") or DEFAULT_PARALLEL)
    rows = budget.get("by_executor") or list((budget.get("executors") or {}).values())
    lines = [
        f"Skillbench matrix budget estimate (parallel={parallel}):",
        f"{'Executor':<28} {'Rollouts':>8} {'Est. wall (min)':>16} {'Est. tokens':>14}  Source",
        f"{'-' * 28} {'-' * 8} {'-' * 16} {'-' * 14}  {'-' * 20}",
    ]
    for r in rows:
        ex = str(r.get("executor") or "")
        n_roll = int(r.get("rollouts") or 0)
        est_min = float(r.get("est_minutes") or 0.0)
        est_tok = int(r.get("est_tokens") or 0)
        tok_str = f"{est_tok:,}" if r.get("median_tokens") is not None or est_tok > 0 else "-"
        src = "history" if r.get("from_history") else f"default ({DEFAULT_ROLLOUT_MINUTES:.1f}m/rep)"
        lines.append(f"{ex:<28} {n_roll:>8} {est_min:>16.1f} {tok_str:>14}  {src}")
    lines.append(f"{'-' * 28} {'-' * 8} {'-' * 16} {'-' * 14}  {'-' * 20}")
    tot_roll = int(budget.get("rollouts") or 0)
    tot_min = float(budget.get("est_minutes") or 0.0)
    tot_tok = int(budget.get("est_tokens") or 0)
    tot_tok_str = f"{tot_tok:,}" if tot_tok > 0 else "0"
    lines.append(f"{'TOTAL':<28} {tot_roll:>8} {tot_min:>16.1f} {tot_tok_str:>14}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Invalid-rollout counting & matrix.json
# --------------------------------------------------------------------------------------------


def collect_invalid_counts(
    results_cell_dir: Path,
    adapt_ns_dir: Path | None = None,
) -> tuple[int, int, dict[str, int]]:
    """Return `(total_rollouts, invalid_total, invalid_by_reason)` for one cell."""
    seen_dirs: set[Path] = set()
    invalid_by_reason: dict[str, int] = {}
    total = 0
    invalid = 0

    search_roots = [p for p in (results_cell_dir, adapt_ns_dir) if p is not None and p.is_dir()]
    summary_files: list[Path] = []
    for root in search_roots:
        summary_files.extend(sorted(root.glob("*/summary.json")))

    for sp in summary_files:
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary, Mapping):
            continue
        for entry in summary.get("base") or []:
            if not isinstance(entry, Mapping):
                continue
            raw_dir = entry.get("run_dir")
            if not isinstance(raw_dir, str) or not raw_dir:
                continue
            rdir = Path(raw_dir)
            if not rdir.is_absolute():
                rdir = sp.parent.parent / rdir
            resolved = rdir.resolve()
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            tf = rdir / SKILL_TIMING_FILE
            if not tf.is_file():
                continue
            total += 1
            try:
                timing = json.loads(tf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid += 1
                invalid_by_reason["unreadable skill_timing.json"] = (
                    invalid_by_reason.get("unreadable skill_timing.json", 0) + 1
                )
                continue
            if isinstance(timing, Mapping) and not timing.get("valid", True):
                invalid += 1
                reason = str(timing.get("invalid_reason") or "invalid rollout")
                invalid_by_reason[reason] = invalid_by_reason.get(reason, 0) + 1

    if total == 0 and results_cell_dir.is_dir():
        for tf in sorted(results_cell_dir.rglob(SKILL_TIMING_FILE)):
            resolved = tf.parent.resolve()
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            total += 1
            try:
                timing = json.loads(tf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid += 1
                invalid_by_reason["unreadable skill_timing.json"] = (
                    invalid_by_reason.get("unreadable skill_timing.json", 0) + 1
                )
                continue
            if isinstance(timing, Mapping) and not timing.get("valid", True):
                invalid += 1
                reason = str(timing.get("invalid_reason") or "invalid rollout")
                invalid_by_reason[reason] = invalid_by_reason.get(reason, 0) + 1

    return total, invalid, invalid_by_reason


def write_matrix_json(
    runs_root: Path,
    *,
    suite: SuiteConfig,
    manifest_hash: str,
    k: int,
    arms_mode: str,
    arms: Sequence[str],
    executors: Sequence[str],
    judges: Mapping[str, str],
    budget: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> Path:
    """Write `<runs_root>/matrix.json` (and mirror to `<runs_root>/results/matrix.json`)."""
    invalid_by_reason: dict[str, int] = {}
    invalid_total = 0
    for cell in cells:
        for reason, count in (cell.get("invalid_by_reason") or {}).items():
            invalid_by_reason[str(reason)] = invalid_by_reason.get(str(reason), 0) + int(count)
        invalid_total += int(cell.get("invalid_total") or 0)

    cell_status = {f"{c['executor']}/{c['arm']}": c["status"] for c in cells}
    payload = {
        "suite": suite.name,
        "suite_hash": run_sweep._fresh_suite_hash(suite),
        "manifest_hash": manifest_hash,
        "k": int(k),
        "arms_mode": arms_mode,
        "arms": list(arms),
        "executors": list(executors),
        "judges": dict(judges),
        "budget": dict(budget),
        "cells": [dict(c) for c in cells],
        "cell_status": cell_status,
        "invalid_total": invalid_total,
        "invalid_by_reason": invalid_by_reason,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    out = runs_root / MATRIX_JSON
    text = json.dumps(payload, indent=2, default=str) + "\n"
    out.write_text(text, encoding="utf-8")
    res_dir = runs_root / "results"
    if res_dir.is_dir():
        (res_dir / MATRIX_JSON).write_text(text, encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------------
# CLI parser & execution driver
# --------------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skillbench-matrix",
        description=__doc__,
        formatter_class=C.HelpFormatter,
    )
    C.add_suite_arg(p)
    C.add_suites_arg(p)
    p.add_argument(
        "--executors",
        default="",
        help="comma list of executor model ids (default: [matrix].executors from suite.toml)",
    )
    p.add_argument("--k", type=int, default=DEFAULT_K, help="rollouts per (executor, arm, task)")
    p.add_argument(
        "--arms",
        choices=ARMS_MODES,
        default="both",
        help="arms to schedule: skill (with/without) | suite (baseline/suite-<name>) | "
        "both | baseline (control arms only)",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help="domains to run concurrently in each sweep and for wall-time budget estimation",
    )
    p.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help="hard cap on pending rollouts; abort before any rollout if exceeded",
    )
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="skip interactive TTY confirmation prompt",
    )
    p.add_argument("--seed", type=int, default=None, help="stream-order seed")
    p.add_argument("--split", choices=("all", "train", "test"), default="all")
    p.add_argument("--split-file", type=Path, default=None)
    p.add_argument(
        "--only-domains",
        default="",
        help="comma list restricting the sweep to these domains",
    )
    p.add_argument("--max-tasks", type=int, default=None, help="per-domain task cap")
    p.add_argument(
        "--pilot",
        action="store_true",
        help="restrict the stream to pilot tasks (pilot = true)",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="skillbench runs root (default: $HARNESS2_RUNS_DIR/skillbench)",
    )
    p.add_argument(
        "--history",
        type=Path,
        default=None,
        help="optional history.jsonl or results directory for budget estimation",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the budget table and exit without running any rollouts",
    )
    return p


def _sweep_args_namespace(args: argparse.Namespace, arm: str) -> argparse.Namespace:
    return argparse.Namespace(
        bench="skillbench",
        subset=None,
        substrate="agy",
        arm=arm,
        suite=args.suite,
        suites=args.suites,
        k=args.k,
        seed=args.seed,
        modes="parallel",
        split=args.split,
        split_file=args.split_file,
        workers=args.parallel,
        only_domains=args.only_domains,
        max_tasks=args.max_tasks,
        bases_only=True,
        pilot=args.pilot,
        dry_run=False,
    )


def _sweep_argv(args: argparse.Namespace, arm: str) -> list[str]:
    argv = [
        "--bench",
        "skillbench",
        "--substrate",
        "agy",
        "--arm",
        arm,
        "--k",
        str(args.k),
        "--workers",
        str(args.parallel),
        "--split",
        args.split,
    ]
    if args.suite is not None:
        argv += ["--suite", str(args.suite)]
    if args.suites:
        argv += ["--suites", str(args.suites)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.split_file is not None:
        argv += ["--split-file", str(args.split_file)]
    if args.only_domains:
        argv += ["--only-domains", str(args.only_domains)]
    if args.max_tasks is not None:
        argv += ["--max-tasks", str(args.max_tasks)]
    if args.pilot:
        argv.append("--pilot")
    return argv


def _judge_argv(args: argparse.Namespace, arm: str) -> list[str]:
    argv = [
        "--bench",
        "skillbench",
        "--substrate",
        "agy",
        "--arm",
        arm,
        "--k",
        str(args.k),
        "--bases",
    ]
    if args.suite is not None:
        argv += ["--suite", str(args.suite)]
    if args.suites:
        argv += ["--suites", str(args.suites)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    return argv


def _confirm_or_abort(yes: bool) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "refusing to run matrix without confirmation: stdin is not a TTY (pass --yes to confirm)"
        )
    print("Proceed with matrix run? [y/N]: ", end="", flush=True)
    answer = sys.stdin.readline().strip().lower()
    if answer not in ("y", "yes"):
        raise RuntimeError("matrix run aborted (confirmation declined)")


def _mark_cell_done(cell_dir: Path, jobs: Sequence[MatrixJob]) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    marker = cell_dir / CELL_DONE_JSON
    completed = [[j.task_id, j.rep] for j in jobs]
    marker.write_text(
        json.dumps({"completed": completed}, indent=2) + "\n",
        encoding="utf-8",
    )


def _main(args: argparse.Namespace) -> int:
    if args.k < 1 or args.parallel < 1:
        raise ValueError("--k and --parallel must be >= 1")
    if args.max_rollouts is not None and args.max_rollouts < 0:
        raise ValueError("--max-rollouts must be >= 0")

    main_suite, extra_suites = _resolve_suite_and_extras(args.suite, args.suites)
    exec_list = _resolve_executors(args.executors, main_suite)
    judges = {ex: resolve_judge(ex, main_suite) for ex in exec_list}
    base_runs = (
        Path(args.runs_root).expanduser().resolve()
        if args.runs_root is not None
        else _default_runs_root()
    )
    t_root = resolve_tasks_root(suite=main_suite)
    manifest_hash = check_manifest(t_root)

    # Check identity on any existing cell directories before expanding / resuming so a changed
    # suite_hash, manifest_hash, judge, or task stream is caught immediately.
    arm_labels = arms_for_mode(args.arms, main_suite, extra_suites)
    for ex in exec_list:
        judge_id = judges[ex]
        with executor_env(ex, judge_id, suite=main_suite, runs_root=base_runs, matrix=True):
            for arm in arm_labels:
                adapter = SkillbenchAdapter(
                    arm,
                    split_file=args.split_file,
                    pilot=True if args.pilot else None,
                    root=t_root,
                    runs_root=base_runs,
                    suite=main_suite,
                    suites=extra_suites,
                    executor=ex,
                )
                results_cell_dir, adapt_ns_dir = _cell_dirs(
                    base_runs, ex, arm, args.k, args.seed
                )
                sweep_ns = _sweep_args_namespace(args, arm)
                for candidate_dir in (adapt_ns_dir, results_cell_dir):
                    if (candidate_dir / "config.json").is_file():
                        run_sweep._check_identity(adapter, sweep_ns, candidate_dir, tag=arm)

    all_jobs = expand_jobs(
        main_suite,
        executors=exec_list,
        k=args.k,
        arms=args.arms,
        suites=extra_suites,
        split=args.split,
        split_file=args.split_file,
        seed=args.seed,
        only_domains=args.only_domains,
        max_tasks=args.max_tasks,
        pilot=args.pilot,
        runs_root=base_runs,
        tasks_root=t_root,
    )
    if not all_jobs:
        raise ValueError(f"no tasks matched --arms {args.arms!r} in corpus {t_root}")

    history_records = load_history(args.history, suite=main_suite, runs_root=base_runs)
    budget = estimate_budget(all_jobs, history_records, parallel=args.parallel)
    print(format_budget(budget), flush=True)

    if args.max_rollouts is not None and budget["rollouts"] > args.max_rollouts:
        raise RuntimeError(
            f"estimated rollouts ({budget['rollouts']}) exceeds --max-rollouts "
            f"({args.max_rollouts}); aborting before any rollout"
        )

    if args.dry_run:
        return 0

    if budget["rollouts"] > 0:
        _confirm_or_abort(args.yes)

    active_arms = tuple(dict.fromkeys(j.arm for j in all_jobs))
    cells: list[dict[str, Any]] = []
    any_failed = False
    any_partial = False

    for ex in exec_list:
        judge_id = judges[ex]
        for arm in active_arms:
            cell_jobs = [j for j in all_jobs if j.executor == ex and j.arm == arm]
            if not cell_jobs:
                continue
            results_cell_dir, adapt_ns_dir = _cell_dirs(base_runs, ex, arm, args.k, args.seed)
            pending_cell_jobs = [j for j in cell_jobs if not j.done]
            if not pending_cell_jobs:
                tot_seen, inv_tot, inv_reasons = collect_invalid_counts(
                    results_cell_dir, adapt_ns_dir
                )
                cells.append(
                    {
                        "executor": ex,
                        "judge": judge_id,
                        "arm": arm,
                        "status": "skipped",
                        "resumed": True,
                        "sweep_rc": 0,
                        "judge_rc": 0,
                        "rollouts": len(cell_jobs),
                        "pending_rollouts": 0,
                        "observed_rollouts": tot_seen,
                        "invalid_total": inv_tot,
                        "invalid_by_reason": inv_reasons,
                        "ns_dir": str(results_cell_dir),
                    }
                )
                continue

            with executor_env(ex, judge_id, suite=main_suite, runs_root=base_runs, matrix=True):
                adapter = SkillbenchAdapter(
                    arm,
                    split_file=args.split_file,
                    pilot=True if args.pilot else None,
                    root=t_root,
                    runs_root=base_runs,
                    suite=main_suite,
                    suites=extra_suites,
                    executor=ex,
                )
                sweep_ns = _sweep_args_namespace(args, arm)
                run_sweep._check_identity(adapter, sweep_ns, adapt_ns_dir, tag=arm)
                run_sweep.write_run_json(adapter, adapt_ns_dir)
                C.sync_results_cell(adapter, adapt_ns_dir)

                sweep_rc = int(run_sweep.main(_sweep_argv(args, arm)) or 0)
                if sweep_rc in (0, 2):
                    C.sync_results_cell(adapter, adapt_ns_dir)
                    judge_rc = int(judge.main(_judge_argv(args, arm)) or 0)
                else:
                    judge_rc = 1

                C.sync_results_cell(adapter, adapt_ns_dir)

            if sweep_rc == 0 and judge_rc == 0:
                _mark_cell_done(results_cell_dir, cell_jobs)
                status = "ok"
            elif sweep_rc in (0, 2) and judge_rc in (0, 2):
                status = "partial"
                any_partial = True
            else:
                status = "failed"
                any_failed = True

            tot_seen, inv_tot, inv_reasons = collect_invalid_counts(
                results_cell_dir, adapt_ns_dir
            )
            cells.append(
                {
                    "executor": ex,
                    "judge": judge_id,
                    "arm": arm,
                    "status": status,
                    "resumed": False,
                    "sweep_rc": sweep_rc,
                    "judge_rc": judge_rc,
                    "rollouts": len(cell_jobs),
                    "pending_rollouts": len(pending_cell_jobs),
                    "observed_rollouts": tot_seen,
                    "invalid_total": inv_tot,
                    "invalid_by_reason": inv_reasons,
                    "ns_dir": str(results_cell_dir),
                }
            )

    matrix_path = write_matrix_json(
        base_runs,
        suite=main_suite,
        manifest_hash=manifest_hash,
        k=args.k,
        arms_mode=args.arms,
        arms=active_arms,
        executors=exec_list,
        judges=judges,
        budget=budget,
        cells=cells,
    )
    print(f"wrote {matrix_path}", flush=True)
    if any_failed:
        return 1
    if any_partial:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `skillbench-matrix`."""
    args = _build_parser().parse_args(argv)
    try:
        return _main(args)
    except (config.ConfigError, RuntimeError, ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
