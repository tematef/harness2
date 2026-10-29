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

"""Targeted skillbench improver (`skillbench-improve`): Propose -> Probe -> Judge -> 3-Way Report.

Given an existing `skillbench` run directory (`results/with/` and `results/without/`, or
matrix cells `results/<executor>/with/` and `results/<executor>/without/`), inspects
weak or high-overhead skills, builds a contrastive evidence packet from the `WITH` and `WITHOUT`
trajectories and deliverables (without leaking `rubric.json` or judge verdicts unless
`--include-verdicts` is set), asks an isolated `agy` improver session for a revised `SKILL.md`,
validates the candidate skill (frontmatter, headless fallback, suite `banned_tools`, and
`--max-growth` size cap), probes it with the **same executor model(s)** as the baseline across
the skill's tasks (`k` reps), runs deterministic `verify` and the blinded judge, and emits a
3-way comparison (`WITHOUT` vs `Current WITH` vs `Candidate WITH`) in
`improve_report.{json,md,html}` plus a unified diff (`best.patch`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import html
import json
import os
import re
import shlex
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import config
from ..benchmarks.skillbench import (
    DEFAULT_JUDGE_TIMEOUT_S,
    JUDGMENTS_FILE,
    SKILL_TIMING_FILE,
    VERIFY_FILE,
    SkillbenchAdapter,
    SkillbenchTask,
    _default_runs_root,
    _env_pilot,
    _path_segment,
    build_prompt,
    deliverable_text,
    diff_stats,
    judge_model,
    load_tasks,
    rollout_valid,
    score,
    tasks_root,
)
from ..benchmarks.skillbench_suite import (
    DEFAULT_BANNED_TOOLS,
    SuiteConfig,
    SuiteError,
    active_suite,
    load_suite,
)
from ..substrate.agy import (
    _kill_group,
    agy_argv,
    agy_bin_missing_message,
    auth_files,
    exec_effort,
    exec_model,
    executor_env,
    model_guard,
    parse_event,
    read_events,
    render_trajectory,
    resolve_agy_bin,
    run_rollout,
    target_skill_file,
)
from . import _common as C
from .skillbench_report import (
    CEILING_FLAG_SHARE,
    HARDNESS_CEILING,
    HARDNESS_INFORMATIVE,
    HARDNESS_UNCLASSIFIED,
    MIN_BASELINE_REPS,
    NEEDS_HARDER_TASKS,
    build_report,
)
from .skillbench_summary import (
    REC_CONSIDER_REMOVING,
    REC_FIX,
    REC_NEEDS_HARDER_TASKS,
    _extract_tokens_delta_pct,
    _extract_wall_delta_pct,
    recommend,
)


def safe_tid(task_id: str) -> str:
    """Filesystem-safe single-segment identifier for `<skill>/<task-id>`."""
    return task_id.replace("/", "__")


def _rep_from_summary_json(base_dir: Path, task_id: str, rep_idx: int) -> Path | None:
    """Resolve `base<rep_idx>` from `<base_dir>/<task>/summary.json` if present."""
    for sub in (safe_tid(task_id), task_id):
        sp = base_dir / sub / "summary.json"
        if not sp.is_file():
            continue
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, Mapping):
            continue
        for entry in data.get("base") or []:
            if not isinstance(entry, Mapping):
                continue
            lbl = str(entry.get("label") or "")
            if lbl in (f"base{rep_idx}", f"b{rep_idx}"):
                raw_rd = entry.get("run_dir")
                if isinstance(raw_rd, str) and raw_rd:
                    p = Path(raw_rd)
                    cand = p if p.is_absolute() else (base_dir / p)
                    if cand.is_dir():
                        return cand
    return None


def resolve_rep_dir(results_dir: Path, task_id: str, rep_idx: int) -> Path:
    """Locate `<task_id>/base<rep_idx>` under `results_dir` (direct, safe_tid, flat, or namespaced)."""
    direct = results_dir / task_id / f"base{rep_idx}"
    if direct.is_dir():
        return direct
    safe_sub = results_dir / safe_tid(task_id) / f"base{rep_idx}"
    if safe_sub.is_dir():
        return safe_sub
    flat = results_dir / f"{safe_tid(task_id)}__b{rep_idx}"
    if flat.is_dir():
        return flat
    from_summary = _rep_from_summary_json(results_dir, task_id, rep_idx)
    if from_summary is not None:
        return from_summary
    for ns_dir in sorted(results_dir.glob("skillbench-*")):
        for cand in (
            ns_dir / task_id / f"base{rep_idx}",
            ns_dir / safe_tid(task_id) / f"base{rep_idx}",
            ns_dir / f"{safe_tid(task_id)}__b{rep_idx}",
        ):
            if cand.is_dir():
                return cand
        from_ns_summary = _rep_from_summary_json(ns_dir, task_id, rep_idx)
        if from_ns_summary is not None:
            return from_ns_summary
    return direct


IMPROVER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["diagnosis", "changes_summary", "skill_markdown"],
    "properties": {
        "diagnosis": {"type": "string", "minLength": 1},
        "changes_summary": {"type": "string", "minLength": 1},
        "skill_markdown": {"type": "string", "minLength": 1},
    },
}
IMPROVER_SCHEMA_JSON = json.dumps(IMPROVER_SCHEMA, separators=(",", ":"))

# Default of a suite's `banned_tools` (Claude Code tool names; AGY has its own).
BANNED_CLAUDE_TOOLS = DEFAULT_BANNED_TOOLS

EVIDENCE_TRAJ_CHARS = 1800
EVIDENCE_DELIV_CHARS = 1800
MAX_EVIDENCE_CHARS = 70_000
EFFICIENCY_GAIN_RATIO = 0.85
DEFAULT_MAX_GROWTH_PCT = 50.0


def improver_model(override: str | None = None) -> str:
    """Model used to propose candidate `SKILL.md` edits."""
    if override and override.strip():
        return override.strip()
    raw = config.env("SKILLBENCH_IMPROVER_MODEL")
    if raw and str(raw).strip():
        return str(raw).strip()
    return exec_model()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _str_or_none(val: Any) -> str | None:
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def _parse_models(
    models: Sequence[str] | str | None = None,
    executor: str | None = None,
) -> tuple[str, ...]:
    """Normalize `--models` / `models` / `executor` into a deduplicated tuple of model IDs."""
    out: list[str] = []
    if models is not None:
        if isinstance(models, str):
            items = [s.strip() for s in models.split(",") if s.strip()]
        else:
            items = []
            for m in models:
                for s in str(m).split(","):
                    if s.strip():
                        items.append(s.strip())
        out.extend(items)
    if executor and str(executor).strip():
        out.append(str(executor).strip())
    return tuple(dict.fromkeys(out))


def detect_cell_executor(
    results_dir: Path,
    tasks: Sequence[SkillbenchTask] = (),
    k: int = 3,
) -> str | None:
    """Detect the executor model recorded in a baseline cell (`run.json`, `config.json`, or
    rollout `skill_timing.json`). Returns `None` when the cell records no model."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return None
    for meta_name in ("run.json", "config.json"):
        meta = _read_json(results_dir / meta_name)
        ex = (
            _str_or_none(meta.get("executor"))
            or _str_or_none(meta.get("requested_exec_model"))
            or _str_or_none(meta.get("actual_exec_model"))
            or _str_or_none(meta.get("model"))
        )
        if ex:
            return ex
    for ns_dir in sorted(results_dir.glob("skillbench-*")):
        for meta_name in ("run.json", "config.json"):
            meta = _read_json(ns_dir / meta_name)
            ex = (
                _str_or_none(meta.get("executor"))
                or _str_or_none(meta.get("requested_exec_model"))
                or _str_or_none(meta.get("actual_exec_model"))
                or _str_or_none(meta.get("model"))
            )
            if ex:
                return ex
    for task in tasks:
        for i in range(1, max(1, k) + 1):
            rdir = resolve_rep_dir(results_dir, task.task_id, i)
            timing = _read_json(rdir / SKILL_TIMING_FILE)
            ex = (
                _str_or_none(timing.get("requested_model"))
                or _str_or_none(timing.get("actual_model"))
                or _str_or_none(timing.get("executor"))
                or _str_or_none(timing.get("model"))
            )
            if ex:
                return ex
    for tp in sorted(results_dir.rglob(SKILL_TIMING_FILE)):
        timing = _read_json(tp)
        ex = (
            _str_or_none(timing.get("requested_model"))
            or _str_or_none(timing.get("actual_model"))
            or _str_or_none(timing.get("executor"))
            or _str_or_none(timing.get("model"))
        )
        if ex:
            return ex
    if (
        results_dir.name in ("with", "without", "baseline")
        or results_dir.name.startswith("suite-")
    ) and results_dir.parent.parent.name in ("results", "adapt"):
        parent_name = results_dir.parent.name
        if parent_name not in ("results", "adapt", ""):
            return parent_name
    return None


def detect_cell_judge(
    results_dir: Path,
    tasks: Sequence[SkillbenchTask] = (),
    k: int = 3,
) -> str | None:
    """Detect the judge model recorded in a cell (`run.json`, `config.json`, or `judgments.json`)."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return None
    for meta_name in ("run.json", "config.json"):
        meta = _read_json(results_dir / meta_name)
        jm = _str_or_none(meta.get("judge")) or _str_or_none(meta.get("requested_judge_model"))
        if jm:
            return jm
    for ns_dir in sorted(results_dir.glob("skillbench-*")):
        for meta_name in ("run.json", "config.json"):
            meta = _read_json(ns_dir / meta_name)
            jm = _str_or_none(meta.get("judge")) or _str_or_none(meta.get("requested_judge_model"))
            if jm:
                return jm
    for task in tasks:
        for i in range(1, max(1, k) + 1):
            rdir = resolve_rep_dir(results_dir, task.task_id, i)
            judg = _read_json(rdir / JUDGMENTS_FILE)
            jm = _str_or_none(judg.get("judge_model")) or _str_or_none(judg.get("actual_model"))
            if jm:
                return jm
    return None


def _row_is_ceiling(row: Mapping[str, Any]) -> tuple[bool, int | None, int | None]:
    """Whether a skill row from `report.json` or `summarize_arm_runs` is flagged `needs harder tasks`."""
    rec = str(row.get("recommendation") or "").strip().lower()
    if rec == NEEDS_HARDER_TASKS:
        return True, row.get("ceiling_tasks"), row.get("total_tasks")
    if row.get("needs_harder_tasks") is True:
        return True, row.get("ceiling_tasks"), row.get("total_tasks")
    ceil_t = row.get("ceiling_tasks")
    tot_t = row.get("total_tasks", row.get("n_tasks"))
    if (
        isinstance(ceil_t, int)
        and not isinstance(ceil_t, bool)
        and isinstance(tot_t, int)
        and not isinstance(tot_t, bool)
        and tot_t > 0
        and (ceil_t / tot_t) > CEILING_FLAG_SHARE
    ):
        return True, ceil_t, tot_t
    share = row.get("ceiling_share")
    if isinstance(share, (int, float)) and not isinstance(share, bool):
        s_val = float(share) / 100.0 if float(share) > 1.0 else float(share)
        if s_val > CEILING_FLAG_SHARE:
            return True, ceil_t, tot_t
    return False, ceil_t, tot_t


def _report_skill_ceiling(
    report: Mapping[str, Any],
    skill: str,
    executor: str | None = None,
) -> tuple[bool, str]:
    """Check whether `report` flags `skill` as `needs harder tasks` for `executor`."""
    models = report.get("models")
    if isinstance(models, list) and models:
        for m_block in models:
            if not isinstance(m_block, Mapping):
                continue
            m_exec = _str_or_none(m_block.get("executor"))
            if executor is not None and m_exec is not None and m_exec != executor:
                continue
            by_skill = m_block.get("by_skill")
            row = by_skill.get(skill) if isinstance(by_skill, Mapping) else None
            if not isinstance(row, Mapping):
                raw_sk = m_block.get("skills")
                if isinstance(raw_sk, Mapping):
                    row = raw_sk.get(skill)
                elif isinstance(raw_sk, list):
                    row = next(
                        (
                            r
                            for r in raw_sk
                            if isinstance(r, Mapping) and r.get("skill") == skill
                        ),
                        None,
                    )
            if isinstance(row, Mapping):
                flagged, c_cnt, t_cnt = _row_is_ceiling(row)
                if flagged:
                    ex_str = m_exec or executor or "default"
                    counts = f" ({c_cnt}/{t_cnt} tasks at ceiling)" if c_cnt is not None and t_cnt is not None else ""
                    return (
                        True,
                        f"skill {skill!r} is flagged '{NEEDS_HARDER_TASKS}' for executor {ex_str!r}{counts}",
                    )
            if executor is not None and m_exec == executor:
                return False, ""

    summary_rows = report.get("summary")
    if isinstance(summary_rows, list):
        for s_row in summary_rows:
            if not isinstance(s_row, Mapping) or s_row.get("skill") != skill:
                continue
            by_ex = s_row.get("by_executor")
            if executor is not None and isinstance(by_ex, Mapping) and executor in by_ex:
                ex_row = by_ex[executor]
                if isinstance(ex_row, Mapping):
                    flagged, c_cnt, t_cnt = _row_is_ceiling(ex_row)
                    if flagged:
                        counts = (
                            f" ({c_cnt}/{t_cnt} tasks at ceiling)"
                            if c_cnt is not None and t_cnt is not None
                            else ""
                        )
                        return (
                            True,
                            f"skill {skill!r} is flagged '{NEEDS_HARDER_TASKS}' for executor {executor!r}{counts}",
                        )
                    return False, ""
            row_model = _str_or_none(s_row.get("model")) or _str_or_none(s_row.get("executor"))
            if executor is not None and row_model is not None and row_model != executor:
                continue
            flagged, c_cnt, t_cnt = _row_is_ceiling(s_row)
            if flagged:
                ex_str = row_model or executor or "default"
                counts = f" ({c_cnt}/{t_cnt} tasks at ceiling)" if c_cnt is not None and t_cnt is not None else ""
                return (
                    True,
                    f"skill {skill!r} is flagged '{NEEDS_HARDER_TASKS}' for executor {ex_str!r}{counts}",
                )

    raw_skills = report.get("by_skill") or report.get("skills") or {}
    row = None
    if isinstance(raw_skills, Mapping):
        row = raw_skills.get(skill)
    elif isinstance(raw_skills, list):
        row = next(
            (r for r in raw_skills if isinstance(r, Mapping) and r.get("skill") == skill),
            None,
        )
    if isinstance(row, Mapping):
        flagged, c_cnt, t_cnt = _row_is_ceiling(row)
        if flagged:
            ex_str = executor or _str_or_none(report.get("requested_exec_model")) or "default"
            counts = f" ({c_cnt}/{t_cnt} tasks at ceiling)" if c_cnt is not None and t_cnt is not None else ""
            return (
                True,
                f"skill {skill!r} is flagged '{NEEDS_HARDER_TASKS}' for executor {ex_str!r}{counts}",
            )
    return False, ""


def _row_has_cost_overhead(row: Mapping[str, Any]) -> bool:
    """Whether a summary or skill row shows token or wall-time cost overhead over control."""
    if row.get("cost_overhead") is True:
        return True
    tok_dp = _extract_tokens_delta_pct(row)
    if tok_dp is not None and tok_dp > 1e-9:
        return True
    wall_dp = _extract_wall_delta_pct(row)
    if wall_dp is not None and wall_dp > 1e-9:
        return True
    tok_w = row.get("tokens_with_median", row.get("tokens_median_with"))
    tok_wo = row.get("tokens_without_median", row.get("tokens_median_without"))
    if (
        isinstance(tok_w, (int, float))
        and not isinstance(tok_w, bool)
        and isinstance(tok_wo, (int, float))
        and not isinstance(tok_wo, bool)
        and tok_wo > 0
        and float(tok_w) > float(tok_wo) + 1e-9
    ):
        return True
    t_w = row.get("t_total_with_median", row.get("t_total_median_s_with"))
    t_wo = row.get("t_total_without_median", row.get("t_total_median_s_without"))
    if (
        isinstance(t_w, (int, float))
        and not isinstance(t_w, bool)
        and isinstance(t_wo, (int, float))
        and not isinstance(t_wo, bool)
        and t_wo > 0
        and float(t_w) > float(t_wo) + 1e-9
    ):
        return True
    return False


def _select_from_summary_row(row: Mapping[str, Any], *, all_weak: bool = False) -> bool:
    """Whether a v2 `summary[]` row is selected (`fix` + `consider removing` by default;
    `all_weak=True` also selects `inconclusive` with cost overhead)."""
    rec = str(row.get("recommendation") or "").strip().lower()
    if not rec:
        rec = recommend(row)[0].strip().lower()
    if rec == REC_NEEDS_HARDER_TASKS:
        return False
    if rec in (REC_FIX, REC_CONSIDER_REMOVING):
        return True
    if all_weak:
        verdict = str(row.get("verdict") or "").strip().lower()
        is_inconclusive = verdict == "inconclusive" or rec == "inconclusive" or (
            not verdict and rec not in ("helps", "hurts")
        )
        if is_inconclusive and _row_has_cost_overhead(row):
            return True
    return False


def select_weak_skills(
    report: Mapping[str, Any],
    *,
    executor: str | None = None,
    all_weak: bool = False,
    skill: str | Sequence[str] | None = None,
) -> list[str]:
    """Identify skills from a `skillbench` `report.json` that can benefit from improvement.

    - If `skill` is provided (`--skill`), forces selection of the given skill name(s).
    - When `report` contains v2 `summary[]`: selects skills recommended as `fix` or
      `consider removing` by default; `all_weak=True` (`--all-weak`) also adds `inconclusive`
      skills that have token or wall-time cost overhead; skills flagged `needs harder tasks`
      are skipped.
    - Falls back to the v1 per-skill heuristic when `summary[]` is absent.
    """
    if skill is not None:
        if isinstance(skill, str):
            forced = [s.strip() for s in skill.split(",") if s.strip()]
        else:
            forced = [
                part.strip()
                for item in skill
                for part in str(item).split(",")
                if part.strip()
            ]
        if forced:
            return list(dict.fromkeys(forced))

    summary_rows = report.get("summary")
    if isinstance(summary_rows, list):
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for s_row in summary_rows:
            if not isinstance(s_row, Mapping):
                continue
            if str(s_row.get("kind") or "").strip().lower() == "suite":
                continue
            sk_name = _str_or_none(s_row.get("skill"))
            if not sk_name:
                continue
            grouped.setdefault(sk_name, []).append(s_row)

        weak_v2: list[str] = []
        for sk_name, rows_for_sk in grouped.items():
            eff_row: Mapping[str, Any] | None = None
            if executor is not None:
                for cand_r in rows_for_sk:
                    by_ex = cand_r.get("by_executor")
                    if isinstance(by_ex, Mapping) and isinstance(by_ex.get(executor), Mapping):
                        eff_row = {**cand_r, **by_ex[executor]}
                        break
                if eff_row is None:
                    for cand_r in rows_for_sk:
                        r_mod = _str_or_none(cand_r.get("model")) or _str_or_none(
                            cand_r.get("executor")
                        )
                        if r_mod == executor:
                            eff_row = cand_r
                            break
                if eff_row is None and isinstance(report.get("models"), list):
                    for m_block in report["models"]:
                        if isinstance(m_block, Mapping) and m_block.get("executor") == executor:
                            m_by = m_block.get("by_skill")
                            if isinstance(m_by, Mapping) and isinstance(m_by.get(sk_name), Mapping):
                                eff_row = {**rows_for_sk[0], **m_by[sk_name]}
                                break
                            for m_r in m_block.get("skills") or []:
                                if isinstance(m_r, Mapping) and m_r.get("skill") == sk_name:
                                    eff_row = {**rows_for_sk[0], **m_r}
                                    break
                if eff_row is None:
                    has_explicit_models = any(
                        _str_or_none(r.get("model")) or _str_or_none(r.get("executor"))
                        for r in rows_for_sk
                    )
                    if not has_explicit_models or len(rows_for_sk) == 1:
                        eff_row = rows_for_sk[0]
            else:
                eff_row = rows_for_sk[0]

            if eff_row is None:
                continue

            flagged, reason = _report_skill_ceiling(report, sk_name, executor=executor)
            if not flagged:
                flagged, c_cnt, t_cnt = _row_is_ceiling(eff_row)
                if flagged:
                    ex_str = (
                        executor
                        or _str_or_none(eff_row.get("model"))
                        or _str_or_none(eff_row.get("executor"))
                        or "default"
                    )
                    counts = (
                        f" ({c_cnt}/{t_cnt} tasks at ceiling)"
                        if c_cnt is not None and t_cnt is not None
                        else ""
                    )
                    reason = (
                        f"skill {sk_name!r} is flagged '{NEEDS_HARDER_TASKS}' "
                        f"for executor {ex_str!r}{counts}"
                    )
            if flagged:
                print(f"skipping {sk_name}: {reason}", flush=True)
                continue

            if _select_from_summary_row(eff_row, all_weak=all_weak):
                weak_v2.append(sk_name)
        return weak_v2

    weak: list[str] = []
    raw_skills: Any = None
    if executor is not None and isinstance(report.get("models"), list):
        for m_block in report["models"]:
            if isinstance(m_block, Mapping) and m_block.get("executor") == executor:
                raw_skills = m_block.get("skills") or m_block.get("by_skill")
                break
    if raw_skills is None:
        raw_skills = report.get("skills") or report.get("by_skill") or {}

    if isinstance(raw_skills, list):
        items = [
            (str(s.get("skill") or ""), s)
            for s in raw_skills
            if isinstance(s, Mapping) and s.get("skill")
        ]
    elif isinstance(raw_skills, Mapping):
        items = [(str(k), v) for k, v in raw_skills.items() if isinstance(v, Mapping)]
    else:
        items = []

    for sk_name, s in sorted(items, key=lambda pair: pair[0]):
        flagged, reason = _report_skill_ceiling(report, sk_name, executor=executor)
        if not flagged:
            flagged, c_cnt, t_cnt = _row_is_ceiling(s)
            if flagged:
                ex_str = executor or _str_or_none(report.get("requested_exec_model")) or "default"
                counts = f" ({c_cnt}/{t_cnt} tasks at ceiling)" if c_cnt is not None and t_cnt is not None else ""
                reason = f"skill {sk_name!r} is flagged '{NEEDS_HARDER_TASKS}' for executor {ex_str!r}{counts}"
        if flagged:
            print(f"skipping {sk_name}: {reason}", flush=True)
            continue

        if s.get("recommendation"):
            if _select_from_summary_row(s, all_weak=all_weak):
                weak.append(sk_name)
            continue

        mean_w = s.get("with_mean", s.get("mean_with"))
        lift = s.get("lift")
        vw = s.get("verify_with_rate")
        if vw is None and isinstance(s.get("verify_with"), Mapping):
            vw = s["verify_with"].get("rate")
        tok_w = s.get("tokens_with_median", s.get("tokens_median_with"))
        tok_wo = s.get("tokens_without_median", s.get("tokens_median_without"))
        t_w = s.get("t_total_with_median", s.get("t_total_median_s_with"))
        t_wo = s.get("t_total_without_median", s.get("t_total_median_s_without"))

        is_weak = False
        if isinstance(mean_w, (int, float)) and mean_w < 1.0 - 1e-6:
            is_weak = True
        elif isinstance(lift, (int, float)) and lift <= 1e-6:
            is_weak = True
        elif isinstance(vw, (int, float)) and vw < 1.0 - 1e-6:
            is_weak = True
        elif (
            isinstance(tok_w, (int, float))
            and isinstance(tok_wo, (int, float))
            and tok_wo > 0
            and tok_w > 1.25 * tok_wo
        ):
            is_weak = True
        elif (
            isinstance(t_w, (int, float))
            and isinstance(t_wo, (int, float))
            and t_wo > 0
            and t_w > 1.25 * t_wo
        ):
            is_weak = True
        elif all_weak and _row_has_cost_overhead(s):
            is_weak = True
        if is_weak:
            weak.append(sk_name)
    return weak


def validate_candidate_skill(
    candidate_md: str,
    *,
    target_skill: str,
    original_md: str,
    banned_tools: Iterable[str] | None = None,
    max_growth: float | None = None,
    max_growth_pct: float | None = None,
    suite: SuiteConfig | None = None,
) -> list[str]:
    """Validate a proposed `SKILL.md` before staging it. Returns a list of errors (empty = valid).

    `banned_tools` defaults to the suite's `banned_tools` (`BANNED_CLAUDE_TOOLS` when no
    suite is configured). `max_growth` (in percent, default `DEFAULT_MAX_GROWTH_PCT` = 50%)
    rejects candidates whose character length grows beyond the threshold relative to
    `original_md`."""
    errors: list[str] = []
    if banned_tools is not None:
        banned = tuple(banned_tools)
    elif suite is not None:
        banned = tuple(suite.banned_tools)
    elif config.env("SKILLBENCH_SUITE"):
        try:
            banned = tuple(active_suite(quiet=True).banned_tools)
        except config.ConfigError:
            banned = tuple(BANNED_CLAUDE_TOOLS)
    else:
        banned = tuple(BANNED_CLAUDE_TOOLS)

    growth_cap: float | None
    if max_growth is not None:
        growth_cap = float(max_growth)
    elif max_growth_pct is not None:
        growth_cap = float(max_growth_pct)
    else:
        growth_cap = DEFAULT_MAX_GROWTH_PCT
    if growth_cap is not None and growth_cap < 0:
        raise ValueError(f"max_growth must be >= 0, got {growth_cap!r}")

    text = (candidate_md or "").strip()
    if not text:
        return ["candidate SKILL.md is empty"]
    if not text.startswith("---\n"):
        errors.append("missing YAML frontmatter opening '---'")
    else:
        end = text.find("\n---", 4)
        if end == -1:
            errors.append("missing YAML frontmatter closing '---'")
        else:
            fm = text[4:end]
            name_m = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
            desc_m = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
            if not name_m:
                errors.append("frontmatter missing 'name:' field")
            else:
                name_val = name_m.group(1).strip().strip("'\"")
                if name_val != target_skill:
                    errors.append(
                        f"frontmatter name {name_val!r} does not match target skill {target_skill!r}"
                    )
            if not desc_m or not desc_m.group(1).strip():
                errors.append("frontmatter missing non-empty 'description:' field")

    if "## When no one can answer" in original_md and "## When no one can answer" not in text:
        errors.append("removed required '## When no one can answer' headless fallback section")

    for name in banned:
        if re.search(rf"\b{re.escape(name)}\b", text):
            errors.append(f"contains banned tool name {name!r} (use AGY tool names)")

    orig_stripped = original_md.strip()
    if text == orig_stripped:
        errors.append("candidate SKILL.md is identical to the current SKILL.md")

    orig_len = len(orig_stripped)
    cand_len = len(text)
    if growth_cap is not None and orig_len > 0:
        growth_pct = ((cand_len - orig_len) / orig_len) * 100.0
        if growth_pct > growth_cap + 1e-9:
            errors.append(
                f"candidate SKILL.md grew by {growth_pct:.1f}% ({orig_len} -> {cand_len} chars), "
                f"exceeding --max-growth {growth_cap:g}%"
            )

    return errors


def _format_rep_evidence(
    run_dir: Path,
    label: str,
    *,
    include_verdicts: bool = False,
) -> str:
    if not run_dir.is_dir():
        return f"#### {label}: (missing rollout directory)\n"
    timing = _read_json(run_dir / SKILL_TIMING_FILE)
    verify = _read_json(run_dir / VERIFY_FILE)
    dstats = diff_stats(run_dir) or {"files": 0, "added": 0, "removed": 0}
    events_p = run_dir / "events.jsonl"
    traj = (
        render_trajectory(
            read_events(events_p),
            obs_trunc=600,
            max_chars=EVIDENCE_TRAJ_CHARS,
        )
        if events_p.is_file()
        else "(no trajectory)"
    )
    deliv = deliverable_text(run_dir)
    if len(deliv) > EVIDENCE_DELIV_CHARS:
        deliv = deliv[:EVIDENCE_DELIV_CHARS] + "\n...[truncated]..."

    parts = [
        f"#### {label}",
        (
            f"- valid={timing.get('valid')} | invoked={timing.get('invoked')} "
            f"| t_skill_s={timing.get('t_skill_s')} | t_total_s={timing.get('t_total_s')} "
            f"| tokens={timing.get('total_tokens')} "
            f"| diff: {dstats['files']} file(s), +{dstats['added']}/-{dstats['removed']}"
        ),
    ]
    if verify:
        v_rc = verify.get("exit_code") if "exit_code" in verify else verify.get("returncode")
        parts.append(
            f"- deterministic verify: passed={verify.get('passed')} "
            f"(rc={v_rc}, command={verify.get('command')!r})"
        )
        v_err = str(
            verify.get("output_tail") or verify.get("stderr") or verify.get("stdout") or ""
        ).strip()
        if v_err and not verify.get("passed"):
            parts.append(f"  verify output: `{v_err[:400]}`")

    if include_verdicts:
        judg = _read_json(run_dir / JUDGMENTS_FILE)
        j_list = judg.get("judgments")
        if isinstance(j_list, list) and j_list and isinstance(j_list[0], dict):
            first = j_list[0]
            parts.append(
                f"- judge score: {first.get('n_passed')}/{first.get('n_criteria')}"
            )
            for c in first.get("criteria") or []:
                if isinstance(c, dict):
                    passed = c.get("pass") if "pass" in c else c.get("passed")
                    if not passed:
                        parts.append(
                            f"  - FAILED criterion `{c.get('id')}`: {c.get('rationale') or c.get('reason')}"
                        )

    parts.append("<trajectory>")
    parts.append(traj)
    parts.append("</trajectory>")
    parts.append("<deliverable>")
    parts.append(deliv)
    parts.append("</deliverable>")
    return "\n".join(parts) + "\n"


def _locate_arm_dir_for_evidence(
    runs_dir: Path,
    arm: str,
    executor: str | None = None,
) -> Path:
    """Find the directory holding `arm` rollouts under `runs_dir / 'results'`."""
    results_root = runs_dir / "results"
    if executor:
        cand = results_root / _path_segment(executor) / arm
        if cand.is_dir():
            return cand
        cand_raw = results_root / executor / arm
        if cand_raw.is_dir():
            return cand_raw
    direct = results_root / arm
    if direct.is_dir():
        return direct
    if results_root.is_dir():
        for sub in sorted(p for p in results_root.iterdir() if p.is_dir()):
            cand = sub / arm
            if cand.is_dir():
                return cand
    return direct


def _format_prior_round_block(pr: Mapping[str, Any]) -> list[str]:
    """Render one prior round's candidate, 3-way numbers, and gate reasons for the evidence packet."""
    r_num = pr.get("round")
    accepted = bool(pr.get("accepted"))
    status_str = "ACCEPTED" if accepted else "REJECTED"
    reason = pr.get("decision_reason") or pr.get("reason") or ""
    lines = [
        f"### Round {r_num} ({status_str})",
        (
            f"- Round {r_num}: accepted={pr.get('accepted')} | "
            f"reason={reason} | "
            f"mean_score={pr.get('mean_candidate')} | "
            f"tokens_median={pr.get('tokens_median')} | "
            f"wall_median_s={pr.get('t_total_median_s')}"
        ),
    ]
    if pr.get("diagnosis"):
        lines.append(f"- diagnosis: {pr.get('diagnosis')}")
    if pr.get("changes_summary"):
        lines.append(f"- summary: {pr.get('changes_summary')}")

    wo = pr.get("without") if isinstance(pr.get("without"), Mapping) else {}
    cur = pr.get("current_with") if isinstance(pr.get("current_with"), Mapping) else {}
    cand = pr.get("candidate") if isinstance(pr.get("candidate"), Mapping) else {}

    wo_mean = wo.get("mean", pr.get("mean_without"))
    cur_mean = cur.get("mean", pr.get("mean_current_with"))
    cand_mean = cand.get("mean", pr.get("mean_candidate"))

    wo_ver = (wo.get("verify") or {}).get("rate") if isinstance(wo.get("verify"), Mapping) else pr.get("verify_without")
    cur_ver = (cur.get("verify") or {}).get("rate") if isinstance(cur.get("verify"), Mapping) else pr.get("verify_current_with")
    cand_ver = (cand.get("verify") or {}).get("rate") if isinstance(cand.get("verify"), Mapping) else pr.get("verify_candidate")

    wo_tok = wo.get("tokens_median", pr.get("tokens_without"))
    cur_tok = cur.get("tokens_median", pr.get("tokens_current_with"))
    cand_tok = cand.get("tokens_median", pr.get("tokens_median"))

    wo_wall = wo.get("t_total_median_s", pr.get("wall_without_s"))
    cur_wall = cur.get("t_total_median_s", pr.get("wall_current_with_s"))
    cand_wall = cand.get("t_total_median_s", pr.get("t_total_median_s"))

    lines.append(
        "- 3-way numbers (WITHOUT -> Current WITH -> Candidate WITH): "
        f"mean={_fmt_num(wo_mean)} -> {_fmt_num(cur_mean)} -> {_fmt_num(cand_mean)} | "
        f"verify={_fmt_num(wo_ver, 'pct')} -> {_fmt_num(cur_ver, 'pct')} -> {_fmt_num(cand_ver, 'pct')} | "
        f"tokens={_fmt_num(wo_tok, 'int')} -> {_fmt_num(cur_tok, 'int')} -> {_fmt_num(cand_tok, 'int')} | "
        f"wall={_fmt_num(wo_wall, '.1f', 's')} -> {_fmt_num(cur_wall, '.1f', 's')} -> {_fmt_num(cand_wall, '.1f', 's')}"
    )

    model_gates = pr.get("model_gates")
    if isinstance(model_gates, list) and model_gates:
        lines.append("- Per-model gate results:")
        for mg in model_gates:
            if not isinstance(mg, Mapping):
                continue
            m_ex = mg.get("executor")
            m_ok = "ACCEPTED" if mg.get("accepted") else "REJECTED"
            m_rsn = mg.get("decision_reason") or ""
            m_wo = mg.get("without") if isinstance(mg.get("without"), Mapping) else {}
            m_cur = mg.get("current_with") if isinstance(mg.get("current_with"), Mapping) else {}
            m_cand = mg.get("candidate") if isinstance(mg.get("candidate"), Mapping) else {}
            lines.append(
                f"  - `{m_ex}`: {m_ok} ({m_rsn}) | "
                f"3-way mean={_fmt_num(m_wo.get('mean'))} -> {_fmt_num(m_cur.get('mean'))} -> {_fmt_num(m_cand.get('mean'))} | "
                f"verify={_fmt_num((m_wo.get('verify') or {}).get('rate'), 'pct')} -> "
                f"{_fmt_num((m_cur.get('verify') or {}).get('rate'), 'pct')} -> "
                f"{_fmt_num((m_cand.get('verify') or {}).get('rate'), 'pct')} | "
                f"tokens={_fmt_num(m_wo.get('tokens_median'), 'int')} -> "
                f"{_fmt_num(m_cur.get('tokens_median'), 'int')} -> "
                f"{_fmt_num(m_cand.get('tokens_median'), 'int')} | "
                f"wall={_fmt_num(m_wo.get('t_total_median_s'), '.1f', 's')} -> "
                f"{_fmt_num(m_cur.get('t_total_median_s'), '.1f', 's')} -> "
                f"{_fmt_num(m_cand.get('t_total_median_s'), '.1f', 's')}"
            )

    cand_md = (
        pr.get("skill_markdown")
        or pr.get("candidate_markdown")
        or pr.get("candidate_skill_md")
        or ""
    )
    if isinstance(cand_md, str) and cand_md.strip():
        lines.append(f"- Round {r_num} Candidate `SKILL.md`:")
        lines.append("```markdown")
        lines.append(cand_md.strip())
        lines.append("```")
    elif isinstance(pr.get("diff"), str) and pr["diff"].strip():
        lines.append(f"- Round {r_num} Candidate Diff:")
        lines.append("```diff")
        lines.append(pr["diff"].strip())
        lines.append("```")
    return lines


def build_evidence_packet(
    *,
    skill: str,
    current_skill_md: str,
    tasks: Sequence[SkillbenchTask],
    runs_dir: Path,
    k: int,
    include_verdicts: bool = False,
    prior_rounds: Sequence[Mapping[str, Any]] = (),
    executor: str | None = None,
) -> str:
    """Assemble the contrastive evidence packet for `skill` across its tasks."""
    lines = [
        f"# Skill Improvement Evidence Packet: `{skill}`",
        "",
        "## Current `SKILL.md`",
        "```markdown",
        current_skill_md.strip(),
        "```",
        "",
        "## Contrastive Behavioral Evidence (`WITH` vs `WITHOUT`)",
        (
            "Compare how the agent behaved WITH the skill vs WITHOUT the skill on each task. "
            "Look for: (1) missing steps, edge cases, or verification failures in WITH; "
            "(2) unnecessary turns, redundant searches, or token/latency bloat in WITH; "
            "(3) unclear or overly verbose instructions in `SKILL.md`."
        ),
        "",
    ]

    for task in tasks:
        lines.append(f"### Task `{task.task_id}`")
        lines.append("**Prompt:**")
        lines.append("```text")
        lines.append(task.prompt.strip())
        lines.append("```")
        lines.append("")
        for arm in ("with", "without"):
            arm_root = _locate_arm_dir_for_evidence(runs_dir, arm, executor=executor)
            for i in range(1, k + 1):
                rdir = resolve_rep_dir(arm_root, task.task_id, i)
                lines.append(
                    _format_rep_evidence(
                        rdir,
                        f"arm={arm.upper()} rep=b{i}",
                        include_verdicts=include_verdicts,
                    )
                )

    prior_lines: list[str] = []
    if prior_rounds:
        prior_lines.append("## Prior Improvement Rounds in This Session")
        for pr in prior_rounds:
            prior_lines.extend(_format_prior_round_block(pr))
        prior_lines.append("")

    body = "\n".join(lines)
    tail = ("\n" + "\n".join(prior_lines)) if prior_lines else ""
    if len(body) + len(tail) > MAX_EVIDENCE_CHARS:
        budget = max(1000, MAX_EVIDENCE_CHARS - len(tail))
        body = body[:budget] + "\n\n...[evidence packet truncated to fit CLI limit]...\n"
    return body + tail


def build_improver_prompt(
    *,
    skill: str,
    evidence_packet: str,
    validation_error: str = "",
    suite_name: str = "",
    banned_tools: Iterable[str] = BANNED_CLAUDE_TOOLS,
    max_growth: float | None = DEFAULT_MAX_GROWTH_PCT,
) -> str:
    """Build the single-turn prompt sent to the isolated `agy` improver. `suite_name` and
    `banned_tools` come from the suite config; nothing suite-specific is hard-coded here."""
    retry_block = (
        f"\n## Previous Candidate Was Rejected\nFix these validation errors:\n{validation_error}\n"
        if validation_error
        else ""
    )
    suite_phrase = f"the `{suite_name}` plugin suite" if suite_name else "an AGY plugin suite"
    banned_list = ", ".join(f"`{b}`" for b in banned_tools) or "(none)"
    growth_rule = (
        f" Keep the candidate concise (at most +{max_growth:g}% character growth over the original)."
        if max_growth is not None
        else ""
    )
    return (
        f"You are an expert Agent Harness Engineer optimizing a `SKILL.md` file in {suite_phrase} "
        "for Google Antigravity (AGY) / Jetski.\n\n"
        "Do not call any tools or inspect any files; analyze the Evidence Packet below and return "
        "your structured JSON output immediately via `finish`.\n\n"
        "## Optimization Rules\n"
        f"1. Keep YAML frontmatter valid with `name: {skill}` and a clear `description:`.\n"
        "2. If the current `SKILL.md` has a `## When no one can answer` section, preserve that "
        "section heading and its headless fallback behavior.\n"
        "3. Use only AGY/Jetski tool names (`view_file`, `write_to_file`, `replace_file_content`, "
        "`find_by_name`, `grep_search`, `run_command`, `invoke_subagent`, `ask_question`). Never "
        f"use these banned tool names: {banned_list}.\n"
        "4. Fix any quality gaps where `WITH` missed root causes, edge cases, or verification checks.\n"
        "5. Streamline instructions that caused unnecessary tool calls, redundant searches for "
        f"other skills in headless/leave-one-out runs, or high token/time overhead.{growth_rule}\n"
        "6. Return the complete, ready-to-save `SKILL.md` content in `skill_markdown`.\n"
        f"{retry_block}\n"
        "## Evidence Packet\n\n"
        f"{evidence_packet}\n"
    )


def _stage_isolated_auth(home: Path) -> list[Path]:
    real = Path.home()
    copied: list[Path] = []
    for rel in auth_files():
        src = real / rel
        if not src.is_file():
            continue
        dst = home / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst, follow_symlinks=True)
        dst.chmod(0o600)
        copied.append(dst)
    return copied


def run_improver_once(
    prompt: str,
    *,
    model: str,
    timeout: int = DEFAULT_JUDGE_TIMEOUT_S,
) -> dict[str, Any] | None:
    """Run one isolated `agy` improver call and return `{diagnosis, changes_summary, skill_markdown}`."""
    binary = resolve_agy_bin()
    if binary is None:
        raise RuntimeError(agy_bin_missing_message())
    tmp = Path(tempfile.mkdtemp(prefix="skillbench-improver-"))
    home = tmp / "home"
    work = tmp / "work"
    for d in (
        home / ".config",
        home / ".local" / "share",
        home / ".local" / "state",
        home / ".cache",
        work,
    ):
        d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        HOME=str(home),
        EDITOR="true",
        VISUAL="true",
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        XDG_STATE_HOME=str(home / ".local" / "state"),
        XDG_CACHE_HOME=str(home / ".cache"),
    )
    argv = agy_argv(
        binary,
        prompt,
        model=model,
        timeout=timeout,
        effort=exec_effort(),
        json_schema=IMPROVER_SCHEMA_JSON,
    )
    t0 = time.monotonic()
    try:
        _stage_isolated_auth(home)
        try:
            proc = subprocess.Popen(
                argv,
                cwd=work,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError:
            return None
        try:
            out_bytes, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return None
        _kill_group(proc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    wall_s = round(time.monotonic() - t0, 2)
    if proc.returncode != 0:
        return None
    text = out_bytes.decode("utf-8", errors="replace")
    events = [ev for ev in map(parse_event, text.splitlines()) if ev is not None]
    actual, mismatch = model_guard(events, model)
    if mismatch:
        return None
    outputs = [ev.structured_output for ev in events if ev.structured_output is not None]
    if not outputs or not isinstance(outputs[-1], dict):
        return None
    payload = outputs[-1]
    for key in ("diagnosis", "changes_summary", "skill_markdown"):
        val = payload.get(key)
        if not isinstance(val, str) or not val.strip():
            return None
    tokens = [ev.total_tokens for ev in events if ev.total_tokens is not None]
    return {
        "diagnosis": payload["diagnosis"].strip(),
        "changes_summary": payload["changes_summary"].strip(),
        "skill_markdown": payload["skill_markdown"].strip() + "\n",
        "model": model,
        "actual_model": actual,
        "wall_s": wall_s,
        "total_tokens": tokens[-1] if tokens else None,
    }


def propose_skill(
    *,
    skill: str,
    original_md: str,
    evidence_packet: str,
    model: str,
    timeout: int = DEFAULT_JUDGE_TIMEOUT_S,
    max_attempts: int = 2,
    suite: SuiteConfig | None = None,
    max_growth: float | None = None,
) -> dict[str, Any]:
    """Propose and validate a candidate `SKILL.md` (with up to `max_attempts` retries). The
    prompt names `suite` and its `banned_tools` (the active suite when None)."""
    suite = suite if suite is not None else active_suite()
    eff_growth = max_growth if max_growth is not None else DEFAULT_MAX_GROWTH_PCT
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        prompt = build_improver_prompt(
            skill=skill,
            evidence_packet=evidence_packet,
            validation_error=last_err,
            suite_name=suite.name,
            banned_tools=suite.banned_tools,
            max_growth=eff_growth,
        )
        res = run_improver_once(prompt, model=model, timeout=timeout)
        if res is None:
            last_err = f"attempt {attempt}: agy improver returned no valid structured output"
            continue
        errs = validate_candidate_skill(
            res["skill_markdown"],
            target_skill=skill,
            original_md=original_md,
            banned_tools=suite.banned_tools,
            max_growth=eff_growth,
        )
        if errs:
            last_err = "; ".join(errs)
            continue
        res["attempts"] = attempt
        return res
    raise RuntimeError(f"failed to propose a valid SKILL.md for {skill!r}: {last_err}")


def unified_skill_diff(original_md: str, candidate_md: str, *, skill: str) -> str:
    """Unified diff between original and candidate `SKILL.md`."""
    cand = candidate_md if not candidate_md or candidate_md.endswith("\n") else candidate_md + "\n"
    return "".join(
        difflib.unified_diff(
            original_md.splitlines(keepends=True),
            cand.splitlines(keepends=True),
            fromfile=f"a/skills/{skill}/SKILL.md",
            tofile=f"b/skills/{skill}/SKILL.md",
        )
    )


def patch_diff_stat(diff_text: str) -> dict[str, int]:
    """Count `{files, added, removed}` from a unified diff string (`best.patch`)."""
    files = 0
    added = 0
    removed = 0
    for line in (diff_text or "").splitlines():
        if line.startswith("+++ "):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    if files == 0 and (added > 0 or removed > 0):
        files = 1
    return {"files": files, "added": added, "removed": removed}


def format_diff_stat(diff_text: str) -> str:
    """Format a human-readable diff stat (e.g. `1 file changed, +8/-4 lines`)."""
    st_map = patch_diff_stat(diff_text)
    f_cnt = st_map["files"]
    f_word = "file" if f_cnt == 1 else "files"
    return f"{f_cnt} {f_word} changed, +{st_map['added']}/-{st_map['removed']} lines"


def _judged_scores_map(results_dir: Path) -> dict[tuple[str, str], dict[str, int]]:
    """Fallback `(task_id, label) -> {n_passed, n_criteria}` from `base_judged.json`."""
    jp = Path(results_dir) / "base_judged.json"
    if not jp.is_file():
        return {}
    try:
        rows = json.loads(jp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(rows, list):
        return {}
    out: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        tid = str(row.get("task") or row.get("task_id") or "")
        lbl = str(row.get("label") or "")
        sc = row.get("score")
        if tid and lbl and isinstance(sc, Mapping):
            np = sc.get("n_passed")
            nc = sc.get("n_criteria")
            if isinstance(np, int) and isinstance(nc, int) and nc > 0:
                out[(tid, lbl)] = {"n_passed": np, "n_criteria": nc}
    return out


def summarize_arm_runs(
    tasks: Sequence[SkillbenchTask],
    results_dir: Path,
    *,
    k: int,
    executor: str | None = None,
) -> dict[str, Any]:
    """Compute aggregate metrics over `<results_dir>/<safe_tid>__b1..bk` for `tasks`."""
    results_dir = Path(results_dir)
    detected_exec = detect_cell_executor(results_dir, tasks, k=k)
    fallback_scores = _judged_scores_map(results_dir)

    task_means: dict[str, float] = {}
    all_fracs: list[float] = []
    invoked_count = 0
    valid_count = 0
    total_reps = 0
    t_skills: list[float] = []
    tokens_list: list[int] = []
    wall_list: list[float] = []
    diff_lines_list: list[int] = []
    verify_passed = 0
    verify_total = 0
    criteria_totals: dict[str, list[bool]] = {}
    invalid_reasons: list[str] = []
    per_task: dict[str, Any] = {}
    task_hardness_labels: list[str] = []
    eff_min_reps = min(MIN_BASELINE_REPS, max(1, k))

    for task in tasks:
        t_fracs: list[float] = []
        t_missed: list[int] = []
        t_reps: list[dict[str, Any]] = []
        for i in range(1, k + 1):
            total_reps += 1
            rdir = resolve_rep_dir(results_dir, task.task_id, i)
            timing = _read_json(rdir / SKILL_TIMING_FILE)
            judg = _read_json(rdir / JUDGMENTS_FILE)
            verify = _read_json(rdir / VERIFY_FILE)
            dstats = (diff_stats(rdir) if rdir.is_dir() else None) or {
                "files": 0,
                "added": 0,
                "removed": 0,
            }
            valid = bool(timing.get("valid"))
            if not valid and timing.get("invalid_reason"):
                invalid_reasons.append(str(timing["invalid_reason"]))
            frac: float | None = None
            n_missed: int | None = None
            if valid:
                valid_count += 1
                if timing.get("invoked"):
                    invoked_count += 1
                    if isinstance(timing.get("t_skill_s"), (int, float)):
                        t_skills.append(float(timing["t_skill_s"]))
                if isinstance(timing.get("total_tokens"), int):
                    tokens_list.append(int(timing["total_tokens"]))
                if isinstance(timing.get("t_total_s"), (int, float)):
                    wall_list.append(float(timing["t_total_s"]))
                diff_lines_list.append(dstats["added"] + dstats["removed"])
                if isinstance(verify.get("passed"), bool):
                    verify_total += 1
                    if verify["passed"]:
                        verify_passed += 1
                j_list = judg.get("judgments")
                first: Mapping[str, Any] | None = None
                if isinstance(j_list, list) and j_list and isinstance(j_list[0], dict):
                    first = j_list[0]
                elif (task.task_id, f"base{i}") in fallback_scores:
                    first = fallback_scores[(task.task_id, f"base{i}")]
                elif (task.task_id, f"b{i}") in fallback_scores:
                    first = fallback_scores[(task.task_id, f"b{i}")]
                if first is not None:
                    np = first.get("n_passed")
                    nc = first.get("n_criteria")
                    if isinstance(np, int) and isinstance(nc, int) and nc > 0:
                        frac = np / nc
                        n_missed = nc - np
                        t_fracs.append(frac)
                        t_missed.append(n_missed)
                        all_fracs.append(frac)
                    for c in first.get("criteria") or []:
                        if isinstance(c, dict) and isinstance(c.get("id"), str):
                            cid = f"{task.task_id}:{c['id']}"
                            c_pass = c.get("pass") if "pass" in c else c.get("passed")
                            criteria_totals.setdefault(cid, []).append(bool(c_pass))
            t_reps.append(
                {
                    "label": f"b{i}",
                    "valid": valid,
                    "invoked": bool(timing.get("invoked")),
                    "t_skill_s": timing.get("t_skill_s"),
                    "t_total_s": timing.get("t_total_s"),
                    "total_tokens": timing.get("total_tokens"),
                    "frac": frac,
                    "n_missed": n_missed,
                    "verify_passed": verify.get("passed") if verify else None,
                    "diff_lines": dstats["added"] + dstats["removed"],
                }
            )
        t_mean = st.fmean(t_fracs) if t_fracs else None
        if t_mean is not None:
            task_means[task.task_id] = t_mean
        min_fail = max(1, int(getattr(task, "min_baseline_fail", 1) or 1))
        if len(t_missed) < eff_min_reps:
            thard = HARDNESS_UNCLASSIFIED
        elif any(m >= min_fail for m in t_missed):
            thard = HARDNESS_INFORMATIVE
        else:
            thard = HARDNESS_CEILING
        task_hardness_labels.append(thard)
        per_task[task.task_id] = {
            "mean": t_mean,
            "n_valid": sum(1 for r in t_reps if r["valid"]),
            "hardness": thard,
            "baseline_missed": t_missed,
            "reps": t_reps,
        }

    mean_score = st.fmean(task_means.values()) if task_means else None
    total_tasks = len(task_hardness_labels)
    ceiling_tasks = sum(1 for lbl in task_hardness_labels if lbl == HARDNESS_CEILING)
    informative_tasks = sum(1 for lbl in task_hardness_labels if lbl == HARDNESS_INFORMATIVE)
    unclassified_tasks = sum(1 for lbl in task_hardness_labels if lbl == HARDNESS_UNCLASSIFIED)
    ceiling_share = (ceiling_tasks / total_tasks) if total_tasks > 0 else None
    needs_harder = ceiling_share is not None and ceiling_share > CEILING_FLAG_SHARE

    return {
        "executor": detected_exec or executor,
        "recorded_executor": detected_exec,
        "mean": mean_score,
        "n_valid": valid_count,
        "n_total": total_reps,
        "invocation_rate": (invoked_count / valid_count) if valid_count > 0 else None,
        "t_skill_p50_s": st.median(t_skills) if t_skills else None,
        "tokens_median": st.median(tokens_list) if tokens_list else None,
        "t_total_median_s": st.median(wall_list) if wall_list else None,
        "diff_lines_median": st.median(diff_lines_list) if diff_lines_list else None,
        "verify": {
            "passed": verify_passed,
            "total": verify_total,
            "rate": (verify_passed / verify_total) if verify_total > 0 else None,
        },
        "ceiling_tasks": ceiling_tasks,
        "informative_tasks": informative_tasks,
        "unclassified_tasks": unclassified_tasks,
        "total_tasks": total_tasks,
        "ceiling_share": ceiling_share,
        "needs_harder_tasks": needs_harder,
        "criteria_pass_rates": {
            cid: sum(1 for v in vals if v) / len(vals)
            for cid, vals in sorted(criteria_totals.items())
            if vals
        },
        "invalid_reasons": sorted(set(invalid_reasons)),
        "tasks": per_task,
    }


def probe_candidate(
    *,
    skill: str,
    candidate_skill_path: Path,
    tasks: Sequence[SkillbenchTask],
    out_results_dir: Path,
    k: int,
    root: Path | None = None,
    executor: str | None = None,
    judge: str | None = None,
    baseline_executor: str | None = None,
    allow_model_mismatch: bool = False,
) -> dict[str, Any]:
    """Stage `candidate_skill_path`, run all `tasks` for `k` reps, and score each rollout."""
    eff_exec = executor.strip() if executor and executor.strip() else (baseline_executor or exec_model())
    if (
        baseline_executor
        and eff_exec != baseline_executor
        and not allow_model_mismatch
    ):
        raise ValueError(
            f"model mismatch: probe executor {eff_exec!r} does not match baseline executor "
            f"{baseline_executor!r} (pass --allow-model-mismatch to override)"
        )
    with executor_env(eff_exec, judge, matrix=False):
        adapter = SkillbenchAdapter(arm="with", root=root, executor=eff_exec)
        install = adapter.substrate.materialise(
            adapter.base_harness(),
            adapter.spec,
            {
                "arm": "with",
                "target_skill": skill,
                "skill_override": str(candidate_skill_path),
            },
        )
        out_results_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[tuple[SkillbenchTask, str, Path]] = []
        for task in tasks:
            prompt = build_prompt(task)
            for i in range(1, k + 1):
                rdir = out_results_dir / task.task_id / f"base{i}"
                jobs.append((task, prompt, rdir))

        def _run_one(job: tuple[SkillbenchTask, str, Path]) -> None:
            t, p, rd = job
            if not (rd / SKILL_TIMING_FILE).is_file() or not rollout_valid(rd):
                run_rollout(
                    install,
                    rd,
                    prompt=p,
                    fixture=t.fixture,
                    timeout=t.timeout,
                )
            score(rd, t, root=root)

        try:
            workers = max(1, min(len(jobs), 3))
            if workers == 1:
                for j in jobs:
                    _run_one(j)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(_run_one, jobs))
        finally:
            shutil.rmtree(install.staging_dir, ignore_errors=True)

        summary = summarize_arm_runs(tasks, out_results_dir, k=k, executor=eff_exec)
        if not summary.get("executor"):
            summary["executor"] = eff_exec
        return summary


def evaluate_acceptance(
    current_with: Mapping[str, Any],
    candidate_with: Mapping[str, Any],
) -> tuple[bool, str]:
    """Decide whether `candidate_with` should replace `current_with`."""
    m_cur = current_with.get("mean")
    m_cand = candidate_with.get("mean")
    if not isinstance(m_cur, (int, float)) or not isinstance(m_cand, (int, float)):
        reasons = candidate_with.get("invalid_reasons") or []
        detail = f" ({'; '.join(reasons)})" if reasons else ""
        return (
            False,
            f"missing rubric score in current or candidate arm "
            f"(candidate valid={candidate_with.get('n_valid')}/{candidate_with.get('n_total')}{detail})",
        )

    v_cur = (current_with.get("verify") or {}).get("rate")
    v_cand = (candidate_with.get("verify") or {}).get("rate")
    if (
        isinstance(v_cur, (int, float))
        and isinstance(v_cand, (int, float))
        and v_cand < v_cur - 1e-6
    ):
        return False, f"deterministic verify rate regressed ({v_cand:.2f} < {v_cur:.2f})"

    inv_cur = current_with.get("invocation_rate")
    inv_cand = candidate_with.get("invocation_rate")
    if (
        isinstance(inv_cur, (int, float))
        and isinstance(inv_cand, (int, float))
        and inv_cand < inv_cur - 1e-6
    ):
        return False, f"skill invocation rate regressed ({inv_cand:.2f} < {inv_cur:.2f})"

    if m_cand < m_cur - 1e-6:
        return False, f"rubric score regressed ({m_cand:.3f} < {m_cur:.3f})"

    if m_cand > m_cur + 1e-6:
        return True, f"rubric score improved ({m_cur:.3f} -> {m_cand:.3f}, +{m_cand - m_cur:.3f})"

    tok_cur = current_with.get("tokens_median")
    tok_cand = candidate_with.get("tokens_median")
    if (
        isinstance(tok_cur, (int, float))
        and isinstance(tok_cand, (int, float))
        and tok_cur > 0
        and tok_cand <= EFFICIENCY_GAIN_RATIO * tok_cur
    ):
        pct = (1.0 - tok_cand / tok_cur) * 100.0
        return (
            True,
            f"equal rubric score ({m_cand:.3f}) with {pct:.1f}% lower median tokens "
            f"({int(tok_cur):,} -> {int(tok_cand):,})",
        )

    t_cur = current_with.get("t_total_median_s")
    t_cand = candidate_with.get("t_total_median_s")
    if (
        isinstance(t_cur, (int, float))
        and isinstance(t_cand, (int, float))
        and t_cur > 0
        and t_cand <= EFFICIENCY_GAIN_RATIO * t_cur
    ):
        pct = (1.0 - t_cand / t_cur) * 100.0
        return (
            True,
            f"equal rubric score ({m_cand:.3f}) with {pct:.1f}% faster median wall time "
            f"({t_cur:.1f}s -> {t_cand:.1f}s)",
        )

    return (
        False,
        f"no quality gain ({m_cur:.3f} -> {m_cand:.3f}) and efficiency improvement < 15%",
    )


def _summary_rank_tuple(summary: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """`(verify, mean, -tokens, -wall)` for one candidate summary dict."""
    ver_obj = summary.get("verify")
    ver_rate = ver_obj.get("rate") if isinstance(ver_obj, Mapping) else summary.get("verify_rate")
    v_val = (
        float(ver_rate)
        if isinstance(ver_rate, (int, float)) and not isinstance(ver_rate, bool)
        else -1.0
    )
    m_raw = summary.get("mean", summary.get("mean_candidate"))
    m_val = (
        float(m_raw)
        if isinstance(m_raw, (int, float)) and not isinstance(m_raw, bool)
        else -1.0
    )
    tok_raw = summary.get("tokens_median")
    neg_tok = (
        -float(tok_raw)
        if isinstance(tok_raw, (int, float)) and not isinstance(tok_raw, bool)
        else float("-inf")
    )
    wall_raw = summary.get("t_total_median_s")
    neg_wall = (
        -float(wall_raw)
        if isinstance(wall_raw, (int, float)) and not isinstance(wall_raw, bool)
        else float("-inf")
    )
    return (v_val, m_val, neg_tok, neg_wall)


def round_rank_key(summary_or_round: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Lexicographic key `(verify, mean, -tokens, -wall)` for selecting the best round."""
    gates = summary_or_round.get("model_gates")
    if isinstance(gates, list) and gates:
        tuples = [
            _summary_rank_tuple(g["candidate"])
            for g in gates
            if isinstance(g, Mapping) and isinstance(g.get("candidate"), Mapping)
        ]
        if tuples:
            v_vals = [t[0] for t in tuples if t[0] >= 0.0]
            m_vals = [t[1] for t in tuples if t[1] >= 0.0]
            tok_vals = [t[2] for t in tuples if t[2] != float("-inf")]
            wall_vals = [t[3] for t in tuples if t[3] != float("-inf")]
            return (
                st.fmean(v_vals) if v_vals else -1.0,
                st.fmean(m_vals) if m_vals else -1.0,
                st.fmean(tok_vals) if tok_vals else float("-inf"),
                st.fmean(wall_vals) if wall_vals else float("-inf"),
            )
    cand = summary_or_round.get("candidate")
    if isinstance(cand, Mapping):
        return _summary_rank_tuple(cand)
    return _summary_rank_tuple(summary_or_round)


def _fmt_num(v: Any, fmt: str = ".2f", suffix: str = "") -> str:
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        return "—"
    if fmt == "int":
        return f"{int(round(v)):,}{suffix}"
    if fmt == "pct":
        return f"{v * 100:.0f}%"
    return f"{v:{fmt}}{suffix}"


def _arm_exec_str(arm_summary: Mapping[str, Any], fallback: Any) -> str:
    return (
        _str_or_none(arm_summary.get("executor"))
        or _str_or_none(arm_summary.get("recorded_executor"))
        or _str_or_none(fallback)
        or "—"
    )


def _model_gates_for_skill_record(
    s: Mapping[str, Any],
    fallback_exec: Any,
    fallback_judge: Any,
) -> list[dict[str, Any]]:
    gates = s.get("model_gates")
    if isinstance(gates, list) and gates:
        return [dict(g) for g in gates if isinstance(g, Mapping)]
    wo = s.get("without") if isinstance(s.get("without"), Mapping) else {}
    cur = s.get("current_with") if isinstance(s.get("current_with"), Mapping) else {}
    cand = s.get("best_candidate") if isinstance(s.get("best_candidate"), Mapping) else {}
    wo_ex = _arm_exec_str(wo, fallback_exec)
    cur_ex = _arm_exec_str(cur, fallback_exec)
    cand_ex = _arm_exec_str(cand, fallback_exec)
    return [
        {
            "executor": cand_ex if cand_ex != "—" else (cur_ex if cur_ex != "—" else wo_ex),
            "judge": _str_or_none(fallback_judge) or "—",
            "without_executor": wo_ex,
            "current_with_executor": cur_ex,
            "candidate_executor": cand_ex,
            "accepted": bool(s.get("accepted")),
            "skipped": bool(s.get("skipped")),
            "decision_reason": str(s.get("decision_reason") or ""),
            "without": wo,
            "current_with": cur,
            "candidate": cand,
        }
    ]


def format_decision_line(
    s: Mapping[str, Any],
    *,
    fallback_exec: Any = None,
    fallback_judge: Any = None,
) -> str:
    """Format the top per-skill Decision line:
    `ACCEPTED on [models] — <reason> (<diff_stat>)` or `REJECTED: <reason> (<diff_stat>)`."""
    gates = _model_gates_for_skill_record(s, fallback_exec, fallback_judge)
    raw_models = s.get("models")
    models_list: list[str] = []
    if isinstance(raw_models, list):
        models_list = [str(m).strip() for m in raw_models if str(m).strip()]
    if not models_list:
        models_list = [
            str(g.get("executor")).strip()
            for g in gates
            if g.get("executor") and str(g.get("executor")).strip() != "—"
        ]
    if not models_list and fallback_exec:
        models_list = [str(fallback_exec).strip()]
    models_str = ", ".join(models_list) or "default"
    dstat = str(s.get("diff_stat_str") or format_diff_stat(str(s.get("diff") or "")))
    reason = str(s.get("decision_reason") or s.get("skip_reason") or "").strip()
    if s.get("accepted"):
        if reason:
            return f"ACCEPTED on [{models_str}] — {reason} ({dstat})"
        return f"ACCEPTED on [{models_str}] ({dstat})"
    reason_out = reason or "candidate did not pass gate"
    return f"REJECTED: {reason_out} ({dstat})"


def render_improve_markdown(record: Mapping[str, Any]) -> str:
    """Render `improve_report.md` for one or more improved skills."""
    fallback_exec = record.get("executor_model")
    fallback_judge = record.get("judge_model")
    models_hdr = record.get("models")
    models_line = (
        f"- **Models**: `{', '.join(str(m) for m in models_hdr)}`"
        if isinstance(models_hdr, list) and models_hdr
        else None
    )
    lines = [
        "# skillbench Skill Improvement Report (3-Way Comparison)",
        "",
        f"- **Improver model**: `{record.get('improver_model')}`",
        f"- **Executor model**: `{fallback_exec}`",
        f"- **Judge model**: `{fallback_judge}`",
    ]
    if models_line:
        lines.append(models_line)
    lines.extend(
        [
            f"- **Include verdicts in evidence**: `{record.get('include_verdicts')}`",
            "",
            "## Summary: `WITHOUT` vs `Current WITH` vs `Candidate WITH`",
            "",
            "| Skill | Status | WITHOUT | Current WITH | Candidate WITH | Lift (Cand vs Cur) | Verify (W/O → Cur → Cand) | Tokens (W/O → Cur → Cand) | Wall s (W/O → Cur → Cand) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for skill, s in sorted((record.get("skills") or {}).items()):
        wo = s.get("without") or {}
        cur = s.get("current_with") or {}
        cand = s.get("best_candidate") or {}
        delta_cur = (
            cand["mean"] - cur["mean"]
            if isinstance(cand.get("mean"), (int, float))
            and isinstance(cur.get("mean"), (int, float))
            else None
        )
        if s.get("skipped"):
            status = "SKIPPED"
        elif s.get("accepted"):
            status = "ACCEPTED"
        else:
            status = "REJECTED"
        lines.append(
            f"| `{skill}` | **{status}** | {_fmt_num(wo.get('mean'))} | "
            f"{_fmt_num(cur.get('mean'))} | **{_fmt_num(cand.get('mean'))}** | "
            f"{_fmt_num(delta_cur, '+.2f')} | "
            f"{_fmt_num((wo.get('verify') or {}).get('rate'), 'pct')} → "
            f"{_fmt_num((cur.get('verify') or {}).get('rate'), 'pct')} → "
            f"{_fmt_num((cand.get('verify') or {}).get('rate'), 'pct')} | "
            f"{_fmt_num(wo.get('tokens_median'), 'int')} → "
            f"{_fmt_num(cur.get('tokens_median'), 'int')} → "
            f"{_fmt_num(cand.get('tokens_median'), 'int')} | "
            f"{_fmt_num(wo.get('t_total_median_s'), '.1f', 's')} → "
            f"{_fmt_num(cur.get('t_total_median_s'), '.1f', 's')} → "
            f"{_fmt_num(cand.get('t_total_median_s'), '.1f', 's')} |"
        )

    for skill, s in sorted((record.get("skills") or {}).items()):
        wo = s.get("without") or {}
        cur = s.get("current_with") or {}
        cand = s.get("best_candidate") or {}
        wo_ex = _arm_exec_str(wo, fallback_exec)
        cur_ex = _arm_exec_str(cur, fallback_exec)
        cand_ex = _arm_exec_str(cand, fallback_exec)
        if s.get("skipped"):
            status = "SKIPPED"
        elif s.get("accepted"):
            status = "ACCEPTED"
        else:
            status = "REJECTED"
        gates = _model_gates_for_skill_record(s, fallback_exec, fallback_judge)
        decision_line = format_decision_line(
            s, fallback_exec=fallback_exec, fallback_judge=fallback_judge
        )
        lines.extend(
            [
                "",
                f"## Skill: `{skill}` ({status})",
                "",
                f"- **Decision**: {decision_line}",
                (
                    f"- **Arm executors**: WITHOUT=`{wo_ex}` | "
                    f"Current WITH=`{cur_ex}` | Candidate WITH=`{cand_ex}`"
                ),
                f"- **Winning round**: `{s.get('best_round')}` (of {len(s.get('rounds') or [])})",
                f"- **Diagnosis**: {s.get('diagnosis')}",
                f"- **Changes summary**: {s.get('changes_summary')}",
                "",
                "### Per-Model Gate (`WITHOUT` vs `Current WITH` vs `Candidate WITH`)",
                "",
                "| Executor | Gate | WITHOUT (executor) | Current WITH (executor) | Candidate WITH (executor) | Lift (Cand vs Cur) | Verify (W/O → Cur → Cand) | Tokens (W/O → Cur → Cand) | Wall s (W/O → Cur → Cand) | Reason |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
            ]
        )
        for g in gates:
            g_wo = g.get("without") or {}
            g_cur = g.get("current_with") or {}
            g_cand = g.get("candidate") or {}
            g_wo_ex = g.get("without_executor") or _arm_exec_str(g_wo, g.get("executor") or fallback_exec)
            g_cur_ex = g.get("current_with_executor") or _arm_exec_str(g_cur, g.get("executor") or fallback_exec)
            g_cand_ex = g.get("candidate_executor") or _arm_exec_str(g_cand, g.get("executor") or fallback_exec)
            g_delta = (
                g_cand["mean"] - g_cur["mean"]
                if isinstance(g_cand.get("mean"), (int, float))
                and isinstance(g_cur.get("mean"), (int, float))
                else None
            )
            if g.get("skipped"):
                g_status = "SKIPPED"
            elif g.get("accepted"):
                g_status = "ACCEPTED"
            else:
                g_status = "REJECTED"
            lines.append(
                f"| `{g.get('executor')}` | **{g_status}** | "
                f"{_fmt_num(g_wo.get('mean'))} (`{g_wo_ex}`) | "
                f"{_fmt_num(g_cur.get('mean'))} (`{g_cur_ex}`) | "
                f"**{_fmt_num(g_cand.get('mean'))}** (`{g_cand_ex}`) | "
                f"{_fmt_num(g_delta, '+.2f')} | "
                f"{_fmt_num((g_wo.get('verify') or {}).get('rate'), 'pct')} → "
                f"{_fmt_num((g_cur.get('verify') or {}).get('rate'), 'pct')} → "
                f"{_fmt_num((g_cand.get('verify') or {}).get('rate'), 'pct')} | "
                f"{_fmt_num(g_wo.get('tokens_median'), 'int')} → "
                f"{_fmt_num(g_cur.get('tokens_median'), 'int')} → "
                f"{_fmt_num(g_cand.get('tokens_median'), 'int')} | "
                f"{_fmt_num(g_wo.get('t_total_median_s'), '.1f', 's')} → "
                f"{_fmt_num(g_cur.get('t_total_median_s'), '.1f', 's')} → "
                f"{_fmt_num(g_cand.get('t_total_median_s'), '.1f', 's')} | "
                f"{g.get('decision_reason') or ''} |"
            )
        lines.extend(
            [
                "",
                "### Proposed `SKILL.md` Diff (`best.patch`)",
                "",
                "```diff",
                (s.get("diff") or "").rstrip(),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def render_improve_html(record: Mapping[str, Any]) -> str:
    """Render a self-contained interactive 3-way HTML dashboard (`improve_report.html`)."""
    e = html.escape
    fallback_exec = record.get("executor_model")
    fallback_judge = record.get("judge_model")
    rows_html: list[str] = []
    cards_html: list[str] = []
    for skill, s in sorted((record.get("skills") or {}).items()):
        wo = s.get("without") or {}
        cur = s.get("current_with") or {}
        cand = s.get("best_candidate") or {}
        wo_ex = _arm_exec_str(wo, fallback_exec)
        cur_ex = _arm_exec_str(cur, fallback_exec)
        cand_ex = _arm_exec_str(cand, fallback_exec)
        skipped = bool(s.get("skipped"))
        accepted = bool(s.get("accepted"))
        if skipped:
            badge_cls = "bg-slate-500/15 text-slate-400"
            badge_txt = "SKIPPED"
        elif accepted:
            badge_cls = "bg-emerald-500/15 text-emerald-500"
            badge_txt = "ACCEPTED"
        else:
            badge_cls = "bg-amber-500/15 text-amber-500"
            badge_txt = "REJECTED"
        delta_cur = (
            cand["mean"] - cur["mean"]
            if isinstance(cand.get("mean"), (int, float))
            and isinstance(cur.get("mean"), (int, float))
            else None
        )
        rows_html.append(
            f"<tr class='border-b border-[var(--border,#334155)]'>"
            f"<td class='py-2.5 px-3 font-mono text-xs'>{e(skill)}</td>"
            f"<td class='py-2.5 px-3'><span class='inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold {badge_cls}'>{badge_txt}</span></td>"
            f"<td class='py-2.5 px-3 text-right tabular-nums'>{e(_fmt_num(wo.get('mean')))} <span class='text-xs text-[var(--muted-foreground,#94a3b8)] font-mono'>({e(wo_ex)})</span></td>"
            f"<td class='py-2.5 px-3 text-right tabular-nums'>{e(_fmt_num(cur.get('mean')))} <span class='text-xs text-[var(--muted-foreground,#94a3b8)] font-mono'>({e(cur_ex)})</span></td>"
            f"<td class='py-2.5 px-3 text-right tabular-nums font-semibold'>{e(_fmt_num(cand.get('mean')))} <span class='text-xs text-[var(--muted-foreground,#94a3b8)] font-mono'>({e(cand_ex)})</span></td>"
            f"<td class='py-2.5 px-3 text-right tabular-nums'>{e(_fmt_num(delta_cur, '+.2f'))}</td>"
            f"<td class='py-2.5 px-3 text-right tabular-nums'>{e(_fmt_num((wo.get('verify') or {}).get('rate'), 'pct'))} → "
            f"{e(_fmt_num((cur.get('verify') or {}).get('rate'), 'pct'))} → "
            f"<strong>{e(_fmt_num((cand.get('verify') or {}).get('rate'), 'pct'))}</strong></td>"
            f"<td class='py-2.5 px-3 text-right tabular-nums'>{e(_fmt_num(cur.get('t_skill_p50_s'), '.1f', 's'))} → "
            f"<strong>{e(_fmt_num(cand.get('t_skill_p50_s'), '.1f', 's'))}</strong></td>"
            f"<td class='py-2.5 px-3 text-right tabular-nums'>{e(_fmt_num(wo.get('tokens_median'), 'int'))} → "
            f"{e(_fmt_num(cur.get('tokens_median'), 'int'))} → "
            f"<strong>{e(_fmt_num(cand.get('tokens_median'), 'int'))}</strong></td>"
            f"<td class='py-2.5 px-3 text-right tabular-nums'>{e(_fmt_num(wo.get('t_total_median_s'), '.1f', 's'))} → "
            f"{e(_fmt_num(cur.get('t_total_median_s'), '.1f', 's'))} → "
            f"<strong>{e(_fmt_num(cand.get('t_total_median_s'), '.1f', 's'))}</strong></td>"
            f"</tr>"
        )

        gates = _model_gates_for_skill_record(s, fallback_exec, fallback_judge)
        decision_line = format_decision_line(
            s, fallback_exec=fallback_exec, fallback_judge=fallback_judge
        )
        gate_rows_html: list[str] = []
        for g in gates:
            g_wo = g.get("without") or {}
            g_cur = g.get("current_with") or {}
            g_cand = g.get("candidate") or {}
            g_wo_ex = g.get("without_executor") or _arm_exec_str(g_wo, g.get("executor") or fallback_exec)
            g_cur_ex = g.get("current_with_executor") or _arm_exec_str(g_cur, g.get("executor") or fallback_exec)
            g_cand_ex = g.get("candidate_executor") or _arm_exec_str(g_cand, g.get("executor") or fallback_exec)
            g_delta = (
                g_cand["mean"] - g_cur["mean"]
                if isinstance(g_cand.get("mean"), (int, float))
                and isinstance(g_cur.get("mean"), (int, float))
                else None
            )
            if g.get("skipped"):
                g_cls = "bg-slate-500/15 text-slate-400"
                g_txt = "SKIPPED"
            elif g.get("accepted"):
                g_cls = "bg-emerald-500/15 text-emerald-500"
                g_txt = "ACCEPTED"
            else:
                g_cls = "bg-amber-500/15 text-amber-500"
                g_txt = "REJECTED"
            gate_rows_html.append(
                f"<tr class='border-b border-[var(--border,#334155)]' data-executor='{e(str(g.get('executor') or ''))}'>"
                f"<td class='py-1.5 px-2 font-mono text-xs'>{e(str(g.get('executor') or ''))}</td>"
                f"<td class='py-1.5 px-2'><span class='inline-block rounded-full px-2 py-0.5 text-xs font-semibold {g_cls}'>{g_txt}</span></td>"
                f"<td class='py-1.5 px-2 text-right tabular-nums'>{e(_fmt_num(g_wo.get('mean')))} <code class='text-xs'>({e(str(g_wo_ex))})</code></td>"
                f"<td class='py-1.5 px-2 text-right tabular-nums'>{e(_fmt_num(g_cur.get('mean')))} <code class='text-xs'>({e(str(g_cur_ex))})</code></td>"
                f"<td class='py-1.5 px-2 text-right tabular-nums font-semibold'>{e(_fmt_num(g_cand.get('mean')))} <code class='text-xs'>({e(str(g_cand_ex))})</code></td>"
                f"<td class='py-1.5 px-2 text-right tabular-nums'>{e(_fmt_num(g_delta, '+.2f'))}</td>"
                f"<td class='py-1.5 px-2 text-left text-xs'>{e(str(g.get('decision_reason') or ''))}</td>"
                f"</tr>"
            )

        cards_html.append(
            f"<div class='bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-xl p-5 mb-4 shadow-sm'>"
            f"<h3 class='text-base font-semibold mb-2'><code class='font-mono'>{e(skill)}</code> — "
            f"<span class='inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold {badge_cls}'>{badge_txt}</span></h3>"
            f"<p class='text-sm mb-1' data-decision-line='true'><strong>Decision:</strong> {e(decision_line)}</p>"
            f"<p class='text-xs text-[var(--muted-foreground,#94a3b8)] mb-3'><strong>Arm executors:</strong> "
            f"WITHOUT <code class='font-mono'>{e(wo_ex)}</code> · "
            f"Current WITH <code class='font-mono'>{e(cur_ex)}</code> · "
            f"Candidate WITH <code class='font-mono'>{e(cand_ex)}</code></p>"
            f"<table class='w-full text-xs border-collapse mb-3'>"
            f"<thead><tr class='border-b border-[var(--border,#334155)] text-[var(--muted-foreground,#94a3b8)]'>"
            f"<th class='py-1 px-2 text-left'>Executor</th><th class='py-1 px-2 text-left'>Gate</th>"
            f"<th class='py-1 px-2 text-right'>WITHOUT</th><th class='py-1 px-2 text-right'>Current WITH</th>"
            f"<th class='py-1 px-2 text-right'>Candidate WITH</th><th class='py-1 px-2 text-right'>Δ</th>"
            f"<th class='py-1 px-2 text-left'>Reason</th></tr></thead>"
            f"<tbody>{''.join(gate_rows_html)}</tbody></table>"
            f"<p class='text-sm text-[var(--muted-foreground,#94a3b8)] mb-2 whitespace-pre-line'><strong>Diagnosis:</strong>\n{e(str(s.get('diagnosis') or ''))}</p>"
            f"<p class='text-sm text-[var(--muted-foreground,#94a3b8)] mb-3 whitespace-pre-line'><strong>Changes Summary:</strong>\n{e(str(s.get('changes_summary') or ''))}</p>"
            f"<details open><summary class='text-xs font-semibold cursor-pointer mb-2'>Unified Diff (<code>best.patch</code>)</summary>"
            f"<pre class='bg-[var(--background,#0f172a)] border border-[var(--border,#334155)] rounded-lg p-3 overflow-x-auto text-xs font-mono'>{e(str(s.get('diff') or ''))}</pre></details>"
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>skillbench-improve — 3-Way Skill Improvement Report</title>
<script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-[var(--background,#0f172a)] text-[var(--foreground,#f8fafc)] antialiased p-6">
<div class="max-w-6xl mx-auto">
  <h1 class="text-xl font-bold mb-1">skillbench-improve: 3-Way Skill Optimization Report</h1>
  <div class="text-xs text-[var(--muted-foreground,#94a3b8)] mb-5">
    Improver: <code class="font-mono">{e(str(record.get('improver_model')))}</code> ·
    Executor: <code class="font-mono">{e(str(fallback_exec))}</code> ·
    Judge: <code class="font-mono">{e(str(fallback_judge))}</code> ·
    Verdicts in evidence: <code class="font-mono">{e(str(record.get('include_verdicts')))}</code>
  </div>
  <div class="bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-xl p-5 mb-5 shadow-sm overflow-x-auto">
    <h2 class="text-base font-semibold mb-3">3-Way Comparison (WITHOUT → Current WITH → Candidate WITH)</h2>
    <table class="w-full text-sm border-collapse">
      <thead>
        <tr class="border-b border-[var(--border,#334155)] text-xs text-[var(--muted-foreground,#94a3b8)]">
          <th class="py-2 px-3 text-left">Skill</th>
          <th class="py-2 px-3 text-left">Gate</th>
          <th class="py-2 px-3 text-right">WITHOUT</th>
          <th class="py-2 px-3 text-right">Current WITH</th>
          <th class="py-2 px-3 text-right">Candidate WITH</th>
          <th class="py-2 px-3 text-right">Δ (Cand - Cur)</th>
          <th class="py-2 px-3 text-right">Verify</th>
          <th class="py-2 px-3 text-right">t_skill p50</th>
          <th class="py-2 px-3 text-right">Median Tokens</th>
          <th class="py-2 px-3 text-right">Median Wall</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
  </div>
  {"".join(cards_html)}
</div>
</body>
</html>
"""


def _has_cell_content(cell_dir: Path, tasks: Sequence[SkillbenchTask], k: int) -> bool:
    if not cell_dir.is_dir():
        return False
    if (cell_dir / "run.json").is_file() or (cell_dir / "config.json").is_file():
        return True
    for task in tasks:
        for i in range(1, max(1, k) + 1):
            if resolve_rep_dir(cell_dir, task.task_id, i).is_dir():
                return True
    return any(cell_dir.glob("**/skill_timing.json"))


def _discover_matrix_exec_dirs(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(results_root.iterdir()):
        if not p.is_dir() or p.name in ("with", "without", "baseline") or p.name.startswith("suite-"):
            continue
        if (p / "without").is_dir() or (p / "with").is_dir():
            out.append(p)
    return out


def _resolve_model_cells(
    runs_dir: Path,
    tasks: Sequence[SkillbenchTask],
    k: int,
    *,
    models: Sequence[str] | str | None = None,
    executor: str | None = None,
    allow_model_mismatch: bool = False,
) -> list[tuple[str, Path, Path, str | None, str | None]]:
    """Resolve `[(probe_executor, without_dir, with_dir, recorded_wo_exec, recorded_w_exec), ...]`.

    Enforces that the probe executor matches the executor recorded in the baseline cell(s) unless
    `allow_model_mismatch=True`."""
    results_root = runs_dir / "results"
    explicit = _parse_models(models, executor)
    env_exec = _str_or_none(config.env("SKILLBENCH_EXEC_MODEL"))

    flat_wo = results_root / "without"
    flat_w = results_root / "with"
    has_flat = _has_cell_content(flat_wo, tasks, k) or _has_cell_content(flat_w, tasks, k)
    matrix_dirs = _discover_matrix_exec_dirs(results_root)

    if not explicit:
        if has_flat:
            wo_exec = detect_cell_executor(flat_wo, tasks, k)
            w_exec = detect_cell_executor(flat_w, tasks, k)
            if wo_exec and w_exec and wo_exec != w_exec and not allow_model_mismatch:
                raise ValueError(
                    f"model mismatch between baseline cells: WITHOUT recorded {wo_exec!r}, "
                    f"Current WITH recorded {w_exec!r} (pass --allow-model-mismatch to override)"
                )
            recorded_base = wo_exec or w_exec
            if recorded_base is not None:
                if env_exec is not None and env_exec != recorded_base:
                    if not allow_model_mismatch:
                        raise ValueError(
                            f"model mismatch: probe executor {env_exec!r} does not match "
                            f"baseline cell executor {recorded_base!r} "
                            f"(pass --allow-model-mismatch to override)"
                        )
                    probe_exec = env_exec
                else:
                    probe_exec = recorded_base
            else:
                probe_exec = env_exec or exec_model()
            return [(probe_exec, flat_wo, flat_w, wo_exec, w_exec)]

        if matrix_dirs:
            by_name: dict[str, tuple[Path, Path, str | None, str | None]] = {}
            for mdir in matrix_dirs:
                wo_d, w_d = mdir / "without", mdir / "with"
                wo_ex = detect_cell_executor(wo_d, tasks, k) or mdir.name
                w_ex = detect_cell_executor(w_d, tasks, k) or mdir.name
                if wo_ex and w_ex and wo_ex != w_ex and not allow_model_mismatch:
                    raise ValueError(
                        f"model mismatch between baseline cells in {mdir}: WITHOUT recorded "
                        f"{wo_ex!r}, Current WITH recorded {w_ex!r} "
                        f"(pass --allow-model-mismatch to override)"
                    )
                key = wo_ex or w_ex or mdir.name
                by_name[key] = (wo_d, w_d, wo_ex, w_ex)
                by_name.setdefault(mdir.name, (wo_d, w_d, wo_ex, w_ex))

            if env_exec is not None:
                if env_exec in by_name:
                    wo_d, w_d, wo_ex, w_ex = by_name[env_exec]
                    rec_base = wo_ex or w_ex
                    if rec_base and rec_base != env_exec and not allow_model_mismatch:
                        raise ValueError(
                            f"model mismatch: probe executor {env_exec!r} does not match "
                            f"baseline cell executor {rec_base!r} "
                            f"(pass --allow-model-mismatch to override)"
                        )
                    return [(env_exec, wo_d, w_d, wo_ex, w_ex)]
                first_mdir = matrix_dirs[0]
                wo_d, w_d, wo_ex, w_ex = by_name[first_mdir.name]
                rec_base = wo_ex or w_ex or first_mdir.name
                if not allow_model_mismatch:
                    raise ValueError(
                        f"model mismatch: probe executor {env_exec!r} does not match "
                        f"baseline cell executor {rec_base!r} "
                        f"(pass --allow-model-mismatch to override)"
                    )
                return [(env_exec, wo_d, w_d, wo_ex, w_ex)]

            first_mdir = matrix_dirs[0]
            wo_d, w_d, wo_ex, w_ex = by_name[first_mdir.name]
            probe_exec = wo_ex or w_ex or first_mdir.name
            return [(probe_exec, wo_d, w_d, wo_ex, w_ex)]

        probe_exec = env_exec or exec_model()
        return [(probe_exec, flat_wo, flat_w, None, None)]

    resolved: list[tuple[str, Path, Path, str | None, str | None]] = []
    for m in explicit:
        seg = _path_segment(m)
        cand_dirs = [results_root / seg, results_root / m]
        matched_mdir = next(
            (
                d
                for d in cand_dirs
                if _has_cell_content(d / "without", tasks, k)
                or _has_cell_content(d / "with", tasks, k)
            ),
            None,
        )
        if matched_mdir is not None:
            wo_d = matched_mdir / "without"
            w_d = matched_mdir / "with"
            wo_ex = detect_cell_executor(wo_d, tasks, k)
            w_ex = detect_cell_executor(w_d, tasks, k)
            if wo_ex and w_ex and wo_ex != w_ex and not allow_model_mismatch:
                raise ValueError(
                    f"model mismatch between baseline cells for {m!r}: WITHOUT recorded "
                    f"{wo_ex!r}, Current WITH recorded {w_ex!r} "
                    f"(pass --allow-model-mismatch to override)"
                )
            for arm_lbl, rec_ex in (("WITHOUT", wo_ex), ("Current WITH", w_ex)):
                if rec_ex is not None and rec_ex != m and not allow_model_mismatch:
                    raise ValueError(
                        f"model mismatch: probe executor {m!r} does not match {arm_lbl} "
                        f"baseline cell executor {rec_ex!r} "
                        f"(pass --allow-model-mismatch to override)"
                    )
            resolved.append((m, wo_d, w_d, wo_ex or m, w_ex or m))
            continue

        if has_flat:
            wo_ex = detect_cell_executor(flat_wo, tasks, k)
            w_ex = detect_cell_executor(flat_w, tasks, k)
            if wo_ex and w_exec_mismatch(wo_ex, w_ex) and not allow_model_mismatch:
                raise ValueError(
                    f"model mismatch between baseline cells: WITHOUT recorded {wo_ex!r}, "
                    f"Current WITH recorded {w_ex!r} (pass --allow-model-mismatch to override)"
                )
            rec_base = wo_ex or w_ex
            if rec_base is not None and rec_base != m and not allow_model_mismatch:
                raise ValueError(
                    f"model mismatch: probe executor {m!r} does not match baseline cell "
                    f"executor {rec_base!r} (pass --allow-model-mismatch to override)"
                )
            resolved.append((m, flat_wo, flat_w, wo_ex or m, w_ex or m))
            continue

        if matrix_dirs:
            first_mdir = matrix_dirs[0]
            wo_d, w_d = first_mdir / "without", first_mdir / "with"
            wo_ex = detect_cell_executor(wo_d, tasks, k) or first_mdir.name
            w_ex = detect_cell_executor(w_d, tasks, k) or first_mdir.name
            avail = [d.name for d in matrix_dirs]
            if not allow_model_mismatch:
                raise ValueError(
                    f"model mismatch: no baseline cell for probe executor {m!r} under "
                    f"{results_root} (found {avail}; pass --allow-model-mismatch to override)"
                )
            resolved.append((m, wo_d, w_d, wo_ex, w_ex))
            continue

        resolved.append((m, flat_wo, flat_w, None, None))

    return resolved


def w_exec_mismatch(wo_ex: str | None, w_ex: str | None) -> bool:
    return bool(wo_ex and w_ex and wo_ex != w_ex)


def _resolve_judge_for_probe(
    executor: str,
    wo_dir: Path,
    w_dir: Path,
    tasks: Sequence[SkillbenchTask],
    k: int,
    suite: SuiteConfig,
) -> str:
    """Resolve the judge model for `executor` from the baseline cell, suite `[matrix.judge_for]`,
    or `judge_model()`."""
    cell_j = detect_cell_judge(w_dir, tasks, k) or detect_cell_judge(wo_dir, tasks, k)
    if cell_j:
        return cell_j
    if suite.judge_for:
        try:
            return suite.judge_for_executor(executor)
        except SuiteError:
            pass
    return judge_model()


def _call_probe_candidate(
    *,
    skill: str,
    candidate_skill_path: Path,
    tasks: Sequence[SkillbenchTask],
    out_results_dir: Path,
    k: int,
    root: Path | None,
    executor: str,
    judge: str,
    baseline_executor: str | None,
    allow_model_mismatch: bool,
) -> dict[str, Any]:
    """Invoke `probe_candidate` inside `executor_env(executor, judge)` and verify model guard."""
    with executor_env(executor, judge, matrix=False):
        raw = probe_candidate(
            skill=skill,
            candidate_skill_path=candidate_skill_path,
            tasks=tasks,
            out_results_dir=out_results_dir,
            k=k,
            root=root,
            executor=executor,
            judge=judge,
            baseline_executor=baseline_executor,
            allow_model_mismatch=allow_model_mismatch,
        )
    summary = dict(raw) if isinstance(raw, Mapping) else {}
    rec_cand_exec = _str_or_none(summary.get("recorded_executor")) or _str_or_none(
        summary.get("executor")
    )
    if (
        rec_cand_exec is not None
        and baseline_executor is not None
        and rec_cand_exec != baseline_executor
        and not allow_model_mismatch
    ):
        raise ValueError(
            f"model mismatch: candidate probe reported executor {rec_cand_exec!r}, "
            f"baseline cell recorded {baseline_executor!r} "
            f"(pass --allow-model-mismatch to override)"
        )
    if not summary.get("executor"):
        summary["executor"] = executor
    return summary


def _git_toplevel(target_dir: Path) -> Path | None:
    """Return the realpath-resolved git work tree root for `target_dir`, or `None`."""
    if not target_dir.is_dir():
        return None
    proc = subprocess.run(
        ["git", "-C", str(target_dir), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or proc.stdout.strip().lower() != "true":
        return None
    top_proc = subprocess.run(
        ["git", "-C", str(target_dir), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if top_proc.returncode != 0 or not top_proc.stdout.strip():
        return None
    return Path(os.path.realpath(top_proc.stdout.strip()))


def resolve_git_worktree(
    plugins_dir: Path,
    skill_path: Path | None = None,
    *,
    require_clean: bool = True,
) -> tuple[Path, Path]:
    """Resolve `realpath(plugins_dir)` (and symlinked plugin/skill target when applicable),
    verify it sits inside a git work tree, and verify `git status --porcelain` is clean when
    `require_clean=True`. Returns `(repo_root, real_plugins_dir)`."""
    real_plugins_dir = Path(os.path.realpath(plugins_dir))
    repo_root = _git_toplevel(real_plugins_dir)
    if repo_root is None and skill_path is not None:
        real_skill = Path(os.path.realpath(skill_path))
        if real_skill != skill_path.resolve(strict=False) or skill_path.is_symlink() or any(
            p.is_symlink() for p in skill_path.parents
        ):
            repo_root = _git_toplevel(real_skill.parent)
            if repo_root is not None:
                # Re-anchor real_plugins_dir to the checkout's plugins directory
                real_plugin_dir = real_skill.parent.parent.parent
                if real_plugin_dir.parent.is_dir():
                    real_plugins_dir = real_plugin_dir.parent
    if repo_root is None:
        raise ValueError(
            f"cannot --apply: realpath(plugins_dir) {real_plugins_dir} is not inside a git work tree"
        )
    if require_clean:
        status_proc = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if status_proc.returncode != 0:
            raise ValueError(
                f"cannot --apply: git status failed in {repo_root}: {status_proc.stderr.strip()}"
            )
        dirty = status_proc.stdout.strip()
        if dirty:
            raise ValueError(
                f"cannot --apply: git work tree at {repo_root} has dirty status "
                f"(uncommitted changes):\n{dirty}"
            )
    return repo_root, real_plugins_dir


def build_apply_commit_message(skill: str, skill_record: Mapping[str, Any]) -> str:
    """Build the git commit message carrying the 3-way numbers for `--apply`."""
    fallback_exec = (skill_record.get("executors") or {}).get("candidate") or (
        (skill_record.get("models") or [None])[0]
    )
    decision_line = str(
        skill_record.get("decision_line")
        or format_decision_line(skill_record, fallback_exec=fallback_exec)
    )
    gates = _model_gates_for_skill_record(skill_record, fallback_exec, None)
    lines = [
        f"skillbench-improve({skill}): apply accepted SKILL.md candidate",
        "",
        f"Decision: {decision_line}",
    ]
    if skill_record.get("best_round") is not None:
        lines.append(f"Winning round: {skill_record.get('best_round')}")
    if skill_record.get("changes_summary"):
        lines.append(f"Changes: {skill_record.get('changes_summary')}")
    lines.append("")
    lines.append("3-way comparison (WITHOUT -> Current WITH -> Candidate WITH):")
    for g in gates:
        ex = g.get("executor") or fallback_exec or "default"
        wo = g.get("without") if isinstance(g.get("without"), Mapping) else {}
        cur = g.get("current_with") if isinstance(g.get("current_with"), Mapping) else {}
        cand = g.get("candidate") if isinstance(g.get("candidate"), Mapping) else {}
        wo_m = wo.get("mean")
        cur_m = cur.get("mean")
        cand_m = cand.get("mean")
        delta = (
            cand_m - cur_m
            if isinstance(cand_m, (int, float)) and isinstance(cur_m, (int, float))
            else None
        )
        wo_v = (wo.get("verify") or {}).get("rate") if isinstance(wo.get("verify"), Mapping) else None
        cur_v = (cur.get("verify") or {}).get("rate") if isinstance(cur.get("verify"), Mapping) else None
        cand_v = (cand.get("verify") or {}).get("rate") if isinstance(cand.get("verify"), Mapping) else None
        wo_t = wo.get("tokens_median")
        cur_t = cur.get("tokens_median")
        cand_t = cand.get("tokens_median")
        wo_w = wo.get("t_total_median_s")
        cur_w = cur.get("t_total_median_s")
        cand_w = cand.get("t_total_median_s")
        lines.append(
            f"- [{ex}] WITHOUT -> Current WITH -> Candidate WITH: "
            f"mean={_fmt_num(wo_m)} -> {_fmt_num(cur_m)} -> {_fmt_num(cand_m)} "
            f"(lift {_fmt_num(delta, '+.2f')}) | "
            f"verify={_fmt_num(wo_v, 'pct')} -> {_fmt_num(cur_v, 'pct')} -> {_fmt_num(cand_v, 'pct')} | "
            f"tokens={_fmt_num(wo_t, 'int')} -> {_fmt_num(cur_t, 'int')} -> {_fmt_num(cand_t, 'int')} | "
            f"wall={_fmt_num(wo_w, '.1f', 's')} -> {_fmt_num(cur_w, '.1f', 's')} -> {_fmt_num(cand_w, '.1f', 's')}"
        )
    return "\n".join(lines)


def apply_candidate_patch(
    *,
    skill: str,
    patch_path: Path,
    skill_record: Mapping[str, Any],
    suite: SuiteConfig | None = None,
    plugins_dir: Path | None = None,
    skill_path: Path | None = None,
    date: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Apply `best.patch` onto branch `skillbench/improve-<skill>-<date>` in the git work tree
    containing `realpath(plugins_dir)`, and commit with the 3-way numbers (never pushing).

    `dry_run=True` is the default: prints the git commands without mutating the checkout.
    When `dry_run=False` (`--apply`), enforces that `realpath(plugins_dir)` sits inside a git
    work tree with a clean status (`git status --porcelain`)."""
    resolved_suite = suite if suite is not None else active_suite()
    raw_plugins_dir = Path(plugins_dir) if plugins_dir is not None else resolved_suite.plugins_dir
    raw_skill_path = (
        Path(skill_path)
        if skill_path is not None
        else target_skill_file(skill, resolved_suite)
    )
    real_plugins_dir = Path(os.path.realpath(raw_plugins_dir))
    real_skill_path = Path(os.path.realpath(raw_skill_path))
    patch_path = Path(os.path.realpath(patch_path))

    date_str = str(date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"))
    branch = f"skillbench/improve-{skill}-{date_str}"
    commit_msg = build_apply_commit_message(skill, skill_record)

    if not dry_run:
        repo_root, real_plugins_dir = resolve_git_worktree(
            raw_plugins_dir, raw_skill_path, require_clean=True
        )
    else:
        try:
            repo_root, real_plugins_dir = resolve_git_worktree(
                raw_plugins_dir, raw_skill_path, require_clean=False
            )
        except ValueError:
            repo_root = real_plugins_dir

    try:
        rel_skill_path = real_skill_path.relative_to(repo_root)
    except ValueError:
        rel_skill_path = Path("skills") / skill / "SKILL.md"

    real_plugin_dir = real_skill_path.parent.parent.parent
    try:
        rel_plugin_dir = real_plugin_dir.relative_to(repo_root)
    except ValueError:
        rel_plugin_dir = Path(".")

    checkout_argv = ["git", "-C", str(repo_root), "checkout", "-b", branch]
    apply_argv = ["git", "-C", str(repo_root), "apply"]
    if str(rel_plugin_dir) not in ("", "."):
        apply_argv.extend(["--directory", str(rel_plugin_dir)])
    apply_argv.append(str(patch_path))
    add_argv = ["git", "-C", str(repo_root), "add", str(rel_skill_path)]
    commit_argv = [
        "git",
        "-C",
        str(repo_root),
        "commit",
        "--no-gpg-sign",
        "-m",
        commit_msg,
    ]

    cmd_strings = [
        shlex.join(checkout_argv),
        shlex.join(apply_argv),
        shlex.join(add_argv),
        shlex.join(commit_argv),
    ]
    prefix = "[dry-run --apply]" if dry_run else "[--apply]"
    for c_str in cmd_strings:
        print(f"{prefix} {c_str}", flush=True)

    if dry_run:
        return {
            "dry_run": True,
            "branch": branch,
            "commit_message": commit_msg,
            "commands": cmd_strings,
            "repo_root": str(repo_root),
            "real_plugins_dir": str(real_plugins_dir),
            "applied_to": None,
            "commit": None,
        }

    orig_ref_proc = subprocess.run(
        ["git", "-C", str(repo_root), "symbolic-ref", "--quiet", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if orig_ref_proc.returncode != 0 or not orig_ref_proc.stdout.strip():
        orig_ref_proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    orig_ref = orig_ref_proc.stdout.strip()

    def _rollback_branch() -> None:
        subprocess.run(
            ["git", "-C", str(repo_root), "reset", "--hard"],
            capture_output=True,
            check=False,
        )
        if orig_ref:
            subprocess.run(
                ["git", "-C", str(repo_root), "checkout", orig_ref],
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "-C", str(repo_root), "branch", "-D", branch],
                capture_output=True,
                check=False,
            )

    co_proc = subprocess.run(checkout_argv, capture_output=True, text=True, check=False)
    if co_proc.returncode != 0:
        raise RuntimeError(
            f"git command failed ({shlex.join(checkout_argv)}): "
            f"{co_proc.stderr.strip() or co_proc.stdout.strip()}"
        )

    for argv in (apply_argv, add_argv):
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            _rollback_branch()
            raise RuntimeError(
                f"git command failed ({shlex.join(argv)}): {proc.stderr.strip() or proc.stdout.strip()}"
            )

    # Ensure git user.name / user.email have fallbacks if not configured in the environment/repo.
    name_chk = subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.name"],
        capture_output=True,
        text=True,
        check=False,
    )
    email_chk = subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.email"],
        capture_output=True,
        text=True,
        check=False,
    )
    exec_commit_argv = ["git", "-C", str(repo_root)]
    if not name_chk.stdout.strip():
        exec_commit_argv.extend(["-c", "user.name=skillbench-improve"])
    if not email_chk.stdout.strip():
        exec_commit_argv.extend(["-c", "user.email=skillbench-improve@local"])
    exec_commit_argv.extend(["commit", "--no-gpg-sign", "-m", commit_msg])
    c_proc = subprocess.run(exec_commit_argv, capture_output=True, text=True, check=False)
    if c_proc.returncode != 0:
        _rollback_branch()
        raise RuntimeError(
            f"git commit failed in {repo_root}: {c_proc.stderr.strip() or c_proc.stdout.strip()}"
        )

    sha_proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit_sha = sha_proc.stdout.strip() if sha_proc.returncode == 0 else None
    return {
        "dry_run": False,
        "branch": branch,
        "commit_message": commit_msg,
        "commands": cmd_strings,
        "repo_root": str(repo_root),
        "real_plugins_dir": str(real_plugins_dir),
        "applied_to": str(real_skill_path),
        "commit": commit_sha,
    }


def improve_skill(
    skill: str,
    *,
    runs_dir: Path,
    k: int = 3,
    rounds: int = 1,
    pilot: bool | None = None,
    include_verdicts: bool = False,
    apply_accepted: bool = False,
    improver_model_name: str | None = None,
    root: Path | None = None,
    models: Sequence[str] | str | None = None,
    executor: str | None = None,
    allow_model_mismatch: bool = False,
    max_growth: float | None = DEFAULT_MAX_GROWTH_PCT,
    suite: SuiteConfig | Path | str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """Run up to `rounds` of Propose -> Probe -> Judge for `skill`."""
    all_tasks = load_tasks(root)
    pilot_only = _env_pilot() if pilot is None else bool(pilot)
    tasks = [
        t
        for t in all_tasks.values()
        if t.target_skill == skill and (not pilot_only or t.pilot)
    ]
    if not tasks:
        raise ValueError(f"no skillbench tasks found for skill {skill!r} (pilot={pilot_only})")

    resolved_suite = (
        suite
        if isinstance(suite, SuiteConfig)
        else (load_suite(suite) if suite is not None else active_suite())
    )
    skill_path = target_skill_file(skill, resolved_suite)
    if apply_accepted:
        resolve_git_worktree(resolved_suite.plugins_dir, skill_path, require_clean=True)
    original_md = skill_path.read_text(encoding="utf-8")

    model_cells = _resolve_model_cells(
        runs_dir,
        tasks,
        k,
        models=models,
        executor=executor,
        allow_model_mismatch=allow_model_mismatch,
    )
    models_list = [mc[0] for mc in model_cells]

    without_by_model: dict[str, dict[str, Any]] = {}
    current_with_by_model: dict[str, dict[str, Any]] = {}
    judge_by_model: dict[str, str] = {}
    baseline_exec_by_model: dict[str, str | None] = {}

    for m, wo_dir, w_dir, wo_ex, w_ex in model_cells:
        wo_sum = summarize_arm_runs(tasks, wo_dir, k=k, executor=wo_ex or m)
        w_sum = summarize_arm_runs(tasks, w_dir, k=k, executor=w_ex or m)
        without_by_model[m] = wo_sum
        current_with_by_model[m] = w_sum
        judge_by_model[m] = _resolve_judge_for_probe(m, wo_dir, w_dir, tasks, k, resolved_suite)
        baseline_exec_by_model[m] = wo_ex or w_ex

    # Check if the skill is flagged `needs harder tasks` for any chosen executor.
    report_json = _read_json(runs_dir / "report.json")
    for m in models_list:
        wo_sum = without_by_model[m]
        flagged, reason = False, ""
        if report_json:
            flagged, reason = _report_skill_ceiling(report_json, skill, executor=m)
        if not flagged and wo_sum.get("needs_harder_tasks"):
            c_cnt = wo_sum.get("ceiling_tasks")
            t_cnt = wo_sum.get("total_tasks")
            flagged = True
            reason = (
                f"skill {skill!r} is flagged '{NEEDS_HARDER_TASKS}' for executor {m!r} "
                f"({c_cnt}/{t_cnt} tasks at ceiling, > 50%)"
            )
        if flagged:
            print(f"skipping {skill}: {reason}", flush=True)
            primary = models_list[0]
            skipped_gates = [
                {
                    "executor": mod,
                    "judge": judge_by_model[mod],
                    "without_executor": without_by_model[mod].get("executor") or mod,
                    "current_with_executor": current_with_by_model[mod].get("executor") or mod,
                    "candidate_executor": mod,
                    "accepted": False,
                    "skipped": True,
                    "decision_reason": reason,
                    "without": without_by_model[mod],
                    "current_with": current_with_by_model[mod],
                    "candidate": current_with_by_model[mod],
                }
                for mod in models_list
            ]
            skipped_rec: dict[str, Any] = {
                "skill": skill,
                "skipped": True,
                "skip_reason": reason,
                "accepted": False,
                "decision_reason": reason,
                "best_round": 0,
                "models": models_list,
                "executors": {
                    "without": without_by_model[primary].get("executor") or primary,
                    "current_with": current_with_by_model[primary].get("executor") or primary,
                    "candidate": primary,
                },
                "diagnosis": "",
                "changes_summary": "",
                "diff": "",
                "diff_stat": patch_diff_stat(""),
                "diff_stat_str": format_diff_stat(""),
                "applied_to": None,
                "without": without_by_model[primary],
                "current_with": current_with_by_model[primary],
                "best_candidate": current_with_by_model[primary],
                "by_model": {g["executor"]: g for g in skipped_gates},
                "model_gates": skipped_gates,
                "rounds": [],
            }
            skipped_rec["decision_line"] = format_decision_line(
                skipped_rec, fallback_exec=primary
            )
            return skipped_rec

    skill_out = runs_dir / "improve" / skill
    skill_out.mkdir(parents=True, exist_ok=True)

    imp_model = improver_model(improver_model_name)
    round_records: list[dict[str, Any]] = []
    best_round = 1
    best_rec: dict[str, Any] | None = None
    best_proposal: dict[str, Any] | None = None
    working_md = original_md
    primary_model = models_list[0]

    for r in range(1, max(1, rounds) + 1):
        rdir = skill_out / f"round_{r}"
        rdir.mkdir(parents=True, exist_ok=True)
        evidence = build_evidence_packet(
            skill=skill,
            current_skill_md=working_md,
            tasks=tasks,
            runs_dir=runs_dir,
            k=k,
            include_verdicts=include_verdicts,
            prior_rounds=round_records,
            executor=primary_model,
        )
        (rdir / "evidence.md").write_text(evidence, encoding="utf-8")

        cand_path = rdir / "SKILL.md"
        prop_path = rdir / "proposal.json"
        cached_prop = _read_json(prop_path)
        if (
            cand_path.is_file()
            and cached_prop.get("diagnosis")
            and cached_prop.get("changes_summary")
            and cached_prop.get("skill_markdown")
            and not validate_candidate_skill(
                cached_prop["skill_markdown"],
                target_skill=skill,
                original_md=original_md,
                banned_tools=resolved_suite.banned_tools,
                max_growth=max_growth,
            )
        ):
            proposal = cached_prop
        else:
            proposal = propose_skill(
                skill=skill,
                original_md=original_md,
                evidence_packet=evidence,
                model=imp_model,
                suite=resolved_suite,
                max_growth=max_growth,
            )
            cand_path.write_text(proposal["skill_markdown"], encoding="utf-8")
            (rdir / "proposal.json").write_text(
                json.dumps(proposal, indent=2) + "\n", encoding="utf-8"
            )
        patch_txt = unified_skill_diff(original_md, proposal["skill_markdown"], skill=skill)
        (rdir / "skill.patch").write_text(patch_txt, encoding="utf-8")

        val_errs = validate_candidate_skill(
            proposal["skill_markdown"],
            target_skill=skill,
            original_md=original_md,
            banned_tools=resolved_suite.banned_tools,
            max_growth=max_growth,
        )

        model_gates: list[dict[str, Any]] = []
        cand_by_model: dict[str, dict[str, Any]] = {}
        for m in models_list:
            out_res_dir = (
                rdir / "results"
                if len(models_list) == 1
                else rdir / "results" / _path_segment(m)
            )
            cand_m = _call_probe_candidate(
                skill=skill,
                candidate_skill_path=cand_path,
                tasks=tasks,
                out_results_dir=out_res_dir,
                k=k,
                root=root,
                executor=m,
                judge=judge_by_model[m],
                baseline_executor=baseline_exec_by_model[m],
                allow_model_mismatch=allow_model_mismatch,
            )
            cand_by_model[m] = cand_m
            if val_errs:
                acc_m = False
                rsn_m = f"candidate validation failed: {'; '.join(val_errs)}"
            else:
                acc_m, rsn_m = evaluate_acceptance(current_with_by_model[m], cand_m)
            model_gates.append(
                {
                    "executor": m,
                    "judge": judge_by_model[m],
                    "without_executor": without_by_model[m].get("executor") or m,
                    "current_with_executor": current_with_by_model[m].get("executor") or m,
                    "candidate_executor": cand_m.get("executor") or m,
                    "accepted": acc_m,
                    "decision_reason": rsn_m,
                    "without": without_by_model[m],
                    "current_with": current_with_by_model[m],
                    "candidate": cand_m,
                }
            )

        accepted = bool(model_gates) and all(g["accepted"] for g in model_gates)
        if len(model_gates) == 1:
            reason = model_gates[0]["decision_reason"]
        else:
            parts = [
                f"{g['executor']}: {'ACCEPTED' if g['accepted'] else 'REJECTED'} ({g['decision_reason']})"
                for g in model_gates
            ]
            reason = "; ".join(parts)

        primary_cand = cand_by_model[primary_model]
        primary_wo = without_by_model[primary_model]
        primary_cur = current_with_by_model[primary_model]
        rec: dict[str, Any] = {
            "round": r,
            "accepted": accepted,
            "decision_reason": reason,
            "diagnosis": proposal["diagnosis"],
            "changes_summary": proposal["changes_summary"],
            "skill_markdown": proposal["skill_markdown"],
            "mean_without": primary_wo.get("mean"),
            "mean_current_with": primary_cur.get("mean"),
            "mean_candidate": primary_cand.get("mean"),
            "verify_without": (primary_wo.get("verify") or {}).get("rate"),
            "verify_current_with": (primary_cur.get("verify") or {}).get("rate"),
            "verify_candidate": (primary_cand.get("verify") or {}).get("rate"),
            "tokens_without": primary_wo.get("tokens_median"),
            "tokens_current_with": primary_cur.get("tokens_median"),
            "tokens_median": primary_cand.get("tokens_median"),
            "wall_without_s": primary_wo.get("t_total_median_s"),
            "wall_current_with_s": primary_cur.get("t_total_median_s"),
            "t_total_median_s": primary_cand.get("t_total_median_s"),
            "without": primary_wo,
            "current_with": primary_cur,
            "candidate": primary_cand,
            "by_model": {g["executor"]: g for g in model_gates},
            "model_gates": model_gates,
            "diff": patch_txt,
        }
        round_records.append(rec)

        if best_rec is None or (
            bool(rec["accepted"]),
            *round_rank_key(rec),
        ) > (
            bool(best_rec["accepted"]),
            *round_rank_key(best_rec),
        ):
            best_round = r
            best_rec = rec
            best_proposal = proposal

        # Early stop as soon as a round is accepted on every requested executor.
        if accepted:
            break
        working_md = proposal["skill_markdown"]

    assert best_rec is not None and best_proposal is not None
    best_summary = best_rec["candidate"]
    best_accepted = bool(best_rec["accepted"])
    best_reason = str(best_rec["decision_reason"])
    best_gates = best_rec["model_gates"]
    best_by_model = best_rec["by_model"]
    best_wo = without_by_model[primary_model]
    best_cur = current_with_by_model[primary_model]

    best_diff = unified_skill_diff(original_md, best_proposal["skill_markdown"], skill=skill)
    best_patch_path = skill_out / "best.patch"
    (skill_out / "best_SKILL.md").write_text(best_proposal["skill_markdown"], encoding="utf-8")
    best_patch_path.write_text(best_diff, encoding="utf-8")

    result_rec: dict[str, Any] = {
        "skill": skill,
        "skipped": False,
        "skip_reason": None,
        "accepted": best_accepted,
        "decision_reason": best_reason,
        "best_round": best_round,
        "models": models_list,
        "executors": {
            "without": best_wo.get("executor") or primary_model,
            "current_with": best_cur.get("executor") or primary_model,
            "candidate": best_summary.get("executor") or primary_model,
        },
        "diagnosis": best_proposal["diagnosis"],
        "changes_summary": best_proposal["changes_summary"],
        "diff": best_diff,
        "diff_stat": patch_diff_stat(best_diff),
        "diff_stat_str": format_diff_stat(best_diff),
        "applied_to": None,
        "without": best_wo,
        "current_with": best_cur,
        "best_candidate": best_summary,
        "by_model": best_by_model,
        "model_gates": best_gates,
        "rounds": round_records,
    }
    result_rec["decision_line"] = format_decision_line(
        result_rec, fallback_exec=primary_model
    )

    if apply_accepted and best_accepted:
        apply_info = apply_candidate_patch(
            skill=skill,
            patch_path=best_patch_path,
            skill_record=result_rec,
            suite=resolved_suite,
            skill_path=skill_path,
            date=date,
            dry_run=False,
        )
    else:
        apply_info = apply_candidate_patch(
            skill=skill,
            patch_path=best_patch_path,
            skill_record=result_rec,
            suite=resolved_suite,
            skill_path=skill_path,
            date=date,
            dry_run=True,
        )
    result_rec["applied_to"] = apply_info["applied_to"]
    result_rec["apply"] = apply_info
    result_rec["branch"] = apply_info["branch"]
    result_rec["commit_message"] = apply_info["commit_message"]
    result_rec["apply_commands"] = apply_info["commands"]
    return result_rec


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="skillbench-improve",
        description="Propose, probe, and verify improvements for weak skills of a plugin suite.",
    )
    C.add_suite_arg(ap)
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="skillbench runs directory (default: $HARNESS2_RUNS_DIR/skillbench)",
    )
    ap.add_argument(
        "--skill",
        default="",
        help="comma-separated skill name(s) to force improvement on (overrides summary selection)",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="select weak skills from report.json summary[] (default: fix + consider removing)",
    )
    ap.add_argument(
        "--all-weak",
        action="store_true",
        help=(
            "when selecting weak skills from summary[], also include inconclusive skills "
            "that have token or wall-time cost overhead"
        ),
    )
    ap.add_argument(
        "--models",
        default=None,
        help=(
            "comma-separated executor model IDs to probe against (reusing matrix cells for "
            "WITHOUT and Current WITH; gate must pass on every model; default: baseline cell executor)"
        ),
    )
    ap.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="allow probing with an executor model different from the one recorded in the baseline cell",
    )
    ap.add_argument(
        "--max-growth",
        type=float,
        default=DEFAULT_MAX_GROWTH_PCT,
        help=(
            "maximum allowed candidate SKILL.md size growth in percent over the original "
            "(default: 50; prevents prompt bloat that inflates token cost and latency "
            "on every rollout that reads the skill)"
        ),
    )
    ap.add_argument("--k", type=int, default=3, help="reps per task (default: 3)")
    ap.add_argument("--rounds", type=int, default=1, help="max propose-probe rounds (default: 1)")
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="restrict tasks to pilot = true",
    )
    ap.add_argument(
        "--include-verdicts",
        action="store_true",
        help="include failed rubric criteria from judgments.json in the improver evidence",
    )
    ap.add_argument(
        "--improver-model",
        default=None,
        help="override $SKILLBENCH_IMPROVER_MODEL for proposing candidate SKILL.md",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help=(
            "apply best.patch onto branch skillbench/improve-<skill>-<date> of the "
            "realpath-resolved plugins_dir git checkout and commit with 3-way numbers "
            "(requires clean git status; never pushes; dry-run printing the commands is the default)"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write evidence packet and print selected skills and apply commands without calling agy",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    C.apply_suite(args.suite)

    runs_dir = (
        args.runs_dir.expanduser().resolve() if args.runs_dir else _default_runs_root()
    )
    req_models = _parse_models(args.models)
    primary_exec = req_models[0] if req_models else None

    skills = [s.strip() for s in args.skill.split(",") if s.strip()]
    if not skills and (args.auto or args.all_weak or (runs_dir / "report.json").is_file()):
        report_json = runs_dir / "report.json"
        try:
            rep = (
                json.loads(report_json.read_text(encoding="utf-8"))
                if report_json.is_file()
                else build_report(runs_dir, k=args.k, pilot=args.pilot or None)
            )
            skills = select_weak_skills(
                rep, executor=primary_exec, all_weak=bool(args.all_weak)
            )
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if not skills:
        print(
            "error: specify --skill <name> or --auto (with an existing skillbench run)",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        all_tasks = load_tasks(tasks_root())
        pilot_only = bool(args.pilot or _env_pilot())
        resolved_suite = active_suite()
        for sk in skills:
            tlist = [
                t
                for t in all_tasks.values()
                if t.target_skill == sk and (not pilot_only or t.pilot)
            ]
            sk_path = target_skill_file(sk, resolved_suite)
            md = sk_path.read_text(encoding="utf-8")
            ev = build_evidence_packet(
                skill=sk,
                current_skill_md=md,
                tasks=tlist,
                runs_dir=runs_dir,
                k=args.k,
                include_verdicts=args.include_verdicts,
                executor=primary_exec,
            )
            out_p = runs_dir / "improve" / sk / "round_1" / "evidence.md"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(ev, encoding="utf-8")
            print(f"[dry-run] {sk}: {len(tlist)} task(s) -> {out_p} ({len(ev)} chars)")
            patch_p = runs_dir / "improve" / sk / "best.patch"
            apply_candidate_patch(
                skill=sk,
                patch_path=patch_p,
                skill_record={
                    "skill": sk,
                    "accepted": True,
                    "decision_reason": "dry-run preview",
                    "models": list(req_models) if req_models else [primary_exec or exec_model()],
                },
                suite=resolved_suite,
                skill_path=sk_path,
                dry_run=True,
            )
        return 0

    results: dict[str, Any] = {}
    recorded_models: list[str] = []
    try:
        for sk in skills:
            print(f"improving skill {sk!r} (k={args.k}, rounds={args.rounds})...", flush=True)
            res = improve_skill(
                sk,
                runs_dir=runs_dir,
                k=args.k,
                rounds=args.rounds,
                pilot=args.pilot or None,
                include_verdicts=args.include_verdicts,
                apply_accepted=args.apply,
                improver_model_name=args.improver_model,
                models=req_models or None,
                allow_model_mismatch=args.allow_model_mismatch,
                max_growth=args.max_growth,
            )
            results[sk] = res
            for m in res.get("models") or []:
                if m not in recorded_models:
                    recorded_models.append(m)
            status = (
                "SKIPPED"
                if res.get("skipped")
                else ("ACCEPTED" if res["accepted"] else "REJECTED")
            )
            print(f"  [{sk}] {status}: {res['decision_line']}", flush=True)
    except (ValueError, RuntimeError, config.ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = runs_dir / "improve"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_exec = recorded_models[0] if recorded_models else exec_model()
    record = {
        "improver_model": improver_model(args.improver_model),
        "executor_model": rep_exec,
        "models": recorded_models or [rep_exec],
        "judge_model": judge_model(),
        "include_verdicts": bool(args.include_verdicts),
        "max_growth": args.max_growth,
        "k": args.k,
        "rounds": args.rounds,
        "skills": results,
    }
    (out_dir / "improve_report.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "improve_report.md").write_text(
        render_improve_markdown(record), encoding="utf-8"
    )
    (out_dir / "improve_report.html").write_text(
        render_improve_html(record), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'improve_report.json'}")
    print(f"wrote {out_dir / 'improve_report.md'}")
    print(f"wrote {out_dir / 'improve_report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
