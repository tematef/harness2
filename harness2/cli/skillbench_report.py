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

Reads the `with` and `without` arm namespaces produced by `harness2 --bench skillbench` and
`harness2-judge --bench skillbench --bases`:

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
- `report.html`: the same data as a self-contained dashboard (KPI cards, per-skill and per-task
  tables, criteria pass rates, side-by-side WITH / WITHOUT answers, diagnostics).
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
    _default_runs_root,
    _env_pilot,
    check_manifest,
    diff_stats,
    judge_model,
)
from ..core import schedule as SCHED
from ..substrate.agy import DEFAULT_EXEC_MODEL, exec_model
from . import _common as C
from .skillbench_html import render_html

DEFAULT_BOOTSTRAP_DRAWS = 10_000
DEFAULT_BOOTSTRAP_SEED = 0
MIN_TASKS_FULL = 3
MIN_TASKS_PILOT = 2
MIN_VALID_REPS_PER_ARM = 2

MODEL_UNVERIFIED = "model unverified (agy reports no model)"
POWER_NOTE = (
    "With 4 tasks per skill, the task-level bootstrap has only 35 distinct resamples, so only "
    "large lifts (roughly >= 0.2 of the rubric) will get a `helps`/`hurts` verdict; most skills "
    "are expected to land `inconclusive` in the first full run."
)

_IDENTITY_KEYS = ("executor", "judge", "effort", "manifest_hash")
FINAL_EXCERPT_CHARS = 6000


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
            "frac": self.frac,
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
    return n_passed / n_criteria


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        raw_judged = _read_json(judged_path)
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

    cfg: dict[str, Any] = {}
    cfg_path = ns_dir / "config.json"
    if cfg_path.is_file():
        try:
            loaded = _read_json(cfg_path)
            if isinstance(loaded, dict):
                cfg = loaded
        except (OSError, json.JSONDecodeError):
            pass

    records: list[_RepRecord] = []
    for summary_path in sorted(ns_dir.glob("*/summary.json")):
        try:
            summary = _read_json(summary_path)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"{summary_path}: unreadable JSON ({e})") from None
        if not isinstance(summary, Mapping):
            continue
        task_id = str(summary.get("task_id") or "")
        if not task_id:
            continue
        skill = str(
            summary.get("domain") or (task_id.split("/", 1)[0] if "/" in task_id else task_id)
        )
        for entry in summary.get("base") or []:
            if not isinstance(entry, Mapping):
                continue
            label = str(entry.get("label") or "")
            run_id = str(entry.get("run_id") or "")
            raw_run_dir = entry.get("run_dir")
            run_dir = Path(raw_run_dir) if isinstance(raw_run_dir, str) and raw_run_dir else None
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

    if run_dir is not None:
        timing_path = run_dir / SKILL_TIMING_FILE
        if not timing_path.is_file():
            valid = False
            invalid_reason = "missing skill_timing.json"
        else:
            try:
                timing = _read_json(timing_path)
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

        v_path = run_dir / VERIFY_FILE
        if v_path.is_file():
            try:
                v_data = _read_json(v_path)
            except (OSError, json.JSONDecodeError):
                v_data = None
            if isinstance(v_data, Mapping) and isinstance(v_data.get("passed"), bool):
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
                j_data = _read_json(j_path)
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
    )


def _median(vals: Iterable[float | int | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return float(st.median(clean)) if clean else None


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


def _check_arm_configs(with_cfg: Mapping[str, Any], without_cfg: Mapping[str, Any]) -> None:
    if not with_cfg or not without_cfg:
        return
    for key in _IDENTITY_KEYS:
        v_with = with_cfg.get(key)
        v_without = without_cfg.get(key)
        if v_with is not None and v_without is not None and v_with != v_without:
            raise ValueError(
                f"WITH and WITHOUT namespaces disagree on {key}: {v_with!r} vs {v_without!r}"
            )


def build_report(
    with_ns_dir: Path,
    without_ns_dir: Path,
    *,
    k: int | None = None,
    seed: int | None = None,
    bootstrap_seed: int | None = None,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    pilot: bool | None = None,
) -> dict[str, Any]:
    """Compute the full skillbench report dictionary from the `with` and `without` namespaces."""
    with_cfg, with_recs = _load_arm_records(with_ns_dir)
    without_cfg, without_recs = _load_arm_records(without_ns_dir)
    _check_arm_configs(with_cfg, without_cfg)

    is_pilot = (
        bool(pilot)
        if pilot is not None
        else _env_pilot()
    )
    eff_seed = (
        seed
        if seed is not None
        else (
            with_cfg.get("seed")
            if with_cfg.get("seed") is not None
            else without_cfg.get("seed")
        )
    )
    eff_boot_seed = (
        int(bootstrap_seed)
        if bootstrap_seed is not None
        else (int(eff_seed) if eff_seed is not None else DEFAULT_BOOTSTRAP_SEED)
    )

    all_recs = with_recs + without_recs
    if k is None:
        counts_per_task: dict[str, int] = {}
        for r in with_recs:
            counts_per_task[r.task_id] = counts_per_task.get(r.task_id, 0) + 1
        eff_k = max(counts_per_task.values(), default=1)
    else:
        eff_k = int(k)

    manifest_hash = with_cfg.get("manifest_hash") or without_cfg.get("manifest_hash")
    if not manifest_hash:
        try:
            manifest_hash = check_manifest()
        except ManifestError:
            manifest_hash = "unverified"

    req_exec = (
        with_cfg.get("executor")
        or without_cfg.get("executor")
        or next((r.requested_model for r in all_recs if r.requested_model), None)
        or exec_model()
    )
    req_judge = (
        with_cfg.get("judge")
        or without_cfg.get("judge")
        or next((r.judge_model for r in all_recs if r.judge_model), None)
        or judge_model()
    )

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

    skills = sorted({r.skill for r in all_recs})
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
            tasks_detail[tid] = {
                "with_fracs": w_fracs,
                "without_fracs": u_fracs,
                "with_mean": w_mean,
                "without_mean": u_mean,
                "delta": delta,
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
            "tokens_with_median": _median(r.total_tokens for r in valid_with),
            "tokens_without_median": _median(r.total_tokens for r in valid_without),
            "t_total_with_median": _median(r.t_total_s for r in valid_with),
            "t_total_without_median": _median(r.t_total_s for r in valid_without),
            "task_deltas": task_deltas,
            "tasks": tasks_detail,
        }
        skill_rows.append(row)
        by_skill[skill] = row

    return {
        "k": eff_k,
        "seed": eff_seed,
        "bootstrap_seed": eff_boot_seed,
        "bootstrap_draws": draws,
        "pilot": is_pilot,
        "manifest_hash": manifest_hash,
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
        "invalid_total": overall_invalid_total,
        "invalid_by_reason": dict(sorted(overall_invalid_by_reason.items())),
        "power_note": POWER_NOTE,
        "skills": skill_rows,
        "by_skill": by_skill,
    }


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


def render_markdown(report: Mapping[str, Any]) -> str:
    """Format `report` as `report.md` (header metadata, power note, and per-skill table)."""
    req_exec = report.get("requested_exec_model") or DEFAULT_EXEC_MODEL
    act_exec = report.get("actual_exec_model")
    exec_str = (
        f"requested `{req_exec}`, actual `{act_exec}`"
        if act_exec
        else f"requested `{req_exec}`, actual: {MODEL_UNVERIFIED}"
    )

    req_judge = report.get("requested_judge_model") or DEFAULT_JUDGE_MODEL
    act_judge = report.get("actual_judge_model")
    judge_str = (
        f"requested `{req_judge}`, actual `{act_judge}`"
        if act_judge
        else f"requested `{req_judge}`, actual: {MODEL_UNVERIFIED}"
    )

    noise_rate = report.get("judge_noise_rate")
    if noise_rate is None:
        noise_str = "n/a (single judgment per rollout)"
    else:
        dis = report.get("judge_noise_disagreed", 0)
        tot = report.get("judge_noise_total", 0)
        noise_str = f"{noise_rate * 100:.1f}% ({dis}/{tot} criteria disagreed)"

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

    lines = [
        "# skillbench report",
        "",
        f"- **Executor model**: {exec_str}",
        f"- **Judge model**: {judge_str}",
        f"- **Manifest hash**: `{manifest_hash}`",
        f"- **Run scale / seed**: `k={k}`, `seed={seed_str}` "
        f"(bootstrap `seed={boot_seed}`, `{draws}` draws, mode `{mode_label}`, "
        f"min `{min_tasks}` tasks with >= `{MIN_VALID_REPS_PER_ARM}` valid reps/arm)",
        f"- **Judge-noise rate**: {noise_str}",
        f"- **Statistical power**: {power_note}",
        "",
        "| skill | tasks | WITH | WITHOUT | lift [95% CI] | verdict | invoked | t_skill p50/p90 | invalid |",
        "| :--- | ---: | ---: | ---: | :---: | :---: | :---: | :---: | :--- |",
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
                        row.get("diff_lines_with_median"), row.get("diff_lines_without_median")
                    ),
                    _fmt_pair(row.get("tokens_with_median"), row.get("tokens_without_median")),
                    _fmt_pair(
                        row.get("t_total_with_median"),
                        row.get("t_total_without_median"),
                        suffix="s",
                    ),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


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


def write_report(report: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path, Path]:
    """Write `report.json`, `report.md` and `report.html` inside `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    html_path = out_dir / "report.html"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return (json_path, md_path, html_path)


def _resolve_ns_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.with_dir is not None or args.without_dir is not None:
        if args.with_dir is None or args.without_dir is None:
            raise ValueError("pass both --with-dir and --without-dir, or neither")
        return (Path(args.with_dir), Path(args.without_dir))

    with_ns = SCHED.ns("skillbench", "agy", "agg", args.k, seed=args.seed, tag="with")
    without_ns = SCHED.ns("skillbench", "agy", "agg", args.k, seed=args.seed, tag="without")
    base = Path(args.runs_root) if args.runs_root is not None else _default_runs_root()
    adapt = base / "adapt"
    return (adapt / with_ns, adapt / without_ns)


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
        "--out-dir",
        type=Path,
        default=None,
        help="directory where report.json, report.md and report.html are written",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.k < 1:
            raise ValueError("--k must be positive")
        with_dir, without_dir = _resolve_ns_dirs(args)
        report = build_report(
            with_dir,
            without_dir,
            k=args.k,
            seed=args.seed,
            bootstrap_seed=args.bootstrap_seed,
            draws=args.draws,
            pilot=True if args.pilot else None,
        )
        out_dir = (
            args.out_dir
            if args.out_dir is not None
            else (with_dir.parent.parent if with_dir.parent.name == "adapt" else with_dir.parent)
        )
        json_path, md_path, html_path = write_report(report, out_dir)
        print(md_path.read_text(encoding="utf-8"), end="")
        print(f"wrote {md_path}, {json_path} and {html_path}")
        return 0
    except (FileNotFoundError, ValueError, config.ConfigError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
