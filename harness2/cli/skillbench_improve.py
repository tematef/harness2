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

Given an existing `skillbench` run directory (`results/with/` and `results/without/`), inspects
weak or high-overhead skills, builds a contrastive evidence packet from the `WITH` and `WITHOUT`
trajectories and deliverables (without leaking `rubric.json` or judge verdicts unless
`--include-verdicts` is set), asks an isolated `agy` improver session for a revised `SKILL.md`,
validates the candidate skill, probes it across the skill's tasks (`k` reps), runs deterministic
`verify` and the blinded judge, and emits a 3-way comparison (`WITHOUT` vs `Current WITH` vs
`Candidate WITH`) in `improve_report.{json,md,html}` plus a unified diff (`best.patch`).
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import config
from ..benchmarks.skillbench import (
    DEFAULT_JUDGE_TIMEOUT_S,
    JUDGMENTS_FILE,
    SKILL_TIMING_FILE,
    TASKS_ROOT,
    VERIFY_FILE,
    SkillbenchAdapter,
    SkillbenchTask,
    _default_runs_root,
    _env_pilot,
    build_prompt,
    deliverable_text,
    diff_stats,
    load_tasks,
    rollout_valid,
    score,
)
from ..substrate.agy import (
    _kill_group,
    agy_argv,
    agy_bin_missing_message,
    auth_files,
    exec_effort,
    exec_model,
    model_guard,
    parse_event,
    read_events,
    render_trajectory,
    resolve_agy_bin,
    run_rollout,
    target_skill_file,
)
from .skillbench_report import build_report


def safe_tid(task_id: str) -> str:
    """Filesystem-safe single-segment identifier for `<skill>/<task-id>`."""
    return task_id.replace("/", "__")


def resolve_rep_dir(results_dir: Path, task_id: str, rep_idx: int) -> Path:
    """Locate `<task_id>/base<rep_idx>` under `results_dir` (direct, flat, or namespaced)."""
    direct = results_dir / task_id / f"base{rep_idx}"
    if direct.is_dir():
        return direct
    flat = results_dir / f"{safe_tid(task_id)}__b{rep_idx}"
    if flat.is_dir():
        return flat
    for ns_dir in sorted(results_dir.glob("skillbench-*")):
        cand = ns_dir / task_id / f"base{rep_idx}"
        if cand.is_dir():
            return cand
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

BANNED_CLAUDE_TOOLS = (
    "AskUserQuestion",
    "EnterPlanMode",
    "TodoWrite",
)

EVIDENCE_TRAJ_CHARS = 1800
EVIDENCE_DELIV_CHARS = 1800
MAX_EVIDENCE_CHARS = 70_000
EFFICIENCY_GAIN_RATIO = 0.85


def improver_model(override: str | None = None) -> str:
    """Model used to propose candidate `SKILL.md` edits."""
    if override and override.strip():
        return override.strip()
    raw = config.env("SKILLBENCH_IMPROVER_MODEL")
    if raw and str(raw).strip():
        return str(raw).strip()
    return exec_model()


def select_weak_skills(report: Mapping[str, Any]) -> list[str]:
    """Identify skills from a `skillbench` `report.json` that can benefit from improvement.

    A skill is selected when any of the following holds:
    - `with_mean < 1.0` (missed rubric criteria)
    - `lift <= 0.0` (no positive lift over WITHOUT)
    - `verify_with_rate < 1.0` (deterministic test failure)
    - `tokens_with_median > 1.25 * tokens_without_median` or
      `t_total_with_median > 1.25 * t_total_without_median` (excessive overhead)
    """
    weak: list[str] = []
    raw_skills = report.get("skills") or {}
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

    for skill, s in sorted(items, key=lambda pair: pair[0]):
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
        if is_weak:
            weak.append(skill)
    return weak


def validate_candidate_skill(
    candidate_md: str,
    *,
    target_skill: str,
    original_md: str,
) -> list[str]:
    """Validate a proposed `SKILL.md` before staging it. Returns a list of errors (empty = valid)."""
    errors: list[str] = []
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

    for banned in BANNED_CLAUDE_TOOLS:
        if re.search(rf"\b{re.escape(banned)}\b", text):
            errors.append(f"contains banned Claude Code tool name {banned!r} (use AGY tool names)")

    if text == original_md.strip():
        errors.append("candidate SKILL.md is identical to the current SKILL.md")

    return errors


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


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
    dstats = diff_stats(run_dir)
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
        parts.append(
            f"- deterministic verify: passed={verify.get('passed')} "
            f"(rc={verify.get('returncode')}, command={verify.get('command')!r})"
        )
        v_err = str(verify.get("stderr") or verify.get("stdout") or "").strip()
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
                            f"  - FAILED criterion `{c.get('id')}`: {c.get('rationale')}"
                        )

    parts.append("<trajectory>")
    parts.append(traj)
    parts.append("</trajectory>")
    parts.append("<deliverable>")
    parts.append(deliv)
    parts.append("</deliverable>")
    return "\n".join(parts) + "\n"


def build_evidence_packet(
    *,
    skill: str,
    current_skill_md: str,
    tasks: Sequence[SkillbenchTask],
    runs_dir: Path,
    k: int,
    include_verdicts: bool = False,
    prior_rounds: Sequence[Mapping[str, Any]] = (),
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
            arm_root = runs_dir / "results" / arm
            for i in range(1, k + 1):
                rdir = resolve_rep_dir(arm_root, task.task_id, i)
                lines.append(
                    _format_rep_evidence(
                        rdir,
                        f"arm={arm.upper()} rep=b{i}",
                        include_verdicts=include_verdicts,
                    )
                )

    if prior_rounds:
        lines.append("## Prior Improvement Rounds in This Session")
        for pr in prior_rounds:
            lines.append(
                f"- Round {pr.get('round')}: accepted={pr.get('accepted')} | "
                f"reason={pr.get('decision_reason')} | "
                f"mean_score={pr.get('mean_candidate')} | "
                f"tokens_median={pr.get('tokens_median')} | "
                f"wall_median_s={pr.get('t_total_median_s')}"
            )
            if pr.get("changes_summary"):
                lines.append(f"  summary: {pr.get('changes_summary')}")
        lines.append("")

    out = "\n".join(lines)
    if len(out) > MAX_EVIDENCE_CHARS:
        out = out[:MAX_EVIDENCE_CHARS] + "\n\n...[evidence packet truncated to fit CLI limit]...\n"
    return out


def build_improver_prompt(
    *,
    skill: str,
    evidence_packet: str,
    validation_error: str = "",
) -> str:
    """Build the single-turn prompt sent to the isolated `agy` improver."""
    retry_block = (
        f"\n## Previous Candidate Was Rejected\nFix these validation errors:\n{validation_error}\n"
        if validation_error
        else ""
    )
    return (
        "You are an expert Agent Harness Engineer optimizing a `SKILL.md` file in the `cc-thingz` "
        "plugin suite for Google Antigravity (AGY) / Jetski.\n\n"
        "Do not call any tools or inspect any files; analyze the Evidence Packet below and return "
        "your structured JSON output immediately via `finish`.\n\n"
        "## Optimization Rules\n"
        f"1. Keep YAML frontmatter valid with `name: {skill}` and a clear `description:`.\n"
        "2. If the current `SKILL.md` has a `## When no one can answer` section, preserve that "
        "section heading and its headless fallback behavior.\n"
        "3. Use only AGY/Jetski tool names (`view_file`, `write_to_file`, `replace_file_content`, "
        "`find_by_name`, `grep_search`, `run_command`, `invoke_subagent`, `ask_question`). Never "
        "use Claude Code tool names (`AskUserQuestion`, `EnterPlanMode`, `TodoWrite`).\n"
        "4. Fix any quality gaps where `WITH` missed root causes, edge cases, or verification checks.\n"
        "5. Streamline instructions that caused unnecessary tool calls, redundant searches for "
        "other skills in headless/leave-one-out runs, or high token/time overhead.\n"
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
) -> dict[str, Any]:
    """Propose and validate a candidate `SKILL.md` (with up to `max_attempts` retries)."""
    last_err = ""
    for attempt in range(1, max_attempts + 1):
        prompt = build_improver_prompt(
            skill=skill,
            evidence_packet=evidence_packet,
            validation_error=last_err,
        )
        res = run_improver_once(prompt, model=model, timeout=timeout)
        if res is None:
            last_err = f"attempt {attempt}: agy improver returned no valid structured output"
            continue
        errs = validate_candidate_skill(
            res["skill_markdown"],
            target_skill=skill,
            original_md=original_md,
        )
        if errs:
            last_err = "; ".join(errs)
            continue
        res["attempts"] = attempt
        return res
    raise RuntimeError(f"failed to propose a valid SKILL.md for {skill!r}: {last_err}")


def unified_skill_diff(original_md: str, candidate_md: str, *, skill: str) -> str:
    """Unified diff between original and candidate `SKILL.md`."""
    return "".join(
        difflib.unified_diff(
            original_md.splitlines(keepends=True),
            candidate_md.splitlines(keepends=True),
            fromfile=f"a/skills/{skill}/SKILL.md",
            tofile=f"b/skills/{skill}/SKILL.md",
        )
    )


def summarize_arm_runs(
    tasks: Sequence[SkillbenchTask],
    results_dir: Path,
    *,
    k: int,
) -> dict[str, Any]:
    """Compute aggregate metrics over `<results_dir>/<safe_tid>__b1..bk` for `tasks`."""
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

    for task in tasks:
        t_fracs: list[float] = []
        t_reps: list[dict[str, Any]] = []
        for i in range(1, k + 1):
            total_reps += 1
            rdir = resolve_rep_dir(results_dir, task.task_id, i)
            timing = _read_json(rdir / SKILL_TIMING_FILE)
            judg = _read_json(rdir / JUDGMENTS_FILE)
            verify = _read_json(rdir / VERIFY_FILE)
            dstats = diff_stats(rdir) if rdir.is_dir() else {"files": 0, "added": 0, "removed": 0}
            valid = bool(timing.get("valid"))
            if not valid and timing.get("invalid_reason"):
                invalid_reasons.append(str(timing["invalid_reason"]))
            frac: float | None = None
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
                if isinstance(j_list, list) and j_list and isinstance(j_list[0], dict):
                    first = j_list[0]
                    np = first.get("n_passed")
                    nc = first.get("n_criteria")
                    if isinstance(np, int) and isinstance(nc, int) and nc > 0:
                        frac = np / nc
                        t_fracs.append(frac)
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
                    "verify_passed": verify.get("passed") if verify else None,
                    "diff_lines": dstats["added"] + dstats["removed"],
                }
            )
        t_mean = st.fmean(t_fracs) if t_fracs else None
        if t_mean is not None:
            task_means[task.task_id] = t_mean
        per_task[task.task_id] = {
            "mean": t_mean,
            "n_valid": sum(1 for r in t_reps if r["valid"]),
            "reps": t_reps,
        }

    mean_score = st.fmean(task_means.values()) if task_means else None
    return {
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
    root: Path = TASKS_ROOT,
) -> dict[str, Any]:
    """Stage `candidate_skill_path`, run all `tasks` for `k` reps, and score each rollout."""
    adapter = SkillbenchAdapter(arm="with", root=root)
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

    return summarize_arm_runs(tasks, out_results_dir, k=k)


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


def _fmt_num(v: Any, fmt: str = ".2f", suffix: str = "") -> str:
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        return "—"
    if fmt == "int":
        return f"{int(round(v)):,}{suffix}"
    if fmt == "pct":
        return f"{v * 100:.0f}%"
    return f"{v:{fmt}}{suffix}"


def render_improve_markdown(record: Mapping[str, Any]) -> str:
    """Render `improve_report.md` for one or more improved skills."""
    lines = [
        "# skillbench Skill Improvement Report (3-Way Comparison)",
        "",
        f"- **Improver model**: `{record.get('improver_model')}`",
        f"- **Executor model**: `{record.get('executor_model')}`",
        f"- **Judge model**: `{record.get('judge_model')}`",
        f"- **Include verdicts in evidence**: `{record.get('include_verdicts')}`",
        "",
        "## Summary: `WITHOUT` vs `Current WITH` vs `Candidate WITH`",
        "",
        "| Skill | Status | WITHOUT | Current WITH | Candidate WITH | Lift (Cand vs Cur) | Verify (W/O → Cur → Cand) | Tokens (W/O → Cur → Cand) | Wall s (W/O → Cur → Cand) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for skill, s in sorted((record.get("skills") or {}).items()):
        wo = s["without"]
        cur = s["current_with"]
        cand = s["best_candidate"]
        delta_cur = (
            cand["mean"] - cur["mean"]
            if isinstance(cand.get("mean"), (int, float))
            and isinstance(cur.get("mean"), (int, float))
            else None
        )
        status = "ACCEPTED" if s.get("accepted") else "REJECTED"
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
        lines.extend(
            [
                "",
                f"## Skill: `{skill}` ({'ACCEPTED' if s.get('accepted') else 'REJECTED'})",
                "",
                f"- **Decision**: {s.get('decision_reason')}",
                f"- **Winning round**: `{s.get('best_round')}` (of {len(s.get('rounds') or [])})",
                f"- **Diagnosis**: {s.get('diagnosis')}",
                f"- **Changes summary**: {s.get('changes_summary')}",
                "",
                "### Proposed `SKILL.md` Diff (`best.patch`)",
                "",
                "```diff",
                (s.get("diff") or "").rstrip(),
                "```",
            ]
        )
    out = "\n".join(lines)
    if len(out) > MAX_EVIDENCE_CHARS:
        out = out[:MAX_EVIDENCE_CHARS] + "\n\n...[evidence packet truncated to fit CLI limit]...\n"
    return out + "\n"


def render_improve_html(record: Mapping[str, Any]) -> str:
    """Render a self-contained interactive 3-way HTML dashboard (`improve_report.html`)."""
    e = html.escape
    rows_html: list[str] = []
    cards_html: list[str] = []
    for skill, s in sorted((record.get("skills") or {}).items()):
        wo = s["without"]
        cur = s["current_with"]
        cand = s["best_candidate"]
        accepted = bool(s.get("accepted"))
        badge_cls = (
            "bg-emerald-500/15 text-emerald-500"
            if accepted
            else "bg-amber-500/15 text-amber-500"
        )
        badge_txt = "ACCEPTED" if accepted else "REJECTED"
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
            f"<td class='py-2.5 px-3 text-right tabular-nums'>{e(_fmt_num(wo.get('mean')))}</td>"
            f"<td class='py-2.5 px-3 text-right tabular-nums'>{e(_fmt_num(cur.get('mean')))}</td>"
            f"<td class='py-2.5 px-3 text-right tabular-nums font-semibold'>{e(_fmt_num(cand.get('mean')))}</td>"
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
        cards_html.append(
            f"<div class='bg-[var(--card,#1e293b)] border border-[var(--border,#334155)] rounded-xl p-5 mb-4 shadow-sm'>"
            f"<h3 class='text-base font-semibold mb-2'><code class='font-mono'>{e(skill)}</code> — "
            f"<span class='inline-block rounded-full px-2.5 py-0.5 text-xs font-semibold {badge_cls}'>{badge_txt}</span></h3>"
            f"<p class='text-sm mb-2'><strong>Decision:</strong> {e(str(s.get('decision_reason') or ''))}</p>"
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
    Executor: <code class="font-mono">{e(str(record.get('executor_model')))}</code> ·
    Judge: <code class="font-mono">{e(str(record.get('judge_model')))}</code> ·
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
    root: Path = TASKS_ROOT,
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

    skill_path = target_skill_file(skill)
    original_md = skill_path.read_text(encoding="utf-8")

    without_summary = summarize_arm_runs(tasks, runs_dir / "results" / "without", k=k)
    current_with_summary = summarize_arm_runs(tasks, runs_dir / "results" / "with", k=k)

    skill_out = runs_dir / "improve" / skill
    skill_out.mkdir(parents=True, exist_ok=True)

    imp_model = improver_model(improver_model_name)
    round_records: list[dict[str, Any]] = []
    best_round = 1
    best_summary: dict[str, Any] | None = None
    best_proposal: dict[str, Any] | None = None
    best_accepted = False
    best_reason = ""
    working_md = original_md

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
            )
        ):
            proposal = cached_prop
        else:
            proposal = propose_skill(
                skill=skill,
                original_md=original_md,
                evidence_packet=evidence,
                model=imp_model,
            )
            cand_path.write_text(proposal["skill_markdown"], encoding="utf-8")
            (rdir / "proposal.json").write_text(
                json.dumps(proposal, indent=2) + "\n", encoding="utf-8"
            )
        patch_txt = unified_skill_diff(original_md, proposal["skill_markdown"], skill=skill)
        (rdir / "skill.patch").write_text(patch_txt, encoding="utf-8")

        cand_summary = probe_candidate(
            skill=skill,
            candidate_skill_path=cand_path,
            tasks=tasks,
            out_results_dir=rdir / "results",
            k=k,
            root=root,
        )
        accepted, reason = evaluate_acceptance(current_with_summary, cand_summary)
        rec = {
            "round": r,
            "accepted": accepted,
            "decision_reason": reason,
            "diagnosis": proposal["diagnosis"],
            "changes_summary": proposal["changes_summary"],
            "mean_candidate": cand_summary.get("mean"),
            "tokens_median": cand_summary.get("tokens_median"),
            "t_total_median_s": cand_summary.get("t_total_median_s"),
            "candidate": cand_summary,
            "diff": patch_txt,
        }
        round_records.append(rec)

        if best_summary is None or (
            accepted and not best_accepted
        ) or (
            isinstance(cand_summary.get("mean"), (int, float))
            and isinstance(best_summary.get("mean"), (int, float))
            and cand_summary["mean"] > best_summary["mean"]
        ):
            best_round = r
            best_summary = cand_summary
            best_proposal = proposal
            best_accepted = accepted
            best_reason = reason

        if accepted and (cand_summary.get("mean") or 0.0) >= 1.0 - 1e-6:
            break
        working_md = proposal["skill_markdown"]

    assert best_summary is not None and best_proposal is not None
    best_diff = unified_skill_diff(original_md, best_proposal["skill_markdown"], skill=skill)
    (skill_out / "best_SKILL.md").write_text(best_proposal["skill_markdown"], encoding="utf-8")
    (skill_out / "best.patch").write_text(best_diff, encoding="utf-8")

    applied_to: str | None = None
    if apply_accepted and best_accepted:
        skill_path.write_text(best_proposal["skill_markdown"], encoding="utf-8")
        applied_to = str(skill_path)

    return {
        "skill": skill,
        "accepted": best_accepted,
        "decision_reason": best_reason,
        "best_round": best_round,
        "diagnosis": best_proposal["diagnosis"],
        "changes_summary": best_proposal["changes_summary"],
        "diff": best_diff,
        "applied_to": applied_to,
        "without": without_summary,
        "current_with": current_with_summary,
        "best_candidate": best_summary,
        "rounds": round_records,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="skillbench-improve",
        description="Propose, probe, and verify improvements for weak cc-thingz skills.",
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="skillbench runs directory (default: $HARNESS2_RUNS_DIR/skillbench)",
    )
    ap.add_argument(
        "--skill",
        default="",
        help="comma-separated skill name(s) to improve (or pass --auto)",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="automatically select skills with score < 1.0, lift <= 0, or >25%% overhead",
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
        help="write accepted candidate SKILL.md back to $SKILLBENCH_PLUGINS_DIR",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write evidence packet and print selected skills without calling agy",
    )
    args = ap.parse_args(argv)

    runs_dir = (
        args.runs_dir.expanduser().resolve() if args.runs_dir else _default_runs_root()
    )
    skills = [s.strip() for s in args.skill.split(",") if s.strip()]
    if args.auto and not skills:
        report_json = runs_dir / "report.json"
        rep = (
            json.loads(report_json.read_text(encoding="utf-8"))
            if report_json.is_file()
            else build_report(runs_dir, k=args.k, pilot=args.pilot or None)
        )
        skills = select_weak_skills(rep)
    if not skills:
        print(
            "error: specify --skill <name> or --auto (with an existing skillbench run)",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        all_tasks = load_tasks(TASKS_ROOT)
        pilot_only = bool(args.pilot or _env_pilot())
        for sk in skills:
            tlist = [
                t
                for t in all_tasks.values()
                if t.target_skill == sk and (not pilot_only or t.pilot)
            ]
            md = target_skill_file(sk).read_text(encoding="utf-8")
            ev = build_evidence_packet(
                skill=sk,
                current_skill_md=md,
                tasks=tlist,
                runs_dir=runs_dir,
                k=args.k,
                include_verdicts=args.include_verdicts,
            )
            out_p = runs_dir / "improve" / sk / "round_1" / "evidence.md"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(ev, encoding="utf-8")
            print(f"[dry-run] {sk}: {len(tlist)} task(s) -> {out_p} ({len(ev)} chars)")
        return 0

    from ..benchmarks.skillbench import judge_model

    results: dict[str, Any] = {}
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
        )
        results[sk] = res
        status = "ACCEPTED" if res["accepted"] else "REJECTED"
        print(f"  [{sk}] {status}: {res['decision_reason']}", flush=True)

    out_dir = runs_dir / "improve"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "improver_model": improver_model(args.improver_model),
        "executor_model": exec_model(),
        "judge_model": judge_model(),
        "include_verdicts": bool(args.include_verdicts),
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
