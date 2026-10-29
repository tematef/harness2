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

"""Run a full skillbench evaluation and improvement cycle (`skillbench-cycle`).

Orchestrates the four stages in order:
    `matrix -> report -> history append -> improve (optional)`

Every stage is resumable and skippable (`--from <stage>` or `--skip <stage,...>`).
Writes `<runs_root>/cycle.log` with per-stage wall time and token totals, and prints
the Summary table alongside the paths of `report.html`, `summary.md`, and
`improve_report.html`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .. import config
from ..benchmarks.skillbench import SKILL_TIMING_FILE, _default_runs_root
from ..benchmarks.skillbench_suite import SuiteError, active_suite
from . import _common as C
from . import skillbench_improve, skillbench_matrix, skillbench_report, skillbench_summary

STAGES: tuple[str, ...] = ("matrix", "report", "history", "improve")
CYCLE_LOG_FILENAME = "cycle.log"

_STAGE_ALIASES: dict[str, str] = {
    "matrix": "matrix",
    "report": "report",
    "history": "history",
    "history append": "history",
    "history-append": "history",
    "history_append": "history",
    "improve": "improve",
}


def normalize_stage(stage: str | None) -> str:
    """Normalize a stage name (`matrix`, `report`, `history` / `history append`, `improve`)."""
    if stage is None:
        return "matrix"
    key = str(stage).strip().lower()
    if not key:
        return "matrix"
    resolved = _STAGE_ALIASES.get(key)
    if resolved is None:
        valid = ", ".join(STAGES)
        raise ValueError(f"unknown stage {stage!r}; expected one of: {valid}")
    return resolved


_csv = C.csv_list


# --------------------------------------------------------------------------------------------
# Stage argv builders & default stage runners
# --------------------------------------------------------------------------------------------


def _matrix_argv(args: argparse.Namespace, runs_root: Path) -> list[str]:
    argv = [
        "--k",
        str(args.k),
        "--arms",
        str(args.arms),
        "--parallel",
        str(args.parallel),
        "--split",
        str(args.split),
        "--runs-root",
        str(runs_root),
    ]
    if args.suite is not None:
        argv += ["--suite", str(args.suite)]
    if args.suites:
        argv += ["--suites", str(args.suites)]
    if args.executors:
        argv += ["--executors", str(args.executors)]
    if args.max_rollouts is not None:
        argv += ["--max-rollouts", str(args.max_rollouts)]
    if args.yes:
        argv.append("--yes")
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
    if args.history is not None:
        argv += ["--history", str(args.history)]
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def _report_argv(args: argparse.Namespace, runs_root: Path) -> list[str]:
    results_dir = runs_root / "results"
    argv = [
        "--k",
        str(args.k),
        "--results-root",
        str(results_dir),
        "--out-dir",
        str(runs_root),
    ]
    if args.suite is not None:
        argv += ["--suite", str(args.suite)]
    if args.suites:
        argv += ["--suites", str(args.suites)]
    if args.history is not None:
        argv += ["--history", str(args.history)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.pilot:
        argv.append("--pilot")
    if args.arms == "baseline":
        argv.append("--hardness-only")
    return argv


def _history_argv(args: argparse.Namespace, runs_root: Path) -> list[str]:
    argv = ["history", "append"]
    report_json = runs_root / "report.json"
    results_dir = runs_root / "results"
    if report_json.is_file():
        argv += ["--report", str(report_json)]
    elif results_dir.is_dir():
        argv += ["--results-root", str(results_dir)]
    else:
        argv += ["--runs-root", str(runs_root)]
    argv += ["--k", str(args.k)]
    if args.suite is not None:
        argv += ["--suite", str(args.suite)]
    if args.suites:
        argv += ["--suites", str(args.suites)]
    if args.history is not None:
        argv += ["--history", str(args.history)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.pilot:
        argv.append("--pilot")
    return argv


def _improve_argv(args: argparse.Namespace, runs_root: Path) -> list[str]:
    argv = [
        "--runs-dir",
        str(runs_root),
        "--k",
        str(args.k),
        "--rounds",
        str(args.rounds),
        "--max-growth",
        str(args.max_growth),
    ]
    if args.skill:
        argv += ["--skill", str(args.skill)]
    else:
        argv.append("--auto")
    if args.all_weak:
        argv.append("--all-weak")
    if args.suite is not None:
        argv += ["--suite", str(args.suite)]
    if args.executors:
        argv += ["--models", str(args.executors)]
    if args.allow_model_mismatch:
        argv.append("--allow-model-mismatch")
    if args.pilot:
        argv.append("--pilot")
    if args.include_verdicts:
        argv.append("--include-verdicts")
    if args.improver_model:
        argv += ["--improver-model", str(args.improver_model)]
    if args.apply:
        argv.append("--apply")
    if args.dry_run:
        argv.append("--dry-run")
    return argv


def stage_matrix(args: argparse.Namespace, runs_root: Path) -> int:
    """Execute the `matrix` stage via `skillbench_matrix.main`."""
    rc = skillbench_matrix.main(_matrix_argv(args, runs_root))
    return 0 if rc is None else int(rc)


def stage_report(args: argparse.Namespace, runs_root: Path) -> int:
    """Execute the `report` stage via `skillbench_report.main`."""
    rc = skillbench_report.main(_report_argv(args, runs_root))
    return 0 if rc is None else int(rc)


def stage_history(args: argparse.Namespace, runs_root: Path) -> int:
    """Execute the `history append` stage via `skillbench_report.main(['history', 'append', ...])`."""
    if args.arms == "baseline" and not (runs_root / "report.json").is_file():
        return 0
    rc = skillbench_report.main(_history_argv(args, runs_root))
    rc_int = 0 if rc is None else int(rc)
    if rc_int == 0 and (runs_root / "report.json").is_file():
        try:
            rep = json.loads((runs_root / "report.json").read_text(encoding="utf-8"))
            hpath = (
                Path(args.history).expanduser().resolve()
                if args.history is not None
                else skillbench_summary.default_history_path(args.suite, fallback_dir=runs_root)
            )
            if isinstance(rep, dict) and hpath.is_file():
                rep["summary"] = skillbench_summary.build_summary(rep, history=hpath)
                skillbench_report.write_report(rep, runs_root, history=hpath)
        except (OSError, ValueError, json.JSONDecodeError, config.ConfigError):
            pass
    return rc_int


def stage_improve(args: argparse.Namespace, runs_root: Path) -> int:
    """Execute the optional `improve` stage via `skillbench_improve.main`."""
    rc = skillbench_improve.main(_improve_argv(args, runs_root))
    return 0 if rc is None else int(rc)


# --------------------------------------------------------------------------------------------
# Token accounting & cycle.log helpers
# --------------------------------------------------------------------------------------------


def _sum_dir_tokens(root: Path) -> int:
    """Sum valid `total_tokens` across unique rollout directories under `root`."""
    seen_run_dirs: set[Path] = set()
    total = 0
    if not root.is_dir():
        return 0
    for tf in sorted(root.rglob(SKILL_TIMING_FILE)):
        try:
            run_dir = tf.parent.resolve()
            if run_dir in seen_run_dirs:
                continue
            data = json.loads(tf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, Mapping) or data.get("valid") is False:
            continue
        tok = data.get("total_tokens")
        if isinstance(tok, int) and not isinstance(tok, bool) and tok >= 0:
            seen_run_dirs.add(run_dir)
            total += int(tok)
    return total


def _compute_stage_tokens(stage: str, runs_root: Path) -> int:
    if stage == "matrix":
        return _sum_dir_tokens(runs_root / "results") or _sum_dir_tokens(runs_root / "adapt")
    if stage == "improve":
        return _sum_dir_tokens(runs_root / "improve")
    return 0


def format_cycle_log(
    stages: Sequence[Mapping[str, Any]],
    *,
    status: str,
    rc: int,
    total_wall_s: float,
    total_tokens: int,
    failed_stage: str | None = None,
) -> str:
    """Format `cycle.log` lines with per-stage and total wall time and token totals."""
    lines: list[str] = []
    for entry in stages:
        ts = str(
            entry.get("timestamp")
            or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        st_name = str(entry.get("stage") or "")
        st_status = str(entry.get("status") or "ok")
        st_rc = int(entry.get("rc") or 0)
        wall_s = float(entry.get("wall_s") or 0.0)
        tokens = int(entry.get("tokens") or 0)
        extra = ""
        if entry.get("reason"):
            extra += f" reason={entry['reason']}"
        lines.append(
            f"[{ts}] stage={st_name} status={st_status} rc={st_rc} "
            f"wall_s={wall_s:.3f} tokens={tokens}{extra}"
        )
    ts_end = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fail_part = f" failed_stage={failed_stage}" if failed_stage else ""
    lines.append(
        f"[{ts_end}] total status={status} rc={int(rc)}{fail_part} "
        f"wall_s={float(total_wall_s):.3f} tokens={int(total_tokens)}"
    )
    return "\n".join(lines) + "\n"


_LOG_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s+"
    r"(?:stage=(?P<stage>\S+)|(?P<total>total))\s+"
    r"status=(?P<status>\S+)\s+"
    r"rc=(?P<rc>-?\d+)"
    r"(?:\s+failed_stage=(?P<failed_stage>\S+))?\s+"
    r"wall_s=(?P<wall_s>[0-9.]+)\s+"
    r"tokens=(?P<tokens>\d+)"
    r"(?:\s+reason=(?P<reason>.*))?$"
)


def parse_cycle_log(log_path_or_text: Path | str) -> dict[str, Any]:
    """Parse a `cycle.log` file or string into `{stages, by_stage, total}`."""
    if isinstance(log_path_or_text, Path):
        text = log_path_or_text.read_text(encoding="utf-8")
    else:
        cand = Path(log_path_or_text)
        try:
            is_file = cand.is_file()
        except OSError:
            is_file = False
        text = cand.read_text(encoding="utf-8") if is_file else str(log_path_or_text)

    stages: list[dict[str, Any]] = []
    by_stage: dict[str, dict[str, Any]] = {}
    total: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LOG_LINE_RE.match(line)
        if not m:
            continue
        entry: dict[str, Any] = {
            "timestamp": m.group("ts"),
            "status": m.group("status"),
            "rc": int(m.group("rc")),
            "wall_s": float(m.group("wall_s")),
            "tokens": int(m.group("tokens")),
        }
        if m.group("reason"):
            entry["reason"] = m.group("reason")
        if m.group("total"):
            if m.group("failed_stage"):
                entry["failed_stage"] = m.group("failed_stage")
            total = entry
        else:
            st_name = str(m.group("stage"))
            entry["stage"] = st_name
            stages.append(entry)
            by_stage[st_name] = entry
    return {"stages": stages, "by_stage": by_stage, "total": total}


# --------------------------------------------------------------------------------------------
# Cycle execution
# --------------------------------------------------------------------------------------------


def _invoke_stage_runner(
    runner: Callable[..., Any],
    args: argparse.Namespace,
    runs_root: Path,
) -> tuple[int, str | None]:
    """Call `runner(args, runs_root)` and return `(rc, error_message)`."""
    try:
        raw = runner(args, runs_root)
    except (
        config.ConfigError,
        SuiteError,
        RuntimeError,
        ValueError,
        FileNotFoundError,
        OSError,
    ) as exc:
        return (1, str(exc))
    return (0 if raw is None else int(raw), None)


def _render_or_read_summary(args: argparse.Namespace, runs_root: Path) -> str:
    """Read `summary.md` if present, or render from `report.json`, or produce a minimal header."""
    summary_md_path = runs_root / "summary.md"
    if summary_md_path.is_file():
        try:
            text = summary_md_path.read_text(encoding="utf-8")
            if text.strip():
                return text
        except OSError:
            pass

    report_json_path = runs_root / "report.json"
    if report_json_path.is_file():
        try:
            rep = json.loads(report_json_path.read_text(encoding="utf-8"))
            if isinstance(rep, Mapping):
                return skillbench_summary.render_summary_markdown(
                    rep, history=args.history
                )
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    suite_name = "suite"
    if args.suite is not None:
        suite_name = Path(str(args.suite)).stem or str(args.suite)
    else:
        try:
            suite_name = active_suite(quiet=True).name
        except (config.ConfigError, OSError):
            pass
    stub_report: dict[str, Any] = {
        "schema_version": 2,
        "suite": {"name": suite_name, "hash": ""},
        "k": args.k,
        "seed": args.seed,
        "pilot": bool(args.pilot),
        "manifest_hash": "unverified",
        "models": [],
        "summary": [],
    }
    return skillbench_summary.render_summary_markdown(stub_report)


def _default_stage_runners() -> dict[str, Callable[..., Any]]:
    return {
        "matrix": stage_matrix,
        "report": stage_report,
        "history": stage_history,
        "improve": stage_improve,
    }


def run_cycle(
    args: argparse.Namespace,
    *,
    stage_runners: Mapping[str, Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    """Execute the `skillbench-cycle` pipeline and write `cycle.log`."""
    if args.k < 1 or args.parallel < 1:
        raise ValueError("--k and --parallel must be >= 1")
    if args.suite is not None and str(args.suite):
        C.apply_suite(args.suite)

    start_stage = normalize_stage(getattr(args, "from_stage", None))
    start_idx = STAGES.index(start_stage)
    skip_set = {normalize_stage(s) for s in _csv(getattr(args, "skip", None))}
    improve_enabled = bool(getattr(args, "improve", False) or start_stage == "improve")

    runs_root = (
        Path(args.runs_root).expanduser().resolve()
        if getattr(args, "runs_root", None) is not None
        else _default_runs_root()
    )
    runs_root.mkdir(parents=True, exist_ok=True)
    cycle_log_path = (
        Path(args.cycle_log).expanduser().resolve()
        if getattr(args, "cycle_log", None) is not None
        else (runs_root / CYCLE_LOG_FILENAME)
    )
    cycle_log_path.parent.mkdir(parents=True, exist_ok=True)

    runners = _default_stage_runners()
    if stage_runners:
        for k, fn in stage_runners.items():
            runners[normalize_stage(k)] = fn

    stage_records: list[dict[str, Any]] = []
    cycle_rc = 0
    failed_stage: str | None = None

    for idx, stage in enumerate(STAGES):
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if idx < start_idx:
            stage_records.append(
                {
                    "timestamp": ts,
                    "stage": stage,
                    "status": "skipped",
                    "rc": 0,
                    "wall_s": 0.0,
                    "tokens": 0,
                    "reason": f"--from={start_stage}",
                }
            )
            continue

        if stage in skip_set:
            stage_records.append(
                {
                    "timestamp": ts,
                    "stage": stage,
                    "status": "skipped",
                    "rc": 0,
                    "wall_s": 0.0,
                    "tokens": 0,
                    "reason": "--skip",
                }
            )
            continue

        if stage == "improve" and not improve_enabled:
            stage_records.append(
                {
                    "timestamp": ts,
                    "stage": stage,
                    "status": "skipped",
                    "rc": 0,
                    "wall_s": 0.0,
                    "tokens": 0,
                    "reason": "improve_off",
                }
            )
            continue

        t0 = time.monotonic()
        runner = runners[stage]
        rc, err_msg = _invoke_stage_runner(runner, args, runs_root)
        wall_s = max(0.0, time.monotonic() - t0)
        tokens = _compute_stage_tokens(stage, runs_root)

        status = "ok" if rc == 0 else "failed"
        rec: dict[str, Any] = {
            "timestamp": ts,
            "stage": stage,
            "status": status,
            "rc": rc,
            "wall_s": wall_s,
            "tokens": tokens,
        }
        if err_msg:
            rec["error"] = err_msg
        stage_records.append(rec)

        total_wall_so_far = sum(float(r["wall_s"]) for r in stage_records)
        total_tok_so_far = sum(int(r["tokens"]) for r in stage_records)
        cycle_log_path.write_text(
            format_cycle_log(
                stage_records,
                status="failed" if rc != 0 else "running",
                rc=rc,
                total_wall_s=total_wall_so_far,
                total_tokens=total_tok_so_far,
                failed_stage=stage if rc != 0 else None,
            ),
            encoding="utf-8",
        )

        if rc != 0:
            cycle_rc = rc
            failed_stage = stage
            detail = f": {err_msg}" if err_msg else ""
            msg = f"skillbench-cycle stopped: stage {stage!r} failed (exit code {rc}){detail}"
            print(msg, flush=True)
            print(f"error: {msg}", file=sys.stderr, flush=True)
            break

    total_wall_s = sum(float(r["wall_s"]) for r in stage_records)
    total_tokens = sum(int(r["tokens"]) for r in stage_records)
    final_status = "ok" if cycle_rc == 0 else "failed"
    cycle_log_path.write_text(
        format_cycle_log(
            stage_records,
            status=final_status,
            rc=cycle_rc,
            total_wall_s=total_wall_s,
            total_tokens=total_tokens,
            failed_stage=failed_stage,
        ),
        encoding="utf-8",
    )

    report_html_path = runs_root / "report.html"
    summary_md_path = runs_root / "summary.md"
    improve_html_path = runs_root / "improve" / "improve_report.html"

    if cycle_rc == 0:
        summary_text = _render_or_read_summary(args, runs_root)
        if not summary_text.endswith("\n"):
            summary_text += "\n"
        print(summary_text, end="", flush=True)
        print(f"report.html: {report_html_path}", flush=True)
        print(f"summary.md: {summary_md_path}", flush=True)
        print(f"improve_report.html: {improve_html_path}", flush=True)
        print(f"cycle.log: {cycle_log_path}", flush=True)

    return {
        "rc": cycle_rc,
        "status": final_status,
        "failed_stage": failed_stage,
        "stages": stage_records,
        "total_wall_s": total_wall_s,
        "total_tokens": total_tokens,
        "report_html": report_html_path,
        "summary_md": summary_md_path,
        "improve_report_html": improve_html_path,
        "cycle_log": cycle_log_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skillbench-cycle",
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
    p.add_argument("--k", type=int, default=3, help="rollouts per (executor, arm, task)")
    p.add_argument(
        "--arms",
        choices=skillbench_matrix.ARMS_MODES,
        default="both",
        help="arms to schedule: skill | suite | both | baseline",
    )
    p.add_argument(
        "--improve",
        action="store_true",
        default=False,
        help="run the optional skillbench-improve stage after history append (off by default)",
    )
    p.add_argument(
        "--from",
        "--from-stage",
        dest="from_stage",
        default="matrix",
        help="stage to start from, skipping earlier stages: matrix | report | history | improve",
    )
    p.add_argument(
        "--skip",
        default="",
        help="comma-separated stage names to skip",
    )
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="skip interactive TTY confirmation prompt in matrix stage",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=skillbench_matrix.DEFAULT_PARALLEL,
        help="domains to run concurrently in each sweep",
    )
    p.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help="hard cap on pending rollouts in matrix stage",
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
        help="restrict tasks to pilot = true",
    )
    p.add_argument(
        "--runs-root",
        "--runs-dir",
        dest="runs_root",
        type=Path,
        default=None,
        help="skillbench runs root (default: $HARNESS2_RUNS_DIR/skillbench)",
    )
    p.add_argument(
        "--history",
        type=Path,
        default=None,
        help="explicit history.jsonl path (default: history.jsonl next to suite.toml)",
    )
    p.add_argument(
        "--cycle-log",
        type=Path,
        default=None,
        help="explicit path for cycle.log (default: <runs-root>/cycle.log)",
    )
    p.add_argument("--rounds", type=int, default=1, help="max propose-probe rounds in improve")
    p.add_argument(
        "--skill",
        default="",
        help="comma-separated skill name(s) to force in improve stage",
    )
    p.add_argument(
        "--all-weak",
        action="store_true",
        help="include inconclusive skills with token/wall overhead in improve stage",
    )
    p.add_argument(
        "--max-growth",
        type=float,
        default=skillbench_improve.DEFAULT_MAX_GROWTH_PCT,
        help="maximum allowed candidate SKILL.md size growth in percent (default: 50)",
    )
    p.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help="allow probing with an executor different from the baseline cell in improve stage",
    )
    p.add_argument(
        "--include-verdicts",
        action="store_true",
        help="include failed rubric criteria in improver evidence",
    )
    p.add_argument(
        "--improver-model",
        default=None,
        help="override $SKILLBENCH_IMPROVER_MODEL in improve stage",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="apply accepted best.patch onto a branch in the suite checkout",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="pass --dry-run to matrix and improve stages",
    )
    return p


def main(
    argv: list[str] | None = None,
    *,
    stage_runners: Mapping[str, Callable[..., Any]] | None = None,
) -> int:
    """CLI entry point for `skillbench-cycle`."""
    args = _build_parser().parse_args(argv)
    args.__dict__["from"] = args.from_stage
    try:
        res = run_cycle(args, stage_runners=stage_runners)
        return int(res["rc"])
    except (config.ConfigError, SuiteError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
