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

"""skillbench-tasks: scaffold and check a skillbench task corpus (the authoring loop).

    skillbench-tasks new --skill X --id Y [--tasks ROOT] [--suite S] [--prompt ...]
    skillbench-tasks new --kind suite --family F --id Y [...]
    skillbench-tasks check [--tasks ROOT] [--suite S]

`new` writes `<root>/<skill>/<id>/` (or `<root>/_suite/<id>/` for suite tasks): a `task.toml`
with `forbidden_terms` pre-filled from the skill name and the script names its `SKILL.md`
mentions, `asks_user` pre-set from the SKILL.md frontmatter, a commented `verify` stub, a
`rubric.json` with four criterion stubs and `fixture/README.md`. Nothing is run.

`check` loads every task under the root and prints one row per task: prompt hygiene against
every skill of the active suite plus `forbidden_terms`, rubric size, `user_answers` present
for asking tasks, `kind` / `family` against the suite, scaffold placeholders left behind, an
empty fixture, and the task's hash versus `manifest.json` (information only: drift is what
`freeze` is for). Exit status 1 when any task fails.

The corpus root is `--tasks` -> `suite.tasks_root` -> `$SKILLBENCH_TASKS_ROOT` -> the bundled
corpus (`skillbench.tasks_root`).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .. import config
from ..benchmarks import skillbench as SB
from ..benchmarks.skillbench_suite import SuiteConfig, active_suite, skill_asks_user
from . import _common as C

DEFAULT_TIMEOUT = 900
RUBRIC_STUBS = 4
PROMPT_PLACEHOLDER = "TODO: describe the task in natural words, without naming any skill."
CRITERION_PLACEHOLDER = "TODO: one binary, outcome-only criterion the deliverable must meet."
FIXTURE_README = """\
# Fixture

This directory is copied into the working directory of every rollout of this task.
Replace this file with a small, self-contained repository that has a real failure mode the
built-in behaviour is likely to miss (hidden coupling, misleading symptom, multi-file
dependency). Keep it free of `.git/` and `__pycache__/`; both are dropped when copied.
"""

# Script names a SKILL.md may reference: `scripts/foo.sh`, `/abs/path/foo.py`, `bar.sh`. The
# stem (`foo`) goes into `forbidden_terms` so a prompt cannot hint at the skill's helpers.
_SCRIPT_REF = re.compile(r"(?<![\w\-])([A-Za-z0-9][A-Za-z0-9_\-]*)\.(?:sh|py)\b")
_TRIVIAL_STEMS = frozenset({"setup", "main", "test", "tests", "install", "run", "__init__"})


class TasksCliError(ValueError):
    """Bad arguments or an unusable corpus root; the message says what to fix."""


# --------------------------------------------------------------------------------------------
# Suite lookup (optional: both commands work without one, with fewer checks)
# --------------------------------------------------------------------------------------------


def _optional_suite(path: Path | str | None) -> SuiteConfig | None:
    """The suite to check against: `--suite` / `SKILLBENCH_SUITE` / env fallback, or None
    when nothing is configured (a broken suite file is still an error)."""
    try:
        return active_suite(path, quiet=True)
    except config.ConfigError as e:
        if path is not None or config.env(C.SUITE_ENV):
            raise TasksCliError(str(e)) from None
        return None


def _skill_md(suite: SuiteConfig | None, skill: str) -> str | None:
    if suite is None or not skill:
        return None
    p = suite.skill_files().get(skill)
    return p.read_text(encoding="utf-8", errors="replace") if p is not None else None


def script_stems(skill_md_text: str) -> list[str]:
    """Sorted stems of every `*.sh` / `*.py` the SKILL.md mentions (trivial names dropped)."""
    stems = {m.group(1) for m in _SCRIPT_REF.finditer(skill_md_text or "")}
    return sorted(s for s in stems if s.lower() not in _TRIVIAL_STEMS)


# --------------------------------------------------------------------------------------------
# new
# --------------------------------------------------------------------------------------------


def _toml_str(v: str) -> str:
    return json.dumps(v, ensure_ascii=False)


def _toml_list(items: Sequence[str]) -> str:
    if not items:
        return "[]"
    return "[\n" + "".join(f"  {_toml_str(i)},\n" for i in items) + "]"


def render_task_toml(
    *,
    kind: str,
    skill: str,
    family: str | None,
    prompt: str,
    timeout: int,
    asks_user: bool | None,
    forbidden_terms: Sequence[str],
) -> str:
    """The scaffolded task.toml: every key `load_task` knows, the optional ones commented."""
    lines: list[str] = []
    if kind == "suite":
        lines.append(f"kind = {_toml_str(kind)}")
        lines.append(f"family = {_toml_str(family or '')}")
    else:
        lines.append(f"target_skill = {_toml_str(skill)}")
    lines.append("pilot = false")
    lines.append(f"timeout = {int(timeout)}")
    lines.append(f'prompt = """{prompt}"""')
    if asks_user is None:
        lines.append(
            "# asks_user = true   # set when the skill asks the user (frontmatter allowed-tools);"
            " the simulated user then needs user_answers"
        )
        lines.append("# user_answers = []")
    elif asks_user:
        lines.append("asks_user = true")
        lines.append(
            "# One scripted answer per question the skill asks, in order; identical in every arm."
        )
        lines.append("user_answers = []")
    else:
        lines.append("asks_user = false")
    lines.append("# Terms that must not appear in the prompt (skill name, its helper scripts).")
    lines.append(f"forbidden_terms = {_toml_list(forbidden_terms)}")
    lines.append("# Deterministic check run against the agent's work tree (exit 0 = pass):")
    lines.append('# verify = "python -m unittest discover -s tests"')
    lines.append("# Fixture files put back before verify runs, so edits to tests cannot pass it:")
    lines.append('# verify_restore = ["tests/test_core.py"]')
    lines.append("# min_baseline_fail = 1   # criteria the baseline must miss for 'informative'")
    return "\n".join(lines) + "\n"


def rubric_stubs(n: int = RUBRIC_STUBS) -> list[dict[str, str]]:
    return [{"id": f"c{i}", "criterion": CRITERION_PLACEHOLDER} for i in range(1, n + 1)]


@dataclass(frozen=True)
class Scaffold:
    task_dir: Path
    files: tuple[Path, ...]
    asks_user: bool | None
    forbidden_terms: tuple[str, ...]
    notes: tuple[str, ...] = ()


def scaffold_task(
    root: Path,
    *,
    task_id: str,
    kind: str = "skill",
    skill: str | None = None,
    family: str | None = None,
    prompt: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    suite: SuiteConfig | None = None,
    force: bool = False,
) -> Scaffold:
    """Write a new task directory under `root`; refuses to overwrite unless `force`."""
    if kind not in SB.KINDS:
        raise TasksCliError(f"--kind must be one of {', '.join(SB.KINDS)}, got {kind!r}")
    if not SB._TASK_ID.fullmatch(task_id or ""):
        raise TasksCliError(f"--id must match {SB._TASK_ID.pattern}, got {task_id!r}")
    if timeout <= 0:
        raise TasksCliError(f"--timeout must be a positive number of seconds, got {timeout}")
    notes: list[str] = []
    if kind == "suite":
        if skill:
            raise TasksCliError("--kind suite tasks benchmark the whole suite: drop --skill")
        if not family or not SB._NAME.fullmatch(family):
            raise TasksCliError("--kind suite needs --family <name> (e.g. planning)")
        if suite is not None and suite.families and family not in suite.families:
            raise TasksCliError(
                f"family {family!r} is not defined in suite {suite.name!r} "
                f"(known: {', '.join(sorted(suite.families))})"
            )
        parent = SB.SUITE_DIR
        skill = ""
        asks_user: bool | None = None
        forbidden: list[str] = []
    else:
        if family:
            raise TasksCliError("--family applies to --kind suite only")
        if not skill or not SB._NAME.fullmatch(skill):
            raise TasksCliError("--skill <name> is required for a skill task (e.g. make-plan)")
        parent = skill
        md = _skill_md(suite, skill)
        if md is None:
            if suite is not None:
                raise TasksCliError(
                    f"skill {skill!r} is not provided by suite {suite.name!r} "
                    f"(known: {', '.join(sorted(suite.skill_names())) or '-'})"
                )
            asks_user = None
            notes.append("no suite configured: asks_user and script names left for you")
            stems: list[str] = []
        else:
            asks_user = skill_asks_user(md)
            stems = script_stems(md)
        forbidden = [skill] + [s for s in stems if s != skill]
    text = (prompt or PROMPT_PLACEHOLDER).strip()
    problems = SB.check_prompt(text, [skill] if skill else [], forbidden)
    if problems:
        raise TasksCliError(f"--prompt invokes the skill explicitly ({'; '.join(problems)})")

    task_dir = Path(root) / parent / task_id
    if task_dir.exists() and not force:
        raise TasksCliError(f"{task_dir} already exists (use --force to overwrite its scaffold)")
    (task_dir / SB.FIXTURE_DIR).mkdir(parents=True, exist_ok=True)
    toml_path = task_dir / SB.TASK_FILE
    toml_path.write_text(
        render_task_toml(
            kind=kind,
            skill=skill,
            family=family,
            prompt=text,
            timeout=timeout,
            asks_user=asks_user,
            forbidden_terms=forbidden,
        ),
        encoding="utf-8",
    )
    rubric_path = task_dir / SB.RUBRIC_FILE
    rubric_path.write_text(json.dumps(rubric_stubs(), indent=2) + "\n", encoding="utf-8")
    readme = task_dir / SB.FIXTURE_DIR / "README.md"
    readme.write_text(FIXTURE_README, encoding="utf-8")
    return Scaffold(
        task_dir=task_dir,
        files=(toml_path, rubric_path, readme),
        asks_user=asks_user,
        forbidden_terms=tuple(forbidden),
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------------


@dataclass
class TaskCheck:
    """The result of checking one task directory; `ok` when `problems` is empty."""

    task_id: str
    kind: str = "?"
    rubric: int | None = None
    asks: bool | None = None
    answers: int = 0
    manifest: str = "-"
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _manifest_hashes(root: Path) -> dict[str, str] | None:
    p = root / SB.MANIFEST
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    tasks = data.get("tasks") if isinstance(data, dict) else None
    return tasks if isinstance(tasks, dict) else {}


def _manifest_status(frozen: dict[str, str] | None, task_id: str, task_dir: Path) -> str:
    if frozen is None:
        return "-"
    current = SB.hash_task_dir(task_dir)
    if task_id not in frozen:
        return "new"
    return "ok" if frozen[task_id] == current else "changed"


def check_task(
    task_id: str,
    task_dir: Path,
    *,
    suite: SuiteConfig | None,
    frozen: dict[str, str] | None,
) -> TaskCheck:
    """Every check for one task directory; never raises for a malformed task."""
    res = TaskCheck(task_id=task_id)
    try:
        task = SB.load_task(task_dir)
    except SB.TaskError as e:
        res.problems.append(str(e).replace(f"{task_dir}/", ""))
        res.manifest = _manifest_status(frozen, task_id, task_dir)
        return res
    res.kind = task.kind
    res.rubric = len(task.rubric)
    res.answers = len(task.user_answers)
    res.manifest = _manifest_status(frozen, task_id, task_dir)

    suite_skills = sorted(suite.skill_names()) if suite is not None else []
    names = set(suite_skills) | ({task.target_skill} if task.target_skill else set())
    problems = SB.check_prompt(task.prompt, names, task.forbidden_terms)
    if problems:
        res.problems.append("prompt: " + "; ".join(problems))
    if task.prompt.strip() == PROMPT_PLACEHOLDER:
        res.problems.append("prompt is still the scaffold placeholder")
    stubs = [c.id for c in task.rubric if c.criterion.strip() == CRITERION_PLACEHOLDER]
    if stubs:
        res.problems.append(f"rubric criteria still stubs: {', '.join(stubs)}")
    if not SB.MIN_CRITERIA <= len(task.rubric) <= SB.MAX_CRITERIA:  # load_task enforces this
        res.problems.append(f"rubric has {len(task.rubric)} criteria")

    if task.kind == "suite":
        if suite is not None and suite.families and task.family not in suite.families:
            res.problems.append(
                f"family {task.family!r} not in suite {suite.name!r} "
                f"({', '.join(sorted(suite.families))})"
            )
        md = None
    else:
        if suite is not None and task.target_skill not in suite.skill_names():
            res.problems.append(
                f"target_skill {task.target_skill!r} is not provided by suite {suite.name!r}"
            )
        md = _skill_md(suite, task.target_skill)
    asks = SB.task_asks_user(task, md)
    res.asks = asks if (task.asks_user is not None or md is not None) else None
    if asks and not task.user_answers:
        res.problems.append("asking task without user_answers (script the simulated user)")

    fixture_files = [p for p in task.fixture.rglob("*") if p.is_file()]
    if not fixture_files:
        res.problems.append("fixture/ is empty")
    return res


def check_corpus(root: Path, *, suite: SuiteConfig | None) -> list[TaskCheck]:
    """Check every `<root>/*/*/task.toml`; an empty or missing root is a single failing row."""
    root = Path(root)
    dirs = SB.task_dirs(root) if root.is_dir() else {}
    if not dirs:
        return [TaskCheck(task_id="(corpus)", problems=[f"no tasks under {root}"])]
    frozen = _manifest_hashes(root)
    return [
        check_task(tid, d, suite=suite, frozen=frozen) for tid, d in dirs.items()
    ]


def render_table(rows: Sequence[TaskCheck]) -> str:
    """A fixed-width table, one row per task, problems indented below their row."""
    head = ("task", "kind", "rubric", "asks", "answers", "manifest", "status")
    body = [
        (
            r.task_id,
            r.kind,
            "-" if r.rubric is None else str(r.rubric),
            "-" if r.asks is None else ("yes" if r.asks else "no"),
            str(r.answers),
            r.manifest,
            "ok" if r.ok else "FAIL",
        )
        for r in rows
    ]
    widths = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(head)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*head).rstrip(), fmt.format(*("-" * w for w in widths))]
    for r, b in zip(rows, body):
        out.append(fmt.format(*b).rstrip())
        out.extend(f"    - {p}" for p in r.problems)
    failed = sum(1 for r in rows if not r.ok)
    out.append(f"{len(rows)} task(s), {failed} failing")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _add_root_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="task corpus root (default: the suite's tasks_root, else $SKILLBENCH_TASKS_ROOT, "
        "else the bundled corpus)",
    )
    C.add_suite_arg(p)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="skillbench-tasks", description=__doc__, formatter_class=C.HelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser(
        "new", help="scaffold a task directory", formatter_class=C.HelpFormatter
    )
    new.add_argument("--skill", default=None, help="target skill of a skill task")
    new.add_argument("--id", dest="task_id", required=True, help="task directory name")
    new.add_argument("--kind", choices=SB.KINDS, default="skill", help="skill | suite")
    new.add_argument("--family", default=None, help="family of a suite task (suite.toml)")
    new.add_argument("--prompt", default=None, help="task prompt (default: a placeholder)")
    new.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="rollout timeout, s")
    new.add_argument("--force", action="store_true", help="overwrite an existing scaffold")
    _add_root_args(new)
    chk = sub.add_parser(
        "check", help="validate every task under the root", formatter_class=C.HelpFormatter
    )
    _add_root_args(chk)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        C.apply_suite(args.suite)
        suite = _optional_suite(args.suite)
        root = SB.tasks_root(args.tasks, suite)
        if args.cmd == "new":
            sc = scaffold_task(
                root,
                task_id=args.task_id,
                kind=args.kind,
                skill=args.skill,
                family=args.family,
                prompt=args.prompt,
                timeout=args.timeout,
                suite=suite,
                force=args.force,
            )
            print(f"scaffolded {sc.task_dir}")
            for f in sc.files:
                print(f"  {f.relative_to(sc.task_dir)}")
            asks = "unknown" if sc.asks_user is None else ("yes" if sc.asks_user else "no")
            print(f"  asks_user: {asks}; forbidden_terms: {', '.join(sc.forbidden_terms) or '-'}")
            for n in sc.notes:
                print(f"  note: {n}")
            print("next: write the prompt, the rubric and the fixture, then `skillbench-tasks check`")
            return 0
        rows = check_corpus(root, suite=suite)
        print(f"corpus: {root}" + (f"  suite: {suite.name}" if suite else "  suite: (none)"))
        print(render_table(rows))
        return 0 if all(r.ok for r in rows) else 1
    except (TasksCliError, config.ConfigError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
