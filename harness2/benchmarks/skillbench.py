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

"""skillbench: leave-one-out skill lift of the cc-thingz suite on the agy substrate.

Task model (this part; the judge and the benchmark adapter build on it):

    load_task       <root>/<skill>/<task-id>/ -> SkillbenchTask (task.toml + rubric.json + fixture/)
    load_tasks      every task under a corpus root, keyed `<skill>/<task-id>`
    check_prompt    explicit skill invocations / forbidden terms found in a prompt
    build_prompt    task prompt + the simulated-user preamble (identical in both arms)
    freeze          sha256 per task dir -> <root>/manifest.json
    check_manifest  refuses a corpus that drifted from its manifest

Task layout::

    <root>/<skill>/<task-id>/
        task.toml    target_skill, prompt, timeout, [user_answers], [forbidden_terms], [pilot]
        rubric.json  [{"id": ..., "criterion": ...}], 4-8 binary outcome-based criteria
        fixture/     small repo copied into every rollout's working directory

CLI: `python -m harness2.benchmarks.skillbench freeze [--force] [--root DIR]` and
`python -m harness2.benchmarks.skillbench check [--root DIR]`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..substrate.agy import AgyTask

# The bundled corpus. A `skillbench/` directory next to this module holds only data (no
# `__init__.py`), so `harness2.benchmarks.skillbench` still imports this file.
TASKS_ROOT = Path(__file__).resolve().with_name("skillbench") / "tasks"
MANIFEST = "manifest.json"
MANIFEST_VERSION = 1

TASK_FILE = "task.toml"
RUBRIC_FILE = "rubric.json"
FIXTURE_DIR = "fixture"

# Skills whose workflow calls `ask_question`: their tasks must script the simulated user.
ASK_QUESTION_SKILLS = frozenset(
    {"brainstorm", "exec", "make-plan", "dialectic", "backlog", "clarify", "learn", "wrong"}
)

MIN_CRITERIA = 4
MAX_CRITERIA = 8

_REQUIRED_KEYS = ("target_skill", "prompt", "timeout")
_OPTIONAL_KEYS = ("user_answers", "forbidden_terms", "pilot")
_NAME = re.compile(r"[a-z0-9][a-z0-9_\-]*")
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")
# Never part of a task's identity: copying a fixture drops them too (agy.init_work).
_HASH_IGNORED = frozenset({".git", "__pycache__"})

PREAMBLE_HEAD = "The user is unavailable."
PREAMBLE_TAIL = (
    "Never open an editor or an interactive review; write your results to files or to your "
    "final answer instead."
)


class TaskError(ValueError):
    """A skillbench task directory is malformed; the message names the file and the fix."""


class ManifestError(RuntimeError):
    """The task corpus differs from its frozen manifest (or has none)."""


@dataclass(frozen=True)
class Criterion:
    id: str
    criterion: str


@dataclass(frozen=True)
class SkillbenchTask:
    """One frozen skillbench task. `task_id` is `<skill>/<task-id>`; its domain is the skill."""

    task_id: str
    target_skill: str
    prompt: str
    timeout: int
    rubric: tuple[Criterion, ...]
    user_answers: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    pilot: bool = False
    dir: Path = Path()

    @property
    def fixture(self) -> Path:
        return self.dir / FIXTURE_DIR


# --------------------------------------------------------------------------------------------
# Prompt check and simulated user
# --------------------------------------------------------------------------------------------


def check_prompt(
    prompt: str, skill_names: Iterable[str], forbidden_terms: Iterable[str] = ()
) -> list[str]:
    """Explicit skill invocations in `prompt` (empty list = clean).

    Rejected, case-insensitively, for every name: `<name> skill`, a slash command `/<name>`,
    `skills/<name>`; plus any `SKILL.md` and every forbidden term (substring). Ordinary uses
    of a skill's name ("add this to the backlog", "what went wrong") pass.
    """
    found: list[str] = []
    if re.search(r"skill\.md", prompt, re.IGNORECASE):
        found.append("mentions SKILL.md")
    for name in sorted(set(skill_names)):
        n = re.escape(name)
        checks = (
            (rf"(?<![\w-]){n}\s+skills?\b", f"names the skill ('{name} skill')"),
            (rf"(?<![\w/.\-])/{n}(?![\w-])", f"slash command '/{name}'"),
            (rf"skills/{n}(?![\w-])", f"skill path 'skills/{name}'"),
        )
        for pattern, why in checks:
            if re.search(pattern, prompt, re.IGNORECASE):
                found.append(why)
    for term in forbidden_terms:
        if term and term.casefold() in prompt.casefold():
            found.append(f"forbidden term {term!r}")
    return found


def simulated_user_preamble(user_answers: Iterable[str]) -> str:
    """The instructions standing in for the absent user; depends only on the answers."""
    answers = list(user_answers)
    if not answers:
        return (
            f"{PREAMBLE_HEAD} When you would ask the user a question, do not ask: proceed "
            f"with your recommended option. {PREAMBLE_TAIL}"
        )
    lines = [
        f"{PREAMBLE_HEAD} When you would ask the user a question, do not wait for a reply: "
        "take these answers, in order, one per question:"
    ]
    lines += [f"{i}. {a}" for i, a in enumerate(answers, 1)]
    lines.append(
        f"Once these answers run out, proceed with your recommended option. {PREAMBLE_TAIL}"
    )
    return "\n".join(lines)


def build_prompt(task: SkillbenchTask) -> str:
    """The prompt exactly as sent to agy: task prompt + simulated-user preamble. Takes no arm,
    so both arms of a task always see the same text."""
    return f"{task.prompt.rstrip()}\n\n{simulated_user_preamble(task.user_answers)}"


def agy_task(task: SkillbenchTask) -> AgyTask:
    """What the agy substrate needs for one rollout of `task`."""
    return AgyTask(prompt=build_prompt(task), fixture=task.fixture)


# --------------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------------


def _str_list(meta: dict, key: str, where: Path) -> tuple[str, ...]:
    raw = meta.get(key, [])
    if not isinstance(raw, list) or any(not isinstance(v, str) or not v.strip() for v in raw):
        raise TaskError(f"{where}: {key} must be a list of non-empty strings")
    return tuple(v.strip() for v in raw)


def _load_rubric(path: Path) -> tuple[Criterion, ...]:
    if not path.is_file():
        raise TaskError(f"{path}: missing (a JSON list of {{'id', 'criterion'}} objects)")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise TaskError(f"{path}: not valid JSON ({e})") from None
    if not isinstance(data, list):
        raise TaskError(f"{path}: must be a JSON list of criteria")
    if not MIN_CRITERIA <= len(data) <= MAX_CRITERIA:
        raise TaskError(
            f"{path}: has {len(data)} criteria; write {MIN_CRITERIA}-{MAX_CRITERIA} binary, "
            f"outcome-based criteria"
        )
    out: list[Criterion] = []
    seen: set[str] = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict) or set(item) != {"id", "criterion"}:
            raise TaskError(f"{path}: criterion #{i} must have exactly the keys id, criterion")
        cid, text = item["id"], item["criterion"]
        if not isinstance(cid, str) or not cid.strip():
            raise TaskError(f"{path}: criterion #{i} id must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise TaskError(f"{path}: criterion {cid!r} text must be a non-empty string")
        cid = cid.strip()
        if cid in seen:
            raise TaskError(f"{path}: duplicate criterion id {cid!r}")
        seen.add(cid)
        out.append(Criterion(cid, text.strip()))
    return tuple(out)


def load_task(task_dir: Path) -> SkillbenchTask:
    """Load and validate `<root>/<skill>/<task-id>/`; raises `TaskError` on any defect."""
    task_dir = Path(task_dir)
    toml_path = task_dir / TASK_FILE
    if not toml_path.is_file():
        raise TaskError(f"{toml_path}: missing")
    try:
        meta = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise TaskError(f"{toml_path}: not valid TOML ({e})") from None
    missing = [k for k in _REQUIRED_KEYS if k not in meta]
    if missing:
        raise TaskError(f"{toml_path}: missing required field(s): {', '.join(missing)}")
    unknown = sorted(set(meta) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
    if unknown:
        raise TaskError(
            f"{toml_path}: unknown field(s) {', '.join(unknown)}; allowed: "
            f"{', '.join(_REQUIRED_KEYS + _OPTIONAL_KEYS)}"
        )

    skill = meta["target_skill"]
    if not isinstance(skill, str) or not _NAME.fullmatch(skill):
        raise TaskError(f"{toml_path}: target_skill must be a skill name like 'make-plan'")
    if task_dir.parent.name != skill:
        raise TaskError(
            f"{toml_path}: target_skill {skill!r} but the task lives under "
            f"{task_dir.parent.name!r}; tasks go in <root>/<target_skill>/<task-id>/"
        )
    if not _TASK_ID.fullmatch(task_dir.name):
        raise TaskError(f"{task_dir}: task directory name must match {_TASK_ID.pattern}")
    prompt = meta["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise TaskError(f"{toml_path}: prompt must be a non-empty string")
    timeout = meta["timeout"]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise TaskError(f"{toml_path}: timeout must be a positive integer (seconds)")
    pilot = meta.get("pilot", False)
    if not isinstance(pilot, bool):
        raise TaskError(f"{toml_path}: pilot must be true or false")
    answers = _str_list(meta, "user_answers", toml_path)
    if skill in ASK_QUESTION_SKILLS and not answers:
        raise TaskError(
            f"{toml_path}: {skill!r} asks the user questions; script the simulated user with "
            f"user_answers = [...]"
        )
    forbidden = _str_list(meta, "forbidden_terms", toml_path)
    problems = check_prompt(prompt, [skill], forbidden)
    if problems:
        raise TaskError(
            f"{toml_path}: prompt invokes the skill explicitly ({'; '.join(problems)}); "
            f"describe the task in natural words"
        )
    rubric = _load_rubric(task_dir / RUBRIC_FILE)
    if not (task_dir / FIXTURE_DIR).is_dir():
        raise TaskError(f"{task_dir / FIXTURE_DIR}: missing (the repo copied into each rollout)")
    return SkillbenchTask(
        task_id=f"{skill}/{task_dir.name}",
        target_skill=skill,
        prompt=prompt,
        timeout=timeout,
        rubric=rubric,
        user_answers=answers,
        forbidden_terms=forbidden,
        pilot=pilot,
        dir=task_dir,
    )


def task_dirs(root: Path) -> dict[str, Path]:
    """`<skill>/<task-id>` -> directory for every `<root>/*/*/task.toml` (no validation)."""
    root = Path(root)
    return {
        p.parent.relative_to(root).as_posix(): p.parent
        for p in sorted(root.glob(f"*/*/{TASK_FILE}"))
    }


def load_tasks(root: Path = TASKS_ROOT) -> dict[str, SkillbenchTask]:
    """Every task under `root`, validated, keyed `<skill>/<task-id>` (sorted)."""
    return {tid: load_task(d) for tid, d in task_dirs(root).items()}


# --------------------------------------------------------------------------------------------
# Freeze and manifest check
# --------------------------------------------------------------------------------------------


def hash_task_dir(task_dir: Path) -> str:
    """sha256 over every file's relative path and bytes (symlinks by target, not followed);
    `.git` and `__pycache__` are skipped, as when a fixture is copied into a rollout."""
    task_dir = Path(task_dir)
    entries: list[tuple[str, bytes]] = []
    for dirpath, dirnames, filenames in os.walk(task_dir, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in _HASH_IGNORED)
        here = Path(dirpath)
        for name in filenames:
            p = here / name
            rel = p.relative_to(task_dir).as_posix()
            if p.is_symlink():
                entries.append((rel, b"link:" + os.readlink(p).encode()))
            else:
                entries.append((rel, b"file:" + hashlib.sha256(p.read_bytes()).digest()))
        for name in dirnames:
            p = here / name
            if p.is_symlink():  # os.walk lists dir symlinks as dirs but never enters them
                rel = p.relative_to(task_dir).as_posix()
                entries.append((rel, b"link:" + os.readlink(p).encode()))
    h = hashlib.sha256()
    for rel, payload in sorted(entries):
        h.update(rel.encode() + b"\0" + payload + b"\0")
    return h.hexdigest()


def _manifest_hash(tasks: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()


def freeze(root: Path = TASKS_ROOT, *, force: bool = False) -> dict:
    """Validate every task under `root` and write `<root>/manifest.json` (sha256 per task dir).

    Refuses to overwrite an existing manifest unless `force`, and to freeze an empty corpus.
    """
    root = Path(root)
    path = root / MANIFEST
    if path.exists() and not force:
        raise ManifestError(f"{path} already exists; re-freeze with --force to overwrite it")
    tasks = load_tasks(root)
    if not tasks:
        raise ManifestError(f"no skillbench tasks under {root} (expected <skill>/<id>/task.toml)")
    hashes = {tid: hash_task_dir(t.dir) for tid, t in tasks.items()}
    manifest = {"version": MANIFEST_VERSION, "hash": _manifest_hash(hashes), "tasks": hashes}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def check_manifest(root: Path = TASKS_ROOT) -> str:
    """The manifest hash when every task dir under `root` matches `manifest.json`; otherwise
    `ManifestError` listing the changed, added and removed tasks."""
    root = Path(root)
    path = root / MANIFEST
    remedy = f"run `python -m harness2.benchmarks.skillbench freeze --root {root}`"
    if not path.is_file():
        raise ManifestError(f"{path}: missing; the skillbench corpus is not frozen ({remedy})")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ManifestError(f"{path}: not valid JSON ({e}); {remedy} --force") from None
    frozen = data.get("tasks") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or data.get("version") != MANIFEST_VERSION
        or not isinstance(frozen, dict)
        or data.get("hash") != _manifest_hash(frozen)
    ):
        raise ManifestError(f"{path}: malformed or edited by hand; {remedy} --force")
    current = {tid: hash_task_dir(d) for tid, d in task_dirs(root).items()}
    changed = sorted(t for t in current.keys() & frozen.keys() if current[t] != frozen[t])
    added = sorted(current.keys() - frozen.keys())
    removed = sorted(frozen.keys() - current.keys())
    if changed or added or removed:
        parts = [
            f"{label}: {', '.join(ids)}"
            for label, ids in (("changed", changed), ("added", added), ("removed", removed))
            if ids
        ]
        raise ManifestError(
            f"skillbench corpus drifted from {path} ({'; '.join(parts)}); review the tasks, "
            f"then {remedy} --force"
        )
    return data["hash"]


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m harness2.benchmarks.skillbench",
        description="Freeze or check the skillbench task corpus.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    fz = sub.add_parser("freeze", help="validate all tasks and write manifest.json")
    fz.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    ck = sub.add_parser("check", help="verify the corpus against manifest.json")
    for p in (fz, ck):
        p.add_argument("--root", type=Path, default=TASKS_ROOT, help="task corpus root")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "freeze":
            m = freeze(args.root, force=args.force)
            print(f"froze {len(m['tasks'])} task(s) -> {args.root / MANIFEST} ({m['hash'][:12]})")
        else:
            h = check_manifest(args.root)
            print(f"ok: {args.root / MANIFEST} matches ({h[:12]})")
    except (TaskError, ManifestError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
