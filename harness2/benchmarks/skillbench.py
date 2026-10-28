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

"""skillbench: leave-one-out skill lift of a plugin suite (see skillbench_suite) on the agy
substrate.

Task model and agy judge (the benchmark adapter builds on both):

    tasks_root         the corpus root: explicit path -> suite.tasks_root -> $SKILLBENCH_TASKS_ROOT
                       -> the bundled corpus (`TASKS_ROOT`)
    load_task          <root>/<skill>/<task-id>/ -> SkillbenchTask (task.toml + rubric.json + fixture/)
    load_tasks         every task under a corpus root, keyed `<skill>/<task-id>`
    check_prompt       explicit skill invocations / forbidden terms found in a prompt
    build_prompt       task prompt + the simulated-user preamble (identical in both arms)
    freeze             sha256 per task dir -> <root>/manifest.json
    check_manifest     refuses a corpus that drifted from its manifest
    deliverable_files  final.md, diff.patch, new_files/, side_effects/ from a rollout dir
    deliverable_text   combined deliverable text for a rollout dir
    scrub_deliverable  redacts skill names, SKILL.md paths and plugin paths to `[redacted]`
                       (the names come from `skillbench_suite.redaction_names`)
    task_asks_user     whether a task's skill asks the user (`asks_user` override, else the
                       SKILL.md frontmatter via `skillbench_suite.skill_asks_user`)
    score              runs the isolated agy judge (`--model $SKILLBENCH_JUDGE_MODEL --json-schema`)
                       and caches judgments in `<run_dir>/judgments.json`
    run_verify         optional deterministic check (`verify` command) -> `<run_dir>/verify.json`
    diff_stats         files / lines added / lines removed in a rollout's `diff.patch`
    SkillbenchAdapter  BenchmarkAdapter for one arm: skill arms (`with` / `without`, leave-one-out
                       lift) run `kind = "skill"` tasks, suite arms (`baseline` / `suite-<name>`,
                       suite vs built-in behaviour) run `kind = "suite"` tasks; rollouts land in
                       `results/<executor>/<arm label>/`

Task layout::

    <root>/<skill>/<task-id>/          kind = "skill" (default): leave-one-out tasks
    <root>/_suite/<task-id>/           kind = "suite": whole-suite tasks, grouped by `family`
        task.toml    target_skill (skill tasks), prompt, timeout, [kind], [family], [user_answers],
                     [forbidden_terms], [pilot], [verify], [verify_restore], [asks_user],
                     [min_baseline_fail]
        rubric.json  [{"id": ..., "criterion": ...}], 4-8 binary outcome-based criteria
        fixture/     small repo copied into every rollout's working directory

The corpus root is resolved by `tasks_root()`; `skillbench-tasks new` scaffolds a task and
`skillbench-tasks check` validates a corpus against the active suite.

CLI: `python -m harness2.benchmarks.skillbench freeze [--force] [--root DIR]` and
`python -m harness2.benchmarks.skillbench check [--root DIR]`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .. import config
from ..core import runmark as RM
from ..core import tree as HT
from ..core.spec import BenchPrompts, BenchSpec, ChannelSpec
from ..substrate.agy import (
    AGY_SURFACE,
    INSTALL_MODE,
    PLUGINS_REL,
    AgyEvent,
    AgySubstrate,
    AgyTask,
    ArmSpec,
    _kill_group,
    agy_argv,
    agy_bin_missing_message,
    auth_files,
    checked_plugins,
    exec_effort,
    exec_model,
    model_guard,
    parse_arm_label,
    parse_event,
    resolve_agy_bin,
)
from .skillbench_suite import (
    SuiteConfig,
    SuiteError,
    active_suite,
    load_suite,
    redaction_names,
    skill_asks_user,
)
from .tasks import discover, select

# The bundled corpus. A `skillbench/` directory next to this module holds only data (no
# `__init__.py`), so `harness2.benchmarks.skillbench` still imports this file.
TASKS_ROOT = Path(__file__).resolve().with_name("skillbench") / "tasks"
TASKS_ROOT_ENV = "SKILLBENCH_TASKS_ROOT"
MANIFEST = "manifest.json"
MANIFEST_VERSION = 1

TASK_FILE = "task.toml"
RUBRIC_FILE = "rubric.json"
FIXTURE_DIR = "fixture"
# Whole-suite tasks (`kind = "suite"`) live under `<root>/_suite/<task-id>/`.
SUITE_DIR = "_suite"
KINDS = ("skill", "suite")

MIN_CRITERIA = 4
MAX_CRITERIA = 8

_REQUIRED_KEYS = ("prompt", "timeout")
_OPTIONAL_KEYS = (
    "target_skill",
    "kind",
    "family",
    "user_answers",
    "forbidden_terms",
    "pilot",
    "verify",
    "verify_restore",
    "asks_user",
    "min_baseline_fail",
)
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
    """One frozen skillbench task. `task_id` is `<skill>/<task-id>` (`_suite/<task-id>` for
    suite tasks); its domain is the skill, or the family for suite tasks."""

    task_id: str
    target_skill: str
    prompt: str
    timeout: int
    rubric: tuple[Criterion, ...]
    user_answers: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    pilot: bool = False
    dir: Path = Path()
    verify: tuple[str, ...] = ()
    verify_restore: tuple[str, ...] = ()
    # `asks_user` from task.toml; None = decide from the skill's SKILL.md (`task_asks_user`).
    asks_user: bool | None = None
    # "skill" (leave-one-out, `target_skill` set) or "suite" (whole suite, `family` set).
    kind: str = "skill"
    family: str | None = None
    # Criteria the baseline arm must miss (in >= 1 valid rep) for the task to count as
    # informative for an executor (the hardness flag; never an exclusion).
    min_baseline_fail: int = 1

    @property
    def fixture(self) -> Path:
        return self.dir / FIXTURE_DIR

    @property
    def domain(self) -> str:
        """The per-task grouping key: the target skill, or the family of a suite task."""
        if self.kind == "suite" and self.family:
            return self.family
        return self.target_skill


# --------------------------------------------------------------------------------------------
# Corpus root resolution
# --------------------------------------------------------------------------------------------


def tasks_root(
    explicit: Path | str | None = None, suite: SuiteConfig | None = None
) -> Path:
    """The task corpus root: `explicit` (a `--tasks` flag) -> `suite.tasks_root` ->
    `$SKILLBENCH_TASKS_ROOT` -> the bundled corpus. `suite` defaults to the active suite when
    one is configured; no suite config at all is not an error here (the bundled corpus
    needs none), but a configured suite file that fails to load is (`SuiteError`)."""
    if explicit is not None and str(explicit) != "":
        return Path(explicit).expanduser()
    if suite is None:
        try:
            suite = active_suite(quiet=True)
        except SuiteError:
            raise
        except config.ConfigError:
            suite = None
    if suite is not None and suite.tasks_root is not None:
        return Path(suite.tasks_root).expanduser()
    raw = config.env(TASKS_ROOT_ENV)
    if raw:
        return Path(str(raw)).expanduser()
    return TASKS_ROOT


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


def _load_kind(meta: dict, toml_path: Path, task_dir: Path) -> tuple[str, str, str | None]:
    """`(kind, target_skill, family)` from task.toml, checked against the directory layout:
    skill tasks live in `<root>/<target_skill>/`, suite tasks in `<root>/_suite/` with a
    `family` and no `target_skill`."""
    kind = meta.get("kind", "skill")
    if kind not in KINDS:
        raise TaskError(f"{toml_path}: kind must be one of {', '.join(KINDS)}, got {kind!r}")
    family = meta.get("family")
    if family is not None and (not isinstance(family, str) or not _NAME.fullmatch(family)):
        raise TaskError(f"{toml_path}: family must be a family name like 'planning'")
    if kind == "suite":
        if "target_skill" in meta:
            raise TaskError(
                f"{toml_path}: a kind = \"suite\" task benchmarks the whole suite; drop "
                f"target_skill and name its family instead"
            )
        if family is None:
            raise TaskError(f"{toml_path}: a kind = \"suite\" task needs family = \"<name>\"")
        if task_dir.parent.name != SUITE_DIR:
            raise TaskError(
                f"{toml_path}: kind = \"suite\" tasks go in <root>/{SUITE_DIR}/<task-id>/, not "
                f"under {task_dir.parent.name!r}"
            )
        return kind, "", family
    if "target_skill" not in meta:
        raise TaskError(f"{toml_path}: missing required field(s): target_skill")
    if family is not None:
        raise TaskError(
            f"{toml_path}: family applies to kind = \"suite\" tasks only (skill families are "
            f"defined in suite.toml)"
        )
    skill = meta["target_skill"]
    if not isinstance(skill, str) or not _NAME.fullmatch(skill):
        raise TaskError(f"{toml_path}: target_skill must be a skill name like 'make-plan'")
    if task_dir.parent.name != skill:
        raise TaskError(
            f"{toml_path}: target_skill {skill!r} but the task lives under "
            f"{task_dir.parent.name!r}; tasks go in <root>/<target_skill>/<task-id>/"
        )
    return kind, skill, None


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

    kind, skill, family = _load_kind(meta, toml_path, task_dir)
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
    asks_user = meta.get("asks_user")
    if asks_user is not None and not isinstance(asks_user, bool):
        raise TaskError(f"{toml_path}: asks_user must be true or false")
    min_fail = meta.get("min_baseline_fail", 1)
    if isinstance(min_fail, bool) or not isinstance(min_fail, int) or min_fail < 1:
        raise TaskError(f"{toml_path}: min_baseline_fail must be a positive integer")
    forbidden = _str_list(meta, "forbidden_terms", toml_path)
    problems = check_prompt(prompt, [skill] if skill else [], forbidden)
    if problems:
        raise TaskError(
            f"{toml_path}: prompt invokes the skill explicitly ({'; '.join(problems)}); "
            f"describe the task in natural words"
        )
    rubric = _load_rubric(task_dir / RUBRIC_FILE)
    if not (task_dir / FIXTURE_DIR).is_dir():
        raise TaskError(f"{task_dir / FIXTURE_DIR}: missing (the repo copied into each rollout)")
    verify, verify_restore = _load_verify(meta, toml_path, task_dir / FIXTURE_DIR)
    return SkillbenchTask(
        task_id=f"{task_dir.parent.name}/{task_dir.name}",
        target_skill=skill,
        prompt=prompt,
        timeout=timeout,
        rubric=rubric,
        user_answers=answers,
        forbidden_terms=forbidden,
        pilot=pilot,
        dir=task_dir,
        verify=verify,
        verify_restore=verify_restore,
        asks_user=asks_user,
        kind=kind,
        family=family,
        min_baseline_fail=min_fail,
    )


def task_asks_user(task: SkillbenchTask, skill_md_text: str | None = None) -> bool:
    """Whether the simulated user must be scripted for `task`: the task's `asks_user` when
    set, else whether the skill's `SKILL.md` frontmatter (`skill_md_text`) allows
    `ask_question`; False when neither is known. A missing `user_answers` on an asking task
    is reported by `skillbench-tasks check`, not by `load_task`."""
    if task.asks_user is not None:
        return task.asks_user
    return skill_asks_user(skill_md_text) if skill_md_text else False


def _load_verify(
    meta: dict, toml_path: Path, fixture: Path
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`verify` (a command, split like a shell would, never run through one) and
    `verify_restore` (fixture-relative files put back before it runs)."""
    raw = meta.get("verify")
    restore = _str_list(meta, "verify_restore", toml_path)
    if raw is None:
        if restore:
            raise TaskError(f"{toml_path}: verify_restore needs a verify command")
        return ((), ())
    if not isinstance(raw, str) or not raw.strip():
        raise TaskError(f"{toml_path}: verify must be a non-empty command string")
    try:
        argv = tuple(shlex.split(raw))
    except ValueError as e:
        raise TaskError(f"{toml_path}: verify is not a valid command ({e})") from None
    for rel in restore:
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or not (fixture / p).is_file():
            raise TaskError(
                f"{toml_path}: verify_restore entry {rel!r} must be a file under fixture/"
            )
    return (argv, restore)


def task_dirs(root: Path) -> dict[str, Path]:
    """`<skill>/<task-id>` -> directory for every `<root>/*/*/task.toml` (no validation)."""
    root = Path(root)
    return {
        p.parent.relative_to(root).as_posix(): p.parent
        for p in sorted(root.glob(f"*/*/{TASK_FILE}"))
    }


def load_tasks(root: Path | None = None) -> dict[str, SkillbenchTask]:
    """Every task under `root` (default: `tasks_root()`), validated, keyed `<skill>/<task-id>`
    (sorted)."""
    return {tid: load_task(d) for tid, d in task_dirs(tasks_root(root)).items()}


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


def freeze(root: Path | None = None, *, force: bool = False) -> dict:
    """Validate every task under `root` (default: `tasks_root()`) and write
    `<root>/manifest.json` (sha256 per task dir).

    Refuses to overwrite an existing manifest unless `force`, and to freeze an empty corpus.
    """
    root = tasks_root(root)
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


def check_manifest(root: Path | None = None) -> str:
    """The manifest hash when every task dir under `root` (default: `tasks_root()`) matches
    `manifest.json`; otherwise `ManifestError` listing the changed, added and removed tasks."""
    root = tasks_root(root)
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
# Judge (fable via agy): deliverable extraction, scrubbing, schema validation, and cache
# --------------------------------------------------------------------------------------------

# A jet model id: the effort is part of the id, there is no plain `fable`.
DEFAULT_JUDGE_MODEL = "fable-5.1-high"
DEFAULT_JUDGE_REPEAT = 1
DEFAULT_JUDGE_TIMEOUT_S = 300
JUDGMENTS_FILE = "judgments.json"
SKILL_TIMING_FILE = "skill_timing.json"

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["criteria"],
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "pass", "reason"],
                "properties": {
                    "id": {"type": "string"},
                    "pass": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}
JUDGE_SCHEMA_JSON = json.dumps(JUDGE_SCHEMA, separators=(",", ":"))

_PATH_PATTERNS = (
    re.compile(r"[^\s\"'`()\[\]<>]*\.gemini/config/plugins[^\s\"'`()\[\]<>]*", re.IGNORECASE),
    re.compile(
        r"[^\s\"'`()\[\]<>]*\bplugins/[A-Za-z0-9._\-]+/skills/[^\s\"'`()\[\]<>]*", re.IGNORECASE
    ),
    re.compile(r"[^\s\"'`()\[\]<>]*\bskills/[A-Za-z0-9._\-]+/[^\s\"'`()\[\]<>]*", re.IGNORECASE),
    re.compile(r"[^\s\"'`()\[\]<>]*SKILL\.md\b", re.IGNORECASE),
)


def judge_model() -> str:
    """The judge model requested from agy (SKILLBENCH_JUDGE_MODEL, default `fable-5.1-high`)."""
    return str(config.env("SKILLBENCH_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))


def judge_repeat() -> int:
    """How many repeated judge calls to store per rollout (SKILLBENCH_JUDGE_REPEAT, default 1)."""
    n = config.env_int("SKILLBENCH_JUDGE_REPEAT", DEFAULT_JUDGE_REPEAT)
    if n < 1:
        raise config.ConfigError(f"SKILLBENCH_JUDGE_REPEAT must be >= 1, got {n}")
    return n


def rollout_valid(run_dir: Path) -> bool:
    """Whether `run_dir` holds a completed, valid rollout (`skill_timing.json` has `valid=true`)."""
    p = Path(run_dir) / SKILL_TIMING_FILE
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and data.get("valid") is True


def _is_bytecode_path(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    return "__pycache__" in parts or rel_path.endswith((".pyc", ".pyo"))


def _strip_bytecode_diffs(patch: str) -> str:
    """Drop `diff --git` sections for `__pycache__` / `.pyc` files and strip NUL bytes."""
    kept: list[str] = []
    skip = False
    for line in patch.replace("\x00", "").splitlines(keepends=True):
        if line.startswith("diff --git "):
            tokens = line.split()
            a_path = tokens[2][2:] if len(tokens) > 2 and tokens[2].startswith("a/") else ""
            b_path = tokens[3][2:] if len(tokens) > 3 and tokens[3].startswith("b/") else ""
            skip = _is_bytecode_path(a_path) or _is_bytecode_path(b_path)
        if not skip:
            kept.append(line)
    return "".join(kept)


def deliverable_files(run_dir: Path) -> dict[str, str]:
    """Extract agent deliverables from `run_dir` (`final.md`, `diff.patch`, `new_files/`,
    `side_effects/`). Never reads `events.jsonl`, `home/`, or Python bytecode (`__pycache__`)."""
    run_dir = Path(run_dir)
    out: dict[str, str] = {}
    for name in ("final.md", "diff.patch"):
        p = run_dir / name
        if p.is_file():
            raw = p.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
            out[name] = _strip_bytecode_diffs(raw) if name == "diff.patch" else raw
    for sub in ("new_files", "side_effects"):
        base = run_dir / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.is_symlink():
                rel = p.relative_to(base).as_posix()
                if sub == "new_files" and _is_bytecode_path(rel):
                    continue
                out[f"{sub}/{rel}"] = (
                    p.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
                )
    return out


def deliverable_text(run_dir: Path) -> str:
    """Combined readable text of `final.md`, `diff.patch`, `new_files/` and any `side_effects/`."""
    files = deliverable_files(run_dir)
    sections: list[str] = []
    for key in ("final.md", "diff.patch"):
        body = files.get(key, "").strip()
        sections.append(f"### {key}\n{body or '(empty)'}")
    new_keys = [k for k in files if k.startswith("new_files/")]
    if new_keys:
        parts = [f"#### {k}\n{files[k].strip() or '(empty)'}" for k in new_keys]
        sections.append("### new_files\n" + "\n\n".join(parts))
    else:
        sections.append("### new_files\n(none)")
    side_keys = [k for k in files if k.startswith("side_effects/")]
    if side_keys:
        parts = [f"#### {k}\n{files[k].strip() or '(empty)'}" for k in side_keys]
        sections.append("### side_effects\n" + "\n\n".join(parts))
    return "\n\n".join(sections)


VERIFY_FILE = "verify.json"
VERIFY_TIMEOUT_S = 300
_VERIFY_TAIL = 4000


def diff_stats(run_dir: Path) -> dict[str, int] | None:
    """`{files, added, removed}` of `<run_dir>/diff.patch` (None when the file is missing)."""
    p = Path(run_dir) / "diff.patch"
    if not p.is_file():
        return None
    files = added = removed = 0
    cleaned = _strip_bytecode_diffs(p.read_text(encoding="utf-8", errors="replace"))
    for line in cleaned.splitlines():
        if line.startswith("diff --git "):
            files += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"files": files, "added": added, "removed": removed}


def run_verify(
    run_dir: Path,
    task: SkillbenchTask | str | Path,
    *,
    root: Path | None = None,
    timeout: int = VERIFY_TIMEOUT_S,
) -> dict[str, Any] | None:
    """Run the task's deterministic `verify` command against the agent's work tree.

    Copies `<run_dir>/work` to a scratch tree, puts the `verify_restore` files back from the
    fixture (so an agent cannot pass by editing the tests), runs the command there and caches
    `{command, restored, exit_code, passed, timed_out, t_s, output_tail}` in
    `<run_dir>/verify.json`. Returns None when the task has no `verify` or the rollout is
    invalid; never calls a model.
    """
    run_dir = Path(run_dir)
    resolved = _resolve_task(task, root)
    if not resolved.verify or not rollout_valid(run_dir):
        return None
    cache = run_dir / VERIFY_FILE
    command = list(resolved.verify)
    restored = list(resolved.verify_restore)
    if cache.is_file():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("command") == command
            and cached.get("restored") == restored
            and isinstance(cached.get("passed"), bool)
        ):
            return cached
    work = run_dir / "work"
    if not work.is_dir():
        return None
    scratch = run_dir / "verify"
    if scratch.exists():
        shutil.rmtree(scratch)
    tree = scratch / "work"
    home = scratch / "home"
    shutil.copytree(
        work, tree, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__")
    )
    home.mkdir(parents=True)
    for rel in restored:
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolved.fixture / rel, dst)
    env = dict(os.environ)
    env.update(HOME=str(home), PYTHONDONTWRITEBYTECODE="1", EDITOR="true", VISUAL="true")
    start = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            command,
            cwd=tree,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        exit_code: int | None = proc.returncode
        output = proc.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        timed_out = True
        exit_code = None
        output = (e.stdout or b"").decode("utf-8", errors="replace")
    except OSError as e:
        exit_code = None
        output = f"could not run verify command: {e}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    record = {
        "command": command,
        "restored": restored,
        "exit_code": exit_code,
        "passed": exit_code == 0,
        "timed_out": timed_out,
        "t_s": round(time.monotonic() - start, 3),
        "output_tail": output[-_VERIFY_TAIL:],
    }
    cache.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def scrub_deliverable(
    text: str,
    skill_names: Iterable[str] = (),
    *,
    redact_paths: Iterable[str] = (),
) -> str:
    """Replace `SKILL.md` / plugin paths, run-dir paths, and `skill_names` with `[redacted]`.

    Takes no suite itself: callers pass the names (`score` builds them with
    `skillbench_suite.redaction_names` from the rollout's staged plugins and the active suite).
    """
    out = text
    for raw_path in sorted({p for p in redact_paths if p}, key=len, reverse=True):
        out = out.replace(raw_path, "[redacted]")
    for pat in _PATH_PATTERNS:
        out = pat.sub("[redacted]", out)
    names = sorted({s for s in skill_names if s}, key=len, reverse=True)
    for name in names:
        n = re.escape(name)
        for pat in (
            rf"(?<![\w/.\-])/(?:[A-Za-z0-9._\-]+:)?{n}(?![\w-])",
            rf"(?<![\w-]){n}\s+skills?\b",
            rf"skills/{n}(?![\w-])",
            rf"(?<![\w-]){n}(?![\w-])",
        ):
            out = re.sub(pat, "[redacted]", out, flags=re.IGNORECASE)
    return out


def build_judge_prompt(rubric: Iterable[Criterion], scrubbed_deliverable: str) -> str:
    """Judge prompt containing only the rubric criteria and the scrubbed deliverable."""
    crit_lines = [f"- {c.id}: {c.criterion}" for c in rubric]
    return (
        "Evaluate the deliverable below against each rubric criterion.\n"
        "Return a JSON object matching the required schema: an object containing `criteria`, "
        "where each entry has `id` (the criterion id), `pass` (true or false), and `reason` "
        "(a brief explanation grounded in the deliverable).\n"
        "Do not call any tools or inspect any files; grade solely from the Deliverable section "
        "below and call finish immediately.\n\n"
        "## Rubric\n"
        + "\n".join(crit_lines)
        + "\n\n## Deliverable\n"
        + scrubbed_deliverable
    )


def _resolve_task(task: SkillbenchTask | str | Path, root: Path | None) -> SkillbenchTask:
    if isinstance(task, SkillbenchTask):
        return task
    if isinstance(task, Path):
        return load_task(task)
    if isinstance(task, str):
        p = Path(task)
        if (p / TASK_FILE).is_file():
            return load_task(p)
        return load_task(tasks_root(root) / task)
    raise TypeError(f"task must be SkillbenchTask, str, or Path, got {type(task).__name__}")


def _rubric_sha256(rubric: tuple[Criterion, ...]) -> str:
    payload = json.dumps(
        [{"id": c.id, "criterion": c.criterion} for c in rubric],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_judgment(
    events: list[AgyEvent], rubric: tuple[Criterion, ...], requested_model: str
) -> dict[str, Any] | None:
    if not any(ev.is_result for ev in events):
        return None
    actual, mismatch = model_guard(events, requested_model)
    if mismatch is not None:
        return None
    structured = None
    for ev in reversed(events):
        if ev.is_result and ev.structured_output is not None:
            structured = ev.structured_output
            break
    if not isinstance(structured, dict):
        return None
    raw_criteria = structured.get("criteria")
    if not isinstance(raw_criteria, list):
        return None
    expected_ids = [c.id for c in rubric]
    if len(raw_criteria) != len(expected_ids):
        return None
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_criteria:
        if not isinstance(item, dict):
            return None
        cid = item.get("id")
        passed = item.get("pass")
        reason = item.get("reason")
        if (
            not isinstance(cid, str)
            or not isinstance(passed, bool)
            or not isinstance(reason, str)
            or cid in by_id
        ):
            return None
        by_id[cid] = {"id": cid, "pass": passed, "reason": reason}
    if set(by_id) != set(expected_ids):
        return None
    ordered = [by_id[cid] for cid in expected_ids]
    n_passed = sum(1 for c in ordered if c["pass"])
    tokens = [ev.total_tokens for ev in events if ev.total_tokens is not None]
    return {
        "criteria": ordered,
        "n_passed": n_passed,
        "n_criteria": len(ordered),
        "actual_model": actual,
        "total_tokens": tokens[-1] if tokens else None,
    }


def _stage_judge_auth(home: Path) -> list[Path]:
    """Copy the optional SKILLBENCH_AUTH_FILES (none by default) into the judge's HOME."""
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


def _cleanup_judge_auth(home: Path, copied: list[Path]) -> None:
    for dst in copied:
        try:
            dst.unlink()
        except OSError:
            pass
        parent = dst.parent
        while parent != home and parent.is_relative_to(home):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def _run_judge_once(
    run_dir: Path,
    prompt: str,
    rubric: tuple[Criterion, ...],
    *,
    model: str,
    timeout: int,
) -> dict[str, Any] | None:
    binary = resolve_agy_bin()
    if binary is None:
        raise RuntimeError(agy_bin_missing_message())
    judge_root = run_dir / "judge"
    home = judge_root / "home"
    work = judge_root / "work"
    for d in (home, work):
        if d.exists():
            shutil.rmtree(d)
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
        json_schema=JUDGE_SCHEMA_JSON,
    )
    copied = _stage_judge_auth(home)
    try:
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
        _cleanup_judge_auth(home, copied)
    if proc.returncode != 0:
        return None
    text = out_bytes.decode("utf-8", errors="replace")
    events = [ev for ev in map(parse_event, text.splitlines()) if ev is not None]
    return _validate_judgment(events, rubric, model)


def _read_cached_judgments(
    cache_path: Path,
    *,
    cache_key: str,
    model: str,
    rubric_sha: str,
    deliverable_sha: str,
    repeat: int,
) -> dict[str, int] | None:
    if not cache_path.is_file():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(data, dict)
        or data.get("cache_key") != cache_key
        or data.get("judge_model") != model
        or data.get("rubric_sha256") != rubric_sha
        or data.get("deliverable_sha256") != deliverable_sha
    ):
        return None
    judgments = data.get("judgments")
    if not isinstance(judgments, list) or len(judgments) < repeat:
        return None
    first = judgments[0]
    if not isinstance(first, dict):
        return None
    n_passed = first.get("n_passed")
    n_criteria = first.get("n_criteria")
    if (
        isinstance(n_passed, bool)
        or isinstance(n_criteria, bool)
        or not isinstance(n_passed, int)
        or not isinstance(n_criteria, int)
        or n_criteria <= 0
    ):
        return None
    return {"n_passed": n_passed, "n_criteria": n_criteria}


def _redaction_names_for(run_dir: Path, suite: SuiteConfig | None) -> frozenset[str]:
    """Skill names to scrub for one rollout: whatever was staged into its HOME plus the
    active suite's skills and name (the suite is optional: the judge needs no plugins)."""
    if suite is None:
        try:
            suite = active_suite(quiet=True)
        except config.ConfigError:
            suite = None
    staged = run_dir / "home" / PLUGINS_REL
    return redaction_names(staged if staged.is_dir() else None, suite)


def score(
    run_dir: Path,
    task: SkillbenchTask | str | Path,
    *,
    root: Path | None = None,
    timeout: int | None = None,
    suite: SuiteConfig | None = None,
) -> dict | None:
    """Grade one skillbench rollout in `run_dir` with `agy --model $SKILLBENCH_JUDGE_MODEL`.

    Returns `{"n_passed": int, "n_criteria": int}` from the first judgment, or `None` when the
    rollout itself is invalid or any judge call fails (timeout, non-zero exit, malformed output,
    unknown/missing criterion ids, or a disagreeing model field). Valid judgments are cached in
    `<run_dir>/judgments.json` keyed by judge model + rubric sha256 + scrubbed-deliverable sha256.
    The task's deterministic `verify` check (if any) runs first, even when judgments are cached;
    its result goes to `verify.json` and is never shown to the judge.
    """
    run_dir = Path(run_dir)
    if not rollout_valid(run_dir):
        return None
    resolved = _resolve_task(task, root)
    try:
        run_verify(run_dir, resolved, root=root)
    except OSError as e:
        print(f"skillbench: verify failed for {run_dir}: {e}", file=sys.stderr)
    model = judge_model()
    repeat = judge_repeat()
    rubric = resolved.rubric
    rubric_sha = _rubric_sha256(rubric)
    scrubbed = scrub_deliverable(
        deliverable_text(run_dir),
        _redaction_names_for(run_dir, suite) | {resolved.target_skill},
        redact_paths=(str(run_dir), str(run_dir.resolve()), run_dir.name),
    )
    deliverable_sha = hashlib.sha256(scrubbed.encode("utf-8")).hexdigest()
    cache_key = hashlib.sha256(
        f"{model}\0{rubric_sha}\0{deliverable_sha}".encode("utf-8")
    ).hexdigest()
    cache_path = run_dir / JUDGMENTS_FILE

    cached = _read_cached_judgments(
        cache_path,
        cache_key=cache_key,
        model=model,
        rubric_sha=rubric_sha,
        deliverable_sha=deliverable_sha,
        repeat=repeat,
    )
    if cached is not None:
        return cached

    eff_timeout = (
        int(timeout)
        if timeout is not None and int(timeout) > 0
        else (resolved.timeout if resolved.timeout > 0 else DEFAULT_JUDGE_TIMEOUT_S)
    )
    prompt = build_judge_prompt(rubric, scrubbed)
    judgments: list[dict[str, Any]] = []
    for _ in range(repeat):
        j = _run_judge_once(run_dir, prompt, rubric, model=model, timeout=eff_timeout)
        if j is None:
            return None
        judgments.append(j)

    result = {
        "n_passed": judgments[0]["n_passed"],
        "n_criteria": judgments[0]["n_criteria"],
    }
    record = {
        "cache_key": cache_key,
        "judge_model": model,
        "actual_model": judgments[0]["actual_model"],
        "rubric_sha256": rubric_sha,
        "deliverable_sha256": deliverable_sha,
        "score": result,
        "judgments": judgments,
    }
    cache_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------------------------
# BenchmarkAdapter surface
# --------------------------------------------------------------------------------------------

_BASE_LOCK = threading.Lock()
BASE_GEMINI_MD = "<!-- skillbench base harness -->\n"

SKILLBENCH_PROMPTS = BenchPrompts(
    work=(
        "Software engineering assignments in a local git repository: planning, debugging, "
        "reviewing, refactoring and documenting code, and producing structured project "
        "artifacts."
    ),
    domain_word="skill domain",
    deliverable_word="repository changes and final answer",
    inv_levers=(
        "- inspect the repository layout, relevant source files, and git state before editing\n"
        "- follow applicable workflow skills when present and complete every requested step\n"
        "- write requested files to the exact paths expected in the workspace and verify them\n"
        "- summarize findings, decisions, and changes clearly in the final response"
    ),
    dom_levers=(
        "- tailor the depth of analysis, plan structure, or code change to the task type\n"
        "- preserve existing tests and behavior when refactoring unless asked to change them\n"
        "- ground diagnostic claims and design choices in concrete file and line evidence"
    ),
    yardstick=(
        "Did the run inspect the workspace, perform the requested engineering operation, "
        "leave the working tree and any requested files in a clean, usable state, and "
        "provide a complete final response?"
    ),
    selection_checks=(
        "Prefer the candidate that satisfies every stated requirement in the prompt, grounds "
        "its claims in the repository files, leaves clean and working code or documentation "
        "artifacts, and gives a clear, actionable final response."
    ),
)


def _default_runs_root() -> Path:
    raw = config.env("HARNESS2_RUNS_DIR")
    base = Path(raw).expanduser().resolve() if raw else config.RUNS
    return base / "skillbench"


def _env_pilot() -> bool:
    raw = str(config.env("SKILLBENCH_PILOT") or "").strip().lower()
    return raw in ("1", "true", "yes")


def skillbench_spec(*, runs_root: Path | None = None) -> BenchSpec:
    """The skillbench `BenchSpec` on the `agy` substrate."""
    effort = str(config.env("SKILLBENCH_EFFORT") or "high")
    channel = ChannelSpec("claude_cli", exec_model(), effort=effort)
    return BenchSpec(
        name="skillbench",
        runs_root=Path(runs_root) if runs_root is not None else _default_runs_root(),
        install_mode=INSTALL_MODE,
        prompts=SKILLBENCH_PROMPTS,
        channels={"proposer": channel, "selector": channel},
        domain_word="skill domain",
        leak_globs=(RUBRIC_FILE, TASK_FILE, MANIFEST),
        allow_subagents=False,
        proposer_deliverables="contents",
        obs_trunc=50_000,
    )


def _path_segment(model_id: str) -> str:
    """A model id as one results-directory segment (`/` and whitespace replaced)."""
    return re.sub(r"[\s/\\]+", "_", model_id.strip()) or "unknown-executor"


class SkillbenchAdapter:
    """The `BenchmarkAdapter` surface for skillbench on the `agy` substrate.

    Each adapter instance represents one arm of the suite under test (`suite`, from `--suite`
    / `SKILLBENCH_SUITE` / the env fallback): a skill arm (`"with"` / `"without"`, leave-one-
    out per task) or a suite arm (`"baseline"` = no plugins, `"suite-<name>"` = the whole
    named suite -- the suite under test or one of `suites`, the `--suites` extras). `arm` is
    a label or a resolved `ArmSpec`. The task corpus is `root` when given, else
    `tasks_root(suite=suite)`; constructing the adapter verifies `check_manifest` against it
    and refuses on drift. Skill arms schedule `kind = "skill"` tasks only, suite arms
    `kind = "suite"` tasks only. Rollouts live under
    `<runs_root>/results/<executor>/<arm label>/`.
    """

    def __init__(
        self,
        arm: ArmSpec | str,
        substrate: str = "agy",
        split_file: Path | None = None,
        *,
        pilot: bool | None = None,
        root: Path | None = None,
        runs_root: Path | None = None,
        suite: SuiteConfig | None = None,
        suites: Iterable[SuiteConfig | Path | str] = (),
        executor: str | None = None,
    ):
        if substrate != "agy":
            raise ValueError(f"skillbench requires substrate 'agy', got {substrate!r}")
        self.suite = suite if suite is not None else active_suite()
        self.extra_suites: tuple[SuiteConfig, ...] = tuple(
            s if isinstance(s, SuiteConfig) else load_suite(s) for s in suites
        )
        if isinstance(arm, ArmSpec):
            self.arm_spec = arm
        else:
            try:
                self.arm_spec = parse_arm_label(
                    arm, suite=self.suite, extra_suites=self.extra_suites
                )
            except ValueError as e:
                raise ValueError(f"arm: {e}") from None
        self.arm = self.arm_spec.label
        self.arm_kind = self.arm_spec.kind
        self.split_file = split_file
        self._pilot = pilot
        self._root = tasks_root(root, self.suite)
        self.manifest_hash = check_manifest(self._root)
        self._tasks = load_tasks(self._root)
        self.spec = skillbench_spec(runs_root=runs_root)
        self.executor = (
            executor.strip() if executor and executor.strip() else exec_model()
        )
        self._results_root = (
            self.spec.runs_root / "results" / _path_segment(self.executor) / self.arm
        )
        self.substrate = AgySubstrate(
            tasks=self._agy_task,
            runs_root=self._results_root,
        )

    def _task_kind(self) -> str:
        """The task kind this arm schedules: `skill` tasks for skill arms, `suite` tasks for
        suite arms."""
        return "skill" if self.arm_kind == "skill" else "suite"

    def _get_task(self, task_id: str) -> SkillbenchTask:
        if task_id not in self._tasks:
            raise KeyError(f"unknown skillbench task {task_id!r}")
        return self._tasks[task_id]

    def _agy_task(self, task_id: str) -> AgyTask:
        return agy_task(self._get_task(task_id))

    def tasks(self, split: str = "all", *, pilot: bool | None = None) -> list[str]:
        pilot_only = (
            (self._pilot if self._pilot is not None else _env_pilot())
            if pilot is None
            else bool(pilot)
        )
        if split == "pilot":
            split = "all"
            pilot_only = True
        discovered = (
            discover("skillbench")
            if self._root == tasks_root(suite=self.suite)
            else {tid: t.domain for tid, t in self._tasks.items()}
        )
        # Skill arms run `kind = "skill"` tasks, suite arms `kind = "suite"` tasks.
        kind = self._task_kind()
        discovered = {tid: d for tid, d in discovered.items() if self._tasks[tid].kind == kind}
        selected = select(discovered, split, self.split_file, benchmark="skillbench")
        if pilot_only:
            return [tid for tid in selected if self._tasks[tid].pilot]
        return selected

    def domain_of(self, task_id: str) -> str:
        if task_id in self._tasks:
            return self._tasks[task_id].domain
        if "/" in task_id:
            skill, rest = task_id.split("/", 1)
            if _NAME.fullmatch(skill) and _TASK_ID.fullmatch(rest):
                return skill
        raise ValueError(f"malformed skillbench task_id {task_id!r} (expected '<skill>/<task-id>')")

    def reference_domain_order(self) -> list[str]:
        kind = self._task_kind()
        return sorted({t.domain for t in self._tasks.values() if t.kind == kind})

    def task_prompt(self, task_id: str) -> str:
        return build_prompt(self._get_task(task_id))

    def task_files(self, task_id: str) -> Path | None:
        fix = self._get_task(task_id).fixture
        return fix if fix.is_dir() else None

    def _run_dir(self, run_id: str) -> Path:
        return self._results_root / run_id

    def materialise_ctx(self, task: SkillbenchTask | str) -> dict[str, str]:
        """The `materialise` ctx for one task of this arm: the arm label, the task's target
        skill (skill arms) and, for a `suite-<name>` arm of a further suite, its path."""
        resolved = self._get_task(task) if isinstance(task, str) else task
        ctx = {"arm": self.arm, "target_skill": resolved.target_skill}
        staged = self.arm_spec.suite
        if (
            self.arm_kind == "suite"
            and staged is not None
            and staged is not self.suite
            and staged.path is not None
        ):
            ctx["suite_path"] = str(staged.path)
        return ctx

    def _output_fingerprint(self, run_dir: Path) -> str:
        run_dir = Path(run_dir)
        h = hashlib.sha1()
        for name in (SKILL_TIMING_FILE, "final.md", "diff.patch"):
            p = run_dir / name
            if p.is_file():
                h.update(name.encode() + b"\0" + p.read_bytes() + b"\0")
        h.update(b"new_files\0" + RM.fingerprint(run_dir / "new_files").encode() + b"\0")
        h.update(b"side_effects\0" + RM.fingerprint(run_dir / "side_effects").encode() + b"\0")
        return h.hexdigest()[:16]

    def artifacts_intact(self, run_dir: Path) -> bool:
        run_dir = Path(run_dir)
        if not (
            (run_dir / SKILL_TIMING_FILE).is_file()
            and (run_dir / "final.md").is_file()
            and (run_dir / "diff.patch").is_file()
            and (run_dir / "events.jsonl").is_file()
            and (run_dir / "new_files").is_dir()
        ):
            return False
        marker = RM.read_mark(run_dir)
        if marker is not None and "output_fingerprint" in marker:
            return marker["output_fingerprint"] == self._output_fingerprint(run_dir)
        return True

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        task = self._get_task(task_id)
        out = self._run_dir(run_id)
        mark = RM.read_mark(out) or {}
        if (
            RM.is_complete(out, harness_dir=harness_dir)
            and mark.get("arm") == self.arm
            and mark.get("target_skill") == task.target_skill
            and self.artifacts_intact(out)
        ):
            return out
        src = Path(harness_dir) if harness_dir is not None else self.base_harness()
        install = self.substrate.materialise(src, self.spec, self.materialise_ctx(task))
        eff_timeout = (
            int(timeout) if timeout is not None and int(timeout) > 0 else task.timeout
        )
        try:
            dirs = self.substrate.execute(task_id, install, [run_id], eff_timeout)
        finally:
            shutil.rmtree(install.staging_dir, ignore_errors=True)
        out = dirs[0]
        RM.mark_complete(
            out,
            run_id=run_id,
            harness_dir=harness_dir,
            extra={
                "arm": self.arm,
                "target_skill": task.target_skill,
                "output_fingerprint": self._output_fingerprint(out),
            },
        )
        return out

    def delivered(self, run_dir: Path) -> bool:
        run_dir = Path(run_dir)
        if not rollout_valid(run_dir):
            return False
        final_p = run_dir / "final.md"
        if not final_p.is_file():
            return False
        try:
            return bool(final_p.read_text(encoding="utf-8", errors="replace").strip())
        except OSError:
            return False

    def trajectory(self, run_dir: Path, *, obs_trunc: int = 50_000) -> str:
        stream = Path(run_dir) / "events.jsonl"
        if not stream.is_file():
            return f"(no trajectory recorded for {Path(run_dir).name})"
        return self.substrate.read_trajectory(stream, obs_trunc=obs_trunc, max_chars=10**9)

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        return deliverable_files(run_dir)

    def deliverable_text(self, run_dir: Path) -> str:
        return deliverable_text(run_dir)

    def base_harness(self) -> Path:
        d = self.spec.runs_root / f"_base_harness_{os.getpid()}"
        with _BASE_LOCK:
            if not HT.is_tree(AGY_SURFACE, d):
                HT.write_tree(AGY_SURFACE, {"GEMINI.md": BASE_GEMINI_MD}, d, stubs=False)
        return d

    def validate_tree(self, tree: dict[str, str]) -> None:
        if not isinstance(tree, dict) or not any(k in tree for k in AGY_SURFACE.always_on):
            raise ValueError("skillbench harness tree must define GEMINI.md")
        bad = [k for k in tree if not HT.is_allowed(AGY_SURFACE, k)]
        if bad:
            raise ValueError(f"path(s) off the agy editable surface: {bad}")
        HT.validate(
            AGY_SURFACE,
            tree,
            allow_subagents=self.spec.allow_subagents,
            reserved_scripts=HT.RESERVED_SCRIPTS | self.spec.reserved_scripts,
        )

    def preflight(self) -> None:
        self.substrate.preflight()
        if self.arm_spec.suite is not None and self.arm_spec.suite is not self.suite:
            checked_plugins(self.arm_spec.suite)
        check_manifest(self._root)
        judge_repeat()

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        task = self._tasks.get(task_id, task_id)
        return score(
            run_dir,
            task,
            root=self._root,
            suite=self.arm_spec.suite or self.suite,
        )


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
        p.add_argument(
            "--root",
            type=Path,
            default=None,
            help="task corpus root (default: the active suite's tasks_root, else "
            "$SKILLBENCH_TASKS_ROOT, else the bundled corpus)",
        )
    args = ap.parse_args(argv)
    root = tasks_root(args.root)
    try:
        if args.cmd == "freeze":
            m = freeze(root, force=args.force)
            print(f"froze {len(m['tasks'])} task(s) -> {root / MANIFEST} ({m['hash'][:12]})")
        else:
            h = check_manifest(root)
            print(f"ok: {root / MANIFEST} matches ({h[:12]})")
    except (TaskError, ManifestError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
