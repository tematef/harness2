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

Task model and agy judge (the benchmark adapter builds on both):

    load_task          <root>/<skill>/<task-id>/ -> SkillbenchTask (task.toml + rubric.json + fixture/)
    load_tasks         every task under a corpus root, keyed `<skill>/<task-id>`
    check_prompt       explicit skill invocations / forbidden terms found in a prompt
    build_prompt       task prompt + the simulated-user preamble (identical in both arms)
    freeze             sha256 per task dir -> <root>/manifest.json
    check_manifest     refuses a corpus that drifted from its manifest
    deliverable_files  final.md, diff.patch, new_files/, side_effects/ from a rollout dir
    deliverable_text   combined deliverable text for a rollout dir
    scrub_deliverable  redacts skill names, SKILL.md paths and plugin paths to `[redacted]`
    score              runs the isolated agy judge (`--model $SKILLBENCH_JUDGE_MODEL --json-schema`)
                       and caches judgments in `<run_dir>/judgments.json`

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
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .. import config
from ..substrate.agy import (
    AgyEvent,
    AgyTask,
    _kill_group,
    agy_argv,
    agy_bin,
    auth_files,
    model_guard,
    parse_event,
    resolve_agy_bin,
)

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
_LOADED_RUBRIC_HASH: dict[tuple[Path, tuple[Criterion, ...]], str] = {}

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
    rubric_path = task_dir / RUBRIC_FILE
    rubric = _load_rubric(rubric_path)
    if not (task_dir / FIXTURE_DIR).is_dir():
        raise TaskError(f"{task_dir / FIXTURE_DIR}: missing (the repo copied into each rollout)")
    _LOADED_RUBRIC_HASH[(task_dir, rubric)] = hashlib.sha256(rubric_path.read_bytes()).hexdigest()
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
# Judge (fable via agy): deliverable extraction, scrubbing, schema validation, and cache
# --------------------------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL = "fable"
DEFAULT_JUDGE_REPEAT = 1
DEFAULT_JUDGE_TIMEOUT_S = 300
JUDGMENTS_FILE = "judgments.json"
SKILL_TIMING_FILE = "skill_timing.json"

KNOWN_SKILLS = frozenset(
    {
        "backlog",
        "brainstorm",
        "clarify",
        "code-agentic-refactor",
        "code-review-best-practices",
        "code-review-refactor",
        "dialectic",
        "exec",
        "git-review",
        "learn",
        "make-plan",
        "md-copy",
        "pr",
        "root-cause-investigator",
        "skill-eval",
        "txt-copy",
        "wiki-builder",
        "writing-style",
        "wrong",
    }
)

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
    """The judge model requested from agy (SKILLBENCH_JUDGE_MODEL, default `fable`)."""
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


def _discovered_skills() -> set[str]:
    names = set(KNOWN_SKILLS)
    raw = config.env("SKILLBENCH_PLUGINS_DIR")
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir():
            names.update(s.parent.name for s in p.glob("*/skills/*/SKILL.md"))
    return names


def deliverable_files(run_dir: Path) -> dict[str, str]:
    """Extract agent deliverables from `run_dir` (`final.md`, `diff.patch`, `new_files/`,
    `side_effects/`). Never reads `events.jsonl` or `home/`."""
    run_dir = Path(run_dir)
    out: dict[str, str] = {}
    for name in ("final.md", "diff.patch"):
        p = run_dir / name
        if p.is_file():
            out[name] = p.read_text(encoding="utf-8", errors="replace")
    for sub in ("new_files", "side_effects"):
        base = run_dir / sub
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.is_symlink():
                rel = p.relative_to(base).as_posix()
                out[f"{sub}/{rel}"] = p.read_text(encoding="utf-8", errors="replace")
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


def scrub_deliverable(
    text: str,
    skill_names: Iterable[str] = (),
    *,
    redact_paths: Iterable[str] = (),
) -> str:
    """Replace `SKILL.md` / plugin paths, run-dir paths, and skill names with `[redacted]`."""
    out = text
    for raw_path in sorted({p for p in redact_paths if p}, key=len, reverse=True):
        out = out.replace(raw_path, "[redacted]")
    for pat in _PATH_PATTERNS:
        out = pat.sub("[redacted]", out)
    names = sorted(_discovered_skills() | {s for s in skill_names if s}, key=len, reverse=True)
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
        "(a brief explanation grounded in the deliverable).\n\n"
        "## Rubric\n"
        + "\n".join(crit_lines)
        + "\n\n## Deliverable\n"
        + scrubbed_deliverable
    )


def _resolve_task(task: SkillbenchTask | str | Path, root: Path) -> SkillbenchTask:
    if isinstance(task, SkillbenchTask):
        return task
    if isinstance(task, Path):
        return load_task(task)
    if isinstance(task, str):
        p = Path(task)
        if (p / TASK_FILE).is_file():
            return load_task(p)
        return load_task(Path(root) / task)
    raise TypeError(f"task must be SkillbenchTask, str, or Path, got {type(task).__name__}")


def _effective_rubric(task: SkillbenchTask) -> tuple[Criterion, ...]:
    """Use updated on-disk `rubric.json` when `task` came from `load_task` and the file changed."""
    rubric_path = task.dir / RUBRIC_FILE if task.dir != Path() else None
    loaded_hash = _LOADED_RUBRIC_HASH.get((task.dir, task.rubric))
    if rubric_path is not None and loaded_hash is not None and rubric_path.is_file():
        try:
            current_hash = hashlib.sha256(rubric_path.read_bytes()).hexdigest()
            if current_hash != loaded_hash:
                return _load_rubric(rubric_path)
        except (OSError, TaskError):
            pass
    return task.rubric


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
    if not config.env("SKILLBENCH_AUTH_FILES"):
        return []
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
        raise RuntimeError(
            f"agy binary {agy_bin()!r} not found or not executable: install AGY/Jetski or set "
            f"HARNESS2_AGY_BIN to the agy (or jetski) executable"
        )
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
        effort=config.env("SKILLBENCH_EFFORT"),
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


def score(
    run_dir: Path,
    task: SkillbenchTask | str | Path,
    *,
    root: Path = TASKS_ROOT,
    timeout: int | None = None,
) -> dict | None:
    """Grade one skillbench rollout in `run_dir` with `agy --model $SKILLBENCH_JUDGE_MODEL`.

    Returns `{"n_passed": int, "n_criteria": int}` from the first judgment, or `None` when the
    rollout itself is invalid or any judge call fails (timeout, non-zero exit, malformed output,
    unknown/missing criterion ids, or a disagreeing model field). Valid judgments are cached in
    `<run_dir>/judgments.json` keyed by judge model + rubric sha256 + scrubbed-deliverable sha256.
    """
    run_dir = Path(run_dir)
    if not rollout_valid(run_dir):
        return None
    resolved = _resolve_task(task, root)
    model = judge_model()
    repeat = judge_repeat()
    rubric = _effective_rubric(resolved)
    rubric_sha = _rubric_sha256(rubric)
    scrubbed = scrub_deliverable(
        deliverable_text(run_dir),
        [resolved.target_skill],
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
