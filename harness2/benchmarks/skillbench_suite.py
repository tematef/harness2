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

"""Suite config: the one place anything specific to a benchmarked skill set lives.

A *suite* is a plugin checkout (`plugins_dir`) plus the plugins of it that are staged into
every rollout. `suite.toml` describes one::

    name = "my-suite"
    plugins_dir = "~/projects/my-suite/plugins"   # a checkout; `~` and relative paths resolve
    plugins = ["planning", "review"]              # or "*" = every dir holding a plugin.json
    tasks_root = "~/projects/my-suite/benchmarks/skillbench"   # optional external corpus
    banned_tools = ["AskUserQuestion", "EnterPlanMode", "TodoWrite"]   # optional
    [families]                                    # optional, suite-mode groupings
    planning = ["brainstorm", "make-plan", "exec"]
    [matrix]                                      # optional, model matrix defaults
    executors = ["opus-5.5-high", "gemini-3.8-flash-high"]
    [matrix.judge_for]                            # keyed by executor model-id prefix
    "opus-" = "gemini-3.8-flash-high"
    "gemini-" = "fable-5.1-high"

Public surface:

    SuiteConfig        the parsed, validated suite (frozen dataclass)
    load_suite         suite.toml -> SuiteConfig (raises SuiteError on any defect)
    env_suite          the deprecated env fallback (SKILLBENCH_PLUGINS_DIR / SKILLBENCH_PLUGINS)
    active_suite       `path` -> SKILLBENCH_SUITE -> env fallback
    discover_plugins   every plugin directory (holds a plugin.json) under a checkout
    skill_names        every `skills/<name>/SKILL.md` under the given plugins
    redaction_names    judge scrubber set: every staged skill name + the suite name
    skill_asks_user    whether a SKILL.md's frontmatter `allowed-tools` lists `ask_question`
    compute_suite_hash sha256 over the staged plugin files (what a rollout actually sees)

Nothing here imports the substrate or the benchmark: both import this module.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import config

SUITE_ENV = "SKILLBENCH_SUITE"
PLUGINS_DIR_ENV = "SKILLBENCH_PLUGINS_DIR"
PLUGINS_ENV = "SKILLBENCH_PLUGINS"
PLUGIN_MANIFEST = "plugin.json"
SKILL_FILE = "SKILL.md"
ASK_TOOL = "ask_question"

# Tool names a candidate SKILL.md must not mention (Claude Code names; AGY has its own). A
# suite overrides this with `banned_tools`.
DEFAULT_BANNED_TOOLS: tuple[str, ...] = ("AskUserQuestion", "EnterPlanMode", "TodoWrite")

_SUITE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")
_PLUGIN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")
_SKILL_NAME = re.compile(r"[a-z0-9][a-z0-9_\-]*")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")
_HASH_IGNORED = frozenset({"__pycache__", ".git"})
_TOP_KEYS = frozenset(
    {"name", "plugins_dir", "plugins", "tasks_root", "banned_tools", "families", "matrix"}
)
_MATRIX_KEYS = frozenset({"executors", "judge_for"})

_FRONTMATTER = re.compile(r"\A\ufeff?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_ALLOWED_TOOLS = re.compile(r"(?im)^allowed[-_]tools\s*:\s*(.*)$")

_env_note_shown = False


class SuiteError(config.ConfigError):
    """A suite config is malformed or names things that do not exist; the message says what."""


@dataclass(frozen=True)
class SuiteConfig:
    """One validated suite. `plugins` is the ordered tuple of plugin directory names staged
    into rollouts; `path` is the `suite.toml` it came from (`None` for the env fallback)."""

    name: str
    plugins_dir: Path
    plugins: tuple[str, ...]
    tasks_root: Path | None = None
    banned_tools: tuple[str, ...] = DEFAULT_BANNED_TOOLS
    families: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    executors: tuple[str, ...] = ()
    judge_for: Mapping[str, str] = field(default_factory=dict)
    path: Path | None = None
    source: str = "file"

    @cached_property
    def suite_hash(self) -> str:
        """Digest of every staged plugin file (path + bytes); moves when the checkout does."""
        return compute_suite_hash(self.plugins_dir, self.plugins)

    def skill_files(self) -> dict[str, Path]:
        """`skill name -> SKILL.md path` over the suite's plugins (first plugin wins)."""
        return skill_files(self.plugins_dir, self.plugins)

    def skill_names(self) -> frozenset[str]:
        return frozenset(self.skill_files())

    def judge_for_executor(self, executor: str) -> str:
        """The judge paired with `executor` by the longest matching `judge_for` prefix."""
        prefix = _judge_prefix(self.judge_for, executor)
        if prefix is None:
            raise SuiteError(
                f"suite {self.name!r}: no [matrix.judge_for] prefix matches executor "
                f"{executor!r}; add e.g. \"{_family_prefix(executor)}\" = \"<judge model id>\""
            )
        judge = self.judge_for[prefix]
        fam = _family_prefix(executor)
        if judge.startswith(prefix) or judge.startswith(fam) or judge == executor:
            shared = prefix if judge.startswith(prefix) else fam
            raise SuiteError(
                f"suite {self.name!r}: judge {judge!r} shares the prefix {shared!r} with "
                f"executor {executor!r}; a model must not grade its own family"
            )
        return judge

    def describe(self) -> str:
        where = str(self.path) if self.path is not None else f"env ({PLUGINS_DIR_ENV})"
        return f"{self.name} [{', '.join(self.plugins)}] from {where}"


# --------------------------------------------------------------------------------------------
# Discovery, hashing, frontmatter
# --------------------------------------------------------------------------------------------


def discover_plugins(plugins_dir: Path) -> tuple[str, ...]:
    """Sorted names of every `<plugins_dir>/<name>/plugin.json` directory (symlinks followed)."""
    root = Path(plugins_dir)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            p.name
            for p in root.iterdir()
            if p.is_dir() and (p / PLUGIN_MANIFEST).is_file() and _PLUGIN_NAME.fullmatch(p.name)
        )
    )


def skill_files(plugins_dir: Path, plugins: Iterable[str]) -> dict[str, Path]:
    """`skill name -> SKILL.md` for every `skills/<name>/SKILL.md` under the listed plugins."""
    out: dict[str, Path] = {}
    root = Path(plugins_dir)
    for plugin in plugins:
        skills = root / plugin / "skills"
        if not skills.is_dir():
            continue
        for d in sorted(skills.iterdir()):
            md = d / SKILL_FILE
            if d.is_dir() and md.is_file() and _SKILL_NAME.fullmatch(d.name):
                out.setdefault(d.name, md)
    return out


def skill_names(plugins_dir: Path, plugins: Iterable[str] | None = None) -> frozenset[str]:
    """Every skill directory name under `plugins_dir` (all plugins when `plugins` is None)."""
    root = Path(plugins_dir)
    if plugins is None:
        plugins = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    return frozenset(skill_files(root, plugins))


def redaction_names(
    staged_plugins_dir: Path | None, suite: SuiteConfig | None = None
) -> frozenset[str]:
    """What the judge scrubber redacts: every `skills/<name>/` under the staged plugins (any
    plugin), every skill of `suite` (so a leave-one-out arm still hides the omitted skill's
    name) and the suite name itself."""
    names: set[str] = set()
    if staged_plugins_dir is not None:
        names.update(skill_names(staged_plugins_dir))
    if suite is not None:
        names.update(suite.skill_names())
        names.add(suite.name)
    return frozenset(n for n in names if n)


def compute_suite_hash(plugins_dir: Path, plugins: Iterable[str]) -> str:
    """sha256 (16 hex chars) over `<plugin>/<rel path>` + bytes of every file that staging
    copies: symlinks followed (staging resolves them), `__pycache__`, `.pyc` and `.git`
    skipped (staging drops them)."""
    root = Path(plugins_dir)
    h = hashlib.sha256()
    for plugin in sorted(set(plugins)):
        entries: list[tuple[str, bytes]] = []
        base = root / plugin
        for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
            dirnames[:] = sorted(d for d in dirnames if d not in _HASH_IGNORED)
            here = Path(dirpath)
            for name in sorted(filenames):
                if name.endswith((".pyc", ".pyo")):
                    continue
                p = here / name
                if not p.is_file():
                    continue
                rel = f"{plugin}/{p.relative_to(base).as_posix()}"
                entries.append((rel, hashlib.sha256(p.read_bytes()).digest()))
        for rel, digest in sorted(entries):
            h.update(rel.encode() + b"\0" + digest + b"\0")
    return h.hexdigest()[:16]


def skill_frontmatter(skill_md_text: str) -> str:
    """The YAML block between the leading `---` fences (empty when there is none)."""
    m = _FRONTMATTER.match(skill_md_text or "")
    return m.group(1) if m else ""


def skill_asks_user(skill_md_text: str) -> bool:
    """True when the frontmatter's `allowed-tools` lists `ask_question`. Only the frontmatter
    counts: a skill whose body merely mentions the tool is not an asking skill."""
    fm = skill_frontmatter(skill_md_text)
    m = _ALLOWED_TOOLS.search(fm)
    if not m:
        return False
    raw = m.group(1).strip().strip("[]")
    tools = {t.strip().strip("'\"").lower() for t in raw.split(",")}
    return ASK_TOOL in tools


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


def _expand(raw: str, base: Path) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (base / p)


def _str_list(value: Any, key: str, where: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        raise SuiteError(f"{where}: {key} must be a list of strings")
    bad = [v for v in value if not pattern.fullmatch(v)]
    if bad:
        raise SuiteError(f"{where}: {key} has invalid name(s) {bad}")
    return tuple(dict.fromkeys(value))


def _judge_prefix(judge_for: Mapping[str, str], executor: str) -> str | None:
    matches = [k for k in judge_for if executor.startswith(k)]
    return max(matches, key=len) if matches else None


def _family_prefix(model_id: str) -> str:
    return model_id.split("-", 1)[0] + "-"


def _resolve_plugins(
    plugins_dir: Path, raw: Any, where: str
) -> tuple[str, ...]:
    if raw is None or raw == "*":
        found = discover_plugins(plugins_dir)
        if not found:
            raise SuiteError(
                f"{where}: plugins = \"*\" found no directory with a {PLUGIN_MANIFEST} under "
                f"{plugins_dir}; point plugins_dir at a plugin checkout or list the plugins"
            )
        return found
    names = _str_list(raw, "plugins", where, _PLUGIN_NAME)
    if not names:
        raise SuiteError(f'{where}: plugins must be a non-empty list or "*"')
    missing = [n for n in names if not (plugins_dir / n).is_dir()]
    if missing:
        raise SuiteError(
            f"{where}: plugin(s) {missing} not found under {plugins_dir}; fix `plugins` or "
            f"update the checkout"
        )
    return names


def _check_plugins_dir(plugins_dir: Path, where: str) -> Path:
    if not plugins_dir.is_dir():
        raise SuiteError(
            f"{where}: plugins_dir {str(plugins_dir)!r} is not a directory; clone the plugin "
            f"suite and point it at its plugins/ directory"
        )
    return plugins_dir


def _load_families(
    raw: Any, where: str, known_skills: frozenset[str]
) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SuiteError(f"{where}: [families] must be a table of name = [skill, ...]")
    out: dict[str, tuple[str, ...]] = {}
    for fam, skills in raw.items():
        if not _SKILL_NAME.fullmatch(str(fam)):
            raise SuiteError(f"{where}: family name {fam!r} must match {_SKILL_NAME.pattern}")
        names = _str_list(skills, f"families.{fam}", where, _SKILL_NAME)
        if not names:
            raise SuiteError(f"{where}: families.{fam} must list at least one skill")
        unknown = [s for s in names if s not in known_skills]
        if unknown:
            raise SuiteError(
                f"{where}: families.{fam} names skill(s) {unknown} that no staged plugin "
                f"provides (known: {sorted(known_skills)})"
            )
        out[str(fam)] = names
    return out


def _load_matrix(raw: Any, where: str) -> tuple[tuple[str, ...], dict[str, str]]:
    if raw is None:
        return (), {}
    if not isinstance(raw, dict):
        raise SuiteError(f"{where}: [matrix] must be a table")
    unknown = sorted(set(raw) - _MATRIX_KEYS)
    if unknown:
        raise SuiteError(f"{where}: unknown [matrix] key(s) {unknown}; allowed: executors, judge_for")
    executors = _str_list(raw.get("executors", []), "matrix.executors", where, _MODEL_ID)
    jf_raw = raw.get("judge_for", {})
    if not isinstance(jf_raw, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not k or not v
        for k, v in jf_raw.items()
    ):
        raise SuiteError(f"{where}: [matrix.judge_for] must map model-id prefixes to model ids")
    judge_for = dict(jf_raw)
    for ex in executors:
        prefix = _judge_prefix(judge_for, ex)
        if prefix is None:
            raise SuiteError(
                f"{where}: executor {ex!r} has no [matrix.judge_for] prefix; add e.g. "
                f"\"{_family_prefix(ex)}\" = \"<judge model id>\""
            )
        fam = _family_prefix(ex)
        judge = judge_for[prefix]
        if judge.startswith(prefix) or judge.startswith(fam) or judge == ex:
            raise SuiteError(
                f"{where}: judge_for[{prefix!r}] = {judge!r} shares the executor's "
                f"prefix; a model must not grade its own family"
            )
    return executors, judge_for


def load_suite(path: Path | str) -> SuiteConfig:
    """Parse and validate a `suite.toml`. Relative paths resolve against the file's directory,
    `~` is expanded, every listed plugin must exist and `"*"` must find at least one plugin."""
    path = Path(path).expanduser()
    where = str(path)
    if not path.is_file():
        raise SuiteError(f"{where}: suite file not found")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise SuiteError(f"{where}: not valid TOML ({e})") from None
    unknown = sorted(set(data) - _TOP_KEYS)
    if unknown:
        raise SuiteError(
            f"{where}: unknown key(s) {unknown}; allowed: {', '.join(sorted(_TOP_KEYS))}"
        )
    name = data.get("name")
    if not isinstance(name, str) or not _SUITE_NAME.fullmatch(name):
        raise SuiteError(f"{where}: name must be a short identifier like 'my-suite'")
    raw_dir = data.get("plugins_dir")
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        raise SuiteError(f"{where}: plugins_dir is required (a plugin checkout's plugins/ dir)")
    base = path.resolve().parent
    plugins_dir = _check_plugins_dir(_expand(raw_dir.strip(), base), where)
    plugins = _resolve_plugins(plugins_dir, data.get("plugins"), where)
    tasks_root: Path | None = None
    raw_tasks = data.get("tasks_root")
    if raw_tasks is not None:
        if not isinstance(raw_tasks, str) or not raw_tasks.strip():
            raise SuiteError(f"{where}: tasks_root must be a path string")
        tasks_root = _expand(raw_tasks.strip(), base)
    banned = (
        DEFAULT_BANNED_TOOLS
        if data.get("banned_tools") is None
        else _str_list(data["banned_tools"], "banned_tools", where, _PLUGIN_NAME)
    )
    known = frozenset(skill_files(plugins_dir, plugins))
    families = _load_families(data.get("families"), where, known)
    executors, judge_for = _load_matrix(data.get("matrix"), where)
    return SuiteConfig(
        name=name,
        plugins_dir=plugins_dir,
        plugins=plugins,
        tasks_root=tasks_root,
        banned_tools=banned,
        families=families,
        executors=executors,
        judge_for=judge_for,
        path=path,
        source="file",
    )


def env_suite(*, quiet: bool = False) -> SuiteConfig:
    """The suite implied by SKILLBENCH_PLUGINS_DIR (+ optional SKILLBENCH_PLUGINS comma list;
    unset = every plugin with a plugin.json). Deprecated in favour of a `suite.toml`: a note
    is printed once per process unless `quiet`."""
    global _env_note_shown
    raw = config.env(PLUGINS_DIR_ENV)
    if not raw:
        raise config.ConfigError(
            f"no skillbench suite configured: pass --suite <suite.toml>, set {SUITE_ENV}, or "
            f"(deprecated) point {PLUGINS_DIR_ENV} at a plugin suite's plugins/ checkout"
        )
    plugins_dir = Path(str(raw)).expanduser()
    if not plugins_dir.is_dir():
        raise config.ConfigError(
            f"{PLUGINS_DIR_ENV}={raw!r} is not a directory: clone the plugin suite and point "
            f"it at its plugins/ directory (or write a suite.toml and pass --suite)"
        )
    raw_list = config.env(PLUGINS_ENV)
    if raw_list:
        names = tuple(dict.fromkeys(n.strip() for n in str(raw_list).split(",") if n.strip()))
        bad = [n for n in names if not _PLUGIN_NAME.fullmatch(n)]
        if bad or not names:
            raise config.ConfigError(
                f"{PLUGINS_ENV} must be a comma list of plugin directory names, got {raw_list!r}"
            )
        missing = [n for n in names if not (plugins_dir / n).is_dir()]
        if missing:
            raise config.ConfigError(
                f"allowlisted plugin(s) {missing} not found in {PLUGINS_DIR_ENV}={plugins_dir}; "
                f"fix {PLUGINS_ENV} or update the checkout"
            )
    else:
        names = discover_plugins(plugins_dir)
        if not names:
            raise config.ConfigError(
                f"{PLUGINS_DIR_ENV}={plugins_dir} holds no directory with a {PLUGIN_MANIFEST}; "
                f"list the plugins in {PLUGINS_ENV} or write a suite.toml"
            )
    name = plugins_dir.parent.name if plugins_dir.name == "plugins" else plugins_dir.name
    if not _SUITE_NAME.fullmatch(name or ""):
        name = "suite"
    if not quiet and not _env_note_shown:
        _env_note_shown = True
        print(
            f"skillbench: no --suite / {SUITE_ENV} given; using the env fallback suite "
            f"{name!r} from {PLUGINS_DIR_ENV} (deprecated: describe the suite in a suite.toml)",
            file=sys.stderr,
        )
    return SuiteConfig(
        name=name, plugins_dir=plugins_dir, plugins=names, path=None, source="env"
    )


def active_suite(path: Path | str | None = None, *, quiet: bool = False) -> SuiteConfig:
    """`path` -> `$SKILLBENCH_SUITE` -> env fallback (`env_suite`)."""
    if path is not None:
        return load_suite(path)
    raw = config.env(SUITE_ENV)
    if raw:
        return load_suite(str(raw))
    return env_suite(quiet=quiet)
