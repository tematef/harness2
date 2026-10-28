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

"""AgySubstrate: AGY / Jetski (`agy -p ... --output-format stream-json`) as a solver substrate.

The stream-json decoder:

    parse_event        one stdout line -> AgyEvent | None (non-JSON / non-event lines skipped)
    read_events        events.jsonl -> list[AgyEvent]
    skill_timing       first read of skills/<target>/SKILL.md (any skill when target is None)
                       -> {invoked, t_skill_s, step_idx, skills_read}
    render_trajectory  AgyEvent list -> the shared [ASSISTANT]/[ACTION]/... trajectory grammar

Arms (`ArmSpec`, `parse_arm_label`): two kinds share one label grammar

    skill   `with` / `without`   leave-one-out of `target_skill` in the suite under test
    suite   `baseline`           no plugins at all (AGY's built-in behaviour)
            `suite-<name>`       the whole named suite (the active one or a `--suites` extra)

Staging (install mode `agy_home`, one Install per arm):

    AgySubstrate.materialise  the arm's plugins (nothing for `baseline`, the full named suite
                              for `suite-<name>`, leave-one-out for `with` / `without`),
                              copied auth files and side-effect shims -> a HOME template
    prepare_rollout           fresh per-rollout copy of that HOME + the isolated environment

Execution (one rollout per run_id):

    run_rollout               fixture -> `<run>/work` git repo, one headless agy call, then
                              events.jsonl, final.md, diff.patch, new_files/, skill_timing.json
    AgySubstrate.execute      run_rollout for each run_id (task resolver + runs root supplied
                              by the benchmark adapter)
    AgySubstrate.preflight    binary, git, effort value, plugins dir and any listed auth files

The event shape is verified against real `jet` streams (tests/fixtures/agy/real/, see
docs/skillbench-spike.md):

    init         {event: "init", conversation_id, init{model, cwd, tools, permission_mode}}
                 -- the first line; `init.model` is the only place the stream names the model
                 (a full jet id such as `opus-5.5-high`)
    step_update  {step_type, state, tool_name, text_delta, tool_info{name, parameters, output?}}
                 -- step types `user_input`, `unknown`, `agent_response`, `tool`; a skill read
                 is a `read_file` tool step whose `Path` is <HOME>/.gemini/config/plugins/
                 <plugin>/skills/<skill>/SKILL.md; a tool's DONE step repeats its parameters
                 and adds `output`
    result       {response, status, num_turns, duration_seconds, usage.total_tokens,
                 structured_output?} -- the last line; `response` holds the full final answer
                 (the last DONE text delta is often just "\n") and there is no model field

Older shapes stay decodable (`result.model`, `view_file`/`AbsolutePath` reads, DONE tool steps
without `tool_info`). Lines in a run's events.jsonl additionally carry `t`, seconds since the
agy process spawned.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any, Callable, Iterable, Iterator, Mapping

from .. import config
from ..benchmarks.skillbench_suite import SuiteConfig, active_suite, load_suite
from ..core import runmark as RM
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install

# Tools whose call counts as reading a skill (spike question 3: jet reads a skill with an
# ordinary `read_file` of <HOME>/.gemini/config/plugins/<plugin>/skills/<skill>/SKILL.md;
# `view_file` is the older AGY name for the same tool).
READ_TOOLS = frozenset({"view_file", "read_file"})

_THINKING_STEPS = frozenset({"thinking", "reasoning", "planner_thought"})
_ERROR_STATES = frozenset({"ERROR", "FAILED", "CANCELED", "CANCELLED"})


@dataclass(frozen=True)
class AgyEvent:
    """One decoded agy stream-json event. Absent or wrongly typed fields are `None`."""

    event: str
    step_type: str | None = None
    state: str | None = None
    tool_name: str | None = None
    tool_args: Mapping[str, Any] | None = None
    text: str | None = None
    structured_output: Any = None
    total_tokens: int | None = None
    model: str | None = None
    error: str | None = None
    t: float | None = None
    response: str | None = None
    tool_output: str | None = None

    @property
    def is_result(self) -> bool:
        return self.event == "result"

    @property
    def is_tool(self) -> bool:
        return self.step_type == "tool" or self.tool_name is not None


def _str(v) -> str | None:
    return v if isinstance(v, str) and v else None


def _dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def parse_event(line: str) -> AgyEvent | None:
    """Decode one stream-json line; `None` for blank, non-JSON or non-event lines."""
    line = (line or "").strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not _str(obj.get("event")):
        return None
    t = obj.get("t")
    t = float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else None
    kind = obj["event"]
    su = _dict(obj.get("step_update"))
    res = _dict(obj.get("result"))
    init = _dict(obj.get("init"))
    info = _dict(su.get("tool_info"))
    params = info.get("parameters")
    structured = res.get("structured_output")
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except json.JSONDecodeError:
            pass
    tokens = _dict(res.get("usage")).get("total_tokens")
    err = obj.get("error")
    if isinstance(err, dict):
        err = err.get("message") or json.dumps(err, default=str)[:300]
    return AgyEvent(
        event=kind,
        step_type=_str(su.get("step_type")),
        state=_str(su.get("state")),
        tool_name=_str(su.get("tool_name")) or _str(info.get("name")),
        tool_args=params if isinstance(params, dict) else None,
        text=su.get("text_delta") if isinstance(su.get("text_delta"), str) else None,
        structured_output=structured,
        total_tokens=tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None,
        model=_str(res.get("model")) or _str(obj.get("model")) or _str(init.get("model")),
        error=_str(err) if err is not None else None,
        t=t,
        response=res.get("response") if isinstance(res.get("response"), str) else None,
        tool_output=info.get("output") if isinstance(info.get("output"), str) else None,
    )


def read_events(path: Path) -> list[AgyEvent]:
    """Parse every decodable line of an events.jsonl (missing file -> [])."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [ev for ev in map(parse_event, raw.splitlines()) if ev is not None]


_ANY_SKILL_READ = re.compile(r"(?:^|[/\\])skills[/\\]([a-z0-9][a-z0-9_\-]*)[/\\]SKILL\.md$")


def _under_plugins_dir(raw_path: str, plugins_dir: Path | str | None) -> bool:
    """Whether `raw_path` lies under `plugins_dir` (and exists on disk when `plugins_dir`
    itself exists on disk, so an empty `baseline` plugins dir matches nothing)."""
    if plugins_dir is None:
        return True
    root = Path(plugins_dir)
    cand = Path(raw_path)
    if not (
        cand.is_relative_to(root)
        or (root.exists() and cand.exists() and cand.resolve().is_relative_to(root.resolve()))
    ):
        return False
    if root.exists():
        return cand.is_file()
    return True


def _skill_read(ev: AgyEvent, plugins_dir: Path | str | None = None) -> str | None:
    """The skill whose `SKILL.md` this event reads (`skills/<name>/SKILL.md` under
    `plugins_dir` when given), else None."""
    if ev.tool_name not in READ_TOOLS or not ev.tool_args:
        return None
    for v in ev.tool_args.values():
        if isinstance(v, str):
            m = _ANY_SKILL_READ.search(v)
            if m and _under_plugins_dir(v, plugins_dir):
                return m.group(1)
    return None


def skills_read(
    events: Iterable[AgyEvent], *, plugins_dir: Path | str | None = None
) -> list[str]:
    """Sorted names of every skill whose `SKILL.md` the stream read (re-reads collapsed)."""
    return sorted({name for ev in events if (name := _skill_read(ev, plugins_dir)) is not None})


def skill_timing(
    events: Iterable[AgyEvent],
    target_skill: str | None = None,
    *,
    plugins_dir: Path | str | None = None,
) -> dict:
    """Locate the first read of `skills/<target_skill>/SKILL.md` -- or of any skill's `SKILL.md`
    under `plugins_dir` when `target_skill` is None (suite arms).

    Returns `{invoked, t_skill_s, step_idx}` for a skill arm (`target_skill` given), and adds
    `skills_read` (sorted set of every skill read in the stream) when `target_skill` is None.
    Reads of other skills (for a target), reads outside `plugins_dir` (when given), later
    re-reads and writes to the path do not count as invocation.
    """
    events = list(events)
    hit: dict = {"invoked": False, "t_skill_s": None, "step_idx": None}
    for idx, ev in enumerate(events):
        name = _skill_read(ev, plugins_dir)
        if name is None or (target_skill is not None and name != target_skill):
            continue
        hit = {"invoked": True, "t_skill_s": ev.t, "step_idx": idx}
        break
    if target_skill is None:
        hit["skills_read"] = skills_read(events, plugins_dir=plugins_dir)
    return hit


_TRUNC_FMT = "…(+{n} chars — render truncation)"


def _clip(s, n: int) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\x00", "")
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + _TRUNC_FMT.format(n=len(s) - n)


def render_trajectory(
    events: Iterable[AgyEvent], *, obs_trunc: int, max_chars: int, args_trunc: int = 400
) -> str:
    """Render decoded agy events into the shared trajectory grammar.

    Streamed `ACTIVE` text deltas of one response are merged into a single line; a `DONE`
    response delta gets its own line. A tool call renders as `[ACTION]` when it starts; its
    `DONE` step (which repeats the parameters) renders only its `output`, as `[OBSERVATION]`,
    and a failed step renders `[OBSERVATION ERROR]`.
    """
    lines: list[str] = []
    buf: list[str] = []
    buf_tag = "ASSISTANT"

    def flush() -> None:
        txt = "".join(buf).strip()
        if txt:
            lines.append(f"[{buf_tag}] {_clip(txt, obs_trunc)}")
        buf.clear()

    for ev in events:
        if ev.error is not None:
            flush()
            lines.append(f"[ERROR] {_clip(ev.error, 300)}")
            continue
        if ev.is_tool:
            flush()
            if ev.state in _ERROR_STATES:
                detail = ev.text or ev.state
                lines.append(
                    f"[OBSERVATION ERROR] {ev.tool_name or 'tool'}: {_clip(detail, obs_trunc)}"
                )
            elif ev.state == "DONE":
                if ev.tool_output and ev.tool_output.strip():
                    lines.append(f"[OBSERVATION] {_clip(ev.tool_output.strip(), obs_trunc)}")
            elif ev.tool_args is not None or ev.state == "ACTIVE":
                args = (
                    json.dumps(ev.tool_args, default=str, ensure_ascii=False)
                    if ev.tool_args is not None
                    else ""
                )
                lines.append(f"[ACTION] {ev.tool_name or 'tool'}({_clip(args, args_trunc)})")
            elif ev.text and ev.text.strip():
                lines.append(f"[OBSERVATION] {_clip(ev.text.strip(), obs_trunc)}")
            continue
        if ev.text is not None:
            tag = "THINKING" if ev.step_type in _THINKING_STEPS else "ASSISTANT"
            if tag != buf_tag:
                flush()
                buf_tag = tag
            if ev.state == "DONE":
                flush()
                buf.append(ev.text)
                flush()
            else:
                buf.append(ev.text)
            continue
        if ev.is_result:
            flush()
            if ev.structured_output is not None:
                so = json.dumps(ev.structured_output, default=str, ensure_ascii=False)
                lines.append(f"[OBSERVATION] structured_output: {_clip(so, obs_trunc)}")
    flush()
    rendered = "\n".join(lines).strip()
    if not rendered:
        return "(no parseable events in trajectory)"
    if len(rendered) > max_chars:
        dropped = len(rendered) - max_chars
        rendered = rendered[:max_chars] + "\n" + _TRUNC_FMT.format(n=dropped)
    return rendered


# --------------------------------------------------------------------------------------------
# Arms: two kinds, one label grammar.
# --------------------------------------------------------------------------------------------

INSTALL_MODE = "agy_home"
# Skill arms: leave-one-out of `target_skill` in the suite under test.
ARMS = ("with", "without")
# Suite arms: `baseline` stages no plugins at all; `suite-<name>` stages the whole named suite.
BASELINE_ARM = "baseline"
SUITE_ARM_PREFIX = "suite-"
ARM_KINDS = ("skill", "suite")
ARM_LABEL_HELP = f"with | without | {BASELINE_ARM} | {SUITE_ARM_PREFIX}<name>"
# What the control arm of each treatment label is (report pairing).
CONTROL_OF = {"with": "without", SUITE_ARM_PREFIX + "<name>": BASELINE_ARM}
_SUITE_ARM_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")
_SKILL_NAME = re.compile(r"[a-z0-9][a-z0-9_\-]*")


def arm_kind(label: str) -> str:
    """`"skill"` for `with` / `without`, `"suite"` for `baseline` / `suite-<name>`;
    `ValueError` (naming the grammar) for anything else."""
    if isinstance(label, str):
        if label in ARMS:
            return "skill"
        if label == BASELINE_ARM:
            return "suite"
        if label.startswith(SUITE_ARM_PREFIX) and _SUITE_ARM_NAME.fullmatch(
            label[len(SUITE_ARM_PREFIX) :]
        ):
            return "suite"
    raise ValueError(f"arm label must be one of {ARM_LABEL_HELP}, got {label!r}")


def control_label(label: str) -> str:
    """The control arm a treatment label is compared against (`without` for `with`,
    `baseline` for `suite-<name>`); control labels map to themselves."""
    return "without" if arm_kind(label) == "skill" else BASELINE_ARM


def is_control_label(label: str) -> bool:
    return label in ("without", BASELINE_ARM)


def suite_arm_label(suite_name: str) -> str:
    """`suite-<name>` for a suite."""
    return SUITE_ARM_PREFIX + suite_name


@dataclass(frozen=True)
class ArmSpec:
    """One arm, resolved: its `kind`, its `label`, the suite staged into the rollout HOME
    (`None` for `baseline` = no plugins) and, for skill arms, the leave-one-out
    `target_skill` (set per task with `for_task`)."""

    kind: str
    label: str
    suite: SuiteConfig | None = None
    target_skill: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ARM_KINDS:
            raise ValueError(f"ArmSpec.kind must be one of {ARM_KINDS}, got {self.kind!r}")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError(f"ArmSpec.label must be a non-empty string, got {self.label!r}")

    @property
    def is_control(self) -> bool:
        return is_control_label(self.label)

    @property
    def stages_plugins(self) -> bool:
        return self.suite is not None

    def for_task(self, target_skill: str | None) -> ArmSpec:
        """The same arm for a task: skill arms take the task's `target_skill`, suite arms
        carry none."""
        if self.kind == "suite":
            return self if self.target_skill is None else replace(self, target_skill=None)
        if not isinstance(target_skill, str) or not _SKILL_NAME.fullmatch(target_skill):
            raise ValueError(
                f"arm {self.label!r} requires a non-empty target_skill directory name, got "
                f"{target_skill!r}"
            )
        return replace(self, target_skill=target_skill)


def _as_suite(s: SuiteConfig | Path | str) -> SuiteConfig:
    return s if isinstance(s, SuiteConfig) else load_suite(s)


def parse_arm_label(
    label: str,
    *,
    suite: SuiteConfig | Path | str | None = None,
    extra_suites: Iterable[SuiteConfig | Path | str] = (),
    target_skill: str | None = None,
) -> ArmSpec:
    """`with` / `without` / `baseline` / `suite-<name>` -> `ArmSpec`.

    `suite` is the suite under test (the active suite when None; resolved lazily, so
    `baseline` needs none). `suite-<name>` must name that suite or one of `extra_suites`
    (`--suites`); the suite's own name is checked, not its file name.
    """
    kind = arm_kind(label)
    if kind == "skill":
        main = _as_suite(suite) if suite is not None else active_suite()
        spec = ArmSpec("skill", label, main)
        return spec.for_task(target_skill) if target_skill is not None else spec
    if target_skill not in (None, ""):
        raise ValueError(
            f"target_skill is not valid for suite arm {label!r}, got {target_skill!r}"
        )
    if label == BASELINE_ARM:
        return ArmSpec("suite", label, None)
    name = label[len(SUITE_ARM_PREFIX) :]
    main = _as_suite(suite) if suite is not None else active_suite()
    loaded_extras = [_as_suite(s) for s in extra_suites]
    candidates = [main, *loaded_extras]
    for cand in candidates:
        if cand.name == name:
            return ArmSpec("suite", label, cand)
    raise ValueError(
        f"arm {label!r} names suite {name!r}, but the suite under test is {main.name!r}"
        + (
            f" and the further suites are {[s.name for s in loaded_extras]}"
            if loaded_extras
            else " and no further suite was given"
        )
        + f"; pass --suites <path/to/{name}.toml> to compare a further suite"
    )


# --------------------------------------------------------------------------------------------
# Staging: the arm's plugins, isolated HOME, side-effect shims.
# --------------------------------------------------------------------------------------------

SHIM_TOOLS = ("xclip", "xsel", "wl-copy", "pbcopy", "revdiff")
SIDE_EFFECTS_ENV = "SKILLBENCH_SIDE_EFFECTS_DIR"
# Where agy discovers plugins, relative to HOME (spike question 9).
PLUGINS_REL = Path(".gemini") / "config" / "plugins"
# Where the editable tree's always-on file lands, relative to HOME (AGY's global rules file).
RULES_REL = Path(".gemini") / "GEMINI.md"

# Bases-only surface: the always-on GEMINI.md is installed as AGY's global rules file; the full
# c1..c8 mapping (skills, hooks, scripts as an AGY plugin) is deferred to Option B.
AGY_SURFACE = SurfaceSpec(
    name="agy",
    always_on=("GEMINI.md",),
    always_on_max=24_000,
    always_on_cap_reason="AGY loads ~/.gemini/GEMINI.md into every turn",
    skill_dir="skills",
    skill_name_rule="must_match",
    trigger_hint=(
        'Fix it by adding a trigger line naming `{name}` to GEMINI.md, e.g. "When '
        '<the situation this skill handles>, read the `{name}` skill and follow it."'
    ),
    autotrigger_file="GEMINI.md",
    trigger_line="- When {desc}, read the `{name}` skill and follow it.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=None,
)

# Records stdin instead of touching the clipboard or opening an interactive review. The target
# directory comes from the rollout env, so one shim dir serves every rollout of an install.
_SHIM = """#!/bin/sh
# skillbench side-effect shim for `{tool}`: appends stdin to ${env}/{tool}.txt
# instead of touching the real clipboard or opening an interactive review UI.
for a in "$@"; do
  case "$a" in
    -o|-out|--output) exit 0 ;;
  esac
done
out="${{{env}:-}}"
if [ -z "$out" ]; then
  [ -t 0 ] || cat > /dev/null
  exit 0
fi
mkdir -p "$out"
if [ -t 0 ]; then
  : >> "$out/{tool}.txt"
else
  cat >> "$out/{tool}.txt"
fi
exit 0
"""


def plugins_dir(suite: SuiteConfig | None = None) -> Path:
    """The active suite's plugin checkout (`suite.plugins_dir`)."""
    return (suite or active_suite()).plugins_dir


def plugin_allowlist(suite: SuiteConfig | None = None) -> tuple[str, ...]:
    """The active suite's staged plugin names (`suite.plugins`)."""
    return (suite or active_suite()).plugins


def auth_files() -> tuple[str, ...]:
    """SKILLBENCH_AUTH_FILES as normalised paths relative to `$HOME` (optional, default none).

    Internal Jetski authenticates without any file under `$HOME` (spike question 8), so an
    unset or empty value copies nothing. A listed entry must be a path inside `$HOME`.
    """
    raw = str(config.env("SKILLBENCH_AUTH_FILES") or "")
    out = []
    for item in (s.strip() for s in raw.split(",")):
        if not item:
            continue
        try:
            rel = HT.normalize_rel(item)
        except ValueError:
            rel = ""
        if not rel:
            raise config.ConfigError(
                f"SKILLBENCH_AUTH_FILES entries must be paths relative to $HOME, got {item!r}"
            )
        out.append(rel)
    return tuple(dict.fromkeys(out))


def _target_dirs(root: Path, allowlist: tuple[str, ...], skill: str) -> list[Path]:
    return [
        root / p / "skills" / skill
        for p in allowlist
        if (root / p / "skills" / skill / "SKILL.md").is_file()
    ]


def checked_plugins(suite: SuiteConfig | None = None) -> tuple[Path, tuple[str, ...]]:
    """`(plugins_dir, plugins)` of the active suite after checking every plugin exists."""
    suite = suite or active_suite()
    root, allow = suite.plugins_dir, suite.plugins
    missing = [p for p in allow if not (root / p).is_dir()]
    if missing:
        raise config.ConfigError(
            f"suite {suite.name!r}: plugin(s) {missing} not found under {root}; fix the "
            f"suite's `plugins` or update the checkout"
        )
    return root, allow


def target_skill_file(target_skill: str, suite: SuiteConfig | None = None) -> Path:
    """Path to `skills/<target_skill>/SKILL.md` in the active suite's plugins checkout."""
    root, allow = checked_plugins(suite)
    targets = _target_dirs(root, allow, target_skill)
    if not targets:
        raise ValueError(
            f"target skill {target_skill!r} not found as skills/{target_skill}/SKILL.md in the "
            f"suite's plugins {list(allow)} under {root}"
        )
    return targets[0] / "SKILL.md"


def _copy_plugin(src: Path, dst: Path, omit_skill: str | None = None) -> None:
    """Copy one plugin directory (symlinks resolved, bytecode dropped), without
    `skills/<omit_skill>/` when given."""

    def ignore(d: str, entries: list[str]) -> set[str]:
        drop = {e for e in entries if e == "__pycache__" or e.endswith(".pyc")}
        if omit_skill is not None and Path(d) == src / "skills":
            drop.add(omit_skill)
        return drop

    shutil.copytree(src, dst, symlinks=False, ignore=ignore)


def stage_plugins(
    dest: Path,
    arm: ArmSpec | str,
    *,
    target_skill: str | None = None,
    skill_override: Path | str | None = None,
    suite: SuiteConfig | None = None,
    extra_suites: Iterable[SuiteConfig | Path | str] = (),
) -> list[str]:
    """Copy an arm's plugins (symlinks resolved) into `dest`; returns the omitted paths
    relative to the plugins checkout.

    `arm` is an `ArmSpec` or a label (`with` / `without` take `target_skill`; `suite` is the
    suite under test, default the active one). Arm `baseline` stages nothing (`dest` is
    created empty); `suite-<name>` stages every plugin of the named suite; `with` stages the
    suite under test; `without` omits `skills/<target_skill>/` from every staged plugin that
    has it -- hooks and every other file stay. `skill_override` (a `Path` to a candidate
    `SKILL.md` or raw markdown text) replaces the staged `skills/<target_skill>/SKILL.md` in
    place and is accepted for arm `with` only.
    """
    spec = (
        arm
        if isinstance(arm, ArmSpec)
        else parse_arm_label(
            arm, suite=suite, extra_suites=extra_suites, target_skill=target_skill
        )
    )
    if isinstance(arm, ArmSpec) and target_skill is not None:
        spec = spec.for_task(target_skill)
    dest = Path(dest)
    if skill_override is not None and spec.label != "with":
        raise ValueError(f"skill_override requires arm='with', got arm={spec.label!r}")
    dest.mkdir(parents=True, exist_ok=True)
    if spec.kind == "suite":
        if spec.suite is None:
            return []
        root, allow = checked_plugins(spec.suite)
        for name in allow:
            _copy_plugin(root / name, dest / name)
        return []
    if spec.target_skill is None:
        raise ValueError(f"arm {spec.label!r} needs a target_skill")
    target = spec.target_skill
    root, allow = checked_plugins(spec.suite)
    targets = _target_dirs(root, allow, target)
    if not targets:
        raise ValueError(
            f"target skill {target!r} not found as skills/{target}/SKILL.md in the "
            f"suite's plugins {list(allow)} under {root}"
        )
    omit = set(targets) if spec.label == "without" else set()
    for name in allow:
        src = root / name
        _copy_plugin(src, dest / name, target if (src / "skills" / target) in omit else None)
    if skill_override is not None:
        text = (
            skill_override.read_text(encoding="utf-8")
            if isinstance(skill_override, Path)
            else Path(skill_override).read_text(encoding="utf-8")
            if os.sep in skill_override and Path(skill_override).is_file()
            else str(skill_override)
        )
        for t in targets:
            (dest / t.relative_to(root) / "SKILL.md").write_text(text, encoding="utf-8")
    return sorted(str(t.relative_to(root)) for t in omit)


def checked_auth_files() -> tuple[Path, tuple[str, ...]]:
    """`(real $HOME, auth_files())` after checking every listed auth file exists there (an
    empty list passes)."""
    real = Path.home()
    rels = auth_files()
    missing = [r for r in rels if not (real / r).is_file()]
    if missing:
        raise config.ConfigError(
            f"auth file(s) {missing} listed in SKILLBENCH_AUTH_FILES not found under {real}: "
            f"log in with `agy` once, or fix SKILLBENCH_AUTH_FILES (internal Jetski needs "
            f"none; leave it empty)"
        )
    return real, rels


def stage_auth(home: Path) -> list[str]:
    """Copy (never link) the SKILLBENCH_AUTH_FILES, if any, from the real `$HOME` into `home`."""
    real, rels = checked_auth_files()
    for rel in rels:
        dst = home / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(real / rel, dst, follow_symlinks=True)
        dst.chmod(0o600)
    return list(rels)


def write_shims(shim_dir: Path) -> None:
    """One executable side-effect shim per SHIM_TOOLS entry."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    for tool in SHIM_TOOLS:
        p = shim_dir / tool
        p.write_text(_SHIM.format(tool=tool, env=SIDE_EFFECTS_ENV), encoding="utf-8")
        p.chmod(0o755)


def prepare_rollout(
    install: Install, run_dir: Path, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Give one rollout a fresh copy of the staged HOME and return its environment.

    `<run>/home` is rebuilt from the install (stale files and refreshed tokens from an earlier
    attempt are dropped); `<run>/side_effects/` receives what the shims capture. The env puts
    the shims first on PATH, sets `EDITOR=VISUAL=true` and points HOME and the XDG dirs into the
    throwaway home. `base_env` (default `os.environ`) is not mutated.
    """
    run_dir = Path(run_dir)
    home = run_dir / "home"
    if home.exists():
        shutil.rmtree(home)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(install.staging_dir) / "home", home, symlinks=False)
    side = run_dir / "side_effects"
    side.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ if base_env is None else base_env)
    shims = Path(install.staging_dir) / "shims"
    path = env.get("PATH") or os.defpath
    env.update(
        HOME=str(home),
        PATH=f"{shims}{os.pathsep}{path}",
        EDITOR="true",
        VISUAL="true",
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        XDG_STATE_HOME=str(home / ".local" / "state"),
        XDG_CACHE_HOME=str(home / ".cache"),
        **{SIDE_EFFECTS_ENV: str(side)},
    )
    return env


def _arm_ctx(ctx: Mapping[str, str]) -> ArmSpec:
    """The arm a `materialise` ctx asks for: `arm` (a label), `target_skill` (skill arms;
    empty or absent for suite arms) and, for a `suite-<name>` arm of a further suite, its
    `suite_path`."""
    files = sorted(k for k in ctx if k.startswith("file:"))
    if files:
        raise ValueError(
            f"install mode {INSTALL_MODE!r} takes no `file:` ctx entries (got {files}); the "
            f"rollout's files come from the task fixture"
        )
    label = ctx.get("arm")
    try:
        kind = arm_kind(label)  # type: ignore[arg-type]
    except ValueError:
        raise ValueError(f"ctx['arm'] must be one of {ARM_LABEL_HELP}, got {label!r}") from None
    target = ctx.get("target_skill")
    raw_path = ctx.get("suite_path")
    extra = (load_suite(raw_path),) if raw_path else ()
    if kind == "skill":
        if not isinstance(target, str) or not _SKILL_NAME.fullmatch(target):
            raise ValueError(f"ctx['target_skill'] must be a skill directory name, got {target!r}")
        suite = extra[0] if extra else None
        return parse_arm_label(label, suite=suite, target_skill=target)  # type: ignore[arg-type]
    if target not in (None, ""):
        raise ValueError(
            f"ctx['target_skill'] applies to skill arms (with | without); arm {label!r} takes "
            f"none, got {target!r}"
        )
    return parse_arm_label(label, extra_suites=extra)  # type: ignore[arg-type]


def _digest(*parts: str) -> str:
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------------
# Execution: fixture repo, one headless agy call, run-dir outputs, model guard, preflight.
# --------------------------------------------------------------------------------------------

DEFAULT_AGY_BIN = "agy"
# The Google-internal Jetski CLI (same flags as `agy`); `jet` is usually a shell alias for it.
JETSKI_CLI = "/google/bin/releases/jetski-devs/tools/cli"
# jet model ids embed the effort (spike question 1): there is no plain `opus`.
DEFAULT_EXEC_MODEL = "opus-5.5-high"
# Values `--effort` accepts (spike question 7).
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Used when a caller passes no positive timeout (adapters pass `int(timeout or 0)`).
DEFAULT_TIMEOUT_S = 1800
# Upper bound on draining agy's stdout after the process (group) is gone.
_READER_JOIN_S = 10.0
# Fixed identity for the fixture's initial commit (and any commit the agent makes): the
# rollout HOME has no ~/.gitconfig, and the real one must never be read.
GIT_ENV = {
    "GIT_AUTHOR_NAME": "skillbench",
    "GIT_AUTHOR_EMAIL": "skillbench@localhost",
    "GIT_COMMITTER_NAME": "skillbench",
    "GIT_COMMITTER_EMAIL": "skillbench@localhost",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
}


@dataclass(frozen=True)
class AgyTask:
    """What a rollout needs from the benchmark: the prompt exactly as sent, and the fixture repo
    copied into `<run>/work` (`None` -> an empty repository)."""

    prompt: str
    fixture: Path | None = None


def agy_bin() -> str:
    """The configured agy command (HARNESS2_AGY_BIN, default `agy`)."""
    return str(config.env("HARNESS2_AGY_BIN", DEFAULT_AGY_BIN))


def agy_bin_missing_message() -> str:
    """The remediation for an unusable HARNESS2_AGY_BIN (shared with the skillbench judge)."""
    return (
        f"agy binary {agy_bin()!r} not found or not executable: install AGY so `agy` is on "
        f"PATH, or set HARNESS2_AGY_BIN to the executable's path (Google-internal Jetski CLI: "
        f"HARNESS2_AGY_BIN={JETSKI_CLI}); a shell alias or function such as `jet` cannot be "
        f"used because harness2 starts the binary directly"
    )


def resolve_agy_bin() -> Path | None:
    """HARNESS2_AGY_BIN as an executable path (a bare name is looked up on PATH), else None."""
    raw = agy_bin()
    if os.sep in raw or raw.startswith("~"):
        p = Path(raw).expanduser()
        return p if p.is_file() and os.access(p, os.X_OK) else None
    found = shutil.which(raw)
    return Path(found) if found else None


def exec_model() -> str:
    """The executor model requested from agy (SKILLBENCH_EXEC_MODEL, default `opus-5.5-high`)."""
    return str(config.env("SKILLBENCH_EXEC_MODEL", DEFAULT_EXEC_MODEL))


def exec_effort() -> str:
    """SKILLBENCH_EFFORT (optional): `""` passes no `--effort`, since a jet model id such as
    `opus-5.5-high` already pins the effort. A set value must be one of EFFORTS."""
    raw = str(config.env("SKILLBENCH_EFFORT") or "").strip()
    if raw and raw not in EFFORTS:
        raise config.ConfigError(
            f"SKILLBENCH_EFFORT must be empty or one of {'|'.join(EFFORTS)}, got {raw!r}"
        )
    return raw


@contextlib.contextmanager
def executor_env(
    executor: str,
    judge: str | None = None,
    *,
    suite: SuiteConfig | Path | str | None = None,
    runs_root: Path | None = None,
    matrix: bool = True,
) -> Iterator[dict[str, str]]:
    """Temporarily set `SKILLBENCH_EXEC_MODEL`, `SKILLBENCH_JUDGE_MODEL`, `SKILLBENCH_SUITE`,
    `SKILLBENCH_RUNS_ROOT` and `SKILLBENCH_MATRIX` in `os.environ` for one matrix executor."""
    keys = (
        "SKILLBENCH_EXEC_MODEL",
        "SKILLBENCH_JUDGE_MODEL",
        "SKILLBENCH_SUITE",
        "SKILLBENCH_RUNS_ROOT",
        "SKILLBENCH_MATRIX",
    )
    prev = {k: os.environ.get(k) for k in keys}
    updates: dict[str, str] = {"SKILLBENCH_EXEC_MODEL": executor}
    if judge is not None:
        updates["SKILLBENCH_JUDGE_MODEL"] = judge
    if suite is not None:
        if isinstance(suite, SuiteConfig):
            if suite.path is not None:
                updates["SKILLBENCH_SUITE"] = str(suite.path)
        elif str(suite):
            updates["SKILLBENCH_SUITE"] = str(Path(suite).expanduser())
    if runs_root is not None:
        updates["SKILLBENCH_RUNS_ROOT"] = str(Path(runs_root).expanduser().resolve())
    if matrix:
        updates["SKILLBENCH_MATRIX"] = "1"
    try:
        os.environ.update(updates)
        yield updates
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def agy_argv(
    binary: Path | str,
    prompt: str,
    *,
    model: str,
    timeout: int,
    effort: str | None = None,
    json_schema: str | None = None,
) -> list[str]:
    """The headless agy command line (revmux's executor flags)."""
    argv = [
        str(binary),
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--dangerously-skip-permissions",
        "--disable-slash-commands",
        "--model",
        model,
    ]
    if effort:
        argv += ["--effort", effort]
    if json_schema is not None:
        argv += ["--json-schema", json_schema]
    argv += ["--print-timeout", f"{int(timeout)}s"]
    return argv


def observed_model(events: Iterable[AgyEvent]) -> str | None:
    """The last model name the stream reported, or None when agy reports none."""
    seen = None
    for ev in events:
        if ev.model is not None:
            seen = ev.model
    return seen


def model_matches(requested: str, actual: str) -> bool:
    """`actual` names the requested model: equal, or the requested alias is one of its
    dash/dot-separated tokens (`opus` matches `claude-opus-4-1`). Case-insensitive."""
    req, act = requested.strip().lower(), actual.strip().lower()
    return bool(req) and (req == act or req in re.split(r"[^a-z0-9]+", act))


def model_guard(events: Iterable[AgyEvent], requested: str) -> tuple[str | None, str | None]:
    """`(actual_model, invalid_reason)`: a reported model that disagrees invalidates the rep; no
    reported model is `actual_model=None` and keeps the rep valid ("model unverified")."""
    actual = observed_model(events)
    if actual is not None and not model_matches(requested, actual):
        return actual, f"model mismatch: requested {requested!r}, agy reported {actual!r}"
    return actual, None


def final_text(events: Iterable[AgyEvent]) -> str:
    """agy's final message: the last non-blank `result.response` (jet's full answer), else the
    last `DONE` response delta, else the response text streamed after the last tool call."""
    last_done: str | None = None
    last_response: str | None = None
    tail: list[str] = []
    for ev in events:
        if ev.response is not None and ev.response.strip():
            last_response = ev.response
        if ev.is_tool:
            tail = []
            continue
        if ev.text is None or ev.step_type in _THINKING_STEPS:
            continue
        if ev.state == "DONE":
            last_done = ev.text
        else:
            tail.append(ev.text)
    if last_response is not None:
        return last_response
    return last_done if last_done is not None else "".join(tail)


def _git(work: Path, env: Mapping[str, str], *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=work,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {work}: {proc.stderr.strip()[:500]}")
    return proc.stdout


def init_work(work: Path, fixture: Path | None, env: Mapping[str, str]) -> str:
    """Copy `fixture` (without any `.git`) into `work`, `git init` and commit it; returns the
    base commit sha the deliverable diff is taken against."""
    if fixture is not None:
        fixture = Path(fixture)
        if not fixture.is_dir():
            raise ValueError(f"task fixture {fixture} is not a directory")
        shutil.copytree(
            fixture, work, symlinks=True, ignore=shutil.ignore_patterns(".git", "__pycache__")
        )
    else:
        work.mkdir(parents=True)
    _git(work, env, "init", "-q")
    exclude = work / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as f:
        f.write("\n__pycache__/\n*.py[cod]\n")
    _git(work, env, "add", "-A")
    _git(
        work,
        env,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "--allow-empty",
        "--no-verify",
        "-m",
        "skillbench fixture",
    )
    return _git(work, env, "rev-parse", "HEAD").strip()


def collect_changes(work: Path, base: str, run_dir: Path, env: Mapping[str, str]) -> list[str]:
    """Write `diff.patch` (everything since `base`, incl. agent commits and untracked files)
    and copy files added since `base` into `new_files/`. Returns the added paths."""
    _git(work, env, "add", "-A")
    (run_dir / "diff.patch").write_text(
        _git(work, env, "diff", "--cached", "--binary", base), encoding="utf-8"
    )
    names = _git(work, env, "diff", "--cached", "--name-only", "--diff-filter=A", "-z", base)
    added = [n for n in names.split("\0") if n]
    out = run_dir / "new_files"
    out.mkdir(exist_ok=True)
    for rel in added:
        src = work / rel
        if src.is_symlink() or not src.is_file():
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    return added


def _pump(stdout: IO[bytes], sink: IO[str], start: float, events: list[AgyEvent]) -> None:
    """Copy agy's stdout into events.jsonl, stamping each JSON object line with `t`."""
    for raw in iter(stdout.readline, b""):
        t = round(time.monotonic() - start, 3)
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            obj["t"] = t
            line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        sink.write(line + "\n")
        sink.flush()
        ev = parse_event(line)
        if ev is not None:
            events.append(ev)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _scrub_auth(home: Path) -> None:
    """Drop the copied credentials from a finished rollout's HOME (run dirs are kept)."""
    try:
        rels = auth_files()
    except config.ConfigError:
        return
    for rel in rels:
        try:
            (home / rel).unlink()
        except (FileNotFoundError, IsADirectoryError):
            pass


def run_rollout(
    install: Install,
    run_dir: Path,
    *,
    prompt: str,
    fixture: Path | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    base_env: Mapping[str, str] | None = None,
) -> Path:
    """One isolated agy rollout into a fresh `run_dir`.

    Writes `work/` (the fixture as a git repo, cwd of agy), `home/` (credentials removed
    afterwards), `side_effects/`, `events.jsonl` (+`t`), `agy.stderr.log`, `final.md`,
    `diff.patch`, `new_files/` and `skill_timing.json`. A timeout, a non-zero exit, a reported
    model that disagrees with the requested one, a stream without a result event or an
    unreadable work tree mark the rep invalid (`valid=false`, `invalid_reason`); the run dir is
    written either way. `skill_timing.json` records the invocation (first read of
    `skills/<target>/SKILL.md`, or of any skill's `SKILL.md` for a suite arm), the set of
    skills read, validity, model and cost. Infrastructure failures (missing binary, fixture or
    config) raise.
    """
    kind = install.extras.get("arm_kind") or "skill"
    target: str | None = install.extras.get("target_skill") or None
    if kind == "skill" and not target:
        raise ValueError("install has no target_skill; materialise it with ctx['target_skill']")
    if kind == "suite":
        target = None
    binary = resolve_agy_bin()
    if binary is None:
        raise RuntimeError(agy_bin_missing_message())
    model, effort = exec_model(), exec_effort()
    timeout = int(timeout) if timeout and int(timeout) > 0 else DEFAULT_TIMEOUT_S
    run_dir = Path(run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    env = prepare_rollout(install, run_dir, base_env)
    env.update(GIT_ENV)
    work = run_dir / "work"
    try:
        base = init_work(work, fixture, env)
        argv = agy_argv(binary, prompt, model=model, timeout=timeout, effort=effort)
        events: list[AgyEvent] = []
        timed_out = False
        with (
            open(run_dir / "events.jsonl", "w", encoding="utf-8") as sink,
            open(run_dir / "agy.stderr.log", "wb") as err,
        ):
            start = time.monotonic()
            try:
                proc = subprocess.Popen(
                    argv,
                    cwd=work,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=err,
                    start_new_session=True,
                )
            except OSError as e:
                raise RuntimeError(f"could not start agy ({binary}): {e}") from e
            reader = threading.Thread(
                target=_pump, args=(proc.stdout, sink, start, events), daemon=True
            )
            reader.start()
            try:
                exit_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
                exit_code = proc.wait()
            t_total = round(time.monotonic() - start, 3)
            _kill_group(proc)  # leftovers of the session must not keep the pipe open
            reader.join(_READER_JOIN_S)
            if proc.stdout is not None:
                proc.stdout.close()
        (run_dir / "final.md").write_text(final_text(events), encoding="utf-8")
        diff_error = None
        try:
            collect_changes(work, base, run_dir, env)
        except RuntimeError as e:
            diff_error = str(e)
            (run_dir / "diff.patch").write_text("", encoding="utf-8")
            (run_dir / "new_files").mkdir(exist_ok=True)

        actual, mismatch = model_guard(events, model)
        if timed_out:
            reason = f"timeout after {timeout}s"
        elif exit_code != 0:
            reason = f"agy exited with code {exit_code}"
        elif mismatch:
            reason = mismatch
        elif not any(ev.is_result for ev in events):
            reason = "no result event in stream"
        elif diff_error:
            reason = f"work tree unreadable: {diff_error}"
        else:
            reason = None
        tokens = [ev.total_tokens for ev in events if ev.total_tokens is not None]
        staged_plugins = run_dir / "home" / PLUGINS_REL
        timing = skill_timing(events, target, plugins_dir=staged_plugins)
        record = {
            **timing,
            "skills_read": timing.get("skills_read")
            if "skills_read" in timing
            else skills_read(events, plugins_dir=staged_plugins),
            "t_total_s": t_total,
            "requested_model": model,
            "actual_model": actual,
            "valid": reason is None,
            "invalid_reason": reason,
            "exit_code": exit_code,
            "total_tokens": tokens[-1] if tokens else None,
        }
        (run_dir / "skill_timing.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        _scrub_auth(run_dir / "home")
    return run_dir


class AgySubstrate:
    """AGY / Jetski behind the Substrate interface (bases-only staging for skillbench).

    `tasks` maps a task_ref to its `AgyTask` and `runs_root / <run_id>` is each rollout's run
    dir; both come from the benchmark adapter (staging and preflight work without them).
    """

    name = "agy"

    def __init__(
        self,
        tasks: Callable[[str], AgyTask] | None = None,
        runs_root: Path | None = None,
    ):
        self._tasks = tasks
        self._runs_root = Path(runs_root) if runs_root is not None else None

    def surface(self) -> SurfaceSpec:
        return AGY_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        """Stage a HOME template for one arm.

        `ctx` names the arm (`arm`: `with` / `without` / `baseline` / `suite-<name>`), the
        leave-one-out `target_skill` (skill arms) and, for a further suite's `suite-<name>`
        arm, its `suite_path`. Layout of `staging_dir`: `home/.gemini/config/plugins/<plugin>/`
        (the arm's plugins: the suite under test, minus the target skill for `without`; the
        named suite for `suite-<name>`; nothing for `baseline`), `home/.gemini/GEMINI.md` (the
        tree's always-on text, when non-blank), `home/<auth files>` (copies, if any) and
        `shims/<tool>`. The fingerprint digests the source tree, the staged plugins and the
        shims -- never the auth files, so a token refresh does not move the harness identity.
        `extras` records the arm label and kind, the target skill, the suite under test
        (`suite`, `suite_hash`) and the staged suite (`staged_suite`, `staged_suite_hash`;
        empty for `baseline`).
        """
        if spec.install_mode != INSTALL_MODE:
            raise ValueError(
                f"install_mode {spec.install_mode!r} is not the agy mode; this substrate has "
                f"exactly one: {INSTALL_MODE!r}"
            )
        arm = _arm_ctx(ctx)
        override = ctx.get("skill_override")
        tree_dir = Path(tree_dir)
        if not HT.is_tree(AGY_SURFACE, tree_dir):
            raise ValueError(f"{tree_dir} is not an agy harness tree (no GEMINI.md)")
        suite = active_suite()
        staged = Path(tempfile.mkdtemp(prefix=f"adapt-agy-{arm.label}-"))
        try:
            home = staged / "home"
            omitted = stage_plugins(home / PLUGINS_REL, arm=arm, skill_override=override)
            rules = (tree_dir / "GEMINI.md").read_text(encoding="utf-8", errors="replace")
            if rules.strip():
                (home / RULES_REL).write_text(rules, encoding="utf-8")
            stage_auth(home)
            write_shims(staged / "shims")
            fp = _digest(
                RM.fingerprint(tree_dir),
                RM.fingerprint(home / PLUGINS_REL),
                RM.fingerprint(staged / "shims"),
            )
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return Install(
            staging_dir=staged,
            fingerprint=fp,
            extras={
                "install_mode": INSTALL_MODE,
                "arm": arm.label,
                "arm_kind": arm.kind,
                "target_skill": arm.target_skill or "",
                "omitted": ",".join(omitted),
                "suite": suite.name,
                "suite_hash": suite.suite_hash,
                "staged_suite": arm.suite.name if arm.suite is not None else "",
                "staged_suite_hash": arm.suite.suite_hash if arm.suite is not None else "",
            },
        )

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        """One `run_rollout` per run_id into `runs_root / <run_id>`, in order."""
        if self._tasks is None or self._runs_root is None:
            raise RuntimeError(
                "AgySubstrate was constructed without a task resolver and runs root; the "
                "benchmark adapter must supply both (it owns prompts, fixtures and run dirs)"
            )
        task = self._tasks(task_ref)
        return [
            run_rollout(
                install,
                self._runs_root / rid,
                prompt=task.prompt,
                fixture=task.fixture,
                timeout=int(timeout or 0),
            )
            for rid in run_ids
        ]

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 400
    ) -> str:
        return render_trajectory(
            read_events(Path(stream_path)),
            obs_trunc=obs_trunc,
            max_chars=max_chars,
            args_trunc=args_trunc,
        )

    def preflight(self) -> None:
        """Checks the agy binary, git, the effort value, the active suite's plugins checkout
        and any listed auth files; starts nothing (agy is not even asked for --version)."""
        if resolve_agy_bin() is None:
            raise RuntimeError(agy_bin_missing_message())
        if shutil.which("git") is None:
            raise RuntimeError("git not found on PATH: install git (fixtures run as git repos)")
        exec_effort()
        checked_plugins()
        checked_auth_files()
