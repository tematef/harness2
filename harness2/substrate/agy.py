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

This module currently holds the stream-json decoder:

    parse_event        one stdout line -> AgyEvent | None (non-JSON / non-event lines skipped)
    read_events        events.jsonl -> list[AgyEvent]
    skill_timing       first read of skills/<target>/SKILL.md -> {invoked, t_skill_s, step_idx}
    render_trajectory  AgyEvent list -> the shared [ASSISTANT]/[ACTION]/... trajectory grammar

The event shape follows revmux's agy decoder (`event`, `result{structured_output, model?,
usage.total_tokens}`, `step_update{step_type, state, tool_name, text_delta, tool_info{name,
parameters}}`). Fields marked "assumed" in docs/skillbench-spike.md (the optional
`result.model`, skill reads as plain `view_file` steps) are verified in the pre-pilot spike.
Lines in a run's events.jsonl additionally carry `t`, seconds since the agy process spawned.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

# Tools whose call counts as reading a skill (spike question 3: a skill read is an ordinary
# file read of <HOME>/.gemini/config/plugins/<plugin>/skills/<skill>/SKILL.md).
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
        model=_str(res.get("model")) or _str(obj.get("model")),
        error=_str(err) if err is not None else None,
        t=t,
    )


def read_events(path: Path) -> list[AgyEvent]:
    """Parse every decodable line of an events.jsonl (missing file -> [])."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [ev for ev in map(parse_event, raw.splitlines()) if ev is not None]


def _reads_skill(ev: AgyEvent, pattern: re.Pattern[str]) -> bool:
    if ev.tool_name not in READ_TOOLS or not ev.tool_args:
        return False
    return any(isinstance(v, str) and pattern.search(v) for v in ev.tool_args.values())


def skill_timing(events: Iterable[AgyEvent], target_skill: str) -> dict:
    """Locate the first read of `skills/<target_skill>/SKILL.md`.

    Returns `{invoked, t_skill_s, step_idx}`; `step_idx` is the 0-based index in `events`,
    `t_skill_s` that event's `t` (`None` when the stream carries no timestamps). Reads of other
    skills, later re-reads and writes to the path are ignored.
    """
    pattern = re.compile(rf"(?:^|[/\\])skills[/\\]{re.escape(target_skill)}[/\\]SKILL\.md$")
    for idx, ev in enumerate(events):
        if _reads_skill(ev, pattern):
            return {"invoked": True, "t_skill_s": ev.t, "step_idx": idx}
    return {"invoked": False, "t_skill_s": None, "step_idx": None}


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
    response delta (agy's final message) gets its own line. agy streams no tool output, so a
    tool call renders as `[ACTION]` only, plus `[OBSERVATION ERROR]` when the step fails.
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
