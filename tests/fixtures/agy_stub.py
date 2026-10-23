#!/usr/bin/env python3
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

"""Fake ``agy`` binary for tests (stdlib only); select it with ``HARNESS2_AGY_BIN``.

It accepts any ``agy`` command line, records the call and replays a synthetic stream-json
fixture on stdout. Behaviour is driven by environment variables:

- ``AGY_STUB_SCENARIO`` — fixture to replay: a name under ``tests/fixtures/agy/``
  (``skill_read`` → ``skill_read.jsonl``) or a path to any ``.jsonl`` file. Default
  ``plain_answer``. Unknown scenarios exit 2.
- ``AGY_STUB_LOG`` — file that gets one JSON line per call, appended *before* the replay:
  ``{argv, cwd, home, env, flags}``; ``flags`` holds the parsed ``-p`` / ``--model`` /
  ``--effort`` / ``--json-schema`` / ``--output-format`` / ``--print-timeout`` values.
- ``AGY_STUB_SLEEP`` — seconds to sleep before each emitted line (default 0).
- ``AGY_STUB_HANG`` — when truthy, sleep ``AGY_STUB_HANG_S`` seconds (default 3600) after the
  replay instead of exiting, so callers can exercise their timeout.
- ``AGY_STUB_EXIT_CODE`` — exit status after the replay (default 0); non-zero also prints an
  error to stderr.

The fixtures use the placeholder rollout root ``/tmp/skillbench-run/``: ``home/`` is rewritten
to ``$HOME`` and ``work/`` to the current directory in every emitted event. A ``write_to_file``
tool call whose ``TargetFile`` resolves inside the current directory is applied (its
``CodeContent`` is written), so a scenario can produce a diff; targets outside cwd are skipped
with a warning on stderr. Malformed fixture lines are emitted verbatim.
"""

import json
import os
import sys
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "agy"
DEFAULT_SCENARIO = "plain_answer"
PLACEHOLDER_HOME = "/tmp/skillbench-run/home"
PLACEHOLDER_WORK = "/tmp/skillbench-run/work"
VALUE_FLAGS = {
    "-p": "prompt",
    "--prompt": "prompt",
    "--model": "model",
    "--effort": "effort",
    "--json-schema": "json_schema",
    "--output-format": "output_format",
    "--print-timeout": "print_timeout",
}
TRUTHY = {"1", "true", "yes", "on"}


def parse_flags(argv):
    """Return the known ``agy`` flags in ``argv`` as a dict; unknown tokens are ignored."""
    flags = {}
    i = 0
    while i < len(argv):
        token = argv[i]
        name, eq, inline = token.partition("=")
        if name in VALUE_FLAGS and eq:
            flags[VALUE_FLAGS[name]] = inline
        elif token in VALUE_FLAGS and i + 1 < len(argv):
            flags[VALUE_FLAGS[token]] = argv[i + 1]
            i += 1
        elif token.startswith("--"):
            flags[token[2:].replace("-", "_")] = True
        i += 1
    return flags


def resolve_scenario(name):
    """Map a scenario name or path to a fixture file, or ``None`` if it does not exist."""
    path = Path(name)
    if path.suffix == ".jsonl" and path.is_file():
        return path
    candidate = FIXTURES / f"{name}.jsonl"
    return candidate if candidate.is_file() else None


def write_log(log_path, argv, home):
    record = {
        "argv": argv,
        "cwd": os.getcwd(),
        "home": home,
        "env": dict(os.environ),
        "flags": parse_flags(argv),
    }
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def rewrite(value, mapping):
    """Replace placeholder path prefixes in every string inside ``value``."""
    if isinstance(value, str):
        for old, new in mapping:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [rewrite(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: rewrite(v, mapping) for k, v in value.items()}
    return value


def apply_write(event, cwd):
    """Write the file of an ACTIVE ``write_to_file`` step when its target lies inside ``cwd``."""
    step = event.get("step_update") if isinstance(event, dict) else None
    if not isinstance(step, dict) or step.get("state") != "ACTIVE":
        return
    info = step.get("tool_info")
    if not isinstance(info, dict) or info.get("name") != "write_to_file":
        return
    params = info.get("parameters") or {}
    target, content = params.get("TargetFile"), params.get("CodeContent", "")
    if not isinstance(target, str) or not isinstance(content, str):
        return
    path = (cwd / target).resolve()
    if not path.is_relative_to(cwd):
        print(f"agy-stub: refusing to write outside cwd: {target}", file=sys.stderr)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def emit(line, mapping, cwd):
    """Print one fixture line (rewritten if needed) and apply its side effects."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        print(line, flush=True)
        return
    rewritten = rewrite(event, mapping)
    if rewritten != event:
        line = json.dumps(rewritten, ensure_ascii=False, separators=(",", ":"))
    print(line, flush=True)
    apply_write(rewritten, cwd)


def main(argv):
    home = os.environ.get("HOME", "")
    if log_path := os.environ.get("AGY_STUB_LOG"):
        write_log(log_path, argv, home)
    if "--version" in argv:
        print("agy-stub 0.0.0")
        return 0
    scenario = os.environ.get("AGY_STUB_SCENARIO") or DEFAULT_SCENARIO
    fixture = resolve_scenario(scenario)
    if fixture is None:
        print(f"agy-stub: unknown scenario {scenario!r} (looked in {FIXTURES})", file=sys.stderr)
        return 2
    sleep_s = float(os.environ.get("AGY_STUB_SLEEP") or 0)
    cwd = Path.cwd().resolve()
    mapping = [(PLACEHOLDER_HOME, home), (PLACEHOLDER_WORK, str(cwd))]
    for line in fixture.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        if sleep_s:
            time.sleep(sleep_s)
        emit(line, mapping, cwd)
    if os.environ.get("AGY_STUB_HANG", "").lower() in TRUTHY:
        time.sleep(float(os.environ.get("AGY_STUB_HANG_S") or 3600))
    code = int(os.environ.get("AGY_STUB_EXIT_CODE") or 0)
    if code:
        print(f"agy-stub: simulated failure (exit {code})", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
