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

"""Shape checks for the synthetic agy stream-json fixtures (stdlib only, no harness2 import)."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "agy"
SPIKE_DOC = ROOT / "docs" / "skillbench-spike.md"
SPIKE_SCRIPT = ROOT / "scripts" / "agy_spike.sh"

# Fixtures that deliberately contain non-JSON lines, with the expected count.
MALFORMED = {"malformed_line.jsonl": 1}

STEP_KEYS = {"step_type", "state", "tool_name", "text_delta", "tool_info"}
STEP_TYPES = {"agent_response", "message", "tool", "tool_call"}
STATES = {"ACTIVE", "DONE"}
RESULT_KEYS = {"structured_output", "usage", "model"}
SKILL_PATH = re.compile(r"/skills/([^/]+)/SKILL\.md$")
ASSUMED = "assumed — verify in pre-pilot spike"


def fixture_files():
    return sorted(FIXTURES.glob("*.jsonl"))


def read_lines(path):
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n"), f"{path.name}: missing trailing newline"
    return text[:-1].split("\n")


def decode(path):
    """Return (events, malformed_lines) for one fixture."""
    events, bad = [], []
    for line in read_lines(path):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(line)
    return events, bad


def shape_errors(ev):
    """Return a list of problems with one decoded event against the revmux event shape."""
    if not isinstance(ev, dict):
        return ["event is not an object"]
    errors = []
    kind = ev.get("event")
    if kind == "step_update":
        su = ev.get("step_update")
        if not isinstance(su, dict):
            return ["step_update is not an object"]
        if extra := set(su) - STEP_KEYS:
            errors.append(f"unknown step_update keys {sorted(extra)}")
        if "step_type" in su and su["step_type"] not in STEP_TYPES:
            errors.append(f"bad step_type {su['step_type']!r}")
        if "state" in su and su["state"] not in STATES:
            errors.append(f"bad state {su['state']!r}")
        for key in ("tool_name", "text_delta"):
            if key in su and not isinstance(su[key], str):
                errors.append(f"{key} is not a string")
        if "tool_info" in su:
            info = su["tool_info"]
            if not isinstance(info, dict) or not isinstance(info.get("name"), str):
                errors.append("tool_info needs a string name")
            elif not isinstance(info.get("parameters", {}), dict):
                errors.append("tool_info.parameters is not an object")
    elif kind == "result":
        res = ev.get("result")
        if not isinstance(res, dict):
            return ["result is not an object"]
        if extra := set(res) - RESULT_KEYS:
            errors.append(f"unknown result keys {sorted(extra)}")
        usage = res.get("usage")
        if not isinstance(usage, dict) or not isinstance(usage.get("total_tokens"), int):
            errors.append("result.usage.total_tokens must be an int")
        if "model" in res and not isinstance(res["model"], str):
            errors.append("result.model is not a string")
    else:
        errors.append(f"unknown event {kind!r}")
    return errors


def tool_calls(events, name):
    """ACTIVE tool steps with tool_info for the given tool name."""
    return [
        ev["step_update"]
        for ev in events
        if ev.get("event") == "step_update"
        and ev["step_update"].get("tool_info", {}).get("name") == name
        and ev["step_update"].get("state") == "ACTIVE"
    ]


class FixtureShapeTests(unittest.TestCase):
    def test_fixtures_exist(self):
        self.assertTrue(fixture_files(), f"no fixtures in {FIXTURES}")

    def test_only_deliberate_lines_are_malformed(self):
        for path in fixture_files():
            with self.subTest(fixture=path.name):
                _, bad = decode(path)
                self.assertEqual(len(bad), MALFORMED.get(path.name, 0), bad)

    def test_every_event_matches_revmux_shape(self):
        for path in fixture_files():
            events, _ = decode(path)
            for idx, ev in enumerate(events):
                with self.subTest(fixture=path.name, event=idx):
                    self.assertEqual(shape_errors(ev), [])

    def test_every_fixture_ends_with_one_result(self):
        for path in fixture_files():
            with self.subTest(fixture=path.name):
                events, _ = decode(path)
                kinds = [ev["event"] for ev in events]
                self.assertEqual(kinds.count("result"), 1)
                self.assertEqual(kinds[-1], "result")

    def test_shape_checker_rejects_bad_events(self):
        self.assertTrue(shape_errors({"event": "nope"}))
        self.assertTrue(shape_errors({"event": "step_update", "step_update": {"bogus": 1}}))
        self.assertTrue(shape_errors({"event": "result", "result": {"usage": {}}}))
        self.assertTrue(
            shape_errors(
                {"event": "step_update", "step_update": {"tool_info": {"parameters": {}}}}
            )
        )


class ScenarioTests(unittest.TestCase):
    def load(self, name):
        events, _ = decode(FIXTURES / f"{name}.jsonl")
        return events

    @staticmethod
    def result(events):
        return events[-1]["result"]

    def test_plain_answer_has_prose_and_no_tools(self):
        events = self.load("plain_answer")
        steps = [ev["step_update"] for ev in events if ev["event"] == "step_update"]
        self.assertTrue(all(s.get("step_type") == "agent_response" for s in steps))
        self.assertTrue("".join(s.get("text_delta", "") for s in steps).strip())

    def test_skill_read_reads_target_and_another_skill(self):
        events = self.load("skill_read")
        calls = tool_calls(events, "view_file")
        paths = [c["tool_info"]["parameters"]["AbsolutePath"] for c in calls]
        skills = [m.group(1) for p in paths if (m := SKILL_PATH.search(p))]
        self.assertIn("make-plan", skills)
        self.assertIn("brainstorm", skills)
        self.assertLess(skills.index("brainstorm"), skills.index("make-plan"))
        self.assertGreaterEqual(skills.count("make-plan"), 2, "target read twice")
        self.assertTrue(any(not SKILL_PATH.search(p) for p in paths), "a non-skill read too")

    def test_ask_question_call(self):
        calls = tool_calls(self.load("ask_question"), "ask_question")
        self.assertEqual(len(calls), 1)
        questions = calls[0]["tool_info"]["parameters"]["questions"]
        self.assertIsInstance(questions[0]["options"], list)
        self.assertIsInstance(questions[0]["is_multi_select"], bool)

    def test_json_schema_structured_output(self):
        out = self.result(self.load("json_schema"))["structured_output"]
        self.assertIsInstance(out, dict)
        for crit in out["criteria"]:
            self.assertIsInstance(crit["id"], str)
            self.assertIsInstance(crit["pass"], bool)
            self.assertIsInstance(crit["reason"], str)

    def test_model_field_variants(self):
        self.assertEqual(self.result(self.load("result_with_model"))["model"], "opus")
        self.assertEqual(self.result(self.load("result_model_mismatch"))["model"], "sonnet")
        for ev in self.load("result_without_model"):
            self.assertNotIn("model", ev.get("result", {}))
            self.assertNotIn("model", ev)

    def test_malformed_fixture_still_has_valid_events_around_bad_line(self):
        lines = read_lines(FIXTURES / "malformed_line.jsonl")
        _, bad = decode(FIXTURES / "malformed_line.jsonl")
        idx = lines.index(bad[0])
        self.assertGreater(idx, 0)
        self.assertLess(idx, len(lines) - 1)


class SpikeDocTests(unittest.TestCase):
    def test_every_scenario_in_doc_has_fixture(self):
        text = SPIKE_DOC.read_text(encoding="utf-8")
        named = set(re.findall(r"tests/fixtures/agy/([\w-]+\.jsonl)", text))
        self.assertTrue(named, "doc names no fixture scenarios")
        missing = sorted(n for n in named if not (FIXTURES / n).is_file())
        self.assertEqual(missing, [])

    def test_open_questions_are_marked_assumed(self):
        text = SPIKE_DOC.read_text(encoding="utf-8")
        section = text.split("## Open questions", 1)[1]
        questions = re.findall(r"^\d+\. \*\*", section, flags=re.M)
        self.assertGreaterEqual(len(questions), 9)
        self.assertEqual(section.count(ASSUMED), len(questions))


class SpikeScriptTests(unittest.TestCase):
    def test_script_is_valid_bash(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash not available")
        proc = subprocess.run([bash, "-n", str(SPIKE_SCRIPT)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_dry_run_prints_every_probe_without_calling_agy(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash not available")
        with tempfile.TemporaryDirectory() as tmp:
            plugins = Path(tmp) / "plugins"
            for name in ("planning", "review"):
                (plugins / name).mkdir(parents=True)
            out = Path(tmp) / "out"
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": tmp,
                "TMPDIR": tmp,
                "HARNESS2_AGY_BIN": str(Path(tmp) / "no-such-agy"),
                "SKILLBENCH_PLUGINS_DIR": str(plugins),
                "SKILLBENCH_PLUGINS": "planning,review",
                "SPIKE_OUT": str(out),
            }
            proc = subprocess.run(
                [bash, str(SPIKE_SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(out.exists(), "dry run must not write streams")
        probes = [line for line in proc.stdout.splitlines() if line.startswith("[dry-run] ")]
        for model in ("opus", "fable"):
            for scenario in ("plain_answer", "skill_read", "ask_question", "json_schema"):
                hits = [p for p in probes if p.startswith(f"[dry-run] {scenario}.{model}:")]
                self.assertEqual(len(hits), 1, (scenario, model))
                self.assertIn(f"--model {model}", hits[0])
                self.assertIn("--output-format stream-json", hits[0])
        schema_probe = next(p for p in probes if p.startswith("[dry-run] json_schema.opus:"))
        self.assertIn("--json-schema", schema_probe)

    def test_missing_allowlisted_plugin_fails(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash not available")
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": tmp,
                "TMPDIR": tmp,
                "SKILLBENCH_PLUGINS_DIR": tmp,
                "SKILLBENCH_PLUGINS": "planning",
            }
            proc = subprocess.run(
                [bash, str(SPIKE_SCRIPT), "--dry-run"], capture_output=True, text=True, env=env
            )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("allowlisted plugin not found", proc.stderr)


if __name__ == "__main__":
    unittest.main()
