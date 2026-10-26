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

"""Behaviour of the fake ``agy`` binary (``tests/fixtures/agy_stub.py``) used by later tests."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUB = ROOT / "tests" / "fixtures" / "agy_stub.py"
FIXTURES = ROOT / "tests" / "fixtures" / "agy"
STUB_ENV = (
    "AGY_STUB_SCENARIO",
    "AGY_STUB_LOG",
    "AGY_STUB_SLEEP",
    "AGY_STUB_HANG",
    "AGY_STUB_HANG_S",
    "AGY_STUB_EXIT_CODE",
)
ARGV = [
    "-p",
    "Fix the failing test.",
    "--output-format",
    "stream-json",
    "--dangerously-skip-permissions",
    "--disable-slash-commands",
    "--model",
    "opus",
    "--effort",
    "high",
    "--print-timeout",
    "600s",
]
MODEL_PLACEHOLDER = "<requested-model>"


def replayed(name, work, model="opus"):
    """A fixture as the stub replays it in cwd ``work`` for ``--model <model>``."""
    text = (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8")
    return text.replace(MODEL_PLACEHOLDER, model).replace("/tmp/skillbench-run/work", str(work))


class AgyStubTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.home = self.root / "home"
        self.work = self.root / "work"
        self.home.mkdir()
        self.work.mkdir()
        self.log = self.root / "agy.log"

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if k not in STUB_ENV}
        env.update(HOME=str(self.home), AGY_STUB_LOG=str(self.log), **extra)
        return env

    def run_stub(self, argv=ARGV, timeout=30, **extra):
        return subprocess.run(
            [str(STUB), *argv],
            cwd=self.work,
            env=self.env(**extra),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def log_records(self):
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def test_stub_is_executable_stdlib_only(self):
        self.assertTrue(os.access(STUB, os.X_OK))
        self.assertTrue(STUB.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3\n"))
        tree = ast.parse(STUB.read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module.split(".")[0])
        self.assertTrue(modules)
        self.assertLessEqual(modules, set(sys.stdlib_module_names))

    def test_replays_named_fixture_verbatim(self):
        proc = self.run_stub(AGY_STUB_SCENARIO="plain_answer")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, replayed("plain_answer", self.work))

    def test_default_scenario_is_plain_answer(self):
        proc = self.run_stub()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, replayed("plain_answer", self.work))

    def test_init_model_echoes_requested_model(self):
        argv = [*ARGV]
        argv[argv.index("--model") + 1] = "fable-5.1-high"
        proc = self.run_stub(argv, AGY_STUB_SCENARIO="skill_read")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        first = json.loads(proc.stdout.splitlines()[0])
        self.assertEqual(first["event"], "init")
        self.assertEqual(first["init"]["model"], "fable-5.1-high")
        self.assertNotIn(MODEL_PLACEHOLDER, proc.stdout)

    def test_explicit_init_model_is_kept(self):
        proc = self.run_stub(AGY_STUB_SCENARIO="result_model_mismatch")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout.splitlines()[0])["init"]["model"], "sonnet-4-high")

    def test_replays_fixture_given_as_path(self):
        custom = self.root / "custom.jsonl"
        custom.write_text('{"event":"result","result":{"usage":{"total_tokens":7}}}\n')
        proc = self.run_stub(AGY_STUB_SCENARIO=str(custom))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout, custom.read_text())

    def test_unknown_scenario_fails(self):
        proc = self.run_stub(AGY_STUB_SCENARIO="no_such_scenario")
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertIn("no_such_scenario", proc.stderr)

    def test_malformed_line_is_emitted_verbatim(self):
        proc = self.run_stub(AGY_STUB_SCENARIO="malformed_line")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        fixture_lines = (FIXTURES / "malformed_line.jsonl").read_text().splitlines()
        out_lines = proc.stdout.splitlines()
        self.assertEqual(len(out_lines), len(fixture_lines))
        bad = [line for line in fixture_lines if not line.endswith("}")]
        self.assertEqual(len(bad), 1)
        self.assertIn(bad[0], out_lines)

    def test_placeholders_rewritten_to_home_and_cwd(self):
        proc = self.run_stub(AGY_STUB_SCENARIO="skill_read")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("/tmp/skillbench-run/", proc.stdout)
        events = [json.loads(line) for line in proc.stdout.splitlines()]
        paths = [
            params.get("Path") or params.get("AbsolutePath")
            for ev in events
            if "tool_info" in ev.get("step_update", {})
            for params in [ev["step_update"]["tool_info"]["parameters"]]
        ]
        target = f"{self.home}/.gemini/config/plugins/planning/skills/make-plan/SKILL.md"
        self.assertIn(target, paths)
        self.assertIn(f"{self.work}/app.py", paths)
        self.assertEqual(events[0]["init"]["cwd"], str(self.work))
        result = events[-1]["result"]
        self.assertEqual(result["usage"]["total_tokens"], 8731)
        self.assertTrue(result["response"].strip())

    def test_writes_file_into_cwd_when_scenario_asks(self):
        proc = self.run_stub(AGY_STUB_SCENARIO="skill_read")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        written = self.work / "docs" / "plans" / "20261023-add-cache.md"
        self.assertTrue(written.is_file())
        self.assertIn("add an LRU cache around fetch()", written.read_text())
        self.assertEqual(sorted(p.name for p in self.home.iterdir()), [])

    def test_scenarios_without_writes_leave_cwd_untouched(self):
        for scenario in ("plain_answer", "ask_question", "json_schema", "result_with_model"):
            with self.subTest(scenario=scenario):
                proc = self.run_stub(AGY_STUB_SCENARIO=scenario)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(list(self.work.iterdir()), [])

    def test_refuses_writes_outside_cwd(self):
        custom = self.root / "escape.jsonl"
        params = {"TargetFile": "/tmp/skillbench-run/work/../escaped.txt", "CodeContent": "x"}
        event = {
            "event": "step_update",
            "step_update": {
                "step_type": "tool",
                "state": "ACTIVE",
                "tool_name": "write_to_file",
                "tool_info": {"name": "write_to_file", "parameters": params},
            },
        }
        custom.write_text(json.dumps(event) + "\n")
        proc = self.run_stub(AGY_STUB_SCENARIO=str(custom))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse((self.root / "escaped.txt").exists())
        self.assertIn("refusing to write outside cwd", proc.stderr)

    def test_logs_argv_cwd_home_env_and_flags(self):
        schema = '{"type":"object"}'
        argv = [*ARGV, "--json-schema", schema]
        proc = self.run_stub(argv, AGY_STUB_SCENARIO="json_schema", EDITOR="true")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.run_stub(["--model=fable", "-p", "second"])
        first, second = self.log_records()
        self.assertEqual(first["argv"], argv)
        self.assertEqual(first["cwd"], str(self.work))
        self.assertEqual(first["home"], str(self.home))
        self.assertEqual(first["env"]["EDITOR"], "true")
        self.assertEqual(first["env"]["AGY_STUB_SCENARIO"], "json_schema")
        flags = first["flags"]
        self.assertEqual(flags["prompt"], "Fix the failing test.")
        self.assertEqual(flags["model"], "opus")
        self.assertEqual(flags["effort"], "high")
        self.assertEqual(flags["json_schema"], schema)
        self.assertEqual(flags["output_format"], "stream-json")
        self.assertEqual(flags["print_timeout"], "600s")
        self.assertIs(flags["dangerously_skip_permissions"], True)
        self.assertIs(flags["disable_slash_commands"], True)
        self.assertEqual(second["flags"], {"model": "fable", "prompt": "second"})

    def test_no_log_without_log_env(self):
        env = self.env()
        del env["AGY_STUB_LOG"]
        proc = subprocess.run(
            [str(STUB), *ARGV], cwd=self.work, env=env, capture_output=True, text=True, timeout=30
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(self.log.exists())

    def test_version(self):
        proc = self.run_stub(["--version"])
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(proc.stdout.startswith("agy-stub"))

    def test_per_event_sleep(self):
        n_lines = len((FIXTURES / "plain_answer.jsonl").read_text().splitlines())
        start = time.monotonic()
        proc = self.run_stub(AGY_STUB_SCENARIO="plain_answer", AGY_STUB_SLEEP="0.1")
        elapsed = time.monotonic() - start
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertGreaterEqual(elapsed, 0.1 * n_lines)

    def test_non_zero_exit_still_emits_events_and_logs(self):
        proc = self.run_stub(AGY_STUB_SCENARIO="plain_answer", AGY_STUB_EXIT_CODE="3")
        self.assertEqual(proc.returncode, 3)
        self.assertEqual(proc.stdout, replayed("plain_answer", self.work))
        self.assertIn("simulated failure", proc.stderr)
        self.assertEqual(len(self.log_records()), 1)

    def test_hang_until_killed(self):
        proc = subprocess.Popen(
            [str(STUB), *ARGV],
            cwd=self.work,
            env=self.env(AGY_STUB_SCENARIO="plain_answer", AGY_STUB_HANG="1"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            first = proc.stdout.readline()
            self.assertEqual(json.loads(first)["event"], "init")
            with self.assertRaises(subprocess.TimeoutExpired):
                proc.wait(timeout=1)
            self.assertEqual(len(self.log_records()), 1)
        finally:
            proc.kill()
            proc.communicate(timeout=10)
        self.assertNotEqual(proc.returncode, 0)

    def test_hang_duration_is_configurable(self):
        proc = self.run_stub(
            AGY_STUB_SCENARIO="plain_answer", AGY_STUB_HANG="1", AGY_STUB_HANG_S="0.2", timeout=10
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
