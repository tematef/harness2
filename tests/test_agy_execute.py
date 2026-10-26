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

"""AgySubstrate execution, model guard and preflight, driven by tests/fixtures/agy_stub.py.

Every test runs against its own fake cc-thingz `plugins/` checkout and its own fake `$HOME`
(`HOME=<tmp>`); the real `~/.gemini` is never read or written.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from harness2.config import ConfigError
from harness2.core.spec import BenchPrompts, BenchSpec, ChannelSpec
from harness2.substrate.agy import (
    DEFAULT_EXEC_MODEL,
    INSTALL_MODE,
    JETSKI_CLI,
    PLUGINS_REL,
    AgyTask,
    agy_argv,
    exec_effort,
    exec_model,
    final_text,
    model_guard,
    model_matches,
    parse_event,
)
from harness2.substrate.agy import AgySubstrate as AgyFromModule

STUB = Path(__file__).resolve().parent / "fixtures" / "agy_stub.py"
AUTH = (".gemini/oauth_creds.json", ".gemini/google_accounts.json")
PROMPT = "Plan adding a cache around fetch() in this repo."


def _write(p: Path, body: str = "x\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _skill(plugin_dir: Path, name: str) -> None:
    _write(plugin_dir / "skills" / name / "SKILL.md", f"---\nname: {name}\n---\nDo {name}.\n")


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _scenario(path: Path, *events: dict) -> Path:
    return _write(path, "".join(json.dumps(e) + "\n" for e in events))


class _ExecCase(unittest.TestCase):
    """Fake cc-thingz checkout, fake HOME, fixture repo, env pointing the agy bin at the stub."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="agy-exec-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.plugins = self.tmp / "cc-thingz" / "plugins"
        _skill(self.plugins / "brainstorm", "brainstorm")
        _skill(self.plugins / "planning", "make-plan")
        _skill(self.plugins / "review", "code-review-refactor")
        _write(self.plugins / "skill-eval" / "hooks.json", '{"hooks": {}}\n')
        _skill(self.plugins / "thinking-tools", "root-cause-investigator")
        _skill(self.plugins / "workflow", "backlog")

        self.home = self.tmp / "realhome"
        for rel in AUTH:
            _write(self.home / rel, f'{{"token": "{rel}"}}\n')
        _write(self.home / ".gitconfig", "[user]\n\tname = real user\n")

        self.fixture = self.tmp / "fixture"
        _write(self.fixture / "app.py", "def fetch(key):\n    return key\n")
        _write(self.fixture / "README.md", "# demo\n")

        self.tree = self.tmp / "tree"
        _write(self.tree / "GEMINI.md", "<!-- base harness -->\n")
        self.stub_log = self.tmp / "stub.log"
        self.runs = self.tmp / "runs"

        env = {
            "HOME": str(self.home),
            "SKILLBENCH_PLUGINS_DIR": str(self.plugins),
            "SKILLBENCH_AUTH_FILES": ",".join(AUTH),
            "HARNESS2_AGY_BIN": str(STUB),
            "SKILLBENCH_EXEC_MODEL": "opus-5.5-high",
            "SKILLBENCH_EFFORT": "high",
            "AGY_STUB_LOG": str(self.stub_log),
            "AGY_STUB_SCENARIO": "skill_read",
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in (
            "SKILLBENCH_PLUGINS",
            "AGY_STUB_SLEEP",
            "AGY_STUB_HANG",
            "AGY_STUB_HANG_S",
            "AGY_STUB_EXIT_CODE",
        ):
            os.environ.pop(k, None)

        self.sub = AgyFromModule(
            tasks=lambda ref: AgyTask(prompt=PROMPT, fixture=self.fixture), runs_root=self.runs
        )
        ch = ChannelSpec("claude_cli", "m")
        self.spec = BenchSpec(
            name="skillbench",
            runs_root=self.tmp / "bench-runs",
            install_mode=INSTALL_MODE,
            prompts=BenchPrompts(work="solve coding tasks"),
            channels={"proposer": ch, "selector": ch},
        )
        self.real_home_before = _tree_hash(self.home)

    def tearDown(self) -> None:
        self.assertEqual(_tree_hash(self.home), self.real_home_before, "fake real HOME changed")

    def install(self, arm: str = "with", target: str = "make-plan"):
        inst = self.sub.materialise(self.tree, self.spec, {"arm": arm, "target_skill": target})
        self.addCleanup(shutil.rmtree, inst.staging_dir, True)
        return inst

    def run_one(self, run_id: str = "r0", timeout: int = 60, inst=None) -> Path:
        dirs = self.sub.execute("make-plan/t1", inst or self.install(), [run_id], timeout)
        self.assertEqual(dirs, [self.runs / run_id])
        return dirs[0]

    def timing(self, run_dir: Path) -> dict:
        return json.loads((run_dir / "skill_timing.json").read_text(encoding="utf-8"))

    def stub_calls(self) -> list[dict]:
        if not self.stub_log.is_file():
            return []
        return [json.loads(x) for x in self.stub_log.read_text(encoding="utf-8").splitlines()]

    def git(self, cwd: Path, *args: str) -> str:
        env = {**os.environ, "HOME": str(self.tmp / "githome"), "GIT_CONFIG_NOSYSTEM": "1"}
        return subprocess.run(
            ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
        ).stdout


class TestExecute(_ExecCase):
    def test_work_dir_is_committed_fixture_repo(self) -> None:
        rd = self.run_one()
        work = rd / "work"
        self.assertEqual((work / "app.py").read_text(), "def fetch(key):\n    return key\n")
        self.assertTrue((work / ".git").is_dir())
        log = self.git(work, "log", "--format=%an|%s").splitlines()
        self.assertEqual(log, ["skillbench|skillbench fixture"])
        self.assertIn("README.md", self.git(work, "show", "--name-only", "--format=", "HEAD"))

    def test_agy_invoked_with_expected_argv_home_and_cwd(self) -> None:
        rd = self.run_one(timeout=90)
        (call,) = self.stub_calls()
        self.assertEqual(
            call["argv"],  # the stub logs argv without the program name
            [
                "-p",
                PROMPT,
                "--output-format",
                "stream-json",
                "--dangerously-skip-permissions",
                "--disable-slash-commands",
                "--model",
                "opus-5.5-high",
                "--effort",
                "high",
                "--print-timeout",
                "90s",
            ],
        )
        flags = call["flags"]
        self.assertEqual(flags["prompt"], PROMPT)
        self.assertEqual(flags["model"], "opus-5.5-high")
        self.assertEqual(flags["effort"], "high")
        self.assertEqual(flags["output_format"], "stream-json")
        self.assertEqual(flags["print_timeout"], "90s")
        self.assertIs(flags["dangerously_skip_permissions"], True)
        self.assertIs(flags["disable_slash_commands"], True)
        self.assertNotIn("json_schema", flags)
        self.assertEqual(Path(call["home"]), rd / "home")
        self.assertEqual(Path(call["cwd"]).resolve(), (rd / "work").resolve())
        env = call["env"]
        self.assertEqual(env["EDITOR"], "true")
        self.assertEqual(env["VISUAL"], "true")
        self.assertEqual(Path(env["SKILLBENCH_SIDE_EFFECTS_DIR"]), rd / "side_effects")
        # the rollout home held the arm's plugins while agy ran
        self.assertTrue((rd / "home" / PLUGINS_REL / "planning" / "skills" / "make-plan").is_dir())

    def test_effort_and_model_follow_env(self) -> None:
        os.environ["SKILLBENCH_EFFORT"] = "low"
        os.environ["SKILLBENCH_EXEC_MODEL"] = "fable-5.1-high"
        os.environ["AGY_STUB_SCENARIO"] = "plain_answer"
        rd = self.run_one()
        (call,) = self.stub_calls()
        self.assertEqual(
            (call["flags"]["model"], call["flags"]["effort"]), ("fable-5.1-high", "low")
        )
        t = self.timing(rd)
        self.assertEqual(t["requested_model"], "fable-5.1-high")
        self.assertEqual(t["actual_model"], "fable-5.1-high")  # the stub echoes init.model

    def test_effort_unset_passes_no_effort_flag(self) -> None:
        # jet model ids pin the effort (opus-5.5-high), so SKILLBENCH_EFFORT is optional
        for unset in (None, ""):
            if unset is None:
                os.environ.pop("SKILLBENCH_EFFORT", None)
            else:
                os.environ["SKILLBENCH_EFFORT"] = unset
            with self.subTest(value=unset):
                rd = self.run_one(run_id=f"r-{unset!r}")
                call = self.stub_calls()[-1]
                self.assertNotIn("--effort", call["argv"])
                self.assertNotIn("effort", call["flags"])
                self.assertIs(self.timing(rd)["valid"], True)

    def test_auth_files_unset_runs_without_credentials(self) -> None:
        os.environ.pop("SKILLBENCH_AUTH_FILES")
        rd = self.run_one()
        self.assertIs(self.timing(rd)["valid"], True)
        for rel in AUTH:
            self.assertFalse((rd / "home" / rel).exists(), rel)

    def test_events_jsonl_has_monotonic_t(self) -> None:
        os.environ["AGY_STUB_SLEEP"] = "0.01"
        rd = self.run_one()
        lines = (rd / "events.jsonl").read_text(encoding="utf-8").splitlines()
        fixture_lines = (STUB.parent / "agy" / "skill_read.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), len([x for x in fixture_lines if x.strip()]))
        ts = [json.loads(x)["t"] for x in lines]
        self.assertTrue(all(isinstance(t, float) for t in ts))
        self.assertEqual(ts, sorted(ts))
        self.assertGreater(ts[-1], ts[0])
        self.assertGreaterEqual(ts[0], 0.0)

    def test_final_diff_new_files_and_skill_timing(self) -> None:
        rd = self.run_one()
        # the last DONE delta is "\n"; final.md is the full answer from result.response
        self.assertEqual(
            (rd / "final.md").read_text(),
            "Looking at the repository first.\nWriting the plan file.\n"
            "Wrote the plan to docs/plans/20261023-add-cache.md.\n",
        )
        diff = (rd / "diff.patch").read_text()
        self.assertIn("docs/plans/20261023-add-cache.md", diff)
        self.assertIn("+### Task 1: cache layer", diff)
        new = rd / "new_files" / "docs" / "plans" / "20261023-add-cache.md"
        self.assertTrue(new.read_text().startswith("# Add a cache"))
        self.assertEqual(
            [p.relative_to(rd / "new_files").as_posix() for p in (rd / "new_files").rglob("*.md")],
            ["docs/plans/20261023-add-cache.md"],
        )
        t = self.timing(rd)
        self.assertIs(t["invoked"], True)
        self.assertEqual(t["step_idx"], 6)  # the make-plan read, not the brainstorm one
        self.assertIsInstance(t["t_skill_s"], float)
        self.assertGreaterEqual(t["t_total_s"], t["t_skill_s"])
        self.assertEqual(t["requested_model"], "opus-5.5-high")
        self.assertEqual(t["actual_model"], "opus-5.5-high")
        self.assertIs(t["valid"], True)
        self.assertIsNone(t["invalid_reason"])
        self.assertEqual(t["exit_code"], 0)
        self.assertEqual(t["total_tokens"], 8731)

    def test_write_file_with_path_is_applied(self) -> None:
        scen = _scenario(
            self.tmp / "write_file.jsonl",
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "state": "ACTIVE",
                    "tool_name": "write_file",
                    "tool_info": {
                        "name": "write_file",
                        "parameters": {
                            "Path": "/tmp/skillbench-run/work/app.py",
                            "Content": "def fetch(key):\n    return memo(key)\n",
                        },
                    },
                },
            },
            {"event": "result", "result": {"response": "Done.", "usage": {"total_tokens": 5}}},
        )
        os.environ["AGY_STUB_SCENARIO"] = str(scen)
        rd = self.run_one()
        self.assertIn("+    return memo(key)", (rd / "diff.patch").read_text())
        self.assertEqual((rd / "final.md").read_text(), "Done.")

    def test_modified_file_shows_in_diff_not_new_files(self) -> None:
        scen = _scenario(
            self.tmp / "edit.jsonl",
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "state": "ACTIVE",
                    "tool_name": "write_to_file",
                    "tool_info": {
                        "name": "write_to_file",
                        "parameters": {
                            "TargetFile": "/tmp/skillbench-run/work/app.py",
                            "CodeContent": "def fetch(key):\n    return cache(key)\n",
                        },
                    },
                },
            },
            {"event": "result", "result": {"usage": {"total_tokens": 5}}},
        )
        os.environ["AGY_STUB_SCENARIO"] = str(scen)
        rd = self.run_one()
        diff = (rd / "diff.patch").read_text()
        self.assertIn("+    return cache(key)", diff)
        self.assertIn("-    return key", diff)
        self.assertEqual(list((rd / "new_files").iterdir()), [])
        self.assertEqual((rd / "final.md").read_text(), "")

    def test_plain_answer_not_invoked_and_empty_diff(self) -> None:
        os.environ["AGY_STUB_SCENARIO"] = "plain_answer"
        rd = self.run_one()
        t = self.timing(rd)
        self.assertEqual((t["invoked"], t["t_skill_s"], t["step_idx"]), (False, None, None))
        self.assertIs(t["valid"], True)
        self.assertEqual((rd / "diff.patch").read_text(), "")
        self.assertTrue((rd / "new_files").is_dir())
        self.assertEqual(
            (rd / "final.md").read_text(),
            "`add(a, b)` returns the sum of both arguments.\nNo changes were needed.",
        )

    def test_malformed_line_kept_verbatim_and_rep_valid(self) -> None:
        os.environ["AGY_STUB_SCENARIO"] = "malformed_line"
        rd = self.run_one()
        raw = (rd / "events.jsonl").read_text()
        self.assertIn('"tool_name":"vi\n', raw)
        self.assertIs(self.timing(rd)["valid"], True)

    def test_auth_files_scrubbed_from_run_home(self) -> None:
        rd = self.run_one()
        for rel in AUTH:
            self.assertFalse((rd / "home" / rel).exists(), rel)
        self.assertTrue((rd / "home" / PLUGINS_REL / "planning").is_dir())

    def test_rerun_replaces_run_dir(self) -> None:
        inst = self.install()
        rd = self.run_one(inst=inst)
        _write(rd / "stale.txt")
        _write(rd / "work" / "leftover.py")
        rd = self.run_one(inst=inst)
        self.assertFalse((rd / "stale.txt").exists())
        self.assertFalse((rd / "work" / "leftover.py").exists())
        self.assertEqual(len(self.git(rd / "work", "log", "--format=%H").split()), 1)

    def test_several_run_ids_one_rollout_each(self) -> None:
        dirs = self.sub.execute("make-plan/t1", self.install(), ["a", "b"], 60)
        self.assertEqual(dirs, [self.runs / "a", self.runs / "b"])
        calls = self.stub_calls()
        self.assertEqual(
            [Path(c["cwd"]).resolve() for c in calls], [(d / "work").resolve() for d in dirs]
        )
        self.assertTrue(all((d / "skill_timing.json").is_file() for d in dirs))

    def test_empty_fixture_gives_empty_repo(self) -> None:
        sub = AgyFromModule(tasks=lambda ref: AgyTask(prompt=PROMPT), runs_root=self.runs)
        os.environ["AGY_STUB_SCENARIO"] = "plain_answer"
        (rd,) = sub.execute("make-plan/t1", self.install(), ["e"], 60)
        self.assertEqual(sorted(p.name for p in (rd / "work").iterdir()), [".git"])
        self.assertIs(self.timing(rd)["valid"], True)

    def test_execute_without_resolver_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "task resolver"):
            AgyFromModule().execute("make-plan/t1", self.install(), ["r0"], 60)

    def test_invalid_effort_raises(self) -> None:
        inst = self.install()
        os.environ["SKILLBENCH_EFFORT"] = "extreme"
        with self.assertRaisesRegex(ConfigError, "SKILLBENCH_EFFORT"):
            self.run_one(inst=inst)
        self.assertEqual(self.stub_calls(), [])

    def test_missing_binary_raises(self) -> None:
        inst = self.install()
        os.environ["HARNESS2_AGY_BIN"] = str(self.tmp / "no-such-agy")
        with self.assertRaisesRegex(RuntimeError, "HARNESS2_AGY_BIN") as cm:
            self.run_one(inst=inst)
        self.assertIn(JETSKI_CLI, str(cm.exception))


class TestModelGuardAndFailures(_ExecCase):
    def test_model_field_disagreeing_invalidates(self) -> None:
        os.environ["AGY_STUB_SCENARIO"] = "result_model_mismatch"
        t = self.timing(self.run_one())
        self.assertEqual(t["actual_model"], "sonnet-4-high")
        self.assertIs(t["valid"], False)
        self.assertIn("model mismatch", t["invalid_reason"])
        self.assertIn("'opus-5.5-high'", t["invalid_reason"])
        self.assertIn("'sonnet-4-high'", t["invalid_reason"])

    def test_model_field_agreeing_is_recorded(self) -> None:
        os.environ["AGY_STUB_SCENARIO"] = "result_with_model"
        t = self.timing(self.run_one())
        self.assertEqual(t["actual_model"], "opus-5.5-high")
        self.assertIs(t["valid"], True)

    def test_init_model_of_another_jet_model_invalidates(self) -> None:
        os.environ["SKILLBENCH_EXEC_MODEL"] = "fable-5.1-high"
        os.environ["AGY_STUB_SCENARIO"] = "result_with_model"  # init.model = opus-5.5-high
        t = self.timing(self.run_one())
        self.assertIs(t["valid"], False)
        self.assertIn("model mismatch", t["invalid_reason"])

    def test_model_field_absent_keeps_rep_valid(self) -> None:
        os.environ["AGY_STUB_SCENARIO"] = "result_without_model"
        t = self.timing(self.run_one())
        self.assertIsNone(t["actual_model"])
        self.assertIs(t["valid"], True)
        self.assertIsNone(t["invalid_reason"])

    def test_timeout_invalid_and_run_dir_written(self) -> None:
        os.environ.update(AGY_STUB_HANG="1", AGY_STUB_HANG_S="60")
        t0 = time.monotonic()
        rd = self.run_one(timeout=1)
        self.assertLess(time.monotonic() - t0, 20)
        t = self.timing(rd)
        self.assertIs(t["valid"], False)
        self.assertEqual(t["invalid_reason"], "timeout after 1s")
        self.assertGreaterEqual(t["t_total_s"], 1.0)
        self.assertTrue((rd / "events.jsonl").read_text().strip())
        for name in ("final.md", "diff.patch", "agy.stderr.log"):
            self.assertTrue((rd / name).is_file(), name)
        self.assertTrue((rd / "new_files").is_dir())
        self.assertIs(t["invoked"], True)  # events before the hang still count

    def test_nonzero_exit_invalid_and_run_dir_written(self) -> None:
        os.environ["AGY_STUB_EXIT_CODE"] = "3"
        rd = self.run_one()
        t = self.timing(rd)
        self.assertIs(t["valid"], False)
        self.assertEqual(t["invalid_reason"], "agy exited with code 3")
        self.assertEqual(t["exit_code"], 3)
        self.assertIn("simulated failure", (rd / "agy.stderr.log").read_text())
        self.assertTrue((rd / "diff.patch").read_text())
        self.assertTrue((rd / "final.md").read_text())

    def test_stream_without_result_invalid(self) -> None:
        scen = _scenario(
            self.tmp / "noresult.jsonl",
            {
                "event": "step_update",
                "step_update": {"step_type": "agent_response", "state": "DONE", "text_delta": "hi"},
            },
        )
        os.environ["AGY_STUB_SCENARIO"] = str(scen)
        t = self.timing(self.run_one())
        self.assertIs(t["valid"], False)
        self.assertEqual(t["invalid_reason"], "no result event in stream")


class TestPureHelpers(unittest.TestCase):
    def test_model_matches(self) -> None:
        self.assertTrue(model_matches("opus", "opus"))
        self.assertTrue(model_matches("Opus", "OPUS"))
        self.assertTrue(model_matches("opus", "claude-opus-4-1"))
        self.assertFalse(model_matches("opus", "sonnet"))
        self.assertFalse(model_matches("opus", "opusx"))
        self.assertFalse(model_matches("", "opus"))

    def test_model_guard(self) -> None:
        def ev(model):
            return parse_event(json.dumps({"event": "result", "result": {"model": model}}))

        none = parse_event('{"event": "result", "result": {}}')
        self.assertEqual(model_guard([none], "opus"), (None, None))
        self.assertEqual(model_guard([ev("opus")], "opus"), ("opus", None))
        actual, reason = model_guard([ev("opus"), ev("sonnet")], "opus")
        self.assertEqual(actual, "sonnet")
        self.assertIn("mismatch", reason)

    def test_defaults_match_jet_model_ids(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SKILLBENCH_EXEC_MODEL", None)
            os.environ.pop("SKILLBENCH_EFFORT", None)
            self.assertEqual(DEFAULT_EXEC_MODEL, "opus-5.5-high")
            self.assertEqual(exec_model(), "opus-5.5-high")
            self.assertEqual(exec_effort(), "")
            for value in ("low", "medium", "high", "xhigh", "max"):
                os.environ["SKILLBENCH_EFFORT"] = value
                self.assertEqual(exec_effort(), value)
            os.environ["SKILLBENCH_EFFORT"] = "turbo"
            with self.assertRaisesRegex(ConfigError, "low|medium|high|xhigh|max"):
                exec_effort()

    def test_agy_argv_without_effort(self) -> None:
        argv = agy_argv("agy", "p", model="opus-5.5-high", timeout=5, effort="")
        self.assertNotIn("--effort", argv)
        self.assertEqual(argv[argv.index("--model") + 1], "opus-5.5-high")

    def test_final_text_prefers_result_response(self) -> None:
        done = parse_event(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "agent_response",
                        "state": "DONE",
                        "text_delta": "\n",
                    },
                }
            )
        )
        res = parse_event(json.dumps({"event": "result", "result": {"response": "full answer"}}))
        self.assertEqual(final_text([done, res]), "full answer")
        self.assertEqual(final_text([done]), "\n")

    def test_final_text_fallback_without_done(self) -> None:
        def ev(text, state="ACTIVE", step_type="agent_response"):
            su = {"step_type": step_type, "state": state, "text_delta": text}
            return parse_event(json.dumps({"event": "step_update", "step_update": su}))

        tool = parse_event(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {"step_type": "tool", "state": "DONE", "tool_name": "x"},
                }
            )
        )
        self.assertEqual(final_text([ev("a"), tool, ev("b"), ev("c")]), "bc")
        self.assertEqual(
            final_text([ev("a"), ev("final", "DONE"), ev("t", "x", "thinking")]), "final"
        )
        self.assertEqual(final_text([]), "")


class TestPreflight(_ExecCase):
    def test_passes_and_starts_nothing(self) -> None:
        self.sub.preflight()
        self.assertEqual(self.stub_calls(), [])

    def test_missing_binary_path(self) -> None:
        os.environ["HARNESS2_AGY_BIN"] = str(self.tmp / "nope" / "agy")
        with self.assertRaisesRegex(RuntimeError, "HARNESS2_AGY_BIN") as cm:
            self.sub.preflight()
        # points at the Jetski CLI and warns that a shell alias/function cannot be used
        self.assertIn(JETSKI_CLI, str(cm.exception))
        self.assertIn("alias", str(cm.exception))

    def test_missing_binary_on_path(self) -> None:
        os.environ["HARNESS2_AGY_BIN"] = "agy-definitely-not-installed-xyz"
        with self.assertRaisesRegex(RuntimeError, "not found"):
            self.sub.preflight()

    def test_non_executable_binary(self) -> None:
        plain = _write(self.tmp / "agy-plain", "#!/bin/sh\n")
        plain.chmod(0o644)
        os.environ["HARNESS2_AGY_BIN"] = str(plain)
        with self.assertRaisesRegex(RuntimeError, "not executable"):
            self.sub.preflight()

    def test_missing_auth_file(self) -> None:
        os.environ["SKILLBENCH_AUTH_FILES"] = ",".join((*AUTH, ".gemini/missing.json"))
        with self.assertRaisesRegex(RuntimeError, r"SKILLBENCH_AUTH_FILES.*\n?.*") as cm:
            self.sub.preflight()
        self.assertIn(".gemini/missing.json", str(cm.exception))
        self.assertIn("log in with `agy`", str(cm.exception))

    def test_auth_files_unset_passes(self) -> None:
        del os.environ["SKILLBENCH_AUTH_FILES"]
        self.sub.preflight()
        os.environ["SKILLBENCH_AUTH_FILES"] = ""
        self.sub.preflight()

    def test_bad_auth_entry_fails(self) -> None:
        os.environ["SKILLBENCH_AUTH_FILES"] = "../outside.json"
        with self.assertRaisesRegex(RuntimeError, "SKILLBENCH_AUTH_FILES"):
            self.sub.preflight()

    def test_missing_plugins_dir(self) -> None:
        del os.environ["SKILLBENCH_PLUGINS_DIR"]
        with self.assertRaisesRegex(RuntimeError, "SKILLBENCH_PLUGINS_DIR"):
            self.sub.preflight()
        os.environ["SKILLBENCH_PLUGINS_DIR"] = str(self.tmp / "no-plugins")
        with self.assertRaisesRegex(RuntimeError, "not a directory"):
            self.sub.preflight()

    def test_missing_allowlisted_plugin(self) -> None:
        os.environ["SKILLBENCH_PLUGINS"] = "planning,revmux"
        with self.assertRaisesRegex(RuntimeError, "revmux"):
            self.sub.preflight()

    def test_effort_unset_passes(self) -> None:
        del os.environ["SKILLBENCH_EFFORT"]
        self.sub.preflight()

    def test_invalid_effort(self) -> None:
        os.environ["SKILLBENCH_EFFORT"] = "extreme"
        with self.assertRaisesRegex(RuntimeError, "SKILLBENCH_EFFORT"):
            self.sub.preflight()


class TestRegistration(unittest.TestCase):
    def test_exported_from_substrate_package(self) -> None:
        from harness2.substrate import AGY_SURFACE, AgySubstrate

        self.assertIs(AgySubstrate, AgyFromModule)
        self.assertEqual(AgySubstrate.name, "agy")
        self.assertIs(AgySubstrate().surface(), AGY_SURFACE)

    def test_cli_substrates_include_agy_reserved_for_skillbench(self) -> None:
        from harness2.cli._common import SUBSTRATES, make_adapter

        self.assertEqual(SUBSTRATES[:2], ("oc", "cx"))
        self.assertIn("agy", SUBSTRATES)
        for bench in ("jb", "lab"):
            with self.assertRaisesRegex(ValueError, "skillbench"):
                make_adapter(bench, None, "agy")
        with self.assertRaisesRegex(ValueError, "skillbench"):
            make_adapter("wb", "code", "agy")


if __name__ == "__main__":
    unittest.main()
