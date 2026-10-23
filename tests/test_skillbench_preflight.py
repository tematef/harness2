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

"""Skillbench path through `preflight`, `config`, `run_sweep`, `judge`, and `check`.

Every test runs in temporary directories with a fake HOME and `tests/fixtures/agy_stub.py`;
the user's real `$HOME` and `~/.gemini` are never touched.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness2 import config, preflight
from harness2.benchmarks import skillbench as SB
from harness2.cli import _common as C
from harness2.cli import check as check_cli
from harness2.cli import judge as judge_cli
from harness2.cli import run_sweep as sweep_cli
from harness2.core import schedule
from harness2.substrate.opencode import OpencodeSubstrate
from tests.test_skillbench_tasks import make_task

STUB = Path(__file__).resolve().parent / "fixtures" / "agy_stub.py"
AUTH = (".gemini/oauth_creds.json", ".gemini/google_accounts.json")


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


class _PreflightCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-preflight-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.home = self.tmp / "realhome"
        for rel in AUTH:
            _write(self.home / rel, f'{{"token": "{rel}"}}\n')
        _write(self.home / ".gemini" / "GEMINI.md", "real user rules\n")

        self.plugins = self.tmp / "cc-thingz" / "plugins"
        _skill(self.plugins / "brainstorm", "brainstorm")
        _skill(self.plugins / "planning", "make-plan")
        _skill(self.plugins / "review", "code-review-refactor")
        _write(self.plugins / "skill-eval" / "hooks.json", '{"hooks": {}}\n')
        _skill(self.plugins / "thinking-tools", "root-cause-investigator")
        _skill(self.plugins / "workflow", "backlog")

        self.tasks_root = self.tmp / "tasks"
        make_task(
            self.tasks_root,
            "make-plan",
            "p1",
            pilot=True,
            prompt="Write an implementation plan for adding caching to fetch() in utils.py.",
            user_answers=["Use an in-memory LRU cache", "No external dependencies"],
        )
        make_task(
            self.tasks_root,
            "code-review-refactor",
            "t1",
            pilot=False,
            prompt="Clean up duplicate validation logic in utils.py.",
        )
        SB.freeze(self.tasks_root)

        self.runs_dir = self.tmp / "runs"
        self.stub_log = self.tmp / "stub.log"
        env = {
            "HOME": str(self.home),
            "HARNESS2_RUNS_DIR": str(self.runs_dir),
            "HARNESS2_AGY_BIN": str(STUB),
            "SKILLBENCH_PLUGINS_DIR": str(self.plugins),
            "SKILLBENCH_AUTH_FILES": ",".join(AUTH),
            "SKILLBENCH_EXEC_MODEL": "opus",
            "SKILLBENCH_JUDGE_MODEL": "fable",
            "SKILLBENCH_EFFORT": "high",
            "AGY_STUB_LOG": str(self.stub_log),
            "AGY_STUB_SCENARIO": "skill_read",
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in (
            "HARNESS2_SOLVER_MODEL",
            "HARNESS2_IMPROVER_MODEL",
            "HARNESS2_JUDGE_MODEL",
            "HARNESS2_CODEX_MODEL",
            "GOOGLE_CLOUD_PROJECT",
            "HARNESS2_VERTEX_PROJECT",
            "SKILLBENCH_PLUGINS",
            "SKILLBENCH_PILOT",
            "SKILLBENCH_JUDGE_REPEAT",
            "AGY_STUB_SLEEP",
            "AGY_STUB_HANG",
            "AGY_STUB_HANG_S",
            "AGY_STUB_EXIT_CODE",
        ):
            os.environ.pop(k, None)

        for attr, val in (
            ("RUNS", self.runs_dir),
            ("SOLVER_MODEL_OC", ""),
            ("SOLVER_MODEL_CX", ""),
            ("IMPROVER_MODEL", ""),
            ("JUDGE_MODEL", ""),
            ("VERTEX_PROJECT", ""),
        ):
            p = mock.patch.object(config, attr, val)
            p.start()
            self.addCleanup(p.stop)

        tasks_root_patcher = mock.patch.object(SB, "TASKS_ROOT", self.tasks_root)
        tasks_root_patcher.start()
        self.addCleanup(tasks_root_patcher.stop)

        self.real_home_before = _tree_hash(self.home)

    def tearDown(self) -> None:
        self.assertEqual(_tree_hash(self.home), self.real_home_before, "fake real HOME modified")


class TestSkillbenchPreflight(_PreflightCase):
    def test_preflight_check_runs_only_agy_substrate_and_adapter_preflight(self) -> None:
        adapter = C.make_adapter("skillbench", None, "agy", arm="with")
        with (
            mock.patch.object(
                preflight,
                "missing_runners",
                side_effect=AssertionError("missing_runners must not run for skillbench"),
            ),
            mock.patch.object(
                OpencodeSubstrate,
                "preflight",
                side_effect=AssertionError("OpencodeSubstrate.preflight must not run"),
            ),
            mock.patch.object(
                preflight.subprocess,
                "run",
                side_effect=AssertionError("no git/gcloud subprocess probes for skillbench"),
            ),
            mock.patch.object(
                preflight.subprocess,
                "check_output",
                side_effect=AssertionError("no dependency pin check_output for skillbench"),
            ),
            mock.patch.object(
                adapter.substrate, "preflight", wraps=adapter.substrate.preflight
            ) as spy_sub,
            mock.patch.object(adapter, "preflight", wraps=adapter.preflight) as spy_adapter,
        ):
            self.assertEqual(preflight.check(adapter, judge=False), [])
            self.assertEqual(preflight.check(adapter, judge=True), [])
            preflight.require_ready(adapter, judge=True)

        self.assertGreaterEqual(spy_sub.call_count, 3)
        self.assertEqual(spy_adapter.call_count, 3)

    def test_preflight_check_reports_agy_and_adapter_errors(self) -> None:
        adapter = C.make_adapter("skillbench", None, "agy", arm="with")

        with mock.patch.dict(os.environ, {"HARNESS2_AGY_BIN": str(self.tmp / "no-such-agy")}):
            errs = preflight.check(adapter, judge=True)
            self.assertEqual(len(errs), 1)
            self.assertIn("HARNESS2_AGY_BIN", errs[0])
            with self.assertRaises(config.ConfigError):
                preflight.require_ready(adapter, judge=True)

        with mock.patch.dict(os.environ, {"SKILLBENCH_EFFORT": ""}):
            errs = preflight.check(adapter, judge=True)
            self.assertEqual(len(errs), 1)
            self.assertIn("SKILLBENCH_EFFORT", errs[0])

        with mock.patch.dict(os.environ, {"SKILLBENCH_PLUGINS_DIR": str(self.tmp / "missing")}):
            errs = preflight.check(adapter, judge=True)
            self.assertEqual(len(errs), 1)
            self.assertIn("SKILLBENCH_PLUGINS_DIR", errs[0])

        with mock.patch.dict(os.environ, {"SKILLBENCH_AUTH_FILES": ".gemini/missing_auth.json"}):
            errs = preflight.check(adapter, judge=True)
            self.assertEqual(len(errs), 1)
            self.assertIn("SKILLBENCH_AUTH_FILES", errs[0])

        with mock.patch.dict(os.environ, {"SKILLBENCH_JUDGE_REPEAT": "0"}):
            errs = preflight.check(adapter, judge=True)
            self.assertEqual(len(errs), 1)
            self.assertIn("SKILLBENCH_JUDGE_REPEAT", errs[0])

        # Drift a task file after adapter construction
        _write(self.tasks_root / "make-plan" / "p1" / "fixture" / "extra.txt", "drifted\n")
        errs = preflight.check(adapter, judge=True)
        self.assertEqual(len(errs), 1)
        self.assertIn("drifted", errs[0])


class TestModelsAndProposerSkippedForSkillbench(_PreflightCase):
    def test_validate_models_skipped_for_skillbench_and_required_for_other_benches(self) -> None:
        # Neither HARNESS2_SOLVER_MODEL nor HARNESS2_JUDGE_MODEL is set in _PreflightCase
        config.validate_models(judge=False, bench="skillbench")
        config.validate_models(judge=True, bench="skillbench")

        for bench in (None, "jb", "lab", "wb"):
            with self.subTest(bench=bench):
                with self.assertRaisesRegex(config.ConfigError, "HARNESS2_SOLVER_MODEL"):
                    config.validate_models(judge=False, bench=bench)
                with self.assertRaisesRegex(config.ConfigError, "HARNESS2_SOLVER_MODEL"):
                    config.validate_models(judge=True, bench=bench)

        with (
            mock.patch.dict(
                os.environ,
                {
                    "HARNESS2_SOLVER_MODEL": "openai/gpt-4o",
                    "HARNESS2_IMPROVER_MODEL": "openai/gpt-4o",
                },
            ),
            mock.patch.object(config, "SOLVER_MODEL_OC", "openai/gpt-4o"),
            mock.patch.object(config, "IMPROVER_MODEL", "openai/gpt-4o"),
        ):
            config.validate_models(judge=False, bench="jb")
            with self.assertRaisesRegex(config.ConfigError, "HARNESS2_JUDGE_MODEL"):
                config.validate_models(judge=True, bench="jb")

    def test_skillbench_run_and_judge_succeed_without_solver_or_judge_models(self) -> None:
        # 1. Mocked schedule.run_sweep: sweep_cli.main returns 0, Proposer never constructed,
        # require_ready runs for real without HARNESS2_SOLVER_MODEL / HARNESS2_JUDGE_MODEL.
        fake_rep = {
            "domains": [{"domain": "make-plan", "done": 1, "failed": 0, "tasks": []}],
        }
        buf = io.StringIO()
        with (
            mock.patch.object(
                sweep_cli,
                "Proposer",
                side_effect=AssertionError("Proposer must not be constructed for skillbench"),
            ),
            mock.patch.object(schedule, "run_sweep", return_value=fake_rep) as spy_sched,
            contextlib.redirect_stdout(buf),
        ):
            rc = sweep_cli.main(
                [
                    "--bench",
                    "skillbench",
                    "--arm",
                    "with",
                    "--k",
                    "1",
                    "--only-domains",
                    "make-plan",
                ]
            )
        self.assertEqual(rc, 0)
        self.assertIsNone(spy_sched.call_args.kwargs.get("prop"))

        # 2. Real base rollout + judge --bases without HARNESS2_SOLVER_MODEL / HARNESS2_JUDGE_MODEL
        shutil.rmtree(self.runs_dir, ignore_errors=True)
        buf = io.StringIO()
        with (
            mock.patch.object(
                sweep_cli,
                "Proposer",
                side_effect=AssertionError("Proposer must not be constructed for skillbench"),
            ),
            contextlib.redirect_stdout(buf),
        ):
            rc_sweep = sweep_cli.main(
                [
                    "--bench",
                    "skillbench",
                    "--arm",
                    "with",
                    "--k",
                    "1",
                    "--only-domains",
                    "make-plan",
                ]
            )
        self.assertIn(rc_sweep, (0, 2))

        judge_scen = _scenario(
            self.tmp / "judge_ok.jsonl",
            {
                "event": "result",
                "result": {
                    "structured_output": {
                        "criteria": [
                            {"id": "c0", "pass": True, "reason": "ok"},
                            {"id": "c1", "pass": True, "reason": "ok"},
                            {"id": "c2", "pass": True, "reason": "ok"},
                            {"id": "c3", "pass": False, "reason": "no"},
                        ]
                    },
                    "usage": {"total_tokens": 100},
                },
            },
        )
        os.environ["AGY_STUB_SCENARIO"] = str(judge_scen)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_judge = judge_cli.main(
                ["--bench", "skillbench", "--arm", "with", "--k", "1", "--bases"]
            )
        self.assertEqual(rc_judge, 0)

        ns_dir = self.runs_dir / "skillbench" / "adapt" / "skillbench-agy-agg1-with"
        judged = json.loads((ns_dir / "base_judged.json").read_text(encoding="utf-8"))
        self.assertEqual(len(judged), 1)
        self.assertEqual(judged[0]["score"], {"n_passed": 3, "n_criteria": 4})

    def test_other_benches_still_require_solver_and_judge_models(self) -> None:
        with mock.patch("harness2.benchmarks.jobbench.discover", return_value={"acc/t1": "acc"}):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = sweep_cli.main(["--bench", "jb", "--k", "1"])
            self.assertEqual(rc, 1)
            self.assertIn("HARNESS2_SOLVER_MODEL", err.getvalue())

            with self.assertRaisesRegex(config.ConfigError, "HARNESS2_SOLVER_MODEL"):
                judge_cli.main(["--bench", "jb", "--bases"])


class TestSkillbenchIdentityCheck(_PreflightCase):
    def _run_mocked_sweep(self, arm: str = "with") -> int:
        fake_rep = {
            "domains": [{"domain": "make-plan", "done": 1, "failed": 0, "tasks": []}],
        }
        buf = io.StringIO()
        with (
            mock.patch.object(schedule, "run_sweep", return_value=fake_rep),
            contextlib.redirect_stdout(buf),
        ):
            return sweep_cli.main(
                [
                    "--bench",
                    "skillbench",
                    "--arm",
                    arm,
                    "--k",
                    "1",
                    "--only-domains",
                    "make-plan",
                ]
            )

    def test_check_identity_records_skillbench_fields_and_refuses_resume_on_change(self) -> None:
        expected_hash = SB.check_manifest(self.tasks_root)
        self.assertEqual(self._run_mocked_sweep("with"), 0)

        ns_dir = self.runs_dir / "skillbench" / "adapt" / "skillbench-agy-agg1-with"
        cfg = json.loads((ns_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["executor"], "opus")
        self.assertEqual(cfg["judge"], "fable")
        self.assertEqual(cfg["effort"], "high")
        self.assertEqual(cfg["manifest_hash"], expected_hash)
        self.assertEqual(cfg["tag"], "with")

        # Identical settings resume cleanly
        self.assertEqual(self._run_mocked_sweep("with"), 0)

        # Changing SKILLBENCH_EXEC_MODEL refuses resume
        with mock.patch.dict(os.environ, {"SKILLBENCH_EXEC_MODEL": "sonnet"}):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(self._run_mocked_sweep("with"), 1)
            self.assertIn("different models or tasks", err.getvalue())

        # Changing SKILLBENCH_JUDGE_MODEL refuses resume
        with mock.patch.dict(os.environ, {"SKILLBENCH_JUDGE_MODEL": "opus"}):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(self._run_mocked_sweep("with"), 1)
            self.assertIn("different models or tasks", err.getvalue())

        # Changing SKILLBENCH_EFFORT refuses resume
        with mock.patch.dict(os.environ, {"SKILLBENCH_EFFORT": "low"}):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(self._run_mocked_sweep("with"), 1)
            self.assertIn("different models or tasks", err.getvalue())

        # Changing corpus + re-freezing (manifest_hash change) refuses resume
        _write(self.tasks_root / "make-plan" / "p1" / "fixture" / "changed.py", "x = 2\n")
        SB.freeze(self.tasks_root, force=True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(self._run_mocked_sweep("with"), 1)
        self.assertIn("different models or tasks", err.getvalue())

    def test_judge_identity_check_compares_skillbench_judge_model(self) -> None:
        self.assertEqual(self._run_mocked_sweep("with"), 0)
        adapter = C.make_adapter("skillbench", None, "agy", arm="with")
        ns = C.summary_namespace(adapter, 1, seed=None, tag="with")
        ns_dir = C.runs_root_for(adapter) / ns
        run_dir = adapter.execute("make-plan/p1", None, f"{ns}/make-plan/p1/base1")
        _write(
            ns_dir / "make-plan__p1" / "summary.json",
            json.dumps(
                {
                    "task_id": "make-plan/p1",
                    "tag": "with",
                    "status": "ok",
                    "base": [
                        {
                            "label": "base1",
                            "run_id": f"{ns}/make-plan/p1/base1",
                            "run_dir": str(run_dir),
                            "ok": True,
                            "delivered": True,
                        }
                    ],
                },
                indent=2,
            ),
        )
        judge_scen = _scenario(
            self.tmp / "judge_id.jsonl",
            {
                "event": "result",
                "result": {
                    "structured_output": {
                        "criteria": [
                            {"id": "c0", "pass": True, "reason": "ok"},
                            {"id": "c1", "pass": True, "reason": "ok"},
                            {"id": "c2", "pass": True, "reason": "ok"},
                            {"id": "c3", "pass": True, "reason": "ok"},
                        ]
                    },
                    "usage": {"total_tokens": 80},
                },
            },
        )
        os.environ["AGY_STUB_SCENARIO"] = str(judge_scen)

        # Even if config.JUDGE_MODEL differs, skillbench compares SKILLBENCH_JUDGE_MODEL
        with (
            mock.patch.object(config, "JUDGE_MODEL", "unrelated-vertex-judge"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc = judge_cli.main(["--bench", "skillbench", "--arm", "with", "--bases"])
        self.assertEqual(rc, 0)

        # Changing SKILLBENCH_JUDGE_MODEL fails the identity check
        with mock.patch.dict(os.environ, {"SKILLBENCH_JUDGE_MODEL": "opus"}):
            with self.assertRaisesRegex(SystemExit, "Judge model changed"):
                judge_cli.main(["--bench", "skillbench", "--arm", "with", "--bases"])


class TestCheckCliSkillbench(_PreflightCase):
    def test_harness2_check_skillbench_agy_works(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = check_cli.main(["--bench", "skillbench", "--substrate", "agy"])
        self.assertEqual(rc, 0)
        self.assertIn("Ready: skillbench/agy; 2 tasks.", buf.getvalue())

        # Omitted --substrate defaults to agy for skillbench
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = check_cli.main(["--bench", "skillbench"])
        self.assertEqual(rc, 0)
        self.assertIn("Ready: skillbench/agy; 2 tasks.", buf.getvalue())

        # Explicit --arm without also works
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = check_cli.main(
                ["--bench", "skillbench", "--substrate", "agy", "--arm", "without"]
            )
        self.assertEqual(rc, 0)
        self.assertIn("Ready: skillbench/agy; 2 tasks.", buf.getvalue())

    def test_harness2_check_skillbench_reports_failures(self) -> None:
        with mock.patch.dict(os.environ, {"SKILLBENCH_EFFORT": ""}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = check_cli.main(["--bench", "skillbench", "--substrate", "agy"])
            self.assertEqual(rc, 1)
            self.assertIn("FAIL:", buf.getvalue())
            self.assertIn("SKILLBENCH_EFFORT", buf.getvalue())

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = check_cli.main(["--bench", "skillbench", "--substrate", "oc"])
        self.assertEqual(rc, 1)
        self.assertIn("FAIL:", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
