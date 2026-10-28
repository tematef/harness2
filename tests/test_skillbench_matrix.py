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

"""Tests for Task 6: `skillbench-matrix` runner, budget estimator, resume and identity."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from harness2 import config
from harness2.benchmarks import skillbench as SB
from harness2.benchmarks import skillbench_suite as SS
from harness2.cli import judge as judge_cli
from harness2.cli import run_sweep as sweep_cli
from harness2.cli import skillbench_matrix as SM
from harness2.cli import skillbench_report as SR
from harness2.core import schedule
from harness2.substrate import agy

ROOT = Path(__file__).resolve().parents[1]
STUB = ROOT / "tests" / "fixtures" / "agy_stub.py"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_plugin(plugins_dir: Path, plugin_name: str, skills: list[str]) -> None:
    pdir = plugins_dir / plugin_name
    _write(pdir / "plugin.json", json.dumps({"name": f"test-{plugin_name}"}) + "\n")
    for s in skills:
        _write(
            pdir / "skills" / s / "SKILL.md",
            f"---\nname: {s}\ndescription: Skill {s}\n---\n# {s}\nFollow {s}.\n",
        )


def _make_skill_task(tasks_root: Path, skill: str, tid: str, *, pilot: bool = True) -> None:
    tdir = tasks_root / skill / tid
    _write(
        tdir / "task.toml",
        f'target_skill = "{skill}"\n'
        f'kind = "skill"\n'
        f"pilot = {'true' if pilot else 'false'}\n"
        f"timeout = 60\n"
        f'prompt = "Implement {tid} in app.py."\n',
    )
    _write(
        tdir / "rubric.json",
        json.dumps([{"id": f"c{i}", "criterion": f"Criterion {i} passes."} for i in range(4)]),
    )
    _write(tdir / "fixture" / "app.py", "def run():\n    return 1\n")


def _make_suite_task(tasks_root: Path, family: str, tid: str, *, pilot: bool = True) -> None:
    tdir = tasks_root / "_suite" / tid
    _write(
        tdir / "task.toml",
        f'kind = "suite"\n'
        f'family = "{family}"\n'
        f"pilot = {'true' if pilot else 'false'}\n"
        f"timeout = 60\n"
        f'prompt = "Plan and implement {tid} in app.py."\n',
    )
    _write(
        tdir / "rubric.json",
        json.dumps(
            [{"id": f"c{i}", "criterion": f"Suite criterion {i} passes."} for i in range(4)]
        ),
    )
    _write(tdir / "fixture" / "app.py", "def run():\n    return 1\n")


class _MatrixCase(unittest.TestCase):
    def setUp(self) -> None:
        self._env_snapshot = os.environ.copy()
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-matrix-test-"))
        self.runs_dir = self.tmp / "runs"
        self.runs_root = self.runs_dir / "skillbench"
        self.plugins_dir = self.tmp / "alpha-plugins"
        _make_plugin(self.plugins_dir, "planning", ["plan-skill", "exec-skill"])
        _make_plugin(self.plugins_dir, "review", ["review-skill"])

        self.beta_plugins_dir = self.tmp / "beta-plugins"
        _make_plugin(self.beta_plugins_dir, "core", ["beta-plan"])

        self.tasks_root = self.tmp / "tasks"
        _make_skill_task(self.tasks_root, "plan-skill", "t1")
        _make_skill_task(self.tasks_root, "plan-skill", "t2")
        _make_skill_task(self.tasks_root, "review-skill", "r1")
        _make_suite_task(self.tasks_root, "planning", "s1")
        _make_suite_task(self.tasks_root, "planning", "s2")
        SB.freeze(self.tasks_root)

        self.suite_toml = _write(
            self.tmp / "alpha.toml",
            f'name = "alpha"\n'
            f'plugins_dir = "{self.plugins_dir}"\n'
            f'plugins = ["planning", "review"]\n'
            f'tasks_root = "{self.tasks_root}"\n'
            f"[families]\n"
            f'planning = ["plan-skill", "exec-skill"]\n'
            f"[matrix]\n"
            f'executors = ["opus-5.5-high", "gemini-3.8-flash-high"]\n'
            f"[matrix.judge_for]\n"
            f'"opus-" = "gemini-3.8-flash-high"\n'
            f'"gemini-" = "fable-5.1-high"\n'
            f'"fable-" = "gemini-3.8-flash-high"\n',
        )
        self.beta_toml = _write(
            self.tmp / "beta.toml",
            f'name = "beta"\n'
            f'plugins_dir = "{self.beta_plugins_dir}"\n'
            f'plugins = ["core"]\n'
            f'tasks_root = "{self.tasks_root}"\n',
        )
        self.suite = SS.load_suite(self.suite_toml)
        self.beta_suite = SS.load_suite(self.beta_toml)

        os.environ.update(
            HARNESS2_RUNS_DIR=str(self.runs_dir),
            HARNESS2_AGY_BIN=str(STUB),
            SKILLBENCH_SUITE=str(self.suite_toml),
            SKILLBENCH_EFFORT="high",
        )
        for k in (
            "SKILLBENCH_PLUGINS_DIR",
            "SKILLBENCH_PLUGINS",
            "SKILLBENCH_TASKS_ROOT",
            "SKILLBENCH_MATRIX",
            "SKILLBENCH_RUNS_ROOT",
        ):
            os.environ.pop(k, None)

        p = mock.patch.object(config, "RUNS", self.runs_dir)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_snapshot)
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)


class TestJobExpansion(_MatrixCase):
    def test_expand_jobs_for_each_arms_value(self) -> None:
        # 3 skill tasks (plan-skill/t1, plan-skill/t2, review-skill/r1)
        # 2 suite tasks (_suite/s1, _suite/s2)
        # 2 executors (opus-5.5-high, gemini-3.8-flash-high), k=2

        # 1) --arms skill -> ("with", "without") over 3 skill tasks = 2 * 2 * 3 * 2 = 24 jobs
        skill_jobs = SM.expand_jobs(self.suite, k=2, arms="skill")
        self.assertEqual(len(skill_jobs), 24)
        self.assertEqual({j.arm for j in skill_jobs}, {"with", "without"})
        self.assertEqual({j.arm_kind for j in skill_jobs}, {"skill"})
        self.assertEqual(
            {j.task_id for j in skill_jobs},
            {"plan-skill/t1", "plan-skill/t2", "review-skill/r1"},
        )
        self.assertEqual({j.rep for j in skill_jobs}, {1, 2})
        for j in skill_jobs:
            if j.executor == "opus-5.5-high":
                self.assertEqual(j.judge, "gemini-3.8-flash-high")
            else:
                self.assertEqual(j.executor, "gemini-3.8-flash-high")
                self.assertEqual(j.judge, "fable-5.1-high")

        # 2) --arms suite -> ("baseline", "suite-alpha") over 2 suite tasks = 2 * 2 * 2 * 2 = 16 jobs
        suite_jobs = SM.expand_jobs(self.suite, k=2, arms="suite")
        self.assertEqual(len(suite_jobs), 16)
        self.assertEqual({j.arm for j in suite_jobs}, {"baseline", "suite-alpha"})
        self.assertEqual({j.arm_kind for j in suite_jobs}, {"suite"})
        self.assertEqual({j.task_id for j in suite_jobs}, {"_suite/s1", "_suite/s2"})

        # --arms suite with --suites beta.toml -> ("baseline", "suite-alpha", "suite-beta") = 2 * 3 * 2 * 2 = 24
        multi_suite_jobs = SM.expand_jobs(
            self.suite, k=2, arms="suite", suites=[self.beta_toml]
        )
        self.assertEqual(len(multi_suite_jobs), 24)
        self.assertEqual(
            {j.arm for j in multi_suite_jobs},
            {"baseline", "suite-alpha", "suite-beta"},
        )

        # 3) --arms both -> 24 skill jobs + 16 suite jobs = 40 jobs
        both_jobs = SM.expand_jobs(self.suite, k=2, arms="both")
        self.assertEqual(len(both_jobs), 40)
        self.assertEqual(
            {j.arm for j in both_jobs},
            {"with", "without", "baseline", "suite-alpha"},
        )

        # 4) --arms baseline -> control arms only: "without" on 3 skill tasks + "baseline" on 2 suite tasks
        # = (3 + 2) * 2 executors * 2 reps = 20 jobs
        base_jobs = SM.expand_jobs(self.suite, k=2, arms="baseline")
        self.assertEqual(len(base_jobs), 20)
        self.assertEqual({j.arm for j in base_jobs}, {"without", "baseline"})
        without_tasks = {j.task_id for j in base_jobs if j.arm == "without"}
        baseline_tasks = {j.task_id for j in base_jobs if j.arm == "baseline"}
        self.assertEqual(without_tasks, {"plan-skill/t1", "plan-skill/t2", "review-skill/r1"})
        self.assertEqual(baseline_tasks, {"_suite/s1", "_suite/s2"})

    def test_expand_jobs_respects_custom_executors_and_domain_filters(self) -> None:
        jobs = SM.expand_jobs(
            self.suite,
            executors="fable-5.1-high",
            k=3,
            arms="skill",
            only_domains="review-skill",
        )
        # 1 executor * 2 arms ("with", "without") * 1 task ("review-skill/r1") * 3 reps = 6
        self.assertEqual(len(jobs), 6)
        self.assertEqual({j.executor for j in jobs}, {"fable-5.1-high"})
        self.assertEqual({j.judge for j in jobs}, {"gemini-3.8-flash-high"})
        self.assertEqual({j.task_id for j in jobs}, {"review-skill/r1"})


class TestJudgeValidation(_MatrixCase):
    def test_judge_same_family_refused(self) -> None:
        # Direct mapping with same prefix
        with self.assertRaisesRegex(SS.SuiteError, "must not grade its own family"):
            SM.resolve_judge("opus-5.5-high", {"opus-": "opus-5.5-low"})

        # Longer executor prefix mapped to another model in the same family
        with self.assertRaisesRegex(SS.SuiteError, "must not grade its own family"):
            SM.resolve_judge("opus-5.5-high", {"opus-5.5": "opus-4.6-high"})

        # Via expand_jobs
        with self.assertRaisesRegex(SS.SuiteError, "must not grade its own family"):
            SM.expand_jobs(
                self.suite,
                executors=["opus-5.5-high"],
                judge_for={"opus-": "opus-5.5-high"},
            )

        # Via CLI main with a suite whose judge_for maps a custom executor to its own family
        bad_suite_toml = _write(
            self.tmp / "bad-judge.toml",
            f'name = "bad"\n'
            f'plugins_dir = "{self.plugins_dir}"\n'
            f'plugins = ["planning"]\n'
            f'tasks_root = "{self.tasks_root}"\n'
            f"[matrix.judge_for]\n"
            f'"sonnet-" = "sonnet-4.6-low"\n',
        )
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = SM.main(
                [
                    "--suite",
                    str(bad_suite_toml),
                    "--executors",
                    "sonnet-4.6-high",
                    "--dry-run",
                ]
            )
        self.assertEqual(rc, 1)
        self.assertIn("must not grade its own family", err.getvalue())

    def test_missing_judge_mapping_error(self) -> None:
        with self.assertRaisesRegex(SS.SuiteError, "no \\[matrix.judge_for\\] prefix matches"):
            SM.resolve_judge("sonnet-4.6-high", self.suite)

        with self.assertRaisesRegex(SS.SuiteError, "no \\[matrix.judge_for\\] prefix matches"):
            SM.expand_jobs(self.suite, executors=["sonnet-4.6-high"])

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = SM.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--executors",
                    "sonnet-4.6-high",
                    "--dry-run",
                ]
            )
        self.assertEqual(rc, 1)
        self.assertIn("no [matrix.judge_for] prefix matches", err.getvalue())


class TestBudgetEstimation(_MatrixCase):
    def test_estimate_budget_numbers_from_synthetic_history_and_fallback(self) -> None:
        # 2 executors: opus-5.5-high (12 jobs), gemini-3.8-flash-high (12 jobs)
        jobs = SM.expand_jobs(self.suite, k=2, arms="skill")
        self.assertEqual(len(jobs), 24)

        # Synthetic history:
        # - opus-5.5-high has 3 valid rollouts: t_total_s = [60, 120, 240] (median = 120s = 2.0m),
        #   total_tokens = [10_000, 20_000, 50_000] (median = 20_000), plus 1 invalid rollout
        #   (valid=False, ignored).
        # - gemini-3.8-flash-high has NO history -> falls back to 3.0m (180s) per rollout, 0 tokens.
        synthetic_history = [
            {"executor": "opus-5.5-high", "t_total_s": 60.0, "total_tokens": 10_000, "valid": True},
            {"executor": "opus-5.5-high", "t_total_s": 120.0, "total_tokens": 20_000, "valid": True},
            {"executor": "opus-5.5-high", "t_total_s": 240.0, "total_tokens": 50_000, "valid": True},
            {"executor": "opus-5.5-high", "t_total_s": 9999.0, "total_tokens": 999_999, "valid": False},
        ]

        budget = SM.estimate_budget(jobs, synthetic_history, parallel=3)
        self.assertEqual(budget["rollouts"], 24)
        self.assertEqual(budget.rollouts, 24)

        opus_b = budget["executors"]["opus-5.5-high"]
        self.assertEqual(opus_b["rollouts"], 12)
        self.assertAlmostEqual(opus_b["median_wall_s"], 120.0)
        self.assertAlmostEqual(opus_b["median_wall_min"], 2.0)
        # 12 rollouts * 2.0 min / 3 parallel = 8.0 min (480s)
        self.assertAlmostEqual(opus_b["est_minutes"], 8.0)
        self.assertAlmostEqual(opus_b["est_wall_s"], 480.0)
        # 12 rollouts * 20,000 median tokens = 240,000
        self.assertEqual(opus_b["est_tokens"], 240_000)
        self.assertTrue(opus_b["from_history"])

        gem_b = budget["executors"]["gemini-3.8-flash-high"]
        self.assertEqual(gem_b["rollouts"], 12)
        self.assertAlmostEqual(gem_b["median_wall_min"], 3.0)
        # 12 rollouts * 3.0 min / 3 parallel = 12.0 min (720s)
        self.assertAlmostEqual(gem_b["est_minutes"], 12.0)
        self.assertAlmostEqual(gem_b["est_wall_s"], 720.0)
        self.assertEqual(gem_b["est_tokens"], 0)
        self.assertFalse(gem_b["from_history"])

        # Total: 24 rollouts, 8.0 + 12.0 = 20.0 min, 240,000 tokens
        self.assertAlmostEqual(budget["est_minutes"], 20.0)
        self.assertAlmostEqual(budget["est_wall_s"], 1200.0)
        self.assertEqual(budget["est_tokens"], 240_000)

        table = SM.format_budget(budget)
        self.assertIn("opus-5.5-high", table)
        self.assertIn("gemini-3.8-flash-high", table)
        self.assertIn("240,000", table)
        self.assertIn("TOTAL", table)

    def test_estimate_budget_from_history_jsonl_file(self) -> None:
        jobs = SM.expand_jobs(self.suite, executors=["opus-5.5-high"], k=1, arms="skill")
        # 1 executor * 2 arms * 3 tasks * 1 rep = 6 rollouts
        self.assertEqual(len(jobs), 6)
        hist_path = self.tmp / "history.jsonl"
        _write(
            hist_path,
            json.dumps({"executor": "opus-5.5-high", "t_total_s": 90.0, "total_tokens": 15_000})
            + "\n"
            + json.dumps({"executor": "opus-5.5-high", "t_total_s": 150.0, "total_tokens": 25_000})
            + "\n",
        )
        budget = SM.estimate_budget(jobs, hist_path, parallel=2)
        # median_wall_s = 120s (2m), 6 * 2m / 2 parallel = 6.0 min; median_tokens = 20_000 -> 120_000
        self.assertEqual(budget["rollouts"], 6)
        self.assertAlmostEqual(budget["est_minutes"], 6.0)
        self.assertEqual(budget["est_tokens"], 120_000)


class TestMatrixGatesAndExecution(_MatrixCase):
    def test_max_rollouts_abort_before_any_runner_call(self) -> None:
        with (
            mock.patch.object(SM.run_sweep, "main", return_value=0) as spy_sweep,
            mock.patch.object(SM.judge, "main", return_value=0) as spy_judge,
        ):
            out = io.StringIO()
            err = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = SM.main(
                    [
                        "--suite",
                        str(self.suite_toml),
                        "--arms",
                        "skill",
                        "--k",
                        "2",
                        "--max-rollouts",
                        "10",
                        "--yes",
                    ]
                )
            self.assertEqual(rc, 1)
            self.assertIn("--max-rollouts", err.getvalue())
            self.assertEqual(spy_sweep.call_count, 0)
            self.assertEqual(spy_judge.call_count, 0)
            self.assertFalse((self.runs_root / "matrix.json").exists())

    def test_confirmation_gate_aborts_without_yes_or_tty(self) -> None:
        with (
            mock.patch.object(SM.run_sweep, "main", return_value=0) as spy_sweep,
            mock.patch.object(SM.judge, "main", return_value=0) as spy_judge,
        ):
            # Non-TTY stdin without --yes -> aborts
            non_tty = io.StringIO("y\n")
            err = io.StringIO()
            with (
                mock.patch.object(SM.sys, "stdin", non_tty),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(err),
            ):
                rc = SM.main(["--suite", str(self.suite_toml), "--arms", "skill", "--k", "1"])
            self.assertEqual(rc, 1)
            self.assertIn("not a TTY", err.getvalue())
            self.assertEqual(spy_sweep.call_count, 0)

            # TTY stdin with "n" -> aborts
            tty_no = io.StringIO("n\n")
            tty_no.isatty = lambda: True  # type: ignore[attr-defined]
            err2 = io.StringIO()
            with (
                mock.patch.object(SM.sys, "stdin", tty_no),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(err2),
            ):
                rc2 = SM.main(["--suite", str(self.suite_toml), "--arms", "skill", "--k", "1"])
            self.assertEqual(rc2, 1)
            self.assertIn("aborted", err2.getvalue())
            self.assertEqual(spy_sweep.call_count, 0)

            # TTY stdin with "y" -> proceeds
            tty_yes = io.StringIO("y\n")
            tty_yes.isatty = lambda: True  # type: ignore[attr-defined]
            with (
                mock.patch.object(SM.sys, "stdin", tty_yes),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rc3 = SM.main(
                    [
                        "--suite",
                        str(self.suite_toml),
                        "--executors",
                        "opus-5.5-high",
                        "--arms",
                        "skill",
                        "--k",
                        "1",
                    ]
                )
            self.assertEqual(rc3, 0)
            self.assertEqual(spy_sweep.call_count, 2)
            self.assertEqual(spy_judge.call_count, 2)

    def test_resume_skips_finished_cells_and_writes_matrix_json_with_invalid_counts(
        self,
    ) -> None:
        observed_envs: list[tuple[str, str, str]] = []

        def fake_sweep(argv: list[str] | None = None) -> int:
            p = sweep_cli._build_parser().parse_args(argv)
            observed_envs.append(
                (
                    os.environ.get("SKILLBENCH_EXEC_MODEL", ""),
                    os.environ.get("SKILLBENCH_JUDGE_MODEL", ""),
                    str(p.arm),
                )
            )
            return 0

        # Seed one invalid skill_timing.json in opus-5.5-high / with to verify invalid_by_reason
        bad_timing_dir = (
            self.runs_root
            / "results"
            / "opus-5.5-high"
            / "with"
            / "plan-skill__t1"
            / "base1"
        )
        _write(
            bad_timing_dir / "skill_timing.json",
            json.dumps(
                {
                    "valid": False,
                    "invalid_reason": "timeout after 60s",
                    "t_total_s": 60.0,
                    "total_tokens": 5000,
                }
            )
            + "\n",
        )

        # First run: only opus-5.5-high across ("with", "without") -> 2 cells run
        with (
            mock.patch.object(SM.run_sweep, "main", side_effect=fake_sweep) as spy_sweep,
            mock.patch.object(SM.judge, "main", return_value=0) as spy_judge,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc1 = SM.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--executors",
                    "opus-5.5-high",
                    "--arms",
                    "skill",
                    "--k",
                    "2",
                    "--yes",
                ]
            )
        self.assertEqual(rc1, 0)
        self.assertEqual(spy_sweep.call_count, 2)
        self.assertEqual(spy_judge.call_count, 2)
        self.assertEqual(
            observed_envs,
            [
                ("opus-5.5-high", "gemini-3.8-flash-high", "with"),
                ("opus-5.5-high", "gemini-3.8-flash-high", "without"),
            ],
        )

        matrix_path = self.runs_root / "matrix.json"
        self.assertTrue(matrix_path.is_file())
        m1 = json.loads(matrix_path.read_text(encoding="utf-8"))
        self.assertEqual(m1["executors"], ["opus-5.5-high"])
        self.assertEqual(m1["judges"], {"opus-5.5-high": "gemini-3.8-flash-high"})
        self.assertEqual(
            m1["cell_status"],
            {"opus-5.5-high/with": "ok", "opus-5.5-high/without": "ok"},
        )
        self.assertEqual(m1["invalid_by_reason"], {"timeout after 60s": 1})
        self.assertEqual(m1["invalid_total"], 1)

        # Second run with both executors (opus-5.5-high + gemini-3.8-flash-high):
        # opus-5.5-high cells are skipped; ONLY gemini-3.8-flash-high cells run!
        observed_envs.clear()
        with (
            mock.patch.object(SM.run_sweep, "main", side_effect=fake_sweep) as spy_sweep2,
            mock.patch.object(SM.judge, "main", return_value=0) as spy_judge2,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc2 = SM.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--arms",
                    "skill",
                    "--k",
                    "2",
                    "--yes",
                ]
            )
        self.assertEqual(rc2, 0)
        self.assertEqual(spy_sweep2.call_count, 2)
        self.assertEqual(spy_judge2.call_count, 2)
        self.assertEqual(
            observed_envs,
            [
                ("gemini-3.8-flash-high", "fable-5.1-high", "with"),
                ("gemini-3.8-flash-high", "fable-5.1-high", "without"),
            ],
        )
        m2 = json.loads(matrix_path.read_text(encoding="utf-8"))
        self.assertEqual(
            m2["cell_status"],
            {
                "opus-5.5-high/with": "skipped",
                "opus-5.5-high/without": "skipped",
                "gemini-3.8-flash-high/with": "ok",
                "gemini-3.8-flash-high/without": "ok",
            },
        )

        # Third run with both executors: ALL cells finished -> 0 new rollouts scheduled,
        # 0 calls to run_sweep.main or judge.main!
        out3 = io.StringIO()
        with (
            mock.patch.object(SM.run_sweep, "main", side_effect=fake_sweep) as spy_sweep3,
            mock.patch.object(SM.judge, "main", return_value=0) as spy_judge3,
            contextlib.redirect_stdout(out3),
        ):
            rc3 = SM.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--arms",
                    "skill",
                    "--k",
                    "2",
                ]
            )
        self.assertEqual(rc3, 0)
        self.assertEqual(spy_sweep3.call_count, 0)
        self.assertEqual(spy_judge3.call_count, 0)

    def test_identity_mismatch_on_changed_suite_hash_refuses_resume(self) -> None:
        # 1) Test via skillbench-matrix with mocked runner
        with (
            mock.patch.object(SM.run_sweep, "main", return_value=0),
            mock.patch.object(SM.judge, "main", return_value=0),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc1 = SM.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--executors",
                    "opus-5.5-high",
                    "--arms",
                    "skill",
                    "--k",
                    "1",
                    "--yes",
                ]
            )
        self.assertEqual(rc1, 0)

        # Modify a staged plugin file so suite_hash changes
        skill_md = self.plugins_dir / "planning" / "skills" / "plan-skill" / "SKILL.md"
        skill_md.write_text(
            skill_md.read_text(encoding="utf-8") + "\nAdditional rule.\n",
            encoding="utf-8",
        )

        err = io.StringIO()
        with (
            mock.patch.object(SM.run_sweep, "main", return_value=0) as spy_sweep,
            mock.patch.object(SM.judge, "main", return_value=0),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(err),
        ):
            rc2 = SM.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--executors",
                    "opus-5.5-high",
                    "--arms",
                    "skill",
                    "--k",
                    "1",
                    "--yes",
                ]
            )
        self.assertEqual(rc2, 1)
        self.assertIn("different models or tasks", err.getvalue())
        self.assertEqual(spy_sweep.call_count, 0)

    def test_run_sweep_check_identity_includes_suite_hash_and_namespaces_per_executor(
        self,
    ) -> None:
        fake_rep = {
            "domains": [{"domain": "plan-skill", "done": 2, "failed": 0, "tasks": []}],
        }
        with (
            mock.patch("harness2.preflight.require_ready"),
            mock.patch.object(schedule, "run_sweep", return_value=fake_rep),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with agy.executor_env("opus-5.5-high", "gemini-3.8-flash-high", suite=self.suite):
                self.assertEqual(
                    sweep_cli.main(
                        [
                            "--bench",
                            "skillbench",
                            "--arm",
                            "with",
                            "--k",
                            "1",
                            "--only-domains",
                            "plan-skill",
                        ]
                    ),
                    0,
                )
            with agy.executor_env("gemini-3.8-flash-high", "fable-5.1-high", suite=self.suite):
                self.assertEqual(
                    sweep_cli.main(
                        [
                            "--bench",
                            "skillbench",
                            "--arm",
                            "with",
                            "--k",
                            "1",
                            "--only-domains",
                            "plan-skill",
                        ]
                    ),
                    0,
                )

        opus_cfg_path = (
            self.runs_root
            / "adapt"
            / "opus-5.5-high"
            / "skillbench-agy-agg1-with"
            / "config.json"
        )
        gem_cfg_path = (
            self.runs_root
            / "adapt"
            / "gemini-3.8-flash-high"
            / "skillbench-agy-agg1-with"
            / "config.json"
        )
        self.assertTrue(opus_cfg_path.is_file())
        self.assertTrue(gem_cfg_path.is_file())
        opus_cfg = json.loads(opus_cfg_path.read_text(encoding="utf-8"))
        gem_cfg = json.loads(gem_cfg_path.read_text(encoding="utf-8"))
        self.assertEqual(opus_cfg["executor"], "opus-5.5-high")
        self.assertEqual(opus_cfg["suite_hash"], self.suite.suite_hash)
        self.assertEqual(gem_cfg["executor"], "gemini-3.8-flash-high")
        self.assertEqual(gem_cfg["suite_hash"], self.suite.suite_hash)

        # Mutate plugin file -> suite_hash changes -> run_sweep refuses resume
        skill_md = self.plugins_dir / "planning" / "skills" / "plan-skill" / "SKILL.md"
        skill_md.write_text(
            skill_md.read_text(encoding="utf-8") + "\nMutated.\n",
            encoding="utf-8",
        )
        err = io.StringIO()
        with (
            mock.patch("harness2.preflight.require_ready"),
            mock.patch.object(schedule, "run_sweep", return_value=fake_rep),
            agy.executor_env("opus-5.5-high", "gemini-3.8-flash-high", suite=self.suite),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(err),
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
                    "plan-skill",
                ]
            )
        self.assertEqual(rc, 1)
        self.assertIn("different models or tasks", err.getvalue())

    def test_in_process_matrix_with_agy_stub_populates_discoverable_cells(self) -> None:
        # Run 2 executors x 2 skill arms x 1 task x k=1 with agy_stub.py
        with (
            mock.patch.dict(
                os.environ,
                {
                    "AGY_STUB_SCENARIO": "skill_read",
                    "AGY_STUB_WITHOUT_SCENARIO": "plain_answer",
                },
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc = SM.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--arms",
                    "skill",
                    "--k",
                    "1",
                    "--only-domains",
                    "review-skill",
                    "--max-tasks",
                    "1",
                    "--yes",
                ]
            )
        self.assertEqual(rc, 0)
        cells = SR.discover_cells(self.runs_root / "results")
        self.assertEqual(
            [(c.executor, c.judge, c.arm_label) for c in cells],
            [
                ("gemini-3.8-flash-high", "fable-5.1-high", "with"),
                ("gemini-3.8-flash-high", "fable-5.1-high", "without"),
                ("opus-5.5-high", "gemini-3.8-flash-high", "with"),
                ("opus-5.5-high", "gemini-3.8-flash-high", "without"),
            ],
        )

    def test_pyproject_registers_skillbench_matrix_console_script(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = pyproject["project"]["scripts"]
        self.assertEqual(
            scripts.get("skillbench-matrix"),
            "harness2.cli.skillbench_matrix:main",
        )
        self.assertIs(judge_cli.main, SM.judge.main)


if __name__ == "__main__":
    unittest.main()
