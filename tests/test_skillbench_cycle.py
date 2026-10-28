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

"""Tests for Task 11: `skillbench-cycle` runner (`matrix -> report -> history -> improve`)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

from harness2 import config
from harness2.benchmarks import skillbench as SB
from harness2.cli import skillbench_cycle as SC

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


def _make_skill_task(tasks_root: Path, skill: str, tid: str) -> None:
    tdir = tasks_root / skill / tid
    _write(
        tdir / "task.toml",
        f'target_skill = "{skill}"\n'
        f'kind = "skill"\n'
        f"pilot = true\n"
        f"timeout = 60\n"
        f'prompt = "Implement {tid} in app.py."\n',
    )
    _write(
        tdir / "rubric.json",
        json.dumps([{"id": f"c{i}", "criterion": f"Criterion {i} passes."} for i in range(4)]),
    )
    _write(tdir / "fixture" / "app.py", "def run():\n    return 1\n")


def _seed_judged_cell(
    runs_root: Path,
    *,
    executor: str,
    judge: str,
    arm: str,
    suite_name: str,
    suite_hash: str,
    manifest_hash: str,
    skill: str,
    tasks: list[str],
    n_passed: int,
    tokens: int,
    wall_s: float,
) -> Path:
    cell_dir = runs_root / "results" / executor / arm
    _write(
        cell_dir / "config.json",
        json.dumps(
            {
                "executor": executor,
                "judge": judge,
                "effort": "high",
                "manifest_hash": manifest_hash,
                "suite_hash": suite_hash,
                "k": 2,
                "seed": None,
            }
        )
        + "\n",
    )
    _write(
        cell_dir / "run.json",
        json.dumps(
            {
                "suite": suite_name,
                "suite_hash": suite_hash,
                "manifest_hash": manifest_hash,
                "arm": arm,
                "executor": executor,
                "judge": judge,
                "effort": "high",
            }
        )
        + "\n",
    )
    judged_rows = []
    for tid in tasks:
        full_tid = f"{skill}/{tid}"
        slug = f"{skill}__{tid}"
        base_entries = []
        for rep in (1, 2):
            lbl = f"base{rep}"
            rid = f"{arm}-{tid}-{rep}"
            rdir = cell_dir / slug / lbl
            _write(
                rdir / "skill_timing.json",
                json.dumps(
                    {
                        "valid": True,
                        "invoked": arm == "with",
                        "t_skill_s": 2.5 if arm == "with" else None,
                        "requested_model": executor,
                        "actual_model": executor,
                        "total_tokens": tokens,
                        "t_total_s": wall_s,
                        "skills_read": [skill] if arm == "with" else [],
                    }
                )
                + "\n",
            )
            _write(rdir / "verify.json", json.dumps({"passed": True}) + "\n")
            _write(rdir / "final.md", f"Completed {full_tid} ({arm} {lbl}).\n")
            criteria = [
                {"id": f"c{i}", "pass": i < n_passed, "reason": "ok"} for i in range(4)
            ]
            _write(
                rdir / "judgments.json",
                json.dumps(
                    {
                        "judge_model": judge,
                        "actual_model": judge,
                        "judgments": [{"criteria": criteria}],
                    }
                )
                + "\n",
            )
            base_entries.append(
                {"label": lbl, "run_id": rid, "run_dir": str(rdir), "ok": True}
            )
            judged_rows.append(
                {
                    "task": full_tid,
                    "label": lbl,
                    "run_id": rid,
                    "score": {"n_passed": n_passed, "n_criteria": 4},
                }
            )
        _write(
            cell_dir / slug / "summary.json",
            json.dumps(
                {
                    "task_id": full_tid,
                    "domain": skill,
                    "status": "bases-only",
                    "base": base_entries,
                }
            )
            + "\n",
        )
    _write(cell_dir / "base_judged.json", json.dumps(judged_rows, indent=2) + "\n")
    return cell_dir


class _CycleCase(unittest.TestCase):
    def setUp(self) -> None:
        self._env_snapshot = os.environ.copy()
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-cycle-test-"))
        self.runs_dir = self.tmp / "runs"
        self.runs_root = self.runs_dir / "skillbench"
        self.plugins_dir = self.tmp / "plugins"
        _make_plugin(self.plugins_dir, "planning", ["plan-skill"])

        self.tasks_root = self.tmp / "tasks"
        _make_skill_task(self.tasks_root, "plan-skill", "t1")
        _make_skill_task(self.tasks_root, "plan-skill", "t2")
        SB.freeze(self.tasks_root)
        self.manifest_hash = SB.check_manifest(self.tasks_root)

        self.suite_toml = _write(
            self.tmp / "suite.toml",
            f'name = "alpha"\n'
            f'plugins_dir = "{self.plugins_dir}"\n'
            f'plugins = ["planning"]\n'
            f'tasks_root = "{self.tasks_root}"\n'
            f"[matrix]\n"
            f'executors = ["opus-5.5-high"]\n'
            f"[matrix.judge_for]\n"
            f'"opus-" = "gemini-3.8-flash-high"\n',
        )

        os.environ.update(
            HARNESS2_RUNS_DIR=str(self.runs_dir),
            HARNESS2_AGY_BIN=str(STUB),
            SKILLBENCH_SUITE=str(self.suite_toml),
        )
        p = mock.patch.object(config, "RUNS", self.runs_dir)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_snapshot)
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestStageOrderingAndFlags(_CycleCase):
    def test_improve_off_by_default_runs_matrix_report_history_in_order(self) -> None:
        calls: list[str] = []

        def fake_stage(name: str):
            def _run(args, runs_root):
                calls.append(name)
                return 0

            return _run

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = SC.main(
                ["--suite", str(self.suite_toml), "--runs-root", str(self.runs_root), "--yes"],
                stage_runners={
                    "matrix": fake_stage("matrix"),
                    "report": fake_stage("report"),
                    "history": fake_stage("history"),
                    "improve": fake_stage("improve"),
                },
            )
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["matrix", "report", "history"])

        parsed = SC.parse_cycle_log(self.runs_root / "cycle.log")
        self.assertEqual(
            [s["stage"] for s in parsed["stages"]],
            ["matrix", "report", "history", "improve"],
        )
        self.assertEqual(parsed["by_stage"]["matrix"]["status"], "ok")
        self.assertEqual(parsed["by_stage"]["report"]["status"], "ok")
        self.assertEqual(parsed["by_stage"]["history"]["status"], "ok")
        self.assertEqual(parsed["by_stage"]["improve"]["status"], "skipped")
        self.assertEqual(parsed["total"]["status"], "ok")

        stdout_txt = out.getvalue()
        self.assertIn("# skillbench summary", stdout_txt)
        self.assertIn(str(self.runs_root / "report.html"), stdout_txt)
        self.assertIn(str(self.runs_root / "summary.md"), stdout_txt)
        self.assertIn(str(self.runs_root / "improve" / "improve_report.html"), stdout_txt)

    def test_improve_flag_enables_improve_stage_fourth(self) -> None:
        calls: list[str] = []

        with (
            mock.patch.object(
                SC.skillbench_matrix,
                "main",
                side_effect=lambda argv: (calls.append("matrix"), 0)[1],
            ),
            mock.patch.object(
                SC.skillbench_report,
                "main",
                side_effect=lambda argv: (
                    calls.append("history" if argv and argv[0] == "history" else "report"),
                    0,
                )[1],
            ),
            mock.patch.object(
                SC.skillbench_improve,
                "main",
                side_effect=lambda argv: (calls.append("improve"), 0)[1],
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc = SC.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--runs-root",
                    str(self.runs_root),
                    "--executors",
                    "opus-5.5-high",
                    "--k",
                    "2",
                    "--arms",
                    "skill",
                    "--improve",
                    "--yes",
                ]
            )
        self.assertEqual(rc, 0)
        self.assertEqual(calls, ["matrix", "report", "history", "improve"])
        parsed = SC.parse_cycle_log(self.runs_root / "cycle.log")
        for st_name in ("matrix", "report", "history", "improve"):
            self.assertEqual(parsed["by_stage"][st_name]["status"], "ok")

    def test_from_flag_skips_prior_stages_and_resumes_at_target_stage(self) -> None:
        calls: list[str] = []
        runners = {
            name: (lambda n=name: (calls.append(n), 0)[1])
            for name in ("matrix", "report", "history", "improve")
        }

        # 1) --from report (without --improve) -> runs report -> history
        with contextlib.redirect_stdout(io.StringIO()):
            rc1 = SC.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--runs-root",
                    str(self.runs_root),
                    "--from",
                    "report",
                ],
                stage_runners=runners,
            )
        self.assertEqual(rc1, 0)
        self.assertEqual(calls, ["report", "history"])
        log1 = SC.parse_cycle_log(self.runs_root / "cycle.log")
        self.assertEqual(log1["by_stage"]["matrix"]["status"], "skipped")
        self.assertEqual(log1["by_stage"]["report"]["status"], "ok")
        self.assertEqual(log1["by_stage"]["history"]["status"], "ok")
        self.assertEqual(log1["by_stage"]["improve"]["status"], "skipped")

        # 2) --from "history append" --improve -> runs history -> improve
        calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            rc2 = SC.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--runs-root",
                    str(self.runs_root),
                    "--from",
                    "history append",
                    "--improve",
                ],
                stage_runners=runners,
            )
        self.assertEqual(rc2, 0)
        self.assertEqual(calls, ["history", "improve"])

        # 3) --from improve (implicitly enables improve even without --improve)
        calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            rc3 = SC.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--runs-root",
                    str(self.runs_root),
                    "--from",
                    "improve",
                ],
                stage_runners=runners,
            )
        self.assertEqual(rc3, 0)
        self.assertEqual(calls, ["improve"])

        # 4) --skip report -> runs matrix -> history
        calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            rc4 = SC.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--runs-root",
                    str(self.runs_root),
                    "--skip",
                    "report",
                ],
                stage_runners=runners,
            )
        self.assertEqual(rc4, 0)
        self.assertEqual(calls, ["matrix", "history"])

        # 5) Invalid --from value -> returns 1 and writes error to stderr
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc_bad = SC.main(
                ["--suite", str(self.suite_toml), "--from", "nonexistent-stage"]
            )
        self.assertEqual(rc_bad, 1)
        self.assertIn("unknown stage", err.getvalue())


class TestFailureHandlingAndCycleLog(_CycleCase):
    def test_failing_stage_stops_cycle_and_reports(self) -> None:
        calls: list[str] = []

        def ok_matrix():
            calls.append("matrix")
            return 0

        def fail_report():
            calls.append("report")
            return 2

        def never_called(name: str):
            def _fn():
                calls.append(name)
                return 0

            return _fn

        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = SC.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--runs-root",
                    str(self.runs_root),
                    "--improve",
                    "--yes",
                ],
                stage_runners={
                    "matrix": ok_matrix,
                    "report": fail_report,
                    "history": never_called("history"),
                    "improve": never_called("improve"),
                },
            )
        self.assertEqual(rc, 2)
        self.assertEqual(calls, ["matrix", "report"])
        self.assertIn("stage 'report' failed", out.getvalue())
        self.assertIn("stage 'report' failed", err.getvalue())

        parsed = SC.parse_cycle_log(self.runs_root / "cycle.log")
        self.assertEqual([s["stage"] for s in parsed["stages"]], ["matrix", "report"])
        self.assertEqual(parsed["by_stage"]["matrix"]["status"], "ok")
        self.assertEqual(parsed["by_stage"]["report"]["status"], "failed")
        self.assertEqual(parsed["by_stage"]["report"]["rc"], 2)
        self.assertEqual(parsed["total"]["status"], "failed")
        self.assertEqual(parsed["total"]["failed_stage"], "report")

    def test_exception_in_stage_stops_cycle_and_logs_failure(self) -> None:
        calls: list[str] = []

        def boom_matrix():
            calls.append("matrix")
            raise RuntimeError("quota exhausted during matrix")

        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = SC.main(
                ["--suite", str(self.suite_toml), "--runs-root", str(self.runs_root)],
                stage_runners={
                    "matrix": boom_matrix,
                    "report": lambda: (calls.append("report"), 0)[1],
                    "history": lambda: (calls.append("history"), 0)[1],
                    "improve": lambda: (calls.append("improve"), 0)[1],
                },
            )
        self.assertEqual(rc, 1)
        self.assertEqual(calls, ["matrix"])
        self.assertIn("quota exhausted during matrix", err.getvalue())
        parsed = SC.parse_cycle_log(self.runs_root / "cycle.log")
        self.assertEqual(parsed["by_stage"]["matrix"]["status"], "failed")
        self.assertEqual(parsed["total"]["failed_stage"], "matrix")

    def test_cycle_log_tracks_wall_time_and_token_totals_with_real_report_and_history(
        self,
    ) -> None:
        # Seed a judged matrix (2 tasks x 2 reps x 2 arms = 8 rollouts, 5,000 tokens each = 40,000)
        _seed_judged_cell(
            self.runs_root,
            executor="opus-5.5-high",
            judge="gemini-3.8-flash-high",
            arm="with",
            suite_name="alpha",
            suite_hash="hash12345678",
            manifest_hash=self.manifest_hash,
            skill="plan-skill",
            tasks=["t1", "t2"],
            n_passed=4,
            tokens=5000,
            wall_s=10.0,
        )
        _seed_judged_cell(
            self.runs_root,
            executor="opus-5.5-high",
            judge="gemini-3.8-flash-high",
            arm="without",
            suite_name="alpha",
            suite_hash="hash12345678",
            manifest_hash=self.manifest_hash,
            skill="plan-skill",
            tasks=["t1", "t2"],
            n_passed=2,
            tokens=5000,
            wall_s=8.0,
        )

        def fake_improve(args, runs_root: Path) -> int:
            imp_timing = (
                runs_root / "improve" / "plan-skill" / "round_1" / "results" / "t1" / "base1"
            )
            _write(
                imp_timing / "skill_timing.json",
                json.dumps({"valid": True, "total_tokens": 7500, "t_total_s": 6.0}) + "\n",
            )
            _write(
                runs_root / "improve" / "improve_report.html",
                "<html><body>improve</body></html>\n",
            )
            return 0

        out = io.StringIO()
        with (
            mock.patch.object(SC.skillbench_matrix, "main", return_value=0),
            mock.patch.object(SC, "stage_improve", side_effect=fake_improve),
            contextlib.redirect_stdout(out),
        ):
            rc = SC.main(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--runs-root",
                    str(self.runs_root),
                    "--k",
                    "2",
                    "--pilot",
                    "--improve",
                    "--yes",
                ]
            )
        self.assertEqual(rc, 0)

        # Verify report.html, summary.md, history.jsonl, improve_report.html, cycle.log exist
        self.assertTrue((self.runs_root / "report.html").is_file())
        self.assertTrue((self.runs_root / "summary.md").is_file())
        self.assertTrue((self.runs_root / "improve" / "improve_report.html").is_file())
        self.assertTrue((self.tmp / "history.jsonl").is_file())
        self.assertTrue((self.runs_root / "cycle.log").is_file())

        stdout_txt = out.getvalue()
        self.assertIn("# skillbench summary", stdout_txt)
        self.assertIn("plan-skill", stdout_txt)
        self.assertIn("`keep`", stdout_txt)
        self.assertIn(str(self.runs_root / "report.html"), stdout_txt)
        self.assertIn(str(self.runs_root / "summary.md"), stdout_txt)
        self.assertIn(str(self.runs_root / "improve" / "improve_report.html"), stdout_txt)

        parsed = SC.parse_cycle_log(self.runs_root / "cycle.log")
        self.assertEqual(parsed["by_stage"]["matrix"]["tokens"], 40_000)
        self.assertEqual(parsed["by_stage"]["report"]["tokens"], 0)
        self.assertEqual(parsed["by_stage"]["history"]["tokens"], 0)
        self.assertEqual(parsed["by_stage"]["improve"]["tokens"], 7_500)
        self.assertEqual(parsed["total"]["tokens"], 47_500)
        self.assertGreaterEqual(parsed["total"]["wall_s"], 0.0)

    def test_pyproject_registers_skillbench_cycle_and_helpers(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        scripts = pyproject["project"]["scripts"]
        self.assertEqual(
            scripts.get("skillbench-cycle"),
            "harness2.cli.skillbench_cycle:main",
        )
        self.assertEqual(SC.normalize_stage("history append"), "history")
        self.assertEqual(SC.normalize_stage("history-append"), "history")
        self.assertEqual(SC.normalize_stage(None), "matrix")

    def test_stage_improve_argv_and_baseline_history_noop(self) -> None:
        captured: list[list[str]] = []
        with mock.patch.object(
            SC.skillbench_improve,
            "main",
            side_effect=lambda argv: (captured.append(list(argv)), 0)[1],
        ):
            args = SC._build_parser().parse_args(
                [
                    "--suite",
                    str(self.suite_toml),
                    "--runs-root",
                    str(self.runs_root),
                    "--executors",
                    "opus-5.5-high",
                    "--k",
                    "2",
                    "--rounds",
                    "3",
                    "--skill",
                    "plan-skill",
                    "--all-weak",
                    "--apply",
                ]
            )
            rc = SC.stage_improve(args, self.runs_root)
        self.assertEqual(rc, 0)
        self.assertEqual(len(captured), 1)
        self.assertIn("--skill", captured[0])
        self.assertIn("plan-skill", captured[0])
        self.assertIn("--rounds", captured[0])
        self.assertIn("3", captured[0])
        self.assertIn("--apply", captured[0])

        # Baseline-only arms without report.json is a clean no-op in stage_history
        base_args = SC._build_parser().parse_args(
            ["--suite", str(self.suite_toml), "--arms", "baseline"]
        )
        self.assertEqual(SC.stage_history(base_args, self.runs_root), 0)

        # Invalid --k returns exit code 1
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(SC.main(["--k", "0"]), 1)
        self.assertIn("--k and --parallel must be >= 1", err.getvalue())


if __name__ == "__main__":
    unittest.main()
