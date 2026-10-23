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

"""End-to-end stub tests for `scripts/run_skillbench.sh` and `--pilot` CLI wiring (Task 14).

Every test runs against `tests/fixtures/agy_stub.py`, temporary directories, and an isolated
fake `HOME`; the user's real `$HOME` and `~/.gemini` are never touched.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness2.benchmarks import skillbench as SB
from harness2.cli import _common as C
from harness2.cli import check as check_cli
from harness2.cli import run_sweep as sweep_cli
from harness2.substrate.agy import PLUGINS_REL
from tests.test_skillbench_tasks import make_task

ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = ROOT / "scripts" / "run_skillbench.sh"
STUB = ROOT / "tests" / "fixtures" / "agy_stub.py"
VENV_BIN = ROOT / ".venv" / "bin"
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


def _table_rows(md: str) -> list[list[str]]:
    """Return parsed cells of every data row in the markdown report table."""
    rows: list[list[str]] = []
    in_table = False
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("| skill |"):
            in_table = True
            continue
        if in_table:
            if not s.startswith("|"):
                break
            if set(s.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            rows.append(cells)
    return rows


class _E2ECase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-e2e-test-"))
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

        self.runs_dir = self.tmp / "runs"
        self.stub_log = self.tmp / "stub.log"
        self.home_before = _tree_hash(self.home)

    def tearDown(self) -> None:
        self.assertEqual(_tree_hash(self.home), self.home_before, "fake HOME was modified")

    def base_env(self, **overrides: str) -> dict[str, str]:
        env = dict(os.environ)
        for key in (
            "HARNESS2_SOLVER_MODEL",
            "HARNESS2_IMPROVER_MODEL",
            "HARNESS2_JUDGE_MODEL",
            "HARNESS2_CODEX_MODEL",
            "GOOGLE_CLOUD_PROJECT",
            "HARNESS2_VERTEX_PROJECT",
            "SKILLBENCH_PLUGINS",
            "SKILLBENCH_PILOT",
            "SKILLBENCH_JUDGE_REPEAT",
            "AGY_STUB_SCENARIO",
            "AGY_STUB_JUDGE_SCENARIO",
            "AGY_STUB_WITH_SCENARIO",
            "AGY_STUB_WITHOUT_SCENARIO",
            "AGY_STUB_SLEEP",
            "AGY_STUB_HANG",
            "AGY_STUB_HANG_S",
            "AGY_STUB_EXIT_CODE",
            "K",
            "SEED",
            "PILOT",
            "SUBSTRATE",
            "MODE",
        ):
            env.pop(key, None)
        path = env.get("PATH") or os.defpath
        env.update(
            HOME=str(self.home),
            PATH=f"{VENV_BIN}{os.pathsep}{path}",
            HARNESS2_RUNS_DIR=str(self.runs_dir),
            HARNESS2_AGY_BIN=str(STUB),
            SKILLBENCH_PLUGINS_DIR=str(self.plugins),
            SKILLBENCH_AUTH_FILES=",".join(AUTH),
            SKILLBENCH_EXEC_MODEL="opus",
            SKILLBENCH_JUDGE_MODEL="fable",
            SKILLBENCH_EFFORT="high",
            AGY_STUB_LOG=str(self.stub_log),
            AGY_STUB_SCENARIO="skill_read",
            AGY_STUB_WITHOUT_SCENARIO="plain_answer",
        )
        env.update(overrides)
        return env

    def run_script(self, *args: str, **env_overrides: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(RUN_SCRIPT), *args],
            cwd=ROOT,
            env=self.base_env(**env_overrides),
            capture_output=True,
            text=True,
            timeout=120,
        )

    def stub_calls(self) -> list[dict]:
        if not self.stub_log.is_file():
            return []
        return [json.loads(x) for x in self.stub_log.read_text(encoding="utf-8").splitlines()]


class TestPilotCliWiring(_E2ECase):
    def test_make_adapter_and_run_sweep_wire_pilot_flag(self) -> None:
        tasks_root = self.tmp / "custom_tasks"
        make_task(
            tasks_root,
            "make-plan",
            "p1",
            pilot=True,
            prompt="Write an implementation plan for caching fetch() in utils.py.",
            user_answers=["Use an in-memory LRU cache", "No external dependencies"],
        )
        make_task(
            tasks_root,
            "code-review-refactor",
            "t1",
            pilot=False,
            prompt="Refactor duplicate validation logic in utils.py.",
        )
        SB.freeze(tasks_root)

        with (
            mock.patch.dict(os.environ, self.base_env()),
            mock.patch.object(SB, "TASKS_ROOT", tasks_root),
        ):
            # pilot_only alias and pilot kwarg on SkillbenchAdapter
            a_alias = SB.SkillbenchAdapter("with", pilot_only=True)
            self.assertEqual(a_alias.tasks(), ["make-plan/p1"])

            # make_adapter(..., pilot=True) filters to pilot tasks
            a_pilot = C.make_adapter("skillbench", None, "agy", arm="with", pilot=True)
            self.assertEqual(a_pilot.tasks(), ["make-plan/p1"])

            a_all = C.make_adapter("skillbench", None, "agy", arm="with")
            self.assertEqual(a_all.tasks(), ["code-review-refactor/t1", "make-plan/p1"])

            # --pilot rejected on other benches
            with self.assertRaisesRegex(ValueError, "--pilot"):
                C.make_adapter("jb", None, "oc", pilot=True)

            # run_sweep --pilot restricts stream to pilot tasks only
            buf_pilot = io.StringIO()
            with contextlib.redirect_stdout(buf_pilot):
                rc = sweep_cli.main(
                    ["--bench", "skillbench", "--arm", "with", "--pilot", "--dry-run"]
                )
            self.assertEqual(rc, 0)
            self.assertIn("make-plan/p1", buf_pilot.getvalue())
            self.assertNotIn("code-review-refactor/t1", buf_pilot.getvalue())

            # harness2-check --pilot counts only pilot tasks
            buf_check = io.StringIO()
            with contextlib.redirect_stdout(buf_check):
                rc_chk = check_cli.main(["--bench", "skillbench", "--pilot"])
            self.assertEqual(rc_chk, 0)
            self.assertIn("1 tasks", buf_check.getvalue())

            # run_sweep --pilot on non-skillbench fails cleanly
            err_sweep = io.StringIO()
            with contextlib.redirect_stderr(err_sweep):
                rc_bad = sweep_cli.main(["--bench", "jb", "--pilot", "--dry-run"])
            self.assertEqual(rc_bad, 1)
            self.assertIn("--pilot", err_sweep.getvalue())


class TestRunSkillbenchEndToEnd(_E2ECase):
    def test_full_chain_one_skill_one_task_k1_produces_report_and_separate_namespaces(
        self,
    ) -> None:
        self.assertTrue(RUN_SCRIPT.is_file(), f"missing {RUN_SCRIPT}")
        proc = self.run_script("--only-domains", "make-plan", "--max-tasks", "1", K="1")
        self.assertEqual(
            proc.returncode,
            0,
            f"run_skillbench.sh failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )

        adapt_dir = self.runs_dir / "skillbench" / "adapt"
        with_ns = adapt_dir / "skillbench-agy-agg1-with"
        without_ns = adapt_dir / "skillbench-agy-agg1-without"
        self.assertTrue(with_ns.is_dir(), f"missing {with_ns}")
        self.assertTrue(without_ns.is_dir(), f"missing {without_ns}")
        self.assertFalse((adapt_dir / "skillbench-agy-agg1").exists())

        # Both namespaces have config.json, 1 task summary, and base_judged.json
        with_summaries = list(with_ns.glob("*/summary.json"))
        without_summaries = list(without_ns.glob("*/summary.json"))
        self.assertEqual(len(with_summaries), 1)
        self.assertEqual(len(without_summaries), 1)

        with_summary = json.loads(with_summaries[0].read_text(encoding="utf-8"))
        without_summary = json.loads(without_summaries[0].read_text(encoding="utf-8"))
        self.assertEqual(with_summary["task_id"], "make-plan/token-bucket-limiter")
        self.assertEqual(without_summary["task_id"], "make-plan/token-bucket-limiter")
        self.assertEqual(with_summary["status"], "bases-only")
        self.assertEqual(without_summary["status"], "bases-only")

        with_run_dir = Path(with_summary["base"][0]["run_dir"])
        without_run_dir = Path(without_summary["base"][0]["run_dir"])
        self.assertNotEqual(with_run_dir, without_run_dir)

        # Leave-one-out staging verified in the rollout homes
        self.assertTrue(
            (with_run_dir / "home" / PLUGINS_REL / "planning" / "skills" / "make-plan" / "SKILL.md").is_file()
        )
        self.assertFalse(
            (without_run_dir / "home" / PLUGINS_REL / "planning" / "skills" / "make-plan").exists()
        )
        self.assertTrue(
            (without_run_dir / "home" / PLUGINS_REL / "review" / "skills" / "code-review-refactor" / "SKILL.md").is_file()
        )

        # Both base_judged.json files exist and have 1 graded row
        with_judged = json.loads((with_ns / "base_judged.json").read_text(encoding="utf-8"))
        without_judged = json.loads((without_ns / "base_judged.json").read_text(encoding="utf-8"))
        self.assertEqual(len(with_judged), 1)
        self.assertEqual(len(without_judged), 1)
        self.assertEqual(with_judged[0]["score"], {"n_passed": 6, "n_criteria": 6})
        self.assertEqual(without_judged[0]["score"], {"n_passed": 3, "n_criteria": 6})

        # Report files written to skillbench root and adapt dir
        report_md_path = self.runs_dir / "skillbench" / "report.md"
        report_json_path = self.runs_dir / "skillbench" / "report.json"
        self.assertTrue(report_md_path.is_file())
        self.assertTrue(report_json_path.is_file())

        report = json.loads(report_json_path.read_text(encoding="utf-8"))
        self.assertEqual(report["k"], 1)
        self.assertFalse(report["pilot"])
        self.assertEqual(len(report["skills"]), 1)
        skill_row = report["skills"][0]
        self.assertEqual(skill_row["skill"], "make-plan")
        self.assertEqual(skill_row["n_tasks"], 1)
        self.assertAlmostEqual(skill_row["with_mean"], 1.0)
        self.assertAlmostEqual(skill_row["without_mean"], 0.5)
        self.assertAlmostEqual(skill_row["lift"], 0.5)
        self.assertIsNotNone(skill_row["ci_low"])
        self.assertIsNotNone(skill_row["ci_high"])
        self.assertEqual(skill_row["verdict"], "insufficient data")
        self.assertEqual(skill_row["invoked"], 1)
        self.assertEqual(skill_row["valid_with_reps"], 1)
        self.assertAlmostEqual(skill_row["invocation_rate"], 1.0)
        self.assertIsNotNone(skill_row["t_skill_p50"])
        self.assertIsNotNone(skill_row["t_skill_p90"])
        self.assertEqual(skill_row["invalid_total"], 0)

        md = report_md_path.read_text(encoding="utf-8")
        rows = _table_rows(md)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "make-plan")
        self.assertEqual(rows[0][1], "1")
        self.assertEqual(rows[0][2], "1.00")
        self.assertEqual(rows[0][3], "0.50")
        self.assertIn("+0.50", rows[0][4])
        self.assertEqual(rows[0][5], "insufficient data")
        self.assertEqual(rows[0][6], "1/1 (100%)")

        # 2 executor calls (WITH + WITHOUT) + 2 judge calls (WITH + WITHOUT) = 4 stub calls
        calls = self.stub_calls()
        self.assertEqual(len(calls), 4)
        exec_calls = [c for c in calls if "json_schema" not in c["flags"]]
        judge_calls = [c for c in calls if "json_schema" in c["flags"]]
        self.assertEqual(len(exec_calls), 2)
        self.assertEqual(len(judge_calls), 2)

    def test_pilot_and_seed_and_k_settings_honoured(self) -> None:
        proc = self.run_script(
            "--only-domains",
            "make-plan",
            "--max-tasks",
            "1",
            K="1",
            SEED="7",
            PILOT="1",
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"run_skillbench.sh PILOT=1 failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )

        adapt_dir = self.runs_dir / "skillbench" / "adapt"
        with_ns = adapt_dir / "skillbench-agy-agg1-s7-with"
        without_ns = adapt_dir / "skillbench-agy-agg1-s7-without"
        self.assertTrue(with_ns.is_dir())
        self.assertTrue(without_ns.is_dir())

        # PILOT=1 sets SKILLBENCH_JUDGE_REPEAT=2 -> 2 judgments stored per rollout
        with_summary = json.loads(
            next(with_ns.glob("*/summary.json")).read_text(encoding="utf-8")
        )
        with_run_dir = Path(with_summary["base"][0]["run_dir"])
        judgments_data = json.loads((with_run_dir / "judgments.json").read_text(encoding="utf-8"))
        self.assertEqual(len(judgments_data["judgments"]), 2)

        report = json.loads(
            (self.runs_dir / "skillbench" / "report.json").read_text(encoding="utf-8")
        )
        self.assertTrue(report["pilot"])
        self.assertEqual(report["seed"], 7)
        self.assertEqual(report["bootstrap_seed"], 7)
        self.assertEqual(report["k"], 1)
        self.assertIsNotNone(report["judge_noise_rate"])
        self.assertEqual(report["judge_noise_total"], 12)

        # 2 executor calls + 4 judge calls (2 arms * repeat=2) = 6 stub calls
        calls = self.stub_calls()
        self.assertEqual(len(calls), 6)

    def test_dry_run_stops_before_judging_and_reporting(self) -> None:
        proc = self.run_script(
            "--only-domains",
            "make-plan",
            "--max-tasks",
            "1",
            "--dry-run",
            K="2",
            SEED="11",
            PILOT="1",
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("skillbench-agy-agg2-s11-with", proc.stdout)
        self.assertIn("skillbench-agy-agg2-s11-without", proc.stdout)
        self.assertFalse((self.runs_dir / "skillbench" / "report.md").exists())
        self.assertEqual(self.stub_calls(), [])

    def test_preflight_failure_stops_script_before_rollouts(self) -> None:
        proc = self.run_script(
            "--only-domains",
            "make-plan",
            "--max-tasks",
            "1",
            SKILLBENCH_EFFORT="",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("SKILLBENCH_EFFORT", proc.stdout + proc.stderr)
        self.assertEqual(self.stub_calls(), [])
        self.assertFalse((self.runs_dir / "skillbench" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
