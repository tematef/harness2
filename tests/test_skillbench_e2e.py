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
    _plugin(plugin_dir)
    _write(plugin_dir / "skills" / name / "SKILL.md", f"---\nname: {name}\n---\nDo {name}.\n")


def _plugin(plugin_dir: Path) -> None:
    """A plugin.json makes the directory a plugin the env-fallback suite discovers."""
    if not (plugin_dir / "plugin.json").exists():
        _write(plugin_dir / "plugin.json", f'{{"name": "fake-{plugin_dir.name}"}}\n')


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
        _plugin(self.plugins / "skill-eval")
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
            # pilot kwarg on SkillbenchAdapter
            a_direct = SB.SkillbenchAdapter("with", pilot=True)
            self.assertEqual(a_direct.tasks(), ["make-plan/p1"])

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

        # Isolation: fake HOME and plugins dir are byte-identical before/after, and nothing was
        # written outside self.runs_dir (except self.stub_log).
        self.assertEqual(_tree_hash(self.home), self.home_before)
        for p in self.tmp.rglob("*"):
            if not p.is_file() or p == self.stub_log:
                continue
            self.assertTrue(
                p.is_relative_to(self.home)
                or p.is_relative_to(self.plugins)
                or p.is_relative_to(self.runs_dir),
                f"unexpected file written outside run root: {p}",
            )
        for call in calls:
            self.assertTrue(
                Path(call["home"]).resolve().is_relative_to(self.runs_dir.resolve()),
                f"stub HOME outside runs_dir: {call['home']}",
            )
            self.assertTrue(
                Path(call["cwd"]).resolve().is_relative_to(self.runs_dir.resolve()),
                f"stub cwd outside runs_dir: {call['cwd']}",
            )

    def test_multi_skill_stub_chain_produces_lift_ci_verdict_and_time_to_skill_per_skill(
        self,
    ) -> None:
        # Custom WITH scenario that reads both target skills and writes a deliverable file.
        multi_read = self.tmp / "multi_skill_read.jsonl"
        events = [
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "state": "ACTIVE",
                    "tool_name": "view_file",
                    "tool_info": {
                        "name": "view_file",
                        "parameters": {
                            "AbsolutePath": (
                                "/tmp/skillbench-run/home/.gemini/config/plugins/"
                                "planning/skills/make-plan/SKILL.md"
                            )
                        },
                    },
                },
            },
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "state": "ACTIVE",
                    "tool_name": "view_file",
                    "tool_info": {
                        "name": "view_file",
                        "parameters": {
                            "AbsolutePath": (
                                "/tmp/skillbench-run/home/.gemini/config/plugins/"
                                "review/skills/code-review-refactor/SKILL.md"
                            )
                        },
                    },
                },
            },
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "state": "ACTIVE",
                    "tool_name": "write_to_file",
                    "tool_info": {
                        "name": "write_to_file",
                        "parameters": {
                            "TargetFile": "/tmp/skillbench-run/work/DELIVERABLE.md",
                            "CodeContent": "# Deliverable\nDone.\n",
                        },
                    },
                },
            },
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "state": "DONE",
                    "text_delta": "Completed task.",
                },
            },
            {"event": "result", "result": {"usage": {"total_tokens": 1024}}},
        ]
        _write(multi_read, "".join(json.dumps(e) + "\n" for e in events))

        plugins_before = _tree_hash(self.plugins)
        proc = self.run_script(
            "--only-domains",
            "make-plan,code-review-refactor",
            K="2",
            SEED="7",
            PILOT="1",
            SKILLBENCH_JUDGE_REPEAT="1",
            AGY_STUB_WITH_SCENARIO=str(multi_read),
            AGY_STUB_WITHOUT_SCENARIO="plain_answer",
            AGY_STUB_JUDGE_SCENARIO="auto",
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"multi-skill run_skillbench.sh failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        self.assertEqual(_tree_hash(self.home), self.home_before)
        self.assertEqual(_tree_hash(self.plugins), plugins_before)

        report = json.loads(
            (self.runs_dir / "skillbench" / "report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(sorted(report["by_skill"]), ["code-review-refactor", "make-plan"])
        expected_by_skill = {
            "code-review-refactor": (0.6, 0.4, "0.60", "+0.40 [+0.40, +0.40]"),
            "make-plan": (0.5, 0.5, "0.50", "+0.50 [+0.50, +0.50]"),
        }
        for skill_name, (exp_without, exp_lift, _, _) in expected_by_skill.items():
            row = report["by_skill"][skill_name]
            self.assertEqual(row["n_tasks"], 2)
            self.assertEqual(row["n_qualifying_tasks"], 2)
            self.assertAlmostEqual(row["with_mean"], 1.0)
            self.assertAlmostEqual(row["without_mean"], exp_without)
            self.assertAlmostEqual(row["lift"], exp_lift)
            self.assertAlmostEqual(row["ci_low"], exp_lift)
            self.assertAlmostEqual(row["ci_high"], exp_lift)
            self.assertEqual(row["verdict"], "helps")
            self.assertEqual(row["invoked"], 4)
            self.assertEqual(row["valid_with_reps"], 4)
            self.assertAlmostEqual(row["invocation_rate"], 1.0)
            self.assertIsNotNone(row["t_skill_p50"])
            self.assertIsNotNone(row["t_skill_p90"])
            self.assertEqual(row["invalid_total"], 0)

        md = (self.runs_dir / "skillbench" / "report.md").read_text(encoding="utf-8")
        self.assertIn("model unverified (agy reports no model)", md)
        table = {r[0]: r for r in _table_rows(md)}
        self.assertEqual(sorted(table), ["code-review-refactor", "make-plan"])
        for skill_name, (_, _, exp_without_str, exp_lift_str) in expected_by_skill.items():
            cells = table[skill_name]
            self.assertEqual(cells[1], "2", skill_name)
            self.assertEqual(cells[2], "1.00", skill_name)
            self.assertEqual(cells[3], exp_without_str, skill_name)
            self.assertEqual(cells[4], exp_lift_str, skill_name)
            self.assertEqual(cells[5], "helps", skill_name)
            self.assertEqual(cells[6], "4/4 (100%)", skill_name)
            self.assertNotEqual(cells[7], "—", skill_name)
            self.assertEqual(cells[8], "0", skill_name)

    def test_model_guard_fallback_judge_none_and_invalid_reps_excluded_not_zeroed(self) -> None:
        from harness2.cli import judge as judge_cli
        from harness2.cli import skillbench_report as SR

        # Roll k=3 on 1 skill x 1 task via the CLI:
        # - base1: valid rollout with no model field (model-guard fallback -> valid, score=6/6=1.0)
        # - base2: invalid rollout due to model mismatch (result_model_mismatch -> invalid, score=None)
        # - base3: valid rollout whose judge call fails (malformed output -> score=None)
        # Verify base2 and base3 are excluded (not zeroed), so WITH mean is 1.0 (not 1/3 or 1/2).
        # The stub echoes --model into init.model by default; the WITH arm replays skill_read
        # without any model field to exercise the fallback.
        no_model = self.tmp / "skill_read_no_model.jsonl"
        lines = (STUB.parent / "agy" / "skill_read.jsonl").read_text(encoding="utf-8").splitlines()
        first = json.loads(lines[0])
        del first["init"]["model"]
        _write(no_model, "\n".join([json.dumps(first), *lines[1:]]) + "\n")
        env = {**self.base_env(), "AGY_STUB_WITH_SCENARIO": str(no_model)}
        with mock.patch.dict(os.environ, env):
            with contextlib.redirect_stdout(io.StringIO()):
                rc_with = sweep_cli.main(
                    [
                        "--bench",
                        "skillbench",
                        "--arm",
                        "with",
                        "--k",
                        "3",
                        "--only-domains",
                        "make-plan",
                        "--max-tasks",
                        "1",
                    ]
                )
                rc_without = sweep_cli.main(
                    [
                        "--bench",
                        "skillbench",
                        "--arm",
                        "without",
                        "--k",
                        "3",
                        "--only-domains",
                        "make-plan",
                        "--max-tasks",
                        "1",
                    ]
                )
            self.assertEqual((rc_with, rc_without), (0, 0))

            adapt_dir = self.runs_dir / "skillbench" / "adapt"
            with_ns = adapt_dir / "skillbench-agy-agg3-with"
            without_ns = adapt_dir / "skillbench-agy-agg3-without"
            with_summary = json.loads(
                next(with_ns.glob("*/summary.json")).read_text(encoding="utf-8")
            )

            # Mutate base2's skill_timing.json to simulate a model-guard mismatch rollout
            base2_dir = Path(with_summary["base"][1]["run_dir"])
            timing2 = json.loads((base2_dir / "skill_timing.json").read_text(encoding="utf-8"))
            timing2["valid"] = False
            timing2["actual_model"] = "gemini-2.5-flash"
            timing2["invalid_reason"] = (
                "model mismatch: requested 'opus', agy reported 'gemini-2.5-flash'"
            )
            (base2_dir / "skill_timing.json").write_text(
                json.dumps(timing2, indent=2) + "\n", encoding="utf-8"
            )

            # Judge WITH arm: base1 gets 6/6, base2 is invalid -> None without calling judge,
            # and force base3's judge call to return None (malformed judge stream).
            real_score = SB.SkillbenchAdapter.score
            base3_dir = Path(with_summary["base"][2]["run_dir"]).resolve()

            def score_with_one_judge_none(adapter_self, run_dir: Path, task_id: str):
                if Path(run_dir).resolve() == base3_dir:
                    with mock.patch.dict(
                        os.environ, {"AGY_STUB_JUDGE_SCENARIO": "malformed_line"}
                    ):
                        return real_score(adapter_self, run_dir, task_id)
                return real_score(adapter_self, run_dir, task_id)

            with (
                mock.patch.object(SB.SkillbenchAdapter, "score", score_with_one_judge_none),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                rc_jw = judge_cli.main(
                    ["--bench", "skillbench", "--arm", "with", "--k", "3", "--bases"]
                )
                rc_jwo = judge_cli.main(
                    ["--bench", "skillbench", "--arm", "without", "--k", "3", "--bases"]
                )
            # judge_bases returns 2 when some rows are ungraded (base2, base3) and 0 when all graded
            self.assertEqual((rc_jw, rc_jwo), (2, 0))

            rep = SR.build_report(with_ns, without_ns, draws=200)
            self.assertFalse(rep["exec_model_verified"])
            row = rep["by_skill"]["make-plan"]
            t_entry = row["tasks"]["make-plan/token-bucket-limiter"]
            # Only base1 (1.0) is in with_fracs; base2 (invalid) and base3 (judge None) are excluded!
            self.assertEqual(t_entry["with_fracs"], [1.0])
            self.assertAlmostEqual(row["with_mean"], 1.0)
            self.assertEqual(row["invalid_with"], 1)
            self.assertEqual(row["ungraded_with"], 1)
            self.assertEqual(
                row["invalid_by_reason"],
                {"model mismatch: requested 'opus', agy reported 'gemini-2.5-flash'": 1},
            )
            # Valid WITH reps = 2 (base1 and base3); base2 (invalid) is excluded from invocation rate too
            self.assertEqual(row["valid_with_reps"], 2)
            self.assertEqual(row["invoked"], 2)

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
            SKILLBENCH_EFFORT="extreme",
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("SKILLBENCH_EFFORT", proc.stdout + proc.stderr)
        self.assertEqual(self.stub_calls(), [])
        self.assertFalse((self.runs_dir / "skillbench" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
