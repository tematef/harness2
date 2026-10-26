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

"""Arm/tag threading through `_common`, `run_sweep`, and `judge` CLIs for skillbench.

All tests run in temporary directories with a fake HOME and `tests/fixtures/agy_stub.py`;
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

from harness2 import config
from harness2.benchmarks import skillbench as SB
from harness2.cli import _common as C
from harness2.cli import judge as judge_cli
from harness2.cli import report as report_cli
from harness2.cli import run_sweep as sweep_cli
from harness2.core import schedule
from harness2.core.proposer import Reply
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


class _FakeProposer:
    """In-memory proposer stub so full sweeps never invoke external model CLIs."""

    def __init__(self, spec, *, audit_dir: Path | None = None, **kwargs) -> None:
        self.spec = spec
        self.audit_dir = audit_dir
        self.attempts = 2
        self.project = "test-project"
        self.region = "global"

    def call(self, mission: str, context_dir: Path | None, label: str, **kwargs) -> Reply:
        if label.startswith("select."):
            candidates = sorted((Path(context_dir) / "candidates").glob("*.md"))
            chosen = candidates[0].stem if candidates else "base1"
            text = f"<why>Valid candidate.</why>\n```choice\n{chosen}\n```"
            return Reply(label=label, text=text, ok=True, cost_usd=0.0, attempts=1)
        if label.startswith("experience."):
            text = (
                "<edits>\ncomponent: GEMINI.md\ncandidates: inv1\n"
                "change: kept base\nobserved: base completed\n"
                "verdict: no observed effect\n</edits>\n\nNO_ENTRY"
            )
            return Reply(label=label, text=text, ok=True, cost_usd=0.0, attempts=1)
        ws = Path(context_dir) / "harness" if context_dir is not None else None
        return Reply(
            label=label,
            text="<notes>kept base harness</notes>",
            ok=True,
            cost_usd=0.0,
            attempts=1,
            workspace=ws,
        )


class _CliCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-cli-test-"))
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
            "HARNESS2_SOLVER_MODEL": "google-vertex/gemini-2.5-pro",
            "HARNESS2_IMPROVER_MODEL": "google-vertex/gemini-2.5-pro",
            "HARNESS2_JUDGE_MODEL": "fable",
            "GOOGLE_CLOUD_PROJECT": "test-project",
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
            ("SOLVER_MODEL_OC", "google-vertex/gemini-2.5-pro"),
            ("IMPROVER_MODEL", "google-vertex/gemini-2.5-pro"),
            ("JUDGE_MODEL", "fable"),
            ("VERTEX_PROJECT", "test-project"),
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


class TestCommonAndAdapterConstruction(_CliCase):
    def test_benches_includes_skillbench(self) -> None:
        self.assertIn("skillbench", C.BENCHES)

    def test_arm_required_for_skillbench_and_rejected_for_other_benches(self) -> None:
        with self.assertRaisesRegex(ValueError, "--arm"):
            C.make_adapter("skillbench", None, "agy")
        with self.assertRaisesRegex(ValueError, "--arm"):
            C.make_adapter("skillbench", None, "agy", arm=None)
        with self.assertRaisesRegex(ValueError, "--arm"):
            C.make_adapter("skillbench", None, "agy", arm="both")

        for bench, subset in (("jb", None), ("lab", None), ("wb", "code")):
            with self.subTest(bench=bench):
                with self.assertRaisesRegex(ValueError, "--arm"):
                    C.make_adapter(bench, subset, "oc", arm="with")
                with self.assertRaisesRegex(ValueError, "--arm"):
                    C.make_adapter(bench, subset, "oc", arm="without")

    def test_make_adapter_builds_skillbench_adapter_with_requested_arm(self) -> None:
        without_adapter = C.make_adapter("skillbench", None, "agy", arm="without")
        self.assertIsInstance(without_adapter, SB.SkillbenchAdapter)
        self.assertEqual(without_adapter.arm, "without")
        self.assertEqual(without_adapter.substrate.name, "agy")

        with_adapter = C.make_adapter("skillbench", None, "agy", arm="with")
        self.assertIsInstance(with_adapter, SB.SkillbenchAdapter)
        self.assertEqual(with_adapter.arm, "with")
        self.assertEqual(with_adapter.substrate.name, "agy")

    def test_skillbench_rejects_oc_and_cx_substrates_and_subset(self) -> None:
        for bad_sub in ("oc", "cx"):
            with self.subTest(substrate=bad_sub):
                with self.assertRaisesRegex(ValueError, "agy"):
                    C.make_adapter("skillbench", None, bad_sub, arm="with")
        for bench, subset in (("jb", None), ("lab", None), ("wb", "code")):
            with self.subTest(bench=bench):
                with self.assertRaisesRegex(ValueError, "skillbench"):
                    C.make_adapter(bench, subset, "agy")
        with self.assertRaisesRegex(ValueError, "--subset"):
            C.make_adapter("skillbench", "code", "agy", arm="with")

    def test_omitted_substrate_resolves_to_agy_for_skillbench_and_oc_otherwise(self) -> None:
        # Via make_adapter with substrate=None or omitted
        sb = C.make_adapter("skillbench", None, None, arm="without")
        self.assertEqual(sb.substrate.name, "agy")
        self.assertEqual(sb.arm, "without")

        jb = C.make_adapter("jb", None, None)
        self.assertEqual(jb.substrate.name, "oc")
        wb = C.make_adapter("wb", "code", None)
        self.assertEqual(wb.substrate.name, "oc")
        lab = C.make_adapter("lab", None, None)
        self.assertEqual(lab.substrate.name, "oc")

        # Via CLI parser (add_cell_args)
        parser = sweep_cli._build_parser()
        sb_args = parser.parse_args(["--bench", "skillbench", "--arm", "with"])
        self.assertEqual(sb_args.substrate, "agy")
        self.assertEqual(sb_args.arm, "with")

        jb_args = parser.parse_args(["--bench", "jb"])
        self.assertEqual(jb_args.substrate, "oc")
        self.assertIsNone(jb_args.arm)

        # Explicit --substrate is preserved (both before and after --bench) so make_adapter rejects bad combos
        explicit_after = parser.parse_args(
            ["--bench", "skillbench", "--arm", "with", "--substrate", "oc"]
        )
        self.assertEqual(explicit_after.substrate, "oc")
        explicit_before = parser.parse_args(
            ["--substrate", "cx", "--bench", "skillbench", "--arm", "with"]
        )
        self.assertEqual(explicit_before.substrate, "cx")


class TestRunSweepThreading(_CliCase):
    def test_dry_run_threads_arm_tag_and_rejects_bad_arm_or_substrate(self) -> None:
        for arm in ("with", "without"):
            buf = io.StringIO()
            with (
                mock.patch.object(
                    C, "summary_namespace", wraps=C.summary_namespace
                ) as spy_ns,
                mock.patch.object(
                    C, "library_root_for", wraps=C.library_root_for
                ) as spy_lib,
                mock.patch.object(
                    sweep_cli, "_print_stream", wraps=sweep_cli._print_stream
                ) as spy_print,
                contextlib.redirect_stdout(buf),
            ):
                rc = sweep_cli.main(["--bench", "skillbench", "--arm", arm, "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertEqual(spy_print.call_count, 1)
            self.assertEqual(spy_print.call_args.kwargs.get("tag"), arm)
            self.assertEqual(spy_ns.call_args.kwargs.get("tag"), arm)
            self.assertEqual(spy_lib.call_args.kwargs.get("tag"), arm)
            out = buf.getvalue()
            self.assertIn(f"skillbench-agy-agg1-{arm}", out)
            self.assertIn(f"agy-{arm}", out)

        # Missing --arm on skillbench fails with error
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = sweep_cli.main(["--bench", "skillbench", "--dry-run"])
        self.assertEqual(rc, 1)
        self.assertIn("--arm", err.getvalue())

        # --arm on non-skillbench fails with error
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = sweep_cli.main(["--bench", "jb", "--arm", "with", "--dry-run"])
        self.assertEqual(rc, 1)
        self.assertIn("--arm", err.getvalue())

        # --substrate oc on skillbench fails with error
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = sweep_cli.main(
                ["--bench", "skillbench", "--arm", "with", "--substrate", "oc", "--dry-run"]
            )
        self.assertEqual(rc, 1)
        self.assertIn("agy", err.getvalue())

    def test_run_sweep_passes_tag_arm_to_all_consumers_and_keeps_empty_tag_for_other_benches(
        self,
    ) -> None:
        fake_rep = {
            "domains": [{"domain": "make-plan", "done": 1, "failed": 0, "tasks": []}],
        }
        for arm in ("with", "without"):
            buf = io.StringIO()
            with (
                mock.patch("harness2.preflight.require_ready"),
                mock.patch.object(sweep_cli, "Proposer", _FakeProposer),
                mock.patch.object(
                    C, "summary_namespace", wraps=C.summary_namespace
                ) as spy_ns,
                mock.patch.object(
                    C, "library_root_for", wraps=C.library_root_for
                ) as spy_lib,
                mock.patch.object(
                    sweep_cli, "_check_identity", wraps=sweep_cli._check_identity
                ) as spy_id,
                mock.patch.object(
                    schedule, "run_sweep", return_value=fake_rep
                ) as spy_sched,
                contextlib.redirect_stdout(buf),
            ):
                rc = sweep_cli.main(
                    [
                        "--bench",
                        "skillbench",
                        "--arm",
                        arm,
                        "--k",
                        "1",
                        "--seed",
                        "7",
                        "--only-domains",
                        "make-plan",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(spy_ns.call_args.kwargs.get("tag"), arm)
            self.assertEqual(spy_lib.call_args.kwargs.get("tag"), arm)
            self.assertEqual(spy_id.call_args.kwargs.get("tag"), arm)
            self.assertEqual(spy_sched.call_args.kwargs.get("tag"), arm)
            expected_ns = f"skillbench-agy-agg1-s7-{arm}"
            self.assertEqual(spy_id.call_args.args[2].name, expected_ns)

        # Non-skillbench keeps tag=""
        buf = io.StringIO()
        with (
            mock.patch(
                "harness2.benchmarks.jobbench.discover",
                return_value={"acc/t1": "acc"},
            ),
            mock.patch("harness2.preflight.require_ready"),
            mock.patch.object(sweep_cli, "Proposer", _FakeProposer),
            mock.patch.object(
                C, "summary_namespace", wraps=C.summary_namespace
            ) as spy_ns,
            mock.patch.object(
                C, "library_root_for", wraps=C.library_root_for
            ) as spy_lib,
            mock.patch.object(
                sweep_cli, "_check_identity", wraps=sweep_cli._check_identity
            ) as spy_id,
            mock.patch.object(
                schedule,
                "run_sweep",
                return_value={"domains": [{"domain": "acc", "done": 1, "failed": 0}]},
            ) as spy_sched,
            contextlib.redirect_stdout(buf),
        ):
            rc = sweep_cli.main(["--bench", "jb", "--k", "1"])
        self.assertEqual(rc, 0)
        self.assertEqual(spy_ns.call_args.kwargs.get("tag", ""), "")
        self.assertEqual(spy_lib.call_args.kwargs.get("tag", ""), "")
        self.assertEqual(spy_id.call_args.kwargs.get("tag", ""), "")
        self.assertEqual(spy_sched.call_args.kwargs.get("tag"), "")

    def test_with_and_without_sweeps_write_separate_namespaces(self) -> None:
        for arm in ("with", "without"):
            buf = io.StringIO()
            with (
                mock.patch("harness2.preflight.require_ready"),
                mock.patch.object(sweep_cli, "Proposer", _FakeProposer),
                contextlib.redirect_stdout(buf),
            ):
                rc = sweep_cli.main(
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
            self.assertIn(rc, (0, 2))

        adapt_root = self.runs_dir / "skillbench" / "adapt"
        with_ns_dir = adapt_root / "skillbench-agy-agg1-with"
        without_ns_dir = adapt_root / "skillbench-agy-agg1-without"
        untagged_ns_dir = adapt_root / "skillbench-agy-agg1"

        self.assertTrue(with_ns_dir.is_dir())
        self.assertTrue(without_ns_dir.is_dir())
        self.assertFalse(untagged_ns_dir.exists())
        self.assertTrue((with_ns_dir / "config.json").is_file())
        self.assertTrue((without_ns_dir / "config.json").is_file())

        with_summary = json.loads(
            (with_ns_dir / "make-plan__p1" / "summary.json").read_text(encoding="utf-8")
        )
        without_summary = json.loads(
            (without_ns_dir / "make-plan__p1" / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            with_summary["base"][0]["run_id"],
            "skillbench-agy-agg1-with/make-plan/p1/base1",
        )
        self.assertEqual(
            without_summary["base"][0]["run_id"],
            "skillbench-agy-agg1-without/make-plan/p1/base1",
        )
        self.assertNotEqual(
            with_summary["base"][0]["run_dir"],
            without_summary["base"][0]["run_dir"],
        )

        sweeps_dir = adapt_root / "_sweeps"
        self.assertEqual(len(list(sweeps_dir.glob("skillbench-agy-agg1-with-*.json"))), 1)
        self.assertEqual(len(list(sweeps_dir.glob("skillbench-agy-agg1-without-*.json"))), 1)


class TestJudgeCliArmThreading(_CliCase):
    def _seed_arm_namespace(
        self, arm: str, *, passed: bool, task_id: str = "make-plan/p1"
    ) -> tuple[Path, Path]:
        adapter = C.make_adapter("skillbench", None, "agy", arm=arm)
        ns = C.summary_namespace(adapter, 1, seed=None, tag=arm)
        ns_dir = C.runs_root_for(adapter) / ns
        run_dir = adapter.execute(task_id, None, f"{ns}/{task_id}/base1")
        task_dir = ns_dir / task_id.replace("/", "__")
        task_dir.mkdir(parents=True, exist_ok=True)
        _write(
            task_dir / "summary.json",
            json.dumps(
                {
                    "task_id": task_id,
                    "tag": arm,
                    "status": "ok",
                    "base": [
                        {
                            "label": "base1",
                            "run_id": f"{ns}/{task_id}/base1",
                            "run_dir": str(run_dir),
                            "ok": True,
                            "delivered": True,
                        }
                    ],
                },
                indent=2,
            ),
        )
        return ns_dir, run_dir

    def test_judge_bases_reads_without_namespace_only(self) -> None:
        with_ns_dir, with_run_dir = self._seed_arm_namespace("with", passed=True)
        without_ns_dir, without_run_dir = self._seed_arm_namespace("without", passed=False)

        judge_scen = _scenario(
            self.tmp / "judge_without.jsonl",
            {
                "event": "result",
                "result": {
                    "structured_output": {
                        "criteria": [
                            {"id": "c0", "pass": True, "reason": "ok"},
                            {"id": "c1", "pass": False, "reason": "no"},
                            {"id": "c2", "pass": True, "reason": "ok"},
                            {"id": "c3", "pass": False, "reason": "no"},
                        ]
                    },
                    "usage": {"total_tokens": 120},
                },
            },
        )
        os.environ["AGY_STUB_SCENARIO"] = str(judge_scen)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # Omit --substrate to also verify judge defaults --substrate to agy for skillbench
            rc = judge_cli.main(
                ["--bench", "skillbench", "--arm", "without", "--bases"]
            )
        self.assertEqual(rc, 0)

        without_judged = without_ns_dir / "base_judged.json"
        with_judged = with_ns_dir / "base_judged.json"
        self.assertTrue(without_judged.is_file())
        self.assertFalse(with_judged.exists())
        self.assertTrue((without_run_dir / "judgments.json").is_file())
        self.assertFalse((with_run_dir / "judgments.json").exists())

        rows = json.loads(without_judged.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task"], "make-plan/p1")
        self.assertEqual(rows[0]["label"], "base1")
        self.assertEqual(rows[0]["score"], {"n_passed": 2, "n_criteria": 4})

    def test_judge_requires_arm_for_skillbench_and_rejects_for_other_benches(self) -> None:
        with self.assertRaisesRegex(ValueError, "--arm"):
            judge_cli.main(["--bench", "skillbench", "--bases"])
        with self.assertRaisesRegex(ValueError, "--arm"):
            judge_cli.main(["--bench", "jb", "--arm", "without", "--bases"])


class TestReportCliArmThreading(_CliCase):
    """The upstream `harness2-report` reads a skillbench arm's reference executions."""

    def _seed_base_judged(self, arm: str, scores: list[dict | None], k: int = 1) -> Path:
        adapter = C.make_adapter("skillbench", None, "agy", arm=arm)
        ns_dir = C.runs_root_for(adapter) / C.summary_namespace(adapter, k, seed=None, tag=arm)
        ns_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "task": "make-plan/p1",
                "label": f"base{i}",
                "run_id": f"{ns_dir.name}/make-plan/p1/base{i}",
                "score": score,
            }
            for i, score in enumerate(scores, start=1)
        ]
        _write(ns_dir / "base_judged.json", json.dumps(rows, indent=1))
        return ns_dir

    def _report(self, *argv: str) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = report_cli.main(list(argv))
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_report_reads_each_arm_namespace_bases(self) -> None:
        self._seed_base_judged("with", [{"n_passed": 4, "n_criteria": 4}] * 2)
        self._seed_base_judged(
            "without", [{"n_passed": 2, "n_criteria": 4}, {"n_passed": 1, "n_criteria": 4}]
        )

        with_out = self._report("--bench", "skillbench", "--arm", "with", "--k", "1")
        without_out = self._report("--bench", "skillbench", "--arm", "without", "--k", "1")

        self.assertIn("arm=with", with_out)
        self.assertIn("bases k=1: n=  2 | score 100.00%", with_out)
        self.assertIn("arm=without", without_out)
        self.assertIn("bases k=1: n=  2 | score  37.50%", without_out)
        for out in (with_out, without_out):
            self.assertNotIn("picked output", out)
            self.assertNotIn("WARNING", out)

    def test_report_counts_ungraded_bases_and_explicit_bases_flag(self) -> None:
        self._seed_base_judged("with", [{"n_passed": 3, "n_criteria": 4}, None])
        out = self._report("--bench", "skillbench", "--arm", "with", "--k", "1", "--bases")
        self.assertIn("bases k=1: n=  1 | score  75.00% | ungraded 1", out)

    def test_report_without_judged_bases_has_nothing_to_report(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
            report_cli.main(["--bench", "skillbench", "--arm", "with", "--k", "1"])
        self.assertIn("base_judged.json", buf.getvalue())
        self.assertIn("--bases", buf.getvalue())

    def test_report_requires_arm_for_skillbench_and_rejects_for_other_benches(self) -> None:
        with self.assertRaisesRegex(ValueError, "--arm"):
            report_cli.main(["--bench", "skillbench", "--k", "1"])
        with self.assertRaisesRegex(ValueError, "--arm"):
            report_cli.main(["--bench", "jb", "--arm", "with", "--k", "1"])


if __name__ == "__main__":
    unittest.main()
