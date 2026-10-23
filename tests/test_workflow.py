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

"""Focused regressions for the paper's execution, evidence, and adoption contracts."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness2 import config
from harness2.cli import _common as C
from harness2.cli import judge as judge_cli
from harness2.cli import run_sweep as sweep_cli
from harness2.core import context, judge, recursion, schedule, tree
from harness2.core.library import Library
from harness2.core.proposer import Proposer, ProposerError, Reply
from harness2.core.spec import resolve_channel_model
from harness2.demo import BASE_SCRIPT, DemoAdapter, DemoProposer


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = DemoAdapter(self.root)
        self.proposer = DemoProposer(self.adapter.spec.channels["proposer"])

    def run_stream(self, k=2, proposer=None):
        return schedule.run_domain(
            "accounting",
            self.adapter.tasks(),
            k,
            adapter=self.adapter,
            prop=proposer or self.proposer,
            runs_root=self.adapter.spec.runs_root,
            library_root=self.root / "library" / str(k),
        )

    def test_recursion_budget_and_resume(self):
        with patch.dict(config.PROPOSER_MODEL_BY_CHANNEL, {"opencode": "", "claude_cli": ""}):
            unresolved = resolve_channel_model(None, None)
            self.assertEqual(unresolved.model, "")
            with self.assertRaises(ValueError):
                Proposer(unresolved)
            with self.assertRaises(ValueError):
                resolve_channel_model("opencode", "")
        for k in (1, 2):
            with self.subTest(k=k):
                result = self.run_stream(k)
                self.assertEqual((result["done"], result["failed"]), (2, 0))
                for task_id in self.adapter.tasks():
                    summary = json.loads(
                        (
                            self.adapter.spec.runs_root
                            / f"demo-oc-agg{k}"
                            / task_id
                            / "summary.json"
                        ).read_text()
                    )
                    for mode in ("agg", "seq"):
                        ledger = summary["modes"][mode]["ledger"]
                        self.assertEqual((ledger["R"], ledger["P"]), (4 * k, 3 * k))
                        selection = summary["selections"][mode]
                        expected = "Total: 85.15\n" if task_id == "ledger-1" else "Total: 20.25\n"
                        self.assertEqual(
                            self.adapter.deliverable_text(Path(selection["chosen_run_dir"])),
                            expected,
                        )
                executions, calls = self.adapter.execution_count, len(self.proposer._calls)
                resumed = self.run_stream(k)
                self.assertEqual(
                    (self.adapter.execution_count, len(self.proposer._calls)), (executions, calls)
                )
                self.assertEqual(
                    [row["status"] for row in resumed["tasks"]], ["skipped", "skipped"]
                )
        self.assertFalse(
            any(
                reply.label.endswith(".agg1") or reply.label.endswith(".seq1")
                for reply in self.proposer._calls
                if reply.label.startswith("select.")
            )
        )
        work = self.adapter.spec.runs_root / "demo-oc-seq2" / "ledger-1"
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        chosen = json.loads(summary_path.read_text())["selections"]["seq"]
        (work / "harnesses" / chosen["chosen_label"] / "scripts/summarize.py").write_text(
            "changed after selection\n"
        )
        repaired = self.run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])
        chosen = json.loads(summary_path.read_text())["selections"]["agg"]
        (Path(chosen["chosen_run_dir"]) / "answer.txt").unlink()
        repaired = self.run_stream()
        self.assertEqual([row["status"] for row in repaired["tasks"]], ["ok", "skipped"])

    def test_complete_evidence_and_library_prefix(self):
        (self.adapter.task_files("ledger-1") / "reference_answer.txt").write_text('"pass_rate": 1')
        self.run_stream()
        work = self.adapter.spec.runs_root / "demo-oc-seq2" / "ledger-1"
        reflection = work / "reflection/seq2"
        required = [
            "source/ledger.csv",
            "rollouts/base1.md",
            "rollouts/c2.md",
            "pool/inv1/harness.diff",
            "decisions/c1.md",
            "adoption.json",
            "base_harness/scripts/summarize.py",
            "harness/scripts/summarize.py",
        ]
        self.assertEqual([name for name in required if not (reflection / name).is_file()], [])
        self.assertTrue((work / "context/inv2/pool/c1/rollout.md").is_file())
        self.assertEqual(list(work.rglob("reference_answer.txt")), [])
        next_context = work.parent / "ledger-2/context/inv1/library/harnesses"
        self.assertEqual([path.name for path in next_context.iterdir()], ["ledger-1"])
        with self.assertRaises(ValueError):
            context.assert_tree_has_no_grader_verdicts({"EXPERIENCE.md": '"pass_rate": 1'})
        previous_sessions = []

        def run_opencode(command, **kwargs):
            database = Path(kwargs["env"]["XDG_DATA_HOME"]) / "opencode.db"
            previous_sessions.append(database.exists())
            database.parent.mkdir(parents=True, exist_ok=True)
            database.write_text("private evidence from this session")
            return Mock(stdout='{"type":"text","part":{"text":"done"}}', stderr="", returncode=0)

        proposer = Proposer(self.proposer.spec)
        with (
            patch("harness2.core.proposer._opencode_dir", return_value=self.root),
            patch("harness2.core.proposer.subprocess.run", side_effect=run_opencode),
        ):
            for index in range(2):
                proposer.call("Read this task only", None, f"isolation{index}")
        self.assertEqual(previous_sessions, [False, False])

    def test_invalid_proposal_retries_with_feedback_and_accumulates_cost(self):
        class RepairProposer(DemoProposer):
            feedback = ""

            def call(self, mission, context_dir, label, **kwargs):
                script = Path(context_dir) / "harness/scripts/summarize.py"
                if label.endswith(".a1"):
                    script.write_text("def broken(:\n")
                    return Reply(
                        label,
                        "invalid edit",
                        True,
                        cost_usd=1,
                        attempts=1,
                        workspace=script.parent.parent,
                    )
                self.feedback = mission
                script.write_text(BASE_SCRIPT)
                reply = super().call(mission, context_dir, label, **kwargs)
                reply.cost_usd = 2
                return reply

        proposer = RepairProposer(self.proposer.spec)
        stage = recursion._run_stage(
            label="inv1",
            dim="INV",
            job="Repair the script",
            adapter=self.adapter,
            prop=proposer,
            ctx_dir=self.root / "context",
            task_id="ledger-1",
            rollouts=[],
            fallback=tree.read_tree(self.adapter.substrate.surface(), self.adapter.base_harness()),
            harness_dir=self.adapter.base_harness(),
            library_dir=None,
        )
        self.assertEqual((stage.fell_back, stage.attempts, stage.usd), (False, 2, 3))
        self.assertIn("PREVIOUS ATTEMPT WAS REJECTED", proposer.feedback)

    def test_no_entry_stays_out_of_library_on_resume(self):
        class NoExperience(DemoProposer):
            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("experience."):
                    if label.endswith("seq1"):
                        raise ProposerError("reflection failed", attempts=2, cost_usd=2)
                    return Reply(label, "NO_ENTRY", True, cost_usd=1)
                return super().call(mission, context_dir, label, **kwargs)

        proposer = NoExperience(self.proposer.spec)
        self.run_stream(k=1, proposer=proposer)
        self.run_stream(k=1, proposer=proposer)
        for mode in ("agg", "seq"):
            library = Library(
                self.root / "library/1" / mode, "accounting", self.adapter.substrate.surface()
            )
            self.assertEqual(library.entries(), [])
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg1/ledger-1/summary.json"
        selections = json.loads(summary_path.read_text())["selections"]
        self.assertEqual([selections[mode]["usd"] for mode in ("agg", "seq")], [1, 2])

    def test_failed_selection_is_never_judged(self):
        class FailedSelector(DemoProposer):
            def call(self, mission, context_dir, label, **kwargs):
                if label.startswith("select."):
                    raise ProposerError("selection failed", attempts=2, cost_usd=3)
                return super().call(mission, context_dir, label, **kwargs)

        self.run_stream(proposer=FailedSelector(self.proposer.spec))
        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        summary = json.loads(summary_path.read_text())
        self.assertEqual([summary["selections"][mode]["usd"] for mode in ("agg", "seq")], [3, 3])
        self.adapter.score = Mock(return_value={"n_passed": 1, "n_criteria": 1})
        rows = judge.judge_selections(
            self.adapter,
            list(summary["selections"].values()),
        )
        self.adapter.score.assert_not_called()
        self.assertEqual((rows[0].graded, rows[0].n_passed), (False, None))

    def test_run_task_bases_only_rolls_bases_and_skips_proposer_and_selector(self):
        res = schedule.run_task(
            "ledger-1",
            2,
            adapter=self.adapter,
            prop=self.proposer,
            runs_root=self.adapter.spec.runs_root,
            bases_only=True,
        )
        self.assertEqual(res.status, "bases-only")
        self.assertEqual(self.adapter.execution_count, 2)
        self.assertEqual(self.proposer._calls, [])

        summary_path = self.adapter.spec.runs_root / "demo-oc-agg2/ledger-1/summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "bases-only")
        self.assertEqual([b["label"] for b in summary["base"]], ["base1", "base2"])
        self.assertEqual(summary["modes"], {})
        self.assertEqual(summary["selections"], {})
        self.assertEqual(summary["errors"], [])
        for b in summary["base"]:
            self.assertTrue((Path(b["run_dir"]) / "answer.txt").is_file())

        # prop=None also works when bases_only=True
        res2 = schedule.run_task(
            "ledger-2",
            1,
            adapter=self.adapter,
            prop=None,
            runs_root=self.adapter.spec.runs_root,
            bases_only=True,
        )
        self.assertEqual(res2.status, "bases-only")
        self.assertEqual(self.adapter.execution_count, 3)

        # Default path (bases_only=False) does not treat bases-only summary as completed recursion
        full_res = schedule.run_task(
            "ledger-1",
            2,
            adapter=self.adapter,
            prop=self.proposer,
            runs_root=self.adapter.spec.runs_root,
        )
        self.assertEqual(full_res.status, "ok")
        self.assertGreater(len(self.proposer._calls), 0)
        full_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(full_summary["status"], "ok")

    def test_bases_only_domain_resume_retry_and_run_sweep(self):
        dom = schedule.run_domain(
            "accounting",
            self.adapter.tasks(),
            2,
            adapter=self.adapter,
            prop=self.proposer,
            runs_root=self.adapter.spec.runs_root,
            library_root=self.root / "library/2",
            bases_only=True,
        )
        self.assertEqual((dom["done"], dom["failed"]), (2, 0))
        self.assertEqual([t["status"] for t in dom["tasks"]], ["bases-only", "bases-only"])
        self.assertEqual(self.adapter.execution_count, 4)
        self.assertEqual(self.proposer._calls, [])

        # Second run skips via _resume without re-rolling
        dom_resumed = schedule.run_domain(
            "accounting",
            self.adapter.tasks(),
            2,
            adapter=self.adapter,
            prop=self.proposer,
            runs_root=self.adapter.spec.runs_root,
            library_root=self.root / "library/2",
            bases_only=True,
        )
        self.assertEqual((dom_resumed["done"], dom_resumed["failed"]), (2, 0))
        self.assertEqual([t["status"] for t in dom_resumed["tasks"]], ["skipped", "skipped"])
        self.assertEqual(self.adapter.execution_count, 4)

        # A base-failed task is retried on the next run
        real_execute = self.adapter.execute

        def fail_ledger_1(task_id, harness_dir, run_id, **kwargs):
            if task_id == "ledger-1":
                raise RuntimeError("transient base failure")
            return real_execute(task_id, harness_dir, run_id, **kwargs)

        with patch.object(self.adapter, "execute", side_effect=fail_ledger_1):
            dom_failed = schedule.run_domain(
                "accounting",
                self.adapter.tasks(),
                1,
                adapter=self.adapter,
                prop=self.proposer,
                runs_root=self.adapter.spec.runs_root,
                library_root=self.root / "library/1",
                bases_only=True,
            )
        self.assertEqual((dom_failed["done"], dom_failed["failed"]), (1, 1))
        self.assertEqual(
            [t["status"] for t in dom_failed["tasks"]], ["base-failed", "bases-only"]
        )

        dom_retried = schedule.run_domain(
            "accounting",
            self.adapter.tasks(),
            1,
            adapter=self.adapter,
            prop=self.proposer,
            runs_root=self.adapter.spec.runs_root,
            library_root=self.root / "library/1",
            bases_only=True,
        )
        self.assertEqual((dom_retried["done"], dom_retried["failed"]), (2, 0))
        self.assertEqual(
            [t["status"] for t in dom_retried["tasks"]], ["bases-only", "skipped"]
        )
        retried_summary = json.loads(
            (self.adapter.spec.runs_root / "demo-oc-agg1/ledger-1/summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(retried_summary["status"], "bases-only")
        self.assertEqual(retried_summary["errors"], [])

        # run_sweep CLI: --bases-only is False by default, True when passed, and implied by --bench skillbench
        parser = sweep_cli._build_parser()
        self.assertFalse(parser.parse_args(["--bench", "jb"]).bases_only)
        self.assertTrue(parser.parse_args(["--bench", "jb", "--bases-only"]).bases_only)
        self.assertTrue(parser.parse_args(["--bench", "skillbench", "--arm", "with"]).bases_only)

        # run_sweep CLI with --bases-only exits 0 and never constructs Proposer
        with (
            patch.object(C, "make_adapter", return_value=self.adapter),
            patch("harness2.preflight.require_ready"),
            patch.object(
                sweep_cli,
                "Proposer",
                side_effect=AssertionError("Proposer must not be constructed for --bases-only"),
            ),
            patch.object(schedule, "run_sweep", wraps=schedule.run_sweep) as spy_sweep,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc = sweep_cli.main(["--bench", "jb", "--k", "2", "--bases-only"])
        self.assertEqual(rc, 0)
        self.assertTrue(spy_sweep.call_args.kwargs.get("bases_only"))

        # --bench skillbench also implies bases_only=True in schedule.run_sweep
        with (
            patch.object(C, "make_adapter", return_value=self.adapter),
            patch("harness2.preflight.require_ready"),
            patch.object(sweep_cli, "_check_identity"),
            patch.object(
                sweep_cli,
                "Proposer",
                side_effect=AssertionError("Proposer must not be constructed for skillbench"),
            ),
            patch.object(schedule, "run_sweep", wraps=schedule.run_sweep) as spy_sb_sweep,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc_sb = sweep_cli.main(["--bench", "skillbench", "--arm", "with", "--k", "1"])
        self.assertEqual(rc_sb, 0)
        self.assertTrue(spy_sb_sweep.call_args.kwargs.get("bases_only"))

    def test_judge_bases_scores_bases_only_namespace(self):
        with (
            patch.object(C, "make_adapter", return_value=self.adapter),
            patch("harness2.preflight.require_ready"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc_sweep = sweep_cli.main(["--bench", "jb", "--k", "2", "--bases-only"])
        self.assertEqual(rc_sweep, 0)

        def score_base(run_dir: Path, task_id: str) -> dict[str, int]:
            self.assertTrue((run_dir / "answer.txt").is_file())
            return {"n_passed": 1 if task_id == "ledger-1" else 2, "n_criteria": 2}

        self.adapter.score = Mock(side_effect=score_base)
        with (
            patch.object(C, "make_adapter", return_value=self.adapter),
            patch.object(config, "validate_models"),
            patch.object(config, "JUDGE_MODEL", ""),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc_judge = judge_cli.main(["--bench", "jb", "--k", "2", "--bases"])
        self.assertEqual(rc_judge, 0)
        self.assertEqual(self.adapter.score.call_count, 4)

        ns_dir = C.runs_root_for(self.adapter) / "demo-oc-agg2"
        base_judged = json.loads((ns_dir / "base_judged.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [(r["task"], r["label"], r["score"]) for r in base_judged],
            [
                ("ledger-1", "base1", {"n_passed": 1, "n_criteria": 2}),
                ("ledger-1", "base2", {"n_passed": 1, "n_criteria": 2}),
                ("ledger-2", "base1", {"n_passed": 2, "n_criteria": 2}),
                ("ledger-2", "base2", {"n_passed": 2, "n_criteria": 2}),
            ],
        )


if __name__ == "__main__":
    unittest.main()
