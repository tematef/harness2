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

"""Per-executor hardness flag of `harness2.cli.skillbench_report`: `task_hardness`
classification, per-skill counts and the `needs harder tasks` flag, per-executor independence,
the lift being unchanged by ceiling tasks, `build_hardness` on control-arm cells alone,
`--hardness-only` output and the HTML / markdown badges.

Synthetic namespaces are built with `seed_namespace` from the schema-v2 tests.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from unittest import mock

from harness2 import config
from harness2.cli import skillbench_report as SR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_skillbench_report_v2 import Spec, seed_namespace  # noqa: E402

# Control arm (WITHOUT) hits the ceiling on t1 and t2, misses on t3; WITH is perfect.
TWO_CEILING: Spec = {
    "alpha/t1": {"with": [4, 4], "without": [4, 4]},
    "alpha/t2": {"with": [4, 4], "without": [4, 4]},
    "alpha/t3": {"with": [4, 4], "without": [2, 3]},
}
# The same tasks, informative for a weaker control arm.
INFORMATIVE: Spec = {
    "alpha/t1": {"with": [4, 4], "without": [3, 4]},
    "alpha/t2": {"with": [4, 4], "without": [2, 2]},
    "alpha/t3": {"with": [4, 4], "without": [2, 3]},
}


def rec(n_missed: int | None, *, valid: bool = True, task_id: str = "s/t") -> SR._RepRecord:
    """A minimal control-arm record with `n_missed` criteria failed (None = ungraded)."""
    return SR._RepRecord(
        task_id=task_id,
        skill=task_id.split("/", 1)[0],
        label="base1",
        run_id="r",
        valid=valid,
        invalid_reason=None if valid else "exit 1",
        invoked=False,
        t_skill_s=None,
        requested_model=None,
        actual_model=None,
        frac=None if n_missed is None or not valid else 1.0 - n_missed / 4,
        n_missed=None if not valid else n_missed,
    )


class TestTaskHardness(unittest.TestCase):
    def test_classification(self) -> None:
        self.assertEqual(SR.task_hardness([rec(0), rec(0)]), "ceiling")
        self.assertEqual(SR.task_hardness([rec(0), rec(1)]), "informative")
        self.assertEqual(SR.task_hardness([rec(2), rec(3), rec(0)]), "informative")
        # Fewer than min_reps valid graded reps: unclassified, whatever they scored.
        self.assertEqual(SR.task_hardness([rec(3)]), "unclassified")
        self.assertEqual(SR.task_hardness([]), "unclassified")
        self.assertEqual(SR.task_hardness([rec(0)]), "unclassified")
        # Invalid and ungraded reps do not count towards min_reps.
        self.assertEqual(SR.task_hardness([rec(0), rec(2, valid=False)]), "unclassified")
        self.assertEqual(SR.task_hardness([rec(0), rec(None)]), "unclassified")
        self.assertEqual(
            SR.task_hardness([rec(0), rec(1, valid=False), rec(0)]), "ceiling"
        )

    def test_min_reps_and_min_baseline_fail(self) -> None:
        self.assertEqual(SR.task_hardness([rec(1)], min_reps=1), "informative")
        self.assertEqual(SR.task_hardness([rec(0)], min_reps=1), "ceiling")
        self.assertEqual(SR.task_hardness([rec(0), rec(0), rec(1)], min_reps=4), "unclassified")
        # A task that demands two missed criteria treats a single miss as ceiling.
        self.assertEqual(SR.task_hardness([rec(1), rec(1)], min_baseline_fail=2), "ceiling")
        self.assertEqual(SR.task_hardness([rec(1), rec(2)], min_baseline_fail=2), "informative")
        for bad in ({"min_baseline_fail": 0}, {"min_reps": 0}, {"min_baseline_fail": True}):
            with self.subTest(bad), self.assertRaises(ValueError):
                SR.task_hardness([rec(1), rec(1)], **bad)
        self.assertEqual(
            SR.baseline_missed([rec(2), rec(0, valid=False), rec(None), rec(1)]), [2, 1]
        )

    def test_skill_hardness_counts_and_flag_threshold(self) -> None:
        self.assertEqual(
            SR.skill_hardness(["ceiling", "ceiling", "informative"]),
            {
                "ceiling_tasks": 2,
                "informative_tasks": 1,
                "unclassified_tasks": 0,
                "total_tasks": 3,
                "ceiling_share": 2 / 3,
                "needs_harder_tasks": True,
            },
        )
        # Exactly half is not "more than 50 %".
        self.assertFalse(SR.skill_hardness(["ceiling", "informative"])["needs_harder_tasks"])
        self.assertTrue(SR.skill_hardness(["ceiling", "ceiling", "informative", "ceiling"])[
            "needs_harder_tasks"
        ])
        # Unclassified tasks count in the denominator, not as ceiling.
        half = SR.skill_hardness(["ceiling", "unclassified"])
        self.assertEqual((half["ceiling_share"], half["needs_harder_tasks"]), (0.5, False))
        self.assertEqual(half["unclassified_tasks"], 1)
        empty = SR.skill_hardness([])
        self.assertEqual((empty["total_tasks"], empty["ceiling_share"]), (0, None))
        self.assertFalse(empty["needs_harder_tasks"])

    def test_thresholds_spec(self) -> None:
        self.assertEqual(SR._thresholds(2), (2, {}))
        self.assertEqual(SR._thresholds({"a/t1": 3}), (1, {"a/t1": 3}))
        with self.assertRaises(ValueError):
            SR._thresholds(0)
        with self.assertRaises(ValueError):
            SR._thresholds({"a/t1": 0})
        with mock.patch.object(SR, "corpus_thresholds", return_value={"x/y": 2}):
            self.assertEqual(SR._thresholds(None), (1, {"x/y": 2}))
        rows = SR.hardness_rows(
            [
                rec(1, task_id="a/t1"),
                rec(1, task_id="a/t1"),
                rec(1, task_id="a/t2"),
                rec(0, task_id="a/t2"),
            ],
            (1, {"a/t1": 2}),
        )
        self.assertEqual(rows["a/t1"]["hardness"], "ceiling")
        self.assertEqual(rows["a/t1"]["min_baseline_fail"], 2)
        self.assertEqual(rows["a/t2"]["hardness"], "informative")
        self.assertEqual(rows["a/t2"]["baseline_missed"], [1, 0])
        self.assertEqual(rows["a/t2"]["valid_reps"], 2)

    def test_corpus_thresholds_reads_the_corpus_or_stays_empty(self) -> None:
        with mock.patch.object(SR, "load_tasks", side_effect=config.ConfigError("no suite")):
            self.assertEqual(SR.corpus_thresholds(), {})
        with mock.patch.object(SR, "load_tasks", side_effect=FileNotFoundError("gone")):
            self.assertEqual(SR.corpus_thresholds(), {})
        fake = {"a/t1": mock.Mock(min_baseline_fail=2), "a/t2": mock.Mock(min_baseline_fail=1)}
        with mock.patch.object(SR, "load_tasks", return_value=fake):
            self.assertEqual(SR.corpus_thresholds(), {"a/t1": 2, "a/t2": 1})


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-hardness-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.runs_dir = self.tmp / "runs"
        env = {
            "HOME": str(self.home),
            "HARNESS2_RUNS_DIR": str(self.runs_dir),
            "SKILLBENCH_EXEC_MODEL": "env-exec",
            "SKILLBENCH_JUDGE_MODEL": "env-judge",
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ("SKILLBENCH_PILOT", "SKILLBENCH_SUITE", "SKILLBENCH_PLUGINS_DIR"):
            os.environ.pop(key, None)
        p = mock.patch.object(config, "RUNS", self.runs_dir)
        p.start()
        self.addCleanup(p.stop)

    def matrix(self, executors: dict[str, tuple[Spec, dict[str, Any]]]) -> Path:
        """`results/<executor>/<arm>` namespaces from a per-executor `(spec, kwargs)`."""
        root = self.runs_dir / "run-1" / "results"
        for executor, (spec, kw) in executors.items():
            kw = {"executor": executor, **kw}
            for arm in ("with", "without"):
                seed_namespace(root / executor / arm, arm, spec, **kw)
        return root


class TestReportHardness(_Case):
    def test_per_executor_independence_and_flag(self) -> None:
        root = self.matrix(
            {"exec-a": (TWO_CEILING, {}), "exec-c": (INFORMATIVE, {"judge": "judge-d"})}
        )
        rep = SR.build_report(root, draws=50)
        a, c = rep["models"]
        self.assertEqual(
            a["task_hardness"],
            {"alpha/t1": "ceiling", "alpha/t2": "ceiling", "alpha/t3": "informative"},
        )
        self.assertEqual(
            c["task_hardness"],
            {"alpha/t1": "informative", "alpha/t2": "informative", "alpha/t3": "informative"},
        )
        row_a, row_c = a["by_skill"]["alpha"], c["by_skill"]["alpha"]
        self.assertEqual((row_a["ceiling_tasks"], row_a["total_tasks"]), (2, 3))
        self.assertTrue(row_a["needs_harder_tasks"])
        self.assertEqual((row_c["ceiling_tasks"], row_c["total_tasks"]), (0, 3))
        self.assertFalse(row_c["needs_harder_tasks"])
        self.assertEqual(row_a["total_tasks"], row_a["n_tasks"])
        t1 = row_a["tasks"]["alpha/t1"]
        self.assertEqual(t1["hardness"], "ceiling")
        self.assertEqual(t1["baseline_missed"], [0, 0])
        self.assertEqual(t1["min_baseline_fail"], 1)
        self.assertEqual(row_a["tasks"]["alpha/t3"]["baseline_missed"], [2, 1])
        self.assertEqual([r["n_missed"] for r in t1["reps"]["without"]], [0, 0])
        self.assertIn("hardness_note", rep)
        # The legacy mirror carries the new per-skill fields too.
        self.assertEqual(rep["by_skill"]["alpha"]["ceiling_tasks"], 2)

    def test_ceiling_tasks_stay_in_the_lift(self) -> None:
        root = self.matrix({"exec-a": (TWO_CEILING, {})})
        rep = SR.build_report(root, draws=200, bootstrap_seed=1)
        row = rep["by_skill"]["alpha"]
        # Mean of per-task deltas over ALL three tasks: (0 + 0 + 0.375) / 3.
        self.assertAlmostEqual(row["lift"], 0.125)
        self.assertEqual(row["n_paired_tasks"], 3)
        self.assertEqual(row["n_qualifying_tasks"], 3)
        self.assertEqual(set(row["task_deltas"]), {"alpha/t1", "alpha/t2", "alpha/t3"})
        self.assertEqual(row["ci"][0] <= row["lift"] <= row["ci"][1], True)
        # Same numbers whether or not the hardness fields are looked at: a report built with
        # a threshold that makes every task informative has the identical lift / CI / verdict.
        rep2 = SR.build_report(root, draws=200, bootstrap_seed=1, min_baseline_fail=1)
        strict = SR.build_report(root, draws=200, bootstrap_seed=1, min_baseline_fail=3)
        for key in ("lift", "ci", "verdict", "with_mean", "without_mean", "n_qualifying_tasks"):
            self.assertEqual(rep2["by_skill"]["alpha"][key], row[key], key)
            self.assertEqual(strict["by_skill"]["alpha"][key], row[key], key)
        self.assertEqual(strict["models"][0]["task_hardness"]["alpha/t3"], "ceiling")
        self.assertEqual(strict["by_skill"]["alpha"]["ceiling_tasks"], 3)

    def test_unclassified_and_per_task_thresholds(self) -> None:
        spec: Spec = {
            "alpha/t1": {"with": [4, 4], "without": [4, None]},  # one valid control rep
            "alpha/t2": {"with": [4, 4], "without": [3, 3]},
            "alpha/t3": {"with": [4, 4], "without": [3, 3]},
        }
        root = self.matrix({"exec-a": (spec, {})})
        rep = SR.build_report(root, draws=50, min_baseline_fail={"alpha/t2": 2})
        row = rep["by_skill"]["alpha"]
        self.assertEqual(row["tasks"]["alpha/t1"]["hardness"], "unclassified")
        self.assertEqual(row["tasks"]["alpha/t2"]["hardness"], "ceiling")
        self.assertEqual(row["tasks"]["alpha/t2"]["min_baseline_fail"], 2)
        self.assertEqual(row["tasks"]["alpha/t3"]["hardness"], "informative")
        self.assertEqual(
            (row["ceiling_tasks"], row["informative_tasks"], row["unclassified_tasks"]),
            (1, 1, 1),
        )
        self.assertFalse(row["needs_harder_tasks"])
        # Thresholds default to the corpus; a corpus that cannot be read means 1 everywhere.
        with mock.patch.object(SR, "corpus_thresholds", return_value={"alpha/t3": 2}) as ct:
            rep2 = SR.build_report(root, draws=50)
        ct.assert_called_once()
        self.assertEqual(rep2["by_skill"]["alpha"]["tasks"]["alpha/t3"]["hardness"], "ceiling")
        with self.assertRaises(ValueError):
            SR.build_report(root, draws=50, min_baseline_fail=0)


class TestBuildHardness(_Case):
    def test_control_cells_alone_are_enough(self) -> None:
        root = self.runs_dir / "run-1" / "results"
        seed_namespace(root / "exec-a" / "without", "without", TWO_CEILING)
        seed_namespace(
            root / "exec-c" / "without", "without", INFORMATIVE, executor="exec-c", judge="judge-d"
        )
        h = SR.build_hardness(root)
        self.assertEqual([m["executor"] for m in h["models"]], ["exec-a", "exec-c"])
        self.assertEqual([m["control_arm"] for m in h["models"]], ["without", "without"])
        a, c = h["models"]
        self.assertEqual(a["tasks"]["alpha/t1"]["hardness"], "ceiling")
        self.assertEqual(a["tasks"]["alpha/t3"]["baseline_missed"], [2, 1])
        self.assertEqual(a["skills"]["alpha"]["ceiling_tasks"], 2)
        self.assertTrue(a["skills"]["alpha"]["needs_harder_tasks"])
        self.assertEqual(c["skills"]["alpha"]["ceiling_tasks"], 0)
        self.assertEqual(h["suite"], {"name": "suite-x", "hash": "suite-hash-1"})
        self.assertEqual(h["manifest_hash"], "manifest-1")
        self.assertEqual(h["min_reps"], 2)
        # build_report still demands the pair.
        with self.assertRaisesRegex(ValueError, "no 'with' cell"):
            SR.build_report(root, draws=50)
        # A treatment cell alone cannot judge hardness.
        seed_namespace(root / "exec-e" / "with", "with", TWO_CEILING, executor="exec-e")
        with self.assertRaisesRegex(ValueError, "no control arm cell"):
            SR.build_hardness([SR.ReportCell("exec-e", None, "with", root / "exec-e" / "with")])
        with self.assertRaisesRegex(ValueError, "at least one"):
            SR.build_hardness([])

    def test_pairs_are_accepted_and_match_the_report(self) -> None:
        root = self.matrix(
            {"exec-a": (TWO_CEILING, {}), "exec-c": (INFORMATIVE, {"judge": "judge-d"})}
        )
        h = SR.build_hardness(root)
        rep = SR.build_report(root, draws=50)
        self.assertEqual(SR.hardness_of_report(rep)["models"], h["models"])
        self.assertEqual(
            {tid: t["hardness"] for tid, t in h["models"][0]["tasks"].items()},
            rep["models"][0]["task_hardness"],
        )
        # Explicit thresholds flow through as well.
        strict = SR.build_hardness(root, min_baseline_fail=3)
        self.assertEqual(strict["models"][1]["skills"]["alpha"]["ceiling_tasks"], 3)

    def test_render_hardness_markdown(self) -> None:
        root = self.matrix(
            {"exec-a": (TWO_CEILING, {}), "exec-c": (INFORMATIVE, {"judge": "judge-d"})}
        )
        md = SR.render_hardness_markdown(SR.build_hardness(root))
        self.assertEqual(md.splitlines()[0], "# skillbench task hardness")
        self.assertIn("- **Suite**: `suite-x`", md)
        self.assertIn("`exec-a` (control arm `without`, judge `judge-b`)", md)
        self.assertIn("| skill | task | min fail | exec-a | exec-c |", md)
        self.assertIn(
            "| alpha | t1 | 1 | ceiling (missed 0, 0) | informative (missed 1, 0) |", md
        )
        self.assertIn(
            "| alpha | t3 | 1 | informative (missed 2, 1) | informative (missed 2, 1) |", md
        )
        self.assertIn("| skill | exec-a | exec-c |", md)
        self.assertIn("| alpha | 2/3 — needs harder tasks | 0/3 |", md)
        self.assertIn("flagged, not dropped", md)
        # Unclassified cells name the rep count; a task missing for one executor renders "—".
        spec: Spec = {
            "alpha/t1": {"with": [4], "without": [4]},
            "alpha/t9": {"with": [4, 4], "without": [3, 3]},
        }
        other = self.runs_dir / "run-2" / "results"
        seed_namespace(other / "exec-a" / "without", "without", spec)
        seed_namespace(other / "exec-c" / "without", "without", TWO_CEILING, executor="exec-c")
        md2 = SR.render_hardness_markdown(SR.build_hardness(other))
        self.assertIn(
            "| alpha | t1 | 1 | unclassified (1 valid rep) | ceiling (missed 0, 0) |", md2
        )
        self.assertIn("| alpha | t9 | 1 | informative (missed 1, 1) | — |", md2)
        self.assertIn("| alpha | 0/2 (1 unclassified) | 2/3 — needs harder tasks |", md2)
        empty = SR.render_hardness_markdown({"models": []})
        self.assertIn("- **Executors**: none", empty)


class TestCliAndRenderers(_Case):
    def test_hardness_only_prints_the_table_and_writes_nothing(self) -> None:
        root = self.runs_dir / "run-1" / "results"
        seed_namespace(root / "exec-a" / "without", "without", TWO_CEILING)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = SR.main(["--results-root", str(root), "--hardness-only"])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("# skillbench task hardness", text)
        self.assertIn("| alpha | t1 | 1 | ceiling (missed 0, 0) |", text)
        self.assertIn("| alpha | 2/3 — needs harder tasks |", text)
        self.assertFalse((root.parent / "report.json").exists())
        self.assertFalse((root.parent / "report.md").exists())
        # Without the flag the same root is still an error (no WITH cell).
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(SR.main(["--results-root", str(root)]), 1)
        self.assertIn("no 'with' cell", err.getvalue())

    def test_hardness_only_with_explicit_pair_ignores_the_with_dir(self) -> None:
        pair = self.tmp / "pair"
        seed_namespace(pair / "without", "without", INFORMATIVE)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = SR.main(
                [
                    "--with-dir", str(pair / "with"),  # never swept
                    "--without-dir", str(pair / "without"),
                    "--hardness-only",
                ]
            )
        self.assertEqual(rc, 0)
        self.assertIn("| alpha | 0/3 |", out.getvalue())
        self.assertIn("`exec-a` (control arm `without`", out.getvalue())
        self.assertIn("--hardness-only", SR._build_parser().format_help())

    def test_markdown_report_has_the_ceiling_column_and_note(self) -> None:
        root = self.matrix({"exec-a": (TWO_CEILING, {})})
        md = SR.render_markdown(SR.build_report(root, draws=50))
        self.assertIn("| invalid | ceiling/total |", md)
        self.assertIn("| 2/3 — needs harder tasks |", md)
        self.assertIn("- **Hardness**: Hardness is judged per executor", md)
        v1_row = {"skills": [{"skill": "s", "n_tasks": 1}], "k": 1}
        self.assertIn("| s | 1 | n/a | n/a | n/a | insufficient data |", SR.render_markdown(v1_row))
        self.assertIn("| n/a |\n", SR.render_markdown(v1_row))

    def test_html_shows_badges_and_legend(self) -> None:
        root = self.matrix({"exec-a": (TWO_CEILING, {})})
        rep = SR.build_report(root, draws=50)
        page = SR.render_html(rep)

        class Collector(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.pills: list[tuple[str, str]] = []
                self.text: list[str] = []
                self._pill = False

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                d = dict(attrs)
                if tag == "span" and "rounded-full" in (d.get("class") or ""):
                    self._pill = True
                    self._title = d.get("title") or ""

            def handle_data(self, data: str) -> None:
                if self._pill:
                    self.pills.append((data.strip(), self._title))
                    self._pill = False
                if data.strip():
                    self.text.append(data.strip())

        parser = Collector()
        parser.feed(page)
        labels = [p[0] for p in parser.pills]
        self.assertEqual(labels.count("ceiling"), 2)
        self.assertEqual(labels.count("informative"), 1)
        self.assertEqual(labels.count("needs harder tasks"), 1)
        self.assertIn(
            ("ceiling", "WITHOUT reps missed 0, 0 criteria; need >= 1 in some rep"), parser.pills
        )
        text = " ".join(parser.text)
        self.assertIn("2/3", text)
        self.assertIn("Ceiling tasks are flagged, never dropped", text)
        self.assertIn("hardness", text)
        # v1 dictionaries render n/a instead of badges.
        v1 = {"skills": [{"skill": "s", "n_tasks": 1, "tasks": {"s/t": {"delta": None}}}], "k": 1}
        page_v1 = SR.render_html(v1)
        self.assertNotIn("needs harder tasks</span>", page_v1)
        self.assertIn("n/a", page_v1)

    def test_report_json_round_trips_the_hardness_fields(self) -> None:
        root = self.matrix({"exec-a": (TWO_CEILING, {})})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = SR.main(["--results-root", str(root), "--draws", "50", "--k", "2"])
        self.assertEqual(rc, 0)
        data = json.loads((root.parent / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(data["models"][0]["task_hardness"]["alpha/t1"], "ceiling")
        self.assertTrue(data["by_skill"]["alpha"]["needs_harder_tasks"])
        self.assertEqual(data["hardness_note"], SR.HARDNESS_NOTE)
        self.assertIn("ceiling/total", (root.parent / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
