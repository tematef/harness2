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

"""Schema-v2 tests for `harness2.cli.skillbench_report`: `ReportCell` input, v1 parity against
the stored fixture, one `models[]` entry per executor, identity errors, auto-discovery of
`<executor>/<arm-label>` namespaces and the multi-executor renderers.

Synthetic namespaces live in temporary directories with `run_dir` entries relative to the
namespace (the layout of `tests/fixtures/skillbench/report_v1_fixture/`).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from unittest import mock

from harness2 import config
from harness2.cli import skillbench_report as SR

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "skillbench" / "report_v1_fixture"

# task -> arm -> list of n_passed (None = invalid rollout); 4 criteria per rubric
Spec = dict[str, dict[str, list[int | None]]]

THREE_TASKS: Spec = {
    "alpha/t1": {"with": [4, 3], "without": [2, 1]},
    "alpha/t2": {"with": [3, 4], "without": [2, 2]},
    "alpha/t3": {"with": [4, 4], "without": [3, 1]},
}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def seed_namespace(
    ns_dir: Path,
    arm: str,
    spec: Spec,
    *,
    executor: str | None = "exec-a",
    judge: str | None = "judge-b",
    manifest_hash: str | None = "manifest-1",
    suite: str | None = "suite-x",
    suite_hash: str | None = "suite-hash-1",
    run_json: bool = True,
    config_json: bool = True,
    seed: int | None = 3,
    shift: int = 0,
) -> Path:
    """Write a judged arm namespace whose `run_dir`s are relative to `ns_dir`. `shift` lowers
    every WITH score by that many criteria (to make executors differ)."""
    ns_dir.mkdir(parents=True, exist_ok=True)
    if config_json:
        _write_json(
            ns_dir / "config.json",
            {
                "method_version": 1,
                "executor": executor,
                "judge": judge,
                "effort": "high",
                "manifest_hash": manifest_hash,
                "seed": seed,
                "tag": arm,
            },
        )
    if run_json:
        _write_json(
            ns_dir / "run.json",
            {
                "benchmark": "skillbench",
                "arm": arm,
                "suite": suite,
                "suite_hash": suite_hash,
                "manifest_hash": manifest_hash,
                "executor": executor,
                "judge": judge,
                "effort": "high",
            },
        )
    judged: list[dict[str, Any]] = []
    for task_id, arms in sorted(spec.items()):
        task_dir = ns_dir / task_id.replace("/", "__")
        entries = []
        for idx, n_passed in enumerate(arms[arm], 1):
            label = f"base{idx}"
            run_dir = task_dir / label
            run_dir.mkdir(parents=True, exist_ok=True)
            valid = n_passed is not None
            if valid and arm == "with":
                n_passed = max(0, n_passed - shift)
            _write_json(
                run_dir / "skill_timing.json",
                {
                    "invoked": arm == "with",
                    "t_skill_s": 2.0 if arm == "with" else None,
                    "step_idx": 1 if arm == "with" else None,
                    "t_total_s": 20.0,
                    "requested_model": executor or "env-exec",
                    "actual_model": executor or "env-exec",
                    "valid": valid,
                    "invalid_reason": None if valid else "exit 1",
                    "exit_code": 0 if valid else 1,
                    "total_tokens": 1000,
                },
            )
            score = {"n_passed": n_passed, "n_criteria": 4} if valid else None
            if valid:
                _write_json(
                    run_dir / "judgments.json",
                    {
                        "judge_model": judge or "env-judge",
                        "actual_model": judge or "env-judge",
                        "score": score,
                        "judgments": [
                            {
                                "criteria": [
                                    {"id": f"c{i}", "pass": i < n_passed, "reason": "r"}
                                    for i in range(4)
                                ],
                                "n_passed": n_passed,
                                "n_criteria": 4,
                            }
                        ],
                    },
                )
            run_id = f"{ns_dir.name}/{task_id}/{label}"
            entries.append(
                {
                    "label": label,
                    "run_id": run_id,
                    "run_dir": f"{task_dir.name}/{label}",
                    "ok": valid,
                    "delivered": valid,
                }
            )
            judged.append({"task": task_id, "label": label, "run_id": run_id, "score": score})
        _write_json(
            task_dir / "summary.json",
            {
                "task_id": task_id,
                "domain": task_id.split("/", 1)[0],
                "k": 2,
                "seed": seed,
                "tag": arm,
                "status": "bases-only",
                "base": entries,
            },
        )
    _write_json(ns_dir / "base_judged.json", judged)
    return ns_dir


class _V2Case(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-report-v2-"))
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

    def _pair(
        self, name: str, spec: Spec = THREE_TASKS, **kw: Any
    ) -> tuple[Path, Path]:
        root = self.tmp / name
        return (
            seed_namespace(root / "with", "with", spec, **kw),
            seed_namespace(root / "without", "without", spec, **kw),
        )

    def _matrix(self, executors: dict[str, dict[str, Any]], root: Path | None = None) -> Path:
        """`results/<executor>/<arm>` namespaces; each value is extra kwargs for the executor."""
        root = root or (self.runs_dir / "run-1" / "results")
        for executor, kw in executors.items():
            kw = {"executor": executor, **kw}
            seed_namespace(root / executor / "with", "with", THREE_TASKS, **kw)
            seed_namespace(root / executor / "without", "without", THREE_TASKS, **kw)
        return root


class TestV1Parity(_V2Case):
    def test_stored_fixture_reproduces_v1_numbers(self) -> None:
        expected = json.loads((FIXTURE / "expected_v1.json").read_text(encoding="utf-8"))
        args = expected["build_args"]
        rep = SR.build_report(FIXTURE / "with", FIXTURE / "without", **args)

        self.assertEqual(rep["schema_version"], 2)
        self.assertEqual(rep["suite"], {"name": "fixture-suite", "hash": "fixture-suite-hash"})
        self.assertEqual(len(rep["models"]), 1)
        for key, value in expected["top"].items():
            self.assertEqual(rep[key], value, key)
        for skill, row in expected["by_skill"].items():
            got = rep["by_skill"][skill]
            for key, value in row.items():
                self.assertEqual(got[key], value, f"{skill}.{key}")
        for skill, tasks in expected["task_means"].items():
            for tid, (w, u, d) in tasks.items():
                t = rep["by_skill"][skill]["tasks"][tid]
                self.assertEqual([t["with_mean"], t["without_mean"], t["delta"]], [w, u, d])

    def test_top_level_mirrors_the_single_model_block(self) -> None:
        rep = SR.build_report(FIXTURE / "with", FIXTURE / "without", draws=50)
        model = rep["models"][0]
        self.assertEqual(model["executor"], "exec-a")
        self.assertEqual(model["judge"], "judge-b")
        self.assertEqual(model["arms"]["with"], str(FIXTURE / "with"))
        for key in SR._LEGACY_MODEL_KEYS:
            self.assertEqual(rep[key], model[key], key)
        self.assertEqual(rep["k"], model["k"])
        self.assertEqual(rep["seed"], 3)
        # The legacy renderers read the mirrored keys and the single-model wording.
        md = SR.render_markdown(rep)
        self.assertIn("- **Executor model**: requested `exec-a`, actual `exec-a`", md)
        self.assertIn("- **Suite**: `fixture-suite`", md)
        self.assertNotIn("## executor", md)
        self.assertEqual(len(SR.models_of(rep)), 1)
        self.assertEqual(SR.models_of({"skills": []}), [{"skills": []}])

    def test_cells_from_dirs_and_explicit_cells_agree(self) -> None:
        cells = SR.cells_from_dirs(FIXTURE / "with", FIXTURE / "without")
        self.assertEqual(
            [(c.executor, c.judge, c.arm_label) for c in cells],
            [("exec-a", "judge-b", "with"), ("exec-a", "judge-b", "without")],
        )
        via_cells = SR.build_report(cells, draws=50, bootstrap_seed=1)
        via_dirs = SR.build_report(FIXTURE / "with", FIXTURE / "without", draws=50, bootstrap_seed=1)
        self.assertEqual(via_cells["by_skill"], via_dirs["by_skill"])
        explicit = [
            SR.ReportCell(None, None, "with", FIXTURE / "with"),
            SR.ReportCell(None, None, "without", FIXTURE / "without"),
        ]
        self.assertEqual(SR.build_report(explicit, draws=50)["models"][0]["executor"], "exec-a")

    def test_executor_and_judge_fall_back_to_rollouts_then_env(self) -> None:
        with_ns, without_ns = self._pair(
            "bare", executor=None, judge=None, run_json=False, config_json=False
        )
        rep = SR.build_report(with_ns, without_ns, draws=50)
        self.assertEqual(rep["models"][0]["executor"], "env-exec")
        self.assertEqual(rep["models"][0]["judge"], "env-judge")
        self.assertEqual(rep["suite"], {"name": None, "hash": None})
        # No manifest hash recorded: the bundled corpus manifest is checked instead.
        self.assertTrue(rep["manifest_hash"])
        self.assertNotIn(rep["manifest_hash"], ("", None))


class TestMultiExecutor(_V2Case):
    def test_two_executors_produce_two_models_entries(self) -> None:
        root = self._matrix({"exec-a": {}, "exec-c": {"judge": "judge-d", "shift": 2}})
        cells = [
            SR.ReportCell(ex, j, arm, root / ex / arm)
            for ex, j in (("exec-a", "judge-b"), ("exec-c", "judge-d"))
            for arm in ("with", "without")
        ]
        rep = SR.build_report(cells, draws=200, bootstrap_seed=5)
        self.assertEqual([m["executor"] for m in rep["models"]], ["exec-a", "exec-c"])
        self.assertEqual([m["judge"] for m in rep["models"]], ["judge-b", "judge-d"])
        a, c = rep["models"]
        self.assertAlmostEqual(a["by_skill"]["alpha"]["lift"], 0.4583333333333333)
        self.assertAlmostEqual(c["by_skill"]["alpha"]["lift"], 0.4583333333333333 - 0.5)
        self.assertEqual(a["by_skill"]["alpha"]["verdict"], "helps")
        self.assertEqual(c["by_skill"]["alpha"]["verdict"], "inconclusive")
        # The legacy top-level view is the first executor.
        self.assertEqual(rep["requested_exec_model"], "exec-a")
        self.assertEqual(rep["by_skill"], a["by_skill"])
        self.assertEqual(rep["suite"], {"name": "suite-x", "hash": "suite-hash-1"})
        self.assertEqual(rep["k"], 2)
        # Explicit cells need not carry executor / judge when run.json does.
        bare = [SR.ReportCell(None, None, c.arm_label, c.ns_dir) for c in cells]
        self.assertEqual(
            [m["executor"] for m in SR.build_report(bare, draws=50)["models"]],
            ["exec-a", "exec-c"],
        )

    def test_mixed_judges_for_one_executor_is_an_error(self) -> None:
        with_ns, without_ns = self._pair("mixed")
        cells = [
            SR.ReportCell("exec-a", "judge-b", "with", with_ns),
            SR.ReportCell("exec-a", "judge-z", "without", without_ns),
        ]
        with self.assertRaisesRegex(ValueError, "judge"):
            SR.build_report(cells, draws=50)
        # Also when the disagreement is recorded in run.json rather than on the cell.
        seed_namespace(self.tmp / "mixed2" / "with", "with", THREE_TASKS)
        seed_namespace(self.tmp / "mixed2" / "without", "without", THREE_TASKS, judge="judge-z")
        with self.assertRaisesRegex(ValueError, "different judges"):
            SR.build_report(self.tmp / "mixed2" / "with", self.tmp / "mixed2" / "without", draws=50)

    def test_cell_and_namespace_identity_errors(self) -> None:
        with_ns, without_ns = self._pair("ident")
        cases = {
            "unknown arm label": [
                SR.ReportCell("exec-a", "judge-b", "treatment", with_ns),
                SR.ReportCell("exec-a", "judge-b", "without", without_ns),
            ],
            "no 'without' cell": [SR.ReportCell("exec-a", "judge-b", "with", with_ns)],
            "no 'with' cell": [SR.ReportCell("exec-a", "judge-b", "without", without_ns)],
            "two 'with' cells": [
                SR.ReportCell("exec-a", "judge-b", "with", with_ns),
                SR.ReportCell("exec-a", "judge-b", "with", with_ns),
                SR.ReportCell("exec-a", "judge-b", "without", without_ns),
            ],
            "cell executor 'exec-q'": [
                SR.ReportCell("exec-q", "judge-b", "with", with_ns),
                SR.ReportCell("exec-q", "judge-b", "without", without_ns),
            ],
            "cell judge 'judge-q'": [
                SR.ReportCell("exec-a", "judge-q", "with", with_ns),
                SR.ReportCell("exec-a", "judge-b", "without", without_ns),
            ],
            "at least one": [],
        }
        for pattern, cells in cases.items():
            with self.subTest(pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    SR.build_report(cells, draws=50)
        with self.assertRaisesRegex(ValueError, "either a list"):
            SR.build_report([SR.ReportCell(None, None, "with", with_ns)], without_ns)

    def test_cells_must_share_manifest_and_suite_hash(self) -> None:
        root = self._matrix(
            {"exec-a": {}, "exec-c": {"judge": "judge-d", "manifest_hash": "manifest-2"}}
        )
        with self.assertRaisesRegex(ValueError, "manifest_hash"):
            SR.build_report(root, draws=50)
        root2 = self._matrix(
            {"exec-a": {}, "exec-c": {"judge": "judge-d", "suite_hash": "suite-hash-2"}},
            root=self.runs_dir / "run-2" / "results",
        )
        with self.assertRaisesRegex(ValueError, "suite_hash"):
            SR.build_report(root2, draws=50)
        # Within one executor the two arms must agree too (run.json / config.json identity).
        seed_namespace(self.tmp / "drift" / "with", "with", THREE_TASKS)
        seed_namespace(
            self.tmp / "drift" / "without", "without", THREE_TASKS, suite_hash="suite-hash-9"
        )
        with self.assertRaisesRegex(ValueError, "suite_hash"):
            SR.build_report(self.tmp / "drift" / "with", self.tmp / "drift" / "without", draws=50)


class TestAutoDiscovery(_V2Case):
    def test_discover_cells_finds_judged_namespaces_only(self) -> None:
        root = self._matrix({"exec-a": {}, "exec-c": {"judge": "judge-d"}})
        (root / "exec-e" / "with").mkdir(parents=True)  # swept but never judged
        (root / "notes.txt").write_text("ignored\n", encoding="utf-8")
        cells = SR.discover_cells(root)
        self.assertEqual(
            [(c.executor, c.judge, c.arm_label, c.ns_dir.name) for c in cells],
            [
                ("exec-a", "judge-b", "with", "with"),
                ("exec-a", "judge-b", "without", "without"),
                ("exec-c", "judge-d", "with", "with"),
                ("exec-c", "judge-d", "without", "without"),
            ],
        )
        rep = SR.build_report(root, draws=50)
        self.assertEqual([m["executor"] for m in rep["models"]], ["exec-a", "exec-c"])
        # The directory name wins over a missing run.json; a disagreeing one is an error.
        with self.assertRaisesRegex(ValueError, "cell executor 'exec-c'"):
            SR.build_report(
                [
                    SR.ReportCell("exec-c", None, "with", root / "exec-a" / "with"),
                    SR.ReportCell("exec-c", None, "without", root / "exec-a" / "without"),
                ],
                draws=50,
            )

    def test_discover_cells_errors(self) -> None:
        with self.assertRaises(FileNotFoundError):
            SR.discover_cells(self.tmp / "nowhere")
        empty = self.tmp / "empty"
        (empty / "exec-a" / "with").mkdir(parents=True)
        with self.assertRaisesRegex(FileNotFoundError, "no judged namespaces"):
            SR.discover_cells(empty)
        with self.assertRaisesRegex(FileNotFoundError, "no judged namespaces"):
            SR.build_report(empty, draws=50)

    def test_cli_results_root_writes_next_to_results(self) -> None:
        root = self._matrix({"exec-a": {}, "exec-c": {"judge": "judge-d", "shift": 1}})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = SR.main(["--results-root", str(root), "--draws", "50", "--k", "2"])
        self.assertEqual(rc, 0)
        run_dir = root.parent
        data = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual([m["executor"] for m in data["models"]], ["exec-a", "exec-c"])
        self.assertEqual(data["k"], 2)
        md = (run_dir / "report.md").read_text(encoding="utf-8")
        self.assertIn("## executor `exec-a`", md)
        self.assertIn("## executor `exec-c`", md)
        self.assertIn("- **Executors**: `exec-a` (judge `judge-b`), `exec-c` (judge `judge-d`)", md)
        self.assertTrue((run_dir / "report.html").is_file())
        self.assertIn(str(run_dir / "report.md"), out.getvalue())

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = SR.main(
                ["--results-root", str(root), "--with-dir", str(root / "exec-a" / "with")]
            )
        self.assertEqual(rc, 1)
        self.assertIn("cannot be combined", err.getvalue())

        err2 = io.StringIO()
        with contextlib.redirect_stderr(err2):
            rc = SR.main(["--results-root", str(self.tmp / "missing")])
        self.assertEqual(rc, 1)
        self.assertIn("does not exist", err2.getvalue())

    def test_cli_with_dir_pair_still_works(self) -> None:
        with_ns, without_ns = self._pair("pair")
        out_dir = self.tmp / "out"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = SR.main(
                [
                    "--with-dir", str(with_ns), "--without-dir", str(without_ns),
                    "--out-dir", str(out_dir), "--draws", "50", "--k", "2",
                ]
            )
        self.assertEqual(rc, 0)
        data = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(data["models"][0]["executor"], "exec-a")
        self.assertEqual(data["by_skill"]["alpha"]["verdict"], "helps")


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.optgroups: list[str] = []
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = dict(attrs)
        if tag == "optgroup" and d.get("label"):
            self.optgroups.append(str(d["label"]))
        if d.get("id"):
            self.ids.append(str(d["id"]))

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.text.append(data.strip())


class TestRenderers(_V2Case):
    def test_html_repeats_tabs_per_executor_with_unique_panel_ids(self) -> None:
        root = self._matrix({"exec-a": {}, "exec-c": {"judge": "judge-d"}})
        rep = SR.build_report(root, draws=50)
        page = SR.render_html(rep)
        parser = _TextCollector()
        parser.feed(page)
        text = " ".join(parser.text)
        self.assertIn("2 executors: exec-a (judge judge-b), exec-c (judge judge-d)", text)
        self.assertIn("suite suite-x", text)
        self.assertEqual(text.count("executor exec-a"), 4)  # KPI, skills, tasks, diagnostics
        self.assertEqual(text.count("executor exec-c"), 4)
        self.assertEqual(parser.optgroups, ["exec-a", "exec-c"])
        sbs_ids = [i for i in parser.ids if i.startswith("sbs-") and i != "sbs-select"]
        self.assertEqual(len(sbs_ids), 6)
        self.assertEqual(len(set(sbs_ids)), 6)
        self.assertEqual(page.count("<script"), 2)

    def test_html_single_model_and_v1_dict_keep_the_old_shape(self) -> None:
        rep = SR.build_report(FIXTURE / "with", FIXTURE / "without", draws=50)
        page = SR.render_html(rep)
        self.assertIn("executor exec-a · judge judge-b", page)
        self.assertIn("suite fixture-suite", page)
        self.assertNotIn("<optgroup", page)
        self.assertNotIn("2 executors", page)
        v1 = {k: v for k, v in rep.items() if k not in ("models", "schema_version", "suite")}
        page_v1 = SR.render_html(v1)
        self.assertIn("alpha-skill/t1", page_v1)
        self.assertIn("requested exec-a, actual exec-a", page_v1)
        self.assertNotIn("suite ", page_v1.split("<nav")[0].split("<header>")[1])
        empty = SR.render_html({"skills": [], "k": 1})
        self.assertIn("No task data.", empty)

    def test_markdown_single_model_matches_v1_layout(self) -> None:
        rep = SR.build_report(FIXTURE / "with", FIXTURE / "without", draws=50)
        md = SR.render_markdown(rep)
        lines = md.splitlines()
        self.assertEqual(lines[0], "# skillbench report")
        self.assertIn("- **Judge-noise rate**: n/a (single judgment per rollout)", lines)
        self.assertEqual(md.count("| skill | tasks | WITH | WITHOUT |"), 1)
        self.assertIn("| alpha-skill | 3 | 0.92 | 0.46 | +0.46 [", md)
        v1 = {k: v for k, v in rep.items() if k not in ("models", "schema_version", "suite")}
        md_v1 = SR.render_markdown(v1)
        self.assertNotIn("**Suite**", md_v1)
        self.assertIn("| alpha-skill | 3 | 0.92 | 0.46 |", md_v1)


if __name__ == "__main__":
    unittest.main()
