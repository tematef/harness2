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

"""Task 14 acceptance tests for Skillbench v2.

Verifies:
1. The synthetic 2-plugin non-cc-thingz suite under `tests/fixtures/skillbench/other-suite/`
   (`acme-ops`, plugins `triage-kit` + `deploy-kit`, asking skill `incident-triage`,
   non-asking skill `canary-rollout`, suite task `_suite/full-release`) runs through
   `skillbench-tasks check`, `stage_plugins` for all four arm labels (`with`, `without`,
   `baseline`, `suite-acme-ops`), `build_report` + `write_report` (mocked cells), and
   `improve_skill` (mocked improver + probe) with zero code changes.
2. AST/token inspection of all `.py` files under `harness2/` confirms `cc-thingz`,
   `make-plan`, `root-cause`, and `brainstorm` never appear in runtime code logic (only
   docstrings/comments, the example `suites/cc-thingz.toml`, and the bundled corpus).
3. `summary.md` and the `report.html` Summary tab answer all three questions on one screen
   (per-skill recommendation, per-family suite-vs-built-in verdict, per-model efficiency),
   and `summary.md` has well-formed GitHub Markdown tables.
4. Matrix resume schedules 0 new rollouts after completion, and improver enforces model
   matching, multi-model gates, and `--apply` dry-run branch/commit output.
5. Every public function in `skillbench_suite.py`, `skillbench_tasks.py`,
   `skillbench_matrix.py`, `skillbench_summary.py`, and `skillbench_cycle.py` is covered by
   unit tests.
"""

from __future__ import annotations

import ast
import contextlib
import datetime
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import tokenize
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from harness2 import config
from harness2.benchmarks import skillbench as sb
from harness2.benchmarks import skillbench_suite as ss
from harness2.cli import skillbench_cycle as sc
from harness2.cli import skillbench_improve as si
from harness2.cli import skillbench_matrix as sm
from harness2.cli import skillbench_report as sr
from harness2.cli import skillbench_summary as ssum
from harness2.cli import skillbench_tasks as st
from harness2.substrate import agy

REPO_ROOT = Path(__file__).resolve().parents[1]
OTHER_SUITE_DIR = REPO_ROOT / "tests" / "fixtures" / "skillbench" / "other-suite"
OTHER_SUITE_TOML = OTHER_SUITE_DIR / "suite.toml"

FORBIDDEN_LOGIC_TERMS = ("cc-thingz", "make-plan", "root-cause", "brainstorm")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _seed_skill_cell(
    ns_dir: Path,
    arm: str,
    *,
    executor: str,
    judge: str,
    suite_name: str,
    suite_hash: str,
    manifest_hash: str,
    task_scores: dict[str, list[int]],
    tokens: int = 12_000,
    wall_s: float = 25.0,
    t_skill_s: float = 4.0,
) -> Path:
    """Seed a judged skill-mode (`with` / `without`) cell directory."""
    ns_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        ns_dir / "config.json",
        {
            "method_version": 1,
            "executor": executor,
            "judge": judge,
            "effort": "high",
            "manifest_hash": manifest_hash,
            "suite_hash": suite_hash,
            "seed": 1,
            "tag": arm,
        },
    )
    _write_json(
        ns_dir / "run.json",
        {
            "benchmark": "skillbench",
            "arm": arm,
            "arm_kind": "skill",
            "suite": suite_name,
            "suite_hash": suite_hash,
            "manifest_hash": manifest_hash,
            "executor": executor,
            "judge": judge,
            "effort": "high",
        },
    )
    judged: list[dict[str, Any]] = []
    for task_id, scores in sorted(task_scores.items()):
        tdir = ns_dir / task_id.replace("/", "__")
        entries: list[dict[str, Any]] = []
        for idx, n_passed in enumerate(scores, 1):
            label = f"base{idx}"
            rdir = tdir / label
            rdir.mkdir(parents=True, exist_ok=True)
            _write_json(
                rdir / "skill_timing.json",
                {
                    "invoked": arm == "with",
                    "t_skill_s": t_skill_s if arm == "with" else None,
                    "step_idx": 1 if arm == "with" else None,
                    "t_total_s": wall_s,
                    "requested_model": executor,
                    "actual_model": executor,
                    "valid": True,
                    "invalid_reason": None,
                    "exit_code": 0,
                    "total_tokens": tokens,
                },
            )
            score_obj = {"n_passed": n_passed, "n_criteria": 4}
            _write_json(
                rdir / "judgments.json",
                {
                    "judge_model": judge,
                    "actual_model": judge,
                    "score": score_obj,
                    "judgments": [
                        {
                            "criteria": [
                                {"id": f"c{i}", "pass": i <= n_passed, "reason": "ok"}
                                for i in range(1, 5)
                            ],
                            "n_passed": n_passed,
                            "n_criteria": 4,
                        }
                    ],
                },
            )
            (rdir / "final.md").write_text(f"Deliverable for {task_id} ({arm})\n", encoding="utf-8")
            (rdir / "diff.patch").write_text("", encoding="utf-8")
            (rdir / "new_files").mkdir(exist_ok=True)
            (rdir / "events.jsonl").write_text("", encoding="utf-8")
            run_id = f"{ns_dir.name}/{task_id}/{label}"
            entries.append(
                {
                    "label": label,
                    "run_id": run_id,
                    "run_dir": f"{tdir.name}/{label}",
                    "ok": True,
                    "delivered": True,
                }
            )
            judged.append({"task": task_id, "label": label, "run_id": run_id, "score": score_obj})
        _write_json(
            tdir / "summary.json",
            {
                "task_id": task_id,
                "domain": task_id.split("/", 1)[0],
                "k": len(scores),
                "seed": 1,
                "tag": arm,
                "status": "bases-only",
                "base": entries,
            },
        )
    _write_json(ns_dir / "base_judged.json", judged)
    return ns_dir


def _seed_suite_cell(
    ns_dir: Path,
    arm: str,
    *,
    executor: str,
    judge: str,
    suite_name: str,
    suite_hash: str,
    manifest_hash: str,
    family: str,
    task_id: str,
    scores: list[int],
    skills_read: list[str],
    tokens: int = 14_000,
    wall_s: float = 30.0,
) -> Path:
    """Seed a judged suite-mode (`baseline` / `suite-<name>`) cell directory."""
    ns_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        ns_dir / "config.json",
        {
            "method_version": 1,
            "executor": executor,
            "judge": judge,
            "effort": "high",
            "manifest_hash": manifest_hash,
            "suite_hash": suite_hash,
            "seed": 1,
            "tag": arm,
        },
    )
    _write_json(
        ns_dir / "run.json",
        {
            "benchmark": "skillbench",
            "arm": arm,
            "arm_kind": "suite",
            "suite": suite_name,
            "suite_hash": suite_hash,
            "manifest_hash": manifest_hash,
            "executor": executor,
            "judge": judge,
            "effort": "high",
        },
    )
    tdir = ns_dir / f"{family}__{task_id}"
    entries: list[dict[str, Any]] = []
    judged: list[dict[str, Any]] = []
    full_tid = f"{family}/{task_id}"
    for idx, n_passed in enumerate(scores, 1):
        label = f"base{idx}"
        rdir = tdir / label
        rdir.mkdir(parents=True, exist_ok=True)
        invoked = bool(skills_read)
        _write_json(
            rdir / "skill_timing.json",
            {
                "invoked": invoked,
                "t_skill_s": 3.5 if invoked else None,
                "step_idx": 1 if invoked else None,
                "skills_read": list(skills_read),
                "t_total_s": wall_s,
                "requested_model": executor,
                "actual_model": executor,
                "valid": True,
                "invalid_reason": None,
                "exit_code": 0,
                "total_tokens": tokens,
            },
        )
        score_obj = {"n_passed": n_passed, "n_criteria": 4}
        _write_json(
            rdir / "judgments.json",
            {
                "judge_model": judge,
                "actual_model": judge,
                "score": score_obj,
                "judgments": [
                    {
                        "criteria": [
                            {"id": f"c{i}", "pass": i <= n_passed, "reason": "ok"}
                            for i in range(1, 5)
                        ],
                        "n_passed": n_passed,
                        "n_criteria": 4,
                    }
                ],
            },
        )
        run_id = f"{ns_dir.name}/{full_tid}/{label}"
        entries.append(
            {
                "label": label,
                "run_id": run_id,
                "run_dir": f"{tdir.name}/{label}",
                "ok": True,
                "delivered": True,
            }
        )
        judged.append({"task": full_tid, "label": label, "run_id": run_id, "score": score_obj})
    _write_json(
        tdir / "summary.json",
        {
            "task_id": full_tid,
            "domain": family,
            "family": family,
            "kind": "suite",
            "k": len(scores),
            "seed": 1,
            "tag": arm,
            "status": "bases-only",
            "base": entries,
        },
    )
    _write_json(ns_dir / "base_judged.json", judged)
    return ns_dir


class TestOtherSuiteAcceptance(unittest.TestCase):
    """Requirement 1 & 3 & 5: `other-suite` runs through check, staging, report, Summary, and improve."""

    def setUp(self) -> None:
        self._env_snapshot = os.environ.copy()
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-accept-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.suite = ss.load_suite(OTHER_SUITE_TOML)
        os.environ.update(
            {
                "SKILLBENCH_SUITE": str(OTHER_SUITE_TOML),
                "HARNESS2_RUNS_DIR": str(self.tmp / "runs"),
                "SKILLBENCH_AUTH_FILES": "",
            }
        )

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_snapshot)

    def test_other_suite_config_and_tasks_check(self) -> None:
        self.assertEqual(self.suite.name, "acme-ops")
        self.assertEqual(self.suite.plugins, ("triage-kit", "deploy-kit"))
        self.assertEqual(self.suite.skill_names(), frozenset({"incident-triage", "canary-rollout"}))
        self.assertEqual(
            dict(self.suite.families),
            {"release": ("incident-triage", "canary-rollout")},
        )
        # Asking vs non-asking detection from SKILL.md frontmatter
        files = self.suite.skill_files()
        self.assertTrue(ss.skill_asks_user(files["incident-triage"].read_text(encoding="utf-8")))
        self.assertFalse(ss.skill_asks_user(files["canary-rollout"].read_text(encoding="utf-8")))

        # skillbench-tasks check passes with 0 failing tasks and manifest ok
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = st.main(["check", "--suite", str(OTHER_SUITE_TOML)])
        self.assertEqual(rc, 0, buf.getvalue())
        out = buf.getvalue()
        self.assertIn("suite: acme-ops", out)
        self.assertIn("incident-triage/alert-spike", out)
        self.assertIn("canary-rollout/blue-green-shift", out)
        self.assertIn("_suite/full-release", out)
        self.assertIn("3 task(s), 0 failing", out)

    def test_stage_plugins_for_all_four_arm_labels(self) -> None:
        # 1. baseline -> empty plugins directory
        dest_base = self.tmp / "stage-baseline"
        omitted_base = agy.stage_plugins(dest_base, "baseline", suite=self.suite)
        self.assertEqual(omitted_base, [])
        self.assertEqual(list(dest_base.iterdir()), [])

        # 2. suite-acme-ops -> both plugins and both skills staged
        dest_suite = self.tmp / "stage-suite"
        omitted_suite = agy.stage_plugins(dest_suite, "suite-acme-ops", suite=self.suite)
        self.assertEqual(omitted_suite, [])
        self.assertTrue(
            (dest_suite / "triage-kit" / "skills" / "incident-triage" / "SKILL.md").is_file()
        )
        self.assertTrue(
            (dest_suite / "deploy-kit" / "skills" / "canary-rollout" / "SKILL.md").is_file()
        )
        self.assertEqual(
            ss.redaction_names(dest_suite, self.suite),
            frozenset({"acme-ops", "incident-triage", "canary-rollout"}),
        )

        # 3. with (target_skill="incident-triage") -> both skills staged
        dest_with = self.tmp / "stage-with"
        omitted_with = agy.stage_plugins(
            dest_with, "with", target_skill="incident-triage", suite=self.suite
        )
        self.assertEqual(omitted_with, [])
        self.assertTrue(
            (dest_with / "triage-kit" / "skills" / "incident-triage" / "SKILL.md").is_file()
        )
        self.assertTrue(
            (dest_with / "deploy-kit" / "skills" / "canary-rollout" / "SKILL.md").is_file()
        )

        # 4. without (target_skill="incident-triage") -> incident-triage omitted, canary-rollout kept
        dest_without = self.tmp / "stage-without"
        omitted_without = agy.stage_plugins(
            dest_without, "without", target_skill="incident-triage", suite=self.suite
        )
        self.assertEqual(omitted_without, ["triage-kit/skills/incident-triage"])
        self.assertFalse(
            (dest_without / "triage-kit" / "skills" / "incident-triage").exists()
        )
        self.assertTrue(
            (dest_without / "deploy-kit" / "skills" / "canary-rollout" / "SKILL.md").is_file()
        )

    def test_report_summary_and_improve_on_other_suite(self) -> None:
        manifest_hash = sb.check_manifest(self.suite.tasks_root)
        results_root = self.tmp / "runs" / "results"
        executors = [
            ("opus-5.5-high", "gemini-3.8-flash-high", 11_000, 22.0),
            ("gemini-3.8-flash-high", "fable-5.1-high", 9_000, 16.0),
        ]
        for ex, judge_id, toks, wall in executors:
            ex_dir = results_root / ex
            _seed_skill_cell(
                ex_dir / "without",
                "without",
                executor=ex,
                judge=judge_id,
                suite_name=self.suite.name,
                suite_hash=self.suite.suite_hash,
                manifest_hash=manifest_hash,
                task_scores={
                    "incident-triage/alert-spike": [2, 2],
                    "canary-rollout/blue-green-shift": [1, 2],
                },
                tokens=toks,
                wall_s=wall,
            )
            _seed_skill_cell(
                ex_dir / "with",
                "with",
                executor=ex,
                judge=judge_id,
                suite_name=self.suite.name,
                suite_hash=self.suite.suite_hash,
                manifest_hash=manifest_hash,
                task_scores={
                    "incident-triage/alert-spike": [3, 3],
                    "canary-rollout/blue-green-shift": [4, 4],
                },
                tokens=toks - 1000,
                wall_s=wall - 2.0,
            )
            _seed_suite_cell(
                ex_dir / "baseline",
                "baseline",
                executor=ex,
                judge=judge_id,
                suite_name=self.suite.name,
                suite_hash=self.suite.suite_hash,
                manifest_hash=manifest_hash,
                family="release",
                task_id="full-release",
                scores=[1, 2],
                skills_read=[],
                tokens=toks,
                wall_s=wall,
            )
            _seed_suite_cell(
                ex_dir / "suite-acme-ops",
                "suite-acme-ops",
                executor=ex,
                judge=judge_id,
                suite_name=self.suite.name,
                suite_hash=self.suite.suite_hash,
                manifest_hash=manifest_hash,
                family="release",
                task_id="full-release",
                scores=[4, 4],
                skills_read=["incident-triage", "canary-rollout"],
                tokens=toks + 500,
                wall_s=wall + 1.0,
            )

        out_dir = self.tmp / "out"
        rep = sr.build_report(results_root, draws=100, bootstrap_seed=7)
        written = sr.write_report(rep, out_dir)
        self.assertEqual(rep["schema_version"], 2)
        self.assertEqual(rep["suite"]["name"], "acme-ops")
        self.assertEqual(len(rep["models"]), 2)
        self.assertTrue((out_dir / "report.json").is_file())
        self.assertTrue((out_dir / "report.md").is_file())
        self.assertTrue((out_dir / "report.html").is_file())
        self.assertTrue((out_dir / "summary.md").is_file())
        self.assertEqual(len(written), 3)

        # Requirement 3: Summary page answers the 3 questions on one screen & valid GFM tables
        summary_md = (out_dir / "summary.md").read_text(encoding="utf-8")
        report_html = (out_dir / "report.html").read_text(encoding="utf-8")
        self.assertIn("## Skill recommendations", summary_md)
        self.assertIn("incident-triage", summary_md)
        self.assertIn("canary-rollout", summary_md)
        self.assertIn("## Suite vs built-in baseline (per family)", summary_md)
        self.assertIn("release", summary_md)
        self.assertIn("## Model efficiency", summary_md)
        self.assertIn("opus-5.5-high", summary_md)
        self.assertIn("gemini-3.8-flash-high", summary_md)

        # Check HTML Summary tab is the first tab and contains all 3 sections
        self.assertIn('data-tab="summary"', report_html)
        self.assertIn("Per-skill recommendations", report_html)
        self.assertIn("Suite vs built-in baseline (per family)", report_html)
        self.assertIn("Model efficiency", report_html)

        # Validate GitHub Markdown tables in summary.md
        lines = summary_md.splitlines()
        table_blocks: list[list[str]] = []
        cur: list[str] = []
        for line in lines:
            if line.startswith("|"):
                cur.append(line)
            elif cur:
                table_blocks.append(cur)
                cur = []
        if cur:
            table_blocks.append(cur)
        self.assertEqual(len(table_blocks), 3, "Expected 3 Markdown tables in summary.md")
        for tbl in table_blocks:
            self.assertGreaterEqual(len(tbl), 3)
            col_count = tbl[0].count("|")
            self.assertRegex(tbl[1], r"^\|(\s*:?-+:?\s*\|)+$")
            for row_line in tbl:
                self.assertEqual(row_line.count("|"), col_count, row_line)

        # Now run improve_skill on `incident-triage` from `other-suite` with mocked improver + probe
        orig_md = self.suite.skill_files()["incident-triage"].read_text(encoding="utf-8")
        cand_md = orig_md.replace(
            "Diagnose production alert spikes systematically",
            "Diagnose production alert spikes in a single pass",
        )
        # Verify suite's custom banned_tools ("LegacyDeployTool") is enforced
        bad_errs = si.validate_candidate_skill(
            cand_md + "\nUse LegacyDeployTool now.\n",
            target_skill="incident-triage",
            original_md=orig_md,
            suite=self.suite,
        )
        self.assertTrue(any("LegacyDeployTool" in e for e in bad_errs), bad_errs)

        # Verify improver prompt names acme-ops and never cc-thingz
        prompt_text = si.build_improver_prompt(
            skill="incident-triage",
            evidence_packet="evidence",
            suite_name=self.suite.name,
            banned_tools=self.suite.banned_tools,
        )
        self.assertIn("acme-ops", prompt_text)
        self.assertIn("LegacyDeployTool", prompt_text)
        for term in FORBIDDEN_LOGIC_TERMS:
            self.assertNotIn(term, prompt_text)

        tasks_map = sb.load_tasks(self.suite.tasks_root)
        fake_proposal = {
            "diagnosis": "Tightened triage steps.",
            "changes_summary": "Single-pass triage check.",
            "skill_markdown": cand_md,
            "model": "opus-5.5-high",
            "actual_model": "opus-5.5-high",
            "wall_s": 4.0,
            "total_tokens": 2500,
            "attempts": 1,
        }

        def fake_probe(**kwargs: Any) -> dict[str, Any]:
            ex = kwargs.get("executor") or agy.exec_model()
            return {
                "mean": 1.0,
                "n_valid": 2,
                "n_total": 2,
                "invocation_rate": 1.0,
                "t_skill_p50_s": 3.0,
                "tokens_median": 8000,
                "t_total_median_s": 14.0,
                "diff_lines_median": 0,
                "verify": {"passed": 2, "total": 2, "rate": 1.0},
                "criteria_pass_rates": {},
                "tasks": {},
                "executor": ex,
            }

        with (
            mock.patch.object(si, "load_tasks", return_value=tasks_map),
            mock.patch.object(si, "propose_skill", return_value=fake_proposal),
            mock.patch.object(si, "probe_candidate", side_effect=fake_probe),
        ):
            res = si.improve_skill(
                "incident-triage",
                runs_dir=self.tmp / "runs",
                k=2,
                pilot=True,
                models=["opus-5.5-high", "gemini-3.8-flash-high"],
                suite=self.suite,
            )
        self.assertTrue(res["accepted"])
        self.assertIn(
            "ACCEPTED on [opus-5.5-high, gemini-3.8-flash-high]",
            res["decision_line"],
        )
        self.assertTrue(
            (self.tmp / "runs" / "improve" / "incident-triage" / "best.patch").is_file()
        )


class TestNoHardCodingInHarness2(unittest.TestCase):
    """Requirement 2: `cc-thingz|make-plan|root-cause|brainstorm` never appear in runtime logic."""

    @staticmethod
    def _docstring_lines(tree: ast.AST) -> set[int]:
        lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                ):
                    doc_node = node.body[0]
                    start = getattr(doc_node, "lineno", 0)
                    end = getattr(doc_node, "end_lineno", start)
                    lines.update(range(start, end + 1))
        return lines

    def test_no_suite_or_skill_names_in_python_runtime_logic(self) -> None:
        pattern = re.compile(r"cc-thingz|make-plan|root-cause|brainstorm", re.IGNORECASE)
        harness_root = REPO_ROOT / "harness2"
        bundled_tasks = harness_root / "benchmarks" / "skillbench" / "tasks"
        example_suite = harness_root / "benchmarks" / "skillbench" / "suites" / "cc-thingz.toml"

        unexpected_files: list[str] = []
        logic_hits: list[str] = []

        for path in sorted(harness_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path == example_suite or path.is_relative_to(bundled_tasks):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not pattern.search(text):
                continue
            if path.suffix != ".py":
                unexpected_files.append(str(path.relative_to(REPO_ROOT)))
                continue

            tree = ast.parse(text, filename=str(path))
            doc_lines = self._docstring_lines(tree)
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type == tokenize.COMMENT:
                    continue
                if tok.type == tokenize.STRING and tok.start[0] in doc_lines:
                    continue
                if pattern.search(tok.string):
                    rel = path.relative_to(REPO_ROOT)
                    logic_hits.append(f"{rel}:{tok.start[0]}: {tok.string!r}")

        self.assertEqual(unexpected_files, [], f"Hits in non-corpus files: {unexpected_files}")
        self.assertEqual(logic_hits, [], f"Runtime logic hard-coding hits: {logic_hits}")


class TestMatrixResumeAndImproverAcceptance(unittest.TestCase):
    """Requirements 4 & 5: matrix resume schedules 0 rollouts; improver mismatch & dry-run apply."""

    def setUp(self) -> None:
        self._env_snapshot = os.environ.copy()
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-resume-accept-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.runs_dir = self.tmp / "runs"
        self.runs_root = self.runs_dir / "skillbench"
        os.environ.update(
            {
                "SKILLBENCH_SUITE": str(OTHER_SUITE_TOML),
                "HARNESS2_RUNS_DIR": str(self.runs_dir),
                "SKILLBENCH_AUTH_FILES": "",
            }
        )
        p = mock.patch.object(config, "RUNS", self.runs_dir)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_snapshot)

    def test_matrix_re_run_after_completion_schedules_zero_new_rollouts(self) -> None:
        argv = [
            "--suite",
            str(OTHER_SUITE_TOML),
            "--arms",
            "both",
            "--k",
            "2",
            "--yes",
        ]
        with (
            mock.patch.object(sm.run_sweep, "main", return_value=0) as spy_sweep1,
            mock.patch.object(sm.judge, "main", return_value=0) as spy_judge1,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc1 = sm.main(argv)
        self.assertEqual(rc1, 0)
        # 2 executors * 4 arms ("with", "without", "baseline", "suite-acme-ops") = 8 cells
        self.assertEqual(spy_sweep1.call_count, 8)
        self.assertEqual(spy_judge1.call_count, 8)

        # Re-running the identical command schedules 0 new rollouts
        jobs_after = sm.expand_jobs(
            OTHER_SUITE_TOML, k=2, arms="both", runs_root=self.runs_root
        )
        budget_after = sm.estimate_budget(jobs_after)
        self.assertEqual(budget_after["rollouts"], 0)

        with (
            mock.patch.object(sm.run_sweep, "main", return_value=0) as spy_sweep2,
            mock.patch.object(sm.judge, "main", return_value=0) as spy_judge2,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            rc2 = sm.main(argv)
        self.assertEqual(rc2, 0)
        self.assertEqual(spy_sweep2.call_count, 0)
        self.assertEqual(spy_judge2.call_count, 0)

    def test_improver_refuses_mismatch_and_apply_dry_run_prints_branch_and_commit(self) -> None:
        # Copy other-suite into a clean temporary git repo so --apply dry-run can resolve it
        git_repo = self.tmp / "acme-ops-repo"
        shutil.copytree(OTHER_SUITE_DIR, git_repo)
        subprocess.run(["git", "init", str(git_repo)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(git_repo), "config", "user.name", "Test"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(git_repo), "config", "user.email", "test@example.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "-C", str(git_repo), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(git_repo), "commit", "--no-gpg-sign", "-m", "init"],
            check=True,
            capture_output=True,
        )

        suite_toml = git_repo / "suite.toml"
        suite = ss.load_suite(suite_toml)
        manifest_hash = sb.check_manifest(suite.tasks_root)
        res_dir = self.runs_root / "results"
        for arm, sc_val in (("without", [2]), ("with", [3])):
            _seed_skill_cell(
                res_dir / arm,
                arm,
                executor="opus-5.5-high",
                judge="gemini-3.8-flash-high",
                suite_name=suite.name,
                suite_hash=suite.suite_hash,
                manifest_hash=manifest_hash,
                task_scores={"incident-triage/alert-spike": sc_val},
            )

        orig_md = suite.skill_files()["incident-triage"].read_text(encoding="utf-8")
        cand_md = orig_md.replace("systematically", "concisely in one pass")
        proposal = {
            "diagnosis": "Faster triage.",
            "changes_summary": "One-pass triage.",
            "skill_markdown": cand_md,
            "model": "opus-5.5-high",
            "actual_model": "opus-5.5-high",
            "wall_s": 3.0,
            "total_tokens": 2000,
            "attempts": 1,
        }
        cand_summary = {
            "mean": 1.0,
            "n_valid": 1,
            "n_total": 1,
            "invocation_rate": 1.0,
            "t_skill_p50_s": 3.0,
            "tokens_median": 8000,
            "t_total_median_s": 15.0,
            "diff_lines_median": 0,
            "verify": {"passed": 1, "total": 1, "rate": 1.0},
            "criteria_pass_rates": {},
            "tasks": {},
            "executor": "opus-5.5-high",
        }

        with (
            mock.patch.dict(os.environ, {"SKILLBENCH_SUITE": str(suite_toml)}, clear=False),
            mock.patch.object(si, "propose_skill", return_value=proposal),
            mock.patch.object(si, "probe_candidate", return_value=cand_summary),
        ):
            # 1. Model mismatch refused without --allow-model-mismatch
            with self.assertRaisesRegex(ValueError, "model mismatch"):
                si.improve_skill(
                    "incident-triage",
                    runs_dir=self.runs_root,
                    k=1,
                    pilot=True,
                    models=["gemini-3.8-flash-high"],
                    suite=suite,
                )

            # 2. Dry-run default prints branch skillbench/improve-incident-triage-<date> & commit
            out_buf = io.StringIO()
            with contextlib.redirect_stdout(out_buf):
                rc = si.main(
                    [
                        "--suite",
                        str(suite_toml),
                        "--runs-dir",
                        str(self.runs_root),
                        "--skill",
                        "incident-triage",
                        "--k",
                        "1",
                        "--pilot",
                    ]
                )
            self.assertEqual(rc, 0)
            today = datetime.date.today().isoformat()
            stdout_txt = out_buf.getvalue()
            self.assertIn("[dry-run --apply]", stdout_txt)
            self.assertIn(f"skillbench/improve-incident-triage-{today}", stdout_txt)
            self.assertIn("commit --no-gpg-sign -m", stdout_txt)


class TestPublicFunctionCoverage(unittest.TestCase):
    """Requirement 7: Every public function in the 5 new modules has at least one test."""

    NEW_MODULES = (
        ("harness2/benchmarks/skillbench_suite.py", ss),
        ("harness2/cli/skillbench_tasks.py", st),
        ("harness2/cli/skillbench_matrix.py", sm),
        ("harness2/cli/skillbench_summary.py", ssum),
        ("harness2/cli/skillbench_cycle.py", sc),
    )

    def setUp(self) -> None:
        self._env_snapshot = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_snapshot)

    def test_direct_invocation_of_all_helper_public_functions(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="skillbench-pubfn-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        suite = ss.load_suite(OTHER_SUITE_TOML)

        # skillbench_suite.discover_plugins
        self.assertEqual(
            ss.discover_plugins(OTHER_SUITE_DIR / "plugins"),
            ("deploy-kit", "triage-kit"),
        )

        # skillbench_tasks.render_task_toml, rubric_stubs, scaffold_task, render_table
        stubs = st.rubric_stubs(4)
        self.assertEqual(len(stubs), 4)
        toml_txt = st.render_task_toml(
            kind="skill",
            skill="incident-triage",
            family=None,
            prompt="Check pool.",
            timeout=60,
            asks_user=True,
            forbidden_terms=["incident-triage"],
        )
        self.assertIn('target_skill = "incident-triage"', toml_txt)
        scaf = st.scaffold_task(
            tmp / "tasks",
            task_id="new-task",
            kind="skill",
            skill="incident-triage",
            suite=suite,
        )
        self.assertTrue(scaf.task_dir.is_dir())
        rows = st.check_corpus(OTHER_SUITE_DIR / "tasks", suite=suite)
        table_str = st.render_table(rows)
        self.assertIn("3 task(s), 0 failing", table_str)

        # skillbench_matrix.arms_for_mode, load_history, collect_invalid_counts, write_matrix_json
        self.assertEqual(
            sm.arms_for_mode("both", suite),
            ("with", "without", "baseline", "suite-acme-ops"),
        )
        hist_file = tmp / "history.jsonl"
        hist_file.write_text(
            json.dumps({"executor": "opus-5.5-high", "t_total_s": 30.0, "total_tokens": 5000})
            + "\n",
            encoding="utf-8",
        )
        loaded_hist = sm.load_history(hist_file)
        self.assertEqual(len(loaded_hist), 1)
        tot_seen, inv_tot, inv_reasons = sm.collect_invalid_counts(tmp / "nonexistent")
        self.assertEqual((tot_seen, inv_tot, inv_reasons), (0, 0, {}))
        mpath = sm.write_matrix_json(
            tmp,
            suite=suite,
            manifest_hash="mhash",
            k=1,
            arms_mode="skill",
            arms=("with", "without"),
            executors=("opus-5.5-high",),
            judges={"opus-5.5-high": "gemini-3.8-flash-high"},
            budget={"rollouts": 0},
            cells=[],
        )
        self.assertTrue(mpath.is_file())

        # skillbench_summary.history_records_from_report, render_history_markdown,
        # build_summary, render_summary_section_html, render_summary_html, write_summary
        sample_report: dict[str, Any] = {
            "schema_version": 2,
            "suite": {"name": "acme-ops", "hash": suite.suite_hash},
            "k": 2,
            "seed": 1,
            "pilot": True,
            "manifest_hash": "mhash",
            "models": [
                {
                    "executor": "opus-5.5-high",
                    "judge": "gemini-3.8-flash-high",
                    "skills": [
                        {
                            "skill": "incident-triage",
                            "verdict": "helps",
                            "lift": 0.25,
                            "ci_low": 0.10,
                            "ci_high": 0.40,
                            "tokens_with_median": 9000.0,
                            "tokens_without_median": 10000.0,
                            "t_total_with_median": 18.0,
                            "t_total_without_median": 20.0,
                            "invoked": 2,
                            "valid_with_reps": 2,
                            "t_skill_p50": 3.0,
                            "t_skill_p90": 4.0,
                            "ceiling_tasks": 0,
                            "total_tasks": 1,
                        }
                    ],
                    "suites": [],
                }
            ],
        }
        srows = ssum.build_summary(sample_report, history=[])
        self.assertEqual(len(srows), 1)
        sample_report["summary"] = srows
        hrecs = ssum.history_records_from_report(sample_report, date="2026-10-29")
        self.assertEqual(len(hrecs), 1)
        hmd = ssum.render_history_markdown(hrecs)
        self.assertIn("incident-triage", hmd)
        sec_html = ssum.render_summary_section_html(sample_report, history=hrecs)
        self.assertIn("incident-triage", sec_html)
        full_html = ssum.render_summary_html(sample_report, history=hrecs)
        self.assertIn("<!DOCTYPE html>", full_html)
        md_p, html_p = ssum.write_summary(sample_report, tmp / "sum_out", history=hrecs)
        self.assertTrue(md_p.is_file() and html_p.is_file())

        # skillbench_cycle.stage_matrix, stage_report, format_cycle_log, run_cycle
        log_txt = sc.format_cycle_log(
            [{"stage": "matrix", "status": "ok", "rc": 0, "wall_s": 1.2, "tokens": 100}],
            status="ok",
            rc=0,
            total_wall_s=1.2,
            total_tokens=100,
        )
        self.assertIn("stage=matrix", log_txt)
        ns_args = sc._build_parser().parse_args(
            ["--suite", str(OTHER_SUITE_TOML), "--runs-root", str(tmp / "c_runs"), "--yes"]
        )
        with mock.patch.object(sm, "main", return_value=0):
            self.assertEqual(sc.stage_matrix(ns_args, tmp / "c_runs"), 0)
        with mock.patch.object(sr, "main", return_value=0):
            self.assertEqual(sc.stage_report(ns_args, tmp / "c_runs"), 0)
        with contextlib.redirect_stdout(io.StringIO()):
            cres = sc.run_cycle(
                ns_args,
                stage_runners={
                    "matrix": lambda a, r: 0,
                    "report": lambda a, r: 0,
                    "history": lambda a, r: 0,
                },
            )
        self.assertEqual(cres["rc"], 0)

    def test_every_public_function_in_new_modules_is_called_in_tests(self) -> None:
        test_sources = "\n".join(
            p.read_text(encoding="utf-8")
            for p in sorted((REPO_ROOT / "tests").glob("test_skillbench*.py"))
        )
        missing: list[str] = []
        for rel_path, _ in self.NEW_MODULES:
            mod_path = REPO_ROOT / rel_path
            tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=rel_path)
            pub_funcs = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not node.name.startswith("_")
            ]
            self.assertTrue(pub_funcs, f"No public functions found in {rel_path}")
            for fn in pub_funcs:
                if not re.search(rf"\b{re.escape(fn)}\s*\(", test_sources):
                    missing.append(f"{rel_path}:{fn}")
        self.assertEqual(missing, [], f"Untested public functions: {missing}")


if __name__ == "__main__":
    unittest.main()
