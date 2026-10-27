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

"""Unit tests for `skillbench-improve` (`harness2.cli.skillbench_improve`)."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness2.benchmarks.skillbench import (
    JUDGMENTS_FILE,
    SKILL_TIMING_FILE,
    VERIFY_FILE,
    Criterion,
    SkillbenchTask,
    skillbench_spec,
)
from harness2.cli.skillbench_improve import (
    build_evidence_packet,
    evaluate_acceptance,
    improve_skill,
    main,
    safe_tid,
    select_weak_skills,
    validate_candidate_skill,
)
from harness2.core import tree as HT
from harness2.substrate.agy import AGY_SURFACE, PLUGINS_REL, AgySubstrate, stage_plugins

ORIGINAL_SKILL = """---
name: make-plan
description: Create a structured implementation plan in docs/plans/
allowed-tools: view_file, write_to_file, ask_question
---

# Make Plan

## When no one can answer

Proceed with recommended option.

## Steps

1. Inspect files.
2. Write plan.
"""

CANDIDATE_SKILL = """---
name: make-plan
description: Create a concise, structured implementation plan in docs/plans/
allowed-tools: view_file, write_to_file, ask_question
---

# Make Plan

## When no one can answer

Proceed immediately with recommended option without searching for missing sibling skills.

## Steps

1. Inspect only the target files directly.
2. Write the plan file in one pass.
"""


class TestSkillbenchImprove(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="test-sb-improve-"))
        self.plugins_dir = self.tmp / "plugins"
        skill_dir = self.plugins_dir / "planning" / "skills" / "make-plan"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(ORIGINAL_SKILL, encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stage_plugins_and_materialise_with_skill_override(self) -> None:
        cand_file = self.tmp / "cand_SKILL.md"
        cand_file.write_text(CANDIDATE_SKILL, encoding="utf-8")
        env = {
            "SKILLBENCH_PLUGINS_DIR": str(self.plugins_dir),
            "SKILLBENCH_PLUGINS": "planning",
            "SKILLBENCH_AUTH_FILES": "",
        }
        with mock.patch.dict("os.environ", env, clear=False):
            dest_orig = self.tmp / "staged_orig"
            stage_plugins(dest_orig, arm="with", target_skill="make-plan")
            self.assertEqual(
                (dest_orig / "planning" / "skills" / "make-plan" / "SKILL.md").read_text(),
                ORIGINAL_SKILL,
            )

            dest_cand = self.tmp / "staged_cand"
            stage_plugins(
                dest_cand,
                arm="with",
                target_skill="make-plan",
                skill_override=cand_file,
            )
            self.assertEqual(
                (dest_cand / "planning" / "skills" / "make-plan" / "SKILL.md").read_text(),
                CANDIDATE_SKILL,
            )

            with self.assertRaises(ValueError):
                stage_plugins(
                    self.tmp / "staged_bad",
                    arm="without",
                    target_skill="make-plan",
                    skill_override=cand_file,
                )

            tree_dir = self.tmp / "tree"
            HT.write_tree(AGY_SURFACE, {"GEMINI.md": "<!-- base -->\n"}, tree_dir, stubs=False)
            sub = AgySubstrate()
            spec = skillbench_spec(runs_root=self.tmp / "runs")
            inst1 = sub.materialise(
                tree_dir, spec, {"arm": "with", "target_skill": "make-plan"}
            )
            inst2 = sub.materialise(
                tree_dir,
                spec,
                {
                    "arm": "with",
                    "target_skill": "make-plan",
                    "skill_override": str(cand_file),
                },
            )
            try:
                self.assertNotEqual(inst1.fingerprint, inst2.fingerprint)
                staged_md = (
                    inst2.staging_dir
                    / "home"
                    / PLUGINS_REL
                    / "planning"
                    / "skills"
                    / "make-plan"
                    / "SKILL.md"
                ).read_text(encoding="utf-8")
                self.assertEqual(staged_md, CANDIDATE_SKILL)
            finally:
                shutil.rmtree(inst1.staging_dir, ignore_errors=True)
                shutil.rmtree(inst2.staging_dir, ignore_errors=True)

    def test_validate_candidate_skill(self) -> None:
        errs = validate_candidate_skill(
            CANDIDATE_SKILL, target_skill="make-plan", original_md=ORIGINAL_SKILL
        )
        self.assertEqual(errs, [])

        # Identical skill rejected
        errs_same = validate_candidate_skill(
            ORIGINAL_SKILL, target_skill="make-plan", original_md=ORIGINAL_SKILL
        )
        self.assertTrue(any("identical" in e for e in errs_same))

        # Missing headless section rejected
        no_headless = CANDIDATE_SKILL.replace("## When no one can answer", "## Fallback")
        errs_hl = validate_candidate_skill(
            no_headless, target_skill="make-plan", original_md=ORIGINAL_SKILL
        )
        self.assertTrue(any("When no one can answer" in e for e in errs_hl))

        # Banned Claude tool rejected
        with_claude_tool = CANDIDATE_SKILL + "\nUse AskUserQuestion to ask the user.\n"
        errs_tool = validate_candidate_skill(
            with_claude_tool, target_skill="make-plan", original_md=ORIGINAL_SKILL
        )
        self.assertTrue(any("AskUserQuestion" in e for e in errs_tool))

        # Wrong frontmatter name rejected
        wrong_name = CANDIDATE_SKILL.replace("name: make-plan", "name: other-skill")
        errs_name = validate_candidate_skill(
            wrong_name, target_skill="make-plan", original_md=ORIGINAL_SKILL
        )
        self.assertTrue(any("does not match" in e for e in errs_name))

    def test_select_weak_skills(self) -> None:
        report = {
            "skills": {
                "root-cause-investigator": {
                    "mean_with": 1.0,
                    "lift": 0.07,
                    "verify_with": {"rate": None},
                    "tokens_median_with": 50000,
                    "tokens_median_without": 55000,
                    "t_total_median_s_with": 100.0,
                    "t_total_median_s_without": 120.0,
                },
                "make-plan": {
                    "mean_with": 1.0,
                    "lift": 0.0,
                    "verify_with": {"rate": None},
                    "tokens_median_with": 60000,
                    "tokens_median_without": 90000,
                    "t_total_median_s_with": 150.0,
                    "t_total_median_s_without": 290.0,
                },
                "slow-skill": {
                    "mean_with": 1.0,
                    "lift": 0.05,
                    "verify_with": {"rate": 1.0},
                    "tokens_median_with": 95000,
                    "tokens_median_without": 50000,
                    "t_total_median_s_with": 200.0,
                    "t_total_median_s_without": 100.0,
                },
            }
        }
        weak = select_weak_skills(report)
        self.assertEqual(weak, ["make-plan", "slow-skill"])

    def test_evidence_packet_leak_guard(self) -> None:
        runs_dir = self.tmp / "runs"
        task = SkillbenchTask(
            task_id="make-plan/task-1",
            target_skill="make-plan",
            prompt="Create a plan for adding caching.",
            timeout=120,
            rubric=(Criterion("secret_crit_id", "Must mention secret_criterion_phrase"),),
            pilot=True,
            dir=self.tmp,
        )
        safe = safe_tid(task.task_id)
        for arm in ("with", "without"):
            rdir = runs_dir / "results" / arm / f"{safe}__b1"
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / SKILL_TIMING_FILE).write_text(
                json.dumps(
                    {
                        "valid": True,
                        "invoked": arm == "with",
                        "t_skill_s": 12.0 if arm == "with" else None,
                        "t_total_s": 90.0,
                        "total_tokens": 45000,
                    }
                ),
                encoding="utf-8",
            )
            (rdir / "final.md").write_text(f"Final answer for {arm}", encoding="utf-8")
            (rdir / "diff.patch").write_text("", encoding="utf-8")
            (rdir / "new_files").mkdir(exist_ok=True)
            (rdir / "events.jsonl").write_text("", encoding="utf-8")
            (rdir / JUDGMENTS_FILE).write_text(
                json.dumps(
                    {
                        "judgments": [
                            {
                                "n_passed": 0,
                                "n_criteria": 1,
                                "criteria": [
                                    {
                                        "id": "secret_crit_id",
                                        "passed": False,
                                        "rationale": "secret_judge_rationale_leak",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

        ev_clean = build_evidence_packet(
            skill="make-plan",
            current_skill_md=ORIGINAL_SKILL,
            tasks=[task],
            runs_dir=runs_dir,
            k=1,
            include_verdicts=False,
        )
        self.assertNotIn("secret_crit_id", ev_clean)
        self.assertNotIn("secret_judge_rationale_leak", ev_clean)

        ev_diag = build_evidence_packet(
            skill="make-plan",
            current_skill_md=ORIGINAL_SKILL,
            tasks=[task],
            runs_dir=runs_dir,
            k=1,
            include_verdicts=True,
        )
        self.assertIn("secret_crit_id", ev_diag)
        self.assertIn("secret_judge_rationale_leak", ev_diag)

    def test_evaluate_acceptance(self) -> None:
        cur = {
            "mean": 0.90,
            "invocation_rate": 1.0,
            "tokens_median": 60000,
            "t_total_median_s": 150.0,
            "verify": {"rate": 1.0},
        }
        # Quality improvement -> accepted
        ok, reason = evaluate_acceptance(
            cur,
            {
                "mean": 1.0,
                "invocation_rate": 1.0,
                "tokens_median": 62000,
                "t_total_median_s": 155.0,
                "verify": {"rate": 1.0},
            },
        )
        self.assertTrue(ok)
        self.assertIn("rubric score improved", reason)

        # Efficiency improvement at ceiling -> accepted
        cur_ceil = {**cur, "mean": 1.0}
        ok2, reason2 = evaluate_acceptance(
            cur_ceil,
            {
                "mean": 1.0,
                "invocation_rate": 1.0,
                "tokens_median": 45000,
                "t_total_median_s": 140.0,
                "verify": {"rate": 1.0},
            },
        )
        self.assertTrue(ok2)
        self.assertIn("lower median tokens", reason2)

        # Verify regression -> rejected
        ok3, reason3 = evaluate_acceptance(
            cur,
            {
                "mean": 1.0,
                "invocation_rate": 1.0,
                "tokens_median": 45000,
                "t_total_median_s": 110.0,
                "verify": {"rate": 0.5},
            },
        )
        self.assertFalse(ok3)
        self.assertIn("verify rate regressed", reason3)

    def test_improve_skill_end_to_end_mocked(self) -> None:
        runs_dir = self.tmp / "runs"
        task = SkillbenchTask(
            task_id="make-plan/task-1",
            target_skill="make-plan",
            prompt="Create a plan.",
            timeout=120,
            rubric=(
                Criterion("c1", "Criterion 1"),
                Criterion("c2", "Criterion 2"),
                Criterion("c3", "Criterion 3"),
                Criterion("c4", "Criterion 4"),
            ),
            pilot=True,
            dir=self.tmp,
        )
        safe = safe_tid(task.task_id)
        for arm, np in (("without", 3), ("with", 3)):
            rdir = runs_dir / "results" / arm / f"{safe}__b1"
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / SKILL_TIMING_FILE).write_text(
                json.dumps(
                    {
                        "valid": True,
                        "invoked": arm == "with",
                        "t_skill_s": 10.0,
                        "t_total_s": 120.0,
                        "total_tokens": 60000,
                    }
                ),
                encoding="utf-8",
            )
            (rdir / "final.md").write_text("Plan output", encoding="utf-8")
            (rdir / "diff.patch").write_text("", encoding="utf-8")
            (rdir / "new_files").mkdir(exist_ok=True)
            (rdir / "events.jsonl").write_text("", encoding="utf-8")
            (rdir / VERIFY_FILE).write_text(
                json.dumps({"passed": True, "returncode": 0, "command": "true"}),
                encoding="utf-8",
            )
            (rdir / JUDGMENTS_FILE).write_text(
                json.dumps({"judgments": [{"n_passed": np, "n_criteria": 4, "criteria": []}]}),
                encoding="utf-8",
            )

        env = {
            "SKILLBENCH_PLUGINS_DIR": str(self.plugins_dir),
            "SKILLBENCH_PLUGINS": "planning",
            "SKILLBENCH_AUTH_FILES": "",
        }
        fake_proposal = {
            "diagnosis": "Current skill lacked explicit single-pass guidance.",
            "changes_summary": "Streamlined steps to avoid extra searches.",
            "skill_markdown": CANDIDATE_SKILL,
            "model": "opus-5.5-high",
            "actual_model": "opus-5.5-high",
            "wall_s": 5.2,
            "total_tokens": 3200,
            "attempts": 1,
        }
        fake_cand_summary = {
            "mean": 1.0,
            "n_valid": 1,
            "n_total": 1,
            "invocation_rate": 1.0,
            "t_skill_p50_s": 9.0,
            "tokens_median": 42000,
            "t_total_median_s": 88.0,
            "diff_lines_median": 120,
            "verify": {"passed": 1, "total": 1, "rate": 1.0},
            "criteria_pass_rates": {},
            "tasks": {},
        }
        with (
            mock.patch.dict("os.environ", env, clear=False),
            mock.patch(
                "harness2.cli.skillbench_improve.load_tasks",
                return_value={task.task_id: task},
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.propose_skill",
                return_value=fake_proposal,
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.probe_candidate",
                return_value=fake_cand_summary,
            ),
        ):
            res = improve_skill("make-plan", runs_dir=runs_dir, k=1, rounds=1, pilot=True)
            self.assertTrue(res["accepted"])
            self.assertEqual(res["current_with"]["mean"], 0.75)
            self.assertEqual(res["best_candidate"]["mean"], 1.0)
            self.assertTrue((runs_dir / "improve" / "make-plan" / "best.patch").is_file())

            rc = main(
                [
                    "--runs-dir",
                    str(runs_dir),
                    "--skill",
                    "make-plan",
                    "--k",
                    "1",
                    "--pilot",
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((runs_dir / "improve" / "improve_report.json").is_file())
            self.assertTrue((runs_dir / "improve" / "improve_report.md").is_file())
            self.assertTrue((runs_dir / "improve" / "improve_report.html").is_file())


if __name__ == "__main__":
    unittest.main()
