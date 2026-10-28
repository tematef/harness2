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

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
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
    _build_parser,
    build_evidence_packet,
    evaluate_acceptance,
    improve_skill,
    main,
    propose_skill,
    render_improve_html,
    render_improve_markdown,
    round_rank_key,
    safe_tid,
    select_weak_skills,
    validate_candidate_skill,
)
from harness2.core import tree as HT
from harness2.substrate.agy import AGY_SURFACE, PLUGINS_REL, AgySubstrate, exec_model, stage_plugins

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


def _seed_rollout(
    rdir: Path,
    *,
    arm: str,
    n_passed: int,
    n_criteria: int = 4,
    executor: str | None = None,
    judge: str | None = None,
    tokens: int = 60000,
    wall_s: float = 120.0,
    verify_passed: bool = True,
) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    timing: dict[str, Any] = {
        "valid": True,
        "invoked": arm == "with",
        "t_skill_s": 10.0 if arm == "with" else None,
        "t_total_s": wall_s,
        "total_tokens": tokens,
    }
    if executor is not None:
        timing["requested_model"] = executor
        timing["actual_model"] = executor
    (rdir / SKILL_TIMING_FILE).write_text(json.dumps(timing), encoding="utf-8")
    (rdir / "final.md").write_text(f"Plan output ({arm})", encoding="utf-8")
    (rdir / "diff.patch").write_text("", encoding="utf-8")
    (rdir / "new_files").mkdir(exist_ok=True)
    (rdir / "events.jsonl").write_text("", encoding="utf-8")
    (rdir / VERIFY_FILE).write_text(
        json.dumps({"passed": verify_passed, "returncode": 0 if verify_passed else 1, "command": "true"}),
        encoding="utf-8",
    )
    judg: dict[str, Any] = {
        "judgments": [{"n_passed": n_passed, "n_criteria": n_criteria, "criteria": []}]
    }
    if judge is not None:
        judg["judge_model"] = judge
    (rdir / JUDGMENTS_FILE).write_text(json.dumps(judg), encoding="utf-8")


def _make_proposal(md: str = CANDIDATE_SKILL, summary: str = "Streamlined steps.") -> dict[str, Any]:
    return {
        "diagnosis": "Current skill lacked explicit single-pass guidance.",
        "changes_summary": summary,
        "skill_markdown": md,
        "model": "opus-5.5-high",
        "actual_model": "opus-5.5-high",
        "wall_s": 5.2,
        "total_tokens": 3200,
        "attempts": 1,
    }


def _make_cand_summary(
    *,
    mean: float = 1.0,
    verify_rate: float = 1.0,
    tokens: int = 42000,
    wall_s: float = 88.0,
    executor: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mean": mean,
        "n_valid": 1,
        "n_total": 1,
        "invocation_rate": 1.0,
        "t_skill_p50_s": 9.0,
        "tokens_median": tokens,
        "t_total_median_s": wall_s,
        "diff_lines_median": 120,
        "verify": {"passed": 1 if verify_rate >= 1.0 else 0, "total": 1, "rate": verify_rate},
        "criteria_pass_rates": {},
        "tasks": {},
    }
    if executor is not None:
        out["executor"] = executor
    return out


class TestSkillbenchImprove(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="test-sb-improve-"))
        self.plugins_dir = self.tmp / "plugins"
        PluginDir = self.plugins_dir / "planning"
        skill_dir = PluginDir / "skills" / "make-plan"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (PluginDir / "plugin.json").write_text('{"name": "cc-thingz-planning"}\n', encoding="utf-8")
        (skill_dir / "SKILL.md").write_text(ORIGINAL_SKILL, encoding="utf-8")
        self.task = SkillbenchTask(
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

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _base_env(self, **extra: str) -> dict[str, str]:
        env = {
            "SKILLBENCH_PLUGINS_DIR": str(self.plugins_dir),
            "SKILLBENCH_PLUGINS": "planning",
            "SKILLBENCH_AUTH_FILES": "",
        }
        env.update(extra)
        return env

    def test_stage_plugins_and_materialise_with_skill_override(self) -> None:
        cand_file = self.tmp / "cand_SKILL.md"
        cand_file.write_text(CANDIDATE_SKILL, encoding="utf-8")
        env = self._base_env()
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

    def test_growth_guard_pass_and_fail_and_suite_banned_tools(self) -> None:
        # CANDIDATE_SKILL grows by ~37% over ORIGINAL_SKILL -> passes default --max-growth 50%
        self.assertEqual(
            validate_candidate_skill(
                CANDIDATE_SKILL,
                target_skill="make-plan",
                original_md=ORIGINAL_SKILL,
                max_growth=50,
            ),
            [],
        )
        # Fails stricter --max-growth 20%
        errs_strict = validate_candidate_skill(
            CANDIDATE_SKILL,
            target_skill="make-plan",
            original_md=ORIGINAL_SKILL,
            max_growth=20,
        )
        self.assertTrue(any("--max-growth 20%" in e for e in errs_strict), errs_strict)

        # Candidate growing by > 50% fails default threshold
        bloated = CANDIDATE_SKILL + ("\n- Extra verbose bullet point." * 15)
        errs_bloat = validate_candidate_skill(
            bloated,
            target_skill="make-plan",
            original_md=ORIGINAL_SKILL,
        )
        self.assertTrue(any("--max-growth 50%" in e for e in errs_bloat), errs_bloat)

        # Negative max_growth is an error
        with self.assertRaises(ValueError):
            validate_candidate_skill(
                CANDIDATE_SKILL,
                target_skill="make-plan",
                original_md=ORIGINAL_SKILL,
                max_growth=-5,
            )

        # Help text includes --max-growth and rationale
        help_txt = _build_parser().format_help()
        self.assertIn("--max-growth", help_txt)
        self.assertIn("percent", help_txt)
        self.assertIn("bloat", help_txt)

        # Suite banned_tools enforced via suite.toml in propose_skill and validate_candidate_skill
        suite_toml = self.tmp / "suite.toml"
        suite_toml.write_text(
            f'name = "custom-suite"\nplugins_dir = "{self.plugins_dir}"\n'
            'plugins = ["planning"]\nbanned_tools = ["ForbiddenToolX"]\n',
            encoding="utf-8",
        )
        cand_with_custom_banned = CANDIDATE_SKILL + "\nCall ForbiddenToolX now.\n"
        with mock.patch.dict("os.environ", {"SKILLBENCH_SUITE": str(suite_toml)}, clear=False):
            errs_suite = validate_candidate_skill(
                cand_with_custom_banned,
                target_skill="make-plan",
                original_md=ORIGINAL_SKILL,
            )
            self.assertTrue(any("ForbiddenToolX" in e for e in errs_suite), errs_suite)

            # propose_skill rejects bloated / banned candidates
            with mock.patch(
                "harness2.cli.skillbench_improve.run_improver_once",
                return_value=_make_proposal(md=bloated),
            ):
                with self.assertRaisesRegex(RuntimeError, "--max-growth 50%"):
                    propose_skill(
                        skill="make-plan",
                        original_md=ORIGINAL_SKILL,
                        evidence_packet="ev",
                        model="opus-5.5-high",
                        max_attempts=1,
                    )

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

    def test_same_model_default_mismatch_error_and_override(self) -> None:
        runs_dir = self.tmp / "runs"
        safe = safe_tid(self.task.task_id)
        for arm in ("without", "with"):
            _seed_rollout(
                runs_dir / "results" / arm / f"{safe}__b1",
                arm=arm,
                n_passed=3,
                executor="gemini-3.8-flash-high",
            )

        observed_exec_models: list[str] = []

        def fake_probe(**kwargs: Any) -> dict[str, Any]:
            observed_exec_models.append(exec_model())
            return _make_cand_summary(mean=1.0, executor=kwargs.get("executor"))

        env = self._base_env()
        env_clean = {k: v for k, v in os.environ.items() if k != "SKILLBENCH_EXEC_MODEL"}
        env_clean.update(env)
        with (
            mock.patch.dict("os.environ", env_clean, clear=True),
            mock.patch(
                "harness2.cli.skillbench_improve.load_tasks",
                return_value={self.task.task_id: self.task},
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.propose_skill",
                return_value=_make_proposal(),
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.probe_candidate",
                side_effect=fake_probe,
            ),
        ):
            # 1. Defaults to the executor recorded in the baseline cell ("gemini-3.8-flash-high")
            res_default = improve_skill("make-plan", runs_dir=runs_dir, k=1, pilot=True)
            self.assertEqual(observed_exec_models, ["gemini-3.8-flash-high"])
            self.assertEqual(res_default["models"], ["gemini-3.8-flash-high"])
            self.assertEqual(res_default["executors"]["without"], "gemini-3.8-flash-high")
            self.assertEqual(res_default["executors"]["current_with"], "gemini-3.8-flash-high")
            self.assertEqual(res_default["executors"]["candidate"], "gemini-3.8-flash-high")

            # 2. Requesting a different model via models= or CLI --models is an error
            with self.assertRaisesRegex(ValueError, "model mismatch"):
                improve_skill(
                    "make-plan",
                    runs_dir=runs_dir,
                    k=1,
                    pilot=True,
                    models=["opus-5.5-high"],
                )

            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                rc_err = main(
                    [
                        "--runs-dir",
                        str(runs_dir),
                        "--skill",
                        "make-plan",
                        "--models",
                        "opus-5.5-high",
                        "--k",
                        "1",
                        "--pilot",
                    ]
                )
            self.assertEqual(rc_err, 1)
            self.assertIn("model mismatch", err_buf.getvalue())

            # 3. Override with allow_model_mismatch=True / --allow-model-mismatch succeeds
            #    and the 3-way report prints the executor of every arm.
            rc_ok = main(
                [
                    "--runs-dir",
                    str(runs_dir),
                    "--skill",
                    "make-plan",
                    "--models",
                    "opus-5.5-high",
                    "--allow-model-mismatch",
                    "--k",
                    "1",
                    "--pilot",
                ]
            )
            self.assertEqual(rc_ok, 0)
            rep_md = (runs_dir / "improve" / "improve_report.md").read_text(encoding="utf-8")
            rep_html = (runs_dir / "improve" / "improve_report.html").read_text(encoding="utf-8")
            self.assertIn(
                "WITHOUT=`gemini-3.8-flash-high` | Current WITH=`gemini-3.8-flash-high` | Candidate WITH=`opus-5.5-high`",
                rep_md,
            )
            self.assertIn("gemini-3.8-flash-high", rep_html)
            self.assertIn("opus-5.5-high", rep_html)

    def test_multi_model_gate_pass_fail_mix_and_report_rows(self) -> None:
        runs_dir = self.tmp / "runs"
        safe = safe_tid(self.task.task_id)
        for model in ("opus-5.5-high", "gemini-3.8-flash-high"):
            for arm in ("without", "with"):
                _seed_rollout(
                    runs_dir / "results" / model / arm / f"{safe}__b1",
                    arm=arm,
                    n_passed=3,
                    executor=model,
                )

        # Mix: opus passes (1.0 > 0.75), gemini regresses (0.50 < 0.75) -> overall REJECTED
        def probe_mix(**kwargs: Any) -> dict[str, Any]:
            ex = kwargs.get("executor") or exec_model()
            score_mean = 1.0 if ex == "opus-5.5-high" else 0.50
            return _make_cand_summary(mean=score_mean, executor=ex)

        with (
            mock.patch.dict("os.environ", self._base_env(), clear=False),
            mock.patch(
                "harness2.cli.skillbench_improve.load_tasks",
                return_value={self.task.task_id: self.task},
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.propose_skill",
                return_value=_make_proposal(),
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.probe_candidate",
                side_effect=probe_mix,
            ),
        ):
            res_mix = improve_skill(
                "make-plan",
                runs_dir=runs_dir,
                k=1,
                pilot=True,
                models=["opus-5.5-high", "gemini-3.8-flash-high"],
            )
            self.assertFalse(res_mix["accepted"])
            self.assertTrue(res_mix["by_model"]["opus-5.5-high"]["accepted"])
            self.assertFalse(res_mix["by_model"]["gemini-3.8-flash-high"]["accepted"])
            self.assertEqual(len(res_mix["model_gates"]), 2)
            self.assertIn("opus-5.5-high: ACCEPTED", res_mix["decision_reason"])
            self.assertIn("gemini-3.8-flash-high: REJECTED", res_mix["decision_reason"])

            record_mix = {
                "improver_model": "opus-5.5-high",
                "executor_model": "opus-5.5-high",
                "models": ["opus-5.5-high", "gemini-3.8-flash-high"],
                "judge_model": "fable-5.1-high",
                "include_verdicts": False,
                "skills": {"make-plan": res_mix},
            }
            md_txt = render_improve_markdown(record_mix)
            html_txt = render_improve_html(record_mix)
            self.assertIn("| `opus-5.5-high` | **ACCEPTED** |", md_txt)
            self.assertIn("| `gemini-3.8-flash-high` | **REJECTED** |", md_txt)
            self.assertIn("data-executor='opus-5.5-high'", html_txt)
            self.assertIn("data-executor='gemini-3.8-flash-high'", html_txt)

        # Both pass -> overall ACCEPTED
        def probe_all_pass(**kwargs: Any) -> dict[str, Any]:
            ex = kwargs.get("executor") or exec_model()
            return _make_cand_summary(mean=1.0, executor=ex)

        with (
            mock.patch.dict("os.environ", self._base_env(), clear=False),
            mock.patch(
                "harness2.cli.skillbench_improve.load_tasks",
                return_value={self.task.task_id: self.task},
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.propose_skill",
                return_value=_make_proposal(),
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.probe_candidate",
                side_effect=probe_all_pass,
            ),
        ):
            res_pass = improve_skill(
                "make-plan",
                runs_dir=runs_dir,
                k=1,
                pilot=True,
                models="opus-5.5-high,gemini-3.8-flash-high",
            )
            self.assertTrue(res_pass["accepted"])
            self.assertTrue(res_pass["by_model"]["opus-5.5-high"]["accepted"])
            self.assertTrue(res_pass["by_model"]["gemini-3.8-flash-high"]["accepted"])

    def test_rounds_feedback_early_stop_and_best_round_ranking(self) -> None:
        runs_dir = self.tmp / "runs"
        safe = safe_tid(self.task.task_id)
        for arm, np in (("without", 2), ("with", 3)):
            _seed_rollout(
                runs_dir / "results" / arm / f"{safe}__b1",
                arm=arm,
                n_passed=np,
                executor="opus-5.5-high",
            )

        cand_r1 = CANDIDATE_SKILL.replace("in one pass.", "in round 1 pass.")
        cand_r2 = CANDIDATE_SKILL.replace("in one pass.", "in round 2 pass.")
        proposals = [
            _make_proposal(md=cand_r1, summary="Round 1 attempt"),
            _make_proposal(md=cand_r2, summary="Round 2 attempt"),
            _make_proposal(md=CANDIDATE_SKILL, summary="Round 3 attempt"),
        ]
        # Round 1 is rejected (0.75 == 0.75, no efficiency gain).
        # Round 2 is accepted (0.85 > 0.75, note < 1.0) -> must early-stop before Round 3!
        probe_results = [
            _make_cand_summary(mean=0.75, tokens=58000, wall_s=115.0, executor="opus-5.5-high"),
            _make_cand_summary(mean=0.85, tokens=55000, wall_s=110.0, executor="opus-5.5-high"),
            _make_cand_summary(mean=1.00, tokens=40000, wall_s=80.0, executor="opus-5.5-high"),
        ]
        captured_packets: list[str] = []

        def fake_propose(**kwargs: Any) -> dict[str, Any]:
            captured_packets.append(kwargs["evidence_packet"])
            return proposals[len(captured_packets) - 1]

        with (
            mock.patch.dict("os.environ", self._base_env(), clear=False),
            mock.patch(
                "harness2.cli.skillbench_improve.load_tasks",
                return_value={self.task.task_id: self.task},
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.propose_skill",
                side_effect=fake_propose,
            ) as m_prop,
            mock.patch(
                "harness2.cli.skillbench_improve.probe_candidate",
                side_effect=probe_results,
            ) as m_probe,
        ):
            res = improve_skill("make-plan", runs_dir=runs_dir, k=1, rounds=3, pilot=True)
            # Early stop after Round 2 (Round 3 never called)
            self.assertEqual(m_prop.call_count, 2)
            self.assertEqual(m_probe.call_count, 2)
            self.assertEqual(len(res["rounds"]), 2)
            self.assertTrue(res["accepted"])
            self.assertEqual(res["best_round"], 2)

            # Round 2 evidence packet includes Round 1's candidate, 3-way numbers, and gate reason
            r2_packet = captured_packets[1]
            self.assertIn("in round 1 pass.", r2_packet)
            self.assertIn("3-way numbers (WITHOUT -> Current WITH -> Candidate WITH)", r2_packet)
            self.assertIn("mean=0.50 -> 0.75 -> 0.75", r2_packet)
            self.assertIn("no quality gain (0.750 -> 0.750) and efficiency improvement < 15%", r2_packet)

        # Best round ranking by (verify, mean, -tokens, -wall) when all 4 rounds run
        self.assertGreater(
            round_rank_key(_make_cand_summary(verify_rate=1.0, mean=0.70, tokens=60000, wall_s=120.0)),
            round_rank_key(_make_cand_summary(verify_rate=0.5, mean=0.95, tokens=30000, wall_s=60.0)),
        )
        self.assertGreater(
            round_rank_key(_make_cand_summary(verify_rate=1.0, mean=0.85, tokens=60000, wall_s=120.0)),
            round_rank_key(_make_cand_summary(verify_rate=1.0, mean=0.80, tokens=30000, wall_s=60.0)),
        )
        self.assertGreater(
            round_rank_key(_make_cand_summary(verify_rate=1.0, mean=0.80, tokens=40000, wall_s=120.0)),
            round_rank_key(_make_cand_summary(verify_rate=1.0, mean=0.80, tokens=50000, wall_s=60.0)),
        )
        self.assertGreater(
            round_rank_key(_make_cand_summary(verify_rate=1.0, mean=0.80, tokens=40000, wall_s=80.0)),
            round_rank_key(_make_cand_summary(verify_rate=1.0, mean=0.80, tokens=40000, wall_s=95.0)),
        )

        # End-to-end 3 unaccepted rounds pick best round by (verify, mean, -tokens, -wall)
        runs_dir2 = self.tmp / "runs_rank"
        for arm in ("without", "with"):
            _seed_rollout(
                runs_dir2 / "results" / arm / f"{safe}__b1",
                arm=arm,
                n_passed=3,
                executor="opus-5.5-high",
            )
        unaccepted_probes = [
            # Round 1: verify=0.5, mean=0.75
            _make_cand_summary(verify_rate=0.5, mean=0.75, tokens=64000, wall_s=120.0, executor="opus-5.5-high"),
            # Round 2: verify=1.0, mean=0.75, tokens=63000 (wins over R1 on verify, beat by R3 on tokens)
            _make_cand_summary(verify_rate=1.0, mean=0.75, tokens=63000, wall_s=120.0, executor="opus-5.5-high"),
            # Round 3: verify=1.0, mean=0.75, tokens=62000, wall=115.0 (best by -tokens, still <15% token/wall gain)
            _make_cand_summary(verify_rate=1.0, mean=0.75, tokens=62000, wall_s=115.0, executor="opus-5.5-high"),
        ]
        with (
            mock.patch.dict("os.environ", self._base_env(), clear=False),
            mock.patch(
                "harness2.cli.skillbench_improve.load_tasks",
                return_value={self.task.task_id: self.task},
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.propose_skill",
                side_effect=proposals,
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.probe_candidate",
                side_effect=unaccepted_probes,
            ),
        ):
            res_rank = improve_skill("make-plan", runs_dir=runs_dir2, k=1, rounds=3, pilot=True)
            self.assertFalse(res_rank["accepted"])
            self.assertEqual(res_rank["best_round"], 3)
            self.assertEqual(res_rank["changes_summary"], "Round 3 attempt")

    def test_ceiling_skip_per_executor(self) -> None:
        runs_dir = self.tmp / "runs_ceiling"
        safe = safe_tid(self.task.task_id)
        # For opus-5.5-high, WITHOUT passes 4/4 criteria (1/1 = 100% ceiling tasks -> needs harder tasks)
        for arm in ("without", "with"):
            _seed_rollout(
                runs_dir / "results" / "opus-5.5-high" / arm / f"{safe}__b1",
                arm=arm,
                n_passed=4,
                n_criteria=4,
                executor="opus-5.5-high",
            )
        # For gemini-3.8-flash-high, WITHOUT passes 2/4 criteria (informative)
        for arm, np in (("without", 2), ("with", 3)):
            _seed_rollout(
                runs_dir / "results" / "gemini-3.8-flash-high" / arm / f"{safe}__b1",
                arm=arm,
                n_passed=np,
                n_criteria=4,
                executor="gemini-3.8-flash-high",
            )

        with (
            mock.patch.dict("os.environ", self._base_env(), clear=False),
            mock.patch(
                "harness2.cli.skillbench_improve.load_tasks",
                return_value={self.task.task_id: self.task},
            ),
            mock.patch(
                "harness2.cli.skillbench_improve.propose_skill",
                return_value=_make_proposal(),
            ) as m_prop,
            mock.patch(
                "harness2.cli.skillbench_improve.probe_candidate",
                return_value=_make_cand_summary(mean=1.0, executor="gemini-3.8-flash-high"),
            ) as m_probe,
        ):
            out_buf = io.StringIO()
            with contextlib.redirect_stdout(out_buf):
                res_skip = improve_skill(
                    "make-plan",
                    runs_dir=runs_dir,
                    k=1,
                    pilot=True,
                    models=["opus-5.5-high"],
                )
            self.assertTrue(res_skip["skipped"])
            self.assertFalse(res_skip["accepted"])
            self.assertIn("needs harder tasks", res_skip["skip_reason"])
            self.assertIn("opus-5.5-high", res_skip["skip_reason"])
            self.assertIn("needs harder tasks", out_buf.getvalue())
            m_prop.assert_not_called()
            m_probe.assert_not_called()

            # Same skill on gemini-3.8-flash-high is informative -> NOT skipped
            res_run = improve_skill(
                "make-plan",
                runs_dir=runs_dir,
                k=1,
                pilot=True,
                models=["gemini-3.8-flash-high"],
            )
            self.assertFalse(res_run["skipped"])
            self.assertTrue(res_run["accepted"])
            self.assertEqual(m_prop.call_count, 1)
            self.assertEqual(m_probe.call_count, 1)

        # Also test select_weak_skills skips skills flagged needs_harder_tasks in report.json
        rep_with_ceiling = {
            "skills": {
                "ceiling-skill": {
                    "mean_with": 1.0,
                    "lift": 0.0,
                    "ceiling_tasks": 3,
                    "total_tasks": 4,
                    "needs_harder_tasks": True,
                },
                "informative-weak": {
                    "mean_with": 0.75,
                    "lift": 0.0,
                    "ceiling_tasks": 1,
                    "total_tasks": 4,
                    "needs_harder_tasks": False,
                },
            }
        }
        buf_sel = io.StringIO()
        with contextlib.redirect_stdout(buf_sel):
            weak = select_weak_skills(rep_with_ceiling, executor="opus-5.5-high")
        self.assertEqual(weak, ["informative-weak"])
        self.assertIn("needs harder tasks", buf_sel.getvalue())

    def test_improve_skill_end_to_end_mocked(self) -> None:
        runs_dir = self.tmp / "runs"
        safe = safe_tid(self.task.task_id)
        for arm, np in (("without", 3), ("with", 3)):
            _seed_rollout(
                runs_dir / "results" / arm / f"{safe}__b1",
                arm=arm,
                n_passed=np,
            )

        env = self._base_env()
        fake_proposal = _make_proposal()
        fake_cand_summary = _make_cand_summary(mean=1.0)
        with (
            mock.patch.dict("os.environ", env, clear=False),
            mock.patch(
                "harness2.cli.skillbench_improve.load_tasks",
                return_value={self.task.task_id: self.task},
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
