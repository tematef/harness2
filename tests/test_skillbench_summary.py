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

"""Unit tests for Task 8: One-page Summary, recommendations, and `history.jsonl` trend."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from harness2.cli import skillbench_report as SR
from harness2.cli import skillbench_summary as SS


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _seed_cell_ns(
    ns_dir: Path,
    arm: str,
    spec: dict[str, list[dict[str, Any]]],
    *,
    executor: str = "exec-a",
    judge: str = "judge-b",
    suite_name: str = "demo-suite",
    suite_hash: str = "hash1111aaaa",
    manifest_hash: str = "manifest-xyz",
    seed: int = 7,
    arm_kind: str = "skill",
) -> Path:
    """Create a synthetic judged namespace directory for skill or suite arms."""
    ns_dir.mkdir(parents=True, exist_ok=True)
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
    _write_json(
        ns_dir / "run.json",
        {
            "benchmark": "skillbench",
            "arm": arm,
            "arm_kind": arm_kind,
            "suite": suite_name,
            "suite_hash": suite_hash,
            "manifest_hash": manifest_hash,
            "executor": executor,
            "judge": judge,
            "effort": "high",
        },
    )
    judged: list[dict[str, Any]] = []
    for task_id, reps in sorted(spec.items()):
        domain = task_id.split("/", 1)[0]
        task_dir = ns_dir / task_id.replace("/", "__")
        entries: list[dict[str, Any]] = []
        for idx, rep in enumerate(reps, 1):
            label = f"base{idx}"
            run_dir = task_dir / label
            run_dir.mkdir(parents=True, exist_ok=True)
            n_passed = int(rep.get("n_passed", 2))
            n_criteria = int(rep.get("n_criteria", 4))
            invoked = bool(rep.get("invoked", arm not in ("without", "baseline")))
            t_skill_s = rep.get("t_skill_s", 4.0 if invoked else None)
            skills_read = list(
                rep.get("skills_read") or ([domain] if invoked else [])
            )
            _write_json(
                run_dir / "skill_timing.json",
                {
                    "invoked": invoked,
                    "t_skill_s": t_skill_s,
                    "step_idx": 1 if invoked else None,
                    "skills_read": skills_read,
                    "t_total_s": float(rep.get("t_total_s", 20.0)),
                    "requested_model": executor,
                    "actual_model": executor,
                    "valid": True,
                    "invalid_reason": None,
                    "exit_code": 0,
                    "total_tokens": int(rep.get("tokens", 1000)),
                },
            )
            if "verify_passed" in rep:
                _write_json(
                    run_dir / "verify.json",
                    {"passed": bool(rep["verify_passed"])},
                )
            score = {"n_passed": n_passed, "n_criteria": n_criteria}
            _write_json(
                run_dir / "judgments.json",
                {
                    "judge_model": judge,
                    "actual_model": judge,
                    "score": score,
                    "judgments": [
                        {
                            "criteria": [
                                {"id": f"c{i}", "pass": i < n_passed, "reason": "ok"}
                                for i in range(n_criteria)
                            ],
                            "n_passed": n_passed,
                            "n_criteria": n_criteria,
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
                    "ok": True,
                    "delivered": True,
                }
            )
            judged.append(
                {"task": task_id, "label": label, "run_id": run_id, "score": score}
            )
        summary_payload: dict[str, Any] = {
            "task_id": task_id,
            "domain": domain,
            "k": len(reps),
            "seed": seed,
            "tag": arm,
            "status": "bases-only",
            "base": entries,
        }
        if arm_kind == "suite":
            summary_payload["family"] = domain
            summary_payload["kind"] = "suite"
        _write_json(task_dir / "summary.json", summary_payload)
    _write_json(ns_dir / "base_judged.json", judged)
    return ns_dir


class TestRecommendBranchesAndPrecedence(unittest.TestCase):
    """Test every branch of `recommend(row) -> (label, reason)` and rule precedence."""

    def test_rule_1_needs_harder_tasks_and_50_percent_boundary(self) -> None:
        # > 50% at ceiling (3/4 = 75%) -> needs harder tasks
        label, reason = SR.recommend(
            {
                "skill": "alpha",
                "ceiling_tasks": 3,
                "total_tasks": 4,
                "verdict": "helps",
                "lift": 0.25,
            }
        )
        self.assertEqual(label, SR.REC_NEEDS_HARDER_TASKS)
        self.assertIn("rule 1 (needs harder tasks)", reason)
        self.assertIn("3/4 tasks (75%) at ceiling (> 50%)", reason)

        # Exactly 50% (2/4) -> does NOT trigger rule 1
        label_50, reason_50 = SR.recommend(
            {
                "skill": "alpha",
                "ceiling_tasks": 2,
                "total_tasks": 4,
                "verdict": "helps",
                "lift": 0.25,
            }
        )
        self.assertEqual(label_50, SR.REC_KEEP)
        self.assertIn("rule 4 (keep)", reason_50)

        # Explicit ceiling_share > 0.50 without counts
        label_share, reason_share = SR.recommend(
            {"skill": "alpha", "ceiling_share": 0.60, "verdict": "inconclusive"}
        )
        self.assertEqual(label_share, SR.REC_NEEDS_HARDER_TASKS)
        self.assertIn("60% of tasks at ceiling", reason_share)

        # Explicit boolean flag needs_harder_tasks=True
        label_flag, reason_flag = SR.recommend(
            {"skill": "alpha", "needs_harder_tasks": True, "verdict": "inconclusive"}
        )
        self.assertEqual(label_flag, SR.REC_NEEDS_HARDER_TASKS)
        self.assertIn("rule 1 (needs harder tasks)", reason_flag)

    def test_rule_2_consider_removing_branches(self) -> None:
        # Branch 2a: verdict == "hurts"
        label, reason = SR.recommend(
            {
                "skill": "bad-skill",
                "ceiling_tasks": 0,
                "total_tasks": 4,
                "verdict": "hurts",
                "lift": -0.30,
            }
        )
        self.assertEqual(label, SR.REC_CONSIDER_REMOVING)
        self.assertIn("rule 2 (consider removing)", reason)
        self.assertIn("verdict is hurts (lift -0.30)", reason)

        # Branch 2b: inconclusive + tokens_delta_pct >= +15% + wall_delta_pct >= +15% + no verify gain
        label_inc, reason_inc = SR.recommend(
            {
                "skill": "bloated-skill",
                "ceiling_tasks": 1,
                "total_tasks": 4,
                "verdict": "inconclusive",
                "tokens_with_median": 1150.0,
                "tokens_without_median": 1000.0,  # +15%
                "t_total_with_median": 24.0,
                "t_total_without_median": 20.0,  # +20%
                "verify_with_rate": 0.50,
                "verify_without_rate": 0.50,  # no verify gain
            }
        )
        self.assertEqual(label_inc, SR.REC_CONSIDER_REMOVING)
        self.assertIn("rule 2 (consider removing)", reason_inc)
        self.assertIn("tokens p50 +15% (>= +15%)", reason_inc)
        self.assertIn("wall p50 +20% (>= +15%)", reason_inc)
        self.assertIn("no verify gain", reason_inc)

        # Verify gain present -> does NOT trigger rule 2
        label_vgain, _ = SR.recommend(
            {
                "skill": "verify-helping-skill",
                "ceiling_tasks": 0,
                "total_tasks": 4,
                "verdict": "inconclusive",
                "tokens_delta_pct": 25.0,
                "wall_delta_pct": 25.0,
                "verify_with_rate": 0.75,
                "verify_without_rate": 0.50,
                "invocation_rate": 0.90,
                "t_skill_p90": 10.0,
            }
        )
        self.assertEqual(label_vgain, SR.REC_KEEP)

        # Tokens < +15% -> does NOT trigger rule 2
        label_low_tok, _ = SR.recommend(
            {
                "skill": "mild-tok",
                "verdict": "inconclusive",
                "tokens_delta_pct": 14.0,
                "wall_delta_pct": 25.0,
                "invocation_rate": 0.90,
                "t_skill_p90": 10.0,
            }
        )
        self.assertEqual(label_low_tok, SR.REC_KEEP)

    def test_rule_3_fix_branches(self) -> None:
        # Branch 3a: inconclusive with invocation < 80%
        label_inv, reason_inv = SR.recommend(
            {
                "skill": "hidden-skill",
                "ceiling_tasks": 1,
                "total_tasks": 4,
                "verdict": "inconclusive",
                "invocation_rate": 0.75,
                "t_skill_p90": 12.0,
            }
        )
        self.assertEqual(label_inv, SR.REC_FIX)
        self.assertIn("rule 3 (fix)", reason_inv)
        self.assertIn("invocation 75% < 80%", reason_inv)

        # Boundary: invocation == 80% -> does NOT trigger rule 3
        label_inv80, _ = SR.recommend(
            {
                "skill": "ok-inv",
                "verdict": "inconclusive",
                "invocation_rate": 0.80,
                "t_skill_p90": 45.0,
            }
        )
        self.assertEqual(label_inv80, SR.REC_KEEP)

        # Branch 3b: inconclusive with t_skill p90 > 45s
        label_slow, reason_slow = SR.recommend(
            {
                "skill": "slow-discovery",
                "verdict": "inconclusive",
                "invocation_rate": 0.90,
                "t_skill_p90": 48.2,
            }
        )
        self.assertEqual(label_slow, SR.REC_FIX)
        self.assertIn("rule 3 (fix)", reason_slow)
        self.assertIn("t_skill p90 48.2s > 45s", reason_slow)

        # Branch 3c: improver has an accepted candidate
        label_cand, reason_cand = SR.recommend(
            {
                "skill": "improved-skill",
                "verdict": "helps",
                "lift": 0.20,
                "accepted_candidate": True,
            }
        )
        self.assertEqual(label_cand, SR.REC_FIX)
        self.assertIn("rule 3 (fix)", reason_cand)
        self.assertIn("improver has an accepted candidate", reason_cand)

    def test_rule_4_keep_branches(self) -> None:
        # Branch 4a: verdict == "helps"
        label_helps, reason_helps = SR.recommend(
            {
                "skill": "great-skill",
                "ceiling_tasks": 1,
                "total_tasks": 4,
                "verdict": "helps",
                "lift": 0.35,
            }
        )
        self.assertEqual(label_helps, SR.REC_KEEP)
        self.assertIn("rule 4 (keep): verdict is helps (lift +0.35)", reason_helps)

        # Branch 4b: inconclusive with >= 15% tokens saving
        label_tok_save, reason_tok_save = SR.recommend(
            {
                "skill": "token-saver",
                "verdict": "inconclusive",
                "tokens_delta_pct": -18.0,
                "wall_delta_pct": -5.0,
                "invocation_rate": 0.95,
                "t_skill_p90": 8.0,
            }
        )
        self.assertEqual(label_tok_save, SR.REC_KEEP)
        self.assertIn("rule 4 (keep)", reason_tok_save)
        self.assertIn("tokens p50 -18% (<= -15%)", reason_tok_save)

        # Branch 4c: inconclusive with >= 15% wall-time saving
        label_wall_save, reason_wall_save = SR.recommend(
            {
                "skill": "time-saver",
                "verdict": "inconclusive",
                "tokens_delta_pct": -4.0,
                "wall_delta_pct": -15.0,
                "invocation_rate": 0.90,
                "t_skill_p90": 9.0,
            }
        )
        self.assertEqual(label_wall_save, SR.REC_KEEP)
        self.assertIn("wall p50 -15% (<= -15%)", reason_wall_save)

        # Branch 4d: fallback ("none of the above")
        label_fb, reason_fb = SR.recommend(
            {
                "skill": "neutral",
                "verdict": "inconclusive",
                "tokens_delta_pct": 3.0,
                "wall_delta_pct": 2.0,
                "invocation_rate": 0.85,
                "t_skill_p90": 15.0,
            }
        )
        self.assertEqual(label_fb, SR.REC_KEEP)
        self.assertIn("rule 4 (keep): none of the above (verdict inconclusive)", reason_fb)

    def test_precedence_when_multiple_rules_match(self) -> None:
        # Rule 1 > Rule 2 + Rule 3 + Rule 4
        lbl1, rsn1 = SR.recommend(
            {
                "ceiling_tasks": 3,
                "total_tasks": 4,
                "verdict": "hurts",
                "lift": -0.40,
                "invocation_rate": 0.25,
                "t_skill_p90": 90.0,
                "accepted_candidate": True,
            }
        )
        self.assertEqual(lbl1, SR.REC_NEEDS_HARDER_TASKS)
        self.assertIn("rule 1", rsn1)

        # Rule 2 (hurts) > Rule 3 (accepted_candidate) + Rule 4 (tokens saving)
        lbl2, rsn2 = SR.recommend(
            {
                "ceiling_tasks": 1,
                "total_tasks": 4,
                "verdict": "hurts",
                "lift": -0.20,
                "accepted_candidate": True,
                "tokens_delta_pct": -30.0,
            }
        )
        self.assertEqual(lbl2, SR.REC_CONSIDER_REMOVING)
        self.assertIn("rule 2", rsn2)

        # Rule 2 (inconclusive + overhead) > Rule 3 (low invocation + slow t_skill_p90 + candidate)
        lbl2b, rsn2b = SR.recommend(
            {
                "ceiling_tasks": 0,
                "total_tasks": 4,
                "verdict": "inconclusive",
                "tokens_delta_pct": 20.0,
                "wall_delta_pct": 18.0,
                "verify_gain": False,
                "invocation_rate": 0.50,
                "t_skill_p90": 60.0,
                "accepted_candidate": True,
            }
        )
        self.assertEqual(lbl2b, SR.REC_CONSIDER_REMOVING)
        self.assertIn("rule 2", rsn2b)

        # Rule 3 (low invocation) > Rule 4 (inconclusive with >= 15% token/wall saving)
        lbl3, rsn3 = SR.recommend(
            {
                "ceiling_tasks": 1,
                "total_tasks": 4,
                "verdict": "inconclusive",
                "invocation_rate": 0.60,
                "t_skill_p90": 10.0,
                "tokens_delta_pct": -25.0,
                "wall_delta_pct": -20.0,
            }
        )
        self.assertEqual(lbl3, SR.REC_FIX)
        self.assertIn("rule 3", rsn3)

        with self.assertRaises(TypeError):
            SR.recommend("not-a-mapping")  # type: ignore[arg-type]


class TestSummaryRowsAndHistoryTrend(unittest.TestCase):
    """Test `summary[]` rows from a v2 report, `history.jsonl` idempotence, sparklines, and `--summary-only`."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.suite_dir = self.tmp / "suite"
        self.suite_dir.mkdir(parents=True, exist_ok=True)
        (self.suite_dir / "plugins" / "p" / "skills" / "alpha").mkdir(parents=True)
        (self.suite_dir / "plugins" / "p" / "skills" / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: alpha\n---\n# alpha\n", encoding="utf-8"
        )
        (self.suite_dir / "tasks").mkdir(parents=True, exist_ok=True)
        (self.suite_dir / "manifest.sha256").write_text("", encoding="utf-8")
        self.suite_toml = self.suite_dir / "suite.toml"
        self.suite_toml.write_text(
            'name = "demo-suite"\n'
            'plugins_dir = "plugins"\n'
            'tasks_dir = "tasks"\n'
            'manifest = "manifest.sha256"\n',
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build_v2_results_root(self, root: Path, *, suite_hash: str = "hash1111aaaa") -> Path:
        # Skill 1: "alpha" -> helps on exec-a (keep), inconclusive low-inv on exec-b (fix)
        # Skill 2: "beta" -> ceiling on 2/3 tasks (needs harder tasks)
        # Suite family: "planning" -> hurts on exec-a, helps on exec-b (best_model = exec-b -> keep)
        without_a = {
            "alpha/t1": [{"n_passed": 1, "tokens": 1000, "t_total_s": 20.0}, {"n_passed": 1, "tokens": 1000, "t_total_s": 20.0}],
            "alpha/t2": [{"n_passed": 1, "tokens": 1000, "t_total_s": 20.0}, {"n_passed": 1, "tokens": 1000, "t_total_s": 20.0}],
            "beta/t1": [{"n_passed": 4, "tokens": 900, "t_total_s": 15.0}, {"n_passed": 4, "tokens": 900, "t_total_s": 15.0}],
            "beta/t2": [{"n_passed": 4, "tokens": 900, "t_total_s": 15.0}, {"n_passed": 4, "tokens": 900, "t_total_s": 15.0}],
            "beta/t3": [{"n_passed": 1, "tokens": 900, "t_total_s": 15.0}, {"n_passed": 1, "tokens": 900, "t_total_s": 15.0}],
        }
        with_a = {
            "alpha/t1": [{"n_passed": 4, "tokens": 800, "t_total_s": 16.0, "invoked": True, "t_skill_s": 2.5}, {"n_passed": 4, "tokens": 800, "t_total_s": 16.0, "invoked": True, "t_skill_s": 3.5}],
            "alpha/t2": [{"n_passed": 4, "tokens": 800, "t_total_s": 16.0, "invoked": True, "t_skill_s": 3.0}, {"n_passed": 4, "tokens": 800, "t_total_s": 16.0, "invoked": True, "t_skill_s": 3.0}],
            "beta/t1": [{"n_passed": 4, "tokens": 950, "t_total_s": 16.0, "invoked": True, "t_skill_s": 4.0}, {"n_passed": 4, "tokens": 950, "t_total_s": 16.0, "invoked": True, "t_skill_s": 4.0}],
            "beta/t2": [{"n_passed": 4, "tokens": 950, "t_total_s": 16.0, "invoked": True, "t_skill_s": 4.0}, {"n_passed": 4, "tokens": 950, "t_total_s": 16.0, "invoked": True, "t_skill_s": 4.0}],
            "beta/t3": [{"n_passed": 2, "tokens": 950, "t_total_s": 16.0, "invoked": True, "t_skill_s": 4.0}, {"n_passed": 2, "tokens": 950, "t_total_s": 16.0, "invoked": True, "t_skill_s": 4.0}],
        }
        without_b = {
            "alpha/t1": [{"n_passed": 2, "tokens": 1000, "t_total_s": 20.0}, {"n_passed": 2, "tokens": 1000, "t_total_s": 20.0}],
            "alpha/t2": [{"n_passed": 2, "tokens": 1000, "t_total_s": 20.0}, {"n_passed": 2, "tokens": 1000, "t_total_s": 20.0}],
            "beta/t1": [{"n_passed": 4, "tokens": 900, "t_total_s": 15.0}, {"n_passed": 4, "tokens": 900, "t_total_s": 15.0}],
            "beta/t2": [{"n_passed": 4, "tokens": 900, "t_total_s": 15.0}, {"n_passed": 4, "tokens": 900, "t_total_s": 15.0}],
            "beta/t3": [{"n_passed": 1, "tokens": 900, "t_total_s": 15.0}, {"n_passed": 1, "tokens": 900, "t_total_s": 15.0}],
        }
        with_b = {
            "alpha/t1": [{"n_passed": 2, "tokens": 1100, "t_total_s": 22.0, "invoked": False}, {"n_passed": 2, "tokens": 1100, "t_total_s": 22.0, "invoked": True, "t_skill_s": 6.0}],
            "alpha/t2": [{"n_passed": 2, "tokens": 1100, "t_total_s": 22.0, "invoked": False}, {"n_passed": 2, "tokens": 1100, "t_total_s": 22.0, "invoked": False}],
            "beta/t1": [{"n_passed": 4, "tokens": 920, "t_total_s": 15.0, "invoked": True, "t_skill_s": 3.0}, {"n_passed": 4, "tokens": 920, "t_total_s": 15.0, "invoked": True, "t_skill_s": 3.0}],
            "beta/t2": [{"n_passed": 4, "tokens": 920, "t_total_s": 15.0, "invoked": True, "t_skill_s": 3.0}, {"n_passed": 4, "tokens": 920, "t_total_s": 15.0, "invoked": True, "t_skill_s": 3.0}],
            "beta/t3": [{"n_passed": 1, "tokens": 920, "t_total_s": 15.0, "invoked": True, "t_skill_s": 3.0}, {"n_passed": 1, "tokens": 920, "t_total_s": 15.0, "invoked": True, "t_skill_s": 3.0}],
        }

        suite_base = {
            "planning/p1": [{"n_passed": 1, "tokens": 1000, "t_total_s": 20.0}, {"n_passed": 1, "tokens": 1000, "t_total_s": 20.0}],
            "planning/p2": [{"n_passed": 1, "tokens": 1000, "t_total_s": 20.0}, {"n_passed": 1, "tokens": 1000, "t_total_s": 20.0}],
        }
        suite_arm_a = {
            "planning/p1": [{"n_passed": 4, "tokens": 1100, "t_total_s": 22.0, "invoked": True, "t_skill_s": 2.0}, {"n_passed": 4, "tokens": 1100, "t_total_s": 22.0, "invoked": True, "t_skill_s": 4.0}],
            "planning/p2": [{"n_passed": 4, "tokens": 1100, "t_total_s": 22.0, "invoked": True, "t_skill_s": 3.0}, {"n_passed": 4, "tokens": 1100, "t_total_s": 22.0, "invoked": True, "t_skill_s": 3.0}],
        }
        suite_arm_b = {
            "planning/p1": [{"n_passed": 2, "tokens": 1200, "t_total_s": 25.0, "invoked": True, "t_skill_s": 5.0}, {"n_passed": 2, "tokens": 1200, "t_total_s": 25.0, "invoked": True, "t_skill_s": 5.0}],
            "planning/p2": [{"n_passed": 2, "tokens": 1200, "t_total_s": 25.0, "invoked": True, "t_skill_s": 5.0}, {"n_passed": 2, "tokens": 1200, "t_total_s": 25.0, "invoked": True, "t_skill_s": 5.0}],
        }

        _seed_cell_ns(root / "exec-a" / "without", "without", without_a, executor="exec-a", judge="judge-b", suite_hash=suite_hash)
        _seed_cell_ns(root / "exec-a" / "with", "with", with_a, executor="exec-a", judge="judge-b", suite_hash=suite_hash)
        _seed_cell_ns(root / "exec-a" / "baseline", "baseline", suite_base, executor="exec-a", judge="judge-b", suite_hash=suite_hash, arm_kind="suite")
        _seed_cell_ns(root / "exec-a" / "suite-demo-suite", "suite-demo-suite", suite_arm_a, executor="exec-a", judge="judge-b", suite_hash=suite_hash, arm_kind="suite")

        _seed_cell_ns(root / "exec-b" / "without", "without", without_b, executor="exec-b", judge="judge-a", suite_hash=suite_hash)
        _seed_cell_ns(root / "exec-b" / "with", "with", with_b, executor="exec-b", judge="judge-a", suite_hash=suite_hash)
        _seed_cell_ns(root / "exec-b" / "baseline", "baseline", suite_base, executor="exec-b", judge="judge-a", suite_hash=suite_hash, arm_kind="suite")
        _seed_cell_ns(root / "exec-b" / "suite-demo-suite", "suite-demo-suite", suite_arm_b, executor="exec-b", judge="judge-a", suite_hash=suite_hash, arm_kind="suite")
        return root

    def test_v2_report_summary_rows_markdown_and_html_first_page(self) -> None:
        results_root = self._build_v2_results_root(self.tmp / "runs" / "r1" / "results")
        rep = SR.build_report(results_root, draws=200, pilot=True, bootstrap_seed=7)

        self.assertIn("summary", rep)
        self.assertIn("recommendation_note", rep)
        summary = rep["summary"]
        # 2 skills (alpha, beta) + 1 suite family (planning)
        self.assertEqual(len(summary), 3)

        by_name = {r["name"]: r for r in summary}
        alpha = by_name["alpha"]
        self.assertEqual(alpha["kind"], "skill")
        self.assertEqual(alpha["skill"], "alpha")
        self.assertIsNone(alpha["family"])
        self.assertEqual(alpha["best_model"], "exec-a")
        self.assertEqual(alpha["recommendation"], SR.REC_KEEP)
        self.assertIn("rule 4 (keep)", alpha["reason"])
        self.assertAlmostEqual(alpha["lift"], 0.75)
        self.assertAlmostEqual(alpha["tokens_delta_pct"], -20.0)
        self.assertAlmostEqual(alpha["wall_delta_pct"], -20.0)
        self.assertAlmostEqual(alpha["invocation_rate"], 1.0)
        self.assertAlmostEqual(alpha["t_skill_p50"], 3.0)
        # Per-executor recommendations on alpha: exec-a -> keep, exec-b -> fix (inv 25% < 80%)
        self.assertEqual(alpha["by_executor"]["exec-a"]["recommendation"], SR.REC_KEEP)
        self.assertEqual(alpha["by_executor"]["exec-b"]["recommendation"], SR.REC_FIX)
        self.assertIn("invocation 25% < 80%", alpha["by_executor"]["exec-b"]["reason"])

        beta = by_name["beta"]
        self.assertEqual(beta["kind"], "skill")
        self.assertEqual(beta["skill"], "beta")
        self.assertEqual(beta["ceiling_tasks"], 2)
        self.assertEqual(beta["total_tasks"], 3)
        self.assertEqual(beta["recommendation"], SR.REC_NEEDS_HARDER_TASKS)
        self.assertIn("2/3 tasks (67%) at ceiling", beta["reason"])

        planning = by_name["planning"]
        self.assertEqual(planning["kind"], "suite")
        self.assertEqual(planning["family"], "planning")
        self.assertIsNone(planning["skill"])
        self.assertEqual(planning["best_model"], "exec-a")
        self.assertEqual(planning["recommendation"], SR.REC_KEEP)
        self.assertAlmostEqual(planning["lift"], 0.75)

        # summary.md rendering
        summary_md = SR.render_summary_markdown(rep)
        self.assertTrue(summary_md.startswith("# skillbench summary\n"))
        self.assertIn("## Skill recommendations", summary_md)
        self.assertIn(
            "| skill | recommendation | quality Δ [95% CI] | verdict | tokens Δ % "
            "| time Δ % | discovery (inv %, t_skill p50) | best model | reason |",
            summary_md,
        )
        self.assertIn("| alpha | `keep` | +0.75 [+0.75, +0.75] | helps | -20% | -20% | 100%, 3.0s | exec-a |", summary_md)
        self.assertIn("| beta | `needs harder tasks` |", summary_md)
        self.assertIn("## Suite vs built-in baseline (per family)", summary_md)
        self.assertIn("| planning | `keep` |", summary_md)
        self.assertIn("## Model efficiency", summary_md)
        self.assertIn("## Legend", summary_md)
        self.assertIn("Recommendation rule (precedence order, first match wins)", summary_md)

        # report.html: Summary is the first active tab, dashboard tabs follow
        page = SR.render_html(rep)
        self.assertIn(
            '<button type="button" data-tab="summary" aria-selected="true"', page
        )
        self.assertIn('<section data-panel="summary" class="mt-4 ">', page)
        self.assertIn('<section data-panel="skills" class="mt-4 hidden">', page)
        self.assertIn('data-recommendation="keep"', page)
        self.assertIn('data-recommendation="needs harder tasks"', page)
        self.assertIn('data-summary-legend="true"', page)
        self.assertIn("<svg", page)

    def test_history_append_show_idempotence_and_sparkline_svg(self) -> None:
        r1 = self._build_v2_results_root(
            self.tmp / "runs" / "r1" / "results", suite_hash="hash1111aaaa"
        )
        r2 = self._build_v2_results_root(
            self.tmp / "runs" / "r2" / "results", suite_hash="hash2222bbbb"
        )
        rep1 = SR.build_report(r1, draws=100, pilot=True, bootstrap_seed=7)
        rep2 = SR.build_report(r2, draws=100, pilot=True, bootstrap_seed=7)
        rep1_path = self.tmp / "runs" / "r1" / "report.json"
        rep2_path = self.tmp / "runs" / "r2" / "report.json"
        _write_json(rep1_path, rep1)
        _write_json(rep2_path, rep2)

        hist_path = SR.default_history_path(self.suite_toml)
        self.assertEqual(hist_path, self.suite_dir / "history.jsonl")

        # 1. Append rep1 on 2026-10-26 via CLI
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = SR.main(
                [
                    "history",
                    "append",
                    "--suite",
                    str(self.suite_toml),
                    "--report",
                    str(rep1_path),
                    "--date",
                    "2026-10-26",
                ]
            )
        self.assertEqual(rc, 0)
        self.assertTrue(hist_path.is_file())
        records_after_first = SR.read_history(hist_path)
        # 2 executors * (2 skills + 1 suite family) = 6 records
        self.assertEqual(len(records_after_first), 6)

        # 2. Re-append the exact same (date, suite_hash, executor, skill) -> idempotent!
        with contextlib.redirect_stdout(io.StringIO()):
            rc = SR.main(
                [
                    "history",
                    "append",
                    "--suite",
                    str(self.suite_toml),
                    "--report",
                    str(rep1_path),
                    "--date",
                    "2026-10-26",
                ]
            )
        self.assertEqual(rc, 0)
        records_after_dup = SR.read_history(hist_path)
        self.assertEqual(len(records_after_dup), 6)
        self.assertEqual(
            len(hist_path.read_text(encoding="utf-8").splitlines()),
            6,
        )

        # 3. Append rep1 on 2026-10-27 (same suite_hash, new date) -> +6 = 12 records
        SR.append_history(hist_path, rep1, date="2026-10-27")
        self.assertEqual(len(SR.read_history(hist_path)), 12)

        # 4. Append rep2 on 2026-10-28 (changed suite_hash -> hash2222bbbb) -> +6 = 18 records
        with contextlib.redirect_stdout(io.StringIO()):
            rc = SR.main(
                [
                    "history",
                    "append",
                    "--suite",
                    str(self.suite_toml),
                    "--report",
                    str(rep2_path),
                    "--date",
                    "2026-10-28",
                ]
            )
        self.assertEqual(rc, 0)
        all_records = SR.read_history(hist_path)
        self.assertEqual(len(all_records), 18)

        # 5. CLI `history show` (Markdown and JSON)
        show_md = io.StringIO()
        with contextlib.redirect_stdout(show_md):
            rc = SR.main(
                [
                    "history",
                    "show",
                    "--suite",
                    str(self.suite_toml),
                    "--skill",
                    "alpha",
                    "--executor",
                    "exec-a",
                ]
            )
        self.assertEqual(rc, 0)
        md_text = show_md.getvalue()
        self.assertIn("# skillbench history", md_text)
        self.assertIn("2026-10-26", md_text)
        self.assertIn("2026-10-27", md_text)
        self.assertIn("2026-10-28", md_text)
        # Hash change marker on 2026-10-28
        self.assertIn("hash2222bbbb Δ", md_text)

        show_json = io.StringIO()
        with contextlib.redirect_stdout(show_json):
            rc = SR.main(
                [
                    "history",
                    "show",
                    "--suite",
                    str(self.suite_toml),
                    "--skill",
                    "alpha",
                    "--executor",
                    "exec-a",
                    "--json",
                ]
            )
        self.assertEqual(rc, 0)
        parsed_json = json.loads(show_json.getvalue())
        self.assertEqual(len(parsed_json), 3)

        # 6. Sparkline series & inline SVG
        series = SS.sparkline_series(hist_path, skill="alpha", executor="exec-a")
        self.assertEqual(series.dates, ["2026-10-26", "2026-10-27", "2026-10-28"])
        self.assertEqual(
            series.suite_hashes,
            ["hash1111aaaa", "hash1111aaaa", "hash2222bbbb"],
        )
        self.assertEqual(len(series.lifts), 3)
        self.assertEqual(len(series.tokens_delta_pcts), 3)
        # Index 2 changed suite_hash from hash1111aaaa -> hash2222bbbb
        self.assertEqual(series.hash_change_indices, [2])
        self.assertFalse(series.points[0]["hash_changed"])
        self.assertFalse(series.points[1]["hash_changed"])
        self.assertTrue(series.points[2]["hash_changed"])

        svg = SS.render_sparkline_svg(series)
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn('class="sparkline-lift"', svg)
        self.assertIn('class="sparkline-tokens"', svg)
        self.assertIn('class="sparkline-hash-tick" data-hash-change="2"', svg)
        self.assertIn('data-suite-hash="hash2222bbbb"', svg)

    def test_summary_only_cli_writes_only_summary_md_and_summary_html(self) -> None:
        results_root = self._build_v2_results_root(self.tmp / "runs" / "r1" / "results")
        out_dir = self.tmp / "summary-only-out"

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = SR.main(
                [
                    "--results-root",
                    str(results_root),
                    "--suite",
                    str(self.suite_toml),
                    "--draws",
                    "100",
                    "--pilot",
                    "--summary-only",
                    "--out-dir",
                    str(out_dir),
                ]
            )
        self.assertEqual(rc, 0)
        written_files = sorted(p.name for p in out_dir.iterdir())
        self.assertEqual(written_files, ["summary.html", "summary.md"])

        md_text = (out_dir / "summary.md").read_text(encoding="utf-8")
        html_text = (out_dir / "summary.html").read_text(encoding="utf-8")
        self.assertIn("# skillbench summary", md_text)
        self.assertIn("| alpha | `keep` |", md_text)
        self.assertIn("| beta | `needs harder tasks` |", md_text)
        self.assertTrue(html_text.startswith("<!DOCTYPE html>"))
        self.assertIn("<title>skillbench summary</title>", html_text)
        self.assertIn('data-recommendation="keep"', html_text)
        self.assertIn('data-recommendation="needs harder tasks"', html_text)
        self.assertEqual(html_text.count("<script"), 1)  # Only Tailwind script, no tab switcher needed


if __name__ == "__main__":
    unittest.main()
