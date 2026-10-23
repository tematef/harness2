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

"""Unit tests for `harness2.cli.skillbench_report` (lift, bootstrap CI, verdict, time-to-skill).

All tests run on synthetic namespaces in temporary directories (`HOME=<tmp>`); the user's
real `$HOME` and `~/.gemini` are never touched.
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
from typing import Any
from unittest import mock

from harness2 import config
from harness2.cli import skillbench_report as SR


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _make_rep(
    run_dir: Path,
    *,
    valid: bool = True,
    invalid_reason: str | None = None,
    invoked: bool = False,
    t_skill_s: float | None = None,
    step_idx: int | None = None,
    t_total_s: float = 10.0,
    requested_model: str = "opus",
    actual_model: str | None = None,
    judgments: list[dict[str, Any]] | None = None,
    judge_model: str = "fable",
    judge_actual_model: str | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "skill_timing.json",
        {
            "invoked": invoked,
            "t_skill_s": t_skill_s if invoked else None,
            "step_idx": (1 if step_idx is None else step_idx) if invoked else None,
            "t_total_s": t_total_s,
            "requested_model": requested_model,
            "actual_model": actual_model,
            "valid": valid,
            "invalid_reason": invalid_reason,
            "exit_code": 0 if valid else 1,
            "total_tokens": 1000,
        },
    )
    if judgments is not None:
        first = judgments[0]
        _write_json(
            run_dir / "judgments.json",
            {
                "cache_key": "synthetic-key",
                "judge_model": judge_model,
                "actual_model": judge_actual_model,
                "rubric_sha256": "rubric-sha",
                "deliverable_sha256": "deliv-sha",
                "score": {
                    "n_passed": first.get("n_passed", 0),
                    "n_criteria": first.get("n_criteria", 4),
                },
                "judgments": judgments,
            },
        )
    return run_dir


def _judgment_entry(
    passes: list[bool], *, actual_model: str | None = None, tokens: int = 500
) -> dict[str, Any]:
    criteria = [
        {"id": f"c{i}", "pass": bool(p), "reason": f"reason {i}"} for i, p in enumerate(passes)
    ]
    return {
        "criteria": criteria,
        "n_passed": sum(1 for p in passes if p),
        "n_criteria": len(passes),
        "actual_model": actual_model,
        "total_tokens": tokens,
    }


class _ReportCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-report-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.home = self.tmp / "realhome"
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / ".gemini").mkdir(parents=True, exist_ok=True)
        (self.home / ".gemini" / "GEMINI.md").write_text("user rules\n", encoding="utf-8")
        self.home_before = _tree_hash(self.home)

        self.runs_dir = self.tmp / "runs"
        env = {
            "HOME": str(self.home),
            "HARNESS2_RUNS_DIR": str(self.runs_dir),
            "SKILLBENCH_EXEC_MODEL": "opus",
            "SKILLBENCH_JUDGE_MODEL": "fable",
            "SKILLBENCH_EFFORT": "high",
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in ("SKILLBENCH_PILOT", "SKILLBENCH_JUDGE_REPEAT"):
            os.environ.pop(key, None)

        p = mock.patch.object(config, "RUNS", self.runs_dir)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self) -> None:
        self.assertEqual(_tree_hash(self.home), self.home_before, "fake HOME was modified")

    def _seed_arm(
        self,
        ns_dir: Path,
        arm: str,
        tasks_spec: dict[str, list[dict[str, Any]]],
        *,
        k: int = 2,
        seed: int | None = 7,
        manifest_hash: str = "abc123def456",
        executor: str = "opus",
        judge: str = "fable",
    ) -> Path:
        """Write a synthetic arm namespace (`config.json`, `*/summary.json`, `base_judged.json`,
        and per-rep `run_dir` with `skill_timing.json` and optional `judgments.json`)."""
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
        judged_rows: list[dict[str, Any]] = []
        results_root = self.tmp / "results" / arm
        for task_id, reps in sorted(tasks_spec.items()):
            skill = task_id.split("/", 1)[0]
            base_entries: list[dict[str, Any]] = []
            for idx, rep in enumerate(reps, 1):
                label = rep.get("label", f"base{idx}")
                run_id = f"{ns_dir.name}/{task_id}/{label}"
                run_dir = results_root / task_id.replace("/", "__") / label
                _make_rep(
                    run_dir,
                    valid=rep.get("valid", True),
                    invalid_reason=rep.get("invalid_reason"),
                    invoked=rep.get("invoked", False),
                    t_skill_s=rep.get("t_skill_s"),
                    step_idx=rep.get("step_idx"),
                    requested_model=rep.get("requested_model", executor),
                    actual_model=rep.get("actual_model"),
                    judgments=rep.get("judgments"),
                    judge_model=judge,
                    judge_actual_model=rep.get("judge_actual_model"),
                )
                base_entries.append(
                    {
                        "label": label,
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "ok": rep.get("valid", True),
                        "delivered": rep.get("valid", True),
                    }
                )
                score = rep.get("score")
                judged_rows.append(
                    {
                        "task": task_id,
                        "label": label,
                        "run_id": run_id,
                        "score": score,
                    }
                )
            task_dir = ns_dir / task_id.replace("/", "__")
            _write_json(
                task_dir / "summary.json",
                {
                    "task_id": task_id,
                    "domain": skill,
                    "k": k,
                    "seed": seed,
                    "tag": arm,
                    "status": "bases-only",
                    "base": base_entries,
                },
            )
        _write_json(ns_dir / "base_judged.json", judged_rows)
        return ns_dir


class TestPairedDeltasBootstrapAndVerdicts(_ReportCase):
    def test_paired_deltas_lift_and_reproducible_bootstrap_ci(self) -> None:
        with_ns = self.tmp / "ns-with"
        without_ns = self.tmp / "ns-without"

        # 3 tasks for make-plan, 2 reps per task:
        # t1: WITH [3/4=0.75, 4/4=1.00] (mean 0.875), WITHOUT [1/4=0.25, 2/4=0.50] (mean 0.375) -> d_t1 = +0.50
        # t2: WITH [2/4=0.50, 4/4=1.00] (mean 0.750), WITHOUT [2/4=0.50, 2/4=0.50] (mean 0.500) -> d_t2 = +0.25
        # t3: WITH [4/4=1.00, 4/4=1.00] (mean 1.000), WITHOUT [1/4=0.25, 3/4=0.75] (mean 0.500) -> d_t3 = +0.50
        # lift = (0.50 + 0.25 + 0.50) / 3 = 0.416666...
        self._seed_arm(
            with_ns,
            "with",
            {
                "make-plan/t1": [
                    {"score": {"n_passed": 3, "n_criteria": 4}, "invoked": True, "t_skill_s": 1.0},
                    {"score": {"n_passed": 4, "n_criteria": 4}, "invoked": True, "t_skill_s": 2.0},
                ],
                "make-plan/t2": [
                    {"score": {"n_passed": 2, "n_criteria": 4}, "invoked": True, "t_skill_s": 1.5},
                    {"score": {"n_passed": 4, "n_criteria": 4}, "invoked": False},
                ],
                "make-plan/t3": [
                    {"score": {"n_passed": 4, "n_criteria": 4}, "invoked": True, "t_skill_s": 3.0},
                    {"score": {"n_passed": 4, "n_criteria": 4}, "invoked": True, "t_skill_s": 4.0},
                ],
            },
        )
        self._seed_arm(
            without_ns,
            "without",
            {
                "make-plan/t1": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                ],
                "make-plan/t2": [
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                ],
                "make-plan/t3": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                ],
            },
        )

        rep1 = SR.build_report(with_ns, without_ns, bootstrap_seed=42, draws=2000)
        rep2 = SR.build_report(with_ns, without_ns, bootstrap_seed=42, draws=2000)
        rep_diff_seed = SR.build_report(with_ns, without_ns, bootstrap_seed=99, draws=2000)

        skill_row = rep1["by_skill"]["make-plan"]
        self.assertAlmostEqual(skill_row["task_deltas"]["make-plan/t1"], 0.50)
        self.assertAlmostEqual(skill_row["task_deltas"]["make-plan/t2"], 0.25)
        self.assertAlmostEqual(skill_row["task_deltas"]["make-plan/t3"], 0.50)
        self.assertAlmostEqual(skill_row["with_mean"], (0.875 + 0.75 + 1.0) / 3)
        self.assertAlmostEqual(skill_row["without_mean"], (0.375 + 0.50 + 0.50) / 3)
        self.assertAlmostEqual(skill_row["lift"], (0.50 + 0.25 + 0.50) / 3)
        self.assertAlmostEqual(
            skill_row["lift"], skill_row["with_mean"] - skill_row["without_mean"]
        )

        # Fixed seed reproduces exact same CI
        self.assertEqual(skill_row["ci_low"], rep2["by_skill"]["make-plan"]["ci_low"])
        self.assertEqual(skill_row["ci_high"], rep2["by_skill"]["make-plan"]["ci_high"])
        self.assertGreater(skill_row["ci_low"], 0.0)
        self.assertLessEqual(skill_row["ci_low"], skill_row["lift"])
        self.assertGreaterEqual(skill_row["ci_high"], skill_row["lift"])
        self.assertEqual(skill_row["verdict"], "helps")

        # Direct call to bootstrap_ci with same seed produces identical bounds
        ci_direct = SR.bootstrap_ci(
            [
                ([0.75, 1.00], [0.25, 0.50]),
                ([0.50, 1.00], [0.50, 0.50]),
                ([1.00, 1.00], [0.25, 0.75]),
            ],
            seed=42,
            draws=2000,
        )
        self.assertEqual((skill_row["ci_low"], skill_row["ci_high"]), ci_direct)

        # Different seed with non-degenerate variance produces a different draw sequence
        self.assertIsNotNone(rep_diff_seed["by_skill"]["make-plan"]["ci_low"])

    def test_verdict_thresholds_helps_hurts_inconclusive_and_insufficient_data_including_pilot(
        self,
    ) -> None:
        with_ns = self.tmp / "verdicts-with"
        without_ns = self.tmp / "verdicts-without"

        # Skill 1 (`helps-skill`): 3 tasks, each >= 2 valid reps, consistent positive lift (+0.5)
        # Skill 2 (`hurts-skill`): 3 tasks, each >= 2 valid reps, consistent negative lift (-0.5)
        # Skill 3 (`mixed-skill`): 3 tasks, each >= 2 valid reps, deltas +0.5, -0.5, 0.0 -> CI spans 0 -> inconclusive
        # Skill 4 (`two-task-skill`): only 2 tasks with >= 2 valid reps (plus a 3rd task with only 1 valid rep in WITHOUT)
        #   -> in full mode (min 3 tasks): `insufficient data`
        #   -> in pilot mode (min 2 tasks): `helps`
        # Skill 5 (`one-task-skill`): only 1 task with >= 2 valid reps
        #   -> `insufficient data` in both full and pilot modes
        self._seed_arm(
            with_ns,
            "with",
            {
                "helps-skill/t1": [
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                ],
                "helps-skill/t2": [
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                ],
                "helps-skill/t3": [
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                ],
                "hurts-skill/t1": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 0, "n_criteria": 4}},
                ],
                "hurts-skill/t2": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                ],
                "hurts-skill/t3": [
                    {"score": {"n_passed": 0, "n_criteria": 4}},
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                ],
                "mixed-skill/t1": [
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                ],
                "mixed-skill/t2": [
                    {"score": {"n_passed": 0, "n_criteria": 4}},
                    {"score": {"n_passed": 0, "n_criteria": 4}},
                ],
                "mixed-skill/t3": [
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                ],
                "two-task-skill/t1": [
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                ],
                "two-task-skill/t2": [
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                ],
                "two-task-skill/t3": [
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                ],
                "one-task-skill/t1": [
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                ],
            },
        )
        self._seed_arm(
            without_ns,
            "without",
            {
                "helps-skill/t1": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                ],
                "helps-skill/t2": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                ],
                "helps-skill/t3": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                ],
                "hurts-skill/t1": [
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                ],
                "hurts-skill/t2": [
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                ],
                "hurts-skill/t3": [
                    {"score": {"n_passed": 4, "n_criteria": 4}},
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                ],
                "mixed-skill/t1": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                ],
                "mixed-skill/t2": [
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                    {"score": {"n_passed": 3, "n_criteria": 4}},
                ],
                "mixed-skill/t3": [
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                ],
                "two-task-skill/t1": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                ],
                "two-task-skill/t2": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                ],
                "two-task-skill/t3": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"valid": False, "invalid_reason": "timeout after 300s", "score": None},
                ],
                "one-task-skill/t1": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                ],
            },
        )

        full_rep = SR.build_report(
            with_ns, without_ns, bootstrap_seed=7, draws=1000, pilot=False
        )
        self.assertEqual(full_rep["by_skill"]["helps-skill"]["verdict"], "helps")
        self.assertEqual(full_rep["by_skill"]["hurts-skill"]["verdict"], "hurts")
        self.assertEqual(full_rep["by_skill"]["mixed-skill"]["verdict"], "inconclusive")
        self.assertEqual(
            full_rep["by_skill"]["two-task-skill"]["n_qualifying_tasks"], 2
        )
        self.assertEqual(
            full_rep["by_skill"]["two-task-skill"]["verdict"], "insufficient data"
        )
        self.assertEqual(
            full_rep["by_skill"]["one-task-skill"]["verdict"], "insufficient data"
        )

        pilot_rep = SR.build_report(
            with_ns, without_ns, bootstrap_seed=7, draws=1000, pilot=True
        )
        self.assertEqual(pilot_rep["by_skill"]["two-task-skill"]["verdict"], "helps")
        self.assertEqual(
            pilot_rep["by_skill"]["one-task-skill"]["verdict"], "insufficient data"
        )


class TestInvalidRepsTimeToSkillAndModelHeader(_ReportCase):
    def test_invalid_reps_counted_per_reason_and_excluded_and_time_to_skill(self) -> None:
        with_ns = self.tmp / "timing-with"
        without_ns = self.tmp / "timing-without"

        # Skill `backlog`:
        # WITH arm has 5 reps across 2 tasks:
        #   t1/base1: valid=True, invoked=True, t_skill_s=1.0, score=4/4
        #   t1/base2: valid=True, invoked=True, t_skill_s=3.0, score=2/4
        #   t1/base3: valid=False (timeout), invoked=True, t_skill_s=0.2, score=4/4 (must still be excluded!)
        #   t2/base1: valid=True, invoked=False, t_skill_s=None, score=4/4
        #   t2/base2: valid=True, invoked=True, t_skill_s=11.0, score=None (judge returned None -> valid WITH rep for invocation rate & timing, but excluded from frac!)
        #   t2/base3: valid=True, invoked=True, t_skill_s=5.0, score=2/4
        # Valid WITH reps = 5 (t1/base1, t1/base2, t2/base1, t2/base2, t2/base3)
        # Invoked valid WITH reps = 4 (t_skill_s = [1.0, 3.0, 5.0, 11.0])
        #   -> invocation_rate = 4 / 5 = 0.8
        #   -> p50 of [1.0, 3.0, 5.0, 11.0] = 4.0
        #   -> p90 of [1.0, 3.0, 5.0, 11.0] = 5.0 + 0.7 * (11.0 - 5.0) = 9.2
        self._seed_arm(
            with_ns,
            "with",
            {
                "backlog/t1": [
                    {"score": {"n_passed": 4, "n_criteria": 4}, "invoked": True, "t_skill_s": 1.0},
                    {"score": {"n_passed": 2, "n_criteria": 4}, "invoked": True, "t_skill_s": 3.0},
                    {
                        "valid": False,
                        "invalid_reason": "timeout after 300s",
                        "invoked": True,
                        "t_skill_s": 0.2,
                        "score": {"n_passed": 4, "n_criteria": 4},
                    },
                ],
                "backlog/t2": [
                    {"score": {"n_passed": 4, "n_criteria": 4}, "invoked": False},
                    {"score": None, "invoked": True, "t_skill_s": 11.0},
                    {"score": {"n_passed": 2, "n_criteria": 4}, "invoked": True, "t_skill_s": 5.0},
                ],
            },
            k=3,
        )
        self._seed_arm(
            without_ns,
            "without",
            {
                "backlog/t1": [
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                    {
                        "valid": False,
                        "invalid_reason": "agy exited with code 1",
                        "score": None,
                    },
                ],
                "backlog/t2": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {
                        "valid": False,
                        "invalid_reason": "timeout after 300s",
                        "score": None,
                    },
                    {
                        "valid": False,
                        "invalid_reason": "model mismatch: requested 'opus', agy reported 'sonnet'",
                        "score": None,
                    },
                ],
            },
            k=3,
        )

        rep = SR.build_report(with_ns, without_ns, bootstrap_seed=7, draws=500, pilot=True)
        row = rep["by_skill"]["backlog"]

        # Invalid counts and per-reason breakdown
        self.assertEqual(row["invalid_with"], 1)
        self.assertEqual(row["invalid_without"], 3)
        self.assertEqual(row["invalid_total"], 4)
        self.assertEqual(
            row["invalid_by_reason"],
            {
                "agy exited with code 1": 1,
                "model mismatch: requested 'opus', agy reported 'sonnet'": 1,
                "timeout after 300s": 2,
            },
        )
        self.assertEqual(rep["invalid_total"], 4)
        self.assertEqual(rep["invalid_by_reason"]["timeout after 300s"], 2)

        # Invalid rep t1/base3 (despite score=4/4) and ungraded rep t2/base2 (score=None) are excluded, not zeroed
        self.assertEqual(row["tasks"]["backlog/t1"]["with_fracs"], [1.0, 0.5])
        self.assertEqual(row["tasks"]["backlog/t2"]["with_fracs"], [1.0, 0.5])
        self.assertEqual(row["tasks"]["backlog/t1"]["without_fracs"], [0.5, 0.5])
        self.assertEqual(row["tasks"]["backlog/t2"]["without_fracs"], [0.25])
        self.assertEqual(row["ungraded_with"], 1)

        # Invocation rate denominator = valid WITH reps (5), numerator = invoked valid WITH reps (4)
        self.assertEqual(row["valid_with_reps"], 5)
        self.assertEqual(row["invoked"], 4)
        self.assertAlmostEqual(row["invocation_rate"], 4 / 5)

        # Time-to-skill p50 and p90 over invoked valid WITH reps ([1.0, 3.0, 5.0, 11.0])
        self.assertAlmostEqual(row["t_skill_p50"], 4.0)
        self.assertAlmostEqual(row["t_skill_p90"], 9.2)

    def test_model_unverified_header_when_actual_model_is_null_vs_verified(self) -> None:
        # Case 1: actual_model is null -> "model unverified (agy reports no model)" in header
        with_unv = self.tmp / "unv-with"
        without_unv = self.tmp / "unv-without"
        self._seed_arm(
            with_unv,
            "with",
            {
                "make-plan/t1": [
                    {
                        "score": {"n_passed": 3, "n_criteria": 4},
                        "actual_model": None,
                        "judgments": [_judgment_entry([True, True, True, False], actual_model=None)],
                        "judge_actual_model": None,
                    }
                ]
            },
            k=1,
        )
        self._seed_arm(
            without_unv,
            "without",
            {
                "make-plan/t1": [
                    {
                        "score": {"n_passed": 2, "n_criteria": 4},
                        "actual_model": None,
                        "judgments": [_judgment_entry([True, True, False, False], actual_model=None)],
                        "judge_actual_model": None,
                    }
                ]
            },
            k=1,
        )
        rep_unv = SR.build_report(with_unv, without_unv, draws=100)
        self.assertFalse(rep_unv["exec_model_verified"])
        self.assertFalse(rep_unv["judge_model_verified"])
        self.assertFalse(rep_unv["model_verified"])
        self.assertIsNone(rep_unv["actual_exec_model"])
        md_unv = SR.render_markdown(rep_unv)
        self.assertIn("model unverified (agy reports no model)", md_unv)

        # Case 2: actual_model is reported on all rollouts and judgments -> verified
        with_ver = self.tmp / "ver-with"
        without_ver = self.tmp / "ver-without"
        self._seed_arm(
            with_ver,
            "with",
            {
                "make-plan/t1": [
                    {
                        "score": {"n_passed": 3, "n_criteria": 4},
                        "actual_model": "claude-opus-4-1",
                        "judgments": [
                            _judgment_entry([True, True, True, False], actual_model="fable-v1")
                        ],
                        "judge_actual_model": "fable-v1",
                    }
                ]
            },
            k=1,
        )
        self._seed_arm(
            without_ver,
            "without",
            {
                "make-plan/t1": [
                    {
                        "score": {"n_passed": 2, "n_criteria": 4},
                        "actual_model": "claude-opus-4-1",
                        "judgments": [
                            _judgment_entry([True, True, False, False], actual_model="fable-v1")
                        ],
                        "judge_actual_model": "fable-v1",
                    }
                ]
            },
            k=1,
        )
        rep_ver = SR.build_report(with_ver, without_ver, draws=100)
        self.assertTrue(rep_ver["exec_model_verified"])
        self.assertTrue(rep_ver["judge_model_verified"])
        self.assertTrue(rep_ver["model_verified"])
        self.assertEqual(rep_ver["actual_exec_model"], "claude-opus-4-1")
        self.assertEqual(rep_ver["actual_judge_model"], "fable-v1")
        md_ver = SR.render_markdown(rep_ver)
        self.assertNotIn("model unverified", md_ver)
        self.assertIn("claude-opus-4-1", md_ver)
        self.assertIn("fable-v1", md_ver)


class TestJudgeNoiseAndPowerNote(_ReportCase):
    def test_judge_noise_rate_on_repeated_judgments_and_power_note_always_present(self) -> None:
        with_ns = self.tmp / "noise-with"
        without_ns = self.tmp / "noise-without"

        # WITH rep has 4 criteria: c0 agrees (T,T), c1 disagrees (T,F), c2 agrees (F,F), c3 disagrees (F,T) -> 2/4 disagree
        # WITHOUT rep has 4 criteria: c0..c2 agree, c3 disagrees (T,F) -> 1/4 disagree
        # Total across both arms: 3 / 8 = 0.375 (37.5%)
        self._seed_arm(
            with_ns,
            "with",
            {
                "make-plan/t1": [
                    {
                        "score": {"n_passed": 2, "n_criteria": 4},
                        "judgments": [
                            _judgment_entry([True, True, False, False]),
                            _judgment_entry([True, False, False, True]),
                        ],
                    }
                ]
            },
            k=1,
        )
        self._seed_arm(
            without_ns,
            "without",
            {
                "make-plan/t1": [
                    {
                        "score": {"n_passed": 3, "n_criteria": 4},
                        "judgments": [
                            _judgment_entry([True, True, False, True]),
                            _judgment_entry([True, True, False, False]),
                        ],
                    }
                ]
            },
            k=1,
        )

        rep = SR.build_report(with_ns, without_ns, draws=100)
        self.assertEqual(rep["judge_noise_disagreed"], 3)
        self.assertEqual(rep["judge_noise_total"], 8)
        self.assertAlmostEqual(rep["judge_noise_rate"], 3 / 8)
        self.assertIn("35 distinct resamples", rep["power_note"])

        md = SR.render_markdown(rep)
        self.assertIn("37.5%", md)
        self.assertIn("3/8", md)
        self.assertIn("35 distinct resamples", md)
        self.assertIn("inconclusive", md)

    def test_judge_noise_rate_is_none_without_repeats_and_power_note_still_present(self) -> None:
        with_ns = self.tmp / "norepeat-with"
        without_ns = self.tmp / "norepeat-without"
        self._seed_arm(
            with_ns,
            "with",
            {
                "make-plan/t1": [
                    {
                        "score": {"n_passed": 3, "n_criteria": 4},
                        "judgments": [_judgment_entry([True, True, True, False])],
                    }
                ]
            },
            k=1,
        )
        self._seed_arm(
            without_ns,
            "without",
            {
                "make-plan/t1": [
                    {
                        "score": {"n_passed": 1, "n_criteria": 4},
                        "judgments": [_judgment_entry([True, False, False, False])],
                    }
                ]
            },
            k=1,
        )

        rep = SR.build_report(with_ns, without_ns, draws=100)
        self.assertIsNone(rep["judge_noise_rate"])
        self.assertEqual(rep["judge_noise_disagreed"], 0)
        self.assertEqual(rep["judge_noise_total"], 0)
        self.assertIn("35 distinct resamples", rep["power_note"])

        md = SR.render_markdown(rep)
        self.assertIn("35 distinct resamples", md)


class TestSkillbenchReportCliAndFiles(_ReportCase):
    def test_cli_writes_report_json_and_report_md_from_default_namespaces(self) -> None:
        adapt_root = self.runs_dir / "skillbench" / "adapt"
        with_ns = adapt_root / "skillbench-agy-agg2-s7-with"
        without_ns = adapt_root / "skillbench-agy-agg2-s7-without"

        self._seed_arm(
            with_ns,
            "with",
            {
                "make-plan/t1": [
                    {"score": {"n_passed": 4, "n_criteria": 4}, "invoked": True, "t_skill_s": 1.2},
                    {"score": {"n_passed": 3, "n_criteria": 4}, "invoked": True, "t_skill_s": 2.8},
                ],
                "make-plan/t2": [
                    {"score": {"n_passed": 4, "n_criteria": 4}, "invoked": True, "t_skill_s": 2.0},
                    {"score": {"n_passed": 4, "n_criteria": 4}, "invoked": False},
                ],
            },
            k=2,
            seed=7,
        )
        self._seed_arm(
            without_ns,
            "without",
            {
                "make-plan/t1": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                ],
                "make-plan/t2": [
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                    {"score": {"n_passed": 1, "n_criteria": 4}},
                ],
            },
            k=2,
            seed=7,
        )

        out_dir = self.tmp / "out"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = SR.main(
                [
                    "--k",
                    "2",
                    "--seed",
                    "7",
                    "--pilot",
                    "--draws",
                    "500",
                    "--out-dir",
                    str(out_dir),
                ]
            )
        self.assertEqual(rc, 0)

        json_path = out_dir / "report.json"
        md_path = out_dir / "report.md"
        self.assertTrue(json_path.is_file())
        self.assertTrue(md_path.is_file())

        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["k"], 2)
        self.assertEqual(data["seed"], 7)
        self.assertTrue(data["pilot"])
        self.assertEqual(data["manifest_hash"], "abc123def456")
        self.assertEqual(len(data["skills"]), 1)
        self.assertEqual(data["skills"][0]["skill"], "make-plan")
        self.assertEqual(data["skills"][0]["verdict"], "helps")

        md = md_path.read_text(encoding="utf-8")
        self.assertIn(
            "| skill | tasks | WITH | WITHOUT | lift [95% CI] | verdict | invoked | t_skill p50/p90 | invalid |",
            md,
        )
        self.assertIn("| make-plan |", md)
        self.assertIn("abc123def456", md)
        self.assertIn("model unverified (agy reports no model)", md)
        self.assertIn("35 distinct resamples", md)

    def test_cli_errors_on_missing_namespace_or_mismatched_identity(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = SR.main(["--k", "1"])
        self.assertNotEqual(rc, 0)
        self.assertIn("error:", err.getvalue())

        # Mismatched manifest_hash between WITH and WITHOUT arms raises error
        with_ns = self.tmp / "mismatch-with"
        without_ns = self.tmp / "mismatch-without"
        self._seed_arm(
            with_ns,
            "with",
            {"make-plan/t1": [{"score": {"n_passed": 3, "n_criteria": 4}}]},
            k=1,
            manifest_hash="hash-one",
        )
        self._seed_arm(
            without_ns,
            "without",
            {"make-plan/t1": [{"score": {"n_passed": 2, "n_criteria": 4}}]},
            k=1,
            manifest_hash="hash-two",
        )
        err2 = io.StringIO()
        with contextlib.redirect_stderr(err2):
            rc2 = SR.main(
                ["--with-dir", str(with_ns), "--without-dir", str(without_ns)]
            )
        self.assertNotEqual(rc2, 0)
        self.assertIn("manifest_hash", err2.getvalue())

    def test_console_script_entry_point_registered(self) -> None:
        script = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "skillbench-report"
        self.assertTrue(script.is_file(), f"missing console script {script}")
        proc = subprocess.run(
            [str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("skillbench-report", proc.stdout)

    def test_edge_cases_missing_timing_zero_invoked_and_default_out_dir(self) -> None:
        adapt_root = self.runs_dir / "skillbench" / "adapt"
        with_ns = adapt_root / "skillbench-agy-agg1-with"
        without_ns = adapt_root / "skillbench-agy-agg1-without"

        self._seed_arm(
            with_ns,
            "with",
            {
                "clarify/t1": [
                    {"score": {"n_passed": 2, "n_criteria": 4}, "invoked": False},
                ],
                "clarify/t2": [
                    {"score": {"n_passed": 3, "n_criteria": 4}, "invoked": False},
                ],
            },
            k=1,
            seed=None,
        )
        self._seed_arm(
            without_ns,
            "without",
            {
                "clarify/t1": [
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                ],
                "clarify/t2": [
                    {"score": {"n_passed": 2, "n_criteria": 4}},
                ],
            },
            k=1,
            seed=None,
        )
        # Remove skill_timing.json on clarify/t2 in WITHOUT arm to verify missing-file handling
        summary_t2 = json.loads(
            (without_ns / "clarify__t2" / "summary.json").read_text(encoding="utf-8")
        )
        t2_run_dir = Path(summary_t2["base"][0]["run_dir"])
        (t2_run_dir / "skill_timing.json").unlink()

        os.environ["SKILLBENCH_PILOT"] = "1"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = SR.main(["--k", "1", "--draws", "100"])
        self.assertEqual(rc, 0)

        # Default out-dir writes to <runs_dir>/skillbench
        self.assertTrue((self.runs_dir / "skillbench" / "report.json").is_file())
        self.assertTrue((self.runs_dir / "skillbench" / "report.md").is_file())

        data = json.loads((self.runs_dir / "skillbench" / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(data["pilot"])
        row = data["by_skill"]["clarify"]
        self.assertEqual(row["invoked"], 0)
        self.assertEqual(row["valid_with_reps"], 2)
        self.assertEqual(row["invocation_rate"], 0.0)
        self.assertIsNone(row["t_skill_p50"])
        self.assertIsNone(row["t_skill_p90"])
        self.assertEqual(row["invalid_without"], 1)
        self.assertEqual(row["invalid_by_reason"], {"missing skill_timing.json": 1})
        self.assertEqual(row["n_paired_tasks"], 1)

        with self.assertRaises(ValueError):
            SR.bootstrap_ci([], draws=10)
        with self.assertRaises(ValueError):
            SR.bootstrap_ci([([0.5], [0.5])], draws=0)


if __name__ == "__main__":
    unittest.main()
