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

"""Tests for Task 5: suite-mode arms (`baseline` vs `suite-<name>` vs extra `--suites`)."""

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

from harness2 import config
from harness2.benchmarks import skillbench as sb
from harness2.benchmarks.skillbench_suite import SuiteConfig, load_suite
from harness2.cli import _common as C
from harness2.cli import judge as judge_cli
from harness2.cli import run_sweep
from harness2.cli import skillbench_report as SR
from harness2.substrate import agy

STUB_SCRIPT = Path(__file__).resolve().parent / "fixtures" / "agy_stub.py"
AUTH = (".gemini/oauth_creds.json", ".gemini/google_accounts.json")


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_json(path: Path, data: Any) -> Path:
    return _write(path, json.dumps(data, indent=2) + "\n")


def _make_suite_tree(
    root: Path,
    suite_name: str,
    plugins: dict[str, list[str]],
    *,
    tasks_root: Path | None = None,
    families: dict[str, list[str]] | None = None,
) -> tuple[Path, SuiteConfig]:
    """Create a synthetic plugin checkout + `suite.toml` and return `(toml_path, SuiteConfig)`."""
    pdir = root / f"plugins-{suite_name}"
    for plug, skills in plugins.items():
        _write_json(pdir / plug / "plugin.json", {"name": f"{suite_name}-{plug}"})
        for s in skills:
            _write(
                pdir / plug / "skills" / s / "SKILL.md",
                f"---\nname: {s}\ndescription: Skill {s} from {suite_name}\n---\n# {s}\nBody of {s}.\n",
            )
    lines = [
        f'name = "{suite_name}"',
        f'plugins_dir = "{pdir.as_posix()}"',
        f"plugins = {json.dumps(sorted(plugins))}",
    ]
    if tasks_root is not None:
        lines.append(f'tasks_root = "{tasks_root.as_posix()}"')
    if families:
        lines.append("[families]")
        for fam, fam_skills in families.items():
            lines.append(f"{fam} = {json.dumps(fam_skills)}")
    toml_path = _write(root / f"{suite_name}.toml", "\n".join(lines) + "\n")
    return toml_path, load_suite(toml_path)


def _write_task(
    tasks_root: Path,
    folder: str,
    task_id: str,
    *,
    kind: str = "skill",
    skill: str | None = None,
    family: str | None = None,
) -> Path:
    dir_name = sb.SUITE_DIR if kind == "suite" else folder
    tdir = tasks_root / dir_name / task_id
    (tdir / "fixture").mkdir(parents=True, exist_ok=True)
    _write(tdir / "fixture" / "README.md", f"# Fixture for {dir_name}/{task_id}\n")
    toml_lines = [
        f'kind = "{kind}"',
    ]
    if kind == "suite":
        toml_lines.append(f'family = "{family or folder}"')
    else:
        toml_lines.append(f'target_skill = "{skill or folder}"')
    toml_lines += [
        'prompt = "Deliver the requested analysis in the final response."',
        "timeout = 60",
        "pilot = true",
        'forbidden_terms = ["SKILL.md"]',
    ]
    _write(tdir / "task.toml", "\n".join(toml_lines) + "\n")
    _write_json(
        tdir / "rubric.json",
        [
            {"id": "c1", "criterion": "First check"},
            {"id": "c2", "criterion": "Second check"},
            {"id": "c3", "criterion": "Third check"},
            {"id": "c4", "criterion": "Fourth check"},
        ],
    )
    return tdir


class _SuiteArmsCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-suite-arms-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.runs_dir = self.tmp / "runs"
        self.corpus = self.tmp / "corpus"
        _write_task(self.corpus, "plan-skill", "s1", kind="skill", skill="plan-skill")
        _write_task(self.corpus, "debug-skill", "s2", kind="skill", skill="debug-skill")
        _write_task(self.corpus, "planning", "p1", kind="suite", family="planning")
        _write_task(self.corpus, "analysis", "a1", kind="suite", family="analysis")
        sb.freeze(self.corpus)

        self.alpha_toml, self.alpha_suite = _make_suite_tree(
            self.tmp,
            "alpha",
            {"plan": ["plan-skill", "exec-skill"], "think": ["debug-skill"]},
            tasks_root=self.corpus,
            families={"planning": ["plan-skill", "exec-skill"], "analysis": ["debug-skill"]},
        )
        self.beta_toml, self.beta_suite = _make_suite_tree(
            self.tmp,
            "beta",
            {"core": ["beta-plan", "beta-debug"]},
            tasks_root=self.corpus,
            families={"planning": ["beta-plan"], "analysis": ["beta-debug"]},
        )
        self.gamma_toml, self.gamma_suite = _make_suite_tree(
            self.tmp,
            "gamma",
            {"core": ["gamma-plan"]},
            tasks_root=self.corpus,
            families={"planning": ["gamma-plan"]},
        )

        env = {
            "HOME": str(self.home),
            "HARNESS2_RUNS_DIR": str(self.runs_dir),
            "SKILLBENCH_SUITE": str(self.alpha_toml),
            "SKILLBENCH_EXEC_MODEL": "exec-m",
            "SKILLBENCH_JUDGE_MODEL": "judge-m",
        }
        patcher = mock.patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in ("SKILLBENCH_PILOT", "SKILLBENCH_PLUGINS_DIR", "SKILLBENCH_TASKS_ROOT"):
            os.environ.pop(k, None)
        p = mock.patch.object(config, "RUNS", self.runs_dir)
        p.start()
        self.addCleanup(p.stop)


class TestArmSpecAndParseArmLabel(_SuiteArmsCase):
    def test_parse_all_four_arm_labels(self) -> None:
        w = agy.parse_arm_label("with", suite=self.alpha_suite, target_skill="plan-skill")
        self.assertEqual(
            w,
            agy.ArmSpec(
                kind="skill",
                label="with",
                suite=self.alpha_suite,
                target_skill="plan-skill",
            ),
        )
        self.assertFalse(w.is_control)
        self.assertTrue(w.stages_plugins)

        wo = agy.parse_arm_label("without", suite=self.alpha_suite, target_skill="plan-skill")
        self.assertEqual(wo.kind, "skill")
        self.assertEqual(wo.label, "without")
        self.assertTrue(wo.is_control)
        self.assertTrue(wo.stages_plugins)

        base = agy.parse_arm_label("baseline", suite=self.alpha_suite)
        self.assertEqual(
            base,
            agy.ArmSpec(kind="suite", label="baseline", suite=None, target_skill=None),
        )
        self.assertTrue(base.is_control)
        self.assertFalse(base.stages_plugins)

        sa = agy.parse_arm_label("suite-alpha", suite=self.alpha_suite)
        self.assertEqual(
            sa,
            agy.ArmSpec(
                kind="suite", label="suite-alpha", suite=self.alpha_suite, target_skill=None
            ),
        )
        self.assertFalse(sa.is_control)
        self.assertTrue(sa.stages_plugins)

        sb_spec = agy.parse_arm_label(
            "suite-beta",
            suite=self.alpha_suite,
            extra_suites=[self.beta_toml, self.gamma_suite],
        )
        self.assertEqual(sb_spec.kind, "suite")
        self.assertEqual(sb_spec.label, "suite-beta")
        self.assertEqual(sb_spec.suite.name, "beta")

    def test_parse_arm_label_and_armspec_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "ArmSpec.kind"):
            agy.ArmSpec(kind="invalid", label="with")
        with self.assertRaisesRegex(ValueError, "ArmSpec.label"):
            agy.ArmSpec(kind="skill", label=" ")

        for bad in ("", "   ", "control", "treatment", "suite-", "suite-bad/name", "suite-a b"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "arm"):
                    agy.parse_arm_label(bad, suite=self.alpha_suite)

        with self.assertRaisesRegex(ValueError, "--suites"):
            agy.parse_arm_label("suite-beta", suite=self.alpha_suite)

        with self.assertRaisesRegex(ValueError, "target_skill is not valid"):
            agy.parse_arm_label("baseline", suite=self.alpha_suite, target_skill="plan-skill")
        with self.assertRaisesRegex(ValueError, "target_skill is not valid"):
            agy.parse_arm_label("suite-alpha", suite=self.alpha_suite, target_skill="plan-skill")
        with self.assertRaisesRegex(ValueError, "requires a non-empty target_skill"):
            agy.parse_arm_label("with", suite=self.alpha_suite, target_skill="")

    def test_common_arm_tag_and_make_adapter_validation(self) -> None:
        for lbl in ("with", "without", "baseline", "suite-alpha", "suite-beta"):
            self.assertEqual(C.arm_tag("skillbench", lbl), lbl)
        with self.assertRaisesRegex(ValueError, "arm label must be one of"):
            C.arm_tag("skillbench", "bogus")
        with self.assertRaisesRegex(ValueError, "applies to --bench skillbench only"):
            C.make_adapter("jb", arm="baseline")
        with self.assertRaisesRegex(ValueError, "applies to --bench skillbench only"):
            C.make_adapter("jb", suites=[self.beta_toml])

        ad = C.make_adapter("skillbench", arm="suite-beta", suites=[self.beta_toml])
        self.assertEqual(ad.arm, "suite-beta")
        self.assertEqual(ad.arm_spec.suite.name, "beta")


class TestStagePluginsAndMaterialise(_SuiteArmsCase):
    def test_stage_plugins_per_arm_kind(self) -> None:
        # 1. baseline -> empty plugins dir, no omitted paths
        dest_base = self.tmp / "stage-base"
        omitted_base = agy.stage_plugins(dest_base, "baseline", suite=self.alpha_suite)
        self.assertEqual(omitted_base, [])
        self.assertTrue(dest_base.is_dir())
        self.assertEqual(list(dest_base.iterdir()), [])

        # 2. suite-<active> -> full alpha suite staged, no omitted paths
        dest_alpha = self.tmp / "stage-alpha"
        omitted_alpha = agy.stage_plugins(dest_alpha, "suite-alpha", suite=self.alpha_suite)
        self.assertEqual(omitted_alpha, [])
        self.assertTrue((dest_alpha / "plan" / "skills" / "plan-skill" / "SKILL.md").is_file())
        self.assertTrue((dest_alpha / "plan" / "skills" / "exec-skill" / "SKILL.md").is_file())
        self.assertTrue((dest_alpha / "think" / "skills" / "debug-skill" / "SKILL.md").is_file())

        # 3. suite-<extra> -> full beta suite staged, no omitted paths
        dest_beta = self.tmp / "stage-beta"
        omitted_beta = agy.stage_plugins(
            dest_beta,
            "suite-beta",
            suite=self.alpha_suite,
            extra_suites=[self.beta_suite],
        )
        self.assertEqual(omitted_beta, [])
        self.assertTrue((dest_beta / "core" / "skills" / "beta-plan" / "SKILL.md").is_file())
        self.assertTrue((dest_beta / "core" / "skills" / "beta-debug" / "SKILL.md").is_file())

        # 4. with / without -> leave-one-out for target_skill
        dest_with = self.tmp / "stage-with"
        omitted_with = agy.stage_plugins(
            dest_with, "with", target_skill="plan-skill", suite=self.alpha_suite
        )
        self.assertEqual(omitted_with, [])
        self.assertTrue((dest_with / "plan" / "skills" / "plan-skill" / "SKILL.md").is_file())

        dest_without = self.tmp / "stage-without"
        omitted_without = agy.stage_plugins(
            dest_without, "without", target_skill="plan-skill", suite=self.alpha_suite
        )
        self.assertEqual(omitted_without, ["plan/skills/plan-skill"])
        self.assertFalse((dest_without / "plan" / "skills" / "plan-skill").exists())
        self.assertTrue((dest_without / "plan" / "skills" / "exec-skill" / "SKILL.md").is_file())

        # 5. skill_override rejected on non-with arms
        override_md = _write(self.tmp / "override.md", "---\nname: plan-skill\n---\n")
        for arm in ("without", "baseline", "suite-alpha"):
            with self.subTest(arm=arm):
                with self.assertRaisesRegex(ValueError, "skill_override"):
                    agy.stage_plugins(
                        self.tmp / f"err-{arm}",
                        arm,
                        target_skill="plan-skill" if arm == "without" else None,
                        skill_override=override_md,
                        suite=self.alpha_suite,
                    )

    def test_materialise_passes_arm_spec_and_records_extras(self) -> None:
        sub = agy.AgySubstrate()
        tree_dir = self.tmp / "tree"
        _write(tree_dir / "GEMINI.md", "<!-- base -->\n")
        spec = sb.skillbench_spec()

        inst_base = sub.materialise(tree_dir, spec, {"arm": "baseline"})
        self.addCleanup(shutil.rmtree, inst_base.staging_dir, True)
        self.assertEqual(inst_base.extras["arm"], "baseline")
        self.assertEqual(inst_base.extras["arm_kind"], "suite")
        self.assertEqual(inst_base.extras["staged_suite"], "")
        self.assertEqual(inst_base.extras["staged_suite_hash"], "")
        self.assertEqual(
            list((Path(inst_base.staging_dir) / "home" / agy.PLUGINS_REL).iterdir()), []
        )

        inst_beta = sub.materialise(
            tree_dir,
            spec,
            {"arm": "suite-beta", "suite_path": str(self.beta_toml)},
        )
        self.addCleanup(shutil.rmtree, inst_beta.staging_dir, True)
        self.assertEqual(inst_beta.extras["arm"], "suite-beta")
        self.assertEqual(inst_beta.extras["arm_kind"], "suite")
        self.assertEqual(inst_beta.extras["staged_suite"], "beta")
        self.assertEqual(inst_beta.extras["staged_suite_hash"], self.beta_suite.suite_hash)
        self.assertTrue(
            (
                Path(inst_beta.staging_dir)
                / "home"
                / agy.PLUGINS_REL
                / "core"
                / "skills"
                / "beta-plan"
                / "SKILL.md"
            ).is_file()
        )


class TestSchedulingAndAdapterByKind(_SuiteArmsCase):
    def test_tasks_and_domains_filtered_by_arm_kind(self) -> None:
        skill_ad = sb.SkillbenchAdapter(arm="with", executor="opus-5.5-high")
        self.assertEqual(skill_ad.arm_spec.kind, "skill")
        self.assertEqual(
            skill_ad.tasks(),
            ["debug-skill/s2", "plan-skill/s1"],
        )
        self.assertEqual(
            skill_ad.reference_domain_order(),
            ["debug-skill", "plan-skill"],
        )
        self.assertEqual(
            skill_ad._results_root,
            skill_ad.spec.runs_root / "results" / "opus-5.5-high" / "with",
        )

        base_ad = sb.SkillbenchAdapter(arm="baseline", executor="opus-5.5-high")
        self.assertEqual(base_ad.arm_spec.kind, "suite")
        self.assertEqual(
            base_ad.tasks(),
            ["_suite/a1", "_suite/p1"],
        )
        self.assertEqual(
            base_ad.reference_domain_order(),
            ["analysis", "planning"],
        )
        self.assertEqual(
            base_ad._results_root,
            base_ad.spec.runs_root / "results" / "opus-5.5-high" / "baseline",
        )

        beta_ad = sb.SkillbenchAdapter(
            arm="suite-beta",
            executor="gemini-3.8-flash-high",
            suites=[self.beta_toml],
        )
        self.assertEqual(beta_ad.arm_spec.kind, "suite")
        self.assertEqual(beta_ad.arm_spec.suite.name, "beta")
        self.assertEqual(
            beta_ad.tasks(),
            ["_suite/a1", "_suite/p1"],
        )
        p1 = beta_ad.tasks()[1]
        ctx = beta_ad.materialise_ctx(p1)
        self.assertEqual(
            ctx,
            {
                "arm": "suite-beta",
                "target_skill": "",
                "suite_path": str(self.beta_toml.resolve()),
            },
        )


class TestSuiteInvocationParsing(_SuiteArmsCase):
    def test_suite_arm_first_skill_md_read_and_skills_set(self) -> None:
        plugins_dir = self.tmp / "staged-plugins"
        agy.stage_plugins(plugins_dir, "suite-alpha", suite=self.alpha_suite)
        plan_md = plugins_dir / "plan" / "skills" / "plan-skill" / "SKILL.md"
        exec_md = plugins_dir / "plan" / "skills" / "exec-skill" / "SKILL.md"
        repo_skill_md = self.tmp / "work" / "skills" / "fake-skill" / "SKILL.md"
        _write(repo_skill_md, "# Not a staged plugin skill\n")

        lines = [
            json.dumps(
                {
                    "event": "step_update",
                    "t": 0.1,
                    "step_update": {"step_type": "user_input", "state": "DONE"},
                }
            ),
            # Read of a SKILL.md outside plugins_dir -> must NOT trigger invocation
            json.dumps(
                {
                    "event": "step_update",
                    "t": 2.0,
                    "step_update": {
                        "step_type": "tool",
                        "state": "ACTIVE",
                        "tool_name": "read_file",
                        "tool_info": {
                            "name": "read_file",
                            "parameters": {"Path": str(repo_skill_md)},
                        },
                    },
                }
            ),
            # First read of a staged SKILL.md -> triggers invocation at t=4.5s
            json.dumps(
                {
                    "event": "step_update",
                    "t": 4.5,
                    "step_update": {
                        "step_type": "tool",
                        "state": "ACTIVE",
                        "tool_name": "view_file",
                        "tool_info": {
                            "name": "view_file",
                            "parameters": {"AbsolutePath": str(plan_md)},
                        },
                    },
                }
            ),
            # Second read of another staged SKILL.md -> added to skills_read, t_skill_s stays 4.5s
            json.dumps(
                {
                    "event": "step_update",
                    "t": 9.0,
                    "step_update": {
                        "step_type": "tool",
                        "state": "ACTIVE",
                        "tool_name": "read_file",
                        "tool_info": {
                            "name": "read_file",
                            "parameters": {"Path": str(exec_md)},
                        },
                    },
                }
            ),
        ]
        evs = [ev for line in lines if (ev := agy.parse_event(line)) is not None]
        timing = agy.skill_timing(evs, None, plugins_dir=plugins_dir)
        self.assertEqual(
            timing,
            {
                "invoked": True,
                "t_skill_s": 4.5,
                "step_idx": 2,
                "skills_read": ["exec-skill", "plan-skill"],
            },
        )

        # Baseline empty plugins_dir: even if the model attempts to read a path under plugins_dir,
        # the file does not exist on disk so nothing is invoked.
        empty_plugins = self.tmp / "empty-plugins"
        empty_plugins.mkdir(parents=True, exist_ok=True)
        ghost_md = empty_plugins / "plan" / "skills" / "plan-skill" / "SKILL.md"
        ghost_lines = [
            json.dumps(
                {
                    "event": "step_update",
                    "t": 3.0,
                    "step_update": {
                        "step_type": "tool",
                        "state": "ACTIVE",
                        "tool_name": "read_file",
                        "tool_info": {
                            "name": "read_file",
                            "parameters": {"Path": str(ghost_md)},
                        },
                    },
                }
            ),
        ]
        ghost_evs = [ev for line in ghost_lines if (ev := agy.parse_event(line)) is not None]
        self.assertEqual(
            agy.skill_timing(ghost_evs, None, plugins_dir=empty_plugins),
            {
                "invoked": False,
                "t_skill_s": None,
                "step_idx": None,
                "skills_read": [],
            },
        )


def _seed_suite_namespace(
    ns_dir: Path,
    arm: str,
    spec: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    executor: str = "exec-m",
    judge: str = "judge-m",
    suite_name: str = "alpha",
    suite_hash: str = "alpha-hash",
    manifest_hash: str = "manifest-1",
    seed: int = 7,
) -> Path:
    """Write a judged suite-arm namespace from `task_id -> {"family": ..., "reps": [...]}`."""
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
            "arm_kind": "suite",
            "suite": suite_name,
            "suite_hash": suite_hash,
            "manifest_hash": manifest_hash,
            "executor": executor,
            "judge": judge,
            "effort": "high",
        },
    )
    judged: list[dict[str, Any]] = []
    for task_id, tinfo in sorted(spec.items()):
        family = str(tinfo["family"])
        reps = list(tinfo["reps"])
        task_dir = ns_dir / task_id.replace("/", "__")
        entries: list[dict[str, Any]] = []
        for idx, rep in enumerate(reps, 1):
            label = f"base{idx}"
            run_dir = task_dir / label
            run_dir.mkdir(parents=True, exist_ok=True)
            n_passed = rep.get("n_passed", 4)
            valid = n_passed is not None
            skills_read = list(rep.get("skills_read") or [])
            invoked = bool(rep.get("invoked", bool(skills_read)))
            t_skill_s = rep.get("t_skill_s", 3.0 if invoked else None)
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
                    "valid": valid,
                    "invalid_reason": None if valid else "exit 1",
                    "exit_code": 0 if valid else 1,
                    "total_tokens": int(rep.get("tokens", 1000)),
                },
            )
            score = {"n_passed": int(n_passed), "n_criteria": 4} if valid else None
            if valid:
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
                                    for i in range(4)
                                ],
                                "n_passed": int(n_passed),
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
                "domain": family,
                "family": family,
                "kind": "suite",
                "k": len(reps),
                "seed": seed,
                "tag": arm,
                "status": "bases-only",
                "base": entries,
            },
        )
    _write_json(ns_dir / "base_judged.json", judged)
    return ns_dir


class TestReportSuitesSection(_SuiteArmsCase):
    def _build_arm_specs(
        self,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        baseline_spec = {
            "planning/p1": {
                "family": "planning",
                "reps": [
                    {"n_passed": 1, "tokens": 800, "t_total_s": 12.0, "skills_read": []},
                    {"n_passed": 1, "tokens": 900, "t_total_s": 14.0, "skills_read": []},
                ],
            },
            "planning/p2": {
                "family": "planning",
                "reps": [
                    {"n_passed": 2, "tokens": 850, "t_total_s": 15.0, "skills_read": []},
                    {"n_passed": 1, "tokens": 950, "t_total_s": 15.0, "skills_read": []},
                ],
            },
            "planning/p3": {
                "family": "planning",
                "reps": [
                    {"n_passed": 1, "tokens": 700, "t_total_s": 10.0, "skills_read": []},
                    {"n_passed": 2, "tokens": 900, "t_total_s": 12.0, "skills_read": []},
                ],
            },
        }
        alpha_spec = {
            "planning/p1": {
                "family": "planning",
                "reps": [
                    {
                        "n_passed": 4,
                        "tokens": 1200,
                        "t_total_s": 22.0,
                        "t_skill_s": 2.0,
                        "skills_read": ["plan-skill", "exec-skill"],
                    },
                    {
                        "n_passed": 4,
                        "tokens": 1400,
                        "t_total_s": 26.0,
                        "t_skill_s": 4.0,
                        "skills_read": ["plan-skill"],
                    },
                ],
            },
            "planning/p2": {
                "family": "planning",
                "reps": [
                    {
                        "n_passed": 4,
                        "tokens": 1300,
                        "t_total_s": 24.0,
                        "t_skill_s": 3.0,
                        "skills_read": ["plan-skill"],
                    },
                    {
                        "n_passed": 3,
                        "tokens": 1500,
                        "t_total_s": 28.0,
                        "t_skill_s": 5.0,
                        "skills_read": ["exec-skill"],
                    },
                ],
            },
            "planning/p3": {
                "family": "planning",
                "reps": [
                    {
                        "n_passed": 4,
                        "tokens": 1100,
                        "t_total_s": 20.0,
                        "t_skill_s": 2.5,
                        "skills_read": ["plan-skill"],
                    },
                    {
                        "n_passed": 4,
                        "tokens": 1300,
                        "t_total_s": 24.0,
                        "t_skill_s": 3.5,
                        "skills_read": ["plan-skill", "exec-skill"],
                    },
                ],
            },
        }
        beta_spec = {
            "planning/p1": {
                "family": "planning",
                "reps": [
                    {
                        "n_passed": 2,
                        "tokens": 1600,
                        "t_total_s": 30.0,
                        "t_skill_s": 6.0,
                        "skills_read": ["beta-plan"],
                    },
                    {
                        "n_passed": 2,
                        "tokens": 1800,
                        "t_total_s": 32.0,
                        "t_skill_s": 8.0,
                        "skills_read": ["beta-plan"],
                    },
                ],
            },
            "planning/p2": {
                "family": "planning",
                "reps": [
                    {
                        "n_passed": 2,
                        "tokens": 1700,
                        "t_total_s": 31.0,
                        "t_skill_s": 7.0,
                        "skills_read": ["beta-plan", "beta-debug"],
                    },
                    {
                        "n_passed": 2,
                        "tokens": 1700,
                        "t_total_s": 31.0,
                        "invoked": False,
                        "skills_read": [],
                    },
                ],
            },
            "planning/p3": {
                "family": "planning",
                "reps": [
                    {
                        "n_passed": 3,
                        "tokens": 1500,
                        "t_total_s": 29.0,
                        "t_skill_s": 5.0,
                        "skills_read": ["beta-debug"],
                    },
                    {
                        "n_passed": 2,
                        "tokens": 1700,
                        "t_total_s": 31.0,
                        "t_skill_s": 6.0,
                        "skills_read": ["beta-plan"],
                    },
                ],
            },
        }
        return baseline_spec, alpha_spec, beta_spec

    def test_suites_section_with_two_arms(self) -> None:
        baseline_spec, alpha_spec, _ = self._build_arm_specs()
        root = self.tmp / "res2" / "exec-m"
        _seed_suite_namespace(
            root / "baseline",
            "baseline",
            baseline_spec,
            suite_name=self.alpha_suite.name,
            suite_hash=self.alpha_suite.suite_hash,
        )
        _seed_suite_namespace(
            root / "suite-alpha",
            "suite-alpha",
            alpha_spec,
            suite_name=self.alpha_suite.name,
            suite_hash=self.alpha_suite.suite_hash,
        )

        rep = SR.build_report(root.parent, draws=200, bootstrap_seed=11)
        self.assertEqual(rep["suite_arms"], ["suite-alpha"])
        self.assertEqual(len(rep["suites"]), 1)
        fam = rep["by_family"]["planning"]
        self.assertEqual(fam["family"], "planning")
        self.assertEqual(fam["suite_arm"], "suite-alpha")
        self.assertEqual(fam["n_tasks"], 3)
        self.assertEqual(fam["verdict"], "helps")
        self.assertGreater(fam["lift"], 0.5)
        self.assertGreater(fam["ci_low"], 0.0)
        self.assertEqual(fam["tokens_suite_median"], 1300.0)
        self.assertEqual(fam["tokens_baseline_median"], 875.0)
        self.assertEqual(fam["t_total_suite_median"], 24.0)
        self.assertEqual(fam["t_total_baseline_median"], 13.0)
        self.assertEqual(
            fam["skills_read_histogram"],
            {"exec-skill": 3, "plan-skill": 5},
        )
        # Per-task metrics: lift, CI, verdict, tokens/time medians, skills-read histogram
        p1 = fam["tasks"]["planning/p1"]
        self.assertAlmostEqual(p1["suite_mean"], 1.0)
        self.assertAlmostEqual(p1["baseline_mean"], 0.25)
        self.assertAlmostEqual(p1["lift"], 0.75)
        self.assertIsNotNone(p1["ci_low"])
        self.assertEqual(p1["verdict"], "helps")
        self.assertEqual(p1["tokens_suite_median"], 1300.0)
        self.assertEqual(p1["tokens_baseline_median"], 850.0)
        self.assertEqual(p1["t_total_suite_median"], 24.0)
        self.assertEqual(p1["t_total_baseline_median"], 13.0)
        self.assertEqual(
            p1["skills_read_histogram"],
            {"exec-skill": 1, "plan-skill": 2},
        )

        md = SR.render_markdown(rep)
        self.assertIn("Suite mode (per family vs built-in baseline):", md)
        self.assertIn("| family | tasks | suite-alpha | baseline |", md)
        self.assertIn("exec-skill: 3, plan-skill: 5", md)
        self.assertIn("Suite tasks (per task vs built-in baseline):", md)
        self.assertIn("| planning | planning/p1 |", md)

    def test_suites_section_with_three_arms_and_suites_cli(self) -> None:
        baseline_spec, alpha_spec, beta_spec = self._build_arm_specs()
        results_root = self.tmp / "run3" / "results"
        ex_root = results_root / "exec-m"
        _seed_suite_namespace(
            ex_root / "baseline",
            "baseline",
            baseline_spec,
            suite_name=self.alpha_suite.name,
            suite_hash=self.alpha_suite.suite_hash,
        )
        _seed_suite_namespace(
            ex_root / "suite-alpha",
            "suite-alpha",
            alpha_spec,
            suite_name=self.alpha_suite.name,
            suite_hash=self.alpha_suite.suite_hash,
        )
        _seed_suite_namespace(
            ex_root / "suite-beta",
            "suite-beta",
            beta_spec,
            suite_name=self.beta_suite.name,
            suite_hash=self.beta_suite.suite_hash,
        )

        rep = SR.build_report(
            results_root,
            draws=200,
            bootstrap_seed=11,
            suites=[self.beta_toml],
        )
        self.assertEqual(rep["suite_arms"], ["suite-alpha", "suite-beta"])
        self.assertEqual(
            rep["extra_suites"],
            [{"name": "beta", "hash": self.beta_suite.suite_hash}],
        )
        fam = rep["by_family"]["planning"]
        self.assertIn("suite-alpha", fam["by_arm"])
        self.assertIn("suite-beta", fam["by_arm"])
        beta_fam = fam["by_arm"]["suite-beta"]
        self.assertAlmostEqual(beta_fam["lift"], 0.20833333333333334)
        self.assertEqual(beta_fam["tokens_suite_median"], 1700.0)
        self.assertEqual(beta_fam["t_total_suite_median"], 31.0)
        self.assertEqual(
            beta_fam["skills_read_histogram"],
            {"beta-debug": 2, "beta-plan": 4},
        )
        p2_beta = fam["tasks"]["planning/p2"]["by_arm"]["suite-beta"]
        self.assertEqual(
            p2_beta["skills_read_histogram"],
            {"beta-debug": 1, "beta-plan": 1},
        )
        self.assertEqual(p2_beta["invoked"], 1)
        self.assertAlmostEqual(p2_beta["invocation_rate"], 0.5)

        # CLI with --suites renders one extra column per extra suite
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = SR.main(
                [
                    "--results-root",
                    str(results_root),
                    "--suite",
                    str(self.alpha_toml),
                    "--suites",
                    str(self.beta_toml),
                    "--draws",
                    "100",
                    "--k",
                    "2",
                ]
            )
        self.assertEqual(rc, 0)
        md = out.getvalue()
        self.assertIn(
            "| family | tasks | suite-alpha | baseline | lift [95% CI] | verdict "
            "| invoked | t_skill p50/p90 | tokens p50 S/B | wall time p50 S/B "
            "| skills read | ceiling/total | suite-beta |",
            md,
        )
        self.assertIn(
            "| family | task | suite-alpha | baseline | lift [95% CI] | verdict "
            "| tokens p50 S/B | wall time p50 S/B | skills read | hardness | suite-beta |",
            md,
        )
        self.assertIn("skills: beta-debug: 2, beta-plan: 4", md)

    def test_suite_cell_pairing_errors(self) -> None:
        baseline_spec, alpha_spec, _ = self._build_arm_specs()
        base_ns = _seed_suite_namespace(
            self.tmp / "err" / "baseline", "baseline", baseline_spec
        )
        alpha_ns = _seed_suite_namespace(
            self.tmp / "err" / "suite-alpha", "suite-alpha", alpha_spec
        )
        with self.assertRaisesRegex(ValueError, "no 'baseline' cell"):
            SR.build_report(
                [SR.ReportCell("exec-m", "judge-m", "suite-alpha", alpha_ns)],
                draws=50,
            )
        with self.assertRaisesRegex(ValueError, "no 'suite-<name>' cell"):
            SR.build_report(
                [SR.ReportCell("exec-m", "judge-m", "baseline", base_ns)],
                draws=50,
            )
        with self.assertRaisesRegex(ValueError, "no 'suite-gamma' cell"):
            SR.build_report(
                [
                    SR.ReportCell("exec-m", "judge-m", "baseline", base_ns),
                    SR.ReportCell("exec-m", "judge-m", "suite-alpha", alpha_ns),
                ],
                draws=50,
                suites=[self.gamma_toml],
            )


class TestSweepAndJudgeCliSuiteArms(_SuiteArmsCase):
    def setUp(self) -> None:
        super().setUp()
        for rel in AUTH:
            _write(self.home / rel, f'{{"token": "{rel}"}}\n')
        os.environ["HARNESS2_AGY_BIN"] = str(STUB_SCRIPT)
        os.environ["SKILLBENCH_AUTH_FILES"] = ",".join(AUTH)
        os.environ["AGY_STUB_SCENARIO"] = "plain_answer"
        os.environ["AGY_STUB_JUDGE_SCENARIO"] = "auto"

    def test_sweep_and_judge_accept_baseline_and_suite_labels(self) -> None:
        # 1. Sweep & judge baseline
        with contextlib.redirect_stdout(io.StringIO()):
            rc = run_sweep.main(
                [
                    "--bench",
                    "skillbench",
                    "--arm",
                    "baseline",
                    "--suite",
                    str(self.alpha_toml),
                    "--k",
                    "1",
                    "--bases-only",
                ]
            )
        self.assertEqual(rc, 0)

        with contextlib.redirect_stdout(io.StringIO()):
            rc = judge_cli.main(
                [
                    "--bench",
                    "skillbench",
                    "--arm",
                    "baseline",
                    "--suite",
                    str(self.alpha_toml),
                    "--k",
                    "1",
                    "--bases",
                ]
            )
        self.assertEqual(rc, 0)

        # 2. Sweep & judge suite-beta (extra suite via --suites)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = run_sweep.main(
                [
                    "--bench",
                    "skillbench",
                    "--arm",
                    "suite-beta",
                    "--suite",
                    str(self.alpha_toml),
                    "--suites",
                    str(self.beta_toml),
                    "--k",
                    "1",
                    "--bases-only",
                ]
            )
        self.assertEqual(rc, 0)

        with contextlib.redirect_stdout(io.StringIO()):
            rc = judge_cli.main(
                [
                    "--bench",
                    "skillbench",
                    "--arm",
                    "suite-beta",
                    "--suite",
                    str(self.alpha_toml),
                    "--suites",
                    str(self.beta_toml),
                    "--k",
                    "1",
                    "--bases",
                ]
            )
        self.assertEqual(rc, 0)

        # Verify run.json in adapt/ namespaces and rollout dirs under results/<executor>/<arm>/
        adapt_dir = self.runs_dir / "skillbench" / "adapt"
        base_ns = next(adapt_dir.glob("*-baseline"))
        beta_ns = next(adapt_dir.glob("*-suite-beta"))
        self.assertTrue((base_ns / "_suite__a1" / "summary.json").is_file())
        self.assertTrue((base_ns / "_suite__p1" / "summary.json").is_file())
        self.assertFalse((base_ns / "plan-skill__s1").exists())

        base_res = self.runs_dir / "skillbench" / "results" / "exec-m" / "baseline"
        beta_res = self.runs_dir / "skillbench" / "results" / "exec-m" / "suite-beta"
        self.assertTrue((base_res / base_ns.name / "_suite" / "a1" / "base1").is_dir())
        self.assertTrue((base_res / base_ns.name / "_suite" / "p1" / "base1").is_dir())
        self.assertFalse((base_res / base_ns.name / "plan-skill").exists())

        self.assertTrue((beta_res / beta_ns.name / "_suite" / "a1" / "base1").is_dir())
        self.assertTrue((beta_res / beta_ns.name / "_suite" / "p1" / "base1").is_dir())
        self.assertFalse((beta_res / beta_ns.name / "plan-skill").exists())

        run_meta = json.loads((beta_ns / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run_meta["arm"], "suite-beta")
        self.assertEqual(run_meta["arm_kind"], "suite")
        self.assertEqual(run_meta["suite"], "alpha")
        self.assertEqual(run_meta["staged_suite"], "beta")
        self.assertEqual(run_meta["staged_suite_hash"], self.beta_suite.suite_hash)

        # Build report over the swept & judged baseline + suite-beta cells
        rep = SR.build_report(
            [
                SR.ReportCell("exec-m", "judge-m", "baseline", base_ns),
                SR.ReportCell("exec-m", "judge-m", "suite-beta", beta_ns),
            ],
            draws=50,
            suites=[self.beta_toml],
        )
        self.assertEqual(sorted(rep["by_family"]), ["analysis", "planning"])


if __name__ == "__main__":
    unittest.main()
