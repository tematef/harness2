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

"""SkillbenchAdapter and discover("skillbench") tests.

Every test runs against temp directories, fake cc-thingz plugins, a fake HOME, and
`tests/fixtures/agy_stub.py`; the user's real `$HOME` and `~/.gemini` are never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness2.benchmarks import skillbench as SB
from harness2.benchmarks.tasks import discover
from harness2.config import ConfigError
from harness2.core import tree as HT
from harness2.core.spec import BenchmarkAdapter, BenchPrompts, BenchSpec, ChannelSpec, SubstrateLike
from harness2.substrate.agy import AGY_SURFACE, INSTALL_MODE, PLUGINS_REL
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


class _AdapterCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-adapter-test-"))
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
        make_task(
            self.tasks_root,
            "root-cause-investigator",
            "p2",
            pilot=True,
            prompt="Investigate why utils.py returns None on empty inputs.",
        )
        SB.freeze(self.tasks_root)

        self.runs_dir = self.tmp / "runs"
        self.stub_log = self.tmp / "stub.log"
        env = {
            "HOME": str(self.home),
            "HARNESS2_RUNS_DIR": str(self.runs_dir),
            "HARNESS2_AGY_BIN": str(STUB),
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

        tasks_root_patcher = mock.patch.object(SB, "TASKS_ROOT", self.tasks_root)
        tasks_root_patcher.start()
        self.addCleanup(tasks_root_patcher.stop)

        self.real_home_before = _tree_hash(self.home)

    def tearDown(self) -> None:
        self.assertEqual(_tree_hash(self.home), self.real_home_before, "fake real HOME modified")

    def stub_calls(self) -> list[dict]:
        if not self.stub_log.is_file():
            return []
        return [json.loads(x) for x in self.stub_log.read_text(encoding="utf-8").splitlines()]


class TestDiscoverAndManifest(_AdapterCase):
    def test_discover_skillbench_maps_task_id_to_skill_domain(self) -> None:
        got = discover("skillbench")
        self.assertEqual(
            got,
            {
                "code-review-refactor/t1": "code-review-refactor",
                "make-plan/p1": "make-plan",
                "root-cause-investigator/p2": "root-cause-investigator",
            },
        )
        self.assertEqual(list(got), sorted(got))

    def test_discover_skillbench_rejects_subset_and_empty_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown benchmark/subset"):
            discover("skillbench", subset="code")
        empty = self.tmp / "empty-tasks"
        empty.mkdir()
        with mock.patch.object(SB, "TASKS_ROOT", empty):
            with self.assertRaisesRegex(FileNotFoundError, "No skillbench tasks"):
                discover("skillbench")

    def test_adapter_refuses_to_construct_when_check_manifest_fails(self) -> None:
        # Missing manifest.json
        unfrozen = self.tmp / "unfrozen"
        make_task(unfrozen, "code-review-refactor", "t1")
        with mock.patch.object(SB, "TASKS_ROOT", unfrozen):
            with self.assertRaisesRegex(SB.ManifestError, "not frozen"):
                SB.SkillbenchAdapter("with")

        # Modified task after freeze
        (self.tasks_root / "make-plan" / "p1" / "fixture" / "utils.py").write_text(
            "def f(x):\n    return x + 1\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(SB.ManifestError, "changed: make-plan/p1"):
            SB.SkillbenchAdapter("with")

        # Restore and test added task after freeze
        SB.freeze(self.tasks_root, force=True)
        make_task(self.tasks_root, "code-review-refactor", "t2")
        with self.assertRaisesRegex(SB.ManifestError, "added: code-review-refactor/t2"):
            SB.SkillbenchAdapter("without")

    def test_adapter_rejects_bad_arm_or_substrate(self) -> None:
        for bad_arm in ("", "both", "agg"):
            with self.assertRaisesRegex(ValueError, "arm"):
                SB.SkillbenchAdapter(bad_arm)
        for bad_sub in ("oc", "cx", ""):
            with self.assertRaisesRegex(ValueError, "substrate"):
                SB.SkillbenchAdapter("with", substrate=bad_sub)

    def test_tasks_honours_pilot_filtering_and_splits(self) -> None:
        adapter = SB.SkillbenchAdapter("with")
        self.assertEqual(
            adapter.tasks(),
            ["code-review-refactor/t1", "make-plan/p1", "root-cause-investigator/p2"],
        )
        # Explicit pilot=True on tasks()
        self.assertEqual(
            adapter.tasks(pilot=True),
            ["make-plan/p1", "root-cause-investigator/p2"],
        )
        # split="pilot"
        self.assertEqual(
            adapter.tasks("pilot"),
            ["make-plan/p1", "root-cause-investigator/p2"],
        )
        # Adapter constructed with pilot=True
        pilot_adapter = SB.SkillbenchAdapter("with", pilot=True)
        self.assertEqual(
            pilot_adapter.tasks(),
            ["make-plan/p1", "root-cause-investigator/p2"],
        )
        self.assertEqual(
            pilot_adapter.tasks(pilot=False),
            ["code-review-refactor/t1", "make-plan/p1", "root-cause-investigator/p2"],
        )
        # Env var SKILLBENCH_PILOT=1
        with mock.patch.dict(os.environ, {"SKILLBENCH_PILOT": "1"}):
            env_adapter = SB.SkillbenchAdapter("without")
            self.assertEqual(
                env_adapter.tasks(),
                ["make-plan/p1", "root-cause-investigator/p2"],
            )

        # Split file compatibility
        split_file = _write(
            self.tmp / "split.json",
            json.dumps(
                {
                    "benchmark": "skillbench",
                    "train": ["make-plan/p1", "code-review-refactor/t1"],
                    "test": ["root-cause-investigator/p2"],
                }
            ),
        )
        split_adapter = SB.SkillbenchAdapter("with", split_file=split_file)
        self.assertEqual(
            split_adapter.tasks("train"),
            ["code-review-refactor/t1", "make-plan/p1"],
        )
        self.assertEqual(split_adapter.tasks("test"), ["root-cause-investigator/p2"])
        self.assertEqual(split_adapter.tasks("train", pilot=True), ["make-plan/p1"])


class TestProtocolAndSurface(_AdapterCase):
    def setUp(self) -> None:
        super().setUp()
        self.adapter = SB.SkillbenchAdapter("with")

    def test_satisfies_benchmark_adapter_and_substrate_protocols(self) -> None:
        self.assertIsInstance(self.adapter, BenchmarkAdapter)
        self.assertIsInstance(self.adapter.substrate, SubstrateLike)
        self.assertEqual(self.adapter.substrate.name, "agy")
        self.assertIs(self.adapter.substrate.surface(), AGY_SURFACE)

    def test_spec_is_valid_bench_spec_without_forbidden_prompt_words(self) -> None:
        spec = self.adapter.spec
        self.assertIsInstance(spec, BenchSpec)
        self.assertEqual(spec.name, "skillbench")
        self.assertEqual(spec.install_mode, INSTALL_MODE)
        self.assertIn("proposer", spec.channels)
        self.assertIn("selector", spec.channels)
        for role in ("proposer", "selector"):
            ch = spec.channels[role]
            self.assertIsInstance(ch, ChannelSpec)
            self.assertTrue(ch.model)
        self.assertIsInstance(spec.prompts, BenchPrompts)
        for field_name in (
            "work",
            "inv_levers",
            "dom_levers",
            "yardstick",
            "domain_guidance",
            "selection_checks",
        ):
            text = getattr(spec.prompts, field_name).lower()
            for forbidden in ("benchmark", "evaluation", "grader", "rubric", "score"):
                self.assertNotIn(forbidden, text, f"{forbidden!r} found in prompts.{field_name}")

    def test_domain_of_and_reference_domain_order(self) -> None:
        self.assertEqual(self.adapter.domain_of("make-plan/p1"), "make-plan")
        self.assertEqual(
            self.adapter.domain_of("code-review-refactor/t1"), "code-review-refactor"
        )
        self.assertEqual(
            self.adapter.reference_domain_order(),
            ["code-review-refactor", "make-plan", "root-cause-investigator"],
        )
        with self.assertRaises(ValueError):
            self.adapter.domain_of("not-a-task-id")

    def test_task_prompt_matches_build_prompt_and_is_arm_independent(self) -> None:
        without = SB.SkillbenchAdapter("without")
        for tid in self.adapter.tasks():
            task = SB.load_task(self.tasks_root / tid)
            expected = SB.build_prompt(task)
            self.assertEqual(self.adapter.task_prompt(tid), expected)
            self.assertEqual(without.task_prompt(tid), expected)
        with self.assertRaises(KeyError):
            self.adapter.task_prompt("make-plan/no-such-task")

    def test_task_files_returns_fixture_dir_only(self) -> None:
        fx = self.adapter.task_files("make-plan/p1")
        self.assertEqual(fx, self.tasks_root / "make-plan" / "p1" / "fixture")
        self.assertTrue((fx / "utils.py").is_file())
        self.assertFalse((fx / "rubric.json").exists())
        self.assertFalse((fx / "task.toml").exists())

    def test_base_harness_and_validate_tree(self) -> None:
        base = self.adapter.base_harness()
        self.assertTrue(base.is_dir())
        self.assertTrue(HT.is_tree(AGY_SURFACE, base))
        tree = HT.read_tree(AGY_SURFACE, base)
        self.assertTrue(tree)
        self.assertTrue(tree.get("GEMINI.md", "").strip())
        self.assertIsNone(self.adapter.validate_tree(tree))

        # Invalid trees raise ValueError
        with self.assertRaisesRegex(ValueError, "GEMINI.md"):
            self.adapter.validate_tree({})
        with self.assertRaisesRegex(ValueError, "off the agy editable surface"):
            self.adapter.validate_tree({"GEMINI.md": "ok\n", "unexpected.txt": "bad\n"})
        with self.assertRaisesRegex(ValueError, "sub_agents"):
            self.adapter.validate_tree(
                {
                    "GEMINI.md": "ok\n",
                    "sub_agents/helper/AGENT.md": "---\ndescription: h\n---\nbody\n",
                }
            )
        with self.assertRaisesRegex(ValueError, "over the 24000 cap"):
            self.adapter.validate_tree({"GEMINI.md": "x" * 25_000})


class TestExecuteDeliverablesScoreAndPreflight(_AdapterCase):
    def test_execute_passes_arm_and_target_skill_via_ctx(self) -> None:
        for arm, expect_skill_dir in (("with", True), ("without", False)):
            with self.subTest(arm=arm):
                adapter = SB.SkillbenchAdapter(arm)
                seen_ctx: list[dict] = []
                orig_materialise = adapter.substrate.materialise

                def spy_materialise(tree_dir, spec, ctx, _orig=orig_materialise):
                    seen_ctx.append(dict(ctx))
                    return _orig(tree_dir, spec, ctx)

                with mock.patch.object(adapter.substrate, "materialise", side_effect=spy_materialise):
                    rd = adapter.execute("make-plan/p1", None, f"run-{arm}", timeout=45)

                self.assertEqual(seen_ctx, [{"arm": arm, "target_skill": "make-plan"}])
                staged_skill = rd / "home" / PLUGINS_REL / "planning" / "skills" / "make-plan"
                self.assertEqual(staged_skill.exists(), expect_skill_dir)
                # Other skills in the suite remain staged in both arms.
                self.assertTrue(
                    (
                        rd
                        / "home"
                        / PLUGINS_REL
                        / "review"
                        / "skills"
                        / "code-review-refactor"
                        / "SKILL.md"
                    ).is_file()
                )
                call = self.stub_calls()[-1]
                self.assertEqual(call["flags"]["prompt"], adapter.task_prompt("make-plan/p1"))
                self.assertEqual(call["flags"]["print_timeout"], "45s")

    def test_execute_is_idempotent_until_artifacts_change(self) -> None:
        adapter = SB.SkillbenchAdapter("with")
        rd = adapter.execute("make-plan/p1", None, "idem-run")
        self.assertEqual(len(self.stub_calls()), 1)

        # Second call with same run_id reuses completed rollout without calling agy.
        rd2 = adapter.execute("make-plan/p1", None, "idem-run")
        self.assertEqual(rd2, rd)
        self.assertEqual(len(self.stub_calls()), 1)

        # Corrupting final.md triggers a re-roll.
        (rd / "final.md").unlink()
        adapter.execute("make-plan/p1", None, "idem-run")
        self.assertEqual(len(self.stub_calls()), 2)

    def test_delivered_requires_non_empty_final_md_and_valid_rollout(self) -> None:
        adapter = SB.SkillbenchAdapter("with")
        rd = adapter.execute("make-plan/p1", None, "deliv-ok")
        self.assertTrue(adapter.delivered(rd))

        # Empty final.md -> False
        (rd / "final.md").write_text("   \n", encoding="utf-8")
        self.assertFalse(adapter.delivered(rd))
        (rd / "final.md").write_text("Done.\n", encoding="utf-8")
        self.assertTrue(adapter.delivered(rd))

        # Invalid rollout (non-zero exit) -> False even when final.md is non-empty
        os.environ["AGY_STUB_EXIT_CODE"] = "2"
        rd_bad = adapter.execute("make-plan/p1", None, "deliv-bad")
        self.assertTrue((rd_bad / "final.md").read_text(encoding="utf-8").strip())
        self.assertFalse(adapter.delivered(rd_bad))
        self.assertFalse(adapter.delivered(self.tmp / "nonexistent-run"))

    def test_trajectory_and_deliverables_and_score(self) -> None:
        adapter = SB.SkillbenchAdapter("with")
        rd = adapter.execute("make-plan/p1", None, "full-run")

        traj = adapter.trajectory(rd, obs_trunc=10_000)
        self.assertIn("[ACTION] view_file(", traj)
        self.assertIn("[ASSISTANT] Wrote the plan to docs/plans/20261023-add-cache.md.", traj)
        self.assertIn(
            "no trajectory recorded", adapter.trajectory(self.tmp / "missing", obs_trunc=1000)
        )

        files = adapter.deliverable_files(rd)
        self.assertIn("final.md", files)
        self.assertIn("diff.patch", files)
        self.assertIn("new_files/docs/plans/20261023-add-cache.md", files)
        text = adapter.deliverable_text(rd)
        self.assertEqual(text, SB.deliverable_text(rd))
        self.assertIn("### final.md", text)
        self.assertIn("docs/plans/20261023-add-cache.md", text)

        judge_scen = _scenario(
            self.tmp / "judge.jsonl",
            {
                "event": "result",
                "result": {
                    "structured_output": {
                        "criteria": [
                            {"id": "c0", "pass": True, "reason": "ok"},
                            {"id": "c1", "pass": True, "reason": "ok"},
                            {"id": "c2", "pass": False, "reason": "no"},
                            {"id": "c3", "pass": True, "reason": "ok"},
                        ]
                    },
                    "usage": {"total_tokens": 300},
                },
            },
        )
        os.environ["AGY_STUB_SCENARIO"] = str(judge_scen)
        self.assertEqual(
            adapter.score(rd, "make-plan/p1"),
            {"n_passed": 3, "n_criteria": 4},
        )

    def test_preflight_checks_substrate_and_manifest(self) -> None:
        adapter = SB.SkillbenchAdapter("with")
        adapter.preflight()
        self.assertEqual(self.stub_calls(), [])

        # Missing binary fails preflight
        with mock.patch.dict(os.environ, {"HARNESS2_AGY_BIN": str(self.tmp / "missing-agy")}):
            with self.assertRaisesRegex(RuntimeError, "HARNESS2_AGY_BIN"):
                adapter.preflight()

        # Unset effort is fine (jet model ids pin it); an unknown value fails preflight
        with mock.patch.dict(os.environ, {"SKILLBENCH_EFFORT": ""}):
            adapter.preflight()
        with mock.patch.dict(os.environ, {"SKILLBENCH_EFFORT": "extreme"}):
            with self.assertRaisesRegex(ConfigError, "SKILLBENCH_EFFORT"):
                adapter.preflight()

        # Corpus drift after adapter construction fails preflight
        (self.tasks_root / "code-review-refactor" / "t1" / "fixture" / "utils.py").write_text(
            "changed\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(SB.ManifestError, "changed: code-review-refactor/t1"):
            adapter.preflight()


if __name__ == "__main__":
    unittest.main()
