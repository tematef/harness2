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

"""Pilot corpus verification for skillbench (Task 13).

Verifies that:
- every bundled task in `harness2/benchmarks/skillbench/tasks` loads and validates cleanly,
- `manifest.json` is frozen and matches the corpus on disk,
- each of the 3 pilot skills (`make-plan`, `root-cause-investigator`, `code-review-refactor`)
  has exactly 2 pilot tasks (`pilot = true`),
- every prompt passes `check_prompt` across all 18 cc-thingz skill names and `forbidden_terms`,
- no rubric criterion mentions any skill name or any skill ritual / step (e.g. 5-Why, checkboxes,
  tool names, template section headers),
- each task's `fixture/` repo is self-contained and its `unittest` suite executes as expected
  (passing baseline for `make-plan` and `code-review-refactor`; reproducible seeded test failure
  for `root-cause-investigator`).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness2.benchmarks import skillbench as SB
from harness2.benchmarks.tasks import discover

PILOT_SKILLS = ("make-plan", "root-cause-investigator", "code-review-refactor")

# All 18 skills in the cc-thingz plugin suite plus any legacy names in SB.KNOWN_SKILLS.
CC_THINGZ_18_SKILLS = frozenset(
    {
        "backlog",
        "brainstorm",
        "clarify",
        "code-agentic-refactor",
        "code-review-best-practices",
        "code-review-refactor",
        "dialectic",
        "exec",
        "git-review",
        "learn",
        "make-plan",
        "md-copy",
        "root-cause-investigator",
        "txt-copy",
        "wiki-builder",
        "writing-style",
        "wrong",
        "skill-eval",
    }
)
ALL_SKILL_NAMES = tuple(sorted(CC_THINGZ_18_SKILLS | SB.KNOWN_SKILLS))

# Skill rituals, template headers, helper scripts, and AGY tool names that must never appear in
# outcome-only rubrics.
FORBIDDEN_RUBRIC_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bskill\.md\b", re.IGNORECASE), "mentions SKILL.md"),
    (re.compile(r"\.gemini/config/plugins", re.IGNORECASE), "mentions plugin path"),
    (re.compile(r"\b5[\s\-]*whys?\b", re.IGNORECASE), "mentions 5-Why ritual"),
    (re.compile(r"\bfive[\s\-]+whys?\b", re.IGNORECASE), "mentions Five-Whys ritual"),
    (re.compile(r"\bwhy\s*#\s*[1-5]\b", re.IGNORECASE), "mentions Why #N header"),
    (re.compile(r"\bcheckbox(?:es)?\b", re.IGNORECASE), "mentions checkboxes"),
    (re.compile(r"\[\s*[xX ]?\s*\]"), "mentions markdown checkbox syntax"),
    (re.compile(r"\byyyymmdd\b", re.IGNORECASE), "mentions yyyymmdd filename ritual"),
    (re.compile(r"\bask_question\b", re.IGNORECASE), "mentions ask_question tool"),
    (re.compile(r"\binvoke_subagent\b", re.IGNORECASE), "mentions invoke_subagent tool"),
    (re.compile(r"\bview_file\b", re.IGNORECASE), "mentions view_file tool"),
    (re.compile(r"\bwrite_to_file\b", re.IGNORECASE), "mentions write_to_file tool"),
    (re.compile(r"\breplace_file_content\b", re.IGNORECASE), "mentions replace_file_content tool"),
    (re.compile(r"\bresolve-project-dir\b", re.IGNORECASE), "mentions resolve-project-dir script"),
    (re.compile(r"\bresolve-rules\b", re.IGNORECASE), "mentions resolve-rules script"),
    (re.compile(r"\bplanning-rules\.md\b", re.IGNORECASE), "mentions planning-rules.md"),
    (re.compile(r"\bplan-review\b", re.IGNORECASE), "mentions plan-review agent"),
    (re.compile(r"\brevdiff\b", re.IGNORECASE), "mentions revdiff tool"),
    (re.compile(r"\bplan-annotate\b", re.IGNORECASE), "mentions plan-annotate script"),
    (re.compile(r"\bpost-completion\b", re.IGNORECASE), "mentions Post-Completion section ritual"),
    (re.compile(r"\bwhat goes where\b", re.IGNORECASE), "mentions What Goes Where section ritual"),
    (
        re.compile(r"\bhigh impact refactorings\b", re.IGNORECASE),
        "mentions High Impact Refactorings header",
    ),
    (
        re.compile(r"\bbefore/after examples\b", re.IGNORECASE),
        "mentions Before/After Examples header",
    ),
    (
        re.compile(r"\brecommended investigation areas\b", re.IGNORECASE),
        "mentions Recommended Investigation Areas header",
    ),
    (
        re.compile(r"\broot cause identified\b", re.IGNORECASE),
        "mentions Root Cause Identified header",
    ),
)


def check_rubric_criterion(text: str, skill_names: tuple[str, ...] = ALL_SKILL_NAMES) -> list[str]:
    """Return any skill-name or skill-ritual violations found in a rubric criterion string."""
    problems: list[str] = []
    problems.extend(SB.check_prompt(text, skill_names))
    for name in skill_names:
        pat = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])", re.IGNORECASE)
        if pat.search(text):
            problems.append(f"mentions skill name {name!r}")
    for regex, reason in FORBIDDEN_RUBRIC_PATTERNS:
        if regex.search(text):
            problems.append(reason)
    return problems


class RubricHygieneHelperTest(unittest.TestCase):
    def test_accepts_outcome_only_criterion(self) -> None:
        clean = (
            "Eliminates duplicated discount and tax calculations in orderflow/processor.py by "
            "extracting shared helper functions while keeping existing unit tests passing."
        )
        self.assertEqual(check_rubric_criterion(clean), [])

    def test_rejects_skill_names_and_rituals(self) -> None:
        dirty_cases = (
            ("Follows the make-plan workflow.", "make-plan"),
            ("Applies 5-Why analysis to find the defect.", "5-Why"),
            ("Includes Why #1 through Why #5 sections.", "Why #N"),
            ("Writes tasks with [ ] checkboxes for each step.", "checkbox"),
            ("Calls ask_question before writing the file.", "ask_question"),
            ("Includes a Post-Completion section at the end.", "Post-Completion"),
            ("Groups findings under High Impact Refactorings.", "High Impact Refactorings"),
            ("Reads SKILL.md before editing.", "SKILL.md"),
        )
        for text, expected in dirty_cases:
            with self.subTest(text=text):
                problems = check_rubric_criterion(text)
                self.assertTrue(
                    any(expected.lower() in p.lower() for p in problems),
                    f"expected {expected!r} in {problems}",
                )


class PilotCorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = SB.load_tasks(SB.TASKS_ROOT)

    def test_every_task_loads_and_discover_matches(self) -> None:
        self.assertGreaterEqual(len(self.tasks), 6)
        discovered = discover("skillbench")
        self.assertEqual(set(discovered), set(self.tasks))
        for tid, task in self.tasks.items():
            self.assertEqual(discovered[tid], task.target_skill)
            self.assertTrue(task.fixture.is_dir(), f"{tid}: missing fixture dir")
            self.assertGreaterEqual(len(task.rubric), SB.MIN_CRITERIA)
            self.assertLessEqual(len(task.rubric), SB.MAX_CRITERIA)

    def test_manifest_is_frozen_and_in_sync(self) -> None:
        manifest_hash = SB.check_manifest(SB.TASKS_ROOT)
        self.assertEqual(len(manifest_hash), 64)
        tasks_manifest = SB.TASKS_ROOT / SB.MANIFEST
        self.assertTrue(tasks_manifest.is_file())
        parent_manifest = SB.TASKS_ROOT.parent / SB.MANIFEST
        self.assertTrue(parent_manifest.is_file())
        self.assertEqual(
            json.loads(tasks_manifest.read_text(encoding="utf-8")),
            json.loads(parent_manifest.read_text(encoding="utf-8")),
        )

    def test_each_pilot_skill_has_exactly_two_pilot_tasks(self) -> None:
        by_skill: dict[str, list[SB.SkillbenchTask]] = {s: [] for s in PILOT_SKILLS}
        for task in self.tasks.values():
            if task.pilot:
                self.assertIn(task.target_skill, by_skill)
                by_skill[task.target_skill].append(task)
        for skill in PILOT_SKILLS:
            with self.subTest(skill=skill):
                self.assertEqual(
                    len(by_skill[skill]),
                    2,
                    f"expected 2 pilot tasks for {skill}, got {[t.task_id for t in by_skill[skill]]}",
                )

    def test_every_task_passes_prompt_check_against_all_18_skills(self) -> None:
        self.assertGreaterEqual(len(ALL_SKILL_NAMES), 18)
        for tid, task in self.tasks.items():
            with self.subTest(task_id=tid):
                self.assertEqual(
                    SB.check_prompt(task.prompt, ALL_SKILL_NAMES, task.forbidden_terms),
                    [],
                )
                full_prompt = SB.build_prompt(task)
                self.assertEqual(
                    SB.check_prompt(full_prompt, ALL_SKILL_NAMES, task.forbidden_terms),
                    [],
                )
                for word in ("rubric", "score", "skill.md"):
                    self.assertNotIn(word, full_prompt.lower())
                if task.target_skill in SB.ASK_QUESTION_SKILLS:
                    self.assertGreaterEqual(
                        len(task.user_answers),
                        4,
                        f"{tid}: expected scripted user_answers for {task.target_skill}",
                    )

    def test_no_rubric_criterion_mentions_skill_name_or_skill_step(self) -> None:
        for tid, task in self.tasks.items():
            for crit in task.rubric:
                with self.subTest(task_id=tid, criterion_id=crit.id):
                    self.assertEqual(check_rubric_criterion(crit.id), [])
                    self.assertEqual(check_rubric_criterion(crit.criterion), [])

    def test_fixture_repos_are_self_contained_and_run_unittest(self) -> None:
        for tid, task in self.tasks.items():
            with self.subTest(task_id=tid):
                self.assertTrue((task.fixture / "README.md").is_file(), f"{tid}: missing README.md")
                self.assertTrue((task.fixture / "tests").is_dir(), f"{tid}: missing tests/ dir")
                with tempfile.TemporaryDirectory(prefix="skillbench-fx-") as tmp:
                    work = Path(tmp) / "work"
                    shutil.copytree(task.fixture, work)
                    proc = subprocess.run(
                        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                        cwd=work,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if task.target_skill == "root-cause-investigator":
                        self.assertNotEqual(
                            proc.returncode,
                            0,
                            f"{tid}: expected seeded failing test in fixture",
                        )
                        self.assertIn("FAIL:", proc.stderr)
                    else:
                        self.assertEqual(
                            proc.returncode,
                            0,
                            f"{tid}: fixture tests failed:\n{proc.stderr}",
                        )


if __name__ == "__main__":
    unittest.main()
