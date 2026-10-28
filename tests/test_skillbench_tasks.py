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

"""skillbench task model: loader, prompt check, simulated-user preamble, freeze/manifest.

Every test builds its own task corpus in a temp dir; the bundled corpus is never touched.
"""

import contextlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from harness2.benchmarks import skillbench as SB
from harness2.substrate.agy import AgyTask

REPO = Path(__file__).resolve().parents[1]
# Skills of the bundled corpus whose SKILL.md allows `ask_question`; `make_task` scripts the
# simulated user for them. The loader itself no longer knows any skill names: whether a task
# asks is decided by `asks_user` in task.toml or the skill's frontmatter (`task_asks_user`).
ASKING_SKILLS = frozenset(
    {"brainstorm", "exec", "make-plan", "dialectic", "backlog", "clarify", "learn", "wrong"}
)
ASKING_SKILL_MD = (
    "---\nname: make-plan\ndescription: plan\n"
    "allowed-tools: view_file, write_to_file, run_command, ask_question\n---\n# Plan\n"
)
SILENT_SKILL_MD = (
    "---\nname: code-review-refactor\ndescription: refactor\n"
    "allowed-tools: view_file, replace_file_content\n---\n# Refactor\nask_question in body only\n"
)


def _rubric(n: int = 4) -> list[dict]:
    return [
        {"id": f"c{i}", "criterion": f"The deliverable satisfies outcome {i}."} for i in range(n)
    ]


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, str):
        return json.dumps(v)
    return str(v)


def make_task(
    root: Path,
    skill: str = "code-review-refactor",
    task_id: str = "t1",
    *,
    rubric=None,
    fixture: bool = True,
    **fields,
) -> Path:
    """Write `<root>/<skill>/<task_id>/`; `fields` override task.toml (None drops a key)."""
    d = root / skill / task_id
    d.mkdir(parents=True)
    meta = {
        "target_skill": skill,
        "prompt": "Tidy up utils.py: the helpers are hard to follow.",
        "timeout": 600,
        "pilot": False,
    }
    if skill in ASKING_SKILLS:
        meta["user_answers"] = ["Yes, go ahead.", "Use the simplest option."]
    meta.update(fields)
    body = "".join(f"{k} = {_toml_value(v)}\n" for k, v in meta.items() if v is not None)
    (d / "task.toml").write_text(body, encoding="utf-8")
    (d / "rubric.json").write_text(
        json.dumps(_rubric() if rubric is None else rubric), encoding="utf-8"
    )
    if fixture:
        (d / "fixture").mkdir()
        (d / "fixture" / "utils.py").write_text("def f(x):\n    return x\n", encoding="utf-8")
    return d


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tasks"
        self.root.mkdir()

    def tearDown(self):
        self._tmp.cleanup()


class LoadTaskTest(TmpCase):
    def test_valid_task(self):
        d = make_task(self.root, forbidden_terms=["refactor"], prompt="Tidy up utils.py.")
        t = SB.load_task(d)
        self.assertEqual(t.task_id, "code-review-refactor/t1")
        self.assertEqual(t.target_skill, "code-review-refactor")
        self.assertEqual(t.timeout, 600)
        self.assertEqual(len(t.rubric), 4)
        self.assertEqual(t.rubric[0], SB.Criterion("c0", "The deliverable satisfies outcome 0."))
        self.assertEqual(t.forbidden_terms, ("refactor",))
        self.assertEqual(t.user_answers, ())
        self.assertFalse(t.pilot)
        self.assertEqual(t.fixture, d / "fixture")

    def test_pilot_and_answers(self):
        d = make_task(self.root, "make-plan", "p1", pilot=True, user_answers=["A", "B"])
        t = SB.load_task(d)
        self.assertTrue(t.pilot)
        self.assertEqual(t.user_answers, ("A", "B"))

    def test_pilot_defaults_false(self):
        self.assertFalse(SB.load_task(make_task(self.root, pilot=None)).pilot)

    def test_missing_required_fields(self):
        for i, key in enumerate(("target_skill", "prompt", "timeout")):
            with self.subTest(key=key):
                d = make_task(self.root, task_id=f"m{i}", **{key: None})
                with self.assertRaisesRegex(SB.TaskError, f"missing required field.*{key}"):
                    SB.load_task(d)

    def test_missing_files(self):
        d = make_task(self.root, task_id="nofix", fixture=False)
        with self.assertRaisesRegex(SB.TaskError, "fixture"):
            SB.load_task(d)
        d = make_task(self.root, task_id="norub")
        (d / "rubric.json").unlink()
        with self.assertRaisesRegex(SB.TaskError, "rubric.json: missing"):
            SB.load_task(d)
        d = make_task(self.root, task_id="notoml")
        (d / "task.toml").unlink()
        with self.assertRaisesRegex(SB.TaskError, "task.toml: missing"):
            SB.load_task(d)

    def test_bad_values(self):
        cases = {
            "timeout0": {"timeout": 0},
            "timeoutstr": {"timeout": "600"},
            "timeoutbool": {"timeout": True},
            "emptyprompt": {"prompt": "  "},
            "pilotstr": {"pilot": "yes"},
            "answers": {"user_answers": ["ok", ""]},
            "unknown": {"user_answer": ["typo"]},
        }
        for tid, fields in cases.items():
            with self.subTest(tid=tid):
                with self.assertRaises(SB.TaskError):
                    SB.load_task(make_task(self.root, task_id=tid, **fields))

    def test_target_skill_must_match_directory(self):
        d = make_task(self.root, task_id="x", target_skill="dialectic")
        with self.assertRaisesRegex(SB.TaskError, "<target_skill>/<task-id>"):
            SB.load_task(d)

    def test_malformed_toml_and_json(self):
        d = make_task(self.root, task_id="badtoml")
        (d / "task.toml").write_text("prompt = \n", encoding="utf-8")
        with self.assertRaisesRegex(SB.TaskError, "not valid TOML"):
            SB.load_task(d)
        d = make_task(self.root, task_id="badjson")
        (d / "rubric.json").write_text("[{", encoding="utf-8")
        with self.assertRaisesRegex(SB.TaskError, "not valid JSON"):
            SB.load_task(d)

    def test_criteria_count(self):
        for n in (0, 3, 9):
            with self.subTest(n=n):
                d = make_task(self.root, task_id=f"n{n}", rubric=_rubric(n))
                with self.assertRaisesRegex(SB.TaskError, f"has {n} criteria"):
                    SB.load_task(d)
        for n in (4, 8):
            self.assertEqual(
                len(SB.load_task(make_task(self.root, task_id=f"ok{n}", rubric=_rubric(n))).rubric),
                n,
            )

    def test_duplicate_and_malformed_criteria(self):
        dup = _rubric()
        dup[2]["id"] = "c0"
        with self.assertRaisesRegex(SB.TaskError, "duplicate criterion id 'c0'"):
            SB.load_task(make_task(self.root, task_id="dup", rubric=dup))
        bad = _rubric()
        bad[1] = {"id": "c1"}
        with self.assertRaisesRegex(SB.TaskError, "exactly the keys"):
            SB.load_task(make_task(self.root, task_id="bad", rubric=bad))
        with self.assertRaisesRegex(SB.TaskError, "JSON list"):
            SB.load_task(make_task(self.root, task_id="obj", rubric={"c0": "x"}))

    def test_loader_no_longer_requires_user_answers_for_asking_skills(self):
        # The "asking task without user_answers" check moved to `skillbench-tasks check`;
        # `load_task` accepts the task and records that nothing was scripted.
        t = SB.load_task(
            make_task(
                self.root, "make-plan", "q", user_answers=None, prompt="Help with release notes."
            )
        )
        self.assertEqual(t.user_answers, ())
        self.assertIsNone(t.asks_user)
        # Skills that never ask need none.
        self.assertEqual(
            SB.load_task(
                make_task(
                    self.root, "root-cause-investigator", "r", prompt="Why does make test fail?"
                )
            ).user_answers,
            (),
        )

    def test_missing_user_answers_is_reported_by_skillbench_tasks_check(self):
        from harness2.cli import skillbench_tasks as T

        d = make_task(self.root, "make-plan", "q", user_answers=None, asks_user=True)
        SB.load_task(d)  # loads fine ...
        res = T.check_task("make-plan/q", d, suite=None, frozen=None)
        self.assertFalse(res.ok)  # ... but the corpus check flags it
        self.assertTrue(any("user_answers" in p for p in res.problems), res.problems)
        ok = make_task(self.root, "make-plan", "ok", asks_user=True)
        self.assertTrue(T.check_task("make-plan/ok", ok, suite=None, frozen=None).ok)

    def test_asks_user_override_is_parsed_and_validated(self):
        t = SB.load_task(make_task(self.root, "code-review-refactor", "a", asks_user=True))
        self.assertIs(t.asks_user, True)
        self.assertTrue(SB.task_asks_user(t))
        self.assertTrue(SB.task_asks_user(t, SILENT_SKILL_MD))  # override beats frontmatter
        t2 = SB.load_task(make_task(self.root, "make-plan", "b", asks_user=False))
        self.assertIs(t2.asks_user, False)
        self.assertFalse(SB.task_asks_user(t2, ASKING_SKILL_MD))
        with self.assertRaisesRegex(SB.TaskError, "asks_user must be true or false"):
            SB.load_task(make_task(self.root, "make-plan", "c", asks_user="yes"))

    def test_task_asks_user_falls_back_to_frontmatter(self):
        t = SB.load_task(make_task(self.root, "make-plan", "f"))
        self.assertIsNone(t.asks_user)
        self.assertTrue(SB.task_asks_user(t, ASKING_SKILL_MD))
        self.assertFalse(SB.task_asks_user(t, SILENT_SKILL_MD))
        self.assertFalse(SB.task_asks_user(t))  # no SKILL.md known -> not an asking skill

    def test_prompt_check_runs_in_loader(self):
        d = make_task(self.root, task_id="inv", prompt="Use the code-review-refactor skill.")
        with self.assertRaisesRegex(SB.TaskError, "invokes the skill explicitly"):
            SB.load_task(d)
        d = make_task(
            self.root,
            task_id="forb",
            prompt="Please refactor utils.py.",
            forbidden_terms=["Refactor"],
        )
        with self.assertRaisesRegex(SB.TaskError, "forbidden term"):
            SB.load_task(d)

    def test_load_tasks_and_task_dirs(self):
        make_task(self.root, "make-plan", "b")
        make_task(self.root, "make-plan", "a")
        make_task(self.root, "code-review-refactor", "z")
        (self.root / "make-plan" / "notask").mkdir()
        tasks = SB.load_tasks(self.root)
        self.assertEqual(list(tasks), ["code-review-refactor/z", "make-plan/a", "make-plan/b"])
        self.assertEqual(list(SB.task_dirs(self.root)), list(tasks))


class PromptCheckTest(unittest.TestCase):
    def check(self, prompt, names=("backlog",), forbidden=()):
        return SB.check_prompt(prompt, names, forbidden)

    def test_rejects_explicit_invocations(self):
        for prompt in (
            "Run the backlog skill on this repo.",
            "Use your Backlog Skill please.",
            "/backlog add the flaky test",
            "Please run /backlog now.",
            "Read plugins/workflow/skills/backlog/SKILL.md first.",
            "see skills/backlog for details",
        ):
            with self.subTest(prompt=prompt):
                self.assertTrue(self.check(prompt))

    def test_rejects_skill_md_and_forbidden_terms(self):
        self.assertIn("mentions SKILL.md", self.check("Follow the skill.md steps.", names=()))
        self.assertEqual(
            self.check("Do a 5-Why analysis.", names=(), forbidden=["5-why"]),
            ["forbidden term '5-why'"],
        )

    def test_accepts_ordinary_words(self):
        for prompt in (
            "Add this to the backlog.",
            "Something went wrong, help me learn why.",
            "File it under docs/backlog/ like the others.",
            "What would you learn from this? It all went wrong at /tmp/x.",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(self.check(prompt, ("backlog", "wrong", "learn")), [])

    def test_hyphenated_names_match_whole(self):
        self.assertEqual(self.check("Use the make-plan-ish skill", names=("make-plan",)), [])
        self.assertEqual(self.check("/make-planner", names=("make-plan",)), [])
        self.assertTrue(self.check("/make-plan for this", names=("make-plan",)))


class BuildPromptTest(TmpCase):
    def test_appends_preamble_with_answers_in_order(self):
        t = SB.load_task(
            make_task(self.root, "make-plan", "p", user_answers=["First answer", "Second answer"])
        )
        p = SB.build_prompt(t)
        self.assertTrue(p.startswith(t.prompt.rstrip() + "\n\n" + SB.PREAMBLE_HEAD))
        self.assertLess(p.index("1. First answer"), p.index("2. Second answer"))
        self.assertIn("recommended option", p)
        self.assertIn("Never open an editor", p)
        self.assertTrue(p.endswith(SB.simulated_user_preamble(t.user_answers)))

    def test_no_answers(self):
        t = SB.load_task(make_task(self.root))
        p = SB.build_prompt(t)
        self.assertIn(SB.PREAMBLE_HEAD, p)
        self.assertIn("recommended option", p)
        self.assertNotIn("1.", p)

    def test_identical_for_both_arms(self):
        # build_prompt takes no arm: the text cannot differ between WITH and WITHOUT.
        self.assertEqual(list(inspect.signature(SB.build_prompt).parameters), ["task"])
        t = SB.load_task(make_task(self.root, "make-plan", "p"))
        self.assertEqual(SB.build_prompt(t), SB.build_prompt(SB.load_task(t.dir)))
        self.assertEqual(SB.agy_task(t), AgyTask(SB.build_prompt(t), t.dir / "fixture"))

    def test_preamble_passes_prompt_check(self):
        pre = SB.simulated_user_preamble(["yes"])
        self.assertEqual(SB.check_prompt(pre, sorted(ASKING_SKILLS)), [])
        for word in ("rubric", "score", "skill"):
            self.assertNotIn(word, pre.lower())


class FreezeTest(TmpCase):
    def setUp(self):
        super().setUp()
        self.a = make_task(self.root, "make-plan", "a")
        self.b = make_task(self.root, "code-review-refactor", "b")

    def test_freeze_writes_manifest_and_check_passes(self):
        m = SB.freeze(self.root)
        data = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(data, m)
        self.assertEqual(sorted(m["tasks"]), ["code-review-refactor/b", "make-plan/a"])
        self.assertTrue(all(len(h) == 64 for h in m["tasks"].values()))
        self.assertEqual(SB.check_manifest(self.root), m["hash"])

    def test_refuses_overwrite_without_force(self):
        SB.freeze(self.root)
        with self.assertRaisesRegex(SB.ManifestError, "--force"):
            SB.freeze(self.root)
        (self.a / "fixture" / "new.txt").write_text("n\n")
        m = SB.freeze(self.root, force=True)
        self.assertEqual(SB.check_manifest(self.root), m["hash"])

    def test_freeze_validates_tasks(self):
        (self.b / "rubric.json").write_text("[]")
        with self.assertRaises(SB.TaskError):
            SB.freeze(self.root)
        self.assertFalse((self.root / "manifest.json").exists())

    def test_freeze_empty_corpus(self):
        empty = self.root.parent / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(SB.ManifestError, "no skillbench tasks"):
            SB.freeze(empty)

    def test_missing_manifest(self):
        with self.assertRaisesRegex(SB.ManifestError, "not frozen"):
            SB.check_manifest(self.root)

    def test_changed_task(self):
        SB.freeze(self.root)
        (self.a / "fixture" / "utils.py").write_text("def f(x):\n    return x + 1\n")
        with self.assertRaisesRegex(SB.ManifestError, "changed: make-plan/a"):
            SB.check_manifest(self.root)

    def test_changed_rubric_or_prompt(self):
        SB.freeze(self.root)
        (self.b / "rubric.json").write_text(json.dumps(_rubric(5)))
        with self.assertRaisesRegex(SB.ManifestError, "changed: code-review-refactor/b"):
            SB.check_manifest(self.root)

    def test_added_task(self):
        SB.freeze(self.root)
        make_task(self.root, "make-plan", "c")
        with self.assertRaisesRegex(SB.ManifestError, "added: make-plan/c"):
            SB.check_manifest(self.root)

    def test_removed_task(self):
        SB.freeze(self.root)
        (self.b / "task.toml").unlink()
        with self.assertRaisesRegex(SB.ManifestError, "removed: code-review-refactor/b"):
            SB.check_manifest(self.root)

    def test_hand_edited_manifest(self):
        SB.freeze(self.root)
        path = self.root / "manifest.json"
        data = json.loads(path.read_text())
        data["tasks"]["make-plan/a"] = "0" * 64
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(SB.ManifestError, "malformed or edited"):
            SB.check_manifest(self.root)
        path.write_text("{")
        with self.assertRaisesRegex(SB.ManifestError, "not valid JSON"):
            SB.check_manifest(self.root)

    def test_hash_ignores_git_and_pycache_but_sees_symlinks(self):
        h = SB.hash_task_dir(self.a)
        (self.a / "fixture" / "__pycache__").mkdir()
        (self.a / "fixture" / "__pycache__" / "x.pyc").write_bytes(b"\0")
        (self.a / "fixture" / ".git").mkdir()
        (self.a / "fixture" / ".git" / "HEAD").write_text("ref\n")
        self.assertEqual(SB.hash_task_dir(self.a), h)
        os.symlink("utils.py", self.a / "fixture" / "alias.py")
        h2 = SB.hash_task_dir(self.a)
        self.assertNotEqual(h2, h)
        os.symlink("..", self.a / "fixture" / "up")  # a dir symlink: recorded, never followed
        self.assertNotEqual(SB.hash_task_dir(self.a), h2)

    def test_hash_stable_across_copies(self):
        import shutil

        other = self.root.parent / "copy" / "make-plan" / "a"
        shutil.copytree(self.a, other)
        self.assertEqual(SB.hash_task_dir(other), SB.hash_task_dir(self.a))


class CliTest(TmpCase):
    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = SB.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_freeze_force_and_check(self):
        make_task(self.root)
        rc, out, _ = self.run_main("freeze", "--root", str(self.root))
        self.assertEqual(rc, 0)
        self.assertIn("froze 1 task", out)
        before = (self.root / "manifest.json").read_text()
        rc, _, err = self.run_main("freeze", "--root", str(self.root))
        self.assertEqual(rc, 1)
        self.assertIn("--force", err)
        self.assertEqual((self.root / "manifest.json").read_text(), before)
        rc, _, _ = self.run_main("freeze", "--force", "--root", str(self.root))
        self.assertEqual(rc, 0)
        rc, out, _ = self.run_main("check", "--root", str(self.root))
        self.assertEqual((rc, out.startswith("ok:")), (0, True))
        make_task(self.root, task_id="t2")
        rc, _, err = self.run_main("check", "--root", str(self.root))
        self.assertEqual(rc, 1)
        self.assertIn("added", err)

    def test_module_entry_point(self):
        make_task(self.root)
        cmd = [
            sys.executable,
            "-m",
            "harness2.benchmarks.skillbench",
            "freeze",
            "--root",
            str(self.root),
        ]
        first = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=60)
        self.assertEqual(first.returncode, 0, first.stderr)
        second = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=60)
        self.assertEqual(second.returncode, 1)
        self.assertIn("--force", second.stderr)
        self.assertNotIn("RuntimeWarning", first.stderr)


if __name__ == "__main__":
    unittest.main()
