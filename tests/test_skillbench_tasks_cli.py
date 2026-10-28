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

"""skillbench-tasks (scaffold + check) and the corpus-root resolution it builds on.

Everything runs against synthetic suites and corpora in a temp dir; no model is called and
the bundled corpus is never written.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness2.benchmarks import skillbench as SB
from harness2.benchmarks import tasks as TASKS
from harness2.benchmarks.skillbench_suite import SuiteError
from harness2.cli import skillbench_tasks as T

ASKING_MD = (
    "---\nname: planner\ndescription: plans\n"
    "allowed-tools: view_file, write_to_file, ask_question\n---\n# Planner\n"
    "Run `${PLUGIN_ROOT}/scripts/resolve-dir.sh` first, then `scripts/annotate.py`.\n"
    "Never run setup.py here.\n"
)
SILENT_MD = (
    "---\nname: tidier\ndescription: tidies\nallowed-tools: view_file, replace_file_content\n---\n"
    "# Tidier\nMentions ask_question in the body only.\n"
)
RUBRIC = [{"id": f"c{i}", "criterion": f"The deliverable meets outcome {i}."} for i in range(4)]
ENV_KEYS = ("SKILLBENCH_SUITE", "SKILLBENCH_PLUGINS_DIR", "SKILLBENCH_PLUGINS", SB.TASKS_ROOT_ENV)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def make_suite(tmp: Path, *, tasks_root: Path | None = None, families: bool = True) -> Path:
    """A two-plugin suite with an asking skill (`planner`) and a silent one (`tidier`)."""
    plugins = tmp / "checkout" / "plugins"
    _write(plugins / "alpha" / "plugin.json", '{"name": "alpha"}')
    _write(plugins / "alpha" / "skills" / "planner" / "SKILL.md", ASKING_MD)
    _write(plugins / "beta" / "plugin.json", '{"name": "beta"}')
    _write(plugins / "beta" / "skills" / "tidier" / "SKILL.md", SILENT_MD)
    body = f'name = "other"\nplugins_dir = "{plugins}"\nplugins = ["alpha", "beta"]\n'
    if tasks_root is not None:
        body += f'tasks_root = "{tasks_root}"\n'
    if families:
        body += '[families]\nplanning = ["planner"]\n'
    return _write(tmp / "suite.toml", body)


def make_task(
    root: Path,
    skill: str,
    task_id: str,
    *,
    prompt: str = "Tidy up utils.py; the helpers are hard to follow.",
    rubric=None,
    fixture_files: dict[str, str] | None = None,
    **fields,
) -> Path:
    d = root / skill / task_id
    meta: dict = {"target_skill": skill, "prompt": prompt, "timeout": 60}
    meta.update(fields)
    lines = []
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (list, tuple)):
            lines.append(f"{k} = [{', '.join(json.dumps(x) for x in v)}]")
        elif isinstance(v, str):
            lines.append(f"{k} = {json.dumps(v)}")
        else:
            lines.append(f"{k} = {v}")
    _write(d / "task.toml", "\n".join(lines) + "\n")
    _write(d / "rubric.json", json.dumps(RUBRIC if rubric is None else rubric))
    files = {"utils.py": "def f(x):\n    return x\n"} if fixture_files is None else fixture_files
    (d / "fixture").mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        _write(d / "fixture" / rel, text)
    return d


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        clean = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
        patcher = mock.patch.dict(os.environ, clean, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = T.main(list(argv))
        return rc, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------------------------
# tasks_root resolution
# --------------------------------------------------------------------------------------------


class TasksRootTest(TmpCase):
    def test_resolution_order_explicit_suite_env_bundled(self):
        suite_root = self.tmp / "from_suite"
        env_root = self.tmp / "from_env"
        explicit = self.tmp / "explicit"
        suite_file = make_suite(self.tmp, tasks_root=suite_root)
        os.environ["SKILLBENCH_SUITE"] = str(suite_file)
        os.environ[SB.TASKS_ROOT_ENV] = str(env_root)
        # 1. explicit flag wins over everything
        self.assertEqual(SB.tasks_root(explicit), explicit)
        self.assertEqual(SB.tasks_root(""), suite_root)  # empty string = not given
        # 2. suite.tasks_root
        self.assertEqual(SB.tasks_root(), suite_root)
        # 3. env var when the suite sets no tasks_root
        os.environ["SKILLBENCH_SUITE"] = str(make_suite(self.tmp, tasks_root=None))
        self.assertEqual(SB.tasks_root(), env_root)
        # 4. bundled corpus when nothing is configured
        del os.environ[SB.TASKS_ROOT_ENV]
        del os.environ["SKILLBENCH_SUITE"]
        self.assertEqual(SB.tasks_root(), SB.TASKS_ROOT)
        # the bundled fallback follows the module constant (tests patch it)
        with mock.patch.object(SB, "TASKS_ROOT", self.tmp / "patched"):
            self.assertEqual(SB.tasks_root(), self.tmp / "patched")

    def test_explicit_suite_object_and_tilde(self):
        from harness2.benchmarks.skillbench_suite import load_suite

        suite = load_suite(make_suite(self.tmp, tasks_root=self.tmp / "s"))
        self.assertEqual(SB.tasks_root(None, suite), self.tmp / "s")
        os.environ[SB.TASKS_ROOT_ENV] = "~/corpus"
        self.assertEqual(SB.tasks_root(), Path("~/corpus").expanduser())

    def test_broken_suite_file_is_an_error_not_a_fallback(self):
        os.environ["SKILLBENCH_SUITE"] = str(_write(self.tmp / "bad.toml", "name = 1\n"))
        with self.assertRaises(SuiteError):
            SB.tasks_root()

    def test_discover_freeze_check_and_adapter_honour_the_root(self):
        root = self.tmp / "corpus"
        make_task(root, "tidier", "t1")
        os.environ["SKILLBENCH_SUITE"] = str(make_suite(self.tmp, tasks_root=root))
        self.assertEqual(TASKS.discover("skillbench"), {"tidier/t1": "tidier"})
        m = SB.freeze()
        self.assertTrue((root / "manifest.json").is_file())
        self.assertEqual(SB.check_manifest(), m["hash"])
        self.assertEqual(list(SB.load_tasks()), ["tidier/t1"])
        with mock.patch.dict(os.environ, {"HARNESS2_RUNS_DIR": str(self.tmp / "runs")}):
            adapter = SB.SkillbenchAdapter("with", runs_root=self.tmp / "runs")
        self.assertEqual(adapter._root, root)
        self.assertEqual(adapter.tasks(), ["tidier/t1"])
        # the module CLI resolves --root the same way
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = SB.main(["check"])
        self.assertEqual((rc, out.getvalue().startswith("ok:")), (0, True))
        self.assertIn(str(root), out.getvalue())


# --------------------------------------------------------------------------------------------
# task.toml: kind / family / min_baseline_fail
# --------------------------------------------------------------------------------------------


class KindAndFamilyTest(TmpCase):
    def test_suite_task_loads_from_suite_dir(self):
        d = make_task(
            self.tmp, SB.SUITE_DIR, "e2e", target_skill=None, kind="suite", family="planning"
        )
        t = SB.load_task(d)
        self.assertEqual((t.kind, t.family, t.target_skill), ("suite", "planning", ""))
        self.assertEqual(t.task_id, "_suite/e2e")
        self.assertEqual(t.domain, "planning")
        self.assertEqual(t.min_baseline_fail, 1)

    def test_skill_task_defaults(self):
        t = SB.load_task(make_task(self.tmp, "tidier", "t", min_baseline_fail=2))
        self.assertEqual((t.kind, t.family, t.domain, t.min_baseline_fail), (
            "skill", None, "tidier", 2
        ))

    def test_kind_and_family_errors(self):
        cases = {
            "badkind": dict(kind="other"),
            "suite_with_skill": dict(kind="suite", family="planning"),  # target_skill kept
            "family_on_skill_task": dict(family="planning"),
            "badfamily": dict(kind="suite", target_skill=None, family="Bad Name"),
            "nofamily": dict(kind="suite", target_skill=None),
            "minfail0": dict(min_baseline_fail=0),
            "minfailbool": dict(min_baseline_fail=True),
        }
        for tid, fields in cases.items():
            with self.subTest(tid=tid):
                with self.assertRaises(SB.TaskError):
                    SB.load_task(make_task(self.tmp, "tidier", tid, **fields))
        # a suite task outside _suite/
        d = make_task(self.tmp, "tidier", "misplaced", target_skill=None, kind="suite",
                      family="planning")
        with self.assertRaisesRegex(SB.TaskError, "_suite"):
            SB.load_task(d)
        # a skill task without target_skill names the missing key
        with self.assertRaisesRegex(SB.TaskError, "target_skill"):
            SB.load_task(make_task(self.tmp, "tidier", "noskill", target_skill=None))

    def test_adapter_schedules_skill_tasks_only(self):
        root = self.tmp / "corpus"
        make_task(root, "tidier", "t1")
        make_task(root, SB.SUITE_DIR, "e2e", target_skill=None, kind="suite", family="planning")
        os.environ["SKILLBENCH_SUITE"] = str(make_suite(self.tmp, tasks_root=root))
        SB.freeze(root)
        with mock.patch.dict(os.environ, {"HARNESS2_RUNS_DIR": str(self.tmp / "runs")}):
            adapter = SB.SkillbenchAdapter("without", runs_root=self.tmp / "runs")
        self.assertEqual(adapter.tasks(), ["tidier/t1"])
        self.assertEqual(adapter.reference_domain_order(), ["tidier"])
        self.assertEqual(adapter.domain_of("_suite/e2e"), "planning")
        self.assertIn("_suite/e2e", SB.load_tasks(root))


# --------------------------------------------------------------------------------------------
# new
# --------------------------------------------------------------------------------------------


class ScaffoldTest(TmpCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "corpus"
        self.suite_file = make_suite(self.tmp, tasks_root=self.root)

    def test_scaffold_asking_skill(self):
        rc, out, err = self.run_main(
            "new", "--skill", "planner", "--id", "plan-1", "--suite", str(self.suite_file)
        )
        self.assertEqual(rc, 0, err)
        d = self.root / "planner" / "plan-1"
        self.assertIn(str(d), out)
        for rel in ("task.toml", "rubric.json", "fixture/README.md"):
            self.assertTrue((d / rel).is_file(), rel)
        toml = (d / "task.toml").read_text()
        self.assertIn('target_skill = "planner"', toml)
        self.assertIn("asks_user = true", toml)
        self.assertIn("user_answers = []", toml)
        self.assertIn("# verify =", toml)
        self.assertIn("# verify_restore =", toml)
        t = SB.load_task(d)  # the scaffold is a loadable task
        self.assertEqual(t.forbidden_terms, ("planner", "annotate", "resolve-dir"))
        self.assertIs(t.asks_user, True)
        self.assertEqual(t.timeout, T.DEFAULT_TIMEOUT)
        self.assertEqual(len(t.rubric), 4)
        self.assertTrue(all(c.criterion == T.CRITERION_PLACEHOLDER for c in t.rubric))

    def test_scaffold_silent_skill_and_prompt(self):
        rc, _, err = self.run_main(
            "new", "--skill", "tidier", "--id", "t-1", "--suite", str(self.suite_file),
            "--prompt", "Clean up utils.py.", "--timeout", "120",
        )
        self.assertEqual(rc, 0, err)
        t = SB.load_task(self.root / "tidier" / "t-1")
        self.assertIs(t.asks_user, False)
        self.assertEqual(t.user_answers, ())
        self.assertEqual(t.prompt, "Clean up utils.py.")
        self.assertEqual(t.timeout, 120)
        self.assertEqual(t.forbidden_terms, ("tidier",))

    def test_scaffold_suite_task(self):
        rc, _, err = self.run_main(
            "new", "--kind", "suite", "--family", "planning", "--id", "e2e",
            "--suite", str(self.suite_file),
        )
        self.assertEqual(rc, 0, err)
        t = SB.load_task(self.root / SB.SUITE_DIR / "e2e")
        self.assertEqual((t.kind, t.family, t.target_skill), ("suite", "planning", ""))

    def test_scaffold_without_suite_leaves_asks_user_open(self):
        rc, out, _ = self.run_main("new", "--skill", "anything", "--id", "x", "--tasks",
                                   str(self.root))
        self.assertEqual(rc, 0)
        self.assertIn("note: no suite configured", out)
        toml = (self.root / "anything" / "x" / "task.toml").read_text()
        self.assertIn("# asks_user = true", toml)
        self.assertIsNone(SB.load_task(self.root / "anything" / "x").asks_user)

    def test_scaffold_errors(self):
        s = str(self.suite_file)
        cases = {
            "unknown skill": ["new", "--skill", "nope", "--id", "x", "--suite", s],
            "unknown family": ["new", "--kind", "suite", "--family", "nope", "--id", "x",
                               "--suite", s],
            "suite needs family": ["new", "--kind", "suite", "--id", "x", "--suite", s],
            "suite rejects skill": ["new", "--kind", "suite", "--family", "planning",
                                    "--skill", "planner", "--id", "x", "--suite", s],
            "family on skill": ["new", "--skill", "planner", "--family", "planning", "--id",
                                "x", "--suite", s],
            "no skill": ["new", "--id", "x", "--suite", s],
            "bad id": ["new", "--skill", "planner", "--id", "bad id", "--suite", s],
            "bad timeout": ["new", "--skill", "planner", "--id", "x", "--timeout", "0",
                            "--suite", s],
            "prompt names skill": ["new", "--skill", "planner", "--id", "x", "--suite", s,
                                   "--prompt", "Use the planner skill."],
            "broken suite": ["new", "--skill", "planner", "--id", "x", "--suite",
                             str(_write(self.tmp / "bad.toml", "name = 1\n"))],
        }
        for name, argv in cases.items():
            with self.subTest(name=name):
                rc, _, err = self.run_main(*argv)
                self.assertEqual(rc, 2, err)
                self.assertTrue(err.startswith("error:"), err)
        self.assertFalse((self.root / "planner" / "x").exists())

    def test_refuses_overwrite_without_force(self):
        argv = ["new", "--skill", "tidier", "--id", "t", "--suite", str(self.suite_file)]
        self.assertEqual(self.run_main(*argv)[0], 0)
        marker = _write(self.root / "tidier" / "t" / "fixture" / "keep.py", "x = 1\n")
        rc, _, err = self.run_main(*argv)
        self.assertEqual(rc, 2)
        self.assertIn("--force", err)
        self.assertEqual(self.run_main(*argv, "--force")[0], 0)
        self.assertTrue(marker.is_file())  # --force rewrites the scaffold files only

    def test_script_stems(self):
        self.assertEqual(T.script_stems(ASKING_MD), ["annotate", "resolve-dir"])
        self.assertEqual(T.script_stems(""), [])
        self.assertEqual(T.script_stems("run tests.py and do_it.sh"), ["do_it"])


# --------------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------------


class CheckTest(TmpCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "corpus"
        self.suite_file = make_suite(self.tmp, tasks_root=self.root)
        self.s = str(self.suite_file)

    def good_corpus(self) -> None:
        make_task(self.root, "planner", "p1", user_answers=["Yes.", "Simplest."],
                  prompt="Write a plan for adding retries to the mailer.")
        make_task(self.root, "tidier", "t1")
        make_task(self.root, SB.SUITE_DIR, "e2e", target_skill=None, kind="suite",
                  family="planning", prompt="Plan, implement and review a retry helper.")

    def test_check_success_table_and_manifest_column(self):
        self.good_corpus()
        rc, out, err = self.run_main("check", "--suite", self.s)
        self.assertEqual(rc, 0, err + out)
        self.assertIn("suite: other", out)
        lines = out.splitlines()
        self.assertTrue(lines[1].startswith("task"))
        rows = {ln.split()[0]: ln.split() for ln in lines[3:] if "/" in ln}
        self.assertEqual(set(rows), {"planner/p1", "tidier/t1", "_suite/e2e"})
        # task kind rubric asks answers manifest status
        self.assertEqual(rows["planner/p1"][1:], ["skill", "4", "yes", "2", "-", "ok"])
        self.assertEqual(rows["tidier/t1"][1:], ["skill", "4", "no", "0", "-", "ok"])
        self.assertEqual(rows["_suite/e2e"][1:], ["suite", "4", "-", "0", "-", "ok"])
        self.assertIn("3 task(s), 0 failing", out)
        # manifest column: ok / changed / new once frozen; drift is not a failure
        SB.freeze(self.root)
        _write(self.root / "tidier" / "t1" / "fixture" / "more.py", "y = 2\n")
        make_task(self.root, "tidier", "t2")
        rc, out, _ = self.run_main("check", "--suite", self.s)
        self.assertEqual(rc, 0)
        rows = {ln.split()[0]: ln.split() for ln in out.splitlines() if "/" in ln}
        self.assertEqual(rows["tidier/t1"][5], "changed")
        self.assertEqual(rows["tidier/t2"][5], "new")
        self.assertEqual(rows["planner/p1"][5], "ok")

    def test_each_failure_path(self):
        self.good_corpus()
        # 1. asking task without user_answers
        make_task(self.root, "planner", "noanswers", prompt="Plan the retry feature.")
        # 2. asks_user override forces the check on a silent skill
        make_task(self.root, "tidier", "override", asks_user=True)
        # 3. prompt names another suite skill (load_task only knows the task's own skill)
        make_task(self.root, "tidier", "crossref", prompt="Run /planner then tidy utils.py.")
        # 4. malformed task (rubric too small) is reported, not raised
        make_task(self.root, "tidier", "smallrubric", rubric=RUBRIC[:3])
        # 5. skill not in the suite
        make_task(self.root, "ghost", "g1")
        # 6. unknown family
        make_task(self.root, SB.SUITE_DIR, "badfam", target_skill=None, kind="suite",
                  family="ops", prompt="Do the ops thing.")
        # 7. empty fixture
        make_task(self.root, "tidier", "nofixture", fixture_files={})
        # 8. scaffold placeholders left behind
        self.assertEqual(
            self.run_main("new", "--skill", "tidier", "--id", "stub", "--suite", self.s)[0], 0
        )
        rc, out, _ = self.run_main("check", "--suite", self.s)
        self.assertEqual(rc, 1)
        rows = {ln.split()[0]: ln for ln in out.splitlines() if "/" in ln}
        for tid in ("planner/p1", "tidier/t1", "_suite/e2e"):
            self.assertTrue(rows[tid].endswith("ok"), rows[tid])
        expect = {
            "planner/noanswers": "asking task without user_answers",
            "tidier/override": "asking task without user_answers",
            "tidier/crossref": "slash command '/planner'",
            "tidier/smallrubric": "has 3 criteria",
            "ghost/g1": "not provided by suite 'other'",
            "_suite/badfam": "family 'ops' not in suite 'other'",
            "tidier/nofixture": "fixture/ is empty",
            "tidier/stub": "still the scaffold placeholder",
        }
        for tid, needle in expect.items():
            with self.subTest(tid=tid):
                self.assertTrue(rows[tid].endswith("FAIL"), rows[tid])
                self.assertIn(needle, out)
        self.assertIn("rubric criteria still stubs: c1, c2, c3, c4", out)
        self.assertIn(f"{len(expect) + 3} task(s), {len(expect)} failing", out)

    def test_check_without_suite_uses_overrides_only(self):
        make_task(self.root, "planner", "p1", prompt="Plan the retry feature.")  # no answers
        make_task(self.root, "tidier", "t1", asks_user=True)
        rc, out, _ = self.run_main("check", "--tasks", str(self.root))
        self.assertEqual(rc, 1)
        self.assertIn("suite: (none)", out)
        rows = {ln.split()[0]: ln.split() for ln in out.splitlines() if "/" in ln}
        self.assertEqual(rows["planner/p1"][3:], ["-", "0", "-", "ok"])  # frontmatter unknown
        self.assertEqual(rows["tidier/t1"][-1], "FAIL")

    def test_empty_or_missing_root_fails(self):
        rc, out, _ = self.run_main("check", "--tasks", str(self.tmp / "nowhere"))
        self.assertEqual(rc, 1)
        self.assertIn("no tasks under", out)
        (self.tmp / "empty").mkdir()
        self.assertEqual(self.run_main("check", "--tasks", str(self.tmp / "empty"))[0], 1)

    def test_bundled_corpus_passes_check(self):
        suite = Path(SB.TASKS_ROOT).parent / "suites" / "cc-thingz.toml"
        try:
            from harness2.benchmarks.skillbench_suite import load_suite

            load_suite(suite)
        except SuiteError:
            self.skipTest("the cc-thingz checkout is not available on this machine")
        rc, out, err = self.run_main("check", "--suite", str(suite), "--tasks",
                                     str(SB.TASKS_ROOT))
        self.assertEqual(rc, 0, out + err)
        self.assertIn("0 failing", out)

    def test_parser_and_console_script(self):
        import tomllib

        from harness2.cli import skillbench_tasks

        p = skillbench_tasks._build_parser()
        ns = p.parse_args(["check", "--suite", "s.toml", "--tasks", "t"])
        self.assertEqual((ns.cmd, ns.suite, ns.tasks), ("check", Path("s.toml"), Path("t")))
        ns = p.parse_args(["new", "--skill", "a", "--id", "b"])
        self.assertEqual((ns.kind, ns.timeout, ns.force), ("skill", T.DEFAULT_TIMEOUT, False))
        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            pyproject["project"]["scripts"]["skillbench-tasks"],
            "harness2.cli.skillbench_tasks:main",
        )


if __name__ == "__main__":
    unittest.main()
