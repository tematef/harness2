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

"""Suite config (skillbench_suite): loading, discovery, env fallback, redaction names,
`skill_asks_user`, `--suite` plumbing and the absence of suite names in prompts.

Every test builds its own fake plugin checkout in a temp dir; nothing under `~` is read.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from harness2 import config
from harness2.benchmarks import skillbench as SB
from harness2.benchmarks import skillbench_suite as SU
from harness2.cli import _common as C
from harness2.cli import skillbench_improve as IMP
from harness2.substrate import agy

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SUITE = ROOT / "harness2" / "benchmarks" / "skillbench" / "suites" / "cc-thingz.toml"

ASKING_MD = (
    "---\nname: ask-it\ndescription: asks\n"
    "allowed-tools: view_file, write_to_file, ask_question\n---\n# Ask\n"
)
SILENT_MD = (
    "---\nname: quiet\ndescription: never asks\nallowed-tools: view_file, run_command\n---\n"
    "# Quiet\nThe body may say ask_question without being an asking skill.\n"
)


def _write(p: Path, body: str = "x\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _plugin(root: Path, name: str, skills: dict[str, str], manifest: bool = True) -> Path:
    d = root / name
    if manifest:
        _write(d / "plugin.json", json.dumps({"name": f"x-{name}"}) + "\n")
    for skill, md in skills.items():
        _write(d / "skills" / skill / "SKILL.md", md)
    return d


class _SuiteCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-suite-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.plugins = self.tmp / "other-suite" / "plugins"
        _plugin(self.plugins, "alpha", {"ask-it": ASKING_MD, "quiet": SILENT_MD})
        _plugin(self.plugins, "beta", {"third": SILENT_MD})
        _plugin(self.plugins, "nomanifest", {"ghost": SILENT_MD}, manifest=False)
        _write(self.plugins / "beta" / "scripts" / "__pycache__" / "x.cpython-313.pyc", "")
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in ("SKILLBENCH_SUITE", "SKILLBENCH_PLUGINS_DIR", "SKILLBENCH_PLUGINS"):
            os.environ.pop(k, None)

    def suite_toml(self, body: str, name: str = "suite.toml") -> Path:
        return _write(self.tmp / name, body)

    def minimal(self, extra: str = "", plugins: str = '["alpha", "beta"]') -> Path:
        return self.suite_toml(
            f'name = "other"\nplugins_dir = "{self.plugins}"\nplugins = {plugins}\n{extra}'
        )


class TestLoadSuite(_SuiteCase):
    def test_explicit_list(self) -> None:
        s = SU.load_suite(self.minimal())
        self.assertEqual(s.name, "other")
        self.assertEqual(s.plugins, ("alpha", "beta"))
        self.assertEqual(s.plugins_dir, self.plugins)
        self.assertIsNone(s.tasks_root)
        self.assertEqual(s.banned_tools, SU.DEFAULT_BANNED_TOOLS)
        self.assertEqual(s.families, {})
        self.assertEqual(s.executors, ())
        self.assertEqual(s.source, "file")
        self.assertEqual(s.skill_names(), {"ask-it", "quiet", "third"})
        self.assertRegex(s.suite_hash, r"^[0-9a-f]{16}$")

    def test_star_discovers_only_plugin_json_dirs(self) -> None:
        s = SU.load_suite(self.minimal(plugins='"*"'))
        self.assertEqual(s.plugins, ("alpha", "beta"))
        self.assertEqual(SU.load_suite(self.suite_toml(
            f'name = "d"\nplugins_dir = "{self.plugins}"\n'
        )).plugins, ("alpha", "beta"))  # plugins omitted = "*"

    def test_star_without_any_plugin_json_is_an_error(self) -> None:
        empty = self.tmp / "empty-plugins"
        (empty / "just-a-dir").mkdir(parents=True)
        p = self.suite_toml(f'name = "e"\nplugins_dir = "{empty}"\nplugins = "*"\n')
        with self.assertRaisesRegex(SU.SuiteError, "no directory with a plugin.json"):
            SU.load_suite(p)

    def test_bad_names_and_missing_things_are_errors(self) -> None:
        cases = {
            "bad suite name": f'name = "bad name"\nplugins_dir = "{self.plugins}"\n',
            "bad plugin name": (
                f'name = "x"\nplugins_dir = "{self.plugins}"\nplugins = ["../alpha"]\n'
            ),
            "missing plugin": (
                f'name = "x"\nplugins_dir = "{self.plugins}"\nplugins = ["alpha", "nope"]\n'
            ),
            "empty plugin list": f'name = "x"\nplugins_dir = "{self.plugins}"\nplugins = []\n',
            "plugins_dir missing": f'name = "x"\nplugins_dir = "{self.tmp / "nowhere"}"\n',
            "no plugins_dir": 'name = "x"\n',
            "unknown key": f'name = "x"\nplugins_dir = "{self.plugins}"\ncolour = "red"\n',
            "unknown family skill": (
                f'name = "x"\nplugins_dir = "{self.plugins}"\n[families]\nfam = ["nope"]\n'
            ),
            "empty family": (
                f'name = "x"\nplugins_dir = "{self.plugins}"\n[families]\nfam = []\n'
            ),
            "executor without judge": (
                f'name = "x"\nplugins_dir = "{self.plugins}"\n[matrix]\n'
                'executors = ["opus-5.5-high"]\n[matrix.judge_for]\n"gemini-" = "fable-5.1-high"\n'
            ),
            "same-prefix judge": (
                f'name = "x"\nplugins_dir = "{self.plugins}"\n[matrix]\n'
                'executors = ["opus-5.5-high"]\n[matrix.judge_for]\n"opus-" = "opus-5.5-low"\n'
            ),
            "bad banned_tools": f'name = "x"\nplugins_dir = "{self.plugins}"\nbanned_tools = "x"\n',
        }
        for label, body in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(SU.SuiteError):
                    SU.load_suite(self.suite_toml(body, name=f"{label.replace(' ', '_')}.toml"))
        with self.assertRaisesRegex(SU.SuiteError, "not found"):
            SU.load_suite(self.tmp / "missing.toml")
        with self.assertRaisesRegex(SU.SuiteError, "not valid TOML"):
            SU.load_suite(self.suite_toml("name = [unclosed\n", name="broken.toml"))
        self.assertTrue(issubclass(SU.SuiteError, config.ConfigError))

    def test_families_matrix_banned_tools_and_relative_paths(self) -> None:
        # plugins_dir relative to the suite file; `~` expanded for tasks_root.
        p = self.suite_toml(
            'name = "other"\nplugins_dir = "other-suite/plugins"\nplugins = ["alpha"]\n'
            'tasks_root = "~/some/corpus"\nbanned_tools = ["Foo", "Bar"]\n'
            "[families]\nasking = [\"ask-it\"]\n"
            '[matrix]\nexecutors = ["opus-5.5-high", "gemini-3.8-flash-high"]\n'
            '[matrix.judge_for]\n"opus-" = "gemini-3.8-flash-high"\n"gemini-" = "fable-5.1-high"\n'
        )
        s = SU.load_suite(p)
        self.assertEqual(s.plugins_dir, self.tmp / "other-suite" / "plugins")
        self.assertEqual(s.tasks_root, Path("~/some/corpus").expanduser())
        self.assertEqual(s.banned_tools, ("Foo", "Bar"))
        self.assertEqual(s.families, {"asking": ("ask-it",)})
        self.assertEqual(s.executors, ("opus-5.5-high", "gemini-3.8-flash-high"))
        self.assertEqual(s.judge_for_executor("opus-5.5-high"), "gemini-3.8-flash-high")
        self.assertEqual(s.judge_for_executor("gemini-3.8-flash-high"), "fable-5.1-high")
        with self.assertRaisesRegex(SU.SuiteError, "no .*prefix matches"):
            s.judge_for_executor("fable-5.1-high")
        self.assertIn("other [alpha]", s.describe())

    def test_suite_hash_tracks_staged_files_only(self) -> None:
        s1 = SU.load_suite(self.minimal())
        h1 = s1.suite_hash
        # bytecode is not staged, so it does not move the hash
        _write(self.plugins / "alpha" / "__pycache__" / "a.pyc", "zz")
        self.assertEqual(SU.load_suite(self.minimal()).suite_hash, h1)
        # a plugin outside the suite does not either
        _write(self.plugins / "nomanifest" / "skills" / "ghost" / "SKILL.md", "changed\n")
        self.assertEqual(SU.load_suite(self.minimal()).suite_hash, h1)
        # a staged file does
        _write(self.plugins / "alpha" / "skills" / "quiet" / "SKILL.md", SILENT_MD + "more\n")
        self.assertNotEqual(SU.load_suite(self.minimal()).suite_hash, h1)
        # and so does the plugin list
        self.assertNotEqual(SU.load_suite(self.minimal(plugins='["alpha"]')).suite_hash, h1)
        self.assertEqual(SU.compute_suite_hash(self.plugins, ["alpha", "beta"]),
                         SU.compute_suite_hash(self.plugins, ["beta", "alpha"]))

    def test_example_suite_file_parses_when_its_checkout_exists(self) -> None:
        self.assertTrue(EXAMPLE_SUITE.is_file())
        import tomllib

        data = tomllib.loads(EXAMPLE_SUITE.read_text(encoding="utf-8"))
        self.assertEqual(data["name"], "cc-thingz")
        self.assertEqual(set(data["matrix"]["judge_for"]), {"opus-", "gemini-", "fable-"})
        if not Path(data["plugins_dir"]).expanduser().is_dir():
            self.skipTest("cc-thingz checkout not present")
        s = SU.load_suite(EXAMPLE_SUITE)
        self.assertEqual(s.name, "cc-thingz")
        self.assertEqual(len(s.plugins), 6)
        for ex in s.executors:
            self.assertNotEqual(s.judge_for_executor(ex).split("-")[0], ex.split("-")[0])


class TestEnvFallbackAndActiveSuite(_SuiteCase):
    def test_env_fallback_discovers_plugins_and_prints_note_once(self) -> None:
        os.environ["SKILLBENCH_PLUGINS_DIR"] = str(self.plugins)
        SU._env_note_shown = False
        err = io.StringIO()
        with redirect_stderr(err):
            s = SU.env_suite()
            SU.env_suite()
        self.assertEqual(s.plugins, ("alpha", "beta"))
        self.assertEqual(s.name, "other-suite")  # parent of plugins/
        self.assertEqual(s.source, "env")
        self.assertIsNone(s.path)
        self.assertEqual(err.getvalue().count("deprecated"), 1)
        self.assertIn("suite.toml", err.getvalue())

    def test_env_fallback_honours_plugins_list_and_validates_it(self) -> None:
        os.environ["SKILLBENCH_PLUGINS_DIR"] = str(self.plugins)
        os.environ["SKILLBENCH_PLUGINS"] = "beta, nomanifest"
        s = SU.env_suite(quiet=True)
        self.assertEqual(s.plugins, ("beta", "nomanifest"))  # an explicit list needs no manifest
        os.environ["SKILLBENCH_PLUGINS"] = "beta,nope"
        with self.assertRaisesRegex(config.ConfigError, "nope"):
            SU.env_suite(quiet=True)
        os.environ["SKILLBENCH_PLUGINS"] = "../x"
        with self.assertRaisesRegex(config.ConfigError, "SKILLBENCH_PLUGINS must be"):
            SU.env_suite(quiet=True)

    def test_env_fallback_errors_name_the_remedy(self) -> None:
        with self.assertRaisesRegex(config.ConfigError, "--suite"):
            SU.env_suite(quiet=True)
        os.environ["SKILLBENCH_PLUGINS_DIR"] = str(self.tmp / "missing")
        with self.assertRaisesRegex(config.ConfigError, "not a directory"):
            SU.env_suite(quiet=True)
        empty = self.tmp / "empty"
        empty.mkdir()
        os.environ["SKILLBENCH_PLUGINS_DIR"] = str(empty)
        with self.assertRaisesRegex(config.ConfigError, "no directory with a plugin.json"):
            SU.env_suite(quiet=True)

    def test_active_suite_precedence(self) -> None:
        file_a = self.minimal(plugins='["alpha"]')
        file_b = self.suite_toml(
            f'name = "b"\nplugins_dir = "{self.plugins}"\nplugins = ["beta"]\n', name="b.toml"
        )
        os.environ["SKILLBENCH_PLUGINS_DIR"] = str(self.plugins)
        self.assertEqual(SU.active_suite(quiet=True).source, "env")
        os.environ["SKILLBENCH_SUITE"] = str(file_b)
        self.assertEqual(SU.active_suite().name, "b")
        self.assertEqual(SU.active_suite(file_a).name, "other")  # explicit path wins

    def test_agy_helpers_read_the_active_suite(self) -> None:
        os.environ["SKILLBENCH_SUITE"] = str(self.minimal(plugins='["alpha"]'))
        self.assertEqual(agy.plugins_dir(), self.plugins)
        self.assertEqual(agy.plugin_allowlist(), ("alpha",))
        self.assertEqual(agy.checked_plugins(), (self.plugins, ("alpha",)))
        self.assertEqual(
            agy.target_skill_file("quiet"), self.plugins / "alpha" / "skills" / "quiet" / "SKILL.md"
        )
        with self.assertRaisesRegex(ValueError, "third"):
            agy.target_skill_file("third")  # beta is not in this suite
        self.assertFalse(hasattr(agy, "DEFAULT_PLUGINS"))
        staged = self.tmp / "staged"
        omitted = agy.stage_plugins(staged, arm="without", target_skill="quiet")
        self.assertEqual(omitted, ["alpha/skills/quiet"])
        self.assertEqual(sorted(p.name for p in staged.iterdir()), ["alpha"])
        self.assertTrue((staged / "alpha" / "skills" / "ask-it" / "SKILL.md").is_file())


class TestRedactionAndAskDetection(_SuiteCase):
    def test_redaction_names_from_staged_plugins_and_suite(self) -> None:
        suite = SU.load_suite(self.minimal(plugins='["alpha"]'))
        staged = self.tmp / "staged"
        agy.stage_plugins(staged, arm="without", target_skill="quiet", suite=suite)
        self.assertEqual(SU.redaction_names(staged), {"ask-it"})
        self.assertEqual(
            SU.redaction_names(staged, suite), {"ask-it", "quiet", "other"}
        )  # the omitted skill and the suite name are still redacted
        self.assertEqual(SU.redaction_names(None, suite), {"ask-it", "quiet", "other"})
        self.assertEqual(SU.redaction_names(None, None), frozenset())
        self.assertEqual(SU.redaction_names(self.tmp / "nowhere"), frozenset())
        self.assertFalse(hasattr(SB, "KNOWN_SKILLS"))
        self.assertFalse(hasattr(SB, "_discovered_skills"))

    def test_scrub_deliverable_uses_the_given_names_only(self) -> None:
        text = "Used the quiet skill, read skills/ask-it/SKILL.md and other-suite notes; third."
        out = SB.scrub_deliverable(text, {"quiet", "ask-it", "other-suite"})
        for leaked in ("quiet", "ask-it", "other-suite", "SKILL.md"):
            self.assertNotIn(leaked, out)
        self.assertIn("third", out)  # not in the names: untouched

    def test_score_redacts_staged_and_suite_skill_names(self) -> None:
        suite = SU.load_suite(self.minimal(plugins='["alpha"]'))
        run_dir = self.tmp / "run"
        agy.stage_plugins(run_dir / "home" / agy.PLUGINS_REL, arm="with", target_skill="quiet",
                          suite=suite)
        self.assertEqual(
            SB._redaction_names_for(run_dir, suite), {"ask-it", "quiet", "other"}
        )
        # no suite configured at all: only what was staged into the rollout
        self.assertEqual(SB._redaction_names_for(run_dir, None), {"ask-it", "quiet"})
        self.assertEqual(SB._redaction_names_for(self.tmp / "no-run", None), frozenset())

    def test_skill_asks_user_reads_frontmatter_only(self) -> None:
        self.assertTrue(SU.skill_asks_user(ASKING_MD))
        self.assertFalse(SU.skill_asks_user(SILENT_MD))
        self.assertFalse(SU.skill_asks_user("# no frontmatter\nask_question everywhere\n"))
        self.assertFalse(SU.skill_asks_user(""))
        self.assertTrue(SU.skill_asks_user("---\nallowed_tools: [ask_question]\n---\n"))
        self.assertTrue(SU.skill_asks_user("---\nAllowed-Tools: \"ask_question\", view_file\n---\n"))
        self.assertFalse(SU.skill_asks_user("---\nallowed-tools: ask_questions\n---\n"))
        self.assertFalse(hasattr(SB, "ASK_QUESTION_SKILLS"))
        self.assertEqual(SU.skill_frontmatter(ASKING_MD).splitlines()[0], "name: ask-it")

    def test_task_asks_user_override(self) -> None:
        base = dict(
            task_id="s/t", target_skill="s", prompt="p", timeout=1, rubric=(),
        )
        self.assertTrue(SB.task_asks_user(SB.SkillbenchTask(**base, asks_user=True), SILENT_MD))
        self.assertFalse(SB.task_asks_user(SB.SkillbenchTask(**base, asks_user=False), ASKING_MD))
        self.assertTrue(SB.task_asks_user(SB.SkillbenchTask(**base), ASKING_MD))
        self.assertFalse(SB.task_asks_user(SB.SkillbenchTask(**base)))


class TestSuitePlumbingAndPrompts(_SuiteCase):
    def test_apply_suite_exports_env_and_add_suite_arg(self) -> None:
        import argparse

        self.assertIsNone(C.apply_suite(None))
        self.assertNotIn("SKILLBENCH_SUITE", os.environ)
        p = argparse.ArgumentParser()
        C.add_suite_arg(p)
        args = p.parse_args(["--suite", "~/x/suite.toml"])
        self.assertEqual(C.apply_suite(args.suite), Path("~/x/suite.toml").expanduser())
        self.assertEqual(os.environ["SKILLBENCH_SUITE"], str(Path("~/x/suite.toml").expanduser()))

    def test_cell_args_carry_suite_and_non_skillbench_benches_reject_it(self) -> None:
        import argparse

        p = argparse.ArgumentParser()
        C.add_cell_args(p)
        args = p.parse_args(["--bench", "skillbench", "--arm", "with", "--suite", "s.toml"])
        self.assertEqual(args.suite, Path("s.toml"))
        with self.assertRaisesRegex(ValueError, "--suite applies to --bench skillbench only"):
            C.make_adapter("jb", substrate="oc", suite=Path("s.toml"))

    def test_make_adapter_uses_the_suite_flag(self) -> None:
        suite_file = self.minimal(plugins='["alpha"]')
        tasks_root = self.tmp / "tasks"
        d = tasks_root / "quiet" / "t1"
        _write(d / "task.toml", 'target_skill = "quiet"\nprompt = "Tidy utils.py."\ntimeout = 60\n')
        _write(
            d / "rubric.json",
            json.dumps([{"id": f"c{i}", "criterion": f"Outcome {i}."} for i in range(4)]),
        )
        (d / "fixture").mkdir()
        SB.freeze(tasks_root)
        with mock.patch.object(SB, "TASKS_ROOT", tasks_root):
            adapter = C.make_adapter(
                "skillbench", arm="with", suite=suite_file
            )
        self.assertEqual(adapter.suite.name, "other")
        self.assertEqual(adapter.suite.plugins, ("alpha",))
        self.assertEqual(os.environ["SKILLBENCH_SUITE"], str(suite_file))

    def test_every_skillbench_cli_parser_accepts_suite(self) -> None:
        from harness2.cli import run_sweep, skillbench_report

        for build in (run_sweep._build_parser, skillbench_report._build_parser):
            with self.subTest(parser=build.__module__):
                self.assertIn("--suite", build().format_help())
        self.assertIn("--suite", self._improve_help())
        # harness2-judge / harness2-check / harness2-report build their parser inline from
        # the shared cell args, covered by test_cell_args_carry_suite_... above.

    def _improve_help(self) -> str:
        import argparse

        captured: dict[str, argparse.ArgumentParser] = {}
        real_parse = argparse.ArgumentParser.parse_args

        def grab(self_, *a, **kw):
            captured["p"] = self_
            raise SystemExit(0)

        with mock.patch.object(argparse.ArgumentParser, "parse_args", grab):
            with self.assertRaises(SystemExit):
                IMP.main([])
        with mock.patch.object(argparse.ArgumentParser, "parse_args", real_parse):
            return captured["p"].format_help()

    def test_run_json_records_suite_and_hash(self) -> None:
        from harness2.cli.run_sweep import write_run_json

        suite = SU.load_suite(self.minimal())

        class _Spec:
            name = "skillbench"

        class _Adapter:
            spec = _Spec()
            arm = "with"
            manifest_hash = "m" * 64

        _Adapter.suite = suite
        os.environ["SKILLBENCH_EXEC_MODEL"] = "opus-5.5-high"
        os.environ["SKILLBENCH_JUDGE_MODEL"] = "fable-5.1-high"
        self.addCleanup(os.environ.pop, "SKILLBENCH_EXEC_MODEL", None)
        self.addCleanup(os.environ.pop, "SKILLBENCH_JUDGE_MODEL", None)
        ns = self.tmp / "ns"
        path = write_run_json(_Adapter(), ns)
        rec = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(path.name, "run.json")
        self.assertEqual(rec["suite"], "other")
        self.assertEqual(rec["suite_hash"], suite.suite_hash)
        self.assertEqual(rec["suite_path"], str(suite.path))
        self.assertEqual(rec["arm"], "with")
        self.assertEqual(rec["executor"], "opus-5.5-high")
        self.assertEqual(rec["judge"], "fable-5.1-high")
        self.assertEqual(rec["manifest_hash"], "m" * 64)

    def test_prompts_and_help_carry_no_suite_name(self) -> None:
        prompt = IMP.build_improver_prompt(skill="quiet", evidence_packet="E")
        self.assertNotIn("cc-thingz", prompt)
        self.assertIn("an AGY plugin suite", prompt)
        for tool in SU.DEFAULT_BANNED_TOOLS:
            self.assertIn(f"`{tool}`", prompt)
        named = IMP.build_improver_prompt(
            skill="quiet", evidence_packet="E", suite_name="other", banned_tools=("Foo",)
        )
        self.assertIn("the `other` plugin suite", named)
        self.assertIn("`Foo`", named)
        self.assertNotIn("AskUserQuestion", named)
        bp = SB.SKILLBENCH_PROMPTS
        for text in (bp.work, bp.dom_levers, bp.inv_levers, bp.yardstick, bp.selection_checks):
            for word in ("cc-thingz", "root-cause", "make-plan", "brainstorm"):
                self.assertNotIn(word, text)
        self.assertNotIn("cc-thingz", self._improve_help())
        self.assertEqual(IMP.BANNED_CLAUDE_TOOLS, SU.DEFAULT_BANNED_TOOLS)

    def test_validate_candidate_skill_uses_suite_banned_tools(self) -> None:
        original = "---\nname: quiet\ndescription: d\n---\nBody\n"
        cand = "---\nname: quiet\ndescription: d\n---\nUse Foo here, never AskUserQuestion.\n"
        default_errs = IMP.validate_candidate_skill(cand, target_skill="quiet", original_md=original)
        self.assertTrue(any("AskUserQuestion" in e for e in default_errs))
        self.assertFalse(any("'Foo'" in e for e in default_errs))
        suite_errs = IMP.validate_candidate_skill(
            cand, target_skill="quiet", original_md=original, banned_tools=("Foo",)
        )
        self.assertTrue(any("'Foo'" in e for e in suite_errs))
        self.assertFalse(any("AskUserQuestion" in e for e in suite_errs))

    def test_no_logic_hard_codes_the_example_suite(self) -> None:
        # The example suite file and the bundled corpus are data; the code may mention the
        # suite only in docstrings (Task 14 widens this to a grep over harness2/).
        import ast

        for mod in (SU, agy, SB, IMP, C):
            tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
            docstrings = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    body = getattr(node, "body", [])
                    if body and isinstance(body[0], ast.Expr):
                        docstrings.add(id(body[0].value))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                ):
                    with self.subTest(module=mod.__name__, line=node.lineno):
                        self.assertNotIn("cc-thingz", node.value)


if __name__ == "__main__":
    unittest.main()
