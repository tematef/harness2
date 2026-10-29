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

"""Documentation and `.env.example` verification for skillbench (Task 16 & v2 Task 15)."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import unittest
from pathlib import Path
from unittest import mock

from harness2 import config
from harness2.cli import (
    skillbench_cycle,
    skillbench_improve,
    skillbench_matrix,
    skillbench_report,
    skillbench_tasks,
)

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_MD = ROOT / "docs" / "BENCHMARKS.md"
SETUP_MD = ROOT / "docs" / "SETUP.md"
README_MD = ROOT / "README.md"
AGENTS_MD = ROOT / ".agents" / "AGENTS.md"
ENV_EXAMPLE = ROOT / ".env.example"
JETSKI_CLI = "/google/bin/releases/jetski-devs/tools/cli"

REQUIRED_ENV_KEYS = (
    "HARNESS2_AGY_BIN",
    "SKILLBENCH_SUITE",
    "SKILLBENCH_TASKS_ROOT",
    "SKILLBENCH_PLUGINS_DIR",
    "SKILLBENCH_PLUGINS",
    "SKILLBENCH_AUTH_FILES",
    "SKILLBENCH_EXEC_MODEL",
    "SKILLBENCH_JUDGE_MODEL",
    "SKILLBENCH_EFFORT",
    "SKILLBENCH_JUDGE_REPEAT",
    "SKILLBENCH_PILOT",
)

NEW_ENV_KEYS = (
    "SKILLBENCH_SUITE",
    "SKILLBENCH_TASKS_ROOT",
)

NEW_FLAGS = (
    "--suite",
    "--suites",
    "--results-root",
    "--hardness-only",
    "--summary-only",
)

SKILLBENCH_CLIS = (
    "skillbench-tasks",
    "skillbench-matrix",
    "skillbench-report",
    "skillbench-improve",
    "skillbench-cycle",
)


class TestSkillbenchDocumentation(unittest.TestCase):
    def test_benchmarks_doc_covers_skillbench(self) -> None:
        text = BENCHMARKS_MD.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^##\s+skillbench\b")
        for term in (
            # task format & freeze
            "task.toml",
            "rubric.json",
            "fixture/",
            "manifest.json",
            "user_answers",
            "forbidden_terms",
            # suite.toml reference & schema v2
            "suite.toml",
            "plugins_dir",
            "[families]",
            "[matrix.judge_for]",
            "schema_version: 2",
            "summary.md",
            "history.jsonl",
            # arms & leave-one-out & suite arms
            "--arm with",
            "--arm without",
            "baseline",
            "suite-<name>",
            "leave-one-out",
            # allowlist
            "SKILLBENCH_PLUGINS_DIR",
            "SKILLBENCH_PLUGINS",
            "brainstorm,planning,review,skill-eval,thinking-tools,workflow",
            # shims
            "xclip",
            "xsel",
            "wl-copy",
            "pbcopy",
            "revdiff",
            "side_effects/",
            # metrics & report & recommendation labels
            "skillbench-report",
            "t_skill_s",
            "helps",
            "hurts",
            "inconclusive",
            "insufficient data",
            "needs harder tasks",
            "consider removing",
            "fix",
            "keep",
            # statistical power note
            "35",
            "0.2",
            # env vars
            "HARNESS2_AGY_BIN",
            "SKILLBENCH_SUITE",
            "SKILLBENCH_TASKS_ROOT",
            "SKILLBENCH_AUTH_FILES",
            "SKILLBENCH_EXEC_MODEL",
            "SKILLBENCH_JUDGE_MODEL",
            "SKILLBENCH_EFFORT",
            "SKILLBENCH_JUDGE_REPEAT",
            # new flags & CLIs
            *NEW_FLAGS,
            *SKILLBENCH_CLIS,
            # cost gates / pilot / spike
            "agy_spike.sh",
            "run_skillbench.sh",
            "PILOT=1",
        ):
            with self.subTest(term=term):
                self.assertIn(term, text)

    def test_setup_doc_covers_agy_substrate(self) -> None:
        text = SETUP_MD.read_text(encoding="utf-8")
        for term in (
            "--substrate agy",
            "HARNESS2_AGY_BIN",
            "SKILLBENCH_SUITE",
            "SKILLBENCH_TASKS_ROOT",
            "SKILLBENCH_PLUGINS_DIR",
            "SKILLBENCH_AUTH_FILES",
            "SKILLBENCH_EFFORT",
            "SKILLBENCH_EXEC_MODEL",
            "SKILLBENCH_JUDGE_MODEL",
            "harness2-check --bench skillbench",
            "--suite",
            "suite.toml",
            *SKILLBENCH_CLIS,
        ):
            with self.subTest(term=term):
                self.assertIn(term, text)

    def test_readme_benchmark_table_lists_skillbench(self) -> None:
        text = README_MD.read_text(encoding="utf-8")
        table_match = re.search(
            r"(?ms)^\| Benchmark \| Tasks \| Install \| Guide \|\n(?:\|[^\n]*\n)+",
            text,
        )
        self.assertIsNotNone(table_match, "Benchmark table not found in README.md")
        table = table_match.group(0)
        self.assertIn("skillbench", table)
        self.assertIn("docs/BENCHMARKS.md#skillbench", table)

    def test_readme_covers_generic_skillbench_v2_workflows_and_clis(self) -> None:
        text = README_MD.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^###\s+Benchmark any skill set\b")
        self.assertRegex(text, r"(?m)^###\s+Improve a weak skill\b")
        for term in (
            "suite.toml",
            "skillbench-tasks new",
            "skillbench-tasks check",
            "--arms baseline",
            "--hardness-only",
            "--results-root",
            "history.jsonl",
            "summary.md",
            # recommendation rule in order
            "needs harder tasks",
            "consider removing",
            "fix",
            "keep",
            # improver v2 flags
            "--models",
            "--rounds",
            "--max-growth",
            "--apply",
            # all 5 skillbench CLIs
            *SKILLBENCH_CLIS,
        ):
            with self.subTest(term=term):
                self.assertIn(term, text)

    def test_agents_md_covers_v2_rules(self) -> None:
        text = AGENTS_MD.read_text(encoding="utf-8")
        for term in (
            "No paid runs without explicit user approval.",
            "suite.toml",
            "no suite hard-coding",
            "schema_version: 2",
            "Staging only reads the checkout; only",
            "`--apply` writes, on a branch, never on a dirty tree",
            *SKILLBENCH_CLIS,
        ):
            with self.subTest(term=term):
                self.assertIn(term, text)
        self.assertNotIn("touch the `cc-thingz` checkout", text)

    def test_env_example_contains_agy_and_skillbench_keys_and_loads_cleanly(self) -> None:
        self.assertTrue(ENV_EXAMPLE.is_file(), f"missing {ENV_EXAMPLE}")
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        for key in REQUIRED_ENV_KEYS:
            with self.subTest(key=key):
                self.assertRegex(
                    text,
                    rf"(?m)^#?\s*{re.escape(key)}=",
                    f"{key} missing from .env.example",
                )
        with mock.patch.dict(os.environ, {}, clear=True):
            config.load_dotenv(ENV_EXAMPLE)

    def test_env_example_matches_jet_defaults(self) -> None:
        text = ENV_EXAMPLE.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^HARNESS2_AGY_BIN=agy$")
        self.assertRegex(text, rf"(?m)^#.*{re.escape(JETSKI_CLI)}")
        self.assertRegex(text, r"(?m)^SKILLBENCH_EXEC_MODEL=opus-5\.5-high$")
        self.assertRegex(text, r"(?m)^SKILLBENCH_JUDGE_MODEL=fable-5\.1-high$")
        self.assertRegex(text, r"(?m)^SKILLBENCH_EFFORT=$")
        self.assertRegex(text, r"(?m)^SKILLBENCH_AUTH_FILES=$")

    def test_docs_mark_auth_and_effort_optional_with_jet_defaults(self) -> None:
        for doc in (SETUP_MD, BENCHMARKS_MD):
            text = doc.read_text(encoding="utf-8")
            with self.subTest(doc=doc.name):
                for key in ("SKILLBENCH_AUTH_FILES", "SKILLBENCH_EFFORT"):
                    row = next(ln for ln in text.splitlines() if ln.startswith(f"| `{key}`"))
                    self.assertNotIn("**required**", row)
                    self.assertIn("optional", row)
                self.assertIn("opus-5.5-high", text)
                self.assertIn("fable-5.1-high", text)
                self.assertIn(JETSKI_CLI, text)

    def test_all_cli_flags_documented_and_code_block_invocations_parse_cleanly(self) -> None:
        def _flags_of(parser: argparse.ArgumentParser) -> set[str]:
            out: set[str] = set()
            for action in parser._actions:
                for opt in action.option_strings:
                    if opt.startswith("--") and opt != "--help":
                        out.add(opt)
                if isinstance(action, argparse._SubParsersAction):
                    for sp in action.choices.values():
                        out |= _flags_of(sp)
            return out

        cli_parsers = {
            "skillbench-tasks": skillbench_tasks._build_parser(),
            "skillbench-matrix": skillbench_matrix._build_parser(),
            "skillbench-report": skillbench_report._build_parser(),
            "skillbench-improve": skillbench_improve._build_parser(),
            "skillbench-cycle": skillbench_cycle._build_parser(),
        }
        history_parser = skillbench_report._build_history_parser()

        all_flags: set[str] = set()
        for p in cli_parsers.values():
            all_flags |= _flags_of(p)
        all_flags |= _flags_of(history_parser)

        bench_text = BENCHMARKS_MD.read_text(encoding="utf-8")
        readme_text = README_MD.read_text(encoding="utf-8")
        for flag in sorted(all_flags):
            with self.subTest(flag=flag):
                self.assertIn(flag, bench_text, f"{flag} missing from docs/BENCHMARKS.md")
                self.assertIn(flag, readme_text, f"{flag} missing from README.md")

        # Parse every skillbench-* invocation inside fenced bash code blocks in docs
        code_block_re = re.compile(r"```bash\n(.*?)```", re.DOTALL)
        for doc in (BENCHMARKS_MD, SETUP_MD, README_MD):
            doc_text = doc.read_text(encoding="utf-8")
            for block_match in code_block_re.finditer(doc_text):
                block = re.sub(r"\\\n\s*", " ", block_match.group(1))
                for raw_line in block.splitlines():
                    line = raw_line.split("#", 1)[0].strip()
                    if not line:
                        continue
                    tokens = shlex.split(line)
                    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
                        tokens.pop(0)
                    if not tokens or tokens[0] not in cli_parsers:
                        continue
                    prog = tokens[0]
                    argv = tokens[1:]
                    with self.subTest(doc=doc.name, cmd=line):
                        if prog == "skillbench-report" and argv and argv[0] == "history":
                            ns = history_parser.parse_args(argv[1:])
                        else:
                            ns = cli_parsers[prog].parse_args(argv)
                        self.assertIsNotNone(ns)


if __name__ == "__main__":
    unittest.main()
