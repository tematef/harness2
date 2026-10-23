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

"""Documentation and `.env.example` verification for skillbench (Task 16)."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from unittest import mock

from harness2 import config

ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_MD = ROOT / "docs" / "BENCHMARKS.md"
SETUP_MD = ROOT / "docs" / "SETUP.md"
README_MD = ROOT / "README.md"
ENV_EXAMPLE = ROOT / ".env.example"

REQUIRED_ENV_KEYS = (
    "HARNESS2_AGY_BIN",
    "SKILLBENCH_PLUGINS_DIR",
    "SKILLBENCH_PLUGINS",
    "SKILLBENCH_AUTH_FILES",
    "SKILLBENCH_EXEC_MODEL",
    "SKILLBENCH_JUDGE_MODEL",
    "SKILLBENCH_EFFORT",
    "SKILLBENCH_JUDGE_REPEAT",
    "SKILLBENCH_PILOT",
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
            # arms & leave-one-out
            "--arm with",
            "--arm without",
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
            # metrics & report
            "skillbench-report",
            "t_skill_s",
            "helps",
            "hurts",
            "inconclusive",
            "insufficient data",
            # statistical power note
            "35",
            "0.2",
            # env vars
            "HARNESS2_AGY_BIN",
            "SKILLBENCH_AUTH_FILES",
            "SKILLBENCH_EXEC_MODEL",
            "SKILLBENCH_JUDGE_MODEL",
            "SKILLBENCH_EFFORT",
            "SKILLBENCH_JUDGE_REPEAT",
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
            "SKILLBENCH_PLUGINS_DIR",
            "SKILLBENCH_AUTH_FILES",
            "SKILLBENCH_EFFORT",
            "SKILLBENCH_EXEC_MODEL",
            "SKILLBENCH_JUDGE_MODEL",
            "harness2-check --bench skillbench",
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


if __name__ == "__main__":
    unittest.main()
