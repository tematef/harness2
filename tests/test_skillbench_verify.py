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

"""skillbench deterministic checks: `verify` / `verify_restore` loading, `run_verify`, `diff_stats`.

Every test builds its own task corpus and fake rollout dirs in a temp dir; no model is called.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness2.benchmarks import skillbench as SB

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_skillbench_tasks import make_task  # noqa: E402

PASSING_TEST = (
    "import unittest\n"
    "from utils import f\n\n"
    "class T(unittest.TestCase):\n"
    "    def test_f(self):\n"
    "        self.assertEqual(f(2), 2)\n"
)
CHEATING_TEST = (
    "import unittest\n\n"
    "class T(unittest.TestCase):\n"
    "    def test_f(self):\n"
    "        pass\n"
)


class _Case(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "tasks"
        self.root.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _task(self, **fields) -> SB.SkillbenchTask:
        d = make_task(self.root, **fields)
        (d / "fixture" / "test_utils.py").write_text(PASSING_TEST, encoding="utf-8")
        return SB.load_task(d)

    def _rollout(self, task: SB.SkillbenchTask, *, utils: str, test: str | None = None) -> Path:
        """A fake valid rollout dir whose work tree has `utils.py` (and maybe an edited test)."""
        run_dir = self.tmp / "run"
        work = run_dir / "work"
        work.mkdir(parents=True)
        (work / "utils.py").write_text(utils, encoding="utf-8")
        (work / "test_utils.py").write_text(test or PASSING_TEST, encoding="utf-8")
        (run_dir / SB.SKILL_TIMING_FILE).write_text(json.dumps({"valid": True}), encoding="utf-8")
        return run_dir


class LoadVerifyTest(_Case):
    def test_task_without_verify_has_empty_fields(self) -> None:
        t = self._task()
        self.assertEqual(t.verify, ())
        self.assertEqual(t.verify_restore, ())

    def test_verify_is_split_like_a_shell_and_restore_files_are_kept(self) -> None:
        t = self._task(verify="python3 -m unittest -q 'test_utils'")
        self.assertEqual(t.verify, ("python3", "-m", "unittest", "-q", "test_utils"))
        d = make_task(self.root, task_id="t2", verify="true", verify_restore=["utils.py"])
        self.assertEqual(SB.load_task(d).verify_restore, ("utils.py",))

    def test_bad_verify_settings_are_rejected(self) -> None:
        cases = {
            "empty": {"verify": "  "},
            "restore-without-verify": {"verify_restore": ["utils.py"]},
            "missing-file": {"verify": "true", "verify_restore": ["nope.py"]},
            "escape": {"verify": "true", "verify_restore": ["../task.toml"]},
            "unbalanced-quote": {"verify": "python3 'oops"},
        }
        for name, fields in cases.items():
            with self.subTest(name):
                d = make_task(self.root, task_id=name, **fields)
                with self.assertRaises(SB.TaskError):
                    SB.load_task(d)


class RunVerifyTest(_Case):
    def _verify_task(self) -> SB.SkillbenchTask:
        return self._task(
            verify=f"{sys.executable} -m unittest -q test_utils",
            verify_restore=["test_utils.py"],
        )

    def test_passing_work_tree_is_cached(self) -> None:
        task = self._verify_task()
        run_dir = self._rollout(task, utils="def f(x):\n    return x\n")
        rec = SB.run_verify(run_dir, task)
        assert rec is not None
        self.assertTrue(rec["passed"])
        self.assertEqual(rec["exit_code"], 0)
        self.assertEqual(rec["restored"], ["test_utils.py"])
        self.assertFalse((run_dir / "verify").exists(), "scratch tree must be removed")
        with mock.patch.object(SB.subprocess, "run") as run:
            again = SB.run_verify(run_dir, task)
        run.assert_not_called()
        self.assertEqual(again, json.loads((run_dir / SB.VERIFY_FILE).read_text()))

    def test_edited_tests_are_restored_so_broken_code_fails(self) -> None:
        task = self._verify_task()
        run_dir = self._rollout(task, utils="def f(x):\n    return -x\n", test=CHEATING_TEST)
        rec = SB.run_verify(run_dir, task)
        assert rec is not None
        self.assertFalse(rec["passed"])
        self.assertNotEqual(rec["exit_code"], 0)
        self.assertIn("FAIL", rec["output_tail"])
        # The agent's own work tree is left untouched.
        self.assertEqual((run_dir / "work" / "test_utils.py").read_text(), CHEATING_TEST)

    def test_no_verify_or_invalid_rollout_returns_none(self) -> None:
        plain = self._task()
        run_dir = self._rollout(plain, utils="def f(x):\n    return x\n")
        self.assertIsNone(SB.run_verify(run_dir, plain))
        task = SB.load_task(
            make_task(self.root, task_id="t2", verify=f"{sys.executable} -c pass")
        )
        (run_dir / SB.SKILL_TIMING_FILE).write_text(json.dumps({"valid": False}))
        self.assertIsNone(SB.run_verify(run_dir, task))
        self.assertFalse((run_dir / SB.VERIFY_FILE).exists())

    def test_timeout_is_a_failed_check(self) -> None:
        task = SB.load_task(
            make_task(
                self.root,
                task_id="slow",
                verify=f"{sys.executable} -c 'import time; time.sleep(5)'",
            )
        )
        run_dir = self._rollout(task, utils="")
        rec = SB.run_verify(run_dir, task, timeout=1)
        assert rec is not None
        self.assertTrue(rec["timed_out"])
        self.assertFalse(rec["passed"])

    def test_score_runs_verify_before_the_judge(self) -> None:
        task = self._verify_task()
        run_dir = self._rollout(task, utils="def f(x):\n    return x\n")
        with mock.patch.object(SB, "_run_judge_once", return_value=None):
            self.assertIsNone(SB.score(run_dir, task))
        self.assertTrue(json.loads((run_dir / SB.VERIFY_FILE).read_text())["passed"])


class DiffStatsTest(_Case):
    def test_counts_files_and_lines_but_not_headers(self) -> None:
        run_dir = self.tmp / "run"
        run_dir.mkdir()
        self.assertIsNone(SB.diff_stats(run_dir))
        (run_dir / "diff.patch").write_text(
            "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n-a\n+b\n+c\n"
            "diff --git a/y.py b/y.py\nnew file mode 100644\n--- /dev/null\n+++ b/y.py\n"
            "@@ -0,0 +1 @@\n+d\n",
            encoding="utf-8",
        )
        self.assertEqual(SB.diff_stats(run_dir), {"files": 2, "added": 3, "removed": 1})


if __name__ == "__main__":
    unittest.main()
