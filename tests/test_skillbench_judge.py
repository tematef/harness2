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

"""skillbench judge (fable via agy): deliverable scrubbing, schema validation, and cache.

Every test runs against `tests/fixtures/agy_stub.py` and temp directories (`HOME=<tmp>`); the
user's real `$HOME` and `~/.gemini` are never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from harness2.benchmarks import skillbench as SB
from harness2.config import ConfigError
from tests.test_skillbench_tasks import make_task

STUB = Path(__file__).resolve().parent / "fixtures" / "agy_stub.py"
AUTH = (".gemini/oauth_creds.json", ".gemini/google_accounts.json")


def _write(p: Path, body: str = "x\n") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return p


def _tree_hash(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _scenario(path: Path, *events: dict) -> Path:
    return _write(path, "".join(json.dumps(e) + "\n" for e in events))


def _judge_result(criteria: list[dict], *, model: str | None = None, tokens: int = 1200) -> dict:
    res: dict = {"structured_output": {"criteria": criteria}, "usage": {"total_tokens": tokens}}
    if model is not None:
        res["model"] = model
    return {"event": "result", "result": res}


class _JudgeCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="skillbench-judge-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

        self.home = self.tmp / "realhome"
        for rel in AUTH:
            _write(self.home / rel, f'{{"token": "{rel}"}}\n')
        _write(self.home / ".gemini" / "GEMINI.md", "real user rules\n")
        _write(
            self.home / ".gemini" / "config" / "plugins" / "planning" / "skills" / "make-plan"
            / "SKILL.md",
            "real skill\n",
        )

        self.plugins = self.tmp / "cc-thingz" / "plugins"
        _write(self.plugins / "planning" / "skills" / "make-plan" / "SKILL.md", "make-plan\n")
        _write(
            self.plugins / "review" / "skills" / "code-review-refactor" / "SKILL.md", "refactor\n"
        )

        self.tasks_root = self.tmp / "tasks"
        self.task_dir = make_task(
            self.tasks_root,
            "make-plan",
            "t1",
            prompt="Write an implementation plan for caching fetch() in utils.py.",
            user_answers=["In-memory LRU cache", "Keep TTL simple"],
        )
        self.task = SB.load_task(self.task_dir)

        self.stub_log = self.tmp / "stub.log"
        self.default_scenario = _scenario(
            self.tmp / "judge_ok.jsonl",
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "agent_response",
                    "state": "DONE",
                    "text_delta": "graded",
                },
            },
            _judge_result(
                [
                    {"id": "c0", "pass": True, "reason": "ok 0"},
                    {"id": "c1", "pass": True, "reason": "ok 1"},
                    {"id": "c2", "pass": False, "reason": "missing 2"},
                    {"id": "c3", "pass": True, "reason": "ok 3"},
                ]
            ),
        )

        env = {
            "HOME": str(self.home),
            "HARNESS2_AGY_BIN": str(STUB),
            "SKILLBENCH_PLUGINS_DIR": str(self.plugins),
            "SKILLBENCH_AUTH_FILES": ",".join(AUTH),
            "SKILLBENCH_JUDGE_MODEL": "fable",
            "SKILLBENCH_EFFORT": "high",
            "AGY_STUB_LOG": str(self.stub_log),
            "AGY_STUB_SCENARIO": str(self.default_scenario),
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in (
            "SKILLBENCH_JUDGE_REPEAT",
            "AGY_STUB_SLEEP",
            "AGY_STUB_HANG",
            "AGY_STUB_HANG_S",
            "AGY_STUB_EXIT_CODE",
        ):
            os.environ.pop(k, None)

        self.real_home_before = _tree_hash(self.home)

    def tearDown(self) -> None:
        self.assertEqual(_tree_hash(self.home), self.real_home_before, "fake real HOME modified")

    def make_run_dir(
        self,
        name: str = "run0",
        *,
        valid: bool = True,
        invalid_reason: str | None = None,
        final_md: str = "Wrote plan to docs/plans/20261023-cache.md.\n",
        diff_patch: str = "--- a/utils.py\n+++ b/utils.py\n@@ -1 +1,2 @@\n+# cached\n",
        new_files: dict[str, str] | None = None,
    ) -> Path:
        rd = self.tmp / "runs" / name
        rd.mkdir(parents=True, exist_ok=True)
        _write(rd / "final.md", final_md)
        _write(rd / "diff.patch", diff_patch)
        nf = rd / "new_files"
        nf.mkdir(exist_ok=True)
        for rel, content in (
            {"docs/plans/20261023-cache.md": "# Cache plan\n### Task 1: Add LRU\n"}
            if new_files is None
            else new_files
        ).items():
            _write(nf / rel, content)
        # Put rollout home plugins and trajectory in run_dir to verify the judge ignores them.
        _write(
            rd / "home" / ".gemini" / "config" / "plugins" / "planning" / "skills" / "make-plan"
            / "SKILL.md",
            "staged skill\n",
        )
        _write(
            rd / "events.jsonl",
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "tool",
                        "state": "ACTIVE",
                        "tool_name": "view_file",
                        "tool_info": {
                            "name": "view_file",
                            "parameters": {
                                "AbsolutePath": "/secret/trajectory/only/marker.py",
                            },
                        },
                    },
                    "t": 0.5,
                }
            )
            + "\n",
        )
        _write(
            rd / "skill_timing.json",
            json.dumps(
                {
                    "invoked": True,
                    "t_skill_s": 0.5,
                    "step_idx": 0,
                    "t_total_s": 2.0,
                    "requested_model": "opus",
                    "actual_model": None,
                    "valid": valid,
                    "invalid_reason": invalid_reason,
                    "exit_code": 0 if valid else 1,
                    "total_tokens": 500,
                }
            )
            + "\n",
        )
        return rd

    def stub_calls(self) -> list[dict]:
        if not self.stub_log.is_file():
            return []
        return [json.loads(x) for x in self.stub_log.read_text(encoding="utf-8").splitlines()]


class TestJudgeIsolationAndPrompt(_JudgeCase):
    def test_judge_runs_in_empty_home_with_model_and_json_schema(self) -> None:
        rd = self.make_run_dir()
        got = SB.score(rd, self.task)
        self.assertEqual(got, {"n_passed": 3, "n_criteria": 4})

        (call,) = self.stub_calls()
        flags = call["flags"]
        self.assertEqual(flags["model"], "fable")
        self.assertEqual(flags["output_format"], "stream-json")
        self.assertIs(flags["dangerously_skip_permissions"], True)
        self.assertIs(flags["disable_slash_commands"], True)
        self.assertIn("json_schema", flags)
        schema = json.loads(flags["json_schema"])
        self.assertEqual(schema["required"], ["criteria"])
        self.assertEqual(
             sorted(schema["properties"]["criteria"]["items"]["required"]),
            ["id", "pass", "reason"],
        )

        judge_home = Path(call["home"])
        self.assertNotEqual(judge_home.resolve(), self.home.resolve())
        self.assertNotEqual(judge_home.resolve(), (rd / "home").resolve())
        self.assertFalse((judge_home / ".gemini" / "config" / "plugins").exists())
        self.assertEqual(list(judge_home.rglob("SKILL.md")), [])
        self.assertEqual(list(judge_home.iterdir()), [])

        judge_cwd = Path(call["cwd"])
        self.assertNotEqual(judge_cwd.resolve(), (rd / "work").resolve())
        self.assertEqual(list(judge_cwd.iterdir()), [])

    def test_judge_model_follows_env(self) -> None:
        os.environ["SKILLBENCH_JUDGE_MODEL"] = "custom-judge"
        rd = self.make_run_dir()
        self.assertEqual(SB.score(rd, self.task), {"n_passed": 3, "n_criteria": 4})
        (call,) = self.stub_calls()
        self.assertEqual(call["flags"]["model"], "custom-judge")

    def test_prompt_contains_rubric_and_scrubbed_deliverables_without_leaks(self) -> None:
        rd = self.make_run_dir(
            name="with_arm_run",
            final_md=(
                "Read /tmp/run/home/.gemini/config/plugins/planning/skills/make-plan/SKILL.md "
                "and consulted brainstorm, root-cause-investigator, and code-review-refactor "
                "via /planning:make-plan and the make-plan skill. "
                "Final summary: added LRU cache around fetch()."
            ),
            diff_patch=(
                "--- a/utils.py\n+++ b/utils.py\n@@ -1 +1,3 @@\n"
                "+# see plugins/planning/skills/make-plan/SKILL.md and skills/backlog/SKILL.md\n"
                "+def cached_fetch(k):\n+    return fetch(k)\n"
            ),
            new_files={
                "docs/plans/20261023-cache.md": (
                    "# Cache plan\n"
                    "Created per SKILL.md from make-plan and dialectic.\n"
                    "### Task 1: Add cache\n"
                )
            },
        )
        self.assertEqual(SB.score(rd, self.task), {"n_passed": 3, "n_criteria": 4})
        (call,) = self.stub_calls()
        prompt = call["flags"]["prompt"]

        # Rubric criteria are present.
        for crit in self.task.rubric:
            self.assertIn(crit.id, prompt)
            self.assertIn(crit.criterion, prompt)

        # Deliverable content from final.md, diff.patch, and new_files is present.
        self.assertIn("Final summary: added LRU cache around fetch().", prompt)
        self.assertIn("def cached_fetch(k):", prompt)
        self.assertIn("docs/plans/20261023-cache.md", prompt)
        self.assertIn("### Task 1: Add cache", prompt)
        self.assertIn("[redacted]", prompt)

        # No trajectory, arm label, skill names, or SKILL.md / plugin paths.
        self.assertNotIn("/secret/trajectory/only/marker.py", prompt)
        self.assertNotIn("view_file", prompt)
        self.assertNotIn("[ACTION]", prompt)
        self.assertNotIn("with_arm_run", prompt)
        self.assertNotRegex(prompt.lower(), r"\b(with|without)\b")
        self.assertNotIn("SKILL.md", prompt)
        self.assertNotIn("skill.md", prompt.lower())
        self.assertNotIn(".gemini/config/plugins", prompt)
        for skill_name in (
            "make-plan",
            "brainstorm",
            "root-cause-investigator",
            "code-review-refactor",
            "backlog",
            "dialectic",
        ):
            self.assertNotIn(skill_name, prompt.lower(), skill_name)


class TestJudgeValidationAndFailures(_JudgeCase):
    def test_valid_bundled_json_schema_fixture(self) -> None:
        os.environ["AGY_STUB_SCENARIO"] = "json_schema"
        two_crit_task = replace(
            self.task,
            rubric=(
                SB.Criterion("c1", "Plan file exists."),
                SB.Criterion("c2", "Tests are listed."),
            ),
        )
        rd = self.make_run_dir()
        self.assertEqual(SB.score(rd, two_crit_task), {"n_passed": 1, "n_criteria": 2})

    def test_unknown_or_missing_or_duplicate_criterion_ids_return_none(self) -> None:
        bad_cases = {
            "unknown_id": [
                {"id": "c0", "pass": True, "reason": "r"},
                {"id": "c1", "pass": True, "reason": "r"},
                {"id": "c2", "pass": True, "reason": "r"},
                {"id": "c99", "pass": True, "reason": "r"},
            ],
            "missing_id": [
                {"id": "c0", "pass": True, "reason": "r"},
                {"id": "c1", "pass": True, "reason": "r"},
                {"id": "c2", "pass": True, "reason": "r"},
            ],
            "duplicate_id": [
                {"id": "c0", "pass": True, "reason": "r"},
                {"id": "c0", "pass": False, "reason": "r"},
                {"id": "c1", "pass": True, "reason": "r"},
                {"id": "c2", "pass": True, "reason": "r"},
            ],
            "extra_id": [
                {"id": "c0", "pass": True, "reason": "r"},
                {"id": "c1", "pass": True, "reason": "r"},
                {"id": "c2", "pass": True, "reason": "r"},
                {"id": "c3", "pass": True, "reason": "r"},
                {"id": "c4", "pass": True, "reason": "r"},
            ],
            "non_bool_pass": [
                {"id": "c0", "pass": "yes", "reason": "r"},
                {"id": "c1", "pass": True, "reason": "r"},
                {"id": "c2", "pass": True, "reason": "r"},
                {"id": "c3", "pass": True, "reason": "r"},
            ],
            "missing_reason": [
                {"id": "c0", "pass": True},
                {"id": "c1", "pass": True, "reason": "r"},
                {"id": "c2", "pass": True, "reason": "r"},
                {"id": "c3", "pass": True, "reason": "r"},
            ],
        }
        for label, criteria in bad_cases.items():
            with self.subTest(label=label):
                scen = _scenario(self.tmp / f"{label}.jsonl", _judge_result(criteria))
                os.environ["AGY_STUB_SCENARIO"] = str(scen)
                rd = self.make_run_dir(name=f"run_{label}")
                self.assertIsNone(SB.score(rd, self.task))
                self.assertFalse((rd / "judgments.json").exists())

    def test_malformed_json_or_missing_structured_output_returns_none(self) -> None:
        bad_streams = {
            "plain_answer": "plain_answer",
            "bad_json_str": str(
                _scenario(
                    self.tmp / "bad_so.jsonl",
                    {
                        "event": "result",
                        "result": {
                            "structured_output": "{not valid json",
                            "usage": {"total_tokens": 1},
                        },
                    },
                )
            ),
            "no_result": str(
                _scenario(
                    self.tmp / "no_res.jsonl",
                    {
                        "event": "step_update",
                        "step_update": {
                            "step_type": "agent_response",
                            "state": "DONE",
                            "text_delta": "x",
                        },
                    },
                )
            ),
        }
        for label, scen in bad_streams.items():
            with self.subTest(label=label):
                os.environ["AGY_STUB_SCENARIO"] = scen
                rd = self.make_run_dir(name=f"run_{label}")
                self.assertIsNone(SB.score(rd, self.task))
                self.assertFalse((rd / "judgments.json").exists())

    def test_timeout_and_nonzero_exit_return_none(self) -> None:
        rd_exit = self.make_run_dir(name="run_exit")
        os.environ["AGY_STUB_EXIT_CODE"] = "2"
        self.assertIsNone(SB.score(rd_exit, self.task))
        self.assertFalse((rd_exit / "judgments.json").exists())
        os.environ.pop("AGY_STUB_EXIT_CODE", None)

        rd_hang = self.make_run_dir(name="run_hang")
        os.environ.update(AGY_STUB_HANG="1", AGY_STUB_HANG_S="60")
        t0 = time.monotonic()
        self.assertIsNone(SB.score(rd_hang, self.task, timeout=1))
        self.assertLess(time.monotonic() - t0, 20)
        self.assertFalse((rd_hang / "judgments.json").exists())

    def test_disagreeing_model_field_returns_none_agreeing_records_model(self) -> None:
        mismatch = _scenario(
            self.tmp / "judge_mismatch.jsonl",
            _judge_result(
                [{"id": f"c{i}", "pass": True, "reason": "ok"} for i in range(4)],
                model="sonnet",
            ),
        )
        os.environ["AGY_STUB_SCENARIO"] = str(mismatch)
        rd_bad = self.make_run_dir(name="run_mismatch")
        self.assertIsNone(SB.score(rd_bad, self.task))
        self.assertFalse((rd_bad / "judgments.json").exists())

        agree = _scenario(
            self.tmp / "judge_agree.jsonl",
            _judge_result(
                [{"id": f"c{i}", "pass": True, "reason": "ok"} for i in range(4)],
                model="fable",
            ),
        )
        os.environ["AGY_STUB_SCENARIO"] = str(agree)
        rd_ok = self.make_run_dir(name="run_agree")
        self.assertEqual(SB.score(rd_ok, self.task), {"n_passed": 4, "n_criteria": 4})
        cached = json.loads((rd_ok / "judgments.json").read_text(encoding="utf-8"))
        self.assertEqual(cached["actual_model"], "fable")
        self.assertEqual(cached["judgments"][0]["actual_model"], "fable")

    def test_invalid_rollout_returns_none_without_calling_judge(self) -> None:
        rd_invalid = self.make_run_dir(
            name="run_invalid", valid=False, invalid_reason="timeout after 60s"
        )
        self.assertIsNone(SB.score(rd_invalid, self.task))
        self.assertEqual(self.stub_calls(), [])
        self.assertFalse((rd_invalid / "judgments.json").exists())

        rd_missing = self.make_run_dir(name="run_no_timing")
        (rd_missing / "skill_timing.json").unlink()
        self.assertIsNone(SB.score(rd_missing, self.task))
        self.assertEqual(self.stub_calls(), [])


class TestJudgeCacheAndRepeats(_JudgeCase):
    def test_second_score_hits_judgments_cache_without_calling_agy(self) -> None:
        rd = self.make_run_dir()
        first = SB.score(rd, self.task)
        self.assertEqual(first, {"n_passed": 3, "n_criteria": 4})
        self.assertEqual(len(self.stub_calls()), 1)
        self.assertTrue((rd / "judgments.json").is_file())

        # Even if stub would now fail, the cache serves the existing judgment without calling agy.
        os.environ["AGY_STUB_EXIT_CODE"] = "3"
        second = SB.score(rd, "make-plan/t1", root=self.tasks_root)
        self.assertEqual(second, first)
        self.assertEqual(len(self.stub_calls()), 1)

    def test_changing_judge_model_or_rubric_or_deliverable_invalidates_cache(self) -> None:
        rd = self.make_run_dir()
        self.assertEqual(SB.score(rd, self.task), {"n_passed": 3, "n_criteria": 4})
        self.assertEqual(len(self.stub_calls()), 1)

        # Changing judge model invalidates cache.
        os.environ["SKILLBENCH_JUDGE_MODEL"] = "opus"
        self.assertEqual(SB.score(rd, self.task), {"n_passed": 3, "n_criteria": 4})
        self.assertEqual(len(self.stub_calls()), 2)

        # Changing rubric in task object invalidates cache.
        changed_task = replace(
            self.task,
            rubric=(
                SB.Criterion("c0", "Changed criterion text 0."),
                *self.task.rubric[1:],
            ),
        )
        self.assertEqual(SB.score(rd, changed_task), {"n_passed": 3, "n_criteria": 4})
        self.assertEqual(len(self.stub_calls()), 3)

        # Changing rubric.json on disk also invalidates cache.
        disk_rubric = [
            {"id": f"c{i}", "criterion": f"Updated disk criterion {i}."} for i in range(4)
        ]
        (self.task_dir / "rubric.json").write_text(json.dumps(disk_rubric), encoding="utf-8")
        self.assertEqual(SB.score(rd, self.task_dir), {"n_passed": 3, "n_criteria": 4})
        self.assertEqual(len(self.stub_calls()), 4)
        self.assertIn("Updated disk criterion 0.", self.stub_calls()[-1]["flags"]["prompt"])

        # Changing deliverable on disk invalidates cache.
        (rd / "final.md").write_text("Different final summary.\n", encoding="utf-8")
        self.assertEqual(SB.score(rd, self.task_dir), {"n_passed": 3, "n_criteria": 4})
        self.assertEqual(len(self.stub_calls()), 5)

    def test_judge_repeat_stores_multiple_judgments_and_uses_first_for_score(self) -> None:
        scen_second = _scenario(
            self.tmp / "judge_second.jsonl",
            _judge_result(
                [
                    {"id": "c0", "pass": True, "reason": "ok 0"},
                    {"id": "c1", "pass": False, "reason": "no 1"},
                    {"id": "c2", "pass": False, "reason": "no 2"},
                    {"id": "c3", "pass": False, "reason": "no 3"},
                ]
            ),
        )
        counter = self.tmp / "call_count.txt"
        wrapper = _write(
            self.tmp / "agy_repeat_wrapper.sh",
            f"#!/bin/sh\n"
            f'if [ ! -f "{counter}" ]; then\n'
            f'  echo 1 > "{counter}"\n'
            f'  export AGY_STUB_SCENARIO="{self.default_scenario}"\n'
            f"else\n"
            f'  export AGY_STUB_SCENARIO="{scen_second}"\n'
            f"fi\n"
            f'exec "{STUB}" "$@"\n',
        )
        wrapper.chmod(0o755)
        os.environ["HARNESS2_AGY_BIN"] = str(wrapper)
        os.environ["SKILLBENCH_JUDGE_REPEAT"] = "2"

        rd = self.make_run_dir()
        got = SB.score(rd, self.task)
        self.assertEqual(got, {"n_passed": 3, "n_criteria": 4})
        self.assertEqual(len(self.stub_calls()), 2)

        data = json.loads((rd / "judgments.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data["judgments"]), 2)
        self.assertEqual(data["score"], {"n_passed": 3, "n_criteria": 4})
        self.assertEqual(data["judgments"][0]["n_passed"], 3)
        self.assertEqual(data["judgments"][1]["n_passed"], 1)
        for j in data["judgments"]:
            self.assertEqual([c["id"] for c in j["criteria"]], ["c0", "c1", "c2", "c3"])
            self.assertEqual(j["n_criteria"], 4)

        # Second call with repeat=2 hits cache and makes no further agy calls.
        self.assertEqual(SB.score(rd, self.task), {"n_passed": 3, "n_criteria": 4})
        self.assertEqual(len(self.stub_calls()), 2)

    def test_invalid_judge_repeat_raises(self) -> None:
        rd = self.make_run_dir()
        for bad in ("0", "-1", "two"):
            with mock.patch.dict(os.environ, {"SKILLBENCH_JUDGE_REPEAT": bad}):
                with self.assertRaises(ConfigError):
                    SB.score(rd, self.task)


if __name__ == "__main__":
    unittest.main()
