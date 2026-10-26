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

"""agy stream-json decoder: parse_event, read_events, skill_timing, render_trajectory."""

import json
import tempfile
import unittest
from pathlib import Path

from harness2.substrate.agy import (
    AgyEvent,
    final_text,
    parse_event,
    read_events,
    render_trajectory,
    skill_timing,
)

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "agy"


def _fixture_lines(name: str) -> list[str]:
    return (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()


def _timed(name: str, step: float = 0.5) -> list[AgyEvent]:
    """Fixture events with a synthetic `t` (seconds since spawn) like events.jsonl carries."""
    out = []
    for i, ln in enumerate(_fixture_lines(name)):
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        obj["t"] = round((i + 1) * step, 3)
        ev = parse_event(json.dumps(obj))
        if ev is not None:
            out.append(ev)
    return out


class ParseEventTest(unittest.TestCase):
    def test_init_event_carries_model(self):
        ev = parse_event(_fixture_lines("result_with_model")[0])
        self.assertEqual(ev.event, "init")
        self.assertEqual(ev.model, "opus-5.5-high")
        self.assertIsNone(ev.text)
        self.assertFalse(ev.is_tool)
        self.assertFalse(ev.is_result)

    def test_agent_response_step(self):
        ev = parse_event(_fixture_lines("plain_answer")[2])
        self.assertIsNotNone(ev)
        self.assertEqual(ev.event, "step_update")
        self.assertEqual(ev.step_type, "agent_response")
        self.assertEqual(ev.state, "ACTIVE")
        self.assertEqual(ev.text, "`add(a, b)` returns ")
        self.assertIsNone(ev.tool_name)
        self.assertIsNone(ev.tool_args)
        self.assertFalse(ev.is_result)

    def test_read_file_call_with_path(self):
        ev = parse_event(_fixture_lines("skill_read")[6])
        self.assertEqual(ev.step_type, "tool")
        self.assertEqual(ev.tool_name, "read_file")
        self.assertEqual(
            ev.tool_args,
            {
                "Path": "/tmp/skillbench-run/home/.gemini/config/plugins/planning/skills/make-plan/SKILL.md"
            },
        )
        self.assertIsNone(ev.tool_output)

    def test_tool_done_carries_output(self):
        ev = parse_event(_fixture_lines("skill_read")[3])
        self.assertEqual((ev.tool_name, ev.state), ("read_file", "DONE"))
        self.assertEqual(ev.tool_args, {"Path": "/tmp/skillbench-run/work/app.py"})
        self.assertEqual(ev.tool_output, "1: def fetch(key):\n2:     return key\n")

    def test_tool_call_with_parameters(self):
        ev = parse_event(_fixture_lines("skill_read")[8])
        self.assertEqual(ev.step_type, "tool")
        self.assertEqual(ev.tool_name, "view_file")
        self.assertEqual(
            ev.tool_args,
            {
                "AbsolutePath": "/tmp/skillbench-run/home/.gemini/config/plugins/planning/"
                "skills/make-plan/SKILL.md",
                "StartLine": 40,
                "EndLine": 120,
            },
        )

    def test_tool_done_without_tool_info(self):
        ev = parse_event(_fixture_lines("skill_read")[5])
        self.assertEqual(ev.tool_name, "read_file")
        self.assertEqual(ev.state, "DONE")
        self.assertIsNone(ev.tool_args)

    def test_tool_name_falls_back_to_tool_info_name(self):
        line = json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "state": "ACTIVE",
                    "tool_info": {"name": "read_file", "parameters": {"Path": "/x"}},
                },
            }
        )
        self.assertEqual(parse_event(line).tool_name, "read_file")

    def test_result_with_usage_and_structured_output(self):
        ev = parse_event(_fixture_lines("json_schema")[-1])
        self.assertTrue(ev.is_result)
        self.assertEqual(ev.total_tokens, 2150)
        self.assertEqual([c["id"] for c in ev.structured_output["criteria"]], ["c1", "c2"])
        self.assertTrue(ev.response.startswith("checking each criterion"))

    def test_result_response_captured(self):
        ev = parse_event(_fixture_lines("plain_answer")[-1])
        self.assertEqual(
            ev.response, "`add(a, b)` returns the sum of both arguments.\nNo changes were needed."
        )
        bad = json.dumps({"event": "result", "result": {"response": 3}})
        self.assertIsNone(parse_event(bad).response)
        self.assertIsNone(parse_event(_fixture_lines("plain_answer")[2]).response)

    def test_structured_output_as_json_string_is_decoded(self):
        line = json.dumps(
            {"event": "result", "result": {"structured_output": json.dumps({"a": 1})}}
        )
        self.assertEqual(parse_event(line).structured_output, {"a": 1})

    def test_model_field_present(self):
        self.assertEqual(parse_event(_fixture_lines("result_with_model")[0]).model, "opus-5.5-high")
        self.assertEqual(
            parse_event(_fixture_lines("result_model_mismatch")[0]).model, "sonnet-4-high"
        )
        # jet's result carries no model; the init event is the source
        self.assertIsNone(parse_event(_fixture_lines("result_with_model")[-1]).model)

    def test_model_sources_in_priority_order(self):
        def model(obj):
            return parse_event(json.dumps(obj)).model

        self.assertEqual(model({"event": "init", "init": {"model": "a"}}), "a")
        self.assertEqual(model({"event": "x", "model": "b", "init": {"model": "a"}}), "b")
        self.assertEqual(model({"event": "result", "result": {"model": "c"}, "model": "b"}), "c")
        self.assertIsNone(model({"event": "init", "init": {"model": ""}}))
        self.assertIsNone(model({"event": "init", "init": "opus"}))

    def test_model_field_absent(self):
        lines = _fixture_lines("result_without_model")
        self.assertIsNone(parse_event(lines[0]).model)
        ev = parse_event(lines[-1])
        self.assertTrue(ev.is_result)
        self.assertIsNone(ev.model)
        self.assertIsNone(ev.structured_output)
        self.assertEqual(ev.total_tokens, 980)

    def test_t_is_read_when_present(self):
        obj = json.loads(_fixture_lines("plain_answer")[0])
        self.assertIsNone(parse_event(json.dumps(obj)).t)
        obj["t"] = 1.25
        self.assertEqual(parse_event(json.dumps(obj)).t, 1.25)

    def test_malformed_lines_ignored(self):
        lines = _fixture_lines("malformed_line")
        parsed = [parse_event(ln) for ln in lines]
        self.assertEqual(sum(p is None for p in parsed), 1)
        self.assertIsNone(parsed[3])
        for bad in ("", "   ", "not json", "[1, 2]", '"str"', "{broken", '{"no_event": 1}'):
            self.assertIsNone(parse_event(bad), bad)

    def test_wrong_typed_payloads_do_not_raise(self):
        line = json.dumps({"event": "step_update", "step_update": "oops"})
        ev = parse_event(line)
        self.assertEqual(ev.event, "step_update")
        self.assertIsNone(ev.step_type)
        line = json.dumps({"event": "result", "result": {"usage": "x", "model": 3}, "t": "x"})
        ev = parse_event(line)
        self.assertIsNone(ev.total_tokens)
        self.assertIsNone(ev.model)
        self.assertIsNone(ev.t)

    def test_read_events_skips_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.jsonl"
            p.write_text((FIXTURES / "malformed_line.jsonl").read_text(encoding="utf-8"))
            evs = read_events(p)
            self.assertEqual(len(evs), 5)
            self.assertTrue(evs[-1].is_result)
            self.assertEqual(read_events(Path(d) / "missing.jsonl"), [])


class SkillTimingTest(unittest.TestCase):
    def test_first_read_of_target(self):
        evs = _timed("skill_read")
        st = skill_timing(evs, "make-plan")
        # line index 6 is the first make-plan read (t = 7 * 0.5); its DONE echo and the
        # later view_file re-read are ignored
        self.assertEqual(st, {"invoked": True, "t_skill_s": 3.5, "step_idx": 6})

    def test_legacy_view_file_read_counts(self):
        line = json.dumps(
            {
                "event": "step_update",
                "step_update": {
                    "step_type": "tool",
                    "state": "ACTIVE",
                    "tool_name": "view_file",
                    "tool_info": {
                        "name": "view_file",
                        "parameters": {"AbsolutePath": "/h/plugins/p/skills/make-plan/SKILL.md"},
                    },
                },
            }
        )
        self.assertTrue(skill_timing([parse_event(line)], "make-plan")["invoked"])

    def test_other_skill_reads_ignored(self):
        evs = _timed("skill_read")
        st = skill_timing(evs, "brainstorm")
        self.assertEqual(st["step_idx"], 4)
        # a skill that is only a suffix/prefix of a read skill name never matches
        self.assertFalse(skill_timing(evs, "plan")["invoked"])
        self.assertFalse(skill_timing(evs, "make")["invoked"])

    def test_never_read(self):
        for name in ("plain_answer", "ask_question", "malformed_line"):
            st = skill_timing(_timed(name), "make-plan")
            self.assertEqual(st, {"invoked": False, "t_skill_s": None, "step_idx": None})

    def test_write_to_skill_path_is_not_a_read(self):
        for tool, key in (("write_to_file", "TargetFile"), ("write_file", "Path")):
            line = json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "tool",
                        "state": "ACTIVE",
                        "tool_name": tool,
                        "tool_info": {
                            "name": tool,
                            "parameters": {key: "/h/skills/make-plan/SKILL.md"},
                        },
                    },
                    "t": 1.0,
                }
            )
            self.assertFalse(skill_timing([parse_event(line)], "make-plan")["invoked"], tool)

    def test_read_without_t(self):
        evs = [parse_event(ln) for ln in _fixture_lines("skill_read")]
        st = skill_timing(evs, "make-plan")
        self.assertTrue(st["invoked"])
        self.assertIsNone(st["t_skill_s"])
        self.assertEqual(st["step_idx"], 6)


class RenderTrajectoryTest(unittest.TestCase):
    def test_shared_grammar(self):
        out = render_trajectory(_timed("skill_read"), obs_trunc=2000, max_chars=100_000)
        lines = out.splitlines()
        self.assertEqual(lines[0], "[ASSISTANT] Looking at the repository first.")
        self.assertEqual(
            lines[1], '[ACTION] read_file({"Path": "/tmp/skillbench-run/work/app.py"})'
        )
        # a DONE step echoing its parameters renders its output, not a second [ACTION]
        self.assertEqual(lines[2], "[OBSERVATION] 1: def fetch(key):")
        self.assertEqual(sum(ln.startswith("[ACTION] read_file(") for ln in lines), 3)
        self.assertEqual(sum(ln.startswith("[ACTION] view_file(") for ln in lines), 1)
        self.assertEqual(sum(ln.startswith("[ACTION] write_file(") for ln in lines), 1)
        self.assertIn("[OBSERVATION] Created file docs/plans/20261023-add-cache.md", lines)
        self.assertEqual(
            lines[-1], "[ASSISTANT] Wrote the plan to docs/plans/20261023-add-cache.md."
        )
        for ln in lines:
            if ln.startswith("["):
                self.assertRegex(
                    ln, r"^\[(ASSISTANT|THINKING|ACTION|OBSERVATION|OBSERVATION ERROR|ERROR)\] "
                )

    def test_streamed_text_deltas_are_merged(self):
        out = render_trajectory(_timed("plain_answer"), obs_trunc=2000, max_chars=100_000)
        self.assertEqual(
            out,
            "[ASSISTANT] `add(a, b)` returns the sum of both arguments.\n"
            "[ASSISTANT] No changes were needed.",
        )

    def test_structured_result_rendered(self):
        out = render_trajectory(_timed("json_schema"), obs_trunc=2000, max_chars=100_000)
        self.assertIn('[OBSERVATION] structured_output: {"criteria": [', out)

    def test_error_states(self):
        evs = [
            parse_event(
                json.dumps(
                    {
                        "event": "step_update",
                        "step_update": {
                            "step_type": "tool",
                            "state": "ERROR",
                            "tool_name": "run_command",
                            "text_delta": "exit status 2",
                        },
                    }
                )
            ),
            parse_event(json.dumps({"event": "error", "error": {"message": "quota exceeded"}})),
        ]
        out = render_trajectory(evs, obs_trunc=2000, max_chars=100_000)
        self.assertEqual(
            out, "[OBSERVATION ERROR] run_command: exit status 2\n[ERROR] quota exceeded"
        )

    def test_truncation(self):
        evs = _timed("skill_read")
        out = render_trajectory(evs, obs_trunc=10, max_chars=100_000, args_trunc=20)
        self.assertIn("[ASSISTANT] Looking at…(+", out)
        self.assertIn("chars — render truncation)", out)
        capped = render_trajectory(evs, obs_trunc=2000, max_chars=50)
        head, tail = capped.rsplit("\n", 1)
        self.assertEqual(len(head), 50)
        self.assertRegex(tail, r"^…\(\+\d+ chars — render truncation\)$")

    def test_blank_tool_output_renders_nothing(self):
        su = {
            "step_type": "tool",
            "state": "DONE",
            "tool_name": "run_command",
            "tool_info": {
                "name": "run_command",
                "parameters": {"CommandLine": "true"},
                "output": " ",
            },
        }
        ev = parse_event(json.dumps({"event": "step_update", "step_update": su}))
        self.assertEqual(
            render_trajectory([ev], obs_trunc=100, max_chars=1000),
            "(no parseable events in trajectory)",
        )

    def test_empty(self):
        self.assertEqual(
            render_trajectory([], obs_trunc=10, max_chars=10), "(no parseable events in trajectory)"
        )


class FinalTextTest(unittest.TestCase):
    def test_prefers_result_response_over_done_delta(self):
        # skill_read ends like a real jet stream: the last DONE delta is just "\n"
        evs = _timed("skill_read")
        self.assertEqual(
            final_text(evs),
            "Looking at the repository first.\nWriting the plan file.\n"
            "Wrote the plan to docs/plans/20261023-add-cache.md.\n",
        )

    def test_last_non_empty_response_wins(self):
        def result(resp):
            return parse_event(json.dumps({"event": "result", "result": {"response": resp}}))

        done = parse_event(
            json.dumps(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_type": "agent_response",
                        "state": "DONE",
                        "text_delta": "delta",
                    },
                }
            )
        )
        self.assertEqual(final_text([result("first"), result("second"), done]), "second")
        self.assertEqual(final_text([result("first"), result("  \n")]), "first")
        self.assertEqual(final_text([done, result("")]), "delta")


if __name__ == "__main__":
    unittest.main()
