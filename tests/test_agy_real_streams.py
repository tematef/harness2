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

"""Decoder regression over the real `jet` streams in tests/fixtures/agy/real/.

The streams were recorded by scripts/agy_spike.sh (files `<probe>.<model>.jsonl`). They carry
no `t` (the spike saves raw stdout), so `t_skill_s` is always None here.
"""

import json
import unittest
from pathlib import Path

from harness2.substrate.agy import (
    final_text,
    model_guard,
    parse_event,
    read_events,
    render_trajectory,
    skill_timing,
)

REAL = Path(__file__).resolve().parent / "fixtures" / "agy" / "real"
SKILL_PROBES = ("skill_read", "skill_read_slash_enabled")


def streams() -> list[Path]:
    return sorted(REAL.glob("*.jsonl"))


def probe_and_model(path: Path) -> tuple[str, str]:
    probe, model = path.stem.split(".", 1)
    return probe, model


def raw_result(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    results = [json.loads(ln)["result"] for ln in lines if '"event":"result"' in ln]
    assert len(results) == 1, path.name
    return results[0]


class RealStreamTest(unittest.TestCase):
    def test_streams_present(self) -> None:
        self.assertEqual(len(streams()), 10)

    def test_every_line_decodes(self) -> None:
        for path in streams():
            with self.subTest(stream=path.name):
                lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                dropped = [ln for ln in lines if parse_event(ln) is None]
                self.assertEqual(dropped, [])
                self.assertEqual(len(read_events(path)), len(lines))

    def test_final_text_is_the_result_response(self) -> None:
        for path in streams():
            with self.subTest(stream=path.name):
                expected = raw_result(path)["response"]
                got = final_text(read_events(path))
                self.assertEqual(got, expected)
                self.assertTrue(got.strip())

    def test_skill_read_detected_only_for_skill_probes(self) -> None:
        for path in streams():
            probe, _ = probe_and_model(path)
            with self.subTest(stream=path.name):
                st = skill_timing(read_events(path), "make-plan")
                self.assertIs(st["invoked"], probe in SKILL_PROBES)
                self.assertIsNone(st["t_skill_s"])
                if probe in SKILL_PROBES:
                    ev = read_events(path)[st["step_idx"]]
                    self.assertEqual((ev.tool_name, ev.state), ("read_file", "ACTIVE"))
                    self.assertTrue(ev.tool_args["Path"].endswith("/make-plan/SKILL.md"))

    def test_model_guard_verifies_init_model(self) -> None:
        for path in streams():
            _, model = probe_and_model(path)
            with self.subTest(stream=path.name):
                self.assertEqual(model_guard(read_events(path), model), (model, None))

    def test_model_guard_flags_the_other_model(self) -> None:
        events = read_events(REAL / "plain_answer.opus-5.5-high.jsonl")
        actual, reason = model_guard(events, "fable-5.1-high")
        self.assertEqual(actual, "opus-5.5-high")
        self.assertIn("model mismatch", reason)

    def test_json_schema_structured_output(self) -> None:
        for path in sorted(REAL.glob("json_schema.*.jsonl")):
            with self.subTest(stream=path.name):
                results = [ev for ev in read_events(path) if ev.is_result]
                self.assertEqual(len(results), 1)
                out = results[0].structured_output
                self.assertIsInstance(out, dict)
                self.assertEqual([c["id"] for c in out["criteria"]], ["c1", "c2"])
                for crit in out["criteria"]:
                    self.assertIsInstance(crit["pass"], bool)
                    self.assertIsInstance(crit["reason"], str)
                self.assertIsInstance(results[0].total_tokens, int)

    def test_trajectory_has_no_duplicate_actions_and_shows_output(self) -> None:
        for path in streams():
            with self.subTest(stream=path.name):
                out = render_trajectory(read_events(path), obs_trunc=500, max_chars=1_000_000)
                actions = [ln for ln in out.splitlines() if ln.startswith("[ACTION] ")]
                n_active = sum(1 for ev in read_events(path) if ev.is_tool and ev.state == "ACTIVE")
                self.assertEqual(len(actions), n_active)
                if n_active:
                    self.assertIn("[OBSERVATION] ", out)


if __name__ == "__main__":
    unittest.main()
