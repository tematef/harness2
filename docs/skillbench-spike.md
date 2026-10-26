# skillbench: agy spike notes

Every skillbench test runs offline against two sets of stream-json fixtures:

- **real** streams in `tests/fixtures/agy/real/`, recorded once with the Google-internal Jetski
  CLI (`jet`, `/google/bin/releases/jetski-devs/tools/cli`, same flags as `agy`) on
  2026-10-26 by `scripts/agy_spike.sh`;
- **synthetic** fixtures in `tests/fixtures/agy/`, small hand-written streams in the same shape
  that the stub `tests/fixtures/agy_stub.py` replays for the substrate, judge and end-to-end
  tests.

`tests/test_agy_real_streams.py` runs the decoder over the real streams, and
`tests/test_agy_fixtures.py` checks that every synthetic event stays within the shape of the
real ones.

## Re-running the spike

The spike makes about ten short paid model calls; the test suite never runs it. Re-run it after
a Jetski upgrade, then compare the new streams with the synthetic fixtures and keep both test
files green:

```bash
HARNESS2_AGY_BIN=/google/bin/releases/jetski-devs/tools/cli \
SKILLBENCH_PLUGINS_DIR=~/projects/cc-thingz/plugins \
bash scripts/agy_spike.sh            # --dry-run prints the probes without calling agy
```

`HARNESS2_AGY_BIN` must be the executable itself: `jet` is a shell alias or function, which a
subprocess cannot start. `SKILLBENCH_AUTH_FILES` is optional (see question 8) and
`SPIKE_MODELS` defaults to `opus-5.5-high fable-5.1-high`.

Each probe runs in a fresh throwaway `HOME` holding only the allowlisted plugins (and copies of
any listed auth files), in a fresh git repo with a small `app.py`. The script writes to
`tests/fixtures/agy/real/`:

| File | Content |
| :--- | :------ |
| `<probe>.<model>.jsonl` | stdout of one probe, scrubbed of secrets, the temp dir (`<TMP>`), the real home (`<HOME>`) and the local user and group names |
| `<probe>.<model>.stderr.txt` | stderr of that probe (empty in the recorded run) |
| `help.txt` | `agy --help` |
| `auth-files.txt` | per probe: files written and opened under the throwaway `HOME`, files opened under the real `HOME`, and whether listed real-`HOME` auth files changed |

Probes: `plain_answer`, `skill_read` (asks for the `make-plan` skill, with
`--disable-slash-commands`), `skill_read_slash_enabled` (same, without the flag),
`ask_question`, and `json_schema` (`--json-schema` with the judge schema). The saved streams
carry no `t`: the spike stores raw stdout, while real rollouts get `t` stamped by the substrate.

## Event shape (verified)

One JSON object per line, no other lines:

- `{"event": "init", "conversation_id", "init": {model, cwd, tools, permission_mode, json_schema?}}`
  — always the first line; `init.model` is the full model id that ran (`opus-5.5-high`).
- `{"event": "step_update", "step_update": {conversation_id, step_index, step_type, state, tool_name?, text_delta?, tool_info?, duration_seconds?, usage?}}`
  - `step_type`: `user_input`, `unknown`, `agent_response` (prose) or `tool`
  - `state`: `ACTIVE` while streaming, `DONE` on completion. `agent_response` text arrives as
    `ACTIVE` deltas; the `DONE` delta carries `usage` and is often just `"\n"` or empty.
  - a tool's `ACTIVE` step has `tool_info: {name, parameters}`; its `DONE` step repeats them and
    adds `output`
- `{"event": "result", "result": {conversation_id, status, response, duration_seconds, num_turns, usage: {total_tokens, ...}, json_schema?, structured_output?}}`
  — always the last line. `response` holds the whole answer; there is no model field.

The decoder also still reads the older shapes the synthetic fixtures keep for coverage:
`result.model`, `view_file` reads with `AbsolutePath`, `write_to_file` with `TargetFile`, and
`DONE` tool steps without `tool_info`.

## Fixture scenarios

| Scenario | Fixture | What it encodes |
| :------- | :------ | :-------------- |
| plain answer | `tests/fixtures/agy/plain_answer.jsonl` | `user_input`, prose as `agent_response` deltas, `result.response` |
| skill read | `tests/fixtures/agy/skill_read.jsonl` | `read_file` (`Path`) of a repo file, of `brainstorm/SKILL.md` (another skill), then of the target `make-plan/SKILL.md`, a `view_file` re-read of the target, a `write_file` (`Path`, `Content`), and a last `DONE` delta of just `"\n"` |
| `ask_question` call | `tests/fixtures/agy/ask_question.jsonl` | an `ask_question` tool call that gets no answer; kept for decoder coverage, real models do not call it under `-p` (question 4) |
| `--json-schema` result | `tests/fixtures/agy/json_schema.jsonl` | a `finish` tool step and `result.structured_output` in the judge schema |
| stream with matching model | `tests/fixtures/agy/result_with_model.jsonl` | `init.model` = `opus-5.5-high` |
| stream with another model | `tests/fixtures/agy/result_model_mismatch.jsonl` | `init.model` = `sonnet-4-high` (disagrees with `opus-5.5-high`) |
| stream without model | `tests/fixtures/agy/result_without_model.jsonl` | no `model` key anywhere |
| malformed line | `tests/fixtures/agy/malformed_line.jsonl` | one truncated, non-JSON line between valid events; a legacy `view_file` read |

Paths inside the fixtures use the placeholder rollout root `/tmp/skillbench-run/` (`home/` =
isolated `HOME`, `work/` = rollout cwd), and `init.model` is the placeholder
`<requested-model>`, which the stub replaces with the `--model` it was given. The three model
fixtures name their model literally.

## Spike questions

1. **Model ids.** Does `agy --model` accept `opus` and `fable`?
   **Verified:** no plain aliases exist. `jet models` lists ids that embed the effort:
   `opus-5.5-high`, `fable-5.1-high` (also `-low`, `-medium`, `-max`). skillbench defaults to
   `opus-5.5-high` (executor) and `fable-5.1-high` (judge).
2. **Model field in the stream.** Does stream-json say which model actually ran?
   **Verified:** yes, in the leading `init` event as `init.model`, equal to the requested id.
   `result` has no model field. The model guard compares `init.model` with the requested model,
   so real rollouts are no longer "model unverified". Fixtures: `result_with_model`,
   `result_model_mismatch`, `result_without_model`.
3. **How a skill read appears.** **Verified:** an ordinary tool step, `step_type: tool`,
   `tool_name: read_file`, with `tool_info.parameters.Path` =
   `<HOME>/.gemini/config/plugins/planning/skills/make-plan/SKILL.md`. There is no dedicated
   "skill invoked" event; time-to-skill is the first such `ACTIVE` step. Writes are
   `write_file` steps with `Path` (the stream carries no file content). Fixture: `skill_read`.
4. **`ask_question` under `-p`.** **Verified:** `ask_question` is listed in `init.tools`, but in
   print mode neither model called it; both said they had no such tool. `opus-5.5-high` asked in
   plain text and stopped, `fable-5.1-high` picked a default and went on. skillbench therefore
   keeps the simulated-user preamble in the prompt. Fixture: `ask_question` (decoder coverage
   only).
5. **`PreToolUse` answer hook.** Can a `PreToolUse` hook answer `ask_question` for the user?
   **Not verified:** the spike did not test it, and it is not needed: the models never call
   `ask_question` under `-p` (question 4), and the simulated-user preamble covers the case. No
   fixture.
6. **Skills under `--disable-slash-commands`.** **Verified:** the flag does not hide skills.
   Both models read `make-plan/SKILL.md` with the flag (`skill_read`) and without it
   (`skill_read_slash_enabled`), so skillbench keeps passing it.
7. **Default `--effort`.** **Verified:** `--effort` accepts `low|medium|high|xhigh|max`
   (`help.txt`), but the model id already pins the effort. `SKILLBENCH_EFFORT` is optional:
   empty passes no `--effort`; a set value must be one of those five and is sent to both arms
   and the judge. Its value (or `""`) stays in the run identity, so a resume with a different
   effort is refused.
8. **Auth files.** **Verified:** internal Jetski needs none; a probe in an empty `HOME`
   authenticated without any file. `SKILLBENCH_AUTH_FILES` is optional: empty copies nothing, a
   listed file that is missing (or a path outside `$HOME`) still fails preflight. Listed files
   are copied, never symlinked, and the real-`HOME` copies stayed unchanged in every probe.
9. **Plugin discovery only from the throwaway `HOME`.** **Verified:** Jetski discovered plugins
   only from the throwaway `HOME` (`auth-files.txt` lists the plugin files it opened there).
   The only real-`HOME` file it opened was `~/.gemini/jetski/bin/agentapi`, a helper binary, not
   configuration or credentials.
10. **Non-JSON lines on stdout.** **Verified:** none were seen, and stderr was empty for every
    probe. The decoder still skips non-JSON lines (truncated writes on kill). Fixture:
    `malformed_line`.

Also found: the last `agent_response` `DONE` delta is often just `"\n"`, so a final message
built from it was empty. `final.md` now comes from `result.response`.
