# skillbench: agy spike notes

`agy` is not installed on the development host, so every skillbench test runs against
**synthetic** stream-json fixtures in `tests/fixtures/agy/`. Their shape is derived from the
revmux decoder (`~/projects/revmux/app/executor/agy.go`), and every behaviour they encode is an
assumption until the pre-pilot spike checks it against real streams.

The spike is `scripts/agy_spike.sh`. It is **not run during implementation**; run it once on a
host with `agy` installed and authenticated, before the paid pilot:

```bash
SKILLBENCH_PLUGINS_DIR=~/projects/cc-thingz/plugins \
SKILLBENCH_AUTH_FILES=<comma list relative to $HOME> \
bash scripts/agy_spike.sh            # --dry-run prints the probes without calling agy
```

It stages the allowlisted plugins into a throwaway `HOME`, runs each probe with
`--model opus` and `--model fable`, and writes scrubbed streams, stderr, `agy --help` and an
auth-file report to `tests/fixtures/agy/real/`. Compare those streams with the synthetic
fixtures below, then update the fixtures, the decoder and this file wherever an assumption was
wrong.

## Event shape (from revmux)

One JSON object per line:

- `{"event": "step_update", "step_update": {step_type, state, tool_name, text_delta, tool_info: {name, parameters}}}`
  - `step_type`: `agent_response` / `message` (prose) or `tool` / `tool_call` (tool use)
  - `state`: `ACTIVE` while streaming, `DONE` on completion; a tool's `DONE` event carries no `tool_info`
- `{"event": "result", "result": {structured_output, usage: {total_tokens}, model?}}` — the last event
- lines that are not JSON are ignored by the decoder

## Fixture scenarios

| Scenario | Fixture | What it encodes |
| :------- | :------ | :-------------- |
| plain answer | `tests/fixtures/agy/plain_answer.jsonl` | prose as `agent_response` text deltas, `result` with usage only |
| skill read | `tests/fixtures/agy/skill_read.jsonl` | `view_file` of a repo file, of `brainstorm/SKILL.md` (another skill), then of the target `make-plan/SKILL.md` twice, then a `write_to_file` |
| `ask_question` call | `tests/fixtures/agy/ask_question.jsonl` | `ask_question` tool call that returns no answer; the agent proceeds with its recommended option |
| `--json-schema` result | `tests/fixtures/agy/json_schema.jsonl` | `result.structured_output` as a pre-parsed object in the judge schema |
| result with model field | `tests/fixtures/agy/result_with_model.jsonl` | `result.model` = `opus` (agrees with `--model opus`) |
| result with disagreeing model | `tests/fixtures/agy/result_model_mismatch.jsonl` | `result.model` = `sonnet` (disagrees with `--model opus`) |
| result without model field | `tests/fixtures/agy/result_without_model.jsonl` | no `model` key anywhere |
| malformed line | `tests/fixtures/agy/malformed_line.jsonl` | exactly one truncated, non-JSON line between valid events |

Paths inside the fixtures use the placeholder rollout root `/tmp/skillbench-run/`
(`home/` = isolated `HOME`, `work/` = rollout cwd).

## Open questions and assumed answers

1. **Model aliases.** Does `agy --model` accept the aliases `opus` and `fable`?
   Assumed: yes, both are accepted verbatim as aliases, and an unknown alias is not rejected
   (agy may silently fall back to its default model, as `claude` does).
   *assumed — verify in pre-pilot spike*
2. **Model field in the stream.** Does stream-json say which model actually ran?
   Assumed: at most an optional string `result.model`; usually absent (revmux hardcodes
   `ActualModel = "agy"`). Present ⇒ compare with the requested model; absent ⇒
   `actual_model = null` and the report says "model unverified". Fixtures:
   `result_with_model`, `result_model_mismatch`, `result_without_model`.
   *assumed — verify in pre-pilot spike*
3. **How a skill read appears.** Assumed: an ordinary `view_file` tool step
   (`step_type: tool`, `tool_name: view_file`) whose `tool_info.parameters.AbsolutePath` is
   `<HOME>/.gemini/config/plugins/<plugin>/skills/<skill>/SKILL.md`; there is no dedicated
   "skill invoked" event. Time-to-skill is the first such `ACTIVE` event for the target skill.
   Fixture: `skill_read`.
   *assumed — verify in pre-pilot spike*
4. **`ask_question` under `-p`.** Assumed: the tool call is emitted as a normal tool step and
   returns immediately without an answer in non-interactive print mode; the run does not block
   and the agent continues. Fixture: `ask_question`.
   *assumed — verify in pre-pilot spike*
5. **`PreToolUse` answer hook.** Can a `PreToolUse` hook answer `ask_question` on the user's
   behalf? Assumed: no — AGY's documented `PreToolUse` contract returns only a `decision`
   (`allow` / `ask` / `deny` / `force_ask`), so skillbench uses the simulated-user preamble in
   the prompt instead and builds no answer hook. No fixture.
   *assumed — verify in pre-pilot spike*
6. **Skills under `--disable-slash-commands`.** Assumed: the flag only stops `/name` parsing of
   the prompt; plugin skills stay listed to the model and readable, so skill reads still happen.
   The synthetic `skill_read` fixture stands for a run with the flag set; the spike also runs a
   `skill_read_slash_enabled` probe without the flag for comparison.
   *assumed — verify in pre-pilot spike*
7. **Default `--effort`.** Assumed: unknown and model-dependent; skillbench never relies on it
   and always passes `SKILLBENCH_EFFORT` explicitly, identical in both arms. The spike records
   `agy --help` to confirm the accepted values.
   *assumed — verify in pre-pilot spike*
8. **Auth files and whether agy writes them.** Assumed: agy reads OAuth credentials from a small
   set of files under `~/.gemini/` and may rewrite them on token refresh. The exact list is not
   known, so `SKILLBENCH_AUTH_FILES` is required; files are always **copied** (never symlinked)
   into the throwaway `HOME`, so a refresh cannot write through to the real `$HOME`. The spike
   records which files under the throwaway `HOME` agy opened, created or modified.
   *assumed — verify in pre-pilot spike*
9. **Plugin discovery only from the throwaway `HOME`.** Assumed: agy discovers plugins only
   through `$HOME/.gemini/config/` (the `plugins/` directory and `plugins.json`) plus the
   workspace `.agents/` of its cwd, and never reads the real home by absolute path. With
   `HOME=<tmp>` and a fixture repo without `.agents/`, only staged plugins are visible.
   *assumed — verify in pre-pilot spike*
10. **Non-JSON lines on stdout.** Assumed: agy may interleave non-JSON lines (warnings,
    truncated writes on kill) with events; decoders skip them. Fixture: `malformed_line`.
    *assumed — verify in pre-pilot spike*
