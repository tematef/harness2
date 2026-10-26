#!/usr/bin/env bash
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

# Pre-pilot spike: record real agy stream-json for the scenarios the synthetic fixtures in
# tests/fixtures/agy/ encode (see docs/skillbench-spike.md). Makes paid model calls; never run
# by the test suite. With the Google-internal Jetski CLI (a `jet` alias cannot be used):
#   HARNESS2_AGY_BIN=/google/bin/releases/jetski-devs/tools/cli \
#     SKILLBENCH_PLUGINS_DIR=~/projects/cc-thingz/plugins bash scripts/agy_spike.sh
#   bash scripts/agy_spike.sh --dry-run     # print the probes, call nothing, write nothing
# Settings: HARNESS2_AGY_BIN (default agy), SKILLBENCH_PLUGINS (allowlist),
#   SKILLBENCH_AUTH_FILES (optional comma list relative to $HOME, copied, never symlinked;
#   internal Jetski needs none), SPIKE_MODELS (default "opus-5.5-high fable-5.1-high", jet
#   model ids that pin the effort), SPIKE_TIMEOUT (default 5m),
#   SPIKE_OUT (default tests/fixtures/agy/real).
# Every probe runs under a fresh throwaway HOME holding only the staged plugins and copies of
# any auth files; streams and stderr are scrubbed of secrets, home paths and the local user and
# group names before saving.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  "") ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

AGY=${HARNESS2_AGY_BIN:-agy}
PLUGINS_DIR=${SKILLBENCH_PLUGINS_DIR:?set SKILLBENCH_PLUGINS_DIR to a cc-thingz plugins/ checkout}
PLUGINS=${SKILLBENCH_PLUGINS:-brainstorm,planning,review,skill-eval,thinking-tools,workflow}
AUTH_FILES=${SKILLBENCH_AUTH_FILES:-}
MODELS=${SPIKE_MODELS:-opus-5.5-high fable-5.1-high}
TIMEOUT=${SPIKE_TIMEOUT:-5m}
OUT=${SPIKE_OUT:-tests/fixtures/agy/real}
REAL_HOME=$HOME
USER_NAME=$(id -un 2>/dev/null || true)
GROUP_NAME=$(id -gn 2>/dev/null || true)
PROBES=(plain_answer skill_read skill_read_slash_enabled ask_question json_schema)
SCHEMA='{"type":"object","required":["criteria"],"properties":{"criteria":{"type":"array","items":{"type":"object","required":["id","pass","reason"],"properties":{"id":{"type":"string"},"pass":{"type":"boolean"},"reason":{"type":"string"}}}}}}'

die() { echo "agy_spike: $*" >&2; exit 1; }

[[ -d $PLUGINS_DIR ]] || die "SKILLBENCH_PLUGINS_DIR is not a directory: $PLUGINS_DIR"
IFS=, read -ra PLUGIN_NAMES <<<"$PLUGINS"
IFS=, read -ra AUTH_NAMES <<<"$AUTH_FILES"
for p in "${PLUGIN_NAMES[@]}"; do
  [[ -d $PLUGINS_DIR/$p ]] || die "allowlisted plugin not found: $PLUGINS_DIR/$p"
done
if [[ $DRY_RUN == 0 ]]; then
  command -v "$AGY" >/dev/null ||
    die "agy binary not found: $AGY (set HARNESS2_AGY_BIN to its path, e.g." \
      "/google/bin/releases/jetski-devs/tools/cli; a shell alias such as jet cannot be used)"
  for f in "${AUTH_NAMES[@]}"; do
    [[ -z $f || -f $REAL_HOME/$f ]] || die "auth file missing: ~/$f"
  done
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/agy-spike.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
SPIKE_HOME=$TMP/home
WORK=$TMP/work

# Fresh throwaway HOME: allowlisted plugins (symlinks resolved) + copied auth files only.
stage_home() {
  rm -rf "$SPIKE_HOME"
  mkdir -p "$SPIKE_HOME/.gemini/config/plugins"
  local p f
  for p in "${PLUGIN_NAMES[@]}"; do
    cp -RL "$PLUGINS_DIR/$p" "$SPIKE_HOME/.gemini/config/plugins/$p"
  done
  for f in "${AUTH_NAMES[@]}"; do
    [[ -z $f ]] && continue
    mkdir -p "$(dirname "$SPIKE_HOME/$f")"
    cp -pL "$REAL_HOME/$f" "$SPIKE_HOME/$f"
  done
}

# Fresh git repo as the rollout cwd.
make_work() {
  rm -rf "$WORK"
  mkdir -p "$WORK"
  cat >"$WORK/app.py" <<'EOF'
import json
import urllib.request


def fetch(url):
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)
EOF
  git -C "$WORK" init -q
  git -C "$WORK" add -A
  git -C "$WORK" -c user.name=spike -c user.email=spike@example.invalid commit -qm init
}

probe_prompt() {
  case $1 in
    plain_answer)
      echo "What does fetch() in app.py return? Answer in one sentence. Do not edit any file." ;;
    skill_read | skill_read_slash_enabled)
      echo "Use the make-plan skill to write an implementation plan for adding an LRU cache" \
        "to fetch() in app.py." ;;
    ask_question)
      echo "Before changing anything, use the ask_question tool to ask me which cache backend" \
        "to use (in-memory, Redis or SQLite), then add that cache to fetch() in app.py." ;;
    json_schema)
      echo "Grade the file app.py in the current directory against this rubric." \
        "c1: app.py defines fetch(). c2: fetch() results are cached." ;;
  esac
}

scrub() {
  local ids=()
  [[ -n $USER_NAME ]] && ids+=(-e "s#\\b${USER_NAME}\\b#<user>#g")
  [[ -n $GROUP_NAME && $GROUP_NAME != "$USER_NAME" ]] && ids+=(-e "s#\\b${GROUP_NAME}\\b#<group>#g")
  sed -E \
    -e "s#${TMP}#<TMP>#g" \
    -e "s#${REAL_HOME}#<HOME>#g" \
    "${ids[@]}" \
    -e 's#("(access_token|refresh_token|id_token|client_secret|api_key|apikey|token|authorization)"[[:space:]]*:[[:space:]]*")[^"]*"#\1[scrubbed]"#gI' \
    -e 's#ya29\.[A-Za-z0-9._-]+#[scrubbed]#g' \
    -e 's#1//[A-Za-z0-9._-]{20,}#[scrubbed]#g' \
    -e 's#AIza[0-9A-Za-z_-]{35}#[scrubbed]#g' \
    -e 's#Bearer [A-Za-z0-9._~+/=-]+#Bearer [scrubbed]#g'
}

auth_digest() {
  local f
  for f in "${AUTH_NAMES[@]}"; do
    [[ -n $f ]] && sha256sum "$REAL_HOME/$f" 2>/dev/null | sed "s#${REAL_HOME}#<HOME>#"
  done
  return 0
}

# Which files under the throwaway HOME agy wrote (mtime) and opened (strace, when available),
# and whether any real-HOME auth file changed. Appended to $OUT/auth-files.txt.
report_auth() {
  local label=$1 rc=$2 marker=$3 before=$4
  {
    echo "## $label (exit $rc)"
    echo "written under throwaway HOME (outside plugins/):"
    find "$SPIKE_HOME" -type f -newer "$marker" -not -path '*/.gemini/config/plugins/*' \
      | sed "s#${SPIKE_HOME}#  <tmp-home>#" | sort
    if [[ -s $TMP/strace.log ]]; then
      echo "opened under throwaway HOME (outside plugins/):"
      grep -o '"[^"]*"' "$TMP/strace.log" | tr -d '"' | grep "^${SPIKE_HOME}/" \
        | grep -v '/.gemini/config/plugins/' | sed "s#${SPIKE_HOME}#  <tmp-home>#" | sort -u || true
      echo "plugin files opened: $(grep -o '"[^"]*"' "$TMP/strace.log" | tr -d '"' \
        | grep -c "^${SPIKE_HOME}/.gemini/config/plugins/" || true)"
      echo "opened under real HOME (expected: none):"
      grep -o '"[^"]*"' "$TMP/strace.log" | tr -d '"' | grep "^${REAL_HOME}/" \
        | grep -v "^${TMP}/" | sed "s#${REAL_HOME}#  <real-home>#" | sort -u || true
    else
      echo "opened files: not recorded (strace unavailable)"
    fi
    if [[ -z $AUTH_FILES ]]; then
      echo "real-HOME auth files: none listed (SKILLBENCH_AUTH_FILES empty)"
    elif [[ $(auth_digest) == "$before" ]]; then
      echo "real-HOME auth files: unchanged"
    else
      echo "real-HOME auth files: CHANGED"
    fi
    echo
  } >>"$OUT/auth-files.txt"
}

run_probe() {
  local name=$1 model=$2
  local argv=(-p "$(probe_prompt "$name")" --output-format stream-json
    --dangerously-skip-permissions)
  [[ $name != skill_read_slash_enabled ]] && argv+=(--disable-slash-commands)
  argv+=(--model "$model")
  [[ $name == json_schema ]] && argv+=(--json-schema "$SCHEMA")
  argv+=(--print-timeout "$TIMEOUT")

  if [[ $DRY_RUN == 1 ]]; then
    printf '[dry-run] %s.%s: HOME=<tmp>/home' "$name" "$model"
    printf ' %q' "$AGY" "${argv[@]}"
    printf '\n'
    return
  fi

  stage_home
  make_work
  local marker=$TMP/marker before rc
  touch "$marker"
  before=$(auth_digest)
  local tracer=()
  rm -f "$TMP/strace.log"
  if command -v strace >/dev/null; then
    tracer=(strace -f -qq -e trace=open,openat,creat,rename,renameat,renameat2 -o "$TMP/strace.log")
  fi
  echo "probe $name.$model ..." >&2
  set +e
  (cd "$WORK" && env HOME="$SPIKE_HOME" XDG_CONFIG_HOME="$SPIKE_HOME/.config" \
    XDG_CACHE_HOME="$SPIKE_HOME/.cache" XDG_DATA_HOME="$SPIKE_HOME/.local/share" \
    XDG_STATE_HOME="$SPIKE_HOME/.local/state" EDITOR=true VISUAL=true \
    "${tracer[@]}" "$AGY" "${argv[@]}") >"$TMP/stdout" 2>"$TMP/stderr"
  rc=$?
  set -e
  scrub <"$TMP/stdout" >"$OUT/$name.$model.jsonl"
  scrub <"$TMP/stderr" >"$OUT/$name.$model.stderr.txt"
  report_auth "$name.$model" "$rc" "$marker" "$before"
}

if [[ $DRY_RUN == 0 ]]; then
  mkdir -p "$OUT"
  : >"$OUT/auth-files.txt"
  { "$AGY" --help 2>&1 || true; } | scrub >"$OUT/help.txt"
else
  echo "[dry-run] plugins: ${PLUGIN_NAMES[*]}; auth files copied: ${AUTH_FILES:-<none>}"
fi
for model in $MODELS; do
  for probe in "${PROBES[@]}"; do
    run_probe "$probe" "$model"
  done
done
[[ $DRY_RUN == 1 ]] || echo "saved streams to $OUT; compare with tests/fixtures/agy/*.jsonl" >&2
