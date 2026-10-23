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

# Run skillbench end to end: check, roll WITH and WITHOUT arms, judge bases, report.
#   bash scripts/run_skillbench.sh                                       # every task
#   PILOT=1 K=3 bash scripts/run_skillbench.sh                           # paid pilot (3 skills)
#   bash scripts/run_skillbench.sh --only-domains make-plan --max-tasks 1
# Settings: SUBSTRATE=agy  K=1  SEED=<int>  PILOT=0|1  MODE=parallel|sequential
# Extra arguments go to the harness2 sweep commands; --dry-run stops after them.
set -uo pipefail
cd "$(dirname "$0")/.."
PATH="$PWD/.venv/bin:$PATH"
SUBSTRATE=${SUBSTRATE:-agy}
K=${K:-1}
SEED=${SEED:-}
PILOT=${PILOT:-0}
MODE=${MODE:-parallel}

prev=""
for arg in "$@"; do
  case "$prev" in
    --k) K="$arg" ;;
    --seed) SEED="$arg" ;;
  esac
  case "$arg" in
    --k=*) K="${arg#--k=}" ;;
    --seed=*) SEED="${arg#--seed=}" ;;
  esac
  prev="$arg"
done

run=(--bench skillbench --substrate "$SUBSTRATE" --k "$K" --mode "$MODE")
pilot_args=()
report=(--k "$K")
check_args=(--bench skillbench --substrate "$SUBSTRATE")

if [[ -n "$SEED" ]]; then
  run+=(--seed "$SEED")
  report+=(--seed "$SEED")
fi

if [[ "$PILOT" == "1" || "$PILOT" == "true" || "$PILOT" == "yes" || " $* " == *" --pilot "* ]]; then
  export SKILLBENCH_JUDGE_REPEAT=${SKILLBENCH_JUDGE_REPEAT:-2}
  pilot_args+=(--pilot)
  report+=(--pilot)
  check_args+=(--pilot)
fi

harness2-check "${check_args[@]}" || exit
harness2 "${run[@]}" "${pilot_args[@]}" --arm with "$@"
status_with=$?
harness2 "${run[@]}" "${pilot_args[@]}" --arm without "$@"
status_without=$?
status=$((status_with ? status_with : status_without))
[[ " $* " == *" --dry-run "* ]] && exit $status
harness2-judge "${run[@]}" --arm with --bases
judged_with=$?
harness2-judge "${run[@]}" --arm without --bases
judged_without=$?
judged=$((judged_with ? judged_with : judged_without))
skillbench-report "${report[@]}"
reported=$?
exit $((status ? status : (judged ? judged : reported)))
