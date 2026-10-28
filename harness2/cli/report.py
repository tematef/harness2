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

"""Render the score report for one cell from cached judged.json rows — read-only.

The figure each benchmark reports per (mode, k) — partial credit and all-criteria pass on LAB,
one score elsewhere — with pick provenance, the k=1 noise-floor note and the sanity warnings;
fell-back counts and improver spend come from the task summaries.

    python3 -m harness2.cli.report --bench jb --substrate oc --k 1,2

With --bases it reports the reference executions scored by `cli.judge --bases`
(base_judged.json) instead. skillbench runs reference executions only, so --bases is implied
for it and --arm picks the arm namespace:

    python3 -m harness2.cli.report --bench skillbench --arm with --k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..core import report as R
from ..core import schedule as SCHED
from ..core.judge import Scored
from . import _common as C
from .judge import _load_rows


def _load_base_rows(path: Path, k: int) -> list[Scored]:
    """Rows of a base_judged.json (written by `cli.judge --bases`) as `mode="base"` Scored rows;
    an unscored execution (`score: null`) becomes an ungraded row."""
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable base judged cache {path}: {exc}; re-run cli.judge --bases")
    rows = []
    for row in raw:
        score = row.get("score") if isinstance(row.get("score"), dict) else None
        n_passed = score.get("n_passed") if score else None
        n_criteria = score.get("n_criteria") if score else None
        rows.append(
            Scored(
                task_id=str(row.get("task", "")),
                mode=R.BASE_MODE,
                k=k,
                label=str(row.get("label", "")),
                run_id=str(row.get("run_id", "")),
                chosen_is_base=True,
                n_passed=n_passed,
                n_criteria=n_criteria,
                all_pass=bool(score) and n_criteria is not None and n_passed == n_criteria,
                raw=score or {},
                error="" if score else "ungraded",
            )
        )
    return rows


def _spend_and_fallbacks(ns_dir: Path, modes: tuple[str, ...], k: int) -> tuple[dict, dict]:
    """(fell_back, usd) keyed (mode token, k), from the summaries the sweep checkpointed."""
    fell_back: dict[tuple[str, int], int] = {}
    usd: dict[tuple[str, int], float] = {}
    for summary_path in sorted(ns_dir.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! skipping unreadable {summary_path}: {exc}", flush=True)
            continue
        records = SCHED._modes_of(summary)
        for mode in modes:
            key = (mode, k)
            rec = (records or {}).get(mode) or {}
            fell_back[key] = fell_back.get(key, 0) + len(rec.get("fell_back") or [])
            spend = sum(float(s.get("usd") or 0.0) for s in rec.get("stages") or [])
            sel = (summary.get("selections") or {}).get(mode) or {}
            usd[key] = usd.get(key, 0.0) + spend + float(sel.get("usd") or 0.0)
    return fell_back, usd


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="harness2.cli.report", description=__doc__, formatter_class=C.HelpFormatter
    )
    C.add_cell_args(p.add_argument_group("cell"))
    g = p.add_argument_group("run identity (names the sweeps)")
    g.add_argument("--k", default="1", help="comma list of scales to report, e.g. 1,2,3")
    g.add_argument(
        "--seed",
        type=int,
        default=None,
        help="stream-order seed of the sweep; omit for the seedless namespace",
    )
    C.add_modes_arg(g)
    p.add_argument(
        "--bases",
        action="store_true",
        help="report the reference executions scored by `cli.judge --bases` "
        "(base_judged.json) instead of the selected runs; implied for skillbench",
    )
    args = p.parse_args(argv)
    modes = C.parse_modes(args.modes)
    try:
        ks = sorted({int(s) for s in args.k.split(",") if s.strip()})
    except ValueError:
        raise SystemExit(f"--k takes a comma list of ints, got {args.k!r}")
    if not ks:
        raise SystemExit("--k named no scale")

    adapter = C.make_adapter(
        args.bench, args.subset, args.substrate, arm=args.arm, suite=args.suite
    )
    tag = C.arm_tag(args.bench, args.arm)
    bases = args.bases or args.bench == "skillbench"
    rows, fell_back, usd = [], {}, {}
    for k in ks:
        ns_dir = C.runs_root_for(adapter) / C.summary_namespace(adapter, k, seed=args.seed, tag=tag)
        if bases:
            judged = ns_dir / "base_judged.json"
            got = _load_base_rows(judged, k)
            if not got:
                print(
                    f"  ! no judged reference executions at {judged} — run "
                    f"cli.judge --bases for k={k} first",
                    flush=True,
                )
                continue
            rows.extend(got)
            continue
        judged = ns_dir / "judged.json"
        got = _load_rows(judged)
        if not got:
            print(f"  ! no judged rows at {judged} — run cli.judge for k={k} first", flush=True)
            continue
        rows.extend(r for r in got if r.mode in modes)
        fb, sp = _spend_and_fallbacks(ns_dir, modes, k)
        fell_back.update(fb)
        usd.update(sp)
    if not rows:
        raise SystemExit("nothing to report")

    title = (
        f"{adapter.spec.name}-{adapter.substrate.name}"
        + (f" arm={tag}" if tag else "")
        + (f" seed={args.seed}" if args.seed is not None else "")
        + (
            " reference executions (bases)"
            if bases
            else f" modes={','.join(C.mode_label(m) for m in modes)}"
        )
    )
    summaries = R.summarise(rows, fell_back=fell_back, usd=usd)
    print(R.render(summaries, title=title, bench=adapter.spec.name))
    spend = sum(s.usd for s in summaries)
    if spend:
        print(
            f"\n  improver (editors, composer, selection, reflection) spend across reported modes: "
            f"${spend:.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
