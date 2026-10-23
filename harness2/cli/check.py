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

"""Check one installed benchmark before running adaptation and judging."""

from __future__ import annotations

import argparse

from harness2.cli._common import add_cell_args, make_adapter
from harness2.preflight import check


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_cell_args(p)
    p.add_argument(
        "--pilot",
        action="store_true",
        help="skillbench only: restrict task count to pilot tasks (pilot = true)",
    )
    args = p.parse_args(argv)
    arm = args.arm or ("with" if args.bench == "skillbench" else None)
    try:
        adapter_kw = {"arm": arm}
        if args.pilot:
            adapter_kw["pilot"] = True
        adapter = make_adapter(args.bench, args.subset, args.substrate, **adapter_kw)
        errors = check(adapter, judge=True)
    except (OSError, RuntimeError, ValueError) as exc:
        errors = [str(exc)]
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        print(f"Setup incomplete: {len(errors)} issue(s). See docs/BENCHMARKS.md.")
        return 1
    print(
        f"Ready: {adapter.spec.name}/{adapter.substrate.name}; {len(adapter.tasks())} tasks. "
        "Model access is not exercised; the first real run is the first model call."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
