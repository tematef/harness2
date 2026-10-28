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

"""Shared plumbing for the CLIs: adapter construction and the on-disk conventions.

The conventions are defined here once, because run_sweep writes them and
judge/report read them back.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..core import library as LIB
from ..core import schedule as SCHED
from ..core.spec import BenchmarkAdapter

SUITE_ENV = "SKILLBENCH_SUITE"
MATRIX_ENV = "SKILLBENCH_MATRIX"

BENCHES = ("jb", "wb", "lab", "skillbench")
WB_SUBSETS = ("office", "web", "code")
SUBSTRATES = ("oc", "cx", "agy")
# Skill arms; the full label grammar (`baseline`, `suite-<name>`) lives in substrate.agy.
ARMS = ("with", "without")
ARM_LABEL_HELP = "with | without | baseline | suite-<name>"
# Substrates that only some benches support; every other bench rejects them.
SUBSTRATE_BENCHES: dict[str, tuple[str, ...]] = {"agy": ("skillbench",)}

MODES_DEFAULT = "parallel"
MODES_HELP = (
    "parallel | sequential (agg / seq accepted: the on-disk tokens); "
    "comma list, e.g. 'parallel' or 'parallel,sequential'"
)


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Defaults in the help text, module docstring kept verbatim as the description."""


def default_substrate(bench: str) -> str:
    """Default substrate when `--substrate` is omitted: `agy` for skillbench, `oc` otherwise."""
    return "agy" if bench == "skillbench" else "oc"


def arm_tag(bench: str, arm: str | None) -> str:
    """Run tag for `(bench, arm)`: the validated arm label for skillbench, empty for other
    benches."""
    if bench != "skillbench" or not arm:
        return ""
    from ..substrate.agy import arm_kind

    arm_kind(arm)
    return arm


class _BenchAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        setattr(namespace, self.dest, values)
        if getattr(namespace, "substrate", None) is None:
            namespace.substrate = default_substrate(values)


def apply_suite(suite: Path | str | None) -> Path | None:
    """Make `--suite` the active skillbench suite for this process by exporting it as
    `SKILLBENCH_SUITE` (every suite lookup reads that variable, so the substrate, the judge
    and the report all see the same suite). Returns the exported path, or None when the
    flag was not given (the env variable or the env fallback then applies)."""
    if suite is None or str(suite) == "":
        return None
    path = Path(suite).expanduser()
    os.environ[SUITE_ENV] = str(path)
    return path


def parse_suites(raw: object) -> tuple[object, ...]:
    """Comma list or iterable of extra `suite.toml` paths / `SuiteConfig`s (`--suites a.toml,b.toml`)."""
    if raw is None or raw == "":
        return ()
    if isinstance(raw, Path):
        return (raw.expanduser(),)
    if isinstance(raw, str):
        return tuple(Path(s.strip()).expanduser() for s in raw.split(",") if s.strip())
    if isinstance(raw, (list, tuple, set, frozenset)):
        out: list[object] = []
        for item in raw:
            if isinstance(item, str):
                out.extend(Path(s.strip()).expanduser() for s in item.split(",") if s.strip())
            elif isinstance(item, Path):
                out.append(item.expanduser())
            elif item is not None:
                out.append(item)
        return tuple(out)
    return (raw,)


def add_suite_arg(p: argparse._ActionsContainer) -> None:
    """`--suite suite.toml` (skillbench only; alternative: `SKILLBENCH_SUITE`)."""
    p.add_argument(
        "--suite",
        type=Path,
        default=None,
        help="skillbench only: suite.toml describing the plugin suite under test "
        "(default: $SKILLBENCH_SUITE, else the SKILLBENCH_PLUGINS_DIR env fallback)",
    )


def add_suites_arg(p: argparse._ActionsContainer) -> None:
    """`--suites a.toml,b.toml` (skillbench only: further suites for `suite-<name>` arms)."""
    p.add_argument(
        "--suites",
        default="",
        help="skillbench only: comma list of extra suite.toml files for `suite-<name>` "
        "comparisons against `baseline`",
    )


def make_adapter(
    bench: str,
    subset: str | None = None,
    substrate: str | None = None,
    split_file: Path | None = None,
    arm: str | None = None,
    *,
    pilot: bool | None = None,
    suite: Path | str | None = None,
    suites: object = None,
) -> BenchmarkAdapter:
    """The one place a CLI turns (--bench, --subset, --substrate, --arm, --suite) into an
    adapter."""
    if bench not in BENCHES:
        raise ValueError(f"unknown bench {bench!r}; expected one of {BENCHES}")
    if substrate is None:
        substrate = default_substrate(bench)
    if substrate not in SUBSTRATES:
        raise ValueError(f"unknown substrate {substrate!r}; expected one of {SUBSTRATES}")
    if substrate in SUBSTRATE_BENCHES and bench not in SUBSTRATE_BENCHES[substrate]:
        raise ValueError(
            f"--substrate {substrate} is not supported by --bench {bench}; it is reserved for "
            f"the skillbench benchmark"
        )
    extra_suites = parse_suites(suites)
    if bench == "skillbench":
        if subset:
            raise ValueError(f"--subset applies to --bench wb only, got {subset!r} for {bench}")
        if substrate != "agy":
            raise ValueError(f"--bench skillbench requires --substrate agy, got {substrate!r}")
        if not arm:
            raise ValueError(
                f"--bench skillbench requires --arm ({ARM_LABEL_HELP}), got {arm!r}"
            )
        from ..substrate.agy import arm_kind

        try:
            arm_kind(arm)
        except ValueError as e:
            raise ValueError(f"--arm: {e}") from None
        apply_suite(suite)
        from ..benchmarks.skillbench import SkillbenchAdapter

        return SkillbenchAdapter(
            arm,
            substrate=substrate,
            split_file=split_file,
            pilot=pilot,
            suites=extra_suites,
        )
    if arm is not None:
        raise ValueError(f"--arm applies to --bench skillbench only, got {arm!r} for {bench}")
    if pilot:
        raise ValueError(f"--pilot applies to --bench skillbench only, got {bench!r}")
    if suite is not None:
        raise ValueError(f"--suite applies to --bench skillbench only, got {suite!r} for {bench}")
    if extra_suites:
        raise ValueError(
            f"--suites applies to --bench skillbench only, got {suites!r} for {bench}"
        )
    if bench == "wb":
        if subset not in WB_SUBSETS:
            raise ValueError(f"--bench wb needs --subset from {WB_SUBSETS}, got {subset!r} ")
        from ..benchmarks.workbuddy import WbAdapter

        return WbAdapter(subset, substrate=substrate, split_file=split_file)
    if subset:
        raise ValueError(f"--subset applies to --bench wb only, got {subset!r} for {bench}")
    if bench == "jb":
        from ..benchmarks.jobbench import JobBenchAdapter

        return JobBenchAdapter(substrate=substrate, split_file=split_file)
    from ..benchmarks.lab import LabAdapter

    return LabAdapter(substrate=substrate, split_file=split_file)


def _matrix_executor_segment(adapter: BenchmarkAdapter) -> str:
    """Executor path segment when running a skillbench matrix cell (`SKILLBENCH_MATRIX` set),
    else empty so single-model CLI namespaces stay at `<runs_root>/adapt/<ns>`."""
    if getattr(adapter.spec, "name", "") != "skillbench":
        return ""
    if not str(os.environ.get(MATRIX_ENV) or "").strip():
        return ""
    from ..benchmarks.skillbench import _path_segment
    from ..substrate.agy import exec_model

    raw = getattr(adapter, "executor", None) or exec_model()
    return _path_segment(str(raw))


def runs_root_for(adapter: BenchmarkAdapter) -> Path:
    """Scheduler state: summaries, harnesses and selections, not the rollout store."""
    base = adapter.spec.runs_root / "adapt"
    seg = _matrix_executor_segment(adapter)
    return base / seg if seg else base


def library_root_for(adapter: BenchmarkAdapter, k: int, *, seed: int | None, tag: str = "") -> Path:
    """Per-(benchmark, substrate, k, seed, tag) experience-library root (the improver's R).

    Separate seeds and tags keep independent task streams from sharing entries.
    """
    SCHED.validate_tag(tag)
    variant = adapter.substrate.name
    seg = _matrix_executor_segment(adapter)
    if seg:
        variant += f"-{seg}"
    if seed is not None:
        variant += f"-s{int(seed)}"
    if tag:
        variant += f"-{tag}"
    return LIB.root_for(
        adapter.spec.runs_root / "library", benchmark=adapter.spec.name, k=k, variant=variant
    )


def summary_namespace(adapter: BenchmarkAdapter, k: int, *, seed: int | None, tag: str = "") -> str:
    """The namespace whose task dirs hold summary.json for BOTH modes: run_task keeps the
    summary in the parallel mode's namespace (on-disk token "agg") whichever modes ran."""
    return SCHED.ns(adapter.spec.name, adapter.substrate.name, "agg", k, seed=seed, tag=tag)


def add_cell_args(p: argparse._ActionsContainer) -> None:
    """The (benchmark, substrate) cell selectors; `p` may be a parser or an argument group."""
    p.add_argument(
        "--bench",
        required=True,
        choices=BENCHES,
        action=_BenchAction,
        help="jb | wb | lab | skillbench",
    )
    p.add_argument(
        "--subset", default=None, choices=WB_SUBSETS, help="WB only: office | web | code"
    )
    p.add_argument(
        "--substrate",
        default=None,
        choices=SUBSTRATES,
        help="task-executor substrate: oc (opencode) | cx (codex) | agy (AGY / Jetski; "
        "default: agy for skillbench, oc otherwise)",
    )
    p.add_argument(
        "--arm",
        default=None,
        metavar="LABEL",
        help=f"skillbench only (required): {ARM_LABEL_HELP}",
    )
    add_suite_arg(p)
    add_suites_arg(p)


def add_modes_arg(p: argparse._ActionsContainer) -> None:
    p.add_argument("--mode", "--modes", dest="modes", default=MODES_DEFAULT, help=MODES_HELP)


def add_run_identity_args(p: argparse._ActionsContainer) -> None:
    """The run-identity knobs that enter the namespace."""
    p.add_argument(
        "--k", type=int, default=1, help="scale: R=4k rollouts, P=3k improver sessions per task"
    )
    p.add_argument(
        "--seed", type=int, default=None, help="stream-order seed; omit for sorted task order"
    )
    add_modes_arg(p)


def parse_modes(raw: str) -> tuple[str, ...]:
    """'parallel,sequential' (or the on-disk tokens 'agg,seq') -> tokens in MODES order."""
    names = [s.strip() for s in raw.split(",") if s.strip()]
    if not names:
        raise ValueError(
            f"--modes takes a comma list of parallel | sequential (agg | seq), got {raw!r}"
        )
    tokens = set()
    for name in names:
        try:
            tokens.add(SCHED.mode_token(name))
        except ValueError as exc:
            raise ValueError(f"--modes: {exc} (got {raw!r})") from None
    return tuple(t for t in SCHED.MODES if t in tokens)


def mode_label(token: str) -> str:
    """Human name for an on-disk token ('agg' -> 'parallel'); unknown tokens pass through."""
    return SCHED.MODE_NAMES.get(token, token)
