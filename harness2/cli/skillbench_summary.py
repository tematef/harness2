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

"""One-page skillbench Summary, per-skill recommendation rule, and `history.jsonl` trend.

Implements:
- `recommend(row) -> (label, reason)` with the 4-step precedence in Solution Overview §5:
  1. `needs harder tasks` — > 50 % of the skill's tasks at the ceiling for this executor;
  2. `consider removing` — verdict `hurts`, or `inconclusive` with tokens p50 >= +15 % and
     wall p50 >= +15 % and no verify gain;
  3. `fix` — `inconclusive` with invocation < 80 % or `t_skill` p90 > 45 s, or the improver
     has an accepted candidate;
  4. `keep` — `helps`, or `inconclusive` with >= 15 % tokens or time saving, or none of the
     above.
- `build_summary(report, history=...)` — constructs the schema-v2 `summary[]` list (one row
  per skill and per family in suite mode) and annotates per-model rows with `recommendation`
  and `reason`.
- `append_history` / `read_history` / `render_history_markdown` on `history.jsonl` next to
  `suite.toml`, idempotent per `(date, suite_hash, executor, skill)`.
- `sparkline_series` and `render_sparkline_svg` — dependency-free inline `<svg>` polylines
  for `lift` and `tokens_delta_pct` with vertical tick markers where `suite_hash` changed.
- `render_summary_markdown`, `render_summary_section_html`, `render_summary_html` and
  `write_summary` (`--summary-only`).
"""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import config
from ..benchmarks.skillbench_suite import SuiteConfig, active_suite

REC_NEEDS_HARDER_TASKS = "needs harder tasks"
REC_CONSIDER_REMOVING = "consider removing"
REC_FIX = "fix"
REC_KEEP = "keep"
RECOMMENDATION_LABELS = (
    REC_NEEDS_HARDER_TASKS,
    REC_CONSIDER_REMOVING,
    REC_FIX,
    REC_KEEP,
)

CEILING_SHARE_THRESHOLD = 0.50
OVERHEAD_PCT_THRESHOLD = 15.0
SAVING_PCT_THRESHOLD = -15.0
MIN_INVOCATION_RATE = 0.80
MAX_T_SKILL_P90_S = 45.0

HISTORY_FILENAME = "history.jsonl"

RECOMMENDATION_RULE_NOTE = (
    "Recommendation rule (precedence order, first match wins): "
    "(1) `needs harder tasks` — > 50% of the skill's tasks are at the ceiling for this executor; "
    "(2) `consider removing` — verdict is `hurts`, or `inconclusive` with tokens p50 >= +15% "
    "and wall p50 >= +15% and no verify gain; "
    "(3) `fix` — `inconclusive` with invocation < 80% or `t_skill` p90 > 45s, or the improver "
    "has an accepted candidate; "
    "(4) `keep` — verdict is `helps`, or `inconclusive` with >= 15% tokens or time saving, "
    "or none of the above."
)


class SparklineSeries(dict):
    """Dictionary of sparkline series data that also supports attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _opt_float(val: Any) -> float | None:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def _opt_int(val: Any) -> int | None:
    if isinstance(val, int) and not isinstance(val, bool):
        return int(val)
    return None


def _delta_pct(treatment: float | None, control: float | None) -> float | None:
    if (
        treatment is None
        or control is None
        or isinstance(treatment, bool)
        or isinstance(control, bool)
        or control <= 0
    ):
        return None
    return (float(treatment) - float(control)) / float(control) * 100.0


def _extract_ceiling(row: Mapping[str, Any]) -> tuple[bool, int | None, int | None, float | None]:
    ceiling = _opt_int(row.get("ceiling_tasks"))
    total = _opt_int(row.get("total_tasks"))
    if total is None:
        total = _opt_int(row.get("n_tasks"))
    share = _opt_float(row.get("ceiling_share"))
    if share is not None and share > 1.0:
        share = share / 100.0
    if ceiling is not None and total is not None and total > 0:
        share = ceiling / total
    explicit_flag = row.get("needs_harder_tasks")
    if isinstance(explicit_flag, bool):
        is_ceiling = explicit_flag or (share is not None and share > CEILING_SHARE_THRESHOLD)
    else:
        is_ceiling = share is not None and share > CEILING_SHARE_THRESHOLD
    return is_ceiling, ceiling, total, share


def _extract_tokens_delta_pct(row: Mapping[str, Any]) -> float | None:
    direct = _opt_float(row.get("tokens_delta_pct", row.get("tokens_delta")))
    if direct is not None:
        return direct
    saving = _opt_float(row.get("tokens_saving_pct"))
    if saving is not None:
        return -saving
    tok_w = _opt_float(
        row.get(
            "tokens_with_median",
            row.get("tokens_suite_median", row.get("tokens_p50_with")),
        )
    )
    tok_wo = _opt_float(
        row.get(
            "tokens_without_median",
            row.get("tokens_baseline_median", row.get("tokens_p50_without")),
        )
    )
    return _delta_pct(tok_w, tok_wo)


def _extract_wall_delta_pct(row: Mapping[str, Any]) -> float | None:
    direct = _opt_float(
        row.get("wall_delta_pct", row.get("time_delta_pct", row.get("wall_delta")))
    )
    if direct is not None:
        return direct
    saving = _opt_float(row.get("time_saving_pct", row.get("wall_saving_pct")))
    if saving is not None:
        return -saving
    wall_w = _opt_float(
        row.get(
            "t_total_with_median",
            row.get(
                "t_total_suite_median",
                row.get("wall_with_median", row.get("wall_p50_with")),
            ),
        )
    )
    wall_wo = _opt_float(
        row.get(
            "t_total_without_median",
            row.get(
                "t_total_baseline_median",
                row.get("wall_without_median", row.get("wall_p50_without")),
            ),
        )
    )
    return _delta_pct(wall_w, wall_wo)


def _has_verify_gain(row: Mapping[str, Any]) -> bool:
    if "verify_gain" in row and row["verify_gain"] is not None:
        vg = row["verify_gain"]
        if isinstance(vg, bool):
            return vg
        if isinstance(vg, (int, float)):
            return float(vg) > 1e-9
    vd = _opt_float(row.get("verify_delta"))
    if vd is not None:
        return vd > 1e-9

    vw = _opt_float(row.get("verify_with_rate", row.get("verify_suite_rate")))
    vu = _opt_float(row.get("verify_without_rate", row.get("verify_baseline_rate")))
    if vw is None:
        vw_p = _opt_int(row.get("verify_with_passed", row.get("verify_suite_passed")))
        vw_t = _opt_int(row.get("verify_with_total", row.get("verify_suite_total")))
        if vw_p is not None and vw_t is not None and vw_t > 0:
            vw = vw_p / vw_t
    if vu is None:
        vu_p = _opt_int(row.get("verify_without_passed", row.get("verify_baseline_passed")))
        vu_t = _opt_int(row.get("verify_without_total", row.get("verify_baseline_total")))
        if vu_p is not None and vu_t is not None and vu_t > 0:
            vu = vu_p / vu_t
    if vw is not None and vu is not None:
        return vw > vu + 1e-9
    return False


def _extract_invocation_rate(row: Mapping[str, Any]) -> float | None:
    raw = _opt_float(
        row.get(
            "invocation_rate",
            row.get("invocation", row.get("inv_rate", row.get("invocation_pct"))),
        )
    )
    if raw is not None:
        return raw / 100.0 if raw > 1.0 else raw
    invoked = _opt_int(row.get("invoked"))
    valid_with = _opt_int(row.get("valid_with_reps", row.get("valid_suite_reps")))
    if invoked is not None and valid_with is not None and valid_with > 0:
        return invoked / valid_with
    return None


def _has_accepted_candidate(row: Mapping[str, Any]) -> bool:
    for key in (
        "accepted_candidate",
        "improver_accepted",
        "has_accepted_candidate",
        "candidate_accepted",
    ):
        val = row.get(key)
        if isinstance(val, bool) and val:
            return True
        if isinstance(val, str) and val.strip().lower() in ("true", "1", "yes", "accepted"):
            return True
        if isinstance(val, Mapping) and val.get("accepted") is True:
            return True
    return False


def recommend(row: Mapping[str, Any]) -> tuple[str, str]:
    """Compute `(recommendation_label, reason)` for a skill or family row.

    Precedence (first match wins):
    1. `needs harder tasks` — > 50 % of tasks at the ceiling for this executor;
    2. `consider removing` — verdict `hurts`, or `inconclusive` with tokens p50 >= +15 %
       and wall p50 >= +15 % and no verify gain;
    3. `fix` — `inconclusive` with invocation < 80 % or `t_skill` p90 > 45 s, or the
       improver has an accepted candidate;
    4. `keep` — `helps`, or `inconclusive` with >= 15 % tokens or time saving, or none of
       the above.
    """
    if not isinstance(row, Mapping):
        raise TypeError(f"recommend expects a Mapping row, got {type(row)!r}")

    # Rule 1: needs harder tasks (> 50 % of tasks at ceiling).
    is_ceiling, ceiling, total, share = _extract_ceiling(row)
    if is_ceiling:
        if ceiling is not None and total is not None and total > 0:
            pct = (ceiling / total) * 100.0
            return (
                REC_NEEDS_HARDER_TASKS,
                f"rule 1 ({REC_NEEDS_HARDER_TASKS}): {ceiling}/{total} tasks "
                f"({pct:.0f}%) at ceiling (> 50%)",
            )
        if share is not None:
            return (
                REC_NEEDS_HARDER_TASKS,
                f"rule 1 ({REC_NEEDS_HARDER_TASKS}): {share * 100:.0f}% of tasks "
                f"at ceiling (> 50%)",
            )
        return (
            REC_NEEDS_HARDER_TASKS,
            f"rule 1 ({REC_NEEDS_HARDER_TASKS}): > 50% of tasks at ceiling",
        )

    verdict = str(row.get("verdict") or "inconclusive").strip().lower()
    lift = _opt_float(row.get("lift"))
    tok_dp = _extract_tokens_delta_pct(row)
    wall_dp = _extract_wall_delta_pct(row)
    verify_gain = _has_verify_gain(row)

    # Rule 2: consider removing (hurts, or inconclusive with tokens >= +15%, wall >= +15%, no verify gain).
    if verdict == "hurts":
        lift_str = f" (lift {lift:+.2f})" if lift is not None else ""
        return (
            REC_CONSIDER_REMOVING,
            f"rule 2 ({REC_CONSIDER_REMOVING}): verdict is hurts{lift_str}",
        )
    if (
        verdict == "inconclusive"
        and tok_dp is not None
        and tok_dp >= OVERHEAD_PCT_THRESHOLD - 1e-9
        and wall_dp is not None
        and wall_dp >= OVERHEAD_PCT_THRESHOLD - 1e-9
        and not verify_gain
    ):
        return (
            REC_CONSIDER_REMOVING,
            f"rule 2 ({REC_CONSIDER_REMOVING}): inconclusive with tokens p50 "
            f"{tok_dp:+.0f}% (>= +15%), wall p50 {wall_dp:+.0f}% (>= +15%) and no verify gain",
        )

    # Rule 3: fix (inconclusive with invocation < 80% or t_skill p90 > 45s, or accepted candidate).
    inv_rate = _extract_invocation_rate(row)
    t_p90 = _opt_float(row.get("t_skill_p90", row.get("t_skill_p90_s")))
    accepted = _has_accepted_candidate(row)
    low_inv = (
        verdict == "inconclusive"
        and inv_rate is not None
        and inv_rate < MIN_INVOCATION_RATE - 1e-9
    )
    slow_skill = (
        verdict == "inconclusive"
        and t_p90 is not None
        and t_p90 > MAX_T_SKILL_P90_S + 1e-9
    )
    if low_inv or slow_skill or accepted:
        reasons: list[str] = []
        if low_inv and inv_rate is not None:
            reasons.append(f"invocation {inv_rate * 100:.0f}% < 80%")
        if slow_skill and t_p90 is not None:
            reasons.append(f"t_skill p90 {t_p90:.1f}s > 45s")
        if accepted:
            reasons.append("improver has an accepted candidate")
        prefix = "inconclusive with " if (low_inv or slow_skill) else ""
        return (
            REC_FIX,
            f"rule 3 ({REC_FIX}): {prefix}{' and '.join(reasons)}",
        )

    # Rule 4: keep (helps, or inconclusive with >= 15% tokens or time saving, or none of the above).
    if verdict == "helps":
        lift_str = f" (lift {lift:+.2f})" if lift is not None else ""
        return (
            REC_KEEP,
            f"rule 4 ({REC_KEEP}): verdict is helps{lift_str}",
        )
    tok_saved = tok_dp is not None and tok_dp <= SAVING_PCT_THRESHOLD + 1e-9
    wall_saved = wall_dp is not None and wall_dp <= SAVING_PCT_THRESHOLD + 1e-9
    if verdict == "inconclusive" and (tok_saved or wall_saved):
        savings: list[str] = []
        if tok_saved and tok_dp is not None:
            savings.append(f"tokens p50 {tok_dp:+.0f}% (<= -15%)")
        if wall_saved and wall_dp is not None:
            savings.append(f"wall p50 {wall_dp:+.0f}% (<= -15%)")
        return (
            REC_KEEP,
            f"rule 4 ({REC_KEEP}): inconclusive with >= 15% efficiency saving "
            f"({', '.join(savings)})",
        )
    return (
        REC_KEEP,
        f"rule 4 ({REC_KEEP}): none of the above (verdict {verdict})",
    )


# --------------------------------------------------------------------------------------------
# History (history.jsonl next to suite.toml) and sparklines
# --------------------------------------------------------------------------------------------


def _history_key(rec: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Idempotence key `(date, suite_hash, executor, skill)` for a `history.jsonl` record."""
    date = str(rec.get("date") or "")
    suite_hash = str(rec.get("suite_hash") or "")
    executor = str(rec.get("executor") or rec.get("model") or "")
    target = str(rec.get("skill") or rec.get("family") or "")
    return (date, suite_hash, executor, target)


def default_history_path(
    suite: SuiteConfig | Path | str | None = None,
    *,
    fallback_dir: Path | None = None,
) -> Path:
    """Resolve `history.jsonl` next to `suite.toml` (or from an explicit file/directory path)."""
    if isinstance(suite, SuiteConfig):
        if suite.path is not None:
            return suite.path.expanduser().resolve().parent / HISTORY_FILENAME
    elif isinstance(suite, (str, Path)) and str(suite).strip():
        p = Path(suite).expanduser()
        if p.suffix == ".jsonl":
            return p.resolve()
        if p.suffix == ".toml" or p.is_file():
            return p.resolve().parent / HISTORY_FILENAME
        if p.is_dir():
            return p.resolve() / HISTORY_FILENAME
        return p.resolve()
    else:
        try:
            active = active_suite(quiet=True)
            if active.path is not None:
                return active.path.expanduser().resolve().parent / HISTORY_FILENAME
        except (config.ConfigError, OSError):
            pass
    if fallback_dir is not None:
        return Path(fallback_dir).expanduser().resolve() / HISTORY_FILENAME
    raise ValueError(
        "cannot locate history.jsonl: pass --suite <suite.toml> or --history <path>"
    )


def read_history(history_path: Path | str | None) -> list[dict[str, Any]]:
    """Read `history.jsonl` records in file order; returns `[]` when the file is absent."""
    if history_path is None:
        return []
    path = Path(history_path).expanduser()
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _normalize_history_record(
    rec: Mapping[str, Any],
    *,
    default_date: str | None = None,
    default_suite: str | None = None,
    default_suite_hash: str | None = None,
) -> dict[str, Any]:
    date = str(rec.get("date") or default_date or dt. datetime.now(dt.timezone.utc).strftime("%Y-%m-%d"))
    suite_name = _str_or_none(rec.get("suite")) or default_suite
    suite_hash = str(rec.get("suite_hash") if rec.get("suite_hash") is not None else (default_suite_hash or ""))
    executor = str(rec.get("executor") or rec.get("model") or "executor")
    kind = str(rec.get("kind") or ("suite" if rec.get("family") and not rec.get("skill") else "skill"))
    skill = _str_or_none(rec.get("skill"))
    family = _str_or_none(rec.get("family"))
    target = skill or family or ""
    rec_label = _str_or_none(rec.get("recommendation"))
    rec_reason = _str_or_none(rec.get("reason"))
    if not rec_label or not rec_reason:
        rec_label, rec_reason = recommend(rec)
    tok_dp = _extract_tokens_delta_pct(rec)
    wall_dp = _extract_wall_delta_pct(rec)
    tok_w = _opt_float(rec.get("tokens_with_median", rec.get("tokens_suite_median")))
    tok_wo = _opt_float(rec.get("tokens_without_median", rec.get("tokens_baseline_median")))
    wall_w = _opt_float(rec.get("t_total_with_median", rec.get("t_total_suite_median")))
    wall_wo = _opt_float(rec.get("t_total_without_median", rec.get("t_total_baseline_median")))
    ci_low = _opt_float(rec.get("ci_low"))
    ci_high = _opt_float(rec.get("ci_high"))
    ci = rec.get("ci")
    if not isinstance(ci, list) and ci_low is not None and ci_high is not None:
        ci = [ci_low, ci_high]
    elif isinstance(ci, list) and len(ci) == 2:
        ci_low = _opt_float(ci[0])
        ci_high = _opt_float(ci[1])
    else:
        ci = None

    out: dict[str, Any] = {
        "date": date,
        "suite": suite_name,
        "suite_hash": suite_hash,
        "executor": executor,
        "kind": kind,
        "skill": target,
        "family": family if kind == "suite" else None,
        "lift": _opt_float(rec.get("lift")),
        "ci": ci,
        "verdict": str(rec.get("verdict") or "inconclusive"),
        "recommendation": rec_label,
        "reason": rec_reason,
        "tokens_delta_pct": tok_dp,
        "wall_delta_pct": wall_dp,
        "invocation_rate": _extract_invocation_rate(rec),
        "t_skill_p50": _opt_float(rec.get("t_skill_p50")),
        "t_skill_p90": _opt_float(rec.get("t_skill_p90")),
        "ceiling_tasks": _opt_int(rec.get("ceiling_tasks")),
        "total_tasks": _opt_int(rec.get("total_tasks", rec.get("n_tasks"))),
    }
    if tok_w is not None:
        out["tokens_with_median"] = tok_w
        out["median_tokens"] = tok_w
    elif tok_wo is not None:
        out["median_tokens"] = tok_wo
    if tok_wo is not None:
        out["tokens_without_median"] = tok_wo
    if wall_w is not None:
        out["t_total_with_median"] = wall_w
        out["median_wall_s"] = wall_w
    elif wall_wo is not None:
        out["median_wall_s"] = wall_wo
    if wall_wo is not None:
        out["t_total_without_median"] = wall_wo
    return out


def _str_or_none(val: Any) -> str | None:
    return val if isinstance(val, str) and val else None


def history_records_from_report(
    report: Mapping[str, Any],
    *,
    date: str | None = None,
) -> list[dict[str, Any]]:
    """Extract one `history.jsonl` record per `(date, suite_hash, executor, skill | family)`
    from a report dictionary."""
    eff_date = str(
        date
        or report.get("date")
        or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    )
    suite_meta = report.get("suite") if isinstance(report.get("suite"), Mapping) else {}
    suite_name = _str_or_none(suite_meta.get("name"))
    suite_hash = str(suite_meta.get("hash") or "")

    raw_models = report.get("models")
    if isinstance(raw_models, list) and raw_models:
        models = [m for m in raw_models if isinstance(m, Mapping)]
    elif isinstance(report.get("skills"), list) or isinstance(report.get("suites"), list):
        models = [report]
    else:
        models = []

    records: list[dict[str, Any]] = []
    if models:
        for m in models:
            executor = str(
                m.get("executor") or m.get("requested_exec_model") or "executor"
            )
            for srow in m.get("skills") or []:
                if not isinstance(srow, Mapping) or not srow.get("skill"):
                    continue
                records.append(
                    _normalize_history_record(
                        {"executor": executor, "kind": "skill", **srow},
                        default_date=eff_date,
                        default_suite=suite_name,
                        default_suite_hash=suite_hash,
                    )
                )
            for frow in m.get("suites") or []:
                if not isinstance(frow, Mapping) or not (
                    frow.get("family") or frow.get("skill")
                ):
                    continue
                fam_name = str(frow.get("family") or frow.get("skill"))
                records.append(
                    _normalize_history_record(
                        {
                            "executor": executor,
                            "kind": "suite",
                            **frow,
                            "family": fam_name,
                            "skill": fam_name,
                        },
                        default_date=eff_date,
                        default_suite=suite_name,
                        default_suite_hash=suite_hash,
                    )
                )
        return records

    for srow in report.get("summary") or []:
        if isinstance(srow, Mapping) and (srow.get("skill") or srow.get("family")):
            records.append(
                _normalize_history_record(
                    srow,
                    default_date=eff_date,
                    default_suite=suite_name,
                    default_suite_hash=suite_hash,
                )
            )
    return records


def append_history(
    history_path: Path | str,
    report_or_records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    date: str | None = None,
) -> list[dict[str, Any]]:
    """Append records to `history_path` (`history.jsonl`), idempotent per
    `(date, suite_hash, executor, skill)`.

    Accepts either a `build_report` dictionary or an iterable of record mappings (or a single
    record mapping). Returns the complete list of records now in `history_path`.
    """
    path = Path(history_path).expanduser()
    existing = read_history(path)

    if isinstance(report_or_records, Mapping):
        if any(k in report_or_records for k in ("models", "skills", "suites", "summary")):
            incoming = history_records_from_report(report_or_records, date=date)
        else:
            incoming = [_normalize_history_record(report_or_records, default_date=date)]
    else:
        incoming = [
            _normalize_history_record(r, default_date=date)
            for r in report_or_records
            if isinstance(r, Mapping)
        ]

    index_by_key: dict[tuple[str, str, str, str], int] = {}
    merged: list[dict[str, Any]] = []
    for rec in existing:
        norm = _normalize_history_record(rec)
        key = _history_key(norm)
        if key in index_by_key:
            merged[index_by_key[key]] = norm
        else:
            index_by_key[key] = len(merged)
            merged.append(norm)

    for rec in incoming:
        key = _history_key(rec)
        if key in index_by_key:
            merged[index_by_key[key]] = rec
        else:
            index_by_key[key] = len(merged)
            merged.append(rec)

    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in merged)
    path.write_text(body, encoding="utf-8")
    return merged


def render_history_markdown(
    records: Sequence[Mapping[str, Any]] | Path | str,
    *,
    skill: str | None = None,
    executor: str | None = None,
) -> str:
    """Format `history.jsonl` records as a Markdown table (`skillbench-report history show`)."""
    raw_list = (
        read_history(records)
        if isinstance(records, (str, Path))
        else [r for r in records if isinstance(r, Mapping)]
    )
    filtered: list[dict[str, Any]] = []
    for r in raw_list:
        norm = _normalize_history_record(r)
        target = norm.get("skill") or norm.get("family") or ""
        if skill and target != skill:
            continue
        if executor and norm.get("executor") != executor:
            continue
        filtered.append(norm)

    lines = ["# skillbench history", ""]
    if not filtered:
        lines.append("(no history records)")
        return "\n".join(lines) + "\n"

    lines += [
        "| date | suite_hash | executor | skill | lift | tokens Δ % | time Δ % | verdict | recommendation |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    prev_hash_by_pair: dict[tuple[str, str], str] = {}
    for r in filtered:
        ex = str(r.get("executor") or "")
        target = str(r.get("skill") or r.get("family") or "")
        sh = str(r.get("suite_hash") or "")
        pair = (ex, target)
        prev_sh = prev_hash_by_pair.get(pair)
        changed = bool(prev_sh and sh and prev_sh != sh)
        if sh:
            prev_hash_by_pair[pair] = sh
        sh_cell = f"{sh[:12]} Δ" if changed else (sh[:12] if sh else "n/a")
        lift = _opt_float(r.get("lift"))
        tok_dp = _opt_float(r.get("tokens_delta_pct"))
        wall_dp = _opt_float(r.get("wall_delta_pct"))
        lines.append(
            "| "
            + " | ".join(
                (
                    str(r.get("date") or ""),
                    sh_cell,
                    ex,
                    target,
                    "n/a" if lift is None else f"{lift:+.2f}",
                    "n/a" if tok_dp is None else f"{tok_dp:+.0f}%",
                    "n/a" if wall_dp is None else f"{wall_dp:+.0f}%",
                    str(r.get("verdict") or "inconclusive"),
                    str(r.get("recommendation") or REC_KEEP),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def sparkline_series(
    entries: Sequence[Mapping[str, Any]] | Path | str,
    *,
    skill: str | None = None,
    executor: str | None = None,
) -> SparklineSeries:
    """Build the ordered trend series for `(skill, executor)` from `history.jsonl` entries.

    Returns a `SparklineSeries` (dict with attribute access):
    - `points`: `[{index, date, suite_hash, executor, lift, tokens_delta_pct, hash_changed}, ...]`
    - `lifts`: `[float | None, ...]`
    - `tokens_delta_pcts`: `[float | None, ...]`
    - `dates`: `[str, ...]`
    - `suite_hashes`: `[str, ...]`
    - `hash_change_indices`: indices `i >= 1` where `suite_hash` changed from point `i - 1`.
    """
    raw_list = (
        read_history(entries)
        if isinstance(entries, (str, Path))
        else [e for e in entries if isinstance(e, Mapping)]
    )
    matched: list[tuple[str, int, dict[str, Any]]] = []
    seen_keys: dict[tuple[str, str, str, str], int] = {}
    deduped: list[dict[str, Any]] = []
    for e in raw_list:
        target = str(e.get("skill") or e.get("family") or "")
        if skill is not None and target != skill:
            continue
        ex = str(e.get("executor") or e.get("model") or "")
        if executor is not None and ex and ex != executor:
            continue
        date = str(e.get("date") or "")
        sh = str(e.get("suite_hash") or "")
        key = (date, sh, ex, target)
        item = {
            "date": date,
            "suite_hash": sh,
            "executor": ex,
            "skill": target,
            "lift": _opt_float(e.get("lift")),
            "tokens_delta_pct": _extract_tokens_delta_pct(e),
        }
        if key in seen_keys and (date or sh):
            deduped[seen_keys[key]] = item
        else:
            seen_keys[key] = len(deduped)
            deduped.append(item)

    for idx, item in enumerate(deduped):
        matched.append((item["date"], idx, item))
    matched.sort(key=lambda t: (t[0], t[1]))

    points: list[dict[str, Any]] = []
    prev_hash: str | None = None
    for i, (_, _, item) in enumerate(matched):
        cur_hash = item["suite_hash"]
        hash_changed = bool(i > 0 and cur_hash and prev_hash and cur_hash != prev_hash)
        if cur_hash:
            prev_hash = cur_hash
        points.append(
            {
                "index": i,
                "date": item["date"],
                "suite_hash": cur_hash,
                "executor": item["executor"],
                "lift": item["lift"],
                "tokens_delta_pct": item["tokens_delta_pct"],
                "hash_changed": hash_changed,
            }
        )

    return SparklineSeries(
        skill=skill,
        executor=executor,
        points=points,
        dates=[p["date"] for p in points],
        suite_hashes=[p["suite_hash"] for p in points],
        lifts=[p["lift"] for p in points],
        tokens_delta_pcts=[p["tokens_delta_pct"] for p in points],
        hash_change_indices=[p["index"] for p in points if p["hash_changed"]],
    )


def render_sparkline_svg(
    series_or_entries: Mapping[str, Any] | Sequence[Mapping[str, Any]] | Path | str,
    *,
    skill: str | None = None,
    executor: str | None = None,
    width: int = 120,
    height: int = 28,
) -> str:
    """Render an inline `<svg>` sparkline for `lift` and `tokens_delta_pct` with vertical ticks
    where `suite_hash` changed."""
    if isinstance(series_or_entries, Mapping) and "points" in series_or_entries:
        series = series_or_entries
    else:
        series = sparkline_series(
            series_or_entries,  # type: ignore[arg-type]
            skill=skill,
            executor=executor,
        )
    points = list(series.get("points") or [])
    if not points:
        return '<span class="text-[var(--app-muted-foreground,#59636e)] text-xs">n/a</span>'

    pad = 3.0
    n = len(points)
    usable_w = max(1.0, float(width) - 2.0 * pad)
    usable_h = max(1.0, float(height) - 2.0 * pad)

    def x_of(idx: int) -> float:
        if n <= 1:
            return float(width) / 2.0
        return pad + (idx / (n - 1)) * usable_w

    lifts = [p["lift"] for p in points if p.get("lift") is not None]
    l_lo = min([-0.1, *(float(v) for v in lifts)]) if lifts else -1.0
    l_hi = max([0.1, *(float(v) for v in lifts)]) if lifts else 1.0
    if l_hi <= l_lo:
        l_lo, l_hi = l_lo - 0.5, l_hi + 0.5

    def y_lift(val: float) -> float:
        return float(height) - pad - ((val - l_lo) / (l_hi - l_lo)) * usable_h

    toks = [p["tokens_delta_pct"] for p in points if p.get("tokens_delta_pct") is not None]
    t_lo = min([-15.0, *(float(v) for v in toks)]) if toks else -50.0
    t_hi = max([15.0, *(float(v) for v in toks)]) if toks else 50.0
    if t_hi <= t_lo:
        t_lo, t_hi = t_lo - 10.0, t_hi + 10.0

    def y_tok(val: float) -> float:
        return float(height) - pad - ((val - t_lo) / (t_hi - t_lo)) * usable_h

    lift_coords = [
        (x_of(int(p["index"])), y_lift(float(p["lift"])))
        for p in points
        if p.get("lift") is not None
    ]
    tok_coords = [
        (x_of(int(p["index"])), y_tok(float(p["tokens_delta_pct"])))
        for p in points
        if p.get("tokens_delta_pct") is not None
    ]

    def fmt_points(coords: list[tuple[float, float]]) -> str:
        if len(coords) == 1:
            x0, y0 = coords[0]
            return f"{x0:.1f},{y0:.1f} {x0:.1f},{y0:.1f}"
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)

    zero_y = y_lift(0.0)
    parts: list[str] = [
        f'<svg class="inline-block align-middle" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
    ]
    summary_bits: list[str] = []
    for p in points:
        d = p.get("date") or f"#{p['index']}"
        lv = "n/a" if p.get("lift") is None else f"{float(p['lift']):+.2f}"
        tv = "n/a" if p.get("tokens_delta_pct") is None else f"{float(p['tokens_delta_pct']):+.0f}%"
        mark = f" [hash→{p['suite_hash'][:8]}]" if p.get("hash_changed") else ""
        summary_bits.append(f"{d}: lift {lv}, tok {tv}{mark}")
    parts.append(f"<title>{html.escape('; '.join(summary_bits), quote=True)}</title>")

    # Zero baseline axis
    parts.append(
        f'<line x1="{pad:.1f}" y1="{zero_y:.1f}" x2="{width - pad:.1f}" y2="{zero_y:.1f}" '
        'stroke="currentColor" stroke-opacity="0.2" stroke-width="1"/>'
    )

    # Vertical ticks where suite_hash changed
    for p in points:
        if p.get("hash_changed"):
            xi = x_of(int(p["index"]))
            sh = html.escape(str(p.get("suite_hash") or ""), quote=True)
            parts.append(
                f'<line class="sparkline-hash-tick" data-hash-change="{p["index"]}" '
                f'data-suite-hash="{sh}" x1="{xi:.1f}" y1="1" x2="{xi:.1f}" y2="{height - 1}" '
                'stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="2,1"/>'
            )

    if tok_coords:
        parts.append(
            f'<polyline class="sparkline-tokens" data-metric="tokens_delta_pct" '
            f'fill="none" stroke="#6366f1" stroke-width="1.2" stroke-dasharray="2,2" '
            f'points="{fmt_points(tok_coords)}"/>'
        )
    if lift_coords:
        parts.append(
            f'<polyline class="sparkline-lift" data-metric="lift" '
            f'fill="none" stroke="#10b981" stroke-width="1.5" '
            f'points="{fmt_points(lift_coords)}"/>'
        )
        for x, y in lift_coords:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8" fill="#10b981"/>'
            )

    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------------------------
# Summary rows (`report["summary"]`) and renderers (`summary.md`, `summary.html`, `report.html`)
# --------------------------------------------------------------------------------------------


def _pick_representative_row(
    per_exec: Sequence[tuple[str, Mapping[str, Any]]],
    best_model: str | None,
) -> tuple[str, Mapping[str, Any]]:
    if best_model:
        for ex, row in per_exec:
            if ex == best_model:
                return ex, row
    return per_exec[0]


def _build_one_summary_row(
    *,
    kind: str,
    name: str,
    per_exec: Sequence[tuple[str, Mapping[str, Any]]],
    best_model: str | None,
    history_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rep_exec, rep_row = _pick_representative_row(per_exec, best_model)
    rec_label = _str_or_none(rep_row.get("recommendation"))
    rec_reason = _str_or_none(rep_row.get("reason"))
    if not rec_label or not rec_reason:
        rec_label, rec_reason = recommend(rep_row)

    tok_w = _opt_float(rep_row.get("tokens_with_median", rep_row.get("tokens_suite_median")))
    tok_wo = _opt_float(
        rep_row.get("tokens_without_median", rep_row.get("tokens_baseline_median"))
    )
    tok_dp = _extract_tokens_delta_pct(rep_row)

    wall_w = _opt_float(rep_row.get("t_total_with_median", rep_row.get("t_total_suite_median")))
    wall_wo = _opt_float(
        rep_row.get("t_total_without_median", rep_row.get("t_total_baseline_median"))
    )
    wall_dp = _extract_wall_delta_pct(rep_row)

    ci_low = _opt_float(rep_row.get("ci_low"))
    ci_high = _opt_float(rep_row.get("ci_high"))
    ci = rep_row.get("ci")
    if not isinstance(ci, list) and ci_low is not None and ci_high is not None:
        ci = [ci_low, ci_high]
    elif isinstance(ci, list) and len(ci) == 2:
        ci_low = _opt_float(ci[0])
        ci_high = _opt_float(ci[1])
    else:
        ci = None

    invoked = int(rep_row.get("invoked", 0) or 0)
    valid_with = int(
        rep_row.get("valid_with_reps", rep_row.get("valid_suite_reps", 0)) or 0
    )
    inv_rate = _extract_invocation_rate(rep_row)
    t_p50 = _opt_float(rep_row.get("t_skill_p50"))
    t_p90 = _opt_float(rep_row.get("t_skill_p90"))
    ceiling_tasks = _opt_int(rep_row.get("ceiling_tasks"))
    total_tasks = _opt_int(rep_row.get("total_tasks", rep_row.get("n_tasks")))
    unclassified_tasks = int(rep_row.get("unclassified_tasks") or 0)
    is_ceiling, _, _, _ = _extract_ceiling(rep_row)

    by_executor: dict[str, dict[str, Any]] = {}
    for ex, erow in per_exec:
        elabel = _str_or_none(erow.get("recommendation"))
        ereason = _str_or_none(erow.get("reason"))
        if not elabel or not ereason:
            elabel, ereason = recommend(erow)
        eci_low = _opt_float(erow.get("ci_low"))
        eci_high = _opt_float(erow.get("ci_high"))
        eci = erow.get("ci")
        if not isinstance(eci, list) and eci_low is not None and eci_high is not None:
            eci = [eci_low, eci_high]
        by_executor[ex] = {
            "model": ex,
            "executor": ex,
            "verdict": str(erow.get("verdict", "insufficient data")),
            "recommendation": elabel,
            "reason": ereason,
            "lift": _opt_float(erow.get("lift")),
            "ci_low": eci_low,
            "ci_high": eci_high,
            "ci": eci,
            "tokens_delta_pct": _extract_tokens_delta_pct(erow),
            "wall_delta_pct": _extract_wall_delta_pct(erow),
            "invocation_rate": _extract_invocation_rate(erow),
            "t_skill_p50": _opt_float(erow.get("t_skill_p50")),
            "t_skill_p90": _opt_float(erow.get("t_skill_p90")),
            "ceiling_tasks": _opt_int(erow.get("ceiling_tasks")),
            "total_tasks": _opt_int(erow.get("total_tasks", erow.get("n_tasks"))),
            "needs_harder_tasks": _extract_ceiling(erow)[0],
        }

    row_entries = [
        e
        for e in history_entries
        if str(e.get("skill") or e.get("family") or "") == name
    ]
    if not row_entries:
        row_entries = [
            {
                "date": "current",
                "suite_hash": "",
                "executor": rep_exec,
                "skill": name,
                "lift": _opt_float(rep_row.get("lift")),
                "tokens_delta_pct": tok_dp,
            }
        ]
    trend = sparkline_series(row_entries, skill=name, executor=rep_exec)
    if not trend["points"]:
        trend = sparkline_series(row_entries, skill=name)

    out: dict[str, Any] = {
        "kind": kind,
        "skill": name if kind == "skill" else None,
        "family": name if kind == "suite" else None,
        "name": name,
        "model": rep_exec,
        "executor": rep_exec,
        "best_model": best_model,
        "verdict": str(rep_row.get("verdict", "insufficient data")),
        "recommendation": rec_label,
        "reason": rec_reason,
        "lift": _opt_float(rep_row.get("lift")),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci": ci,
        "with_mean": _opt_float(rep_row.get("with_mean", rep_row.get("suite_mean"))),
        "without_mean": _opt_float(rep_row.get("without_mean", rep_row.get("baseline_mean"))),
        "tokens_with_median": tok_w,
        "tokens_without_median": tok_wo,
        "tokens_delta_pct": tok_dp,
        "t_total_with_median": wall_w,
        "t_total_without_median": wall_wo,
        "wall_delta_pct": wall_dp,
        "time_delta_pct": wall_dp,
        "invoked": invoked,
        "valid_with_reps": valid_with,
        "invocation_rate": inv_rate,
        "t_skill_p50": t_p50,
        "t_skill_p90": t_p90,
        "ceiling_tasks": ceiling_tasks,
        "total_tasks": total_tasks,
        "unclassified_tasks": unclassified_tasks,
        "needs_harder_tasks": is_ceiling,
        "verify_with_rate": _opt_float(
            rep_row.get("verify_with_rate", rep_row.get("verify_suite_rate"))
        ),
        "verify_without_rate": _opt_float(
            rep_row.get("verify_without_rate", rep_row.get("verify_baseline_rate"))
        ),
        "by_executor": by_executor,
        "trend": dict(trend),
    }
    return out


def build_summary(
    report: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]] | Path | str | None = None,
) -> list[dict[str, Any]]:
    """Build the schema-v2 `summary[]` list (one row per skill, plus one row per family in
    suite mode) from a report dictionary, and annotate per-executor rows in place when mutable."""
    from .skillbench_report import models_of, pick_best_model

    models = models_of(report)
    if history is None:
        raw_hist = report.get("history")
        if isinstance(raw_hist, (list, tuple)):
            hist_list: list[dict[str, Any]] = [
                dict(x) for x in raw_hist if isinstance(x, Mapping)
            ]
        else:
            try:
                hpath = default_history_path()
                hist_list = read_history(hpath)
            except ValueError:
                hist_list = []
    elif isinstance(history, (str, Path)):
        hist_list = read_history(history)
    else:
        hist_list = [dict(x) for x in history if isinstance(x, Mapping)]

    # Annotate recommendation & reason on every per-model skill and suite row.
    for m in models:
        for srow in m.get("skills") or []:
            if isinstance(srow, dict):
                lbl, rsn = recommend(srow)
                srow["recommendation"] = lbl
                srow["reason"] = rsn
        for frow in m.get("suites") or []:
            if isinstance(frow, dict):
                lbl, rsn = recommend(frow)
                frow["recommendation"] = lbl
                frow["reason"] = rsn

    skills = sorted(
        {
            str(r.get("skill"))
            for m in models
            for r in (m.get("skills") or [])
            if isinstance(r, Mapping) and r.get("skill")
        }
    )
    families = sorted(
        {
            str(r.get("family") or r.get("skill"))
            for m in models
            for r in (m.get("suites") or [])
            if isinstance(r, Mapping) and (r.get("family") or r.get("skill"))
        }
    )

    summary_rows: list[dict[str, Any]] = []
    for skill in skills:
        per_exec: list[tuple[str, Mapping[str, Any]]] = []
        candidates: list[dict[str, Any]] = []
        for m in models:
            ex = str(m.get("executor") or m.get("requested_exec_model") or "executor")
            for r in m.get("skills") or []:
                if isinstance(r, Mapping) and str(r.get("skill")) == skill:
                    per_exec.append((ex, r))
                    candidates.append({"executor": ex, **r})
                    break
        if not per_exec:
            continue
        best = _str_or_none(per_exec[0][1].get("best_model")) or pick_best_model(candidates)
        summary_rows.append(
            _build_one_summary_row(
                kind="skill",
                name=skill,
                per_exec=per_exec,
                best_model=best,
                history_entries=hist_list,
            )
        )

    for family in families:
        per_exec_fam: list[tuple[str, Mapping[str, Any]]] = []
        fam_candidates: list[dict[str, Any]] = []
        for m in models:
            ex = str(m.get("executor") or m.get("requested_exec_model") or "executor")
            for r in m.get("suites") or []:
                if isinstance(r, Mapping) and str(r.get("family") or r.get("skill")) == family:
                    per_exec_fam.append((ex, r))
                    fam_candidates.append({"executor": ex, **r})
                    break
        if not per_exec_fam:
            continue
        best_fam = _str_or_none(per_exec_fam[0][1].get("best_model")) or pick_best_model(
            fam_candidates
        )
        summary_rows.append(
            _build_one_summary_row(
                kind="suite",
                name=family,
                per_exec=per_exec_fam,
                best_model=best_fam,
                history_entries=hist_list,
            )
        )

    return summary_rows


def _fmt_lift_ci(lift: float | None, lo: float | None, hi: float | None) -> str:
    if lift is None or lo is None or hi is None:
        return "n/a"
    return f"{lift:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def _fmt_delta_pct(val: float | None) -> str:
    if val is None:
        return "n/a"
    return f"{val:+.0f}%"


def _fmt_discovery(row: Mapping[str, Any]) -> str:
    rate = _extract_invocation_rate(row)
    p50 = _opt_float(row.get("t_skill_p50"))
    rate_str = "n/a" if rate is None else f"{rate * 100:.0f}%"
    p50_str = "n/a" if p50 is None else f"{p50:.1f}s"
    if rate is None and p50 is None:
        return "n/a"
    return f"{rate_str}, {p50_str}"


def render_summary_markdown(
    report: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]] | Path | str | None = None,
) -> str:
    """Render `summary.md`: the one-page Markdown twin of the Summary page."""
    from .skillbench_report import (
        _fmt_invalid_rate,
        _fmt_signed_frac,
        _fmt_wall_sum,
        build_model_summary,
        judge_pairing_note,
        models_of,
    )

    models = models_of(report)
    raw_summary = report.get("summary")
    if isinstance(raw_summary, list) and raw_summary and history is None:
        summary_rows = [dict(r) for r in raw_summary if isinstance(r, Mapping)]
    else:
        summary_rows = build_summary(report, history=history)

    skill_rows = [
        r for r in summary_rows if r.get("kind") != "suite" and r.get("skill")
    ]
    suite_rows = [
        r for r in summary_rows if r.get("kind") == "suite" or (r.get("family") and not r.get("skill"))
    ]

    suite_meta = report.get("suite") if isinstance(report.get("suite"), Mapping) else {}
    k = report.get("k", 1)
    seed = report.get("seed")
    seed_str = "none" if seed is None else str(seed)
    mode_label = "pilot" if report.get("pilot") else "full"
    manifest_hash = report.get("manifest_hash") or "unverified"

    lines: list[str] = ["# skillbench summary", ""]
    if suite_meta.get("name") or suite_meta.get("hash"):
        lines.append(
            f"- **Suite**: `{suite_meta.get('name') or 'unnamed'}` "
            f"(hash `{str(suite_meta.get('hash') or 'unknown')[:12]}`)"
        )
    lines.append(
        "- **Executors**: "
        + ", ".join(
            f"`{m.get('executor') or m.get('requested_exec_model') or 'executor'}` "
            f"(judge `{m.get('judge') or m.get('requested_judge_model') or 'judge'}`)"
            for m in models
        )
    )
    lines.append(
        f"- **Run scale**: `k={k}`, `seed={seed_str}`, mode `{mode_label}`, "
        f"manifest `{manifest_hash}`"
    )

    if skill_rows or not suite_rows:
        lines += [
            "",
            "## Skill recommendations",
            "",
            "| skill | recommendation | quality Δ [95% CI] | verdict | tokens Δ % "
            "| time Δ % | discovery (inv %, t_skill p50) | best model | reason |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |",
        ]
        for r in skill_rows:
            ci = r.get("ci") if isinstance(r.get("ci"), list) and len(r.get("ci")) == 2 else [r.get("ci_low"), r.get("ci_high")]
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(r.get("skill") or r.get("name") or ""),
                        f"`{r.get('recommendation') or REC_KEEP}`",
                        _fmt_lift_ci(_opt_float(r.get("lift")), _opt_float(ci[0]), _opt_float(ci[1])),
                        str(r.get("verdict") or "insufficient data"),
                        _fmt_delta_pct(_extract_tokens_delta_pct(r)),
                        _fmt_delta_pct(_extract_wall_delta_pct(r)),
                        _fmt_discovery(r),
                        str(r.get("best_model") or r.get("model") or "n/a"),
                        str(r.get("reason") or ""),
                    )
                )
                + " |"
            )

    if suite_rows:
        lines += [
            "",
            "## Suite vs built-in baseline (per family)",
            "",
            "| family | recommendation | quality Δ [95% CI] | verdict | tokens Δ % "
            "| time Δ % | discovery (inv %, t_skill p50) | best model | reason |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |",
        ]
        for r in suite_rows:
            ci = r.get("ci") if isinstance(r.get("ci"), list) and len(r.get("ci")) == 2 else [r.get("ci_low"), r.get("ci_high")]
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(r.get("family") or r.get("name") or ""),
                        f"`{r.get('recommendation') or REC_KEEP}`",
                        _fmt_lift_ci(_opt_float(r.get("lift")), _opt_float(ci[0]), _opt_float(ci[1])),
                        str(r.get("verdict") or "insufficient data"),
                        _fmt_delta_pct(_extract_tokens_delta_pct(r)),
                        _fmt_delta_pct(_extract_wall_delta_pct(r)),
                        _fmt_discovery(r),
                        str(r.get("best_model") or r.get("model") or "n/a"),
                        str(r.get("reason") or ""),
                    )
                )
                + " |"
            )

    raw_ms = report.get("model_summary")
    model_rows = (
        [r for r in raw_ms if isinstance(r, Mapping)]
        if isinstance(raw_ms, list) and raw_ms
        else build_model_summary(report)
    )
    if model_rows:
        lines += [
            "",
            "## Model efficiency",
            "",
            "| executor | judge | skills | mean lift | total tokens | total wall | invalid rate |",
            "| :--- | :--- | ---: | :---: | ---: | ---: | :---: |",
        ]
        for mrow in model_rows:
            tot_tok = mrow.get("total_tokens")
            tok_str = (
                str(int(tot_tok))
                if isinstance(tot_tok, (int, float)) and not isinstance(tot_tok, bool)
                else "n/a"
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(mrow.get("executor", "executor")),
                        str(mrow.get("judge", "judge")),
                        str(mrow.get("n_skills", 0)),
                        _fmt_signed_frac(_opt_float(mrow.get("mean_lift"))),
                        tok_str,
                        _fmt_wall_sum(_opt_float(mrow.get("total_wall_s", mrow.get("total_wall")))),
                        _fmt_invalid_rate(
                            int(mrow.get("invalid_total", 0)),
                            int(mrow.get("total_rollouts", 0)),
                            _opt_float(mrow.get("invalid_rate")),
                        ),
                    )
                )
                + " |"
            )

    rule_note = str(report.get("recommendation_note") or RECOMMENDATION_RULE_NOTE)
    pairing_note = str(report.get("judge_pairing_note") or judge_pairing_note(report))
    lines += [
        "",
        "## Legend",
        "",
        f"- **Recommendation rule**: {rule_note}",
        f"- **Models & judges**: {pairing_note}",
    ]
    return "\n".join(lines) + "\n"


_REC_BADGE_CLASS = {
    REC_KEEP: "bg-emerald-500/15 text-emerald-600 border-emerald-500/30",
    REC_FIX: "bg-amber-500/15 text-amber-600 border-amber-500/30",
    REC_CONSIDER_REMOVING: "bg-rose-500/15 text-rose-600 border-rose-500/30",
    REC_NEEDS_HARDER_TASKS: "bg-amber-500/15 text-amber-600 border-amber-500/30",
}
_REC_BADGE_DEFAULT = "bg-slate-500/15 text-slate-500 border-slate-500/30"


def _rec_badge(label: str) -> str:
    """Recommendation badge (`rounded-md`, never `rounded-full`, so hardness-pill regexes in
    the Skills/Tasks tabs stay isolated)."""
    cls = _REC_BADGE_CLASS.get(label, _REC_BADGE_DEFAULT)
    return (
        f'<span data-recommendation="{html.escape(label, quote=True)}" '
        f'class="inline-block rounded-md border px-2 py-0.5 text-xs font-medium {cls}">'
        f"{html.escape(label, quote=True)}</span>"
    )


def _verdict_badge(verdict: str) -> str:
    cls_map = {
        "helps": "bg-emerald-500/15 text-emerald-600",
        "hurts": "bg-rose-500/15 text-rose-600",
        "inconclusive": "bg-amber-500/15 text-amber-600",
    }
    cls = cls_map.get(verdict, "bg-slate-500/15 text-slate-500")
    return (
        f'<span class="inline-block rounded-md px-2 py-0.5 text-xs font-medium {cls}">'
        f"{html.escape(verdict, quote=True)}</span>"
    )


def _summary_rows_table_html(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_header: str,
    target_key: str,
) -> str:
    from .skillbench_html import BORDER, CARD, FG, MUTED, MUTED_BG, _e

    headers = [
        (target_header, "text-left"),
        ("Recommendation", "text-center"),
        ("Quality Δ [95% CI]", "text-center"),
        ("Tokens Δ %", "text-center"),
        ("Time Δ %", "text-center"),
        ("Discovery (inv %, t_skill p50)", "text-center"),
        ("Best model", "text-left"),
        ("Trend (lift / tok Δ)", "text-center"),
        ("Reason", "text-left"),
    ]
    th_cells = "".join(
        f'<th class="px-3 py-2 text-xs font-medium uppercase tracking-wide {MUTED} {align}">'
        f"{_e(h)}</th>"
        for h, align in headers
    )
    body_rows: list[str] = []
    for r in rows:
        target_name = str(r.get(target_key) or r.get("name") or "")
        rec_label = str(r.get("recommendation") or REC_KEEP)
        verdict = str(r.get("verdict") or "insufficient data")
        lift = _opt_float(r.get("lift"))
        ci = (
            r.get("ci")
            if isinstance(r.get("ci"), list) and len(r.get("ci")) == 2
            else [r.get("ci_low"), r.get("ci_high")]
        )
        lift_ci_str = _fmt_lift_ci(lift, _opt_float(ci[0]), _opt_float(ci[1]))
        tok_dp = _extract_tokens_delta_pct(r)
        wall_dp = _extract_wall_delta_pct(r)
        disc_str = _fmt_discovery(r)
        best_model = str(r.get("best_model") or r.get("model") or "n/a")
        reason = str(r.get("reason") or "")
        trend_series = r.get("trend")
        if isinstance(trend_series, Mapping) and "points" in trend_series:
            svg = render_sparkline_svg(trend_series)
        else:
            svg = render_sparkline_svg(
                [
                    {
                        "date": "current",
                        "suite_hash": "",
                        "executor": best_model,
                        "skill": target_name,
                        "lift": lift,
                        "tokens_delta_pct": tok_dp,
                    }
                ],
                skill=target_name,
            )

        body_rows.append(
            f'<tr class="border-t {BORDER}" data-summary-row="{_e(target_name)}">'
            f'<td class="px-3 py-2.5 text-sm font-medium {FG}"><code>{_e(target_name)}</code></td>'
            f'<td class="px-3 py-2.5 text-sm text-center">{_rec_badge(rec_label)}</td>'
            f'<td class="px-3 py-2.5 text-sm text-center font-mono">{_e(lift_ci_str)} '
            f"{_verdict_badge(verdict)}</td>"
            f'<td class="px-3 py-2.5 text-sm text-center font-mono">{_e(_fmt_delta_pct(tok_dp))}</td>'
            f'<td class="px-3 py-2.5 text-sm text-center font-mono">{_e(_fmt_delta_pct(wall_dp))}</td>'
            f'<td class="px-3 py-2.5 text-sm text-center font-mono">{_e(disc_str)}</td>'
            f'<td class="px-3 py-2.5 text-sm {FG}"><code>{_e(best_model)}</code></td>'
            f'<td class="px-3 py-2.5 text-sm text-center">{svg}</td>'
            f'<td class="px-3 py-2.5 text-xs {MUTED}">{_e(reason)}</td>'
            "</tr>"
        )
    if not body_rows:
        body_rows.append(
            f'<tr><td colspan="{len(headers)}" class="px-3 py-4 text-sm {MUTED}">'
            "No summary rows.</td></tr>"
        )
    return (
        f'<div class="overflow-x-auto rounded-lg border {BORDER} {CARD}">'
        '<table class="w-full border-collapse">'
        f'<thead class="{MUTED_BG}"><tr>{th_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table></div>'
    )


def render_summary_section_html(
    report: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]] | Path | str | None = None,
) -> str:
    """Render the one-page Summary HTML section (used as the first tab in `report.html` and
    as the body of `summary.html`)."""
    from .skillbench_html import BORDER, CARD, FG, MUTED, MUTED_BG, _e
    from .skillbench_report import (
        _fmt_invalid_rate,
        _fmt_signed_frac,
        _fmt_wall_sum,
        build_model_summary,
        judge_pairing_note,
    )

    raw_summary = report.get("summary")
    if isinstance(raw_summary, list) and raw_summary and history is None:
        summary_rows = [dict(r) for r in raw_summary if isinstance(r, Mapping)]
    else:
        summary_rows = build_summary(report, history=history)

    skill_rows = [
        r for r in summary_rows if r.get("kind") != "suite" and r.get("skill")
    ]
    suite_rows = [
        r
        for r in summary_rows
        if r.get("kind") == "suite" or (r.get("family") and not r.get("skill"))
    ]

    blocks: list[str] = []
    if skill_rows or not suite_rows:
        blocks.append(
            f'<div><h3 class="{FG} text-sm font-semibold mb-2">Per-skill recommendations</h3>'
            + _summary_rows_table_html(
                skill_rows, target_header="Skill", target_key="skill"
            )
            + "</div>"
        )

    if suite_rows:
        blocks.append(
            f'<div><h3 class="{FG} text-sm font-semibold mb-2">'
            "Suite vs built-in baseline (per family)</h3>"
            + _summary_rows_table_html(
                suite_rows, target_header="Family", target_key="family"
            )
            + "</div>"
        )

    raw_ms = report.get("model_summary")
    model_rows = (
        [r for r in raw_ms if isinstance(r, Mapping)]
        if isinstance(raw_ms, list) and raw_ms
        else build_model_summary(report)
    )
    if model_rows:
        m_headers = [
            ("Model", "text-left"),
            ("Judge", "text-left"),
            ("Skills", "text-right"),
            ("Mean lift", "text-center"),
            ("Total tokens", "text-right"),
            ("Total wall", "text-right"),
            ("Invalid rate", "text-center"),
        ]
        m_th = "".join(
            f'<th class="px-3 py-2 text-xs font-medium uppercase tracking-wide {MUTED} {align}">'
            f"{_e(h)}</th>"
            for h, align in m_headers
        )
        m_trs: list[str] = []
        for mrow in model_rows:
            tot_tok = mrow.get("total_tokens")
            tok_str = (
                f"{int(tot_tok):,}"
                if isinstance(tot_tok, (int, float)) and not isinstance(tot_tok, bool)
                else "n/a"
            )
            m_trs.append(
                f'<tr class="border-t {BORDER}">'
                f'<td class="px-3 py-2 text-sm font-medium {FG}"><code>{_e(mrow.get("executor"))}</code></td>'
                f'<td class="px-3 py-2 text-sm {MUTED}"><code>{_e(mrow.get("judge"))}</code></td>'
                f'<td class="px-3 py-2 text-sm text-right">{_e(mrow.get("n_skills", 0))}</td>'
                f'<td class="px-3 py-2 text-sm text-center font-mono">'
                f'{_e(_fmt_signed_frac(_opt_float(mrow.get("mean_lift"))))}</td>'
                f'<td class="px-3 py-2 text-sm text-right font-mono">{_e(tok_str)}</td>'
                f'<td class="px-3 py-2 text-sm text-right font-mono">'
                f'{_e(_fmt_wall_sum(_opt_float(mrow.get("total_wall_s", mrow.get("total_wall")))))}</td>'
                f'<td class="px-3 py-2 text-sm text-center font-mono">'
                f'{_e(_fmt_invalid_rate(int(mrow.get("invalid_total", 0)), int(mrow.get("total_rollouts", 0)), _opt_float(mrow.get("invalid_rate"))))}</td>'
                "</tr>"
            )
        blocks.append(
            f'<div><h3 class="{FG} text-sm font-semibold mb-2">Model efficiency</h3>'
            f'<div class="overflow-x-auto rounded-lg border {BORDER} {CARD}">'
            '<table class="w-full border-collapse">'
            f'<thead class="{MUTED_BG}"><tr>{m_th}</tr></thead>'
            f'<tbody>{"".join(m_trs)}</tbody></table></div></div>'
        )

    rule_note = str(report.get("recommendation_note") or RECOMMENDATION_RULE_NOTE)
    pairing_note = str(report.get("judge_pairing_note") or judge_pairing_note(report))
    legend_html = (
        f'<div class="rounded-lg border {BORDER} {CARD} p-4 text-xs {MUTED} space-y-1.5" '
        'data-summary-legend="true">'
        f'<div class="{FG} font-semibold text-xs">Legend &amp; decision rules</div>'
        f"<div>{_e(rule_note)}</div>"
        f"<div>{_e(pairing_note)}</div>"
        "<div>Sparkline: solid green line = quality lift; dashed indigo line = tokens Δ %; "
        "vertical dashed amber tick = <code>suite_hash</code> change.</div>"
        "</div>"
    )
    blocks.append(legend_html)
    return f'<div class="space-y-4">{"".join(blocks)}</div>'


def render_summary_html(
    report: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]] | Path | str | None = None,
) -> str:
    """Render minimal standalone `summary.html` (`--summary-only`)."""
    from .skillbench_html import (
        _FALLBACK_CSS,
        BG,
        FG,
        MUTED,
        TAILWIND_SRC,
        _e,
        _executor_name,
        _judge_name,
        _kpis,
        _models,
    )

    models = _models(report)
    multi = len(models) > 1
    mode = "pilot" if report.get("pilot") else "full"
    suite = report.get("suite") if isinstance(report.get("suite"), Mapping) else {}
    suite_part = f'suite {suite.get("name")} · ' if suite.get("name") else ""
    if multi:
        models_part = (
            f"{len(models)} executors: "
            + ", ".join(f"{_executor_name(m)} (judge {_judge_name(m)})" for m in models)
        )
    else:
        models_part = (
            f"executor {_executor_name(models[0])} · judge {_judge_name(models[0])}"
        )
    subtitle = (
        f'{suite_part}{models_part} · k={report.get("k")} · {mode} mode · '
        f'manifest {str(report.get("manifest_hash") or "unverified")[:12]}'
    )
    kpis = (
        "".join(
            f'<div><h3 class="{FG} text-sm font-semibold mb-2">executor {_e(_executor_name(m))} '
            f'<span class="{MUTED} font-normal">judge {_e(_judge_name(m))}</span></h3>{_kpis(m)}'
            "</div>"
            for m in models
        )
        if multi
        else _kpis(models[0])
    )
    body = render_summary_section_html(report, history=history)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>skillbench summary</title>\n"
        f'<script src="{TAILWIND_SRC}"></script>\n'
        f"<style>{_FALLBACK_CSS}</style>\n"
        "</head>\n"
        f'<body class="{BG} {FG} font-[var(--vscode-font-family,system-ui)] antialiased p-6">\n'
        '<main class="max-w-7xl mx-auto space-y-5">\n'
        f'<header><h1 class="{FG} text-xl font-semibold">skillbench summary</h1>'
        f'<p class="{MUTED} text-sm">{_e(subtitle)}</p></header>\n'
        f"{kpis}\n"
        f'<section data-panel="summary" class="mt-4">{body}</section>\n'
        "</main>\n"
        "</body>\n</html>\n"
    )


def write_summary(
    report: Mapping[str, Any],
    out_dir: Path | str,
    *,
    history: Sequence[Mapping[str, Any]] | Path | str | None = None,
) -> tuple[Path, Path]:
    """Write only `summary.md` and minimal `summary.html` inside `out_dir` (`--summary-only`)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    md_path = out_path / "summary.md"
    html_path = out_path / "summary.html"
    md_path.write_text(render_summary_markdown(report, history=history), encoding="utf-8")
    html_path.write_text(render_summary_html(report, history=history), encoding="utf-8")
    return (md_path, html_path)

