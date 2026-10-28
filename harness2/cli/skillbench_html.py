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

"""Render the skillbench report dictionary as a self-contained `report.html` dashboard.

One file, no build step: styled with the Tailwind script that AGY / Jetski allowlists for
generative UI, coloured with the host's `--app-*` theme variables (each with an inline fallback,
so the file also reads well in a plain browser), and driven by a few lines of vanilla JS
(tabs and the task picker). Every value is rendered server-side and HTML-escaped; the page
works without JS except for switching tabs. Accepts a schema-v1 dictionary or a schema-v2 one
(`models[]`: one block per executor, each rendered under its own heading inside every tab);
v2 rows add the per-task hardness badge and the per-skill `ceiling/total` column.
Open it in a browser or embed it in AGY chat with
`<agent-embed src="file:///.../report.html" height="800px"></agent-embed>`.
"""

from __future__ import annotations

import html
import statistics as st
from typing import Any, Mapping, Sequence

TAILWIND_SRC = "https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"

# Theme tokens: the host injects `--app-*`; the fallbacks keep a standalone file readable.
BG = "bg-[var(--app-background,#ffffff)]"
CARD = "bg-[var(--app-card,#f6f8fa)]"
FG = "text-[var(--app-foreground,#1f2328)]"
MUTED = "text-[var(--app-muted-foreground,#59636e)]"
BORDER = "border-[var(--app-border,#d1d9e0)]"
MUTED_BG = "bg-[var(--app-muted,#eaeef2)]"

_VERDICT_CLASS = {
    "helps": "bg-emerald-500/15 text-emerald-600",
    "hurts": "bg-rose-500/15 text-rose-600",
    "inconclusive": "bg-amber-500/15 text-amber-600",
}
_VERDICT_DEFAULT = "bg-slate-500/15 text-slate-500"

_HARDNESS_CLASS = {
    "informative": "bg-emerald-500/15 text-emerald-600",
    "ceiling": "bg-amber-500/15 text-amber-600",
    "unclassified": "bg-slate-500/15 text-slate-500",
}
_FLAG_CLASS = "bg-amber-500/15 text-amber-600"
_HARDNESS_LEGEND = (
    "Hardness (per executor, from its control arm alone): <em>informative</em> — the WITHOUT "
    "arm missed at least <code>min_baseline_fail</code> criteria (default 1) in some valid rep; "
    "<em>ceiling</em> — every valid WITHOUT rep passed the whole rubric, so the task cannot show "
    "a lift for this executor; <em>unclassified</em> — fewer than 2 valid WITHOUT reps. Ceiling "
    "tasks are flagged, never dropped: they stay in the lift, CI and verdict, because excluding "
    "them would select tasks on the control arm's luck and bias the lift upward. A skill with "
    "more than 50% of its tasks at the ceiling is marked <em>needs harder tasks</em>."
)


def _e(val: Any) -> str:
    return html.escape("" if val is None else str(val), quote=True)


def _pct(val: float | None) -> str:
    return "n/a" if val is None else f"{val * 100:.0f}%"


def _pp(val: float | None) -> str:
    return "n/a" if val is None else f"{val * 100:+.1f} pp"


def _num(val: float | None, suffix: str = "") -> str:
    if val is None:
        return "n/a"
    return f"{val:,.0f}{suffix}"


def _ratio(passed: Any, total: Any) -> str:
    if not isinstance(total, int) or total <= 0 or not isinstance(passed, int):
        return "n/a"
    return f"{passed}/{total} ({passed / total * 100:.0f}%)"


def _verdict_pill(verdict: str) -> str:
    cls = _VERDICT_CLASS.get(verdict, _VERDICT_DEFAULT)
    return (
        f'<span class="inline-block rounded-full px-2 py-0.5 text-xs font-medium {cls}">'
        f"{_e(verdict)}</span>"
    )


def _hardness_pill(task: Mapping[str, Any]) -> str:
    """Hardness badge of a task row; rows without the field (schema v1) get `n/a`."""
    label = task.get("hardness")
    if not isinstance(label, str) or not label:
        return f'<span class="{MUTED} text-xs">n/a</span>'
    missed = list(task.get("baseline_missed") or [])
    title = (
        f"WITHOUT reps missed {', '.join(str(m) for m in missed)} criteria; "
        f'need >= {task.get("min_baseline_fail", 1)} in some rep'
        if missed
        else f"{len(missed)} valid WITHOUT reps"
    )
    cls = _HARDNESS_CLASS.get(label, _VERDICT_DEFAULT)
    return (
        f'<span class="inline-block rounded-full px-2 py-0.5 text-xs font-medium {cls}" '
        f'title="{_e(title)}">{_e(label)}</span>'
    )


def _ceiling_cell(row: Mapping[str, Any]) -> str:
    """`ceiling/total` of a per-skill row plus the `needs harder tasks` flag."""
    ceiling, total = row.get("ceiling_tasks"), row.get("total_tasks")
    if not isinstance(ceiling, int) or not isinstance(total, int):
        return f'<span class="{MUTED}">n/a</span>'
    out = _e(f"{ceiling}/{total}")
    unclassified = row.get("unclassified_tasks") or 0
    if unclassified:
        out += f'<div class="{MUTED} text-xs">{_e(unclassified)} unclassified</div>'
    if row.get("needs_harder_tasks"):
        out += (
            f'<div><span class="inline-block rounded-full px-2 py-0.5 text-xs font-medium '
            f'{_FLAG_CLASS}">needs harder tasks</span></div>'
        )
    return out


def _ci_bar(lift: float | None, lo: float | None, hi: float | None) -> str:
    """Horizontal CI bar on a fixed [-1, +1] axis with a tick at zero and a dot at the lift."""
    if lift is None or lo is None or hi is None:
        return f'<span class="{MUTED}">n/a</span>'

    def x(v: float) -> float:
        return max(0.0, min(100.0, (v + 1.0) * 50.0))

    left, right, dot = x(lo), x(hi), x(lift)
    colour = "bg-emerald-500" if lo > 0 else ("bg-rose-500" if hi < 0 else "bg-amber-500")
    return (
        f'<div class="relative h-3 w-40 rounded {MUTED_BG}" '
        f'title="lift {_pp(lift)}, 95% CI [{_pp(lo)}, {_pp(hi)}]">'
        f'<div class="absolute top-0 h-3 w-px bg-[var(--app-foreground,#1f2328)] opacity-40" '
        f'style="left:50%"></div>'
        f'<div class="absolute top-1 h-1 rounded {colour} opacity-60" '
        f'style="left:{left:.1f}%;width:{max(right - left, 0.5):.1f}%"></div>'
        f'<div class="absolute top-0.5 h-2 w-2 -ml-1 rounded-full {colour}" '
        f'style="left:{dot:.1f}%"></div></div>'
    )


def _card(title: str, value: str, sub: str) -> str:
    return (
        f'<div class="{CARD} {BORDER} border rounded-xl p-4">'
        f'<div class="{MUTED} text-xs uppercase tracking-wide">{_e(title)}</div>'
        f'<div class="{FG} text-2xl font-semibold mt-1">{_e(value)}</div>'
        f'<div class="{MUTED} text-xs mt-1">{_e(sub)}</div></div>'
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], align: str = "") -> str:
    """`rows` cells are pre-escaped HTML; `align` is one char per column (l / r / c)."""
    def cls(i: int) -> str:
        a = align[i] if i < len(align) else "l"
        return {"r": "text-right", "c": "text-center"}.get(a, "text-left")

    head = "".join(
        f'<th class="px-3 py-2 {cls(i)} font-medium {MUTED} text-xs">{_e(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = "".join(
        f'<tr class="border-t {BORDER}">'
        + "".join(f'<td class="px-3 py-2 {cls(i)} align-middle">{c}</td>' for i, c in enumerate(r))
        + "</tr>"
        for r in rows
    )
    return (
        f'<div class="overflow-x-auto {CARD} {BORDER} border rounded-xl">'
        f'<table class="w-full text-sm"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _kpis(report: Mapping[str, Any]) -> str:
    rows = list(report.get("skills") or [])
    verdicts = [str(r.get("verdict", "insufficient data")) for r in rows]
    counts = ", ".join(
        f"{verdicts.count(v)} {v}"
        for v in ("helps", "hurts", "inconclusive", "insufficient data")
        if verdicts.count(v)
    )
    lifts = [r["lift"] for r in rows if r.get("lift") is not None]
    mean_lift = st.mean(lifts) if lifts else None
    invoked = sum(int(r.get("invoked", 0)) for r in rows)
    valid_with = sum(int(r.get("valid_with_reps", 0)) for r in rows)
    inv = (invoked / valid_with) if valid_with else None
    vw_p = sum(int(r.get("verify_with_passed") or 0) for r in rows)
    vw_t = sum(int(r.get("verify_with_total") or 0) for r in rows)
    vu_p = sum(int(r.get("verify_without_passed") or 0) for r in rows)
    vu_t = sum(int(r.get("verify_without_total") or 0) for r in rows)
    tok_w = [r["tokens_with_median"] for r in rows if r.get("tokens_with_median") is not None]
    tok_u = [
        r["tokens_without_median"] for r in rows if r.get("tokens_without_median") is not None
    ]
    tests_value = (
        f"{_pct(vw_p / vw_t if vw_t else None)} vs {_pct(vu_p / vu_t if vu_t else None)}"
        if (vw_t or vu_t)
        else "n/a"
    )
    tok_value = (
        f"{_num(st.median(tok_w) if tok_w else None)} vs "
        f"{_num(st.median(tok_u) if tok_u else None)}"
    )
    cards = [
        _card("Skills evaluated", str(len(rows)), counts or "no data"),
        _card("Mean rubric lift", _pp(mean_lift), "WITH minus WITHOUT, averaged over skills"),
        _card("Skill invocation", _pct(inv), f"{invoked}/{valid_with} valid WITH reps read SKILL.md"),
        _card("Tests pass", tests_value, f"WITH vs WITHOUT ({vw_t}+{vu_t} checked reps)"),
        _card("Median tokens", tok_value, "WITH vs WITHOUT, per valid rep"),
    ]
    return f'<div class="grid grid-cols-2 md:grid-cols-5 gap-3">{"".join(cards)}</div>'


def _skills_tab(report: Mapping[str, Any]) -> str:
    rows = []
    for r in report.get("skills") or []:
        rows.append(
            [
                f'<span class="font-medium">{_e(r.get("skill"))}</span>',
                _e(r.get("n_tasks")),
                _e(_pct(r.get("with_mean"))),
                _e(_pct(r.get("without_mean"))),
                f'<span class="font-medium">{_e(_pp(r.get("lift")))}</span>'
                f'<div class="{MUTED} text-xs">[{_e(_pp(r.get("ci_low")))}, '
                f'{_e(_pp(r.get("ci_high")))}]</div>',
                _ci_bar(r.get("lift"), r.get("ci_low"), r.get("ci_high")),
                _verdict_pill(str(r.get("verdict", "insufficient data"))),
                _e(f'{r.get("invoked", 0)}/{r.get("valid_with_reps", 0)}'),
                _e(
                    "n/a"
                    if r.get("t_skill_p50") is None
                    else f'{r["t_skill_p50"]:.1f}s / {r.get("t_skill_p90") or 0:.1f}s'
                ),
                _e(_ratio(r.get("verify_with_passed"), r.get("verify_with_total"))),
                _e(_ratio(r.get("verify_without_passed"), r.get("verify_without_total"))),
                _e(
                    f'{_num(r.get("tokens_with_median"))} / '
                    f'{_num(r.get("tokens_without_median"))}'
                ),
                _e(r.get("invalid_total", 0)),
                _ceiling_cell(r),
            ]
        )
    table = _table(
        [
            "skill",
            "tasks",
            "WITH",
            "WITHOUT",
            "lift [95% CI]",
            "CI on [-100, +100] pp",
            "verdict",
            "invoked",
            "t_skill p50/p90",
            "tests WITH",
            "tests WITHOUT",
            "tokens p50 W/WO",
            "invalid",
            "ceiling/total",
        ],
        rows,
        "lrrrlcccccccrc",
    )
    note = (
        f'<p class="{MUTED} text-xs mt-3">WITH / WITHOUT: mean share of rubric criteria passed. '
        "Lift: per-task paired difference, 95% CI from a two-level bootstrap (tasks, then reps). "
        "Tests: the task's deterministic <code>verify</code> command, original tests restored. "
        "Verdict <em>helps</em> / <em>hurts</em> only when the whole CI is above / below 0. "
        "Ceiling/total: tasks whose WITHOUT arm already passed every criterion (see the "
        "Tasks tab); they stay in the lift.</p>"
    )
    return table + note


def _tasks_tab(report: Mapping[str, Any]) -> str:
    rows = []
    crit_blocks = []
    for r in report.get("skills") or []:
        for tid, t in sorted((r.get("tasks") or {}).items()):
            rows.append(
                [
                    _e(r.get("skill")),
                    f'<span class="font-medium">{_e(tid.split("/", 1)[-1])}</span>',
                    _e(_pct(t.get("with_mean"))),
                    _e(_pct(t.get("without_mean"))),
                    f'<span class="font-medium">{_e(_pp(t.get("delta")))}</span>',
                    _hardness_pill(t),
                    _e(f'{len(t.get("with_fracs") or [])} / {len(t.get("without_fracs") or [])}'),
                    _e(_ratio(*(t.get("verify_with") or [0, 0]))),
                    _e(_ratio(*(t.get("verify_without") or [0, 0]))),
                    _e(
                        f'{_num(t.get("diff_lines_with_median"))} / '
                        f'{_num(t.get("diff_lines_without_median"))}'
                    ),
                ]
            )
            cw = t.get("criteria_with") or {}
            cu = t.get("criteria_without") or {}
            ids = list(dict.fromkeys(list(cw) + list(cu)))
            if ids:
                crows = []
                for cid in ids:
                    w, u = cw.get(cid), cu.get(cid)
                    d = (w - u) if (w is not None and u is not None) else None
                    crows.append([_e(cid), _e(_pct(w)), _e(_pct(u)), _e(_pp(d))])
                crit_blocks.append(
                    f'<div><h4 class="{FG} text-sm font-medium mb-2">{_e(tid)}</h4>'
                    + _table(["criterion", "WITH", "WITHOUT", "Δ"], crows, "lrrr")
                    + "</div>"
                )
    table = _table(
        [
            "skill",
            "task",
            "WITH",
            "WITHOUT",
            "Δ",
            "hardness",
            "graded reps W/WO",
            "tests WITH",
            "tests WITHOUT",
            "diff lines p50 W/WO",
        ],
        rows,
        "llrrrccccc",
    )
    legend = f'<p class="{MUTED} text-xs mt-3">{_HARDNESS_LEGEND}</p>'
    crit = (
        f'<h3 class="{FG} text-base font-semibold mt-6 mb-3">Criteria pass rate per arm</h3>'
        f'<div class="grid md:grid-cols-2 gap-4">{"".join(crit_blocks)}</div>'
        if crit_blocks
        else ""
    )
    return table + legend + crit


def _rep_panel(rep: Mapping[str, Any], idx: int, total: int) -> str:
    diff = rep.get("diff") or {}
    facts = [
        ("score", _pct(rep.get("frac"))),
        (
            "skill read",
            (
                f'yes, {rep["t_skill_s"]:.1f}s'
                if rep.get("invoked") and rep.get("t_skill_s") is not None
                else ("yes" if rep.get("invoked") else "no")
            ),
        ),
        (
            "tests",
            "n/a"
            if rep.get("verify_passed") is None
            else ("pass" if rep["verify_passed"] else "fail"),
        ),
        (
            "diff",
            f'{diff.get("files", 0)} files, +{diff.get("added", 0)} / -{diff.get("removed", 0)}'
            if diff
            else "n/a",
        ),
        ("tokens", _num(rep.get("total_tokens"))),
        ("wall", _num(rep.get("t_total_s"), "s")),
    ]
    if not rep.get("valid"):
        facts.insert(0, ("invalid", str(rep.get("invalid_reason") or "invalid rollout")))
    fact_html = "".join(
        f'<span class="{MUTED_BG} rounded px-2 py-0.5 text-xs">{_e(k)}: '
        f'<span class="font-medium">{_e(v)}</span></span>'
        for k, v in facts
    )
    crit = "".join(
        f'<li class="text-xs"><span class="{"text-emerald-600" if c.get("pass") else "text-rose-600"}'
        f' font-semibold">{"✓" if c.get("pass") else "✗"}</span> '
        f'<span class="font-medium">{_e(c.get("id"))}</span> '
        f'<span class="{MUTED}">— {_e(c.get("reason"))}</span></li>'
        for c in rep.get("criteria") or []
        if isinstance(c, Mapping)
    )
    excerpt = rep.get("final_excerpt") or "(no final answer)"
    return (
        f'<details class="{BORDER} border rounded-lg p-3" {"open" if idx == 0 else ""}>'
        f'<summary class="cursor-pointer text-sm font-medium">rep {idx + 1} of {total} '
        f'<span class="{MUTED} text-xs">{_e(rep.get("label"))}</span></summary>'
        f'<div class="flex flex-wrap gap-1.5 mt-2">{fact_html}</div>'
        + (f'<ul class="mt-2 space-y-1">{crit}</ul>' if crit else "")
        + f'<pre class="mt-2 max-h-96 overflow-auto whitespace-pre-wrap text-xs {MUTED_BG} '
        f'rounded p-2">{_e(excerpt)}</pre></details>'
    )


def _side_by_side_tab(report: Mapping[str, Any]) -> str:
    models = _models(report)
    multi = len(models) > 1
    options = []
    panels = []
    n = 0
    for model in models:
        exec_name = _executor_name(model)
        model_options = []
        for r in model.get("skills") or []:
            for tid, t in sorted((r.get("tasks") or {}).items()):
                reps = t.get("reps") or {}
                cols = []
                for arm in ("with", "without"):
                    arm_reps = list(reps.get(arm) or [])
                    arm_reps.sort(key=lambda x: (not x.get("valid"), str(x.get("label"))))
                    inner = "".join(
                        _rep_panel(x, i, len(arm_reps)) for i, x in enumerate(arm_reps)
                    )
                    if not inner:
                        inner = f'<p class="{MUTED} text-xs">no reps</p>'
                    title = "WITH skill" if arm == "with" else "WITHOUT skill"
                    cols.append(
                        f'<div class="space-y-2"><h4 class="{FG} text-sm font-semibold">{title} '
                        f'<span class="{MUTED} font-normal">mean '
                        f'{_e(_pct(t.get(arm + "_mean")))}</span></h4>{inner}</div>'
                    )
                label = f"{exec_name} · {tid}" if multi else tid
                model_options.append(f'<option value="sbs-{n}">{_e(label)}</option>')
                panels.append(
                    f'<div class="sbs-panel grid md:grid-cols-2 gap-4 {"" if n == 0 else "hidden"}" '
                    f'id="sbs-{n}">{"".join(cols)}</div>'
                )
                n += 1
        if multi and model_options:
            options.append(
                f'<optgroup label="{_e(exec_name)}">{"".join(model_options)}</optgroup>'
            )
        else:
            options.extend(model_options)
    if not panels:
        return f'<p class="{MUTED} text-sm">No task data.</p>'
    return (
        f'<label class="{MUTED} text-xs">Task <select id="sbs-select" '
        f'class="{CARD} {BORDER} {FG} border rounded px-2 py-1 text-sm ml-1">'
        f'{"".join(options)}</select></label>'
        f'<p class="{MUTED} text-xs mt-2">Final answers are shown as the agent wrote them; the '
        "judge saw them with skill names and paths redacted, and never saw test results.</p>"
        f'<div class="mt-3">{"".join(panels)}</div>'
    )


def _model_diag_rows(model: Mapping[str, Any]) -> list[tuple[str, str]]:
    noise = model.get("judge_noise_rate")
    noise_str = (
        "n/a (single judgment per rollout)"
        if noise is None
        else f'{noise * 100:.1f}% ({model.get("judge_noise_disagreed", 0)}/'
        f'{model.get("judge_noise_total", 0)} criteria disagreed)'
    )
    invalid = model.get("invalid_by_reason") or {}
    return [
        (
            "Executor model",
            f'requested {model.get("requested_exec_model") or model.get("executor")}, actual '
            f'{model.get("actual_exec_model") or "unverified"}',
        ),
        (
            "Judge model",
            f'requested {model.get("requested_judge_model") or model.get("judge")}, actual '
            f'{model.get("actual_judge_model") or "unverified"}',
        ),
        ("Judge noise", noise_str),
        (
            "Invalid rollouts",
            f'{model.get("invalid_total", 0)}'
            + (
                " (" + ", ".join(f"{k}: {v}" for k, v in sorted(invalid.items())) + ")"
                if invalid
                else ""
            ),
        ),
    ]


def _diagnostics_tab(report: Mapping[str, Any]) -> str:
    models = _models(report)
    multi = len(models) > 1
    suite = report.get("suite") if isinstance(report.get("suite"), Mapping) else {}
    items: list[tuple[str, str]] = []
    if suite.get("name") or suite.get("hash"):
        items.append(
            ("Suite", f'{suite.get("name") or "unnamed"} (hash {suite.get("hash") or "unknown"})')
        )
    items += [
        ("Manifest hash", str(report.get("manifest_hash"))),
        (
            "Run scale",
            f'k={report.get("k")}, seed={report.get("seed")}, '
            f'{"pilot" if report.get("pilot") else "full"} mode',
        ),
        (
            "Bootstrap",
            f'{report.get("bootstrap_draws")} draws, seed {report.get("bootstrap_seed")}',
        ),
        ("Statistical power", str(report.get("power_note") or "")),
    ]
    if multi:
        items.insert(
            0,
            (
                "Executors",
                ", ".join(f"{_executor_name(m)} (judge {_judge_name(m)})" for m in models),
            ),
        )
    else:
        items = _model_diag_rows(models[0]) + items
    rows = [[f'<span class="font-medium">{_e(k)}</span>', _e(v)] for k, v in items]
    out = _table(["item", "value"], rows, "ll")
    if multi:
        for model in models:
            mrows = [
                [f'<span class="font-medium">{_e(k)}</span>', _e(v)]
                for k, v in _model_diag_rows(model)
            ]
            out += _section_title(_executor_name(model)) + _table(["item", "value"], mrows, "ll")
    return out


def _models(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Per-executor blocks (`models[]`); a schema-v1 dictionary is its own single block."""
    models = report.get("models")
    if isinstance(models, list) and models:
        return [m for m in models if isinstance(m, Mapping)]
    return [report]


def _executor_name(model: Mapping[str, Any]) -> str:
    return str(model.get("executor") or model.get("requested_exec_model") or "executor")


def _judge_name(model: Mapping[str, Any]) -> str:
    return str(model.get("judge") or model.get("requested_judge_model") or "judge")


def _section_title(text: str) -> str:
    return f'<h3 class="{FG} text-base font-semibold mt-6 mb-3">executor {_e(text)}</h3>'


def _per_model(report: Mapping[str, Any], render) -> str:
    """Concatenate `render(model)` over the executors, titled per executor when there are
    several."""
    models = _models(report)
    if len(models) == 1:
        return render(models[0])
    return "".join(_section_title(_executor_name(m)) + render(m) for m in models)


_JS = """
document.querySelectorAll('[data-tab]').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var name = btn.getAttribute('data-tab');
    document.querySelectorAll('[data-tab]').forEach(function (b) {
      b.setAttribute('aria-selected', String(b === btn));
      b.classList.toggle('font-semibold', b === btn);
      b.classList.toggle('border-b-2', b === btn);
    });
    document.querySelectorAll('[data-panel]').forEach(function (p) {
      p.classList.toggle('hidden', p.getAttribute('data-panel') !== name);
    });
  });
});
var sel = document.getElementById('sbs-select');
if (sel) {
  sel.addEventListener('change', function () {
    document.querySelectorAll('.sbs-panel').forEach(function (p) {
      p.classList.toggle('hidden', p.id !== sel.value);
    });
  });
}
"""

# Minimal styling for viewers that block the Tailwind script (no colour tokens on :root).
_FALLBACK_CSS = """
.hidden{display:none}
table{border-collapse:collapse}
th,td{padding:4px 8px}
pre{white-space:pre-wrap}
"""


def render_html(report: Mapping[str, Any]) -> str:
    """Self-contained `report.html` for the skillbench report dictionary (schema v1 or v2); a
    v2 report with several executors repeats every tab's content once per executor."""
    models = _models(report)
    multi = len(models) > 1
    tabs = [
        ("skills", "Skills", _per_model(report, _skills_tab)),
        ("tasks", "Tasks & criteria", _per_model(report, _tasks_tab)),
        ("sbs", "Side by side", _side_by_side_tab(report)),
        ("diag", "Diagnostics", _diagnostics_tab(report)),
    ]
    buttons = "".join(
        f'<button type="button" data-tab="{key}" aria-selected="{str(i == 0).lower()}" '
        f'class="px-3 py-2 text-sm {FG} border-[var(--app-primary,#0969da)] '
        f'{"font-semibold border-b-2" if i == 0 else ""}">{_e(label)}</button>'
        for i, (key, label, _) in enumerate(tabs)
    )
    panels = "".join(
        f'<section data-panel="{key}" class="mt-4 {"" if i == 0 else "hidden"}">{body}</section>'
        for i, (key, _, body) in enumerate(tabs)
    )
    mode = "pilot" if report.get("pilot") else "full"
    suite = report.get("suite") if isinstance(report.get("suite"), Mapping) else {}
    suite_part = f'suite {suite.get("name")} · ' if suite.get("name") else ""
    if multi:
        models_part = (
            f'{len(models)} executors: '
            + ", ".join(f"{_executor_name(m)} (judge {_judge_name(m)})" for m in models)
        )
    else:
        models_part = (
            f'executor {_executor_name(models[0])} · judge {_judge_name(models[0])}'
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
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>skillbench report</title>\n"
        f'<script src="{TAILWIND_SRC}"></script>\n'
        f"<style>{_FALLBACK_CSS}</style>\n"
        "</head>\n"
        f'<body class="{BG} {FG} font-[var(--vscode-font-family,system-ui)] antialiased p-6">\n'
        '<main class="max-w-7xl mx-auto space-y-5">\n'
        f'<header><h1 class="{FG} text-xl font-semibold">skillbench report</h1>'
        f'<p class="{MUTED} text-sm">{_e(subtitle)}</p></header>\n'
        f"{kpis}\n"
        f'<nav class="flex gap-1 border-b {BORDER}" role="tablist">{buttons}</nav>\n'
        f"{panels}\n"
        "</main>\n"
        f"<script>{_JS}</script>\n"
        "</body>\n</html>\n"
    )
