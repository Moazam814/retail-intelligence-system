"""Pure HTML builders for the dashboard (no Streamlit import, so they are unit-testable)."""
from __future__ import annotations

from html import escape
from typing import Callable

import pandas as pd

BADGE_KINDS = {"success", "warning", "danger", "info", "neutral"}


def _attr(text: str) -> str:
    return escape(text or "", quote=True)


def badge(text: str, kind: str = "neutral") -> str:
    kind = kind if kind in BADGE_KINDS else "neutral"
    return f'<span class="ri-badge {kind}">{escape(text)}</span>'


def page_header(title: str, subtitle: str = "", chips: list[str] | None = None) -> str:
    sub = f'<div class="ri-page-sub">{escape(subtitle)}</div>' if subtitle else ""
    chip_html = "".join(f'<span class="ri-chip">{escape(c)}</span>' for c in (chips or []))
    chips_block = f'<div class="ri-chips">{chip_html}</div>' if chip_html else ""
    return f'<div class="ri-page-head"><div class="ri-page-title">{escape(title)}</div>{sub}{chips_block}</div>'


def section_header(title: str, description: str = "") -> str:
    desc = f'<div class="ri-section-desc">{escape(description)}</div>' if description else ""
    return f'<div class="ri-section"><div class="ri-section-title">{escape(title)}</div>{desc}</div>'


def _delta_html(delta, unit: str, label: str) -> str:
    if delta is None:
        return ""
    if abs(delta) < 0.05:
        cls, arrow = "flat", "●"
    else:
        cls, arrow = ("up", "▲") if delta > 0 else ("down", "▼")
    return (f'<div class="ri-delta {cls}">{arrow} {abs(delta):.1f}{escape(unit)} '
            f'<span>{escape(label)}</span></div>')


def kpi_grid(cards: list[dict]) -> str:
    """cards: label, value, caption, help (tooltip), delta (float|None), delta_unit, delta_label."""
    items = []
    for c in cards:
        delta = _delta_html(c.get("delta"), c.get("delta_unit", "%"),
                            c.get("delta_label", "vs previous period"))
        items.append(
            f'<div class="ri-kpi" title="{_attr(c.get("help", ""))}">'
            f'<div class="ri-kpi-label">{escape(c["label"])}</div>'
            f'<div class="ri-kpi-value">{escape(c["value"])}</div>{delta}'
            f'<div class="ri-kpi-caption">{escape(c.get("caption", ""))}</div></div>')
    return '<div class="ri-kpi-grid">' + "".join(items) + "</div>"


def table_html(df: pd.DataFrame, columns: list[dict],
               row_class: Callable[[pd.Series], str] | None = None) -> str:
    """Render a DataFrame as a styled, horizontally scrollable table.

    Each column spec: key, label, align ('left'|'right'), fmt (callable), help (tooltip),
    or html (callable(row) -> trusted HTML string, used for badges and bars).
    """
    head = "".join(
        f'<th class="{c.get("align", "left")}" title="{_attr(c.get("help", ""))}">{escape(c["label"])}</th>'
        for c in columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in columns:
            if "html" in c:
                content = c["html"](row)
            else:
                value = row[c["key"]]
                fmt = c.get("fmt")
                content = escape(fmt(value) if fmt else str(value))
            cells.append(f'<td class="{c.get("align", "left")}">{content}</td>')
        cls = row_class(row) if row_class else ""
        rows.append(f'<tr class="{cls}">' + "".join(cells) + "</tr>")
    return (f'<div class="ri-table-wrap"><table class="ri-table"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def cover_bar(days, max_days: float, kind: str) -> str:
    """Inline progress bar showing days of stock cover."""
    if days is None or pd.isna(days):
        return '<span class="ri-bar-text">—</span>'
    width = max(4.0, min(100.0, 100.0 * float(days) / max(max_days, 1e-9)))
    return (f'<div class="ri-bar"><span class="ri-bar-fill {kind}" style="width:{width:.0f}%"></span></div>'
            f'<span class="ri-bar-text">{float(days):.1f} d</span>')


def insight_list(insights: list[dict]) -> str:
    items = "".join(
        f'<div class="ri-insight {escape(i.get("tone", "info"))}">'
        f'<div class="ri-insight-title">{escape(i["title"])}</div>'
        f'<div class="ri-insight-text">{escape(i["text"])}</div></div>' for i in insights)
    return f'<div class="ri-insights">{items}</div>'


def empty_state(title: str, message: str) -> str:
    return (f'<div class="ri-empty"><div class="ri-empty-title">{escape(title)}</div>'
            f'<div class="ri-empty-text">{escape(message)}</div></div>')


def alert(text: str, kind: str = "warning") -> str:
    return f'<div class="ri-alert {escape(kind)}">{escape(text)}</div>'


def key_value_list(pairs: list[tuple[str, str]]) -> str:
    rows = "".join(f'<div class="ri-kv"><span>{escape(k)}</span><strong>{escape(v)}</strong></div>'
                   for k, v in pairs)
    return f'<div class="ri-kv-list">{rows}</div>'
