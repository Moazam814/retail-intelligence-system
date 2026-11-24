"""Streamlit wrappers around the pure HTML builders in ui.html."""
from __future__ import annotations

import inspect
from pathlib import Path

import streamlit as st

from ui import html

ASSETS = Path(__file__).resolve().parent
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def load_css() -> None:
    css = (ASSETS / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def page_header(title, subtitle="", chips=None):
    _md(html.page_header(title, subtitle, chips))


def section_header(title, description=""):
    _md(html.section_header(title, description))


def kpi_grid(cards):
    _md(html.kpi_grid(cards))


def table(df, columns, row_class=None):
    _md(html.table_html(df, columns, row_class))


def insight_list(items):
    _md(html.insight_list(items))


def empty_state(title, message):
    _md(html.empty_state(title, message))


def alert(text, kind="warning"):
    _md(html.alert(text, kind))


def key_values(pairs):
    _md(html.key_value_list(pairs))


def note(text):
    st.caption(text)


def card():
    """Bordered container used as a white card."""
    return st.container(border=True)


def show_chart(fig):
    """Render a Plotly figure across Streamlit versions (width='stretch' vs use_container_width)."""
    config = {"displayModeBar": False}
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", config=config, theme=None)
    else:
        st.plotly_chart(fig, use_container_width=True, config=config, theme=None)


def download(label, data, file_name, key=None):
    st.download_button(label, data, file_name, "text/csv", key=key)
