"""Smoke test for every dashboard page using stand-in streamlit/plotly modules.

Run:  python tests/ui_smoke_stub.py
Uses real packages when installed; otherwise installs light stubs. It executes each page
function against the real dataset and checks the generated HTML, so it catches runtime
errors (bad column names, NaN formatting, wrong data shapes) but NOT visual problems.
"""
import contextlib
import importlib
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MARKUP: list[str] = []
FAKE = None


class _Ctx(contextlib.AbstractContextManager):
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(FAKE, name)


class FakeSt(types.ModuleType):
    session_state: dict = {}
    pressed = False

    def __init__(self):
        super().__init__("streamlit")

    def cache_data(self, *a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

    def markdown(self, body, **k): MARKUP.append(str(body))
    def caption(self, body, **k): MARKUP.append(str(body))
    def columns(self, spec, **k): return [_Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    def container(self, **k): return _Ctx()
    def expander(self, *a, **k): return _Ctx()
    def spinner(self, *a, **k): return _Ctx()
    def radio(self, label, options, index=0, **k): return options[index]
    def selectbox(self, label, options, index=0, **k): return options[index]
    def multiselect(self, *a, **k): return []
    def slider(self, label, a, b, value=None, **k): return value
    def select_slider(self, label, options, value=None, **k): return value
    def button(self, *a, **k): return FakeSt.pressed
    def download_button(self, label, data, *a, **k): assert len(data) > 0
    def metric(self, label, value, **k): MARKUP.append(f"{label}: {value}")
    def plotly_chart(self, fig, width=None, use_container_width=None, **k): assert fig is not None


def install_stubs():
    try:
        import streamlit, plotly.graph_objects  # noqa: F401
        return False
    except ImportError:
        pass
    global FAKE
    FAKE = sys.modules["streamlit"] = FakeSt()

    class Fig:
        def __init__(self, *a, **k): self.n = 0
        def add_trace(self, *a, **k): self.n += 1; return self
        def __getattr__(self, name): return lambda *a, **k: self
    go = types.ModuleType("plotly.graph_objects")
    go.Figure = Fig
    for name in ("Bar", "Scatter", "Histogram"):
        setattr(go, name, lambda **k: k)
    pkg = types.ModuleType("plotly"); pkg.graph_objects = go
    sys.modules.update({"plotly": pkg, "plotly.graph_objects": go})
    return True


def main():
    stubbed = install_stubs()
    import streamlit as st
    import pandas as pd
    from ui import views
    from analytics import load_data

    df = load_data()
    full = views.FilterContext(df, df, df["Date"].min(), df["Date"].max(), "All")
    q4 = df[df["Date"] >= "2025-10-01"]
    scenarios = {
        "full year": full,
        "Q4 + Dairy": views.FilterContext(df, q4[q4["Category"] == "Dairy"], pd.Timestamp("2025-10-01"),
                                          df["Date"].max(), "Dairy"),
        "single day": views.FilterContext(df, df[df["Date"] == df["Date"].max()], df["Date"].max(), df["Date"].max(), "All"),
        "empty": views.FilterContext(df, df.iloc[0:0], df["Date"].min(), df["Date"].max(), "All"),
    }
    pages = [views.executive_overview, views.sales_analytics, views.product_intelligence,
             views.inventory_planning, views.demand_forecasting, views.data_and_methodology]
    for pressed in (False, True):                       # forecast page before / after pressing Run
        st.session_state.clear(); st.pressed = pressed
        for scenario, ctx in scenarios.items():
            for page in pages:
                MARKUP.clear()
                page(ctx)
                text = "\n".join(MARKUP)
                bad = re.findall(r"\b(nan|NaN|None|inf)\b", re.sub(r"<[^>]+>", " ", text))
                assert not bad, f"{page.__name__} [{scenario}] shows {set(bad)}"
        print(f"pages OK (Run pressed={pressed}, stubs={'yes' if stubbed else 'no'})")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
