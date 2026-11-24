import numpy as np
import pandas as pd
import pytest

from forecasting import holdout_predictions
from helpers import make_df
from insights import generate_insights, kpi_values, pct_change, previous_period
from ui import html
from ui.formatting import pct, pkr, pkr_compact


def test_currency_formatting():
    assert pkr(1234567) == "Rs. 1,234,567"
    assert pkr_compact(1_320_000) == "Rs. 1.32M"
    assert pkr_compact(270_500) == "Rs. 270.5K"
    assert pkr(float("nan")) == "—" and pct(None) == "—"


def test_pct_change_never_invents_values():
    assert pct_change(110, 100) == pytest.approx(10.0)
    assert pct_change(100, 0) is None
    assert pct_change(100, None) is None


def test_previous_period_requires_full_coverage():
    df = make_df(days=200)
    assert previous_period(df, df["Date"].min(), df["Date"].max()) is None
    start = df["Date"].min() + pd.Timedelta(days=100)
    prev = previous_period(df, start, df["Date"].max())
    assert prev["Date"].max() == start - pd.Timedelta(days=1)


def test_kpi_values_margin():
    df = make_df(days=10)
    k = kpi_values(df)
    assert k["transactions"] == len(df)
    assert k["margin"] == pytest.approx(0.4)


def test_insights_are_derived_and_handle_empty():
    df = make_df(days=120)
    assert generate_insights(df.iloc[0:0]) == []
    titles = {i["title"] for i in generate_insights(df)}
    assert {"Category leader", "Weekend effect"} <= titles


def test_html_escapes_untrusted_text():
    out = html.table_html(pd.DataFrame({"Product": ["<script>x</script>"]}),
                          [{"key": "Product", "label": "Product"}])
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "ri-kpi" in html.kpi_grid([{"label": "A", "value": "1", "caption": "c"}])


def test_holdout_predictions_shape_and_dates():
    df = make_df(days=200)
    actual, predicted = holdout_predictions(df, horizon=14)
    assert actual.shape == predicted.shape == (14, 3)
    assert (actual.index == predicted.index).all()
    assert (predicted.values >= 0).all()


def test_chart_builders_return_figures():
    go = pytest.importorskip("plotly.graph_objects")
    from analytics import category_performance, product_performance
    from ui import charts
    df = make_df(days=120)
    assert isinstance(charts.revenue_profit_trend(df, "W"), go.Figure)
    assert isinstance(charts.category_comparison(category_performance(df)), go.Figure)
    assert isinstance(charts.margin_scatter(product_performance(df)), go.Figure)
    assert isinstance(charts.daily_revenue_histogram(df), go.Figure)
